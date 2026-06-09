# Usage Reports — Implementation Plan

> Status: **Plan / design** (no feature code written yet). Goal: add a **Usage Report** to
> gpuplatform's web console with **Export to PDF** and **Export to DOCX**, modelled on the
> Reports feature already shipping in the sibling **SlurmUI** repo.
>
> Every file path / column name below was verified against current `main` of both repos
> (pulled 2026-06-08).

## Decisions locked (2026-06-08)

From the product owner:

1. **Content focus** — the report leads with: **which models are being used / submitted**,
   **per-model request volume**, **total request counts**, **4xx / 5xx (error-class) breakdown**,
   and **request volume over time, granular per model** (e.g. "at 14:00, model X had N requests").
   (Latency/GPU-hours are secondary.)
2. **Cost** — **include a "Resource spend" section** sourced from the **Benchmark / TrainingRun /
   ComputePod** tables (`cost_per_hr` + timestamps already exist). Serverless per-request $ is *not*
   in scope (not tracked — see §3d).
3. **Tokens** — **best-effort extraction from the stored `output` JSON** (no schema migration in v1).
4. **Surface** — one **platform-wide admin page** (`/admin/usage-reports`), admin-only. A
   per-endpoint "Usage" tab is an easy follow-up that reuses the same gateway endpoint (§4).

The one genuinely open call is **how to source 4xx/5xx** (durable-approximate vs live-exact) — see §3e and §9.

---

## 1. What we're copying — how SlurmUI's Reports work

SlurmUI's report is **one client page** (`web/app/(admin)/admin/reports/page.tsx`, ~1450 lines) +
a thin data route (`web/app/api/reports/route.ts`).

| Concern | SlurmUI implementation |
|---|---|
| **Page** | A `"use client"` page: filter bar (date presets + cluster/partition/status/user), stat cards, `recharts` charts, expandable per-day breakdown. |
| **Data** | `GET /api/reports` (Next route) aggregates **Prisma → Postgres `Job` table** by day/cluster/user/status; GPU hardware charts come separately from **Prometheus `query_range`**. |
| **PDF export** | **No PDF library.** Button → `window.print()`. Page renders *twice*: screen UI (`print:hidden`) and a **print-only DOM mirror** (`hidden print:block .report-document`, inline styles). `globals.css` `@media print { aside, header { display:none } … }` strips the chrome → "Save as PDF". |
| **DOCX export** | **`docx`** npm pkg (`^9.7.1`), **dynamically imported**. Builds `Document → Table/Paragraph/TextRun`. Charts rasterised to **PNG via a hand-written Canvas 2D renderer** (`generateChartPng`, ~80 lines, no chart lib) → embedded as `ImageRun`. Download via `Packer.toBlob()` + `URL.createObjectURL` + `a.click()`. |

Functions to port almost verbatim (only the data shape changes):
- `exportDocx(...)` — `SlurmUI/web/app/(admin)/admin/reports/page.tsx:594`
- `generateChartPng(...)` — same file, `:508`
- `@media print` block — `SlurmUI/web/app/globals.css:229`

---

## 2. The architectural difference that drives this plan

**In SlurmUI the Next app owns the database** (Prisma) → aggregation lives in a Next route.
**In gpuplatform the web owns nothing** — it is a thin client that proxies everything to the
**Python FastAPI gateway**, the only process with DB access.

So:
1. **Aggregation lives in the gateway**, as a new router (`usage_api.py`), following the
   `global_env_api.py` / `training_api.py` pattern. The web fetches finished JSON via its proxy
   (`web/src/app/api/proxy/[...path]/route.ts`, which injects the `Authorization: Bearer` token
   from the httpOnly `sgpu_token` cookie).
2. **There is no web→Prometheus query path today.** The "live metrics" tab
   (`serverless/[id]/tabs/metrics.tsx`) scrapes the gateway's `GET /{app_id}/metrics` exposition
   endpoint — **counters since process start**, not time-series. The real Prometheus/Loki/Grafana
   stack (`deploy/monitoring/`) is a **separate, not-always-on docker-compose** (15d / 7d
   retention). The report's **durable core must not depend on it**.
3. **The durable historical source is the Postgres `requests` table** — gpuplatform's analogue of
   SlurmUI's `Job` table.

---

## 3. Data sources

### 3a. Backbone: `requests` table (`gateway/gateway/db.py:411`)

```
request_id   PK str
app_id       FK apps.app_id      (indexed)   ← the endpoint/fleet
owner_id     FK users.id         (indexed)   ← the user
endpoint     str                             ← v1/chat/completions, v1/audio/transcriptions, …
payload      JSON                            ← request body — CONTAINS payload["model"]  ★
status       str  (indexed)                  ← pending|completed|failed|timeout|cancelled|error
output       JSON (nullable)                 ← vLLM response (carries usage.* tokens on success)
is_stream    bool
created_at   datetime tz       (indexed)
completed_at datetime tz (nullable)
```

Fully answers the locked content with **zero new instrumentation**:

| Report metric | From `requests` |
|---|---|
| **Models used / submitted** ★ | `payload["model"]` per row → `GROUP BY model` (durable; works for multi-model fleets where the member model is in the payload). Join `apps.model` / `apps.models` for the served set. |
| **Total requests** | `COUNT(*)` over `created_at ∈ [from,to]` (+ filters) |
| Per-endpoint / per-user / daily | `GROUP BY app_id` / `owner_id` / day-bucket(created_at, tz) |
| **Requests over time, per model** ★ | bucket `created_at` into an adaptive time step (hour/day) × `GROUP BY payload["model"]` → a time-series of per-model request counts. Pure Postgres, no instrumentation. |
| Outcome breakdown | `GROUP BY status` (see §3e for the 4xx/5xx mapping) |
| Tokens (best-effort) | extract from `output["usage"]` (§3c) |
| Latency (secondary) | `completed_at − created_at` (end-to-end; includes queue/cold-start) |

**Verified status vocabulary** (`main.py`): `pending`, `completed`, `failed`, `timeout`,
`cancelled`, `error`. Terminal-success = `completed`. Finalisation happens in
`_mirror_status_to_db()` (`main.py:2630`), which writes `status`/`output`/`completed_at` from the
worker's Redis result blob `{status, output}`.

### 3b. Model identity

- **Single-model endpoint:** served model = `apps.model`; the client's `payload["model"]` doubles
  as the model name (`main.py:74`).
- **Multi-model fleet** (`apps.mode == "multi"`): `apps.models = [{model, tp, …}]`; the targeted
  member is resolved from `payload["model"]` (`_resolve_model_to_app`, `main.py:2649`). So
  `payload["model"]` is the correct per-request model key for the breakdown.

### 3c. Tokens — best-effort from `output` (locked: no migration)

Gateway does **not** persist token counts. But non-streaming chat/completions responses carry
`output["usage"] = {prompt_tokens, completion_tokens, total_tokens}`. Plan: sum these in the
gateway aggregation when present; streaming (`is_stream=True`) and audio usually lack `usage` →
count as "tokens unknown" and report a **coverage %** caveat so totals aren't read as exact.

### 3d. Cost — "Resource spend" section (locked: include benchmark/training/compute)

Serverless **worker** lifetimes live only in Redis TTL (never persisted) → serverless GPU-hours/$
are **not computable**; out of scope for v1.

But **`Benchmark`**, **`TrainingRun`**, **`ComputePod`** each carry `cost_per_hr` + start/end
timestamps, and `audit.cost_breakdown()` already computes `final_cost = elapsed_h × cost_per_hr`.
The report includes a **Resource spend** section: per-resource-type and per-user $ over the period,
computed the same way (`cost_per_hr × elapsed`, `cost_per_hr` may be NULL for FakeProvider → show
"—"). This is a separate gateway endpoint/section from inference usage.

### 3e. ⚠ 4xx / 5xx — the one real data-source decision

The literal HTTP status code (incl. 4xx/5xx) is computed in the gateway **HTTP middleware**
(`main.py:544-581`: `status_code = resp.status_code`) and emitted **only** to:
- Prometheus `serverless_http_requests_total{method, route, http_status, app_id}` (process-since-start counter), and
- the JSON access log → Loki (7d).

It is **NOT stored on the `requests` row** (which keeps only the internal `status`). Two
consequences:
- The `requests` table can give an **outcome class** but not literal codes.
- **Genuine 4xx are systematically under-counted** by the `requests` table, because many are
  raised *before* a row exists (e.g. `400 "missing 'model'"` at `main.py:3007`, `404` unknown
  model in `_resolve_model_to_app`, `401` auth). Those never become rows.

**Options (recommend A for v1, offer C as follow-up):**

- **A — Durable outcome-class from `requests` (no new infra).** Map internal status →
  `success` (`completed`) / `server_error≈5xx` (`failed`,`error`,`timeout`) / `client/cancelled≈4xx`
  (`cancelled`), optionally refined by sniffing `output` for an error code (same best-effort spirit
  as tokens). Durable, date-rangeable, per-model. **Caveat (state it in the UI):** approximate, and
  excludes door-rejection 4xx. ✅ recommended v1 core.
- **B — Live-exact from Prometheus.** Read real `serverless_http_requests_total{http_status}` for an
  exact 2xx/4xx/5xx mix — but it's *current totals since gateway start*, not historical, and resets
  on restart. Optionally show as a labelled "current" panel alongside A.
- **C — Durable-exact (follow-up, needs investment).** Either (i) persist `http_status` per request
  (worker returns its upstream code in the result blob → store in a new `requests.http_status`
  column in `_mirror_status_to_db`; still misses door-rejections), or (ii) add a durable HTTP
  access-log/rollup table written from the middleware (captures *all* requests incl. rejections, but
  adds a per-request DB write on the hot path — a perf/scale decision to make deliberately).

**Recommendation:** v1 = **A** (durable, honest labels) and optionally show **B** as a live "exact,
current" cross-check. Defer **C** unless literal historical HTTP codes are a hard requirement.

---

## 4. Scope (locked: admin platform-wide page; per-endpoint tab = follow-up)

- **v1 — Platform-wide admin Usage Report** at `/admin/usage-reports` (admin-only): aggregates
  across **all** apps/users. Matches the "what models are being used across the platform" framing.
- **Follow-up — per-endpoint Usage tab** on `serverless/[id]/endpoint-detail.tsx`, scoped to one
  `app_id` (owner-or-admin). Cheap once the gateway endpoint supports `app_id` scoping.

The gateway endpoint is built to serve both: admin → all; non-admin → forced to own `owner_id`
(mirrors `/apps/{app_id}/requests`).

---

## 5. Target architecture

```
web (Next.js)
  /admin/usage-reports/page.tsx          server: admin guard + ConsoleTopbar
    └─ usage-report.tsx                  "use client": filters, stat cards, recharts,
         ├─ print-only DOM mirror          per-model + status-class + daily breakdown,
         ├─ export-docx.ts (docx)          Export PDF / Export DOCX
         └─ generateChartPng (Canvas→PNG, ported)
  lib/gateway.ts  getUsageReport(params) ─┐  lib/types.ts  UsageReport, UsageByModel, …
  globals.css     + @media print          │  components/console/sidebar.tsx + nav entry
                                          ▼  /api/proxy/[...path]  (Bearer inject)
gateway (FastAPI)
  usage_api.py (NEW, prefix /v1/usage)
    GET /v1/usage/report?from&to&tz&app_id?&owner_id?&model?&status?   (admin or owner-scoped)
        → aggregate `requests` (+ join apps/users): byModel, byApp, byUser, byEndpoint,
          outcome-class (§3e-A), tokens best-effort, daily breakdown
    GET /v1/usage/spend?from&to&tz   (admin)
        → benchmark/training/compute cost_per_hr × elapsed  (§3d)
  main.py  app.include_router(usage_module.router)   (1 line, near :526)
```

New deps: **`docx`** (npm). PDF needs none. No Prometheus dependency for the core.

---

## 6. Implementation plan — phased

### Phase 1 — Gateway usage API (data first)
**New `gateway/gateway/usage_api.py`** (`APIRouter(prefix="/v1/usage", tags=["usage"])`, Pydantic
models, `Depends(current_user)`, `require_admin` for platform scope).

- `GET /v1/usage/report`
  - **Params:** `from`,`to` (YYYY-MM-DD), `tz` (IANA), optional `app_id`, `owner_id` (admin only),
    `model` (csv), `status` (csv), `bucket` (`auto`|`hour`|`day`, default `auto`).
  - **Scope:** admin → all; non-admin → forced `owner_id = current_user.id`.
  - **Query:** one ranged `SELECT` over `requests WHERE created_at ∈ [from,to]` (+ filters), join
    `apps` (name/model/mode) & `users` (username); aggregate in Python (indexes on
    created_at/app_id/owner_id/status already exist):
    - `summary`: totalRequests, success/serverError/clientCancelled counts (§3e-A), successRate,
      tokensIn/Out/total + `tokenCoveragePct`, distinct models/endpoints/users, avg/median/p95
      latency (secondary).
    - `byModel[]`: {model, requests, success, serverError, clientCancelled, tokensTotal,
      avgLatencySec} — sorted desc. ★ primary table.
    - `timeSeries[]`: ★ per-bucket points `{ts, label, total, byModel:{<model>:count}, byStatusClass:{success,serverError,clientCancelled}}`.
      Bucket size from an adaptive step (port SlurmUI's `promStep` idea, `page.tsx:196`): ≤1 day →
      hourly; ≤7 days → 6-hourly or hourly; ≤30 days → daily; longer → daily. `bucket` param overrides.
      Drives the "requests over time, by model" chart and answers "at HH:00, model X had N".
    - `byApp[]`, `byUser[]`, `byEndpoint[]`: analogous rollups.
    - `daily[]`: per calendar day (client TZ) {date, dayLabel, requests, success, serverError,
      tokensTotal, requests:[{requestId, appId, model, endpoint, username, outcome, startTime,
      endTime, elapsedLabel}]} (cap/paginate per-day rows).
    - `models[]`/`apps[]`/`users[]`: filter-dropdown lists. `period`, `scope`.
  - **TZ:** port SlurmUI `route.ts` Intl day-bucketing into Python (`zoneinfo`) so day totals match
    the browser TZ.
  - **Token/outcome helpers:** `usage=(output or {}).get("usage") or {}`; outcome map per §3e-A.
- `GET /v1/usage/spend` (admin): benchmark/training/compute `cost_per_hr × elapsed` rollup by
  resource type + user (§3d).
- **(Optional B)** include exact live HTTP-status mix by reading `metrics.render(...)` /
  `serverless_http_requests_total` and returning a `liveHttpStatus` block, labelled "current".
- **Wire-up:** `main.py` — `from . import usage_api as usage_module` +
  `app.include_router(usage_module.router)` (~`main.py:526`).

### Phase 2 — Web data layer
- `web/src/lib/types.ts` — `UsageReport`, `UsageSummary`, `UsageByModel`, `UsageByApp`,
  `UsageByUser`, `UsageDay`, `UsageDayRequest`, `ResourceSpend`.
- `web/src/lib/gateway.ts` — `getUsageReport(params)` + `getUsageSpend(params)` (browser → `/api/proxy`).

### Phase 3 — Web Reports page (screen UI)
- `web/src/app/(app)/admin/usage-reports/page.tsx` — server component; copy the admin guard +
  layout from `admin/audit/page.tsx` (redirect if `me.role !== "admin"`, `ConsoleTopbar`
  breadcrumb, header, render client component).
- `web/src/app/(app)/admin/usage-reports/usage-report.tsx` — `"use client"`, port of SlurmUI's body:
  - **UI primitives:** `components/ui/{card,button,badge,select,dropdown-menu}` (exist). **No
    `ui/table`** in gpuplatform → use raw `<table>` + Tailwind (like `metrics.tsx`) or add the primitive.
  - **Charts:** `recharts@3.8.1` (present) + `--chart-1..5` CSS vars. Daily requests (Area), tokens (Line).
  - **Filters:** date presets + model multiselect + app multiselect + user select (admin) + status
    multiselect (port SlurmUI `DATE_PRESETS` + `MultiSelect`).
  - **Sections (content-locked order):** stat cards (Total requests · Success rate · ~4xx · ~5xx ·
    Tokens · Distinct models) → **Requests over time, by model** ★ (stacked Area/Bar over
    `timeSeries`, one series per model, with a **granularity toggle** hour/day + the date presets) →
    **By model** table ★ → tokens chart → By-endpoint table → Top users table → **Resource spend**
    table → expandable per-day breakdown (each day drills into its hourly buckets). Show the §3e-A
    approximation caveat near the status-class numbers.

### Phase 4 — PDF export (`window.print()`, no lib)
- Add `@media print` to `web/src/app/globals.css` (none today): hide the console chrome (confirm the
  sidebar/topbar landmark/class names; SlurmUI keys off `aside, header`) + neutralise
  `.report-document` styling.
- Render the **print-only DOM mirror** (`<div className="hidden print:block report-document">`,
  inline styles) mirroring SlurmUI's `P*Table` components with usage data. Button → `window.print()`.

### Phase 5 — DOCX export (`docx`)
- `npm install docx` in `web/` (absent; verified). Port `exportDocx()` + `generateChartPng()` into
  `web/src/app/(app)/admin/usage-reports/export-docx.ts`, swapping section builders to usage data
  (summary, by-model, by-endpoint, top-users, resource-spend tables; the **requests-over-time-by-model**
  chart + tokens chart via the Canvas PNG renderer). Keep the lazy `await import("docx")`.

### Phase 6 — Navigation
- `web/src/components/console/sidebar.tsx` — add to the **ADMIN** list:
  `{ label: "Usage Reports", href: "/admin/usage-reports", icon: BarChart3 }` (`BarChart3` from
  `lucide-react`). Admin gating already handled by the ADMIN list.

### Phase 7 — Optional / later
- Per-endpoint Usage tab (§4). · True historical HTTP codes (§3e-C). · Indexed token columns. ·
  Serverless worker-lifetime → GPU-hour/$ instrumentation. · Prometheus latency-percentile charts.

---

## 7. Report content mapping (SlurmUI → gpuplatform)

| SlurmUI (jobs) | gpuplatform (serverless usage) |
|---|---|
| Cluster | Endpoint (`app_id` / app name) |
| Job | Inference request |
| — | **Model** (`payload["model"]`) — primary breakdown ★ |
| GPU charts (intra-day Prometheus time-series) | **Requests over time, by model** (intra-day hourly buckets from `created_at`, Postgres) ★ |
| Completed / Failed / Cancelled | success (`completed`) / ~5xx (`failed,error,timeout`) / ~4xx (`cancelled`) (§3e) |
| GPU-hours / CPU-hours | **Tokens** (best-effort) + **Resource spend $** (benchmark/training/compute) |
| Top users (by job count) | Top users (by request / token count) |
| vLLM serving jobs | Per-model fleet breakdown |
| Per-cluster table | Per-endpoint table |
| Daily job history | Daily request history (per-model, outcome) |
| GPU metric charts (Prometheus) | Daily requests/tokens charts (Postgres) |

---

## 8. File checklist

**Gateway:** `usage_api.py` *(new)* · `main.py` *(+1 include_router)*.
**Web:** `admin/usage-reports/page.tsx` *(new)* · `admin/usage-reports/usage-report.tsx` *(new)* ·
`admin/usage-reports/export-docx.ts` *(new)* · `lib/types.ts` *(edit)* · `lib/gateway.ts` *(edit)* ·
`globals.css` *(+@media print)* · `components/console/sidebar.tsx` *(+nav)* · `package.json` *(+docx)*.

---

## 9. Remaining open decision

**4xx/5xx fidelity (§3e):** ship v1 with **A** (durable outcome-class approximation from
`requests`, honest "approx / excludes pre-validation rejections" caveat) — recommended — and
optionally surface **B** (exact live counts from Prometheus, labelled "current")? Or is literal,
historical, complete 4xx/5xx a hard requirement, making **C** (persist HTTP status / durable access
log) part of v1?

*(Everything else is decided — see "Decisions locked".)*

---

## 10. Risks & gotchas

- **4xx under-count** (§3e) — the `requests` table misses door-rejection 4xx; label the status-class
  numbers as approximate, or adopt option C for exact counts.
- **Token coverage** — streaming/audio lack `usage`; always show coverage %.
- **Latency semantics** — `completed_at − created_at` is end-to-end (queue + cold-start), not
  inference-only; label accordingly.
- **No web DB access** — all aggregation is gateway-side; don't port SlurmUI's Prisma route into Next.
- **No `ui/table`** — use raw `<table>` (like `metrics.tsx`) or add the primitive.
- **`cost_per_hr` may be NULL** (FakeProvider) → render "—" in Resource spend.
- **Dev auth** — `AUTH_DISABLED=1` makes the local user a seeded admin (page works locally); verify
  the prod `me.role === "admin"` guard too.
- **Timezones** — port SlurmUI's Intl day-bucketing exactly (gateway-side, Python) or day totals drift.
- **Volume/pagination** — cap per-day request rows and surface any truncation.
- **`docx` lazy-load** — keep the dynamic `import("docx")` so the bundle isn't bloated.
```
