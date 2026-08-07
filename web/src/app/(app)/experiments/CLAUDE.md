# Claude guide — Experiments UI (`web/src/app/(app)/experiments/`)

Agent-observability section: replay captured requests against endpoints and score every reply.
Server pages fetch via `gateway.*` (`web/src/lib/gateway.ts`); client components mutate through
`/api/proxy/v1/{experiments,eval-datasets}/*`. All logic lives in the Python gateway
(`gateway/gateway/experiments_api.py` + `evaluators.py` — see the gateway CLAUDE.md section).

## Layout

| Route | File | What it is |
|---|---|---|
| `/experiments` | `page.tsx` + `experiments-list.tsx` | run list (status pills, search/filter/sort, `_page` pagination, 4s poll while any run is active) |
| `/experiments/new` | `new/experiment-form.tsx` | **mirrors `/benchmark/new`** — see below |
| `/experiments/[id]` | `[id]/experiment-detail.tsx` | header KPIs + `?tab=`-less Tabs: Overview / Tradeoff / Samples / Config |
| `/experiments/evaluators` | `evaluators/*` | your reusable evaluator library (author/test/edit) + the built-ins for reference |
| `/experiments/optimize` | `optimize/*` | GEPA prompt optimization — see below |
| — | `tradeoff-plot.tsx` | the parallel-coordinates chart, shared by the detail tabs |
| — | `section-tabs.tsx` | Runs · Optimize · Evaluators. **No Datasets tab on purpose** — see below |
| — | `new/capture-dialog.tsx` | capture requests (Langfuse / served traffic) into a NEW platform dataset |

## The one rule for `new/`: mirror `/benchmark/new`

The create form is built on the same shell as `benchmark/new/benchmark-form.tsx` and
`autotrain/new/training-form.tsx`. **Diff against benchmark-form.tsx before changing anything
here**, and follow it when those patterns evolve. What that means concretely:

- `<FormShell>` wraps a real `<form onSubmit>` (not a button `onClick`), so Enter submits and the
  browser validates. Submit is `type="submit"`.
- Each section is a `SectionCard` (icon chip + title + `CardDescription`) carrying
  **`data-form-section="Title"` + `scroll-mt-6`**. That attribute is the ONLY thing feeding the
  scrollspy rail — FormShell discovers sections from the DOM and re-scans on mutation, so
  conditional cards appear in the rail automatically. There is no section registry to update.
  Current sections: Dataset · Targets · Variants · Evaluators · Run.
- Fields live in a 4-column `Grid` of `FieldWrap`s (uppercase tracking-wide label + `hint`,
  `wide` spans two columns). `SectionCard` / `Grid` / `FieldWrap` are defined locally at the bottom
  of the file, copied from benchmark-form — keep them in sync rather than importing across routes
  (that's the existing convention).
- `<FormFooter>` is the LAST child inside the `<form>`: sticky action bar carrying the submit
  error, a `hint` explaining *why* submit is disabled (the `blocked` discriminant), and
  Cancel/submit. The page's scroller is `relative` so Radix `<Switch>`'s absolutely-positioned
  form-bubble input can't escape the clip and stretch the document past the bar.
- `page.tsx` mirrors benchmark's: crumbs → `relative flex-1 overflow-y-auto px-6 py-6 lg:px-10
  lg:py-8 scrollbar-thin` → `NoAccessAlert` or the form. It also supports **`?from=<exp-id>`**
  to clone an earlier run's whole matrix (name gets `-copy`; API keys are never returned by the
  API, so a clone always re-asks for them).
- The rail is **client-only** (populated in an effect after the DOM scan), so it is legitimately
  absent from the SSR HTML — `/benchmark/new` behaves identically. Don't "fix" that.

## Datasets come from `/datasets` — there is no Experiments dataset store

An earlier cut had a parallel `eval_datasets` concept with its own tab. It's gone: a corpus of
captured requests is a dataset like any other, so it lives in the **Datasets** section where it
can be browsed, published, packed and reused. Consequences to preserve:

- The form's picker is `GET /v1/experiments/datasets` — the platform's datasets filtered to the
  chat-shaped kinds (`llm`/`hf`/`upload`/`s3` **with a messages column mapped**). Dataset links
  across the section point at `/datasets/<id>`, not an Experiments-local page.
- **Capture writes a real dataset.** `new/capture-dialog.tsx` → `/v1/experiments/capture/*`
  creates a `kind=upload` chat dataset on a chosen S3 storage and selects it. Don't reintroduce a
  private store for captured requests.
- **Generating a corpus is a DATASETS feature, not an Experiments one** — red teaming has nothing
  to capture, so `/datasets/new` → source "Generate (synthetic)" has a model write the rows
  (`/v1/datasets/generate`), and the dataset then appears in this form's picker like any other. The
  form links out to it; do NOT reintroduce a generate tab in `capture-dialog.tsx`. Score those runs
  with the `red_team` evaluator (its per-row direction comes from `expected.attack`, which the
  generator writes).
- **Defaults follow the dataset's size, and that is load-bearing.** The two uses of this feature
  want opposite settings: a captured trace (≤ `sweep_case_threshold` rows) is replayed MANY times
  to catch intermittent failures, while a corpus is swept ONCE over a sample. `defaultsFor()`
  picks between them, and `initialDefaults` applies the same choice to the dataset that's selected
  on first paint — otherwise loading the page on a 19k-row corpus opened at 19k units while
  picking that same dataset by hand opened at 200. Ceilings come from
  `GET /v1/experiments/limits`, never hardcoded here.
- **Submit is blocked over `max_units`** (`blocked === "over-cap"`), with a `fitToCap()` action
  that drops repeats to 1 and samples cases while leaving targets and variants alone — those are
  the comparison the user came for. Without the block the form happily composed 382,980 requests
  and only failed server-side.
- The unit-count footer says **`~N cases`**: a dataset reports `num_rows`, but the true case count
  is only known once rows are read (rows without a usable messages cell are skipped).

## Vocabulary: "row", not "case"

A dataset **row** is the unit the UI talks about — one row → one replayed request.
`case` was internal jargon that leaked into the copy and confused people, so nothing
user-facing says it any more: the form says **Max rows**, the footer reads
`N requests will be sent — N dataset rows × targets × variants × repeats`, and the API
uses `max_rows` / `n_rows` / `GET /v1/experiments/datasets/{id}/rows`.

The Python side still calls the object `Case` (`resolve_cases()`, `experiment_samples.case_id`)
because there it IS a distinct thing — a row *resolved into* a replayable request, after the
rows without a usable messages cell are dropped. That distinction is why the footer says
`~N` and why `n_planned` is an estimate. Don't reintroduce "case" into UI copy.

## Contracts

- `web/src/lib/types.ts` — `ExperimentRecord` / `ExperimentSummary` / `ExperimentCell` /
  `EvalDatasetRecord` / `EvalCaseRecord` / `EvaluatorRegistry` must stay in sync with the pydantic
  models of the same shape in `experiments_api.py`.
- **The evaluator list is server-driven.** The form renders whatever
  `GET /v1/experiments/evaluators` returns — id, label, description, and an `options[]` schema
  (`number` / `boolean` / `text` / `list` / `select`). **Adding a detector in `evaluators.py`
  needs no change in this directory.** Don't hardcode evaluator ids here; the only list that
  exists is `DEFAULT_EVALUATORS` (which detectors start checked on a fresh form).
- **Custom evaluators are AUTHORED on the Evaluators tab, not in the run form.** They're reusable
  across experiments, so `evaluators/custom-evaluator-editor.tsx` lives there; the run form only
  *selects* from the library and links out. Don't move the editor back into the form.
  `GET /v1/experiments/evaluators` carries `custom` (the library) and `custom_context`
  (variables, helpers, safe methods, api defaults, examples, and whether python mode is allowed)
  so both surfaces need one fetch and the help text can't drift from the validator. Selecting one
  sends `custom:<ce-id>`. The **Test** button (`POST /v1/custom-evaluators/test`) is load-bearing,
  not decoration — `fail_when_true` silently inverts every result if chosen wrong.
  Three modes in the editor: **expression** (textarea), **api** (the `API_FIELDS` grid — URL plus
  dotted response paths), **python** (greyed out unless `custom_context.python_allowed`, with the
  reason inline — that flag is now `admin role` alone, since python mode is on by default;
  `EXPERIMENT_ALLOW_PYTHON_EVALUATORS=0` still turns it off platform-wide). api mode stores its settings in `config`, not `code`, so `codeReady` gates the
  Test/Save buttons on the URL instead.
- Target suggestions come from `GET /v1/experiments/targets` (this platform's apps + proxy
  endpoints) but they only *prefill* the form — an experiment always stores plain
  `{base_url, model, key}`, so a third-party endpoint is equally valid. Don't add a
  "platform target" code path.
- Section access: key `experiments` — registered in `Section` (me.ts), `SectionKey` (types.ts),
  roles-manager, organization table, user-profile `SECTION_LABEL`, and the sidebar item (which
  IS platform-gated via `section: "experiments"`, unlike Autotrain's). Page check is
  `me.is_admin || sections?.experiments`.
- Live updates: the list polls every 4s while any run is `running`/`queued`; the detail polls
  every 3s while its own run is active. No SSE — a run writes progress to the row every ~2s and
  there's no log stream to follow.

## Optimize (`optimize/`) — GEPA, and the loop back into Runs

`optimize/page.tsx` + `optimize-list.tsx` (runs) · `optimize/new/optimize-form.tsx` (create) ·
`optimize/[id]/optimize-detail.tsx` (result). All three mirror their Runs counterparts —
diff `optimize-list.tsx` against `experiments-list.tsx` and `optimize-form.tsx` against
`experiment-form.tsx` before changing structure. Backend + the algorithm's gotchas are in
**`gateway/gateway/CLAUDE.md` → "Prompt optimization — GEPA"**. What's UI-specific:

- **It's a peer tab, not a sub-page of Runs.** A search *uses* the same datasets and evaluators, and
  its output is a prompt you then run an experiment with — siblings in one loop.
- **The form PRICES the run, and that number must match the server.** GEPA's budget is real billed
  requests, so the Budget card resolves it client-side with the same arithmetic as
  `resolve_budget()` in `prompt_opt_api.py` (preset × validation rows, floored at one full
  iteration, clamped to the ceiling) and prints "Up to N billed requests" above the submit bar.
  Ceilings come from `GET /v1/prompt-optimizations/limits` — never hardcode them here. Change the
  Python, change this.
- **The component list is server-driven** (`limits.components`), same convention as the evaluator
  registry: adding a mutable slot in `prompt_opt.COMPONENTS` needs no change in this directory.
- **Two warnings on the detail page are load-bearing, not decoration.** `unscored_rollouts > 0`
  means replies no evaluator graded (all of them = the run measured nothing however green it looks);
  `in_sample` means the minibatches reused the validation rows so the gain is in-sample. Both come
  straight off `result_json`; don't quietly drop them to tidy the layout.
- **The loop closes through `?prompt=opt-…`.** `/experiments/new` fetches the run server-side and
  seeds **two** variants — `baseline` and `optimized` (the winner in `system_override`) — on the
  dataset the search used, with a banner naming its origin. A GEPA result is a comparison, not a
  single prompt: the optimizer scored it on a validation slice, and that run confirms it on the
  whole corpus. Don't "simplify" it to one variant.
- **`system_override` is a new VariantSpec field** (replaces the row's system message; prefix/suffix
  still decorate on top) and is a normal part of the experiment form now — it is where an optimized
  prompt lands, but it's useful on its own.
- `react-hooks/set-state-in-effect` fires on `optimize-form.tsx`'s seed-prompt fetch, same as the
  ~44 pre-existing instances noted below. The build does not gate on it.

## The two benchmark unit tests

`Function Call Unit Tests` / `Multilingual Unit Tests` are ports of external benchmarks — the
provenance, the parity requirements, and the single-turn caveat are in the **root `CLAUDE.md`**
(always loaded) and `gateway/gateway/CLAUDE.md`. UI-side there's nothing special: they arrive
through the same server-driven registry as every other built-in, so the Evaluators tab and the run
form list them with no per-evaluator code. The only UI-visible difference is that they emit
corpus-level `metrics`, rendered by `BenchmarkMetrics` below.

## Corpus-level metrics (`BenchmarkMetrics` in the detail view)

`function_call_units` and `multilingual_units` report numbers that are pooled across every sample
in a cell — an F1 or a per-language accuracy can't be recovered from per-sample rates. They arrive
on `cell.evals[id].metrics` with a `headline` ordering, and render as their own block under the
Overview table rather than as columns in it (they aren't failure counts). Any evaluator that
declares an `aggregate` in `evaluators.py` shows up here automatically — no change needed in the UI.

## The tradeoff plot (`tradeoff-plot.tsx`)

Parallel coordinates, hand-rolled SVG (recharts has no such chart, and the highlight-one-line
interaction needs direct control). One polyline per `(target, variant)` cell; axes are built from
the summary — categorical `target`/`variant` first, then pass rate, each evaluator's pass rate,
error rate, p95 latency, p95 TTFT, output tokens, cost.

**⚠ The colours are validated, not chosen by eye — don't "tidy" them.** `SERIES_CLASS` holds the
app's existing categorical hues **re-ordered** so every adjacent pair separates under simulated
colour-vision deficiency, with a **separate dark set stepped into the dark lightness band**
(0.48–0.67) rather than an automatic flip. Re-run the validator if you touch them:

```
node <dataviz-skill>/scripts/validate_palette.js "#2563eb,#f59e0b,#10b981,#8b5cf6,#06b6d4,#ec4899" --mode light
node <dataviz-skill>/scripts/validate_palette.js "#3b82f6,#d97706,#059669,#8b5cf6,#0891b2,#ec4899" --mode dark
```

Both sets carry one WARN that is **discharged by secondary encoding**: the direct labels down the
left edge and the cells table rendered under the plot. Removing either leaves colour doing the
identity work alone — that's the thing the WARN is about. The table is not a duplicate of the
plot; it carries the exact numbers the axes only position.

## Gotchas

- `_id` on target/variant drafts is a **client-only React key**. `onSubmit` builds each payload
  field explicitly rather than spreading-and-deleting, so `_id` never reaches the API and there's
  no unused-var disable to maintain.
- The paste importer accepts the shape the old hand-written stress scripts consumed —
  `{messages, tools}` — so an existing `input.json` works unchanged. It also accepts a bare
  messages array and an array of case objects. Keep that tolerance.
- `react-hooks/set-state-in-effect` fires on the fetch-on-mount effects here. The repo has ~44
  pre-existing instances of the same pattern and the build does not gate on it; the list uses the
  `bootedRef` guard (mirroring `quantization-list.tsx`) so it doesn't refetch over SSR data.
