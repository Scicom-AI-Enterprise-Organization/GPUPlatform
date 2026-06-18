# GPUPlatform Agent Notes

## Local Dev

- `docker-compose.yml` must build the `gateway` service from the repo root with
  `gateway/Dockerfile`. That Dockerfile copies both `gateway/` and
  `worker-agent/`, so `build: ./gateway` is invalid and breaks `docker compose up gateway`.
- The compose `gateway` service is intentionally local-dev friendly:
  `AUTH_DISABLED=1`, `ADMIN_USERNAME=admin`, `ADMIN_PASSWORD=admin` by default.
  Override them from the shell when you need stricter auth behavior.
- `postgres` is exposed on host port `5436`, while the compose `gateway`
  connects to the internal service port `5432`.
- The top-left sidebar brand uses two assets intentionally:
  `/public/logos/scicom-logo-light-v2.svg` in light mode and the inline
  `ScicomLogo` SVG component in dark mode. Keep both theme states readable
  against the sidebar background if the branding changes again.

## GPU Timeline Data

- The admin GPU Timeline is sourced from `/admin/worker-events`, which joins:
  `worker_events` + `apps` + `providers` + `benchmarks` + `users`.
- The analytics page does not fetch `/api/analytics/worker-events` up front
  anymore. Core charts/tables load first, and the worker-events payload is only
  fetched when the `GPU Timeline` tab becomes active for the current date
  window. If you touch that tab, preserve the lazy-fetch boundary unless there
  is a deliberate product reason to regress initial load time.
- The GPU Platform side is split into two web proxy routes now:
  `/api/analytics/gpuplatform-overview` for the summary cards/charts and
  `/api/analytics/gpuplatform-records` for Running now + Jobs + node-oriented
  tables. Keep their filter semantics aligned; a date/source/app filter change
  must still mean the same thing across both payloads.
- The analytics UI defaults to the `GPU hours` tab so the overview can stay on
  the lighter overview payload. The record payload is intentionally deferred
  until the `Running now` section is near the viewport or the user opens a
  record-driven tab (`Jobs`, `Node timeline`, `Nodes`). Preserve that behavior
  unless product explicitly wants initial-load regressions in exchange for
  immediate record visibility.
- The timeline is rendered as a week-style calendar: day columns across, hour
  rows downward, and concurrent workloads side-by-side within each day. The top
  controls scope the view to one GPU node at a time and one selected week at a
  time. Inference blocks are blue, benchmark blocks are yellow, and status is
  shown with badges / hover content instead of additional block colours. The
  blocks have a client-side animated hover popout on an opaque dark surface. The
  week grid should fit all 7 day columns within the analytics panel on standard
  desktop widths; keep horizontal scrolling only as a narrow-screen fallback, not
  the default desktop behavior. Default the selected week to the current week
  when it has blocks for the active node; otherwise fall back to the latest
  populated week so the live view does not open on an empty calendar. If you
  change the block shape, keep the hover-card fields in sync so node, GPU lane,
  duration, and worker / benchmark ids still render correctly.
- Inference statuses are intentionally derived from worker lifecycle events, not
  request outcomes. Keep `running` for open spans, `served` for normal
  `terminated` / `idle_terminated`, and `failed` for `terminate_failed`. Expose
  the raw `endReason` in the hover card so a red inference dot is understood as
  a teardown/lifecycle failure rather than a guaranteed request failure.
- Minimal useful dummy dataset for the UI:
  one admin user, one or more VM providers, one or more apps with
  `provider_id` + `visible_devices`, worker lifecycle rows in `worker_events`,
  and optional benchmark rows with `started_at`, `ended_at`, `provider_id`,
  and `visible_devices`.
- The client collapses worker lifecycle rows into spans using:
  `provisioned|registered|scaled_up` as ON and
  `terminated|idle_terminated|terminate_failed` as OFF.

## Verification

- For local UI verification, make sure both succeed before debugging the page:
  `curl http://localhost:8080/health`
  `curl http://localhost:3000/api/analytics/worker-events?...`
- If the Analytics page is empty, inspect seeded rows first rather than the UI:
  check `apps.provider_id`, `apps.visible_devices`, `worker_events.created_at`,
  and `benchmarks.started_at/ended_at`.
