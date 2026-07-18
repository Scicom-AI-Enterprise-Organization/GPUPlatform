# Gateway API Latency Report

**Date:** 2026-07-18
**Environment:** local dev — gateway `.venv/bin/gateway` (uvicorn, single process) on macOS,
Postgres 16 + Redis 7 in Docker Compose, loopback HTTP (`127.0.0.1:8080`), auth via `sgpu_` API key.
**Method:** 150 sequential requests per endpoint after 10 warm-ups (httpx, keep-alive);
concurrency pass = C parallel workers × 40 requests each. Numbers are milliseconds.
**Dataset shape at measurement time:** 272 training runs, ~60 benchmarks, 1 app, 3 proxy upstreams.

> Loopback numbers exclude network/TLS/ingress. Treat them as the gateway's *floor*:
> prod adds RTT + ingress, but the relative ranking and the outliers hold.

## Sequential latency by endpoint

| Endpoint | Category | p50 | p90 | p95 | p99 | max |
|---|---|---:|---:|---:|---:|---:|
| `GET /` | probe | 0.70 | 0.78 | 0.80 | 0.90 | 1.2 |
| `GET /health` (liveness) | probe | 0.73 | 1.3 | 1.9 | 11.0 | 19.7 |
| `GET /ready` (redis+pg) | probe | 2.5 | 3.5 | 3.8 | 4.2 | 5.4 |
| `GET /auth/me` | control | 2.2 | 2.9 | 3.2 | 5.0 | 7.4 |
| `GET /openapi.json` | schema | 2.5 | 2.8 | 2.9 | 4.4 | 5.7 |
| `GET /api-keys` | control | 2.5 | 3.6 | 4.2 | 6.4 | 9.5 |
| `GET /apps` | control | 3.2 | 4.2 | 4.7 | 6.7 | 8.3 |
| `GET /admin/audit-logs?limit=50` | analytics | 3.7 | 6.0 | 7.5 | 30.3 | 46.9 |
| `GET /v1/providers` | control | 4.2 | 6.3 | 7.3 | 14.1 | 17.6 |
| `GET /v1/storage` | control | 4.8 | 7.4 | 8.8 | 10.7 | 12.5 |
| `GET /v1/datasets` | control | 6.5 | 10.7 | 13.0 | 32.3 | 76.5 |
| `GET /v1/history/activity` | analytics | 7.8 | 10.6 | 12.3 | 62.1 | 87.7 |
| `GET /metrics` (Prom scrape) | metrics | 25.8 | 39.8 | 45.1 | 66.3 | 82.1 |
| `GET /benchmarks/_page?limit=20` | control | 86.1 | 116.9 | 129.9 | 187.2 | 233.9 |
| `GET /metrics/resources` | metrics | 107.4 | 158.4 | 177.9 | 228.3 | 261.0 |
| `GET /v1/training-runs` | control | 145.9 → **107.7** | — | 231.7 → 220.2 | 439.3 → 386.6 | — |

## Concurrency (sustained parallel load, zero errors in every run)

| Endpoint | Concurrency | Requests | Throughput | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| `GET /health` | 50 | 2000 | 732 req/s | 61.8 | 191.7 | 236.9 |
| `GET /auth/me` | 50 | 2000 | 322 req/s | 145.6 | 500.6 | 578.3 |
| `GET /apps` | 20 | 800 | 293 req/s | 63.3 | 193.1 | 229.8 |
| `GET /apps` | 50 | 2000 | 249 req/s | 190.5 | 644.0 | 754.9 |
| `GET /v1/history/activity` | 20 | 800 | 125 req/s | 153.9 | 380.9 | 442.4 |

Zero 5xx and zero connection errors across all 7,600 concurrent requests. Throughput is
bounded by the single uvicorn process (one event loop); horizontal scale-out is the
supported path (`gateway.replicaCount` + `GATEWAY_HA=1` in the Helm chart).

## Findings

1. **Auth + control-plane reads are cheap.** The `sgpu_` key path (SHA-256 lookup +
   throttled `last_used_at` write) costs ~2ms; typical list endpoints sit at 3–8ms p50.
   The DB pool (`pool_size=20, max_overflow=20`, pre-ping) is not a bottleneck at this load —
   `gateway_db_pool_checked_out` stayed ≤ 3 of 40 during the concurrency pass.

2. **`GET /v1/training-runs` was the slowest control-plane endpoint (146ms p50)** because the
   legacy list serialized every run's full `result_json` (per-step training curves — thousands
   of points per run) for all 272 runs. **Fixed in this pass:** the legacy list now slims
   `result_json` to `{"best": …}` exactly like the newer `/_page` endpoint (the full record
   remains at `GET /v1/training-runs/{id}`), and the per-owner username lookup became one
   `IN` query. Result: **p50 146 → 108ms (−26%), p99 439 → 387ms**; remaining cost is 272 ×
   `config_json` rows in one response — pagination (`/_page`) is the real fix and is what the
   web UI uses. Expect a larger win in prod where runs carry bigger step histories.

3. **`GET /metrics` (26ms p50)** samples queue/worker gauges from Redis per scrape. This pass
   pipelined the per-app sampling (was 2 + 2·apps + workers sequential round-trips, now 3
   pipelined batches) and made the scrape survive a Redis outage (`gateway_redis_up 0` instead
   of an HTTP 500). At 1 app the pipelining is invisible; it matters at fleet scale.

4. **`GET /metrics/resources` (107ms p50)** does full-inventory Postgres GROUP-BYs plus one
   series per training run (capped at 500). This is by design — it is scraped at 60s intervals
   by Prometheus, not called interactively. No change needed; keep it off hot paths.

5. **`GET /benchmarks/_page` (86ms p50)** pays for `count(*)` over a subquery plus JSON-heavy
   benchmark rows. Acceptable for a paged UI endpoint; if it grows, the same `result_json`
   slimming used for training runs applies.

6. **Probes are sub-3ms** — `/ready` (now Redis **and** Postgres, each behind a 2s timeout)
   costs 2.5ms p50, safe for 5s Kubernetes probe intervals.

## Changes made in this hardening pass that affect latency

- `/ready` now also pings Postgres (adds ~1.5ms to the probe; a replica with an exhausted
  pool is pulled from the LB instead of serving 500s).
- `/metrics` Redis sampling pipelined + outage-tolerant.
- `/v1/training-runs` slimmed (−26% p50 measured, larger payload savings at prod data shapes).
- Request-id `ContextVar` + per-request log filter: no measurable overhead (sub-0.1ms,
  within noise on `/`).
- Global exception handler + optional `MAX_REQUEST_BODY_MB` guard: zero cost on the happy path
  (header check only).

## Reproduction

```bash
# gateway running locally, key with platform access:
.venv/bin/python scratchpad/latency_bench.py   # see docs note below
```

The benchmark script lives in the session scratchpad (`latency_bench.py`); it is a ~100-line
httpx script — sequential percentiles + semaphore-bounded concurrency — easy to re-create or
lift into `gateway/tests/` as a perf harness if you want it versioned.
