# Gateway API-layer observability (Prometheus + Alertmanager + Loki + Tempo + Grafana)

A self-contained monitoring stack for the gateway API layer — the same shape as
SlurmUI's: RED metrics, latency percentiles, by-route tables, and a **filterable
request-log panel**. It does not depend on any cluster's Grafana.

```
gateway /metrics             ──scrape──▶ Prometheus ──┬──▶ Alertmanager (:9093, alerts.yml rules)
gateway JSON access log file ──tail──▶ Promtail ▶ Loki┤──▶ Grafana (GPUPlatform — API Layer)
gateway endpoint (vLLM) log  ──tail──▶ Promtail ▶ Loki┤   ({service="vllm"})
Redis queue:{app_id} (sampled into /metrics)─────────┤
LLM-proxy requests ──OTLP──▶ Tempo (:4418 in, :3200 out)┘   (one span per proxied request)
```

## Run it locally

1. Start the gateway with JSON logs teed to files Promtail can tail:

   ```bash
   LOG_JSON=1 \
     GATEWAY_ACCESS_LOG=deploy/monitoring/logs/gateway-access.log \
     GATEWAY_ENDPOINT_LOG=deploy/monitoring/logs/endpoint.log \
     .venv/bin/gateway
   ```

   - `GATEWAY_ACCESS_LOG` → per-request HTTP access lines (`service="gateway"`).
   - `GATEWAY_ENDPOINT_LOG` → serverless-endpoint **vLLM** logs (`service="vllm"`),
     the same lines shown on `/serverless/<app>?tab=logs`, re-emitted from the
     `/workers/logs` ingest path. Tagged with `app_id` / `model` / `level` labels
     and `machine` / `session` JSON fields.

   (`LOG_JSON` unset → human-readable access lines for the terminal; the access
   file is still written if `GATEWAY_ACCESS_LOG` is set. The endpoint re-emit is a
   no-op unless `LOG_JSON=1` **or** `GATEWAY_ENDPOINT_LOG` is set, so plain dev
   pays nothing.)

   `LOG_JSON=1` makes the **whole stdout stream** JSON, not just the access lines:
   module logs and tracebacks are rendered as `kind="app_log"` records carrying the
   same `requestId`, and uvicorn's own access line — a lesser duplicate of ours,
   which used to bypass the format entirely because uvicorn sets `propagate=False`
   on its loggers — is turned off. Set `LOG_UVICORN_ACCESS=1` to keep it (only
   useful when you suspect requests are dying *before* the middleware sees them).

2. Bring up the stack:

   ```bash
   docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
   ```

   ⚠ **On a machine that also runs SlurmUI's monitoring stack** (`~/Documents/SlurmUI/
   monitoring/`), that stack already owns :3001 / :9091 / :9093, so bringing all of
   these up will fail on port conflicts — start only what's missing, e.g.
   `… up -d tempo`, and add the Tempo datasource to the Grafana you already have
   (URL `http://host.docker.internal:3200`, since the two stacks are on different
   Docker networks). This compose sets an explicit `name: gpuplatform-monitoring`;
   without it Compose derives the project from the parent directory — `monitoring`
   for BOTH files — and a `down` here would tear down SlurmUI's stack.

3. Open Grafana at <http://localhost:3001> (admin / admin) → **GPUPlatform — API
   Layer**. Prometheus + Loki + Tempo datasources and the dashboard are
   auto-provisioned.

## Request tracing (Tempo) — the LLM proxy's request record

Every proxied request can be exported as one OTLP span, which is what lets
`proxy_requests` stop being a row per request in Postgres (that table is
append-only, unbounded, and read almost never — the wrong shape for a
transactional store at prod volume). Start the gateway with:

```bash
PROXY_TRACING=1 \
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4418 \
  TEMPO_URL=http://localhost:3200 \
  PROXY_REQUEST_STORE=off \
  .venv/bin/gateway
```

- `OTEL_EXPORTER_OTLP_ENDPOINT` is where spans are **written** (Tempo's OTLP
  receiver); `TEMPO_URL` is where the Queue tab **reads them back** (Tempo's query
  API). Different ports, different directions — swapping them looks like "tracing
  works but the history is empty".
- ⚠ The OTLP/HTTP host port defaults to **4418**, not the standard 4318, because
  4318 is already taken on this laptop. Override with `TEMPO_OTLP_HTTP_PORT`
  (inside the compose network Tempo still listens on 4318).
- `PROXY_REQUEST_STORE=all|sampled|errors|off` decides how much Postgres keeps;
  `all` (the default) changes nothing. `PROXY_HISTORY_SOURCE=auto` makes
  `/proxy/<id>?tab=queue` read from whichever store actually holds the history.
- `PROXY_TRACE_SAMPLE_RATIO` samples **successful** requests only — failures,
  blocks, cancels and anything slower than `PROXY_TRACE_SLOW_MS` are always kept.
- `TRACE_UI_URL` (e.g.
  `http://localhost:3001/explore?...&traceId={trace}`) turns on the per-row
  **trace** link in the Queue tab.

In Grafana → Explore → Tempo, search with TraceQL:

```
{ span.sgpu.proxy.name="for-agentic" && span.sgpu.status="failed" }
{ span.sgpu.request.id="pxr-4ce76aeb0409d8b5" }
{ span.sgpu.latency_ms > 10000 }
```

The datasource is wired for **trace → logs** (jumps to that request's Loki lines by
`requestId`) and **trace → metrics**.

## What you get

- **Overview (RED):** request rate, error rate, p95 latency, 5xx rate.
- **Latency:** p50/p95/p99 by route, heatmap, slowest-routes table.
- **Queue:** depth per app, depth-vs-workers saturation, total queued.
- **Logs (Loki):** the request-log panel, filterable by the `Log status (regex)`,
  `Min latency (ms)`, `app_id`, `route`, and `method` dashboard variables. E.g. set
  Log status to `5..` to see only 5xx, or Min latency to `1000` for slow requests.
- **Alerts:** `prometheus/alerts.yml` — gateway down, Redis unreachable
  (`gateway_redis_up`), DB-pool saturation, stalled background loops
  (`gateway_loop_last_tick_timestamp_seconds`), 5xx/latency, queue backlog with
  zero workers, provision/terminate failures, autotrain run failed/stuck, and
  LLM-proxy failure rate. Routed to **Alertmanager** at <http://localhost:9093>;
  wire Slack/Telegram/webhook receivers in `alertmanager/alertmanager.yml` (the
  file ships with commented examples and a no-op default so the stack boots
  without secrets). The Helm chart mirrors these rules as a `PrometheusRule` —
  **keep the two files in sync**.

## Fleet URLs

The gateway already serves OpenAI-compatible, per-fleet paths:

```
https://serverlessgpu.aies.scicom.dev/{app_id}/v1/chat/completions
https://serverlessgpu.aies.scicom.dev/{app_id}/v1/audio/transcriptions
```

`{app_id}` is the fleet/endpoint id (e.g. `tm-fleet`). It is captured as the
`app_id` label on the access logs, so the Logs panel and the `app_id` dashboard
variable filter per fleet.

## In-cluster

The Helm chart already ships a `ServiceMonitor` scraping `/metrics`. For logs in
the cluster, set `LOG_JSON=1` on the gateway deployment and point Alloy/Promtail
(from `kube-prometheus-stack` / `grafana/loki-stack`) at the gateway pod stdout,
then import `deploy/grafana/api-layer.json`. With `LOG_JSON=1` the gateway writes
**both** streams to stdout as JSON; distinguish them by the `service` field
(`service="gateway"` access lines vs `service="vllm"` endpoint lines) — e.g. an
Alloy `loki.process` stage that promotes `service`, `app_id`, and (for vLLM)
`model` / `level` to labels. No `GATEWAY_ENDPOINT_LOG` file is needed in-cluster;
that file tee is only for tailing a host-process gateway in local dev.
