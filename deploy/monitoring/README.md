# Gateway API-layer observability (Prometheus + Alertmanager + Loki + Grafana)

A self-contained monitoring stack for the gateway API layer — the same shape as
SlurmUI's: RED metrics, latency percentiles, by-route tables, and a **filterable
request-log panel**. It does not depend on any cluster's Grafana.

```
gateway /metrics             ──scrape──▶ Prometheus ──┬──▶ Alertmanager (:9093, alerts.yml rules)
gateway JSON access log file ──tail──▶ Promtail ▶ Loki┤──▶ Grafana (GPUPlatform — API Layer)
gateway endpoint (vLLM) log  ──tail──▶ Promtail ▶ Loki┤   ({service="vllm"})
Redis queue:{app_id} (sampled into /metrics)─────────┘
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

2. Bring up the stack:

   ```bash
   docker compose -f deploy/monitoring/docker-compose.monitoring.yml up -d
   ```

3. Open Grafana at <http://localhost:3001> (admin / admin) → **GPUPlatform — API
   Layer**. Prometheus + Loki datasources and the dashboard are auto-provisioned.

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
