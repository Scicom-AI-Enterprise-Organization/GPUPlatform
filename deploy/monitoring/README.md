# Gateway API-layer observability (Prometheus + Loki + Grafana)

A self-contained monitoring stack for the gateway API layer — the same shape as
SlurmUI's: RED metrics, latency percentiles, by-route tables, and a **filterable
request-log panel**. It does not depend on any cluster's Grafana.

```
gateway /metrics            ──scrape──▶ Prometheus ──┐
gateway JSON access log file ──tail──▶ Promtail ▶ Loki┤──▶ Grafana (GPUPlatform — API Layer)
Redis queue:{app_id} (sampled into /metrics)─────────┘
```

## Run it locally

1. Start the gateway with JSON access logs teed to a file Promtail can tail:

   ```bash
   LOG_JSON=1 GATEWAY_ACCESS_LOG=deploy/monitoring/logs/gateway-access.log .venv/bin/gateway
   ```

   (`LOG_JSON` unset → human-readable `POST /tm-fleet/v1/chat/completions → 200
   (842ms)` lines for the terminal; the file is still written if `GATEWAY_ACCESS_LOG`
   is set.)

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
(from `kube-prometheus-stack` / `grafana/loki-stack`) at the gateway pod stdout
with `service=gateway`, then import `deploy/grafana/api-layer.json`.
