# Helm chart overview — what `helm install` actually creates

Scoped to the chart. For the whole-system architecture (including PI workers
and request lifecycle) see [`docs/ARCHITECTURE.md`](../../../docs/ARCHITECTURE.md).

## What runs WHERE

```
                                    INTERNET
                                       │
                                       │ HTTPS (TLS via cert-manager)
                                       ▼
        ┌──────────── Your k8s cluster ──────────────────────────────┐
        │                                                             │
        │   ┌─ Ingress ─────────────────────────────────────┐        │
        │   │  host: api.your-domain.com  (class: nginx)    │        │
        │   │  TLS: cert-manager auto-provisioned           │        │
        │   │  annotations: SSE-friendly (proxy-buffering off)│      │
        │   └────────────────────┬──────────────────────────┘        │
        │                        │                                    │
        │                        ▼                                    │
        │   ┌─ Service: gateway (ClusterIP) ──────────────────┐      │
        │   │  port 80  → targetPort 8080                      │      │
        │   └────────────────────┬─────────────────────────────┘      │
        │                        │                                    │
        │                        ▼                                    │
        │   ┌─ Deployment: gateway ────────────────────────────┐     │
        │   │  replicas: 2 (configurable)                       │     │
        │   │  image: ghcr.io/<owner>/serverlessgpu-gateway:tag │     │
        │   │  envFrom Secret: PI_API_KEY, GATEWAY_API_KEYS     │     │
        │   │  /health (liveness)  /ready (readiness, pings    │     │
        │   │  redis below)                                     │     │
        │   └────────────────────┬─────────────────────────────┘     │
        │                        │ redis://serverlessgpu-redis:6379  │
        │                        ▼                                    │
        │   ┌─ Service: redis (ClusterIP, headless) ───────────┐     │
        │   │  port 6379                                        │     │
        │   └────────────────────┬─────────────────────────────┘     │
        │                        │                                    │
        │                        ▼                                    │
        │   ┌─ StatefulSet: redis ─────────────────────────────┐     │
        │   │  replicas: 1                                      │     │
        │   │  image: redis:7-alpine + --appendonly yes (AOF)   │     │
        │   │  PVC: 5Gi (storageClassName from values)          │     │
        │   └───────────────────────────────────────────────────┘     │
        │                                                             │
        │   ┌─ Secrets ─────────────────────────────────────────┐     │
        │   │  -pi:    PI_API_KEY                               │     │
        │   │  -auth:  GATEWAY_API_KEYS (comma-separated)       │     │
        │   └───────────────────────────────────────────────────┘     │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘
                                       ▲
                                       │ HTTPS dial-back
                                       │ (Bearer registration_token,
                                       │  then heartbeats every 5s)
                        ┌──────────────┴──────────────────┐
                        │                                  │
                        ▼                                  ▼
       ┌─ Prime Intellect pod ─────┐    ┌─ Prime Intellect pod ─────┐
       │  GPU: H100 / RTX 3090 / …  │    │  ...                     │
       │  Image: customTemplateId    │    │  (provisioned by gateway │
       │  vllm/vllm-openai:latest    │    │   autoscaler on demand,  │
       │   + worker-agent            │    │   torn down when idle)   │
       │   + entrypoint-pi.sh        │    │                          │
       │                              │    │                          │
       │   vllm serves :8000 ◄──┐     │    └──────────────────────────┘
       │   worker-agent ────────┘     │
       │     - BRPOPs queue:{app_id} │
       │     - calls localhost:8000  │
       │     - heartbeat to gateway  │
       └──────────────────────────────┘
```

**Key fact**: GPU pods are NOT in your cluster. They run on Prime Intellect.
The cluster only hosts the **control plane** (gateway + redis), which is
small and CPU-only.

## k8s objects this chart creates

Render with `helm template` to see exact names. With release name `sgpu`:

| Kind | Name | Purpose |
|---|---|---|
| `Deployment` | `sgpu-serverlessgpu-gateway` | FastAPI control plane, 2 replicas by default |
| `Service` | `sgpu-serverlessgpu-gateway` | ClusterIP, port 80 → 8080 |
| `Ingress` | `sgpu-serverlessgpu` | One ingress for everything; nginx class; cert-manager TLS |
| `StatefulSet` | `sgpu-serverlessgpu-redis` | Redis 7 with AOF persistence |
| `Service` | `sgpu-serverlessgpu-redis` | Headless ClusterIP for stable DNS |
| `PersistentVolumeClaim` | `data-sgpu-serverlessgpu-redis-0` | 5Gi, default StorageClass |
| `Secret` | `sgpu-serverlessgpu-pi` | `PI_API_KEY` (only when `provider=primeintellect`) |
| `Secret` | `sgpu-serverlessgpu-auth` | `GATEWAY_API_KEYS` (only when set) |

That's it. **5 templates → 7-8 k8s objects** depending on values.

## What we deliberately don't create

- **No HorizontalPodAutoscaler** — gateway is stateless, scale manually with `replicaCount`
  (+ `gateway.ha: true` so only one replica runs the mutating controllers)
- **No NetworkPolicy** — operator's choice, depends on cluster security model
- **No per-app Ingress** — one ingress routes everything in-gateway by path or body
- **No GPU node pools** — workers run on PI, not in your cluster

(Formerly on this list, now shipped: a **PodDisruptionBudget** renders
automatically at `replicaCount >= 2` — see `gateway.pdb` in values; a
**ServiceMonitor** and a **PrometheusRule** carrying the gateway alert rules
render when the `monitoring.coreos.com/v1` CRDs exist — see
`metrics.serviceMonitor` / `metrics.prometheusRule`.)

## Component responsibilities

### Gateway (in-cluster)
- Accepts HTTP requests (auth via Bearer token)
- Stores app definitions in Redis
- Enqueues jobs into `queue:{app_id}` Redis lists
- Runs the **autoscaler** loop (1Hz) that calls PI API to spawn workers
- Runs the **reconciler** loop (5s) that GCs disappeared workers
- Validates **registration tokens** (one-shot, 5min TTL)
- Serves SSE streams via Redis pubsub
- Exposes `/metrics` (Prometheus), `/health`, `/ready`

### Redis (in-cluster, persistent)
- Job queues per app
- Worker registry + heartbeats (TTL'd)
- App definition records
- Result cache (3600s TTL)
- Registration token storage (one-shot)
- Pubsub channels for streaming
- Cancel signaling

### PI workers (out-of-cluster, ephemeral)
- Provisioned on-demand by gateway autoscaler
- Run vLLM + worker-agent in one container
- Dial back to gateway via the public Ingress URL
- BRPOP jobs, forward to localhost vLLM, write results / publish stream chunks
- Heartbeat every 5s; gateway TTL'd if heartbeat stops
- Self-terminate when receiving drain signal

## Request flow through this chart

A user sends `POST /run/qwen` to the public ingress URL. Trace:

```
1. Ingress controller terminates TLS, forwards to gateway Service
2. Service load-balances to one of the gateway pods
3. Gateway pod's FastAPI matches /run/{app_id}
4. Auth middleware validates Authorization header against GATEWAY_API_KEYS
5. Gateway LPUSHes job to Redis queue:qwen
6. Returns 202 with request_id immediately
7. Autoscaler tick (1Hz): sees qlen=1, current_workers=0, calls PI API
8. PI provisions a pod with our customTemplateId image
9. ~60s later: worker-agent inside that pod calls /workers/register on the
   gateway URL, gets validated, starts BRPOPing queue:qwen
10. Worker picks up the job, calls vLLM localhost, writes result to Redis
11. Client polls GET /result/<request_id>, gets the completion
```

Workers stay alive until `app.autoscaler.idle_timeout_s` passes with no
requests, then autoscaler tells PI to terminate.

## Resource sizing defaults

| Component | requests | limits | Notes |
|---|---|---|---|
| Gateway pod | 100m CPU / 256Mi mem | 1 CPU / 1Gi mem | Stateless, handles ~100 RPS comfortably |
| Redis pod | 100m CPU / 256Mi mem | 500m CPU / 1Gi mem | OK for ~10k apps + queues |
| Redis storage | 5Gi PVC (gp3 / default SC) | — | Bump for high-throughput workloads |

These are sized for "few hundred apps, tens of thousands of requests/day".
For higher scale, bump `gateway.replicaCount` and `redis.storage.size`.

## Files in this chart

```
deploy/helm/serverlessgpu/
├── Chart.yaml                          metadata
├── values.yaml                         all knobs, with comments
├── values-prod.example.yaml            template — `cp` and fill TODOs
├── README.md                           helm install command
├── overview.md                         this file
└── templates/
    ├── _helpers.tpl                    name + labels helpers
    ├── gateway.yaml                    Deployment + Service
    ├── ingress.yaml                    Ingress (gated on values.ingress.enabled)
    ├── redis.yaml                      StatefulSet + Service
    └── secret.yaml                     PI key + auth keys (gated)
```
