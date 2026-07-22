# Local multi-container HA harness

Spins up **2 gateway replicas** behind an **nginx LB**, sharing one Redis +
Postgres, plus a mock OpenAI upstream — enough to exercise both HA switches:

| Switch | What it does | `leader.py` / `proxy_cluster.py` |
|---|---|---|
| `GATEWAY_HA=1` | every replica serves HTTP, only the **leader** runs the singleton controllers (autoscaler/reconciler/janitors/…). Leader = Redis lock `gateway:leader`. | `leader.py` |
| `PROXY_CLUSTER=1` | the LLM proxy's `max_concurrency`, live-queue view and cancel become **cluster-wide** (Redis ZSET lease + `proxy:live:*` + `proxy:cancel` pub/sub) instead of per-replica. Fails **open** if Redis is down. | `proxy_cluster.py` |

Ports: LB `:8090` (what clients hit) · gw1 `:8091` · gw2 `:8092` (direct, for
per-replica inspection).

```bash
docker compose -f deploy/ha-local/docker-compose.yml up --build -d
docker compose -f deploy/ha-local/docker-compose.yml ps    # wait for healthy
```

## Test A — leader election + failover

```bash
curl -s localhost:8091/leader; echo   # {"ha_enabled":true,"is_leader":true, ...}  <- gw1 came up first
curl -s localhost:8092/leader; echo   # {"ha_enabled":true,"is_leader":false, ...}

docker kill sgpu-ha-gateway1-1        # kill the leader
sleep 8                               # > LEADER_TTL_S (6s) -> lock lapses
curl -s localhost:8092/leader; echo   # gw2 now is_leader:true  (promoted in ~TTL)

docker start sgpu-ha-gateway1-1       # rejoins as a follower
```

## Test B — cluster-wide proxy concurrency cap

Create a proxy endpoint with `max_concurrency: 2`, then fire 6 concurrent
requests through the LB (round-robined across both replicas). With
`PROXY_CLUSTER=1` the mock must never see **>2** at once; without it you'd see up
to 4 (2 per replica).

```bash
# create endpoint 'demo' -> mock upstream, cluster-wide cap of 2
curl -s localhost:8090/v1/proxy -X POST -H 'content-type: application/json' -d '{
  "name": "demo", "max_concurrency": 2, "enabled": true,
  "upstreams": [{"name":"mock","base_url":"http://mock-upstream:9000/v1","models":{"mock":"mock"}}]
}'; echo

# 6 concurrent chat requests through the LB
seq 6 | xargs -P6 -I{} curl -s -o /dev/null -w "req {} -> %{http_code} %{time_total}s\n" \
  localhost:8090/proxy/demo/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}'

# cap held cluster-wide iff peak == 2 (six 3s requests => ~9s total, 3 waves of 2)
docker compose -f deploy/ha-local/docker-compose.yml logs mock-upstream | grep peak | tail -1
```

**Negative control** — repeat with the proxy cluster OFF and the same 6 requests
now peak at **4** (2 per replica), ~6s in 2 waves, proving the coordination is
what holds the cap:

```bash
docker compose -f deploy/ha-local/docker-compose.yml \
               -f deploy/ha-local/docker-compose.nocluster.yml up -d
# ... re-fire the 6 requests ... grep peak  => peak=4
docker compose -f deploy/ha-local/docker-compose.yml up -d   # restore PROXY_CLUSTER=1
```

## Test C — cross-replica live view + cancel/flush

`flush` cancels only requests still **queued** (waiting for a slot); already
in-flight non-stream requests can't abort mid-upstream-read (documented caveat in
`gateway/gateway/CLAUDE.md`). Use the slow-mock override so requests sit queued
long enough for the 2s cluster sync loop to mirror them to Redis before you flush
from the *other* replica:

```bash
docker compose -f deploy/ha-local/docker-compose.yml \
               -f deploy/ha-local/docker-compose.slowmock.yml up -d mock-upstream

pid=$(curl -s localhost:8092/v1/proxy | python3 -c 'import sys,json;print(json.load(sys.stdin)[0]["id"])')
# fire 4 at gw1 (:8091) in the background: 2 go in-flight, 2 queue
seq 4 | xargs -P4 -I{} curl -s -o /dev/null -w "req {} -> %{http_code}\n" \
  localhost:8091/proxy/$pid/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"mock","messages":[{"role":"user","content":"hi"}]}' &
sleep 5   # let the sync loop mirror the queued entries to Redis
# gw2 sees gw1's queue (cross-replica global registry):
curl -s localhost:8092/v1/proxy/$pid   # -> inflight:2, queued:2
# flush from gw2 -> cancels the 2 queued via proxy:cancel pub/sub -> clients get 499
curl -s -X POST localhost:8092/v1/proxy/$pid/flush   # -> {"ok":true,"flushed":2}
```

The 2 queued requests return **HTTP 499**; the 2 in-flight complete **200**.

> The proxy management API needs an admin. `AUTH_DISABLED=1` maps anonymous to a
> seeded admin, so the compose sets `ADMIN_USERNAME`/`ADMIN_PASSWORD=admin` — a
> fresh DB has no admin otherwise and create/flush 500 with a clear message.
>
> First request(s) may lag if they land before the per-replica
> `proxy_health_loop` first probes the upstream — it's treated as alive once probed.

## Teardown

```bash
docker compose -f deploy/ha-local/docker-compose.yml down -v
```

## Iterating on HA *code* (faster than rebuilding the image)

This compose builds the gateway **image**, so backend edits need `--build`. To
iterate on your working tree instead, run 2 gateway **processes** against your
existing local Redis+Postgres — same env, no image:

```bash
# terminal 1
GATEWAY_HA=1 PROXY_CLUSTER=1 LEADER_TTL_S=6 LEADER_RENEW_S=2 \
  GATEWAY_BIND=127.0.0.1:8081 .venv/bin/gateway
# terminal 2
GATEWAY_HA=1 PROXY_CLUSTER=1 LEADER_TTL_S=6 LEADER_RENEW_S=2 \
  GATEWAY_BIND=127.0.0.1:8082 .venv/bin/gateway
# then: curl 127.0.0.1:8081/leader ; curl 127.0.0.1:8082/leader ; kill one, watch the other promote
```

(Stop any single `.venv/bin/gateway` you already have running first — a plain
one has `GATEWAY_HA` unset, so it's an unconditional leader and would run
controllers alongside the elected one.)
