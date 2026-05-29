# Multi-model VM fleet — curl runbook

Stand up several vLLM servers that **time-share one SSH VM's GPUs** (sleep/wake), then
drive them over the OpenAI-compatible API — entirely with `curl`. This is the exact
flow used to deploy the 5-model fleet on prod (`serverlessgpu.aies.scicom.dev`, VM
provider `TM-H20`, 8× NVIDIA H20-3e).

## 0. Setup

```bash
export SGPU_URL=https://serverlessgpu.aies.scicom.dev   # the gateway (NOT the web UI host)
export SGPU_API_KEY=sgpu_...                            # mint at /api-keys in the web UI
auth=(-H "Authorization: Bearer $SGPU_API_KEY")
```

> The web UI lives at `serverless.aies.scicom.dev`; the **gateway API** is the separate
> `serverlessgpu.aies.scicom.dev` host. Point curl / any OpenAI client at the gateway.

Sanity-check the gateway and your key:

```bash
curl -s "$SGPU_URL/health"      # {"ok":true}
curl -s "$SGPU_URL/ready"       # {"ok":true,"redis":"ok"}
curl -s "${auth[@]}" "$SGPU_URL/v1/providers"   # find your kind:"vm" provider id
curl -s "${auth[@]}" "$SGPU_URL/v1/storage"     # (optional) s3 backends
```

Grab the `id` of your `kind: "vm"` provider (here `prov-3318238b`) — you'll pin the
fleet to it. The provider already carries the VM's SSH host/user and GPU list.

## 1. Create the fleet — `POST /apps`

One app, `mode: "multi"`, with a `models[]` array. Each member is pinned to a
tensor-parallel slice; the gateway packs them onto `visible_devices` (here 6 GPUs →
`27b[0,1] 35B[2,3] 122B[0,1,2,3] Mistral[0,1,2,3] gemma[4,5]`). The 10 `env_vars`
are exported into **every** vLLM process (cache/HOME dirs on the VM's `/share` mount).

```bash
cat > /tmp/fleet.json <<'JSON'
{
  "name": "tm-fleet",
  "gpu": "VM",
  "gpu_count": 6,
  "mode": "multi",
  "provider_id": "prov-3318238b",
  "visible_devices": "0,1,2,3,4,5",
  "venv_path": "/share/vllm-venv",
  "vllm_version": "0.19.1",
  "sleep_level": 1,
  "env_vars": {
    "HOME": "/share/home",
    "XDG_CACHE_HOME": "/share/.cache",
    "TRITON_CACHE_DIR": "/share/triton_cache",
    "TORCHINDUCTOR_CACHE_DIR": "/share/torchinductor_cache",
    "FLASHINFER_WORKSPACE_DIR": "/share/flashinfer_cache",
    "HF_HOME": "/share/huggingface",
    "TRANSFORMERS_CACHE": "/share/huggingface",
    "VLLM_CACHE_ROOT": "/share/vllm_cache",
    "CUDA_CACHE_PATH": "/share/nv_cache",
    "NUMBA_CACHE_DIR": "/share/numba_cache"
  },
  "models": [
    { "model": "qwen/qwen3.6-27b", "tp": 2,
      "extra_args": "--max-model-len 262144 --reasoning-parser qwen3 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data --mm-processor-cache-type shm" },
    { "model": "Qwen/Qwen3.6-35B-A3B", "tp": 2,
      "extra_args": "--max-model-len 262144 --reasoning-parser qwen3 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data --mm-processor-cache-type shm" },
    { "model": "Qwen/Qwen3.5-122B-A10B", "tp": 4,
      "extra_args": "--max-model-len 262144 --reasoning-parser qwen3 --gpu-memory-utilization 0.80 --enable-auto-tool-choice --tool-call-parser qwen3_coder --mm-encoder-tp-mode data --mm-processor-cache-type shm" },
    { "model": "mistralai/Mistral-Small-4-119B-2603", "tp": 4,
      "extra_args": "--tool-call-parser mistral --enable-auto-tool-choice --gpu-memory-utilization 0.80 --reasoning-parser mistral" },
    { "model": "google/gemma-4-31b-it", "tp": 2,
      "extra_args": "--tool-call-parser gemma4 --enable-auto-tool-choice --gpu-memory-utilization 0.90 --reasoning-parser gemma4" }
  ]
}
JSON

curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/apps" --data @/tmp/fleet.json
# → {"app_id":"tm-fleet","url":"/run/tm-fleet"}
```

Field notes:
- `mode: "multi"` is what switches the app from single-endpoint to a fleet.
- `tp` = tensor-parallel size = GPUs that member occupies. `sum over a GPU` may exceed
  the physical count — that's the point: overlapping members **sleep/wake** to share.
- `sleep_level`: `1` = offload weights to CPU RAM (fast wake); `2` = discard weights
  (smaller footprint, slow wake = reload from disk).
- Two big models (`122B`, `Mistral 119B`) run at `--gpu-memory-utilization 0.80` so a
  resident model leaves headroom; the small ones at `0.90`.

## 2. Watch it load — `GET /apps/{id}/status`

```bash
curl -s "${auth[@]}" "$SGPU_URL/apps/tm-fleet/status" | python3 -m json.tool
```

Each member reports `state` (`launching` → `asleep`/`awake`, or `dead` with a `reason`),
its packed `gpus`, `tp`, `inflight`, and `port`. Non-overlapping members load
concurrently (wave-loading); overlapping ones serialize. First full load of 5 large
models is a few minutes. `workers: 1` means the VM worker registered.

## 3. List served models — `GET /v1/models` (public, no key)

```bash
curl -s "$SGPU_URL/v1/models" | python3 -m json.tool
```

Returns the OpenAI `{"object":"list","data":[...]}` shape with every fleet member —
so any OpenAI client can discover them without auth.

## 4. Request a model — `POST /v1/chat/completions`

The `model` field routes to (and wakes, if asleep) that exact member. A request to a
still-loading model returns `503 {"error":"warming_up"}`; a failed one returns
`503` with the dead `reason`. Just retry warming-up after a few seconds.

> **`model` = member id, not the endpoint name.** Single-model endpoints route by the
> *endpoint name* (`model: "my-endpoint"`) — but a fleet hosts many models under one
> endpoint, so each member is addressed by its **model id** (what `/v1/models` lists;
> `owned_by` there is the parent endpoint). Sending the endpoint name to a fleet is a
> deliberate `400`:
> ```json
> {"error":"this is a multi-model endpoint — set 'model' to one of its member models, not the endpoint name",
>  "endpoint":"tm-fleet","models":["qwen/qwen3.6-27b","Qwen/Qwen3.6-35B-A3B", ...]}
> ```

```bash
# Qwen / Gemma — support the enable_thinking chat-template kwarg
curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/v1/chat/completions" -d '{
    "model": "qwen/qwen3.6-27b",
    "messages": [{"role":"user","content":"Capital of France? One word."}],
    "max_tokens": 16,
    "chat_template_kwargs": {"enable_thinking": false}
  }'

# Mistral — DO NOT send chat_template_kwargs: Mistral tokenizers reject any
# chat_template ("chat_template is not supported for Mistral tokenizers" → 400).
curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/v1/chat/completions" -d '{
    "model": "mistralai/Mistral-Small-4-119B-2603",
    "messages": [{"role":"user","content":"Capital of France? One word."}],
    "max_tokens": 16
  }'
```

The other members route the same way: `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.5-122B-A10B`,
`google/gemma-4-31b-it` (model strings are case-sensitive — match `/v1/models`).

### Per-endpoint base URL (run many endpoints side by side)

The global `/v1/chat/completions` routes by the `model` field across **all** your
endpoints — which means a model id has to be globally unique to you (two endpoints
serving the same model → `409 ambiguous`). To deploy **multiple** inferences, address
each by its id in the path instead:

```
POST /{endpoint_id}/v1/chat/completions
POST /{endpoint_id}/v1/completions
POST /{endpoint_id}/v1/embeddings
GET  /{endpoint_id}/v1/models
```

The endpoint is fixed by the URL, so `model` only selects the **member** of that
fleet (and for a single-model endpoint it's ignored). Point any OpenAI client at
`base_url = $SGPU_URL/{endpoint_id}/v1`:

```bash
curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/tm-fleet/v1/chat/completions" -d '{
    "model": "qwen/qwen3.6-27b",
    "messages": [{"role":"user","content":"Capital of France? One word."}],
    "max_tokens": 16
  }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://serverlessgpu.aies.scicom.dev/tm-fleet/v1",
                api_key="sgpu_...")
client.chat.completions.create(model="qwen/qwen3.6-27b",
                               messages=[{"role":"user","content":"hi"}])
```

Resolution rules: unknown endpoint → `404`; a multi-model endpoint with no `model`
→ `400` (lists members); a `model` the endpoint doesn't serve → `404`.

## 5. Operate the fleet — `POST /apps/{id}/model-action`

Drive sleep/wake/lifecycle without redeploying. `action` ∈ `sleep | restart | kill`
(needs `model`), or `sleep_all` (no `model`):

```bash
# sleep one awake model (frees its GPUs for an overlapping one)
curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/apps/tm-fleet/model-action" \
  -d '{"model":"qwen/qwen3.6-27b","action":"sleep"}'

# restart / kill one member
curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/apps/tm-fleet/model-action" \
  -d '{"model":"google/gemma-4-31b-it","action":"restart"}'

# sleep every awake model at once
curl -s "${auth[@]}" -H "Content-Type: application/json" \
  -X POST "$SGPU_URL/apps/tm-fleet/model-action" -d '{"action":"sleep_all"}'
```

Commands are queued on the worker's next heartbeat (a second or two), then reflected in
`/status`. Waking is request-driven too: a `/v1/chat/completions` to an asleep model
sleeps whatever holds its GPUs (LRU) and wakes the target.

## 6. Per-model logs — `GET /apps/{id}/models/logs?model=`

```bash
curl -s "${auth[@]}" \
  "$SGPU_URL/apps/tm-fleet/models/logs?model=mistralai/Mistral-Small-4-119B-2603&tail=200" \
  | python3 -c 'import sys,json;[print(l) for l in json.load(sys.stdin)["lines"]]'
```

## 7. Tear down — `DELETE /apps/{id}`

Kills every vLLM process on the VM (incl. the renamed `VLLM::Worker_TP` tp-workers) and
deregisters the worker.

```bash
curl -s "${auth[@]}" -X DELETE "$SGPU_URL/apps/tm-fleet"
```
