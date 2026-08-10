# Turn detector (LiveKit EOU) through the LLM proxy

How to serve a LiveKit **turn detector** (end-of-utterance prediction) through the
GPUPlatform LLM proxy, and how to verify it by hand.

A LiveKit voice agent decides when the user has stopped talking by asking a small model
one question: *how likely is the end-of-turn token right now?* With
`LIVEKIT_REMOTE_EOT_URL` set, it POSTs that question to an HTTP endpoint instead of
running the model in-process. This document covers pointing that URL at the proxy.

> **The agent must point at the PROXY, never at the engine directly.** The proxy is what
> holds the engine's credential and gives you failover, history and metrics.

---

## 1. The request

The client sends **raw** `/v1/completions` — *not* `/v1/chat/completions`:

```json
{
  "model": "livekit/turn-detector",
  "prompt": "<pre-rendered ChatML string>",
  "max_tokens": 1,
  "logprobs": 1,
  "allowed_token_ids": [151645]
}
```

`151645` is `<|im_end|>` for Qwen-family detectors. `allowed_token_ids` and `logprobs`
are vLLM-specific and **must reach the backend**.

The client reads exactly one value back — `choices[0].logprobs.token_logprobs[0]` — and
converts it with `exp()` to a probability.

**Why `allowed_token_ids` matters.** With `logprobs: 1` you only get the logprob of the
token the model actually emitted. Mid-sentence the model emits the *next word*, so
`<|im_end|>` would never appear in the response. Forcing it to be sampled is what
guarantees its logprob lands in `token_logprobs[0]`. Nothing is generated — `text` comes
back empty, which is correct.

**If the client can't parse the response it falls back to `1.0`** — "the user has
finished talking". A broken pipeline therefore does not error; it makes the agent talk
over people. Every check below exists because of that.

---

## 2. Registering the backend

Routing lives in Postgres (`proxy_endpoints`), not in code. No code change is needed —
the proxy forwards request bodies verbatim (`body = {**payload, **extra_body, "model":
…}`) and returns `r.json()` untouched, so `allowed_token_ids` / `logprobs` survive in
both directions.

Store the engine key as a global secret, then register the endpoint:

```bash
curl -X PUT "$GW/v1/global-env/TURN_DETECTOR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"value":"<engine key>","is_secret":true}'

curl -X POST "$GW/v1/proxy" -H "Content-Type: application/json" -d '{
  "name": "turn-detector",
  "timeout_s": 30,
  "upstreams": [{
    "name": "tm-l40",
    "base_url": "https://turn-detector-engine-tm-l40.aies.scicom.dev/v1",
    "api_key_secret": "TURN_DETECTOR_API_KEY",
    "models": {"livekit/turn-detector": "livekit/turn-detector"}
  }]
}'
```

⚠️ **`base_url` must end in `/v1`.** The data-plane routes append `/completions` (not
`/v1/completions`). Omit it and the ENGINE returns `{"detail":"Not Found"}`, which reads
exactly like a gateway routing bug. The `x-upstream-url` response header tells the two
apart: present = the request reached an upstream, so the 404 came from the engine.

⚠️ **Never enable red-teaming on this endpoint.** The guard covers `/completions`
(`_RT_GUARDED_PATHS`) and returns a blocked body with `logprobs: null` — which the client
reads as the 1.0 fallback. It is off by default; leave it off.

The agent's env var is then — **base only, no `/v1/completions`**:

```
LIVEKIT_REMOTE_EOT_URL=<gateway>/proxy/turn-detector
```

The client appends the path itself (`return f"{url_base}/v1/completions"` in
`multilingual.py`), so including it here produces `…/v1/completions/v1/completions` and a
404. The resolved URL must be `<gateway>/proxy/turn-detector/v1/completions` — NOT the
top-level `/v1/completions`, which resolves `model` to a *serverless App* and cannot reach
an external host.

⚠️ **The client's remote-inference timeout is ~2 s** (`REMOTE_INFERENCE_TIMEOUT`), and any
exception — including a timeout — returns `1.0` ("user finished talking"). The proxy hop
must stay well inside that budget; a slow upstream degrades to constant interruption
rather than to an error. Local measurements: ~135 ms.

---

## 3. Manual testing — browser

Two places, for two different jobs.

**Quick check — the upstream editor's `Turn` probe.** In the backend editor (routing graph
→ a node → *Editing backend*), the Test row has a **Turn** button, preselected whenever the
model name looks like a turn detector. It scores an unfinished utterance and its finished
form and asserts all three things that matter:

```
turn ok (livekit/turn-detector): p(EOU) 3.63e-09 → 6.38e-04 (175,507x), forced '<|im_end|>' over ' you'
```

It FAILS (rather than reporting a cheerful 200) when logprobs are missing, when the sampled
token was already the model's own top choice — which would mean `allowed_token_ids` cannot
be confirmed as applied — or when the two utterances don't separate. Note the other probes
(Chat/Embedding/Rerank/Transcribe/TTS) hit different endpoints and never send
`allowed_token_ids` or `logprobs`, so a green **Chat** proves connectivity only.

**Exploring — the playground.** *Playground tab → mode `Turn detector`*, the only mode on
the raw completions path. Type your own utterances, one per line:

| Do this | Expect |
|---|---|
| Click **Predict** on the defaults | Finished and unfinished utterances separated by orders of magnitude |
| Read the `model wanted …` line | On unfinished text: sampled `<|im_end|>` while the model wanted the next word — this is `allowed_token_ids` being enforced |
| Paste the same utterance 3× | 🟠 amber "scores barely differ" (spread ≈ 0) |
| Leave one utterance only | No spread warning — one line can't show separation |
| Set **EOU token id** to `13` | Works; scores a different token (`.`) — the id is configurable |
| Set **EOU token id** to `9999999` | 🔴 red error from vLLM: *out-of-vocab token id* |
| Emoji, bare text, 2k-char input | All score normally |

Empty and whitespace-only lines are dropped before sending.

**Reading a result.** Judge by *separation*, not by whether anything is displayed:

- `text` empty → **correct** (nothing is meant to be generated)
- tiny `p` like `3.6e-09` → **correct**, a confident "still talking"
- 🔴 "no logprobs" / "BLOCKED by red-teaming guard" → real failure, client falls back to 1.0
- 🟠 "scores barely differ" → prompt template probably doesn't match the client's

---

## 4. Manual testing — terminal

The playground always sends a well-formed request, so the two silent failure modes can
only be reproduced by hand.

```bash
GW=http://localhost:8080
P=$GW/proxy/turn-detector/v1/completions
B='{"model":"livekit/turn-detector","prompt":"<|im_start|>user\nhello how are","max_tokens":1,"logprobs":1,"allowed_token_ids":[151645]}'

# healthy: empty text, forced token, real logprob
curl -s $P -H 'Content-Type: application/json' -d "$B" | jq '.choices[0]'

# FAILURE SIGNATURE — allowed_token_ids stripped.
# Returns text " you" and p≈0.97 mid-sentence: the agent would interrupt constantly.
curl -s $P -H 'Content-Type: application/json' \
  -d '{"model":"livekit/turn-detector","prompt":"<|im_start|>user\nhello how are","max_tokens":1,"logprobs":1}' \
  | jq -c '{text:.choices[0].text, logprob:.choices[0].logprobs.token_logprobs[0]}'

# FAILURE SIGNATURE — logprobs omitted: logprobs comes back null.
curl -s $P -H 'Content-Type: application/json' \
  -d '{"model":"livekit/turn-detector","prompt":"hi","max_tokens":1,"allowed_token_ids":[151645]}' \
  | jq -c '.choices[0].logprobs'
```

---

## 5. Proving the fields really reach the engine

Three independent legs. Together they cover the full round trip.

**Leg 1 — the response is untouched.** Same prompt, direct vs through the proxy:

```bash
curl -s $ENGINE/v1/completions -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d "$B" | jq -c '.choices[0].logprobs.token_logprobs'
curl -s $P -H 'Content-Type: application/json' -d "$B" | jq -c '.choices[0].logprobs.token_logprobs'
# => identical arrays
```

**Leg 2 — the request physically arrives.** Send a deliberately invalid token id:

```bash
curl -s $P -H 'Content-Type: application/json' \
  -d '{"model":"livekit/turn-detector","prompt":"hi","max_tokens":1,"logprobs":1,"allowed_token_ids":[9999999]}'
# => "allowed_token_ids contains out-of-vocab token id! (parameter=allowed_token_ids, …)"
```

That message comes from **vLLM's own validator**. A validator cannot reject a field it
never received — if the proxy stripped it, this would return 200 with no error. This is
the strongest single check.

**Leg 3 — the constraint is enforced.** On an unfinished utterance the model's own argmax
is the next word, yet `<|im_end|>` is what gets sampled:

```
model wanted : " you"       at  -0.027
actually got : "<|im_end|>" at -19.433
```

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `{"detail":"Not Found"}` **with** `x-upstream-url` | Engine 404 — `base_url` missing `/v1` | Append `/v1` |
| `{"detail":"Not Found"}` **without** it | Gateway route miss | Use `/proxy/<name>/v1/completions` |
| `proxy endpoint '…' not found or disabled` | Wrong name, or disabled | Check name / `enabled` |
| `model '…' is not served by endpoint` | Alias missing from the models map | Add it as the alias (left side) |
| `Unauthorized` | Engine key wrong/absent | Re-check the global secret |
| `502 all upstreams failed` | Engine down | Curl the engine directly |
| `text` non-empty, token ≠ `<|im_end|>` | `allowed_token_ids` stripped | p≈0.97 → agent interrupts |
| `logprobs: null` + `finish_reason: content_filter` | Red-team guard blocked it | Disable red-teaming here |
| All utterances score the same | Template mismatch | Match the client's chat template |
| Agent interrupts constantly | p pinned at 1.0 (fallback) | See the two failure signatures above |
| Agent never stops listening | p always tiny | Threshold, or template mismatch |
| Works locally, not in prod | Endpoint is per-environment | Register it in prod; prod also needs `sgpu_…` auth |

---

## 7. Scope

Registering the backend is **all** that the proxy side requires — no gateway code change.
Pointing tm-voice-agent at the URL (setting `LIVEKIT_REMOTE_EOT_URL`, switching the agent
from local to remote) is separate work.

The prompt template used by the playground (`<|im_start|>user\n{text}`) is a
**placeholder**. Absolute p(EOU) only matches the agent when it matches the template
`predict_end_of_turn` renders in the client plugin; until then, trust the *separation*
between utterances rather than the absolute value.
