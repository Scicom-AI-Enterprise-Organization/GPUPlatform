# Test server for `api`-mode sandboxes

A stand-in tool-response service, so you can wire up
`/experiments/sandboxes/new` and watch the whole path work before pointing a
sandbox at anything real.

Background: a **sandbox** is the thing that answers a model's tool call during an
Experiments replay, which is what turns "one request per row" into a
conversation. `mode=api` POSTs each call to a service you run. This is that
service, in one FastAPI file — it uses the gateway venv, so there's nothing to
install.

```bash
.venv/bin/python sandbox/server.py     # → http://127.0.0.1:8077
```

`GET /docs` documents the request contract (`ToolRequest` is exactly what
`sandbox.ApiProvider.payload()` posts). ⚠ That model is deliberately permissive —
every field optional, extras allowed — because a 422 from request validation
would reach the gateway as `sandbox_http_error` and read like a broken endpoint
rather than a shape mismatch.

Then on **`/experiments/sandboxes/new`**:

| Field | Value |
|---|---|
| Mode | HTTP endpoint |
| Endpoint URL | `http://127.0.0.1:8077/tool` |
| Result path | `content` (or `result.output` to try a nested path) |

and in the **Test** card, put `get_balance` in *Tool* with
`{"account_id": "1001"}` as arguments, then press **Test**.

## What it answers

`get_balance` · `get_bill` · `get_usage` · `lookup_customer` · `list_plans` ·
`create_ticket`, over three fake accounts (`1001`, `1002`, `1003`).

Answers are **deterministic** on purpose. Within a run the gateway caches each
`(name, arguments)` so every variant faces the same world — a random simulator
would make the comparison measure luck instead of model quality, and this keeps
that true across runs too.

An unknown tool comes back as a normal 200 carrying a structured
`{"error": "unknown_tool", …}`. That's deliberate: an unknown call is something
the **model** should get to react to, not a transport failure — and a sandbox
must never fabricate a plausible success for a call it can't answer.

## Fault injection

Call these tool names to exercise the error paths, each of which the gateway
reports differently:

| Tool name | Server does | Gateway reports |
|---|---|---|
| `_http_500` | HTTP 500 | `sandbox_http_error` |
| `_not_json` | 200, non-JSON body | `sandbox_bad_response` (or succeeds if *Result path* is blank) |
| `_wrong_key` | 200, JSON without the configured path | `sandbox_bad_response` |
| `_slow` | sleeps 30s | `sandbox_unreachable` (set *Timeout* low) |
| `_empty` | 200, empty string | a valid, empty tool result |

All five verified end-to-end through `POST /v1/custom-sandboxes/test`.

## The `send_expected` toggle

The server prints, per call, whether the dataset row's `expected` block arrived:

```
→ get_bill({"account_id": "1001"})  [1 msg(s) of context · expected: withheld]
→ get_bill({"account_id": "1001"})  [1 msg(s) of context · expected: SENT ⚠]
```

`expected` holds the gold answer the evaluators grade against, so a simulator
that can read it can return exactly the reference result and inflate the score
with nothing in the trajectory showing why. It is **off by default** for that
reason — this line is how you confirm the toggle does what you think.

## Options

```
--host 127.0.0.1     --port 8077
--api-key SECRET     require `Authorization: Bearer SECRET`
                     (to exercise the form's API-key-secret field)
--quiet              don't print each call
```

`GET /health` lists the tools and fault triggers it knows; `GET /docs` is the
typed request contract.
