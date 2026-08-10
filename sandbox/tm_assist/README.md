# TM text-assist sandbox

The 18 real TM agent tools, answered deterministically, so `/experiments` can
evaluate the TM text-assist models **automatically** instead of one turn at a time.

`sandbox/server.py` (the sibling) is a 6-tool fixture for checking the wiring.
This is the real world: `kbms_search` over 20k KB chunks harvested from the
simulation corpus, plus account / billing / order / case / assurance lookups.

```bash
.venv/bin/python sandbox/tm_assist/build_kb.py     # once → kb.json (~11 MB, gitignored)
.venv/bin/python sandbox/tm_assist/server.py       # → http://127.0.0.1:8078
```

Registered on this gateway as **`sb-07b3ac7d`** (`tm-text-assist`). To recreate:

| Field | Value |
|---|---|
| Mode | HTTP endpoint |
| Endpoint URL | `http://127.0.0.1:8078/tool` |
| Result path | `content` |

Verified through the platform's own `POST /v1/custom-sandboxes/test`.

## Why a sandbox is what makes this corpus evaluable

Experiments replays **one request per row**. On an agentic corpus that scores the
first turn only (root `CLAUDE.md` caveat 3) — the model emits a tool call and the
replay stops. A sandbox answers that call, so the trajectory continues: search the
KB, read what came back, answer from it.

It is also the only way the **grounding** metric means anything. "Did the model
assert something the tool results don't support?" needs real tool results to
compare against. That is why the KB is harvested from the corpus rather than
invented — see `build_kb.py`.

## Three properties the numbers depend on

**1. Deterministic.** Same `(tool, arguments)` → same answer, always. A random
simulator makes a comparison measure luck instead of model quality. Records derive
from `sha256` of the identifier, *not* `hash()` — Python salts string hashing per
process (`PYTHONHASHSEED`), so `hash()`-derived ids silently change on restart.
Verified: identical response hash across a server restart.

> ⚠️ The sibling `sandbox/server.py` has exactly this bug —
> `tool_create_ticket` uses `abs(hash(subject))`, so its ticket ids are *not*
> stable across restarts despite the file claiming determinism. Harmless for a
> wiring fixture, wrong for scoring.

**2. A frozen clock.** `--today` (default `2026-08-10`) anchors every date, so
`list_available_slots` returns the same 21-day window whenever you run it. With a
live clock, two runs a fortnight apart face different worlds and their numbers
aren't comparable.

**3. It refuses to invent customers.** An identifier is real only if it appears in
the conversation the model was given (matched on the corpus's documented formats:
`60…`/`…@unifi`, `1100234567`, `1-OR-30412345`, `CTT-2049581`, bare 7–9 digit
ids). Ask about an account nobody mentioned and you get the tool's documented
empty/`not_found` shape.

This one is load-bearing. A model that hallucinates an account number must not be
handed a plausible record — an evaluator downstream cannot tell the difference,
and the fabrication would score as a success. Formatting drift (added spaces or
dashes) is tolerated; invention is not.

```
get_subscriber_base_info(msisdn=60123456789), conversation never mentions it
  → {"error": "not_found", "reason": "no subscriber matches the supplied identifier"}
same call, conversation says "my line 60123456789 is down"
  → {"cust_id": "28082215", "acct_nbr": "1129482603", "subs_status": "A", …}
```

## The tools

Response shapes follow each tool's declared `returns` contract in the corpus's
`tm_text_assist_functions.json` — money in **sen**, `'A'|'S'|'T'` status codes,
letter order-state codes, and `dd/mm/yyyy HH:mm:ss` for field-force timestamps
(distinct from `YYYY-MM-DD HH:MM:SS` everywhere else).

| Stage | Tools |
|---|---|
| knowledge | `kbms_search` |
| identify/verify | `query_subscription_list` · `get_subscriber_base_info` |
| billing | `retrieve_billing_details` · `query_payment_records` |
| fulfilment | `query_customer_order_list` · `query_customer_order_detail` |
| case | `query_all_cases` · `query_case_detail` · `create_case` · `add_case_comments` · `cancel_case` · `close_case` |
| assurance | `get_ticket_info` · `list_available_slots` · `set_appointment` · `query_ntt_info` |
| account | `update_customer_contact` |

Write tools reject incomplete calls with
`{"error": "missing_required_parameters", "missing": [...]}` rather than
succeeding, so a model that drops required arguments — the exact regression
`quality_guards.missing_required_params` was written for — shows up in the
trajectory instead of vanishing.

`kbms_search` honours `top_k` (clamped 1–15, per the schema) and the `tags`
filter, and **is allowed to return nothing**: the corpus's own trajectories
contain `{"results": []}`, and how a model behaves on an empty KB hit is worth
measuring.

## Fault injection

| Tool name | Server does | Gateway reports |
|---|---|---|
| `_http_500` | HTTP 500 | `sandbox_http_error` |
| `_not_json` | 200, non-JSON body | `sandbox_bad_response` |
| `_wrong_key` | 200, JSON without the result path | `sandbox_bad_response` |
| `_slow` | sleeps 30s | `sandbox_unreachable` (set Timeout low) |
| `_empty` | 200, empty string | a valid, empty tool result |
| `_kb_outage` | `kbms_search`'s documented outage payload | a normal, empty tool result |

`_kb_outage` is deliberately **not** an error: the schema says the KB answers
`{"results": [], "message": "knowledge base is unavailable, answer from general
policy knowledge"}`, and whether the model actually falls back is a real
behaviour to test.

An unknown tool name returns `{"error": "unknown_tool", "known": [...]}` at HTTP
200 — an unknown call is something the **model** should react to, not a transport
failure, and a sandbox must never fabricate success for a call it can't answer.

## `send_expected` stays off

`row.expected` holds the gold reference the evaluators grade against. A simulator
that can read it can return exactly the reference result and inflate the score
with nothing in the trajectory showing why. The server logs whether it arrived:

```
→ kbms_search({"query": "…"})  [3 msg(s) · expected: withheld]
→ kbms_search({"query": "…"})  [3 msg(s) · expected: SENT ⚠]
```

## Options

```
--host 127.0.0.1   --port 8078
--kb PATH          default sandbox/tm_assist/kb.json
--today YYYY-MM-DD frozen clock (default 2026-08-10)
--api-key SECRET   require `Authorization: Bearer SECRET`
--quiet            don't print each call
```

`GET /health` reports the tool list, fault triggers, KB size and the frozen date;
`GET /docs` is the typed request contract.
