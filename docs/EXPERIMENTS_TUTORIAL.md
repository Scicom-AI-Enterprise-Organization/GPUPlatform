# Experiments — a walkthrough

Everything in this guide was run against a local gateway before it was written;
the numbers and transcripts below are real output, not illustrations.

**What Experiments is for:** you have a served model and a question like *did
this finetune get worse?*, *does this prompt leak control tokens?*, *is the new
checkpoint slower?*. Answering it by hand means writing the same script again —
load some captured requests, replay them across a couple of endpoints, eyeball
the replies, tally. Experiments is that script, with the eyeballing replaced by
detectors.

---

## The mental model

Five nouns. Everything else is detail.

| Noun | What it is | Where it lives |
|---|---|---|
| **Dataset** | the requests you replay — rows with a `messages` column | `/datasets` (**not** an Experiments-owned store) |
| **Target** | an endpoint under test: `{base_url, model, key}` | typed into the run form |
| **Variant** | a mutation of every request — a different system prompt, params, no tools | typed into the run form |
| **Evaluator** | a check that scores each reply | built-in, or yours at `/experiments/evaluators` |
| **Sandbox** | answers the model's *tool calls*, so a row becomes a conversation | yours at `/experiments/sandboxes` |

One run = **dataset × targets × variants × repeats**. Each combination is a
**cell**, and a cell is a row in the results matrix. That's the whole shape:

```
row ──► request ──► reply ──► detectors ──► one sample
                                              └── pooled per (target, variant) = a cell
```

The thing to hold onto: **a target is always just `{base_url, model, key}`**.
There is no "platform endpoint" special case — your own app, a proxy endpoint, a
third-party API and a mock server are all equally first-class. That's why the
walkthrough below can run for free.

---

## Run one in five minutes, for free

You need no GPU and no pod. Two throwaway servers stand in for a model and a
tool backend.

### 1. Start a fake model

```bash
.venv/bin/python sandbox/mock_model.py       # → http://127.0.0.1:8078/v1
```

It serves seven models, each broken in a way a *different* detector catches —
so your first run shows real red and green instead of a wall of passes:

| model | what it does | detector that catches it |
|---|---|---|
| `good` | clean replies | — (all pass) |
| `leaky` | emits `<\|channel\|>` markers | `control_token_leak` |
| `empty` | returns `""`, 0 tokens | `empty_response` |
| `loopy` | repeats one phrase 40× | `degeneration` |
| `fenced` | wraps JSON in a ``` fence | `json_output` |
| `flaky` | HTTP 500 on ~1 prompt in 3 | `request_error` (always on) |
| `toolish` | calls a tool, then answers | used for the sandbox run below |

### 2. Create the experiment

`/experiments/new`, or the API:

```bash
curl -X POST http://127.0.0.1:8080/v1/experiments \
  -H 'Content-Type: application/json' -d '{
  "name": "tutorial-first-run",
  "dataset_id": "ds-f0a4aab1",
  "targets": [
    {"label": "good",  "base_url": "http://127.0.0.1:8078/v1", "model": "good"},
    {"label": "leaky", "base_url": "http://127.0.0.1:8078/v1", "model": "leaky"},
    {"label": "loopy", "base_url": "http://127.0.0.1:8078/v1", "model": "loopy"}
  ],
  "evaluators": [
    {"id": "control_token_leak", "options": {}},
    {"id": "empty_response",     "options": {}},
    {"id": "degeneration",       "options": {}}
  ],
  "repeats": 1, "concurrency": 4, "max_rows": 6
}'
```

6 rows × 3 targets × 1 variant × 1 repeat = **18 units**.

### 3. Read the matrix

That run, verbatim:

```
target     pass   err  failing detectors
good       100%    0%  —
leaky        0%    0%  control_token_leak
loopy        0%    0%  degeneration

totals: 18 samples, pass_rate=0.3333
```

That's the output the whole feature exists to produce: **one row per cell, and a
column per detector**, so "which endpoint is worse, and *how*" is one glance.
The Tradeoff tab draws the same cells as parallel coordinates when you have more
axes than fit in a table.

---

## The parts, in the order you'll meet them

### Datasets — Experiments owns no corpus

The picker lists `/datasets` filtered to chat-shaped kinds **with a messages
column mapped**. A dataset that shows as unusable tells you why (usually: no
`messages_field`).

Per row, optionally:

- `tools` / `functions` → the tool declarations sent with the request
- `params` (or flat `temperature`, `top_k`, …) → replayed sampling parameters,
  because replaying with library defaults reproduces a *different* request than
  the one that misbehaved
- `expected` → the reference data evaluators grade against

No corpus yet? Two ways in, both producing an ordinary dataset:

- **Capture** (button on the run form) — from a Langfuse trace, or from this
  platform's own served traffic. ⚠ Proxy-mode traffic **can't** be captured: those
  rows are stored deliberately slim (model + usage only), so there's no body to
  replay.
- **Generate** — `/datasets/new` → "Generate (synthetic)". This is where red-team
  corpora come from: nobody has a log of the attacks nobody has tried yet.

### Targets — one per thing you're comparing

`{label, base_url, model, key}`. The label names the row in the matrix. Keys are
global-secret references or inline (encrypted at rest, never returned by the API
— which is why cloning a run re-asks for them).

⚠ **`base_url` already containing `/v1` is handled** — `…/proxy/foo/v1` plus the
default path won't become `/v1/v1/chat/completions`.

### Variants — the other axis

A variant mutates every request the same way, so `targets × variants` is a grid:

| field | effect |
|---|---|
| `system_override` | **replaces** the row's system message |
| `system_prefix` / `system_suffix` | decorate it instead |
| `user_suffix` | appended to the last user turn |
| `assistant_prefill` | seeds the assistant turn ("force prefill") |
| `params` | override sampling per variant |
| `strip_tools` | send the request with no tools |
| `response_format` | `json_object`, or a full schema object |

Comparing two prompts = one target, two variants. Comparing two checkpoints =
two targets, one variant. Both at once = a 2×2 grid of cells.

### Evaluators — the scoring

Built-ins cover the failures that show up on nearly every endpoint:
`control_token_leak`, `empty_response`, `degeneration`, `json_output`,
`structure_tags`, `regex`, `tool_calls`, `finish_length`, `latency`, `cost`,
plus `red_team`, `function_call_units`, `multilingual_units`, and `llm_judge`.

`request_error` is **always on** — a failed HTTP call must never be invisible.

Write your own at `/experiments/evaluators/new`, in three modes:

- **expression** — one Python expression (`re_search("```", content)`), AST-whitelisted
- **api** — POST the completion to a scorer you already run; nothing executes on the gateway
- **python** — a real `def check(c)`, admin-only, subprocess-isolated

⚠ **Use the Test button.** `fail_when_true` silently inverts every result if you
pick it wrong, and an evaluator that throws is reported as a non-failure — both
are invisible until a whole run has been spent.

### Sandboxes — when one request per row isn't enough

Without a sandbox, each row is **one request**. That's fine for "did the reply
leak a control token", and wrong for "did the agent handle this conversation" —
which needs the model to call tools, get results, and continue.

A sandbox answers those tool calls. Two modes work today:

- **replay** — answers from a `tool_seed` on the dataset row (free, offline,
  totally reproducible)
- **api** — POSTs each call to a service you run

Try it with the second throwaway server:

```bash
.venv/bin/python sandbox/server.py           # → http://127.0.0.1:8077
```

Register it (`/experiments/sandboxes/new`, or the API):

```bash
curl -X POST http://127.0.0.1:8080/v1/custom-sandboxes \
  -H 'Content-Type: application/json' -d '{
  "name": "local-care-api",
  "mode": "api",
  "config": {
    "api":  {"url": "http://127.0.0.1:8077/tool", "response_field": "content"},
    "loop": {"max_tool_rounds": 4, "force_final": true}
  }
}'
```

⚠ **The config nesting is the thing people get wrong**: mode settings under
`config.api`, loop policy under `config.loop`. A flat `{"url": …}` is ignored and
every call comes back `sandbox_bad_response`. (The form writes the nesting for
you.)

Then add `"sandbox": {"id": "sb-…"}` to the experiment. A real trajectory from
that run:

```
assistant → get_balance({"account_id": "1001"})
tool      ← [api] {"balance_myr": 128.4, "due_date": "2026-08-22", "plan": "Fibre 500"}
assistant   Your outstanding balance is RM 128.40, due 22 Aug 2026.
```

and the per-cell rollup it produced:

```
trajectories 3 · rounds_mean 1.0 · tool_calls 3
provenance {api: 2, cache: 1} · novel_call_rate 0.67 · forced_final_rate 0.0
```

`cache: 1` is the response cache doing its job — within a run the same
`(name, arguments)` always returns the same answer, or two variants would face
different worlds and the comparison would measure simulator luck instead of
model quality.

---

## Reading a result without fooling yourself

This is the part that matters, and it's why several numbers exist that look
redundant.

**1. A green pass rate can mean nothing was scored.** Detectors that need
reference data (`function_call_units` needs `expected.tool_calls`,
`multilingual_units` needs `expected.language`, `red_team` needs
`expected.attack`) **skip** a row that lacks it rather than inventing a verdict —
and a skip counts as a pass. A dataset with no `expected` column therefore
reports a clean 100% having compared nothing.

> Always check **`scored`** against the row count before believing a result.
> `scored: 0` with a green rate means the corpus carries no reference data.

**2. Corpus metrics aren't averages.** An F1 or a per-language accuracy can't be
recovered by averaging per-sample rates, so those detectors pool raw counts
across the cell and report under `metrics` (rendered as their own block, not as
table columns).

**3. With a sandbox, three more numbers decide whether the run is real** — a
broken sandbox still produces trajectories, and the detectors score them happily:

- `novel_call_rate` high → the model called tools your seed doesn't cover; you
  measured seed coverage, not the model
- `forced_final_rate` high → most rows hit the round limit; you measured the limit
- `all_errors` → every tool call failed; treat it as a failed run, not a 0%

**4. Errors are data.** `retries` defaults to **1 — no retry, deliberately**.
Retrying masks the failure you're measuring: a fast 500 or a 0-token reply *is*
the finding.

**5. Guardrails can take credit for the model.** Replaying through a red-teamed
proxy, the guard's canned block matches every refusal pattern — so read
`model_refusal_rate`, not `refusal_rate`. "The endpoint is safe" and "the model
is safe" are different claims; only the second survives turning the guard off.

**6. Don't let a model grade itself.** Pointing `llm_judge` at the model that
produced your preference labels flatters it by construction.

---

## Cost, and how it's bounded

Two ceilings, both enforced at create *and* re-checked at run time:

- **units** — rows × targets × variants × repeats (default cap 20,000)
- **billed calls** — units × (1 + `max_tool_rounds`) (default cap 60,000)

The second exists because a sandboxed row stopped being one request. At 6 tool
rounds, 20k units is up to 140k calls — the unit cap alone no longer bounds
spend. The run form prices this before you submit and blocks over the cap, with
a **Fit to cap** action that drops repeats to 1 and samples rows while leaving
your targets and variants alone (those are the comparison you came for).

Defaults follow the dataset's size, and that's load-bearing: a captured trace of
a few rows defaults to **20 repeats** (hunting an intermittent failure), while a
corpus defaults to **1 pass over a sample** (sweeping breadth). Guessing wrong is
how a 19k-row dataset becomes 380k requests.

Cancel is cooperative — in-flight units finish, queued ones are skipped, and the
partial results are still summarized.

---

## Optimize (GEPA) — the sibling tab

`/experiments/optimize` searches for a better **system prompt**: replay the
dataset under a candidate, score every reply with the *same* evaluators, show
the failures and their written critiques to a reflection model, keep a rewrite
only if it measurably wins, and sample the next parent from a Pareto frontier so
the search doesn't collapse onto one strategy.

Two things to know:

- **The evaluators are the metric and their `reason` strings are the feedback.**
  Every evaluator you can select in a run is a GEPA feedback source with no extra
  authoring.
- **The winner is just a variant.** A finished search links to
  `/experiments/new?prompt=opt-…`, which opens the run form with **two** variants —
  baseline and optimized — on the same dataset. GEPA's own number comes from a
  validation slice; that run is what confirms it on the whole corpus.

⚠ **Optimize cannot use a sandbox.** Its rollouts are single-shot completions —
`prompt_opt_api` only ever calls `EvaluatorStack.evaluate()`, never the
trajectory path. That's deliberate, for two reasons. The scoring path is shared
with experiments precisely so an optimized prompt's reported gain reproduces in
the experiment meant to confirm it, and a silent divergence there is the failure
mode the sharing was built to prevent. And the budget is denominated in real
billed calls: a sandboxed rollout would be up to `max_tool_rounds + 1` calls, so
a search would quietly cost ~7× its estimate. Optimizing a prompt for an agentic
loop means optimizing against single-turn rollouts today, then confirming with a
sandboxed experiment.

---

## Where things live

| Route | What |
|---|---|
| `/experiments` | run list |
| `/experiments/new` | create a run (Dataset · Targets · Variants · Evaluators · Sandbox · Run) |
| `/experiments/{id}` | Overview · Tradeoff · Samples · Config |
| `/experiments/evaluators` · `/new` | your evaluator library |
| `/experiments/sandboxes` · `/new` | your sandbox library |
| `/experiments/optimize` · `/new` | GEPA searches |

API: `POST /v1/experiments`, `GET /v1/experiments/{id}`,
`GET /v1/experiments/{id}/samples`, `POST /v1/custom-evaluators`,
`POST /v1/custom-sandboxes`, and the two dry-run routes
`POST /v1/custom-{evaluators,sandboxes}/test`.

Everything is owner-labelled but **not owner-scoped** — anyone with the
Experiments section sees every run, evaluator and sandbox.

---

## Cheat sheet

```bash
# free local rig
.venv/bin/python sandbox/mock_model.py    # fake model      → :8078/v1
.venv/bin/python sandbox/server.py        # fake tool API   → :8077

# what can I replay?
curl -s localhost:8080/v1/experiments/datasets | jq '.[] | select(.usable)'

# what can score it?
curl -s localhost:8080/v1/experiments/evaluators | jq '.evaluators[].id'

# what are the ceilings?
curl -s localhost:8080/v1/experiments/limits
```

Before believing any result: **check `scored` against the row count**, check
`error_rate`, and — if you used a sandbox — check `novel_call_rate` and
`forced_final_rate`.
