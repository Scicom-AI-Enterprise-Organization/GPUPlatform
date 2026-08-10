# Design — Sandboxes for Experiments (multi-turn tool replay)

**Status:** **stages 1–3 implemented** — backend (`mode=replay` + `mode=api`) and the full UI,
41 unit tests in `gateway/tests/unit/test_sandbox.py`. Stages 4–5 (autotrain tab, `mode=llm`,
cache persistence, `mode=python`) are still proposal.
**Scope:** `gateway/gateway/{sandbox,experiments_api,evaluators}.py`, `gateway/gateway/db.py`
(one idempotent ALTER), `gateway/gateway/training_api.py` (the autotrain hook),
`web/src/app/(app)/experiments/*`, `web/src/app/(app)/autotrain/[runId]/*`.

What exists today: the `custom_sandboxes` table + CRUD + `POST /v1/custom-sandboxes/test`,
`GET /v1/experiments/sandboxes` (server-driven mode descriptors), `sandbox.py` (spec, loop policy,
call identity, the replay + api providers, the run-scoped response cache), `ev.Trajectory` +
`turn_expected()` + three-valued `wants` + `run_evaluators_trajectory()`,
`experiments_api.run_trajectory()` (the loop), `EXPERIMENT_MAX_CALLS`,
`experiment_samples.trajectory_json`, the per-cell sandbox rollup in `summarize()`, and web-side:
the **Sandboxes** tab (list / editor / test-on-one-row), the Sandbox card + call-denominated
pricing on the run form, the trajectory viewer in Samples, and the sandbox-health table on
Overview. `sandbox=None` remains the default and that path is byte-identical to before.

## The problem

Experiments replays **one request per row**. A row is turned into a single chat completion, wrapped
in `ev.Completion`, and scored. That is enough for a single-turn detector and it is what
`root CLAUDE.md` already warns about for `function_call_units`:

> ⚠ Experiments replays ONE request per row; the function-call benchmark replays a whole
> conversation. […] Multi-turn replay is unimplemented — it needs the runner to thread tool results
> back into the request, which is the one genuinely missing piece.

An agentic evaluation cannot be expressed this way. To score "did the model handle this care
conversation correctly" you must let it **call tools, answer those calls, and continue** until it
produces a final reply — then judge the whole trajectory. Today that lives in out-of-tree harnesses
(`gateway/gateway/fc_eval.py`'s conversation replay, and the TM
`synthetic-generation/tm-text-assist-simulation/eval_testset.py`), each re-implementing the loop.

A **Sandbox** is the missing primitive: the thing that answers a model's tool call during a replay.
Adding it fixes the documented gap for everyone and removes the first-turn-only caveat from
`function_call_units` as a side effect — **provided the per-turn scoring rule below lands with the
runner** (§`Trajectory`), because a naive last-turn projection makes that metric worse, not better.

## Non-goals

- Not a general agent framework. A sandbox answers tool calls; it does not orchestrate the model.
- Not a scorer. A sandbox produces a trajectory; **evaluators score it** (existing concept, reused).
- No TM-specific or benchmark-specific logic in the gateway. TM ends up as one row in
  `custom_sandboxes` whose `api` endpoint lives in the TM repo.

## Shape: `CustomSandbox` is `CustomEvaluator`'s twin

`custom_evaluators` already solved "let users bring their own logic, safely, with three trust
levels, and don't let library edits rewrite finished runs". A sandbox is the same problem with a
different payload, so it copies that design rather than inventing one — including the invariant that
matters most here:

> The library entry is editable; an experiment **snapshots** it into its own config at create time,
> so editing it later can never change what an already-finished run meant.

That snapshot is also the answer to eval comparability: a sandbox's simulation prompt or mock URL is
part of the environment. Change it and old numbers are no longer comparable — the same rule the repo
already enforces for the fastText detector ("never compare two runs whose detector differs").

### Model (`experiments_api.py`, beside `CustomEvaluator`)

```python
class CustomSandbox(Base):
    """A user-defined tool-response provider, reusable across experiments.

    Mirrors CustomEvaluator, including the snapshot-at-create rule: an experiment copies the
    sandbox definition into its config, so editing the library entry can never change what a
    finished run meant. The definition hash lands on every sample for the same reason.
    """
    __tablename__ = "custom_sandboxes"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)          # sb-<hex8>
    name: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)
    # replay | api | llm | python   (see MODES below; same trust ladder as custom_eval)
    mode: Mapped[str] = mapped_column(String(16), default="replay",
                                      server_default="replay", nullable=False)
    # python mode only: `def respond(convo, call) -> str`.
    code: Mapped[str] = mapped_column(Text, default="", server_default="", nullable=False)
    # Mode config + loop policy. No secret is stored — `api_key_secret` NAMES a global secret.
    #   api:    {url, method, headers, auth_header, api_key_secret, response_field,
    #            timeout_s, concurrency, send_expected: false}
    #   llm:    {base_url, model, prompt, temperature, max_tokens, api_key_secret}
    #   replay: {seed_field: "tool_seed", unknown_call: "error" | "empty"}
    #   loop:   {max_tool_rounds: 6, force_final: true, parallel_tools: false,
    #            trajectory_timeout_s: 600}
    config: Mapped[dict] = mapped_column(JSON, default=dict, server_default="{}", nullable=False)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
```

**No migration for this table.** `create_all` builds it with every column; the repo's
`ALTER TABLE … IF NOT EXISTS` statements in `db.init_db` exist only for columns added to a table
that already shipped. (An earlier draft of this doc showed an `ALTER` adding `config` to
`custom_sandboxes` — it is a no-op and would teach the wrong pattern. The one real ALTER this
feature needs is on `experiment_samples`, in §Storage.)

### Pydantic + routes (mirror `/custom-evaluators` exactly)

```python
class CustomSandboxSpec(BaseModel):
    name: str
    description: str = ""
    mode: str = "replay"
    code: str = ""
    config: dict[str, Any] = Field(default_factory=dict)

class CustomSandboxUpdate(BaseModel):   # every field Optional, PATCH semantics
    ...
class CustomSandboxRecord(BaseModel):   # id, name, description, mode, code, config, timestamps
    ...
```

| Route | Mirrors | Notes |
|---|---|---|
| `GET /v1/custom-sandboxes` | `list_custom_evaluators` | owner-scoped list |
| `POST /v1/custom-sandboxes` | `create_custom_evaluator` | `mode=python` re-checks admin role |
| `PATCH /v1/custom-sandboxes/{sb_id}` | … | never mutates finished experiments |
| `DELETE /v1/custom-sandboxes/{sb_id}` | … | snapshots keep old runs readable |
| `POST /v1/custom-sandboxes/test` | `/custom-evaluators/test` | **dry-run ONE row**, stream the trajectory back. The cost guard: you validate a sandbox on 1 row before spending 45 rollouts |
| `GET /v1/experiments/sandboxes` | `GET /v1/experiments/evaluators` | server-driven option descriptors, so the web editor renders new modes with no web change |

## The modes (same trust ladder as `custom_eval`)

| Mode | User supplies | Trust | Cost | Determinism |
|---|---|---|---|---|
| **`replay`** (default) | nothing | none — pure data | free | total |
| **`api`** | a URL | network only, via `netsafe.assert_safe_fetch_url` | user's own | theirs |
| **`llm`** | prompt + model | model key + budget | **per call** | cache-dependent |
| **`python`** | `def respond(convo, call)` | **admin only**, subprocess + rlimits + scrubbed env | free | **author's** — the child can read the clock, the network and the filesystem the gateway user can read |

- **`replay`** answers from the dataset row itself: a `tool_seed` field mapping
  `(name, canonical_args) → content`, harvested from the row's reference trajectory. A call that
  isn't in the seed gets a deterministic gateway error (`unknown_function` / `no_fixture`), never a
  fabricated success. This is the mode to ship first: free, offline, reproducible, and it already
  matches the TM test set's shape.
- **`api`** is what makes the feature general. `POST {conversation, tool_call, row}` →
  `{content}` (path configurable via `response_field`, like custom_eval's api mode). Any existing
  mock service, staging API, fixture server or simulated environment becomes a sandbox with **zero
  gateway code**. Same config vocabulary as custom_eval's api mode so the editor is a copy.
  ⚠ **`row.expected` is withheld unless `send_expected: true`.** That column holds the gold
  reference the evaluators grade against (`expected.tool_calls`, the reference reply). A simulator
  that can read it can return exactly the reference tool result — inflating the score with nothing
  in the trajectory showing why. Default it off, make the opt-in explicit in the editor, and stamp
  `provenance="api+expected"` when it's on so a drill-down can tell the two apart.
- **`llm`** simulates a plausible response. It is the only mode that spends money per tool call, so
  it must not ship before the response cache and a cost estimate in the UI.
- **`python`** exists for symmetry with custom_eval but `api` dominates it on both power and safety.
  Ship it last, or not at all. Inherit custom_eval's honest caveat verbatim: the subprocess bounds
  blast radius, it does not stop a determined author reading files the gateway user can read.

### Response caching is a contract, not an optimization

Within a run, the same `(name, canonical_args)` **must** return the same response — otherwise two
variants in one experiment face different worlds and the comparison measures simulator luck rather
than model quality. Canonicalize arguments by `json.dumps(…, sort_keys=True)` so key order doesn't
fork the cache.

⚠ **That rule has a direction, and it must be stated: the first caller to make a novel call defines
that call's answer for every other target, variant and repeat in the run.** So a comparison is
order-dependent whenever the variants call different tools — variant A's exploration builds the
world variant B is graded in. Two consequences:

- The cache entry records **which cell seeded it** (`seeded_by: "target/variant"`), and the
  drill-down shows it. A cell whose novel calls were mostly seeded by another cell is not an
  independent measurement.
- For a comparison that must be exact (checkpoint A vs B, base vs finetune), **freeze the cache
  first**: run one reference pass, persist its cache, and run the comparison with the sandbox in
  replay-from-cache mode. That is the same discipline as `detector: fasttext|builtin` — the
  environment is part of the measurement.

**Where the cache lives.** Not in `Experiment.config_json` — that blob is the snapshot and is
returned by every list/read of the row; a 200-row × 6-round cache is megabytes of JSON on a hot
path. Persist it as an **S3 artifact** beside the experiment (the section already writes artifacts
through `s3_put_file`), with `config_json.sandbox.cache_uri` pointing at it. `replay` gets
reproducibility for free and needs no artifact; `api` and `llm` need one.

## The `Trajectory` contract

The central compatibility decision, and the one the first draft got wrong. Evaluators take
`ev.Completion` (a single reply). The naive fix — project the trajectory to its **final** assistant
turn — silently breaks the metrics this feature exists to fix:

> ⚠ **The model's parsed tool calls are not a `Completion` field.** The runner side-channels them as
> `expected["_tool_calls"]`, with the request's own tools as `expected["_tools"]`
> (`evaluators.py:610,774`). The final assistant turn of a trajectory is the *text answer* — it has
> no tool calls. Project to it and `function_call_units` compares an empty model-call list against a
> non-empty reference on every row: `fn = len(ref)`, `tool_call_f1 → 0`. Today's caveat is "reads
> **higher** than the benchmark"; the naive adapter turns it into "reads catastrophically lower",
> and it looks exactly like a model regression.

The same projection hides mid-trajectory failures for every content detector: a model that
degenerated in round 2 and recovered in round 5 reads clean under `degeneration`,
`control_token_leak` and `empty_response`.

So the adapter is per-turn, and `EvaluatorSpec` gains a **three-valued** `wants`:

```python
@dataclass
class Trajectory:
    """One sandboxed replay: the model's turns plus the tool results it was given.

    `messages` is OpenAI-shaped (assistant turns with tool_calls, interleaved role=tool results).
    `provenance` is per tool response: "seed" | "api" | "api+expected" | "llm" | "python" | "error"
    — a trajectory mostly answered by simulation supports a weaker claim than one replayed from
    reference data, and the drill-down renders the difference.
    """
    messages: list[dict[str, Any]] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    rounds: int = 0
    forced_final: bool = False          # hit max_tool_rounds and was asked for a final answer
    tool_calls_total: int = 0
    novel_calls: int = 0                # calls not answered from seed/cache
    error: Optional[str] = None
    error_round: Optional[int] = None   # which turn aborted it
    latency_ms: Optional[int] = None    # wall-clock for the whole trajectory
    ttft_ms: Optional[int] = None       # FIRST turn only
    usage: Optional[dict[str, Any]] = None   # summed across turns — see the caveat below
    expected: dict[str, Any] = field(default_factory=dict)

    # One Completion per ASSISTANT turn, each carrying that turn's own content, reasoning,
    # finish_reason, usage, and — critically — its `expected._tool_calls` / `expected._tools`.
    turns: list["Completion"] = field(default_factory=list)

    def as_completion(self) -> "Completion":
        """The FINAL assistant turn, plus trajectory-level latency/usage. What `wants="completion"`
        evaluators see, so an author who never thought about multi-turn gets today's meaning."""
```

| `wants` | Gets | Fold | Set on |
|---|---|---|---|
| `"completion"` (default) | `as_completion()` — final turn + trajectory latency/usage | n/a | everything not listed below, incl. every existing custom evaluator |
| `"turn"` | one call per assistant turn | **any-fail**; worst turn's score; `flags.failed_turn` names the offender; every turn's flags feed `aggregate` via `flags.turn_flags` | `function_call_units`, `control_token_leak`, `degeneration` |
| `"trajectory"` | the `Trajectory` | n/a | nothing yet — for trajectory-native detectors (round count, tool-call ordering, recovery-after-error) |

Per-turn + the existing `aggregate` hook **is** the benchmark's own behaviour — the real
function-call harness scores every turn and pools tp/fp/fn across the conversation. So this isn't a
stopgap for the projection problem; it's what parity actually requires.

### ⚠ Two rules that only showed up once it was built

**1. A detector may declare `wants="turn"` only if it is meaningful on an assistant turn whose
content is empty because the model *only called tools*.** That turn is normal and frequent, and it
disqualifies more detectors than the first draft assumed: `empty_response` would flag every tool
turn as empty; `json_output`, `structure_tags` and `regex` fail on `""` by construction; and
`multilingual_units` / `red_team` are about the final reply to the user, not the intermediate
mechanics. `degeneration` is safe only because its `min_tokens` guard already refuses to judge a
short reply. The surviving set is the three in the table.

**2. The row's reference data describes ONE turn — the one the row is about.** Scoring every model
turn against `expected.tool_calls` would compare round 2 against round 1's gold answer and invent
failures. `ev.turn_expected()` therefore narrows: turn 0 sees the row's reference, later turns see
it **only** if the row carries `expected.turns[i]`, and otherwise get the reference keys stripped so
the detector abstains. `TURN_REFERENCE_KEYS` is that list. This is what keeps the platform's
standing rule intact one level up — a corpus with no per-turn references still reports `scored`
equal to the number of turns that genuinely had one.

A corollary: the fold publishes **no "turns scored" count**. Abstention vocabulary is per-detector
(`multilingual_units` says `flags.skipped`; `function_call_units` abstains implicitly, with neither
a reference nor a model call), so a generic count read 2-of-2 for a trajectory the function-call
detector actually scored once. The authoritative number stays the detector's own
`aggregate` → `metrics.scored`, which `summarize()` pools out of `turn_flags`.

Deliberately left at `"completion"`: `request_error`, `latency`, `cost`, `finish_length`. Their
trajectory-level values are the meaningful ones (wall-clock, summed spend), and folding them
per-turn by any-fail would be wrong (a per-turn `cost` outcome folded by "worst" is not the bill).

- `EvalOutcome` is unchanged — `flags` already carries drill-down detail.
- **`sandbox=None` (the default) is exactly today's behaviour**: one request, `Completion`, no
  trajectory, `wants` never consulted. This must stay the default so nothing existing changes.

⚠ **Summed `usage` changes what two stored numbers mean.** `completion_tokens` becomes "tokens
across N turns" and `prompt_tokens` **double-counts context** (turn N's prompt contains turn N−1's).
Cost is right to sum — it is the real bill — but `prompt_tokens_mean` on a cell stops being
comparable to a non-sandboxed run. Store `rounds` on the sample so the UI can normalize, and label
the column "billed tokens (all turns)" whenever a sandbox is set.

## Runner integration

One layer above the existing dispatch, in `experiments_api.py`:

```
row ──► [sandbox loop]                                  ──► EvaluatorStack.evaluate(traj)
        assistant → tool_calls? → provider.respond(…)         (unchanged concept; gains a
        └── repeat ≤ max_tool_rounds, then force_final          Trajectory-aware branch)
```

- The loop reuses the existing `_dispatch`-style call for each assistant turn, so streaming,
  reasoning normalization (`reasoning` vs `reasoning_content`), guard detection, latency and usage
  accounting all keep working per turn.
- **A trajectory needs its own deadline.** `timeout_s` (default 300) is per *request*; six rounds of
  it is a 30-minute row, and the run's heartbeat/cancel semantics have never been exercised at that
  duration. `loop.trajectory_timeout_s` (default 600) bounds the whole thing; hitting it is
  `error="trajectory_timeout"`, not a silent truncation.
- Concurrency: rows stay the unit of parallelism. A trajectory is N sequential model calls, so a
  sandboxed run is inherently slower — the planned-vs-completed counters already cover progress, but
  `n_planned` should mean rows, not requests, and the UI must say "rows".
- Failure semantics: a provider error becomes a `role=tool` message carrying a structured error
  (the model gets to react to it, which is realistic and worth scoring) while `provenance="error"`.
  A *transport* failure on a model turn aborts the trajectory and sets `Trajectory.error`, matching
  today's short-circuit where `comp.error` decides the verdict.
  ⚠ That short-circuit (`evaluators.py:1577`) throws away the rounds that DID succeed — a
  trajectory that died at round 5 scores as a total failure. That's the right call (a partial
  trajectory is not a partial answer), but it makes corpus F1 a function of endpoint flakiness, so
  the aggregate must report **`n_aborted` and the round histogram** separately.

### ⚠ `MAX_UNITS` stops bounding spend, and must be joined by a call budget

`EXPERIMENT_MAX_UNITS` (20k) counts units = rows × targets × variants × repeats
(`experiments_api.py:105-109`), and its docstring is explicitly about *billed calls* — "a 4-target ×
3-variant × 200-repeat × 20-case sweep is 48k real billed calls". With `max_tool_rounds: 6` one unit
becomes up to **7** calls, so the same cap silently permits ~140k. Add, mirroring what GEPA already
does (`PROMPT_OPT_MAX_METRIC_CALLS`, budget denominated in real billed calls, enforced *before* each
iteration so a run cannot overshoot what was approved):

- `EXPERIMENT_MAX_CALLS` — ceiling on `units × (max_tool_rounds + 1)`, checked at create AND
  re-checked at run time like `MAX_UNITS` is.
- The form prices it client-side with the same arithmetic and prints it above the submit bar. The
  Experiments UI CLAUDE.md already makes this a rule for the GEPA budget card: *"the form PRICES the
  run, and that number must match the server. Change the Python, change this."*
- `llm` mode multiplies again (one simulator call per tool call). Its estimate is a separate line.

### ⚠ `EvaluatorStack` is shared with GEPA — decide, don't drift

`EvaluatorStack` (`experiments_api.py:872-880`) was extracted so prompt optimization grades a
rollout with **exactly** the detectors an experiment would; the docstring says why: *"Two scoring
paths that drift apart would make an optimized prompt's reported gain unreproducible in the
experiment that's supposed to confirm it."* Adding a trajectory branch to `evaluate()` puts that at
risk, because `prompt_opt_api`'s rollouts are single-shot `call_once` completions.

- **Stages 2–4: refuse the combination.** `prompt_opt_api` rejects a `wants="trajectory"` evaluator
  at create time with a real message. A silent `as_completion()` fallback is the exact failure mode
  the extraction was meant to prevent.
- **Later: give GEPA the loop.** Its `evaluate(texts, row_ids)` already runs through
  `resolve_cases()` → `build_request()` → `call_once()`; the sandbox loop wraps the last of those.
  Note the budget interacts — a GEPA rollout would become up to `max_tool_rounds + 1` billed calls,
  so `resolve_budget()` needs the same multiplier or a search silently costs 7× its estimate.

## Storage — the one real migration

The UI renders a trajectory; nothing stores one today. `ExperimentSample` is deliberately narrow
with a capped text body (`experiments_api.py:199-201`) so a 20k-unit run stays "a few hundred MB
rather than multiple GB", and `EXPERIMENT_MAX_STORED_CHARS` is 8k per sample. A 6-round trajectory
with tool results is roughly an order of magnitude larger — the same run becomes multiple GB, which
is precisely what that column comment exists to prevent.

```python
# Sandbox trajectories: the multi-turn transcript for one replay. NULL for every
# non-sandboxed sample (the default), so existing rows are valid unchanged.
await conn.execute(text(
    "ALTER TABLE experiment_samples ADD COLUMN IF NOT EXISTS trajectory_json JSON"
))
```

- `EXPERIMENT_MAX_TRAJECTORY_CHARS` (default 32k) caps the stored transcript.
- **Truncate tool results first, model turns last** — the model's own output is the thing under
  test; a fixture's 40 KB JSON payload is not. Record `truncated: true` per message so the viewer
  says so instead of implying the model saw less than it did.
- The provenance list, round count and counters are cheap and always stored, even when the
  transcript is dropped: the aggregates below must survive truncation.

## Anti-silent-no-op

The repo's existing rule ("`scored: 0` with a green pass rate means the dataset carries no reference
data") applies doubly here, because a broken sandbox still produces trajectories and the detectors
will happily score them:

- Sandbox aggregates must report **`rows`, `scored`, `novel_calls`, `n_aborted`, and a provenance
  histogram**.
- A sandbox whose every call errored is a **failed run**, not a 100% pass.
- **`novel_call_rate` is a first-class warning.** In `replay` mode a high rate means the model is
  calling tools the seed doesn't cover — the run is measuring seed coverage, not the model. Warn
  above a threshold on the detail page, the same way the optimize page warns on
  `unscored_rollouts`.
- The detail view surfaces `forced_final` count — a corpus where most trajectories ran out of tool
  rounds is measuring the round limit, not the model.

## UI — Experiments

This is the larger half of the work, not a thin wrapper. Existing conventions do a lot of it for
free (server-driven options, DOM-discovered scrollspy sections, the `aggregate`→`BenchmarkMetrics`
path), but four things genuinely change.

| File | Copied from | Change |
|---|---|---|
| `experiments/sandboxes/page.tsx` | `evaluators/page.tsx` | new route |
| `experiments/sandboxes/sandboxes-manager.tsx` | `evaluators-manager.tsx` | list/create/edit/delete |
| `experiments/sandboxes/custom-sandbox-editor.tsx` | `custom-evaluator-editor.tsx` | mode switch + config form + **Test on one row** |
| `experiments/section-tabs.tsx` | — | add **Sandboxes** beside Runs · Optimize · Evaluators |
| `experiments/new/experiment-form.tsx` | `benchmark/new/benchmark-form.tsx` | one `SectionCard` with `data-form-section="Sandbox"`, placed between Targets and Evaluators |
| `experiments/new/experiment-form.tsx` | `optimize/new/optimize-form.tsx` | **the unit-count footer becomes call-denominated** — see below |
| `experiments/[id]/experiment-detail.tsx` | — | Samples tab renders a trajectory when present; Overview gains the provenance / `forced_final` / `novel_call_rate` / `n_aborted` warnings |
| `experiments/experiments-list.tsx` | — | a provenance badge + filter — see the autotrain section |

1. **The footer's arithmetic is now wrong, and it is load-bearing.** It reads
   `N requests will be sent — N dataset rows × targets × variants × repeats`, and `blocked ===
   "over-cap"` gates submit on it. With a sandbox each row is up to `max_tool_rounds + 1` calls, so
   both the copy and the cap need the multiplier — and per the section's own rule, that client
   arithmetic must match `EXPERIMENT_MAX_CALLS` on the server. `fitToCap()` keeps its current
   priority (drop repeats, then sample rows; never touch targets/variants — that's the comparison
   the user came for), and gains one more lever: lower `max_tool_rounds`.
2. **The Sandbox card needs no registry entry.** FormShell discovers sections from
   `data-form-section` and re-scans on mutation, so a conditional card appears in the scrollspy rail
   automatically.
3. **Options stay server-driven** (`specs_payload()` + `GET /v1/experiments/sandboxes`), so a new
   mode ships without a web change — same convention as the evaluator registry and quantization
   schemes. Don't hardcode mode ids in the editor.
4. **The Samples drill-down is where the value lands** — the model's trajectory beside the row's
   reference, tool calls aligned and marked matched / missed / extra, each tool result badged with
   its provenance, and `truncated` shown where the transcript was capped.

`web/src/lib/types.ts` gains `CustomSandboxRecord` + an optional `sandbox` on the experiment config
and `trajectory` on the sample record — keep in sync with the pydantic models, per the datasets-UI
rule.

## Integrating with autotrain

The question a training run cannot answer today: **did this finetune actually get better?** The
Metrics tab has loss curves and the trainer's own eval; neither compares the checkpoint against its
base model on a real corpus with real detectors. Experiments does exactly that, and there is no link
between the two sections today (no `run_id` anywhere in `experiments_api.py`, no experiments
reference in `training-detail.tsx`).

The visible piece is one new tab. The load-bearing pieces are the target, the lease and the link.

### 1. The target — a forward tunnel to the Try-it server

Experiment targets are always plain `{base_url, model, key}` (`experiments_api.py:263`), and a
checkpoint is not an endpoint. But **Try-it already serves the merged checkpoint on real vLLM** at
`_llm_tryit_port(run_id)` (`training_api.py:6289`) once `…/playground/start` has run. The gateway
just can't reach it over HTTP: `playground_chat` shells `curl -N` through one paramiko exec channel
per request (`training_api.py:7880`), which is fine for a chat box and useless at concurrency 8 with
no usage/TTFT accounting.

One call fixes that:

```python
local = vm_tunnel.ensure_forward(host, port, user, key, vm_port=_llm_tryit_port(run_id))
target = {"label": run_id, "base_url": f"http://127.0.0.1:{local}/v1", "model": run_id}
```

`ensure_forward` (`vm_tunnel.py:530`) is the same idempotent autossh `-L` that compute sessions and
proxy-mode endpoints already use, keyed by `(host, port, vm_host, vm_port)` so two runs on one box
don't collide. No new serving path, and the runner gets its normal streaming/usage/TTFT accounting.

The heavier alternative — deploy the checkpoint as a real serverless app or proxy endpoint and
target that — stays the right answer for a **recurring** regression suite, and the HF-export path
already feeds it. The tab uses the tunnel.

### 2. ⚠ The idle reaper will kill a long run mid-flight

`_reap_idle_tryit_pods` (`training_api.py:6259`) terminates a cloud Try-it pod on `expires_at`, and
that stamp is bumped only by playground traffic (`training_api.py:7844`). A 200-row experiment
outlives the window; the pod dies, every remaining row becomes a transport error, and since
`retries` is deliberately 1 those are force-failed — so an infrastructure timeout is reported as a
catastrophic model regression. Either bump `expires_at` from the runner's existing ~2s progress
flush, or take an explicit lease for the run and release it on finalize/cancel.

### 3. The link — two JSON fields, no migration

- Forward: `result_json["eval"] = {experiment_id, status, headline}`, merged by a `_set_eval_state`
  twin of `_set_label_export_state` (`training_api.py:979`) — the same pattern `tryit`,
  `hf_export` and `label_export` already use.
- Reverse: `Experiment.config_json["source"] = {"kind": "training_run", "run_id": …}`, filtered
  server-side with a JSON path exactly as `training_api.py:2850` already does on
  `result_json[...].as_string()`. That drives a provenance badge + filter on
  `experiments-list.tsx`; without it, auto-created eval runs swamp the list.

### 4. The tab

`training-detail.tsx:582-588`. Tabs are already URL-driven (`:157` + `tabHref`), so `?tab=evals`
deep-links and ⌘-clicks for free.

- Trigger gated like `canTryIt` (terminal + LLM task type), content in a new
  `[runId]/evals-tab.tsx` shaped like `hf-export-tab.tsx`.
- Pickers reuse the Experiments endpoints as-is: `GET /v1/experiments/datasets` (already filtered to
  chat-shaped datasets with a messages column), `GET /v1/experiments/evaluators`,
  `GET /v1/experiments/sandboxes`.
- **Compare against the base model by default.** Two targets — the checkpoint via the tunnel and
  `cfg["base_model"]` wherever it's served — is what makes this a tab rather than a link: the
  tradeoff matrix answers the question the loss curve structurally can't.
- **Sweeps**: `result_json.trials` → "evaluate best trial" (default) or "all trials as targets".
  Each trial is a separate Try-it load, so the form has to say what that costs.
- Render the linked experiment's headline cells inline (`summary.cells[].evals[id].metrics` +
  `headline` already exist) and link out to `/experiments/{id}`.
- **Seed the run form instead of duplicating it** where the user wants the full matrix:
  `/experiments/new?from_run=train-…` follows the two existing loop-back conventions
  (`?from=<exp-id>` clone, `?prompt=opt-…` from GEPA). Don't grow a second experiment form inside
  autotrain.

### 5. Auto-evaluate on finish

`_finalize` already carries this exact pattern at `training_api.py:2431`
(`if status == "done" and cfg.get("label_export")`). Add
`if status == "done" and cfg.get("eval_on_finish")` → load Try-it, create the experiment, stash the
id. Best-effort, never fails the run — the same rule the label/HF hooks follow.

### Two traps to encode in the form

1. **Evaluating on the training data.** Default the dataset picker to anything but
   `cfg["dataset_id"]`, and warn loudly when they match. The platform already has `test_split_pct`
   for carving eval, and the whisper multi-test-set lesson (a merged eval hid a 22.6 vs 2.8 WER gap)
   is the same failure one level up.
2. **Grading a checkpoint with its own teacher.** For a DPO'd run, `llm_judge` must not default to
   the model that produced the preference labels — that flatters it by construction. Warn (or
   refuse) when judge model == target model or == the run's base model.

**Not covered: ASR/TTS runs.** `TargetSpec.path` can pin an audio endpoint
(`experiments_api.py:272`) but there is no WER/CER evaluator in `SPECS`, so gate the tab to LLM
initially. Adding one — scored per test split, macro-selected — would make this the natural home for
the multi-test-set discipline that a merged eval already hid once.

## Migration / back-compat

1. New `custom_sandboxes` table (created by metadata, no ALTER) + **one** real ALTER adding
   `experiment_samples.trajectory_json`.
2. `EvaluatorSpec.wants` defaults to `"completion"`; `sandbox` defaults to `None` → existing
   experiments, configs and clones behave identically, and `wants` is never consulted without a
   sandbox.
3. `EXPERIMENT_MAX_CALLS` defaults to a value that cannot bind a non-sandboxed run (rounds = 0), so
   nothing existing starts failing the cap.
4. `web/src/lib/types.ts` gains `CustomSandboxRecord`, `config.sandbox`, `sample.trajectory` —
   keep in sync with the pydantic models, per the datasets-UI rule.

## Staging

1. ✅ **Done.** `CustomSandbox` model + CRUD + `/test`, **`mode=replay` only**. No network, no cost.
2. ✅ **Done.** `Trajectory` + `turns` + `as_completion()` + three-valued `wants` + the runner loop
   + `EXPERIMENT_MAX_CALLS`, `sandbox=None` default. `function_call_units` flipped to `wants="turn"`
   **in this stage, not later** — shipping the loop while it still scored the last turn only would
   post a metric far *worse* than today's first-turn-only number and read as a model regression.
   GEPA is untouched: `EvaluatorStack.evaluate_trajectory()` is a separate entry point, so prompt
   optimization still calls `evaluate()` and cannot drift.
3. ✅ **Done.** `mode=api` (with `send_expected` off by default) — the mode that makes it general —
   plus the whole UI in §UI: the Sandboxes tab, the run form's Sandbox card and call-denominated
   footer, the trajectory viewer, and the sandbox-health table.
4. The autotrain **Evaluate** tab: tunnel target, reaper lease, `result_json["eval"]`, base-model
   comparison. Independent of stage 5 — it is useful with no sandbox at all.
5. `mode=llm` + cache persistence (S3 artifact) + cost estimate. `mode=python` last, if ever.
   Trajectory-native evaluators and GEPA's own sandbox loop after that.

## Worked example: the TM care sandbox

Nothing TM-specific enters the gateway.

| Piece | Where it lives |
|---|---|
| tool schemas | dataset column `functions` (existing `tools_field`) |
| tool responses | `mode=api` → a small service in `synthetic-generation/tm-text-assist-simulation` wrapping its existing frozen-then-simulated `ToolWorld` |
| gold reference | the row's `expected` column (`expected.tool_calls` + reference text) — the same convention `function_call_units` already reads, and **not** sent to the sandbox unless `send_expected` is on |
| judge (Relevance/Accuracy/Tone) | the **existing** `llm_judge` evaluator with a custom `prompt`; no new code |
| tool-call F1 | `function_call_units` at `wants="turn"` (stage 2) |
| Indonesian rate, drafter-format compliance | `api`-mode custom evaluators against the same service, so the metric definitions stay versioned in the TM repo |

⚠ Whoever configures the judge must not point it at the model that produced the corpus's preference
labels — grading a DPO'd checkpoint with its own teacher flatters it by construction. Worth a warning
in the form when judge model == target model.
