"""Sandboxes — the thing that answers a model's tool call during a replay.

Experiments replays ONE request per dataset row. That is enough for a single-turn
detector, but an agentic evaluation needs the model to **call tools, receive
results, and continue** until it produces a final reply. The missing primitive is
not the loop (that's ~80 lines in `experiments_api.run_trajectory`) — it's the
thing that answers the call. That's a Sandbox.

This module is the sibling of `custom_eval.py` and copies its shape deliberately:
a snapshot-able spec, a trust ladder of modes, an SSRF guard on anything that
leaves the box, and the rule that an author's mistake degrades a run rather than
silently corrupting it. Design notes in `docs/EXPERIMENTS_SANDBOX.md`.

Two things here are load-bearing and easy to get wrong:

- **The response cache is a contract, not an optimization.** Within one run the
  same `(name, canonical_args)` MUST return the same content, or two variants in
  one experiment face different worlds and the comparison measures simulator luck
  instead of model quality. That rule has a direction — the first cell to make a
  novel call defines the answer for every other cell — so the cache records who
  seeded each entry (`seeded_by`) and the drill-down can show it.
- **`replay` matches by NAME by default, not by exact arguments.** See
  `ReplayProvider` for why; requiring byte-identical arguments makes almost every
  call novel and the trajectory dies at round 1, measuring nothing.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from . import netsafe

logger = logging.getLogger("gateway.sandbox")

# The trust ladder, mirroring custom_eval.MODES. Only the modes in
# IMPLEMENTED_MODES can be saved or run — the rest are declared so the registry,
# the editor and the docs share one vocabulary as they land.
MODES = ("replay", "api", "llm", "python")
IMPLEMENTED_MODES = ("replay", "api")

# Loop policy defaults. `max_tool_rounds` is the one that costs money: every
# round is another billed model call for every row × target × variant × repeat.
DEFAULT_MAX_TOOL_ROUNDS = 6
MAX_TOOL_ROUNDS_CAP = int(os.environ.get("EXPERIMENT_MAX_TOOL_ROUNDS", "20") or "20")
# A trajectory needs its own deadline: `timeout_s` is per REQUEST, and six rounds
# of a 300s timeout is a 30-minute row.
DEFAULT_TRAJECTORY_TIMEOUT_S = 600.0
TRAJECTORY_TIMEOUT_CAP_S = 3600.0

# Deterministic errors handed back to the model as a tool result. Never a
# fabricated success: a sandbox that invents plausible output for a call it has
# no fixture for is scoring the simulator, not the model.
ERR_NO_FIXTURE = "no_fixture"
ERR_UNKNOWN_FUNCTION = "unknown_function"


class SandboxError(ValueError):
    """Author-facing configuration error (bad mode, bad seed field, bad URL)."""


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass
class SandboxSpec:
    """A resolved sandbox, snapshotted into the experiment config so a later edit
    to the library entry can't change what a finished run means — the same
    invariant `custom_eval.CustomSpec` carries, for the same reason. A sandbox is
    part of the ENVIRONMENT a model was measured in: change its seed field or its
    mock URL and old numbers stop being comparable, exactly like swapping the
    fastText detector for the built-in one."""
    id: str
    name: str
    mode: str = "replay"
    code: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SandboxSpec":
        return cls(
            id=str(d.get("id") or "sandbox"),
            name=str(d.get("name") or d.get("id") or "sandbox"),
            mode=str(d.get("mode") or "replay"),
            code=str(d.get("code") or ""),
            config=dict(d.get("config") or {}),
        )


def loop_config(cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The loop policy, clamped. Bad input is corrected rather than raised: this
    runs inside the runner, where a config that outlived a validation change must
    degrade, not kill a run."""
    raw = dict((cfg or {}).get("loop") or {})

    def _int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            val = int(raw.get(key, default))
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, val))

    try:
        timeout = float(raw.get("trajectory_timeout_s", DEFAULT_TRAJECTORY_TIMEOUT_S))
    except (TypeError, ValueError):
        timeout = DEFAULT_TRAJECTORY_TIMEOUT_S
    return {
        "max_tool_rounds": _int("max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS, 0, MAX_TOOL_ROUNDS_CAP),
        "force_final": bool(raw.get("force_final", True)),
        "trajectory_timeout_s": max(1.0, min(TRAJECTORY_TIMEOUT_CAP_S, timeout)),
    }


def replay_config(cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """`replay`-mode settings. `match` defaults to "name" — see ReplayProvider."""
    raw = dict((cfg or {}).get("replay") or {})
    match = str(raw.get("match") or "name").strip().lower()
    if match not in ("name", "exact"):
        match = "name"
    unknown = str(raw.get("unknown_call") or "error").strip().lower()
    if unknown not in ("error", "empty"):
        unknown = "error"
    return {
        "seed_field": str(raw.get("seed_field") or "tool_seed"),
        "match": match,
        "unknown_call": unknown,
    }


# api-mode defaults. `response_field` is a dotted path (custom_eval's `dig`), and
# `""` legitimately means "the whole response" — for a service that answers with a
# bare string. See api_config() for why that distinction is load-bearing.
API_TIMEOUT_CAP_S = 120.0
DEFAULT_API_CONFIG: dict[str, Any] = {
    "url": "",
    "method": "POST",
    "headers": {},
    "auth_header": "Authorization",
    "auth_prefix": "Bearer ",
    "api_key_secret": "",
    "response_field": "content",
    "timeout_s": 30.0,
    "concurrency": 4,
    # ⚠ OFF by default, and that is a correctness setting, not a privacy nicety.
    # `row.expected` holds the gold reference the evaluators grade against
    # (`expected.tool_calls`, the reference reply). A simulator that can read it
    # can return exactly the reference result and inflate the score, with nothing
    # in the trajectory showing why.
    "send_expected": False,
}


def api_config(spec_config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge a stored api config over the defaults.

    ⚠ Only `None` means "not set" — copied deliberately from
    `custom_eval.api_config`, where treating `""` as unset silently reinstated
    the defaults for the two keys where empty string is a MEANINGFUL value:
    `response_field: ""` (read the whole response) and `auth_prefix: ""` (send the
    key with no `Bearer `).
    """
    cfg = dict(DEFAULT_API_CONFIG)
    for key, val in ((spec_config or {}).get("api") or {}).items():
        if val is not None:
            cfg[key] = val
    return cfg


def validate_api_config(cfg: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Check an api-mode config, including the SSRF guard on the URL."""
    merged = api_config(cfg)
    url = str(merged.get("url") or "").strip()
    if not url:
        raise SandboxError("an API sandbox needs a URL")
    try:
        netsafe.assert_safe_fetch_url(url)
    except ValueError as exc:
        raise SandboxError(f"unsafe URL: {exc}") from None
    method = str(merged.get("method") or "POST").upper()
    if method not in ("POST", "PUT", "PATCH"):
        raise SandboxError("method must be POST, PUT or PATCH (the tool call is the body)")
    headers = merged.get("headers")
    if headers and not isinstance(headers, dict):
        raise SandboxError("headers must be an object")
    for hk, hv in (headers or {}).items():
        if not isinstance(hk, str) or not isinstance(hv, (str, int, float)):
            raise SandboxError(f"header {hk!r} must be a string value")
    try:
        timeout = float(merged.get("timeout_s") or 30)
    except (TypeError, ValueError):
        raise SandboxError("timeout must be a number") from None
    if not (0 < timeout <= API_TIMEOUT_CAP_S):
        raise SandboxError(f"timeout must be between 0 and {API_TIMEOUT_CAP_S:g}s")
    merged["url"], merged["method"], merged["timeout_s"] = url, method, timeout
    return merged


def validate_spec(
    mode: str,
    code: str,
    config: Optional[dict[str, Any]] = None,
    *,
    allow_python: bool = False,
) -> None:
    """Check a sandbox definition. Raises `SandboxError` with an author-facing
    message. Called at save time AND at experiment-create time, because a mode can
    be disabled (or a role revoked) between the two."""
    mode = (mode or "").strip()
    if mode not in MODES:
        raise SandboxError(f"unknown mode {mode!r} (expected one of {', '.join(MODES)})")
    if mode not in IMPLEMENTED_MODES:
        raise SandboxError(
            f"mode {mode!r} is not implemented yet — only "
            f"{', '.join(IMPLEMENTED_MODES)} can run today"
        )
    if mode == "python" and not allow_python:
        raise SandboxError("python sandboxes need admin role")
    if mode == "replay":
        rcfg = replay_config(config)
        if not rcfg["seed_field"].strip():
            raise SandboxError("a replay sandbox needs a seed field name")
    if mode == "api":
        validate_api_config(config)
    # Clamping is silent by design (see loop_config), but a value that could only
    # be a mistake is worth saying out loud at authoring time.
    raw_rounds = ((config or {}).get("loop") or {}).get("max_tool_rounds")
    if raw_rounds is not None:
        try:
            if int(raw_rounds) > MAX_TOOL_ROUNDS_CAP:
                raise SandboxError(
                    f"max_tool_rounds above the {MAX_TOOL_ROUNDS_CAP} cap — every round is "
                    f"another billed call for every row × target × variant × repeat"
                )
        except (TypeError, ValueError):
            raise SandboxError("max_tool_rounds must be a whole number") from None


# --------------------------------------------------------------------------- #
# Call identity
# --------------------------------------------------------------------------- #


def canonical_args(args: Any) -> str:
    """Normalize a tool call's `arguments` into a stable string.

    OpenAI puts arguments on the wire as a JSON **string**, so `{"a":1,"b":2}` and
    `{"b":2,"a":1}` are the same call with different bytes. Sorting keys is what
    stops key order forking the response cache and handing two identical calls
    two different worlds. Unparseable arguments fall back to the raw text —
    a malformed call is still a call, and the detectors want to see it.
    """
    if args is None:
        return "{}"
    if isinstance(args, str):
        text = args.strip()
        if not text:
            return "{}"
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            return text
        return canonical_args(parsed)
    try:
        return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(args)


def call_name(call: dict[str, Any]) -> str:
    """Tool name from either shape: OpenAI's `{function: {name}}` or a bare
    `{name}` (what a hand-built fixture usually carries)."""
    fn = call.get("function") if isinstance(call, dict) else None
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return str((call or {}).get("name") or "")


def call_arguments(call: dict[str, Any]) -> Any:
    fn = call.get("function") if isinstance(call, dict) else None
    if isinstance(fn, dict) and "arguments" in fn:
        return fn.get("arguments")
    return (call or {}).get("arguments")


def call_key(call: dict[str, Any]) -> str:
    """The cache/seed key for one tool call."""
    return f"{call_name(call)}({canonical_args(call_arguments(call))})"


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


@dataclass
class ToolResponse:
    """One answer to one tool call.

    `provenance` is the honesty channel: a trajectory mostly answered by
    simulation supports a weaker claim than one replayed from reference data, and
    the drill-down renders the difference.
    """
    content: str
    provenance: str = "seed"       # seed | cache | api | llm | python | error
    error: bool = False
    novel: bool = False            # not answered from seed or cache
    detail: str = ""               # e.g. how the seed matched


def error_response(kind: str, message: str) -> ToolResponse:
    """A structured error the MODEL sees as a tool result.

    Deliberately handed back rather than aborting: reacting to a failed tool call
    is realistic behaviour and worth scoring. The trajectory records
    `provenance="error"` so a run answered entirely by errors can be told apart
    from one that worked.
    """
    return ToolResponse(
        content=json.dumps({"error": kind, "message": message}),
        provenance="error", error=True, novel=True, detail=kind,
    )


class ResponseCache:
    """Per-run memo of `(name, canonical_args) → content`.

    ⚠ This is what makes two variants comparable, and it has a DIRECTION: the
    first cell to make a novel call defines that call's answer for every other
    target, variant and repeat in the run. `seeded_by` records which cell that
    was, so a drill-down can say when a cell was measured in a world someone else
    built. For an exact comparison (checkpoint A vs B), freeze a cache from a
    reference pass instead of letting the run build one.
    """

    def __init__(self, initial: Optional[dict[str, Any]] = None) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        for key, val in (initial or {}).items():
            if isinstance(val, dict) and "content" in val:
                self._entries[str(key)] = dict(val)
            else:
                self._entries[str(key)] = {"content": str(val), "provenance": "seed"}

    def get(self, key: str) -> Optional[ToolResponse]:
        entry = self._entries.get(key)
        if entry is None:
            return None
        return ToolResponse(
            content=str(entry.get("content") or ""),
            # The response is a repeat of one this run already resolved — that is
            # a different claim from having resolved it, so it gets its own tag.
            provenance="cache",
            error=bool(entry.get("error")),
            novel=False,
            detail=str(entry.get("seeded_by") or ""),
        )

    def put(self, key: str, resp: ToolResponse, seeded_by: str = "") -> None:
        self._entries.setdefault(key, {
            "content": resp.content,
            "provenance": resp.provenance,
            "error": resp.error,
            "seeded_by": seeded_by,
        })

    def as_dict(self) -> dict[str, Any]:
        return dict(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------- #
# replay — answer from the dataset row's own reference trajectory
# --------------------------------------------------------------------------- #


class ReplayProvider:
    """Answers tool calls from a `tool_seed` carried on the dataset row.

    Free, offline and totally reproducible — the mode to reach for first. The
    seed is harvested from the row's reference trajectory and is accepted in
    either shape a real corpus produces:

        [{"name": "get_bill", "arguments": {...}, "content": "…"}, …]   # ordered
        {"get_bill": "…", "get_usage": "…"}                             # by name

    ⚠ **Matching is by NAME by default, not by exact arguments.** Requiring
    byte-identical arguments sounds stricter and is actually useless: a model
    under test almost never reproduces the reference call's arguments verbatim,
    so nearly every call would be `no_fixture`, the trajectory would die at round
    one, and the run would report a catastrophic score for a seed-coverage
    problem. The out-of-tree harness this ports has the same behaviour — it
    injects the reference result for the turn regardless of argument fidelity,
    because scoring the ARGUMENTS is the evaluator's job (`function_call_units`
    reads `expected.tool_calls`), not the environment's. Set `match: "exact"` when
    the fixture really does depend on the arguments.
    """

    def __init__(self, spec: SandboxSpec) -> None:
        self.spec = spec
        self.cfg = replay_config(spec.config)

    def seed_for(self, row_expected: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize whatever the row carries into a list of seed entries."""
        raw = (row_expected or {}).get(self.cfg["seed_field"])
        if raw is None:
            return []
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                return []
        entries: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            for name, content in raw.items():
                entries.append({"name": str(name), "arguments": None,
                                "content": _as_text(content)})
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                name = call_name(item)
                if not name:
                    continue
                content = item.get("content", item.get("result", item.get("output")))
                entries.append({"name": name, "arguments": call_arguments(item),
                                "content": _as_text(content)})
        return entries

    def respond(
        self,
        call: dict[str, Any],
        row_expected: Optional[dict[str, Any]],
        used: Optional[set[int]] = None,
    ) -> ToolResponse:
        """Resolve one call against the row's seed.

        `used` lets repeated calls to the same tool walk down the seed in order
        (a conversation that queries the same endpoint twice gets two different
        reference results), while still falling back to the last match rather
        than erroring once the seed runs out.
        """
        name = call_name(call)
        if not name:
            return error_response(ERR_UNKNOWN_FUNCTION, "the tool call carries no function name")

        entries = self.seed_for(row_expected)
        if not entries:
            return error_response(
                ERR_NO_FIXTURE,
                f"the dataset row carries no {self.cfg['seed_field']!r} for {name!r}",
            )

        want_args = canonical_args(call_arguments(call))
        by_name = [(i, e) for i, e in enumerate(entries) if e["name"] == name]
        if not by_name:
            return error_response(
                ERR_UNKNOWN_FUNCTION,
                f"{name!r} is not in this row's reference trajectory",
            )

        exact = [(i, e) for i, e in by_name if canonical_args(e.get("arguments")) == want_args]
        if exact:
            idx, entry = _pick(exact, used)
            return ToolResponse(entry["content"], provenance="seed", detail="exact")

        if self.cfg["match"] == "exact":
            if self.cfg["unknown_call"] == "empty":
                return ToolResponse("", provenance="seed", novel=True, detail="no-exact-match")
            return error_response(
                ERR_NO_FIXTURE,
                f"no reference result for {name!r} with these arguments",
            )

        idx, entry = _pick(by_name, used)
        return ToolResponse(entry["content"], provenance="seed", detail="name")


def _pick(candidates: list[tuple[int, dict[str, Any]]], used: Optional[set[int]]):
    """First unused candidate, else the last one. Consuming the seed in order is
    what lets a conversation call the same tool twice and get two results; falling
    back to the last (rather than erroring) keeps a model that calls one more time
    than the reference did from killing the trajectory."""
    if used is not None:
        for idx, entry in candidates:
            if idx not in used:
                used.add(idx)
                return idx, entry
    return candidates[-1]


def _as_text(value: Any) -> str:
    """Tool results reach the model as a string; JSON-encode anything else so a
    dict fixture doesn't arrive as a Python repr with single quotes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


# --------------------------------------------------------------------------- #
# api — answer from a service the user already runs
# --------------------------------------------------------------------------- #


class ApiProvider:
    """Answers tool calls by POSTing them to an endpoint the user owns.

    This is the mode that makes the feature general: any existing mock service,
    staging API, fixture server or simulated environment becomes a sandbox with
    **zero gateway code**, and the metric/environment definitions stay versioned
    in the repo that owns them rather than leaking in here.

    Nothing executes on the gateway, so there is no sandbox to reason about —
    only `netsafe.assert_safe_fetch_url` (which permits internal hosts, because an
    in-cluster simulator is legitimate, but blocks link-local / cloud-metadata),
    re-checked on first use because a saved hostname can be re-pointed, paired
    with `follow_redirects=False` so a 3xx can't bounce onto a blocked host.

    Request body:

        {"conversation": [...], "tool_call": {...}, "call": {"name", "arguments"},
         "row": {...}, "sandbox": "<name>"}

    ⚠ `row` carries the dataset row's `expected` block ONLY when
    `send_expected` is on — see DEFAULT_API_CONFIG for why that default is a
    correctness setting.
    """

    def __init__(
        self,
        spec: SandboxSpec,
        client: Optional[httpx.AsyncClient],
        api_key: str = "",
    ) -> None:
        self.spec = spec
        self.cfg = api_config(spec.config)
        self.client = client
        self.api_key = api_key
        # Bound in flight so a slow simulator can't starve the replay connections.
        self._sem = asyncio.Semaphore(max(1, int(self.cfg.get("concurrency") or 4)))
        self._url_checked = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        for key, val in (self.cfg.get("headers") or {}).items():
            headers[str(key)] = str(val)
        if self.api_key:
            hdr = str(self.cfg.get("auth_header") or "Authorization")
            prefix = self.cfg.get("auth_prefix")
            prefix = "Bearer " if prefix is None else str(prefix)
            headers[hdr] = f"{prefix}{self.api_key}"
        return headers

    def payload(
        self,
        call: dict[str, Any],
        conversation: Optional[list[dict[str, Any]]],
        row_expected: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "sandbox": self.spec.name,
            "conversation": list(conversation or []),
            "tool_call": call,
            # Pre-parsed too: a fixture server shouldn't have to re-implement
            # "arguments is a JSON string sometimes".
            "call": {
                "name": call_name(call),
                "arguments": _parsed_args(call_arguments(call)),
            },
        }
        if self.cfg.get("send_expected"):
            body["row"] = dict(row_expected or {})
        return body

    async def respond(
        self,
        call: dict[str, Any],
        conversation: Optional[list[dict[str, Any]]],
        row_expected: Optional[dict[str, Any]],
    ) -> ToolResponse:
        url = str(self.cfg.get("url") or "")
        if self.client is None:
            return error_response("sandbox_error", "no HTTP client available")
        if not self._url_checked:
            try:
                netsafe.assert_safe_fetch_url(url)
            except ValueError as exc:
                return error_response("sandbox_error", f"sandbox URL rejected: {exc}")
            self._url_checked = True

        try:
            async with self._sem:
                resp = await self.client.request(
                    str(self.cfg.get("method") or "POST"),
                    url,
                    headers=self._headers(),
                    json=self.payload(call, conversation, row_expected),
                    timeout=float(self.cfg.get("timeout_s") or 30),
                    follow_redirects=False,
                )
        except Exception as exc:  # noqa: BLE001 — the endpoint is the user's, not ours
            return error_response(
                "sandbox_unreachable", f"{exc.__class__.__name__}: {exc}",
            )

        if resp.status_code >= 400:
            return error_response(
                "sandbox_http_error",
                f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        field_path = str(self.cfg.get("response_field") or "")
        try:
            data = resp.json()
        except ValueError:
            # A service answering plain text is fine — but only when the config
            # asked for the whole response. Otherwise the field is genuinely
            # missing and saying so beats feeding the model a stray HTML page.
            if not field_path:
                return ToolResponse(resp.text, provenance="api", novel=True, detail="text")
            return error_response(
                "sandbox_bad_response",
                f"the sandbox did not return JSON: {resp.text[:200]}",
            )

        from .custom_eval import dig
        value = dig(data, field_path)
        if value is None:
            return error_response(
                "sandbox_bad_response",
                f"no {field_path!r} in the sandbox response "
                f"(got keys: {sorted(data)[:8] if isinstance(data, dict) else type(data).__name__})",
            )
        return ToolResponse(_as_text(value), provenance="api", novel=True, detail=field_path)


def _parsed_args(args: Any) -> Any:
    """Tool-call arguments as an object where possible, raw text otherwise."""
    if isinstance(args, str):
        try:
            return json.loads(args or "{}")
        except (ValueError, TypeError):
            return args
    return args if args is not None else {}


# --------------------------------------------------------------------------- #
# Runtime — one per run, shared by every unit
# --------------------------------------------------------------------------- #


class SandboxRuntime:
    """What the runner holds: the snapshotted spec, the loop policy, the shared
    response cache, and the counters the anti-silent-no-op rules need.

    One instance per experiment run — the cache is run-scoped on purpose (see
    ResponseCache) and the counters roll up into the summary.
    """

    def __init__(
        self,
        spec: SandboxSpec,
        *,
        cache: Optional[dict[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
        api_key: str = "",
    ) -> None:
        self.spec = spec
        self.loop = loop_config(spec.config)
        self.cache = ResponseCache(cache)
        self._replay = ReplayProvider(spec) if spec.mode == "replay" else None
        self._api = ApiProvider(spec, client, api_key) if spec.mode == "api" else None

    @property
    def max_tool_rounds(self) -> int:
        return int(self.loop["max_tool_rounds"])

    @property
    def force_final(self) -> bool:
        return bool(self.loop["force_final"])

    @property
    def trajectory_timeout_s(self) -> float:
        return float(self.loop["trajectory_timeout_s"])

    async def respond(
        self,
        call: dict[str, Any],
        row_expected: Optional[dict[str, Any]],
        *,
        cell: str = "",
        used: Optional[set[int]] = None,
        conversation: Optional[list[dict[str, Any]]] = None,
    ) -> ToolResponse:
        """Answer one tool call. Never raises — a provider fault becomes a
        structured error the model sees, exactly like an unknown function."""
        key = call_key(call)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        try:
            if self._replay is not None:
                resp = self._replay.respond(call, row_expected, used)
                resp.novel = resp.novel or resp.provenance != "seed"
            elif self._api is not None:
                resp = await self._api.respond(call, conversation, row_expected)
            else:
                resp = error_response(
                    ERR_NO_FIXTURE, f"sandbox mode {self.spec.mode!r} cannot answer calls",
                )
        except Exception as exc:  # noqa: BLE001 — a broken sandbox must not kill the run
            logger.warning("sandbox %s failed on %s: %s", self.spec.name, key, exc)
            resp = error_response("sandbox_error", f"{exc.__class__.__name__}: {exc}")
        # Errors are cached too: within one run the same bad call must fail the
        # same way for every variant, or the comparison measures retry luck.
        self.cache.put(key, resp, seeded_by=cell)
        return resp


# --------------------------------------------------------------------------- #
# Registry payload (server-driven UI, same convention as the evaluator registry)
# --------------------------------------------------------------------------- #


def specs_payload() -> dict[str, Any]:
    """Mode descriptors + the loop-option schema the web editor renders directly,
    so a new mode ships without a web change."""
    return {
        "modes": [
            {
                "id": "replay",
                "label": "Replay (from the dataset row)",
                "implemented": True,
                "description": "Answers each tool call from a reference trajectory carried on the "
                               "row. Free, offline and totally reproducible — start here.",
                "options": [
                    {"name": "seed_field", "type": "text", "label": "Seed column",
                     "default": "tool_seed",
                     "help": "Key on the row's `expected` block holding the reference results."},
                    {"name": "match", "type": "select", "label": "Match calls by",
                     "default": "name", "options": ["name", "exact"],
                     "help": "name = the reference result for that tool (what the benchmark "
                             "does); exact = also require identical arguments."},
                    {"name": "unknown_call", "type": "select", "label": "Unseeded call",
                     "default": "error", "options": ["error", "empty"],
                     "help": "What the model gets for a call the seed doesn't cover. Never a "
                             "fabricated success."},
                ],
            },
            {
                "id": "api",
                "label": "HTTP endpoint",
                "implemented": True,
                "description": "POSTs each tool call to a service you already run and reads the "
                               "result out of its JSON. Any mock service, staging API or "
                               "simulated environment becomes a sandbox with no gateway code.",
                "options": [
                    {"name": "url", "type": "text", "label": "Endpoint URL", "default": "",
                     "help": "Receives {conversation, tool_call, call}. Internal hosts are "
                             "allowed; link-local and cloud-metadata addresses are not."},
                    {"name": "method", "type": "select", "label": "Method",
                     "default": "POST", "options": ["POST", "PUT", "PATCH"]},
                    {"name": "response_field", "type": "text", "label": "Result path",
                     "default": "content",
                     "help": "Dotted path into the response, e.g. result.output. Blank = the "
                             "whole response, for a service that answers with a bare string."},
                    {"name": "api_key_secret", "type": "text", "label": "API key secret",
                     "default": "",
                     "help": "Name of a global secret. The key itself is never stored here."},
                    {"name": "auth_header", "type": "text", "label": "Auth header",
                     "default": "Authorization"},
                    {"name": "auth_prefix", "type": "text", "label": "Auth prefix",
                     "default": "Bearer ", "help": "Blank sends the key with no prefix."},
                    {"name": "timeout_s", "type": "number", "label": "Timeout (s)", "default": 30},
                    {"name": "concurrency", "type": "number", "label": "Max in flight",
                     "default": 4,
                     "help": "Bounded so a slow simulator can't starve the replay connections."},
                    {"name": "send_expected", "type": "boolean",
                     "label": "Send the row's expected block", "default": False,
                     "help": "⚠ Off by default on purpose: `expected` holds the gold answer the "
                             "evaluators grade against, so a sandbox that can read it can return "
                             "the reference result and inflate the score."},
                ],
            },
            {"id": "llm", "label": "Simulated by a model", "implemented": False,
             "description": "A model writes a plausible tool result. Billed per call. "
                            "Not implemented yet.",
             "options": []},
            {"id": "python", "label": "Python (admin)", "implemented": False,
             "description": "A `respond(convo, call)` function. Not implemented yet.",
             "options": []},
        ],
        "loop_options": [
            {"name": "max_tool_rounds", "type": "number", "label": "Max tool rounds",
             "default": DEFAULT_MAX_TOOL_ROUNDS,
             "help": f"Each round is another billed model call per row. Cap {MAX_TOOL_ROUNDS_CAP}."},
            {"name": "force_final", "type": "boolean", "label": "Force a final answer", "default": True,
             "help": "At the round limit, re-ask with the tools removed so the reply is text."},
            {"name": "trajectory_timeout_s", "type": "number", "label": "Trajectory timeout (s)",
             "default": DEFAULT_TRAJECTORY_TIMEOUT_S,
             "help": "Whole-trajectory deadline. The request timeout is per turn."},
        ],
        "implemented_modes": list(IMPLEMENTED_MODES),
        "max_tool_rounds_cap": MAX_TOOL_ROUNDS_CAP,
    }
