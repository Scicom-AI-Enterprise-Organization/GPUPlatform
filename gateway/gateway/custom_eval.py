"""User-defined evaluators for the Experiments feature.

`evaluators.py` ships the detectors that recur across studies. This module is the
escape hatch for the one-off check that doesn't: the user writes it in the UI
instead of editing the repo and restarting the gateway.

Three modes, deliberately different in power AND in trust:

**`expression`** (default, always available). One Python *expression* over the
completion, evaluated by a whitelisting AST walker — no imports, no statements,
no attribute access to anything dunder, and calls only into a fixed helper
registry. Everything in scope is plain data (str / int / list / dict / None), so
there is no object graph to climb toward `__subclasses__`. This covers the large
majority of real detectors, which are regex + threshold logic.

**`api`** (always available). POST the completion to an HTTP endpoint you already
own — an existing scorer, a classifier service, a hosted guardrail — and read the
verdict out of the JSON response. No code runs on the gateway at all, so this is
the right answer whenever the logic already lives somewhere else. URLs go through
`netsafe.assert_safe_fetch_url`, which permits internal hosts (an in-cluster
scorer is a legitimate target) but blocks link-local / cloud-metadata addresses.

**`python`** (**admin-only, on by default**). A real `def check(c)`. This one
executes arbitrary user code, so it is restricted to **admin role** (re-checked at
save AND at experiment-create time) and runs in a separate process with a scrubbed
environment (none of the gateway's DB / Fernet / cloud credentials), CPU +
address-space + file-descriptor rlimits, and a wall-clock kill. Set
`EXPERIMENT_ALLOW_PYTHON_EVALUATORS=0` to disable the mode outright.

⚠️ **Be honest about what the `python` sandbox is and isn't.** It removes the
credentials from the environment, bounds CPU and memory, and keeps user code out
of the gateway's own process — a real reduction in blast radius. It does NOT
prevent a determined author from reading the filesystem the gateway user can read
or opening a socket. With the mode on by default, **admin role is the whole
control**: anyone who is an admin (or who can reach an endpoint authenticated as
one) has code execution on the gateway host. Deployments where that is not
acceptable must set the env var to 0. The expression mode has no such caveat.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import sys
import zlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from . import evaluators as ev
from . import netsafe

logger = logging.getLogger("gateway.custom_eval")

MODES = ("expression", "api", "python")

# `python` mode executes arbitrary user code — see the module docstring.
PYTHON_MODE_ENV = "EXPERIMENT_ALLOW_PYTHON_EVALUATORS"
# Wall-clock ceiling for one python-mode call, and the child's own CPU rlimit.
PY_CALL_TIMEOUT_S = float(os.environ.get("EXPERIMENT_PY_EVAL_TIMEOUT_S", "5") or "5")
PY_CPU_SECONDS = int(os.environ.get("EXPERIMENT_PY_EVAL_CPU_S", "5") or "5")
PY_MEM_BYTES = int(os.environ.get("EXPERIMENT_PY_EVAL_MEM_MB", "512") or "512") * 1024 * 1024


def python_mode_enabled() -> bool:
    """Whether python-mode evaluators may be authored/run at all.

    ⚠ **Default ON (opt-OUT)** since 2026-08-07, by explicit product decision.
    The other half of the gate — **admin role** — still applies and is checked
    separately at both save and experiment-create time, so this is not
    "anyone can run code": it is "an admin can, without an env flag first".

    Set `EXPERIMENT_ALLOW_PYTHON_EVALUATORS=0` (or false/no/off) to restore the
    old refuse-by-default behaviour on a deployment where even admin-authored
    code execution on the gateway host is unacceptable.
    """
    raw = (os.environ.get(PYTHON_MODE_ENV, "") or "").strip().lower()
    if not raw:
        return True
    return raw not in ("0", "false", "no", "off")


class CustomEvalError(ValueError):
    """The evaluator is invalid — surfaced to the author, not a 500."""


# --------------------------------------------------------------------------- #
# The variables a custom evaluator sees
# --------------------------------------------------------------------------- #


def completion_vars(c: ev.Completion) -> dict[str, Any]:
    """Plain-data view of one completion. Everything here is str/int/float/None/
    list/dict — never a live object — which is what makes attribute access in
    expression mode safe to allow."""
    usage = c.usage or {}
    tool_calls = []
    for call in (c.expected or {}).get("_tool_calls") or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments")
        parsed: Any = None
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                parsed = None
        elif isinstance(args, dict):
            parsed = args
        tool_calls.append({
            "name": fn.get("name") or "",
            "arguments": args if isinstance(args, str) else json.dumps(args or {}),
            "parsed_arguments": parsed,
        })
    expected = {k: v for k, v in (c.expected or {}).items() if not k.startswith("_")}
    return {
        "content": c.content or "",
        "reasoning": c.reasoning or "",
        "finish_reason": c.finish_reason,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_ms": c.latency_ms,
        "ttft_ms": c.ttft_ms,
        "status_code": c.status_code,
        "error": c.error,
        "tool_calls": tool_calls,
        "expected": expected,
    }


# --------------------------------------------------------------------------- #
# Helper registry — the only callables an expression can reach
# --------------------------------------------------------------------------- #


def _re_search(pattern: str, text: Any, ignore_case: bool = True) -> bool:
    try:
        return bool(re.search(str(pattern), str(text or ""), re.IGNORECASE if ignore_case else 0))
    except re.error as exc:
        raise CustomEvalError(f"bad regex {pattern!r}: {exc}") from None


def _re_findall(pattern: str, text: Any, ignore_case: bool = True) -> list[str]:
    try:
        return re.findall(str(pattern), str(text or ""), re.IGNORECASE if ignore_case else 0)
    except re.error as exc:
        raise CustomEvalError(f"bad regex {pattern!r}: {exc}") from None


def _re_count(pattern: str, text: Any, ignore_case: bool = True) -> int:
    return len(_re_findall(pattern, text, ignore_case))


def _json_loads(text: Any) -> Any:
    """Parse, or None — never raises, so an expression can test `is None`."""
    try:
        return json.loads(str(text or ""))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _is_json_object(text: Any) -> bool:
    return isinstance(_json_loads(text), dict)


def _distinct_ratio(text: Any) -> float:
    return float(ev.degeneration_metrics(str(text or ""))["distinct_ratio"])


def _max_repeat(text: Any) -> int:
    return int(ev.degeneration_metrics(str(text or ""))["max_consecutive_repeat"])


def _compression_ratio(text: Any) -> float:
    return float(ev.degeneration_metrics(str(text or ""))["compression_ratio"])


def _top_token_ratio(text: Any) -> float:
    return float(ev.degeneration_metrics(str(text or ""))["top_token_ratio"])


def _word_count(text: Any) -> int:
    return len(str(text or "").split())


def _control_tokens(text: Any) -> list[str]:
    """The same control/channel-marker scan the built-in leak detector uses."""
    return sorted(set(ev._CONTROL_TOKEN_RE.findall(str(text or ""))))


def _json_keys(text: Any) -> list[str]:
    obj = _json_loads(text)
    return sorted(obj.keys()) if isinstance(obj, dict) else []


HELPERS: dict[str, Any] = {
    # regex
    "re_search": _re_search,
    "re_findall": _re_findall,
    "re_count": _re_count,
    # json
    "json_loads": _json_loads,
    "is_json_object": _is_json_object,
    "json_keys": _json_keys,
    # text stats (the degeneration building blocks, reusable on their own)
    "distinct_ratio": _distinct_ratio,
    "max_repeat": _max_repeat,
    "compression_ratio": _compression_ratio,
    "top_token_ratio": _top_token_ratio,
    "word_count": _word_count,
    "control_tokens": _control_tokens,
    # builtins, hand-picked
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "any": any,
    "all": all,
    "sorted": sorted,
    "round": round,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "set": set,
    "list": list,
}

# Methods an expression may call on the plain data in scope. Anything not here —
# and anything starting with "_" — is rejected at compile time.
SAFE_METHODS = frozenset({
    "lower", "upper", "strip", "lstrip", "rstrip", "title", "casefold",
    "startswith", "endswith", "count", "find", "rfind", "index",
    "split", "rsplit", "splitlines", "join", "replace", "removeprefix", "removesuffix",
    "get", "keys", "values", "items",
    "isdigit", "isalpha", "isspace",
})

# ⚠ `ast.Pow` is deliberately ABSENT. Expression mode runs in the gateway's own
# process, and `2**(10**10)` is a three-character CPU/memory bomb that would take
# the gateway down. Without `**`, and with no loops or comprehensions, the only
# remaining lever is repeated multiplication — bounded below by _MAX_REPEAT_CONST.
_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot, ast.Call, ast.keyword, ast.Name, ast.Load, ast.Constant,
    ast.Subscript, ast.Slice, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.IfExp,
    ast.Attribute,
)

# Largest constant allowed as a multiplier: `content * 100000000` allocates a
# gigabyte. Real detectors never multiply by a big literal.
_MAX_REPEAT_CONST = 1000


def validate_expression(source: str, variables: Optional[set[str]] = None) -> ast.Expression:
    """Parse + whitelist-check an expression. Raises CustomEvalError on anything
    outside the allowed grammar."""
    src = (source or "").strip()
    if not src:
        raise CustomEvalError("the expression is empty")
    if len(src) > 4000:
        raise CustomEvalError("expression too long (4000 char limit)")
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as exc:
        raise CustomEvalError(f"syntax error: {exc.msg} (offset {exc.offset})") from None

    known = set(variables or set()) | set(HELPERS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Pow):
            raise CustomEvalError(
                "** is not allowed (it makes CPU/memory bombs trivial) — use x*x"
            )
        if not isinstance(node, _ALLOWED_NODES):
            # Comprehensions are excluded on purpose: nested ones make it easy to
            # write an accidental O(n²) blowup over an 8k-char completion.
            raise CustomEvalError(
                f"{type(node).__name__} is not allowed in an expression evaluator"
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            for side in (node.left, node.right):
                if (
                    isinstance(side, ast.Constant)
                    and isinstance(side.value, int)
                    and not isinstance(side.value, bool)
                    and abs(side.value) > _MAX_REPEAT_CONST
                ):
                    raise CustomEvalError(
                        f"multiplying by {side.value} is not allowed "
                        f"(limit {_MAX_REPEAT_CONST}) — it can allocate gigabytes"
                    )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in SAFE_METHODS:
                raise CustomEvalError(f"attribute .{node.attr} is not allowed")
        if isinstance(node, ast.Name) and node.id not in known:
            raise CustomEvalError(
                f"unknown name {node.id!r} — available: "
                + ", ".join(sorted(known))
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in HELPERS:
                raise CustomEvalError(f"{node.func.id!r} is not a callable helper")
    return tree


def compile_expression(source: str, variables: Optional[set[str]] = None):
    tree = validate_expression(source, variables)
    return compile(tree, "<custom-evaluator>", "eval")


_VAR_NAMES = set(completion_vars(ev.Completion()).keys())


def eval_expression(code, vars_: dict[str, Any]) -> Any:
    """Run a compiled expression. `__builtins__` is emptied so even a name that
    slipped the walker has nothing to reach."""
    return eval(code, {"__builtins__": {}}, {**HELPERS, **vars_})  # noqa: S307


# --------------------------------------------------------------------------- #
# Normalizing whatever the user returned
# --------------------------------------------------------------------------- #


def normalize_result(raw: Any, fail_when_true: bool, name: str) -> ev.EvalOutcome:
    """Accept a bool, a number, or a {passed, score, reason, flags} dict.

    `fail_when_true` flips the sense so an author can write either "this is what
    good looks like" or "this is the bug I'm hunting" — both read naturally, and
    guessing wrong silently inverts every result.
    """
    score: Optional[float] = None
    reason: Optional[str] = None
    flags: dict[str, Any] = {}

    if isinstance(raw, dict):
        truth = bool(raw.get("passed", raw.get("ok", False)))
        if raw.get("score") is not None:
            try:
                score = float(raw["score"])
            except (TypeError, ValueError):
                score = None
        if raw.get("reason") is not None:
            reason = str(raw["reason"])[:500]
        if isinstance(raw.get("flags"), dict):
            flags = raw["flags"]
        passed = truth
        # An explicit dict states its own verdict; flipping it too would be a trap.
    else:
        truth = bool(raw)
        passed = (not truth) if fail_when_true else truth
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            score = float(raw)

    if score is None:
        score = 1.0 if passed else 0.0
    if reason is None and not passed:
        reason = f"{name}: matched" if fail_when_true else f"{name}: condition not met"
    return ev.EvalOutcome(id=name, passed=passed, score=score, reason=reason, flags=flags)


# --------------------------------------------------------------------------- #
# Expression evaluator
# --------------------------------------------------------------------------- #


@dataclass
class CustomSpec:
    """A resolved custom evaluator, snapshotted into the experiment config so a
    later edit to the library entry can't change what a finished run means."""
    id: str
    name: str
    mode: str
    code: str
    fail_when_true: bool = False
    # api mode only: {url, method, headers, auth_*, *_field, timeout_s, concurrency}
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustomSpec":
        return cls(
            id=str(d.get("id") or "custom"),
            name=str(d.get("name") or d.get("id") or "custom"),
            mode=str(d.get("mode") or "expression"),
            code=str(d.get("code") or ""),
            fail_when_true=bool(d.get("fail_when_true", False)),
            config=dict(d.get("config") or {}),
        )


def run_expression_evaluator(spec: CustomSpec, c: ev.Completion) -> ev.EvalOutcome:
    """Evaluate one expression-mode evaluator. Never raises: an author error is
    reported as a non-failing outcome so one bad expression can't void a run."""
    try:
        code = compile_expression(spec.code, _VAR_NAMES)
        raw = eval_expression(code, completion_vars(c))
        return normalize_result(raw, spec.fail_when_true, spec.name)
    except CustomEvalError as exc:
        return ev.EvalOutcome(
            id=spec.name, passed=True, score=None,
            reason=f"evaluator invalid: {exc}", flags={"evaluator_error": True},
        )
    except Exception as exc:  # noqa: BLE001 — author bug, not a platform bug
        return ev.EvalOutcome(
            id=spec.name, passed=True, score=None,
            reason=f"evaluator raised {exc.__class__.__name__}: {exc}",
            flags={"evaluator_error": True},
        )


# --------------------------------------------------------------------------- #
# API-mode evaluator
# --------------------------------------------------------------------------- #

DEFAULT_API_CONFIG: dict[str, Any] = {
    "url": "",
    "method": "POST",
    "headers": {},
    # Resolved from a global secret at run time; never stored inline.
    "api_key_secret": "",
    "auth_header": "Authorization",
    "auth_prefix": "Bearer ",
    "timeout_s": 30.0,
    "concurrency": 4,
    # Dotted paths into the JSON response. "a.b.0.c" walks dicts and lists.
    "passed_field": "passed",
    "score_field": "score",
    "reason_field": "reason",
    "flags_field": "flags",
    # Send the reasoning channel too (off for endpoints that only want the answer).
    "include_reasoning": True,
}

API_TIMEOUT_CAP_S = 120.0


def api_config(spec_config: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Merge a stored config over the defaults.

    ⚠ Only `None` means "not set". An empty string is a **meaningful value** for
    two keys — `passed_field: ""` reads the whole response (for an endpoint that
    answers a bare `true`), and `auth_prefix: ""` sends the key with no `Bearer `.
    Treating "" as unset silently reinstated the defaults for both.
    """
    cfg = dict(DEFAULT_API_CONFIG)
    for k, v in (spec_config or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def dig(obj: Any, path: str) -> Any:
    """Walk a dotted path through dicts and lists. Missing → None.

    `""` means "the whole response", which is what you want when the endpoint
    answers with a bare `true` rather than an object.
    """
    if not path:
        return obj
    cur = obj
    for part in str(path).split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def validate_api_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Check an api-mode config, including the SSRF guard on the URL."""
    merged = api_config(cfg)
    url = str(merged.get("url") or "").strip()
    if not url:
        raise CustomEvalError("an API evaluator needs a URL")
    try:
        netsafe.assert_safe_fetch_url(url)
    except ValueError as exc:
        raise CustomEvalError(f"unsafe URL: {exc}") from None
    method = str(merged.get("method") or "POST").upper()
    if method not in ("POST", "PUT", "PATCH"):
        raise CustomEvalError("method must be POST, PUT or PATCH (the completion is the body)")
    headers = merged.get("headers")
    if headers and not isinstance(headers, dict):
        raise CustomEvalError("headers must be an object")
    for key in ("headers",):
        for hk, hv in (merged.get(key) or {}).items():
            if not isinstance(hk, str) or not isinstance(hv, (str, int, float)):
                raise CustomEvalError(f"header {hk!r} must be a string value")
    try:
        t = float(merged.get("timeout_s") or 30)
    except (TypeError, ValueError):
        raise CustomEvalError("timeout must be a number") from None
    if not (0 < t <= API_TIMEOUT_CAP_S):
        raise CustomEvalError(f"timeout must be between 0 and {API_TIMEOUT_CAP_S:g}s")
    merged["url"], merged["method"], merged["timeout_s"] = url, method, t
    return merged


class ApiEvaluatorClient:
    """Scores completions by calling an HTTP endpoint the user already owns.

    Nothing executes on the gateway, so this mode needs no sandbox — only the
    SSRF guard on the URL and a bound on how many calls are in flight.
    """

    def __init__(self, spec: CustomSpec, client: httpx.AsyncClient, api_key: str = ""):
        self.spec = spec
        self.cfg = api_config(spec.config)
        self.client = client
        self.api_key = api_key
        self._sem = asyncio.Semaphore(max(1, int(self.cfg.get("concurrency") or 4)))
        self._url_checked = False

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        for k, v in (self.cfg.get("headers") or {}).items():
            headers[str(k)] = str(v)
        if self.api_key:
            hdr = str(self.cfg.get("auth_header") or "Authorization")
            prefix = str(self.cfg.get("auth_prefix") or "")
            headers[hdr] = f"{prefix}{self.api_key}"
        return headers

    def payload(self, c: ev.Completion) -> dict[str, Any]:
        body = completion_vars(c)
        if not self.cfg.get("include_reasoning", True):
            body.pop("reasoning", None)
        body["evaluator"] = self.spec.name
        return body

    async def evaluate(self, c: ev.Completion) -> ev.EvalOutcome:
        name = self.spec.name
        url = str(self.cfg.get("url") or "")
        # Re-check on first use: a stored evaluator's hostname could have been
        # re-pointed at a metadata address since it was saved.
        if not self._url_checked:
            try:
                netsafe.assert_safe_fetch_url(url)
            except ValueError as exc:
                return ev.EvalOutcome(
                    id=name, passed=True, score=None,
                    reason=f"evaluator URL rejected: {exc}",
                    flags={"evaluator_error": True},
                )
            self._url_checked = True

        try:
            async with self._sem:
                resp = await self.client.request(
                    str(self.cfg.get("method") or "POST"),
                    url,
                    headers=self._headers(),
                    json=self.payload(c),
                    timeout=float(self.cfg.get("timeout_s") or 30),
                    # A 3xx could bounce a validated host onto a blocked one.
                    follow_redirects=False,
                )
        except Exception as exc:  # noqa: BLE001 — the endpoint is the user's, not ours
            return ev.EvalOutcome(
                id=name, passed=True, score=None,
                reason=f"evaluator API unreachable: {exc.__class__.__name__}: {exc}",
                flags={"evaluator_error": True},
            )

        if resp.status_code >= 400:
            return ev.EvalOutcome(
                id=name, passed=True, score=None,
                reason=f"evaluator API returned HTTP {resp.status_code}: {resp.text[:200]}",
                flags={"evaluator_error": True, "status_code": resp.status_code},
            )
        try:
            data = resp.json()
        except ValueError:
            return ev.EvalOutcome(
                id=name, passed=True, score=None,
                reason=f"evaluator API did not return JSON: {resp.text[:200]}",
                flags={"evaluator_error": True},
            )

        return self.parse_response(data)

    def parse_response(self, data: Any) -> ev.EvalOutcome:
        """Extract the verdict from the endpoint's JSON via the configured paths."""
        name = self.spec.name
        raw_passed = dig(data, str(self.cfg.get("passed_field") or ""))
        score_raw = dig(data, str(self.cfg.get("score_field") or ""))
        reason_raw = dig(data, str(self.cfg.get("reason_field") or ""))
        flags_raw = dig(data, str(self.cfg.get("flags_field") or ""))

        if raw_passed is None and score_raw is None:
            return ev.EvalOutcome(
                id=name, passed=True, score=None,
                reason=(
                    f"evaluator API response has no {self.cfg.get('passed_field')!r} field "
                    f"(got keys: {sorted(data)[:8] if isinstance(data, dict) else type(data).__name__})"
                ),
                flags={"evaluator_error": True},
            )

        # A string verdict is common ("pass" / "PASS" / "fail" / "yes" / "no").
        if isinstance(raw_passed, str):
            token = raw_passed.strip().lower()
            if token in ("pass", "passed", "true", "yes", "ok", "good", "1"):
                truth = True
            elif token in ("fail", "failed", "false", "no", "bad", "0"):
                truth = False
            else:
                truth = bool(token)
        elif raw_passed is None:
            truth = bool(score_raw)
        else:
            truth = bool(raw_passed)

        passed = (not truth) if self.spec.fail_when_true else truth
        score: Optional[float] = None
        if score_raw is not None:
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = None
        if score is None:
            score = 1.0 if passed else 0.0

        reason = str(reason_raw)[:500] if reason_raw is not None else None
        if reason is None and not passed:
            reason = f"{name}: endpoint returned a failing verdict"
        flags = flags_raw if isinstance(flags_raw, dict) else {}
        return ev.EvalOutcome(id=name, passed=passed, score=score, reason=reason, flags=flags)


# --------------------------------------------------------------------------- #
# Python-mode worker
# --------------------------------------------------------------------------- #

# The child program. It sets its OWN rlimits before touching user code (doing it
# via preexec_fn is documented as unsafe in a threaded parent), then serves one
# JSON request per line so process startup is paid once per run, not per sample.
_CHILD_SOURCE = r'''
import json, os, resource, sys

def _limit():
    cpu, mem, nofile = int(sys.argv[1]), int(sys.argv[2]), 64
    try: resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception: pass
    try: resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    except Exception: pass
    try: resource.setrlimit(resource.RLIMIT_NOFILE, (nofile, nofile))
    except Exception: pass
    try: resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception: pass

_limit()
source = sys.stdin.readline()
try:
    user_code = json.loads(source)["code"]
except Exception as exc:
    sys.stdout.write(json.dumps({"fatal": "could not read evaluator code: %s" % exc}) + "\n")
    sys.stdout.flush(); sys.exit(1)

ns = {}
try:
    exec(compile(user_code, "<custom-evaluator>", "exec"), ns)
except Exception as exc:
    sys.stdout.write(json.dumps({"fatal": "%s: %s" % (type(exc).__name__, exc)}) + "\n")
    sys.stdout.flush(); sys.exit(1)

fn = ns.get("check")
if not callable(fn):
    sys.stdout.write(json.dumps({"fatal": "define a function named check(c)"}) + "\n")
    sys.stdout.flush(); sys.exit(1)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        payload = json.loads(line)
    except Exception:
        continue
    try:
        out = fn(payload)
        if not isinstance(out, (bool, int, float, dict)) and out is not None:
            out = bool(out)
        sys.stdout.write(json.dumps({"result": out}, default=str) + "\n")
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": "%s: %s" % (type(exc).__name__, exc)}) + "\n")
    sys.stdout.flush()
'''

# Environment handed to the child: no DATABASE_URL, no PROVIDER_SECRET_KEY, no
# cloud keys — just enough to start an interpreter.
_CHILD_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TMPDIR")


class PythonEvaluatorWorker:
    """One long-lived child process per python-mode evaluator per run.

    Spawn cost is paid once instead of per sample (a 10k-sample run would
    otherwise be 10k process launches). A hang or crash kills and respawns, so a
    single bad sample can't wedge the whole run.
    """

    def __init__(self, spec: CustomSpec):
        self.spec = spec
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._fatal: Optional[str] = None

    async def _spawn(self) -> None:
        env = {k: os.environ[k] for k in _CHILD_ENV_KEYS if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self._proc = await asyncio.create_subprocess_exec(
            sys.executable, "-I", "-c", _CHILD_SOURCE,
            str(PY_CPU_SECONDS), str(PY_MEM_BYTES),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        assert self._proc.stdin is not None
        self._proc.stdin.write((json.dumps({"code": self.spec.code}) + "\n").encode())
        await self._proc.stdin.drain()

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
            await asyncio.wait_for(proc.wait(), timeout=5)
        except (ProcessLookupError, asyncio.TimeoutError):
            pass

    async def evaluate(self, c: ev.Completion) -> ev.EvalOutcome:
        name = self.spec.name
        if self._fatal:
            return ev.EvalOutcome(id=name, passed=True, score=None,
                                  reason=self._fatal, flags={"evaluator_error": True})
        payload = json.dumps(completion_vars(c), default=str)
        async with self._lock:
            try:
                if self._proc is None or self._proc.returncode is not None:
                    await self._spawn()
                assert self._proc is not None and self._proc.stdin and self._proc.stdout
                self._proc.stdin.write((payload + "\n").encode())
                await self._proc.stdin.drain()
                raw = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=PY_CALL_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                await self._kill()
                return ev.EvalOutcome(
                    id=name, passed=True, score=None,
                    reason=f"evaluator timed out after {PY_CALL_TIMEOUT_S:g}s",
                    flags={"evaluator_error": True, "timeout": True},
                )
            except Exception as exc:  # noqa: BLE001
                await self._kill()
                return ev.EvalOutcome(
                    id=name, passed=True, score=None,
                    reason=f"evaluator process failed: {exc.__class__.__name__}: {exc}",
                    flags={"evaluator_error": True},
                )

        if not raw:
            # Child died (rlimit kill, syntax error in user code, sys.exit).
            await self._kill()
            return ev.EvalOutcome(
                id=name, passed=True, score=None,
                reason="evaluator process exited (check the code, or it hit the memory/CPU limit)",
                flags={"evaluator_error": True},
            )
        try:
            msg = json.loads(raw.decode("utf-8", "replace"))
        except json.JSONDecodeError:
            return ev.EvalOutcome(id=name, passed=True, score=None,
                                  reason="evaluator returned unreadable output",
                                  flags={"evaluator_error": True})

        if msg.get("fatal"):
            # A broken definition never becomes valid — stop retrying it.
            self._fatal = f"evaluator invalid: {msg['fatal']}"
            await self._kill()
            return ev.EvalOutcome(id=name, passed=True, score=None,
                                  reason=self._fatal, flags={"evaluator_error": True})
        if msg.get("error"):
            return ev.EvalOutcome(id=name, passed=True, score=None,
                                  reason=f"evaluator raised {msg['error']}",
                                  flags={"evaluator_error": True})
        return normalize_result(msg.get("result"), self.spec.fail_when_true, name)

    async def close(self) -> None:
        await self._kill()


# --------------------------------------------------------------------------- #
# Validation used by the API before anything is stored or run
# --------------------------------------------------------------------------- #


def validate_spec(
    mode: str,
    code: str,
    *,
    allow_python: bool,
    config: Optional[dict[str, Any]] = None,
) -> None:
    """Raise CustomEvalError if this evaluator can't be stored/run as written."""
    if mode not in MODES:
        raise CustomEvalError(f"mode must be one of {', '.join(MODES)}")
    if mode == "api":
        # api mode carries no code — the endpoint is the logic.
        validate_api_config(config or {})
        return
    if not (code or "").strip():
        raise CustomEvalError("the evaluator has no code")
    if mode == "expression":
        validate_expression(code, _VAR_NAMES)
        return
    if not allow_python:
        raise CustomEvalError(
            "python-mode evaluators are not available to you. They execute arbitrary "
            "code on the gateway host, so they require admin role"
            f" (and are blocked entirely when {PYTHON_MODE_ENV}=0)."
        )
    if len(code) > 20000:
        raise CustomEvalError("python evaluator too long (20000 char limit)")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CustomEvalError(f"syntax error on line {exc.lineno}: {exc.msg}") from None
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "check" not in names:
        raise CustomEvalError("define a function named check(c) that returns a bool or a dict")


def describe_context() -> dict[str, Any]:
    """What an author can reference — served to the UI so the help text and the
    validator can never disagree."""
    sample = completion_vars(ev.Completion())
    return {
        "variables": [
            {"name": k, "type": type(v).__name__ if v is not None else "str | None"}
            for k, v in sorted(sample.items())
        ],
        "helpers": sorted(HELPERS.keys()),
        "safe_methods": sorted(SAFE_METHODS),
        "modes": list(MODES),
        "python_enabled": python_mode_enabled(),
        "python_env_var": PYTHON_MODE_ENV,
        # api mode: the request body is exactly the `variables` above (plus an
        # `evaluator` name), and the response is read via these dotted paths.
        "api_defaults": {
            k: v for k, v in DEFAULT_API_CONFIG.items() if k != "api_key_secret"
        },
        "examples": [
            {
                "name": "no-markdown-fence",
                "mode": "expression",
                "fail_when_true": True,
                "code": 're_search("```", content)',
                "note": "Fails a reply that wraps its output in a code fence.",
            },
            {
                "name": "confidence-in-range",
                "mode": "expression",
                "fail_when_true": False,
                "code": 'is_json_object(content) and 0 <= (json_loads(content).get("confidence") or -1) <= 1',
                "note": "Passes only when the reply is a JSON object with a sane confidence.",
            },
            {
                "name": "answers-in-malay",
                "mode": "expression",
                "fail_when_true": False,
                "code": 're_search("(saya|anda|tidak|boleh|akaun)", content)',
                "note": "Cheap language check without a judge call.",
            },
            {
                "name": "short-and-single-paragraph",
                "mode": "expression",
                "fail_when_true": False,
                "code": 'word_count(content) <= 80 and content.count("\\n\\n") == 0',
            },
            {
                "name": "external-scorer (api)",
                "mode": "api",
                "fail_when_true": False,
                "code": "",
                "note": (
                    "POSTs the completion to your endpoint and reads {passed, score, reason} "
                    "out of the JSON reply. Nothing runs on the gateway."
                ),
            },
            {
                "name": "scored-check (python)",
                "mode": "python",
                "fail_when_true": False,
                "code": (
                    "def check(c):\n"
                    "    text = c[\"content\"]\n"
                    "    bullets = text.count(\"\\n- \")\n"
                    "    return {\n"
                    "        \"passed\": bullets >= 3,\n"
                    "        \"score\": bullets,\n"
                    "        \"reason\": f\"only {bullets} bullets\" if bullets < 3 else None,\n"
                    "        \"flags\": {\"bullets\": bullets},\n"
                    "    }\n"
                ),
                "note": "Return a dict to record a numeric score and your own flags.",
            },
        ],
    }
