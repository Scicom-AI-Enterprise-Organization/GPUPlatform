#!/usr/bin/env python3
"""A tiny tool-response server for testing Experiments' `api`-mode sandboxes.

A sandbox is the thing that ANSWERS a model's tool call during a replay
(`gateway/gateway/sandbox.py`). `mode=api` POSTs each call to a service you run
and reads the result out of the JSON — this is a stand-in for that service, so
you can wire up `/experiments/sandboxes/new` and see the whole path work before
pointing it at anything real.

FastAPI, so the request contract is typed and `/docs` documents it. Uses the
gateway venv (fastapi + uvicorn are already there — no install):

    .venv/bin/python sandbox/server.py        # listens on 127.0.0.1:8077

Then on /experiments/sandboxes/new:

    Mode         HTTP endpoint
    Endpoint URL http://127.0.0.1:8077/tool
    Result path  content
    Tool         get_balance          ← in the Test card, then press Test

⚠ `ToolRequest` is deliberately permissive (`extra="allow"`, everything
optional). A 422 from request validation would reach the gateway as
`sandbox_http_error` and read like a broken endpoint rather than a shape
mismatch, so this server accepts whatever arrives and decides for itself.

⚠ `row` carries the dataset row's `expected` block — the gold answer the
evaluators grade against. It is off by default (`send_expected`) for a reason: a
simulator that can read it can return exactly the reference result and inflate
the score. This server logs whether it arrived, so you can confirm the toggle
does what you think it does.

**Fault injection.** Call one of these tool names to exercise the error paths the
gateway has to survive — each maps to a distinct failure it reports:

    _http_500   → HTTP 500                     (sandbox_http_error)
    _not_json   → 200 with a non-JSON body     (sandbox_bad_response, unless
                                                the result path is blank)
    _wrong_key  → 200 with JSON lacking the
                  configured result path        (sandbox_bad_response)
    _slow       → sleeps 30s                    (timeout → sandbox_unreachable)
    _empty      → 200 with an empty string      (a valid, empty tool result)

Anything else that isn't a known tool comes back as a structured "unknown tool"
result — a normal 200, because an unknown call is something the MODEL should get
to react to, not a transport failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

# Set from the CLI; empty = no auth.
API_KEY = ""
QUIET = False

# --------------------------------------------------------------------------- #
# The fake world
# --------------------------------------------------------------------------- #
#
# Deterministic on purpose: the same (name, arguments) must always produce the
# same answer, or two variants in one experiment face different worlds and the
# comparison measures simulator luck rather than model quality. The gateway
# caches responses per run for exactly this reason — this server not being
# random keeps it true ACROSS runs too.

ACCOUNTS: dict[str, dict[str, Any]] = {
    "1001": {"name": "Aina Rahman", "plan": "Fibre 500", "balance": 128.40, "due": "2026-08-22"},
    "1002": {"name": "Wei Ling Tan", "plan": "Mobile 5G Max", "balance": 0.00, "due": None},
    "1003": {"name": "Suresh Kumar", "plan": "Fibre 100", "balance": 89.90, "due": "2026-08-15"},
}
PLANS = [
    {"id": "fibre-100", "name": "Fibre 100", "speed_mbps": 100, "price_myr": 89.90},
    {"id": "fibre-500", "name": "Fibre 500", "speed_mbps": 500, "price_myr": 159.00},
    {"id": "mobile-5g-max", "name": "Mobile 5G Max", "data_gb": None, "price_myr": 98.00},
]


def _account(args: dict) -> Optional[dict]:
    acc = str(args.get("account_id") or args.get("account") or args.get("id") or "").strip()
    return ACCOUNTS.get(acc)


def tool_get_balance(args: dict) -> Any:
    acc = _account(args)
    if acc is None:
        return {"error": "unknown_account", "known": sorted(ACCOUNTS)}
    return {"balance_myr": acc["balance"], "due_date": acc["due"], "plan": acc["plan"]}


def tool_get_bill(args: dict) -> Any:
    acc = _account(args)
    if acc is None:
        return {"error": "unknown_account", "known": sorted(ACCOUNTS)}
    month = str(args.get("month") or "latest")
    # Vary by month so a conversation that calls twice gets two answers —
    # multi-turn replays are the point of a sandbox.
    factor = {"latest": 1.0, "jan": 0.9, "feb": 1.1}.get(month.lower(), 1.0)
    return {
        "month": month,
        "amount_myr": round(acc["balance"] * factor, 2),
        "plan": acc["plan"],
        "status": "unpaid" if acc["balance"] else "paid",
    }


def tool_get_usage(args: dict) -> Any:
    acc = _account(args)
    if acc is None:
        return {"error": "unknown_account", "known": sorted(ACCOUNTS)}
    return {"data_gb_used": 412.7, "quota_gb": None, "cycle_ends": "2026-08-31"}


def tool_lookup_customer(args: dict) -> Any:
    needle = str(args.get("name") or args.get("query") or "").strip().lower()
    hits = [
        {"account_id": k, **{kk: vv for kk, vv in v.items() if kk != "balance"}}
        for k, v in ACCOUNTS.items()
        if not needle or needle in v["name"].lower()
    ]
    return {"results": hits, "count": len(hits)}


def tool_list_plans(_args: dict) -> Any:
    return {"plans": PLANS}


def tool_create_ticket(args: dict) -> Any:
    subject = str(args.get("subject") or args.get("issue") or "unspecified")
    # Deterministic id — a random one would make two runs incomparable.
    return {"ticket_id": f"TKT-{abs(hash(subject)) % 90000 + 10000}", "status": "open",
            "subject": subject}


TOOLS = {
    "get_balance": tool_get_balance,
    "get_bill": tool_get_bill,
    "get_usage": tool_get_usage,
    "lookup_customer": tool_lookup_customer,
    "list_plans": tool_list_plans,
    "create_ticket": tool_create_ticket,
}

FAULTS = ("_http_500", "_not_json", "_wrong_key", "_slow", "_empty")


# --------------------------------------------------------------------------- #
# Contract
# --------------------------------------------------------------------------- #


class ParsedCall(BaseModel):
    """The pre-parsed convenience copy the gateway sends alongside `tool_call`,
    so a fixture server doesn't have to re-implement "arguments is sometimes a
    JSON string"."""
    model_config = ConfigDict(extra="allow")
    name: str = ""
    arguments: Any = Field(default_factory=dict)


class ToolRequest(BaseModel):
    """Exactly what `sandbox.ApiProvider.payload()` posts.

    Every field is optional and extras are allowed — see the module docstring:
    a 422 here would surface to the gateway as `sandbox_http_error`.
    """
    model_config = ConfigDict(extra="allow")
    sandbox: str = ""
    conversation: list[dict[str, Any]] = Field(default_factory=list)
    tool_call: dict[str, Any] = Field(default_factory=dict)
    call: ParsedCall = Field(default_factory=ParsedCall)
    # Present ONLY when `send_expected` is on. See the warning above.
    row: Optional[dict[str, Any]] = None


class ToolResponse(BaseModel):
    """`content` is the gateway's default result path; `result.output` mirrors it
    so you can try a nested path without editing this file."""
    content: str
    result: dict[str, Any]
    tool: str


app = FastAPI(
    title="sgpu sandbox",
    description="Fake tool-response service for testing api-mode sandboxes.",
    version="1.0",
)


def _check_key(authorization: Optional[str]) -> None:
    if not API_KEY:
        return
    got = (authorization or "").removeprefix("Bearer ").strip()
    if got != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing API key")


def _args_of(call: ParsedCall) -> dict:
    args = call.arguments
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


@app.get("/health")
@app.get("/healthz")
async def health() -> dict:
    return {"ok": True, "tools": sorted(TOOLS), "faults": list(FAULTS)}


@app.post("/tool", response_model=None)
@app.post("/", response_model=None)
async def answer_tool_call(
    req: ToolRequest,
    authorization: Optional[str] = Header(default=None),
) -> Any:
    _check_key(authorization)

    name = req.call.name or str((req.tool_call.get("function") or {}).get("name") or "")
    args = _args_of(req.call)

    if not QUIET:
        # ⚠ Whether the gold reference reached us. `send_expected` is off by
        # default precisely so this prints "withheld".
        gold = "SENT ⚠" if req.row is not None else "withheld"
        print(f"→ {name}({json.dumps(args, sort_keys=True)})  "
              f"[{len(req.conversation)} msg(s) of context · expected: {gold}]", flush=True)

    # ---- fault injection -------------------------------------------------- #
    if name == "_http_500":
        return JSONResponse(status_code=500, content={"error": "injected failure"})
    if name == "_not_json":
        return PlainTextResponse("this is not JSON")
    if name == "_wrong_key":
        return JSONResponse({"something_else": "the configured result path is missing"})
    if name == "_slow":
        await asyncio.sleep(30)
        return JSONResponse({"content": "…eventually"})
    if name == "_empty":
        return JSONResponse({"content": ""})

    # ---- the real answer -------------------------------------------------- #
    fn = TOOLS.get(name)
    if fn is None:
        # A normal 200: an unknown call is something the MODEL should get to
        # react to, not a transport failure. Never fabricate a success.
        result: Any = {"error": "unknown_tool", "tool": name, "known": sorted(TOOLS)}
    else:
        result = fn(args)

    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    return ToolResponse(content=text, result={"output": text}, tool=name)


def main() -> int:
    global API_KEY, QUIET
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--api-key", default="",
                    help="require this bearer token (to test the form's API-key secret field)")
    ap.add_argument("--quiet", action="store_true", help="don't print each call")
    a = ap.parse_args()

    API_KEY, QUIET = a.api_key, a.quiet

    import uvicorn
    print(f"sandbox server on http://{a.host}:{a.port}")
    print(f"  endpoint URL for the form:  http://{a.host}:{a.port}/tool")
    print(f"  result path:                content        (or result.output)")
    print(f"  tools:                      {', '.join(sorted(TOOLS))}")
    print(f"  fault triggers:             {', '.join(FAULTS)}")
    print(f"  request contract:           http://{a.host}:{a.port}/docs")
    if a.api_key:
        print(f"  auth:                       Bearer {a.api_key}")
    print("Ctrl-C to stop.\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
