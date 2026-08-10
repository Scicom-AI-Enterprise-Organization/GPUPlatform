#!/usr/bin/env python3
"""A fake OpenAI-compatible endpoint, so you can run a real experiment for free.

Every Experiments target is a plain `{base_url, model, key}` — there is no
"platform target" code path — which means a mock endpoint is a first-class
target. This one exists so the tutorial (`docs/EXPERIMENTS_TUTORIAL.md`) is
runnable end to end with no GPU, no pod and no bill.

    .venv/bin/python sandbox/mock_model.py        # → http://127.0.0.1:8078/v1

Serves `POST /v1/chat/completions` (streaming and not) plus `GET /v1/models`.

**The models are the point.** Each one fails in a way a different detector is
built to catch, so a two-target experiment shows real red and green instead of
a uniform wall of passes:

    good        clean, well-formed replies                     → everything passes
    leaky       emits `<|channel|>` control markers            → control_token_leak
    empty       returns "" with 0 completion tokens            → empty_response
    loopy       repeats one phrase ~40 times                   → degeneration
    fenced      wraps JSON in a ```json fence                  → json_output
    flaky       fails ~1 request in 3 with HTTP 500            → request_error
    toolish     always answers with a tool call, then text     → for sandbox runs

`toolish` is what makes a sandbox demo work: give it `tools` and it calls one,
which the sandbox answers, and it then produces a final text reply.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

MODELS = ("good", "leaky", "empty", "loopy", "fenced", "flaky", "toolish")

app = FastAPI(title="sgpu mock model", version="1.0")


class ChatRequest(BaseModel):
    # Permissive for the same reason as the sandbox server: a 422 would reach the
    # runner as an HTTP error and read like a broken endpoint.
    model_config = ConfigDict(extra="allow")
    model: str = "good"
    messages: list[dict[str, Any]] = Field(default_factory=list)
    tools: Optional[list[dict[str, Any]]] = None
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None


def _last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):  # multimodal content parts
                return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
            return str(c or "")
    return ""


def _reply(model: str, messages: list[dict[str, Any]], has_tools: bool) -> dict[str, Any]:
    """(content, reasoning, tool_calls, finish_reason, completion_tokens)."""
    user = _last_user(messages)
    # Deterministic per prompt so two runs of the same experiment agree.
    seed = zlib.crc32(user.encode()) if user else 0

    if model == "leaky":
        return {"content": f"<|channel|>thought I should answer.<channel|>{user[:60] or 'Hello'}!",
                "tokens": 24}
    if model == "empty":
        return {"content": "", "tokens": 0, "finish_reason": "stop"}
    if model == "loopy":
        return {"content": ("I can help with that. " * 40).strip(), "tokens": 200}
    if model == "fenced":
        return {"content": '```json\n{"intent": "billing", "confidence": 0.91}\n```', "tokens": 30}
    if model == "toolish":
        # A tool-calling turn ONLY when tools are offered AND none has answered
        # yet; once a role=tool result is in the conversation, produce the final
        # text. That's what lets a sandbox loop terminate.
        answered = any(m.get("role") == "tool" for m in messages)
        if has_tools and not answered:
            return {"tool_calls": [{
                "id": "call_mock_1", "type": "function",
                "function": {"name": "get_balance",
                             "arguments": json.dumps({"account_id": "1001"})},
            }], "content": "", "tokens": 18, "finish_reason": "tool_calls"}
        return {"content": "Your outstanding balance is RM 128.40, due 22 Aug 2026.", "tokens": 22}
    # good
    return {"content": f"Sure — {user[:80] or 'how can I help?'}", "tokens": 20 + (seed % 12)}


def _envelope(model: str, r: dict[str, Any], prompt_tokens: int) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": r.get("content") or ""}
    if r.get("tool_calls"):
        msg["tool_calls"] = r["tool_calls"]
    return {
        "id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": r.get("finish_reason") or "stop"}],
        "usage": {"prompt_tokens": prompt_tokens,
                  "completion_tokens": int(r.get("tokens") or 0),
                  "total_tokens": prompt_tokens + int(r.get("tokens") or 0)},
    }


@app.get("/v1/models")
async def models() -> dict:
    return {"object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    model = req.model or "good"
    prompt_tokens = sum(len(str(m.get("content") or "")) // 4 for m in req.messages) or 8

    # `flaky` fails a deterministic ~1-in-3 of distinct prompts — enough to make
    # request_error and the error-rate column light up without being useless.
    if model == "flaky" and zlib.crc32(_last_user(req.messages).encode()) % 3 == 0:
        return JSONResponse(status_code=500, content={"error": {"message": "mock upstream error"}})

    r = _reply(model, req.messages, bool(req.tools))

    if not req.stream:
        return JSONResponse(_envelope(model, r, prompt_tokens))

    def sse():
        base = {"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model}
        if r.get("tool_calls"):
            # Streamed tool calls arrive fragmented; the runner reassembles them
            # by index (`_merge_stream_tool_calls`), so send them that way.
            for i, tc in enumerate(r["tool_calls"]):
                frag = {"index": i, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": ""}}
                yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [frag]}}]})}\n\n"
                for ch in tc["function"]["arguments"]:
                    part = {"index": i, "function": {"arguments": ch}}
                    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'tool_calls': [part]}}]})}\n\n"
        for word in (r.get("content") or "").split(" "):
            delta = {"content": word + " "}
            yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta}]})}\n\n"
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': r.get('finish_reason') or 'stop'}]})}\n\n"
        # The runner injects stream_options.include_usage, so always emit usage —
        # without it every token-derived metric silently reads zero.
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": int(r.get("tokens") or 0),
                 "total_tokens": prompt_tokens + int(r.get("tokens") or 0)}
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8078)
    a = ap.parse_args()
    import uvicorn
    print(f"mock model on http://{a.host}:{a.port}/v1")
    print(f"  base URL for a target:  http://{a.host}:{a.port}/v1")
    print(f"  models:                 {', '.join(MODELS)}")
    print("Ctrl-C to stop.\n")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
