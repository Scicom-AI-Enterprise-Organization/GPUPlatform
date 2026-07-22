#!/usr/bin/env python3
"""Minimal OpenAI-compatible mock upstream for local proxy-HA testing.

Pure stdlib (no pip install), so it runs on a bare python:3.x image.

Routes:
  GET  /v1/models              -> advertises one model id (env MODEL_ID)
  POST /v1/chat/completions     -> sleeps DELAY_S then returns one choice
                                   (honours "stream": true as SSE)

It tracks concurrent in-flight requests and logs the running peak, so a
cluster-wide concurrency cap is directly observable:
    docker compose ... logs mock-upstream | grep peak | tail -1

Env: MODEL_ID (default "mock"), DELAY_S (default 3), PORT (default 9000).
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get("MODEL_ID", "mock")
DELAY = float(os.environ.get("DELAY_S", "3"))
PORT = int(os.environ.get("PORT", "9000"))

_lock = threading.Lock()
_inflight = 0
_peak = 0


class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        else:
            self._json(200, {"status": "ok"})

    def do_POST(self):
        global _inflight, _peak
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception:
            req = {}
        with _lock:
            _inflight += 1
            _peak = max(_peak, _inflight)
            print(f"[mock] +1 inflight={_inflight} peak={_peak}", flush=True)
        try:
            time.sleep(DELAY)  # hold the slot so the concurrency cap is observable
            if req.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for tok in ["hello", " from", " mock"]:
                    chunk = {"choices": [{"index": 0, "delta": {"content": tok}}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                self._json(200, {
                    "id": "cmpl-mock",
                    "object": "chat.completion",
                    "model": MODEL,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello from mock"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
                })
        finally:
            with _lock:
                _inflight -= 1
                print(f"[mock] -1 inflight={_inflight} peak={_peak}", flush=True)

    def log_message(self, *args):  # silence the default per-request access log
        pass


if __name__ == "__main__":
    print(f"[mock] listening on :{PORT} model={MODEL} delay={DELAY}s", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
