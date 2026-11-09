#!/usr/bin/env python3
"""Minimal OpenAI-compatible relay with a fixed upstream and per-run limits."""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UPSTREAM = "https://openrouter.ai/api/v1/chat/completions"


class ProxyState:
    def __init__(self, model: str, call_budget: int, max_output_tokens: int) -> None:
        self.model = model
        self.remaining = call_budget
        self.max_output_tokens = max_output_tokens
        self.lock = threading.Lock()


def authorize_request(path: str, payload: dict, state: ProxyState) -> tuple[int, dict | None]:
    if path != "/v1/chat/completions":
        return 403, None
    if not isinstance(payload, dict) or payload.get("model") != state.model:
        return 403, None
    with state.lock:
        if state.remaining <= 0:
            return 429, None
        state.remaining -= 1
    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if value is not None:
            if type(value) is not int or value < 1:
                return 400, None
            payload[key] = min(value, state.max_output_tokens)
    if "max_tokens" not in payload and "max_completion_tokens" not in payload:
        payload["max_tokens"] = state.max_output_tokens
    return 200, payload


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "ReviewBotProxy/1"
    state: ProxyState

    def do_GET(self) -> None:
        self._reply(403, {"error": "request_denied"})

    def do_PUT(self) -> None:
        self._reply(403, {"error": "request_denied"})

    def do_DELETE(self) -> None:
        self._reply(403, {"error": "request_denied"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 2_000_000:
                self._reply(400, {"error": "invalid_request"})
                return
            payload = json.loads(self.rfile.read(length))
            status, authorized = authorize_request(self.path, payload, self.state)
            if status != 200:
                if status == 429:
                    print("proxy_budget_exhausted", flush=True)
                self._reply(status, {"error": "request_denied"})
                return
            key = os.environ.get("OPENROUTER_API_KEY", "")
            if not key:
                self._reply(503, {"error": "upstream_unavailable"})
                return
            request = Request(UPSTREAM, data=json.dumps(authorized).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
            try:
                response = urlopen(request, timeout=600)
            except HTTPError as error:
                response = error
            self.send_response(getattr(response, "status", None) or getattr(response, "code", 502))
            self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
            length_header = response.headers.get("Content-Length")
            if length_header and length_header.isdigit():
                self.send_header("Content-Length", length_header)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (ValueError, TypeError, URLError, TimeoutError):
            self._reply(502, {"error": "upstream_unavailable"})

    def _reply(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    state = ProxyState(os.environ["MODEL_ID"], int(os.environ["MAX_PROXY_CALLS"]),
        int(os.environ["MAX_OUTPUT_TOKENS"]))
    handler = type("ConfiguredProxyHandler", (ProxyHandler,), {"state": state})
    ThreadingHTTPServer(("0.0.0.0", 8080), handler).serve_forever()


if __name__ == "__main__":
    main()
