#!/usr/bin/env python3
"""Minimal OpenAI-compatible relay with a fixed upstream and per-run limits."""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

UPSTREAMS = {
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "opencode-go": "https://opencode.ai/inference/go/openai/v1/chat/completions",
}
PI_MAX_OUTPUT_TOKENS = 32000  # pi's own per-request ceiling
_MAX_USAGE_SCAN_BYTES = 16_777_216


class ProxyState:
    def __init__(self, model: str, call_budget: int, max_output_tokens: int | None = None,
        ledger_path: str | None = None, provider: str = "openrouter",
        upstream_session_id: str | None = None) -> None:
        if provider not in UPSTREAMS:
            raise ValueError("unsupported_provider")
        if provider == "opencode-go" and (not isinstance(upstream_session_id, str)
                or not upstream_session_id.startswith("ses_")):
            raise ValueError("upstream_session_id_required")
        self.model = model
        self.remaining = call_budget
        self.max_output_tokens = max_output_tokens
        self.ledger_path = ledger_path
        self.provider = provider
        self.upstream_url = UPSTREAMS[provider]
        self.upstream_session_id = upstream_session_id
        self.calls = 0
        self.lock = threading.Lock()

    def next_call(self) -> int:
        with self.lock:
            self.calls += 1
            return self.calls

    def record(self, row: dict) -> None:
        if self.ledger_path is None:
            return
        with self.lock:
            with open(self.ledger_path, "a", encoding="utf-8") as ledger:
                ledger.write(json.dumps(row, sort_keys=True) + "\n")


def authorize_request(path: str, payload: dict, state: ProxyState) -> tuple[int, dict | None]:
    if path != "/v1/chat/completions":
        return 403, None
    if not isinstance(payload, dict) or payload.get("model") != state.model:
        return 403, None
    with state.lock:
        if state.remaining <= 0:
            return 429, None
        state.remaining -= 1
    output_ceiling = (state.max_output_tokens if state.max_output_tokens is not None
        else PI_MAX_OUTPUT_TOKENS)
    for key in ("max_tokens", "max_completion_tokens"):
        if key in payload:
            value = payload[key]
            if type(value) is not int or value < 1:
                return 400, None
            payload[key] = min(value, output_ceiling)
    if (state.max_output_tokens is not None and "max_tokens" not in payload
            and "max_completion_tokens" not in payload):
        payload["max_tokens"] = state.max_output_tokens
    usage_options = payload.get("usage")
    if state.provider == "openrouter":
        payload["usage"] = ({**usage_options, "include": True}
            if isinstance(usage_options, dict) else {"include": True})
    elif isinstance(usage_options, dict):
        usage_options.pop("include", None)
        if not usage_options:
            payload.pop("usage", None)
    return 200, payload


def build_upstream_request(state: ProxyState, payload: dict, api_key: str) -> Request:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if state.provider == "opencode-go":
        headers["User-Agent"] = "review-bot-agent/0.1.0"
        headers["x-opencode-session"] = state.upstream_session_id
    return Request(state.upstream_url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")


def usage_record(payload: object) -> dict | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("usage"), dict):
        return None
    usage = payload["usage"]
    result = {}
    for source, destination in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = usage.get(source)
        if type(value) is int and value >= 0:
            result[destination] = value
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
        if type(reasoning) is int and reasoning >= 0:
            result["reasoning_tokens"] = reasoning
    cost = usage.get("cost")
    if isinstance(cost, dict):
        result["cost"] = {str(key): value for key, value in cost.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}
    elif isinstance(cost, (int, float)) and not isinstance(cost, bool):
        result["cost"] = {"total": cost}
    opencode = usage.get("opencode")
    if isinstance(opencode, dict):
        opencode_cost = result.get("cost", {})
        if not isinstance(opencode_cost, dict):
            opencode_cost = {}
        for source, destination in (("list_cost", "list_cost"), ("allowance", "allowance")):
            value = opencode.get(source)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                opencode_cost[destination] = value
        if opencode_cost:
            result["cost"] = opencode_cost
        billing_source = opencode.get("billing_source")
        if isinstance(billing_source, str):
            result["billing_source"] = billing_source
    return result or None


class ResponseUsageCapture:
    def __init__(self, content_type: str) -> None:
        self.event_stream = "text/event-stream" in content_type.lower()
        self.pending = bytearray()
        self.raw = bytearray()
        self.usage: dict | None = None
        self.finish_reason: str | None = None

    def _capture(self, value: object) -> None:
        self.usage = usage_record(value) or self.usage
        if isinstance(value, dict):
            for choice in value.get("choices", []):
                if isinstance(choice, dict) and choice.get("finish_reason") in (
                    "stop", "length", "tool_calls", "content_filter", "function_call"):
                    self.finish_reason = choice["finish_reason"]

    def feed(self, chunk: bytes) -> None:
        if self.event_stream:
            self.pending.extend(chunk)
            while b"\n" in self.pending:
                line, _, remainder = self.pending.partition(b"\n")
                self.pending[:] = remainder
                self._event_line(bytes(line))
        elif len(self.raw) < _MAX_USAGE_SCAN_BYTES:
            self.raw.extend(chunk[:_MAX_USAGE_SCAN_BYTES - len(self.raw)])

    def _event_line(self, line: bytes) -> None:
        line = line.strip()
        if not line.startswith(b"data:"):
            return
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            return
        try:
            value = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        self._capture(value)

    def result(self) -> dict | None:
        if self.event_stream:
            if self.pending:
                self._event_line(bytes(self.pending))
            return {**(self.usage or {}), **({"finish_reason": self.finish_reason} if self.finish_reason else {})} or None
        if len(self.raw) >= _MAX_USAGE_SCAN_BYTES:
            return self.usage
        try:
            self._capture(json.loads(self.raw))
            return {**(self.usage or {}), **({"finish_reason": self.finish_reason} if self.finish_reason else {})} or None
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self.usage


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
            key = os.environ.get("UPSTREAM_API_KEY", "")
            if not key:
                self._reply(503, {"error": "upstream_unavailable"})
                return
            request = build_upstream_request(self.state, authorized, key)
            call = self.state.next_call()
            started = time.monotonic()
            started_at = round(time.time() * 1000)
            # Survive proxy/VM teardown while an upstream call is still pending.
            self.state.record({"call": call, "status": "in_flight", "started_at_unix_ms": started_at})
            usage = None
            status_code = 502
            try:
                response = urlopen(request, timeout=600)
            except HTTPError as error:
                response = error
            except (URLError, TimeoutError):
                self.state.record({"call": call, "status": status_code,
                    "duration_ms": int((time.monotonic() - started) * 1000)})
                self._reply(502, {"error": "upstream_unavailable"})
                return
            try:
                status_code = getattr(response, "status", None) or getattr(response, "code", 502)
                content_type = response.headers.get("Content-Type", "application/json")
                capture = ResponseUsageCapture(content_type)
                first_byte_ms = None
                self.send_response(status_code)
                self.send_header("Content-Type", content_type)
                length_header = response.headers.get("Content-Length")
                if length_header and length_header.isdigit():
                    self.send_header("Content-Length", length_header)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = response.read1(65536) if hasattr(response, "read1") else response.read(65536)
                    if not chunk:
                        break
                    if first_byte_ms is None:
                        first_byte_ms = round((time.monotonic() - started) * 1000)
                    capture.feed(chunk)
                    self.wfile.write(chunk)
                    self.wfile.flush()
                usage = capture.result()
                self.close_connection = True
            finally:
                response.close()
                self.state.record({"call": call, "status": status_code, "duration_ms": int(
                    (time.monotonic() - started) * 1000), "first_byte_ms": first_byte_ms,
                    **(capture.result() or {})})
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
    max_output_tokens = os.environ.get("MAX_OUTPUT_TOKENS")
    state = ProxyState(os.environ["MODEL_ID"], int(os.environ["MAX_PROXY_CALLS"]),
        int(max_output_tokens) if max_output_tokens else None,
        os.environ.get("USAGE_LEDGER_PATH"),
        os.environ.get("PROVIDER", "openrouter"), os.environ.get("UPSTREAM_SESSION_ID"))
    handler = type("ConfiguredProxyHandler", (ProxyHandler,), {"state": state})
    ThreadingHTTPServer(("0.0.0.0", 8080), handler).serve_forever()


if __name__ == "__main__":
    main()
