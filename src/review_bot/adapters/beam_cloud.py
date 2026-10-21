"""Explicit beam:cloud bridge through local Ollama; JSON is checked locally.

Cloud grammar enforcement is not assumed. Every effective prompt contains the
entire response schema, and native one-token evaluations meter that same prompt.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import ipaddress
import json
import math
import threading
import time
from urllib.parse import urlsplit

import httpx

from review_bot.adapters.llamacpp import _check_output_fields, _json_object
from review_bot.adapters.model_http import mark_model_request_sent, model_request_slot
from review_bot.application.architecture_review import (
    ModelOutputLimitError, ModelProfile, PermanentModelError, TransientModelError,
)
from review_bot.domain.models_v1 import ModelRequest, ModelResponse

BEAM_CLOUD_MODEL = "beam:cloud"
BEAM_CONTEXT_TOKENS = 1048576
BEAM_MAX_OUTPUT_TOKENS = 32678
_TRANSPORT_BYTES = 16 * 1024 * 1024
_METADATA_SECONDS = 60
_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0}


def _encode(value):
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PermanentModelError("Invalid beam cloud request JSON") from None


def _object(value):
    try:
        return _json_object(value)
    except PermanentModelError:
        raise PermanentModelError("Invalid beam cloud response JSON") from None


def _positive(value):
    return type(value) is int and value > 0


def _endpoint(value):
    message = "Beam cloud requires an uncredentialed loopback Ollama HTTP origin"
    if (not isinstance(value, str) or not value.isascii() or any(ord(c) <= 32 for c in value)
            or any(c in value for c in "\\?#%")):
        raise ValueError(message)
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
        local = host == "localhost" or host is not None and ipaddress.ip_address(host).is_loopback
        if (not local or parts.scheme != "http" or parts.username is not None or parts.password is not None
                or parts.path not in ("", "/") or port is not None and not 0 < port <= 65535):
            raise ValueError(message)
        return str(httpx.URL(value)).rstrip("/")
    except (ValueError, httpx.InvalidURL):
        raise ValueError(message) from None


class BeamCloudModelClient:
    """One explicit cloud role; metadata checks never generate model tokens."""

    supports_native_input_measurement = True
    supports_bounded_reasoning = False

    def __init__(self, model: str = BEAM_CLOUD_MODEL, *, api_url: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 60, max_output_bytes: int = 32768, max_attempts: int = 1,
    ) -> None:
        if model != BEAM_CLOUD_MODEL:
            raise ValueError("The explicit beam:cloud model is required")
        if not _positive(timeout_seconds) or not math.isfinite(timeout_seconds):
            raise ValueError("Beam cloud timeout must be positive and finite")
        if not _positive(max_output_bytes) or max_output_bytes > _TRANSPORT_BYTES or not _positive(max_attempts):
            raise ValueError("Beam cloud output and attempt limits must be positive and bounded")
        self.model, self.api_url = model, _endpoint(api_url)
        self.timeout_seconds = timeout_seconds
        self.profile = ModelProfile("ollama-cloud", model, max_input_bytes=_TRANSPORT_BYTES,
            max_output_bytes=max_output_bytes, timeout_seconds=timeout_seconds, max_attempts=max_attempts,
            max_input_tokens=BEAM_CONTEXT_TOKENS - BEAM_MAX_OUTPUT_TOKENS,
            max_output_tokens=BEAM_MAX_OUTPUT_TOKENS)
        self._verified_until = 0.0
        self._memo = {}
        self._lock = threading.RLock()
        self._receipts = threading.local()

    @property
    def token_count_receipt(self):
        return deepcopy(getattr(self._receipts, "value", None))

    def _receipt(self, delivery, calls=0, usage=None):
        self._receipts.value = {"delivery": delivery, "model_calls": calls,
            "usage": deepcopy(_ZERO_USAGE if usage is None and calls == 0 else usage)}

    def _deadline(self, request=None):
        timeout = self.timeout_seconds
        if request is not None:
            if not isinstance(request, ModelRequest) or not _positive(request.timeout_seconds):
                raise PermanentModelError("Invalid beam cloud request timeout")
            timeout = min(timeout, request.timeout_seconds)
        return time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Beam cloud request timed out")
        return remaining

    @staticmethod
    def _usage(data, maximum):
        if (data.get("model") not in (BEAM_CLOUD_MODEL, "beam") or data.get("done") is not True
                or data.get("done_reason") not in ("stop", "length")):
            raise PermanentModelError("Invalid beam cloud completion identity or state")
        prompt, output = data.get("prompt_eval_count"), data.get("eval_count")
        if (not _positive(prompt) or type(output) is not int or not 0 <= output <= maximum
                or prompt > BEAM_CONTEXT_TOKENS):
            raise PermanentModelError("Missing or invalid beam cloud actual token usage")
        cached = data.get("prompt_eval_cached_count", 0)
        if type(cached) is not int or not 0 <= cached <= prompt:
            raise PermanentModelError("Invalid beam cloud cached prompt usage")
        return {"input_tokens": prompt, "output_tokens": output}

    def _request(self, method, path, payload, deadline, *, measurement=False):
        content = _encode(payload) if payload is not None else None
        if content is not None and len(content) > _TRANSPORT_BYTES:
            raise PermanentModelError("Beam cloud input exceeds transport byte limit")
        try:
            with model_request_slot(self, deadline, generates=path == "/api/chat") as (ticket, deadline):
                try:
                    with httpx.Client(trust_env=False, follow_redirects=False,
                            timeout=self._remaining(deadline)) as client:
                        if measurement:
                            self._receipt("unknown", None)
                        mark_model_request_sent(ticket)
                        with client.stream(method, self.api_url + path, content=content,
                                headers={"Content-Type": "application/json", "Accept": "application/json"},
                                timeout=self._remaining(deadline)) as response:
                            self._remaining(deadline)
                            body = bytearray()
                            for chunk in response.iter_bytes():
                                self._remaining(deadline)
                                if len(body) + len(chunk) > _TRANSPORT_BYTES:
                                    raise PermanentModelError("Beam cloud response exceeds transport byte limit")
                                body.extend(chunk)
                            self._remaining(deadline)
                            if response.status_code != 200:
                                if response.status_code == 429:
                                    try:
                                        rejected = _object(bytes(body)) == {"error": "too many concurrent requests"}
                                    except PermanentModelError:
                                        rejected = False
                                    if rejected:
                                        # This completed admission rejection frees only this
                                        # ticket. Delivery accounting and usage stay unknown.
                                        ticket.finish()
                                error = TransientModelError if response.status_code in (408, 429) or response.status_code >= 500 else PermanentModelError
                                raise error(f"Beam cloud HTTP {response.status_code}")
                            data = _object(bytes(body))
                            if path == "/api/chat":
                                # Unknown consumption keeps its generating ticket uncertain.
                                usage = self._usage(data, payload["options"]["num_predict"])
                                if measurement:
                                    self._receipt("sent", 1, usage)
                            ticket.finish()
                except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                    ticket.finish()
                    if measurement:
                        self._receipt("not_sent")
                    raise
        except httpx.TimeoutException:
            raise TimeoutError("Beam cloud request timed out") from None
        except httpx.DecodingError:
            raise PermanentModelError("Invalid beam cloud response encoding") from None
        except httpx.RequestError:
            raise TransientModelError("Beam cloud connection failed") from None
        self._remaining(deadline)
        return data

    def _verify(self, deadline):
        with self._lock:
            if time.monotonic() < self._verified_until:
                return
            tags = self._request("GET", "/api/tags", None, deadline)
            models = tags.get("models")
            candidates = [item for item in models if isinstance(item, dict) and item.get("name") == self.model] if isinstance(models, list) else []
            if (len(candidates) != 1 or candidates[0].get("model") != self.model
                    or candidates[0].get("remote_model") != "beam"
                    or candidates[0].get("remote_host") not in ("https://ollama.com", "https://ollama.com:443")):
                raise PermanentModelError("Ollama does not expose the exact pinned beam cloud identity")
            data = self._request("POST", "/api/show", {"model": self.model}, deadline)
            info, details, capabilities = data.get("model_info"), data.get("details"), data.get("capabilities")
            if (not isinstance(info, dict) or info.get("general.architecture") != "beam"
                    or type(info.get("beam.context_length")) is not int or info["beam.context_length"] != BEAM_CONTEXT_TOKENS
                    or not isinstance(details, dict) or details.get("parent_model") != "beam" or details.get("family") != "beam"
                    or not isinstance(capabilities, list) or "completion" not in capabilities
                    or data.get("remote_model") not in (None, "beam")
                    or data.get("remote_host") not in (None, "https://ollama.com", "https://ollama.com:443")):
                raise PermanentModelError("Unsupported beam cloud metadata or context")
            self._memo.clear()
            self._verified_until = min(deadline, time.monotonic() + _METADATA_SECONDS)

    def verify(self):
        self._verify(self._deadline())

    def warmup(self):
        self.verify()

    def _payload(self, request):
        if (not isinstance(request, ModelRequest) or request.model != self.model
                or not isinstance(request.system_prompt, str) or not isinstance(request.user_content, str)
                or not isinstance(request.response_schema, dict) or request.reasoning_budget_tokens is not None):
            raise PermanentModelError("Invalid beam cloud request contract")
        maximum = request.max_output_tokens if request.max_output_tokens is not None else BEAM_MAX_OUTPUT_TOKENS
        if not _positive(maximum) or maximum > BEAM_MAX_OUTPUT_TOKENS:
            raise PermanentModelError("Invalid beam cloud output reservation")
        schema = _encode(request.response_schema).decode("utf-8")
        contract = "\n\nReturn exactly one JSON object matching this entire response schema. Do not add prose or fields.\n" + schema
        payload = {"model": self.model, "messages": [
                {"role": "system", "content": request.system_prompt + contract},
                {"role": "user", "content": request.user_content}],
            "format": request.response_schema, "stream": False, "think": False, "truncate": False, "shift": False,
            "options": {"num_ctx": BEAM_CONTEXT_TOKENS, "num_predict": maximum, "temperature": 0, "seed": 0}}
        if len(_encode(payload)) > _TRANSPORT_BYTES:
            raise PermanentModelError("Beam cloud input exceeds transport byte limit")
        return payload

    def _measure(self, payload, deadline):
        with self._lock:
            self._verify(deadline)
            key = sha256(_encode(payload)).hexdigest()
            if key in self._memo:
                self._receipt("not_sent")
                return self._memo[key]
            measured_payload = deepcopy(payload)
            measured_payload["options"]["num_predict"] = 1
            data = self._request("POST", "/api/chat", measured_payload, deadline, measurement=True)
            count = self._usage(data, 1)["input_tokens"]
            if "truncated" in data and data["truncated"] is not False:
                raise PermanentModelError("Beam cloud input truncation is forbidden")
            # Bounded, per-instance memo; never persisted or shared across metadata refreshes.
            if len(self._memo) >= 128:
                self._memo.clear()
            self._memo[key] = count
            return count

    def count_input_tokens(self, request):
        self._receipt("not_sent")
        return self._measure(self._payload(request), self._deadline(request))

    def complete(self, request):
        deadline = self._deadline(request)
        payload = self._payload(request)
        with self._lock:
            key = sha256(_encode(payload)).hexdigest()
            if time.monotonic() >= self._verified_until or key not in self._memo:
                error = PermanentModelError("Beam cloud requires a fresh exact native input measurement before generation")
                error.delivery, error.usage = "not_sent", deepcopy(_ZERO_USAGE)
                raise error
            measured = self._memo[key]
        if measured > self.profile.max_input_tokens or measured + payload["options"]["num_predict"] > BEAM_CONTEXT_TOKENS:
            error = PermanentModelError("Beam cloud input exceeds actual token reservation; truncation is forbidden")
            error.delivery, error.usage = "not_sent", deepcopy(_ZERO_USAGE)
            raise error
        data = self._request("POST", "/api/chat", payload, deadline)
        usage = self._usage(data, payload["options"]["num_predict"])
        try:
            if usage["input_tokens"] != measured or ("truncated" in data and data["truncated"] is not False):
                raise PermanentModelError("Beam cloud usage does not match the full native prompt measurement")
            message = data.get("message")
            if (not isinstance(message, dict) or message.get("role") != "assistant"
                    or message.get("tool_calls") or message.get("refusal") is not None):
                raise PermanentModelError("Invalid beam cloud assistant output")
            text, thinking = message.get("content"), message.get("thinking")
            if (not isinstance(text, str) and not (data["done_reason"] == "length" and text is None)
                    or thinking is not None and not isinstance(thinking, str)):
                raise PermanentModelError("Invalid beam cloud content or thinking metadata")
            if data["done_reason"] == "length":
                error = ModelOutputLimitError("Beam cloud reached its output limit")
                error.rejected_output = {"finish_reason": "length", "content": text, "reasoning_content": thinking}
                raise error
            if len(text.encode("utf-8")) > self.profile.max_output_bytes:
                raise PermanentModelError("Beam cloud content exceeds configured output byte limit")
            value = _object(text)
            _check_output_fields(value, request.response_schema)
            self._remaining(deadline)
            return ModelResponse(text, usage)
        except (PermanentModelError, TimeoutError) as error:
            error.delivery, error.usage = "sent", usage
            raise
