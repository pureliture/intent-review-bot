"""OpenRouter chat adapter for the validation-phase review LLM.

Converted from the former Beam bridge: the transport skeleton (deadline,
admission slot, bounded byte transport, metadata verify-before-generate)
is reused, but every measurement path is deleted. There is no
``count_input_tokens``, no one-token metering request, no tokenizer
endpoint, and no memo of measured counts. Each ``complete`` performs
exactly one ``POST /api/v1/chat/completions`` after a fail-closed
byte-upper-bound pre-gate, and validates the response usage post-send.

Default model: ``recursive/flint-v1.0:free`` (256K context). The
128K production envelope (input 98,394 + output 32,678) still binds
through ``min(app_budget, model_context)``.
"""
from __future__ import annotations

import json
import math
import threading
import time
from typing import Any

import httpx

from review_bot.adapters.byte_budget import (
    ADMISSION_OUTPUT_FLOOR_TOKENS,
    CHOICE_WIRE_FLOOR_TOKENS,
    _CHOICE_REQUEST_CEILING_TOKENS,
)
from review_bot.adapters.llamacpp import _check_output_fields, _json_object
from review_bot.adapters.model_http import mark_model_request_sent, model_request_slot
from review_bot.adapters.openrouter_common import (
    _METADATA_SECONDS,
    _TRANSPORT_BYTES,
    auth_headers,
    base_url,
    check_response_usage,
    check_text_entry,
    classify_http_error,
    encode,
    fetch_models_catalog,
    find_model_entry,
    fresh,
    precheck_byte_budget,
)
from review_bot.application.architecture_review import (
    ModelInputCapacityError,
    ModelOutputLimitError,
    ModelProfile,
    PermanentModelError,
    TransientModelError,
)
from review_bot.application.model_wire_contract import compact_schema
from review_bot.domain.models_v1 import ModelRequest, ModelResponse

OPENROUTER_REVIEW_PROFILE = "openrouter-review"
REVIEW_MODEL = "recursive/flint-v1.0:free"
REVIEW_CONTEXT_TOKENS = 262144
# Production 128K envelope from the agreed app budget: input + output.
APP_INPUT_TOKENS = 98394
APP_OUTPUT_TOKENS = 32678
_CHAT_PATH = "/api/v1/chat/completions"
_ZERO_USAGE = {"input_tokens": 0, "output_tokens": 0}


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def review_wire_payload(request: ModelRequest) -> tuple[dict[str, Any], int]:
    """Pure OpenRouter chat wire builder shared by transport and fingerprinting.

    No HTTP. Raises PermanentModelError on contract violations.
    Returns (payload, max_output_tokens).
    """
    if (not isinstance(request, ModelRequest) or not isinstance(request.model, str)
            or not isinstance(request.system_prompt, str) or not isinstance(request.user_content, str)
            or not isinstance(request.response_schema, dict) or request.reasoning_budget_tokens is not None):
        raise PermanentModelError("Invalid OpenRouter review request contract")
    maximum = request.max_output_tokens if request.max_output_tokens is not None else APP_OUTPUT_TOKENS
    if type(maximum) is not int or not 0 < maximum <= APP_OUTPUT_TOKENS:
        raise PermanentModelError("Invalid OpenRouter review output reservation")
    # Minimax reasoning is mandatory and billed as completion tokens. Tiny
    # legacy reservations (e.g. 16-token closed choices) cannot fit reasoning
    # plus content, so the wire reservation is floored. The floor participates
    # in the same byte-bound pre-gate and post-send usage check.
    floored = max(maximum, ADMISSION_OUTPUT_FLOOR_TOKENS)
    if maximum <= _CHOICE_REQUEST_CEILING_TOKENS:
        floored = max(floored, CHOICE_WIRE_FLOOR_TOKENS)
    schema = compact_schema(request.response_schema)
    contract = ("\n\nReturn exactly one JSON object matching this entire response schema. "
        "Do not add prose or fields.\n"
        + encode(schema).decode("utf-8"))
    payload = {"model": request.model, "messages": [
            {"role": "system", "content": request.system_prompt + contract},
            {"role": "user", "content": request.user_content}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "review_bot", "strict": True, "schema": schema}},
        "stream": False, "temperature": 0, "max_tokens": floored,
        "reasoning": {"effort": "low"}}
    # Canonicalize through the shared encoder so the fingerprinted bytes
    # equal the transmitted bytes.
    return json.loads(encode(payload).decode("utf-8")), floored


class OpenRouterReviewClient:
    """Validation-phase review LLM; one generation per complete, never metered."""

    def __init__(self, model: str = REVIEW_MODEL, *, api_key: str = "",
            origin: str | None = None, timeout_seconds: int = 60,
            max_output_bytes: int = 32768, max_attempts: int = 1,
            app_input_tokens: int = APP_INPUT_TOKENS,
            app_output_tokens: int = APP_OUTPUT_TOKENS,
            framing_reserve_tokens: int = 8192,
            max_transport_bytes: int = _TRANSPORT_BYTES,
    ) -> None:
        if not isinstance(model, str) or not model.strip() or len(model) > 256:
            raise ValueError("An explicit OpenRouter review model is required")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("OPENROUTER_API_KEY must be set for openrouter-review")
        if not _positive(timeout_seconds) or not math.isfinite(timeout_seconds):
            raise ValueError("OpenRouter review timeout must be positive and finite")
        if not _positive(max_output_bytes) or max_output_bytes > max_transport_bytes or not _positive(max_attempts):
            raise ValueError("OpenRouter review output and attempt limits must be positive and bounded")
        if not _positive(app_input_tokens) or not _positive(app_output_tokens):
            raise ValueError("OpenRouter review app budget must be positive")
        if type(framing_reserve_tokens) is not int or framing_reserve_tokens < 0:
            raise ValueError("OpenRouter review framing reserve must be a non-negative integer")
        if type(max_transport_bytes) is not int or max_transport_bytes <= 0:
            raise ValueError("OpenRouter review transport byte limit must be positive")
        self.model = model
        self._api_key = api_key.strip()
        self.origin = base_url(origin)
        self.timeout_seconds = timeout_seconds
        self._framing_reserve_tokens = framing_reserve_tokens
        self._max_transport_bytes = max_transport_bytes
        self.profile = ModelProfile("openrouter", model, max_input_bytes=max_transport_bytes,
            max_output_bytes=max_output_bytes, timeout_seconds=timeout_seconds, max_attempts=max_attempts,
            max_input_tokens=app_input_tokens, max_output_tokens=app_output_tokens)
        self._verified_until = 0.0
        self._model_context = 0
        self._lock = threading.RLock()

    def _deadline(self, request: ModelRequest | None = None) -> float:
        timeout = self.timeout_seconds
        if request is not None:
            if not isinstance(request, ModelRequest) or not _positive(request.timeout_seconds):
                raise PermanentModelError("Invalid OpenRouter review request timeout")
            timeout = min(timeout, request.timeout_seconds)
        return time.monotonic() + timeout

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OpenRouter review request timed out")
        return remaining

    def _verify(self, deadline: float) -> int:
        with self._lock:
            if fresh(deadline, self._verified_until):
                return self._model_context
            catalog = fetch_models_catalog(self._api_key, self.origin, self._remaining(deadline))
            try:
                entry = find_model_entry(catalog, self.model)
            except PermanentModelError:
                raise PermanentModelError("OpenRouter catalog does not list the pinned review model") from None
            context = check_text_entry(entry, self.model,
                minimum_context=APP_INPUT_TOKENS + APP_OUTPUT_TOKENS)
            self._model_context = context
            self._verified_until = min(deadline, time.monotonic() + _METADATA_SECONDS)
            return context

    def verify(self) -> None:
        self._verify(self._deadline())

    def warmup(self) -> None:
        """Metadata verification only; never generates tokens."""
        self.verify()

    def _payload(self, request: ModelRequest) -> tuple[dict[str, Any], int]:
        if not isinstance(request, ModelRequest) or request.model != self.model:
            raise PermanentModelError("Invalid OpenRouter review request contract")
        return review_wire_payload(request)

    def admission_bytes(self, request: ModelRequest) -> int:
        """Exact chat wire bytes plus framing reserve: a proven token upper bound.

        Pure local computation; no HTTP, no measurement call.
        """
        if not isinstance(request, ModelRequest) or request.model != self.model:
            raise PermanentModelError("Invalid OpenRouter review request contract")
        payload, _ = review_wire_payload(request)
        return len(encode(payload)) + self._framing_reserve_tokens

    def complete(self, request: ModelRequest) -> ModelResponse:
        deadline = self._deadline(request)
        with self._lock:
            model_context = self._verify(deadline)
        payload, maximum = self._payload(request)
        body = encode(payload)
        if len(body) > self._max_transport_bytes:
            raise PermanentModelError("OpenRouter review input exceeds transport byte limit")
        # Fail-closed byte-upper-bound pre-gate. No measurement call exists.
        precheck_byte_budget(len(body), maximum, app_input_tokens=self.profile.max_input_tokens,
            app_output_tokens=self.profile.max_output_tokens, model_context_tokens=model_context,
            framing_reserve_tokens=self._framing_reserve_tokens, role="OpenRouter review")
        data = self._post(payload, body, deadline)
        message = data.get("choices", [{}])[0].get("message") if isinstance(data.get("choices"), list) and len(data.get("choices")) == 1 else None
        if data.get("model") != self.model or not isinstance(message, dict):
            raise PermanentModelError("Invalid OpenRouter review response identity")
        finish = data.get("choices")[0].get("finish_reason")
        text = message.get("content")
        if finish == "length" or text is None:
            error = ModelOutputLimitError("OpenRouter review reached its output limit")
            error.rejected_output = {"finish_reason": finish, "content": text}
            # Preserve the real usage for accounting/diagnosis instead of
            # zeroing it: a length truncation still consumed the reservation.
            try:
                error.usage = check_response_usage(data.get("usage"),
                    app_input_tokens=self.profile.max_input_tokens,
                    app_output_tokens=self.profile.max_output_tokens,
                    model_context_tokens=model_context,
                    max_output_tokens=maximum, role="OpenRouter review")
            except PermanentModelError:
                error.usage = dict(_ZERO_USAGE)
            error.delivery = "sent"
            raise error
        if (finish != "stop" or message.get("role") != "assistant" or not isinstance(text, str)
                or message.get("refusal") is not None):
            raise PermanentModelError("Invalid OpenRouter review assistant output")
        if len(text.encode("utf-8")) > self.profile.max_output_bytes:
            raise PermanentModelError("OpenRouter review content exceeds configured output byte limit")
        try:
            value = _json_object(text)
        except PermanentModelError:
            raise PermanentModelError("OpenRouter review output must be a JSON object") from None
        try:
            _check_output_fields(value, payload["response_format"]["json_schema"]["schema"])
        except PermanentModelError:
            raise PermanentModelError("Unexpected OpenRouter review model output fields") from None
        usage = check_response_usage(data.get("usage"), app_input_tokens=self.profile.max_input_tokens,
            app_output_tokens=self.profile.max_output_tokens, model_context_tokens=model_context,
            max_output_tokens=maximum, role="OpenRouter review")
        self._remaining(deadline)
        return ModelResponse(text, usage)

    def _post(self, payload: dict[str, Any], body: bytes, deadline: float) -> dict[str, Any]:
        try:
            with model_request_slot(self, deadline, generates=True) as (ticket, deadline):
                try:
                    with httpx.Client(trust_env=False, follow_redirects=False,
                            timeout=self._remaining(deadline)) as client:
                        mark_model_request_sent(ticket)
                        with client.stream("POST", self.origin + _CHAT_PATH, content=body,
                                headers=auth_headers(self._api_key),
                                timeout=self._remaining(deadline)) as response:
                            self._remaining(deadline)
                            chunks = bytearray()
                            for chunk in response.iter_bytes():
                                self._remaining(deadline)
                                if len(chunks) + len(chunk) > self._max_transport_bytes:
                                    raise PermanentModelError("OpenRouter review response exceeds byte limit")
                                chunks.extend(chunk)
                            self._remaining(deadline)
                            status = response.status_code
                            raw = bytes(chunks)
                            ticket.finish()
                            if status != 200:
                                raise self._http_error(status, raw, payload)
                except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                    ticket.finish()
                    raise
        except httpx.TimeoutException:
            raise TimeoutError("OpenRouter review request timed out") from None
        except httpx.RequestError:
            raise TransientModelError("OpenRouter review connection failed") from None
        try:
            data = _json_object(raw)
        except PermanentModelError:
            raise PermanentModelError("Invalid OpenRouter review response JSON") from None
        self._remaining(deadline)
        if "error" in data:
            raise PermanentModelError("OpenRouter review returned an error payload")
        return data

    def _http_error(self, status: int, raw: bytes, payload: dict[str, Any]) -> Exception:
        if status == 400:
            text = raw.decode("utf-8", errors="replace")
            lowered = text.lower()
            if "context" in lowered and ("length" in lowered or "limit" in lowered or "exceed" in lowered):
                error = ModelInputCapacityError("OpenRouter review HTTP 400 context exceeded",
                    {"provider": "openrouter", "model": self.model, "status": 400,
                        "response_sha256": __import__("hashlib").sha256(raw).hexdigest(),
                        "wire_sha256": __import__("hashlib").sha256(encode(payload)).hexdigest()})
                error.delivery, error.usage = "not_sent", dict(_ZERO_USAGE)
                return error
            return PermanentModelError(f"OpenRouter review HTTP {status}")
        error_cls = classify_http_error(status, "OpenRouter review")
        if status in (401, 402, 403):
            return error_cls(f"OpenRouter review HTTP {status}: check API key and credits")
        return error_cls(f"OpenRouter review HTTP {status}")
