"""Explicit OpenAI chat-completions connection, independent of serving software.

Only /chat/completions is required. Configuration is operator-owned; server
slot metadata, model catalogs and tokenizer endpoints are not prerequisites.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

from review_bot.adapters.byte_budget import precheck_byte_budget
from review_bot.adapters.chat_json import _check_output_fields, _count, _encode, _json_object
from review_bot.adapters.model_http import mark_model_request_sent, model_request_slot
from review_bot.application.architecture_review import (
    ModelOutputLimitError, ModelProfile, PermanentModelError, TransientModelError,
)
from review_bot.domain.models_v1 import ModelRequest, ModelResponse

OPENAI_COMPATIBLE_PROFILE = "openai-compatible"


class OpenAICompatibleModelClient:
    """Bounded JSON chat adapter; no model-name or server-brand assumptions."""

    supports_bounded_reasoning = False

    def __init__(self, *, api_url: str, model: str, context_tokens: int,
        input_tokens: int = 98394, output_tokens: int = 32678, api_key: str = "",
        timeout_seconds: int = 60, max_output_bytes: int = 32768,
        max_attempts: int = 1, framing_reserve_tokens: int = 8192,
        max_transport_bytes: int = 16 * 1024 * 1024,
        enable_thinking: bool | None = None,
    ):
        message = "Invalid explicit model connection settings"
        try:
            parts = urlsplit(api_url)
            if (not api_url.isascii() or any(ord(c) <= 32 or ord(c) == 127 for c in api_url)
                or any(c in api_url for c in ("\\", "?", "#", "%"))
                or parts.scheme not in ("http", "https") or not parts.hostname
                or parts.username is not None or parts.password is not None
                or parts.port is not None and not 0 < parts.port <= 65535):
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise ValueError(message) from None
        if (not isinstance(model, str) or not model.strip() or len(model) > 256
            or any(ord(c) < 32 or ord(c) == 127 for c in model)
            or not isinstance(api_key, str) or not api_key.isascii()
            or any(ord(c) < 32 or ord(c) == 127 for c in api_key)):
            raise ValueError(message)
        for value in (context_tokens, input_tokens, output_tokens, timeout_seconds,
            max_output_bytes, max_attempts, max_transport_bytes):
            if type(value) is not int or value <= 0:
                raise ValueError(message)
        if (type(framing_reserve_tokens) is not int or framing_reserve_tokens < 0
            or input_tokens + output_tokens > context_tokens or max_attempts > 3
            or max_output_bytes > max_transport_bytes or max_transport_bytes > 16 * 1024 * 1024
            or enable_thinking is not None and type(enable_thinking) is not bool):
            raise ValueError(message)
        self.api_url = api_url.rstrip("/")
        self.model = model.strip()
        self._api_key = api_key.strip()
        self._context_tokens = context_tokens
        self._framing_reserve = framing_reserve_tokens
        self._transport_bytes = max_transport_bytes
        self._enable_thinking = enable_thinking
        self.request_cancelled: Callable[[], bool] | None = None
        self.profile = ModelProfile(OPENAI_COMPATIBLE_PROFILE, self.model,
            max_input_bytes=max_transport_bytes, max_output_bytes=max_output_bytes,
            timeout_seconds=timeout_seconds, max_attempts=max_attempts,
            max_input_tokens=input_tokens, max_output_tokens=output_tokens)

    @property
    def cache_identity(self):
        # Credentials are deliberately absent. Changing a serving contract or
        # wire option must not reuse a result from another configuration.
        return {"wire": "openai-json-chat/v1", "context_tokens": self._context_tokens,
            "framing_reserve_tokens": self._framing_reserve,
            "enable_thinking": self._enable_thinking}

    def set_cancellation_callback(self, callback: Callable[[], bool]) -> None:
        self.request_cancelled = callback

    def warmup(self) -> None:
        """Construction validates settings; readiness requires a real response.

        Do not call a provider-specific catalog, /props or tokenizer, and do
        not spend a generation merely to prepare a client.
        """

    def _payload(self, request: ModelRequest) -> dict:
        if (not isinstance(request, ModelRequest) or type(request.schema_version) is not int
            or request.schema_version != 1 or request.model != self.model
            or not isinstance(request.system_prompt, str) or not isinstance(request.user_content, str)
            or not isinstance(request.response_schema, dict) or request.reasoning_budget_tokens is not None
            or type(request.timeout_seconds) is not int or request.timeout_seconds <= 0):
            raise PermanentModelError("Invalid configured chat request")
        maximum = request.max_output_tokens if request.max_output_tokens is not None else self.profile.max_output_tokens
        if type(maximum) is not int or not 0 < maximum <= self.profile.max_output_tokens:
            raise PermanentModelError("Invalid configured chat output reservation")
        payload = {"model": self.model, "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_content}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "review_bot", "strict": True, "schema": request.response_schema}},
            "max_tokens": maximum, "stream": False, "temperature": 0}
        if self._enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self._enable_thinking}
        if len(_encode(payload)) > self._transport_bytes:
            raise PermanentModelError("Configured chat input exceeds transport byte limit")
        return payload

    def admission_bytes(self, request: ModelRequest) -> int:
        # Full wire data is conservative across serving implementations. No
        # unsupported native tokenizer endpoint, source trimming or guessed
        # token count is substituted when the bound exceeds the app budget.
        return len(_encode(self._payload(request))) + self._framing_reserve

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Configured chat request timed out")
        return remaining

    def _post(self, body, deadline):
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        try:
            with model_request_slot(self, deadline, generates=True) as (ticket, deadline):
                try:
                    with httpx.Client(trust_env=False, follow_redirects=False,
                        timeout=self._remaining(deadline)) as client:
                        mark_model_request_sent(ticket)
                        with client.stream("POST", self.api_url + "/chat/completions",
                            content=body, headers=headers, timeout=self._remaining(deadline)) as response:
                            raw = bytearray()
                            for chunk in response.iter_bytes():
                                self._remaining(deadline)
                                if len(raw) + len(chunk) > self._transport_bytes:
                                    raise PermanentModelError("Configured chat response exceeds byte limit")
                                raw.extend(chunk)
                            ticket.finish()
                            self._remaining(deadline)
                            if response.status_code != 200:
                                cls = TransientModelError if response.status_code in (408, 429) or response.status_code >= 500 else PermanentModelError
                                raise cls(f"Configured chat HTTP {response.status_code}")
                except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                    ticket.finish()
                    raise
        except httpx.TimeoutException:
            raise TimeoutError("Configured chat request timed out") from None
        except httpx.RequestError:
            raise TransientModelError("Configured chat transport failed") from None
        return _json_object(bytes(raw))

    def _usage(self, data, maximum):
        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise PermanentModelError("Configured chat usage is unavailable")
        i, o, total = (_count(usage.get(k)) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        if (i > self.profile.max_input_tokens or o > maximum or total != i + o
            or total > self._context_tokens or data.get("truncated", False) is not False):
            raise PermanentModelError("Configured chat usage exceeds budget or is invalid")
        return {"input_tokens": i, "output_tokens": o}

    def complete(self, request: ModelRequest) -> ModelResponse:
        payload = self._payload(request)
        deadline = time.monotonic() + min(request.timeout_seconds, self.profile.timeout_seconds)
        body = _encode(payload)
        precheck_byte_budget(len(body), payload["max_tokens"], app_input_tokens=self.profile.max_input_tokens,
            app_output_tokens=self.profile.max_output_tokens, model_context_tokens=self._context_tokens,
            framing_reserve_tokens=self._framing_reserve, role="configured chat")
        data = self._post(body, deadline)
        choices = data.get("choices")
        if (data.get("model") != self.model or not isinstance(choices, list) or len(choices) != 1
            or not isinstance(choices[0], dict) or type(choices[0].get("index")) is not int
            or choices[0]["index"] != 0):
            raise PermanentModelError("Configured chat response identity is invalid")
        usage = self._usage(data, payload["max_tokens"])
        try:
            choice = choices[0]
            message = choice.get("message")
            if (not isinstance(message, dict) or message.get("role") != "assistant"
                or set(message) - {"role", "content", "reasoning_content", "refusal"}
                or message.get("refusal") is not None):
                raise PermanentModelError("Configured chat assistant response is invalid")
            text, reasoning = message.get("content"), message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise PermanentModelError("Configured chat reasoning metadata is invalid")
            if choice.get("finish_reason") == "length":
                error = ModelOutputLimitError("Configured chat output reached its limit")
                error.rejected_output = {"finish_reason": "length", "content": text, "reasoning_content": reasoning}
                raise error
            if choice.get("finish_reason") != "stop" or not isinstance(text, str):
                raise PermanentModelError("Configured chat output is incomplete")
            if len(text.encode("utf-8")) > self.profile.max_output_bytes:
                raise PermanentModelError("Configured chat output exceeds byte limit")
            value = _json_object(text)
            _check_output_fields(value, request.response_schema)
            self._remaining(deadline)
        except (PermanentModelError, TimeoutError) as error:
            error.delivery, error.usage = "sent", usage
            raise
        return ModelResponse(text, usage)
