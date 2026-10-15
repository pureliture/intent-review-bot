"""Explicit me707 llama.cpp chat adapters with generation-free native token preflight."""
from __future__ import annotations

import ipaddress
import json
import math
import threading
import time
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit

import httpx

from review_bot.adapters.model_http import mark_model_request_sent, model_request_slot
from review_bot.application.review_a import ModelProfile, PermanentModelError, TransientModelError
from review_bot.domain.models_v1 import ModelRequest, ModelResponse

LLAMACPP_PROFILE = "me707-llamacpp"
REVIEW_MODEL = "bonsai2-27b"
SELECTOR_MODEL = "tev1-4b-q8_0"
_REVIEW_CONTEXT_TOKENS = 262144
_SELECTOR_CONTEXT_TOKENS = 131072
_OUTPUT_RESERVE = 4096
_SELECTOR_OUTPUT_TOKENS = 64
_MAX_INPUT_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_SELECTOR_OUTPUT_BYTES = 4096


def _encode(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PermanentModelError("Invalid llama.cpp request JSON") from None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite JSON number")
    return result


def _check_json_strings(value: Any) -> None:
    if isinstance(value, str):
        value.encode("utf-8")
    elif isinstance(value, dict):
        for key, child in value.items():
            key.encode("utf-8")
            _check_json_strings(child)
    elif isinstance(value, list):
        for child in value:
            _check_json_strings(child)


def _json_object(value: bytes | str) -> dict[str, Any]:
    try:
        text = value.decode("utf-8") if isinstance(value, bytes) else value
        text.encode("utf-8")
        result = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite, parse_float=_finite_float)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object")
        _check_json_strings(result)
        return result
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PermanentModelError("Invalid llama.cpp response JSON") from None


def _count(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise PermanentModelError("Invalid llama.cpp token count")
    return value


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _valid_timeout(value: Any) -> bool:
    try:
        return _positive_integer(value) and math.isfinite(value)
    except OverflowError:
        return False


def _endpoint(api_url: str) -> str:
    message = "llama.cpp endpoint must be an approved uncredentialed HTTP /v1 endpoint"
    if (
        not isinstance(api_url, str) or not api_url.isascii()
        or any(ord(char) <= 32 or ord(char) == 127 for char in api_url)
        or any(char in api_url for char in ("\\", "?", "#", "%"))
    ):
        raise ValueError(message)
    try:
        parts = urlsplit(api_url)
        host = parts.hostname
        port = parts.port
        approved = host in ("localhost", "ubuntu-ms-7e07")
        if not approved and host:
            try:
                approved = ipaddress.ip_address(host).is_loopback
            except ValueError:
                approved = False
        if (
            not approved or parts.scheme != "http" or parts.username is not None or parts.password is not None
            or parts.path not in ("/v1", "/v1/") or (port is not None and not 0 < port <= 65535)
        ):
            raise ValueError(message)
        return str(httpx.URL(api_url)).rstrip("/")
    except (ValueError, httpx.InvalidURL):
        raise ValueError(message) from None


def _check_output_fields(value: Any, schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise PermanentModelError("Invalid llama.cpp output schema")
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            raise PermanentModelError("Unexpected llama.cpp model output fields")
        for key, child in value.items():
            _check_output_fields(child, properties.get(key))
    elif isinstance(value, list):
        for child in value:
            _check_output_fields(child, schema.get("items"))


class _LlamaCppClient:
    def __init__(
        self, model: str, api_url: str, timeout_seconds: int, *, role_model: str,
        context_tokens: int, max_output_tokens: int, max_output_bytes: int,
    ) -> None:
        if model != role_model or not isinstance(model, str):
            raise ValueError("An exact llama.cpp role model is required")
        if not _valid_timeout(timeout_seconds):
            raise ValueError("llama.cpp timeout must be positive and finite")
        self._model = model
        self._api_url = self._validate_endpoint(api_url)
        self._origin = self._api_url[:-len("/v1")]
        self.timeout_seconds = timeout_seconds
        self._context_tokens = context_tokens
        self._max_input_tokens = context_tokens - _OUTPUT_RESERVE
        self._max_output_tokens = max_output_tokens
        self._max_output_bytes = max_output_bytes

    def _validate_endpoint(self, api_url: str) -> str:
        return _endpoint(api_url)

    @property
    def model(self) -> str:
        return self._model

    @property
    def api_url(self) -> str:
        return self._api_url

    def _deadline(self, request: ModelRequest | None = None) -> float:
        timeout = self.timeout_seconds
        if request is not None:
            if not isinstance(request, ModelRequest) or not _valid_timeout(request.timeout_seconds):
                raise PermanentModelError("Invalid llama.cpp request timeout")
            timeout = min(timeout, request.timeout_seconds)
        return time.monotonic() + timeout

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("llama.cpp request timed out")
        return remaining

    def _record_delivery(self, delivery: str) -> None:
        self._receipts.value = {
            "delivery": delivery, "model_calls": 0, "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def _request(
        self, method: str, path: str, deadline: float, payload: dict[str, Any] | None = None, *, measurement: bool = False,
    ) -> dict[str, Any]:
        content = _encode(payload) if payload is not None else None
        if content is not None and len(content) > _MAX_INPUT_BYTES:
            raise PermanentModelError("llama.cpp input exceeds transport byte limit")
        headers = {"Accept": "application/json"}
        if content is not None:
            headers["Content-Type"] = "application/json"
        try:
            with model_request_slot(self, deadline, generates=path == "/v1/chat/completions") as (ticket, deadline):
                try:
                    with httpx.Client(trust_env=False, follow_redirects=False, timeout=self._remaining(deadline)) as client:
                        timeout = self._remaining(deadline)
                        if measurement:
                            self._record_delivery("unknown")
                        mark_model_request_sent(ticket)
                        with client.stream(method, self._origin + path, content=content, headers=headers, timeout=timeout) as response:
                            if measurement:
                                self._record_delivery("sent")
                            status = response.status_code
                            self._remaining(deadline)
                            length = response.headers.get("Content-Length")
                            if length is not None:
                                try:
                                    length_bytes = int(length)
                                except ValueError:
                                    raise PermanentModelError("Invalid llama.cpp response byte length") from None
                                if not 0 <= length_bytes <= _MAX_RESPONSE_BYTES:
                                    raise PermanentModelError("llama.cpp response exceeds byte limit")
                            body = bytearray()
                            for chunk in response.iter_bytes():
                                self._remaining(deadline)
                                if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                                    raise PermanentModelError("llama.cpp response exceeds byte limit")
                                body.extend(chunk)
                            ticket.finish()
                            self._remaining(deadline)
                            if status != 200:
                                error = TransientModelError if status in (408, 429) or status >= 500 else PermanentModelError
                                raise error(f"llama.cpp HTTP {status}")
                except (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
                    ticket.finish()
                    raise
        except httpx.TimeoutException:
            raise TimeoutError("llama.cpp request timed out") from None
        except httpx.DecodingError:
            raise PermanentModelError("Invalid llama.cpp response encoding") from None
        except httpx.RequestError:
            raise TransientModelError("llama.cpp connection failed") from None
        result = _json_object(bytes(body))
        self._remaining(deadline)
        if "error" in result:
            raise PermanentModelError("llama.cpp returned an invalid response")
        return result

    def _verify(self, deadline: float) -> None:
        models = self._request("GET", "/v1/models", deadline)
        entries = models.get("data")
        if (
            models.get("object") != "list" or not isinstance(entries, list) or len(entries) != 1
            or not isinstance(entries[0], dict) or entries[0].get("id") != self.model
        ):
            raise PermanentModelError("llama.cpp server does not expose the exact role model")
        props = self._request("GET", "/props", deadline)
        if type(props.get("total_slots")) is not int or props["total_slots"] != 1:
            raise PermanentModelError("llama.cpp requires exactly one server generation slot")
        settings = props.get("default_generation_settings")
        template = props.get("chat_template")
        if (
            not isinstance(settings, dict) or type(settings.get("n_ctx")) is not int
            or settings["n_ctx"] != self._context_tokens
            or not isinstance(template, str) or not template.strip()
        ):
            raise PermanentModelError("llama.cpp requires the configured per-slot context and chat template")
        _encode(template)

    def verify(self) -> None:
        """Check the exact loaded alias and active slot settings, never training-context metadata."""
        self._verify(self._deadline())

    def warmup(self) -> None:
        """Explicit metadata verification only; no inference or synthetic generation."""
        self.verify()

    def _chat_payload(self, request: ModelRequest) -> dict[str, Any]:
        if not isinstance(request, ModelRequest):
            raise PermanentModelError("Invalid llama.cpp chat request")
        if request.model != self.model:
            raise PermanentModelError("llama.cpp request model does not match its role")
        maximum = request.max_output_tokens if request.max_output_tokens is not None else self._max_output_tokens
        if type(maximum) is not int or not 0 < maximum <= self._max_output_tokens:
            raise PermanentModelError("Invalid llama.cpp output token budget")
        if (
            type(request.schema_version) is not int or request.schema_version != 1
            or not isinstance(request.system_prompt, str) or not isinstance(request.user_content, str)
            or not isinstance(request.response_schema, dict)
        ):
            raise PermanentModelError("Invalid llama.cpp chat input")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": request.system_prompt},
                         {"role": "user", "content": request.user_content}],
            "response_format": {"type": "json_schema", "schema": request.response_schema},
            "stream": False, "temperature": 0, "seed": 0, "n": 1, "max_tokens": maximum,
            "stop": [], "cache_prompt": False, "n_keep": -1,
        }
        encoded = _encode(payload)
        if len(encoded) > _MAX_INPUT_BYTES:
            raise PermanentModelError("llama.cpp input exceeds transport byte limit")
        return _json_object(encoded)

    def _measure(self, payload: dict[str, Any], deadline: float, *, measurement: bool = False) -> int:
        """The server template supplies framing; response_format constrains sampling, not prompt text.

        No tools or template overrides are used. Native string-prompt tokenization
        uses model special-token insertion and parses the template's special tokens.
        Unsupported formatting/tokenization fails closed; counts are never cached.
        """
        template_input = {"messages": payload["messages"]}
        if "chat_template_kwargs" in payload:
            template_input["chat_template_kwargs"] = payload["chat_template_kwargs"]
        formatted = self._request("POST", "/apply-template", deadline, template_input, measurement=measurement)
        prompt = formatted.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise PermanentModelError("Unsupported llama.cpp chat template response")
        tokenization = self._request("POST", "/tokenize", deadline, {
            "content": prompt, "add_special": True, "parse_special": True, "with_pieces": False,
        }, measurement=measurement)
        tokens = tokenization.get("tokens")
        if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise PermanentModelError("Unsupported llama.cpp native tokenization response")
        self._remaining(deadline)
        return len(tokens)

    def _check_output(self, value: dict[str, Any], payload: dict[str, Any]) -> None:
        try:
            _check_output_fields(value, payload["response_format"]["schema"])
        except RecursionError:
            raise PermanentModelError("Invalid llama.cpp model output nesting") from None

    def _response(self, data: dict[str, Any], payload: dict[str, Any], measured: int) -> ModelResponse:
        choices = data.get("choices")
        if data.get("model") != self.model or not isinstance(choices, list) or len(choices) != 1:
            raise PermanentModelError("Invalid llama.cpp chat response model or choices")
        choice = choices[0]
        if (
            not isinstance(choice, dict) or type(choice.get("index")) is not int or choice["index"] != 0
            or choice.get("finish_reason") != "stop"
        ):
            raise PermanentModelError("Invalid or truncated llama.cpp model output")
        message = choice.get("message")
        if (
            not isinstance(message, dict) or message.get("role") != "assistant"
            or set(message) - {"role", "content", "reasoning_content", "refusal"}
            or message.get("reasoning_content") not in (None, "") or message.get("refusal") is not None
        ):
            raise PermanentModelError("Invalid llama.cpp assistant output; tools and hidden reasoning are forbidden")
        text = message.get("content")
        if not isinstance(text, str):
            raise PermanentModelError("Invalid llama.cpp model output text")
        try:
            text_bytes = len(text.encode("utf-8"))
        except UnicodeError:
            raise PermanentModelError("Invalid llama.cpp model output encoding") from None
        if text_bytes > self._max_output_bytes:
            raise PermanentModelError("llama.cpp model output exceeds byte limit")
        value = _json_object(text)
        self._check_output(value, payload)
        usage = data.get("usage")
        if not isinstance(usage, dict):
            raise PermanentModelError("Missing llama.cpp model usage")
        input_tokens = _count(usage.get("prompt_tokens"))
        output_tokens = _count(usage.get("completion_tokens"))
        total_tokens = _count(usage.get("total_tokens"))
        if (
            input_tokens != measured or input_tokens > self._max_input_tokens
            or output_tokens > payload["max_tokens"] or total_tokens != input_tokens + output_tokens
            or ("truncated" in data and data["truncated"] is not False)
            or ("tokens_evaluated" in data and _count(data["tokens_evaluated"]) != measured)
        ):
            raise PermanentModelError("llama.cpp usage or truncation does not match native token preflight")
        details = usage.get("prompt_tokens_details")
        if details is not None:
            if not isinstance(details, dict) or _count(details.get("cached_tokens")) > input_tokens:
                raise PermanentModelError("Invalid llama.cpp cached token usage")
        if "timings" in data:
            timings = data["timings"]
            if (
                not isinstance(timings, dict) or _count(timings.get("cache_n")) + _count(timings.get("prompt_n")) != measured
                or _count(timings.get("predicted_n")) != output_tokens
            ):
                raise PermanentModelError("llama.cpp timings do not match model usage")
        return ModelResponse(text, {"input_tokens": input_tokens, "output_tokens": output_tokens})

    def complete(self, request: ModelRequest) -> ModelResponse:
        deadline = self._deadline(request)
        payload = self._chat_payload(request)
        self._verify(deadline)
        measured = self._measure(payload, deadline)
        if measured > self._max_input_tokens:
            raise PermanentModelError("llama.cpp input exceeds native token budget; truncation is forbidden")
        data = self._request("POST", "/v1/chat/completions", deadline, payload)
        response = self._response(data, payload, measured)
        self._remaining(deadline)
        return response


class LlamaCppModelClient(_LlamaCppClient):
    """Bonsai review client; transport bytes and actual prompt tokens are separate limits."""

    _role_model = REVIEW_MODEL
    _role_context_tokens = _REVIEW_CONTEXT_TOKENS

    def __init__(
        self, model: str = REVIEW_MODEL, *, api_url: str, timeout_seconds: int = 60,
        max_output_bytes: int = 32768, max_attempts: int = 1,
    ) -> None:
        if not _positive_integer(max_output_bytes) or max_output_bytes > _MAX_RESPONSE_BYTES:
            raise ValueError("llama.cpp output byte limit must be positive and bounded")
        if not _positive_integer(max_attempts):
            raise ValueError("llama.cpp attempts must be positive")
        super().__init__(model, api_url, timeout_seconds, role_model=self._role_model,
                         context_tokens=self._role_context_tokens, max_output_tokens=_OUTPUT_RESERVE,
                         max_output_bytes=max_output_bytes)
        self.profile = ModelProfile(
            provider="llamacpp", model=model, max_input_bytes=_MAX_INPUT_BYTES, max_output_bytes=max_output_bytes,
            timeout_seconds=timeout_seconds, max_attempts=max_attempts, max_input_tokens=self._max_input_tokens,
            max_output_tokens=_OUTPUT_RESERVE,
        )
        self._receipts = threading.local()

    @property
    def token_count_receipt(self) -> dict[str, Any] | None:
        """A caller-local receipt for tokenizer transport, never a generation or usage receipt."""
        return deepcopy(getattr(self._receipts, "value", None))

    def count_input_tokens(self, request: ModelRequest) -> int:
        self._record_delivery("not_sent")
        deadline = self._deadline(request)
        payload = self._chat_payload(request)
        self._verify(deadline)
        return self._measure(payload, deadline, measurement=True)


class LlamaCppDecisionClient(_LlamaCppClient):
    """Tev metadata-only mode selection via chat, not a review-plan or general review role."""

    _role_model = SELECTOR_MODEL
    _role_context_tokens = _SELECTOR_CONTEXT_TOKENS

    def __init__(self, model: str = SELECTOR_MODEL, *, api_url: str, timeout_seconds: int = 60) -> None:
        super().__init__(model, api_url, timeout_seconds, role_model=self._role_model,
                         context_tokens=self._role_context_tokens, max_output_tokens=_SELECTOR_OUTPUT_TOKENS,
                         max_output_bytes=_MAX_SELECTOR_OUTPUT_BYTES)

    def _chat_payload(self, request: ModelRequest) -> dict[str, Any]:
        payload = super()._chat_payload(request)
        metadata = _json_object(request.user_content)
        if (
            set(metadata) != {"title", "body", "changed_files", "available_modes"}
            or not isinstance(metadata["title"], str) or not isinstance(metadata["body"], str)
            or not isinstance(metadata["changed_files"], list)
            or any(not isinstance(path, str) for path in metadata["changed_files"])
            or not isinstance(metadata["available_modes"], list) or not 1 <= len(metadata["available_modes"]) <= 2
            or any(not isinstance(mode, str) or mode not in ("architecture", "intent") for mode in metadata["available_modes"])
            or len(metadata["available_modes"]) != len(set(metadata["available_modes"]))
        ):
            raise PermanentModelError("llama.cpp selector only accepts registered PR selection metadata")
        return payload

    def _check_output(self, value: dict[str, Any], payload: dict[str, Any]) -> None:
        super()._check_output(value, payload)
        available = _json_object(payload["messages"][1]["content"])["available_modes"]
        selected = value.get("selected_modes")
        if (
            set(value) != {"selected_modes"} or not isinstance(selected, list)
            or any(not isinstance(mode, str) or mode not in available for mode in selected)
            or len(selected) != len(set(selected))
        ):
            raise PermanentModelError("Invalid llama.cpp selected modes")