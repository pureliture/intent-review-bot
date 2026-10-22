"""Shared OpenRouter transport helpers for validation-phase adapters.

No tokenization, no measurement calls, no inference here. The only
pre-send guarantee is a sound byte upper bound: every token covers at
least one UTF-8 byte, so ``len(wire_bytes)`` is a proven upper bound on
the tokens our own payload can occupy. A fixed framing reserve covers
provider-side chat-template framing we cannot see. Anything beyond the
reserve is caught post-send by the response usage check and fails
closed; it is never reinterpreted as a fitting request.

Exact local tokenizers are unavailable for the validation models
(gemma tokenizer is public; pplx decider tokenizer unreleased),
so this conservative bound is the interim proof. It is fail-closed, not
an estimate: it can only refuse, never wrongly admit.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import httpx

from review_bot.adapters.byte_budget import FRAMING_RESERVE_TOKENS, precheck_byte_budget as _shared_precheck
from review_bot.application.architecture_review import PermanentModelError, TransientModelError

OPENROUTER_BASE_URL = "https://openrouter.ai"
_MODELS_PATH = "/api/v1/models"
_TRANSPORT_BYTES = 16 * 1024 * 1024
_METADATA_SECONDS = 60


def _positive(value: Any) -> bool:
    return type(value) is int and value > 0


def encode(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        raise PermanentModelError("Invalid OpenRouter request JSON") from None


def auth_headers(api_key: str) -> dict[str, str]:
    if not isinstance(api_key, str) or not api_key.strip() or any(
            char.isspace() or not char.isascii() for char in api_key):
        raise ValueError("OPENROUTER_API_KEY must be a non-empty ASCII token")
    return {"Authorization": "Bearer " + api_key.strip(),
        "Content-Type": "application/json", "Accept": "application/json"}


def base_url(value: str | None) -> str:
    raw = (value or OPENROUTER_BASE_URL).strip()
    try:
        parsed = httpx.URL(raw)
    except (ValueError, httpx.InvalidURL):
        raise ValueError("OpenRouter base URL must be an https origin") from None
    if (parsed.scheme != "https" or parsed.userinfo or parsed.query or parsed.fragment
            or parsed.path not in ("", "/")):
        raise ValueError("OpenRouter base URL must be an https origin")
    return str(parsed).rstrip("/")


def precheck_byte_budget(wire_len: int, max_output_tokens: int, *, app_input_tokens: int,
        app_output_tokens: int, model_context_tokens: int, role: str,
        framing_reserve_tokens: int | None = None) -> None:
    """Fail-closed pre-send gate; delegates to the shared byte_budget module."""
    if wire_len > _TRANSPORT_BYTES:
        raise PermanentModelError(f"{role} input exceeds transport byte limit")
    return _shared_precheck(wire_len, max_output_tokens, app_input_tokens=app_input_tokens,
        app_output_tokens=app_output_tokens, model_context_tokens=model_context_tokens, role=role,
        framing_reserve_tokens=FRAMING_RESERVE_TOKENS if framing_reserve_tokens is None else framing_reserve_tokens)


def check_response_usage(usage: Any, *, app_input_tokens: int, app_output_tokens: int,
        model_context_tokens: int, max_output_tokens: int, role: str) -> dict[str, int]:
    """Validate provider-reported usage against the same budget. Fail-closed."""
    if not isinstance(usage, dict):
        raise PermanentModelError(f"Missing {role} model usage")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or prompt < 0 or type(completion) is not int or completion < 0:
        raise PermanentModelError(f"Missing {role} model usage")
    total = usage.get("total_tokens", prompt + completion)
    if type(total) is not int or total != prompt + completion:
        raise PermanentModelError(f"Invalid {role} model usage total")
    total_cap = min(app_input_tokens + app_output_tokens, model_context_tokens)
    if (prompt > app_input_tokens or completion > max_output_tokens
            or prompt + completion > total_cap):
        raise PermanentModelError(f"{role} usage exceeds budget; result is discarded")
    return {"input_tokens": prompt, "output_tokens": completion}


def classify_http_error(status: int, role: str) -> type[Exception]:
    if status == 400:
        return PermanentModelError
    if status in (401, 402, 403):
        return PermanentModelError
    if status in (408, 429) or status >= 500:
        return TransientModelError
    return PermanentModelError


def fetch_models_catalog(api_key: str, origin: str, timeout: float, *, query: str = "") -> list[dict[str, Any]]:
    try:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=timeout) as client:
            response = client.get(origin + _MODELS_PATH + query, headers=auth_headers(api_key))
    except httpx.TimeoutException:
        raise TimeoutError("OpenRouter catalog request timed out") from None
    except httpx.RequestError:
        raise TransientModelError("OpenRouter catalog connection failed") from None
    if response.status_code != 200:
        error = classify_http_error(response.status_code, "catalog")
        raise error(f"OpenRouter catalog HTTP {response.status_code}")
    try:
        data = json.loads(response.content.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise PermanentModelError("Invalid OpenRouter catalog JSON") from None
    if not isinstance(data, dict) or not isinstance(data.get("data"), list):
        raise PermanentModelError("Invalid OpenRouter catalog shape")
    return [entry for entry in data["data"] if isinstance(entry, dict)]


def find_model_entry(catalog: list[dict[str, Any]], model: str) -> dict[str, Any]:
    for entry in catalog:
        if entry.get("id") == model:
            return entry
    raise PermanentModelError("OpenRouter catalog does not list the pinned model")


def check_text_entry(entry: dict[str, Any], model: str, *, minimum_context: int) -> int:
    arch = entry.get("architecture")
    modalities = (arch.get("output_modalities") or []) if isinstance(arch, dict) else []
    if "text" not in modalities:
        raise PermanentModelError("OpenRouter model does not serve text output")
    context = entry.get("context_length")
    if type(context) is not int or context < minimum_context:
        raise PermanentModelError("OpenRouter model context is below the application envelope")
    return context


def check_decisions_entry(entry: dict[str, Any], model: str, *, minimum_context: int) -> int:
    arch = entry.get("architecture")
    modalities = (arch.get("output_modalities") or []) if isinstance(arch, dict) else []
    if "decisions" not in modalities:
        raise PermanentModelError("OpenRouter model does not serve decisions output")
    context = entry.get("context_length")
    if type(context) is not int or context < minimum_context:
        raise PermanentModelError("OpenRouter decision model context is below the application envelope")
    return context


def fresh(deadline: float, verified_until: float) -> bool:
    try:
        return time.monotonic() < verified_until <= deadline
    except (TypeError, ValueError, OverflowError):
        return False
