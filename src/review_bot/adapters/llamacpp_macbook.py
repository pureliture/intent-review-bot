"""Loopback-only MacBook local32k roles over the shared llama.cpp transport."""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from review_bot.adapters.llamacpp import LlamaCppDecisionClient, LlamaCppModelClient
from review_bot.adapters.sqlite.model_admission import SQLiteModelAdmission
from review_bot.domain.models_v1 import ModelRequest

MACBOOK_LLAMACPP_PROFILE = "macbook-llamacpp"
MACBOOK_REVIEW_MODEL = "lfm2.5-2.6b-q8_0"
MACBOOK_SELECTOR_MODEL = "tev1-0.8b-q8_0"
MACBOOK_CONTEXT_TOKENS = 32768


def macbook_model_admission(max_concurrency: int = 2) -> SQLiteModelAdmission:
    state_home = Path.home() / ".local" / "state" / "github-review-bot"
    configured = os.environ.get("REVIEW_BOT_MACBOOK_MODEL_STATE_DIR")
    message = "REVIEW_BOT_MACBOOK_MODEL_STATE_DIR must be an absolute private directory separate from production state"
    if configured is not None and not configured.strip():
        raise ValueError(message)
    root = Path(configured) if configured is not None else state_home / "macbook-model-admission"
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError(message)
    root = Path("/", *root.parts[1:])
    if root in (Path("/"), Path("/root")):
        raise ValueError(message)
    reserved = {state_home / "model-admission"}
    production = os.environ.get("REVIEW_BOT_MODEL_STATE_DIR")
    if production is not None:
        production_root = Path(production)
        if production_root.is_absolute():
            reserved.add(Path("/", *production_root.parts[1:]))
    if root in reserved:
        raise ValueError(message)
    return SQLiteModelAdmission(root / "requests.sqlite", max_concurrency=max_concurrency)


class _MacBookLlamaCppClient:
    def _validate_endpoint(self, api_url: str) -> str:
        canonical = super()._validate_endpoint(api_url)
        host = urlsplit(canonical).hostname
        if host == "localhost":
            return canonical
        try:
            if host is not None and ipaddress.ip_address(host).is_loopback:
                return canonical
        except ValueError:
            pass
        raise ValueError("MacBook llama.cpp endpoint must use loopback HTTP /v1") from None

    def _chat_payload(self, request: ModelRequest) -> dict[str, Any]:
        payload = super()._chat_payload(request)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "review_bot", "strict": True, "schema": payload["response_format"]["schema"]},
        }
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload

    def _check_output(self, value: dict[str, Any], payload: dict[str, Any]) -> None:
        validation_payload = {
            **payload,
            "response_format": {"type": "json_schema", "schema": payload["response_format"]["json_schema"]["schema"]},
        }
        super()._check_output(value, validation_payload)


class MacBookLlamaCppModelClient(_MacBookLlamaCppClient, LlamaCppModelClient):
    """LFM review role with a private MacBook admission group and 32K context."""

    _role_model = MACBOOK_REVIEW_MODEL
    _role_context_tokens = MACBOOK_CONTEXT_TOKENS

    def __init__(
        self, model: str = MACBOOK_REVIEW_MODEL, *, api_url: str, timeout_seconds: int = 60,
        max_output_bytes: int = 32768, max_attempts: int = 1,
    ) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
                         max_output_bytes=max_output_bytes, max_attempts=max_attempts)
        self.request_admission = macbook_model_admission()


class MacBookLlamaCppDecisionClient(_MacBookLlamaCppClient, LlamaCppDecisionClient):
    """Tev metadata-only selected_modes role; no planning or model fallback."""

    _role_model = MACBOOK_SELECTOR_MODEL
    _role_context_tokens = MACBOOK_CONTEXT_TOKENS

    def __init__(self, model: str = MACBOOK_SELECTOR_MODEL, *, api_url: str, timeout_seconds: int = 60) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds)
        self.request_admission = macbook_model_admission()