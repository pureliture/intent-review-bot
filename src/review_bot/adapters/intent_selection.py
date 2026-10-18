"""Thin Tev clients for closed choices about predefined intent perspectives."""
from __future__ import annotations

from review_bot.adapters.llamacpp import SELECTOR_MODEL, LlamaCppModelClient
from review_bot.adapters.llamacpp_macbook import MACBOOK_SELECTOR_MODEL, MacBookLlamaCppModelClient


class LlamaCppIntentSelectionClient(LlamaCppModelClient):
    _role_model = SELECTOR_MODEL
    _role_context_tokens = 131072
    _role_max_output_tokens = 4096

    def __init__(self, model: str = SELECTOR_MODEL, *, api_url: str,
        timeout_seconds: int = 60, max_output_bytes: int = 512, max_attempts: int = 1):
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
            max_output_bytes=min(max_output_bytes, 512), max_attempts=1)

    def _chat_payload(self, request):
        payload = super()._chat_payload(request)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload


class MacBookIntentSelectionClient(MacBookLlamaCppModelClient):
    _role_model = MACBOOK_SELECTOR_MODEL
    _role_context_tokens = 32768

    def __init__(self, model: str = MACBOOK_SELECTOR_MODEL, *, api_url: str,
        timeout_seconds: int = 60, max_output_bytes: int = 512, max_attempts: int = 1):
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
            max_output_bytes=min(max_output_bytes, 512), max_attempts=1)

    def _chat_payload(self, request):
        payload = super()._chat_payload(request)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload
