"""Thin plan-only Tev roles; the application owns review-plan semantics."""
from __future__ import annotations

from review_bot.adapters.llama_cpp_local import SELECTOR_MODEL, LlamaCppModelClient
from review_bot.adapters.llama_cpp_macbook import MACBOOK_SELECTOR_MODEL, MacBookLlamaCppModelClient
from review_bot.domain.models_v1 import ModelRequest


class LlamaCppPlanClient(LlamaCppModelClient):
    """Tev code-review planning role, separate from metadata-only selected_modes selection."""

    _role_model = SELECTOR_MODEL
    _role_context_tokens = 131072
    _role_max_output_tokens = 4096

    def __init__(
        self, model: str = SELECTOR_MODEL, *, api_url: str, timeout_seconds: int = 60,
        max_output_bytes: int = 32768, max_attempts: int = 1,
        allowed_hosts: tuple[str, ...] | list[str] = (),
    ) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
                         max_output_bytes=max_output_bytes, max_attempts=max_attempts, allowed_hosts=allowed_hosts)

    def _chat_payload(self, request: ModelRequest) -> dict:
        payload = super()._chat_payload(request)
        properties = request.response_schema.get("properties")
        if isinstance(properties, dict) and set(properties) == {"choice"}:
            # The closed label allowance must not be spent on hidden reasoning.
            # Keep the closed-choice wire free of hidden reasoning.
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload


class MacBookLlamaCppPlanClient(MacBookLlamaCppModelClient):
    """Local32k Tev code-review planning role, not the metadata-only mode selector."""

    _role_model = MACBOOK_SELECTOR_MODEL
    _role_context_tokens = 32768

    def __init__(
        self, model: str = MACBOOK_SELECTOR_MODEL, *, api_url: str, timeout_seconds: int = 60,
        max_output_bytes: int = 32768, max_attempts: int = 1,
    ) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
                         max_output_bytes=max_output_bytes, max_attempts=max_attempts)
