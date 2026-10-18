"""Thin plan-only Tev roles; the application owns review-plan semantics."""
from __future__ import annotations

from review_bot.adapters.llamacpp import SELECTOR_MODEL, LlamaCppModelClient
from review_bot.adapters.llamacpp_macbook import MACBOOK_SELECTOR_MODEL, MacBookLlamaCppModelClient


class LlamaCppPlanClient(LlamaCppModelClient):
    """Tev code-review planning role, separate from metadata-only selected_modes selection."""

    _role_model = SELECTOR_MODEL
    _role_context_tokens = 131072
    _role_max_output_tokens = 4096

    def __init__(
        self, model: str = SELECTOR_MODEL, *, api_url: str, timeout_seconds: int = 60,
        max_output_bytes: int = 32768, max_attempts: int = 1,
    ) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
                         max_output_bytes=max_output_bytes, max_attempts=max_attempts)


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
