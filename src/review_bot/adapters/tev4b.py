"""Explicit loopback TEV1 4B roles over the existing native llama.cpp transport."""
from __future__ import annotations

from review_bot.adapters.intent_planner import MacBookLlamaCppPlanClient
from review_bot.adapters.intent_selection import MacBookIntentSelectionClient
from review_bot.adapters.llamacpp_macbook import MacBookLlamaCppDecisionClient

TEV4B_PROFILE = "macbook-beam-cloud-tev4b"
TEV4B_MODEL = "tev1-4b-q8_0"
TEV4B_CONTEXT_TOKENS = 131072
TEV4B_DEFAULT_API_URL = "http://127.0.0.1:18139/v1"


class MacBookTev4bPlanClient(MacBookLlamaCppPlanClient):
    _role_model = TEV4B_MODEL
    _role_context_tokens = TEV4B_CONTEXT_TOKENS
    _server_generation_slots = 1

    def __init__(self, model: str = TEV4B_MODEL, *, api_url: str,
        timeout_seconds: int = 60, max_output_bytes: int = 32768, max_attempts: int = 1) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes, max_attempts=max_attempts)


class MacBookTev4bDecisionClient(MacBookLlamaCppDecisionClient):
    _role_model = TEV4B_MODEL
    _role_context_tokens = TEV4B_CONTEXT_TOKENS
    _server_generation_slots = 1

    def __init__(self, model: str = TEV4B_MODEL, *, api_url: str, timeout_seconds: int = 60) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds)


class MacBookTev4bIntentSelectionClient(MacBookIntentSelectionClient):
    _role_model = TEV4B_MODEL
    _role_context_tokens = TEV4B_CONTEXT_TOKENS
    _server_generation_slots = 1

    def __init__(self, model: str = TEV4B_MODEL, *, api_url: str,
        timeout_seconds: int = 60, max_output_bytes: int = 512, max_attempts: int = 1) -> None:
        super().__init__(model, api_url=api_url, timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes, max_attempts=max_attempts)
