"""Explicit local model profile composition; construction performs no HTTP calls."""
from __future__ import annotations

from review_bot.adapters.ollama import OLLAMA_PROFILE, OllamaModelClient
from review_bot.config import ModelConfig


def build_review_client(profile: str | None, settings: ModelConfig | None = None) -> OllamaModelClient | None:
    if profile is None:
        return None
    if profile != OLLAMA_PROFILE:
        raise ValueError(f"Unsupported model profile: {profile}")
    settings = settings or ModelConfig()
    return OllamaModelClient(
        timeout_seconds=settings.call_timeout_seconds, max_output_bytes=settings.max_output_bytes,
        max_attempts=settings.max_attempts,
    )