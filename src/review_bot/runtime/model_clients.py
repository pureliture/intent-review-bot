"""Explicit model profile composition; construction performs no HTTP calls."""
from __future__ import annotations

import os

from review_bot.adapters.llamacpp import LLAMACPP_PROFILE, LlamaCppDecisionClient, LlamaCppModelClient
from review_bot.adapters.ollama import OLLAMA_PROFILE, OllamaDecisionClient, OllamaModelClient
from review_bot.config import ModelConfig
from review_bot.ports import ModelClient

MODEL_PROFILES = (OLLAMA_PROFILE, LLAMACPP_PROFILE)


def _endpoint(variable: str) -> str:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"{variable} must be set for {LLAMACPP_PROFILE}")
    return value


def build_review_client(profile: str | None, settings: ModelConfig | None = None) -> ModelClient | None:
    if profile is None:
        return None
    if profile not in MODEL_PROFILES:
        raise ValueError("Unsupported model profile")
    settings = settings or ModelConfig()
    options = dict(
        timeout_seconds=settings.call_timeout_seconds, max_output_bytes=settings.max_output_bytes,
        max_attempts=settings.max_attempts,
    )
    if profile == LLAMACPP_PROFILE:
        return LlamaCppModelClient(api_url=_endpoint("ME707_BONSAI_BASE_URL"), **options)
    return OllamaModelClient(**options)


def build_selector_client(profile: str | None, settings: ModelConfig | None = None) -> ModelClient | None:
    if profile is None:
        return None
    if profile not in MODEL_PROFILES:
        raise ValueError("Unsupported model profile")
    settings = settings or ModelConfig()
    if profile == LLAMACPP_PROFILE:
        return LlamaCppDecisionClient(
            api_url=_endpoint("ME707_TEV1_BASE_URL"), timeout_seconds=settings.call_timeout_seconds,
        )
    return OllamaDecisionClient(timeout_seconds=settings.call_timeout_seconds)