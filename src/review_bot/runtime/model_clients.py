"""Explicit model profile composition; construction performs no HTTP calls."""
from __future__ import annotations

import os

from review_bot.adapters.llamacpp import LLAMACPP_PROFILE, LlamaCppDecisionClient, LlamaCppModelClient
from review_bot.adapters.llamacpp_macbook import (
    MACBOOK_LLAMACPP_PROFILE,
    MacBookLlamaCppDecisionClient,
    MacBookLlamaCppModelClient,
    macbook_model_admission,
)
from review_bot.adapters.model_http import default_model_admission
from review_bot.adapters.ollama import OLLAMA_PROFILE, OllamaDecisionClient, OllamaModelClient
from review_bot.config import ModelConfig
from review_bot.ports import ModelClient

MODEL_PROFILES = (OLLAMA_PROFILE, LLAMACPP_PROFILE, MACBOOK_LLAMACPP_PROFILE)


def _endpoint(variable: str, profile: str = LLAMACPP_PROFILE) -> str:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"{variable} must be set for {profile}")
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
        client = LlamaCppModelClient(api_url=_endpoint("ME707_BONSAI_BASE_URL"), **options)
    elif profile == MACBOOK_LLAMACPP_PROFILE:
        client = MacBookLlamaCppModelClient(api_url=_endpoint("MACBOOK_LFM_BASE_URL", profile), **options)
    else:
        client = OllamaModelClient(**options)
    client.request_admission = (
        macbook_model_admission(settings.max_concurrency) if profile == MACBOOK_LLAMACPP_PROFILE
        else default_model_admission(settings.max_concurrency)
    )
    return client


def build_selector_client(profile: str | None, settings: ModelConfig | None = None) -> ModelClient | None:
    if profile is None:
        return None
    if profile not in MODEL_PROFILES:
        raise ValueError("Unsupported model profile")
    settings = settings or ModelConfig()
    if profile == LLAMACPP_PROFILE:
        client = LlamaCppDecisionClient(
            api_url=_endpoint("ME707_TEV1_BASE_URL"), timeout_seconds=settings.call_timeout_seconds,
        )
    elif profile == MACBOOK_LLAMACPP_PROFILE:
        client = MacBookLlamaCppDecisionClient(
            api_url=_endpoint("MACBOOK_TEV1_BASE_URL", profile), timeout_seconds=settings.call_timeout_seconds,
        )
    else:
        client = OllamaDecisionClient(timeout_seconds=settings.call_timeout_seconds)
    client.request_admission = (
        macbook_model_admission(settings.max_concurrency) if profile == MACBOOK_LLAMACPP_PROFILE
        else default_model_admission(settings.max_concurrency)
    )
    return client