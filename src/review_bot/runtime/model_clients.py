"""Explicit model profile composition; construction performs no HTTP calls."""
from __future__ import annotations

import os

from review_bot.adapters.llamacpp import LLAMACPP_PROFILE, LlamaCppDecisionClient, LlamaCppModelClient
from review_bot.adapters.llamacpp_macbook import (
    MACBOOK_LLAMACPP_PROFILE,
    MACBOOK_GRANITE_PROFILE,
    MacBookGraniteLlamaCppModelClient,
    MacBookLlamaCppDecisionClient,
    MacBookLlamaCppModelClient,
    macbook_model_admission,
)
from review_bot.adapters.model_http import default_model_admission
from review_bot.adapters.ollama import OLLAMA_PROFILE, OllamaDecisionClient, OllamaModelClient
from review_bot.config import ModelConfig
from review_bot.ports import ModelClient

BEAM_CLOUD_TEV4B_PROFILE = "macbook-beam-cloud-tev4b"
MACBOOK_PROFILES = (MACBOOK_LLAMACPP_PROFILE, MACBOOK_GRANITE_PROFILE, BEAM_CLOUD_TEV4B_PROFILE)
MODEL_PROFILES = (OLLAMA_PROFILE, LLAMACPP_PROFILE, *MACBOOK_PROFILES)


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
    if profile == BEAM_CLOUD_TEV4B_PROFILE:
        from review_bot.adapters.beam_cloud import BeamCloudModelClient
        client = BeamCloudModelClient(api_url=_endpoint("MACBOOK_BEAM_BASE_URL", profile), **options)
    elif profile == LLAMACPP_PROFILE:
        client = LlamaCppModelClient(api_url=_endpoint("ME707_BONSAI_BASE_URL"), **options)
    elif profile == MACBOOK_LLAMACPP_PROFILE:
        client = MacBookLlamaCppModelClient(api_url=_endpoint("MACBOOK_LFM_BASE_URL", profile), **options)
    elif profile == MACBOOK_GRANITE_PROFILE:
        client = MacBookGraniteLlamaCppModelClient(api_url=_endpoint("MACBOOK_GRANITE_BASE_URL", profile), **options)
    else:
        client = OllamaModelClient(**options)
    client.request_admission = (
        macbook_model_admission(settings.max_concurrency) if profile in MACBOOK_PROFILES
        else default_model_admission(settings.max_concurrency)
    )
    return client


def build_selector_client(profile: str | None, settings: ModelConfig | None = None) -> ModelClient | None:
    if profile is None:
        return None
    if profile not in MODEL_PROFILES:
        raise ValueError("Unsupported model profile")
    settings = settings or ModelConfig()
    if profile == BEAM_CLOUD_TEV4B_PROFILE:
        from review_bot.adapters.tev4b import MacBookTev4bDecisionClient
        client = MacBookTev4bDecisionClient(
            api_url=_endpoint("MACBOOK_TEV4B_BASE_URL", profile), timeout_seconds=settings.call_timeout_seconds)
    elif profile == LLAMACPP_PROFILE:
        client = LlamaCppDecisionClient(
            api_url=_endpoint("ME707_TEV1_BASE_URL"), timeout_seconds=settings.call_timeout_seconds,
        )
    elif profile in MACBOOK_PROFILES:
        client = MacBookLlamaCppDecisionClient(
            api_url=_endpoint("MACBOOK_TEV1_BASE_URL", profile), timeout_seconds=settings.call_timeout_seconds,
        )
    else:
        client = OllamaDecisionClient(timeout_seconds=settings.call_timeout_seconds)
    client.request_admission = (
        macbook_model_admission(settings.max_concurrency) if profile in MACBOOK_PROFILES
        else default_model_admission(settings.max_concurrency)
    )
    return client


def build_planning_client(profile: str | None, settings: ModelConfig | None = None) -> ModelClient | None:
    """Opted-in B planning is a separate role; absent endpoints never select another provider."""
    if profile is None or profile == OLLAMA_PROFILE:
        return None
    if profile not in MODEL_PROFILES:
        raise ValueError("Unsupported model profile")
    from review_bot.adapters.intent_planner import LlamaCppPlanClient, MacBookLlamaCppPlanClient

    variable = ("MACBOOK_TEV4B_BASE_URL" if profile == BEAM_CLOUD_TEV4B_PROFILE else
        "MACBOOK_TEV1_BASE_URL" if profile in MACBOOK_PROFILES else "ME707_TEV1_BASE_URL")
    if not os.environ.get(variable, "").strip():
        return None
    settings = settings or ModelConfig()
    options = dict(api_url=_endpoint(variable, profile), timeout_seconds=settings.call_timeout_seconds,
        max_output_bytes=settings.max_output_bytes, max_attempts=settings.max_attempts)
    if profile == BEAM_CLOUD_TEV4B_PROFILE:
        from review_bot.adapters.tev4b import MacBookTev4bPlanClient
        client = MacBookTev4bPlanClient(**options)
    else:
        client = MacBookLlamaCppPlanClient(**options) if profile in MACBOOK_PROFILES else LlamaCppPlanClient(**options)
    client.request_admission = (
        macbook_model_admission(settings.max_concurrency) if profile in MACBOOK_PROFILES
        else default_model_admission(settings.max_concurrency)
    )
    return client


def build_intent_selection_client(profile: str | None, settings: ModelConfig | None = None) -> ModelClient | None:
    """Construct the explicit closed-choice Tev role for single-run intent review."""
    if profile is None:
        return None
    if profile not in (LLAMACPP_PROFILE, *MACBOOK_PROFILES):
        raise ValueError("Intent perspective selection requires an explicit llama.cpp profile")
    from review_bot.adapters.intent_selection import LlamaCppIntentSelectionClient, MacBookIntentSelectionClient

    settings = settings or ModelConfig()
    local = profile in MACBOOK_PROFILES
    variable = ("MACBOOK_TEV4B_BASE_URL" if profile == BEAM_CLOUD_TEV4B_PROFILE else
        "MACBOOK_TEV1_BASE_URL" if local else "ME707_TEV1_BASE_URL")
    if profile == BEAM_CLOUD_TEV4B_PROFILE:
        from review_bot.adapters.tev4b import MacBookTev4bIntentSelectionClient
        cls = MacBookTev4bIntentSelectionClient
    else:
        cls = MacBookIntentSelectionClient if local else LlamaCppIntentSelectionClient
    client = cls(api_url=_endpoint(variable, profile), timeout_seconds=settings.call_timeout_seconds)
    client.request_admission = (macbook_model_admission(settings.max_concurrency) if local
        else default_model_admission(settings.max_concurrency))
    return client
