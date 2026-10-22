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
from review_bot.adapters.openrouter_chat import (
    OPENROUTER_REVIEW_PROFILE,
    OpenRouterReviewClient,
)
from review_bot.adapters.openrouter_decisions import (
    DECISION_MODEL as OPENROUTER_DECISION_MODEL_DEFAULT,
    OPENROUTER_DECISION_PROFILE,
    OpenRouterDecisionClient,
)
from review_bot.adapters.openrouter_chat import REVIEW_MODEL as OPENROUTER_REVIEW_MODEL_DEFAULT
from review_bot.config import ModelConfig
from review_bot.ports import ModelClient

BEAM_CLOUD_TEV4B_PROFILE = "macbook-beam-cloud-tev4b"
MACBOOK_PROFILES = (MACBOOK_LLAMACPP_PROFILE, MACBOOK_GRANITE_PROFILE, BEAM_CLOUD_TEV4B_PROFILE)
# Deployment profile: review/planning/flint-chat and decision/pplx-decisions
# under one name, mirroring me707-llamacpp (bonsai review + tev1 selection).
OPENROUTER_PROFILE = "openrouter"
OPENROUTER_PROFILES = (OPENROUTER_PROFILE, OPENROUTER_REVIEW_PROFILE, OPENROUTER_DECISION_PROFILE)
MODEL_PROFILES = (OLLAMA_PROFILE, LLAMACPP_PROFILE, *MACBOOK_PROFILES, *OPENROUTER_PROFILES)


def _endpoint(variable: str, profile: str = LLAMACPP_PROFILE) -> str:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"{variable} must be set for {profile}")
    return value


def _openrouter_key(profile: str) -> str:
    value = os.environ.get("OPENROUTER_API_KEY", "")
    if not value.strip():
        raise ValueError(f"OPENROUTER_API_KEY must be set for {profile}")
    return value


def _openrouter_model(variable: str, default: str, profile: str) -> str:
    value = os.environ.get(variable, "").strip()
    if not value:
        return default
    if len(value) > 256:
        raise ValueError(f"{variable} must be a valid OpenRouter model id for {profile}")
    return value


def _openrouter_origin() -> str | None:
    value = os.environ.get("OPENROUTER_ORIGIN", "").strip()
    return value or None


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
    if profile in (BEAM_CLOUD_TEV4B_PROFILE, OPENROUTER_PROFILE, OPENROUTER_REVIEW_PROFILE):
        if profile == BEAM_CLOUD_TEV4B_PROFILE and os.environ.get("MACBOOK_BEAM_CONTEXT_TOKENS"):
            raise ValueError("MACBOOK_BEAM_CONTEXT_TOKENS is retired with the Beam bridge; use the OpenRouter app budget")
        client = OpenRouterReviewClient(api_key=_openrouter_key(profile), origin=_openrouter_origin(),
            model=_openrouter_model("OPENROUTER_REVIEW_MODEL", OPENROUTER_REVIEW_MODEL_DEFAULT, profile),
            timeout_seconds=settings.call_timeout_seconds, max_output_bytes=settings.max_output_bytes,
            max_attempts=settings.max_attempts, app_input_tokens=settings.review_input_tokens,
            app_output_tokens=settings.review_output_tokens,
            framing_reserve_tokens=settings.framing_reserve_tokens,
            max_transport_bytes=settings.max_transport_bytes)
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
    elif profile in (OPENROUTER_PROFILE, OPENROUTER_DECISION_PROFILE):
        client = OpenRouterDecisionClient(api_key=_openrouter_key(profile), origin=_openrouter_origin(),
            model=_openrouter_model("OPENROUTER_DECISION_MODEL", OPENROUTER_DECISION_MODEL_DEFAULT, profile),
            timeout_seconds=settings.call_timeout_seconds, app_input_tokens=settings.decision_input_tokens,
            app_output_tokens=settings.decision_output_tokens,
            framing_reserve_tokens=settings.framing_reserve_tokens,
            max_transport_bytes=settings.max_transport_bytes)
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
    """Opted-in planning is a generative role; decision-only profiles cannot plan."""
    if profile is None or profile == OLLAMA_PROFILE or profile == OPENROUTER_DECISION_PROFILE:
        return None
    if profile not in MODEL_PROFILES:
        raise ValueError("Unsupported model profile")
    from review_bot.adapters.intent_planner import LlamaCppPlanClient, MacBookLlamaCppPlanClient

    variable = ("MACBOOK_TEV4B_BASE_URL" if profile == BEAM_CLOUD_TEV4B_PROFILE else
        "MACBOOK_TEV1_BASE_URL" if profile in MACBOOK_PROFILES else
        "OPENROUTER_API_KEY" if profile in OPENROUTER_PROFILES else "ME707_TEV1_BASE_URL")
    if not os.environ.get(variable, "").strip():
        return None
    settings = settings or ModelConfig()
    if profile in OPENROUTER_PROFILES:
        return OpenRouterReviewClient(api_key=_openrouter_key(profile), origin=_openrouter_origin(),
            model=_openrouter_model("OPENROUTER_REVIEW_MODEL", OPENROUTER_REVIEW_MODEL_DEFAULT, profile),
            timeout_seconds=settings.call_timeout_seconds, max_output_bytes=settings.max_output_bytes,
            max_attempts=settings.max_attempts, app_input_tokens=settings.review_input_tokens,
            app_output_tokens=settings.review_output_tokens,
            framing_reserve_tokens=settings.framing_reserve_tokens,
            max_transport_bytes=settings.max_transport_bytes)
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
    if profile == OPENROUTER_DECISION_PROFILE:
        raise ValueError("Intent perspective selection needs text generation; openrouter-decision cannot serve it")
    if profile == OPENROUTER_PROFILE:
        return build_intent_selection_client(OPENROUTER_REVIEW_PROFILE, settings)
    if profile == OPENROUTER_REVIEW_PROFILE:
        settings = settings or ModelConfig()
        from review_bot.adapters.openrouter_chat import OpenRouterReviewClient as _IntentReviewClient
        client = _IntentReviewClient(api_key=_openrouter_key(profile), origin=_openrouter_origin(),
            model=_openrouter_model("OPENROUTER_REVIEW_MODEL", OPENROUTER_REVIEW_MODEL_DEFAULT, profile),
            timeout_seconds=settings.call_timeout_seconds, max_output_bytes=min(settings.max_output_bytes, 512),
            max_attempts=1, app_input_tokens=settings.review_input_tokens,
            app_output_tokens=settings.review_output_tokens,
            framing_reserve_tokens=settings.framing_reserve_tokens,
            max_transport_bytes=settings.max_transport_bytes)
        client.request_admission = default_model_admission(settings.max_concurrency)
        return client
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
