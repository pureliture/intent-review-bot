"""Zero-measurement proofs for the OpenRouter validation adapters.

Each generation-path test asserts the exact HTTP call ledger: one
catalog GET for verify, then exactly one generation POST per complete.
Any second POST, count-only variant, num_predict probe, or tokenizer
endpoint fails the test. Application layers may only record the usage
that rides on the single generation response.
"""
from dataclasses import replace

import httpx
import pytest

from review_bot.adapters.model_http import reset_model_request_context, set_model_request_context
from review_bot.adapters.openrouter_chat import APP_INPUT_TOKENS, OpenRouterReviewClient
from review_bot.adapters.openrouter_decisions import OpenRouterDecisionClient
from review_bot.application.architecture_review import PermanentModelError
from review_bot.domain.models_v1 import ModelRequest, ModelResponse

HTTP_CLIENT = httpx.Client
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False}
REVIEW_MODEL = "google/gemma-4-31b-it:free"
DECISION_MODEL = "perplexity/pplx-decider-flash-v1.2"


def review_request(**changes):
    value = ModelRequest(1, REVIEW_MODEL, "Review the supplied source.", "Untrusted PR source 한글",
        20, 32, SCHEMA)
    return replace(value, **changes)


def selection_request(**changes):
    import json
    body = json.dumps({"title": "t", "body": "b", "changed_files": ["a.py"],
        "available_modes": ["architecture", "intent"]})
    value = ModelRequest(1, DECISION_MODEL, "system", body, 20, 64,
        {"type": "object", "properties": {"selected_modes": {"type": "array", "items": {"type": "string"}}},
            "required": ["selected_modes"], "additionalProperties": False})
    return replace(value, **changes)


@pytest.fixture
def ledger(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(tmp_path / "admission"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    calls = []
    behavior = {}

    def handler(req):
        assert req.headers.get("authorization") == "Bearer test-key"
        assert "num_predict" not in req.content.decode("utf-8", errors="ignore")
        calls.append((req.method, req.url.path))
        if req.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": REVIEW_MODEL, "context_length": 262144,
                    "architecture": {"output_modalities": ["text"]}},
                {"id": DECISION_MODEL, "context_length": 262144,
                    "architecture": {"output_modalities": ["decisions"]}}]})
        if req.url.path == "/api/v1/chat/completions":
            assert behavior.get("chat_status", 200) == 200 or True
            if "chat_status" in behavior:
                return httpx.Response(behavior["chat_status"], json=behavior.get("chat_body", {"error": "x"}))
            return httpx.Response(200, json={
                "model": REVIEW_MODEL,
                "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 64, "completion_tokens": 4, "total_tokens": 68}})
        if req.url.path == "/api/alpha/decisions":
            return httpx.Response(200, json={
                "id": "gen-dec-test", "model": DECISION_MODEL,
                "answers": {"review_mode": {"type": "choice", "choice": "architecture_and_intent",
                    "probabilities": {"architecture": 0.2, "intent": 0.2,
                        "architecture_and_intent": 0.59, "not_applicable": 0.01}}},
                "usage": {"input_tokens": 32, "output_tokens": 2}})
        raise AssertionError(f"unexpected OpenRouter path {req.url.path}")

    def client(**options):
        assert options["trust_env"] is False and options["follow_redirects"] is False
        return HTTP_CLIENT(transport=httpx.MockTransport(handler), **options)

    monkeypatch.setattr(httpx, "Client", client)
    import time
    token = set_model_request_context(lambda: False, time.monotonic() + 60)
    yield calls, behavior
    reset_model_request_context(token)


def test_review_single_generation_no_measurement(ledger):
    calls, _ = ledger
    client = OpenRouterReviewClient(api_key="test-key")
    first = client.complete(review_request())
    assert isinstance(first, ModelResponse) and first.usage == {"input_tokens": 64, "output_tokens": 4}
    second = client.complete(review_request())
    assert second.usage == {"input_tokens": 64, "output_tokens": 4}
    assert calls == [("GET", "/api/v1/models"),
        ("POST", "/api/v1/chat/completions"), ("POST", "/api/v1/chat/completions")]
    assert not hasattr(client, "count_input_tokens")
    assert not hasattr(client, "token_count_receipt")


def test_review_byte_budget_refuses_without_any_generation_post(ledger):
    calls, _ = ledger
    client = OpenRouterReviewClient(api_key="test-key")
    oversized = review_request(user_content="x" * (APP_INPUT_TOKENS + 1))
    with pytest.raises(PermanentModelError):
        client.complete(oversized)
    assert calls == [("GET", "/api/v1/models")]
    assert not any(path == "/api/v1/chat/completions" for _, path in calls)


def test_review_usage_over_budget_discards_result(ledger):
    calls, behavior = ledger
    import json
    def handler_override(req):
        calls.append((req.method, req.url.path))
        if req.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": REVIEW_MODEL, "context_length": 262144,
                    "architecture": {"output_modalities": ["text"]}}]})
        return httpx.Response(200, json={
            "model": REVIEW_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": APP_INPUT_TOKENS + 1, "completion_tokens": 4,
                "total_tokens": APP_INPUT_TOKENS + 5}})
    import review_bot.adapters.openrouter_common as common
    monkeypatch_client = httpx.Client
    # Replace transport-level handler through a fresh client patch.
    import httpx as _httpx_mod
    orig = _httpx_mod.Client
    def client(**options):
        return HTTP_CLIENT(transport=_httpx_mod.MockTransport(handler_override), **options)
    _httpx_mod.Client = client
    try:
        customer = OpenRouterReviewClient(api_key="test-key")
        with pytest.raises(PermanentModelError):
            customer.complete(review_request())
    finally:
        _httpx_mod.Client = orig
    posts = [path for _, path in calls if path == "/api/v1/chat/completions"]
    assert len(posts) == 1


def test_review_auth_failure_is_permanent(ledger):
    calls, behavior = ledger
    behavior["chat_status"] = 401
    client = OpenRouterReviewClient(api_key="test-key")
    with pytest.raises(PermanentModelError):
        client.complete(review_request())
    assert calls.count(("POST", "/api/v1/chat/completions")) == 1


def test_decision_single_request_no_measurement(ledger):
    calls, _ = ledger
    client = OpenRouterDecisionClient(api_key="test-key")
    response = client.complete(selection_request())
    assert isinstance(response, ModelResponse)
    import json
    assert json.loads(response.text) == {"selected_modes": ["architecture", "intent"]}
    assert response.usage == {"input_tokens": 32, "output_tokens": 2}
    assert calls == [("GET", "/api/v1/models"), ("POST", "/api/alpha/decisions")]
    assert not hasattr(client, "count_input_tokens")


def test_decision_rejects_invalid_probabilities(ledger):
    calls, _ = ledger
    import httpx as _httpx_mod
    orig = _httpx_mod.Client
    def handler(req):
        calls.append((req.method, req.url.path))
        if req.url.path == "/api/v1/models":
            return _httpx_mod.Response(200, json={"data": [
                {"id": DECISION_MODEL, "context_length": 262144,
                    "architecture": {"output_modalities": ["decisions"]}}]})
        return _httpx_mod.Response(200, json={
            "id": "gen-dec-bad", "model": DECISION_MODEL,
            "answers": {"review_mode": {"type": "choice", "choice": "not_applicable",
                "probabilities": {"architecture": 0.5, "intent": 0.3, "not_applicable": 0.2}}},
            "usage": {"input_tokens": 32, "output_tokens": 2}})
    def client(**options):
        return HTTP_CLIENT(transport=_httpx_mod.MockTransport(handler), **options)
    _httpx_mod.Client = client
    try:
        customer = OpenRouterDecisionClient(api_key="test-key")
        with pytest.raises(PermanentModelError):
            customer.complete(selection_request())
    finally:
        _httpx_mod.Client = orig


def test_missing_api_key_fails_before_http(ledger):
    calls, _ = ledger
    with pytest.raises(ValueError):
        OpenRouterReviewClient(api_key="")
    with pytest.raises(ValueError):
        OpenRouterDecisionClient(api_key="   ")
    assert calls == []
