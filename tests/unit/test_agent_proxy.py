from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest


def _proxy_module():
    path = Path(__file__).resolve().parents[2] / "deploy" / "agent-sandbox" / "llm_proxy.py"
    spec = importlib.util.spec_from_file_location("agent_llm_proxy", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_proxy_route_model_budget_and_output_clamp():
    proxy = _proxy_module()
    state = proxy.ProxyState("stealth/crouton-alpha", 1, 2048)
    denied, _ = proxy.authorize_request("/v1/embeddings", {"model": "stealth/crouton-alpha"}, state)
    assert denied == 403
    denied, _ = proxy.authorize_request("/v1/chat/completions", {"model": "other/model"}, state)
    assert denied == 403

    payload = {"model": "stealth/crouton-alpha", "max_tokens": 9000, "max_completion_tokens": 9000}
    allowed, bounded = proxy.authorize_request("/v1/chat/completions", payload, state)
    assert allowed == 200
    assert bounded["max_tokens"] == bounded["max_completion_tokens"] == 2048
    exhausted, _ = proxy.authorize_request("/v1/chat/completions",
        deepcopy({"model": "stealth/crouton-alpha", "max_tokens": 1}), state)
    assert exhausted == 429


def test_proxy_adds_a_ceiling_when_request_omits_max_tokens():
    proxy = _proxy_module()
    state = proxy.ProxyState("stealth/crouton-alpha", 2, 1024)
    allowed, payload = proxy.authorize_request("/v1/chat/completions",
        {"model": "stealth/crouton-alpha"}, state)
    assert allowed == 200 and payload["max_tokens"] == 1024
    assert payload["usage"] == {"include": True}


def test_proxy_without_configured_cap_uses_pi_ceiling_without_injecting_tokens():
    proxy = _proxy_module()
    state = proxy.ProxyState("model", 3, None)

    allowed, bounded = proxy.authorize_request("/v1/chat/completions",
        {"model": "model", "max_tokens": 50000}, state)
    assert allowed == 200 and bounded["max_tokens"] == 32000

    allowed, bounded = proxy.authorize_request("/v1/chat/completions",
        {"model": "model", "max_tokens": 1000}, state)
    assert allowed == 200 and bounded["max_tokens"] == 1000

    allowed, bounded = proxy.authorize_request("/v1/chat/completions", {"model": "model"}, state)
    assert allowed == 200
    assert "max_tokens" not in bounded and "max_completion_tokens" not in bounded


def test_proxy_configured_output_cap_keeps_clamping_and_default_injection():
    proxy = _proxy_module()
    state = proxy.ProxyState("model", 2, 4096)

    allowed, bounded = proxy.authorize_request("/v1/chat/completions",
        {"model": "model", "max_tokens": 50000}, state)
    assert allowed == 200 and bounded["max_tokens"] == 4096

    allowed, bounded = proxy.authorize_request("/v1/chat/completions", {"model": "model"}, state)
    assert allowed == 200 and bounded["max_tokens"] == 4096


@pytest.mark.parametrize(("field", "value"), [
    ("max_tokens", 0),
    ("max_tokens", None),
    ("max_tokens", True),
    ("max_completion_tokens", -1),
    ("max_completion_tokens", "100"),
])
def test_proxy_rejects_non_positive_integer_output_limits(field, value):
    proxy = _proxy_module()
    state = proxy.ProxyState("model", 1, None)

    status, payload = proxy.authorize_request("/v1/chat/completions",
        {"model": "model", field: value}, state)

    assert status == 400 and payload is None


@pytest.mark.parametrize("configured_value", [None, ""])
def test_proxy_main_treats_missing_or_empty_output_cap_as_unset(monkeypatch, configured_value):
    proxy = _proxy_module()
    captured = {}

    class Server:
        def __init__(self, _address, handler):
            captured["state"] = handler.state

        def serve_forever(self):
            return None

    monkeypatch.setattr(proxy, "ThreadingHTTPServer", Server)
    monkeypatch.setenv("MODEL_ID", "model")
    monkeypatch.setenv("MAX_PROXY_CALLS", "1")
    monkeypatch.setenv("PROVIDER", "openrouter")
    if configured_value is None:
        monkeypatch.delenv("MAX_OUTPUT_TOKENS", raising=False)
    else:
        monkeypatch.setenv("MAX_OUTPUT_TOKENS", configured_value)

    proxy.main()

    assert captured["state"].max_output_tokens is None


def test_proxy_uses_fixed_provider_upstreams_and_opencode_headers():
    proxy = _proxy_module()
    openrouter = proxy.ProxyState("model-a", 1, 1024)
    openrouter_request = proxy.build_upstream_request(openrouter, {"model": "model-a"}, "test-key")
    assert openrouter_request.full_url == "https://openrouter.ai/api/v1/chat/completions"

    session_id = "ses_0123456789abcdef0123456789abcdef"
    opencode = proxy.ProxyState("model-b", 1, 1024, provider="opencode-go",
        upstream_session_id=session_id)
    request = proxy.build_upstream_request(opencode, {"model": "model-b"}, "test-key")
    headers = {key.lower(): value for key, value in request.header_items()}
    assert request.full_url == "https://opencode.ai/inference/go/openai/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-key"
    assert headers["user-agent"] == "review-bot-agent/0.1.0"
    assert headers["x-opencode-session"] == session_id


def test_opencode_request_omits_usage_include_and_unknown_provider_is_rejected():
    proxy = _proxy_module()
    state = proxy.ProxyState("model-b", 2, 1024, provider="opencode-go",
        upstream_session_id="ses_0123456789abcdef0123456789abcdef")
    allowed, payload = proxy.authorize_request("/v1/chat/completions",
        {"model": "model-b", "usage": {"include": True, "other": "value"}}, state)
    assert allowed == 200
    assert payload["usage"] == {"other": "value"}

    allowed, payload = proxy.authorize_request("/v1/chat/completions",
        {"model": "model-b"}, state)
    assert allowed == 200 and "usage" not in payload

    with pytest.raises(ValueError, match="unsupported_provider"):
        proxy.ProxyState("model", 1, 1024, provider="arbitrary")


def test_proxy_extracts_openrouter_usage_cost_and_writes_a_call_ledger(tmp_path):
    proxy = _proxy_module()
    payload = {"usage": {"prompt_tokens": 23, "completion_tokens": 7,
        "total_tokens": 30, "cost": 0.0025}}

    assert proxy.usage_record(payload) == {
        "prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30,
        "cost": {"total": 0.0025},
    }
    capture = proxy.ResponseUsageCapture("text/event-stream")
    capture.feed(b'data: {"usage":{"prompt_tokens":23,"completion_tokens":7,"cost":{"total":0.0025}}}\n')
    assert capture.result() == {"prompt_tokens": 23, "completion_tokens": 7,
        "cost": {"total": 0.0025}}

    ledger = tmp_path / "usage.jsonl"
    state = proxy.ProxyState("stealth/crouton-alpha", 1, 128, str(ledger))
    state.record({"call": 1, "status": 200, **proxy.usage_record(payload)})
    assert json.loads(ledger.read_text(encoding="utf-8"))["cost"]["total"] == 0.0025


def test_proxy_extracts_opencode_go_usage():
    proxy = _proxy_module()
    payload = {"usage": {"prompt_tokens": 23, "completion_tokens": 7,
        "total_tokens": 30, "cost": 0, "completion_tokens_details": {"reasoning_tokens": 4},
        "opencode": {"billing_source": "go", "list_cost": 0, "allowance": 0,
            "tokens": {"input": 23, "output": 7}}}}
    assert proxy.usage_record(payload) == {
        "prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30,
        "cost": {"total": 0, "list_cost": 0, "allowance": 0},
        "billing_source": "go",
    }
