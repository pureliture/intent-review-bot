from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path


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
