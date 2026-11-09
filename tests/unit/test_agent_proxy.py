from __future__ import annotations

import importlib.util
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
