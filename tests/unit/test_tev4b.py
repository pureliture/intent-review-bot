"""The 4B loopback roles retain native admission and closed-choice transport."""
from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from review_bot.adapters import tev4b
from review_bot.application import partition_choices as choices
from review_bot.application.architecture_review import PermanentModelError
from review_bot.domain.models_v1 import ModelRequest
from tests.unit.test_intent_planner_client import chat_response, mock_http, private_state

API_URL = "http://localhost:18139/v1"
CLIENTS = (tev4b.MacBookTev4bPlanClient, tev4b.MacBookTev4bDecisionClient,
    tev4b.MacBookTev4bIntentSelectionClient)


def role():
    return SimpleNamespace(model=tev4b.TEV4B_MODEL, context=131072, is_macbook=True)


def test_exact_4b_identity_is_offline_and_server_context_must_be_128k(mock_http):
    calls = mock_http(role())
    clients = [client_type(api_url=API_URL) for client_type in CLIENTS]
    assert not calls
    assert tev4b.TEV4B_PROFILE == "macbook-beam-cloud-tev4b"
    assert tev4b.TEV4B_DEFAULT_API_URL == "http://127.0.0.1:18139/v1"
    for client in clients:
        assert client.model == "tev1-4b-q8_0"
        assert client._role_context_tokens == 131072 and client._server_generation_slots == 1
        assert client._max_input_tokens == 131072 - 4096
        client.verify()
        if hasattr(client, "profile"):
            assert client.profile.model == "tev1-4b-q8_0"
            assert client.profile.max_input_tokens == 131072 - 4096
            assert client.profile.max_output_tokens == 4096
    assert {request.url.path for request in calls} == {"/v1/models", "/props"}
    count = len(calls)
    for client_type in CLIENTS:
        for wrong_model in ("tev1-0.8b-q8_0", "tev1-4b-q8_0:cloud"):
            with pytest.raises(ValueError, match="exact llama.cpp role model"):
                client_type(wrong_model, api_url=API_URL)
        with pytest.raises(ValueError, match="endpoint"):
            client_type(api_url="http://example.com:18139/v1")
    assert len(calls) == count
    server = role()
    server.context = 32768
    short_calls = mock_http(server)
    for client_type in CLIENTS:
        with pytest.raises(PermanentModelError):
            client_type(api_url=API_URL).verify()
    assert {request.url.path for request in short_calls} == {"/v1/models", "/props"}


def test_closed_planning_and_perspective_choices_use_exact_wire_without_measurement(mock_http):
    options = choices.mode_options(("architecture", "intent"))
    selected = next(option for option in options if option["key"] == "review:intent")

    def response(request):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=chat_response(model=tev4b.TEV4B_MODEL,
                content=json.dumps({"choice": selected["label"]}), input_tokens=42, output_tokens=8))
        return None

    calls = mock_http(role(), response)
    request = ModelRequest(1, tev4b.TEV4B_MODEL, choices.SYSTEM,
        json.dumps({"stage": "select_review_scope", "mode": "planning", "state": {"field": "global_modes"},
            "options": options}), 10, 8, choices.choice_schema(options))
    for client_type in (tev4b.MacBookTev4bPlanClient, tev4b.MacBookTev4bIntentSelectionClient):
        client = client_type(api_url=API_URL)
        before = len(calls)
        bound = client.admission_bytes(request)
        assert type(bound) is int
        assert len(calls) == before
        result = client.complete(request)
        assert choices.accept_choice(json.loads(result.text), options)["modes"] == ["intent"]
        assert result.usage == {"input_tokens": 42, "output_tokens": 8}
    generations = [json.loads(call.content) for call in calls if call.url.path == "/v1/chat/completions"]
    assert len(generations) == 2
    assert all(payload["chat_template_kwargs"] == {"enable_thinking": False} for payload in generations)
    assert all(payload["model"] == "tev1-4b-q8_0" and payload["max_tokens"] == 8
        and payload["response_format"]["json_schema"]["schema"] == request.response_schema for payload in generations)
    assert [call.url.path for call in calls] == ["/v1/models", "/props", "/v1/chat/completions"] * 2
    assert not any(call.url.path in ("/apply-template", "/tokenize") for call in calls)
    count = len(calls)
    with pytest.raises(PermanentModelError, match="registered PR selection metadata"):
        tev4b.MacBookTev4bDecisionClient(api_url=API_URL).complete(request)
    assert len(calls) == count
