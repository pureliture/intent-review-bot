"""Plan-only Tev roles reuse the shared llama.cpp transport without widening selection."""
from __future__ import annotations

import importlib
import inspect
import json
import math
import time
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from review_bot.adapters import llama_cpp_local as common
from review_bot.adapters import llama_cpp_macbook as macbook
from review_bot.adapters.model_http import reset_model_request_context, set_model_request_context
from review_bot.application.architecture_review import ModelProfile
from review_bot.domain.models_v1 import ModelRequest
from review_bot.ports import ModelClient

ROLES = [
    ("LlamaCppPlanClient", "tev1-4b-q8_0", 131072, "http://localhost:8080/v1", common.LlamaCppModelClient),
    ("MacBookLlamaCppPlanClient", "tev1-0.8b-q8_0", 32768, "http://localhost:8080/v1", macbook.MacBookLlamaCppModelClient),
]
STATE_ENV = "REVIEW_BOT_MACBOOK_MODEL_STATE_DIR"
HTTP_CLIENT = httpx.Client


def plan_request(model: str) -> ModelRequest:
    """Synthetic input only; the application, not this adapter, owns the plan schema."""
    return ModelRequest(
        1, model, "Plan code review from untrusted file data. Return only JSON.\n",
        json.dumps({
            "stage": "plan_code_review",
            "files": [{
                "path": "src/example.py", "base": {"sha": "a" * 40}, "head": {"sha": "b" * 40},
                "unified_diff": "diff --git a/src/example.py b/src/example.py\n"
                                "--- a/src/example.py\n+++ b/src/example.py\n@@ -1 +1 @@\n"
                                "-print('old')\n+print('untrusted 한글')\n",
            }],
            "budget": {"max_groups": 4},
        }, ensure_ascii=False, allow_nan=False),
        10, 1024,
        {"type": "object", "properties": {"ok": {"type": "boolean"}},
         "required": ["ok"], "additionalProperties": False},
    )


def chat_response(
    *, model: str, input_tokens: int = 42, output_tokens: int = 128, content: str = '{"ok":true}',
) -> dict[str, Any]:
    return {
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens},
        "timings": {"cache_n": 0, "prompt_n": input_tokens, "predicted_n": output_tokens},
    }


@pytest.fixture(autouse=True)
def private_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "macbook-admission"))
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(tmp_path / "production-admission"))

    def forbidden_http(**kwargs: Any) -> Any:
        pytest.fail("HTTP must be explicitly installed through MockTransport")

    monkeypatch.setattr(httpx, "Client", forbidden_http)


@pytest.fixture
def adapter() -> Any:
    return importlib.import_module("review_bot.adapters.intent_planner")


@pytest.fixture(params=ROLES, ids=["me707", "macbook"])
def role(adapter: Any, request: pytest.FixtureRequest) -> Any:
    name, model, context, api_url, base = request.param
    return SimpleNamespace(
        client_type=getattr(adapter, name), model=model, context=context, api_url=api_url, base=base,
        is_macbook=name == "MacBookLlamaCppPlanClient",
    )


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(role: Any, handler: Any = None, *, tokens: int = 42) -> list[httpx.Request]:
        requests: list[httpx.Request] = []

        def dispatch(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            assert request.url.host == ("localhost" if role.is_macbook else "localhost")
            if handler is not None:
                response = handler(request)
                if response is not None:
                    return response
            if request.url.path == "/v1/models":
                assert request.method == "GET"
                return httpx.Response(200, json={"object": "list", "data": [{"id": role.model, "object": "model"}]})
            if request.url.path == "/props":
                assert request.method == "GET"
                return httpx.Response(200, json={
                    "default_generation_settings": {"n_ctx": role.context},
                    "total_slots": 1 if role.is_macbook else 2, "chat_template": "{{ messages }}<assistant>",
                })
            assert request.method == "POST"
            assert request.url.path == "/v1/chat/completions"
            return httpx.Response(200, json=chat_response(model=role.model, input_tokens=tokens))

        def client(**kwargs: Any) -> httpx.Client:
            assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
            assert math.isfinite(kwargs["timeout"]) and kwargs["timeout"] > 0
            return HTTP_CLIENT(transport=httpx.MockTransport(dispatch), **kwargs)

        monkeypatch.setattr(httpx, "Client", client)
        return requests

    return install


def test_plan_contracts_and_constructors_are_offline(adapter: Any, tmp_path: Path) -> None:
    for name, model, context, api_url, base in ROLES:
        client_type = getattr(adapter, name)
        signature = inspect.signature(client_type)
        expected = ["model", "api_url", "timeout_seconds", "max_output_bytes", "max_attempts"]
        if name == "LlamaCppPlanClient":
            expected.append("allowed_hosts")
            assert signature.parameters["allowed_hosts"].default == ()
            assert signature.parameters["allowed_hosts"].kind == inspect.Parameter.KEYWORD_ONLY
        assert list(signature.parameters) == expected
        assert signature.parameters["model"].default == model
        assert signature.parameters["model"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert signature.parameters["api_url"].default == inspect.Parameter.empty
        for parameter in ("api_url", "timeout_seconds", "max_output_bytes", "max_attempts"):
            assert signature.parameters[parameter].kind == inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters["timeout_seconds"].default == 60
        assert signature.parameters["max_output_bytes"].default == 32768
        assert signature.parameters["max_attempts"].default == 1
        with pytest.raises(TypeError):
            client_type()
        client = client_type(api_url=api_url + "/")
        assert isinstance(client, base) and isinstance(client, ModelClient)
        assert (client.model, client.api_url, client.timeout_seconds) == (model, api_url, 60)
        assert client_type._role_model == model and client_type._role_context_tokens == context
        for method in ("complete", "verify", "warmup"):
            assert getattr(client_type, method) is getattr(base, method)
        assert not hasattr(client_type, "count_input_tokens")
        assert not hasattr(client_type, "token_count_receipt")
        assert client.profile == ModelProfile(
            provider="llamacpp", model=model, max_input_bytes=16 * 1024 * 1024,
            max_output_bytes=32768, timeout_seconds=60, max_attempts=1,
            max_input_tokens=context - 4096, max_output_tokens=4096,
        )
        assert not hasattr(client, "count_input_tokens")
        assert not hasattr(client, "token_count_receipt")
        configured = client_type(api_url=api_url, timeout_seconds=17, max_output_bytes=12000, max_attempts=2)
        assert (configured.profile.timeout_seconds, configured.profile.max_output_bytes,
                configured.profile.max_attempts) == (17, 12000, 2)
    assert not (tmp_path / "macbook-admission").exists()
    assert not (tmp_path / "production-admission").exists()


def test_plan_roles_reject_review_and_other_selector_aliases(role: Any) -> None:
    other_selector = "tev1-4b-q8_0" if role.is_macbook else "tev1-0.8b-q8_0"
    for wrong_model in ("bonsai2-27b", "lfm2.5-2.6b-q8_0", other_selector, role.model + ":cloud", "", None):
        with pytest.raises(ValueError, match="exact llama.cpp role model"):
            role.client_type(wrong_model, api_url=role.api_url)
    client = role.client_type(api_url=role.api_url)
    for wrong_model in ("bonsai2-27b", "lfm2.5-2.6b-q8_0", other_selector):
        request = replace(plan_request(role.model), model=wrong_model)
        with pytest.raises(common.PermanentModelError, match="does not match its role"):
            client.complete(request)
        with pytest.raises(common.PermanentModelError, match="does not match its role"):
            client.admission_bytes(request)
    assert not hasattr(client, "count_input_tokens")
    assert not hasattr(client, "token_count_receipt")


@pytest.mark.parametrize("kwargs", [
    {"timeout_seconds": 0}, {"timeout_seconds": True}, {"timeout_seconds": math.inf},
    {"max_output_bytes": 0}, {"max_output_bytes": True}, {"max_output_bytes": 16 * 1024 * 1024 + 1},
    {"max_attempts": 0}, {"max_attempts": True},
])
def test_plan_configuration_uses_common_validation(role: Any, kwargs: Any) -> None:
    with pytest.raises(ValueError):
        role.client_type(api_url=role.api_url, **kwargs)


@pytest.mark.parametrize("api_url", ["http://remote-model.internal:8080/v1", "http://100.64.0.1/v1"])
def test_macbook_planning_never_accepts_remote_endpoints(adapter: Any, api_url: str) -> None:
    with pytest.raises(ValueError):
        adapter.MacBookLlamaCppPlanClient(api_url=api_url)


def test_plan_generation_allows_more_than_64_tokens_and_preserves_exact_wire(role: Any, mock_http: Any) -> None:
    requests = mock_http(role)
    request = plan_request(role.model)
    client = role.client_type(api_url=role.api_url, timeout_seconds=17)
    response = client.complete(request)
    assert json.loads(response.text) == {"ok": True}
    assert response.usage == {"input_tokens": 42, "output_tokens": 128}
    assert [item.url.path for item in requests] == [
        "/v1/models", "/props", "/v1/chat/completions",
    ]
    assert sum(item.url.path == "/v1/chat/completions" for item in requests) == 1
    payload = json.loads(requests[2].content)
    messages = [{"role": "system", "content": request.system_prompt}, {"role": "user", "content": request.user_content}]
    assert payload["messages"] == messages and payload["model"] == role.model
    if role.is_macbook:
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    else:
        assert "chat_template_kwargs" not in payload
    assert payload["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "review_bot", "strict": True, "schema": request.response_schema}}
    assert payload["max_tokens"] == 1024
    assert (payload["stream"], payload["temperature"], payload["seed"], payload["n"]) == (False, 0, 0, 1)
    assert payload["stop"] == [] and payload["cache_prompt"] is False and payload["n_keep"] == -1
    assert all(0 < item.extensions["timeout"]["read"] <= 10 for item in requests)


def test_closed_choice_disables_thinking_in_generation_payload(role: Any, mock_http: Any) -> None:
    def reply(request):
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json=chat_response(model=role.model, output_tokens=6, content='{"choice":"A"}'))
        return None
    requests = mock_http(role, reply)
    schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
        "required": ["choice"], "additionalProperties": False}
    request = replace(plan_request(role.model), max_output_tokens=16, response_schema=schema)
    response = role.client_type(api_url=role.api_url).complete(request)
    payload = next(json.loads(item.content) for item in requests if item.url.path == "/v1/chat/completions")
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == 16
    assert payload["response_format"]["json_schema"]["schema"] == schema
    assert json.loads(response.text) == {"choice": "A"}
    assert response.usage == {"input_tokens": 42, "output_tokens": 6}
    assert [item.url.path for item in requests] == ["/v1/models", "/props", "/v1/chat/completions"]


def test_warmup_and_admission_bound_never_generate(role: Any, mock_http: Any) -> None:
    requests = mock_http(role)
    client = role.client_type(api_url=role.api_url)
    client.warmup()
    assert [item.url.path for item in requests] == ["/v1/models", "/props"]
    assert not hasattr(client, "count_input_tokens")
    assert not hasattr(client, "token_count_receipt")
    request = plan_request(role.model)
    bound = client.admission_bytes(request)
    assert type(bound) is int
    assert client.admission_bytes(request) == bound
    assert [item.url.path for item in requests] == ["/v1/models", "/props"]
    assert client.complete(request).usage == {"input_tokens": 42, "output_tokens": 128}
    assert [item.url.path for item in requests] == [
        "/v1/models", "/props", "/v1/models", "/props", "/v1/chat/completions",
    ]


@pytest.mark.parametrize("overflow", [False, True])
def test_plan_context_boundary_keeps_4096_output_reserve(role: Any, mock_http: Any, overflow: bool) -> None:
    tokens = role.context - 4096 + int(overflow)
    requests = mock_http(role, tokens=tokens)
    client = role.client_type(api_url=role.api_url)
    request = replace(plan_request(role.model), max_output_tokens=None)
    if overflow:
        with pytest.raises(common.PermanentModelError, match="budget"):
            client.complete(request)
        assert [item.url.path for item in requests] == ["/v1/models", "/props", "/v1/chat/completions"]
        assert sum(item.url.path == "/v1/chat/completions" for item in requests) == 1
    else:
        assert client.complete(request).usage == {"input_tokens": tokens, "output_tokens": 128}
        assert json.loads(requests[-1].content)["max_tokens"] == 4096


@pytest.mark.parametrize("fault", ["wrong-model", "wrong-context", "wrong-slots"])
def test_plan_verification_keeps_exact_alias_context_and_profile_slots(role: Any, mock_http: Any, fault: str) -> None:
    expected_slots = 1 if role.is_macbook else 2
    def handler(request: httpx.Request) -> Any:
        if fault == "wrong-model" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "wrong-role"}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": role.context + int(fault == "wrong-context")},
                "total_slots": 3 if fault == "wrong-slots" else expected_slots, "chat_template": "template",
            })
        return None

    requests = mock_http(role, handler)
    with pytest.raises(common.PermanentModelError):
        role.client_type(api_url=role.api_url).complete(plan_request(role.model))
    assert [item.url.path for item in requests] == (["/v1/models"] if fault == "wrong-model" else ["/v1/models", "/props"])


@pytest.mark.parametrize("text", [
    "not JSON", "[]", '{"ok":true,"ok":false}', '{"ok":NaN}', '{"ok":1e999}',
    '{"ok":true,"extra":"secret"}', '{"ok":true} trailing', '{"ok":"\\ud800"}',
])
def test_plan_output_keeps_strict_json_and_schema_fields(role: Any, mock_http: Any, text: str) -> None:
    mock_http(role, lambda request: httpx.Response(200, json=chat_response(model=role.model, content=text))
              if request.url.path == "/v1/chat/completions" else None)
    with pytest.raises(common.PermanentModelError):
        role.client_type(api_url=role.api_url).complete(plan_request(role.model))


@pytest.mark.parametrize("path,value", [
    (("model",), "wrong-role"), (("choices", 0, "finish_reason"), "length"),
    (("choices", 0, "message", "reasoning_content"), 42), (("choices", 0, "message", "tool_calls"), []),
    (("usage",), None), (("usage", "prompt_tokens"), 41), (("usage", "completion_tokens"), True),
    (("usage", "completion_tokens"), 1025), (("usage", "total_tokens"), 171),
    (("tokens_evaluated",), 41), (("truncated",), True),
    (("timings", "cache_n"), 1), (("timings", "predicted_n"), 127),
])
def test_plan_response_and_usage_must_match_native_count(role: Any, mock_http: Any, path: Any, value: Any) -> None:
    data = chat_response(model=role.model)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    mock_http(role, lambda request: httpx.Response(200, json=data)
              if request.url.path == "/v1/chat/completions" else None)
    with pytest.raises(common.PermanentModelError):
        role.client_type(api_url=role.api_url).complete(plan_request(role.model))


def test_plan_output_byte_limit_is_inherited(role: Any, mock_http: Any) -> None:
    mock_http(role)
    with pytest.raises(common.PermanentModelError, match="output exceeds byte limit"):
        role.client_type(api_url=role.api_url, max_output_bytes=8).complete(plan_request(role.model))


def test_generation_never_touches_measurement_endpoints(role: Any, mock_http: Any) -> None:
    requests = mock_http(role)
    assert role.client_type(api_url=role.api_url).complete(
        plan_request(role.model)).usage == {"input_tokens": 42, "output_tokens": 128}
    assert [item.url.path for item in requests] == ["/v1/models", "/props", "/v1/chat/completions"]
    assert sum(item.url.path == "/v1/chat/completions" for item in requests) == 1


@pytest.mark.parametrize("stage", ["/v1/models", "/props", "/v1/chat/completions"])
def test_generation_timeouts_fail_closed_without_second_post(role: Any, mock_http: Any, stage: str) -> None:
    def handler(request: httpx.Request) -> Any:
        if request.url.path == stage:
            raise httpx.ReadTimeout("offline timeout", request=request)
        return None

    requests = mock_http(role, handler)
    client = role.client_type(api_url=role.api_url)
    with pytest.raises(TimeoutError):
        client.complete(plan_request(role.model))
    assert not hasattr(client, "token_count_receipt")
    assert sum(item.url.path == "/v1/chat/completions" for item in requests) == (
        1 if stage == "/v1/chat/completions" else 0)
    assert all(item.url.path not in ("/apply-template", "/tokenize") for item in requests)


def test_plan_transport_does_not_retry_or_fall_back(role: Any, mock_http: Any) -> None:
    requests = mock_http(role, lambda request: httpx.Response(503, json={"error": "unavailable"})
                         if request.url.path == "/v1/chat/completions" else None)
    with pytest.raises(common.TransientModelError, match="HTTP 503"):
        role.client_type(api_url=role.api_url, max_attempts=3).complete(plan_request(role.model))
    assert [item.url.path for item in requests] == [
        "/v1/models", "/props", "/v1/chat/completions",
    ]
    assert sum(item.url.path == "/v1/chat/completions" for item in requests) == 1


def test_existing_metadata_selectors_still_reject_plan_requests(role: Any) -> None:
    client_type = macbook.MacBookLlamaCppDecisionClient if role.is_macbook else common.LlamaCppDecisionClient
    client = client_type(api_url=role.api_url)
    with pytest.raises(common.PermanentModelError, match="only accepts registered PR selection metadata"):
        client.complete(replace(plan_request(role.model), max_output_tokens=64))


def test_existing_metadata_selectors_keep_64_token_limit(role: Any) -> None:
    client_type = macbook.MacBookLlamaCppDecisionClient if role.is_macbook else common.LlamaCppDecisionClient
    request = replace(plan_request(role.model), max_output_tokens=65, user_content=json.dumps({
        "title": "Synthetic metadata", "body": "", "changed_files": ["src/example.py"], "available_modes": ["intent"],
    }))
    with pytest.raises(common.PermanentModelError, match="output token budget"):
        client_type(api_url=role.api_url).complete(request)


def test_two_macbook_planners_and_review_share_private_http_admission(
    adapter: Any, mock_http: Any, tmp_path: Path,
) -> None:
    api_url = "http://localhost:8080/v1"
    clients = [adapter.MacBookLlamaCppPlanClient(api_url=api_url),
               adapter.MacBookLlamaCppPlanClient(api_url=api_url),
               macbook.MacBookLlamaCppModelClient(api_url=api_url)]
    root = tmp_path / "macbook-admission"
    assert all(client.request_admission.db_path == root / "requests.sqlite" for client in clients)
    assert all(client.request_admission.max_concurrency == 2 for client in clients)
    assert not root.exists()
    requests = mock_http(SimpleNamespace(model="tev1-0.8b-q8_0", context=32768, is_macbook=True))
    with ExitStack() as stack:
        tickets = [stack.enter_context(client.request_admission.request(
            deadline=time.monotonic() + 5, generates=False,
        )) for client in clients[:2]]
        for ticket in tickets:
            ticket.mark_sent()
        token = set_model_request_context(lambda: False, time.monotonic() + 0.05)
        try:
            with pytest.raises(TimeoutError):
                clients[2].verify()
        finally:
            reset_model_request_context(token)
        assert requests == []
        for ticket in tickets:
            ticket.finish()
    for client in clients:
        requests = mock_http(SimpleNamespace(model=client.model, context=32768, is_macbook=True))
        client.verify()
        assert [item.url.path for item in requests] == ["/v1/models", "/props"]
    assert root.stat().st_mode & 0o777 == 0o700
    assert (root / "requests.sqlite").stat().st_mode & 0o777 == 0o600
    assert not (tmp_path / "production-admission").exists()


def test_macbook_plan_default_state_never_falls_back_to_production(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.delenv(STATE_ENV)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    client = adapter.MacBookLlamaCppPlanClient(api_url="http://localhost:8080/v1")
    expected = tmp_path / "home/.local/state/github-review-bot/macbook-model-admission/requests.sqlite"
    assert client.request_admission.db_path == expected
    assert not expected.parent.exists() and not (tmp_path / "production-admission").exists()


def test_macbook_plan_cannot_use_production_admission_state(
    adapter: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(STATE_ENV, str(tmp_path / "production-admission"))
    with pytest.raises(ValueError, match="separate from production state"):
        adapter.MacBookLlamaCppPlanClient(api_url="http://localhost:8080/v1")
    assert not (tmp_path / "production-admission").exists()


def test_subset_selection_disables_thinking_and_uses_one_post(role, mock_http):
    from review_bot.application.partition_choices import perspective_subset_schema
    from review_bot.domain.intent_perspectives import PERSPECTIVES
    selected = [PERSPECTIVES[0].id, PERSPECTIVES[4].id]
    def handler(request):
        if request.url.path == "/v1/chat/completions":
            wire = json.loads(request.content)
            assert wire["chat_template_kwargs"]["enable_thinking"] is False
            assert wire["max_tokens"] == 256
            return httpx.Response(200, json=chat_response(model=role.model,
                content=json.dumps({"selected_perspective_ids": selected})))
    requests = mock_http(role, handler)
    request = replace(plan_request(role.model), max_output_tokens=256,
        response_schema=perspective_subset_schema(), user_content=json.dumps({"stage": "select_perspectives", "state": {}}))
    response = role.client_type(api_url=role.api_url).complete(request)
    assert json.loads(response.text)["selected_perspective_ids"] == selected
    assert sum(r.url.path == "/v1/chat/completions" for r in requests) == 1
