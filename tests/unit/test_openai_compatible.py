"""Observable wire and factory contracts for replaceable model servers."""
import json
from unittest.mock import Mock

import httpx
import pytest

from review_bot.adapters.openai_compatible import OpenAICompatibleModelClient
from review_bot.application.architecture_review import PermanentModelError, ModelOutputLimitError
from review_bot.adapters.sqlite.model_admission import ModelAdmissionCancelled
from review_bot.config import ModelConfig
from review_bot.domain.models_v1 import ModelRequest
from review_bot.ports import JobStore, ModelInputAdmission, ModelLifecycle
from review_bot.runtime import model_clients
from review_bot.runtime.worker import WorkerLoop
from review_bot.runtime.mode_fingerprints import _profile


SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"], "additionalProperties": False}


@pytest.fixture(autouse=True)
def isolated_admission(tmp_path, monkeypatch):
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(tmp_path / "admission"))


def client(**options):
    return OpenAICompatibleModelClient(**{
        "api_url": "https://configured-server.example/prefix/v1", "model": "configured-model",
        "context_tokens": 262144, "framing_reserve_tokens": 0, **options})


def request(c):
    return ModelRequest(1, c.model, "Return JSON.", "Check this.", 10, 64, SCHEMA)


def response(model="configured-model", **changes):
    return {"model": model, "choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": '{"ok":true}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}, **changes}


def mock_http(monkeypatch, handler):
    actual = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: actual(transport=httpx.MockTransport(handler), **kw))


@pytest.mark.parametrize("model", ["qwen3.8-flash-next-iq3_xxs", "another-model"])
def test_server_and_model_change_without_metadata_or_tokenizer_apis(monkeypatch, model):
    calls = []
    def handler(r):
        calls.append(r)
        assert str(r.url) == "https://configured-server.example/prefix/v1/chat/completions"
        body = json.loads(r.content)
        assert body["model"] == model
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        assert not {"n_keep", "cache_prompt", "reasoning_budget_tokens"} & body.keys()
        return httpx.Response(200, json=response(model))
    mock_http(monkeypatch, handler)
    c = client(model=model)
    c.warmup()
    assert calls == []
    result = c.complete(request(c))
    assert json.loads(result.text) == {"ok": True}
    assert result.usage == {"input_tokens": 10, "output_tokens": 2}
    assert len(calls) == 1
    assert isinstance(c, ModelLifecycle) and isinstance(c, ModelInputAdmission)


def test_optional_auth_and_thinking_are_explicit_wire_settings(monkeypatch):
    def handler(r):
        assert r.headers["authorization"] == "Bearer private-test-key"
        assert json.loads(r.content)["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(200, json=response())
    mock_http(monkeypatch, handler)
    c = client(api_key="private-test-key", enable_thinking=False)
    assert "private-test-key" not in json.dumps(_profile(c))
    c.complete(request(c))


@pytest.mark.parametrize("url", ["ftp://host/v1", "https://user:secret@host/v1",
    "https://host/v1?key=secret", "https://host/v1#fragment", "https://host/ bad"])
def test_bad_endpoint_fails_without_revealing_it(url):
    with pytest.raises(ValueError, match="^Invalid explicit model connection settings$"):
        client(api_url=url)


def test_larger_server_context_does_not_expand_application_budget(monkeypatch):
    c = client(input_tokens=100, output_tokens=64)
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("No HTTP on overflow"))
    with pytest.raises(PermanentModelError, match="byte-upper-bound"):
        c.complete(request(c))
    assert c.profile.max_input_tokens == 100


@pytest.mark.parametrize("changes", [
    {"model": "wrong-model"}, {"usage": None},
    {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 99}},
    {"usage": {"prompt_tokens": True, "completion_tokens": 2, "total_tokens": 3}},
    {"usage": {"prompt_tokens": 10, "completion_tokens": 65, "total_tokens": 75}},
    {"choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": '{"ok":true}', "tool_calls": []}}]},
    {"choices": [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": '{"ok":true,"extra":1}'}}]},
])
def test_invalid_responses_never_become_success(monkeypatch, changes):
    mock_http(monkeypatch, lambda r: httpx.Response(200, json=response(**changes)))
    c = client()
    with pytest.raises(PermanentModelError):
        c.complete(request(c))


def test_truncation_retains_known_usage(monkeypatch):
    data = response()
    data["choices"][0]["finish_reason"] = "length"
    mock_http(monkeypatch, lambda r: httpx.Response(200, json=data))
    c = client()
    with pytest.raises(ModelOutputLimitError) as caught:
        c.complete(request(c))
    assert caught.value.usage == {"input_tokens": 10, "output_tokens": 2}
    assert caught.value.delivery == "sent"


def test_http_error_does_not_leak_response_or_credentials(monkeypatch):
    mock_http(monkeypatch, lambda r: httpx.Response(403, text="private-test-key secret provider body"))
    c = client(api_key="private-test-key")
    with pytest.raises(PermanentModelError, match="^Configured chat HTTP 403$"):
        c.complete(request(c))


def test_cancelled_request_sends_no_http(monkeypatch):
    c = client()
    c.set_cancellation_callback(lambda: True)
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("Cancelled request must not send"))
    with pytest.raises(ModelAdmissionCancelled):
        c.complete(request(c))


def configure(monkeypatch, role, model):
    prefix = "REVIEW_BOT_" + role + "_"
    monkeypatch.setenv(prefix + "BASE_URL", "http://configured-private-server:11435/v1")
    monkeypatch.setenv(prefix + "MODEL", model)
    monkeypatch.setenv(prefix + "CONTEXT_TOKENS", "262144")


def test_all_roles_are_independent_and_construction_is_offline(monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("No construction HTTP"))
    configure(monkeypatch, "REVIEW", "review-model")
    configure(monkeypatch, "PLANNING", "planning-model")
    configure(monkeypatch, "SELECTION", "selection-model")
    settings = ModelConfig()
    review = model_clients.build_review_client("openai-compatible", settings)
    planning = model_clients.build_planning_client("openai-compatible", settings)
    selector = model_clients.build_selector_client("openai-compatible", settings)
    assert [review.model, planning.model, selector.model] == ["review-model", "planning-model", "selection-model"]
    assert review.profile.max_input_tokens == 98394
    assert review.profile.max_output_tokens == 32678
    assert planning.profile.max_output_tokens == 4096
    assert model_clients.build_intent_selection_client("openai-compatible").model == "selection-model"


def test_missing_role_setting_fails_and_never_falls_back(monkeypatch):
    configure(monkeypatch, "REVIEW", "review-model")
    monkeypatch.delenv("REVIEW_BOT_REVIEW_MODEL")
    with pytest.raises(ValueError, match="REVIEW_BOT_REVIEW_MODEL"):
        model_clients.build_review_client("openai-compatible")


def test_generic_client_receives_worker_lifecycle_without_concrete_type_check(monkeypatch):
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)
    configure(monkeypatch, "REVIEW", "review-model")
    c = model_clients.build_review_client("openai-compatible")
    config = Mock()
    config.model = ModelConfig()
    worker = WorkerLoop(config, Mock(spec=JobStore), model_client=c)
    worker.stop()
    assert c.request_cancelled() is True


def test_cache_identity_changes_with_endpoint_model_and_wire_options():
    base = _profile(client())
    assert base != _profile(client(model="different"))
    assert base != _profile(client(api_url="https://another-server.example/v1"))
    assert base != _profile(client(enable_thinking=False))
    assert base == _profile(client(api_key="different-private-key"))


@pytest.mark.parametrize("key", ["bad\nheader", "비밀키"])
def test_invalid_auth_setting_is_rejected_without_echoing_secret(key):
    with pytest.raises(ValueError, match="^Invalid explicit model connection settings$"):
        client(api_key=key)
