"""Synthetic HTTP evidence for the explicitly authorized beam cloud adapter."""
from copy import deepcopy
from dataclasses import replace
import json
import sqlite3

import httpx
import pytest

from review_bot.adapters.beam_cloud import (
    BEAM_CONTEXT_TOKENS, BEAM_MAX_OUTPUT_TOKENS, BeamCloudModelClient,
)
from review_bot.adapters.model_http import reset_model_request_context, set_model_request_context
from review_bot.application.architecture_review import ModelOutputLimitError, PermanentModelError
from review_bot.domain.models_v1 import ModelRequest

HTTP_CLIENT = httpx.Client


def request(**changes):
    value = ModelRequest(1, "beam:cloud", "Review the supplied source.", "Untrusted PR source 한글", 20, 32,
        {"type": "object", "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"], "additionalProperties": False})
    return replace(value, **changes)


def completion(maximum, *, prompt=64):
    return {"model": "beam", "done": True, "done_reason": "length" if maximum == 1 else "stop",
        "prompt_eval_count": prompt, "prompt_eval_cached_count": 0, "eval_count": 1 if maximum == 1 else 4,
        "message": {"role": "assistant", "content": "" if maximum == 1 else '{"ok":true}',
            "thinking": "Provider reasoning is separate from the final JSON."}}


@pytest.fixture
def transport(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(tmp_path / "admission"))
    calls = []
    changes = {}

    def handler(req):
        payload = json.loads(req.content) if req.content else None
        calls.append((req.method, req.url.path, deepcopy(payload)))
        assert req.url.host == "127.0.0.1" and req.url.port == 11434
        if req.url.path == "/api/tags":
            data = {"models": [{"name": "beam:cloud", "model": "beam:cloud",
                "remote_model": "beam", "remote_host": "https://ollama.com"}]}
        elif req.url.path == "/api/show":
            assert payload == {"model": "beam:cloud"}
            data = {"model_info": {"general.architecture": "beam", "beam.context_length": 1048576},
                "details": {"family": "beam", "parent_model": "beam"}, "capabilities": ["completion", "thinking", "tools"]}
        else:
            assert req.url.path == "/api/chat"
            data = completion(payload["options"]["num_predict"], prompt=changes.get("prompt", 64))
        if changes.get("handler"):
            replacement = changes["handler"](req, payload, data)
            if isinstance(replacement, httpx.Response):
                return replacement
            if replacement is not None:
                data = replacement
        return httpx.Response(200, json=data)

    def client(**options):
        assert options["trust_env"] is False and options["follow_redirects"] is False
        assert 0 < options["timeout"] <= 60
        return HTTP_CLIENT(transport=httpx.MockTransport(handler), **options)

    monkeypatch.setattr(httpx, "Client", client)
    return calls, changes, tmp_path / "admission/requests.sqlite"


def test_construction_is_offline_and_identity_is_explicit(monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **kw: pytest.fail("Constructor must remain offline"))
    client = BeamCloudModelClient()
    assert client.supports_native_input_measurement is True
    assert client.supports_bounded_reasoning is False
    assert (client.profile.provider, client.profile.model) == ("ollama-cloud", "beam:cloud")
    assert client.profile.max_input_tokens == 1048576 - 32678
    assert client.profile.max_output_tokens == 32678 and client.profile.max_input_bytes == 16 * 1024 * 1024
    for options in ({"model": "beam"}, {"model": "other:cloud"}, {"api_url": "https://ollama.com"},
            {"api_url": "http://user:secret@localhost:11434"}, {"timeout_seconds": 0}, {"max_output_bytes": True}):
        with pytest.raises(ValueError):
            BeamCloudModelClient(**options)


def test_effective_full_schema_prompt_metering_memo_and_reasoning_success(transport):
    calls, _, database = transport
    client = BeamCloudModelClient()
    client.warmup()
    assert [path for _, path, _ in calls] == ["/api/tags", "/api/show"]
    task = request()
    assert client.count_input_tokens(task) == 64
    assert client.token_count_receipt == {"delivery": "sent", "model_calls": 1,
        "usage": {"input_tokens": 64, "output_tokens": 1}}
    before = len(calls)
    assert client.count_input_tokens(task) == 64 and len(calls) == before
    assert client.token_count_receipt == {"delivery": "not_sent", "model_calls": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0}}
    result = client.complete(task)
    assert json.loads(result.text) == {"ok": True} and result.usage == {"input_tokens": 64, "output_tokens": 4}
    chats = [payload for _, path, payload in calls if path == "/api/chat"]
    assert [item["options"]["num_predict"] for item in chats] == [1, 32]
    assert chats[0]["messages"] == chats[1]["messages"]
    assert chats[0]["format"] == task.response_schema
    assert json.dumps(task.response_schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")) in chats[0]["messages"][0]["content"]
    assert chats[0]["messages"][1]["content"] == task.user_content
    assert all(p["truncate"] is False and p["shift"] is False and p["stream"] is False for p in chats)
    assert all(p["options"]["num_ctx"] == 1048576 for p in chats)
    changed_schema = {**task.response_schema, "description": "Different effective contract"}
    client.count_input_tokens(replace(task, response_schema=changed_schema))
    assert client.token_count_receipt["model_calls"] == 1
    client._verified_until = 0
    client.count_input_tokens(task)
    assert client.token_count_receipt["model_calls"] == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_admission").fetchone()[0] == 0


@pytest.mark.parametrize("fault", ["model", "remote_host", "context"])
def test_metadata_mismatch_has_no_paid_evaluation(transport, fault):
    calls, changes, _ = transport
    def corrupt(req, payload, data):
        if req.url.path == "/api/tags" and fault in ("model", "remote_host"):
            data["models"][0][fault] = "other" if fault == "model" else "https://unapproved.invalid"
        if req.url.path == "/api/show" and fault == "context":
            data["model_info"]["beam.context_length"] = 131072
        return data
    changes["handler"] = corrupt
    with pytest.raises(PermanentModelError):
        BeamCloudModelClient().count_input_tokens(request())
    assert all(path != "/api/chat" for _, path, _ in calls)


def test_native_input_reservation_rejects_before_full_generation(transport):
    calls, changes, _ = transport
    changes["prompt"] = BEAM_CONTEXT_TOKENS - BEAM_MAX_OUTPUT_TOKENS + 1
    client = BeamCloudModelClient()
    assert client.count_input_tokens(request()) == changes["prompt"]
    with pytest.raises(PermanentModelError, match="actual token reservation") as caught:
        client.complete(request())
    assert caught.value.delivery == "not_sent" and caught.value.usage == {"input_tokens": 0, "output_tokens": 0}
    assert [p["options"]["num_predict"] for _, path, p in calls if path == "/api/chat"] == [1]
    assert client.token_count_receipt["usage"]["input_tokens"] == changes["prompt"]


def test_known_length_preserves_exact_content_thinking_and_usage(transport):
    _, changes, database = transport
    def limited(req, payload, data):
        if req.url.path == "/api/chat" and payload["options"]["num_predict"] > 1:
            data.update(done_reason="length", eval_count=32)
            data["message"].update(content='{"partial":', thinking="Original provider thinking.")
        return data
    changes["handler"] = limited
    client = BeamCloudModelClient()
    client.count_input_tokens(request())
    with pytest.raises(ModelOutputLimitError) as caught:
        client.complete(request())
    error = caught.value
    assert error.delivery == "sent" and error.usage == {"input_tokens": 64, "output_tokens": 32}
    assert error.rejected_output == {"finish_reason": "length", "content": '{"partial":',
        "reasoning_content": "Original provider thinking."}
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_admission").fetchone()[0] == 0


@pytest.mark.parametrize("fault", ["missing_usage", "read_timeout"])
def test_unknown_consumption_retains_shared_generating_ticket(transport, fault):
    _, changes, database = transport
    def fail(req, payload, data):
        if req.url.path == "/api/chat":
            if fault == "read_timeout":
                raise httpx.ReadTimeout("secret-provider-detail", request=req)
            data.pop("prompt_eval_count")
        return data
    changes["handler"] = fail
    client = BeamCloudModelClient()
    with pytest.raises((PermanentModelError, TimeoutError)) as caught:
        client.count_input_tokens(request())
    assert "secret-provider-detail" not in str(caught.value)
    assert client.token_count_receipt == {"delivery": "unknown", "model_calls": None, "usage": None}
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT state FROM model_admission").fetchall() == [("uncertain",)]


def test_scope_cancellation_prevents_metadata_or_generation(transport):
    import time
    calls, _, _ = transport
    token = set_model_request_context(lambda: True, time.monotonic() + 20)
    try:
        with pytest.raises(Exception, match="cancel"):
            BeamCloudModelClient().count_input_tokens(request())
    finally:
        reset_model_request_context(token)
    assert calls == []


def test_cloud_json_is_validated_without_claiming_provider_grammar(transport):
    _, changes, _ = transport
    def extra_field(req, payload, data):
        if req.url.path == "/api/chat" and payload["options"]["num_predict"] > 1:
            data["message"]["content"] = '{"ok":true,"unregistered":1}'
        return data
    changes["handler"] = extra_field
    client = BeamCloudModelClient()
    client.count_input_tokens(request())
    with pytest.raises(PermanentModelError, match="Unexpected") as caught:
        client.complete(request())
    assert caught.value.delivery == "sent" and caught.value.usage == {"input_tokens": 64, "output_tokens": 4}


def test_direct_or_expired_completion_never_hides_a_paid_measurement(transport):
    calls, _, _ = transport
    client = BeamCloudModelClient()
    with pytest.raises(PermanentModelError, match="fresh exact native") as direct:
        client.complete(request())
    assert direct.value.delivery == "not_sent" and direct.value.usage == {"input_tokens": 0, "output_tokens": 0}
    assert calls == []
    client.count_input_tokens(request())
    before = deepcopy(calls)
    client._verified_until = 0
    with pytest.raises(PermanentModelError, match="fresh exact native") as expired:
        client.complete(request())
    assert expired.value.delivery == "not_sent" and expired.value.usage == {"input_tokens": 0, "output_tokens": 0}
    assert calls == before
    client.verify()  # Metadata refresh invalidates the old measurement without evaluating a prompt.
    before = deepcopy(calls)
    with pytest.raises(PermanentModelError, match="fresh exact native"):
        client.complete(request())
    assert calls == before
