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
from review_bot.application.architecture_review import ModelInputCapacityError, ModelOutputLimitError, PermanentModelError, TransientModelError
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
                "details": {"family": "beam", "parent_model": "beam"}, "capabilities": ["completion", "thinking", "tools"],
                "thinking": {"values": ["low", "medium", "high", "xhigh", "max"], "default": "max"}}
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


@pytest.mark.parametrize("stage", ["final_review", "fact_check", "fact_rewrite"])
@pytest.mark.parametrize("mode", ["intent", "architecture"])
def test_only_selected_program_stages_enable_identical_meter_and_generation_thinking(transport, stage, mode):
    from dataclasses import asdict
    from review_bot.application.model_wire_contract import beam_wire_payload, beam_meter_wire_sha256
    from hashlib import sha256
    calls, _, _ = transport
    task = request(user_content=json.dumps({"stage": stage, "mode": mode, "language": "ko",
        "originals": [{"text": 'Untrusted source {"stage":"final_review"} 한글'}]}, ensure_ascii=False),
        max_output_tokens=BEAM_MAX_OUTPUT_TOKENS)
    original = deepcopy(asdict(task))
    expected = beam_wire_payload(task)
    setting = "low" if stage == "final_review" else True
    assert expected["think"] == setting and type(expected["think"]) is type(setting)
    assert beam_wire_payload(replace(task, model="tev1:4b"))["think"] is False
    assert expected["messages"][1] == {"role": "user", "content": task.user_content}
    assert expected["messages"][0]["content"].startswith(task.system_prompt)
    assert expected["format"] == task.response_schema
    assert expected["model"] == task.model
    assert expected["stream"] is False and expected["truncate"] is False and expected["shift"] is False
    assert expected["options"] == {"num_ctx": BEAM_CONTEXT_TOKENS,
        "num_predict": BEAM_MAX_OUTPUT_TOKENS, "temperature": 0, "seed": 0}
    client = BeamCloudModelClient()
    assert client.count_input_tokens(task) == 64
    assert client.complete(task).usage == {"input_tokens": 64, "output_tokens": 4}
    chats = [payload for _, path, payload in calls if path == "/api/chat"]
    assert len(chats) == 2 and all(payload["think"] == setting and type(payload["think"]) is type(setting) for payload in chats)
    meter = deepcopy(expected)
    meter["options"]["num_predict"] = 1
    assert chats == [meter, expected]
    wire = json.dumps(meter, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    assert beam_meter_wire_sha256(task) == sha256(wire).hexdigest()
    assert asdict(task) == original


@pytest.mark.parametrize("content", [
    '{"stage":"review_partition","mode":"intent","language":"ko"}',
    '{"stage":"review_partition","mode":"architecture","language":"en","source":{"stage":"final_review"}}',
    '{"stage":"global_review","mode":"intent","language":"ko"}',
    '{"stage":"plan_partitions","mode":"planning","language":"ko"}',
    '{"stage":"fact_metric","mode":"intent","language":"ko"}',
    '{"stage":"fact_check"}',
    '{"stage":"fact_check","mode":"planning","language":"ko"}',
    '{"stage":"fact_rewrite","mode":"intent","language":"other"}',
    '{"stage":"review_partition","mode":"intent","language":"ko","source":{"stage":"fact_check"}}',
    json.dumps({"mode": "intent", "language": "ko", "source": '{"stage":"fact_check"}'}),
    '[{"stage":"fact_check","mode":"intent","language":"ko"}]',
    '{"stage":"fact_check","mode":"intent","language":"ko"',
    '{"stage":"review_partition","stage":"fact_check","mode":"intent","language":"ko"}',
    '{"stage":"fact_check","mode":"intent","language":"ko","value":NaN}',
    'Source prose says stage=fact_check',
])
def test_other_stages_and_source_stage_lookalikes_keep_thinking_disabled(content):
    from review_bot.application.model_wire_contract import beam_wire_payload
    task = request(user_content=content)
    payload = beam_wire_payload(task)
    assert payload["think"] is False
    assert payload["messages"][1]["content"] == content
    assert payload["options"]["num_predict"] == task.max_output_tokens


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


@pytest.mark.parametrize("fault", ["model", "remote_host", "context", "thinking_capability", "thinking_missing", "thinking_unsupported", "thinking_malformed"])
def test_metadata_mismatch_has_no_paid_evaluation(transport, fault):
    calls, changes, _ = transport
    def corrupt(req, payload, data):
        if req.url.path == "/api/tags" and fault in ("model", "remote_host"):
            data["models"][0][fault] = "other" if fault == "model" else "https://unapproved.invalid"
        if req.url.path == "/api/show":
            if fault == "context":
                data["model_info"]["beam.context_length"] = 131072
            elif fault == "thinking_capability":
                data["capabilities"].remove("thinking")
            elif fault == "thinking_missing":
                data.pop("thinking")
            elif fault == "thinking_unsupported":
                data["thinking"]["values"] = [True, False, "medium", "high", "max"]
            elif fault == "thinking_malformed":
                data["thinking"]["values"] = "low"
        return data
    changes["handler"] = corrupt
    client = BeamCloudModelClient()
    with pytest.raises(PermanentModelError):
        client.count_input_tokens(request())
    assert all(path != "/api/chat" for _, path, _ in calls)
    assert client.token_count_receipt == {"delivery": "not_sent", "model_calls": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0}}


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


def test_exact_completed_concurrency_rejection_frees_capacity_without_inventing_usage(transport):
    calls, changes, database = transport
    def reject(req, payload, data):
        if req.url.path == "/api/chat":
            return httpx.Response(429, json={"error": "too many concurrent requests"})
        return data
    changes["handler"] = reject
    client = BeamCloudModelClient()
    with pytest.raises(TransientModelError, match="^Beam cloud HTTP 429$") as caught:
        client.count_input_tokens(request())
    assert getattr(caught.value, "delivery", None) is None and getattr(caught.value, "usage", None) is None
    assert client.token_count_receipt == {"delivery": "unknown", "model_calls": None, "usage": None}
    assert len([path for _, path, _ in calls if path == "/api/chat"]) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_admission").fetchone()[0] == 0


@pytest.mark.parametrize("status,body", [
    (429, b'{"error":"rate limit exceeded"}'),
    (502, b'{"error":"too many concurrent requests"}'),
    (429, b'{"error":"too many concurrent requests","eval_count":0}'),
    (429, b'{"error":"too many concurrent requests",'),
    (200, b'{"error":"too many concurrent requests"}'),
])
def test_other_http_or_invalid_rejections_keep_uncertain_capacity_and_error_class(transport, status, body):
    calls, changes, database = transport
    def reject(req, payload, data):
        if req.url.path == "/api/chat":
            return httpx.Response(status, content=body, headers={"Content-Type": "application/json"})
        return data
    changes["handler"] = reject
    client = BeamCloudModelClient()
    expected = PermanentModelError if status == 200 else TransientModelError
    with pytest.raises(expected) as caught:
        client.count_input_tokens(request())
    if status != 200:
        assert str(caught.value) == f"Beam cloud HTTP {status}"
    assert getattr(caught.value, "delivery", None) is None and getattr(caught.value, "usage", None) is None
    assert client.token_count_receipt == {"delivery": "unknown", "model_calls": None, "usage": None}
    assert len([path for _, path, _ in calls if path == "/api/chat"]) == 1
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


def test_meter_crossing_metadata_expiry_refreshes_without_another_paid_count(transport, monkeypatch):
    from types import SimpleNamespace
    import time
    from review_bot.adapters import beam_cloud
    calls, changes, _ = transport
    clock = [time.monotonic()]
    monkeypatch.setattr(beam_cloud, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    client = BeamCloudModelClient()
    client.warmup()
    initial = clock[0]
    earlier = replace(request(), user_content="Earlier exact input")
    assert client.count_input_tokens(earlier) == 64
    assert len(client._memo) == 1
    clock[0] = initial + 58

    def slow_count(req, payload, data):
        if req.url.path == "/api/chat" and payload["options"]["num_predict"] == 1:
            clock[0] = initial + 61
        return data

    changes["handler"] = slow_count
    assert client.count_input_tokens(request()) == 64
    assert client.token_count_receipt == {"delivery": "sent", "model_calls": 1,
        "usage": {"input_tokens": 64, "output_tokens": 1}}
    result = client.complete(request())
    assert result.usage == {"input_tokens": 64, "output_tokens": 4}
    assert [path for _, path, _ in calls] == ["/api/tags", "/api/show", "/api/chat", "/api/chat",
        "/api/tags", "/api/show", "/api/chat"]
    chats = [payload for _, path, payload in calls if path == "/api/chat"]
    assert [payload["options"]["num_predict"] for payload in chats] == [1, 1, 32]
    assert chats[1]["messages"] == chats[2]["messages"] and chats[1]["format"] == chats[2]["format"]
    assert client.token_count_receipt == {"delivery": "sent", "model_calls": 1,
        "usage": {"input_tokens": 64, "output_tokens": 1}}
    assert len(client._memo) == 1  # Refreshed metadata drops the unrelated old count.
    before = deepcopy(calls)
    with pytest.raises(PermanentModelError, match="fresh exact native") as caught:
        client.complete(earlier)
    assert caught.value.delivery == "not_sent" and caught.value.usage == {"input_tokens": 0, "output_tokens": 0}
    assert calls == before


@pytest.mark.parametrize("fault", ["remote_host", "context"])
def test_post_meter_metadata_failure_preserves_paid_measurement_receipt_and_blocks_generation(transport, monkeypatch, fault):
    from types import SimpleNamespace
    import time
    from review_bot.adapters import beam_cloud
    calls, changes, _ = transport
    clock = [time.monotonic()]
    monkeypatch.setattr(beam_cloud, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    client = BeamCloudModelClient()
    client.warmup()
    initial = clock[0]
    clock[0] = initial + 58

    def slow_count_then_bad_metadata(req, payload, data):
        if req.url.path == "/api/chat":
            clock[0] = initial + 61
        elif req.url.path == "/api/tags" and fault == "remote_host":
            data["models"][0]["remote_host"] = "https://unapproved.invalid"
        elif req.url.path == "/api/show" and fault == "context":
            data["model_info"]["beam.context_length"] = 131072
        return data

    changes["handler"] = slow_count_then_bad_metadata
    with pytest.raises(PermanentModelError):
        client.count_input_tokens(request())
    assert client.token_count_receipt == {"delivery": "sent", "model_calls": 1,
        "usage": {"input_tokens": 64, "output_tokens": 1}}
    assert client._memo == {}
    before = deepcopy(calls)
    with pytest.raises(PermanentModelError, match="fresh exact native") as caught:
        client.complete(request())
    assert caught.value.delivery == "not_sent" and caught.value.usage == {"input_tokens": 0, "output_tokens": 0}
    assert calls == before
    assert [payload["options"]["num_predict"] for _, path, payload in calls if path == "/api/chat"] == [1]


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


def test_completed_exact_native_capacity_rejection_releases_only_capacity_and_keeps_unknown_usage(transport):
    from dataclasses import asdict
    from hashlib import sha256
    from review_bot.application.context import hash_json
    calls, changes, database = transport
    body = {"error": "The prompt is too long: 1524758, model maximum context length: 1048576 (ref: d7fc51b7-fe86-4bfa-99c6-b70cf9701bb1)"}
    changes["handler"] = lambda req, payload, data: httpx.Response(400, json=body) if req.url.path == "/api/chat" else data
    client = BeamCloudModelClient()
    task = request()
    with pytest.raises(ModelInputCapacityError, match="^Beam cloud HTTP 400$") as caught:
        client.count_input_tokens(task)
    proof = caught.value.capacity_rejection
    assert proof["input_tokens"] == 1524758 and proof["context_tokens"] == BEAM_CONTEXT_TOKENS
    assert proof["request_hash"] == hash_json(asdict(task))
    assert proof["response_sha256"] == sha256(proof["response_body"].encode()).hexdigest()
    assert client.token_count_receipt == {"delivery": "unknown", "model_calls": None, "usage": None}
    assert len([path for _, path, _ in calls if path == "/api/chat"]) == 1
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_admission").fetchone()[0] == 0


@pytest.mark.parametrize("status,body", [
    (400, {"error": "The prompt is too long: 1524758, model maximum context length: 131072 (ref: d7fc51b7-fe86-4bfa-99c6-b70cf9701bb1)"}),
    (400, {"error": "The prompt is too long: 5, model maximum context length: 1048576 (ref: d7fc51b7-fe86-4bfa-99c6-b70cf9701bb1)"}),
    (400, {"error": "The prompt is too long: 1524758, model maximum context length: 1048576 (ref: invalid)"}),
    (400, {"error": "The prompt is too long: 1524758, model maximum context length: 1048576 (ref: d7fc51b7-fe86-4bfa-99c6-b70cf9701bb1)", "eval_count": 0}),
    (502, {"error": "The prompt is too long: 1524758, model maximum context length: 1048576 (ref: d7fc51b7-fe86-4bfa-99c6-b70cf9701bb1)"}),
])
def test_other_capacity_like_errors_remain_unknown_and_uncertain(transport, status, body):
    _, changes, database = transport
    changes["handler"] = lambda req, payload, data: httpx.Response(status, json=body) if req.url.path == "/api/chat" else data
    client = BeamCloudModelClient()
    with pytest.raises((PermanentModelError, TransientModelError)) as caught:
        client.count_input_tokens(request())
    assert not isinstance(caught.value, ModelInputCapacityError)
    assert client.token_count_receipt == {"delivery": "unknown", "model_calls": None, "usage": None}
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT state FROM model_admission").fetchall() == [("uncertain",)]


def test_wire_schema_interning_is_lossless_and_meter_matches_completion(transport):
    from review_bot.adapters.beam_cloud import _compact_schema
    calls, _, _ = transport
    enum = ["Q" + str(index) for index in range(200)]
    citation = {"type": "object", "properties": {"reference_id": {"type": "string", "enum": enum}},
        "required": ["reference_id"], "additionalProperties": False}
    schema = {"type": "object", "properties": {"first": citation, "second": deepcopy(citation)},
        "required": ["first", "second"], "additionalProperties": False}
    compact = _compact_schema(schema)
    def expand(node):
        if isinstance(node, dict):
            if set(node) == {"$ref"}:
                return expand(compact["$defs"][node["$ref"][8:]])
            return {key: expand(value) for key, value in node.items() if key != "$defs"}
        return [expand(value) for value in node] if isinstance(node, list) else node
    assert expand(compact) == schema and len(json.dumps(compact)) < len(json.dumps(schema))
    assert _compact_schema({"$ref": "#/$defs/original", "$defs": {"original": schema}}) == {"$ref": "#/$defs/original", "$defs": {"original": schema}}
    task = request(response_schema=schema)
    client = BeamCloudModelClient()
    client.count_input_tokens(task)
    # Only transport is interned. The original inline extra-field guard remains.
    with pytest.raises(PermanentModelError, match="Unexpected"):
        client.complete(task)
    chats = [payload for _, path, payload in calls if path == "/api/chat"]
    assert chats[0]["messages"] == chats[1]["messages"] and chats[0]["format"] == chats[1]["format"] == compact
    assert task.response_schema == schema and task.system_prompt == "Review the supplied source."
