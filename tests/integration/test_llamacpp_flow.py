"""Explicit llama.cpp profile through the real B pipeline and private CLI artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import httpx
import pytest

from review_bot.application.architecture_review import MockModelClient
from review_bot.application.declared_intent_review import STAGES, run_review_b_pipeline
from review_bot.runtime import cli
from tests.fixtures.intent_review import make_task, responses, stage_bodies


@pytest.mark.parametrize("measurement_usage,known", [
    ({"input_tokens": 0, "output_tokens": 0}, True),
    ({"input_tokens": 1, "output_tokens": 0}, False),
])
@pytest.mark.parametrize("delivery", ["sent", "not_sent"])
def test_native_tokenizer_without_generation_has_exact_usage(measurement_usage, known, delivery):
    client = MockModelClient(responses(), input_tokens=100)
    client.profile = replace(client.profile, is_mock=False, provider="llamacpp")
    client.token_count_receipt = {"delivery": delivery, "model_calls": 0, "usage": measurement_usage}
    result = run_review_b_pipeline(make_task(), client)
    assert result.status == "completed"
    assert result.usage["calls"] == 3
    assert result.usage["model_calls"] == (3 if known else None)
    assert all(stage.token_measurement_model_calls == (0 if known else None) for stage in result.stages)
    assert all(stage.token_measurement_delivery == (delivery if known else "unknown") for stage in result.stages)


@pytest.mark.parametrize("failure_stage", [
    None, "observe_changes", "server_capacity_at_start", "server_capacity_after_goals",
])
def test_me707_cli_saves_validated_results_without_posting(tmp_path, monkeypatch, failure_stage):
    requests = []
    client_type = httpx.Client
    current_stage = [None]

    def handler(request):
        requests.append(request)
        assert request.url.host == "ubuntu-ms-7e07"
        assert request.url.port == 11435
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "bonsai2-27b", "object": "model"}]})
        if request.url.path == "/props":
            generated = any(item.url.path == "/v1/chat/completions" for item in requests)
            unsafe_capacity = failure_stage == "server_capacity_at_start" or (
                failure_stage == "server_capacity_after_goals" and generated
            )
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": 262144}, "total_slots": 3 if unsafe_capacity else 2,
                "chat_template": "fixture-chat-template",
            })
        payload = json.loads(request.content)
        if request.url.path == "/apply-template":
            current_stage[0] = json.loads(payload["messages"][1]["content"])["stage"]
            return httpx.Response(200, json={"prompt": json.dumps(payload["messages"])})
        if request.url.path == "/tokenize":
            if current_stage[0] == failure_stage:
                raise httpx.ReadTimeout("private transport details", request=request)
            return httpx.Response(200, json={"tokens": list(range(100))})
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "bonsai2-27b"
        assert payload["stream"] is False
        stage = json.loads(payload["messages"][1]["content"])["stage"]
        return httpx.Response(200, json={
            "model": "bonsai2-27b", "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(stage_bodies()[STAGES.index(stage)]),
            }}], "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return client_type(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("ME707_BONSAI_BASE_URL", "http://ubuntu-ms-7e07:11435/v1")
    monkeypatch.setattr(cli, "PmdRunner", lambda: pytest.fail("B must not initialize PMD"))
    monkeypatch.setattr(cli, "GitRepositoryReader", lambda *_: pytest.fail("Fixed input must not read a repository"))
    source = tmp_path / "input.json"
    source.write_text(json.dumps(asdict(make_task())))
    source.chmod(0o600)
    output = tmp_path / "output"
    assert cli.main(["intent", "--input", str(source), "--output", str(output),
        "--model-profile", "me707-llamacpp"]) == 0
    result = json.loads((output / "B/result.json").read_text())
    calls = [request for request in requests if request.url.path == "/v1/chat/completions"]
    if failure_stage in ("server_capacity_at_start", "server_capacity_after_goals"):
        expected_calls = int(failure_stage == "server_capacity_after_goals")
        assert len(calls) == result["usage"]["calls"] == expected_calls
        assert result["status"] == "not_run"
        assert result["reason"] == "token_count_unavailable"
        assert bool(result["goals"]) == bool(expected_calls)
        assert result["observations"] is None
        assert result["stages"][2]["status"] == "not_attempted"
        assert requests[-1].url.path == "/props"
        assert result["usage"]["input_tokens"] == 100 * expected_calls
        assert result["usage"]["output_tokens"] == 50 * expected_calls
    elif failure_stage:
        assert len(calls) == 1
        assert result["status"] == "not_run"
        assert result["reason"] == "token_count_unavailable"
        assert result["goals"] and result["observations"] is None
        assert result["stages"][2]["status"] == "not_attempted"
        assert "private transport details" not in json.dumps(result)
    else:
        assert len(calls) == 3
        assert result["status"] == "completed"
        assert result["reason"] == "validated"
        assert result["usage"]["model_calls"] == 3
        assert result["usage"]["input_tokens"] == 300
        assert result["usage"]["output_tokens"] == 150
        assert all(stage["token_measurement_model_calls"] == 0 for stage in result["stages"])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["publication"] == {"status": "not_attempted", "reason": "artifact_only"}
    assert output.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in (output / "B").iterdir())
