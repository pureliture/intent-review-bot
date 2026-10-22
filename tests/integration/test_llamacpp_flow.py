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


@pytest.mark.parametrize("measurement_usage", [
    {"input_tokens": 0, "output_tokens": 0},
    {"input_tokens": 1, "output_tokens": 0},
])
@pytest.mark.parametrize("delivery", ["sent", "not_sent"])
def test_legacy_measurement_receipt_is_ignored_and_never_requested(measurement_usage, delivery):
    """Retired metering contract: legacy receipts never drive admission.

    No measurement call exists; only complete() calls are made and only
    response usage is recorded. Legacy receipt attributes are ignored.
    """
    client = MockModelClient(responses(), input_tokens=100)
    client.profile = replace(client.profile, is_mock=False, provider="llamacpp")
    client.token_count_receipt = {"delivery": delivery, "model_calls": 0, "usage": measurement_usage}
    assert not hasattr(client, "count_input_tokens")
    result = run_review_b_pipeline(make_task(), client)
    assert result.status == "completed"
    assert result.usage["calls"] == 3
    assert result.usage["model_calls"] == 3
    assert all(stage.token_measurement_model_calls == 0 for stage in result.stages)
    assert all(stage.token_measurement_delivery == "not_attempted" for stage in result.stages)
    assert all(stage.token_measurement == "not_attempted" for stage in result.stages)


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
                "default_generation_settings": {"n_ctx": 131072}, "total_slots": 3 if unsafe_capacity else 2,
                "chat_template": "fixture-chat-template",
            })
        payload = json.loads(request.content)
        if request.url.path in ("/apply-template", "/tokenize"):
            raise AssertionError(f"measurement endpoint must never be called: {request.url.path}")
        if request.url.path == "/v1/chat/completions" and failure_stage == "observe_changes":
            stage = json.loads(payload["messages"][1]["content"])["stage"]
            if stage == "observe_changes":
                raise httpx.ReadTimeout("private transport details", request=request)
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
    expected_exit = 0 if failure_stage is None else 1
    assert cli.main(["intent", "--input", str(source), "--output", str(output),
        "--model-profile", "me707-llamacpp"]) == expected_exit
    result = json.loads((output / "B/result.json").read_text())
    calls = [request for request in requests if request.url.path == "/v1/chat/completions"]
    assert not any(request.url.path in ("/apply-template", "/tokenize") for request in requests)
    if failure_stage in ("server_capacity_at_start", "server_capacity_after_goals"):
        expected_calls = int(failure_stage == "server_capacity_after_goals")
        assert len(calls) == expected_calls
        # usage calls count attempts: completed generations plus the one
        # failed admission attempt that observed the capacity flip.
        assert result["usage"]["calls"] == expected_calls + 1
        assert result["status"] == "failed"
        assert result["reason"] == "model_error"
        assert bool(result["goals"]) == bool(expected_calls)
        assert result["observations"] is None
        assert result["stages"][2]["status"] == "not_attempted"
        assert requests[-1].url.path == "/props"
        # The failed admission attempt leaves usage unknown: totals stay None.
        assert result["usage"]["input_tokens"] is None
        assert result["usage"]["output_tokens"] is None
        assert result["usage"]["known_input_tokens"] == 100 * expected_calls
        assert result["usage"]["known_output_tokens"] == 50 * expected_calls
    elif failure_stage:
        assert len(calls) == 2
        assert result["status"] == "failed"
        assert result["reason"] == "model_timeout"
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
