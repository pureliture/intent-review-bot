"""MacBook HTTP composition uses the existing B pipeline; traffic is synthetic."""
from __future__ import annotations

import json
from dataclasses import asdict

import httpx
import pytest

from review_bot.application.declared_intent_review import STAGES
from review_bot.runtime import cli
from tests.fixtures.intent_review import make_task, stage_bodies


@pytest.mark.parametrize("failure", [None, "wrong_model", "tokenizer_after_goals", "capacity_after_goals"])
def test_macbook_cli_preserves_validated_results_and_never_falls_back(tmp_path, monkeypatch, failure):
    requests = []
    client_type = httpx.Client
    current_stage = [None]

    def handler(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1" and request.url.port == 18135
        generated = any(item.url.path == "/v1/chat/completions" for item in requests)
        if request.url.path == "/v1/models":
            model = "bonsai2-27b" if failure == "wrong_model" else "lfm2.5-2.6b-q8_0"
            return httpx.Response(200, json={"object": "list", "data": [{"id": model}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={
                "default_generation_settings": {"n_ctx": 32768},
                "total_slots": 2 if failure == "capacity_after_goals" and generated else 1,
                "chat_template": "macbook-synthetic-template",
            })
        payload = json.loads(request.content)
        if request.url.path == "/apply-template":
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            current_stage[0] = json.loads(payload["messages"][1]["content"])["stage"]
            return httpx.Response(200, json={"prompt": json.dumps(payload["messages"])})
        if request.url.path == "/tokenize":
            if failure == "tokenizer_after_goals" and current_stage[0] == "observe_changes":
                raise httpx.ReadTimeout("private transport details", request=request)
            return httpx.Response(200, json={"tokens": list(range(100))})
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == "lfm2.5-2.6b-q8_0"
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert payload["stream"] is False
        stage = json.loads(payload["messages"][1]["content"])["stage"]
        return httpx.Response(200, json={
            "model": payload["model"], "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(stage_bodies()[STAGES.index(stage)]),
            }}], "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    def client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return client_type(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("MACBOOK_LFM_BASE_URL", "http://127.0.0.1:18135/v1")
    monkeypatch.setenv("ME707_BONSAI_BASE_URL", "http://ubuntu-ms-7e07:11435/v1")
    state = tmp_path / "mac-admission"
    monkeypatch.setenv("REVIEW_BOT_MACBOOK_MODEL_STATE_DIR", str(state))
    monkeypatch.setattr(cli, "PmdRunner", lambda: pytest.fail("B must not initialize PMD"))
    monkeypatch.setattr(cli, "GitRepositoryReader", lambda *_: pytest.fail("Fixed input must not read Git"))
    source = tmp_path / "input.json"
    source.write_text(json.dumps(asdict(make_task())))
    source.chmod(0o600)
    output = tmp_path / "output"
    assert cli.main(["intent", "--input", str(source), "--output", str(output),
        "--model-profile", "macbook-llamacpp"]) == 0
    result = json.loads((output / "B/result.json").read_text())
    calls = [request for request in requests if request.url.path == "/v1/chat/completions"]
    expected_calls = 3 if failure is None else (0 if failure == "wrong_model" else 1)
    assert len(calls) == result["usage"]["calls"] == expected_calls
    assert result["usage"]["known_input_tokens"] == 100 * expected_calls
    assert result["usage"]["known_output_tokens"] == 50 * expected_calls
    assert result["usage"]["review_input_tokens"] == 100 * expected_calls
    assert result["usage"]["review_output_tokens"] == 50 * expected_calls
    if failure == "tokenizer_after_goals":
        assert result["usage"]["input_tokens"] is None and result["usage"]["output_tokens"] is None
        assert result["usage"]["usage_complete"] is False and result["usage"]["unknown_usage_attempts"] == 1
        assert result["stages"][0]["attempts"][0]["usage"] == {"input_tokens": 100, "output_tokens": 50}
        assert result["stages"][1]["token_measurement_delivery"] == "unknown"
    else:
        assert result["usage"]["input_tokens"] == 100 * expected_calls
        assert result["usage"]["output_tokens"] == 50 * expected_calls
        assert result["usage"]["usage_complete"] is True and result["usage"]["unknown_usage_attempts"] == 0
    if failure is None:
        assert result["status"] == "completed" and result["reason"] == "validated"
        assert result["usage"]["model_calls"] == 3
        assert all(stage["token_measurement_model_calls"] == 0 for stage in result["stages"])
    else:
        assert result["status"] == "not_run" and result["reason"] == "token_count_unavailable"
        assert bool(result["goals"]) == bool(expected_calls)
        assert result["observations"] is None and result["stages"][2]["status"] == "not_attempted"
        assert "private transport details" not in json.dumps(result)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["publication"] == {"status": "not_attempted", "reason": "artifact_only"}
    assert output.stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in (output / "B").iterdir())
    assert (state / "requests.sqlite").stat().st_mode & 0o077 == 0