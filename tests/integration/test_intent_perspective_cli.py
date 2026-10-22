"""The single-run CLI, real adapters, private outputs, and admission database."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

import httpx
import pytest

from review_bot.runtime import cli, model_clients
from tests.fixtures.intent_perspectives import PerspectiveModelClient, make_task, review_body


def fixed_input(tmp_path, *, repair=False):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(asdict(make_task(repair=repair)), ensure_ascii=False), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_cli_writes_one_private_integrated_review_and_full_histories(tmp_path, monkeypatch):
    main, selector = PerspectiveModelClient(), PerspectiveModelClient(selector=True)
    monkeypatch.setattr(cli, "build_review_client", lambda profile: main)
    monkeypatch.setattr(model_clients, "build_intent_selection_client", lambda profile: selector)
    monkeypatch.setattr(cli, "PmdRunner", lambda: pytest.fail("Single-run intent review does not compose analysis"))
    monkeypatch.setattr(cli, "GitRepositoryReader", lambda *a: pytest.fail("Fixed input must not be recollected"))
    out = tmp_path / "result"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(out),
        "--pipeline", "perspectives-v1", "--model-profile", "macbook-llamacpp"]) == 0
    values = json.loads((out / "B/result.json").read_text())
    assert (values["schema_version"], values["status"]) == (3, "completed")
    assert len(main.requests) == 2 and len(selector.requests) == 8
    expected = {"input.json", "result.json", "scope.json", "usage.json", "validation.json", "comment.md",
        "catalog.json", "selection.json", "perspectives.json", "judgements.json", "context.json", "passes.json"}
    assert {path.name for path in (out / "B").iterdir()} == expected
    assert len(json.loads((out / "B/perspectives.json").read_text())["history"]) == 16
    assert len(json.loads((out / "B/judgements.json").read_text())["history"]) == 16
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["publication"] == {"status": "not_attempted", "reason": "artifact_only"}
    assert manifest["modes"]["intent"]["result_schema_version"] == 3
    assert (out.stat().st_mode & 0o777) == (out / "B").stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (out / "B").iterdir())
    assert "파악한 개발 의도" in (out / "B/comment.md").read_text()
    assert not (out / "A").exists()


def test_unconfigured_single_run_has_zero_calls_and_an_incomplete_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "Client", lambda **k: pytest.fail("No explicit model means no network"))
    out = tmp_path / "not-run"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(out), "--pipeline", "perspectives-v1"]) == 1
    result = json.loads((out / "B/result.json").read_text())
    assert result["reason"] == "model_profile_unset" and result["usage"]["calls"] == 0
    assert result["selection"] == [] and result["passes"] == []
    markdown = (out / "B/comment.md").read_text()
    assert "미실행" in markdown
    assert "검토했습니다" not in markdown and "검토 완료" not in markdown


def test_partial_review_keeps_supported_findings_and_returns_incomplete(tmp_path, monkeypatch):
    def missing(payload, body):
        if payload["stage"] == "draft_review":
            body["context_gaps"] = [{"perspective_ids": ["problem_success"], "description": "목적의 대상 사용자", "blocking": True}]
        return body
    monkeypatch.setattr(cli, "build_review_client", lambda p: PerspectiveModelClient(transform=missing))
    monkeypatch.setattr(model_clients, "build_intent_selection_client", lambda p: PerspectiveModelClient(selector=True))
    out = tmp_path / "partial"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(out), "--pipeline", "perspectives-v1"]) == 1
    result = json.loads((out / "B/result.json").read_text())
    assert result["status"] == "partial_review" and result["scope"]["review_complete"]
    assert result["scope"]["review_sufficient"] is False and result["findings"]
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
    assert "목적의 대상 사용자" in (out / "B/comment.md").read_text()


@pytest.mark.parametrize("profile", ["macbook-llamacpp", "me707-llamacpp"])
def test_actual_adapters_use_closed_choices_native_measurement_and_shared_admission(tmp_path, monkeypatch, profile):
    actual_client = httpx.Client
    requests = []
    local = profile == "macbook-llamacpp"
    main_model, choice_model = ("lfm2.5-2.6b-q8_0", "tev1-0.8b-q8_0") if local else ("bonsai2-27b", "tev1-4b-q8_0")
    def handler(request):
        requests.append(request)
        selector = request.url.port == 18137
        model = choice_model if selector else main_model
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": model}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={"total_slots": 1 if local else 2, "chat_template": "synthetic-intent-template",
                "default_generation_settings": {"n_ctx": 32768 if local else 131072 if selector else 131072}})
        payload = json.loads(request.content)
        if request.url.path in ("/apply-template", "/tokenize"):
            raise AssertionError(f"measurement endpoint must never be called: {request.url.path}")
        assert request.url.path == "/v1/chat/completions"
        task = json.loads(payload["messages"][1]["content"])
        assert set(payload["response_format"]) == {"type", "json_schema"}
        assert payload["response_format"]["json_schema"]["strict"] is True
        if selector:
            assert payload["max_tokens"] == 16 and payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert payload["response_format"]["json_schema"]["schema"] == {
                "type": "object", "properties": {"choice": {"type": "string", "enum": ["A", "B", "C"]}},
                "required": ["choice"], "additionalProperties": False,
            }
            body, tokens = {"choice": "C"}, 5
        else:
            assert len(task["evidence"]) == 6 and task["reference_text"]
            body, tokens = review_body(task), 100
        return httpx.Response(200, json={"model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(body, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": tokens, "total_tokens": 100 + tokens}})
    def client(**options):
        assert options["trust_env"] is False and options["follow_redirects"] is False
        return actual_client(transport=httpx.MockTransport(handler), **options)
    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("MACBOOK_LFM_BASE_URL" if local else "ME707_BONSAI_BASE_URL", "http://127.0.0.1:18135/v1")
    monkeypatch.setenv("MACBOOK_TEV1_BASE_URL" if local else "ME707_TEV1_BASE_URL", "http://127.0.0.1:18137/v1")
    state = tmp_path / "admission"
    monkeypatch.setenv("REVIEW_BOT_MACBOOK_MODEL_STATE_DIR" if local else "REVIEW_BOT_MODEL_STATE_DIR", str(state))
    out = tmp_path / "native-result"
    # me707 (128K budget) completes; macbook (32K budget) stops fail-closed
    # at final_review when the byte-upper-bound exceeds its input budget.
    # No measurement call exists on either profile.
    expected_exit = 0 if not local else 1
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(out),
        "--pipeline", "perspectives-v1", "--model-profile", profile]) == expected_exit
    generations = [request for request in requests if request.url.path == "/v1/chat/completions"]
    assert not any(request.url.path in ("/apply-template", "/tokenize") for request in requests)
    result = json.loads((out / "B/result.json").read_text())
    assert result["model"]["is_mock"] is False
    assert all(row["token_measurement"] == "not_attempted" for row in result["stages"])
    assert all(row["token_measurement_model_calls"] == 0 for row in result["stages"])
    assert all(row["decision"] == "undetermined" for row in result["selection"])
    if local:
        completed = [row for row in result["stages"] if row["attempts"]]
        assert [request.url.port for request in generations] == [18137] * 8 + [18135] * 1
        assert result["status"] == "not_run" and result["reason"] == "input_limit_exceeded"
        assert result["usage"]["calls"] == len(completed) == 9
        # The refused admission leaves no attempt, so known usage stays exact.
        assert result["usage"]["input_tokens"] == 900
        assert result["usage"]["usage_complete"] is True
        assert result["stages"][-1]["stage"] == "final_review"
        assert result["stages"][-1]["reason"] == "input_limit_exceeded"
    else:
        assert [request.url.port for request in generations] == [18137] * 8 + [18135] * 2
        assert result["usage"]["calls"] == result["usage"]["model_calls"] == 10
        assert (result["usage"]["input_tokens"], result["usage"]["output_tokens"]) == (1000, 240)
        assert result["usage"]["usage_complete"] is True
        assert all(row["decision"] == "apply" for row in result["perspectives"])
    with sqlite3.connect(state / "requests.sqlite") as db:
        assert db.execute("SELECT COUNT(*) FROM model_admission").fetchone()[0] == 0


def test_output_language_and_timeout_are_explicit_options(tmp_path, monkeypatch):
    seen = []
    def build_main(profile, settings):
        seen.append((profile, settings.call_timeout_seconds))
        return PerspectiveModelClient(repair=True)
    def build_choice(profile, settings):
        seen.append((profile, settings.call_timeout_seconds))
        return PerspectiveModelClient(selector=True)
    monkeypatch.setattr(cli, "build_review_client", build_main)
    monkeypatch.setattr(model_clients, "build_intent_selection_client", build_choice)
    out = tmp_path / "english"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path, repair=True)), "--output", str(out),
        "--pipeline", "perspectives-v1", "--model-profile", "me707-llamacpp", "--language", "en",
        "--call-timeout-seconds", "120"]) == 0
    assert seen == [("me707-llamacpp", 120)] * 2
    assert "Understood development intent" in (out / "B/comment.md").read_text()


def test_invalid_input_is_rejected_before_client_or_output_creation(tmp_path, monkeypatch):
    source = fixed_input(tmp_path)
    data = json.loads(source.read_text());data["head"]["files"][0]["content"] += "tampered"
    source.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(cli, "build_review_client", lambda p: pytest.fail("No clients for tampered input"))
    out = tmp_path / "invalid"
    assert cli.main(["intent", "--input", str(source), "--output", str(out), "--pipeline", "perspectives-v1"]) == 1
    assert not out.exists()
