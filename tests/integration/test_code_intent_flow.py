"""Code-first B at public CLI and signed Git/SQLite Worker boundaries."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest

from review_bot.application.review_a import MockModelClient
from review_bot.config import ModelConfig
from review_bot.domain.models_v1 import ModelResponse
from review_bot.runtime import cli
from tests.conftest import SyntheticGitRepo
from tests.fixtures.review_b import make_task
from tests.fixtures.review_b_code import (
    BASE_SOURCE,
    CALLER,
    HEAD_SOURCE,
    TEST_SOURCE,
    CodeFirstModelClient,
    make_task as make_code_task,
    stage_body,
)
from tests.integration.test_intent_flow import (
    enable_a_fixture,
    intent_flow,
    output,
    send_event,
    use_modes,
)


@pytest.fixture(scope="module")
def synthetic_repo(tmp_path_factory):
    repo = SyntheticGitRepo(tmp_path_factory.mktemp("code-first-git"))
    repo.write_file("src/value.py", BASE_SOURCE)
    repo.write_file("src/caller.py", CALLER)
    repo.write_file("tests/test_label.py", TEST_SOURCE)
    base = repo.commit("Synthetic string return contract")
    repo.write_file("src/value.py", HEAD_SOURCE)
    head = repo.commit("Synthetic structured return without consumer update")
    return SimpleNamespace(repo_dir=repo.repo_dir, base_sha=base, head_sha=head)


def select_code_first(flow):
    flow.config.intent.pipeline = "code-first-v2"
    flow.config.intent.context_paths = ["src/caller.py", "tests/test_label.py"]


def code_models(flow):
    select_code_first(flow)
    review, planner = CodeFirstModelClient(), CodeFirstModelClient()
    flow.worker.model_client, flow.worker.planning_client = review, planner
    return review, planner


def test_code_first_cli_unconfigured_preserves_v2_not_run_files_and_reports_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_review_client", lambda profile: None)
    monkeypatch.setattr(cli, "build_planning_client", lambda profile: None)
    source = tmp_path / "fixed-input.json"
    source.write_text(json.dumps(asdict(make_task())), encoding="utf-8")
    source.chmod(0o600)
    destination = tmp_path / "private-output"
    assert cli.main(["intent", "--input", str(source), "--output", str(destination),
        "--pipeline", "code-first-v2"]) == 1
    result = json.loads((destination / "B/result.json").read_text())
    assert result["schema_version"] == 2 and result["status"] == "not_run"
    assert result["usage"]["calls"] == 0
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["modes"]["intent"]["result_schema_version"] == 2
    assert manifest["publication"] == {"status": "not_attempted", "reason": "artifact_only"}
    assert not (destination / "B/goals.json").exists()
    for name in ("plan.json", "observations.json", "intents.json", "relations.json", "comment.md"):
        assert (destination / "B" / name).stat().st_mode & 0o777 == 0o600


def test_code_first_failure_preserves_completed_a_files_and_mode_receipts(intent_flow):
    flow = intent_flow
    use_modes(flow, ("architecture", "intent"))
    enable_a_fixture(flow)
    select_code_first(flow)
    flow.worker.model_client = MockModelClient([
        ModelResponse("{}", {"input_tokens": 100, "output_tokens": 50}),
    ], input_tokens=100)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow)
    assert status == "failed"
    assert json.loads((root / "A/result.json").read_text())["schema_version"] == 1
    result = json.loads((root / "B/result.json").read_text())
    assert result["schema_version"] == 2 and result["status"] == "failed"
    assert result["usage"]["calls"] == 1
    receipt = json.loads((root / "B/receipt.json").read_text())
    assert receipt["b_result"] == result
    assert (root / "A/receipt.json").exists()
    assert not flow.comments
    assert flow.analyzer.run_analysis.call_count == 2


def test_code_first_unconfigured_result_cannot_publish_even_in_opted_in_fixture(intent_flow):
    flow = intent_flow
    select_code_first(flow)
    flow.worker.model_client = None
    flow.config.publish.mode = "comment"
    flow.config.publish.b_comments_enabled = True
    flow.issue_status["allow_comments"] = True
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is True
    root, status = output(flow)
    assert status == "succeeded"
    result = json.loads((root / "B/result.json").read_text())
    assert result["schema_version"] == 2 and result["status"] == "not_run"
    assert result["usage"]["calls"] == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["modes"]["intent"]["publication"]["reason"] == "model_profile_unset"
    assert not flow.comments and all(method == "GET" for method, _ in flow.calls)
    flow.analyzer.run_analysis.assert_not_called()


@pytest.mark.parametrize("body", [None, "Documentation-only; no runtime behavior changes."])
def test_code_first_cli_generates_validated_code_defect_without_trusting_body(tmp_path, monkeypatch, body):
    review, planner = CodeFirstModelClient(), CodeFirstModelClient()
    monkeypatch.setattr(cli, "build_review_client", lambda profile: review)
    monkeypatch.setattr(cli, "build_planning_client", lambda profile: planner)
    source = tmp_path / "fixed-code-input.json"
    source.write_text(json.dumps(asdict(make_code_task(body=body))), encoding="utf-8")
    source.chmod(0o600)
    destination = tmp_path / "private-code-output"
    assert cli.main(["intent", "--input", str(source), "--output", str(destination),
        "--pipeline", "code-first-v2", "--model-profile", "macbook-llamacpp"]) == 0
    result = json.loads((destination / "B/result.json").read_text())
    assert (result["schema_version"], result["status"], result["reason"]) == (2, "completed", "validated")
    assert result["findings"][0]["assessment"] == "code_defect"
    assert result["findings"][0]["finding_id"] == "F1"
    assert result["scope"]["review_complete"] is True
    assert result["scope"]["tests_executed_by_bot"] is False
    assert len(planner.requests) == 1 and len(review.requests) == 3
    assert result["usage"]["calls"] == 4
    if body:
        for request in [*review.requests, *review.count_requests, *planner.requests, *planner.count_requests]:
            if json.loads(request.user_content)["stage"] != "assess":
                assert body not in request.user_content
        assert body in review.requests[-1].user_content
    assert result["relations"][0]["reviewed_together"] is True
    assert json.loads((destination / "B/intents.json").read_text()) == result["reconstructed_intents"]
    assert json.loads((destination / "B/plan.json").read_text()) == result["plan"]
    assert not (destination / "B/goals.json").exists()


def test_signed_pr_creates_code_first_result_with_exact_git_citations_and_private_receipt(intent_flow):
    flow = intent_flow
    review, planner = code_models(flow)
    flow.pr["body"] = None
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    root, status = output(flow)
    result = json.loads((root / "B/result.json").read_text())
    receipt = json.loads((root / "B/receipt.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert status == "succeeded" and result["status"] == "completed"
    assert result["findings"][0]["assessment"] == "code_defect"
    citations = result["findings"][0]["citations"]
    assert {row["path"] for row in citations} == {"src/value.py", "src/caller.py"}
    assert all(row["sha"] == flow.pr["head"]["sha"] for row in citations)
    assert any('return {"value": value}' in row["quote"] for row in citations)
    assert any("return render(value).strip()" in row["quote"] for row in citations)
    assert receipt["b_result"] == result and receipt["a_result"] is None
    assert manifest["modes"]["intent"]["result_schema_version"] == 2
    assert manifest["modes"]["intent"]["publication"]["reason"] == "artifact_only"
    assert len(planner.requests) == 1 and len(review.requests) == 3
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in (root / "B").iterdir())
    assert not flow.comments and not (root / "A").exists()
    flow.analyzer.run_analysis.assert_not_called()


def test_code_first_fixture_publishes_updates_same_id_then_reuses_its_own_input_only(intent_flow):
    flow = intent_flow
    review, planner = code_models(flow)
    flow.config.publish.mode = "comment"
    flow.config.publish.b_comments_enabled = True
    flow.issue_status["allow_comments"] = True
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    assert len(flow.comments) == 1 and flow.comments[0]["id"] == 701
    assert len(review.requests) == 3 and len(planner.requests) == 1
    flow.pr["body"] = "Changed optional context, not a requirement."
    assert send_event(flow, "code-first-context", "edited").status_code == 202 and flow.worker.run_once() is True
    assert len(flow.comments) == 1 and flow.comments[0]["id"] == 701
    assert len(review.requests) == 6 and len(planner.requests) == 2
    assert send_event(flow, "code-first-reuse", "edited").status_code == 202 and flow.worker.run_once() is True
    root, status = output(flow)
    mode = json.loads((root / "manifest.json").read_text())["modes"]["intent"]
    assert status == "skipped" and mode["reused"] is True
    assert len(review.requests) == 6 and len(planner.requests) == 2
    assert mode["usage"]["calls"] == 0 and not (root / "B/result.json").exists()
    assert [(method, path) for method, path in flow.calls if method != "GET"] == [
        ("POST", "/repos/org/repo/issues/7/comments"), ("PATCH", "/repos/org/repo/issues/comments/701"),
    ]


def test_code_first_cli_uses_both_real_adapters_native_metering_and_shared_sqlite_with_mock_http(tmp_path, monkeypatch):
    requests = []
    actual_client = httpx.Client

    def handler(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1" and request.url.port in (18135, 18137)
        model = "tev1-0.8b-q8_0" if request.url.port == 18137 else "lfm2.5-2.6b-q8_0"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": model}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 32768},
                "total_slots": 1, "chat_template": "synthetic-code-first-template"})
        payload = json.loads(request.content)
        if request.url.path == "/apply-template":
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            return httpx.Response(200, json={"prompt": json.dumps(payload["messages"])})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": list(range(100))})
        assert request.url.path == "/v1/chat/completions"
        assert payload["model"] == model and payload["stream"] is False
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        body = stage_body(SimpleNamespace(user_content=payload["messages"][1]["content"]))
        return httpx.Response(200, json={"model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(body)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}})

    def client(**options):
        assert options["trust_env"] is False and options["follow_redirects"] is False
        return actual_client(transport=httpx.MockTransport(handler), **options)

    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setenv("MACBOOK_LFM_BASE_URL", "http://127.0.0.1:18135/v1")
    monkeypatch.setenv("MACBOOK_TEV1_BASE_URL", "http://127.0.0.1:18137/v1")
    state = tmp_path / "macbook-admission"
    monkeypatch.setenv("REVIEW_BOT_MACBOOK_MODEL_STATE_DIR", str(state))
    source = tmp_path / "fixed-code-input.json"
    source.write_text(json.dumps(asdict(make_code_task())), encoding="utf-8")
    source.chmod(0o600)
    destination = tmp_path / "native-code-output"
    assert cli.main(["intent", "--input", str(source), "--output", str(destination),
        "--pipeline", "code-first-v2", "--model-profile", "macbook-llamacpp"]) == 0
    result = json.loads((destination / "B/result.json").read_text())
    calls = [request for request in requests if request.url.path == "/v1/chat/completions"]
    assert [request.url.port for request in calls] == [18137, 18135, 18135, 18135]
    assert result["status"] == "completed" and result["usage"]["calls"] == 4
    assert result["usage"]["input_tokens"] == 400 and result["usage"]["output_tokens"] == 200
    assert result["usage"]["model_calls"] == 4 and result["usage"]["usage_complete"] is True
    assert all(record["token_measurement_model_calls"] == 0 for record in result["stages"])
    assert result["findings"][0]["assessment"] == "code_defect"
    with sqlite3.connect(state / "requests.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_admission").fetchone()[0] == 0
    assert (state / "requests.sqlite").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("timeout", [120, 0])
def test_cli_applies_existing_timeout_setting_to_both_roles_without_changing_task_budget(tmp_path, monkeypatch, timeout):
    from review_bot.application import review_b_code
    from review_bot.domain.models_b_code import CodeFirstReviewLimits

    configured = []
    task_budgets = []
    actual_review = review_b_code.run_code_first_review

    def bounded_review(*args, **kwargs):
        task_budgets.append((kwargs.get("limits") or CodeFirstReviewLimits()).task_timeout_seconds)
        return actual_review(*args, **kwargs)

    def factory(profile, settings):
        assert profile == "macbook-llamacpp" and isinstance(settings, ModelConfig)
        configured.append(settings.call_timeout_seconds)
        return CodeFirstModelClient()

    monkeypatch.setattr(cli, "build_review_client", factory)
    monkeypatch.setattr(cli, "build_planning_client", factory)
    monkeypatch.setattr(review_b_code, "run_code_first_review", bounded_review)
    source = tmp_path / "fixed-code-input.json"
    source.write_text(json.dumps(asdict(make_code_task())), encoding="utf-8")
    source.chmod(0o600)
    destination = tmp_path / "configured-output"
    code = cli.main(["intent", "--input", str(source), "--output", str(destination),
        "--pipeline", "code-first-v2", "--model-profile", "macbook-llamacpp",
        "--call-timeout-seconds", str(timeout)])
    if timeout == 0:
        assert code == 1 and configured == [] and task_budgets == []
        assert not (destination / "B/result.json").exists()
    else:
        assert code == 0 and configured == [120, 120] and task_budgets == [900]
        result = json.loads((destination / "B/result.json").read_text())
        assert result["status"] == "completed"