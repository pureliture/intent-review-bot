"""Real Git/SQLite signed requests, isolated A/B comments and successful-mode reuse."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from review_bot.adapters.git import GitRepositoryReader
from review_bot.adapters.github.auth import GitHubAppAuthError
from review_bot.application.publication import publish_review_comment
from review_bot.application.review_a import MockModelClient
from review_bot.application.review_b import SYSTEM_PROMPTS
from review_bot.domain.models_v1 import ModelResponse, PmdViolation
from review_bot.runtime.artifacts import RunArtifacts
from tests.conftest import SyntheticGitRepo
from tests.integration.test_intent_flow import (
    enable_a_fixture,
    intent_flow,
    output,
    send_event,
    synthetic_repo,
    use_modes,
)
from tests.unit.test_delta_adversarial import make_test_run

A_MARKER = "<!-- review-bot:architecture:v1 -->"
B_MARKER = "<!-- review-bot:intent:v1 -->"


def opt_in(flow, modes=("intent",)):
    use_modes(flow, modes)
    flow.config.publish.mode = "comment"
    flow.config.publish.b_comments_enabled = True
    flow.issue_status["allow_comments"] = True
    if "architecture" in modes:
        enable_a_fixture(flow)


def writes(flow):
    return [(method, path) for method, path in flow.calls if method != "GET"]


def receipts(flow):
    root, status = output(flow)
    return root, status, json.loads((root / "manifest.json").read_text())["modes"]


def test_b_opt_in_creates_updates_same_id_and_then_reuses_without_new_model_call(intent_flow):
    flow = intent_flow
    opt_in(flow)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    first, status, modes = receipts(flow)
    assert status == "succeeded" and len(flow.model.requests) == 3
    assert len(flow.comments) == 1 and flow.comments[0]["id"] == 701
    assert flow.comments[0]["body"].count(B_MARKER) == 1
    assert "private evaluation draft" not in flow.comments[0]["body"]
    assert modes["intent"]["publication"]["status"] == "succeeded"
    assert modes["intent"]["input_fingerprint"] == modes["intent"]["publication"]["input_fingerprint"]
    flow.issue["body"] = "Changed explicit requirement"
    assert send_event(flow, "issue-changed", "edited").status_code == 202 and flow.worker.run_once() is True
    second, status, changed = receipts(flow)
    assert status == "succeeded" and len(flow.model.requests) == 6
    assert len(flow.comments) == 1 and flow.comments[0]["id"] == 701
    assert writes(flow) == [("POST", "/repos/org/repo/issues/7/comments"), ("PATCH", "/repos/org/repo/issues/comments/701")]
    assert modes["intent"]["input_fingerprint"] != changed["intent"]["input_fingerprint"]
    assert send_event(flow, "same-input", "edited").status_code == 202 and flow.worker.run_once() is True
    third, status, reused = receipts(flow)
    assert status == "skipped" and len(flow.model.requests) == 6 and len(writes(flow)) == 2
    assert reused["intent"]["reused"] is True and reused["intent"]["usage"]["calls"] == 0
    assert reused["intent"]["model_delivery"] == {"status": "not_attempted"}
    assert not (third / "B/result.json").exists(), "A reuse receipt must not fabricate a new review result"
    assert (first / "B/result.json").exists() and (second / "B/result.json").exists()
    flow.analyzer.run_analysis.assert_not_called()


def test_a_and_b_have_two_comments_independent_update_and_reuse(intent_flow):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    assert [item["id"] for item in flow.comments] == [701, 702]
    assert flow.comments[0]["body"].count(A_MARKER) == flow.comments[1]["body"].count(B_MARKER) == 1
    assert flow.analyzer.run_analysis.call_count == 2 and len(flow.model.requests) == 3
    first_a = flow.comments[0]["body"]
    flow.issue["body"] = "Only B's issue material changed"
    assert send_event(flow, "b-issue-change", "edited").status_code == 202 and flow.worker.run_once() is True
    root, status, modes = receipts(flow)
    assert status == "succeeded" and modes["architecture"]["reused"] is True and modes["intent"]["reused"] is False
    assert flow.comments[0]["body"] == first_a and len(flow.comments) == 2
    assert flow.analyzer.run_analysis.call_count == 2 and len(flow.model.requests) == 6
    assert writes(flow)[-1] == ("PATCH", "/repos/org/repo/issues/comments/702")
    flow.pr["title"] += " with clarified scope"
    assert send_event(flow, "both-change", "edited").status_code == 202 and flow.worker.run_once() is True
    assert [item["id"] for item in flow.comments] == [701, 702]
    assert writes(flow)[-2:] == [("PATCH", "/repos/org/repo/issues/comments/701"), ("PATCH", "/repos/org/repo/issues/comments/702")]
    assert flow.analyzer.run_analysis.call_count == 4 and len(flow.model.requests) == 9
    assert send_event(flow, "both-reuse", "edited").status_code == 202 and flow.worker.run_once() is True
    assert receipts(flow)[1] == "skipped" and flow.analyzer.run_analysis.call_count == 4 and len(flow.model.requests) == 9


@pytest.mark.parametrize("failed_mode", ["architecture", "intent"])
def test_failed_mode_publication_does_not_hide_successful_other_mode_on_retry(intent_flow, failed_mode):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    flow.issue_status["fail_marker"] = A_MARKER if failed_mode == "architecture" else B_MARKER
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status, modes = receipts(flow)
    other = "intent" if failed_mode == "architecture" else "architecture"
    assert status == "failed" and modes[failed_mode]["execution_status"] == "failed"
    assert modes[other]["publication"]["status"] == "succeeded"
    assert modes[failed_mode]["publication"]["input_fingerprint"] is None
    assert "synthetic-write-secret" not in (root / "manifest.json").read_text()
    assert len(flow.comments) == 1
    successful_id = flow.comments[0]["id"]
    flow.issue_status.pop("fail_marker")
    assert send_event(flow, "partial-retry", "edited").status_code == 202 and flow.worker.run_once() is True
    root, status, modes = receipts(flow)
    assert status == "succeeded" and modes[other]["reused"] is True and modes[failed_mode]["reused"] is False
    assert len(flow.comments) == 2 and flow.comments[0]["id"] == successful_id
    assert len(flow.model.requests) == (3 if failed_mode == "architecture" else 6)
    assert flow.analyzer.run_analysis.call_count == (4 if failed_mode == "architecture" else 2)


def test_b_success_is_reused_from_a_failed_execution_run(intent_flow):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    flow.analyzer.run_analysis.side_effect = RuntimeError("synthetic-analysis-secret")
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    assert receipts(flow)[1] == "failed" and len(flow.model.requests) == 3
    enable_a_fixture(flow)
    assert send_event(flow, "analysis-retry", "edited").status_code == 202 and flow.worker.run_once() is True
    root, status, modes = receipts(flow)
    assert status == "succeeded" and modes["intent"]["reused"] is True
    assert len(flow.model.requests) == 3 and len(flow.comments) == 2
    assert modes["architecture"]["execution_status"] == "not_run"


@pytest.mark.parametrize("failure", ["format", "artifact"])
def test_a_output_failure_preserves_called_core_and_usage_without_suppressing_b(intent_flow, monkeypatch, failure):
    flow = intent_flow
    use_modes(flow, ("architecture", "intent"))
    repo_dir = flow.state / "a-source"
    repo_dir.mkdir()
    repo = SyntheticGitRepo(repo_dir)
    source_path = "src/main/java/demo/App.java"
    repo.write_file(source_path, "class App { void handle() {} }\n")
    repo.write_file("tests/test_app.py", "def test_handle():\n    assert handle() == 1\n")
    flow.pr["base"]["sha"] = repo.commit("Synthetic A base")
    head_code = "class App { void handle() { int x = 1; } }\n"
    repo.write_file(source_path, head_code)
    flow.pr["head"]["sha"] = repo.commit("Synthetic A head")
    reader = GitRepositoryReader(repo.repo_dir)
    flow.worker.repo_reader_factory = lambda *args: reader
    violation = PmdViolation(source_path, source_path, 1, 1, 1, 1, "CognitiveComplexity",
        "Design", 3, "App", "handle", "", "Synthetic analysis signal", code_fragment=head_code)
    flow.analyzer.run_analysis.side_effect = lambda sha, *args: make_test_run(sha, source_files=(source_path,),
        pmd_violations=(violation,) if sha == flow.pr["head"]["sha"] else ())
    complete_b = flow.model.complete
    a_requests = []

    def complete(request):
        if "review-a/v1" not in request.system_prompt:
            return complete_b(request)
        flow.model.requests.append(request)
        a_requests.append(request)
        properties = request.response_schema["properties"]
        return ModelResponse(json.dumps({"base_sha": properties["base_sha"]["const"],
            "head_sha": properties["head_sha"]["const"], "review_mode": properties["review_mode"]["const"],
            "summary": "Only the supplied synthetic analysis was assessed.", "findings": [], "questions": []}),
            {"calls": 1, "input_tokens": 100, "output_tokens": 50})

    def fail_output(*args, **kwargs):
        raise OSError("synthetic-output-secret")

    monkeypatch.setattr(flow.model, "complete", complete)
    if failure == "format":
        monkeypatch.setattr("review_bot.runtime.worker.format_review_markdown", fail_output)
    else:
        monkeypatch.setattr(RunArtifacts, "save_review", fail_output)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status, recorded = receipts(flow)
    assert status == "failed" and len(a_requests) == 1 and len(flow.model.requests) == 4
    a = json.loads((root / "A/receipt.json").read_text())
    assert a["a_result"] is not None, a["error_code"]
    assert a["execution_status"] == "failed" and a["a_result"]["status"] == "completed"
    assert a["a_result"]["schema_version"] == 1 and a["a_result"]["review_type"] == "A"
    assert a["usage"] == a["a_result"]["usage"] == {"calls": 1, "input_tokens": 100, "output_tokens": 50}
    assert recorded["architecture"]["usage"] == a["usage"]
    assert a["model_delivery"] == {"status": "sent"}
    assert a["publication"] is None and a["error_code"] == "architecture_pipeline_error:OSError"
    assert recorded["intent"]["execution_status"] == "completed" and recorded["intent"]["usage"]["calls"] == 3
    assert writes(flow) == []
    assert flow.store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0] == 1
    assert "synthetic-output-secret" not in (root / "A/receipt.json").read_text()
    assert "synthetic-output-secret" not in (root / "manifest.json").read_text()


def test_a_cache_does_not_hide_b_retry_after_invalid_model_response(intent_flow):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    bad = MockModelClient([ModelResponse("{}", {"input_tokens": 100, "output_tokens": 10})], input_tokens=100)
    flow.worker.model_client = bad
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    first, status, modes = receipts(flow)
    assert status == "failed" and modes["intent"]["usage"]["calls"] == 1
    assert modes["intent"]["publication"]["status"] == "not_attempted"
    flow.worker.model_client = flow.model
    assert send_event(flow, "b-response-retry", "edited").status_code == 202 and flow.worker.run_once() is True
    root, status, modes = receipts(flow)
    assert status == "succeeded" and modes["architecture"]["reused"] is True
    assert len(flow.model.requests) == 3 and flow.analyzer.run_analysis.call_count == 2
    assert len(flow.comments) == 2 and (first / "B/result.json").exists()


@pytest.mark.parametrize("case", ["disabled", "artifact_only", "draft", "model_unset", "input_missing", "invalid_response"])
def test_b_gate_never_posts_unverified_or_unapproved_opinions(intent_flow, case):
    flow = intent_flow
    opt_in(flow)
    if case == "disabled":
        flow.config.publish.b_comments_enabled = False
    elif case == "artifact_only":
        flow.config.publish.mode = "artifact_only"
    elif case == "draft":
        flow.pr["draft"] = True
    elif case == "model_unset":
        flow.worker.model_client = None
    elif case == "input_missing":
        flow.config.intent.context_paths.append("tests/missing.py")
    else:
        flow.worker.model_client = MockModelClient([ModelResponse("{}", {"input_tokens": 100, "output_tokens": 10})], input_tokens=100)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is (case != "invalid_response")
    root, status, modes = receipts(flow)
    assert writes(flow) == [] and flow.comments == []
    assert modes["intent"]["publication"]["status"] == "not_attempted"
    assert modes["intent"]["publication"]["reason"] == {"disabled": "b_comments_disabled", "artifact_only": "artifact_only",
        "draft": "draft_comments_disabled", "model_unset": "model_profile_unset", "input_missing": "incomplete_code_input",
        "invalid_response": "invalid_model_response"}[case]
    assert (root / "B/result.json").exists()


@pytest.mark.parametrize("mutation", ["head", "base", "closed", "draft", "pr_body", "issue_body", "issue_version", "issue_unavailable"])
def test_b_publication_checks_fixed_sources_before_and_after_write(intent_flow, monkeypatch, mutation):
    flow = intent_flow
    opt_in(flow)

    def mutate():
        if mutation in ("head", "base"):
            flow.pr[mutation]["sha"] = "d" * 40
        elif mutation == "closed":
            flow.pr["state"] = "closed"
        elif mutation == "draft":
            flow.pr["draft"] = True
        elif mutation == "pr_body":
            flow.pr["body"] = "Changed goals"
        elif mutation == "issue_body":
            flow.issue["body"] = "Changed source"
        elif mutation == "issue_version":
            flow.issue["updated_at"] = "2026-10-02T12:00:00Z"
        else:
            flow.issue_status["code"] = 404

    original = flow.model.complete

    def complete(request):
        response = original(request)
        if json.loads(request.user_content)["stage"] == "assess":
            mutate()
        return response

    monkeypatch.setattr(flow.model, "complete", complete)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status, modes = receipts(flow)
    assert status == "superseded" and modes["intent"]["publication"]["status"] == "stale"
    assert writes(flow) == [] and len(flow.model.requests) == 3
    assert modes["intent"]["publication"]["input_fingerprint"] is None


@pytest.mark.parametrize("mutation", ["issue", "issue_version", "issue_unavailable", "head", "draft", "body"])
def test_b_post_write_race_marks_existing_comment_stale_not_success(intent_flow, mutation):
    flow = intent_flow
    opt_in(flow)

    def mutate():
        flow.issue_status.pop("on_comment_write", None)
        if mutation == "issue":
            flow.issue["body"] = "Post-write changed requirement"
        elif mutation == "issue_version":
            flow.issue["updated_at"] = "2026-10-04T00:00:00Z"
        elif mutation == "issue_unavailable":
            flow.issue_status["code"] = 404
        elif mutation == "head":
            flow.pr["head"]["sha"] = "d" * 40
        elif mutation == "draft":
            flow.pr["draft"] = True
        else:
            flow.pr["body"] = "Post-write changed goals"

    flow.issue_status["on_comment_write"] = mutate
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status, modes = receipts(flow)
    assert status == "superseded" and len(flow.comments) == 1
    assert flow.comments[0]["body"].count(B_MARKER) == 1 and "Stale review" in flow.comments[0]["body"]
    assert modes["intent"]["publication"]["status"] == "stale"
    assert modes["intent"]["publication"]["input_fingerprint"] is None
    assert writes(flow) == [("POST", "/repos/org/repo/issues/7/comments"), ("PATCH", "/repos/org/repo/issues/comments/701")]


@pytest.mark.parametrize("case", ["human", "other_bot", "duplicate", "repeated_marker", "modified", "missing"])
def test_b_reuse_requeries_comment_ownership_and_body(intent_flow, case):
    flow = intent_flow
    opt_in(flow)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    initial = len(writes(flow))
    if case == "human":
        flow.comments[0]["user"] = {"login": "human", "type": "User"}
    elif case == "other_bot":
        flow.comments[0]["user"]["login"] = "other-app[bot]"
    elif case == "duplicate":
        flow.comments.append({**deepcopy(flow.comments[0]), "id": 999})
    elif case == "repeated_marker":
        flow.comments[0]["body"] += "\n" + B_MARKER
    elif case == "modified":
        flow.comments[0]["body"] += "\nChanged outside the service"
    else:
        flow.comments.clear()
    assert send_event(flow, "comment-recheck", "edited").status_code == 202
    assert flow.worker.run_once() is (case in ("modified", "missing"))
    root, status, modes = receipts(flow)
    assert modes["intent"].get("reused", False) is False
    if case in ("modified", "missing"):
        assert len(flow.model.requests) == 6 and len(writes(flow)) == initial + 1
    else:
        assert status == "failed" and len(writes(flow)) == initial and len(flow.model.requests) == 3


@pytest.mark.parametrize("method", ["find_marker_comment", "create_comment", "update_comment"])
@pytest.mark.parametrize("modes", [("intent", "architecture"), ("architecture", "intent")])
def test_b_publication_authorization_loss_preserves_usage_and_stops_remaining_modes(intent_flow, monkeypatch, method, modes):
    flow = intent_flow
    opt_in(flow, modes)
    if method == "update_comment":
        flow.comments.append({"id": 700, "body": B_MARKER + "\nPrior review",
            "user": {"login": "review-app[bot]", "type": "Bot"}})
    original = getattr(flow.github, method)

    def reject_intent(*args, **kwargs):
        text = kwargs.get("body", kwargs.get("marker", args[-1] if args else ""))
        if isinstance(text, str) and B_MARKER in text:
            raise GitHubAppAuthError("synthetic-authorization-secret")
        return original(*args, **kwargs)

    monkeypatch.setattr(flow.github, method, reject_intent)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status, recorded = receipts(flow)
    assert status == "failed" and len(flow.model.requests) == 3
    intent = recorded["intent"]
    assert (root / "B/receipt.json").exists()
    receipt = json.loads((root / "B/receipt.json").read_text())
    assert receipt["b_result"]["status"] == "completed" and intent["usage"]["calls"] == 3
    assert receipt["usage"] == intent["usage"]
    assert intent["execution_status"] == "failed" and intent["error_code"] == "publication_authorization_lost"
    assert intent["publication"]["status"] == ("unknown" if method == "update_comment" else "failed")
    assert intent["publication"]["input_fingerprint"] is None
    assert "synthetic-authorization-secret" not in (root / "manifest.json").read_text()
    a_ran = modes[0] == "architecture"
    assert ("architecture" in recorded) is a_ran
    assert flow.analyzer.run_analysis.call_count == (2 if a_ran else 0)
    assert all(path != "/repos/org/repo/issues/comments/700" for _, path in writes(flow))

    monkeypatch.setattr(flow.github, method, original)
    assert send_event(flow, "authorization-restored", "edited").status_code == 202 and flow.worker.run_once() is True
    _, status, retried = receipts(flow)
    assert status == "succeeded" and len(flow.model.requests) == 6
    assert retried["intent"]["reused"] is False
    assert retried["architecture"]["reused"] is a_ran
    assert flow.analyzer.run_analysis.call_count == 2


@pytest.mark.parametrize("method", ["find_marker_comment", "create_comment", "update_comment"])
def test_a_publication_authorization_loss_preserves_core_and_stops_b(intent_flow, monkeypatch, method):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    if method == "update_comment":
        flow.comments.append({"id": 700, "body": A_MARKER + "\nPrior review",
            "user": {"login": "review-app[bot]", "type": "Bot"}})
    original = getattr(flow.github, method)

    def reject_architecture(*args, **kwargs):
        text = kwargs.get("body", kwargs.get("marker", args[-1] if args else ""))
        if isinstance(text, str) and A_MARKER in text:
            raise GitHubAppAuthError("synthetic-authorization-secret")
        return original(*args, **kwargs)

    monkeypatch.setattr(flow.github, method, reject_architecture)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status, recorded = receipts(flow)
    assert status == "failed" and len(flow.model.requests) == 0
    assert (root / "A/receipt.json").exists() and not (root / "B/result.json").exists()
    receipt = json.loads((root / "A/receipt.json").read_text())
    core = json.loads((root / "A/result.json").read_text())
    assert receipt["a_result"] == core and receipt["usage"] == core["usage"]
    assert recorded["architecture"]["error_code"] == "publication_authorization_lost"
    assert receipt["publication"]["status"] == "unknown" and receipt["publication"]["input_fingerprint"] is None
    assert "intent" not in recorded and writes(flow) == []
    assert "synthetic-authorization-secret" not in (root / "A/receipt.json").read_text()


@pytest.mark.parametrize("mode", ["architecture", "intent"])
def test_authorization_loss_remains_common_when_receipt_storage_fails(intent_flow, monkeypatch, caplog, mode):
    flow = intent_flow
    other = "intent" if mode == "architecture" else "architecture"
    opt_in(flow, (mode, other))

    def lose_authorization(*args, **kwargs):
        raise GitHubAppAuthError("synthetic-authorization-secret")

    def storage_failure(*args, **kwargs):
        raise OSError("synthetic-storage-secret")

    monkeypatch.setattr(flow.github, "find_marker_comment", lose_authorization)
    monkeypatch.setattr(RunArtifacts, "save_receipt", storage_failure)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status = output(flow)
    assert status == "failed" and writes(flow) == []
    current_dir, other_dir = ("A", "B") if mode == "architecture" else ("B", "A")
    assert (root / current_dir / "result.json").exists() and not (root / other_dir / "result.json").exists()
    assert len(flow.model.requests) == (0 if mode == "architecture" else 3)
    assert flow.analyzer.run_analysis.call_count == (2 if mode == "architecture" else 0)
    assert flow.store._get_connection().execute("SELECT error FROM runs WHERE run_id = ?", (root.name,)).fetchone()[0] == "publication_authorization_lost"
    assert "synthetic-authorization-secret" not in caplog.text and "synthetic-storage-secret" not in caplog.text


def test_b_publication_guard_failure_hides_raw_exception_without_reads_or_writes(intent_flow, caplog):
    flow = intent_flow

    def failed_guard():
        raise ValueError("synthetic-guard-secret")

    receipt = publish_review_comment(flow.github, 99, 7, flow.pr["head"]["sha"], flow.pr["base"]["sha"],
        B_MARKER + "\nGuarded fixture", marker=B_MARKER, publication_guard=failed_guard, strict_target=True)
    assert receipt.status == "failed" and receipt.reason == "publication_fenced"
    assert flow.calls == [] and flow.comments == []
    assert "synthetic-guard-secret" not in caplog.text


@pytest.mark.parametrize("abort", ["stop", "closed", "generation", "lease", "deadline"])
@pytest.mark.parametrize("boundary", ["on_comment_read", "on_comment_write"])
def test_b_posting_fences_stop_writes_and_remaining_modes(intent_flow, monkeypatch, abort, boundary):
    flow = intent_flow
    opt_in(flow, ("intent", "architecture"))
    flow.config.execution.task_timeout_seconds = 2
    clock = [100.0]
    observed = {}
    monkeypatch.setattr("review_bot.runtime.worker.time", SimpleNamespace(monotonic=lambda: clock[0]))

    def interrupt():
        flow.issue_status.pop(boundary, None)
        observed["run_id"] = flow.worker._current_run_id
        if abort == "stop":
            flow.worker.stop()
        elif abort == "closed":
            flow.pr["state"] = "closed"
            assert send_event(flow, "closed-at-publication", "closed").status_code == 202
        elif abort == "generation":
            assert send_event(flow, "superseded-at-publication", "edited").status_code == 202
        elif abort == "lease":
            flow.store._get_connection().execute("UPDATE runs SET lease_expires_at = 0")
            observed["replacement"] = flow.store.claim_next_run("replacement", 30)
            assert observed["replacement"] is not None
        else:
            clock[0] += 3

    flow.issue_status[boundary] = interrupt
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is False
    root, status = output(flow, observed["run_id"])
    assert len(flow.model.requests) == 3 and flow.analyzer.run_analysis.call_count == 0
    assert json.loads((root / "B/usage.json").read_text())["total"]["calls"] == 3
    assert not (root / "A/result.json").exists()
    assert flow.store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0] == 0
    expected_writes = [("POST", "/repos/org/repo/issues/7/comments")] if boundary == "on_comment_write" else []
    assert writes(flow) == expected_writes
    if expected_writes:
        assert flow.comments[0]["body"].count(B_MARKER) == 1 and "Stale review" not in flow.comments[0]["body"]
    if abort in ("stop", "closed"):
        assert status == "cancelled"
    elif abort == "generation":
        assert status == "superseded"
    elif abort == "deadline":
        assert status == "failed"
        assert flow.store._get_connection().execute("SELECT reason FROM runs WHERE run_id = ?", (root.name,)).fetchone()[0] == "task_deadline_exceeded"
    else:
        assert status == "running" and observed["replacement"].run_id == root.name
        assert flow.store._get_connection().execute("SELECT worker_id FROM runs WHERE run_id = ?", (root.name,)).fetchone()[0] == "replacement"


def test_b_profile_change_does_not_invalidate_or_hide_a_cache(intent_flow, monkeypatch):
    """B's prompt profile is separate; the actual shared model profile is not."""
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    _, status, first = receipts(flow)
    assert status == "succeeded" and len(flow.model.requests) == 3
    prompt = SYSTEM_PROMPTS["assess"] + "\nFixture B prompt profile revision; retain all citation rules."
    monkeypatch.setitem(SYSTEM_PROMPTS, "assess", prompt)
    assert send_event(flow, "only-b-profile-changed", "edited").status_code == 202 and flow.worker.run_once() is True
    _, status, second = receipts(flow)
    assert status == "succeeded" and len(flow.model.requests) == 6
    assert second["architecture"]["reused"] is True and second["intent"]["reused"] is False
    assert first["architecture"]["input_fingerprint"] == second["architecture"]["input_fingerprint"]
    assert first["intent"]["input_fingerprint"] != second["intent"]["input_fingerprint"]
    assert flow.analyzer.run_analysis.call_count == 2 and len(flow.comments) == 2
    assert writes(flow)[-1] == ("PATCH", "/repos/org/repo/issues/comments/702")
    assert flow.model.requests[-1].system_prompt == prompt


def test_shared_model_profile_change_invalidates_both_mode_caches(intent_flow):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    _, _, first = receipts(flow)
    flow.model.profile = replace(flow.model.profile, model="fixture-model-v2")
    assert send_event(flow, "shared-profile-changed", "edited").status_code == 202 and flow.worker.run_once() is True
    _, status, second = receipts(flow)
    assert status == "succeeded" and len(flow.model.requests) == 6
    for mode in ("architecture", "intent"):
        assert second[mode]["reused"] is False
        assert first[mode]["input_fingerprint"] != second[mode]["input_fingerprint"]
    assert flow.analyzer.run_analysis.call_count == 4 and len(flow.comments) == 2
    assert writes(flow)[-2:] == [("PATCH", "/repos/org/repo/issues/comments/701"), ("PATCH", "/repos/org/repo/issues/comments/702")]


def test_b_cache_freshness_is_checked_again_after_marker_lookup(intent_flow):
    flow = intent_flow
    opt_in(flow)
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    flow.issue_status["on_comment_read"] = lambda: flow.issue.update(body="Changed during cache lookup")
    assert send_event(flow, "cache-race", "edited").status_code == 202 and flow.worker.run_once() is False
    root, status = output(flow)
    assert status == "superseded" and len(flow.model.requests) == 3 and len(writes(flow)) == 1


def test_b_custom_marker_stays_separate_from_a_and_needs_known_bot_identity(intent_flow):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    flow.config.publish.b_marker = "<!-- review-bot:intent:team-v1 -->"
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    assert len(flow.comments) == 2 and flow.comments[1]["body"].count(flow.config.publish.b_marker) == 1
    flow.github._bot_login = None
    flow.issue["body"] += " changed"
    assert send_event(flow, "missing-bot-identity", "edited").status_code == 202 and flow.worker.run_once() is False
    root, status, modes = receipts(flow)
    assert status == "failed" and modes["intent"]["error_code"] == "github_bot_identity_unavailable"
    assert len(writes(flow)) == 2


def test_own_comments_changing_generic_pr_updated_at_do_not_invalidate_original_text(intent_flow):
    flow = intent_flow
    opt_in(flow, ("architecture", "intent"))
    flow.pr["updated_at"] = "2026-10-01T12:00:00Z"
    updates = []

    def generic_metadata_update():
        updates.append(True)
        flow.pr["updated_at"] = f"2026-10-0{len(updates) + 1}T12:00:00Z"

    flow.issue_status["on_comment_write"] = generic_metadata_update
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    assert len(flow.comments) == 2 and len(updates) == 2
    assert flow.analyzer.run_analysis.call_count == 2 and len(flow.model.requests) == 3
    first, status, modes = receipts(flow)
    assert status == "succeeded"
    assert send_event(flow, "generic-pr-metadata", "edited").status_code == 202 and flow.worker.run_once() is True
    second, status, reused = receipts(flow)
    assert status == "skipped" and len(flow.model.requests) == 3 and len(updates) == 2
    assert reused["intent"]["input_fingerprint"] == modes["intent"]["input_fingerprint"]
    assert json.loads((first / "snapshot.json").read_text())["snapshot_id"] != json.loads((second / "snapshot.json").read_text())["snapshot_id"]