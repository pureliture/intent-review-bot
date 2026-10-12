"""Real Git/SQLite signed requests, isolated A/B comments and successful-mode reuse."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from review_bot.adapters.github.auth import GitHubAppAuthError
from review_bot.application.review_a import MockModelClient
from review_bot.domain.models_v1 import ModelResponse
from tests.integration.test_intent_flow import (
    enable_a_fixture,
    intent_flow,
    output,
    send_event,
    synthetic_repo,
    use_modes,
)

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


@pytest.mark.parametrize("mutation", ["issue", "head", "draft", "body"])
def test_b_post_write_race_marks_existing_comment_stale_not_success(intent_flow, mutation):
    flow = intent_flow
    opt_in(flow)

    def mutate():
        flow.issue_status.pop("on_comment_write", None)
        if mutation == "issue":
            flow.issue["body"] = "Post-write changed requirement"
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