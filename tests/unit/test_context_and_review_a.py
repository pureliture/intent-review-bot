"""Unit tests for Milestone 4: context.py and review_a.py application components."""
from __future__ import annotations

import hashlib
import json

from review_bot.application.context import (
    build_task_input,
    check_input_gate,
)
from review_bot.application.context import (
    classify_pr as context_classify_pr,
)
from review_bot.application.delta import compute_delta
from review_bot.application.review_a import (
    PROMPT_VERSION,
    MockModelClient,
    ModelProfile,
    PermanentModelError,
    UnconfiguredModelClient,
    build_model_request,
    review_a,
    run_review_a_pipeline,
)
from review_bot.domain.models_v1 import (
    Classification,
    CodeFile,
    CodeSnapshot,
    ModelRequest,
    ModelResponse,
    PrSnapshot,
    Snapshot,
)


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _make_snapshot(
    base_files: dict[str, str],
    head_files: dict[str, str],
    base_complete: bool = True,
    head_complete: bool = True,
    base_errors: tuple[str, ...] = (),
    head_errors: tuple[str, ...] = (),
) -> Snapshot:
    base_sha = "0" * 40
    head_sha = "1" * 40

    base_code_files = tuple(
        CodeFile(path, content, _digest(content), "source")
        for path, content in sorted(base_files.items())
    )
    head_code_files = tuple(
        CodeFile(path, content, _digest(content), "source")
        for path, content in sorted(head_files.items())
    )

    base_snap = CodeSnapshot(
        sha=base_sha,
        files=base_code_files,
        exclusions=(),
        inventory_hash=_digest("base-inv"),
        complete=base_complete,
        errors=base_errors,
    )
    head_snap = CodeSnapshot(
        sha=head_sha,
        files=head_code_files,
        exclusions=(),
        inventory_hash=_digest("head-inv"),
        complete=head_complete,
        errors=head_errors,
    )

    pr = PrSnapshot(
        repository="org/repo",
        number=42,
        base_sha=base_sha,
        head_sha=head_sha,
        source_branch="feature/test",
        labels=(),
        changed_files=tuple(head_files.keys()),
        rename_map={},
    )

    return Snapshot(pr=pr, base=base_snap, head=head_snap)


# ==============================================================================
# Context & Input Gate Tests
# ==============================================================================


class TestContextAndInputGate:
    def test_classify_pr_reexport_works(self) -> None:
        c1 = context_classify_pr("refactor/clean-architecture", ())
        assert c1.review_mode == "refactoring"
        c2 = context_classify_pr("feature/login", ["리팩토링"])
        assert c2.review_mode == "refactoring"
        c3 = context_classify_pr("feature/login", ())
        assert c3.review_mode == "general"

    def test_check_input_gate_complete_passes(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A { int x; }"})
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is True
        assert status == "can_proceed"
        assert reason == ""

    def test_check_input_gate_incomplete_fails(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A {}"}, base_complete=False)
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

    def test_check_input_gate_snapshot_errors_fails(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A {}"}, head_errors=("Blob missing",))
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

    def test_check_input_gate_byte_limit_exceeded(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A {}"})
        limits = {"max_input_bytes": 5}  # Very small
        can_proceed, status, reason = check_input_gate(snap, limits=limits)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "input_limit_exceeded"

    def test_build_task_input_assembles_evidence_and_hashes(self) -> None:
        base_src = "public class Service {\n    void run() {}\n}\n"
        head_src = "public class Service {\n    void run() {\n        int y = 2;\n    }\n}\n"
        snap = _make_snapshot({"src/Service.java": base_src}, {"src/Service.java": head_src})

        delta = compute_delta(
            base_findings=[],
            head_findings=[
                {"rule": "CognitiveComplexity", "path": "src/Service.java", "metric": 20, "method": "run"}
            ],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )

        classification = Classification(review_mode="general", reasons=("General mode",))
        task_input = build_task_input(snap, delta, classification)

        assert task_input.schema_version == 1
        assert task_input.review_mode == "general"
        assert task_input.input_hash != ""
        assert len(task_input.evidence) > 0

        # Check evidence kinds
        kinds = {e.kind for e in task_input.evidence}
        assert "code" in kinds
        assert "signal" in kinds

        # Verify selected signal is review role
        assert len(task_input.signals) == 1
        assert task_input.signals[0].role == "review"


# ==============================================================================
# Review A Pipeline & Model Clients Tests
# ==============================================================================


class TestReviewAPipeline:
    def test_unconfigured_model_client_raises(self) -> None:
        client = UnconfiguredModelClient()
        assert client.profile is None
        try:
            client.complete(ModelRequest(1, "mock", "sys", "usr", 10, None, {}))
            raise AssertionError("Expected PermanentModelError")
        except PermanentModelError:
            pass

    def test_mock_model_client_executes_responses(self) -> None:
        mock_resp = ModelResponse("{\"ok\": true}")
        client = MockModelClient([mock_resp], input_tokens=150)
        req = ModelRequest(1, "mock", "sys", "usr", 10, None, {})
        resp = client.complete(req)
        assert resp.text == "{\"ok\": true}"
        assert client.count_input_tokens(req) == 150
        assert len(client.requests) == 1

    def test_build_model_request_pins_review_a_v1(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A {}"})
        delta = compute_delta(base_findings=[], head_findings=[])
        classification = Classification(review_mode="general")
        task = build_task_input(snap, delta, classification)

        request = build_model_request(task)
        assert PROMPT_VERSION in request.system_prompt
        assert "review-a/v1" in request.system_prompt
        assert request.response_schema["type"] == "object"
        assert "base_sha" in request.response_schema["properties"]

    def test_pipeline_aborts_on_unconfigured_profile(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A {}"})
        delta = compute_delta(base_findings=[], head_findings=[])
        task = build_task_input(snap, delta, Classification(review_mode="general"))

        result = run_review_a_pipeline(task, UnconfiguredModelClient())
        assert result.status == "not_run"
        assert result.reason == "model_profile_unset"

    def test_pipeline_aborts_on_no_review_targets(self) -> None:
        snap = _make_snapshot({"A.java": "class A {}"}, {"A.java": "class A {}"})
        delta = compute_delta(base_findings=[], head_findings=[])
        task = build_task_input(snap, delta, Classification(review_mode="general"))

        profile = ModelProfile(provider="mock", model="test-m")
        result = run_review_a_pipeline(task, MockModelClient([]), profile=profile)
        assert result.status == "not_run"
        assert result.reason == "no_review_targets"

    def test_pipeline_successful_completed_run(self) -> None:
        code_base = "public class Service {\n    void run() {}\n}\n"
        code_head = "public class Service {\n    void run() {\n        int x = 1;\n    }\n}\n"
        snap = _make_snapshot({"src/Service.java": code_base}, {"src/Service.java": code_head})

        delta = compute_delta(
            base_findings=[],
            head_findings=[
                {"rule": "CognitiveComplexity", "path": "src/Service.java", "metric": 20, "method": "run"}
            ],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )

        task = build_task_input(snap, delta, Classification(review_mode="general"))
        code_ev = next(e for e in task.evidence if e.kind == "code" and e.version == "head")
        signal_ev = next(e for e in task.evidence if e.kind == "signal")

        valid_response_data = {
            "base_sha": snap.base.sha,
            "head_sha": snap.head.sha,
            "review_mode": "general",
            "summary": "Valid architecture review completed.",
            "findings": [
                {
                    "signal_id": task.signals[0].signal.signal_id,
                    "judgement": "concern",
                    "change": "new",
                    "basis": "analysis_signal",
                    "title": "Method run complexity concern",
                    "reason": "Cognitive complexity is 20",
                    "recommendation": "Refactor method run",
                    "evidence_ids": [signal_ev.evidence_id, code_ev.evidence_id],
                    "citations": [
                        {
                            "evidence_id": code_ev.evidence_id,
                            "version": "head",
                            "sha": snap.head.sha,
                            "path": "src/Service.java",
                            "begin_line": 1,
                            "end_line": 5,
                            "quote": code_head,
                        }
                    ],
                }
            ],
            "questions": [],
        }

        mock_client = MockModelClient(
            [ModelResponse(json.dumps(valid_response_data), usage={"total_tokens": 100})]
        )
        result = review_a(task, mock_client)
        assert result.status == "completed"
        assert result.reason == "validated"
        assert len(result.findings) == 1
        assert result.findings[0].title == "Method run complexity concern"
