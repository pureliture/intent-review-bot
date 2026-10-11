"""Empirical adversarial stress-test suite for Milestone 4:
- review_bot.application.context (Input gate, TaskInput assembly, Evidence indexing)
- review_bot.application.review_a (Request builder, validate_model_response, Model Clients, Pipeline)

Adversarial Dimensions:
1. Input Gate Stress Testing (corrupted snapshots, missing blobs, empty files, byte/token limits)
2. TaskInput Assembly Integrity (content hashes, inventory hash, SHA mismatch, path traversals)
3. validate_model_response Echo & Schema Bounds (tampered SHAs/mode, limits >20, summary bounds)
4. validate_model_response JSON Injections (duplicate keys, NaN, Infinity, malformed JSON)
5. Citation Grounding & CPD Structural Rule Defense
6. Offline Model Clients & Leased Pipeline Execution
7. Empirical Vulnerabilities & Seam Divergence Identification
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from review_bot.application.context import (
    build_task_input,
    check_input_gate,
)
from review_bot.application.delta import compute_delta
from review_bot.application.review_a import (
    PROMPT_VERSION,
    MockModelClient,
    ModelProfile,
    PermanentModelError,
    TransientModelError,
    UnconfiguredModelClient,
    build_model_request,
    run_review_a_pipeline,
    validate_model_response,
)
from review_bot.domain.models_v1 import (
    Classification,
    CodeFile,
    CodeSnapshot,
    Evidence,
    ModelRequest,
    ModelResponse,
    PrSnapshot,
    Snapshot,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_snapshot(
    base_files: dict[str, str] | None = None,
    head_files: dict[str, str] | None = None,
    *,
    base_sha: str = "a" * 40,
    head_sha: str = "b" * 40,
    base_complete: bool = True,
    head_complete: bool = True,
    base_errors: tuple[str, ...] = (),
    head_errors: tuple[str, ...] = (),
    source_branch: str = "feature/adversarial",
    labels: tuple[str, ...] = (),
    changed_files: tuple[str, ...] | None = None,
    rename_map: dict[str, str] | None = None,
    base_inv_hash: str | None = None,
    head_inv_hash: str | None = None,
) -> Snapshot:
    base_files = base_files or {"Service.java": "public class Service {}"}
    head_files = head_files or {"Service.java": "public class Service { int x; }"}

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
        inventory_hash=base_inv_hash or _digest("base-inv"),
        complete=base_complete,
        errors=base_errors,
    )
    head_snap = CodeSnapshot(
        sha=head_sha,
        files=head_code_files,
        exclusions=(),
        inventory_hash=head_inv_hash or _digest("head-inv"),
        complete=head_complete,
        errors=head_errors,
    )

    pr = PrSnapshot(
        repository="test-org/adversarial-repo",
        number=42,
        base_sha=base_sha,
        head_sha=head_sha,
        source_branch=source_branch,
        labels=labels,
        changed_files=changed_files if changed_files is not None else tuple(head_files.keys()),
        rename_map=rename_map or {},
    )

    return Snapshot(pr=pr, base=base_snap, head=head_snap)


# ==============================================================================
# Category 1: Input Gate Stress Testing
# ==============================================================================


class TestInputGateAdversarial:
    """Stress-test input gate under incomplete snapshots, errors, empty files, and size limits."""

    def test_gate_rejects_incomplete_base_snapshot(self) -> None:
        snap = _make_snapshot(base_complete=False)
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

    def test_gate_rejects_incomplete_head_snapshot(self) -> None:
        snap = _make_snapshot(head_complete=False)
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

    def test_gate_rejects_base_snapshot_with_missing_blob_errors(self) -> None:
        snap = _make_snapshot(base_errors=("fatal: git blob missing: 12345678",))
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

    def test_gate_rejects_head_snapshot_with_read_errors(self) -> None:
        snap = _make_snapshot(head_errors=("fatal: git tree read error: eof",))
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

    def test_gate_accepts_empty_files_without_crashing(self) -> None:
        snap = _make_snapshot(
            base_files={"Empty.java": ""},
            head_files={"Empty.java": ""},
        )
        can_proceed, status, reason = check_input_gate(snap)
        assert can_proceed is True
        assert status == "can_proceed"
        assert reason == ""

        # Verify task input assembly survives zero-line files
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        task = build_task_input(snap, delta, Classification("general"))
        assert task.input_hash != ""
        ev = next(e for e in task.evidence if e.kind == "code" and e.path == "Empty.java")
        assert ev.begin_line == 0
        assert ev.end_line == 0
        assert ev.content == ""

    def test_gate_rejects_byte_limit_overage(self) -> None:
        snap = _make_snapshot(
            base_files={"A.java": "x" * 100},
            head_files={"A.java": "x" * 100},
        )
        can_proceed, status, reason = check_input_gate(snap, limits={"max_input_bytes": 150})
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "input_limit_exceeded"

    def test_gate_rejects_token_limit_overage(self) -> None:
        snap = _make_snapshot()
        can_proceed, status, reason = check_input_gate(
            snap, limits={"max_input_tokens": 1000, "token_count": 1001}
        )
        assert can_proceed is False
        assert status == "not_run"
        assert reason == "input_limit_exceeded"


# ==============================================================================
# Category 2: TaskInput Assembly & Input Validation
# ==============================================================================


class TestTaskInputAssemblyAdversarial:
    """Stress-test input tampering and invariant validation in build_task_input."""

    def test_rejects_tampered_source_content_hash(self) -> None:
        snap = _make_snapshot()
        # Tamper base code file hash
        tampered_file = CodeFile("Service.java", snap.base.files[0].content, "0" * 64, "source")
        tampered_base = snap.base.__class__(
            sha=snap.base.sha,
            files=(tampered_file,),
            exclusions=(),
            inventory_hash=snap.base.inventory_hash,
            complete=True,
            errors=(),
        )
        tampered_snap = Snapshot(pr=snap.pr, base=tampered_base, head=snap.head)
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)

        with pytest.raises(ValueError, match="Invalid base source content hash"):
            build_task_input(tampered_snap, delta, Classification("general"))

    def test_rejects_malformed_inventory_hash(self) -> None:
        snap = _make_snapshot(base_inv_hash="not_a_sha256_hash")
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        with pytest.raises(ValueError, match="Invalid base inventory hash"):
            build_task_input(snap, delta, Classification("general"))

    def test_rejects_mismatched_base_sha(self) -> None:
        snap = _make_snapshot()
        delta = compute_delta(base_findings=[], head_findings=[], base_sha="c" * 40, head_sha=snap.head.sha)
        with pytest.raises(ValueError, match="Mismatched base fixed versions"):
            build_task_input(snap, delta, Classification("general"))

    def test_rejects_path_traversal_in_changed_files(self) -> None:
        snap = _make_snapshot(changed_files=("../secret.txt",))
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        with pytest.raises(ValueError, match="Invalid changed-file path"):
            build_task_input(snap, delta, Classification("general"))

    def test_rejects_absolute_path_in_rename_map(self) -> None:
        snap = _make_snapshot(rename_map={"Old.java": "/etc/passwd"})
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        with pytest.raises(ValueError, match="Invalid rename path"):
            build_task_input(snap, delta, Classification("general"))

    def test_rejects_ambiguous_rename_map(self) -> None:
        snap = _make_snapshot(rename_map={"A.java": "Merged.java", "B.java": "Merged.java"})
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        with pytest.raises(ValueError, match="Ambiguous rename map"):
            build_task_input(snap, delta, Classification("general"))

    def test_rejects_perspective_classification_mismatch(self) -> None:
        snap = _make_snapshot(source_branch="refactor/cleanup")
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        # Expected is refactoring, but passed general
        with pytest.raises(ValueError, match="Classification does not match"):
            build_task_input(snap, delta, Classification("general"))


# ==============================================================================
# Category 3: validate_model_response Echo & Schema Bounds
# ==============================================================================


class TestModelEchoAndSchemaBoundsAdversarial:
    """Stress-test strict echo validation and schema boundary limits."""

    def test_rejects_tampered_base_sha(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        resp = {
            "base_sha": "f" * 40,
            "head_sha": head_sha,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [],
            "questions": [],
        }
        errs = validate_model_response(resp, base_sha, head_sha, "general", {})
        assert any(f"Mismatched base_sha: expected {base_sha}" in e for e in errs)

    def test_rejects_tampered_head_sha(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        resp = {
            "base_sha": base_sha,
            "head_sha": "f" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [],
            "questions": [],
        }
        errs = validate_model_response(resp, base_sha, head_sha, "general", {})
        assert any(f"Mismatched head_sha: expected {head_sha}" in e for e in errs)

    def test_rejects_tampered_review_mode(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        resp = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": "refactoring",
            "summary": "Valid summary",
            "findings": [],
            "questions": [],
        }
        errs = validate_model_response(resp, base_sha, head_sha, "general", {})
        assert any("Mismatched review_mode: expected general" in e for e in errs)

    def test_rejects_findings_count_exceeding_20(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [{"id": f"F-{i}"} for i in range(21)],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert any("Findings must be a list with at most 20 items" in e for e in errs)

    def test_rejects_questions_count_exceeding_20(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [],
            "questions": [{"id": f"Q-{i}"} for i in range(21)],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert any("Questions must be a list with at most 20 items" in e for e in errs)

    @pytest.mark.parametrize("invalid_summary", ["", "a" * 4001, 12345, None, []])
    def test_rejects_out_of_bounds_summary(self, invalid_summary: Any) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": invalid_summary,
            "findings": [],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert any("Summary must be between 1 and 4000 characters" in e for e in errs)


# ==============================================================================
# Category 4: validate_model_response JSON Injections
# ==============================================================================


class TestModelJsonInjectionsAdversarial:
    """Stress-test JSON duplicate keys, non-finite values (NaN, Infinity), and syntax."""

    def test_rejects_duplicate_json_keys(self) -> None:
        raw_json = (
            '{"base_sha": "' + "a" * 40 + '", '
            '"head_sha": "' + "b" * 40 + '", '
            '"review_mode": "general", '
            '"summary": "s1", '
            '"summary": "s2", '
            '"findings": [], "questions": []}'
        )
        errs = validate_model_response(raw_json, "a" * 40, "b" * 40, "general", {})
        assert any("Duplicate key in JSON: summary" in e for e in errs)

    @pytest.mark.parametrize("constant_name", ["NaN", "Infinity", "-Infinity"])
    def test_rejects_nonfinite_numeric_constants_in_json(self, constant_name: str) -> None:
        raw_json = (
            '{"base_sha": "' + "a" * 40 + '", '
            '"head_sha": "' + "b" * 40 + '", '
            '"review_mode": "general", '
            '"summary": "Valid summary", '
            f'"val": {constant_name}, '
            '"findings": [], "questions": []}'
        )
        errs = validate_model_response(raw_json, "a" * 40, "b" * 40, "general", {})
        assert any(f"Non-finite numeric constant in JSON: {constant_name}" in e for e in errs)

    def test_rejects_malformed_unparseable_json(self) -> None:
        errs = validate_model_response('{"base_sha": "broken', "a" * 40, "b" * 40, "general", {})
        assert any("Malformed response JSON" in e for e in errs)


# ==============================================================================
# Category 5: Citation Grounding & CPD Structural Rule Defense
# ==============================================================================


class TestCitationGroundingAndCpdDefenseAdversarial:
    """Stress-test citation attacks and CPD structural rule defense enforcement."""

    def test_rejects_hallucinated_evidence_id(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [
                {
                    "signal_id": "pmd-1",
                    "citations": [{"evidence_id": "ghost-evidence-999", "quote": "class X"}],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {"real-ev": "class X"})
        assert any("Unknown evidence_id referenced: ghost-evidence-999" in e for e in errs)

    def test_rejects_non_verbatim_quote(self) -> None:
        evidence = {"real-ev": "public class OrderService { void process() {} }"}
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [
                {
                    "signal_id": "pmd-1",
                    "citations": [{"evidence_id": "real-ev", "quote": "public class FabricatedService"}],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", evidence)
        assert any("not found verbatim in evidence real-ev" in e for e in errs)

    def test_cpd_signal_concern_with_analysis_signal_basis_is_rejected(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Summary",
            "findings": [
                {
                    "signal_id": "cpd-duplicate-block-1",
                    "judgement": "concern",
                    "basis": "analysis_signal",
                    "title": "Duplicated logic concern",
                    "reason": "Found duplicate code",
                    "recommendation": "Extract common base class",
                    "evidence_ids": [],
                    "citations": [],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert any("CPD count alone cannot justify an architecture concern" in e for e in errs)

    def test_cpd_count_only_basis_is_rejected(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Summary",
            "findings": [
                {
                    "category": "architecture_concern",
                    "basis": "cpd_count_only",
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert any("CPD count alone cannot justify an architecture concern" in e for e in errs)

    def test_cpd_signal_with_observed_structure_basis_is_permitted(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Summary",
            "findings": [
                {
                    "signal_id": "cpd-duplicate-block-1",
                    "judgement": "concern",
                    "basis": "observed_structure",
                    "title": "Responsibility divergence",
                    "reason": "Duplicate classes share lifecycle responsibility",
                    "recommendation": "Unify lifecycle",
                    "evidence_ids": [],
                    "citations": [],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert not any("CPD count alone" in e for e in errs)

    def test_pmd_signal_concern_with_analysis_signal_basis_is_permitted(self) -> None:
        resp = {
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "review_mode": "general",
            "summary": "Summary",
            "findings": [
                {
                    "signal_id": "pmd-cognitive-complexity-1",
                    "judgement": "concern",
                    "basis": "analysis_signal",
                    "title": "Cognitive complexity violation",
                    "reason": "Score 25 exceeds limit",
                    "recommendation": "Split method",
                    "evidence_ids": [],
                    "citations": [],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "a" * 40, "b" * 40, "general", {})
        assert not any("CPD count alone" in e for e in errs)


# ==============================================================================
# Category 6: Offline Model Clients & Leased Pipeline Execution
# ==============================================================================


class TestOfflineClientsAndPipelineAdversarial:
    """Stress-test offline model clients, request builder, and leased review pipeline."""

    def test_unconfigured_client_invariants(self) -> None:
        client = UnconfiguredModelClient()
        assert client.profile is None
        req = ModelRequest(1, "mock", "sys", "usr", 10, None, {})
        with pytest.raises(PermanentModelError, match="No model connection is configured"):
            client.complete(req)
        with pytest.raises(NotImplementedError, match="No model-specific tokenizer is configured"):
            client.count_input_tokens(req)

    def test_mock_client_playback_and_exhaustion(self) -> None:
        client = MockModelClient([ModelResponse("resp 1"), ModelResponse("resp 2")])
        req = ModelRequest(1, "mock", "sys", "usr", 10, None, {})
        assert client.complete(req).text == "resp 1"
        assert client.complete(req).text == "resp 2"
        with pytest.raises(PermanentModelError, match="Mock response sequence is exhausted"):
            client.complete(req)

    def test_request_builder_pins_review_a_v1_and_schema(self) -> None:
        snap = _make_snapshot()
        delta = compute_delta(base_findings=[], head_findings=[], base_sha=snap.base.sha, head_sha=snap.head.sha)
        task = build_task_input(snap, delta, Classification("general"))
        req = build_model_request(task)
        assert f"Version: {PROMPT_VERSION}" in req.system_prompt
        assert req.response_schema["type"] == "object"
        assert "base_sha" in req.response_schema["properties"]
        assert req.response_schema["properties"]["base_sha"]["const"] == snap.pr.base_sha

    def test_pipeline_aborts_when_input_bytes_exceed_profile(self) -> None:
        snap = _make_snapshot(
            base_files={"A.java": "public class A {\n" + "    int x = 1;\n" * 50 + "}\n"},
            head_files={"A.java": "public class A {\n" + "    int x = 2;\n" * 50 + "}\n"},
        )
        delta = compute_delta(
            base_findings=[],
            head_findings=[{"rule": "CognitiveComplexity", "path": "A.java", "metric": 20}],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )
        task = build_task_input(snap, delta, Classification("general"))

        profile = ModelProfile(provider="mock", model="m", max_input_bytes=100)
        client = MockModelClient([], profile=profile)
        res = run_review_a_pipeline(task, client)
        assert res.status == "not_run"
        assert res.reason == "input_limit_exceeded"

    def test_pipeline_aborts_when_input_tokens_exceed_profile(self) -> None:
        snap = _make_snapshot()
        delta = compute_delta(
            base_findings=[],
            head_findings=[{"rule": "CognitiveComplexity", "path": "Service.java", "metric": 20}],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )
        task = build_task_input(snap, delta, Classification("general"))

        profile = ModelProfile(provider="mock", model="m", max_input_tokens=50)
        client = MockModelClient([], profile=profile, input_tokens=100)
        res = run_review_a_pipeline(task, client)
        assert res.status == "not_run"
        assert res.reason == "input_limit_exceeded"

    def test_pipeline_aborts_when_token_counting_unavailable(self) -> None:
        snap = _make_snapshot()
        delta = compute_delta(
            base_findings=[],
            head_findings=[{"rule": "CognitiveComplexity", "path": "Service.java", "metric": 20}],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )
        task = build_task_input(snap, delta, Classification("general"))

        profile = ModelProfile(provider="mock", model="m", max_input_tokens=500)
        # Client without input_tokens configured raises NotImplementedError
        client = MockModelClient([], profile=profile)
        res = run_review_a_pipeline(task, client)
        assert res.status == "not_run"
        assert res.reason == "token_count_unavailable"

    def test_pipeline_timeout_handling(self) -> None:
        snap = _make_snapshot()
        delta = compute_delta(
            base_findings=[],
            head_findings=[{"rule": "CognitiveComplexity", "path": "Service.java", "metric": 20}],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )
        task = build_task_input(snap, delta, Classification("general"))

        profile = ModelProfile(provider="mock", model="m", max_attempts=1)
        client = MockModelClient([TimeoutError("LLM call timed out")], profile=profile)
        res = run_review_a_pipeline(task, client)
        assert res.status == "failed"
        assert res.reason == "model_timeout"

    def test_pipeline_transient_retry_success(self) -> None:
        code_base = "public class Service {\n    void run() {}\n}\n"
        code_head = "public class Service {\n    void run() {\n        int x = 1;\n    }\n}\n"
        snap = _make_snapshot({"src/Service.java": code_base}, {"src/Service.java": code_head})
        delta = compute_delta(
            base_findings=[],
            head_findings=[{"rule": "CognitiveComplexity", "path": "src/Service.java", "metric": 20, "method": "run"}],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )
        task = build_task_input(snap, delta, Classification("general"))
        code_ev = next(e for e in task.evidence if e.kind == "code" and e.version == "head")
        sig_ev = next(e for e in task.evidence if e.kind == "signal")

        valid_response = {
            "base_sha": snap.base.sha,
            "head_sha": snap.head.sha,
            "review_mode": "general",
            "summary": "Review complete",
            "findings": [
                {
                    "signal_id": task.signals[0].signal.signal_id,
                    "judgement": "concern",
                    "change": "new",
                    "basis": "analysis_signal",
                    "title": "Title",
                    "reason": "Reason",
                    "recommendation": "Rec",
                    "evidence_ids": [sig_ev.evidence_id, code_ev.evidence_id],
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

        profile = ModelProfile(provider="mock", model="m", max_attempts=2)
        client = MockModelClient(
            [TransientModelError("503 Server Error"), ModelResponse(json.dumps(valid_response))],
            profile=profile,
        )
        res = run_review_a_pipeline(task, client)
        assert res.status == "completed"
        assert res.reason == "validated"
        assert len(res.findings) == 1

    def test_pipeline_transient_retry_exhaustion(self) -> None:
        snap = _make_snapshot()
        delta = compute_delta(
            base_findings=[],
            head_findings=[{"rule": "CognitiveComplexity", "path": "Service.java", "metric": 20}],
            base_sha=snap.base.sha,
            head_sha=snap.head.sha,
        )
        task = build_task_input(snap, delta, Classification("general"))

        profile = ModelProfile(provider="mock", model="m", max_attempts=2)
        client = MockModelClient(
            [TransientModelError("503"), TransientModelError("503")],
            profile=profile,
        )
        res = run_review_a_pipeline(task, client)
        assert res.status == "failed"
        assert res.reason == "model_error"


# ==============================================================================
# Category 7: Empirical Vulnerabilities & Seam Divergence Identification
# ==============================================================================


class TestEmpiricalVulnerabilities:
    """Verifies that previously identified empirical contract gaps are strictly rejected."""

    def test_vulnerability_inverted_and_out_of_bounds_citation_lines_accepted(self) -> None:
        """contracts.md §7/§9 requires citations to match exact line spans on fixed versions.

        Remediated: validate_model_response verifies begin_line <= end_line and checks
        bounds against referenced evidence bounds.
        """
        ev = Evidence("E1", "code", "head", "1" * 40, "A.java", 1, 5, "public class A {\n    int x;\n}\n")
        resp = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [
                {
                    "signal_id": "sig-1",
                    "judgement": "concern",
                    "change": "new",
                    "basis": "analysis_signal",
                    "title": "Title",
                    "reason": "Reason",
                    "recommendation": "Rec",
                    "evidence_ids": ["E1"],
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "version": "head",
                            "sha": "1" * 40,
                            "path": "A.java",
                            "begin_line": 9999,  # Inverted & completely out of bounds (file has 5 lines)
                            "end_line": 2,
                            "quote": "public class A",
                        }
                    ],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "0" * 40, "1" * 40, "general", {"E1": ev})
        assert len(errs) > 0, "Remediated: inverted and out-of-bounds line ranges must be rejected"
        assert any("Inverted citation line range" in e for e in errs)
        assert any("exceeds evidence E1 end_line" in e for e in errs)

    def test_vulnerability_citation_mismatched_sha_version_path_accepted(self) -> None:
        """contracts.md §7: Citation path, version, and sha must align with referenced Evidence.

        Remediated: validate_model_response checks citation path, version, and sha alignment.
        """
        ev = Evidence("E1", "code", "head", "1" * 40, "A.java", 1, 5, "public class A {\n    int x;\n}\n")
        resp = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [
                {
                    "signal_id": "sig-1",
                    "judgement": "concern",
                    "change": "new",
                    "basis": "analysis_signal",
                    "title": "Title",
                    "reason": "Reason",
                    "recommendation": "Rec",
                    "evidence_ids": ["E1"],
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "version": "base",  # MISMATCH: evidence E1 is head
                            "sha": "9" * 40,   # MISMATCH: evidence E1 sha is 1*40
                            "path": "CompletelyDifferent.java",  # MISMATCH: evidence path is A.java
                            "begin_line": 1,
                            "end_line": 2,
                            "quote": "public class A",
                        }
                    ],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "0" * 40, "1" * 40, "general", {"E1": ev})
        assert len(errs) > 0, "Remediated: citation version/sha/path mismatches must be rejected"
        assert any("path" in e and "does not match" in e for e in errs)
        assert any("version" in e and "does not match" in e for e in errs)
        assert any("sha" in e and "does not match" in e for e in errs)

    def test_vulnerability_questions_citations_unvalidated(self) -> None:
        """contracts.md §7: Question citations must reference real input evidence and quotes.

        Remediated: validate_model_response validates question citations with identical rigor.
        """
        ev = Evidence("E1", "code", "head", "1" * 40, "A.java", 1, 5, "public class A {\n    int x;\n}\n")
        resp = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [],
            "questions": [
                {
                    "signal_id": "sig-1",
                    "question": "Is this tested?",
                    "reason": "Need test verification",
                    "evidence_ids": ["E1"],
                    "citations": [
                        {
                            "evidence_id": "HALLUCINATED_GHOST_EVIDENCE",
                            "quote": "completely fabricated nonexistent code",
                        }
                    ],
                }
            ],
        }
        errs = validate_model_response(resp, "0" * 40, "1" * 40, "general", {"E1": ev})
        assert len(errs) > 0, "Remediated: question citations must be validated"
        assert any("Unknown evidence_id referenced: HALLUCINATED_GHOST_EVIDENCE" in e for e in errs)

    def test_vulnerability_arbitrary_extra_fields_silently_accepted(self) -> None:
        """contracts.md §7: Reject extra fields. PROJECT.md Feature 16: 'no extra fields'.

        Remediated: validate_model_response enforces strictly ALLOWED_TOP_LEVEL_KEYS.
        """
        resp = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid summary",
            "findings": [],
            "questions": [],
            "status": "approved",  # Model attempting to forge bot status
            "confidence": 0.99,    # Forbidden field per system prompt
            "malicious_injection": "arbitrary_payload",
        }
        errs = validate_model_response(resp, "0" * 40, "1" * 40, "general", {})
        assert len(errs) > 0, "Remediated: extra fields must be rejected"
        assert any("Unexpected top-level fields in model response:" in e for e in errs)
        assert any("confidence" in e for e in errs)
        assert any("status" in e for e in errs)

    @pytest.mark.parametrize("bad_line", [0, -1, -99, "1", 1.5, True, False])
    def test_rejects_non_positive_or_non_integer_citation_lines(self, bad_line: Any) -> None:
        ev = Evidence("E1", "code", "head", "1" * 40, "A.java", 1, 10, "class A {}")
        resp = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid",
            "findings": [
                {
                    "signal_id": "sig-1",
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "path": "A.java",
                            "version": "head",
                            "sha": "1" * 40,
                            "begin_line": bad_line,
                            "end_line": 5,
                        }
                    ],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp, "0" * 40, "1" * 40, "general", {"E1": ev})
        assert any("Citation begin_line must be a positive integer" in e for e in errs)

    def test_citation_version_both_compatibility(self) -> None:
        ev_both = Evidence("E-BOTH", "code", "both", "1" * 40, "Shared.java", 1, 10, "class Shared {}")
        # 'head' is compatible with 'both'
        resp_head = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid",
            "findings": [
                {
                    "signal_id": "sig-1",
                    "citations": [
                        {
                            "evidence_id": "E-BOTH",
                            "version": "head",
                            "path": "Shared.java",
                            "sha": "1" * 40,
                            "begin_line": 1,
                            "end_line": 5,
                        }
                    ],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp_head, "0" * 40, "1" * 40, "general", {"E-BOTH": ev_both})
        assert errs == []

        # 'base' is compatible with 'both'
        resp_base = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid",
            "findings": [
                {
                    "signal_id": "sig-1",
                    "citations": [
                        {
                            "evidence_id": "E-BOTH",
                            "version": "base",
                            "path": "Shared.java",
                            "sha": "1" * 40,
                            "begin_line": 1,
                            "end_line": 5,
                        }
                    ],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp_base, "0" * 40, "1" * 40, "general", {"E-BOTH": ev_both})
        assert errs == []

        # 'invalid_version' is incompatible with 'both'
        resp_invalid = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid",
            "findings": [
                {
                    "signal_id": "sig-1",
                    "citations": [
                        {
                            "evidence_id": "E-BOTH",
                            "version": "invalid_version",
                            "path": "Shared.java",
                            "sha": "1" * 40,
                        }
                    ],
                }
            ],
            "questions": [],
        }
        errs = validate_model_response(resp_invalid, "0" * 40, "1" * 40, "general", {"E-BOTH": ev_both})
        assert any("incompatible with evidence E-BOTH version 'both'" in e for e in errs)

    def test_question_citations_comprehensive_validation(self) -> None:
        ev = Evidence("E1", "code", "head", "1" * 40, "A.java", 1, 10, "class A {\n    int x;\n}\n")
        resp = {
            "base_sha": "0" * 40,
            "head_sha": "1" * 40,
            "review_mode": "general",
            "summary": "Valid",
            "findings": [],
            "questions": [
                {
                    "signal_id": "sig-1",
                    "question": "Is this tested?",
                    "reason": "Need test verification",
                    "evidence_ids": ["E1"],
                    "citations": [
                        {
                            "evidence_id": "E1",
                            "version": "base",  # Mismatch
                            "sha": "wrong_sha",  # Mismatch
                            "path": "Wrong.java",  # Mismatch
                            "begin_line": 8,
                            "end_line": 2,  # Inverted
                            "quote": "nonexistent quote",  # Not verbatim
                        }
                    ],
                }
            ],
        }
        errs = validate_model_response(resp, "0" * 40, "1" * 40, "general", {"E1": ev})
        assert any("Inverted citation line range" in e for e in errs)
        assert any("Citation quote 'nonexistent quote' not found verbatim" in e for e in errs)
        assert any("Citation path 'Wrong.java' does not match evidence E1 path 'A.java'" in e for e in errs)
        assert any("Citation version 'base' does not match evidence E1 version 'head'" in e for e in errs)
        assert any("Citation sha 'wrong_sha' does not match evidence E1 sha" in e for e in errs)
