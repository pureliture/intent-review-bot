"""Tier 5 Adversarial Coverage Hardening Suite for Core Domain, Application, and Adapters.

White-box empirical challenge tests covering:
1. review_bot.domain:
   - Frozen dataclass immutability (FrozenInstanceError on mutation & attribute deletion)
   - Default factory isolation (fresh dictionaries/collections per instance)
   - DeltaSignal & AnalysisDelta get() accessors and boundary handling
   - ParsedReport finding counts across PMD, CPD, and unknown tool strings
   - AnalysisDelta counts across 6 canonical states plus unknown/unrecognized states
   - ModeSequence equality/hashing and SelectionRecord dict protocol access
   - Deep nested round-trip serialization (asdict -> json.dumps -> json.loads)
2. review_bot.application:
   - classification.py: case-insensitivity, exact prefix matching, Korean label matching, title strictly ignored
   - selection.py: mode registration validation, unsupported policies, deduplication, deterministic hashing
   - snapshot.py: ChangedFile vs string paths, non-ASCII title/body, unavailable/ambiguous merge-bases
   - context.py: fixed SHA validation, safe path sanitation, span validity, input gate limits, model contract validation
   - review_a.py: prompt specialization, JSON schema rules, duplicate key & non-finite numeric rejection,
     echo verification, citation validation (inverted spans, non-verbatim quotes, evidence bounds),
     CPD structural rule defense, and pipeline execution error paths
   - delta.py: metric pattern extraction (Cognitive, NPath, Cyclomatic, NCSS, GodClass), time range parsing,
     status aggregation matrix, safe path traversal
3. review_bot.adapters.git:
   - fixed SHA and safe path enforcement
   - resolve_merge_base corner cases (identical SHA non-existent commit, criss-cross merge, orphan commits)
   - read_tree_blobs filtering (filtering symlinks, submodules, unsafe paths)
   - get_rename_map injective mapping enforcement (ambiguous rename detection)
   - collect_code_snapshot binary non-UTF-8 exclusion and inventory hashing
4. review_bot.adapters.pmd:
   - StrictTreeBuilder XXE prohibition (DTD & entity declarations rejected)
   - parse_pmd_xml / parse_cpd_xml symlink rejection, 64 MiB size limit, and unsafe source path rejection
   - runner.py environment sanitization (JVM option injection stripping, secret stripping)
   - exit finding concordance full truth matrix
   - AnalysisTimeoutError exception structure
5. review_bot.adapters.sqlite:
   - delivery_id duplicate integrity constraint
   - request coalescing and generation fencing supersession lifecycle
   - worker lease expiration, claim CAS, and renewal rules
   - connection reset, thread-local reopening, and transaction rollback on error
   - multi-threaded concurrency stress test
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from review_bot.adapters.git import (
    _fixed_sha as git_fixed_sha,
)
from review_bot.adapters.git import (
    _safe_path as git_safe_path,
)
from review_bot.adapters.git import (
    collect_code_snapshot,
    get_rename_map,
    read_tree_blobs,
    resolve_merge_base,
)
from review_bot.adapters.pmd.parsers import (
    _read_and_validate_payload,
    _source_path,
    _StrictTreeBuilder,
    parse_pmd_xml,
)
from review_bot.adapters.pmd.runner import (
    check_exit_finding_concordance,
    sanitize_environment,
)
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.application.classification import classify_pr
from review_bot.application.context import (
    _fixed_sha as ctx_fixed_sha,
)
from review_bot.application.context import (
    _fragment_span,
    _valid_span,
    build_task_input,
    check_input_gate,
)
from review_bot.application.context import (
    _safe_path as ctx_safe_path,
)
from review_bot.application.context import (
    _validate as ctx_validate,
)
from review_bot.application.delta import (
    _METRIC_PATTERNS,
    _aggregate,
    _time_range,
)
from review_bot.application.delta import (
    _safe_path as delta_safe_path,
)
from review_bot.application.review_a import (
    ModelProfile,
    build_model_request,
    run_review_a_pipeline,
    validate_model_response,
)
from review_bot.application.selection import ModeSequence, SelectionRecord, select_modes
from review_bot.application.snapshot import (
    assemble_pr_snapshot,
    assemble_snapshot_bundle,
)
from review_bot.domain.changes import Measurement
from review_bot.domain.envelope_v2 import (
    ChangedFile,
    DeliveryRecord,
    EnvelopeEvidence,
    ExecutionReceipt,
    PrRequestRecord,
    PublicationReceipt,
    RunRecord,
    UnavailableSource,
)
from review_bot.domain.models_v1 import (
    AnalysisDelta,
    AnalysisError,
    Citation,
    Classification,
    CodeFile,
    CodeSnapshot,
    CpdDuplication,
    CpdOccurrence,
    DeltaSignal,
    Evidence,
    ModelResponse,
    ParsedReport,
    PmdViolation,
    PrSnapshot,
    ReviewQuestion,
    Snapshot,
    SourceExclusion,
    TaskInput,
)
from tests.conftest import SyntheticGitRepo

# ==============================================================================
# Helper Factories
# ==============================================================================


def _build_minimal_code_snapshot(sha: str, files: tuple[CodeFile, ...] = ()) -> CodeSnapshot:
    inv_str = ":".join(f"{f.path}:{f.content_hash}" for f in files)
    inv_hash = hashlib.sha256(inv_str.encode("utf-8")).hexdigest()
    return CodeSnapshot(sha=sha, files=files, inventory_hash=inv_hash, complete=True)


def _build_minimal_task_input(
    base_sha: str = "a" * 40,
    head_sha: str = "b" * 40,
    review_mode: str = "general",
) -> TaskInput:
    code_file = CodeFile("src/App.java", "public class App {}\n", hashlib.sha256(b"public class App {}\n").hexdigest(), "source")
    base_code = _build_minimal_code_snapshot(base_sha, (code_file,))
    head_code = _build_minimal_code_snapshot(head_sha, (code_file,))
    branch = "refactor/app" if review_mode == "refactoring" else "main"
    labels = ("리팩토링",) if review_mode == "refactoring" else ()
    pr_snap = PrSnapshot("org/repo", 101, base_sha, head_sha, branch, labels=labels, changed_files=("src/App.java",))
    snap = Snapshot(pr=pr_snap, base=base_code, head=head_code)

    violation = PmdViolation("src/App.java", "src/App.java", 1, 1, 1, 10, "Rule", "rs", 1, "App", "m", "p", "msg")
    signal = DeltaSignal("sig-1", "pmd", "new", (), (violation,), ("src/App.java",), "exact")
    delta = AnalysisDelta(1, base_sha, head_sha, "cond_hash", True, (signal,))
    classification = Classification(review_mode)

    return build_task_input(snap, delta, classification)


# ==============================================================================
# 1. Domain Models Adversarial Hardening
# ==============================================================================


class TestDomainAdversarialCore:
    """Stress-test domain immutability, collections isolation, and serialization."""

    def test_domain_frozen_dataclass_mutations(self) -> None:
        """Verify FrozenInstanceError on attribute assignment and deletion across domain records."""
        instances: list[Any] = [
            Measurement("complexity", 10, 15),
            AnalysisError("syntax", "error msg"),
            PmdViolation("p", "p", 1, 2, 1, 5, "R", "RS", 1, "C", "M", "P", "Msg"),
            CpdOccurrence("p", "p", 1, 10),
            CpdDuplication(10, 100, (), "code"),
            ParsedReport("pmd", "rep.xml", True, "valid"),
            SourceExclusion("src/ignore.java", "test file"),
            DeltaSignal("sig-1", "pmd", "new", (), (), ("p",), "exact"),
            CodeFile("src/A.java", "class A{}", "hash", "source"),
            PrSnapshot("repo", 1, "a" * 40, "b" * 40, "branch"),
            Classification("general"),
            Evidence("ev-1", "code", "head", "b" * 40, "src/A.java", 1, 2, "class A{}"),
            Citation("ev-1", "head", "b" * 40, "src/A.java", 1, 2, "class A{}"),
            ReviewQuestion("sig-1", "Why?", "Reason", ("ev-1",)),
            ChangedFile("src/A.java"),
            UnavailableSource("git", "not found"),
            EnvelopeEvidence("ev-1", "source", "1.0", {}, "content", "digest"),
            PublicationReceipt(123, "succeeded", "fp"),
            DeliveryRecord("del-1", "digest", "2026-10-06T12:00:00Z", "opened", 1, 2, 3),
            PrRequestRecord(1, 2, 3, 1),
            RunRecord("run-1", 1, 2, 3, 1, "fp", "queued", "fetch"),
        ]

        for inst in instances:
            # Attribute mutation must raise FrozenInstanceError
            with pytest.raises(FrozenInstanceError):
                inst.nonexistent_attr = "value"

            # Attribute deletion must raise FrozenInstanceError
            first_field = next(iter(inst.__dataclass_fields__))
            with pytest.raises(FrozenInstanceError):
                delattr(inst, first_field)

    def test_domain_default_factory_isolation(self) -> None:
        """Verify default factories produce distinct mutable containers across instances."""
        delta1 = AnalysisDelta(1, "a" * 40, "b" * 40, "hash", True, ())
        delta2 = AnalysisDelta(1, "a" * 40, "b" * 40, "hash", True, ())
        assert delta1.analysis_status is not delta2.analysis_status
        delta1.analysis_status["pmd"] = "clean"
        assert "pmd" not in delta2.analysis_status

        pr1 = PrSnapshot("repo", 1, "a" * 40, "b" * 40, "main")
        pr2 = PrSnapshot("repo", 1, "a" * 40, "b" * 40, "main")
        assert pr1.rename_map is not pr2.rename_map
        pr1.rename_map["old.java"] = "new.java"
        assert "old.java" not in pr2.rename_map

        receipt1 = ExecutionReceipt(1, "arch", "sel-1", "snap-1", 1, "fp", "completed")
        receipt2 = ExecutionReceipt(1, "arch", "sel-1", "snap-1", 1, "fp", "completed")
        assert receipt1.coverage is not receipt2.coverage
        assert receipt1.model_delivery is not receipt2.model_delivery

    def test_domain_delta_signal_and_analysis_delta_get_accessors(self) -> None:
        """Verify .get() dictionary-like accessor fallback behavior."""
        signal = DeltaSignal("sig-42", "pmd", "improved", (), (), ("A.java",), "exact")
        assert signal.get("signal_id") == "sig-42"
        assert signal.get("tool") == "pmd"
        assert signal.get("nonexistent_field") is None
        assert signal.get("nonexistent_field", "fallback_val") == "fallback_val"

        delta = AnalysisDelta(1, "a" * 40, "b" * 40, "hash", True, (signal,))
        assert delta.get("comparable") is True
        assert delta.get("schema_version") == 1
        assert delta.get("missing", 999) == 999

    def test_domain_parsed_report_finding_counts(self) -> None:
        """Verify finding_count logic across PMD, CPD, and exotic tool keys."""
        v = PmdViolation("p", "p", 1, 2, 1, 5, "R", "RS", 1, "C", "M", "P", "Msg")
        d = CpdDuplication(10, 100, (), "code")

        pmd_rep = ParsedReport(tool="pmd", path="r.xml", exists=True, parse_status="valid", violations=(v, v))
        assert pmd_rep.finding_count == 2

        cpd_rep = ParsedReport(tool="cpd", path="r.xml", exists=True, parse_status="valid", duplications=(d, d, d))
        assert cpd_rep.finding_count == 3

        # Non-pmd tool defaults to duplications
        other_rep = ParsedReport(tool="custom_tool", path="r.xml", exists=True, parse_status="valid", duplications=(d,))
        assert other_rep.finding_count == 1

    def test_domain_analysis_delta_counts_handling(self) -> None:
        """Verify AnalysisDelta.counts categorizes all 6 states and unknown entries."""
        signals = (
            DeltaSignal("s1", "pmd", "existing", (), (), (), "exact"),
            DeltaSignal("s2", "pmd", "new", (), (), (), "exact"),
            DeltaSignal("s3", "pmd", "worsened", (), (), (), "exact"),
            DeltaSignal("s4", "pmd", "improved", (), (), (), "exact"),
            DeltaSignal("s5", "pmd", "resolved", (), (), (), "exact"),
            DeltaSignal("s6", "pmd", "unknown", (), (), (), "exact"),
            DeltaSignal("s7", "pmd", "exotic_state_from_future_version", (), (), (), "exact"),
        )
        delta = AnalysisDelta(1, "a" * 40, "b" * 40, "hash", True, signals)
        counts = delta.counts
        assert counts["existing"] == 1
        assert counts["new"] == 1
        assert counts["worsened"] == 1
        assert counts["improved"] == 1
        assert counts["resolved"] == 1
        # s6 ('unknown') and s7 ('exotic_state...') both tally into 'unknown'
        assert counts["unknown"] == 2

    def test_domain_mode_sequence_and_selection_record(self) -> None:
        """Verify ModeSequence sequence invariants and SelectionRecord dict access."""
        seq = ModeSequence(["architecture", "intent"])
        assert seq == ["architecture", "intent"]
        assert seq == ("architecture", "intent")
        assert seq != ["architecture"]
        assert seq != ("architecture",)
        assert seq != ["different_mode"]
        assert seq != ("different_mode",)
        assert hash(seq) == hash(("architecture", "intent"))

        rec = SelectionRecord(
            selection_id="sel-1",
            snapshot_id="snap-1",
            policy_version="all_available",
            selection_digest="digest-123",
            available_modes=seq,
            selected_modes=seq,
        )
        assert rec["selection_id"] == "sel-1"
        assert rec["policy_version"] == "all_available"
        with pytest.raises(KeyError):
            _ = rec["invalid_key"]
        assert rec.get("selection_id") == "sel-1"
        assert rec.get("invalid_key", "default_val") == "default_val"

    def test_domain_roundtrip_serialization_full_graph(self) -> None:
        """Verify full round-trip serialization of deep domain hierarchy to JSON."""
        task_input = _build_minimal_task_input()
        serialized = json.dumps(asdict(task_input), sort_keys=True)
        deserialized = json.loads(serialized)

        assert deserialized["schema_version"] == 1
        assert deserialized["snapshot"]["pr"]["repository"] == "org/repo"
        assert deserialized["classification"]["review_mode"] == "general"
        assert len(deserialized["signals"]) == 1
        assert len(deserialized["evidence"]) >= 1


# ==============================================================================
# 2. Application Core Adversarial Hardening
# ==============================================================================


class TestApplicationCoreAdversarial:
    """Adversarial testing for classification, selection, and snapshot building."""

    @pytest.mark.parametrize(
        ("branch", "labels", "title", "expected_mode"),
        [
            ("refactor/database", (), "Feat: Add DB", "refactoring"),
            ("REFACTOR/DATABASE", (), "Feat: Add DB", "refactoring"),
            ("refact/cleanup", (), None, "refactoring"),
            ("REFACT/CLEANUP", (), None, "refactoring"),
            ("feature/refactor/nested", (), None, "general"),  # must START with refactor/
            ("refactor-branch", (), None, "general"),  # missing slash
            ("  refactor/trimmed  ", (), None, "general"),  # untrimmed whitespace
            ("feature/pay", ("리팩토링",), "Fix bugs", "refactoring"),
            ("feature/pay", ("리펙토링",), "Fix bugs", "general"),  # wrong vowel
            ("feature/pay", ("[리팩토링]",), "Fix bugs", "general"),  # bracketed
            ("feature/pay", (" 리팩토링 ",), "Fix bugs", "general"),  # untrimmed
            ("feature/pay", (), "refactor/payment", "general"),  # title strictly ignored!
            ("feature/pay", (), "리팩토링 대작전", "general"),  # title strictly ignored!
            ("", (), "", "general"),
            (None, (), "", "general"),  # type coercion
            (12345, (), "", "general"),  # type coercion
        ],
    )
    def test_classify_pr_adversarial_inputs(
        self, branch: Any, labels: tuple[str, ...], title: str | None, expected_mode: str
    ) -> None:
        res = classify_pr(source_branch=branch, labels=labels, title=title)
        assert res.review_mode == expected_mode

    def test_select_modes_adversarial_validation(self) -> None:
        """Verify parameter guards on mode selection."""
        with pytest.raises(ValueError, match="snapshot_id must be a non-empty string"):
            select_modes("", ["architecture"])

        with pytest.raises(ValueError, match="snapshot_id must be a non-empty string"):
            select_modes(None, ["architecture"])  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="available_modes must not be empty"):
            select_modes("snap-1", [])

        with pytest.raises(ValueError, match="Unregistered review mode requested: 'unknown_mode'"):
            select_modes("snap-1", ["unknown_mode"])

        with pytest.raises(ValueError, match="Unregistered review mode requested: 'b'"):
            select_modes("snap-1", ["architecture", "b"])

        assert select_modes("snap-1", ["intent"]).selected_modes == ("intent",)
        assert select_modes("snap-1", ["architecture", "intent"]).selected_modes == ("architecture", "intent")

        with pytest.raises(ValueError, match="Unsupported selector policy: 'first_available'"):
            select_modes("snap-1", ["architecture"], policy="first_available")

        # Deduplication check
        res = select_modes("snap-1", ["architecture", "architecture"])
        assert res.available_modes == ("architecture", "architecture")
        assert res.selected_modes == ("architecture",)
        assert res.selection_status == "completed"

    def test_snapshot_assembly_adversarial_paths(self) -> None:
        """Verify PR and ReviewSnapshot handling of non-ASCII text and degraded merge-bases."""
        # 1. ChangedFile mixed with strings
        pr_snap = assemble_pr_snapshot(
            repository="org/repo",
            number=42,
            base_sha="a" * 40,
            head_sha="b" * 40,
            source_branch="refactor/api",
            labels=["리팩토링"],
            changed_files=[ChangedFile("src/A.java"), "src/B.java"],
            rename_map={"src/Old.java": "src/A.java"},
        )
        assert pr_snap.changed_files == ("src/A.java", "src/B.java")
        assert pr_snap.rename_map == {"src/Old.java": "src/A.java"}

        # 2. Korean unicode in title & body determinism
        mock_reader = MagicMock()
        mock_merge = MagicMock(is_comparable=False, comparison_base_sha=None, comparison_status="unavailable")
        mock_reader.resolve_merge_base.return_value = mock_merge

        review_snap, pr_s, _classif, sel = assemble_snapshot_bundle(
            repo_reader=mock_reader,
            repository="org/repo",
            pr_number=99,
            repository_id=10,
            installation_id=20,
            target_base_sha="a" * 40,
            head_sha="b" * 40,
            source_branch="feature/test",
            title="한국어 PR 제목",
            body="이것은 마크다운 본문입니다. Special symbols: <>&'\"",
        )
        assert review_snap.comparison_status == "incomparable"
        assert review_snap.comparison_base_sha is None
        assert len(review_snap.unavailable_sources) == 1
        assert review_snap.unavailable_sources[0].source == "merge_base"
        assert "unavailable" in review_snap.unavailable_sources[0].reason
        assert pr_s.base_sha == ""
        assert pr_s.changed_files == ()
        assert pr_s.rename_map == {}
        mock_reader.get_changed_files.assert_not_called()
        mock_reader.get_rename_map.assert_not_called()
        assert sel.selected_modes == ("architecture",)


# ==============================================================================
# 3. Application Context Assembly & Gate Adversarial Hardening
# ==============================================================================


class TestApplicationContextGateAdversarial:
    """Stress-test SHA validation, path safety, input gate, and contract checking."""

    def test_context_fixed_sha_adversarial_inputs(self) -> None:
        """Verify strict regex rejection on non-canonical SHAs."""
        valid_40 = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
        valid_64 = "a" * 64
        assert ctx_fixed_sha(valid_40) == valid_40
        assert ctx_fixed_sha(valid_40.upper()) == valid_40
        assert ctx_fixed_sha(valid_64) == valid_64

        invalid_shas = [
            "a" * 39,  # too short
            "a" * 41,  # invalid len
            "a" * 63,  # too short
            "a" * 65,  # too long
            "g" * 40,  # non-hex
            "HEAD",  # ref
            "main",  # ref
            "refs/heads/master",  # ref
            "a" * 39 + "; rm -rf /",  # command injection
            "../" + "a" * 37,  # path traversal
            "",  # empty
            12345,  # wrong type
        ]
        for bad in invalid_shas:
            with pytest.raises(ValueError, match="full fixed Git SHA is required"):
                ctx_fixed_sha(bad)  # type: ignore[arg-type]

    def test_context_safe_path_adversarial_inputs(self) -> None:
        """Verify path traversal, control chars, and OS path separator defenses."""
        assert ctx_safe_path("src/main/java/App.java") is True
        assert ctx_safe_path("README.md") is True

        unsafe_paths = [
            "/etc/passwd",  # leading slash
            "src\\App.java",  # windows backslash
            "src/../App.java",  # traversal
            "src/./App.java",  # single dot segment
            "src//App.java",  # double slash empty segment
            "src/\x00App.java",  # null byte
            "src/\x1fApp.java",  # control character
            "src/\x7fApp.java",  # DEL character
            "",  # empty string
            None,  # wrong type
            123,  # wrong type
        ]
        for bad in unsafe_paths:
            assert ctx_safe_path(bad) is False  # type: ignore[arg-type]

    def test_context_valid_span_and_fragment_span(self) -> None:
        """Verify line span bounds and unique code fragment locator."""
        file = CodeFile("A.java", "line 1\nline 2\nline 3\nline 4\n", "hash", "source")
        assert _valid_span(file, 1, 4) is True
        assert _valid_span(file, 2, 3) is True
        assert _valid_span(file, 0, 4) is False  # 0 not valid line
        assert _valid_span(file, 1, 5) is False  # exceeds line count 4
        assert _valid_span(file, 3, 2) is False  # inverted
        assert _valid_span(None, 1, 2) is False

        # Fragment span detection
        assert _fragment_span(file, "line 2\nline 3") == (2, 3)
        assert _fragment_span(file, "line 1") == (1, 1)
        assert _fragment_span(file, "not in content") is None

        # Ambiguous fragment occurring multiple times
        file_dup = CodeFile("B.java", "repeat\nmiddle\nrepeat\n", "hash", "source")
        assert _fragment_span(file_dup, "repeat") is None

    def test_context_check_input_gate_adversarial_limits(self) -> None:
        """Verify Full-or-Not-Run Input Gate on incomplete code, errors, and limits."""
        f = CodeFile("A.java", "x" * 1000, "hash", "source")
        valid_base = _build_minimal_code_snapshot("a" * 40, (f,))
        valid_head = _build_minimal_code_snapshot("b" * 40, (f,))
        pr = PrSnapshot("repo", 1, "a" * 40, "b" * 40, "main")
        snap = Snapshot(pr=pr, base=valid_base, head=valid_head)

        # 1. Clean passes
        ok, status, reason = check_input_gate(snap)
        assert ok is True
        assert status == "can_proceed"

        # 2. Incomplete snapshot
        inc_snap = Snapshot(pr=pr, base=replace(valid_base, complete=False), head=valid_head)
        ok, status, reason = check_input_gate(inc_snap)
        assert ok is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

        # 3. Snapshot with errors
        err_snap = Snapshot(pr=pr, base=valid_base, head=replace(valid_head, errors=("file read error",)))
        ok, status, reason = check_input_gate(err_snap)
        assert ok is False
        assert status == "not_run"
        assert reason == "incomplete_code_input"

        # 4. Byte limit exceeded
        ok, status, reason = check_input_gate(snap, limits={"max_input_bytes": 1500})  # total is 2000
        assert ok is False
        assert status == "not_run"
        assert reason == "input_limit_exceeded"

        # 5. Token limit exceeded
        ok, status, reason = check_input_gate(snap, limits={"max_input_tokens": 500, "token_count": 600})
        assert ok is False
        assert status == "not_run"
        assert reason == "input_limit_exceeded"

    def test_context_validate_adversarial_contract_violations(self) -> None:
        """Verify strict contract validation in context assembly."""
        task = _build_minimal_task_input()

        # 1. Mismatched SHAs
        mismatched_delta = replace(task.delta, base_sha="f" * 40)
        with pytest.raises(ValueError, match="Mismatched base fixed versions"):
            ctx_validate(task.snapshot, mismatched_delta, task.classification)

        # 2. Invalid inventory hash
        bad_inv = replace(task.snapshot.base, inventory_hash="not-a-valid-sha256")
        bad_snap = replace(task.snapshot, base=bad_inv)
        with pytest.raises(ValueError, match="Invalid base inventory hash"):
            ctx_validate(bad_snap, task.delta, task.classification)

        # 3. Content hash mismatch
        tampered_file = CodeFile("src/App.java", "TAMPERED", hashlib.sha256(b"ORIGINAL").hexdigest(), "source")
        tampered_base = _build_minimal_code_snapshot(task.snapshot.base.sha, (tampered_file,))
        tampered_snap = replace(task.snapshot, base=tampered_base)
        with pytest.raises(ValueError, match="Invalid base source content hash"):
            ctx_validate(tampered_snap, task.delta, task.classification)

        # 4. Non-injective rename map
        bad_pr = replace(task.snapshot.pr, rename_map={"A.java": "Target.java", "B.java": "Target.java"})
        with pytest.raises(ValueError, match="Ambiguous rename map"):
            ctx_validate(replace(task.snapshot, pr=bad_pr), task.delta, task.classification)

        # 5. Classification mismatch
        mismatched_class = Classification("refactoring")  # PR is on 'main' branch without label
        with pytest.raises(ValueError, match="Classification does not match the supplied branch and labels"):
            ctx_validate(task.snapshot, task.delta, mismatched_class)

        # 6. Tool mismatch for findings
        cpd_dup = CpdDuplication(10, 100, (), "frag")
        bad_signal = DeltaSignal("sig-err", "pmd", "new", (), (cpd_dup,), ("src/App.java",), "exact")
        bad_delta = replace(task.delta, signals=(bad_signal,))
        with pytest.raises(ValueError, match="Signal findings do not match the analysis tool"):
            ctx_validate(task.snapshot, bad_delta, task.classification)


# ==============================================================================
# 4. Review A Pipeline & Model Validation Adversarial Hardening
# ==============================================================================


class TestReviewAPipelineAdversarial:
    """Stress-test prompt generation, schema constraints, echo checks, and citation verification."""

    def test_model_request_builder_and_system_prompts(self) -> None:
        """Verify review-a/v1 prompt generation across general and refactoring postures."""
        task_general = _build_minimal_task_input(review_mode="general")
        req_general = build_model_request(task_general)
        assert "Version: review-a/v1" in req_general.system_prompt
        assert "General PR: review new and worsened selected signals" in req_general.system_prompt

        task_refactor = _build_minimal_task_input(review_mode="refactoring")
        # Align classification and branch to avoid validation error
        task_refactor = replace(
            task_refactor,
            snapshot=replace(task_refactor.snapshot, pr=replace(task_refactor.snapshot.pr, source_branch="refactor/cleanup")),
        )
        req_refactor = build_model_request(task_refactor)
        assert "Refactoring PR: assess verified improvements/resolutions" in req_refactor.system_prompt

        # Schema constraints check
        schema = req_general.response_schema
        assert schema["properties"]["base_sha"]["const"] == task_general.snapshot.pr.base_sha
        assert schema["properties"]["findings"]["maxItems"] == 20
        assert schema["properties"]["questions"]["maxItems"] == 20

    def test_model_response_json_rejection_rules(self) -> None:
        """Verify rejection of duplicate JSON keys, non-finite floats, and extra top-level fields."""
        base_sha = "a" * 40
        head_sha = "b" * 40

        # 1. Duplicate keys
        dup_json = f'{{"base_sha": "{base_sha}", "base_sha": "{base_sha}", "head_sha": "{head_sha}", "review_mode": "general", "summary": "ok", "findings": [], "questions": []}}'
        errs = validate_model_response(dup_json, base_sha, head_sha, "general", {})
        assert any("Duplicate key in JSON" in e for e in errs)

        # 2. Non-finite numbers (NaN)
        nan_json = f'{{"base_sha": "{base_sha}", "head_sha": "{head_sha}", "review_mode": "general", "summary": "ok", "findings": [], "questions": [], "score": NaN}}'
        errs = validate_model_response(nan_json, base_sha, head_sha, "general", {})
        assert any("Non-finite numeric constant" in e or "Malformed response JSON" in e for e in errs)

        # 3. Extra top-level fields
        extra_json = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": "general",
            "summary": "Everything looks good.",
            "findings": [],
            "questions": [],
            "confidence_score": 0.99,  # disallowed extra field
        }
        errs = validate_model_response(extra_json, base_sha, head_sha, "general", {})
        assert any("Unexpected top-level fields" in e for e in errs)

    def test_model_response_echo_verification_failures(self) -> None:
        """Verify strict rejection on mismatched base_sha, head_sha, or review_mode."""
        payload = {
            "base_sha": "f" * 40,
            "head_sha": "e" * 40,
            "review_mode": "refactoring",
            "summary": "Valid summary text.",
            "findings": [],
            "questions": [],
        }
        errs = validate_model_response(payload, "a" * 40, "b" * 40, "general", {})
        assert any("Mismatched base_sha" in e for e in errs)
        assert any("Mismatched head_sha" in e for e in errs)
        assert any("Mismatched review_mode" in e for e in errs)

    def test_model_response_schema_violations(self) -> None:
        """Verify limits on summary length (>4000), empty summary, and finding/question bounds (>20)."""
        base_sha = "a" * 40
        head_sha = "b" * 40

        # Empty summary
        errs = validate_model_response(
            {"base_sha": base_sha, "head_sha": head_sha, "review_mode": "general", "summary": "", "findings": [], "questions": []},
            base_sha, head_sha, "general", {},
        )
        assert any("Summary must be between 1 and 4000 characters" in e for e in errs)

        # Summary too long (> 4000)
        errs = validate_model_response(
            {"base_sha": base_sha, "head_sha": head_sha, "review_mode": "general", "summary": "x" * 4001, "findings": [], "questions": []},
            base_sha, head_sha, "general", {},
        )
        assert any("Summary must be between 1 and 4000 characters" in e for e in errs)

        # > 20 findings
        errs = validate_model_response(
            {"base_sha": base_sha, "head_sha": head_sha, "review_mode": "general", "summary": "ok", "findings": [{}] * 21, "questions": []},
            base_sha, head_sha, "general", {},
        )
        assert any("Findings must be a list with at most 20 items" in e for e in errs)

        # > 20 questions
        errs = validate_model_response(
            {"base_sha": base_sha, "head_sha": head_sha, "review_mode": "general", "summary": "ok", "findings": [], "questions": [{}] * 21},
            base_sha, head_sha, "general", {},
        )
        assert any("Questions must be a list with at most 20 items" in e for e in errs)

    def test_citation_validation_adversarial_vectors(self) -> None:
        """Verify line range bounds, non-verbatim quotes, path/sha mismatch, and evidence bounds."""
        base_sha = "a" * 40
        head_sha = "b" * 40
        ev = Evidence(
            evidence_id="code-1",
            kind="code",
            version="head",
            sha=head_sha,
            path="src/App.java",
            begin_line=10,
            end_line=20,
            content="int a = 1;\nint b = 2;\n",
        )
        known_ev = {"code-1": ev}

        def make_finding_with_citation(cit: dict[str, Any]) -> dict[str, Any]:
            return {
                "base_sha": base_sha,
                "head_sha": head_sha,
                "review_mode": "general",
                "summary": "Valid summary.",
                "findings": [
                    {
                        "signal_id": "sig-1",
                        "judgement": "concern",
                        "change": "new",
                        "basis": "analysis_signal",
                        "title": "Issue",
                        "reason": "Reason",
                        "recommendation": "Fix",
                        "evidence_ids": ["code-1"],
                        "citations": [cit],
                    }
                ],
                "questions": [],
            }

        # 1. Inverted lines (begin > end)
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "begin_line": 15, "end_line": 12, "quote": "int a = 1;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Inverted citation line range" in e for e in errs)

        # 2. Boolean line numbers (bool is subclass of int in Python)
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "begin_line": True, "end_line": 15, "quote": "int a = 1;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Citation begin_line must be a positive integer" in e for e in errs)

        # 3. Unknown evidence_id
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "unknown-ev", "begin_line": 10, "end_line": 15, "quote": "quote"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Unknown evidence_id referenced" in e for e in errs)

        # 4. Non-verbatim quote
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "begin_line": 10, "end_line": 15, "quote": "int a = 999;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("not found verbatim in evidence" in e for e in errs)

        # 5. Path mismatch
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "path": "src/Other.java", "begin_line": 10, "end_line": 15, "quote": "int a = 1;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Citation path 'src/Other.java' does not match" in e for e in errs)

        # 6. Version mismatch
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "version": "base", "begin_line": 10, "end_line": 15, "quote": "int a = 1;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Citation version 'base' does not match" in e for e in errs)

        # 7. Line numbers exceeding Evidence line bounds (10 to 20)
        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "begin_line": 5, "end_line": 15, "quote": "int a = 1;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Citation begin_line 5 is before evidence code-1 begin_line 10" in e for e in errs)

        errs = validate_model_response(
            make_finding_with_citation({"evidence_id": "code-1", "begin_line": 12, "end_line": 25, "quote": "int a = 1;"}),
            base_sha, head_sha, "general", known_ev,
        )
        assert any("Citation end_line 25 exceeds evidence code-1 end_line 20" in e for e in errs)

    def test_cpd_structural_rule_defense(self) -> None:
        """Verify Feature 18: Model cannot issue architecture concern based solely on CPD count."""
        base_sha = "a" * 40
        head_sha = "b" * 40
        finding_cpd = {
            "signal_id": "cpd-duplication-1",
            "judgement": "concern",
            "change": "new",
            "basis": "analysis_signal",
            "title": "Duplicate code concern",
            "reason": "Too much duplication",
            "recommendation": "Extract common module",
            "evidence_ids": [],
            "citations": [],
        }
        resp = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": "general",
            "summary": "Summary text.",
            "findings": [finding_cpd],
            "questions": [],
        }
        errs = validate_model_response(resp, base_sha, head_sha, "general", {})
        assert any("CPD count alone cannot justify an architecture concern" in e for e in errs)

    def test_review_a_pipeline_execution_error_paths(self) -> None:
        """Verify pipeline graceful failure states (unconfigured, timeouts, over limits)."""
        task = _build_minimal_task_input()

        # 1. Unconfigured model client raises PermanentModelError -> status='failed', reason='model_error'
        res = run_review_a_pipeline(task, model_client=None, profile=ModelProfile("mock", "m1"))
        assert res.status == "failed"
        assert res.reason == "model_error"

        # 2. Timeout error with max_attempts
        timeout_client = MagicMock()
        timeout_client.complete.side_effect = TimeoutError("Timed out")
        res = run_review_a_pipeline(task, model_client=timeout_client, profile=ModelProfile("mock", "m1", max_attempts=1))
        assert res.status == "failed"
        assert res.reason == "model_timeout"

        # 3. Output byte limit exceeded
        giant_client = MagicMock()
        giant_client.complete.return_value = ModelResponse(text="x" * 40000)
        res = run_review_a_pipeline(task, model_client=giant_client, profile=ModelProfile("mock", "m1", max_output_bytes=1000))
        assert res.status == "failed"
        assert res.reason == "invalid_model_response"
        assert any("exceeds output byte limit" in e for e in res.validation_errors)

        # 4. Input byte limit exceeded
        res = run_review_a_pipeline(task, model_client=giant_client, profile=ModelProfile("mock", "m1", max_input_bytes=10))
        assert res.status == "not_run"
        assert res.reason == "input_limit_exceeded"


# ==============================================================================
# 5. Delta Engine Adversarial Hardening
# ==============================================================================


class TestDeltaEngineAdversarial:
    """Stress-test metric regex parsing, path checking, time ranges, and aggregate status."""

    def test_delta_safe_path_and_time_range(self) -> None:
        """Verify path traversal defenses and ISO datetime verification in delta engine."""
        assert delta_safe_path("src/main/java/Clean.java") is True
        assert delta_safe_path("src/../Clean.java") is False
        assert delta_safe_path("/abs/Clean.java") is False
        assert delta_safe_path("src\\Clean.java") is False
        assert delta_safe_path(r"src:\Clean.java") is False
        assert delta_safe_path("src\0Clean.java") is False

        # Time ranges
        t1 = "2026-10-06T12:00:00+00:00"
        t2 = "2026-10-06T12:05:00+00:00"
        assert _time_range(t1, t2) is not None

        # Reversed timestamps
        assert _time_range(t2, t1) is None
        # Naive timestamps (missing timezone)
        assert _time_range("2026-10-06T12:00:00", "2026-10-06T12:05:00") is None
        # Malformed timestamp
        assert _time_range("not-a-date", t2) is None

    def test_delta_metric_patterns_extraction(self) -> None:
        """Verify PMD metric regex parsers on varied rule messages."""
        cog_msg = "The method 'calculate' has a cognitive complexity of 24, current threshold is 15."
        m = _METRIC_PATTERNS["CognitiveComplexity"].search(cog_msg)
        assert m is not None
        assert m.group("cognitive_complexity") == "24"

        npath_msg = "The method 'processOrder' has an NPath complexity of 512, current threshold is 200."
        m = _METRIC_PATTERNS["NPathComplexity"].search(npath_msg)
        assert m is not None
        assert m.group("npath_complexity") == "512"

        cyclomatic_msg = "The class 'PaymentManager' has a total cyclomatic complexity of 45 (highest 12)."
        m = _METRIC_PATTERNS["CyclomaticComplexity"].search(cyclomatic_msg)
        assert m is not None
        assert m.group("cyclomatic_complexity") == "45"
        assert m.group("highest_cyclomatic_complexity") == "12"

        ncss_msg = "The class 'BigService' has a NCSS line count of 1200 (Highest = 350)."
        m = _METRIC_PATTERNS["NcssCount"].search(ncss_msg)
        assert m is not None
        assert m.group("ncss") == "1200"
        assert m.group("highest_ncss") == "350"

        god_msg = "Possible God Class (WMC=55, ATFD=15, TCC=20.5%)"
        m = _METRIC_PATTERNS["GodClass"].search(god_msg)
        assert m is not None
        assert m.group("wmc") == "55"
        assert m.group("atfd") == "15"
        assert m.group("tcc") == "20.5%"

    def test_delta_aggregate_status_combinations(self) -> None:
        """Verify aggregate status resolution logic across tool statuses."""
        mock_run = MagicMock(errors=(), source_state="complete")

        assert _aggregate(mock_run, ("clean", "clean")) == "clean"
        assert _aggregate(mock_run, ("findings", "clean")) == "findings"
        assert _aggregate(mock_run, ("clean", "findings")) == "findings"
        assert _aggregate(mock_run, ("incomplete", "clean")) == "incomplete"
        assert _aggregate(mock_run, ("missing_report", "clean")) == "missing_report"
        assert _aggregate(mock_run, ("failed", "clean")) == "failed"

        mock_err_run = MagicMock(errors=("some_error",), source_state="complete")
        assert _aggregate(mock_err_run, ("clean", "clean")) == "failed"

        mock_no_sources = MagicMock(errors=(), source_state="no_java_sources")
        assert _aggregate(mock_no_sources, ("clean", "clean")) == "no_sources"


# ==============================================================================
# 6. Adapters Git Adversarial Hardening
# ==============================================================================


class TestGitAdapterAdversarialCore:
    """Stress-test git adapter corner cases, symlink filtering, and rename map injective check."""

    def test_git_fixed_sha_and_safe_path_adversarial(self) -> None:
        """Verify adapter SHA and path validation."""
        sha = "1" * 40
        assert git_fixed_sha(sha) == sha

        with pytest.raises(ValueError, match="Invalid full Git SHA"):
            git_fixed_sha("HEAD~1")
        with pytest.raises(ValueError, match="Invalid full Git SHA"):
            git_fixed_sha("--options")

        assert git_safe_path("src/App.java") is True
        assert git_safe_path("../App.java") is False
        assert git_safe_path("/App.java") is False
        assert git_safe_path("src/\x00App.java") is False

    def test_git_resolve_merge_base_identical_missing_commit(self, synthetic_repo: SyntheticGitRepo) -> None:
        """When target base and head SHA are identical but commit does not exist, return unavailable."""
        fake_sha = "f" * 40
        res = resolve_merge_base(synthetic_repo.repo_dir, fake_sha, fake_sha)
        assert res.comparison_status == "unavailable"
        assert res.comparison_base_sha is None

    def test_git_read_tree_blobs_filters_symlinks(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Verify read_tree_blobs reads regular files and completely ignores symlinks."""
        synthetic_repo.write_file("regular.txt", "hello regular")
        # Create a symlink in the git repo
        link_target = synthetic_repo.repo_dir / "link.txt"
        try:
            link_target.symlink_to("regular.txt")
            synthetic_repo.commit("Add regular and symlink")
            head_sha = synthetic_repo.get_head_sha()

            blobs = read_tree_blobs(synthetic_repo.repo_dir, head_sha)
            assert "regular.txt" in blobs
            assert blobs["regular.txt"] == b"hello regular"
            assert "link.txt" not in blobs  # symlink must be ignored
        finally:
            if link_target.is_symlink():
                link_target.unlink()

    def test_git_get_rename_map_injective_enforcement(self) -> None:
        """Verify ambiguous non-injective rename maps raise ValueError."""
        # Mock get_changed_files returning two distinct files renamed to the same destination
        with patch("review_bot.adapters.git.get_changed_files") as mock_changed:
            mock_changed.return_value = (
                ChangedFile(path="dest.java", previous_path="src1.java", change_type="renamed"),
                ChangedFile(path="dest.java", previous_path="src2.java", change_type="renamed"),
            )
            with pytest.raises(ValueError, match="Ambiguous rename map"):
                get_rename_map("/dummy/path", "a" * 40, "b" * 40)

    def test_git_collect_code_snapshot_non_utf8_binary(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Verify collect_code_snapshot excludes non-UTF-8 binary blobs and marks snapshot incomplete."""
        synthetic_repo.write_file("src/Main.java", "public class Main {}")
        synthetic_repo.write_file("data.bin", b"\x80\xff\xfe\x00\x01\x02\x03")
        head_sha = synthetic_repo.commit("Add code and binary")

        snap = collect_code_snapshot(synthetic_repo.repo_dir, head_sha)
        assert snap.complete is False
        assert any(e.path == "data.bin" and e.reason == "unsupported_encoding" for e in snap.exclusions)
        assert any("unsupported encoding" in err for err in snap.errors)
        # Valid java file is still captured
        assert any(f.path == "src/Main.java" for f in snap.files)


# ==============================================================================
# 7. Adapters PMD & CPD Adversarial Hardening
# ==============================================================================


class TestPmdAdapterAdversarialCore:
    """Stress-test XML parser XXE immunity, payload size bounds, and runner environment sanitation."""

    def test_strict_tree_builder_xxe_prohibition(self) -> None:
        """Verify _StrictTreeBuilder rejects DTD / DOCTYPE / entities immediately."""
        builder = _StrictTreeBuilder()
        with pytest.raises(ValueError, match="DTD and entity declarations are not allowed"):
            builder.doctype("pmd", None, None)

    def test_parse_pmd_xml_rejects_symlink(self, tmp_path: Path) -> None:
        """Verify parse_pmd_xml refuses to inspect symlinked report files."""
        real_xml = tmp_path / "real.xml"
        real_xml.write_text("<pmd></pmd>", encoding="utf-8")
        symlink_xml = tmp_path / "symlink.xml"
        symlink_xml.symlink_to(real_xml)

        with pytest.raises(ValueError, match="Report must not be a symbolic link"):
            parse_pmd_xml(symlink_xml)

    def test_parse_pmd_xml_rejects_oversized_payload(self) -> None:
        """Verify 64 MiB payload limit enforcement."""
        oversized = b"<pmd>" + b" " * (64 * 1024 * 1024 + 1) + b"</pmd>"
        with pytest.raises(ValueError, match="Report exceeds the strict parser size limit"):
            _read_and_validate_payload(oversized)

    def test_source_path_traversal_rejections(self, tmp_path: Path) -> None:
        """Verify _source_path rejects directory traversal and absolute path injection."""
        source_root = tmp_path / "src"
        source_root.mkdir()

        # Unsafe characters
        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            _source_path("../../etc/passwd", source_root)

        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            _source_path("src\\Windows\\System32", source_root)

        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            _source_path("C:/autoexec.bat", source_root)

        with pytest.raises(ValueError, match="Report references an unsafe source path"):
            _source_path("src/\x00App.java", source_root)

        # Outside source root
        outside_file = tmp_path / "other" / "File.java"
        outside_file.parent.mkdir()
        outside_file.write_text("class File {}", encoding="utf-8")
        with pytest.raises(ValueError, match="Report references code outside its revision snapshot"):
            _source_path(str(outside_file), source_root)

    def test_pmd_environment_sanitization_strips_secrets_and_jvm_options(self) -> None:
        """Verify subprocess environment sanitization strips JVM injection vectors and tokens."""
        hazardous_env = {
            "JAVA_TOOL_OPTIONS": "-javaagent:/path/to/evil.jar",
            "_JAVA_OPTIONS": "-Dfile.encoding=evil",
            "JDK_JAVA_OPTIONS": "--add-opens=evil",
            "JAVA_OPTS": "-Xmx1024m",
            "CLASSPATH": "/evil/classpath",
            "GITHUB_TOKEN": "ghp_secrettoken12345",
            "OPENAI_API_KEY": "sk-secretkey12345",
            "ANTHROPIC_API_KEY": "sk-ant-secret12345",
            "MODEL_API_KEY": "secret-model-key",
            "GITHUB_WEBHOOK_SECRET": "webhook-secret",
            "SAFE_VAR": "safe_value",
        }
        with patch.dict(os.environ, hazardous_env, clear=True):
            clean = sanitize_environment(Path("/opt/jdk-21"))
            assert clean.get("JAVA_HOME") == "/opt/jdk-21"
            assert clean.get("SAFE_VAR") == "safe_value"
            assert clean.get("LC_ALL") == "C.UTF-8"
            for bad_key in hazardous_env:
                if bad_key != "SAFE_VAR":
                    assert bad_key not in clean

    @pytest.mark.parametrize(
        ("exit_code", "finding_count", "tool", "expected_status", "expected_err_count"),
        [
            (0, 0, "pmd", "clean", 0),
            (0, 5, "pmd", "failed", 1),  # exit 0 with findings is a mismatch
            (4, 5, "pmd", "findings", 0),  # exit 4 with findings is valid
            (4, 0, "pmd", "failed", 1),  # exit 4 with 0 findings is a mismatch
            (5, 0, "pmd", "incomplete", 1),  # exit 5 is recoverable failure
            (5, 5, "pmd", "incomplete", 1),  # exit 5 is recoverable failure
            (1, 0, "pmd", "failed", 1),  # unhandled exit
            (None, 0, "pmd", "failed", 1),  # None exit
            (137, 0, "pmd", "failed", 1),  # SIGKILL exit
        ],
    )
    def test_exit_finding_concordance_adversarial_matrix(
        self, exit_code: int | None, finding_count: int, tool: str, expected_status: str, expected_err_count: int
    ) -> None:
        status, errors = check_exit_finding_concordance(exit_code, finding_count, tool)
        assert status == expected_status
        assert len(errors) == expected_err_count


# ==============================================================================
# 8. Adapters SQLite WAL JobStore Adversarial Hardening
# ==============================================================================


class TestSqliteJobStoreAdversarialCore:
    """Stress-test SQLite JobStore deduplication, fencing, leases, and concurrency."""

    def test_delivery_id_duplicate_integrity_error(self, tmp_path: Path) -> None:
        """Duplicate webhook delivery recording raises sqlite3.IntegrityError."""
        store = SQLiteJobStore(tmp_path / "jobs.db")
        deliv = DeliveryRecord("del-1", "digest", "2026-10-06T12:00:00Z", "opened", 1, 2, 3)
        store.record_delivery(deliv)
        assert store.is_delivery_processed("del-1") is True

        with pytest.raises(sqlite3.IntegrityError):
            store.record_delivery(deliv)

    def test_coalescing_and_generation_fencing_lifecycle(self, tmp_path: Path) -> None:
        """Verify desired_generation increments, coalesces pending runs, and supersedes stale runs."""
        store = SQLiteJobStore(tmp_path / "jobs.db")
        req1 = PrRequestRecord(10, 100, 1, 1)

        # 1. Enqueue generation 1 -> creates queued run with gen 1
        store.enqueue_request(req1)
        r1 = store.get_pr_request(100, 1)
        assert r1 is not None and r1.desired_generation == 1

        # 2. Worker claims run gen 1
        claimed1 = store.claim_next_run("worker-1", lease_seconds=10.0)
        assert claimed1 is not None
        assert claimed1.generation == 1

        # 3. New webhook arrives while worker-1 is executing -> increments desired_generation to 2
        req2 = PrRequestRecord(10, 100, 1, 1)
        store.enqueue_request(req2)
        r2 = store.get_pr_request(100, 1)
        assert r2 is not None and r2.desired_generation == 2

        # 4. Worker-1 checks generation fencing -> run gen 1 < desired gen 2 -> superseded!
        assert store.check_generation_fencing(claimed1.run_id) is False

        # Verify run record in DB is now superseded
        claimed_again = store.claim_next_run("worker-2", lease_seconds=10.0)
        assert claimed_again is not None
        assert claimed_again.generation == 2  # worker-2 picks up the coalesced gen 2 run

    def test_worker_lease_and_renewal_adversarial(self, tmp_path: Path) -> None:
        """Verify worker lease exclusivity, timeout expiration, and renewal CAS."""
        store = SQLiteJobStore(tmp_path / "jobs.db")
        store.enqueue_request(PrRequestRecord(1, 2, 3, 1))

        # Worker-1 claims with short 0.1s lease
        run = store.claim_next_run("worker-1", lease_seconds=0.1)
        assert run is not None

        # Worker-2 cannot claim while Worker-1 lease is active
        assert store.claim_next_run("worker-2", lease_seconds=1.0) is None

        # Worker-2 cannot renew Worker-1's lease
        assert store.renew_lease(run.run_id, "worker-2", lease_seconds=5.0) is False

        # Worker-1 renews lease
        assert store.renew_lease(run.run_id, "worker-1", lease_seconds=0.1) is True

        # Let lease expire
        time.sleep(0.15)

        # Worker-1 cannot renew expired lease
        assert store.renew_lease(run.run_id, "worker-1", lease_seconds=5.0) is False

        # Worker-2 claims expired run (attempts incremented to 2)
        run2 = store.claim_next_run("worker-2", lease_seconds=5.0)
        assert run2 is not None
        assert run2.run_id == run.run_id
        assert run2.attempts == 2

    def test_connection_reset_and_reopen(self, tmp_path: Path) -> None:
        """Verify store.close() cleanly releases and automatically reconnects on demand."""
        store = SQLiteJobStore(tmp_path / "jobs.db")
        store.enqueue_request(PrRequestRecord(1, 2, 3, 1))

        store.close()
        # Next operation transparently establishes fresh connection
        req = store.get_pr_request(2, 3)
        assert req is not None
        assert req.desired_generation == 1

    def test_transaction_rollback_on_error(self, tmp_path: Path) -> None:
        """Verify transaction rollback on unhandled exception inside transaction block."""
        store = SQLiteJobStore(tmp_path / "jobs.db")

        with (
            pytest.raises(RuntimeError, match="Simulated crash"),
            store.transaction(immediate=True) as conn,
        ):
            conn.execute(
                "INSERT INTO deliveries (delivery_id, payload_digest, received_at) VALUES ('del-x', 'dig', 'now');"
            )
            raise RuntimeError("Simulated crash")

        # Rollback ensured del-x was not committed
        assert store.is_delivery_processed("del-x") is False

    def test_concurrent_threads_stress(self, tmp_path: Path) -> None:
        """Verify SQLite WAL handles 10 concurrent threads without deadlocks or corruption."""
        store = SQLiteJobStore(tmp_path / "jobs.db", timeout=10.0)

        # Seed 20 PR requests
        for i in range(20):
            store.enqueue_request(PrRequestRecord(1, 100, i + 1, 1))

        def worker_task(thread_id: int) -> int:
            worker_id = f"worker-thread-{thread_id}"
            claimed_count = 0
            for _ in range(5):
                run = store.claim_next_run(worker_id, lease_seconds=2.0)
                if run:
                    claimed_count += 1
                    store.renew_lease(run.run_id, worker_id, lease_seconds=2.0)
                    store.complete_run(run.run_id)
                time.sleep(0.01)
            return claimed_count

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(worker_task, range(10)))

        # All runs were safely processed or accounted for without SQLite locking exceptions
        assert sum(results) > 0
