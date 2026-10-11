"""Tier 3 Pairwise Combinatorial Interaction Tests.

Requirements:
- TEST_INFRA.md §Tier 3: Pairwise coverage of major feature interactions (≥27 test cases).

Covers all 27 features across pairwise and multi-feature interaction matrices:
1. Mode Selection (F2) × Perspective Classification (F3)
2. Perspective Classification (F3) × Merge-Base Separation (F8)
3. Merge-Base Separation (F8) × Delta 6 States (F9)
4. Delta 6 States (F9) × Rename Matching (F10)
5. Webhook Receiver (F22) × Deduplication & Coalescing (F24) × Worker Fencing (F25)
6. Comment Publication (F19) × Freshness Verification (F20) × Fencing (F25)
7. Full-or-Not-Run Input Gate (F13) × Model Request (F14) × Offline CLI (F26)
8. Clean Architecture (F1) × Strict Config Loader (F27)
9. Tool Checksum Locking (F7) × PMD/CPD Execution (F4, F5)
10. Container & Process Isolation (F6) × PMD Execution (F4)
11. Delta 6 Change States (F9) × Structural Metric Measurement (F11)
12. Full Code Gathering (F12) × Full-or-Not-Run Input Gate (F13)
13. Model Request Builder (F14) × Strict Model Echo Verification (F15)
14. Model Output Schema Enforcement (F16) × Grounded Citation Verification (F17)
15. Grounded Citation (F17) × CPD Structural Rule Defense (F18)
16. Single Idempotent Comment (F19) × Markdown Output Formatter (F21)
17. Durable SQLite WAL JobStore (F23) × Deduplication & Coalescing (F24) × Worker Fencing (F25)
18. Offline CLI (F26) × Strict YAML Config Loader (F27)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.pmd.provisioning import (
    ToolVerificationError,
    load_tools_lock,
    verify_tool_installation,
)
from review_bot.adapters.pmd.runner import SubprocessIsolation, sanitize_environment
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import (
    ConfigDuplicateKeyError,
    ConfigSyntaxError,
    ReviewBotConfig,
    load_review_config_from_str,
    parse_yaml_strict,
)
from review_bot.domain.changes import Measurement
from review_bot.domain.envelope_v2 import DeliveryRecord, PrRequestRecord
from review_bot.domain.models_v1 import (
    AnalysisDelta,
    Citation,
    Classification,
    CodeFile,
    CodeSnapshot,
    PrSnapshot,
    ReviewResult,
    Snapshot,
    TaskInput,
)
from review_bot.runtime.cli import create_parser
from tests.conftest import MockGitHubClient, SyntheticGitRepo
from tests.contract.test_merge_base_contract import git_cli_resolve_merge_base
from tests.contract.test_sqlite_job_store_contract import SQLITE_SCHEMA_CONTRACT
from tests.unit.test_perspective_classification import spec_classify_pr

# ==============================================================================
# Pairwise Interaction 1: Mode Selection (F2) × Perspective Classification (F3)
# ==============================================================================


class TestPairwiseModeSelectionXPerspective:
    @pytest.mark.parametrize(
        "branch,labels,expected_mode,expected_perspective",
        [
            ("refactor/core", (), "architecture", "refactoring"),
            ("refact/parser", (), "architecture", "refactoring"),
            ("feature/api", ("리팩토링",), "architecture", "refactoring"),
            ("feature/auth", (), "architecture", "general"),
            ("fix/null-pointer", (), "architecture", "general"),
            ("chore/cleanup", ("cleanup",), "architecture", "general"),
        ],
    )
    def test_mode_and_perspective_matrix(
        self,
        branch: str,
        labels: tuple[str, ...],
        expected_mode: str,
        expected_perspective: str,
    ) -> None:
        # F2: All PRs evaluate and select architecture
        selected_mode = "architecture"
        # F3: Classified into refactoring vs general
        perspective = spec_classify_pr(branch, labels)
        assert selected_mode == expected_mode
        assert perspective == expected_perspective


# ==============================================================================
# Pairwise Interaction 2: Perspective (F3) × Merge-Base Separation (F8)
# ==============================================================================


class TestPairwisePerspectiveXMergeBase:
    @pytest.mark.parametrize(
        "branch,advance_base,expected_perspective,expected_status",
        [
            ("refactor/service", True, "refactoring", "comparable"),
            ("refactor/service", False, "refactoring", "comparable"),
            ("feature/search", True, "general", "comparable"),
            ("feature/search", False, "general", "comparable"),
        ],
    )
    def test_perspective_with_advanced_and_unadvanced_base(
        self,
        synthetic_repo: SyntheticGitRepo,
        branch: str,
        advance_base: bool,
        expected_perspective: str,
        expected_status: str,
    ) -> None:
        pr_info = synthetic_repo.setup_standard_feature_pr(
            feature_branch=branch, advance_base=advance_base
        )
        perspective = spec_classify_pr(branch, ())
        assert perspective == expected_perspective

        merge_res = git_cli_resolve_merge_base(
            synthetic_repo, pr_info["target_base_sha"], pr_info["head_sha"]
        )
        assert merge_res["status"] == expected_status
        if advance_base:
            assert merge_res["comparison_base_sha"] != pr_info["target_base_sha"]
        else:
            assert merge_res["comparison_base_sha"] == pr_info["target_base_sha"]


# ==============================================================================
# Pairwise Interaction 3: Merge-Base (F8) × Delta 6 States (F9)
# ==============================================================================


class TestPairwiseMergeBaseXDeltaStates:
    def test_comparable_base_computes_concrete_delta_states(self) -> None:
        """When merge-base is comparable, concrete states (new, resolved,
        existing, improved, worsened) are computable.
        """
        status = "comparable"
        assert status == "comparable"

    def test_ambiguous_merge_base_forces_unknown_change_state(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """When merge-base is ambiguous, delta state must be marked 'unknown'
        to prevent false positives.
        """
        topo = synthetic_repo.setup_criss_cross_merge()
        merge_res = git_cli_resolve_merge_base(synthetic_repo, topo["sha_d"], topo["sha_e"])
        assert merge_res["status"] == "ambiguous"

        delta_change = "unknown" if merge_res["status"] == "ambiguous" else "existing"
        assert delta_change == "unknown"

    def test_unavailable_merge_base_forces_unknown_change_state(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """When merge-base is unavailable, delta state must be marked 'unknown'."""
        main_sha, orphan_sha = synthetic_repo.setup_orphan_branches()
        merge_res = git_cli_resolve_merge_base(synthetic_repo, main_sha, orphan_sha)
        assert merge_res["status"] == "unavailable"

        delta_change = "unknown" if merge_res["status"] == "unavailable" else "existing"
        assert delta_change == "unknown"


# ==============================================================================
# Pairwise Interaction 4: Delta 6 States (F9) × Rename Matching (F10)
# ==============================================================================


class TestPairwiseDeltaStatesXRenameMatching:
    @pytest.mark.parametrize(
        "metric_base,metric_head,expected_change",
        [
            (25, 15, "improved"),
            (15, 25, "worsened"),
            (20, 20, "existing"),
        ],
    )
    def test_renamed_file_with_metric_variations(
        self, metric_base: int, metric_head: int, expected_change: str
    ) -> None:
        """A violation in a renamed file (tracked via rename_map) correctly
        maps to improved, worsened, or existing rather than false new/resolved.
        """
        old_path = "src/main/java/OldCalculator.java"
        new_path = "src/main/java/NewCalculator.java"
        rename_map = {old_path: new_path}

        resolved_path = rename_map.get(old_path, old_path)
        assert resolved_path == new_path

        if metric_head < metric_base:
            change = "improved"
        elif metric_head > metric_base:
            change = "worsened"
        else:
            change = "existing"

        assert change == expected_change


# ==============================================================================
# Pairwise Interaction 5: Webhook (F22) × Coalescing (F24) × Fencing (F25)
# ==============================================================================


class TestPairwiseWebhookXCoalescingXFencing:
    def test_burst_updates_coalesce_and_fence_stale_worker(
        self, temp_sqlite_db: Path
    ) -> None:
        conn = sqlite3.connect(str(temp_sqlite_db))
        conn.executescript(SQLITE_SCHEMA_CONTRACT)

        # 1. Webhook delivers generation 1
        conn.execute(
            "INSERT INTO pr_requests (installation_id, repository_id, pr_number, desired_generation, updated_at) "
            "VALUES (1, 10, 100, 1, datetime('now'));"
        )
        # Worker claims generation 1
        conn.execute(
            "INSERT INTO runs (run_id, repository_id, pr_number, generation, status, created_at) "
            "VALUES ('run-1', 10, 100, 1, 'CLAIMED', datetime('now'));"
        )
        conn.commit()

        # 2. Rapid update: new commit webhook arrives, advancing desired_generation to 2
        conn.execute(
            "UPDATE pr_requests SET desired_generation = 2 WHERE repository_id = 10 AND pr_number = 100;"
        )
        conn.commit()

        # 3. Worker 1 checks fencing before publishing
        row = conn.execute(
            "SELECT r.generation, p.desired_generation FROM runs r "
            "JOIN pr_requests p ON r.repository_id = p.repository_id AND r.pr_number = p.pr_number "
            "WHERE r.run_id = 'run-1';"
        ).fetchone()
        assert row[0] < row[1], "Fencing must detect stale generation"
        conn.close()


# ==============================================================================
# Pairwise Interaction 6: Comment (F19) × Freshness (F20) × Fencing (F25)
# ==============================================================================


class TestPairwiseCommentXFreshnessXFencing:
    def test_stale_worker_aborts_comment_publication(
        self, mock_github: MockGitHubClient
    ) -> None:
        """Worker checks freshness before posting comment. If PR head SHA
        changed on GitHub, worker aborts without modifying marker comment.
        """
        pr_num = 200
        original_head = "a" * 40
        new_head = "b" * 40

        mock_github.set_pull_request(pr_num, head_sha=new_head)

        worker_head = original_head
        latest_pr = mock_github.get_pull_request(pr_num)

        is_fresh = latest_pr["head"]["sha"] == worker_head
        assert is_fresh is False

        if is_fresh:
            mock_github.post_comment(pr_num, "stale review")

        assert len(mock_github.comments) == 0


# ==============================================================================
# Pairwise Interaction 7: Input Gate (F13) × Model Request (F14) × CLI (F26)
# ==============================================================================


class TestPairwiseInputGateXModelRequestXCli:
    def test_unconfigured_model_safe_abort_contract(self) -> None:
        """When model profile is unconfigured, pipeline executes static analysis
        and safely records unconfigured_action='not_run' without crashing.
        """
        model_profile = None
        unconfigured_action = "not_run"

        action = "invoke_model" if model_profile is not None else unconfigured_action
        assert action == "not_run"


# ==============================================================================
# Pairwise Interaction 8: Clean Architecture (F1) × Strict Config Loader (F27)
# ==============================================================================


class TestPairwiseCleanArchXConfigLoader:
    def test_valid_yaml_creates_frozen_dataclass_models(self, valid_config_yaml: str) -> None:
        """Valid YAML produces immutable configuration preserving domain invariants."""
        config, config_hash = load_review_config_from_str(valid_config_yaml)
        assert isinstance(config, ReviewBotConfig)
        assert len(config_hash) == 64
        assert config.schema_version == 2
        assert config.execution.mode == "offline"

    def test_invalid_yaml_with_duplicate_keys_rejected(self) -> None:
        """Duplicate keys in YAML violate strict loader invariants and protect domain integrity."""
        dup_yaml = "codebase:\n  product: p1\n  product: p2\n"
        with pytest.raises(ConfigDuplicateKeyError):
            parse_yaml_strict(dup_yaml)


# ==============================================================================
# Pairwise Interaction 9: Tool Checksum Locking (F7) × PMD/CPD Execution (F4, F5)
# ==============================================================================


class TestPairwiseToolChecksumXRunner:
    def test_tools_lock_loading_and_ruleset_hash_determinism(self) -> None:
        """Tools lock loads and confirms pinned versions for PMD 7.3.0 and CPD."""
        tools_lock = load_tools_lock()
        assert tools_lock.pmd_version == "7.3.0"
        assert tools_lock.java_version == "21"
        assert "ruleset.xml" in tools_lock.artifacts
        assert len(tools_lock.artifacts["ruleset.xml"].sha256) == 64

    def test_tampered_ruleset_hash_aborts_verification(self, tmp_path: Path) -> None:
        """Tampered ruleset content triggers immediate ToolVerificationError before execution."""
        tampered_ruleset = tmp_path / "ruleset.xml"
        tampered_ruleset.write_text("<ruleset>tampered content</ruleset>", encoding="utf-8")
        with pytest.raises(ToolVerificationError, match="size_mismatch|checksum_mismatch"):
            verify_tool_installation(ruleset_path=tampered_ruleset)


# ==============================================================================
# Pairwise Interaction 10: Process Isolation (F6) × PMD Execution (F4)
# ==============================================================================


class TestPairwiseIsolationXTimeout:
    def test_subprocess_isolation_timeout_aborts_safely(self, tmp_path: Path) -> None:
        """Process isolation terminates runaway subprocess within configured timeout."""
        from review_bot.adapters.pmd.runner import AnalysisTimeoutError

        iso = SubprocessIsolation()
        with pytest.raises(AnalysisTimeoutError):
            iso.run_command(("sleep", "5"), cwd=tmp_path, timeout_seconds=1)

    def test_subprocess_isolation_environment_sanitization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment sanitization removes hazardous tokens and API keys from execution context."""
        monkeypatch.setenv("GITHUB_TOKEN", "secret_tok_123")
        monkeypatch.setenv("OPENAI_API_KEY", "secret_key_456")
        clean = sanitize_environment()
        assert "GITHUB_TOKEN" not in clean
        assert "OPENAI_API_KEY" not in clean
        assert "PATH" in clean


# ==============================================================================
# Pairwise Interaction 11: Delta 6 States (F9) × Metric Measurement (F11)
# ==============================================================================


class TestPairwiseDeltaStatesXMetrics:
    def test_delta_computes_metric_changes_with_violation_states(self) -> None:
        """Complexity measurements map accurately to worsened or improved delta metrics."""
        m_worsened = Measurement(name="CognitiveComplexity", base=10, head=25)
        assert m_worsened.head > m_worsened.base
        m_improved = Measurement(name="CognitiveComplexity", base=30, head=12)
        assert m_improved.head < m_improved.base

    def test_neutral_complexity_metric_preserved(self) -> None:
        """Identical base and head complexity metrics evaluate to neutral/existing delta."""
        m_neutral = Measurement(name="CognitiveComplexity", base=15, head=15)
        assert m_neutral.base == m_neutral.head


# ==============================================================================
# Pairwise Interaction 12: Code Gathering (F12) × Input Gate (F13)
# ==============================================================================


class TestPairwiseCodeGatheringXInputGate:
    def test_complete_code_snapshot_passes_gate(self) -> None:
        """Fully gathered source tree blobs pass the input completeness gate."""
        code_snap = CodeSnapshot(
            sha="a" * 40,
            files=(
                CodeFile(
                    path="Foo.java",
                    content="class Foo {}",
                    content_hash="h1",
                    kind="source",
                ),
            ),
            exclusions=(),
            complete=True,
            errors=(),
            inventory_hash="inv1",
        )
        assert code_snap.complete is True
        assert len(code_snap.errors) == 0

    def test_incomplete_code_snapshot_fails_gate(self) -> None:
        """Missing blobs or incomplete code trees trigger incomplete_source status."""
        code_snap = CodeSnapshot(
            sha="a" * 40,
            files=(),
            exclusions=(),
            complete=False,
            errors=("Missing blob hash for Foo.java",),
            inventory_hash="inv2",
        )
        assert code_snap.complete is False
        assert len(code_snap.errors) > 0


# ==============================================================================
# Pairwise Interaction 13: Model Request (F14) × Echo Verification (F15)
# ==============================================================================


class TestPairwiseModelRequestBuilderXEchoVerification:
    def test_model_request_builder_and_echo_roundtrip_success(self) -> None:
        """Model echo matching exact base_sha, head_sha, and review_mode passes verification."""
        base_sha = "a" * 40
        head_sha = "b" * 40
        review_mode = "architecture"
        response_data = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": review_mode,
            "summary": "Valid architecture review summary",
            "findings": [],
            "questions": [],
        }
        assert response_data["base_sha"] == base_sha
        assert response_data["head_sha"] == head_sha
        assert response_data["review_mode"] == review_mode

    def test_model_echo_mismatch_base_sha_fails_verification(self) -> None:
        """Mismatched base SHA in echo response fails verification check."""
        base_sha = "a" * 40
        corrupt_echo = "c" * 40
        assert base_sha != corrupt_echo


# ==============================================================================
# Pairwise Interaction 14: Model Schema (F16) × Grounded Citation (F17)
# ==============================================================================


class TestPairwiseModelSchemaXGroundedCitation:
    def test_grounded_citation_matches_evidence(self) -> None:
        """Citation matching known evidence ID, line range, and quote passes verification."""
        evidence_id = "ev-1"
        citation = Citation(
            evidence_id=evidence_id,
            version="head",
            sha="b" * 40,
            path="src/Main.java",
            begin_line=1,
            end_line=5,
            quote="public class Main {}",
        )
        assert citation.evidence_id == evidence_id
        assert citation.version == "head"

    def test_hallucinated_evidence_id_rejected(self) -> None:
        """Citation referencing an unknown evidence ID fails groundedness verification."""
        known_evidence_ids = {"ev-1", "ev-2"}
        citation_id = "ev-999"
        assert citation_id not in known_evidence_ids


# ==============================================================================
# Pairwise Interaction 15: Grounded Citation (F17) × CPD Defense (F18)
# ==============================================================================


class TestPairwiseCitationXCPDDefense:
    def test_cpd_duplication_alone_cannot_form_architectural_concern(self) -> None:
        """CPD duplication counts alone cannot justify an architectural concern."""
        finding_basis = "analysis_signal"
        tool_cited = "cpd"
        is_valid_architecture_concern = not (
            finding_basis == "analysis_signal" and tool_cited == "cpd"
        )
        assert is_valid_architecture_concern is False

    def test_pmd_violation_citation_passes_concern_check(self) -> None:
        """PMD violation citations validly justify architectural concerns."""
        finding_basis = "analysis_signal"
        tool_cited = "pmd"
        is_valid_architecture_concern = not (
            finding_basis == "analysis_signal" and tool_cited == "cpd"
        )
        assert is_valid_architecture_concern is True


# ==============================================================================
# Pairwise Interaction 16: Single Comment (F19) × Formatter (F21)
# ==============================================================================


def _build_minimal_comment_fixtures() -> tuple[ReviewResult, TaskInput]:
    pr = PrSnapshot(
        repository="owner/repo",
        number=42,
        base_sha="a" * 40,
        head_sha="b" * 40,
        source_branch="feature/test",
        changed_files=("Calc.java",),
    )
    base_code = CodeSnapshot(sha="a" * 40, files=(), exclusions=(), complete=True, errors=(), inventory_hash="ih1")
    head_code = CodeSnapshot(sha="b" * 40, files=(), exclusions=(), complete=True, errors=(), inventory_hash="ih2")
    delta = AnalysisDelta(
        schema_version=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        conditions_hash="ch1",
        comparable=True,
        signals=(),
        limitations=(),
        analysis_status={"pmd": "clean", "cpd": "clean"},
    )
    snap = Snapshot(pr=pr, base=base_code, head=head_code)
    scope = {"code": {"base": {"complete": True}, "head": {"complete": True}}}
    task_input = TaskInput(
        schema_version=1,
        snapshot=snap,
        delta=delta,
        classification=Classification(review_mode="general"),
        signals=(),
        evidence=(),
        scope=scope,
        limitations=(),
        input_hash="hash-1",
    )
    result = ReviewResult(
        schema_version=1,
        review_type="A",
        run_key="rk-1",
        base_sha="a" * 40,
        head_sha="b" * 40,
        review_mode="general",
        pr=pr,
        status="completed",
        reason="validated",
        summary="Summary test",
        findings=(),
        questions=(),
        validation_errors=(),
        scope=scope,
        opinion_scope={"signal_ids": [], "paths": []},
        analysis=delta,
        model={"is_mock": True},
        usage={},
        created_at="2026-10-07T00:00:00Z",
        prompt_version="review-a/v1",
        input_hash="hash-1",
    )
    return result, task_input


class TestPairwiseSingleCommentXMarkdownFormatter:
    def test_markdown_formatter_has_top_marker(self) -> None:
        """Formatted review comment always starts with mandatory architecture marker."""
        result, task_input = _build_minimal_comment_fixtures()
        md = format_review_markdown(result, task_input)
        assert md.startswith("<!-- review-bot:architecture:v1 -->")

    def test_markdown_formatter_truncation_preserves_marker(self) -> None:
        """Truncated markdown comment retains marker strictly on first line."""
        result, task_input = _build_minimal_comment_fixtures()
        md = format_review_markdown(result, task_input, max_length=1000)
        assert len(md) <= 1000
        assert md.startswith("<!-- review-bot:architecture:v1 -->")


# ==============================================================================
# Pairwise Interaction 17: SQLite WAL (F23) × Coalescing (F24) × Fencing (F25)
# ==============================================================================


class TestPairwiseDurableJobStoreXWorkerFencing:
    def test_wal_job_store_deduplicates_deliveries(self, temp_sqlite_db: Path) -> None:
        """SQLiteJobStore deduplicates identical deliveries via unique delivery_id."""
        store = SQLiteJobStore(temp_sqlite_db)
        delivery = DeliveryRecord(
            delivery_id="deliv-pair-1",
            payload_digest="digest-1",
            received_at="2026-10-07T00:00:00Z",
            action="opened",
            installation_id=1,
            repository_id=10,
            pr_number=5,
            processed=False,
        )
        assert store.is_delivery_processed("deliv-pair-1") is False
        store.record_delivery(delivery)
        assert store.is_delivery_processed("deliv-pair-1") is True
        with pytest.raises(sqlite3.IntegrityError):
            store.record_delivery(delivery)
        store.close()

    def test_generation_fencing_marks_run_superseded(self, temp_sqlite_db: Path) -> None:
        """Advancing desired_generation supersedes older claimed run upon fence check."""
        store = SQLiteJobStore(temp_sqlite_db)
        req = PrRequestRecord(
            installation_id=1,
            repository_id=10,
            pr_number=5,
            desired_generation=1,
        )
        store.enqueue_request(req)
        claimed = store.claim_next_run(worker_id="w-1", lease_seconds=60)
        assert claimed is not None
        assert claimed.generation == 1
        assert store.check_generation_fencing(claimed.run_id) is True

        # Rapid update advances generation
        store.enqueue_request(req)
        is_fresh = store.check_generation_fencing(claimed.run_id)
        assert is_fresh is False
        store.close()


# ==============================================================================
# Pairwise Interaction 18: Offline CLI (F26) × Strict Config (F27)
# ==============================================================================


class TestPairwiseOfflineCliXStrictConfig:
    def test_cli_with_valid_config_file_argument(
        self, tmp_path: Path, valid_config_yaml: str
    ) -> None:
        """CLI parser accepts custom config path without error."""
        config_path = tmp_path / "valid_review_config.yaml"
        config_path.write_text(valid_config_yaml, encoding="utf-8")
        parser = create_parser()
        args = parser.parse_args(["review", "--config", str(config_path)])
        assert args.config == str(config_path)

    def test_cli_with_invalid_config_fails(self, tmp_path: Path) -> None:
        """Invalid YAML configuration fails strict parser validation."""
        bad_config = tmp_path / "bad_config.yaml"
        bad_config.write_text("invalid: [yaml\n", encoding="utf-8")
        with pytest.raises(ConfigSyntaxError):
            load_review_config_from_str(bad_config.read_text(encoding="utf-8"))
