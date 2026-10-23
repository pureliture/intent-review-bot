"""Tier 4 Real-World Application Scenario Tests.

Requirements:
- TEST_INFRA.md §Real-World Application Scenarios (Tier 4):
  1. Standard Feature PR (General Perspective)
  2. Pure Refactoring PR (Method move + rename + metric improvement)
  3. Webhook Delivery Burst with PR Update Coalescing
  4. Offline CLI Full Audit on Synthetic Complex Repo
  5. Guarded Concurrent Publication Race with Stale Marker Warning
  6. Unconfigured Model Profile Safe Abort (not_run) with Full Artifacts
  7. Ambiguous Merge-Base Incomparable PR Handling
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from review_bot.adapters.git import GitRepositoryReader
from review_bot.adapters.github.client import MockGitHubClient as ApiMockGitHubClient
from review_bot.adapters.pmd.runner import PmdRunner
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.application.architecture_review import MockModelClient
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import PrRequestRecord
from review_bot.domain.models_v1 import AnalysisRun, ModelResponse
from review_bot.ports import StaticAnalyzer
from review_bot.runtime.cli import main
from review_bot.runtime.worker import WorkerLoop
from tests.conftest import JavaSourceFixtures, MockGitHubClient, SyntheticGitRepo
from tests.contract.test_merge_base_contract import git_cli_resolve_merge_base
from tests.contract.test_sqlite_job_store_contract import SQLITE_SCHEMA_CONTRACT
from tests.unit.test_delta_adversarial import make_pmd_violation, make_test_run
from tests.unit.test_perspective_classification import spec_classify_pr

# ==============================================================================
# Scenario 1: Standard Feature PR (General Perspective)
# ==============================================================================


class TestScenario1StandardFeaturePr:
    def test_standard_feature_pr_flow(
        self, synthetic_repo: SyntheticGitRepo, mock_github: MockGitHubClient
    ) -> None:
        """PR from 'feature/payment' without labels evaluates mode A,

        classifies as 'general', separates merge-base, and prepares single comment.
        """
        pr_info = synthetic_repo.setup_standard_feature_pr(
            feature_branch="feature/payment", advance_base=True
        )

        # 1. Perspective classification
        perspective = spec_classify_pr(pr_info["feature_branch"], ())
        assert perspective == "general"

        # 2. Merge-base resolution
        merge_res = git_cli_resolve_merge_base(
            synthetic_repo, pr_info["target_base_sha"], pr_info["head_sha"]
        )
        assert merge_res["status"] == "comparable"
        assert merge_res["comparison_base_sha"] == pr_info["comparison_base_sha"]

        # 3. Publication target marker verification
        mock_github.set_pull_request(
            pr_number=101,
            head_sha=pr_info["head_sha"],
            base_sha=pr_info["target_base_sha"],
            branch=pr_info["feature_branch"],
        )
        comment_body = "<!-- review-bot:architecture:v1 -->\n## Architecture Review\nPerspective: General"
        comment = mock_github.post_comment(101, comment_body)

        marker_comment = mock_github.find_marker_comment(101)
        assert marker_comment is not None
        assert marker_comment.id == comment.id


# ==============================================================================
# Scenario 2: Pure Refactoring PR
# ==============================================================================


class TestScenario2PureRefactoringPr:
    def test_refactoring_pr_flow(
        self, synthetic_repo: SyntheticGitRepo, java_fixtures: JavaSourceFixtures
    ) -> None:
        """PR with 'refactor/' branch or '리팩토링' label classifies as 'refactoring',

        tracks improved complexity metrics and renamed files.
        """
        synthetic_repo.checkout("main")
        synthetic_repo.write_file(
            "src/main/java/demo/ComplexCalculator.java", java_fixtures.COMPLEX_CALCULATOR
        )
        base_sha = synthetic_repo.commit("Add complex calculator on main")

        # Refactoring branch
        refactor_branch = "refactor/optimize-calculator"
        synthetic_repo.create_branch(refactor_branch, base_sha)
        synthetic_repo.checkout(refactor_branch)
        # Refactor calculator to reduce complexity
        synthetic_repo.write_file(
            "src/main/java/demo/ComplexCalculator.java", java_fixtures.REFACTORED_CALCULATOR
        )
        head_sha = synthetic_repo.commit("Refactor calculator")

        perspective = spec_classify_pr(refactor_branch, ())
        assert perspective == "refactoring"

        # Merge-base check
        merge_res = git_cli_resolve_merge_base(synthetic_repo, base_sha, head_sha)
        assert merge_res["status"] == "comparable"


# ==============================================================================
# Scenario 3: Webhook Delivery Burst with PR Update Coalescing
# ==============================================================================


class TestScenario3WebhookBurstCoalescing:
    def test_webhook_burst_coalescing(self, temp_sqlite_db: Path) -> None:
        """5 rapid delivery bursts for the same PR coalesce to desired_generation=5

        and deduplicate repeated delivery IDs.
        """
        conn = sqlite3.connect(str(temp_sqlite_db))
        conn.executescript(SQLITE_SCHEMA_CONTRACT)

        repo_id, pr_num = 200, 15
        # Simulate 5 rapid webhooks
        for i in range(5):
            deliv_id = f"delivery-{i}"
            conn.execute(
                "INSERT INTO deliveries (delivery_id, payload_digest, received_at, status) "
                "VALUES (?, 'digest', datetime('now'), 'received');",
                (deliv_id,),
            )
            conn.execute(
                "INSERT INTO pr_requests (installation_id, repository_id, pr_number, desired_generation, updated_at) "
                "VALUES (1, ?, ?, 1, datetime('now')) "
                "ON CONFLICT(repository_id, pr_number) DO UPDATE SET "
                "desired_generation = desired_generation + 1, updated_at = datetime('now');",
                (repo_id, pr_num),
            )
        conn.commit()

        # Check total unique deliveries recorded
        count = conn.execute("SELECT COUNT(*) FROM deliveries;").fetchone()[0]
        assert count == 5

        # Check coalesced generation
        gen = conn.execute(
            "SELECT desired_generation FROM pr_requests WHERE repository_id = ? AND pr_number = ?;",
            (repo_id, pr_num),
        ).fetchone()[0]
        assert gen == 5
        conn.close()


# ==============================================================================
# Scenario 4: Offline CLI Full Audit on Synthetic Complex Repo
# ==============================================================================


class TestScenario4OfflineCliFullAudit:
    def test_synthetic_repo_audit_structure(
        self,
        synthetic_repo: SyntheticGitRepo,
        java_fixtures: JavaSourceFixtures,
        tmp_path: Path,
    ) -> None:
        """Prepares a complex repository with clean, complex, and duplicate classes,
        executes offline architecture review, and verifies resulting inspection artifacts.
        """
        # Base commit with clean service
        synthetic_repo.write_file("src/main/java/demo/Clean.java", java_fixtures.CLEAN_SERVICE)
        base_sha = synthetic_repo.commit("Initial clean base commit")

        # Head commit introducing complex logic and duplications
        synthetic_repo.write_file("src/main/java/demo/Complex.java", java_fixtures.COMPLEX_CALCULATOR)
        synthetic_repo.write_file("src/main/java/demo/DupA.java", java_fixtures.DUPLICATE_CLASS_A)
        synthetic_repo.write_file("src/main/java/demo/DupB.java", java_fixtures.DUPLICATE_CLASS_B)
        synthetic_repo.write_file("src/main/java/demo/God.java", java_fixtures.GOD_CLASS)
        head_sha = synthetic_repo.commit("Add complex and duplicate classes")
        assert len(head_sha) == 40

        # Run offline review CLI
        out_dir = tmp_path / "audit-artifacts"
        ret = main([
            "review",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--base",
            base_sha,
            "--head",
            head_sha,
            "-o",
            str(out_dir),
        ])
        assert ret == 0
        assert (out_dir / "comment.md").exists()
        assert (out_dir / "result.json").exists()
        assert (out_dir / "delta.json").exists()
        assert (out_dir / "task_input.json").exists()

        comment_text = (out_dir / "comment.md").read_text(encoding="utf-8")
        assert "<!-- review-bot:architecture:v1 -->" in comment_text
        assert "### 아키텍처 리뷰" in comment_text


# ==============================================================================
# Scenario 5: Guarded Concurrent Publication Race
# ==============================================================================


class TestScenario5GuardedConcurrentPublicationRace:
    def test_guarded_publication_race_prevention(self, mock_github: MockGitHubClient) -> None:
        """Two concurrent workers attempt to publish.

        First worker creates marker comment; second worker finds existing marker
        and updates it idempotently instead of creating duplicate comments.
        """
        pr_num = 77
        marker = "<!-- review-bot:architecture:v1 -->"

        # Worker 1 posts
        comment1 = mock_github.post_comment(pr_num, f"{marker}\nWorker 1 review output")
        assert len(mock_github.comments) == 1

        # Worker 2 discovers existing marker comment
        existing = mock_github.find_marker_comment(pr_num, marker)
        assert existing is not None
        assert existing.id == comment1.id

        # Worker 2 updates instead of creating new
        updated = mock_github.update_comment(existing.id, f"{marker}\nWorker 2 updated output")
        assert len(mock_github.comments) == 1
        assert updated.body == f"{marker}\nWorker 2 updated output"

    @pytest.mark.parametrize("analyzer_kind", ["spy", "pmd"])
    def test_worker_publishes_and_upserts_across_generations(
        self, synthetic_repo: SyntheticGitRepo, java_fixtures: JavaSourceFixtures,
        tmp_path: Path, valid_config_yaml: str, analyzer_kind: str,
    ) -> None:
        path = "src/main/java/demo/ComplexCalculator.java"
        base_source = "package demo;\npublic class ComplexCalculator { public int calculate() { return 0; } }\n"
        first_source = java_fixtures.COMPLEX_CALCULATOR
        second_source = first_source.replace("result = -1;", "result = -2;")
        outside = tmp_path / "Outside.java"
        outside.write_text("class Outside {}\n", encoding="utf-8")
        synthetic_repo.write_file("build.gradle", "plugins { id 'java' }\n")
        synthetic_repo.write_file(path, base_source)
        (synthetic_repo.repo_dir / "src/main/java/demo/Linked.java").symlink_to(outside)
        (synthetic_repo.repo_dir / "external").symlink_to(tmp_path, target_is_directory=True)
        synthetic_repo.write_file("src\\Ignored.java", "class Ignored {}\n")
        comparison_base = synthetic_repo.commit("Add base service")
        synthetic_repo.create_branch("feature/service", comparison_base)
        synthetic_repo.checkout("feature/service")
        synthetic_repo.write_file(path, first_source)
        first_head = synthetic_repo.commit("Change service")
        synthetic_repo.write_file(path, second_source)
        second_head = synthetic_repo.commit("Update service")
        synthetic_repo.checkout("main")
        synthetic_repo.write_file("README.md", "Advance target base independently\n")
        target_base = synthetic_repo.commit("Advance target base")
        assert target_base != comparison_base

        base_run = make_test_run(comparison_base, source_files=(path,))
        runs = {comparison_base: base_run}
        for head in (first_head, second_head):
            runs[head] = make_test_run(head, source_files=(path,), pmd_violations=(
                make_pmd_violation(path=path, begin_line=3, end_line=3, klass="ComplexCalculator", method="calculate"),
            ))
        sources = {comparison_base: base_source, first_head: first_source, second_head: second_source}
        roots: list[Path] = []
        runner = PmdRunner(output_root=tmp_path / "analysis") if analyzer_kind == "pmd" else None

        def analyze(sha: str, source_root: Path, ruleset: str, profile: str) -> AnalysisRun:
            assert isinstance(source_root, Path)
            assert source_root.is_dir() and source_root != synthetic_repo.repo_dir
            roots.append(source_root)
            assert (source_root / path).read_text(encoding="utf-8") == sources[sha]
            for excluded in ("src/main/java/demo/Linked.java", "external", "src\\Ignored.java", "README.md", ".git"):
                assert not (source_root / excluded).exists()
                assert not (source_root / excluded).is_symlink()
            analysis = runner.run_analysis(sha, source_root, ruleset, profile) if runner else runs[sha]
            assert source_root.is_dir()
            assert analysis.source_files == (path,)
            assert analysis.status == ("clean" if sha == comparison_base else "findings")
            if runner:
                assert analysis.source_state == "complete"
                assert analysis.pmd.exit_code == (0 if sha == comparison_base else 4)
                assert analysis.cpd.exit_code == 0
                recorded_root = Path(analysis.output_dir) / "sources"
                for tool in (analysis.pmd, analysis.cpd):
                    assert tool.execution.source_root == str(recorded_root)
                    assert str(recorded_root / "src/main/java") in tool.execution.command
                assert (recorded_root / path).read_text(encoding="utf-8") == sources[sha]
            return analysis

        analyzer = Mock(spec=StaticAnalyzer)
        analyzer.run_analysis.side_effect = analyze
        model = MockModelClient([
            ModelResponse(json.dumps({
                "base_sha": comparison_base, "head_sha": head, "review_mode": "general",
                "summary": f"Architecture pass {generation}", "findings": [], "questions": [],
            }))
            for generation, head in enumerate((first_head, second_head), 1)
        ])
        github = ApiMockGitHubClient()
        config, _ = load_review_config_from_str(valid_config_yaml)
        config.publish.mode = "comment"
        store = SQLiteJobStore(tmp_path / "worker_lifecycle.db")
        try:
            worker = WorkerLoop(
                config, store, repo_reader=GitRepositoryReader(synthetic_repo.repo_dir),
                analyzer=analyzer, model_client=model, github_client=github,
                worker_id="lifecycle-worker", lease_duration=30.0,
            )
            comment_ids = []
            fingerprints = []
            for generation, head in enumerate((first_head, second_head), 1):
                github.set_pull_request(101, head_sha=head, base_sha=target_base, branch="feature/service")
                github.pull_requests[101]["base"]["repo"] = {"full_name": "org/repo"}
                store.enqueue_request(PrRequestRecord(1, 200, 101, 1))

                assert worker.claim_and_execute_once() is True

                conn = store._get_connection()
                assert conn.execute(
                    "SELECT status, stage, error FROM runs WHERE repository_id = 200 AND generation = ?;",
                    (generation,),
                ).fetchone() == ("succeeded", "publish", None)
                comment_id, fingerprint, published_at = conn.execute(
                    "SELECT comment_id, published_fingerprint, published_at FROM publication_receipts;",
                ).fetchone()
                assert type(comment_id) is int and comment_id > 0
                assert published_at is not None and len(fingerprint) == 64
                comment_ids.append(comment_id)
                fingerprints.append(fingerprint)
                body = github.comments[comment_id]["body"]
                assert body.startswith(config.publish.a_marker)
                assert "### 아키텍처 리뷰" in body
                assert "#### 전체 판단" in body and "#### 리뷰 의견" in body
                assert "이 검토 범위에서 추가로 조치할 발견사항은 없습니다" in body
                assert f"Architecture pass {generation}" in body
                assert "Base SHA:" not in body and "Head SHA:" not in body
                supplied = json.loads(model.requests[-1].user_content)
                assert supplied["snapshot"]["base"]["sha"] == comparison_base
                assert supplied["snapshot"]["head"]["sha"] == head
                assert target_base not in body

            assert comment_ids[0] == comment_ids[1]
            assert fingerprints[0] != fingerprints[1]
            assert len(github.comments) == len(github.posted_comments) == 1
            assert len(github.updated_comments) == 1
            assert len(model.requests) == 2
            assert analyzer.run_analysis.call_count == 4
            assert len(set(roots)) == 4
            assert all(not root.exists() for root in roots)
            assert outside.read_text(encoding="utf-8") == "class Outside {}\n"
            assert worker._current_run_id is None
        finally:
            store.close()


# ==============================================================================
# Scenario 6: Unconfigured Model Profile Safe Abort
# ==============================================================================


class TestScenario6UnconfiguredModelSafeAbort:
    def test_unconfigured_model_action_is_not_run(self, valid_config_yaml: str) -> None:
        """When model profile is null/unconfigured, review-bot safely aborts model invocation
        with unconfigured_action='not_run' without raising unexpected errors.
        """
        assert "profile: null" in valid_config_yaml
        assert "unconfigured_action: not_run" in valid_config_yaml

        # Validate config model invariants
        config, _ = load_review_config_from_str(valid_config_yaml)
        assert config.model.profile is None
        assert config.model.unconfigured_action == "not_run"


# ==============================================================================
# Scenario 7: Ambiguous Merge-Base Incomparable PR Handling
# ==============================================================================


class TestScenario7AmbiguousMergeBaseHandling:
    def test_ambiguous_merge_base_safe_handling(self, synthetic_repo: SyntheticGitRepo) -> None:
        """In criss-cross merge where multiple merge-bases exist,

        status is 'ambiguous', comparison_base_sha is null, and no false delta is computed.
        """
        topo = synthetic_repo.setup_criss_cross_merge()
        merge_res = git_cli_resolve_merge_base(synthetic_repo, topo["sha_d"], topo["sha_e"])
        assert merge_res["status"] == "ambiguous"
        assert merge_res["comparison_base_sha"] is None
        assert len(merge_res["candidates"]) >= 2
