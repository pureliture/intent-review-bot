"""E2E tests for Feature 26: Offline CLI (review-bot).

Requirements:
- ORIGINAL_REQUEST §R5 & design.md:41:
  - Standalone offline CLI for local execution and artifact inspection.
  - No required external GitHub or live model API connectivity.
  - Subcommands: review, offline, analyze, delta, version, webhook, worker.
  - Deterministic exit codes: 0 for success, 1 for domain/repo error, 2 for CLI parse error.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import SyntheticGitRepo


class TestOfflineCliE2E:
    def test_cli_invocation_help(self) -> None:
        """Verifies CLI entrypoint exposes complete usage help with all subcommands."""
        cmd = [sys.executable, "-m", "review_bot.runtime.cli", "--help"]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 0
        assert "review-bot" in proc.stdout
        assert "review" in proc.stdout
        assert "offline" in proc.stdout
        assert "analyze" in proc.stdout
        assert "delta" in proc.stdout
        assert "version" in proc.stdout

    def test_cli_version_and_tools_lock_json(self) -> None:
        """Verifies version subcommand with and without --json flag."""
        # Text version
        cmd_text = [sys.executable, "-m", "review_bot.runtime.cli", "version"]
        proc_text = subprocess.run(cmd_text, capture_output=True, check=False, text=True)
        assert proc_text.returncode == 0
        assert "review-bot 0.1.0" in proc_text.stdout

        # JSON version
        cmd_json = [sys.executable, "-m", "review_bot.runtime.cli", "version", "--json"]
        proc_json = subprocess.run(cmd_json, capture_output=True, check=False, text=True)
        assert proc_json.returncode == 0
        data = json.loads(proc_json.stdout)
        assert data["version"] == "0.1.0"
        assert "tools_lock" in data

    def test_cli_analyze_e2e_synthetic_repo(
        self, synthetic_repo: SyntheticGitRepo, tmp_path: Path
    ) -> None:
        """Exercises analyze subcommand against a commit in synthetic repository."""
        pr_info = synthetic_repo.setup_standard_feature_pr()
        repo_path = str(synthetic_repo.repo_dir)
        out_file = tmp_path / "analysis.json"

        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "analyze",
            "--repo",
            repo_path,
            "--commit",
            pr_info["head_sha"],
            "-o",
            str(out_file),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 0
        assert out_file.exists()

        content = json.loads(out_file.read_text(encoding="utf-8"))
        assert content["sha"] == pr_info["head_sha"]
        assert "pmd" in content
        assert "cpd" in content

    def test_cli_delta_e2e_synthetic_runs(self, tmp_path: Path) -> None:
        """Exercises delta subcommand comparing base and head analysis JSON files."""
        base_run = {
            "schema_version": 1,
            "run_id": "r1",
            "sha": "a" * 40,
            "profile_hash": "p1",
            "conditions_hash": "c1",
            "conditions": {
                "schema_version": 1,
                "pmd_version": "7.3.0",
                "cpd_version": "7.3.0",
                "ruleset_hash": "rh",
                "minimum_tokens": 100,
                "source_roots": ["src/main/java"],
                "language": "java",
                "language_version": "21",
                "encoding": "UTF-8",
                "type_resolution": "jdk_only",
                "java_version": "21.0.6",
                "java_vendor": "Amazon",
                "classpath_hash": "cp1",
            },
            "output_dir": "",
            "started_at": "",
            "completed_at": "",
            "status": "clean",
            "source_state": "complete",
            "source_files": ["src/main/java/Root.java"],
            "source_hash": "sh1",
            "pmd": {
                "tool": "pmd",
                "status": "clean",
                "exit_code": 0,
                "report": {"tool": "pmd", "violations": [], "duplications": []},
            },
            "cpd": {
                "tool": "cpd",
                "status": "clean",
                "exit_code": 0,
                "report": {"tool": "cpd", "violations": [], "duplications": []},
            },
        }
        head_run = dict(base_run)
        head_run["sha"] = "b" * 40

        base_file = tmp_path / "base.json"
        head_file = tmp_path / "head.json"
        base_file.write_text(json.dumps(base_run), encoding="utf-8")
        head_file.write_text(json.dumps(head_run), encoding="utf-8")

        out_delta = tmp_path / "delta.json"
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "delta",
            "--base-run",
            str(base_file),
            "--head-run",
            str(head_file),
            "-o",
            str(out_delta),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 0
        assert out_delta.exists()
        delta_data = json.loads(out_delta.read_text(encoding="utf-8"))
        assert delta_data["base_sha"] == "a" * 40
        assert delta_data["head_sha"] == "b" * 40

    def test_cli_review_full_lifecycle_artifacts(
        self, synthetic_repo: SyntheticGitRepo, tmp_path: Path
    ) -> None:
        """Executes full review subcommand and verifies all 4 inspection artifacts."""
        pr_info = synthetic_repo.setup_standard_feature_pr()
        repo_path = str(synthetic_repo.repo_dir)
        out_dir = tmp_path / "review-artifacts"

        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "review",
            "--repo",
            repo_path,
            "--base",
            pr_info["comparison_base_sha"],
            "--head",
            pr_info["head_sha"],
            "-o",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 0
        assert (out_dir / "comment.md").exists()
        assert (out_dir / "result.json").exists()
        assert (out_dir / "delta.json").exists()
        assert (out_dir / "task_input.json").exists()

        comment = (out_dir / "comment.md").read_text(encoding="utf-8")
        assert "<!-- review-bot:architecture:v1 -->" in comment
        assert "### PMD/CPD 아키텍처 리뷰 A" in comment

    def test_cli_offline_subcommand_compat(
        self, synthetic_repo: SyntheticGitRepo, tmp_path: Path
    ) -> None:
        """Verifies backward compatibility of 'offline' subcommand with 'review'."""
        pr_info = synthetic_repo.setup_standard_feature_pr()
        repo_path = str(synthetic_repo.repo_dir)
        out_dir = tmp_path / "compat-artifacts"

        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "offline",
            "--repo",
            repo_path,
            "--base",
            pr_info["comparison_base_sha"],
            "--head",
            pr_info["head_sha"],
            "-o",
            str(out_dir),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 0
        assert (out_dir / "comment.md").exists()
        assert (out_dir / "result.json").exists()

    def test_cli_invalid_repo_error_exit(self, tmp_path: Path) -> None:
        """Verifies non-git directory fails gracefully with exit code 1."""
        non_git = tmp_path / "not_git"
        non_git.mkdir()
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "review",
            "--repo",
            str(non_git),
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 1
        assert "is not a git repository" in proc.stderr

    def test_cli_unknown_subcommand_error_exit(self) -> None:
        """Verifies invalid CLI subcommand fails with exit code 2."""
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "nonexistent-subcommand",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
        assert proc.returncode == 2
