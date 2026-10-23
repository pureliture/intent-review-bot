"""Adversarial challenge and stress-test suite for Standalone Offline CLI (review-bot).

Covers:
- Dimension 1: 100% Offline execution & network isolation
- Dimension 2: Backwards compatibility & version inspection
- Dimension 3: Argument parsing & CLI options
- Dimension 4: Non-existent repository handling (empirical defect discovery)
- Dimension 5: Missing commits & unhandled tracebacks (empirical defect discovery)
- Dimension 6: Corrupted and non-existent JSON run files in delta (empirical defect discovery)
- Dimension 7: Malformed schema and asymmetric argument handling in delta (empirical defect discovery)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from review_bot import __version__
from review_bot.runtime.cli import create_parser, main
from tests.conftest import SyntheticGitRepo


class TestCliOfflineAndCompatibility:
    """Tests 100% offline execution and backwards compatibility."""

    def test_offline_execution_without_network_or_api_keys(
        self, synthetic_repo: SyntheticGitRepo, tmp_path: Path
    ) -> None:
        """Verify complete offline review execution with no API keys or network."""
        pr_info = synthetic_repo.setup_standard_feature_pr()
        out_dir = tmp_path / "offline-artifacts"

        # Scrub all network/API tokens from environment
        clean_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", str(tmp_path)),
            "JAVA_HOME": os.environ.get("JAVA_HOME", ""),
            "LANG": "en_US.UTF-8",
        }
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "review",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--base",
            pr_info["comparison_base_sha"],
            "--head",
            pr_info["head_sha"],
            "--output",
            str(out_dir),
        ]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=clean_env,
            check=False,
        )
        assert proc.returncode == 0, f"Offline review failed: {proc.stderr}"
        assert (out_dir / "comment.md").exists()
        assert (out_dir / "result.json").exists()
        assert (out_dir / "delta.json").exists()
        assert (out_dir / "task_input.json").exists()

        comment_text = (out_dir / "comment.md").read_text(encoding="utf-8")
        assert comment_text.splitlines()[0] == "<!-- review-bot:architecture:v1 -->"
        assert "### 아키텍처 리뷰" in comment_text

    def test_version_subcommand_and_flags(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify --version, version, and version --json."""
        ret = main(["version"])
        assert ret == 0
        captured = capsys.readouterr()
        assert f"review-bot {__version__}" in captured.out
        assert "PMD/CPD: 7.3.0" in captured.out

        ret_json = main(["version", "--json"])
        assert ret_json == 0
        captured_json = capsys.readouterr()
        data = json.loads(captured_json.out)
        assert data["version"] == __version__
        assert data["tools_lock"]["pmd_version"] == "7.3.0"
        assert len(data["tools_lock"]["artifacts"]) > 0

    def test_backwards_compatibility_subcommands(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify offline, webhook, and worker subcommands satisfy backwards compatibility."""
        # offline subcommand
        ret = main(["offline"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "review-bot subcommand 'offline' initialized." in captured.out

        # webhook subcommand --help
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_webhook:
            parser.parse_args(["webhook", "--help"])
        assert exc_webhook.value.code == 0

        # worker subcommand --help
        with pytest.raises(SystemExit) as exc_worker:
            parser.parse_args(["worker", "--help"])
        assert exc_worker.value.code == 0

    def test_invalid_cli_arguments_rejected(self) -> None:
        """Verify unrecognized arguments exit with code 2."""
        proc = subprocess.run(
            [sys.executable, "-m", "review_bot.runtime.cli", "--invalid-flag"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2
        assert "unrecognized arguments: --invalid-flag" in proc.stderr

    def test_invalid_format_choice_rejected(self) -> None:
        """Verify invalid choice in --format exits with code 2."""
        proc = subprocess.run(
            [sys.executable, "-m", "review_bot.runtime.cli", "delta", "--format", "invalid_fmt"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 2
        assert "invalid choice: 'invalid_fmt'" in proc.stderr


class TestCliEmpiricalDefects:
    """Verifies remediation of edge cases and empirical defect scenarios.

    Verifies:
    1. review subcommand cleanly fails (code 1) with error message on non-existent repos
    2. analyze subcommand cleanly fails (code 1) on missing commits with no traceback
    3. review subcommand cleanly fails (code 1) on missing commits with no traceback
    4. delta subcommand cleanly fails (code 1) on missing commits with no traceback
    5. delta subcommand cleanly fails (code 1) on corrupted JSON runs with clean error
    6. delta subcommand cleanly fails (code 1) on non-existent JSON runs with clean error
    7. delta subcommand cleanly fails (code 1) on malformed run schema with clean error
    8. delta subcommand rejects asymmetric --base-run / --head-run arguments with clean error
    """

    def test_defect_review_nonexistent_repo_silent_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: review on non-existent repo returns 1 and outputs clean error message."""
        nonexistent = tmp_path / "nonexistent_repo_dir"
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "review",
            "--repo",
            str(nonexistent),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Error:" in proc.stderr
        assert "is not a git repository" in proc.stderr
        assert "Traceback (most recent call last):" not in proc.stderr

        # Verify direct main() invocation returns code 1
        ret = main(["review", "--repo", str(nonexistent)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err
        assert "is not a git repository" in captured.err

    def test_defect_analyze_missing_commit_unhandled_traceback(
        self, synthetic_repo: SyntheticGitRepo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: analyze on missing commit returns 1 with clean error message and no traceback."""
        missing_sha = "deadbeef" * 5
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "analyze",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--commit",
            missing_sha,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Traceback (most recent call last):" not in proc.stderr
        assert f"Error: Commit not found: {missing_sha}" in proc.stderr

        # Inside direct python invocation, returns 1 without unhandled exception
        ret = main(["analyze", "--repo", str(synthetic_repo.repo_dir), "--commit", missing_sha])
        assert ret == 1
        captured = capsys.readouterr()
        assert f"Error: Commit not found: {missing_sha}" in captured.err

    def test_defect_review_missing_base_commit_unhandled_traceback(
        self, synthetic_repo: SyntheticGitRepo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: review on missing base commit returns 1 with clean error message and no traceback."""
        pr_info = synthetic_repo.setup_standard_feature_pr()
        missing_sha = "00000000" * 5
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "review",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--base",
            missing_sha,
            "--head",
            pr_info["head_sha"],
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Traceback (most recent call last):" not in proc.stderr
        assert f"Error: Commit not found: {missing_sha}" in proc.stderr

        ret = main([
            "review",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--base",
            missing_sha,
            "--head",
            pr_info["head_sha"],
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert f"Error: Commit not found: {missing_sha}" in captured.err

    def test_defect_delta_missing_base_commit_unhandled_traceback(
        self, synthetic_repo: SyntheticGitRepo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: delta on missing commit returns 1 with clean error message and no traceback."""
        pr_info = synthetic_repo.setup_standard_feature_pr()
        missing_sha = "11111111" * 5
        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "delta",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--base",
            missing_sha,
            "--head",
            pr_info["head_sha"],
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Traceback (most recent call last):" not in proc.stderr
        assert f"Error: Commit not found: {missing_sha}" in proc.stderr

        ret = main([
            "delta",
            "--repo",
            str(synthetic_repo.repo_dir),
            "--base",
            missing_sha,
            "--head",
            pr_info["head_sha"],
        ])
        assert ret == 1
        captured = capsys.readouterr()
        assert f"Error: Commit not found: {missing_sha}" in captured.err

    def test_defect_delta_corrupted_json_run_unhandled_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: delta on corrupted JSON run returns 1 with clean error message and no traceback."""
        bad_json = tmp_path / "corrupted_run.json"
        bad_json.write_text("{ this is malformed JSON !!! }", encoding="utf-8")
        good_json = tmp_path / "valid_run.json"
        good_json.write_text(json.dumps({"schema_version": 1, "sha": "a" * 40}), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "delta",
            "--base-run",
            str(bad_json),
            "--head-run",
            str(good_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Traceback (most recent call last):" not in proc.stderr
        assert "Error: Malformed JSON in run file:" in proc.stderr

        ret = main(["delta", "--base-run", str(bad_json), "--head-run", str(good_json)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: Malformed JSON in run file:" in captured.err

    def test_defect_delta_nonexistent_json_run_unhandled_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: delta on non-existent JSON run returns 1 with clean error message and no traceback."""
        good_json = tmp_path / "valid_run.json"
        good_json.write_text(json.dumps({"schema_version": 1, "sha": "a" * 40}), encoding="utf-8")
        nonexistent_file = tmp_path / "ghost_base_run.json"

        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "delta",
            "--base-run",
            str(nonexistent_file),
            "--head-run",
            str(good_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Traceback (most recent call last):" not in proc.stderr
        assert "Error: Failed to read analysis run file:" in proc.stderr

        ret = main(["delta", "--base-run", str(nonexistent_file), "--head-run", str(good_json)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: Failed to read analysis run file:" in captured.err

    def test_defect_delta_malformed_run_schema_unhandled_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: delta on malformed run schema returns 1 with clean error message and no traceback."""
        malformed_json = tmp_path / "malformed_schema.json"
        malformed_json.write_text(json.dumps({"pmd": "should_be_dict_not_str"}), encoding="utf-8")
        good_json = tmp_path / "valid_run.json"
        good_json.write_text(json.dumps({"schema_version": 1, "sha": "a" * 40}), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "review_bot.runtime.cli",
            "delta",
            "--base-run",
            str(malformed_json),
            "--head-run",
            str(good_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        assert proc.returncode == 1
        assert "Traceback (most recent call last):" not in proc.stderr
        assert "Error: Invalid analysis run data:" in proc.stderr

        ret = main(["delta", "--base-run", str(malformed_json), "--head-run", str(good_json)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: Invalid analysis run data:" in captured.err

    def test_defect_delta_asymmetric_arguments_silent_fallback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Verify: delta with only --base-run or --head-run rejects with clean error message."""
        base_run_file = tmp_path / "base.json"
        base_run_file.write_text(json.dumps({"schema_version": 1, "sha": "a" * 40}), encoding="utf-8")

        ret = main(["delta", "--base-run", str(base_run_file)])
        assert ret == 1
        captured = capsys.readouterr()
        assert "Error: Both --base-run and --head-run must be provided together" in captured.err

        ret2 = main(["delta", "--head-run", str(base_run_file)])
        assert ret2 == 1
        captured2 = capsys.readouterr()
        assert "Error: Both --base-run and --head-run must be provided together" in captured2.err
