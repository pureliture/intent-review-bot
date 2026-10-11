"""Unit tests for review-bot CLI commands (version, analyze, delta, review)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from review_bot.adapters.git import GitRepositoryReader
from review_bot.runtime import cli
from review_bot.runtime.cli import main
from tests.conftest import SyntheticGitRepo


def test_cli_version_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["version"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "review-bot 0.1.0" in captured.out
    assert "PMD/CPD: 7.3.0" in captured.out


def test_cli_version_json_flag(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["version", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["version"] == "0.1.0"
    assert "tools_lock" in data


def test_cli_analyze_invalid_repo(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    ret = main(["analyze", "--repo", str(tmp_path)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "is not a git repository" in captured.err


@pytest.mark.parametrize("topology", ["orphan", "criss_cross"])
def test_cli_review_halts_incomparable_comparison(
    synthetic_repo: SyntheticGitRepo,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    topology: str,
) -> None:
    if topology == "orphan":
        target_base_sha, head_sha = synthetic_repo.setup_orphan_branches()
        resolution_status = "unavailable"
    else:
        topo = synthetic_repo.setup_criss_cross_merge()
        target_base_sha, head_sha = topo["sha_d"], topo["sha_e"]
        resolution_status = "ambiguous"
    reader = GitRepositoryReader(synthetic_repo.repo_dir)
    code_snapshot = Mock(wraps=reader.collect_code_snapshot)
    changed_files = Mock(wraps=reader.get_changed_files)
    rename_map = Mock(wraps=reader.get_rename_map)
    monkeypatch.setattr(reader, "collect_code_snapshot", code_snapshot)
    monkeypatch.setattr(reader, "get_changed_files", changed_files)
    monkeypatch.setattr(reader, "get_rename_map", rename_map)
    monkeypatch.setattr(cli, "GitRepositoryReader", Mock(return_value=reader))
    runner = Mock(side_effect=AssertionError("Incomparable revisions must not be analyzed"))
    delta_builder = Mock(wraps=cli.build_delta)
    review_pipeline = Mock()
    monkeypatch.setattr(cli, "PmdRunner", runner)
    monkeypatch.setattr(cli, "build_delta", delta_builder)
    monkeypatch.setattr(cli, "run_review_a_pipeline", review_pipeline)
    out_dir = tmp_path / "incomparable-artifacts"

    ret = main([
        "review", "--repo", str(synthetic_repo.repo_dir),
        "--base", target_base_sha, "--head", head_sha, "-o", str(out_dir),
    ])

    assert ret == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "comparison_status=incomparable" in captured.err
    assert resolution_status in captured.err
    assert not out_dir.exists()
    code_snapshot.assert_not_called()
    changed_files.assert_not_called()
    rename_map.assert_not_called()
    runner.assert_not_called()
    delta_builder.assert_not_called()
    review_pipeline.assert_not_called()


def test_cli_analyze_valid_commit(synthetic_repo: SyntheticGitRepo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pr_info = synthetic_repo.setup_standard_feature_pr()
    repo_path = str(synthetic_repo.repo_dir)
    out_file = tmp_path / "analysis.json"

    ret = main(["analyze", "--repo", repo_path, "--commit", pr_info["head_sha"], "-o", str(out_file)])
    assert ret == 0
    assert out_file.exists()

    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["sha"] == pr_info["head_sha"]
    assert "pmd" in content
    assert "cpd" in content


def test_cli_delta_from_json_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Create synthetic base and head analysis JSON files
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
        "pmd": {"tool": "pmd", "status": "clean", "exit_code": 0, "report": {"tool": "pmd", "violations": [], "duplications": []}},
        "cpd": {"tool": "cpd", "status": "clean", "exit_code": 0, "report": {"tool": "cpd", "violations": [], "duplications": []}},
    }
    head_run = dict(base_run)
    head_run["sha"] = "b" * 40

    base_file = tmp_path / "base.json"
    head_file = tmp_path / "head.json"
    base_file.write_text(json.dumps(base_run), encoding="utf-8")
    head_file.write_text(json.dumps(head_run), encoding="utf-8")

    out_delta = tmp_path / "delta.json"
    ret = main(["delta", "--base-run", str(base_file), "--head-run", str(head_file), "-o", str(out_delta)])
    assert ret == 0
    assert out_delta.exists()

    delta_data = json.loads(out_delta.read_text(encoding="utf-8"))
    assert delta_data["base_sha"] == "a" * 40
    assert delta_data["head_sha"] == "b" * 40


def test_cli_delta_markdown_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
        "source_files": [],
        "source_hash": "sh1",
        "pmd": {"tool": "pmd", "status": "clean", "exit_code": 0, "report": {"tool": "pmd", "violations": [], "duplications": []}},
        "cpd": {"tool": "cpd", "status": "clean", "exit_code": 0, "report": {"tool": "cpd", "violations": [], "duplications": []}},
    }
    head_run = dict(base_run)
    head_run["sha"] = "b" * 40

    base_file = tmp_path / "base.json"
    head_file = tmp_path / "head.json"
    base_file.write_text(json.dumps(base_run), encoding="utf-8")
    head_file.write_text(json.dumps(head_run), encoding="utf-8")

    ret = main(["delta", "--base-run", str(base_file), "--head-run", str(head_file), "--format", "table"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "| existing | new | worsened | improved | resolved | unknown |" in captured.out


def test_cli_review_end_to_end_on_synthetic_repo(synthetic_repo: SyntheticGitRepo, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pr_info = synthetic_repo.setup_standard_feature_pr()
    repo_path = str(synthetic_repo.repo_dir)
    out_dir = tmp_path / "review-artifacts"

    ret = main([
        "review",
        "--repo",
        repo_path,
        "--base",
        pr_info["comparison_base_sha"],
        "--head",
        pr_info["head_sha"],
        "--output",
        str(out_dir),
    ])
    assert ret == 0
    assert (out_dir / "comment.md").exists()
    assert (out_dir / "result.json").exists()
    assert (out_dir / "delta.json").exists()
    assert (out_dir / "task_input.json").exists()

    comment_content = (out_dir / "comment.md").read_text(encoding="utf-8")
    assert "<!-- review-bot:architecture:v1 -->" in comment_content
    assert "### PMD/CPD 아키텍처 리뷰 A" in comment_content


def test_cli_review_invalid_repo(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    non_git = tmp_path / "not_git"
    non_git.mkdir()
    ret = main(["review", "--repo", str(non_git)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "is not a git repository" in captured.err


def test_cli_delta_invalid_repo(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    non_git = tmp_path / "not_git"
    non_git.mkdir()
    ret = main(["delta", "--repo", str(non_git)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "is not a git repository" in captured.err


def test_cli_offline_compat_invalid_repo(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    non_git = tmp_path / "not_git"
    non_git.mkdir()
    ret = main(["offline", "--repo", str(non_git)])
    assert ret == 1
    captured = capsys.readouterr()
    assert "is not a git repository" in captured.err


def test_cli_delta_asymmetric_args_validation(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text("{}", encoding="utf-8")

    ret1 = main(["delta", "--base-run", str(run_file)])
    assert ret1 == 1
    assert "Both --base-run and --head-run must be provided together" in capsys.readouterr().err

    ret2 = main(["delta", "--head-run", str(run_file)])
    assert ret2 == 1
    assert "Both --base-run and --head-run must be provided together" in capsys.readouterr().err
