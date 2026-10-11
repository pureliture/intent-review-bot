"""Unit tests for Git adapter (GitRepositoryReader, object reads, diffs, renames)."""
from __future__ import annotations

from pathlib import Path

import pytest

from review_bot.adapters.git import (
    CommitNotFoundError,
    GitRepositoryReader,
    _fixed_sha,
    _safe_path,
    collect_code_snapshot,
    get_changed_files,
    get_rename_map,
    read_tree_blobs,
    resolve_merge_base,
)
from tests.conftest import SyntheticGitRepo


def test_safe_path_validation() -> None:
    assert _safe_path("src/main/App.java") is True
    assert _safe_path("build.gradle") is True
    assert _safe_path("/absolute/path") is False
    assert _safe_path("\\windows\\path") is False
    assert _safe_path("../traversal/path") is False
    assert _safe_path("foo/../bar") is False
    assert _safe_path("foo/./bar") is False
    assert _safe_path("foo//bar") is False
    assert _safe_path("") is False
    assert _safe_path("null\0byte") is False


def test_fixed_sha_validation() -> None:
    valid_40 = "a" * 40
    valid_64 = "b" * 64
    assert _fixed_sha(valid_40) == valid_40
    assert _fixed_sha(valid_64) == valid_64
    assert _fixed_sha(valid_40.upper()) == valid_40

    with pytest.raises(ValueError, match="Invalid full Git SHA"):
        _fixed_sha("short")

    with pytest.raises(ValueError, match="Invalid full Git SHA"):
        _fixed_sha("z" * 40)


def test_invalid_repo_path(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not_a_repo"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="Not a valid git repository"):
        GitRepositoryReader(empty_dir)


def test_git_repository_reader_commit_and_blobs(synthetic_repo: SyntheticGitRepo) -> None:
    reader = GitRepositoryReader(synthetic_repo.repo_dir)

    synthetic_repo.write_file("src/Main.java", "public class Main {}")
    synthetic_repo.write_file("test/MainTest.java", "public class MainTest {}")
    synthetic_repo.write_file("build.gradle", "plugins { id 'java' }")
    commit_sha = synthetic_repo.commit("Add source, test and build files")

    commit = reader.get_commit(commit_sha)
    assert commit.sha == commit_sha
    assert "ReviewBotTester" in commit.author
    assert commit.message == "Add source, test and build files"
    assert commit.tree_sha is not None

    # Test read_tree_blobs
    blobs = reader.read_tree_blobs(commit_sha)
    assert "src/Main.java" in blobs
    assert blobs["src/Main.java"] == b"public class Main {}"
    assert "test/MainTest.java" in blobs
    assert "build.gradle" in blobs

    # Test path filtering
    filtered = reader.read_tree_blobs(commit_sha, paths=["src/Main.java"])
    assert list(filtered.keys()) == ["src/Main.java"]

    # Test collect_code_snapshot
    snapshot = reader.collect_code_snapshot(commit_sha)
    assert snapshot.sha == commit_sha
    assert snapshot.complete is True
    assert len(snapshot.files) == 3
    file_by_path = {f.path: f for f in snapshot.files}
    assert file_by_path["src/Main.java"].kind == "source"
    assert file_by_path["test/MainTest.java"].kind == "test"
    assert file_by_path["build.gradle"].kind == "build"
    assert len(snapshot.inventory_hash) == 64


def test_git_diff_and_rename_map(synthetic_repo: SyntheticGitRepo) -> None:
    reader = GitRepositoryReader(synthetic_repo.repo_dir)

    synthetic_repo.write_file("src/OldName.java", "class Test {}")
    synthetic_repo.write_file("src/DeleteMe.java", "class DeleteMe {}")
    base_sha = synthetic_repo.commit("Base commit")

    # Perform rename, deletion, and addition
    synthetic_repo.delete_file("src/OldName.java")
    synthetic_repo.delete_file("src/DeleteMe.java")
    synthetic_repo.write_file("src/NewName.java", "class Test {}")
    synthetic_repo.write_file("src/Added.java", "class Added {}")
    head_sha = synthetic_repo.commit("Rename and add")

    changed_files = reader.get_changed_files(base_sha, head_sha)
    types = {f.path: f.change_type for f in changed_files}
    assert types.get("src/NewName.java") == "renamed"
    assert types.get("src/DeleteMe.java") == "deleted"
    assert types.get("src/Added.java") == "added"

    rename_map = reader.get_rename_map(base_sha, head_sha)
    assert rename_map == {"src/OldName.java": "src/NewName.java"}


def test_commit_not_found(synthetic_repo: SyntheticGitRepo) -> None:
    reader = GitRepositoryReader(synthetic_repo.repo_dir)
    fake_sha = "0" * 40
    with pytest.raises(CommitNotFoundError):
        reader.get_commit(fake_sha)


def test_standalone_git_functions(synthetic_repo: SyntheticGitRepo) -> None:
    synthetic_repo.write_file("Standalone.java", "public class Standalone {}")
    base_sha = synthetic_repo.commit("Base commit")

    synthetic_repo.write_file("Standalone2.java", "public class Standalone2 {}")
    head_sha = synthetic_repo.commit("Head commit")

    res = resolve_merge_base(synthetic_repo.repo_dir, base_sha, head_sha)
    assert res.is_comparable is True
    assert res.comparison_base_sha == base_sha

    blobs = read_tree_blobs(synthetic_repo.repo_dir, head_sha)
    assert "Standalone.java" in blobs
    assert "Standalone2.java" in blobs

    changes = get_changed_files(synthetic_repo.repo_dir, base_sha, head_sha)
    assert len(changes) == 1
    assert changes[0].path == "Standalone2.java"
    assert get_rename_map(synthetic_repo.repo_dir, base_sha, head_sha) == {}

    snap = collect_code_snapshot(synthetic_repo.repo_dir, head_sha)
    assert snap.sha == head_sha
    assert len(snap.files) >= 2
