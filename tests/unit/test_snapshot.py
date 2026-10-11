"""Unit tests for application/snapshot.py assembly pipeline."""
from __future__ import annotations

from unittest.mock import Mock

import pytest

from review_bot.adapters.git import GitRepositoryReader
from review_bot.application.snapshot import (
    assemble_pr_snapshot,
    assemble_review_snapshot,
    assemble_snapshot_bundle,
)
from review_bot.domain.envelope_v2 import ChangedFile
from review_bot.ports import MergeBaseResult
from tests.conftest import SyntheticGitRepo


def test_assemble_pr_snapshot() -> None:
    changed = [
        ChangedFile(path="src/Foo.java", change_type="modified"),
        ChangedFile(path="src/Bar.java", previous_path="src/OldBar.java", change_type="renamed"),
    ]
    renames = {"src/OldBar.java": "src/Bar.java"}
    pr_snap = assemble_pr_snapshot(
        repository="org/repo",
        number=42,
        base_sha="a" * 40,
        head_sha="b" * 40,
        source_branch="refactor/cleanup",
        labels=("리팩토링",),
        changed_files=changed,
        rename_map=renames,
    )
    assert pr_snap.repository == "org/repo"
    assert pr_snap.number == 42
    assert pr_snap.changed_files == ("src/Foo.java", "src/Bar.java")
    assert pr_snap.rename_map == renames
    assert pr_snap.source_branch == "refactor/cleanup"


def test_assemble_review_snapshot() -> None:
    merge_res = MergeBaseResult(
        target_base_sha="a" * 40,
        comparison_base_sha="c" * 40,
        comparison_status="comparable",
        candidates=("c" * 40,),
    )
    rev_snap = assemble_review_snapshot(
        repository="org/repo",
        pr_number=42,
        repository_id=101,
        installation_id=202,
        target_base_sha="a" * 40,
        head_sha="b" * 40,
        merge_base_result=merge_res,
        title="My PR",
        body="PR Description",
    )
    assert rev_snap.schema_version == 2
    assert rev_snap.snapshot_id is not None
    assert rev_snap.comparison_base_sha == "c" * 40
    assert rev_snap.comparison_status == "comparable"
    assert len(rev_snap.pr_text_digest) == 64
    assert len(rev_snap.context_digest) == 64


def test_assemble_snapshot_bundle_comparable(synthetic_repo: SyntheticGitRepo) -> None:
    pr_info = synthetic_repo.setup_standard_feature_pr(advance_base=True)
    reader = GitRepositoryReader(synthetic_repo.repo_dir)

    rev_snap, pr_snap, classification, selection = assemble_snapshot_bundle(
        repo_reader=reader,
        repository="org/repo",
        pr_number=10,
        repository_id=1,
        installation_id=2,
        target_base_sha=pr_info["target_base_sha"],
        head_sha=pr_info["head_sha"],
        source_branch="refactor/user-module",
        labels=("리팩토링",),
        title="Refactor user module",
    )

    assert rev_snap.comparison_status == "comparable"
    assert rev_snap.comparison_base_sha == pr_info["comparison_base_sha"]
    assert pr_snap.base_sha == pr_info["comparison_base_sha"]
    assert classification.review_mode == "refactoring"
    assert selection.selected_modes == ("architecture",)
    assert len(rev_snap.unavailable_sources) == 0


def test_assemble_snapshot_bundle_orphan(
    synthetic_repo: SyntheticGitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_sha, orphan_sha = synthetic_repo.setup_orphan_branches()
    reader = GitRepositoryReader(synthetic_repo.repo_dir)
    changed_files = Mock(wraps=reader.get_changed_files)
    rename_map = Mock(wraps=reader.get_rename_map)
    monkeypatch.setattr(reader, "get_changed_files", changed_files)
    monkeypatch.setattr(reader, "get_rename_map", rename_map)

    rev_snap, pr_snap, classification, selection = assemble_snapshot_bundle(
        repo_reader=reader,
        repository="org/repo",
        pr_number=11,
        repository_id=1,
        installation_id=2,
        target_base_sha=main_sha,
        head_sha=orphan_sha,
        source_branch="feature/orphan",
        labels=(),
        title="Disjoint branch",
    )

    assert rev_snap.comparison_status == "incomparable"
    assert rev_snap.comparison_base_sha is None
    assert rev_snap.target_base_sha == main_sha
    assert pr_snap.base_sha == ""
    assert rev_snap.changed_files == ()
    assert pr_snap.changed_files == ()
    assert pr_snap.rename_map == {}
    assert classification.review_mode == "general"
    assert selection.selection_status == "completed"
    assert len(rev_snap.unavailable_sources) == 1
    assert rev_snap.unavailable_sources[0].source == "merge_base"
    assert "unavailable" in rev_snap.unavailable_sources[0].reason
    changed_files.assert_not_called()
    rename_map.assert_not_called()


@pytest.mark.parametrize(
    "status, comparison_base_sha",
    [
        ("unavailable", None),
        ("ambiguous", "c" * 40),
        ("comparable", None),
        ("comparable", ""),
    ],
)
def test_review_snapshot_rejects_non_comparable_base(
    status: str, comparison_base_sha: str | None
) -> None:
    merge_res = MergeBaseResult("a" * 40, comparison_base_sha, status)
    rev_snap = assemble_review_snapshot(
        repository="org/repo",
        pr_number=42,
        repository_id=101,
        installation_id=202,
        target_base_sha="a" * 40,
        head_sha="b" * 40,
        merge_base_result=merge_res,
    )

    assert rev_snap.comparison_status == "incomparable"
    assert rev_snap.comparison_base_sha is None
    assert rev_snap.target_base_sha == "a" * 40
