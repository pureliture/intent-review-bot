"""Contract tests for Feature 8: Merge-Base Separation & Feature 12: Full Code Gathering.

Requirements:
- contracts.md §1 & runtime-a-draft.md §5:
  - Strictly distinguish target_base_sha from unique merge-base commit (comparison_base_sha).
  - Missing merge-base -> comparison_status='unavailable'.
  - Multiple merge-base candidates (criss-cross) -> comparison_status='ambiguous'.
  - Neither case may fabricate a false single comparison_base_sha or false findings.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from review_bot.adapters.git import GitRepositoryReader
from review_bot.application.snapshot import assemble_snapshot_bundle
from tests.conftest import SyntheticGitRepo

try:
    from review_bot.adapters.git import resolve_merge_base  # type: ignore
except ImportError:
    resolve_merge_base = None  # type: ignore


# Reference oracle implementation of merge-base resolver using git CLI directly
def git_cli_resolve_merge_base(repo: SyntheticGitRepo, base_sha: str, head_sha: str) -> dict[str, Any]:
    bases = repo.compute_merge_base(base_sha, head_sha, all_bases=True)
    if len(bases) == 1:
        return {
            "comparison_base_sha": bases[0],
            "target_base_sha": base_sha,
            "status": "comparable",
        }
    elif len(bases) > 1:
        return {
            "comparison_base_sha": None,
            "target_base_sha": base_sha,
            "status": "ambiguous",
            "candidates": bases,
        }
    else:
        return {
            "comparison_base_sha": None,
            "target_base_sha": base_sha,
            "status": "unavailable",
        }


# ==============================================================================
# Tier 1: Standard Three-Dot Merge-Base Separation
# ==============================================================================


class TestMergeBaseContractTier1:
    def test_standard_pr_distinguishes_target_base_and_comparison_base(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        pr_info = synthetic_repo.setup_standard_feature_pr(advance_base=True)
        target_base = pr_info["target_base_sha"]
        comparison_base = pr_info["comparison_base_sha"]
        head_sha = pr_info["head_sha"]

        # Crucial contract check: Target base moved ahead, so they must NOT be equal!
        assert target_base != comparison_base

        resolution = git_cli_resolve_merge_base(synthetic_repo, target_base, head_sha)
        assert resolution["status"] == "comparable"
        assert resolution["comparison_base_sha"] == comparison_base
        assert resolution["target_base_sha"] == target_base

    def test_unadvanced_base_shares_sha(self, synthetic_repo: SyntheticGitRepo) -> None:
        pr_info = synthetic_repo.setup_standard_feature_pr(advance_base=False)
        target_base = pr_info["target_base_sha"]
        comparison_base = pr_info["comparison_base_sha"]
        head_sha = pr_info["head_sha"]

        assert target_base == comparison_base
        resolution = git_cli_resolve_merge_base(synthetic_repo, target_base, head_sha)
        assert resolution["status"] == "comparable"
        assert resolution["comparison_base_sha"] == target_base

    def test_adapter_if_available(self, synthetic_repo: SyntheticGitRepo) -> None:
        if resolve_merge_base is None:
            pytest.skip("review_bot.adapters.git not yet implemented by Worker")
        pr_info = synthetic_repo.setup_standard_feature_pr(advance_base=True)
        res = resolve_merge_base(
            repo_path=synthetic_repo.repo_dir,
            target_base_sha=pr_info["target_base_sha"],
            head_sha=pr_info["head_sha"],
        )
        assert res.comparison_base_sha == pr_info["comparison_base_sha"]
        assert res.comparison_status == "comparable"


# ==============================================================================
# Tier 2: Ambiguous & Disjoint Topologies
# ==============================================================================


class TestMergeBaseContractTier2:
    def test_criss_cross_merge_produces_ambiguous_status(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """When multiple merge-bases exist, comparison_base_sha must be None

        and status must be 'ambiguous'.
        """
        topo = synthetic_repo.setup_criss_cross_merge()
        assert topo["is_ambiguous"] is True
        assert len(topo["merge_bases"]) >= 2

        resolution = git_cli_resolve_merge_base(
            synthetic_repo, topo["sha_d"], topo["sha_e"]
        )
        assert resolution["status"] == "ambiguous"
        assert resolution["comparison_base_sha"] is None
        assert len(resolution["candidates"]) >= 2

    def test_orphan_branch_produces_unavailable_status(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """When no common ancestor exists, comparison_base_sha must be None

        and status must be 'unavailable'.
        """
        main_sha, orphan_sha = synthetic_repo.setup_orphan_branches()
        resolution = git_cli_resolve_merge_base(synthetic_repo, main_sha, orphan_sha)
        assert resolution["status"] == "unavailable"
        assert resolution["comparison_base_sha"] is None

    def test_identical_commit_comparison(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Comparing a commit to itself has itself as the single merge-base."""
        head_sha = synthetic_repo.get_head_sha()
        resolution = git_cli_resolve_merge_base(synthetic_repo, head_sha, head_sha)
        assert resolution["status"] == "comparable"
        assert resolution["comparison_base_sha"] == head_sha

    def test_ambiguous_bundle_never_uses_target_base(
        self, synthetic_repo: SyntheticGitRepo, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        topo = synthetic_repo.setup_criss_cross_merge()
        target_base_sha, head_sha = topo["sha_d"], topo["sha_e"]
        assert len(synthetic_repo.compute_merge_base(target_base_sha, head_sha, all_bases=True)) >= 2
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        resolution = reader.resolve_merge_base(target_base_sha, head_sha)
        assert resolution.comparison_status == "ambiguous"
        assert resolution.comparison_base_sha is None
        changed_files = Mock(wraps=reader.get_changed_files)
        rename_map = Mock(wraps=reader.get_rename_map)
        monkeypatch.setattr(reader, "get_changed_files", changed_files)
        monkeypatch.setattr(reader, "get_rename_map", rename_map)

        rev_snap, pr_snap, _, _ = assemble_snapshot_bundle(
            repo_reader=reader,
            repository="org/repo",
            pr_number=12,
            repository_id=1,
            installation_id=2,
            target_base_sha=target_base_sha,
            head_sha=head_sha,
            source_branch="feature/criss-cross",
        )

        assert rev_snap.comparison_status == "incomparable"
        assert rev_snap.target_base_sha == target_base_sha
        assert rev_snap.comparison_base_sha is None
        assert pr_snap.base_sha == ""
        assert rev_snap.changed_files == ()
        assert pr_snap.changed_files == ()
        assert pr_snap.rename_map == {}
        assert rev_snap.unavailable_sources[0].source == "merge_base"
        assert "ambiguous" in rev_snap.unavailable_sources[0].reason
        changed_files.assert_not_called()
        rename_map.assert_not_called()
