"""Adversarial stress-test suite for Git adapter (src/review_bot/adapters/git.py).

Covers:
1. Malformed Git Repositories (empty dirs, corrupted HEAD, missing objects, zero-commit repos)
2. Command & Flag Injection Vectors (argument injection, shell metachars, exotic branch names)
3. Complex Merge Topologies (octopus >3 parents, criss-cross merges, zero-commit branches, disjoint)
4. Path Traversal & Safe Path Invariants (crafted trees, symlinks, traversal paths)
5. Huge Diffs, Binary Files & Resource Pressure (1000+ files, 5MB+ blobs, non-UTF-8 binaries)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from review_bot.adapters.git import (
    CommitNotFoundError,
    GitCommandError,
    GitRepositoryReader,
    _fixed_sha,
    _safe_path,
)
from tests.conftest import SyntheticGitRepo

# ==============================================================================
# Category 1: Malformed Git Repositories
# ==============================================================================


class TestMalformedGitRepos:
    """Stress tests for Git adapter behavior on corrupt, missing, or malformed repos."""

    def test_non_existent_directory(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(ValueError, match="Not a valid git repository"):
            GitRepositoryReader(missing_dir)

    def test_empty_directory_not_git(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="Not a valid git repository"):
            GitRepositoryReader(empty_dir)

    def test_file_as_repository_path(self, tmp_path: Path) -> None:
        dummy_file = tmp_path / "regular_file.txt"
        dummy_file.write_text("not a repo")
        with pytest.raises(ValueError, match="Not a valid git repository"):
            GitRepositoryReader(dummy_file)

    def test_empty_initialized_repo_with_zero_commits(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "zero_commits"
        repo_dir.mkdir()
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)

        reader = GitRepositoryReader(repo_dir)

        # Querying HEAD on zero-commit repo must raise CommitNotFoundError, not uncaught crash
        with pytest.raises(CommitNotFoundError, match="Commit not found: HEAD"):
            reader.get_commit("HEAD")

        # Disjoint non-existent SHAs in zero-commit repo return unavailable status
        sha1 = "0" * 40
        sha2 = "1" * 40
        res = reader.resolve_merge_base(sha1, sha2)
        assert res.comparison_status == "unavailable"
        assert res.comparison_base_sha is None

    def test_corrupted_head_ref(self, synthetic_repo: SyntheticGitRepo) -> None:
        head_file = synthetic_repo.repo_dir / ".git" / "HEAD"
        head_file.write_text("ref: refs/heads/corrupted_non_existent_branch\n")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        with pytest.raises(CommitNotFoundError):
            reader.get_commit("HEAD")

    def test_corrupted_head_garbage_content(self, synthetic_repo: SyntheticGitRepo) -> None:
        head_file = synthetic_repo.repo_dir / ".git" / "HEAD"
        head_file.write_bytes(b"\x00\xff\xfe\x01\x02\x03\x04")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        with pytest.raises(CommitNotFoundError):
            reader.get_commit("HEAD")

    def test_corrupted_objects_directory(self, synthetic_repo: SyntheticGitRepo) -> None:
        obj_dir = synthetic_repo.repo_dir / ".git" / "objects"
        shutil.rmtree(obj_dir)
        obj_dir.write_text("corrupted file replacing objects dir")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        with pytest.raises(CommitNotFoundError):
            reader.get_commit("HEAD")

    def test_non_existent_commit_sha_handling(self, synthetic_repo: SyntheticGitRepo) -> None:
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        fake_sha = "0" * 40
        head_sha = synthetic_repo.get_head_sha()

        # get_commit raises CommitNotFoundError
        with pytest.raises(CommitNotFoundError):
            reader.get_commit(fake_sha)

        # read_tree_blobs raises GitCommandError
        with pytest.raises(GitCommandError) as exc_info:
            reader.read_tree_blobs(fake_sha)
        assert exc_info.value.returncode == 128

        # get_changed_files raises GitCommandError
        with pytest.raises(GitCommandError) as exc_info:
            reader.get_changed_files(fake_sha, head_sha)
        assert exc_info.value.returncode == 128

        # compute_merge_base returns ()
        assert reader.compute_merge_base(fake_sha, head_sha) == ()

        # resolve_merge_base returns unavailable
        res = reader.resolve_merge_base(fake_sha, head_sha)
        assert res.comparison_status == "unavailable"
        assert res.comparison_base_sha is None

    def test_truncated_or_malformed_shas_rejected_with_value_error(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        head_sha = synthetic_repo.get_head_sha()

        invalid_shas = [
            "",
            "short",
            "a" * 39,
            "a" * 41,
            "g" * 40,
            "a" * 63,
            "a" * 65,
            "0" * 39 + "z",
            "HEAD",
            "main",
            "../../master",
            "0" * 40 + " ",
            " 0" * 40,
        ]

        for inv in invalid_shas:
            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                _fixed_sha(inv)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.resolve_merge_base(inv, head_sha)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.read_tree_blobs(inv)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.get_changed_files(inv, head_sha)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.collect_code_snapshot(inv)

    def test_finding_sha_regex_trailing_newline_leak(self) -> None:
        """Verify _SHA_REGEX uses '\\Z' anchor and rejects trailing newlines."""
        sha_with_newline = "0" * 40 + "\n"
        with pytest.raises(ValueError, match="Invalid full Git SHA"):
            _fixed_sha(sha_with_newline)


# ==============================================================================
# Category 2: Command & Flag Injection Vectors
# ==============================================================================


class TestBranchNamesAndCommandInjection:
    """Stress tests verifying subprocess sanitization, argument delimiters, and injection defenses."""

    def test_shell_metacharacters_in_branch_names_do_not_execute(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Verify shell metacharacters in rev strings are never interpreted by shell."""
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        probe_file = synthetic_repo.repo_dir / "injected.txt"

        injection_payloads = [
            f"; touch {probe_file}",
            f"| touch {probe_file}",
            f"& touch {probe_file}",
            f"$(touch {probe_file})",
            f"`touch {probe_file}`",
            f"main; touch {probe_file}",
            f"main && touch {probe_file}",
            "\n touch injected.txt\n",
        ]

        for payload in injection_payloads:
            with pytest.raises(CommitNotFoundError):
                reader.get_commit(payload)
            # Confirm probe file was NOT created
            assert not probe_file.exists(), f"Command injection succeeded for payload: {payload!r}"

    def test_vulnerability_get_commit_argument_injection_output_flag(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Verify get_commit defends against --output argument injection."""
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        pwn_file = synthetic_repo.repo_dir / "pwned_output.txt"
        assert not pwn_file.exists()

        # Calling get_commit with --output=<file> is rejected
        with pytest.raises(CommitNotFoundError):
            reader.get_commit(f"--output={pwn_file}")

        # The defense prevents file creation/overwrite on disk
        assert not pwn_file.exists(), "Argument injection created file on disk!"

    def test_vulnerability_get_commit_flag_argument_returns_head_commit(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Verify get_commit rejects flags like '--no-walk' and does not default to HEAD."""
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        with pytest.raises(CommitNotFoundError):
            reader.get_commit("--no-walk")

    def test_safe_functions_strictly_reject_flag_injection(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Verify resolve_merge_base, read_tree_blobs, get_changed_files, and collect_code_snapshot

        strictly reject flag arguments via _fixed_sha.
        """
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        head_sha = synthetic_repo.get_head_sha()

        dangerous_flags = [
            "--output=/tmp/evil",
            "--upload-pack=evil",
            "--help",
            "--version",
            "--no-walk",
            "-b",
            "-m",
            "--exec=sh",
        ]

        for flag in dangerous_flags:
            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.resolve_merge_base(flag, head_sha)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.resolve_merge_base(head_sha, flag)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.read_tree_blobs(flag)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.get_changed_files(flag, head_sha)

            with pytest.raises(ValueError, match="Invalid full Git SHA"):
                reader.collect_code_snapshot(flag)

    def test_valid_exotic_branch_names_in_git(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Verify git reader handles genuine branches with unicode, hyphens, and slashes."""
        reader = GitRepositoryReader(synthetic_repo.repo_dir)

        exotic_branches = [
            "feat/korean_한국어",
            "refactor/v1.0.0-rc.1",
            "fix/issue-123_test.foo",
        ]

        for branch in exotic_branches:
            synthetic_repo.create_branch(branch)
            commit = reader.get_commit(branch)
            assert commit.sha == synthetic_repo.get_head_sha()

    def test_subprocess_environment_cleaning(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Verify host GIT_* environment variables cannot alter adapter execution."""
        head_sha = synthetic_repo.get_head_sha()
        os.environ["GIT_DIR"] = "/tmp/fake_git_dir"
        os.environ["GIT_WORK_TREE"] = "/tmp/fake_work_tree"
        try:
            reader = GitRepositoryReader(synthetic_repo.repo_dir)
            commit = reader.get_commit("HEAD")
            assert commit.sha == head_sha
        finally:
            os.environ.pop("GIT_DIR", None)
            os.environ.pop("GIT_WORK_TREE", None)


# ==============================================================================
# Category 3: Complex Merge Topologies
# ==============================================================================


class TestComplexMergeTopologies:
    """Stress tests covering octopus merges, criss-cross topologies, and branch edge cases."""

    def test_octopus_merge_four_parents(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Octopus merge with 4 parents must parse all 4 parents and resolve merge-bases cleanly."""
        c0 = synthetic_repo.get_head_sha()

        # Create 4 independent branches from c0
        shas = []
        for i in range(1, 5):
            synthetic_repo.checkout("main")
            branch_name = f"octo_b{i}"
            synthetic_repo.create_branch(branch_name)
            synthetic_repo.checkout(branch_name)
            synthetic_repo.write_file(f"f{i}.txt", f"content {i}")
            shas.append(synthetic_repo.commit(f"Commit on {branch_name}"))

        # Merge branches 2, 3, 4 into branch 1
        synthetic_repo.checkout("octo_b1")
        synthetic_repo._git("merge", "octo_b2", "octo_b3", "octo_b4", "-m", "Octopus 4 parents")
        octo_sha = synthetic_repo.get_head_sha()

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        commit = reader.get_commit(octo_sha)
        assert len(commit.parents) == 4
        assert set(commit.parents) == set(shas)

        # Merge-base between octopus commit and one of its parent branches must be that parent
        res_parent = reader.resolve_merge_base(shas[1], octo_sha)
        assert res_parent.comparison_status == "comparable"
        assert res_parent.comparison_base_sha == shas[1]

        # Merge-base between octopus commit and an external branch off c0 must be c0
        synthetic_repo.checkout("main")
        synthetic_repo.create_branch("external")
        synthetic_repo.checkout("external")
        synthetic_repo.write_file("ext.txt", "external")
        c_ext = synthetic_repo.commit("External commit")

        res_ext = reader.resolve_merge_base(c_ext, octo_sha)
        assert res_ext.comparison_status == "comparable"
        assert res_ext.comparison_base_sha == c0

    def test_criss_cross_merge_multiple_candidates(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Criss-cross merge producing multiple common ancestors must yield ambiguous status."""
        topo = synthetic_repo.setup_criss_cross_merge()
        reader = GitRepositoryReader(synthetic_repo.repo_dir)

        res = reader.resolve_merge_base(topo["sha_d"], topo["sha_e"])
        assert res.comparison_status == "ambiguous"
        assert res.comparison_base_sha is None
        assert res.is_comparable is False
        assert len(res.candidates) >= 2
        assert set(res.candidates) == set(topo["merge_bases"])

    def test_disjoint_orphan_branches(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Orphan branches with zero common ancestors must yield unavailable status."""
        main_sha, orphan_sha = synthetic_repo.setup_orphan_branches()
        reader = GitRepositoryReader(synthetic_repo.repo_dir)

        res = reader.resolve_merge_base(main_sha, orphan_sha)
        assert res.comparison_status == "unavailable"
        assert res.comparison_base_sha is None
        assert res.is_comparable is False
        assert res.candidates == ()

    def test_zero_commit_feature_branch_with_advanced_base(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Feature branch created at C0 with 0 commits while base advances to C1."""
        c0 = synthetic_repo.get_head_sha()
        synthetic_repo.create_branch("zero_feat")

        # Advance main
        synthetic_repo.write_file("main_adv.txt", "adv")
        c1 = synthetic_repo.commit("Main advanced")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        res = reader.resolve_merge_base(c1, c0)
        assert res.comparison_status == "comparable"
        assert res.target_base_sha == c1
        assert res.comparison_base_sha == c0

    def test_identical_existing_commit_comparison(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Target base == head sha for an existing commit."""
        head_sha = synthetic_repo.get_head_sha()
        reader = GitRepositoryReader(synthetic_repo.repo_dir)

        res = reader.resolve_merge_base(head_sha, head_sha)
        assert res.comparison_status == "comparable"
        assert res.target_base_sha == head_sha
        assert res.comparison_base_sha == head_sha
        assert res.candidates == (head_sha,)

    def test_inconsistency_identical_non_existent_commit(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Verify resolve_merge_base verifies commit existence when t_sha == h_sha."""
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        fake_sha = "0" * 40

        # When t_sha != h_sha and both are fake, status is correctly 'unavailable'
        diff_fake = "1" * 40
        res_diff = reader.resolve_merge_base(fake_sha, diff_fake)
        assert res_diff.comparison_status == "unavailable"

        # When t_sha == h_sha and fake, commit does not exist so status must also be 'unavailable'
        res_same = reader.resolve_merge_base(fake_sha, fake_sha)
        assert res_same.comparison_status == "unavailable"
        assert res_same.comparison_base_sha is None
        assert res_same.candidates == ()


# ==============================================================================
# Category 4: Path Traversal & Safe Path Invariants
# ==============================================================================


class TestPathTraversalDefense:
    """Stress tests verifying path traversal prevention in tree reads and diffs."""

    def test_safe_path_comprehensive_validation(self) -> None:
        # Valid safe paths
        assert _safe_path("src/Main.java") is True
        assert _safe_path("build.gradle.kts") is True
        assert _safe_path("src/main/resources/config.xml") is True
        assert _safe_path("filename_with_underscore-and.dot.txt") is True

        # Traversal attempts
        assert _safe_path("../etc/passwd") is False
        assert _safe_path("../../secret.key") is False
        assert _safe_path("src/../../etc/passwd") is False
        assert _safe_path("src/../Main.java") is False
        assert _safe_path("src/./Main.java") is False
        assert _safe_path("./Main.java") is False
        assert _safe_path("Main.java/.") is False
        assert _safe_path("Main.java/..") is False

        # Absolute paths
        assert _safe_path("/etc/passwd") is False
        assert _safe_path("/var/log/syslog") is False
        assert _safe_path("\\Windows\\System32") is False

        # Windows backslashes
        assert _safe_path("src\\Main.java") is False
        assert _safe_path("src/test\\App.java") is False

        # Control characters and null bytes
        assert _safe_path("src/main/\0null.java") is False
        assert _safe_path("src/main/\nnewline.java") is False
        assert _safe_path("src/main/\rreturn.java") is False
        assert _safe_path("src/main/\x7fdel.java") is False

        # Empty and double slashes
        assert _safe_path("") is False
        assert _safe_path("src//Main.java") is False
        assert _safe_path("//etc/passwd") is False

    def test_read_tree_blobs_traversal_paths_arg_defense(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Passing malicious traversal paths in paths parameter must yield empty result."""
        synthetic_repo.write_file("src/App.java", "public class App {}")
        commit_sha = synthetic_repo.commit("Add App")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)

        malicious_paths = [
            "../../../../etc/passwd",
            "/etc/passwd",
            "../App.java",
            "src/../src/App.java",
        ]
        blobs = reader.read_tree_blobs(commit_sha, paths=malicious_paths)
        assert blobs == {}

    def test_read_tree_blobs_symlinks_skipped(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Symlinks in git tree (mode 120000) must be skipped and never read as blobs."""
        synthetic_repo.write_file("regular.java", "class Regular {}")
        repo_dir = synthetic_repo.repo_dir

        # Create symlinks
        link_target = repo_dir / "symlink_to_passwd"
        link_target.symlink_to("/etc/passwd")

        link_rel = repo_dir / "symlink_to_regular"
        link_rel.symlink_to("regular.java")

        commit_sha = synthetic_repo.commit("Add regular file and symlinks")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        blobs = reader.read_tree_blobs(commit_sha)

        assert "regular.java" in blobs
        assert "symlink_to_passwd" not in blobs
        assert "symlink_to_regular" not in blobs

        # Verify collect_code_snapshot excludes symlinks
        snapshot = reader.collect_code_snapshot(commit_sha)
        paths = [f.path for f in snapshot.files]
        assert "regular.java" in paths
        assert "symlink_to_passwd" not in paths
        assert "symlink_to_regular" not in paths

    def test_crafted_tree_with_dotdot_skipped_by_safe_path(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Low-level crafted git tree containing '..' entry must be filtered by _safe_path."""
        td = synthetic_repo.repo_dir
        proc = subprocess.run(
            ["git", "-C", str(td), "hash-object", "-w", "--stdin"],
            input=b"malicious payload",
            capture_output=True,
            check=True,
        )
        oid = proc.stdout.decode().strip()

        # Craft tree with '..' entry using git mktree
        tree_input = f"100644 blob {oid}\t..\n100644 blob {oid}\tvalid.txt\n".encode()
        mktree_proc = subprocess.run(
            ["git", "-C", str(td), "mktree"],
            input=tree_input,
            capture_output=True,
            check=True,
        )
        tree_sha = mktree_proc.stdout.decode().strip()

        commit_proc = subprocess.run(
            ["git", "-C", str(td), "commit-tree", tree_sha, "-m", "Crafted .. tree"],
            capture_output=True,
            check=True,
        )
        commit_sha = commit_proc.stdout.decode().strip()

        reader = GitRepositoryReader(td)
        blobs = reader.read_tree_blobs(commit_sha)
        assert "valid.txt" in blobs
        assert ".." not in blobs


# ==============================================================================
# Category 5: Huge Diffs, Binary Files & Resource Pressure
# ==============================================================================


class TestHugeDiffsAndBinaryFiles:
    """Stress tests verifying handling of large numbers of files, big blobs, and binaries."""

    def test_huge_diff_1000_files_performance_and_integrity(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """Verify get_changed_files and collect_code_snapshot handle 1,000 files in < 2 seconds."""
        # Create 1,000 files
        for i in range(1000):
            synthetic_repo.write_file(f"src/File{i}.java", f"class File{i} {{ int v = 0; }}")
        base_sha = synthetic_repo.commit("Base 1000 files")

        # Modify 500 files and add 500 new files
        for i in range(500):
            synthetic_repo.write_file(f"src/File{i}.java", f"class File{i} {{ int v = 1; }}")
        for i in range(500):
            synthetic_repo.write_file(f"src/New{i}.java", f"class New{i} {{}}")
        head_sha = synthetic_repo.commit("Modify 500 and add 500")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)

        t0 = time.perf_counter()
        changed = reader.get_changed_files(base_sha, head_sha)
        duration_diff = time.perf_counter() - t0
        assert len(changed) == 1000
        assert duration_diff < 2.0, f"get_changed_files took {duration_diff:.2f}s (too slow)"

        t0 = time.perf_counter()
        snapshot = reader.collect_code_snapshot(head_sha)
        duration_snap = time.perf_counter() - t0
        assert len(snapshot.files) == 1500
        assert duration_snap < 5.0, f"collect_code_snapshot took {duration_snap:.2f}s (too slow)"

    def test_large_file_streaming_cat_file_offset_sync(
        self, synthetic_repo: SyntheticGitRepo
    ) -> None:
        """5MB blob with newlines followed by subsequent files must preserve offset synchronization."""
        # 5MB of text with abundant internal newlines
        large_text = ("line of code in large file\n" * 200000).encode("utf-8")
        synthetic_repo.write_file("Large.java", large_text)
        synthetic_repo.write_file("Next.java", "class Next {}")
        commit_sha = synthetic_repo.commit("Add 5MB file and next file")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        blobs = reader.read_tree_blobs(commit_sha)

        assert "Large.java" in blobs
        assert "Next.java" in blobs
        assert len(blobs["Large.java"]) == len(large_text)
        assert blobs["Large.java"] == large_text
        assert blobs["Next.java"] == b"class Next {}"

    def test_empty_files_handling(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Empty (0-byte) files must parse properly without header split errors."""
        synthetic_repo.write_file("empty.txt", b"")
        synthetic_repo.write_file("non_empty.txt", b"hello")
        commit_sha = synthetic_repo.commit("Add empty file")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        blobs = reader.read_tree_blobs(commit_sha)
        assert blobs["empty.txt"] == b""
        assert blobs["non_empty.txt"] == b"hello"

        snap = reader.collect_code_snapshot(commit_sha)
        assert snap.complete is True

    def test_binary_files_in_diff_and_snapshot(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Binary files must be tracked in diffs, read in blobs, and excluded in snapshots."""
        synthetic_repo.write_file("binary.dat", b"\x00\x01\x02\xff\xfe\xfd\x80")
        synthetic_repo.write_file("source.java", "class Source {}")
        base_sha = synthetic_repo.commit("Add binary and source")

        synthetic_repo.write_file("binary.dat", b"\x00\x01\x02\xff\xfe\xfd\x80\x90\xaa")
        head_sha = synthetic_repo.commit("Modify binary")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        changed = reader.get_changed_files(base_sha, head_sha)
        assert len(changed) == 1
        assert changed[0].path == "binary.dat"
        assert changed[0].change_type == "modified"

        blobs = reader.read_tree_blobs(head_sha)
        assert blobs["binary.dat"] == b"\x00\x01\x02\xff\xfe\xfd\x80\x90\xaa"

        # Snapshot must mark complete=False and record SourceExclusion
        snap = reader.collect_code_snapshot(head_sha)
        assert snap.complete is False
        assert len(snap.exclusions) == 1
        assert snap.exclusions[0].path == "binary.dat"
        assert snap.exclusions[0].reason == "unsupported_encoding"
        assert "binary.dat: unsupported encoding" in snap.errors

    def test_renames_with_spaces_and_unicode(self, synthetic_repo: SyntheticGitRepo) -> None:
        """Rename tracking must handle spaces and unicode in filenames correctly."""
        synthetic_repo.write_file("src/Old Name.java", "class Renamed {}")
        synthetic_repo.write_file("src/한국어_이전.java", "class Korean {}")
        base_sha = synthetic_repo.commit("Before rename")

        synthetic_repo.delete_file("src/Old Name.java")
        synthetic_repo.delete_file("src/한국어_이전.java")
        synthetic_repo.write_file("src/New Name.java", "class Renamed {}")
        synthetic_repo.write_file("src/한국어_이후.java", "class Korean {}")
        head_sha = synthetic_repo.commit("After rename")

        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        renames = reader.get_rename_map(base_sha, head_sha)
        assert renames == {
            "src/Old Name.java": "src/New Name.java",
            "src/한국어_이전.java": "src/한국어_이후.java",
        }
