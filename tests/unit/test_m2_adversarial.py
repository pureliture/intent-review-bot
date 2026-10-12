"""Adversarial and empirical stress test suite for Milestone 2 components:
- review_bot.application.classification (PR perspective classification)
- review_bot.application.selection (review mode selection)
- review_bot.application.snapshot (snapshot assembly & invariants)

Stress-tests:
1. Boundary branch names (slashes, case, underscores, unicode, whitespace, RTL overrides, non-strings)
2. Adversarial PR titles (assert title is 100% ignored in classification)
3. Malicious and boundary labels (1000 items, None, non-strings, unicode lookalikes, NFD vs NFC, zero-width chars)
4. Mode selection inputs (null, empty, SQL injection, unknown modes, case sensitivity, duplicate modes)
5. Immutability & hashability invariants across Classification, SelectionRecord, ReviewSnapshot, PrSnapshot
"""
from __future__ import annotations

import unicodedata
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from review_bot.application.classification import classify_pr
from review_bot.application.selection import select_modes
from review_bot.application.snapshot import (
    assemble_pr_snapshot,
    assemble_review_snapshot,
    assemble_snapshot_bundle,
)
from review_bot.domain.envelope_v2 import ChangedFile, UnavailableSource
from review_bot.ports import Commit, MergeBaseResult, RepositoryReader

# ==============================================================================
# 1. Perspective Classification Adversarial Tests
# ==============================================================================


class TestPerspectiveClassificationAdversarial:
    @pytest.mark.parametrize(
        "branch,expected",
        [
            # Exact boundary prefixes
            ("refactor", "general"),               # missing trailing slash
            ("refact", "general"),                 # missing trailing slash
            ("refactor/", "refactoring"),          # minimal prefix
            ("refact/", "refactoring"),            # minimal prefix
            ("refactor//nested", "refactoring"),   # double slash
            ("REFACTOR/BAR", "refactoring"),       # uppercase
            ("ReFaCt/camel", "refactoring"),       # mixed case
            ("refactor_feature", "general"),       # underscore instead of slash
            ("feature/refactor", "general"),       # refactor in middle
            ("refactored/core", "general"),        # refactored != refactor
            ("refactoring", "general"),            # word refactoring without slash
            ("refactor.core", "general"),          # dot instead of slash
            ("refact-cleanup", "general"),         # hyphen instead of slash
            ("refactor/foo ", "refactoring"),      # trailing space in branch
            (" refactor/foo", "general"),          # leading space prevents prefix match
            ("\trefactor/foo", "general"),         # leading tab prevents prefix match
            ("refactor\\foo", "general"),          # backslash is not forward slash
            # Unicode branch names
            ("refactor/기능개선", "refactoring"),
            ("refact/코드_정리", "refactoring"),
            ("feature/리팩토링", "general"),
            ("리팩토링/feature", "general"),
            # RTL character overrides
            ("\u202Erefactor/foo\u202C", "general"),     # RTL override at start hides prefix
            ("refactor/\u202Efoo\u202C", "refactoring"), # RTL inside branch name retains prefix
            # Massive branch name (DoS check)
            ("refactor/" + "a" * 50000, "refactoring"),
            ("feature/" + "b" * 50000, "general"),
            # Empty and non-string inputs
            ("", "general"),
            (None, "general"),
            (12345, "general"),
            ([], "general"),
            ({}, "general"),
        ],
    )
    def test_boundary_branch_names(self, branch: Any, expected: str) -> None:
        result = classify_pr(branch, labels=())
        assert result.review_mode == expected

    @pytest.mark.parametrize(
        "title",
        [
            "[REFACTOR] complete rewrite of architecture",
            "refactor/critical-fix",
            "refact/auth-tokens",
            "리팩토링 진행",
            "refactoring core services",
            "feat: major refactoring across 50 files",
            "\u202Erefactor\u202C",
            "A" * 100000,
            None,
            12345,
            "",
        ],
    )
    def test_pr_title_strictly_ignored(self, title: Any) -> None:
        """Adversarial PR titles must NEVER influence classification posture."""
        res_with_title = classify_pr("feature/payment-v2", labels=(), title=title)
        res_without_title = classify_pr("feature/payment-v2", labels=(), title=None)

        # Classification must be strictly identical regardless of title
        assert res_with_title.review_mode == "general"
        assert res_with_title == res_without_title

        # Title content must not leak into reasons
        if isinstance(title, str) and len(title) > 0:
            for r in res_with_title.reasons:
                assert title not in r

    @pytest.mark.parametrize(
        "label",
        [
            # Whitespace and punctuation variations
            " 리팩토링",
            "리팩토링 ",
            " 리팩토링 ",
            "리팩토링\n",
            "리팩토링\t",
            "리팩토링\u00A0",  # non-breaking space
            "[리팩토링]",
            "리팩토링!",
            "리팩토링/아키텍처",
            # Unicode lookalikes and hidden characters
            "리\u200B팩토링",  # zero-width space
            "리\u200D팩토링",  # zero-width joiner
            "리\u200C팩토링",  # zero-width non-joiner
            "리\xAD팩토링",    # soft hyphen
            "ㄹㅣㅍㅐㄱㅌㅗㄹㅣㅇ",  # Hangul compatibility Jamo
            unicodedata.normalize("NFD", "리팩토링"),  # NFD decomposed Hangul
            # English variations
            "refactor",
            "REFACTOR",
            "refactoring",
            "REFACTORING",
            "Refactor",
        ],
    )
    def test_adversarial_label_variations_rejected(self, label: str) -> None:
        """Only exact precomposed NFC '리팩토링' must trigger refactoring posture."""
        result = classify_pr("feature/orders", labels=(label,))
        assert result.review_mode == "general"

    def test_exact_label_triggers_refactoring(self) -> None:
        result = classify_pr("feature/orders", labels=("리팩토링",))
        assert result.review_mode == "refactoring"
        assert "Label '리팩토링' present" in result.reasons

    def test_massive_label_list_and_non_strings(self) -> None:
        """Classifier must handle large lists, None items, and non-string types safely."""
        # 1. 1000 items without target label
        large_labels = ["enhancement"] * 1000
        res1 = classify_pr("feature/test", labels=large_labels)
        assert res1.review_mode == "general"

        # 2. 1000 items with target label embedded
        large_with_target = ["enhancement"] * 500 + ["리팩토링"] + ["enhancement"] * 499
        res2 = classify_pr("feature/test", labels=large_with_target)
        assert res2.review_mode == "refactoring"

        # 3. None and mixed non-string types inside label list
        mixed_labels = [None, 123, 45.6, False, b"bytes", object()]
        res3 = classify_pr("feature/test", labels=mixed_labels)
        assert res3.review_mode == "general"

        # 4. None and mixed types with target label
        mixed_with_target = [None, 123, "리팩토링", object()]
        res4 = classify_pr("feature/test", labels=mixed_with_target)
        assert res4.review_mode == "refactoring"

    def test_iterable_types_supported(self) -> None:
        """Classifier must accept generators, sets, tuples, and lists."""
        assert classify_pr("main", labels=(x for x in ["리팩토링"])).review_mode == "refactoring"
        assert classify_pr("main", labels={"리팩토링"}).review_mode == "refactoring"
        assert classify_pr("main", labels=("리팩토링",)).review_mode == "refactoring"
        assert classify_pr("main", labels=None).review_mode == "general"

    def test_classification_immutability_and_hashability(self) -> None:
        """Classification instance must be deeply immutable and hashable."""
        cls = classify_pr("refactor/test", labels=("리팩토링",))

        # Reassignment blocked
        with pytest.raises(FrozenInstanceError):
            cls.review_mode = "general"  # type: ignore

        # Deletion blocked
        with pytest.raises(FrozenInstanceError):
            del cls.review_mode  # type: ignore

        # Dynamic attribute injection blocked
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cls.__dynamic_attr__ = "payload"  # type: ignore

        # Reasons is an immutable tuple
        assert isinstance(cls.reasons, tuple)

        # Hashability invariant
        h = hash(cls)
        assert isinstance(h, int)
        s = {cls}
        assert cls in s


# ==============================================================================
# 2. Mode Selection Adversarial Tests
# ==============================================================================


class TestModeSelectionAdversarial:
    @pytest.mark.parametrize("invalid_snap", [None, "", 12345, [], {}])
    def test_rejects_invalid_snapshot_id(self, invalid_snap: Any) -> None:
        with pytest.raises(ValueError, match="snapshot_id must be a non-empty string"):
            select_modes(snapshot_id=invalid_snap, available_modes=["architecture"])

    @pytest.mark.parametrize("empty_modes", [[], (), None])
    def test_rejects_empty_available_modes(self, empty_modes: Any) -> None:
        with pytest.raises(ValueError, match="available_modes must not be empty"):
            select_modes(snapshot_id="snap-valid", available_modes=empty_modes)

    @pytest.mark.parametrize(
        "injected_mode",
        [
            "architecture'; DROP TABLE runs;--",
            "architecture\x00evil",
            "architecture\nmalicious",
            "../etc/passwd",
            "ARCHITECTURE",     # case sensitive: only lowercase registered
            "architecture ",    # trailing space
            " architecture",    # leading space
            "intent'; DROP TABLE runs;--",
            "INTENT",
            "intent ",
            " intent",
            "b",
            "unknown_mode",
            "all",
        ],
    )
    def test_rejects_unregistered_and_malicious_modes(self, injected_mode: str) -> None:
        with pytest.raises(ValueError, match="Unregistered review mode requested"):
            select_modes(snapshot_id="snap-valid", available_modes=[injected_mode])

    @pytest.mark.parametrize("unsupported_policy", ["custom", "random", "", None, 123])
    def test_rejects_unsupported_policy(self, unsupported_policy: Any) -> None:
        with pytest.raises(ValueError, match="Unsupported selector policy"):
            select_modes(
                snapshot_id="snap-valid",
                available_modes=["architecture"],
                policy=unsupported_policy,
            )

    def test_duplicate_modes_behavior(self) -> None:
        """Empirically test that duplicate available modes are deduplicated."""
        rec = select_modes(
            snapshot_id="snap-dup",
            available_modes=["architecture", "architecture"],
        )
        assert "architecture" in rec.selected_modes
        assert len(rec.selected_modes) == 1, "Duplicate modes were not deduplicated in selected_modes"

    def test_idempotent_digest_determinism(self) -> None:
        """Identical inputs must yield bitwise identical selection_id and selection_digest."""
        r1 = select_modes(snapshot_id="snap-stable", available_modes=("architecture",))
        r2 = select_modes(snapshot_id="snap-stable", available_modes=("architecture",))
        assert r1.selection_id == r2.selection_id
        assert r1.selection_digest == r2.selection_digest
        assert r1.selection_digest == (
            "3278d842c43ae9c57cb6b4106d8b4e370dff530e1389bc10e64b7619fb7edd5a"
        )
        assert r1.selection_id == f"sel-{r1.selection_digest[:16]}"

    def test_selection_record_dict_access_and_errors(self) -> None:
        rec = select_modes(snapshot_id="snap-1", available_modes=("architecture",))
        assert rec["snapshot_id"] == "snap-1"
        assert rec.get("policy_version") == "all_available"
        assert rec.get("non_existent_key", "default") == "default"

        with pytest.raises(KeyError):
            _ = rec["non_existent_key"]

    def test_selection_record_immutability_and_hashability_invariants(self) -> None:
        """CRITICAL CHALLENGE: SelectionRecord must maintain immutability and hashability invariants."""
        # 1. When tuple is passed (clean immutable sequence)
        rec_tuple = select_modes(snapshot_id="snap-t", available_modes=("architecture",))
        assert isinstance(rec_tuple.available_modes, tuple)
        assert isinstance(rec_tuple.selected_modes, tuple)

        # Hashability works when backed by tuples
        h = hash(rec_tuple)
        assert isinstance(h, int)
        s = {rec_tuple}
        assert rec_tuple in s

        # Reassignment blocked by frozen dataclass
        with pytest.raises(FrozenInstanceError):
            rec_tuple.selected_modes = ("malicious",)  # type: ignore

        # 2. When list is passed (common caller pattern):
        # ModeSequence preserves immutability & hashability while comparing equal to lists
        rec_list = select_modes(snapshot_id="snap-l", available_modes=["architecture"])

        # When list is used, hash() succeeds because ModeSequence is hashable
        h_list = hash(rec_list)
        assert isinstance(h_list, int)
        assert rec_list in {rec_list}

        # Value equality with lists works for compatibility
        assert rec_list.available_modes == ["architecture"]
        assert rec_list.selected_modes == ["architecture"]

        # Container mutation is blocked
        with pytest.raises((AttributeError, TypeError)):
            rec_list.selected_modes.append("malicious_injected_mode")  # type: ignore

        with pytest.raises((AttributeError, TypeError)):
            rec_list.available_modes.clear()  # type: ignore


# ==============================================================================
# 3. Snapshot Assembly & Invariants Adversarial Tests
# ==============================================================================


class DummyReader(RepositoryReader):
    def __init__(self, comparison_status: str = "comparable") -> None:
        self.comparison_status = comparison_status

    def get_commit(self, sha: str) -> Commit:
        return Commit(sha=sha, message="commit", committed_at="2026-10-06T00:00:00Z", parents=())

    def read_tree_blobs(self, sha: str, paths: tuple[str, ...]) -> dict[str, bytes]:
        return {}

    def compute_merge_base(self, base_sha: str, head_sha: str) -> str | None:
        return base_sha if self.comparison_status == "comparable" else None

    def resolve_merge_base(self, target_base_sha: str, head_sha: str) -> MergeBaseResult:
        if self.comparison_status == "comparable":
            return MergeBaseResult(
                target_base_sha=target_base_sha,
                comparison_base_sha=target_base_sha,
                comparison_status="comparable",
                candidates=(target_base_sha,),
            )
        elif self.comparison_status == "ambiguous":
            return MergeBaseResult(
                target_base_sha=target_base_sha,
                comparison_base_sha=None,
                comparison_status="ambiguous",
                candidates=("c1" * 20, "c2" * 20),
            )
        else:
            return MergeBaseResult(
                target_base_sha=target_base_sha,
                comparison_base_sha=None,
                comparison_status="unavailable",
                candidates=(),
            )

    def get_changed_files(self, base_sha: str, head_sha: str) -> tuple[ChangedFile, ...]:
        return (ChangedFile(path="src/Main.java", change_type="modified"),)

    def get_rename_map(self, base_sha: str, head_sha: str) -> dict[str, str]:
        return {"src/Old.java": "src/New.java"}


class TestSnapshotAssemblyAdversarial:
    def test_assemble_pr_snapshot_invariants(self) -> None:
        """PrSnapshot must store changed_files and labels as immutable tuples."""
        original_labels = ["리팩토링", "backend"]
        original_changes = ["src/A.java", ChangedFile("src/B.java")]
        original_renames = {"old.java": "new.java"}

        snap = assemble_pr_snapshot(
            repository="org/repo",
            number=1,
            base_sha="a" * 40,
            head_sha="b" * 40,
            source_branch="refactor/clean",
            labels=original_labels,
            changed_files=original_changes,
            rename_map=original_renames,
        )

        assert isinstance(snap.labels, tuple)
        assert isinstance(snap.changed_files, tuple)
        assert snap.changed_files == ("src/A.java", "src/B.java")

        # Mutating caller's original collections does not affect PrSnapshot
        original_labels.append("injected")
        assert "injected" not in snap.labels

        original_renames["tamper.java"] = "pwned.java"
        assert "tamper.java" not in snap.rename_map

    def test_assemble_review_snapshot_invariants_and_digests(self) -> None:
        """ReviewSnapshot digests must be deterministic SHA-256 strings."""
        mb = MergeBaseResult("a" * 40, "c" * 40, "comparable", ("c" * 40,))
        snap = assemble_review_snapshot(
            repository="org/repo",
            pr_number=99,
            repository_id=10,
            installation_id=20,
            target_base_sha="a" * 40,
            head_sha="b" * 40,
            merge_base_result=mb,
            title="Adversarial Title \n with linebreaks",
            body="Adversarial Body \x00 with null bytes",
            as_of="2026-10-06T12:00:00Z",
        )

        assert snap.schema_version == 2
        assert len(snap.snapshot_id) == 64
        assert len(snap.pr_text_digest) == 64
        assert len(snap.context_digest) == 64
        assert snap.as_of == "2026-10-06T12:00:00Z"

        # Frozen dataclass reassignment blocked
        with pytest.raises(FrozenInstanceError):
            snap.snapshot_id = "tampered"  # type: ignore

    def test_assemble_review_snapshot_mutable_collections_vulnerability(self) -> None:
        """Verify assemble_review_snapshot converts collections to tuple, preventing in-place mutation."""
        mb = MergeBaseResult("a" * 40, "c" * 40, "comparable", ("c" * 40,))
        mutable_unavail = [UnavailableSource(source="git", reason="timeout")]

        snap = assemble_review_snapshot(
            repository="org/repo",
            pr_number=99,
            repository_id=10,
            installation_id=20,
            target_base_sha="a" * 40,
            head_sha="b" * 40,
            merge_base_result=mb,
            unavailable_sources=mutable_unavail,  # type: ignore
        )

        assert isinstance(snap.unavailable_sources, tuple)
        with pytest.raises((AttributeError, TypeError)):
            snap.unavailable_sources.append(UnavailableSource(source="injected", reason="pwned"))  # type: ignore
        assert len(snap.unavailable_sources) == 1

    def test_assemble_snapshot_bundle_ambiguous_merge_base(self) -> None:
        """Criss-cross merge base must be incomparable and retain the ambiguity diagnostic."""
        reader = DummyReader(comparison_status="ambiguous")
        rev_snap, pr_snap, classification, selection = assemble_snapshot_bundle(
            repo_reader=reader,
            repository="org/repo",
            pr_number=5,
            repository_id=1,
            installation_id=1,
            target_base_sha="a" * 40,
            head_sha="b" * 40,
            source_branch="feature/criss-cross",
        )

        assert rev_snap.comparison_status == "incomparable"
        assert rev_snap.comparison_base_sha is None
        assert pr_snap.base_sha == ""
        assert rev_snap.changed_files == ()
        assert pr_snap.changed_files == ()
        assert pr_snap.rename_map == {}
        assert len(rev_snap.unavailable_sources) == 1
        assert rev_snap.unavailable_sources[0].source == "merge_base"
        assert "ambiguous" in rev_snap.unavailable_sources[0].reason
        assert classification.review_mode == "general"
        assert selection.selection_status == "completed"
