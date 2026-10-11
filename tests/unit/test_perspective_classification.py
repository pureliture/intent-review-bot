"""Unit tests for Feature 3: PR Perspective Classification.

Requirements:
- requirements-a.md §A-02:
  - Branch starts with 'refactor/' or 'refact/' -> 'refactoring'
  - Labels contains '리팩토링' -> 'refactoring'
  - Otherwise -> 'general'
  - Title is strictly ignored.
  - Case-insensitive on branch prefixes (refactor/, REFACTOR/, ReFaCt/).
  - Exact prefix match required ('refactoring/' does NOT match).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

# Try importing classification function if implemented
try:
    from review_bot.application.classification import classify_pr  # type: ignore
except ImportError:
    try:
        # Alternative layout candidate
        from review_bot.application.context import classify_pr  # type: ignore
    except ImportError:
        classify_pr = None  # type: ignore


# Reference oracle function representing the strict authoritative specification
def spec_classify_pr(source_branch: str, labels: tuple[str, ...] | list[str]) -> str:
    norm_branch = (source_branch or "").lower()
    is_refactor_branch = norm_branch.startswith(("refactor/", "refact/"))
    has_refactor_label = "리팩토링" in (labels or [])
    return "refactoring" if (is_refactor_branch or has_refactor_label) else "general"


@pytest.fixture
def classifier() -> Callable[[str, Any], Any]:
    if classify_pr is None:
        pytest.skip("PR perspective classifier not yet implemented in review_bot")
    return classify_pr


def _extract_mode(result: Any) -> str:
    """Extract review_mode or perspective from domain result object or string."""
    if isinstance(result, str):
        return result
    if hasattr(result, "review_mode"):
        return result.review_mode
    if hasattr(result, "perspective"):
        return result.perspective
    raise TypeError(f"Unexpected classification result type: {type(result)}")


# ==============================================================================
# Tier 1: Happy Path Tests
# ==============================================================================


class TestPerspectiveClassificationTier1:
    @pytest.mark.parametrize(
        "branch,labels,expected",
        [
            ("refactor/user-service", (), "refactoring"),
            ("refact/auth-tokens", (), "refactoring"),
            ("feature/payments", ("리팩토링",), "refactoring"),
            ("fix/null-pointer", ("bug", "리팩토링"), "refactoring"),
            ("feature/search-engine", (), "general"),
            ("main", ("enhancement",), "general"),
        ],
    )
    def test_standard_classifications(
        self, classifier: Callable, branch: str, labels: tuple[str, ...], expected: str
    ) -> None:
        result = classifier(branch, labels)
        assert _extract_mode(result) == expected

    def test_oracle_matches_spec(self) -> None:
        assert spec_classify_pr("refactor/core", ()) == "refactoring"
        assert spec_classify_pr("feature/core", ()) == "general"


# ==============================================================================
# Tier 2: Boundary, Edge Cases & Adversarial Tests
# ==============================================================================


class TestPerspectiveClassificationTier2:
    @pytest.mark.parametrize(
        "branch,expected",
        [
            ("REFACTOR/UPPERCASE", "refactoring"),
            ("ReFaCt/MixedCase", "refactoring"),
            ("refact/nested/deep/path", "refactoring"),
            ("refactor/", "refactoring"),
            ("refact/", "refactoring"),
        ],
    )
    def test_case_insensitivity_and_minimal_prefixes(
        self, classifier: Callable, branch: str, expected: str
    ) -> None:
        result = classifier(branch, ())
        assert _extract_mode(result) == expected

    @pytest.mark.parametrize(
        "branch",
        [
            "refactoring/almost",
            "feature/refactor/subpath",
            "refact",
            "refactor",
            "my-refactor/branch",
            "chore/refact",
        ],
    )
    def test_negative_branch_prefixes(self, classifier: Callable, branch: str) -> None:
        result = classifier(branch, ())
        assert _extract_mode(result) == "general"

    @pytest.mark.parametrize(
        "label",
        [
            " 리팩토링",
            "리팩토링 ",
            "refactoring",
            "REFACTORING",
            "Refactor",
            "리팩토링/아키텍처",
        ],
    )
    def test_negative_label_variations(self, classifier: Callable, label: str) -> None:
        result = classifier("feature/test", (label,))
        assert _extract_mode(result) == "general"

    def test_pr_title_is_strictly_ignored(self, classifier: Callable) -> None:
        # Title mentioning refactor must NOT classify as refactoring if branch is normal
        sig = inspect.signature(classifier)
        assert "title" not in sig.parameters or sig.parameters["title"].default is not inspect.Parameter.empty, (
            "Title must not be a required parameter"
        )
        result = classifier("feature/normal-flow", ())
        assert _extract_mode(result) == "general"

    def test_empty_and_minimal_inputs(self, classifier: Callable) -> None:
        assert _extract_mode(classifier("", ())) == "general"
        assert _extract_mode(classifier("main", [])) == "general"

    def test_dynamic_transition_on_label_change(self, classifier: Callable) -> None:
        branch = "feature/checkout"
        # 1. Initially without label
        assert _extract_mode(classifier(branch, ())) == "general"
        # 2. Label added
        assert _extract_mode(classifier(branch, ("리팩토링",))) == "refactoring"
        # 3. Label removed
        assert _extract_mode(classifier(branch, ())) == "general"
