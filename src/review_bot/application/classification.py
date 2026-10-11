"""PR perspective classifier determining refactoring vs general review posture."""
from __future__ import annotations

from collections.abc import Iterable

from review_bot.domain.models_v1 import Classification
from review_bot.domain.status import ReviewMode


def classify_pr(
    source_branch: str,
    labels: tuple[str, ...] | list[str] | Iterable[str] = (),
    title: str | None = None,
) -> Classification:
    """Classify PR perspective into 'refactoring' or 'general'.

    Requirements:
    - Branch prefix 'refactor/' or 'refact/' (case-insensitive) -> refactoring
    - Label containing exact '리팩토링' -> refactoring
    - Otherwise -> general
    - Title is strictly ignored per requirements-a §A-02
    """
    if not isinstance(source_branch, str):
        source_branch = ""

    norm_branch = source_branch.lower()
    is_refactor_branch = norm_branch.startswith(("refactor/", "refact/"))

    labels_tuple = tuple(labels) if labels is not None else ()
    has_refactor_label = "리팩토링" in labels_tuple

    reasons: list[str] = []
    if is_refactor_branch:
        reasons.append(f"Branch prefix matched refactor/ or refact/: {source_branch}")
    if has_refactor_label:
        reasons.append("Label '리팩토링' present")

    if is_refactor_branch or has_refactor_label:
        return Classification(
            review_mode=ReviewMode.REFACTORING.value,
            reasons=tuple(reasons),
        )
    else:
        return Classification(
            review_mode=ReviewMode.GENERAL.value,
            reasons=("No refactoring branch prefix or '리팩토링' label found",),
        )
