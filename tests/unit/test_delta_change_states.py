"""Unit tests for Feature 9: Delta 6 Change States & Feature 10: Conservative Rename Matching.

Requirements:
- requirements-a.md §A-03 & contracts.md §4:
  - 6 Change states: existing, new, worsened, improved, resolved, unknown.
  - Moving/renaming files must not fabricate false 'new' and false 'resolved' findings.
  - If comparison cannot be verified, status must be 'unknown'.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

try:
    from review_bot.application.delta import compute_delta  # type: ignore
except ImportError:
    compute_delta = None  # type: ignore


@pytest.fixture
def delta_engine() -> Callable[..., Any]:
    if compute_delta is None:
        pytest.skip("Delta computation engine review_bot.application.delta not yet implemented by Worker")
    return compute_delta


# ==============================================================================
# Tier 1: 6 Change States Computation
# ==============================================================================


class TestDeltaChangeStatesTier1:
    def test_new_finding_in_head_only(self, delta_engine: Callable) -> None:
        base_findings = []
        head_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 20, "method": "foo"}
        ]
        delta = delta_engine(base_findings=base_findings, head_findings=head_findings)
        signals = getattr(delta, "signals", delta.get("signals"))
        assert any(getattr(s, "change", s.get("change")) == "new" for s in signals)

    def test_resolved_finding_absent_in_head(self, delta_engine: Callable) -> None:
        base_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 20, "method": "foo"}
        ]
        head_findings = []
        delta = delta_engine(base_findings=base_findings, head_findings=head_findings)
        signals = getattr(delta, "signals", delta.get("signals"))
        assert any(getattr(s, "change", s.get("change")) == "resolved" for s in signals)

    def test_existing_finding_identical_in_both(self, delta_engine: Callable) -> None:
        base_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 20, "method": "foo"}
        ]
        head_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 20, "method": "foo"}
        ]
        delta = delta_engine(base_findings=base_findings, head_findings=head_findings)
        signals = getattr(delta, "signals", delta.get("signals"))
        assert any(getattr(s, "change", s.get("change")) == "existing" for s in signals)

    def test_worsened_finding_higher_complexity(self, delta_engine: Callable) -> None:
        base_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 16, "method": "foo"}
        ]
        head_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 28, "method": "foo"}
        ]
        delta = delta_engine(base_findings=base_findings, head_findings=head_findings)
        signals = getattr(delta, "signals", delta.get("signals"))
        assert any(getattr(s, "change", s.get("change")) == "worsened" for s in signals)

    def test_improved_finding_lower_complexity(self, delta_engine: Callable) -> None:
        base_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 28, "method": "foo"}
        ]
        head_findings = [
            {"rule": "CognitiveComplexity", "path": "A.java", "metric": 16, "method": "foo"}
        ]
        delta = delta_engine(base_findings=base_findings, head_findings=head_findings)
        signals = getattr(delta, "signals", delta.get("signals"))
        assert any(getattr(s, "change", s.get("change")) == "improved" for s in signals)


# ==============================================================================
# Tier 2: Rename Matching & Boundary Cases
# ==============================================================================


class TestDeltaRenameMatchingTier2:
    def test_renamed_file_matches_as_existing(self, delta_engine: Callable) -> None:
        base_findings = [
            {"rule": "CyclomaticComplexity", "path": "OldName.java", "metric": 12, "method": "bar"}
        ]
        head_findings = [
            {"rule": "CyclomaticComplexity", "path": "NewName.java", "metric": 12, "method": "bar"}
        ]
        rename_map = {"OldName.java": "NewName.java"}
        delta = delta_engine(
            base_findings=base_findings, head_findings=head_findings, rename_map=rename_map
        )
        signals = getattr(delta, "signals", delta.get("signals"))
        # Must NOT be marked as 'new' or 'resolved'
        changes = [getattr(s, "change", s.get("change")) for s in signals]
        assert "existing" in changes
        assert "resolved" not in changes
        assert "new" not in changes

    def test_incomparable_analysis_produces_unknown(self, delta_engine: Callable) -> None:
        delta = delta_engine(base_findings=[], head_findings=[], comparable=False)
        signals = getattr(delta, "signals", delta.get("signals"))
        for s in signals:
            assert getattr(s, "change", s.get("change")) == "unknown"
