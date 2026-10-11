"""Unit tests verifying Core v1 and Envelope v2 model contracts and immutability."""
from __future__ import annotations

import pytest

from review_bot.domain import (
    AnalysisDelta,
    ChangeState,
    DeltaSignal,
    ExecutionReceipt,
    ModeSelection,
    PrSnapshot,
    ReviewSnapshot,
)


def test_change_states_and_delta() -> None:
    assert ChangeState.EXISTING == "existing"
    assert ChangeState.NEW == "new"
    assert ChangeState.WORSENED == "worsened"
    assert ChangeState.IMPROVED == "improved"
    assert ChangeState.RESOLVED == "resolved"
    assert ChangeState.UNKNOWN == "unknown"

    delta = AnalysisDelta(
        schema_version=1,
        base_sha="0" * 40,
        head_sha="1" * 40,
        conditions_hash="abc",
        comparable=True,
        signals=(
            DeltaSignal("s1", "pmd", "new", (), (), ("A.java",), "exact"),
            DeltaSignal("s2", "cpd", "existing", (), (), ("B.java",), "exact"),
        ),
        limitations=(),
        analysis_status={"pmd": "findings", "cpd": "clean"},
    )
    assert delta.counts["new"] == 1
    assert delta.counts["existing"] == 1
    assert delta.counts["worsened"] == 0
    assert delta.counts["improved"] == 0
    assert delta.counts["resolved"] == 0
    assert delta.counts["unknown"] == 0


def test_envelope_v2_review_snapshot() -> None:
    snap = ReviewSnapshot(
        schema_version=2,
        snapshot_id="snap-123",
        repository="org/repo",
        pr_number=42,
        repository_id=101,
        installation_id=202,
        target_base_sha="a" * 40,
        head_sha="b" * 40,
        comparison_base_sha="c" * 40,
        comparison_status="comparable",
        captured_at="2026-10-06T12:00:00Z",
        as_of="2026-10-06T12:00:00Z",
        pr_text_digest="d1",
        context_digest="d2",
    )
    assert snap.schema_version == 2
    assert snap.target_base_sha != snap.comparison_base_sha
    assert snap.comparison_status == "comparable"


def test_immutability() -> None:
    pr = PrSnapshot(
        repository="org/repo",
        number=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        source_branch="feature",
        labels=(),
        changed_files=(),
        rename_map={},
    )
    with pytest.raises(AttributeError):
        pr.number = 2  # type: ignore[misc]


def test_envelope_v2_receipt_and_selection() -> None:
    selection = ModeSelection(
        selection_id="sel-1",
        snapshot_id="snap-123",
        policy_version="all_available",
        selection_digest="digest-abc",
        available_modes=("architecture",),
        selected_modes=("architecture",),
    )
    assert selection.selected_modes == ("architecture",)

    receipt = ExecutionReceipt(
        schema_version=2,
        mode="architecture",
        selection_id="sel-1",
        snapshot_id="snap-123",
        generation=1,
        fingerprint="fp-xyz",
        execution_status="completed",
        a_result=None,
    )
    assert receipt.execution_status == "completed"

