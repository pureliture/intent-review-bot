from __future__ import annotations

import pytest

from review_bot.application import partition_choices as choices
from review_bot.application.source_partition import SourceOriginal, make_span, PartitionError


def test_candidates_are_deterministic_complete_unique_and_natively_validated():
    original = SourceOriginal("S1", "abcdefghij", "source", "src/A.py", "head", "b" * 40)
    units = [make_span(original, i, i + 2, "line_fallback") for i in range(0, 10, 2)]
    probes = []
    def fits(group):
        probes.append(tuple(u.unit_id for u in group))
        return len(group) <= 2
    first = choices.plan_candidates(units, fits)
    assert first == choices.plan_candidates(units, fits)
    assert len(first) >= 2
    for candidate in first:
        groups = candidate["groups"]
        assert sorted(u for g in groups for u in g) == sorted(u.unit_id for u in units)
        assert all(len(g) <= 2 and tuple(g) in probes for g in groups)


def test_native_rejection_does_not_offer_an_unsafe_candidate():
    original = SourceOriginal("S1", "abc", "source")
    unit = make_span(original, 0, 3, "whole_original")
    with pytest.raises(PartitionError, match="unit_exceeds"):
        choices.plan_candidates([unit], lambda _: False)


def test_global_mode_options_allow_source_based_correction_to_every_eligible_subset():
    options = choices.mode_options(("architecture", "intent"))
    assert len(options) == 4
    assert all(row["key"] != "undetermined" and "uncertain" not in row for row in options)
    assert {tuple(row["modes"]) for row in options} == {
        ("architecture", "intent"), ("architecture",), ("intent",), ()}
    assert all("Java+Gradle" in row["description"] for row in options if "architecture" in row["modes"])
    assert all("PR purpose" in row["description"] for row in options if "intent" in row["modes"])


def test_closed_choice_accepts_exactly_one_known_label():
    options = choices.mode_options(("intent",))
    with pytest.raises(ValueError):
        choices.accept_choice({"choice": "A", "explanation": "not permitted"}, options)
    with pytest.raises(PartitionError, match="invalid_closed_choice"):
        choices.accept_choice({"choice": "Z"}, options)
