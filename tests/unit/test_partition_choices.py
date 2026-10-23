from __future__ import annotations

from collections import Counter

import pytest

from review_bot.application import partition_choices as choices
from review_bot.application.source_partition import SourceOriginal, make_span, PartitionError


def _uncached_candidates(units, fits):
    """Original candidate construction, independent of predicate memoization."""
    paired = sorted(units, key=lambda u: (u.path, u.start_byte, u.version, u.evidence_id))
    strategies = [
        ("paired_pack", "Pack related base/head ranges together where the planning tokenizer permits.", paired, False),
        ("per_original", "Keep source versions and originals in separate bounded groups.", units, True),
        ("individual_units", "Read one validated AST/fallback source unit in each request.", units, None),
    ]
    candidates, seen = [], set()
    for key, description, order, isolate in strategies:
        groups, active = [], []
        for unit in order:
            candidate = [*active, unit]
            split = active and (isolate is None or isolate and active[-1].evidence_id != unit.evidence_id)
            if split or len(candidate) > choices.MAX_GROUP_UNITS or not fits(candidate):
                if active:
                    groups.append(active)
                active = [unit]
                if not fits(active):
                    raise PartitionError("candidate_unit_exceeds_consumer_limit")
            else:
                active = candidate
        if active:
            groups.append(active)
        signature = tuple(sorted(tuple(sorted(u.unit_id for u in group)) for group in groups))
        if signature not in seen:
            seen.add(signature)
            candidates.append({"key": key, "description": description,
                "groups": [[u.unit_id for u in group] for group in groups]})
    return choices.label_options(candidates)


@pytest.mark.parametrize("max_units", [2, 5])
def test_candidate_feasibility_is_deduplicated_only_within_each_invocation(max_units):
    original = SourceOriginal("S1", "abcdefghij", "source", "src/A.py", "head", "b" * 40)
    units = [make_span(original, i, i + 2, "line_fallback") for i in range(0, 10, 2)]
    reference_calls, calls = Counter(), Counter()
    def predicate(group, counts):
        counts[tuple(u.unit_id for u in group)] += 1
        return len(group) <= max_units
    expected = _uncached_candidates(units, lambda group: predicate(group, reference_calls))
    assert max(reference_calls.values()) > 1
    for _ in range(2):
        calls.clear()
        actual = choices.plan_candidates(units, lambda group: predicate(group, calls))
        assert actual == expected
        assert calls.keys() == reference_calls.keys()
        assert set(calls.values()) == {1}
        assert all(len(group) <= max_units for candidate in actual for group in candidate["groups"])
        if max_units == 2:
            assert any(len(group) > max_units for group in calls)


def test_candidate_feasibility_preserves_order_in_its_memo_key():
    original = SourceOriginal("S1", "abcd", "source", "src/A.py", "head", "b" * 40)
    first = make_span(original, 0, 2, "line_fallback")
    second = make_span(original, 2, 4, "line_fallback")
    calls = Counter()
    def fits(group):
        key = tuple(u.unit_id for u in group)
        calls[key] += 1
        return key != (second.unit_id, first.unit_id)
    actual = choices.plan_candidates([second, first], fits)
    assert calls[(first.unit_id, second.unit_id)] == 1
    assert calls[(second.unit_id, first.unit_id)] == 1
    assert set(calls.values()) == {1}
    assert actual == _uncached_candidates([second, first], lambda group:
        tuple(u.unit_id for u in group) != (second.unit_id, first.unit_id))


def test_candidate_feasibility_exceptions_propagate_unchanged():
    original = SourceOriginal("S1", "abc", "source")
    unit = make_span(original, 0, 3, "whole_original")
    failure = RuntimeError("native feasibility failed")
    def fits(_):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        choices.plan_candidates([unit], fits)
    assert caught.value is failure


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


@pytest.mark.parametrize("raw", [None, {}, {"selected_perspective_ids": "problem_success"},
    {"selected_perspective_ids": [], "extra": True}, {"selected_perspective_ids": ["unknown"]},
    {"selected_perspective_ids": ["problem_success", "problem_success"]}])
def test_subset_rejects_invalid_shapes_ids_and_duplicates(raw):
    from review_bot.application.partition_choices import accept_perspective_subset
    with pytest.raises(PartitionError, match="invalid_perspective_subset"):
        accept_perspective_subset(raw)
