"""Synthetic factual error controls exercise the real execution/validation seam."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from review_bot.application.context import hash_json, build_task_input
from review_bot.application import review_fact_check as facts
from review_bot.application.partition_review import run_partition_review, validate_partition_result
from review_bot.application.source_partition import PartitionError
from review_bot.application.delta import compute_delta
from review_bot.domain.models_v1 import Classification
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


def limits(**kwargs):
    return PartitionReviewLimits(max_calls=128, max_total_tokens=1000000, **kwargs)


def _reference(payload, path=None):
    return next(block["reference_id"] for source in payload["originals"]
        if source["version"] == "head" and (path is None or source["path"] == path)
        for block in source["blocks"])


def _metric_task():
    from tests.unit.test_context_and_architecture_review import _make_snapshot
    snapshot = _make_snapshot({"src/A.java": "class A {}\n"},
        {"src/A.java": "class A { int x; void run() { x++; } }\n"})
    delta = compute_delta(base_findings=[], head_findings=[
        {"rule": "GodClass", "path": "src/A.java", "metric": 80, "method": None}],
        base_sha=snapshot.base.sha, head_sha=snapshot.head.sha)
    return build_task_input(snapshot, delta, Classification("general"))


def _metric_run(*, incorrect_rewrite=False, corrupt=None):
    def review_transform(payload, body):
        stage = payload["stage"]
        if stage == "final_review" and payload["review_targets"]:
            head = next(s for s in payload["original_quotes"] if s["version"] == "head" and s["path"] == "src/A.java")
            body["findings"] = [{"verdict": "improvement", "statement": "Inspect cohesion.",
                "reasoning": "Low TCC means coupling is not severe.", "failure_condition": "",
                "basis": "observed_structure",
                "target_ids": list(payload["review_targets"]),
                "citations": [{"reference_id": head["blocks"][0]["reference_id"]}]}]
        elif stage == "fact_rewrite":
            replacement = deepcopy(payload["items"][0]["original_item"])
            replacement["reasoning"] = ("Low TCC means coupling is not severe." if incorrect_rewrite else
                "Low TCC means low cohesion; it cannot establish low coupling.")
            replacement["citations"] = [{"reference_id": _reference(payload, "src/A.java")}]
            body = {"replacement": replacement, "correction_reason": "Correct the metric category against its canonical definition.",
                "correction_citations": replacement["citations"]}
        return corrupt(payload, body) if corrupt else body

    def extractor(payload, body):
        if payload["stage"] == "fact_metric":
            state = payload["state"]
            prose = "\n".join(state["prose"].values())
            category = "coupling" if "coupling is not severe" in prose else "cohesion"
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == category)
        return body

    task = _metric_task()
    client, tev = PartitionModelClient(transform=review_transform), PartitionModelClient(transform=extractor)
    result = run_partition_review(task, "architecture", client, tev, limits=limits())
    return task, result, client, tev


def test_typed_metric_comparison_catches_error_even_when_broad_checker_says_supported():
    task, result, client, tev = _metric_run()
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    ledger = result.partition["fact_verification"]
    entry = next(e for e in ledger["items"] if e["item_id"] == "findings:0")
    assert entry["original"]["reasoning"] == "Low TCC means coupling is not severe."
    first, second = entry["attempts"]
    assert all(a["state"] in ("supported", "non_factual") for a in first["assessments"].values())
    assert first["issues"][0]["expected"] == "cohesion" and first["issues"][0]["asserted"] == "coupling"
    assert second["metric_checks"][0]["selected_key"] == "cohesion"
    assert not second["issues"] and entry["status"] == "model_supported"
    assert result.findings[0]["reasoning"] == second["item"]["reasoning"]
    assert ledger["original_final"] == result.partition["pre_verification_final"]
    assert result.partition["pre_verification_final"] != result.partition["final"]
    assert result.scope["semantic_retention_verified"] is False
    assert len([r for r in client.requests if json.loads(r.user_content)["stage"] == "fact_rewrite"]) == 1
    assert all(set(r.response_schema["properties"]) == {"choice"} for r in tev.requests)


def test_repeated_false_rewrite_remains_explicitly_unverified_and_partial():
    task, result, _, _ = _metric_run(incorrect_rewrite=True)
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    assert not result.scope["review_complete"]
    entry = next(e for e in result.partition["fact_verification"]["items"] if e["item_id"] == "findings:0")
    assert len(entry["attempts"]) == 2 and entry["status"] == "unverified"
    assert entry["original"]["reasoning"] == entry["accepted"]["reasoning"]
    validate_partition_result(result, task)


def test_intent_fact_check_corrects_false_execution_claim_and_rechecks_without_changing_eight_perspectives():
    def transform(payload, body):
        stage = payload["stage"]
        if stage == "final_review":
            body["summary"] = "The target tests passed."
        elif stage == "fact_check" and payload["items"][0]["original_item"] == "The target tests passed.":
            assert payload["execution_facts"]["target_tests_executed_by_bot"] is False
            body["items"]["summary"]["text"].update(state="contradicted", reason="No target tests were executed by this review.")
        elif stage == "fact_rewrite":
            body = {"replacement": "Static test source was reviewed; target tests were not executed.",
                "correction_reason": "Do not claim runtime success from static assertions.",
                "correction_citations": [{"reference_id": _reference(payload)}]}
        return body
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(transform=transform), PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert "not executed" in result.summary and len(result.perspectives) == 8
    ledger = result.partition["fact_verification"]
    assert len(ledger["items"]) == 9 and len(ledger["items"][0]["attempts"]) == 2
    assert ledger["original_final"]["summary"] == "The target tests passed."
    validate_partition_result(result, task)


@pytest.mark.parametrize("state", ["contradicted", "unverifiable"])
def test_source_check_rewrites_only_the_failing_item_and_preserves_all_other_results(state):
    def transform(payload, body):
        if payload["stage"] == "fact_check" and payload["items"][0]["item_id"] == "perspectives:2":
            body["items"]["perspectives:2"]["reasoning"].update(state=state, reason="Precise unresolved or contradictory premise.")
        if payload["stage"] == "fact_rewrite":
            item = deepcopy(payload["items"][0]["original_item"])
            item["reasoning"] = "Explicit limitation remains; this cannot be established."
            item["citations"] = [{"reference_id": _reference(payload)}]
            body = {"replacement": item, "correction_reason": "Preserve the limitation.", "correction_citations": item["citations"]}
        return body
    task = make_task()
    client = PartitionModelClient(transform=transform)
    result = run_partition_review(task, "intent", client, PartitionModelClient(), limits=limits())
    assert result.status == "partial_review"
    rewrites = [json.loads(r.user_content) for r in client.requests if json.loads(r.user_content)["stage"] == "fact_rewrite"]
    assert [r["items"][0]["item_id"] for r in rewrites] == ["perspectives:2"]
    original = result.partition["pre_verification_final"]
    for index, item in enumerate(result.perspectives):
        if index != 2:
            assert item == original["perspectives"][index]
    validate_partition_result(result, task)


def test_fact_checks_cannot_bypass_existing_task_budget_or_promote_unchecked_items():
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_calls=15, max_total_tokens=1000000))
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    assert result.partition["fact_verification"]["interruption_reason"] == "call_limit_exceeded"
    assert result.usage["calls"] <= 15 and result.usage["usage_complete"]
    assert result.partition["final"] == result.partition["pre_verification_final"]
    validate_partition_result(result, task)


@pytest.mark.parametrize("change", ["raw_check", "metric", "accepted", "drop_item", "original", "erase_stage", "focus_without_overflow"])
def test_factual_trace_and_rewrite_tampering_are_rejected(change):
    task, result, _, _ = _metric_run()
    result = deepcopy(result)
    ledger = result.partition["fact_verification"]
    entry = next(e for e in ledger["items"] if e["item_id"] == "findings:0")
    if change == "raw_check":
        entry["attempts"][0]["check"]["response"]["items"]["findings:0"]["reasoning"]["reason"] = "changed"
    elif change == "metric":
        entry["attempts"][0]["metric_checks"][0]["issue"] = None
    elif change == "accepted":
        entry["accepted"]["reasoning"] = "changed"
    elif change == "drop_item":
        ledger["items"].pop()
    elif change == "original":
        ledger["original_final"]["summary"] = "changed"
    elif change == "erase_stage":
        entry["attempts"][0]["check"]["stage_index"] = 0
    else:
        entry["attempts"][0]["check"]["measurements"][0]["delivery"] = "cited_ranges"
    with pytest.raises(PartitionError):
        validate_partition_result(result, task)


@pytest.mark.parametrize("field", ["target_ids", "citations", "correction_reason"])
def test_rewrite_cannot_change_target_identity_or_drop_evidence(field):
    def corrupt(payload, body):
        if payload["stage"] == "fact_rewrite":
            if field == "target_ids":
                body["replacement"][field] = ["foreign-target"]
            elif field == "citations":
                body["replacement"][field] = []
            else:
                body[field] = ""
        return body
    task, result, _, _ = _metric_run(corrupt=corrupt)
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    assert result.partition["final"] == result.partition["pre_verification_final"]
    validate_partition_result(result, task)


def test_every_nested_quote_and_common_purpose_survive_focused_fact_delivery():
    from review_bot.application.partition_review import prepare_originals
    task, result, _, _ = _metric_run()
    originals, meta = prepare_originals(task, "architecture")
    final = result.partition["pre_verification_final"]
    row = next(r for r in facts.items(final) if r["id"] == "findings:0")
    payload, refs, _ = facts.delivery([row], final, originals, meta, kind="cited_ranges")
    by_id = {o.evidence_id: o for o in originals}
    spans = facts.quote_spans(row["item"], by_id)
    assert all("".join(b["text"] for b in s["blocks"]) == by_id[s["evidence_id"]].text
        for s in payload["originals"] if by_id[s["evidence_id"]].kind == "pr_text")
    assert all(any(r.evidence_id == s.evidence_id and r.start_byte <= s.start_byte < s.end_byte <= r.end_byte
        for r in refs.values()) for s in spans)
    assert set(refs) == {r["reference_id"] for source in payload["citation_catalog"] for r in source["ranges"]}


def test_extractor_does_not_receive_canonical_answer_and_non_metric_prose_is_not_classified():
    row = {"id": "findings:0", "item": {"statement": "TCC is low.", "reasoning": "ATFD measures foreign data.", "failure_condition": ""}}
    specs = list(facts.metric_specs(row))
    assert [s["state"]["target_metric"] for s in specs] == ["TCC", "ATFD"]
    assert all("canonical" not in s["state"] and "expected" not in s["state"] for s in specs)
    assert not list(facts.metric_specs({**row, "item": {"statement": "Rename function.", "reasoning": "Structural suggestion.", "failure_condition": ""}}))


@pytest.mark.parametrize("prose", ["TCC=0.07, WMC=80", "ATFD 9, WMC 80, TCC 0.07이다.", "TCC: 7e-2"])
def test_literal_metric_values_do_not_assert_a_meaning_category(prose):
    row = {"id": "summary", "kind": "summary", "item": prose}
    assert list(facts.metric_specs(row)) == []
    schema = facts.check_schema([row], {"Qtest": None})
    assert "summary" in schema["properties"]["items"]["required"]  # value check is still required


def test_literal_grammar_never_skips_a_meaning_claim_next_to_values():
    row = {"id": "summary", "kind": "summary", "item": "TCC 0.07이다. Therefore coupling is low."}
    assert [s["state"]["target_metric"] for s in facts.metric_specs(row)] == ["TCC"]


def test_cpd_rewrite_preserves_both_partner_evidence_requirement():
    from review_bot.application.source_partition import SourceOriginal, make_span
    from review_bot.application.source_citations import citation_spans
    original = SourceOriginal("S-a", "class A {}\n", "code", "A.java", "head", "a" * 40)
    refs = citation_spans([make_span(original, 0, len(original.raw), "test")], {original.evidence_id: original})
    cite = [{"reference_id": next(iter(refs))}]
    row = {"id": "findings:0", "kind": "findings", "item": {"verdict": "improvement", "basis": "observed_structure",
        "statement": "Duplicate structure.", "reasoning": "Both sides need source review.", "failure_condition": "",
        "target_ids": ["cpd-test"], "citations": []}}
    replacement = {**row["item"], "verdict": "concern", "failure_condition": "Concrete conditional failure.", "citations": cite}
    raw = {"replacement": replacement, "correction_reason": "Changed with evidence.", "correction_citations": cite}
    with pytest.raises(PartitionError, match="cpd_concern_without_partner_originals"):
        facts.accept_rewrite(raw, row, refs, "architecture", {"target_paths": {"cpd-test": ["A.java", "B.java"]}})


def test_invalid_checker_reference_is_retained_in_readable_partial_result():
    def corrupt(payload, body):
        if payload["stage"] == "fact_check":
            item = payload["items"][0]["item_id"]
            field = next(iter(body["items"][item]))
            body["items"][item][field]["citations"] = [{"reference_id": "Q-forged"}]
        return body
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(transform=corrupt), PartitionModelClient(), limits=limits())
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    ledger = result.partition["fact_verification"]
    assert ledger["interruption_reason"] == "fact_check_unknown_reference"
    assert ledger["items"][0]["attempts"][0]["check"]["response"]["items"]
    assert result.partition["final"] == result.partition["pre_verification_final"]
    validate_partition_result(result, task)


@pytest.mark.parametrize("component", ["facts", "eligibility"])
def test_worker_mode_fingerprints_include_verification_and_eligibility_implementations(valid_config_yaml, tmp_path, monkeypatch, component):
    from review_bot.config import load_review_config_from_str
    from review_bot.runtime.mode_fingerprints import architecture_input_fingerprint, intent_input_fingerprint
    from review_bot.application import selection
    config, _ = load_review_config_from_str(valid_config_yaml)
    config = config.model_copy(update={"partition_evaluation": config.partition_evaluation.model_copy(update={"enabled": True})})
    task = make_task()
    a_task = _metric_task()
    client = PartitionModelClient()
    def fingerprints():
        return (intent_input_fingerprint(task, config, client),
            architecture_input_fingerprint(a_task.snapshot, Classification("general"), config, client, title="PR", body="Purpose"))
    before = fingerprints()
    assert all(before)
    changed = tmp_path / "changed_fact_check.py"
    changed.write_text("# a different factual verification implementation\n")
    monkeypatch.setattr(facts if component == "facts" else selection, "__file__", str(changed))
    after = fingerprints()
    assert all(after) and all(left != right for left, right in zip(before, after))
