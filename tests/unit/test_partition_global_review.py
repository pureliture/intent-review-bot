"""Synthetic orchestration/provenance proof, not native semantic quality proof."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from review_bot.application.context import hash_json
from review_bot.application.partition_review import run_partition_review, validate_partition_result, prepare_originals
from review_bot.application.source_partition import PartitionError
from review_bot.domain.models_v1 import ModelResponse
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


def _run(*, transform=None, planning_transform=None, full_fits=False, focused_fits=True, max_calls=128, output_cap=512):
    def count(request):
        payload = json.loads(request.user_content)
        if payload["stage"] == "final_review":
            return 200 + 100 * len(payload["observations"])
        if payload["stage"] == "global_review":
            return 100 if (full_fits if "complete_originals" in payload else focused_fits) else 1000
        return 100
    client = PartitionModelClient(counter=count, transform=transform)
    client.profile = replace(client.profile, max_input_tokens=350, max_output_tokens=output_cap)
    client.supports_bounded_reasoning = output_cap > 8192
    task = make_task()
    result = run_partition_review(task, "intent", client, PartitionModelClient(transform=planning_transform),
        limits=PartitionReviewLimits(max_calls=max_calls, max_total_tokens=1000000))
    return task, result, client


def _select_fixed_value_source(payload, body):
    if payload["stage"] == "plan_fixed_sources":
        candidates = payload["state"]["candidate_inventory"]
        selected = next((row["evidence_id"] for row in candidates
            if row["path"] == "src/value.py" and row["version"] == "head"), None)
        body["choice"] = next(row["label"] for row in payload["options"]
            if row["key"] == ("source:" + selected if selected else "no_suitable_source"))
    return body


def _resolve_fixed_gap(payload, body):
    if "recheck" in payload:
        cite = payload["requested_originals"][0]["blocks"][0]["reference_id"]
        body["context_updates"] = {key: {"state": "resolved", "reason": "Synthetic explicit resolution from the full requested original.",
            "citations": [{"reference_id": cite}]} for key in payload["context_items"] if key.startswith("G")}
    return body


def test_bare_global_gap_gets_closed_tev_source_plan_and_original_preserving_recheck():
    calls = []
    def transform(payload, body):
        if payload["stage"] == "global_review":
            calls.append(deepcopy(payload))
            body["missing_context"] = [] if "recheck" in payload else ["Need src/value.py head return contract."]
            _resolve_fixed_gap(payload, body)
        return body
    task, result, _ = _run(transform=transform, planning_transform=_select_fixed_value_source)
    assert result.status == "completed", result.reason
    assert len(calls) == 2
    attempts = result.partition["global_review"]["attempts"]
    plan = attempts[0]["source_plan"]
    assert plan["complete"] and len(plan["requests"]) == 1
    original = next(o for o in prepare_originals(task, "intent")[0]
        if o.path == "src/value.py" and o.version == "head")
    assert plan["requests"][0]["evidence_id"] == original.evidence_id
    assert calls[1]["recheck"]["prior_response"] == attempts[0]["response"]
    assert attempts[0]["response"]["source_requests"] == []
    assert "".join(b["text"] for b in calls[1]["requested_originals"][0]["blocks"]) == original.text
    assert all("state" not in receipt for receipt in calls[1]["recheck"]["source_delivery_plan"]["decision_receipts"])
    assert any(d["stage"] == "plan_fixed_sources" for d in result.partition["decisions"])
    validate_partition_result(result, task)


def test_no_suitable_fixed_source_keeps_a_genuine_gap_without_forcing_a_positive_review():
    def missing(payload, body):
        if payload["stage"] == "global_review":
            body["missing_context"] = ["A necessary external fact is genuinely unavailable."]
        return body
    def no_source(payload, body):
        if payload["stage"] == "plan_fixed_sources":
            body["choice"] = next(row["label"] for row in payload["options"] if row["key"] == "no_suitable_source")
        return body
    task, result, client = _run(transform=missing, planning_transform=no_source)
    assert result.status == "partial_review" and result.partition["final"]["missing_context"]
    assert len([r for r in client.requests if json.loads(r.user_content)["stage"] == "global_review"]) == 1
    plan = result.partition["global_review"]["attempts"][0]["source_plan"]
    assert plan["complete"] and plan["requests"] == []
    validate_partition_result(result, task)


@pytest.mark.parametrize("field", ["requests", "gaps", "parent_plan_id", "complete", "plan_id"])
def test_fixed_source_plan_tampering_is_rejected(field):
    def missing(payload, body):
        if payload["stage"] == "global_review":
            body["missing_context"] = [] if "recheck" in payload else ["Need src/value.py head return contract."]
            _resolve_fixed_gap(payload, body)
        return body
    task, result, _ = _run(transform=missing, planning_transform=_select_fixed_value_source)
    assert result.status == "completed", result.reason
    changed = deepcopy(result)
    plan = changed.partition["global_review"]["attempts"][0]["source_plan"]
    if field in ("requests", "gaps"):
        plan[field].clear()
    elif field == "complete":
        plan[field] = False
    else:
        plan[field] = "changed"
    with pytest.raises(PartitionError, match="stored_fixed_source_"):
        validate_partition_result(changed, task)


def test_fixed_source_planning_cannot_exceed_shared_call_budget_and_preserves_prefix():
    _, baseline, _ = _run()
    def missing(payload, body):
        if payload["stage"] == "global_review":
            body["missing_context"] = ["Need src/value.py head return contract."]
        return body
    task, result, _ = _run(transform=missing, planning_transform=_select_fixed_value_source,
        max_calls=baseline.usage["calls"] - len(baseline.partition["fact_verification"]["items"]))
    assert result.status == "failed" and result.reason == "call_limit_exceeded"
    attempt = result.partition["global_review"]["attempts"][0]
    assert attempt["response"]["missing_context"]
    assert not attempt["source_plan"]["complete"] and not attempt["source_plan"]["decision_ids"]
    assert "final" not in result.partition
    validate_partition_result(result, task)


def test_fixed_source_candidates_cover_entire_inventory_with_bounded_closed_options():
    from review_bot.application import partition_global_review as global_review
    inventory = [{"evidence_id": "S" + str(i), "path": "src/a" + str(i) + ".py", "kind": "source",
        "version": "head", "sha": "h" * 40, "bytes": 100} for i in range(60)]
    raw = {"missing_context": ["Need source A.", "Need source B."]}
    specs = list(global_review.source_specs(inventory, raw, "parent"))
    assert len(specs) == 6
    for gap in raw["missing_context"]:
        pages = [s for s in specs if s["state"]["gap"] == gap]
        assert [o["evidence_id"] for spec in pages for o in spec["options"] if o["evidence_id"]] == [r["evidence_id"] for r in inventory]
        assert all(len(s["options"]) <= 24 and s["options"][-1]["key"] == "no_suitable_source" for s in pages)
        assert all(s["state"]["prior_global_response"] == raw for s in pages)


def test_source_recheck_inherits_an_evidenced_negative_global_context_state():
    calls = []
    def flow(payload, body):
        if payload["stage"] == "review_partition":
            body["missing_context"] = ["Need fixed head implementation."]
        if payload["stage"] == "global_review":
            calls.append(deepcopy(payload))
            if len(calls) == 1:
                key = next(iter(payload["context_items"]))
                cite = next(c for c in payload["citation_catalog"] if c["version"] == "head")
                body["context_updates"][key] = {"state": "unresolved",
                    "reason": "Synthetic new evidence retracts the earlier resolution.",
                    "citations": [{"reference_id": cite["ranges"][0]["reference_id"]}]}
                body["source_requests"] = [{"evidence_id": cite["evidence_id"], "reason": "Synthetic further source check."}]
        return body
    task, result, _ = _run(transform=flow)
    assert len(calls) == 2
    assert result.status == "partial_review" and result.partition["final"]["missing_context"]
    assert any(row["update"]["state"] == "unresolved" and row.get("global_attempt") == 0
        for row in calls[1]["prior_context_updates"].values())
    validate_partition_result(result, task)


@pytest.mark.parametrize("explicit_unresolved", [False, True])
def test_empty_missing_list_never_silently_resolves_a_registered_global_gap(explicit_unresolved):
    def flow(payload, body):
        if payload["stage"] == "global_review":
            body["missing_context"] = [] if "recheck" in payload else ["Need src/value.py head return contract."]
            if "recheck" in payload and explicit_unresolved:
                body["context_updates"] = {key: {"state": "unresolved", "reason": "Synthetic necessary external relation is still unknown.",
                    "citations": []} for key in payload["context_items"] if key.startswith("G")}
        return body
    task, result, _ = _run(transform=flow, planning_transform=_select_fixed_value_source)
    if explicit_unresolved:
        assert result.status == "partial_review" and result.partition["final"]["missing_context"]
        assert all(v["state"] == "unresolved" for k, v in result.partition["final"]["context_updates"].items() if k.startswith("G"))
    else:
        assert result.status == "failed" and result.reason == "global_context_accounting_incomplete"
        assert "final" not in result.partition
    validate_partition_result(result, task)


def test_all_original_page_responses_and_every_nested_cited_byte_reach_global_model():
    task, result, client = _run()
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    pages = result.partition["integration_pages"]
    wire = next(json.loads(r.user_content) for r in client.requests if json.loads(r.user_content)["stage"] == "global_review")
    assert len(pages) == len(wire["observations"]) > 1
    for page, row in zip(pages, wire["observations"]):
        assert row["raw_response"] == page["response"]
        assert row["raw_response_hash"] == hash_json(page["response"])
    known = {o.evidence_id: o for o in prepare_originals(task, "intent")[0]}
    refs = {}
    for source in wire["citation_catalog"]:
        original = known[source["evidence_id"]]
        for span in source["ranges"]:
            assert span["text"] == original.raw[span["start_byte"]:span["end_byte"]].decode()
            refs[span["reference_id"]] = span
    def visit(value):
        if isinstance(value, dict):
            if set(value) == {"reference_id"}:
                assert value["reference_id"] in refs
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(wire["observations"])
    attempts = result.partition["global_review"]["attempts"]
    stages = [result.stages[i] for i in attempts[0]["measurement_stages"]]
    assert len(stages) == 2 and stages[0].reason == "input_limit_exceeded"
    assert stages[0].measured_input_tokens == 1000 and stages[1].measured_input_tokens == 100
    assert "complete_originals" not in wire
    assert result.partition["integration_final"]["target_reviews"] == result.partition["final"]["target_reviews"]
    assert all(p["verdict"] == "aligned" for p in result.perspectives)


def test_fitting_full_request_is_not_replaced_by_citation_only_delivery():
    task, result, client = _run(full_fits=True)
    assert result.status == "completed", result.reason
    wire = next(json.loads(r.user_content) for r in client.requests if json.loads(r.user_content)["stage"] == "global_review")
    assert {o["evidence_id"]: "".join(b["text"] for b in o["blocks"]) for o in wire["complete_originals"]} == {
        o.evidence_id: o.text for o in prepare_originals(task, "intent")[0]}
    assert len(result.partition["global_review"]["attempts"][0]["measurement_stages"]) == 1
    validate_partition_result(result, task)


def _finding(payload, body):
    if payload["stage"] == "final_review" and not payload.get("prior_context_updates"):
        head = next((c for c in payload["citation_catalog"] if c["version"] == "head"), None)
        if head:
            body["findings"] = [{"statement": "Synthetic negative finding.", "reasoning": "Negative fixture judgement.",
                "failure_condition": "Synthetic failing condition.", "verdict": "concern", "basis": "observed_structure",
                "target_ids": payload["review_targets"][:1], "citations": [{"reference_id": head["ranges"][0]["reference_id"]}]}]
    return body


def test_global_model_can_correct_page_finding_without_erasing_its_original():
    def transform(payload, body):
        _finding(payload, body)
        if payload["stage"] == "global_review":
            for row in body["finding_dispositions"].values():
                row.update(disposition="excluded", reason="Synthetic source-backed correction.", result_indices=[])
            body["findings"] = []
            body["correction_reason"] = "Synthetic global correction of the earlier claim."
        return body
    task, result, _ = _run(transform=transform)
    assert result.status == "completed", result.reason
    assert result.partition["integration_final"]["findings"] and not result.findings
    assert result.partition["final"]["finding_dispositions"]
    assert all(r["disposition"] == "excluded" for r in result.partition["final"]["finding_dispositions"].values())
    assert any(p["response"]["findings"] for p in result.partition["integration_pages"])
    validate_partition_result(result, task)


def test_global_fixture_preserves_a_negative_finding_without_forced_positive_result():
    task, result, _ = _run(transform=_finding)
    assert result.status == "completed", result.reason
    assert any(f["verdict"] == "concern" for f in result.findings)
    assert all(r["disposition"] == "retained" for r in result.partition["final"]["finding_dispositions"].values())
    validate_partition_result(result, task)


@pytest.mark.parametrize("field", ["response", "payload_hash", "schema_hash", "stage_index", "delivery", "final"])
def test_stored_global_tampering_is_rejected(field):
    task, result, _ = _run(transform=_finding)
    assert result.status == "completed", result.reason
    changed = deepcopy(result)
    row = changed.partition["global_review"]["attempts"][0]
    if field == "response":
        row[field]["summary"] += "changed"
    elif field == "stage_index":
        row[field] = 0
    elif field == "final":
        row[field]["finding_dispositions"].clear()
    else:
        row[field] = "changed"
    with pytest.raises(PartitionError, match="stored_global_"):
        validate_partition_result(changed, task)


def test_unfit_intact_global_packet_keeps_prefix_without_promoting_page_findings():
    task, result, _ = _run(focused_fits=False, transform=_finding)
    assert result.status == "failed" and result.reason == "global_original_results_exceed_context"
    assert result.partition["integration_final"]["findings"]
    assert "final" not in result.partition and not result.findings
    validate_partition_result(result, task)


def test_unknown_global_usage_never_promotes_a_result_or_restarts_the_job():
    def transform(payload, body):
        if payload["stage"] == "global_review":
            return ModelResponse(json.dumps(body), None)
        return body
    task, result, client = _run(transform=transform)
    assert result.status == "failed" and result.reason == "usage_unavailable"
    assert len([r for r in client.requests if json.loads(r.user_content)["stage"] == "global_review"]) == 1
    assert "final" not in result.partition and result.partition["integration_pages"]
    validate_partition_result(result, task)


def test_global_correction_retains_the_entire_rejected_response():
    calls = []
    def transform(payload, body):
        if payload["stage"] == "global_review":
            calls.append(deepcopy(payload))
            if len(calls) == 1:
                body["finding_dispositions"]["unexpected"] = {}
        return body
    task, result, _ = _run(transform=transform)
    assert result.status == "completed", result.reason
    attempts = result.partition["global_review"]["attempts"]
    assert len(attempts) == 2
    assert calls[1]["recheck"]["rejected_response"] == attempts[0]["response"]
    assert attempts[0]["validation_error"]
    validate_partition_result(result, task)


def test_global_fixed_original_recheck_retains_previous_response_and_exact_source():
    calls = []
    def transform(payload, body):
        if payload["stage"] == "global_review":
            calls.append(deepcopy(payload))
            if len(calls) == 1:
                body["source_requests"] = [{"evidence_id": payload["fixed_original_inventory"][0]["evidence_id"],
                    "reason": "Synthetic fixed-original recheck."}]
        return body
    task, result, _ = _run(transform=transform)
    assert result.status == "completed", result.reason
    attempts = result.partition["global_review"]["attempts"]
    assert len(attempts) == len(calls) == 2
    assert calls[1]["recheck"] == attempts[0]["response"]
    requested = attempts[0]["response"]["source_requests"][0]["evidence_id"]
    source = next(o for o in prepare_originals(task, "intent")[0] if o.evidence_id == requested)
    assert len(calls[1]["requested_originals"]) == 1
    assert "".join(b["text"] for b in calls[1]["requested_originals"][0]["blocks"]) == source.text
    assert not result.partition["final"]["source_requests"]
    validate_partition_result(result, task)


def test_repeated_invalid_global_response_never_becomes_final():
    def transform(payload, body):
        if payload["stage"] == "global_review":
            body["observation_dispositions"].clear()
        return body
    task, result, client = _run(transform=transform)
    assert result.status == "failed"
    assert len([r for r in client.requests if json.loads(r.user_content)["stage"] == "global_review"]) == 2
    assert len(result.partition["global_review"]["attempts"]) == 2
    assert "final" not in result.partition
    validate_partition_result(result, task)


def test_global_stage_uses_the_existing_shared_call_budget():
    _, baseline, _ = _run()
    task, result, client = _run(max_calls=baseline.usage["calls"] - len(baseline.partition["fact_verification"]["items"]) - 1)
    assert result.status == "failed" and result.reason == "call_limit_exceeded"
    assert result.partition["integration_final"] and "final" not in result.partition
    assert not any(json.loads(r.user_content)["stage"] == "global_review" for r in client.requests)
    validate_partition_result(result, task)


def test_global_known_output_limit_uses_same_input_and_output_cap_recovery():
    from review_bot.application.architecture_review import ModelOutputLimitError
    calls = 0
    def transform(payload, body):
        nonlocal calls
        if payload["stage"] == "global_review":
            calls += 1
            if calls == 1:
                error = ModelOutputLimitError("Truncated global output")
                error.delivery = "sent"
                error.usage = {"input_tokens": 100, "output_tokens": 32678}
                error.rejected_output = {"finish_reason": "length", "content": '{"partial":',
                    "reasoning_content": "Original synthetic rejected channel."}
                raise error
        return body
    task, result, client = _run(transform=transform, output_cap=32678)
    assert result.status == "completed", result.reason
    requests = [r for r in client.requests if json.loads(r.user_content)["stage"] == "global_review"]
    assert len(requests) == 2
    assert (requests[0].system_prompt, requests[0].user_content, requests[0].response_schema) == (
        requests[1].system_prompt, requests[1].user_content, requests[1].response_schema)
    assert requests[0].max_output_tokens == requests[1].max_output_tokens == 32678
    assert requests[1].reasoning_budget_tokens == 24486
    assert result.partition["rejected_model_outputs"][0]["output"]["content"] == '{"partial":'
    validate_partition_result(result, task)


def test_malformed_global_citation_retains_a_readable_failed_result():
    def transform(payload, body):
        if payload["stage"] == "global_review":
            next(iter(body["observation_dispositions"].values()))["citations"] = [{}]
        return body
    task, result, _ = _run(transform=transform)
    assert result.status == "failed" and "final" not in result.partition
    assert result.partition["global_review"]["attempts"][0]["response"]
    validate_partition_result(result, task)


def test_global_input_schema_and_reference_order_are_stable_across_process_hash_seeds():
    import os
    import subprocess
    import sys
    code = """
import json
from review_bot.application.context import hash_json
from review_bot.application.source_partition import SourceOriginal, make_span
from review_bot.application.source_citations import citation_spans
from review_bot.application.partition_review import _global_spec
from review_bot.application import partition_global_review as g
original = SourceOriginal('Sfixture', ''.join('value_' + str(i) + ' = 1\\n' for i in range(200)),
    'code', 'src/a.py', 'head', 'h' * 40)
spans = citation_spans([make_span(original, 0, len(original.raw), 'fixture')], {original.evidence_id: original})
quotes = [{'evidence_id': s.evidence_id, 'start_byte': s.start_byte, 'end_byte': s.end_byte, 'quote': s.text}
    for s in spans.values()]
page = {'response': {'findings': [], 'citations': [{'reference_id': ref} for ref in spans]},
    'final': {'context_updates': {}, 'supplied_quotes': quotes}}
data = g.prepare([page], [original], {'target_reviews': {}, 'missing_context': []}, {},
    {'targets': [], 'target_paths': {}})
payload, schema, refs = _global_spec(data, 'intent', full=False, previous=None)
print(json.dumps([hash_json(payload), hash_json(schema), hash_json(list(refs))]))
"""
    values = [subprocess.check_output([sys.executable, "-c", code],
        env={**os.environ, "PYTHONHASHSEED": seed}, text=True).strip() for seed in ("1", "2")]
    assert values[0] == values[1]
