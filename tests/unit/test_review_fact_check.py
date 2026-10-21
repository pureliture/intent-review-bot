"""Synthetic factual error controls exercise the real execution/validation seam."""
from copy import deepcopy
from dataclasses import replace
import json
import hashlib

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


@pytest.mark.parametrize("suffix,content,symbol,cut", [
    ("py", "def visible():\n    return True\n\ndef hidden_guard():\n    return False\n", "hidden_guard", "def hidden_guard"),
    ("java", "class Worker {\n void visible() {}\n void hiddenGuard() {}\n}\n", "hiddenGuard", " void hiddenGuard"),
])
def test_callable_delivery_guard_uses_only_production_head_code_without_changing_raw_state(suffix, content, symbol, cut):
    from review_bot.application.source_partition import SourceOriginal
    head = SourceOriginal("head", content, "source", "src/Worker." + suffix, "head", "a" * 40)
    base = SourceOriginal("base", "def legacy_only(): pass\n", "source", "src/legacy.py", "base", "b" * 40)
    test = SourceOriginal("test", "def commit(): pass\n", "source", "tests/test_fake.py", "head", "a" * 40)
    prose = SourceOriginal("pr", "The claim mentions " + symbol, "pr_text")
    sources = [head, base, test, prose]
    row = {"id": "summary", "kind": "summary", "item": "The route mentions " + symbol + "."}
    raw = {"text": {"state": "supported", "reason": "Original unchanged model assessment.", "citations": []}}
    snapshot = deepcopy(raw)
    partial = [{"evidence_id": "head", "start_byte": 0, "end_byte": content.index(cut)},
        {"evidence_id": "pr", "start_byte": 0, "end_byte": len(prose.raw)}]
    gaps = facts.callable_delivery_issues(row, raw, sources, partial)
    assert raw == snapshot
    assert gaps[0]["state"] == "unverifiable" and gaps[0]["missing_callable_names"] == [symbol]
    assert [request["evidence_id"] for request in gaps[0]["required_sources"]] == ["head"]
    assert not facts.callable_delivery_issues({**row, "item": "commit legacy_only unknown_name"}, raw, sources, partial)
    assert not facts.callable_delivery_issues(row, {"text": {**raw["text"], "state": "non_factual"}}, sources, partial)
    # A complete definition clears only delivery, never proves a call route.
    assert not facts.callable_delivery_issues(row, raw, sources,
        [{"evidence_id": "head", "start_byte": 0, "end_byte": len(head.raw)}])


@pytest.mark.parametrize("delivery,expected_gap", [("callsite_and_base", True), ("one_byte_gap", True), ("adjacent_complete", False)])
@pytest.mark.parametrize("state", ["supported", "contradicted", "unverifiable"])
def test_callable_delivery_requires_same_head_complete_definition_extent(delivery, expected_gap, state):
    from review_bot.application.source_partition import SourceOriginal
    text = 'def trace(func):\n    return func\n\n@trace\nasync def hidden_guard():\n    return "한글"\n\ndef caller():\n    return hidden_guard()\n'
    head = SourceOriginal("head", text, "source", "src/service.py", "head", "a" * 40)
    base = SourceOriginal("base", text, "source", "src/service.py", "base", "b" * 40)
    row = {"id": "summary", "kind": "summary", "item": "The caller uses hidden_guard."}
    raw = {"text": {"state": state, "reason": "Original raw judgement.", "citations": []}}
    bounds = lambda key, start, end: {"evidence_id": key, "start_byte": start, "end_byte": end}
    if delivery == "callsite_and_base":
        ranges = [bounds("head", head.raw.index(b"def caller"), len(head.raw)), bounds("base", 0, len(base.raw))]
    elif delivery == "one_byte_gap":
        end = head.raw.index('"한글"'.encode()) + len('"한글"'.encode())
        ranges = [bounds("head", 0, end - 1), bounds("head", end, len(head.raw))]
    else:
        cut = head.raw.index(b"async def")
        ranges = [bounds("head", 0, cut), bounds("head", cut, len(head.raw))]
    result = facts.callable_delivery_issues(row, raw, [head, base], ranges)
    assert bool(result) is expected_gap
    assert raw["text"]["state"] == state
    if result:
        definition = result[0]["missing_callable_definitions"][0]
        assert definition["evidence_id"] == "head"
        assert head.raw[definition["start_byte"]:definition["end_byte"]].decode() == '@trace\nasync def hidden_guard():\n    return "한글"'
        assert [source["evidence_id"] for source in result[0]["required_sources"]] == ["head"]


def test_program_callable_gap_carries_frozen_source_through_existing_rewrite_and_replay():
    from review_bot.domain.models_v1 import CodeFile, Snapshot
    task = _metric_task()
    path, text = "src/Hidden.java", "class Hidden { void hiddenGuard() {} }\n"
    source = CodeFile(path, text, hashlib.sha256(text.encode()).hexdigest(), "source")
    head = replace(task.snapshot.head, files=(*task.snapshot.head.files, source))
    pr = replace(task.snapshot.pr, changed_files=(*task.snapshot.pr.changed_files, path))
    task = build_task_input(Snapshot(pr, task.snapshot.base, head), task.delta, task.classification)
    literal = "The supplied HEAD source defines hiddenGuard."
    def count(request):
        payload = json.loads(request.user_content)
        if payload["stage"] in ("fact_check", "fact_rewrite") and not payload.get("recheck", {}).get("source_requests"):
            if any(source["path"] == path for source in payload["originals"]):
                return 1000
        return 100
    def transform(payload, body):
        if payload["stage"] == "final_review":
            body["summary"] = literal
        elif payload["stage"] == "fact_rewrite":
            body = {"replacement": payload["items"][0]["original_item"],
                "correction_reason": "Read the missing fixed implementation; preserve the original assertion.",
                "correction_citations": [{"reference_id": _reference(payload, path)}]}
        return body
    review = PartitionModelClient(counter=count, transform=transform, group_all=True)
    review.profile = replace(review.profile, max_input_tokens=500)
    result = run_partition_review(task, "architecture", review, PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    entry = result.partition["fact_verification"]["items"][0]
    first, second = entry["attempts"]
    assert first["assessments"]["text"]["state"] == "supported"
    assert first["check"]["response"]["source_requests"] == []
    assert first["issues"][0]["missing_callable_names"] == ["hiddenGuard"]
    assert first["source_plan"]["requests"] == facts.planned_source_requests(first)
    assert second["input_source_requests"] == first["source_plan"]["requests"]
    assert entry["original"] == entry["accepted"] == result.summary == literal
    validate_partition_result(result, task)
    changed = deepcopy(result)
    changed.partition["fact_verification"]["items"][0]["attempts"][0]["source_plan"]["requests"] = []
    with pytest.raises(PartitionError):
        validate_partition_result(changed, task)


def _capacity_review_client():
    from dataclasses import asdict
    from hashlib import sha256
    from review_bot.adapters.beam_cloud import BeamCloudModelClient
    from review_bot.application.architecture_review import ModelInputCapacityError
    from review_bot.application.model_execution import capacity_input_hash

    class CapacityClient(PartitionModelClient):
        supports_native_input_measurement = True

        def count_input_tokens(self, request):
            payload = json.loads(request.user_content)
            if payload["stage"] == "fact_check" and len(payload["items"]) > 4:
                self.token_count_receipt = {"delivery": "unknown", "model_calls": None, "usage": None}
                body = json.dumps({"error": "The prompt is too long: 1524758, model maximum context length: 1048576 (ref: d7fc51b7-fe86-4bfa-99c6-b70cf9701bb1)"})
                wire = BeamCloudModelClient()._payload(request)
                wire["options"]["num_predict"] = 1
                proof = {"provider": "ollama-cloud", "model": "beam:cloud", "status": 400,
                    "response_body": body, "response_sha256": sha256(body.encode()).hexdigest(),
                    "wire_sha256": sha256(json.dumps(wire, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest(),
                    "input_tokens": 1524758, "context_tokens": 1048576, "output_tokens_reserved": 1,
                    "completed": True, "request_hash": hash_json(asdict(request)),
                    "source_input_hash": capacity_input_hash(request.system_prompt, request.user_content,
                        request.response_schema, request.model, request.max_output_tokens)}
                raise ModelInputCapacityError("Beam cloud HTTP 400", proof)
            count = super().count_input_tokens(request)
            self.token_count_receipt = {"delivery": "sent", "model_calls": 1,
                "usage": {"input_tokens": count, "output_tokens": 1}}
            return count

    client = CapacityClient(name="beam:cloud")
    client.profile = replace(client.profile, provider="ollama-cloud", max_input_tokens=1048576 - 512, is_mock=False)
    return client


@pytest.mark.parametrize("fits", [True, False])
def test_native_capacity_rejection_reserves_unknown_usage_before_lossless_batch_split(fits):
    import time
    from review_bot.application.model_execution import ModelExecutionBudget
    task, client = make_task(), _capacity_review_client()
    settings = PartitionReviewLimits(max_calls=256, max_total_tokens=24000000 if fits else 1000000)
    budget = ModelExecutionBudget(settings, time.monotonic() + settings.task_timeout_seconds, None)
    result = run_partition_review(task, "intent", client, PartitionModelClient(), limits=settings, shared_budget=budget)
    rejected = result.partition["native_capacity_rejections"]
    assert rejected and result.usage["usage_complete"] is False
    assert result.usage["input_tokens"] is None and result.usage["output_tokens"] is None
    assert result.usage["unknown_usage_attempts"] == len(rejected)
    assert result.usage["reserved_model_calls"] == len(rejected)
    assert result.usage["reserved_tokens"] == len(rejected) * 1524759
    actual_calls, actual_tokens, actual_unknown = budget.totals()
    effective_calls, effective_tokens, unbounded_unknown = budget.operational_totals()
    assert actual_unknown is True and unbounded_unknown is False
    assert effective_calls == actual_calls + result.usage["reserved_model_calls"]
    assert effective_tokens == actual_tokens + result.usage["reserved_tokens"]
    assert all(result.stages[item["stage_index"]].token_measurement_usage is None for item in rejected)
    if fits:
        assert result.status == "completed", result.reason
        checks = [json.loads(req.user_content) for req in client.requests if json.loads(req.user_content)["stage"] == "fact_check"]
        assert [row["item_id"] for call in checks for row in call["items"]] == [row["id"] for row in facts.items(result.partition["pre_verification_final"])]
        assert all(len(call["items"]) <= 4 for call in checks)
        for mutation in ("reservation", "body", "wire_hash", "missing", "source_input"):
            changed = deepcopy(result)
            item = changed.partition["native_capacity_rejections"][0]
            if mutation == "reservation":
                item["budget_reservation"]["total_tokens"] -= 1
            elif mutation == "missing":
                changed.partition.pop("native_capacity_rejections")
            else:
                key = {"body": "response_body", "wire_hash": "wire_sha256", "source_input": "source_input_hash"}[mutation]
                item["capacity_rejection"][key] = "forged" if mutation == "body" else "0" * 64
                item["witness_hash"] = hash_json(item["capacity_rejection"])
            with pytest.raises(PartitionError):
                validate_partition_result(changed, task)
    else:
        assert result.status == "partial_review"
        assert result.partition["fact_verification"]["interruption_reason"] == "task_token_limit_exceeded"
        assert not any(json.loads(req.user_content)["stage"] == "fact_check" for req in client.requests)
    validate_partition_result(result, task)


@pytest.mark.parametrize("mode", ["architecture", "intent"])
def test_wire_reference_definitions_preserve_complete_a_b_fact_response_contracts(mode):
    from review_bot.application.model_wire_contract import compact_schema
    if mode == "architecture":
        task, result, client, _ = _metric_run()
    else:
        task, client = make_task(), PartitionModelClient()
        result = run_partition_review(task, mode, client, PartitionModelClient(), limits=limits())
    assert result.status == "completed"
    packets = {packet["stage_index"]: packet for entry in result.partition["fact_verification"]["items"]
        for attempt in entry["attempts"] for key in ("check", "rewrite") if (packet := attempt.get(key))}
    for packet in packets.values():
        request = next(req for req in client.requests if json.loads(req.user_content)["stage"] == packet["call_stage"]
            and [row["item_id"] for row in json.loads(req.user_content)["items"]] == [row["id"] for row in packet["rows"]])
        original = deepcopy(request.response_schema)
        compact = compact_schema(original)
        def expand(value):
            if isinstance(value, dict):
                if set(value) == {"$ref"}:
                    return expand(compact["$defs"][value["$ref"][8:]])
                return {key: expand(child) for key, child in value.items() if key != "$defs"}
            return [expand(child) for child in value] if isinstance(value, list) else value
        restored = expand(compact)
        assert restored == original == request.response_schema
        def reference_ids(value):
            if isinstance(value, dict):
                if set(value) == {"reference_id"}:
                    yield value["reference_id"]
                for child in value.values():
                    yield from reference_ids(child)
            elif isinstance(value, list):
                for child in value:
                    yield from reference_ids(child)
        def enums(value):
            if isinstance(value, dict):
                yield from value.get("enum", [])
                for child in value.values():
                    yield from enums(child)
            elif isinstance(value, list):
                for child in value:
                    yield from enums(child)
        allowed = set(enums(restored))
        assert set(reference_ids(packet["response"])) <= allowed
        assert "__FOREIGN_REFERENCE__" not in allowed
        # All per-item supported-only enums and conditional rules also survive
        # exact expansion, including IDs that exist elsewhere in the catalog.
    validate_partition_result(result, task)


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
        if payload["stage"] == "fact_check" and "perspectives:2" in body["items"]:
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
    baseline = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True), limits=limits())
    fact_index = next(i for i, stage in enumerate(baseline.stages) if stage.stage == "fact_check" and stage.attempts)
    cap = sum(len(stage.attempts) for stage in baseline.stages[:fact_index])
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_calls=cap, max_total_tokens=1000000))
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    assert result.partition["fact_verification"]["interruption_reason"] == "call_limit_exceeded"
    assert result.usage["calls"] <= cap and result.usage["usage_complete"]
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


@pytest.mark.parametrize("component", ["facts", "eligibility", "wire"])
def test_worker_mode_fingerprints_include_verification_and_eligibility_implementations(valid_config_yaml, tmp_path, monkeypatch, component):
    from review_bot.config import load_review_config_from_str
    from review_bot.runtime.mode_fingerprints import architecture_input_fingerprint, intent_input_fingerprint
    from review_bot.application import selection
    from review_bot.application import model_wire_contract
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
    monkeypatch.setattr({"facts": facts, "eligibility": selection, "wire": model_wire_contract}[component], "__file__", str(changed))
    after = fingerprints()
    assert all(after) and all(left != right for left, right in zip(before, after))


@pytest.mark.parametrize("mode", ["architecture", "intent"])
def test_checker_receives_literal_prose_and_exact_claim_rules_without_repairing_input(mode):
    """Wire contract only; native paired controls establish the prompt evidence."""
    literal = "Guard A OR guard B. A named method's property must not be attributed to both."
    def transform(payload, body):
        if payload["stage"] == "final_review":
            body["summary"] = literal
        return body
    task = _metric_task() if mode == "architecture" else make_task()
    client = PartitionModelClient(transform=transform)
    result = run_partition_review(task, mode, client, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    request = next(req for req in client.requests if json.loads(req.user_content)["stage"] == "fact_check")
    payload = json.loads(request.user_content)
    assert payload["items"][0]["original_item"] == literal
    assert payload["items"][0]["fields"]["text"] == literal
    assert result.partition["pre_verification_final"]["summary"] == literal
    assert "not a weaker true paraphrase" in request.system_prompt
    assert "check EACH named component" in request.system_prompt
    assert "necessary conditions, sufficient conditions" in request.system_prompt
    for prompt in (request.system_prompt, facts.REWRITE_PROMPT):
        assert "inspect the exact caller implementation" in prompt
        assert "EVERY claimed wrapper or validator on that route" in prompt
        assert "equivalent check on another route does not support the claimed route" in prompt
    assert "field's reason, account for EVERY literal clause and named symbol" in request.system_prompt
    assert "choose unverifiable and request its exact frozen evidence_id" in request.system_prompt
    assert "source_requests from fixed_original_inventory" in request.system_prompt
    assert "keep the unresolved limitation explicit for the factual recheck" in facts.REWRITE_PROMPT
    assert "silently remove the disputed assertion" in facts.REWRITE_PROMPT
    assert result.partition["identity"]["prompts"]["fact_check"] == hash_json(facts.CHECK_PROMPT)
    assert result.prompt_version == "partitioned-review/v17-shared-service-batched-facts"
    assert result.model["is_mock"] is True  # no semantic accuracy claim from this fixture


def _fixed_test_repair(*, no_suitable=False, cap=128, corrupt_request=None):
    from review_bot.domain.models_v1 import CodeFile, Snapshot
    task = _metric_task()
    path = "tests/ACheckTest.java"
    content = "class ACheckTest { void check() { assert true; } }\n"
    code = CodeFile(path, content, hashlib.sha256(content.encode()).hexdigest(), "test")
    head = replace(task.snapshot.head, files=(*task.snapshot.head.files, code))
    pr = replace(task.snapshot.pr, changed_files=(*task.snapshot.pr.changed_files, path))
    task = build_task_input(Snapshot(pr, task.snapshot.base, head), task.delta, task.classification)
    literal = "The frozen head includes a related test assertion."
    def counter(request):
        payload = json.loads(request.user_content)
        if payload["stage"] in ("fact_check", "fact_rewrite") and not payload.get("recheck", {}).get("source_requests"):
            if any(s["path"] == path for s in payload["originals"]):
                return 1000
        return 100
    def transform(payload, body):
        stage = payload["stage"]
        if stage == "final_review":
            body["summary"] = literal
        elif stage == "fact_check" and payload["items"][0]["item_id"] == "summary":
            if not any(s["path"] == path and s["version"] == "head" for s in payload["originals"]):
                body["items"]["summary"]["text"].update(state="unverifiable",
                    reason="Need the actual frozen HEAD tests/ACheckTest.java assertion, not a reported test result.")
                source = next((s for s in payload["fixed_original_inventory"] if s["path"] == path and s["version"] == "head"), None)
                if source and not no_suitable:
                    body["source_requests"] = [{"evidence_id": source["evidence_id"], "reason": "Read the actual HEAD assertion."}]
                    if corrupt_request == "outside":
                        body["source_requests"][0]["evidence_id"] = "outside-corpus"
                    elif corrupt_request == "already_delivered":
                        body["source_requests"][0]["evidence_id"] = payload["originals"][0]["evidence_id"]
                    elif corrupt_request == "duplicate":
                        body["source_requests"] *= 2
                    elif corrupt_request == "empty_reason":
                        body["source_requests"][0]["reason"] = ""
        elif stage == "fact_rewrite":
            source = path if payload.get("recheck", {}).get("source_requests") else "src/A.java"
            body = {"replacement": payload["items"][0]["original_item"],
                "correction_reason": "Re-read the requested fixed original; keep the item when its premise is supported.",
                "correction_citations": [{"reference_id": _reference(payload, source)}]}
        return body
    review = PartitionModelClient(counter=counter, transform=transform, group_all=True)
    review.profile = replace(review.profile, max_input_tokens=500)
    result = run_partition_review(task, "architecture", review,
        PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_calls=cap, max_total_tokens=1000000))
    return task, result, review, path, content


def test_unverifiable_fact_reads_existing_test_original_and_rechecks_same_item_without_erasing_it():
    task, result, review, path, content = _fixed_test_repair()
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    entry = result.partition["fact_verification"]["items"][0]
    first, second = entry["attempts"]
    assert entry["accepted"] == entry["original"] == result.summary
    assert first["assessments"]["text"]["state"] == "unverifiable"
    assert first["source_plan"]["complete"] and first["source_plan"]["requests"]
    assert second["input_source_requests"] == first["source_plan"]["requests"]
    checks = [json.loads(r.user_content) for r in review.requests if json.loads(r.user_content)["stage"] == "fact_check"]
    assert not any(s["path"] == path for s in checks[0]["originals"])
    supplied = next(s for s in checks[1]["originals"] if s["path"] == path and s["version"] == "head")
    assert "".join(b["text"] for b in supplied["blocks"]) == content
    rewritten = next(json.loads(r.user_content) for r in review.requests if json.loads(r.user_content)["stage"] == "fact_rewrite")
    assert rewritten["recheck"]["original_check_response"] == first["check"]["response"]
    assert rewritten["recheck"]["source_delivery_plan"]["parent_plan_id"] == first["source_plan"]["parent_plan_id"]


def test_no_suitable_fixed_source_does_not_invent_evidence_or_force_a_positive_fact_state():
    task, result, _, _, _ = _fixed_test_repair(no_suitable=True)
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    entry = result.partition["fact_verification"]["items"][0]
    assert entry["attempts"][0]["source_plan"]["requests"] == []
    assert entry["status"] == "unverified"
    validate_partition_result(result, task)


@pytest.mark.parametrize("field", ["parent_plan_id", "requests", "delivered_ranges", "input_source_requests"])
def test_fixed_fact_source_plan_and_delivered_context_tampering_are_rejected(field):
    task, result, _, _, _ = _fixed_test_repair()
    assert result.status == "completed"
    result = deepcopy(result)
    first, second = result.partition["fact_verification"]["items"][0]["attempts"]
    if field == "parent_plan_id":
        first["source_plan"][field] = "changed"
    elif field == "requests":
        first["source_plan"][field] = []
    elif field == "delivered_ranges":
        first["check"][field] = []
    else:
        second[field] = []
    with pytest.raises(PartitionError):
        validate_partition_result(result, task)


def test_fact_source_acquisition_uses_existing_shared_budget_and_keeps_failed_prefix():
    _, baseline, _, _, _ = _fixed_test_repair()
    source_index = next(i for i, s in enumerate(baseline.stages) if s.stage == "fact_rewrite" and s.attempts)
    calls_before = sum(len(s.attempts) for s in baseline.stages[:source_index])
    task, result, _, _, _ = _fixed_test_repair(cap=calls_before)
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    ledger = result.partition["fact_verification"]
    assert ledger["interruption_reason"] == "call_limit_exceeded"
    plan = ledger["items"][0]["attempts"][0]["source_plan"]
    assert plan["complete"] is True and plan["requests"]
    assert result.usage["calls"] == calls_before
    validate_partition_result(result, task)


def test_requested_fact_original_is_complete_even_in_focused_citation_delivery():
    from review_bot.application.source_partition import SourceOriginal, make_span
    code = SourceOriginal("S-code", "one\ntwo\nthree\nfour\n", "source", "src/A.py", "head", "a" * 40)
    test = SourceOriginal("S-test", "def test_a():\n    assert True\n", "test_source", "tests/test_a.py", "head", "a" * 40)
    purpose = SourceOriginal("S-purpose", "Frozen full purpose", "pr_text", "", "", "")
    span = make_span(code, 4, 8, "test")
    quote = {"evidence_id": span.evidence_id, "start_byte": span.start_byte, "end_byte": span.end_byte,
        "sha": span.sha, "path": span.path, "version": span.version, "begin_line": span.begin_line,
        "end_line": span.end_line, "quote": span.text}
    final = {"summary": "Test premise.", "findings": [{"citations": [quote]}], "perspectives": [], "questions": []}
    row = {"id": "summary", "kind": "summary", "item": final["summary"]}
    extra = {"source_requests": [{"evidence_id": test.evidence_id, "reason": "Need the fixed assertion."}],
        "check_cited_evidence": [quote]}
    payload, _, _ = facts.delivery([row], final, [code, test, purpose], {}, kind="cited_ranges", extra=extra)
    supplied = {s["evidence_id"]: "".join(b["text"] for b in s["blocks"]) for s in payload["originals"]}
    assert supplied == {code.evidence_id: "two\n", test.evidence_id: test.text, purpose.evidence_id: purpose.text}
    extra["source_requests"][0]["evidence_id"] = "outside"
    with pytest.raises(PartitionError, match="outside_frozen_corpus"):
        facts.delivery([row], final, [code, test, purpose], {}, kind="cited_ranges", extra=extra)


@pytest.mark.parametrize("corrupt", ["outside", "already_delivered", "duplicate", "empty_reason"])
def test_invalid_fact_source_request_is_a_readable_partial_without_source_expansion(corrupt):
    task, result, review, _, _ = _fixed_test_repair(corrupt_request=corrupt)
    assert result.status == "partial_review" and result.reason == "factual_verification_incomplete"
    assert result.partition["fact_verification"]["interruption_reason"] == "fact_source_request_invalid"
    assert not any(json.loads(req.user_content)["stage"] == "fact_rewrite" for req in review.requests)
    validate_partition_result(result, task)


def test_final_items_share_one_native_admitted_fact_check_without_merge_call():
    task = make_task()
    client = PartitionModelClient()
    result = run_partition_review(task, "intent", client, PartitionModelClient(), limits=limits())
    checks = [json.loads(req.user_content) for req in client.requests if json.loads(req.user_content)["stage"] == "fact_check"]
    assert len(checks) == 1
    assert [row["item_id"] for row in checks[0]["items"]] == [row["id"] for row in facts.items(result.partition["pre_verification_final"])]
    assert result.status == "completed"
    validate_partition_result(result, task)


def test_native_overflow_splits_item_batches_with_complete_deterministic_id_coverage():
    def count(req):
        payload = json.loads(req.user_content)
        return len(payload["items"]) * 100 if payload["stage"] == "fact_check" else 100
    client = PartitionModelClient(counter=count)
    client.profile = replace(client.profile, max_input_tokens=250)
    task = make_task()
    result = run_partition_review(task, "intent", client, PartitionModelClient(), limits=limits())
    checks = [json.loads(req.user_content) for req in client.requests if json.loads(req.user_content)["stage"] == "fact_check"]
    ids = [row["item_id"] for payload in checks for row in payload["items"]]
    assert ids == [row["id"] for row in facts.items(result.partition["pre_verification_final"])]
    assert all(len(payload["items"]) <= 2 for payload in checks)
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)


def _small_source_page_counter(req, *, stage):
    payload = json.loads(req.user_content)
    if payload["stage"] != stage:
        return 100
    size = sum(len(block["text"].encode()) for source in payload["originals"] for block in source["blocks"])
    return 100 if payload.get("source_page") and size <= 20 else 1000


def test_single_oversized_fixed_source_is_losslessly_paged_without_promoting_cross_page_support():
    client = PartitionModelClient(counter=lambda req: _small_source_page_counter(req, stage="fact_check"))
    client.profile = replace(client.profile, max_input_tokens=250)
    task = _metric_task()
    result = run_partition_review(task, "architecture", client, PartitionModelClient(), limits=limits(max_rechecks=0))
    assert result.status == "partial_review"
    assert result.partition["final"] == result.partition["pre_verification_final"]
    ledger = result.partition["fact_verification"]
    call = ledger["items"][0]["attempts"][0]["check"]
    assert call["split"] == "sources" and "interruption_reason" not in ledger
    assert ledger["items"][0]["status"] == "unverified"
    packets = [json.loads(req.user_content) for req in client.requests if json.loads(req.user_content)["stage"] == "fact_check"]
    from review_bot.application.partition_review import prepare_originals
    originals, meta = prepare_originals(task, "architecture")
    _, _, cover = facts.delivery(facts.items(result.partition["pre_verification_final"]), result.partition["pre_verification_final"], originals, meta, kind="complete_originals")
    for span in cover:
        pieces = sorted((source for packet in packets for source in packet["originals"] if source["evidence_id"] == span.evidence_id), key=lambda source: source["start_byte"])
        assert "".join(block["text"] for source in pieces for block in source["blocks"]) == span.text
        assert pieces[0]["start_byte"] == span.start_byte and pieces[-1]["end_byte"] == span.end_byte
    validate_partition_result(result, task)


def test_oversized_rewrite_pages_keep_raw_replies_and_recheck_the_complete_replacement():
    corrected = "Complete corrected assertion."
    def transform(payload, body):
        if payload["stage"] == "fact_check" and payload["items"][0]["original_item"] != corrected:
            body["items"]["summary"]["text"].update(state="contradicted", reason="The original premise is false.")
        if payload["stage"] == "fact_check" and payload.get("source_page") and payload["fixed_original_inventory"]:
            source = payload["fixed_original_inventory"][0]
            body["items"]["summary"]["text"].update(state="unverifiable", reason="Need another exact frozen original for this candidate.")
            body["source_requests"] = [{"evidence_id": source["evidence_id"], "reason": "Carry this actual source into the final recheck."}]
        if payload["stage"] == "fact_rewrite":
            body["replacement"] = corrected
        return body
    client = PartitionModelClient(counter=lambda req: _small_source_page_counter(req, stage="fact_rewrite"), transform=transform)
    client.profile = replace(client.profile, max_input_tokens=250)
    task = _metric_task()
    result = run_partition_review(task, "architecture", client, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    first, second = result.partition["fact_verification"]["items"][0]["attempts"]
    assert first["rewrite"]["split"] == "sources"
    assert all("response" in child for child in first["rewrite"]["children"])
    assert second["item"] == corrected == result.summary
    assert second["input_source_requests"]
    validate_partition_result(result, task)


def test_exact_duplicate_source_text_keeps_every_original_identity_and_byte_on_readback():
    from review_bot.application.source_partition import SourceOriginal
    original = "def run():\n    return 'same exact bytes'\n"
    sources = [SourceOriginal("S-base", original, "source", "a.py", "base", "a" * 40),
        SourceOriginal("S-head", original, "source", "a.py", "head", "b" * 40)]
    final = {"summary": "Same implementation.", "findings": [], "perspectives": [], "questions": []}
    payload, _, refs, spans = facts.request_spec("fact_check", facts.items(final), final, sources, {}, kind="complete_originals")
    assert isinstance(payload["originals"][1]["blocks"][0]["text"], dict)
    materialized = facts.materialize_source_texts(payload)
    assert [source["evidence_id"] for source in materialized] == ["S-base", "S-head"]
    assert [source["version"] for source in materialized] == ["base", "head"]
    assert all("".join(block["text"] for block in source["blocks"]) == original for source in materialized)
    assert len(refs) == 2 and len(spans) == 2


def test_source_capacity_split_preserves_multibyte_unicode_boundaries():
    from review_bot.application.source_partition import SourceOriginal
    source = SourceOriginal("S-unicode", "😀a", "source", "a.py", "head", "a" * 40)
    left, right = facts.split_ranges([{"evidence_id": source.evidence_id, "start_byte": 0, "end_byte": len(source.raw)}], [source])
    assert left[0]["end_byte"] == right[0]["start_byte"] == 4
    assert source.raw[:4].decode() + source.raw[4:].decode() == source.text


def test_citation_limit_applies_to_each_assessment_rather_than_whole_batch():
    from review_bot.application.source_partition import SourceOriginal, make_span
    from review_bot.application.source_citations import citation_spans
    source = SourceOriginal("S-many", "a\n" * 400, "source", "a.py", "head", "a" * 40)
    refs = citation_spans([make_span(source, 0, len(source.raw), "test")], {source.evidence_id: source})
    assert len(refs) > 32
    raw = {"items": {"a": [{"reference_id": ref} for ref in refs]}}
    assert len(facts.response_citations(raw, refs)) == len(refs)


@pytest.mark.parametrize("rewrite_capacity,expected_rewrite_calls", [(250, 1), (150, 2)])
def test_two_factual_errors_share_rewrite_and_batch_recheck_with_id_coverage(rewrite_capacity, expected_rewrite_calls):
    failing = {"perspectives:0", "perspectives:1"}
    corrected = "The complete corrected factual premise."
    def counter(request):
        payload = json.loads(request.user_content)
        return len(payload["items"]) * 100 if payload["stage"] == "fact_rewrite" else 100
    def transform(payload, body):
        if payload["stage"] == "fact_check":
            for row in payload["items"]:
                if row["item_id"] in failing and row["original_item"]["reasoning"] != corrected:
                    body["items"][row["item_id"]]["reasoning"].update(state="contradicted", reason="This exact premise must be corrected.")
        if payload["stage"] == "fact_rewrite":
            replacements = {}
            for row in payload["items"]:
                item = deepcopy(row["original_item"])
                item["reasoning"] = corrected
                item["citations"] = [{"reference_id": _reference(payload)}]
                replacements[row["item_id"]] = {"replacement": item,
                    "correction_reason": "Correct this item's premise with frozen evidence.", "correction_citations": item["citations"]}
            return {"items": replacements} if len(replacements) > 1 else next(iter(replacements.values()))
        return body
    client = PartitionModelClient(counter=counter, transform=transform)
    client.profile = replace(client.profile, max_input_tokens=rewrite_capacity)
    task = make_task()
    result = run_partition_review(task, "intent", client, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    rewrites = [json.loads(req.user_content) for req in client.requests if json.loads(req.user_content)["stage"] == "fact_rewrite"]
    checks = [json.loads(req.user_content) for req in client.requests if json.loads(req.user_content)["stage"] == "fact_check"]
    assert len(rewrites) == expected_rewrite_calls
    assert {row["item_id"] for payload in rewrites for row in payload["items"]} == failing
    assert len(rewrites[0]["recheck"]["checker_outputs"]) == 1
    assert set(next(iter(rewrites[0]["recheck"]["checker_outputs"].values()))["original_check_response"]["items"]) == {
        row["item_id"] for row in checks[0]["items"]}
    assert len(checks) == 2 and {row["item_id"] for row in checks[1]["items"]} == failing
    assert all(row["original_item"]["reasoning"] == corrected for row in checks[1]["items"])
    ledger = result.partition["fact_verification"]
    for entry in ledger["items"]:
        assert len(entry["attempts"]) == (2 if entry["item_id"] in failing else 1)
    validate_partition_result(result, task)


@pytest.mark.parametrize("whole_claim_fits", [True, False])
def test_oversized_original_requires_direct_whole_claim_recheck_after_complete_discovery(whole_claim_fits):
    task = _metric_task()
    declaration = "static final int ANSWER = 42;"
    content = "class A {\n  " + declaration + "\n" + "  // retain this exact frozen context\n" * 100 + "}\n"
    head = replace(task.snapshot.head, files=tuple(replace(file, content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest()) if file.path == "src/A.java" else file
        for file in task.snapshot.head.files))
    task = build_task_input(replace(task.snapshot, head=head), task.delta, task.classification)
    literal = ("The HEAD declaration sets ANSWER to 42." if whole_claim_fits else
        "The HEAD declaration and a separate caller jointly guarantee the same value.")
    def count(request):
        payload = json.loads(request.user_content)
        if payload["stage"] != "fact_check":
            return 100
        if "whole_claim_recheck" in payload:
            return 100 if whole_claim_fits else 1000
        size = sum(len(block["text"].encode()) for source in facts.materialize_source_texts(payload) for block in source["blocks"])
        return 100 if payload.get("source_page") and size <= 250 else 1000
    def transform(payload, body):
        if payload["stage"] == "final_review":
            body["summary"] = literal
        if payload["stage"] == "fact_check":
            proof = next((block for source in facts.materialize_source_texts(payload)
                if source["version"] == "head" and source["path"] == "src/A.java"
                for block in source["blocks"] if declaration in block["text"]), None)
            field = body["items"]["summary"]["text"]
            field.update(state="supported" if proof else "unverifiable",
                reason="The exact HEAD declaration supports the complete literal assertion." if proof else "This page has no evidence of the declaration.",
                citations=[{"reference_id": proof["reference_id"]}] if proof else [])
        return body
    client = PartitionModelClient(counter=count, transform=transform)
    client.profile = replace(client.profile, max_input_tokens=250)
    result = run_partition_review(task, "architecture", client, PartitionModelClient(), limits=limits(max_rechecks=0))
    ledger = result.partition["fact_verification"]
    call = ledger["items"][0]["attempts"][0]["check"]
    assert call["split"] == "sources" and len(call["discovery"]["pages"]) > 1
    assert all(page["response_hash"] == hash_json(page["response"]) for page in call["discovery"]["pages"])
    assert call["whole_claim"]["rows"][0]["item"] == literal
    assert call["discovery_response"]["items"]["summary"]["text"]["state"] == "unverifiable"
    assert result.summary == literal
    pages = [json.loads(request.user_content) for request in client.requests
        if json.loads(request.user_content)["stage"] == "fact_check" and json.loads(request.user_content).get("source_page")]
    head_pieces = sorted((source for page in pages for source in facts.materialize_source_texts(page)
        if source["path"] == "src/A.java" and source["version"] == "head"), key=lambda source: source["start_byte"])
    assert "".join(block["text"] for source in head_pieces for block in source["blocks"]) == content
    if whole_claim_fits:
        assert result.status == "completed", result.reason
        focused = next(json.loads(request.user_content) for request in client.requests
            if json.loads(request.user_content).get("whole_claim_recheck"))
        assert focused["items"][0]["original_item"] == literal
        assert focused["whole_claim_recheck"]["pages"] == call["discovery"]["pages"]
        assert sum(len(block["text"].encode()) for source in facts.materialize_source_texts(focused) for block in source["blocks"]) < len(content.encode())
        for mutation in ("discovery_cover", "focused_evidence"):
            changed = deepcopy(result)
            changed_call = changed.partition["fact_verification"]["items"][0]["attempts"][0]["check"]
            if mutation == "discovery_cover":
                changed_call["discovery"]["pages"][0]["delivered_ranges"][0]["end_byte"] -= 1
            else:
                changed_call["whole_claim"]["source_ranges"][0]["end_byte"] -= 1
            with pytest.raises(PartitionError):
                validate_partition_result(changed, task)
    else:
        assert result.status == "partial_review"
        assert call["whole_claim"]["admission_rejected"]
        assert "response" not in call["whole_claim"]
        assert ledger["items"][0]["status"] == "unverified"
    validate_partition_result(result, task)


@pytest.mark.parametrize("limited_stage", ["fact_check", "fact_rewrite"])
@pytest.mark.parametrize("known_usage", [True, False])
def test_known_native_batch_output_limit_splits_items_after_bounded_recovery_only(limited_stage, known_usage):
    from review_bot.application.architecture_review import ModelOutputLimitError
    failing = {"perspectives:0", "perspectives:1"}
    corrected = "Corrected complete factual premise."
    review = PartitionModelClient()
    review.profile = replace(review.profile, max_output_tokens=32678)
    review.supports_bounded_reasoning = True
    root_size = 9 if limited_stage == "fact_check" else 2
    def transform(payload, body):
        request = review.requests[-1]
        if payload["stage"] == limited_stage and len(payload["items"]) == root_size:
            error = ModelOutputLimitError("Known native length truncation")
            if known_usage:
                error.delivery = "sent"
                error.usage = {"input_tokens": 100, "output_tokens": request.max_output_tokens}
            error.rejected_output = {"finish_reason": "length", "content": '{"unfinished":',
                "reasoning_content": "Exact rejected native output retained."}
            raise error
        if limited_stage == "fact_rewrite" and payload["stage"] == "fact_check":
            for row in payload["items"]:
                if row["item_id"] in failing and row["original_item"]["reasoning"] != corrected:
                    body["items"][row["item_id"]]["reasoning"].update(state="contradicted", reason="Correct this exact factual premise.")
        if payload["stage"] == "fact_rewrite":
            item = deepcopy(payload["items"][0]["original_item"])
            item["reasoning"] = corrected
            item["citations"] = [{"reference_id": _reference(payload)}]
            return {"replacement": item, "correction_reason": "Keep this correction with exact frozen evidence.",
                "correction_citations": item["citations"]}
        return body
    review.transform = transform
    task = make_task()
    result = run_partition_review(task, "intent", review, PartitionModelClient(), limits=limits())
    ledger = result.partition["fact_verification"]
    entry = ledger["items"][0 if limited_stage == "fact_check" else 1]
    call = entry["attempts"][0]["check" if limited_stage == "fact_check" else "rewrite"]
    root_calls = [request for request in review.requests if json.loads(request.user_content)["stage"] == limited_stage
        and len(json.loads(request.user_content)["items"]) == root_size]
    if known_usage:
        assert result.status == "completed", result.reason
        assert len(root_calls) == 2 and root_calls[1].reasoning_budget_tokens is not None
        assert call["split"] == "items" and call["split_cause"] == "known_output_limit"
        assert len(call["children"]) == 2 and all("response" in child for child in call["children"])
        assert result.usage["usage_complete"]
        assert len(result.partition["rejected_model_outputs"]) == 2
        assert len(result.partition["output_recoveries"]) == 1
        assert all(record["output"]["content"] == '{"unfinished":' for record in result.partition["rejected_model_outputs"])
        changed = deepcopy(result)
        changed_call = changed.partition["fact_verification"]["items"][0 if limited_stage == "fact_check" else 1]["attempts"][0]["check" if limited_stage == "fact_check" else "rewrite"]
        changed_call["split_cause"] = "input_limit"
        with pytest.raises(PartitionError):
            validate_partition_result(changed, task)
    else:
        assert result.status == "partial_review"
        assert ledger["interruption_reason"] == "usage_unavailable"
        assert len(root_calls) == 1 and "children" not in call
        assert not result.usage["usage_complete"]
        assert not result.partition["output_recoveries"] and not result.partition["rejected_model_outputs"]
    validate_partition_result(result, task)



def test_fixed_perspective_scope_premises_use_existing_four_item_rewrite_batch():
    from dataclasses import asdict
    from review_bot.domain.intent_perspectives import CATALOG_VERSION, PERSPECTIVES
    excluded = {"problem_success", "outcome_connection", "preservation", "assumptions_tradeoffs"}
    invented = "This perspective is excluded because this page focuses on other perspectives."
    observed_contexts = []
    def plan_transform(payload, body):
        if payload["stage"] == "select_perspective" and payload["state"]["perspective"]["id"] in excluded:
            body["choice"] = next(option["label"] for option in payload["options"] if option["key"] == "not_apply")
        return body
    def review_transform(payload, body):
        stage = payload["stage"]
        if stage == "final_review":
            for item in body["perspectives"]:
                if item["perspective_id"] in excluded:
                    item.update(verdict="not_apply", statement=invented, reasoning=invented)
        elif stage in ("fact_check", "fact_rewrite"):
            context = payload["canonical_facts"]["intent_perspectives"]
            assert context["catalog_version"] == CATALOG_VERSION
            assert context["catalog"] == [asdict(perspective) for perspective in PERSPECTIVES]
            assert context["scope_policy"]["review_all_eight_from_fixed_catalog"] is True
            assert "Provisional" in context["scope_policy"]["initial_tev_applicability"]
            assert "Page focus" in context["scope_policy"]["unsupported_exclusions"]
            observed_contexts.append(context)
            if stage == "fact_check":
                assert payload["originals"] and payload["fixed_original_inventory"] == []
                for row in payload["items"]:
                    if row["item_id"].startswith("perspectives:") and row["original_item"]["reasoning"] == invented:
                        for field in ("statement", "reasoning"):
                            body["items"][row["item_id"]][field].update(state="contradicted",
                                reason="Page focus and a provisional TEV label cannot establish exclusion under the fixed scope policy.")
            else:
                replacements = {}
                for row in payload["items"]:
                    item = deepcopy(row["original_item"])
                    item.update(verdict="aligned", statement="The fixed perspective was independently reviewed against the complete PR originals.",
                        reasoning="The PR purpose and actual HEAD behavior support this judgement, independently of the initial focus hint.",
                        citations=[{"reference_id": _reference(payload)}])
                    replacements[row["item_id"]] = {"replacement": item,
                        "correction_reason": "Correct the unsupported page-focus exclusion using the fixed catalog and original source.",
                        "correction_citations": item["citations"]}
                return {"items": replacements}
        return body
    task = make_task()
    review = PartitionModelClient(transform=review_transform)
    result = run_partition_review(task, "intent", review,
        PartitionModelClient(group_all=True, transform=plan_transform), limits=limits())
    assert result.status == "completed", result.reason
    hints = {row["perspective_id"]: row["decision"] for row in result.partition["scope_selection"]["perspectives"]}
    assert {key for key, value in hints.items() if value == "not_apply"} == excluded
    original = result.partition["pre_verification_final"]
    assert {item["perspective_id"] for item in original["perspectives"] if item["verdict"] == "not_apply"} == excluded
    assert all(item["verdict"] == "aligned" for item in result.perspectives) and len(result.perspectives) == 8
    rewrites = [json.loads(request.user_content) for request in review.requests if json.loads(request.user_content)["stage"] == "fact_rewrite"]
    checks = [json.loads(request.user_content) for request in review.requests if json.loads(request.user_content)["stage"] == "fact_check"]
    assert len(rewrites) == 1 and len(rewrites[0]["items"]) == 4
    assert len(checks) == 2 and len(checks[0]["items"]) == 9 and len(checks[1]["items"]) == 4
    assert {row["original_item"]["perspective_id"] for row in rewrites[0]["items"]} == excluded
    assert len(observed_contexts) == 3
    validate_partition_result(result, task)


def test_source_grounded_perspective_non_applicability_is_still_permitted():
    legitimate = "The fixed change restores the existing local string result contract and introduces no additional operational assumption or accepted cost."
    def transform(payload, body):
        if payload["stage"] == "final_review":
            for item in body["perspectives"]:
                if item["perspective_id"] == "assumptions_tradeoffs":
                    item.update(verdict="not_apply", statement=legitimate, reasoning=legitimate)
        elif payload["stage"] == "fact_check":
            context = payload["canonical_facts"]["intent_perspectives"]
            assert "not_apply remains allowed" in context["scope_policy"]["source_grounded_not_apply"]
            assert "Do not force apply" in context["scope_policy"]["source_grounded_not_apply"]
        return body
    task = make_task(repair=True)
    review = PartitionModelClient(transform=transform)
    result = run_partition_review(task, "intent", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    item = next(item for item in result.perspectives if item["perspective_id"] == "assumptions_tradeoffs")
    assert item["verdict"] == "not_apply" and item["reasoning"] == legitimate
    assert not any(json.loads(request.user_content)["stage"] == "fact_rewrite" for request in review.requests)
    validate_partition_result(result, task)



@pytest.mark.parametrize("mode", ["intent", "architecture"])
def test_empty_field_rule_preserves_original_field_and_strict_response_contract(mode):
    from review_bot.application.source_partition import SourceOriginal
    from review_bot.domain.intent_perspectives import PERSPECTIVES
    source = SourceOriginal("S-scope", "def answer():\n    return 42\n", "source", "a.py", "head", "a" * 40)
    item = {"statement": "The function returns 42.",
        "reasoning": "The exact HEAD return statement supports this assertion.", "failure_condition": "", "citations": []}
    if mode == "intent":
        item.update(perspective_id=PERSPECTIVES[0].id, verdict="aligned")
    else:
        item.update(verdict="improvement", basis="observed_structure", target_ids=[])
    final = {"summary": "Static review.", "findings": [item] if mode == "architecture" else [],
        "perspectives": [item] if mode == "intent" else [], "questions": []}
    row = facts.items(final)[1]
    payload, schema, _, _ = facts.request_spec("fact_check", [row], final, [source], {}, kind="complete_originals")
    rule = payload["canonical_facts"]["empty_field_rule"]
    assert "state=non_factual" in rule and "non-empty reason" in rule and "citations=[]" in rule
    assert "empty_field_rule" not in payload["canonical_facts"].get("intent_perspectives", {})
    assert payload["items"][0]["original_item"] == item
    assert payload["items"][0]["fields"]["failure_condition"] == ""
    assessment = schema["properties"]["items"]["properties"][row["id"]]["properties"]["failure_condition"]
    assert assessment["properties"]["reason"]["minLength"] == 1
    assert "non_factual" in assessment["properties"]["state"]["enum"]
    assert assessment["properties"]["citations"]["minItems"] == 0


def test_true_test_claim_with_header_citation_uses_existing_rewrite_to_repair_binding():
    from review_bot.application.intent_context import build_intent_task_input
    from tests.fixtures.intent_review import digest
    task = make_task()
    path = "tests/test_label.py"
    header = "import os\nimport sys\nimport json\nimport pathlib\nimport subprocess\nimport tempfile\nimport unittest\n\n"
    assertion = '    assert label(7) == "7"\n'
    content = header + "def test_label():\n    from src.caller import label\n" + assertion
    def replace_test(snapshot):
        files = tuple(replace(file, content=content, content_hash=digest(content)) if file.path == path else file for file in snapshot.files)
        inventory = digest(":".join(f"{file.path}:{file.content_hash}" for file in sorted(files, key=lambda file: file.path)))
        return replace(snapshot, files=files, inventory_hash=inventory)
    task = build_intent_task_input(task.snapshot, replace_test(task.base), replace_test(task.head),
        related_paths=("src/caller.py", path))
    literal = "The supplied test source asserts that label(7) equals the string 7."
    checked_fields = []
    def reference(payload, needle):
        sources = facts.materialize_source_texts(payload) if "originals" in payload else payload["original_quotes"]
        return next(block["reference_id"] for source in sources if source["path"] == path and source["version"] == "head"
            for block in source["blocks"] if needle in block["text"])
    def transform(payload, body):
        stage = payload["stage"]
        if stage == "final_review":
            item = next(item for item in body["perspectives"] if item["perspective_id"] == "supporting_evidence")
            item.update(statement=literal, reasoning="The HEAD test contains an explicit assertion for the caller's expected string output.",
                citations=[{"reference_id": reference(payload, "import os")}])
        elif stage == "fact_check":
            rule = payload["canonical_facts"]["citation_grounding_rule"]
            assert "ORIGINAL item's citation set" in rule and "return unverifiable" in rule
            assert "do not silently repair" in rule and "empty or non_factual fields" in rule
            for row in payload["items"]:
                item = row["original_item"]
                if isinstance(item, dict) and item.get("citations"):
                    original_cite = item["citations"][0]
                    own_reference = next(entry["reference_id"] for source in payload["citation_catalog"]
                        if source["evidence_id"] == original_cite["evidence_id"] for entry in source["ranges"]
                        if (entry["start_byte"], entry["end_byte"]) == (original_cite["start_byte"], original_cite["end_byte"]))
                    for field, assessment in body["items"][row["item_id"]].items():
                        if assessment["state"] == "supported" and row["fields"][field]:
                            assessment["citations"] = [{"reference_id": own_reference}]
                if isinstance(item, dict) and item.get("perspective_id") == "supporting_evidence":
                    assert item["failure_condition"] == "" and row["fields"]["failure_condition"] == ""
                    assert body["items"][row["item_id"]]["failure_condition"]["state"] == "non_factual"
                    original_binding_supported = any(assertion.strip() in cite["quote"] for cite in item["citations"])
                    checked_fields.append(original_binding_supported)
                    if not original_binding_supported:
                        for field in ("statement", "reasoning"):
                            body["items"][row["item_id"]][field].update(state="supported",
                                reason="The claim is true in the full HEAD test; the checker selected the actual assertion range.",
                                citations=[{"reference_id": reference(payload, assertion.strip())}])
        elif stage == "fact_rewrite":
            item = deepcopy(payload["items"][0]["original_item"])
            assert item["perspective_id"] == "supporting_evidence"
            item["citations"] = [{"reference_id": reference(payload, assertion.strip())}]
            return {"replacement": item, "correction_reason": "Replace the import-only citation with the exact test assertion range; keep the supported prose.",
                "correction_citations": item["citations"]}
        return body
    review = PartitionModelClient(transform=transform)
    result = run_partition_review(task, "intent", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert checked_fields == [False, True]
    original = next(item for item in result.partition["pre_verification_final"]["perspectives"] if item["perspective_id"] == "supporting_evidence")
    accepted = next(item for item in result.perspectives if item["perspective_id"] == "supporting_evidence")
    assert original["citations"][0]["quote"] == header and original["citations"][0]["end_line"] == 8
    assert accepted["statement"] == original["statement"] == literal
    assert accepted["reasoning"] == original["reasoning"] and accepted["failure_condition"] == ""
    assert any(assertion.strip() in cite["quote"] and cite["begin_line"] == 9 for cite in accepted["citations"])
    entry = next(entry for entry in result.partition["fact_verification"]["items"] if entry["original"] == original)
    first, second = entry["attempts"]
    assert first["assessments"]["statement"]["state"] == "supported"
    assert first["check"]["response"]["items"][entry["item_id"]]["statement"]["state"] == "supported"
    gap = next(issue for issue in first["issues"] if issue["field"] == "statement")
    assert gap["state"] == "unverifiable" and gap["binding_gap_ranges"]
    assert any(assertion.strip() in cite["quote"] for cite in first["check"]["cited_evidence"])
    assert second["assessments"]["statement"]["state"] == "supported"
    assert len([request for request in review.requests if json.loads(request.user_content)["stage"] == "fact_rewrite"]) == 1
    assert result.summary == result.partition["pre_verification_final"]["summary"]
    validate_partition_result(result, task)
    changed = deepcopy(result)
    changed_entry = next(value for value in changed.partition["fact_verification"]["items"] if value["item_id"] == entry["item_id"])
    changed_entry["attempts"][0]["issues"][0]["binding_gap_ranges"] = []
    with pytest.raises(PartitionError, match="stored_fact_assessment_changed"):
        validate_partition_result(changed, task)



def test_citation_binding_union_accepts_adjacent_ranges_and_skips_empty_nonfactual_and_summary():
    from review_bot.application.source_partition import SourceOriginal
    source = SourceOriginal("S-union", "abcdefghij", "source", "a.py", "head", "a" * 40)
    def cite(start, end):
        return {"evidence_id": source.evidence_id, "start_byte": start, "end_byte": end,
            "quote": source.raw[start:end].decode()}
    def reference(start, end):
        return {"reference_id": "Q" + hash_json((source.evidence_id, source.digest, start, end))[:12]}
    row = {"id": "findings:0", "kind": "findings", "item": {"statement": "A supported source assertion.",
        "reasoning": "Could the author consider this alternative?", "failure_condition": "", "citations": [cite(0, 4), cite(4, 8)]}}
    assessments = {"statement": {"state": "supported", "reason": "Exact proof.", "citations": [reference(0, 8)]},
        "reasoning": {"state": "non_factual", "reason": "Pure question.", "citations": [reference(8, 10)]},
        "failure_condition": {"state": "supported", "reason": "Empty field, ignored by binding guard.", "citations": [reference(8, 10)]}}
    unchanged = deepcopy((row, assessments))
    assert facts.citation_binding_issues(row, assessments, [source], [cite(0, 8), cite(8, 10)]) == []
    assert (row, assessments) == unchanged
    assert facts.citation_binding_issues({"id": "summary", "kind": "summary", "item": "No own citations."},
        {"text": assessments["statement"]}, [source], [cite(8, 10)]) == []


def test_citation_binding_distinct_evidence_identity_remains_a_conservative_gap():
    from review_bot.application.source_partition import SourceOriginal
    original = SourceOriginal("S-head", "same bytes", "source", "a.py", "head", "a" * 40)
    other = SourceOriginal("S-base", "same bytes", "source", "a.py", "base", "b" * 40)
    own = {"evidence_id": original.evidence_id, "start_byte": 0, "end_byte": len(original.raw), "quote": original.text}
    witness = {"evidence_id": other.evidence_id, "start_byte": 0, "end_byte": len(other.raw), "quote": other.text}
    reference = {"reference_id": "Q" + hash_json((other.evidence_id, other.digest, 0, len(other.raw)))[:12]}
    row = {"id": "findings:0", "kind": "findings", "item": {"statement": "A source claim.", "reasoning": "", "failure_condition": "", "citations": [own]}}
    assessment = {"statement": {"state": "supported", "reason": "Checker chose the other frozen identity.", "citations": [reference]}}
    gap, = facts.citation_binding_issues(row, assessment, [original, other], [witness])
    assert gap["kind"] == "source_assertion" and gap["state"] == "unverifiable"
    assert gap["binding_gap_ranges"] == [{"evidence_id": other.evidence_id, "start_byte": 0, "end_byte": len(other.raw)}]
    assert assessment["statement"]["state"] == "supported" and row["item"]["citations"] == [own]



def test_architecture_rewrite_declares_existing_concern_guard_and_analysis_attribution():
    from review_bot.application.source_partition import SourceOriginal, make_span
    base = SourceOriginal("S-base", "class A {}\n", "source", "A.java", "base", "a" * 40)
    head = SourceOriginal("S-head", "class A { void lookup() {} }\n", "source", "A.java", "head", "b" * 40)
    span = make_span(head, 0, len(head.raw), "test")
    quote = {"evidence_id": span.evidence_id, "start_byte": span.start_byte, "end_byte": span.end_byte, "quote": span.text}
    item = {"statement": "Inspect the reported lookup complexity.", "reasoning": "The fixed PMD observation reports CognitiveComplexity 17 for lookup.",
        "failure_condition": "", "verdict": "improvement", "basis": "analysis_signal", "target_ids": ["pmd-lookup"], "citations": [quote]}
    final = {"summary": "Static review.", "findings": [item], "perspectives": [], "questions": []}
    row = facts.items(final)[1]
    details = [{"target_id": "pmd-lookup", "tool": "pmd", "change": "new", "paths": ["A.java"],
        "measurements": [{"name": "CognitiveComplexity", "base": None, "head": 17}]}]
    payload, schema, refs, _ = facts.request_spec("fact_rewrite", [row], final, [base, head], {"target_details": details}, kind="complete_originals")
    assert payload["canonical_facts"]["analysis_targets"] == details
    rule = payload["canonical_facts"]["architecture_analysis_grounding_rule"]
    assert "owning member" in rule and "newly reported analysis signal" in rule
    assert "both frozen base and head originals" in rule and "not evidence that its implementation is absent" in rule
    assert "Do not promote an improvement to concern" in payload["canonical_facts"]["architecture_rewrite_contract"]
    replacement_schema = schema["properties"]["replacement"]
    conditional = replacement_schema["allOf"][0]
    assert conditional["if"]["properties"]["verdict"]["const"] == "concern"
    assert conditional["then"]["properties"]["failure_condition"] == {"type": "string", "minLength": 1, "pattern": r"\S"}
    head_refs = [key for key, value in refs.items() if value.path and value.version == "head"]
    assert conditional["then"]["properties"]["citations"]["contains"]["properties"]["reference_id"]["enum"] == head_refs
    cite = [{"reference_id": head_refs[0]}]
    raw = {"replacement": {**item, "verdict": "concern", "citations": cite}, "correction_reason": "Source-based correction.", "correction_citations": cite}
    for empty in ("", "   "):
        raw["replacement"]["failure_condition"] = empty
        with pytest.raises(PartitionError, match="fact_rewrite_concern_without_head_condition"):
            facts.accept_rewrite(raw, row, refs, "architecture")
    raw["replacement"]["failure_condition"] = "When lookup receives an unbounded input, HEAD cannot complete the operation."
    assert facts.accept_rewrite(raw, row, refs, "architecture")["verdict"] == "concern"
    base_ref = next(key for key, value in refs.items() if value.version == "base")
    raw["replacement"]["citations"] = [{"reference_id": base_ref}]
    with pytest.raises(PartitionError, match="fact_rewrite_concern_without_head_condition"):
        facts.accept_rewrite(raw, row, refs, "architecture")
    for verdict in ("improvement", "insufficient_context"):
        raw["replacement"].update(verdict=verdict, failure_condition="")
        assert facts.accept_rewrite(raw, row, refs, "architecture")["verdict"] == verdict


def test_architecture_only_rewrite_schema_is_absent_from_intent_findings_and_wire(monkeypatch):
    from review_bot.application.source_partition import SourceOriginal, make_span
    from review_bot.domain.intent_perspectives import PERSPECTIVES
    source = SourceOriginal("S-wire", "def answer():\n    return 42\n", "source", "a.py", "head", "a" * 40)
    span = make_span(source, 0, len(source.raw), "test")
    quote = {"evidence_id": span.evidence_id, "start_byte": span.start_byte, "end_byte": span.end_byte, "quote": span.text}
    common = {"statement": "A source assertion.", "reasoning": "A source explanation.", "failure_condition": "", "citations": [quote]}
    finding = {**common, "basis": "observed_structure", "verdict": "improvement", "target_ids": []}
    final = {"summary": "Static review.", "findings": [finding], "perspectives": [{**common, "perspective_id": item.id, "verdict": "aligned"} for item in PERSPECTIVES], "questions": []}
    rows = facts.items(final)
    before = []
    for kind in ("complete_originals", "cited_originals", "cited_ranges"):
        payload, schema, refs, _ = facts.request_spec("fact_rewrite", rows[1:], final, [source], {"target_details": [{"tool": "source_review"}]}, kind=kind)
        assert "architecture_analysis_grounding_rule" not in payload["canonical_facts"]
        assert "architecture_rewrite_contract" not in payload["canonical_facts"]
        assert "allOf" not in schema["properties"]["items"]["properties"]["findings:0"]["properties"]["replacement"]
        before.append(json.dumps({"payload": payload, "schema": schema}, ensure_ascii=False, separators=(",", ":")))
    actual_rewrite_schema = facts.rewrite_schema
    def without_architecture_branch(row, refs, *, architecture=False):
        assert architecture is False
        return actual_rewrite_schema(row, refs)
    monkeypatch.setattr(facts, "rewrite_schema", without_architecture_branch)
    for index, kind in enumerate(("complete_originals", "cited_originals", "cited_ranges")):
        payload, schema, _, _ = facts.request_spec("fact_rewrite", rows[1:], final, [source], {"target_details": [{"tool": "source_review"}]}, kind=kind)
        assert json.dumps({"payload": payload, "schema": schema}, ensure_ascii=False, separators=(",", ":")) == before[index]


def test_original_support_reference_ids_use_exact_union_and_gate_only_supported_fields():
    from review_bot.application.source_partition import SourceOriginal
    source = SourceOriginal("S-ids", "abcdefghij", "source", "a.py", "head", "a" * 40)
    citations = [{"evidence_id": source.evidence_id, "start_byte": start, "end_byte": end, "quote": source.raw[start:end].decode()}
        for start, end in ((0, 4), (4, 8))]
    item = {"statement": "A source assertion.", "reasoning": "A pure question?", "failure_condition": "", "citations": citations}
    final = {"summary": "No own citations.", "findings": [item], "perspectives": [], "questions": []}
    rows = facts.items(final)
    for start, end, has_support in ((0, 8, True), (8, 10, False)):
        payload, schema, refs, _ = facts.request_spec("fact_check", rows, final, [source], {}, kind="cited_ranges",
            ranges=[{"evidence_id": source.evidence_id, "start_byte": start, "end_byte": end}])
        summary, finding = payload["items"]
        assert "original_support_reference_ids" not in summary and finding["original_item"] == item
        ids = finding["original_support_reference_ids"]
        assert bool(ids) is has_support
        if has_support:
            combined = "Q" + hash_json((source.evidence_id, source.digest, 0, 8))[:12]
            assert combined in ids
        assert set(ids) == {key for key, span in refs.items() if span.evidence_id == source.evidence_id and 0 <= span.start_byte < span.end_byte <= 8}
        fields_schema = schema["properties"]["items"]["properties"]["findings:0"]["properties"]
        condition = fields_schema["statement"]["allOf"][0]
        assert condition["if"]["properties"]["state"]["const"] == "supported"
        allowed = condition["then"]["properties"]["citations"]["items"]
        assert (allowed["properties"]["reference_id"]["enum"] == ids) if has_support else allowed is False
        assert "allOf" not in fields_schema["failure_condition"]
        assert "allOf" not in schema["properties"]["items"]["properties"]["summary"]["properties"]["text"]
        assert set(fields_schema["statement"]["properties"]["citations"]["items"]["properties"]["reference_id"]["enum"]) == set(refs)
        preservation = payload["canonical_facts"]["citation_grounding_rule"]
        assert "preserve the original statement, reasoning, failure_condition and verdict exactly" in preservation



def test_korean_tcc_metric_extraction_matches_three_recorded_native_controls():
    from review_bot.application import partition_choices
    # Hashes cover the exact saved probe state, Korean options/question and choice schema.
    controls = [{'item_id': 'findings:5',
      'prose': {'statement': 'RagFlowTargetAdapter는 ATFD 9, TCC 0.070, WMC 80으로 외부 데이터 접근이 많고 응집도가 낮은 God Class 경향을 보인다.',
                'reasoning': 'PMD가 보고한 ATFD 9, TCC 0.070, WMC 80은 클래스가 여러 외부 객체/데이터에 접근하면서 메서드 간 응집도가 낮음을 나타낸다. 이는 구조적 관찰이며, '
                             'supplied code만으로는 런타임 실패를 의미하지 않는다.',
                'failure_condition': ''},
      'expected_request_parts_sha256': 'a38d2fbf996af2f9151d2e0032d3e0a249e15e82832e412cc70d33c4c18c62d4'},
     {'item_id': 'findings:5',
      'prose': {'statement': 'TCC가 낮으므로 결합도가 심각하지 않다.', 'reasoning': 'TCC 수치로 결합도가 낮음을 판단한다.', 'failure_condition': ''},
      'expected_request_parts_sha256': 'd060cbd28587b3595dab94b93c501769c30536a70e73477fb5c269b46685012b'},
     {'item_id': 'findings:5',
      'prose': {'statement': 'TCC가 낮아 응집도가 낮다.', 'reasoning': '이 수치는 결합도를 나타내지 않는다.', 'failure_condition': ''},
      'expected_request_parts_sha256': 'efc545fd4d764c05a91d69bfbddde31e4ce7d3c493a23997b4a95345ceb6ae3a'}]
    for control in controls:
        row = {"id": control["item_id"], "kind": "findings", "item": control["prose"]}
        korean = list(facts.metric_specs(row, language="ko"))
        tcc = next(spec for spec in korean if spec["state"]["target_metric"] == "TCC")
        assert hash_json({"state": tcc["state"], "options": tcc["options"], "question": tcc["question"],
            "schema": partition_choices.choice_schema(tcc["options"])}) == control["expected_request_parts_sha256"]
        assert tcc["options"] == facts.KOREAN_TCC_OPTIONS and tcc["question"] == facts.KOREAN_TCC_QUESTION
        english = next(spec for spec in facts.metric_specs(row) if spec["state"]["target_metric"] == "TCC")
        assert english["options"] == facts.CATEGORY_OPTIONS["TCC"] and english["question"] == facts.CATEGORY_QUESTION
        for spec in korean:
            if spec["state"]["target_metric"] != "TCC":
                assert spec["options"] == facts.CATEGORY_OPTIONS[spec["state"]["target_metric"]]
                assert spec["question"] == facts.CATEGORY_QUESTION



@pytest.mark.parametrize("finding_kind", ["method_pmd", "class_pmd", "cpd"])
def test_prepared_analysis_finding_owners_and_raw_details_reach_fact_requests(finding_kind):
    from dataclasses import asdict
    from review_bot.application.delta import build_delta
    from review_bot.application.partition_review import prepare_originals
    from review_bot.application.source_partition import make_span
    from review_bot.domain.models_v1 import CpdDuplication, CpdOccurrence, PmdViolation
    from tests.unit.test_context_and_architecture_review import _make_snapshot
    from tests.unit.test_delta_adversarial import make_test_run
    path = "src/main/java/demo/RagDocumentContentSupport.java"
    partner = "src/main/java/demo/RagDocumentReader.java"
    fragment = 'String listDocumentsByKeyword(String keyword) {\n    return keyword == null ? "" : keyword.trim();\n}\n'
    neighbor = 'boolean matchesContentHashFragment(String value) { return value != null; }\n'
    source = "package demo;\nclass RagDocumentContentSupport {\n" + "// unchanged context\n" * 125 + fragment + neighbor + "}\n"
    files = {path: source, partner: source.replace("RagDocumentContentSupport", "RagDocumentReader"), "build.gradle": "plugins { id 'java' }\n"}
    snapshot = _make_snapshot(files, files)
    base_pmd = head_pmd = base_cpd = head_cpd = ()
    if finding_kind != "cpd":
        method = "listDocumentsByKeyword" if finding_kind == "method_pmd" else None
        rule = "CognitiveComplexity" if method else "GodClass"
        message = ("The method 'listDocumentsByKeyword(String)' has a cognitive complexity of 17, current threshold is 15."
            if method else "Possible God Class (WMC=80, ATFD=9, TCC=0.070)")
        finding = PmdViolation(path=path, raw_path="/workspace/" + path, begin_line=128 if method else 2,
            end_line=130 if method else 132, begin_column=5, end_column=71, rule=rule, ruleset="Design", priority=3,
            klass="RagDocumentContentSupport", method=method, package="demo", message=message,
            code_fragment=fragment if method else source)
        prior_message = message.replace("complexity of 17", "complexity of 16") if method else "Possible God Class (WMC=50, ATFD=5, TCC=0.070)"
        base_pmd, head_pmd = (replace(finding, message=prior_message),), (finding,)
    else:
        finding = CpdDuplication(lines=3, tokens=120, occurrences=tuple(CpdOccurrence(path=value,
            raw_path="/workspace/" + value, begin_line=128, end_line=130, begin_column=5, end_column=71,
            end_line_inferred=False, code_fragment=fragment) for value in (path, partner)), code_fragment=fragment)
        head_cpd = (finding,)
    delta = build_delta(make_test_run(snapshot.base.sha, source_files=(path, partner), pmd_violations=base_pmd, cpd_duplications=base_cpd),
        make_test_run(snapshot.head.sha, source_files=(path, partner), pmd_violations=head_pmd, cpd_duplications=head_cpd))
    task = build_task_input(snapshot, delta, Classification("general"))
    originals, meta = prepare_originals(task, "architecture")
    selected = next(signal.signal for signal in task.signals if signal.role == "review")
    target = next(detail for detail in meta["target_details"] if detail["target_id"] == selected.signal_id)
    assert target["base_findings"] == [asdict(value) for value in selected.base_findings]
    assert target["head_findings"] == [asdict(value) for value in selected.head_findings]
    assert target["head_findings"] == [asdict(finding)]
    head = next(original for original in originals if original.path == path and original.version == "head")
    span = make_span(head, 0, len(head.raw), "test")
    quote = {"evidence_id": span.evidence_id, "start_byte": span.start_byte, "end_byte": span.end_byte, "quote": span.text}
    item = {"statement": "Inspect the reported analysis observation.", "reasoning": "Keep its reported owner, message and exact source fragment available.",
        "failure_condition": "", "verdict": "improvement", "basis": "analysis_signal", "target_ids": [selected.signal_id], "citations": [quote]}
    final = {"summary": "Static analysis review.", "findings": [item], "perspectives": [], "questions": []}
    row = facts.items(final)[1]
    for stage in ("fact_check", "fact_rewrite"):
        payload, _, _, _ = facts.request_spec(stage, [row], final, originals, meta, kind="complete_originals")
        forwarded = next(detail for detail in payload["canonical_facts"]["analysis_targets"] if detail["target_id"] == selected.signal_id)
        assert forwarded == target
        # JSON arrays/null are the actual model wire, not only Python metadata.
        actual = json.loads(json.dumps(forwarded, ensure_ascii=False))
        assert actual["base_findings"] == json.loads(json.dumps([asdict(value) for value in selected.base_findings]))
        assert actual["head_findings"] == json.loads(json.dumps([asdict(finding)]))
        evidence = actual["head_findings"][0]
        assert evidence["code_fragment"] == finding.code_fragment
        if finding_kind == "cpd":
            assert [occurrence["path"] for occurrence in evidence["occurrences"]] == [path, partner]
            assert all(occurrence["begin_column"] == 5 and occurrence["end_column"] == 71 and
                occurrence["code_fragment"] == fragment for occurrence in evidence["occurrences"])
        else:
            assert evidence["method"] == finding.method
            assert evidence["message"] == finding.message and evidence["rule"] == finding.rule
            assert evidence["klass"] == "RagDocumentContentSupport" and evidence["ruleset"] == "Design"
            assert evidence["begin_column"] == 5 and evidence["end_column"] == 71
            if finding_kind == "method_pmd":
                assert evidence["method"] == "listDocumentsByKeyword" and "complexity of 17" in evidence["message"]
                assert "matchesContentHashFragment" not in evidence["message"]
            else:
                assert evidence["method"] is None
