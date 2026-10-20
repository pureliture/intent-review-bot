from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
from review_bot.application.partition_review import run_partition_review, validate_partition_result
from review_bot.application.source_partition import PartitionError
from review_bot.domain.models_v1 import ModelResponse
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


def limits(**kwargs):
    return PartitionReviewLimits(max_total_tokens=1000000, max_calls=128, **kwargs)


def test_native_measured_plan_observation_selection_and_review_reach_eight_perspectives():
    task = make_task()
    review, tev = PartitionModelClient(), PartitionModelClient(name="tev", group_all=True)
    result = run_partition_review(task, "intent", review, tev, limits=limits())
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    assert len(result.perspectives) == 8
    assert result.scope["tests_executed_by_bot"] is False
    assert result.scope["repository_wide"] is False
    assert result.partition["units_prepared"]
    assert len(result.partition["groups"]) == 1
    assert len(result.partition["groups"][0]["unit_ids"]) > 1
    tev_stages = [json.loads(r.user_content)["stage"] for r in tev.requests]
    assert tev_stages == ["plan_partitions", "observe_partition", "observe_partition", "select_review_scope"] + ["select_perspective"] * 8
    assert [json.loads(r.user_content)["stage"] for r in review.requests] == ["review_partition", "final_review"]
    assert result.usage["calls"] == 14 and result.usage["usage_complete"] is True
    import hashlib
    from pathlib import Path
    from review_bot.adapters import intent_planner, llamacpp, llamacpp_macbook
    for module in (intent_planner, llamacpp, llamacpp_macbook):
        path = Path(module.__file__)
        assert result.partition["identity"]["implementation"]["adapters/" + path.name] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "비공개" in format_partition_review_markdown(result, task)
    with pytest.raises(ValueError, match="private_only"):
        format_partition_review_markdown(result, task, publication_mode="comment")


def test_initial_group_judgements_and_all_their_references_reach_final_without_summary():
    from tests.fixtures.intent_review import digest
    from review_bot.application.intent_context import build_intent_task_input
    from review_bot.application.partition_review import PERSPECTIVE_IDS
    task = make_task()
    text = "".join(f"value_{i} = {i}\n" for i in range(600))
    def extend(code):
        files = tuple(replace(f, content=text, content_hash=digest(text)) if f.path == "src/value.py" else f for f in code.files)
        inventory = digest(":".join(f"{f.path}:{f.content_hash}" for f in sorted(files, key=lambda f: f.path)))
        return replace(code, files=files, inventory_hash=inventory)
    task = build_intent_task_input(task.snapshot, extend(task.base), extend(task.head), related_paths=task.reviewed_paths)
    def distinct_evidence(payload, body):
        if payload["stage"] == "review_partition":
            refs = [{"reference_id": row["reference_id"]} for item in payload["citation_catalog"] for row in item["ranges"]]
            assert len(refs) >= 48
            for i, key in enumerate(PERSPECTIVE_IDS):
                body["draft_assessments"][key]["citations"] = refs[i * 6:(i + 1) * 6]
        return body
    review = PartitionModelClient(transform=distinct_evidence)
    result = run_partition_review(task, "intent", review, PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    roots = result.partition["review_observations"]
    assert all(tuple(row["raw_response"]["draft_assessments"]) == PERSPECTIVE_IDS for row in roots)
    assert len(roots[0]["citations"]) >= 48
    final = next(json.loads(r.user_content) for r in review.requests if json.loads(r.user_content)["stage"] == "final_review")
    assert final["observations"][0]["raw_response"] == roots[0]["raw_response"]
    validate_partition_result(result, task)


def test_initial_review_cannot_omit_a_perspective_before_final_review():
    def omit(payload, body):
        if payload["stage"] == "review_partition":
            body["draft_assessments"].pop(next(iter(body["draft_assessments"])))
        return body
    result = run_partition_review(make_task(), "intent", PartitionModelClient(transform=omit),
        PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "failed"
    assert not result.partition.get("integration_pages")


def test_run_identity_binds_the_actual_output_language():
    task = make_task()
    results = [run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True),
        limits=limits(), language=language) for language in ("ko", "en")]
    assert all(result.status == "completed" for result in results)
    assert results[0].run_key != results[1].run_key
    for result in results:
        assert result.partition["identity"]["language"] == result.language
        validate_partition_result(result, task)
    with pytest.raises(PartitionError, match="identity_mismatch"):
        validate_partition_result(replace(results[0], language="en"), task)


def test_code_group_can_cite_the_shared_original_pr_purpose_in_its_draft():
    def cite_shared_purpose(payload, body):
        if payload["stage"] == "review_partition":
            purpose = payload["pr_purpose"][0]["evidence_id"]
            ref = next(row["ranges"][0]["reference_id"] for row in payload["citation_catalog"] if row["evidence_id"] == purpose)
            body["draft_assessments"]["problem_success"]["citations"] = [{"reference_id": ref}]
        return body
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(transform=cite_shared_purpose),
        PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert len(result.partition["groups"]) > 1
    by_unit = {u["unit_id"]: u for u in result.partition["units"]}
    assert any(any(not c["path"] and c["evidence_id"] not in {by_unit[key]["evidence_id"] for key in row["covered_unit_ids"]}
        for c in row["citations"]) for row in result.partition["review_observations"])
    validate_partition_result(result, task)


def test_actual_aggregate_overflow_preserves_results_in_separate_final_pages():
    def count(request):
        payload = json.loads(request.user_content)
        if payload["stage"] == "final_review":
            return 200 + 100 * len(payload["observations"])
        return 100
    task = make_task()
    review, tev = PartitionModelClient(counter=count), PartitionModelClient(counter=count)
    review.profile = replace(review.profile, max_input_tokens=350)
    result = run_partition_review(task, "intent", review, tev, limits=limits())
    assert result.status == "completed", result.reason
    assert not result.partition["reductions"]
    assert len(result.partition["integration_pages"]) > 1
    assert not any(json.loads(r.user_content)["stage"] == "reduce_observations" for r in review.requests)
    assert [row for page in result.partition["integration_pages"] for row in page["inputs"]] == result.partition["review_observations"]
    assert all(row["assessments"] for row in result.perspectives)
    assert result.partition["final_covered_unit_ids"] == sorted(u["unit_id"] for u in result.partition["units"])
    validate_partition_result(result, task)


def test_native_measurement_unavailable_stops_before_generation():
    def unavailable(_):
        raise RuntimeError("no tokenizer")
    client = PartitionModelClient(counter=unavailable)
    result = run_partition_review(make_task(), "intent", client, client, limits=limits())
    assert result.status == "not_run" and result.reason == "token_count_unavailable"
    assert not client.requests


def test_measure_only_prepares_native_fitted_spans_without_generation():
    task = make_task()
    review, tev = PartitionModelClient(), PartitionModelClient()
    result = run_partition_review(task, "intent", review, tev, limits=limits(measure_only=True))
    assert result.status == "measured" and result.reason == "native_input_prepared"
    assert result.usage["calls"] == 0 and not review.requests and not tev.requests
    assert review.count_requests and tev.count_requests
    assert result.partition["native_unit_measurements"]
    validate_partition_result(result, task)


def test_unknown_usage_is_preserved_and_never_retried():
    client = PartitionModelClient(transform=lambda p, b: ModelResponse(json.dumps(b), None))
    result = run_partition_review(make_task(), "intent", client, client, limits=limits())
    assert result.status == "failed" and result.reason == "usage_unavailable"
    assert len(client.requests) == 1
    assert result.usage["input_tokens"] is None and not result.usage["usage_complete"]


def test_outside_span_citation_and_stored_quote_tampering_are_rejected():
    def invalid(payload, body):
        if payload["stage"] == "review_partition":
            body["citations"][0]["reference_id"] = "Q-not-supplied"
        return body
    task = make_task()
    bad = run_partition_review(task, "intent", PartitionModelClient(transform=invalid), PartitionModelClient(), limits=limits())
    assert bad.status == "failed" and bad.reason == "citation_reference_not_supplied"
    assert any(r.reason == "invalid_model_response" for r in bad.stages)
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(), limits=limits())
    result.perspectives[0]["citations"][0]["quote"] = "fabricated"
    with pytest.raises(PartitionError, match="quote_mismatch|assembly_mismatch"):
        validate_partition_result(result, task)


def test_job_fence_cancels_before_any_model_call():
    client = PartitionModelClient()
    result = run_partition_review(make_task(), "intent", client, client, limits=limits(), before_stage=lambda: False)
    assert result.status == "not_run" and result.reason == "stage_gate_rejected"
    assert not client.requests


def test_modes_share_one_call_budget_without_double_counting_usage():
    import time
    from review_bot.application.code_first_intent_review import _Budget
    cap = PartitionReviewLimits(max_total_tokens=1000000, max_calls=15)
    shared = _Budget(cap, time.monotonic() + 900, None)
    review, tev = PartitionModelClient(), PartitionModelClient(group_all=True)
    first = run_partition_review(make_task(), "intent", review, tev, limits=cap, shared_budget=shared)
    assert first.status == "completed" and first.usage["calls"] == 14
    second = run_partition_review(make_task(), "intent", review, tev, limits=cap, shared_budget=shared)
    assert second.reason == "call_limit_exceeded" and second.usage["calls"] == 0
    assert len(review.requests) + len(tev.requests) == 14


def test_architecture_uses_same_executor_and_keeps_analysis_as_evidence():
    from tests.unit.test_context_and_architecture_review import _make_snapshot
    from review_bot.application.context import build_task_input
    from review_bot.application.delta import compute_delta
    from review_bot.domain.models_v1 import Classification
    snapshot = _make_snapshot({"src/A.java": "class A {}\n"}, {"src/A.java": "class A { void run() {} }\n"})
    delta = compute_delta(base_findings=[], head_findings=[{"rule": "CognitiveComplexity", "path": "src/A.java", "metric": 20, "method": "run"}],
        base_sha=snapshot.base.sha, head_sha=snapshot.head.sha)
    task = build_task_input(snapshot, delta, Classification("general"))
    result = run_partition_review(task, "architecture", PartitionModelClient(), PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert result.review_type == "A" and not result.perspectives
    validate_partition_result(result, task)


def test_report_only_group_cannot_request_a_head_concern_but_head_code_still_can():
    from tests.unit.test_context_and_architecture_review import _make_snapshot
    from review_bot.application.context import build_task_input
    from review_bot.application.delta import compute_delta
    from review_bot.domain.models_v1 import Classification
    snapshot = _make_snapshot({"src/A.java": "class A {}\n"}, {"src/A.java": "class A { void run() {} }\n"})
    delta = compute_delta(base_findings=[], head_findings=[{"rule": "CognitiveComplexity", "path": "src/A.java", "metric": 20, "method": "run"}],
        base_sha=snapshot.base.sha, head_sha=snapshot.head.sha)
    task = build_task_input(snapshot, delta, Classification("general"))
    review = PartitionModelClient()
    result = run_partition_review(task, "architecture", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    report_options, head_options = [], []
    for request in review.requests:
        payload = json.loads(request.user_content)
        if payload["stage"] != "review_partition" or not payload["draft_assessment_keys"]:
            continue
        options = [entry["properties"]["verdict"]["enum"] for entry in
            request.response_schema["properties"]["draft_assessments"]["properties"].values()]
        if any(source["path"] and source["version"] == "head" for source in payload["originals"]):
            head_options.extend(options)
        else:
            report_options.extend(options)
    assert report_options and head_options
    assert all("deferred" in options and "concern" not in options and "conditional_risk" not in options
        for options in report_options)
    assert all("concern" in options and "conditional_risk" in options for options in head_options)
    validate_partition_result(result, task)


def test_non_head_intent_groups_offer_only_evidence_compatible_draft_verdicts():
    review = PartitionModelClient()
    result = run_partition_review(make_task(), "intent", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    requests = [request for request in review.requests if json.loads(request.user_content)["stage"] == "review_partition"
        and not any(source["path"] and source["version"] == "head"
            for source in json.loads(request.user_content)["originals"])]
    assert requests
    for request in requests:
        for entry in request.response_schema["properties"]["draft_assessments"]["properties"].values():
            assert "concern" not in entry["properties"]["verdict"]["enum"]
            assert "conditional_risk" not in entry["properties"]["verdict"]["enum"]


def test_out_of_schema_non_head_concern_still_fails_instead_of_bypassing_grounding():
    def unsupported(payload, body):
        if payload["stage"] == "review_partition" and not any(
            source["path"] and source["version"] == "head" for source in payload["originals"]):
            for draft in body["draft_assessments"].values():
                draft.update(verdict="concern", failure_condition="An unsupported head failure.")
        return body
    result = run_partition_review(make_task(), "intent", PartitionModelClient(transform=unsupported),
        PartitionModelClient(), limits=limits())
    assert result.status == "failed" and result.reason == "draft_concern_without_head_failure_evidence"
    assert not result.partition.get("integration_pages")


def test_architecture_draft_can_recommend_structural_improvement_without_inventing_a_runtime_failure():
    from tests.unit.test_context_and_architecture_review import _make_snapshot
    from review_bot.application.context import build_task_input
    from review_bot.application.delta import compute_delta
    from review_bot.domain.models_v1 import Classification
    snapshot = _make_snapshot({"src/A.java": "class A {}\n"}, {"src/A.java": "class A { void run() {} }\n"})
    delta = compute_delta(base_findings=[], head_findings=[{"rule": "CognitiveComplexity", "path": "src/A.java", "metric": 20, "method": "run"}],
        base_sha=snapshot.base.sha, head_sha=snapshot.head.sha)
    task = build_task_input(snapshot, delta, Classification("general"))
    def suggest(payload, body):
        if payload["stage"] == "review_partition":
            for draft in body["draft_assessments"].values():
                draft.update(verdict="improvement", statement="A grounded structural recommendation, not a demonstrated runtime defect.", failure_condition="")
        return body
    result = run_partition_review(task, "architecture", PartitionModelClient(transform=suggest), PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert any(draft["verdict"] == "improvement" for row in result.partition["review_observations"]
        for draft in row["raw_response"]["draft_assessments"].values())
    validate_partition_result(result, task)


def test_draft_insufficient_judgement_cannot_hide_its_gap_from_context_tracking():
    def missing(payload, body):
        if payload["stage"] == "review_partition":
            body["draft_assessments"]["problem_success"]["verdict"] = "insufficient_context"
        return body
    result = run_partition_review(make_task(), "intent", PartitionModelClient(transform=missing),
        PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "failed" and result.reason == "draft_insufficient_judgement_without_missing_context"


def test_bonsai_limit_refines_units_and_only_safe_candidates_are_offered():
    from review_bot.application.intent_context import build_intent_task_input
    from tests.fixtures.intent_review import digest
    task = make_task()
    text = "def render(value):\n" + "    value = value + 1\n" * 40 + "    return value\n"
    def extend(code):
        files = tuple(replace(f, content=text, content_hash=digest(text)) if f.path == "src/value.py" else f for f in code.files)
        inventory = digest(":".join(f"{f.path}:{f.content_hash}" for f in sorted(files, key=lambda f: f.path)))
        return replace(code, files=files, inventory_hash=inventory)
    task = build_intent_task_input(task.snapshot, extend(task.base), extend(task.head), related_paths=task.reviewed_paths)
    def count(request):
        p = json.loads(request.user_content)
        material = p.get("state", p)
        return 200 + sum(len(s["text"]) if "text" in s else sum(len(b["text"]) for b in s["blocks"])
            for s in [*material.get("originals", material.get("original_quotes", [])), *material.get("pr_purpose", [])])
    review = PartitionModelClient(counter=count)
    review.profile = replace(review.profile, max_input_tokens=500)
    tev = PartitionModelClient(counter=count, group_all=True)
    tev.profile = replace(tev.profile, max_input_tokens=2000)
    result = run_partition_review(task, "intent", review, tev, limits=limits())
    assert result.status == "completed", result.reason
    assert any(u["boundary"].startswith("line_fallback") for u in result.partition["units"])
    offered = [json.loads(r.user_content)["state"]["candidate_plans"] for r in tev.requests
        if json.loads(r.user_content)["stage"] == "plan_partitions"]
    assert offered
    assert all(set(r.response_schema["properties"]) == {"choice"} and r.max_output_tokens <= 16 for r in tev.requests)
    assert all(count(r) <= 500 for r in review.requests)
    assert all(count(r) <= 2000 for r in tev.requests)
    validate_partition_result(result, task)


def test_global_plan_precedes_mode_selection_and_is_reused_without_teaching_tev_twice():
    task = make_task()
    review, tev = PartitionModelClient(), PartitionModelClient(name="tev", group_all=True)
    plan = run_partition_review(task, "planning", review, tev, limits=limits(), available_modes=("intent",))
    assert plan.status == "planned", plan.reason
    assert plan.model["model"] == "tev"
    validate_partition_result(plan, task)
    assert not review.requests
    stages = [json.loads(r.user_content)["stage"] for r in tev.requests]
    assert stages.index("plan_partitions") < stages.index("observe_partition") < stages.index("select_review_scope")
    assert plan.partition["selected_modes"] == ["intent"]
    previous = len(tev.requests)
    result = run_partition_review(task, "intent", review, tev, limits=limits(), prepared_plan=plan)
    assert result.status == "completed", result.reason
    assert len(tev.requests) == previous
    assert result.partition["shared_plan_id"] == plan.partition["plan_id"]
    assert result.partition["plan_reuse"]["reused_group_ids"]
    assert not result.partition["plan_reuse"]["amendments"]
    validate_partition_result(result, task)


def test_global_plan_can_select_no_modes_and_nonselection_is_input_bound():
    def select_none(payload, body):
        if payload["stage"] in ("observe_partition", "select_review_scope") and payload["state"]["field"] in ("review_scope", "global_modes"):
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == "review:none")
        return body
    task = make_task()
    review, tev = PartitionModelClient(), PartitionModelClient(group_all=True, transform=select_none)
    plan = run_partition_review(task, "planning", review, tev, limits=limits(), available_modes=("intent",))
    assert plan.status == "planned" and not plan.partition["selected_modes"]
    result = run_partition_review(task, "intent", review, tev, limits=limits(), prepared_plan=plan)
    assert result.status == "not_run" and result.reason == "mode_not_selected_in_shared_plan"
    assert result.input_hash == task.input_hash and not review.requests
    validate_partition_result(result, task)


def test_uncertain_routing_is_preserved_for_the_explicit_final_mode_choice():
    def uncertain(payload, body):
        if payload["stage"] == "observe_partition" and payload["state"]["field"] == "context_sufficiency":
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == "missing_fixed_context")
        if payload["stage"] == "select_review_scope":
            assert {o["key"] for o in payload["options"]} == {
                "review:architecture,intent", "review:architecture", "review:intent", "review:none"}
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == "review:architecture,intent")
        return body
    task = make_task()
    plan = run_partition_review(task, "planning", PartitionModelClient(),
        PartitionModelClient(group_all=True, transform=uncertain), limits=limits(), available_modes=("architecture", "intent"))
    assert plan.status == "planned", plan.reason
    assert plan.partition["selected_modes"] == ["architecture", "intent"]
    assert "routing_override" not in plan.partition
    assert any(row["classification"]["scope_uncertain"] for row in plan.partition["observations"])
    validate_partition_result(plan, task)


def test_offline_architecture_corpus_excludes_unrelated_build_manifests():
    from tests.unit.test_context_and_architecture_review import _make_snapshot
    from review_bot.application.context import build_task_input
    from review_bot.application.delta import compute_delta
    from review_bot.application.partition_review import prepare_originals
    from review_bot.domain.models_v1 import Classification
    files = {"src/A.java": "class A {}\n", "build.gradle": "plugins { id 'java' }\n",
        "unrelated/build.gradle": "unrelated project\n", "other/pom.xml": "<project/>\n"}
    snap = _make_snapshot(files, files)
    snap = replace(snap, pr=replace(snap.pr, changed_files=("src/A.java",)))
    delta = compute_delta(base_findings=[], head_findings=[{"rule": "CognitiveComplexity", "path": "src/A.java", "metric": 20}],
        base_sha=snap.base.sha, head_sha=snap.head.sha)
    from review_bot.application.classification import classify_pr
    task = build_task_input(snap, delta, classify_pr(snap.pr.source_branch, snap.pr.labels))
    originals, _ = prepare_originals(task, "architecture")
    paths = {o.path for o in originals if o.path}
    assert paths == {"src/A.java", "build.gradle"}


def test_tev_cannot_return_a_plan_or_prose_instead_of_a_closed_choice():
    def bad(payload, body):
        if payload["stage"] == "plan_partitions":
            return {"groups": [{"unit_indexes": [0, 0]}]}
        return body
    tev = PartitionModelClient(transform=bad)
    result = run_partition_review(make_task(), "intent", PartitionModelClient(), tev, limits=limits())
    assert result.status == "failed"
    assert len(tev.requests) == 1
    assert set(tev.requests[0].response_schema["properties"]) == {"choice"}


def test_closed_decision_and_categorical_observations_are_replayed():
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    result.partition["observations"][0]["classification"]["mode_counts"]["intent"] += 1
    with pytest.raises(PartitionError, match="classification_mismatch"):
        validate_partition_result(result, task)


def test_routing_aggregation_is_lossless_and_never_uses_tev_prose_generation():
    def count(request):
        p = json.loads(request.user_content)
        if p["stage"] == "select_review_scope":
            return 200 + 100 * len(p["state"]["observations"])
        return 100
    task = make_task()
    review, tev = PartitionModelClient(counter=count), PartitionModelClient(counter=count)
    tev.profile = replace(tev.profile, max_input_tokens=350)
    plan = run_partition_review(task, "planning", review, tev, limits=limits(), available_modes=("intent",))
    assert plan.status == "planned", plan.reason
    assert plan.partition["routing_reductions"]
    assert all(r.max_output_tokens <= 16 and set(r.response_schema["properties"]) == {"choice"} for r in tev.requests)
    assert not any(json.loads(r.user_content)["stage"] == "reduce_observations" for r in tev.requests)
    validate_partition_result(plan, task)
    plan.partition["routing_reductions"][0]["output"]["classification"]["mode_counts"]["intent"] += 1
    with pytest.raises(PartitionError, match="aggregation_mismatch"):
        validate_partition_result(plan, task)


def test_stored_closed_choice_tampering_is_rejected():
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    result.partition["decisions"][0]["choice"] = "X"
    with pytest.raises(PartitionError, match="choice_identity_mismatch"):
        validate_partition_result(result, task)


def test_final_request_preserves_original_results_and_full_sources():
    task = make_task()
    client = PartitionModelClient(group_all=True)
    result = run_partition_review(task, "intent", client, client, limits=limits())
    assert result.status == "completed", result.reason
    final = next(json.loads(r.user_content) for r in client.requests if json.loads(r.user_content)["stage"] == "final_review")
    assert final["original_quotes"] and final["citation_catalog"]
    assert [row["raw_response"] for row in final["observations"]] == [row["raw_response"] for row in result.partition["review_observations"]]
    assert [row["raw_response_hash"] for row in final["observations"]] == [row["raw_response_hash"] for row in result.partition["review_observations"]]
    assert all("raw_response" in row for row in final["observations"])
    delivered = {(row["evidence_id"], row["start_byte"], row["end_byte"]) for row in final["original_quotes"]}
    assert all(any(key == unit["evidence_id"] and begin <= unit["start_byte"] < unit["end_byte"] <= end
        for key, begin, end in delivered) for unit in result.partition["units"])


def test_five_original_citations_are_preserved_without_the_old_four_quote_limit():
    def excessive(payload, body):
        if payload["stage"] == "review_partition":
            body["citations"] = body["citations"] * 5
        return body
    review = PartitionModelClient(transform=excessive)
    result = run_partition_review(make_task(), "intent", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert all(len(row["raw_response"]["citations"]) == 5 for row in result.partition["review_observations"])
    validate_partition_result(result, make_task())


def test_experimental_small_target_does_not_compress_a_fitting_live_request():
    task = make_task()
    review, tev = PartitionModelClient(group_all=True), PartitionModelClient(group_all=True)
    result = run_partition_review(task, "intent", review, tev, limits=limits(input_target_tokens=1))
    assert result.status == "completed", result.reason
    assert len(result.partition["groups"]) == 1
    assert len(result.partition["integration_pages"]) == 1
    assert not result.partition["reductions"]
    final = json.loads(review.requests[-1].user_content)
    assert [row["raw_response"] for row in final["observations"]] == [row["raw_response"] for row in result.partition["review_observations"]]
    assert [row["raw_response_hash"] for row in final["observations"]] == [row["raw_response_hash"] for row in result.partition["review_observations"]]
    validate_partition_result(result, task)


def test_final_tev_can_reject_a_provisional_positive_even_with_uncertain_context():
    def final_none(payload, body):
        if payload["stage"] == "observe_partition" and payload["state"]["field"] == "context_sufficiency":
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == "undetermined")
        if payload["stage"] == "select_review_scope":
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == "review:none")
        return body
    task = make_task()
    tev = PartitionModelClient(group_all=True, transform=final_none)
    plan = run_partition_review(task, "planning", PartitionModelClient(), tev,
        limits=limits(), available_modes=("intent",))
    assert plan.status == "planned", plan.reason
    assert not plan.partition["selected_modes"]
    assert plan.partition["observations"][0]["classification"]["mode_counts"] == {"intent": 1}
    assert "routing_override" not in plan.partition
    validate_partition_result(plan, task)


def test_provisional_missing_context_requires_explicit_evidenced_resolution():
    def provisional(payload, body):
        if payload["stage"] == "review_partition":
            body["missing_context"] = ["A provisional question about the supplied source."]
        return body
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(transform=provisional),
        PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    assert result.partition["context_history"][0]["missing_context"]
    assert result.partition["final"]["context_updates"]
    assert all(row["state"] == "resolved" and row["citations"]
        for row in result.partition["final"]["context_updates"].values())
    validate_partition_result(result, task)
    def erase_resolution(payload, body):
        body = provisional(payload, body)
        if payload["stage"] == "final_review":
            body["context_updates"] = {}
        return body
    bad = run_partition_review(task, "intent", PartitionModelClient(transform=erase_resolution),
        PartitionModelClient(group_all=True), limits=limits())
    assert bad.status == "partial_review"
    assert bad.partition["final"]["missing_context"]


def test_each_code_group_receives_the_same_fixed_pr_purpose():
    client = PartitionModelClient()
    task = make_task(body="Preserve the caller's string contract.")
    result = run_partition_review(task, "intent", client, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    groups = [json.loads(r.user_content) for r in client.requests
        if json.loads(r.user_content)["stage"] == "review_partition"]
    assert len(groups) > 1
    purposes = [group["pr_purpose"] for group in groups]
    assert all(purpose == purposes[0] for purpose in purposes)
    assert "Preserve the caller's string contract." in "".join(block["text"] for block in purposes[0][0]["blocks"])
    for group in groups:
        purpose_id = group["pr_purpose"][0]["evidence_id"]
        assert sum(s["evidence_id"] == purpose_id for s in [*group["originals"], *group["pr_purpose"]]) == 1
        assert group["original_byte_sizes"][purpose_id] == len("".join(
            block["text"] for block in group["pr_purpose"][0]["blocks"]).encode())


def test_rejected_grounding_is_rechecked_once_without_rewriting_the_rejected_response():
    task = make_task(repair=True)
    def invalid_first(payload, body):
        if payload["stage"] == "final_review" and "source_recheck" not in payload:
            prose = next(c for c in payload["citation_catalog"] if not c["path"])
            body["perspectives"][0].update(verdict="concern", failure_condition="A past base failure.",
                citations=[{"reference_id": prose["ranges"][0]["reference_id"]}])
        return body
    review = PartitionModelClient(group_all=True, transform=invalid_first)
    result = run_partition_review(task, "intent", review, PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    finals = [json.loads(r.user_content) for r in review.requests if json.loads(r.user_content)["stage"] == "final_review"]
    assert len(finals) == 2
    assert finals[0]["original_quotes"] == finals[1]["original_quotes"]
    assert finals[0]["observations"] == finals[1]["observations"]
    rejected = result.partition["rejected_final_attempts"]
    assert len(rejected) == 1 and rejected[0]["reason"] == "concern_without_head_failure_evidence"
    assert finals[1]["source_recheck"]["rejected_response"] == rejected[0]["response"]
    assert result.partition["final"]["correction_reason"]
    assert result.usage["usage_complete"]
    validate_partition_result(result, task)


def test_missing_target_review_cannot_be_reported_as_completed():
    def omit(payload, body):
        if payload["stage"] == "final_review" and body["target_reviews"]:
            body["target_reviews"].pop(next(iter(body["target_reviews"])))
        return body
    result = run_partition_review(make_task(), "intent", PartitionModelClient(transform=omit),
        PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "failed"
    assert len(result.partition["rejected_final_attempts"]) == 2


def test_unknown_usage_on_final_review_never_triggers_a_source_recheck():
    def unknown(payload, body):
        return ModelResponse(json.dumps(body), None) if payload["stage"] == "final_review" else body
    review = PartitionModelClient(transform=unknown)
    result = run_partition_review(make_task(), "intent", review, PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "failed" and result.reason == "usage_unavailable"
    assert len([r for r in review.requests if json.loads(r.user_content)["stage"] == "final_review"]) == 1
    assert not result.partition.get("rejected_final_attempts")


def test_model_can_request_one_frozen_original_recheck_and_preserve_the_initial_response():
    def request_once(payload, body):
        if payload["stage"] == "final_review" and "source_recheck" not in payload:
            original = next(row for row in payload["fixed_original_inventory"] if row["version"] == "head")
            body["source_requests"] = [{"evidence_id": original["evidence_id"], "reason": "Recheck the fixed head implementation."}]
        return body
    task = make_task()
    review = PartitionModelClient(transform=request_once)
    result = run_partition_review(task, "intent", review, PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    finals = [json.loads(r.user_content) for r in review.requests if json.loads(r.user_content)["stage"] == "final_review"]
    assert len(finals) == 2
    history = result.partition["source_recheck_requests"]
    assert len(history) == 1 and history[0]["admitted"] is True
    assert finals[1]["source_recheck"]["prior_response"] == history[0]["response"]
    assert finals[1]["observations"] == finals[0]["observations"]
    assert not result.partition["final"]["source_requests"]
    assert result.partition["final"]["correction_reason"]
    validate_partition_result(result, task)


@pytest.mark.parametrize("allowance,fit,expected_calls", [(1, True, 2), (0, True, 1), (1, False, 1)])
def test_unresolved_or_unfittable_source_recheck_is_partial_not_a_silent_success(allowance, fit, expected_calls):
    def always_request(payload, body):
        if payload["stage"] == "final_review":
            row = payload["fixed_original_inventory"][0]
            body["source_requests"] = [{"evidence_id": row["evidence_id"], "reason": "Necessary original recheck remains open."}]
        return body
    def count(request):
        payload = json.loads(request.user_content)
        return 100 if fit or "source_recheck" not in payload else 100000
    task = make_task()
    review = PartitionModelClient(counter=count, transform=always_request)
    result = run_partition_review(task, "intent", review, PartitionModelClient(group_all=True),
        limits=limits(max_rechecks=allowance))
    assert result.status == "partial_review", result.reason
    assert any(gap.startswith("source_recheck_pending:") for gap in result.partition["final"]["missing_context"])
    assert len([r for r in review.requests if json.loads(r.user_content)["stage"] == "final_review"]) == expected_calls
    validate_partition_result(result, task)


def test_source_recheck_cannot_expand_the_frozen_inventory_or_repeat_an_id():
    from review_bot.application.partition_review import _source_requests
    with pytest.raises(PartitionError, match="invalid_fixed_source_request"):
        _source_requests([{"evidence_id": "new-repository-path", "reason": "Explore more."}], {"S1": object()})
    with pytest.raises(PartitionError, match="invalid_fixed_source_request"):
        _source_requests([{"evidence_id": "S1", "reason": "Read."}] * 2, {"S1": object()})


def test_recheck_without_a_remaining_generation_call_retains_the_valid_final_as_partial():
    def request_more(payload, body):
        if payload["stage"] == "final_review":
            body["source_requests"] = [{"evidence_id": payload["fixed_original_inventory"][0]["evidence_id"],
                "reason": "Necessary static recheck."}]
        return body
    task = make_task()
    review = PartitionModelClient(transform=request_more)
    result = run_partition_review(task, "intent", review, PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=14))
    assert result.status == "partial_review", result.reason
    assert result.usage["calls"] == 14 and result.usage["usage_complete"]
    assert result.partition["source_recheck_requests"][0]["reason"] == "call_limit_exceeded"
    assert len([r for r in review.requests if json.loads(r.user_content)["stage"] == "final_review"]) == 1
    validate_partition_result(result, task)


def test_budget_expiry_between_recheck_measurement_and_generation_preserves_prior_final(monkeypatch):
    import time
    from review_bot.application.model_execution import ModelExecutionBudget
    from review_bot.domain.intent_models import IntentStageRecord
    cap = limits()
    budget = ModelExecutionBudget(cap, time.monotonic() + 900, None)
    execute = budget.execute
    def expires(request, client, profile, **kwargs):
        if "source_recheck" in json.loads(request.user_content):
            budget.records.append(IntentStageRecord(kwargs["label"], status="not_run", reason="task_deadline_exceeded"))
            return None, len(budget.records) - 1
        return execute(request, client, profile, **kwargs)
    monkeypatch.setattr(budget, "execute", expires)
    def request_more(payload, body):
        if payload["stage"] == "final_review":
            body["source_requests"] = [{"evidence_id": payload["fixed_original_inventory"][0]["evidence_id"],
                "reason": "Necessary static recheck."}]
        return body
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(transform=request_more),
        PartitionModelClient(group_all=True), limits=cap, shared_budget=budget)
    assert result.status == "partial_review", result.reason
    assert result.partition["source_recheck_requests"][0]["reason"] == "task_deadline_exceeded"
    assert result.summary and result.perspectives
    validate_partition_result(result, task)


def test_paginated_source_recheck_retains_the_original_delivery_plan_and_added_fixed_source():
    requested = False
    def request_once(payload, body):
        nonlocal requested
        if payload["stage"] == "final_review" and not requested:
            original = next(row for row in payload["fixed_original_inventory"] if row["version"] == "head")
            body["source_requests"] = [{"evidence_id": original["evidence_id"], "reason": "Compare this page with frozen head."}]
            requested = True
        return body
    def count(request):
        payload = json.loads(request.user_content)
        return 200 + 100 * len(payload["observations"]) if payload["stage"] == "final_review" else 100
    task = make_task()
    review = PartitionModelClient(transform=request_once, counter=count)
    review.profile = replace(review.profile, max_input_tokens=350)
    result = run_partition_review(task, "intent", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert len(result.partition["integration_pages"]) > 1
    assert result.partition["source_recheck_requests"][0]["admitted"]
    assert result.partition["integration_plan_id"]
    validate_partition_result(result, task)


def test_source_backed_exclusion_of_a_provisional_observation_is_not_missing_evidence():
    def correct(payload, body):
        if payload["stage"] == "final_review":
            key = next(iter(body["observation_dispositions"]))
            body["observation_dispositions"][key].update(disposition="excluded",
                reason="The fixed head code corrects this provisional interpretation; the original evidence remains available.")
        return body
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(transform=correct),
        PartitionModelClient(group_all=True), limits=limits())
    assert result.status == "completed", result.reason
    assert result.partition["excluded_observations"]
    assert not result.partition["final"]["missing_context"]
    assert result.partition["integration_pages"][0]["inputs"] == result.partition["review_observations"]
    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task)
    assert "제외·정정한 초기 관측" in markdown
    assert "불완전한 리뷰" not in markdown


def test_unabridged_diff_is_a_fixed_original_alongside_both_source_versions():
    from difflib import unified_diff
    from review_bot.application.partition_review import prepare_originals
    task = make_task()
    originals, meta = prepare_originals(task, "intent")
    before = next(f for f in task.base.files if f.path == "src/value.py")
    after = next(f for f in task.head.files if f.path == "src/value.py")
    expected = "".join(unified_diff(before.content.splitlines(keepends=True), after.content.splitlines(keepends=True),
        fromfile="a/src/value.py", tofile="b/src/value.py", n=3))
    assert next(o.text for o in originals if o.kind == "diff" and o.path == "src/value.py") == expected
    assert {o.version for o in originals if o.path == "src/value.py"} == {"base", "head", "diff"}
    assert next(row for row in meta["change_metrics"] if row["path"] == "src/value.py")["added_lines"] > 0


def test_a_later_page_can_resolve_an_earlier_gap_and_uncited_uncertainty_cannot_reopen_it():
    final_calls = 0
    def context_flow(payload, body):
        nonlocal final_calls
        if payload["stage"] == "review_partition":
            body["missing_context"] = ["Need the fixed PR description."]
        if payload["stage"] == "final_review":
            final_calls += 1
            purpose = next(c for c in payload["citation_catalog"] if not c["path"])
            for update in body["context_updates"].values():
                if final_calls == 2:
                    update.update(state="resolved", reason="The fixed PR description is present in this request.",
                        citations=[{"reference_id": purpose["ranges"][0]["reference_id"]}])
                else:
                    update.update(state="unresolved", reason="This page is uncertain.", citations=[])
        return body
    def count(request):
        payload = json.loads(request.user_content)
        return 200 + 100 * len(payload["observations"]) if payload["stage"] == "final_review" else 100
    review = PartitionModelClient(counter=count, transform=context_flow)
    review.profile = replace(review.profile, max_input_tokens=350)
    task = make_task()
    result = run_partition_review(task, "intent", review, PartitionModelClient(), limits=limits())
    assert result.status == "completed", result.reason
    assert final_calls > 2
    assert not result.partition["final"]["missing_context"]
    history = result.partition["final"]["context_update_history"]
    assert any(not event["applied"] for event in history)
    assert all(update["state"] == "resolved" for update in result.partition["final"]["context_updates"].values())
    validate_partition_result(result, task)
