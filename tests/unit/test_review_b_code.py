"""Real B v2 core, literal multi-file truth, and mocked external model only."""
from __future__ import annotations

import json
from dataclasses import asdict, replace
from difflib import unified_diff
from unittest.mock import Mock

import pytest

from review_bot.application.intent_context import build_intent_evidence, build_intent_task_input, evidence_digest
from review_bot.application.review_a import ModelProfile, TransientModelError
from review_bot.application.review_b import run_review_b_pipeline
from review_bot.application.review_b_code import (
    PROMPT_VERSION, SYSTEM_PROMPTS, code_first_prompt_hashes, run_code_first_review, validate_code_first_result,
)
from review_bot.domain.envelope_v2 import EnvelopeEvidence
from review_bot.domain.models_b_code import CodeFirstReviewLimits
from review_bot.domain.models_v1 import ModelResponse
from tests.fixtures.review_b_code import (
    BASE_SOURCE, CALLER, HEAD_SOURCE, CodeFirstModelClient, make_task, stage_response,
)


def stages(client):
    return [json.loads(request.user_content)["stage"] for request in client.requests]


def split_counter(request):
    payload = json.loads(request.user_content)
    return 2000 if payload["stage"] == "observe_group" and len(payload["paths"]) >= 3 else 100


def split_client(**kwargs):
    return CodeFirstModelClient(profile=ModelProfile("mock", "grouped", max_input_tokens=1500,
        max_output_tokens=512, is_mock=True), token_counter=split_counter, **kwargs)


@pytest.mark.parametrize("body", [None, "", "FALSE PURPOSE: only improve comments; no behavior changes."])
def test_code_defect_without_declared_goals_and_without_executing_target(body, monkeypatch):
    import subprocess
    import os
    forbidden = Mock(side_effect=AssertionError("target execution is forbidden"))
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    task, client = make_task(body=body), CodeFirstModelClient()
    result = run_code_first_review(task, client)
    assert (result.schema_version, result.status, result.reason, result.prompt_version) == (2, "completed", "validated", PROMPT_VERSION)
    assert stages(client) == ["observe_group", "infer_intent", "assess"]
    assert result.findings[0]["assessment"] == "code_defect"
    assert result.findings[0]["finding_id"] == "F1" and result.questions[0]["question_id"] == "Q1"
    assert result.reconstructed_intents[0]["intent_id"] == "I1"
    assert result.relations[0]["relation_id"] == "R1" and result.relations[0]["status"] == "conflict"
    assert result.scope["code_first"] and not result.scope["repository_wide"]
    assert result.scope["review_complete"] and result.scope["review_sufficient"] and not result.scope["no_findings"]
    assert result.scope["tests_executed_by_bot"] is False
    assert {cite["quote"] for cite in result.findings[0]["citations"]} == {
        HEAD_SOURCE.splitlines(keepends=True)[1], CALLER.splitlines(keepends=True)[3]}
    assert result.usage["calls"] == 3 and result.usage["input_tokens"] == 300 and result.usage["output_tokens"] == 150
    validate_code_first_result(result, task)
    json.dumps(asdict(result), allow_nan=False)
    forbidden.assert_not_called()


def test_pr_body_is_only_reference_text_at_assess_and_planner_gets_full_diff():
    sentinel = "UNTRUSTED PR PURPOSE UNIQUE SENTINEL"
    task, client, planner = make_task(body=sentinel), CodeFirstModelClient(), CodeFirstModelClient()
    result = run_code_first_review(task, client, planning_client=planner)
    assert result.status == "completed"
    assert stages(planner) == ["plan_code_review"]
    assert sentinel not in planner.requests[0].user_content
    assert all(sentinel not in request.user_content for request in client.requests[:-1])
    assert sentinel in client.requests[-1].user_content
    payload = json.loads(planner.requests[0].user_content)
    assert payload["base_sha"] == task.base.sha and payload["head_sha"] == task.head.sha
    assert payload["unified_diffs"][0]["diff"] == "".join(unified_diff(
        BASE_SOURCE.splitlines(keepends=True), HEAD_SOURCE.splitlines(keepends=True),
        fromfile="a/src/value.py", tofile="b/src/value.py", n=3))
    assert {row["path"] for row in payload["files"]} == set(task.reviewed_paths)
    assert payload["changed_files"] == [asdict(row) for row in task.snapshot.changed_files]
    assert result.plan["source"] == "planner" and result.usage["calls"] == 4
    observe = json.loads(client.requests[0].user_content)
    assert {item["kind"] for item in observe["evidence"]} == {"source", "test_source"}
    for version in (task.base, task.head):
        for file in version.files:
            assert any(item["content"] == file.content and item["locator"]["sha"] == version.sha
                and item["locator"]["path"] == file.path for item in observe["evidence"])


@pytest.mark.parametrize("planner_kind", ["absent", "unsupported", "byte_limit", "token_limit", "invalid_json", "invalid_plan", "not_sent"])
def test_full_fit_program_joint_fallback_is_explicit_and_keeps_planning_usage(planner_kind):
    class NotSent(RuntimeError):
        delivery = "not_sent"
    planner = CodeFirstModelClient()
    if planner_kind == "absent":
        planner = None
    elif planner_kind == "unsupported":
        planner = object()
    elif planner_kind == "byte_limit":
        planner.profile = replace(planner.profile, max_input_bytes=100)
    elif planner_kind == "token_limit":
        planner.profile = replace(planner.profile, max_input_tokens=100)
        planner.token_counter = lambda request: 101
    elif planner_kind == "invalid_json":
        planner.transform = lambda payload, body: ModelResponse("not JSON", {"input_tokens": 70, "output_tokens": 20})
    elif planner_kind == "invalid_plan":
        planner.transform = lambda payload, body: {**body, "groups": []}
    elif planner_kind == "not_sent":
        planner.transform = lambda payload, body: NotSent("do not disclose")
    result = run_code_first_review(make_task(), CodeFirstModelClient(), planning_client=planner)
    assert result.status == "completed" and result.plan["source"] == "program_joint"
    if planner_kind != "absent":
        assert result.plan["status"] == "fallback" and result.plan["reason"].startswith("planner_")
    assert result.usage["usage_complete"]
    if planner_kind in ("invalid_json", "invalid_plan", "not_sent"):
        assert result.usage["calls"] == 4
        assert next(row for row in result.stages if row.stage == "plan_code_review").status == "failed"
    if planner_kind == "invalid_json":
        assert result.usage["known_input_tokens"] == 370 and result.usage["known_output_tokens"] == 170


def test_small_planner_is_not_a_blocking_requirement_when_full_review_fits():
    main, planner = CodeFirstModelClient(), CodeFirstModelClient(profile=ModelProfile(
        "mock", "small-planner", max_input_tokens=64, max_output_tokens=32, is_mock=True))
    result = run_code_first_review(make_task(), main, planning_client=planner)
    assert result.status == "completed" and len(main.requests) == 3
    assert planner.requests == [] and result.plan["reason"] == "planner_input_limit_exceeded"


@pytest.mark.parametrize("failure", [TimeoutError("private"), ModelResponse("{}", None)])
def test_unknown_planner_delivery_or_usage_never_falls_back(failure):
    main = CodeFirstModelClient()
    planner = CodeFirstModelClient(transform=lambda payload, body: failure)
    result = run_code_first_review(make_task(), main, planning_client=planner)
    assert result.status != "completed" and main.requests == []
    assert result.usage["input_tokens"] is None and not result.usage["usage_complete"]
    assert result.usage["calls"] == 1
    assert "private" not in repr(result)


def test_split_whole_files_coverage_and_direct_both_endpoint_recheck():
    task, main, planner = make_task(padding=10), split_client(), CodeFirstModelClient(split=True)
    result = run_code_first_review(task, main, planning_client=planner)
    assert result.status == "completed"
    assert stages(planner) == ["plan_code_review"]
    assert stages(main) == ["observe_group", "observe_group", "recheck_relations", "infer_intent", "assess"]
    observed = [json.loads(row.user_content) for row in main.requests[:2]]
    assert sorted(path for group in observed for path in group["paths"]) == list(task.reviewed_paths)
    for code in (task.base, task.head):
        for file in code.files:
            entries = [item for group in observed for item in group["evidence"]
                if item["locator"]["sha"] == code.sha and item["locator"]["path"] == file.path]
            assert len(entries) == 1 and entries[0]["content"] == file.content
    recheck = json.loads(main.requests[2].user_content)
    assert recheck["paths"] == ["src/caller.py", "src/value.py"]
    assert {item["locator"]["path"] for item in recheck["evidence"]} == set(recheck["paths"])
    assert len(recheck["evidence"]) == 4
    assert result.relations[0]["reviewed_together"] and result.relations[0]["status"] == "conflict"
    assert result.usage["calls"] == 6
    validate_code_first_result(result, task)


def test_observer_discovered_cross_group_relation_uses_b_recheck_not_second_planner_call():
    def without_relations(payload, body):
        body["relations"] = []
        return body
    main, planner = split_client(), CodeFirstModelClient(split=True, transform=without_relations)
    result = run_code_first_review(make_task(), main, planning_client=planner)
    assert result.status == "completed" and stages(planner) == ["plan_code_review"]
    assert stages(main).count("recheck_relations") == 1
    assert result.relations[0]["relation_id"] == "R1"


def test_rename_paths_are_one_unit_and_both_original_versions_are_preserved():
    task, main, planner = make_task(rename=True), split_client(), CodeFirstModelClient(split=True)
    result = run_code_first_review(task, main, planning_client=planner)
    assert result.status == "completed"
    group = result.plan["groups"][0]
    assert group["paths"] == ["src/value.py", "src/value_v2.py"]
    originals = json.loads(main.requests[0].user_content)["evidence"]
    assert {(row["locator"]["version"], row["locator"]["path"]) for row in originals} == {
        ("base", "src/value.py"), ("head", "src/value_v2.py")}
    planned = json.loads(planner.requests[0].user_content)
    assert {row["path"] for row in planned["changed_files"]} == {"src/value_v2.py", "src/caller.py"}
    assert any("+from .value_v2 import render" in row["diff"] for row in planned["unified_diffs"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown_path", "too_many", "split_rename", "unknown_relation", "model_id", "bool"])
def test_required_plan_validation_has_no_file_by_file_fallback(mutation):
    task = make_task(rename=mutation == "split_rename")
    def corrupt(payload, body):
        if mutation == "missing":
            body["groups"].pop()
        elif mutation == "duplicate":
            body["groups"][1]["paths"].append(body["groups"][0]["paths"][0])
        elif mutation == "unknown_path":
            body["groups"][0]["paths"] = ["outside.py"]
        elif mutation == "too_many":
            body["groups"] *= 5
        elif mutation == "split_rename":
            body["groups"][1]["paths"].append(body["groups"][0]["paths"].pop())
        elif mutation == "unknown_relation":
            body["relations"][0]["to_path"] = "outside.py"
        elif mutation == "model_id":
            body["groups"][0]["group_id"] = "made-up"
        else:
            body["relations"][0]["important"] = 1
        return body
    main, planner = split_client(), CodeFirstModelClient(split=True, transform=corrupt)
    result = run_code_first_review(task, main, planning_client=planner)
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert main.requests == [] and len(planner.requests) == 1
    assert result.usage["calls"] == 1 and result.usage["input_tokens"] == 100


def test_full_oversize_requires_plan_and_full_diff_oversize_is_explicit():
    main = split_client()
    assert run_code_first_review(make_task(), main).reason == "planning_required"
    assert main.requests == []
    planner = CodeFirstModelClient(profile=ModelProfile("mock", "tiny", max_input_bytes=100, is_mock=True))
    result = run_code_first_review(make_task(), split_client(), planning_client=planner)
    assert (result.status, result.reason) == ("not_run", "plan_input_limit_exceeded")
    assert planner.requests == [] and planner.count_requests == []


@pytest.mark.parametrize("mutation", ["line", "bool_line", "unknown_evidence", "model_id", "extra_quote", "array_limit", "enum"])
def test_observation_validation_stops_all_later_calls_but_keeps_usage(mutation):
    def corrupt(payload, body):
        if payload["stage"] == "observe_group":
            row = body["observations"][0]
            if mutation == "line":
                row["citations"][0]["end_line"] = 9000
            elif mutation == "bool_line":
                row["citations"][0]["start_line"] = True
            elif mutation == "unknown_evidence":
                row["citations"][0]["evidence_id"] = "E-INVENTED"
            elif mutation == "model_id":
                row["observation_id"] = "O1"
            elif mutation == "extra_quote":
                row["citations"][0]["quote"] = "invented source"
            elif mutation == "array_limit":
                body["observations"] = [{**row, "statement": str(index)} for index in range(21)]
            else:
                body["new_relations"][0]["status"] = "approved"
        return body
    client = CodeFirstModelClient(transform=corrupt)
    result = run_code_first_review(make_task(), client)
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert stages(client) == ["observe_group"] and not result.group_observations
    assert result.usage["calls"] == 1 and result.usage["known_input_tokens"] == 100


def test_evidence_from_another_group_is_rejected_even_if_its_id_is_known():
    task = make_task()
    caller = next(item for item in build_intent_evidence(task) if item.locator.get("path") == "src/caller.py")
    def corrupt(payload, body):
        if payload["stage"] == "observe_group":
            body["observations"][0]["citations"][0] = {"evidence_id": caller.id, "start_line": 4, "end_line": 4}
        return body
    main = split_client(transform=corrupt)
    result = run_code_first_review(task, main, planning_client=CodeFirstModelClient(split=True))
    assert result.reason == "invalid_model_response" and stages(main) == ["observe_group"]
    assert result.usage["calls"] == 2


@pytest.mark.parametrize("text", ['{"observations":[],"observations":[]}', '{"x":NaN}', 'null', '[1]', '"bad"'])
def test_strict_json_root_duplicate_keys_and_nonfinite_numbers(text):
    main = CodeFirstModelClient(transform=lambda payload, body: ModelResponse(text, {"input_tokens": 100, "output_tokens": 50}))
    result = run_code_first_review(make_task(), main)
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert len(main.requests) == 1


@pytest.mark.parametrize("stage", ["infer_intent", "assess"])
def test_later_failure_preserves_validated_groups_and_prior_usage(stage):
    def corrupt(payload, body):
        if payload["stage"] == stage:
            if stage == "infer_intent":
                body["intents"][0]["observation_ids"] = ["O999"]
            else:
                body["findings"][0]["intent_ids"] = ["I999"]
        return body
    main = CodeFirstModelClient(transform=corrupt)
    result = run_code_first_review(make_task(), main)
    assert result.reason == "invalid_model_response" and result.group_observations
    assert bool(result.reconstructed_intents) == (stage == "assess")
    assert result.usage["calls"] == (2 if stage == "infer_intent" else 3)
    assert not result.findings and not result.questions


@pytest.mark.parametrize("mutation", ["failure", "one_side", "contradict_relation", "model_id", "no_findings", "bool_sufficient", "explanation_only"])
def test_final_verdict_requires_both_evidence_and_failure_condition(mutation):
    def corrupt(payload, body):
        if payload["stage"] == "assess":
            row = body["findings"][0]
            if mutation == "failure":
                row["failure_condition"] = ""
            elif mutation == "one_side":
                row["citations"] = row["citations"][:1]
            elif mutation == "contradict_relation":
                relation = payload["relations"][0]
                body["relation_updates"] = [{key: value for key, value in relation.items()
                    if key in ("relation_id", "reasoning", "failure_condition", "missing_context")} | {
                    "status": "consistent", "citations": row["citations"]}]
            elif mutation == "model_id":
                row["finding_id"] = "F1"
            elif mutation == "no_findings":
                body["no_findings"] = True
            elif mutation == "bool_sufficient":
                body["review_sufficient"] = 1
            else:
                row["assessment"] = "explanation_mismatch"
        return body
    result = run_code_first_review(make_task(), CodeFirstModelClient(transform=corrupt))
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert result.usage["calls"] == 3 and result.reconstructed_intents and not result.findings


def test_model_cannot_cite_uncarried_original_lines_in_inference():
    def corrupt(payload, body):
        if payload["stage"] == "infer_intent":
            body["intents"][0]["citations"][0]["start_line"] = 1
            body["intents"][0]["citations"][0]["end_line"] = 1
        return body
    result = run_code_first_review(make_task(), CodeFirstModelClient(transform=corrupt))
    assert result.reason == "invalid_model_response" and result.usage["calls"] == 2


def test_relation_code_delivery_does_not_imply_consistency_and_partial_is_separate():
    main = split_client()
    result = run_code_first_review(make_task(), main, planning_client=CodeFirstModelClient(split=True),
        limits=CodeFirstReviewLimits(max_rechecks=0))
    assert result.status == "partial_review" and result.reason == "unresolved_context"
    assert result.relations[0]["status"] == "unreviewed" and not result.relations[0]["reviewed_together"]
    assert result.scope["review_complete"] and not result.scope["review_sufficient"]
    assert result.findings[0]["assessment"] == "insufficient_context"
    validate_code_first_result(result, make_task())


def test_recheck_original_limit_keeps_large_files_and_defers_instead_of_dropping_them():
    def counter(request):
        return 2000 if json.loads(request.user_content)["stage"] == "recheck_relations" else split_counter(request)
    main = split_client()
    main.token_counter = counter
    result = run_code_first_review(make_task(padding=20), main, planning_client=CodeFirstModelClient(split=True))
    assert result.status == "partial_review"
    assert "recheck_relations" not in stages(main)
    request = next(row for row in main.count_requests if json.loads(row.user_content)["stage"] == "recheck_relations")
    assert all("# fixed padding\n" in row["content"] for row in json.loads(request.user_content)["evidence"])
    assert result.relations[0]["status"] == "insufficient_context"
    assert result.scope["missing_context_requests"]


def test_outside_scope_original_request_is_recorded_without_access():
    def request_outside(payload, body):
        if payload["stage"] == "observe_group":
            body["requested_paths"] = ["outside/missing.py"]
        return body
    main = CodeFirstModelClient(transform=request_outside)
    result = run_code_first_review(make_task(), main)
    assert result.status == "partial_review"
    assert result.scope["missing_context_requests"] == ["outside/missing.py"]
    assert "outside_reviewed_paths:outside/missing.py" in result.group_observations[0]["missing_context"]
    assert stages(main) == ["observe_group", "infer_intent", "assess"]


def test_zero_profile_and_invalid_input_make_zero_generation_or_measurement_calls():
    main = CodeFirstModelClient()
    main.profile = None
    result = run_code_first_review(make_task(), main, planning_client=CodeFirstModelClient())
    assert result.status == "not_run" and main.requests == [] and main.count_requests == []
    assert result.usage["calls"] == 0 and result.usage["token_measurements"] == 0
    main = CodeFirstModelClient()
    result = run_code_first_review(replace(make_task(), input_hash="e" * 64), main)
    assert result.reason == "invalid_task_input" and main.requests == [] and main.count_requests == []


@pytest.mark.parametrize("limits", [CodeFirstReviewLimits(max_calls=True), CodeFirstReviewLimits(max_groups=9),
    CodeFirstReviewLimits(max_rechecks=2), CodeFirstReviewLimits(max_total_tokens=0)])
def test_invalid_limits_fail_closed_without_calls(limits):
    main = CodeFirstModelClient()
    result = run_code_first_review(make_task(), main, limits=limits)
    assert result.reason == "invalid_review_limits" and main.count_requests == [] and main.requests == []


def test_call_reservation_leaves_infer_and_assess_even_when_optional_recheck_cannot_fit():
    main = split_client()
    result = run_code_first_review(make_task(), main, planning_client=CodeFirstModelClient(split=True),
        limits=CodeFirstReviewLimits(max_calls=5))
    assert result.status == "partial_review"
    assert stages(main) == ["observe_group", "observe_group", "infer_intent", "assess"]
    assert result.usage["calls"] == 5
    assert next(row for row in result.stages if row.stage == "recheck_relations").reason == "call_limit_exceeded"
    main = CodeFirstModelClient()
    assert run_code_first_review(make_task(), main, limits=CodeFirstReviewLimits(max_calls=2)).reason == "call_limit_exceeded"
    assert not main.requests and not main.count_requests


def test_total_task_tokens_are_cumulative_not_a_per_request_claim():
    profile = ModelProfile("mock", "cumulative", max_output_tokens=100, is_mock=True)
    main = CodeFirstModelClient(profile=profile)
    result = run_code_first_review(make_task(), main, limits=CodeFirstReviewLimits(max_total_tokens=450))
    assert result.reason == "task_token_limit_exceeded"
    assert stages(main) == ["observe_group", "infer_intent"]
    assert result.usage["known_input_tokens"] + result.usage["known_output_tokens"] == 300
    assert result.group_observations and result.reconstructed_intents


def test_optional_recheck_cannot_consume_reserved_final_output_budget():
    def low_usage(payload, body):
        return ModelResponse(json.dumps(body), {"input_tokens": 20, "output_tokens": 20})
    def counter(request):
        return 300 if json.loads(request.user_content)["stage"] == "recheck_relations" else split_counter(request)
    profile = ModelProfile("mock", "reserve", max_input_tokens=1500, max_output_tokens=100, is_mock=True)
    main = CodeFirstModelClient(profile=profile, token_counter=counter, transform=low_usage)
    planner = CodeFirstModelClient(split=True, profile=profile, transform=low_usage)
    result = run_code_first_review(make_task(), main, planning_client=planner, limits=CodeFirstReviewLimits(max_total_tokens=550))
    assert result.status == "partial_review" and stages(main)[-2:] == ["infer_intent", "assess"]
    assert "recheck_relations" not in stages(main)
    assert next(row for row in result.stages if row.stage == "recheck_relations").reason == "task_token_limit_exceeded"
    assert result.usage["known_input_tokens"] + result.usage["known_output_tokens"] == 200


def test_paid_metering_calls_and_tokens_are_in_the_same_task_budget():
    class Metered(CodeFirstModelClient):
        token_count_receipt = {"delivery": "sent", "model_calls": 1, "usage": {"input_tokens": 5, "output_tokens": 1}}
    main = Metered(profile=ModelProfile("fixture-http", "metered", max_output_tokens=512))
    result = run_code_first_review(make_task(), main)
    assert result.status == "completed"
    assert result.usage["calls"] == 3 and result.usage["model_calls"] == 7
    assert result.usage["metering_input_tokens"] == 20 and result.usage["metering_output_tokens"] == 4
    assert result.usage["input_tokens"] == 320 and result.usage["output_tokens"] == 154


def test_unknown_metering_usage_is_not_zero_and_stops_before_generation():
    main = CodeFirstModelClient(profile=ModelProfile("fixture-http", "unknown", max_output_tokens=512))
    result = run_code_first_review(make_task(), main)
    assert result.reason == "usage_unavailable" and main.requests == []
    assert result.usage["input_tokens"] is None and result.usage["model_calls"] is None
    assert result.usage["unknown_usage_attempts"] == 1


def test_known_transient_retry_is_bounded_and_unknown_retry_never_continues():
    class NotSent(TransientModelError):
        delivery = "not_sent"
    def once(payload, body):
        return NotSent("private") if len(main.requests) == 1 else body
    profile = ModelProfile("mock", "retry", max_attempts=2, max_output_tokens=512, is_mock=True)
    main = CodeFirstModelClient(profile=profile, transform=once)
    result = run_code_first_review(make_task(), main)
    assert result.status == "completed" and result.usage["calls"] == 4
    observe = next(row for row in result.stages if row.stage.startswith("observe_group:"))
    assert len(observe.attempts) == 2 and observe.attempts[0].delivery == "not_sent"
    main = CodeFirstModelClient(profile=profile, transform=lambda payload, body: TransientModelError("private"))
    result = run_code_first_review(make_task(), main)
    assert result.reason == "usage_unavailable" and len(main.requests) == 1
    assert result.usage["input_tokens"] is None


@pytest.mark.parametrize("when", ["measure", "complete"])
def test_deadline_after_each_external_call_keeps_usage_and_passes_remaining_time(when, monkeypatch):
    from review_bot.application import review_b_code
    clock = [0.0]
    monkeypatch.setattr(review_b_code.time, "monotonic", lambda: clock[0])
    class Slow(CodeFirstModelClient):
        def count_input_tokens(self, request):
            result = super().count_input_tokens(request)
            if when == "measure":
                clock[0] = 100.0
            return result
        def complete(self, request):
            result = super().complete(request)
            clock[0] = 100.0
            return result
    main = Slow()
    result = run_code_first_review(make_task(), main, limits=CodeFirstReviewLimits(task_timeout_seconds=10))
    assert result.reason == "task_deadline_exceeded"
    assert len(main.requests) == (0 if when == "measure" else 1)
    assert all(row.timeout_seconds <= 10 for row in [*main.count_requests, *main.requests])
    assert result.usage["known_input_tokens"] == (0 if when == "measure" else 100)


@pytest.mark.parametrize("when", ["before_measure", "after_measure", "after_complete"])
def test_fence_checks_pre_and_post_each_external_call(when):
    allowed = [when != "before_measure"]
    class Fenced(CodeFirstModelClient):
        def count_input_tokens(self, request):
            result = super().count_input_tokens(request)
            if when == "after_measure":
                allowed[0] = False
            return result
        def complete(self, request):
            result = super().complete(request)
            allowed[0] = False
            return result
    main = Fenced()
    result = run_code_first_review(make_task(), main, before_stage=lambda: allowed[0])
    assert result.reason == "stage_gate_rejected"
    assert len(main.requests) == (1 if when == "after_complete" else 0)
    assert result.usage["calls"] == len(main.requests)


def test_output_byte_cap_is_not_a_token_estimate():
    profile = ModelProfile("mock", "bytes", max_output_bytes=100, max_output_tokens=512, is_mock=True)
    main = CodeFirstModelClient(profile=profile)
    result = run_code_first_review(make_task(), main)
    assert result.reason == "invalid_model_response" and result.usage["calls"] == 1


def test_public_prompt_hashes_and_run_keys_bind_v2_limits_plan_clients_and_originals():
    task = make_task()
    first = run_code_first_review(task, CodeFirstModelClient())
    second = run_code_first_review(task, CodeFirstModelClient())
    assert first.run_key == second.run_key and first.prompt_hashes == code_first_prompt_hashes()
    assert set(first.prompt_hashes) == set(SYSTEM_PROMPTS)
    changed_limits = run_code_first_review(task, CodeFirstModelClient(), limits=CodeFirstReviewLimits(max_calls=11))
    changed_plan = run_code_first_review(task, CodeFirstModelClient(), planning_client=CodeFirstModelClient())
    changed_text = run_code_first_review(make_task(body="another purpose"), CodeFirstModelClient())
    changed_model = run_code_first_review(task, CodeFirstModelClient(profile=ModelProfile(
        "mock", "different", max_output_tokens=512, is_mock=True)))
    v1 = run_review_b_pipeline(task)
    assert len({first.run_key, changed_limits.run_key, changed_plan.run_key, changed_text.run_key, changed_model.run_key, v1.run_key}) == 6
    assert v1.schema_version == 1


def test_request_stage_response_factory_is_reusable_by_parent_integration():
    class Factory(CodeFirstModelClient):
        def complete(self, request):
            self.requests.append(request)
            return stage_response(request)
    assert run_code_first_review(make_task(), Factory()).status == "completed"


def test_shared_model_request_protocol_stays_v1_while_result_is_v2():
    class V1Transport(CodeFirstModelClient):
        def count_input_tokens(self, request):
            assert request.schema_version == 1
            return super().count_input_tokens(request)
        def complete(self, request):
            assert request.schema_version == 1
            return super().complete(request)
    main = V1Transport()
    result = run_code_first_review(make_task(), main)
    assert result.status == "completed" and result.schema_version == 2
    assert len(main.requests) == 3


def test_profile_none_has_the_existing_unset_reason_and_no_external_calls():
    main = CodeFirstModelClient()
    main.profile = None
    result = run_code_first_review(make_task(), main)
    assert (result.status, result.reason) == ("not_run", "model_profile_unset")
    assert main.count_requests == [] and main.requests == []


@pytest.mark.parametrize("unknown_usage", [False, True])
def test_second_group_failure_preserves_first_group_and_never_calls_infer(unknown_usage):
    def corrupt(payload, body):
        if payload["stage"] == "observe_group" and payload["group_id"] == "group-2":
            return ModelResponse("invalid", None if unknown_usage else {"input_tokens": 100, "output_tokens": 50})
        return body
    main = split_client(transform=corrupt)
    result = run_code_first_review(make_task(), main, planning_client=CodeFirstModelClient(split=True))
    assert result.status == "failed" and len(result.group_observations) == 1
    assert stages(main) == ["observe_group", "observe_group"]
    assert result.usage["calls"] == 3 and result.usage["known_input_tokens"] == (200 if unknown_usage else 300)
    assert result.usage["usage_complete"] is (not unknown_usage)
    validate_code_first_result(result, make_task())


def test_paid_metering_reservation_protects_both_final_stages_from_optional_recheck():
    class Metered(CodeFirstModelClient):
        token_count_receipt = {"delivery": "sent", "model_calls": 1, "usage": {"input_tokens": 5, "output_tokens": 1}}
    main = Metered(profile=ModelProfile("fixture-http", "metered", max_input_tokens=1500, max_output_tokens=512),
        token_counter=split_counter)
    result = run_code_first_review(make_task(), main, planning_client=CodeFirstModelClient(split=True),
        limits=CodeFirstReviewLimits(max_calls=11))
    assert result.status == "partial_review"
    assert stages(main) == ["observe_group", "observe_group", "infer_intent", "assess"]
    assert result.usage["model_calls"] == 10
    assert next(row for row in result.stages if row.stage == "recheck_relations").reason == "call_limit_exceeded"


def test_recheck_limit_counts_attempts_and_does_not_retry_a_relation_stage():
    class NotSent(TransientModelError):
        delivery = "not_sent"
    def fail_recheck(payload, body):
        return NotSent("private") if payload["stage"] == "recheck_relations" else body
    main = split_client(transform=fail_recheck)
    main.profile = replace(main.profile, max_attempts=2)
    result = run_code_first_review(make_task(), main, planning_client=CodeFirstModelClient(split=True))
    assert result.status == "failed"
    assert stages(main) == ["observe_group", "observe_group", "recheck_relations"]
    assert len(next(row for row in result.stages if row.stage == "recheck_relations").attempts) == 1


def test_recorded_run_spanning_groups_is_supplied_separately_to_each_relevant_group():
    task = make_task()
    run = {"run_id": "RUN1", "sha": task.head.sha, "paths": list(task.reviewed_paths),
        "command": ["pytest", "tests/test_label.py"], "exit_code": 0, "status": "completed",
        "started_at": task.snapshot.captured_at, "completed_at": task.snapshot.captured_at,
        "output": "fixed supplied record; not executed by this test\n"}
    run["output_hash"] = evidence_digest(run["output"], "source")
    evidence = EnvelopeEvidence("E-RUN-01", "test_run", task.head.sha,
        {"repository": task.snapshot.repository, "sha": task.head.sha, "run_id": "RUN1"}, run,
        evidence_digest(run, "test_run"))
    task = build_intent_task_input(replace(task.snapshot, evidence=(*task.snapshot.evidence, evidence)),
        task.base, task.head, related_paths=task.reviewed_paths)
    main = split_client()
    result = run_code_first_review(task, main, planning_client=CodeFirstModelClient(split=True))
    assert result.status == "completed" and not result.scope["tests_executed_by_bot"]
    for request in main.requests[:2]:
        payload = json.loads(request.user_content)
        assert [row["id"] for row in payload["test_runs"]] == ["E-RUN-01"]
        assert all(row["kind"] != "test_run" for row in payload["evidence"])


def test_unterminated_original_diff_lines_are_not_stitched_or_dropped():
    from tests.fixtures.review_b import digest
    task = make_task()
    def no_final_lf(code):
        return replace(code, files=tuple(replace(file, content=file.content.rstrip("\n"),
            content_hash=digest(file.content.rstrip("\n"))) for file in code.files), inventory_hash="")
    task = build_intent_task_input(task.snapshot, no_final_lf(task.base), no_final_lf(task.head),
        related_paths=task.reviewed_paths)
    planner = CodeFirstModelClient()
    result = run_code_first_review(task, CodeFirstModelClient(), planning_client=planner)
    assert result.status == "completed"
    diff = json.loads(planner.requests[0].user_content)["unified_diffs"][0]["diff"]
    assert '-    return str(value)\n\\ No newline at end of file\n' in diff
    assert '+    return {"value": value}\n\\ No newline at end of file\n' in diff


def test_more_than_128_prepared_paths_do_not_become_a_silent_group_path_limit():
    from review_bot.domain.models_v1 import CodeFile
    from tests.fixtures.review_b import digest
    task = make_task()
    extra = tuple(CodeFile(f"context/{index:03}.py", "# fixed context\n", digest("# fixed context\n"), "source") for index in range(130))
    task = build_intent_task_input(task.snapshot, replace(task.base, files=(*task.base.files, *extra), inventory_hash=""),
        replace(task.head, files=(*task.head.files, *extra), inventory_hash=""),
        related_paths=(*task.reviewed_paths, *(file.path for file in extra)))
    class BoundedObservations(CodeFirstModelClient):
        def complete(self, request):
            payload = json.loads(request.user_content)
            if payload["stage"] == "observe_group":
                selected = [entry for entry in payload["evidence"] if not entry["locator"]["path"].startswith("context/")]
                changed = replace(request, user_content=json.dumps({**payload, "evidence": selected}))
                response = stage_response(changed)
                self.requests.append(request)
                return response
            return super().complete(request)
    main, planner = BoundedObservations(), CodeFirstModelClient()
    result = run_code_first_review(task, main, planning_client=planner)
    assert result.status == "completed" and len(result.plan["groups"][0]["paths"]) == 133
    assert planner.requests[0].response_schema["properties"]["groups"]["items"]["properties"]["paths"]["maxItems"] == 133
    validate_code_first_result(result, task)


@pytest.mark.parametrize("when", ["count", "complete"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_cancellation_never_becomes_a_safe_planner_fallback_even_if_not_sent(when, asynchronous):
    from asyncio import CancelledError as AsyncCancelledError
    from concurrent.futures import CancelledError
    error_type = AsyncCancelledError if asynchronous else CancelledError
    class CancelledBeforeSend(error_type):
        delivery = "not_sent"
        usage = {"input_tokens": 0, "output_tokens": 0}
    class CancelledPlanner(CodeFirstModelClient):
        def count_input_tokens(self, request):
            result = super().count_input_tokens(request)
            if when == "count":
                raise CancelledBeforeSend("private cancellation")
            return result
        def complete(self, request):
            self.requests.append(request)
            raise CancelledBeforeSend("private cancellation")
    main, planner = CodeFirstModelClient(), CancelledPlanner()
    result = run_code_first_review(make_task(), main, planning_client=planner)
    assert result.status in ("not_run", "failed") and result.reason == "model_cancelled"
    assert main.requests == [] and result.plan is None
    assert len(planner.requests) == (1 if when == "complete" else 0)
    assert "private cancellation" not in repr(result)


def test_received_actual_input_plus_output_cannot_exceed_model_context_limit():
    def changed_usage(payload, body):
        return ModelResponse(json.dumps(body), {"input_tokens": 900, "output_tokens": 200})
    main = CodeFirstModelClient(profile=ModelProfile("mock", "context", max_input_tokens=500,
        max_output_tokens=512, is_mock=True), transform=changed_usage)
    assert 900 > main.profile.max_input_tokens
    assert 900 + 200 > main.profile.max_input_tokens + main.profile.max_output_tokens
    result = run_code_first_review(make_task(), main)
    assert result.status == "failed" and result.reason == "usage_limit_exceeded"
    assert stages(main) == ["observe_group"] and not result.group_observations
    assert result.usage["input_tokens"] == 900 and result.usage["output_tokens"] == 200


def test_definite_nontransmission_is_not_relabelled_as_sent_in_delivery_records():
    class NotSent(RuntimeError):
        delivery = "not_sent"
    planner = CodeFirstModelClient(transform=lambda payload, body: NotSent("private"))
    result = run_code_first_review(make_task(), CodeFirstModelClient(), planning_client=planner)
    assert result.status == "completed" and result.usage["calls"] == 4
    assert result.usage["model_calls"] == 3
    assert result.model_delivery["plan_code_review"]["status"] == "not_sent"