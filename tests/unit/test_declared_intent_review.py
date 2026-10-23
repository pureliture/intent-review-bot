"""Strict B stage boundaries, grounding, and non-fabricated delivery records."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import replace

import pytest

from review_bot.application.architecture_review import MockModelClient, ModelProfile, TransientModelError
from review_bot.application.declared_intent_review import MIN_QUOTE_NONSPACE_CHARS, response_schema, run_review_b_pipeline
from review_bot.domain.intent_models import IntentReviewLimits
from review_bot.domain.models_v1 import ModelResponse
from tests.fixtures.intent_review import GOAL, make_task, responses, stage_bodies


def test_generation_schema_separates_new_result_ids_from_other_entity_types() -> None:
    namespaces = {
        "extract_goals": {"declared_goals": ("goal_id", "G"), "declared_constraints": ("constraint_id", "C")},
        "observe_changes": {"implementation_observations": ("observation_id", "O"),
            "intent_hypotheses": ("hypothesis_id", "H"), "assumptions": ("assumption_id", "S")},
        "assess": {"findings": ("finding_id", "F"), "questions": ("question_id", "Q")},
    }
    for stage, entities in namespaces.items():
        properties = response_schema(stage)["properties"]
        for collection, (key, prefix) in entities.items():
            pattern = properties[collection]["items"]["properties"][key]["pattern"]
            assert re.fullmatch(pattern, f"{prefix}1")
            assert all(not re.fullmatch(pattern, f"{other}1") for other in "GCOHSFQ" if other != prefix)


def test_generation_references_are_bound_to_the_current_stage_originals_and_validated_results() -> None:
    client = MockModelClient(responses(), input_tokens=100)
    assert run_review_b_pipeline(make_task(), client).status == "completed"
    expected_evidence = [["E-PR-01"], ["E-CODE-base-0001", "E-CODE-head-0001", "E-CODE-head-0002"],
        ["E-CODE-base-0001", "E-CODE-head-0001", "E-CODE-head-0002", "E-PR-01"]]
    for request, collection, allowed in zip(client.requests, ["declared_goals", "intent_hypotheses", "findings"], expected_evidence):
        fields = request.response_schema["properties"][collection]["items"]["properties"]
        assert fields["evidence_ids"]["items"]["enum"] == allowed
        assert fields["evidence_ids"]["maxItems"] == len(allowed)
    finding_fields = client.requests[2].response_schema["properties"]["findings"]["items"]["properties"]
    assert finding_fields["goal_ids"]["items"]["enum"] == ["G1"]
    assert finding_fields["constraint_ids"]["maxItems"] == 0
    assert finding_fields["observation_ids"]["items"]["enum"] == ["O1", "O2"]
    question_fields = client.requests[2].response_schema["properties"]["questions"]["items"]["properties"]
    related_pattern = question_fields["related_finding_ids"]["items"]["pattern"]
    assert question_fields["related_finding_ids"]["minItems"] == 1
    assert re.fullmatch(related_pattern, "F1")
    assert not re.fullmatch(related_pattern, "H1") and not re.fullmatch(related_pattern, "S1")


def test_no_declared_goal_generation_cannot_claim_a_conflict_or_consistency() -> None:
    client = MockModelClient(responses(stage_bodies(no_goals=True)), input_tokens=100)
    assert run_review_b_pipeline(make_task(body=""), client).status == "completed"
    fields = client.requests[2].response_schema["properties"]["findings"]["items"]["properties"]
    assert fields["assessment"]["enum"] == ["insufficient_context"]
    assert fields["missing_evidence"]["minItems"] == 1


@pytest.mark.parametrize("collision", ["prior_observation", "finding_question", "repeated_finding"])
def test_final_duplicate_ids_are_rejected_without_rewriting_or_discarding_usage(collision: str) -> None:
    bodies = deepcopy(stage_bodies())
    if collision == "prior_observation":
        bodies[2]["findings"][0]["finding_id"] = "O1"
    elif collision == "finding_question":
        bodies[2]["questions"][0]["question_id"] = "B1"
    else:
        bodies[2]["findings"].append(deepcopy(bodies[2]["findings"][0]))
    result = run_review_b_pipeline(make_task(), MockModelClient(responses(bodies), input_tokens=100))
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert not result.findings and not result.questions
    assert result.goals and result.observations
    assert result.usage["calls"] == 3 and result.usage["output_tokens"] == 150


@pytest.mark.parametrize("language", ["python", "javascript", "java"])
def test_three_validated_stages_are_language_independent_and_separated(language: str) -> None:
    task = make_task(language)
    client = MockModelClient(responses(), input_tokens=100)
    result = run_review_b_pipeline(task, client)
    assert (result.schema_version, result.review_type, result.status, result.reason) == (1, "B", "completed", "validated")
    assert len(client.requests) == 3
    goals, observations, assessment = [json.loads(request.user_content) for request in client.requests]
    assert goals["stage"] == "extract_goals"
    assert {entry["kind"] for entry in goals["evidence"]} == {"pr_text"}
    assert set(goals) == {"stage", "evidence", "unavailable_context"}
    assert observations["stage"] == "observe_changes"
    assert set(observations) == {"stage", "changed_files", "evidence", "unavailable_context"}
    assert {entry["kind"] for entry in observations["evidence"]} == {"source", "test_source"}
    assert GOAL not in client.requests[1].user_content
    assert "declared_goals" not in observations
    assert set(assessment) == {"stage", "goals", "observations", "evidence", "unavailable_context"}
    assert assessment["goals"]["declared_goals"][0]["goal_id"] == "G1"
    assert all(stage.status == "completed" for stage in result.stages)
    assert result.usage["calls"] == 3
    assert result.usage["input_tokens"] == 300
    assert result.usage["output_tokens"] == 150
    assert result.scope["tests_executed_by_bot"] is False
    assert result.model["is_mock"] is True
    assert all(request.response_schema["additionalProperties"] is False for request in client.requests)


def test_missing_profile_makes_no_calls_and_records_all_stages() -> None:
    result = run_review_b_pipeline(make_task())
    assert (result.status, result.reason) == ("not_run", "model_profile_unset")
    assert result.usage["calls"] == 0
    assert all(stage.status == "not_attempted" for stage in result.stages)
    assert all(record["status"] == "not_attempted" for record in result.model_delivery.values())
    assert not result.findings and not result.questions


def test_no_declared_goals_remains_explicit_and_cannot_be_a_conflict() -> None:
    bodies = stage_bodies(no_goals=True)
    task = make_task(body="")
    result = run_review_b_pipeline(task, MockModelClient(responses(bodies), input_tokens=100))
    assert result.status == "completed"
    assert result.goals.declared_goals == ()
    assert "no_declared_goals" in result.limitations
    bodies[2]["findings"][0]["assessment"] = "evidence_conflict"
    client = MockModelClient(responses(bodies), input_tokens=100)
    result = run_review_b_pipeline(task, client)
    assert result.reason == "invalid_model_response"
    assert not result.findings


@pytest.mark.parametrize("stage,change", [
    (0, "forged_quote"), (0, "short_quote_e"), (0, "short_quote_the"),
    (0, "code_as_goal"), (0, "duplicate_id"), (0, "extra_field"),
    (1, "pr_as_observation"), (1, "dangling_id"), (1, "extra_location"),
    (1, "missing_assumption_evidence"), (2, "dangling_goal"), (2, "unknown_enum"),
    (2, "hypothesis_conflict"), (2, "missing_linked_evidence"), (2, "dangling_question"),
])
def test_invalid_stage_stops_without_later_calls(stage: int, change: str) -> None:
    bodies = deepcopy(stage_bodies())
    if change == "forged_quote":
        bodies[0]["declared_goals"][0]["quote"] = "An invented requirement"
    elif change.startswith("short_quote_"):
        bodies[0]["declared_goals"][0]["quote"] = change.removeprefix("short_quote_")
    elif change == "code_as_goal":
        bodies[0]["declared_goals"][0]["evidence_ids"] = ["E-CODE-head-0001"]
    elif change == "duplicate_id":
        bodies[0]["declared_goals"].append(deepcopy(bodies[0]["declared_goals"][0]))
    elif change == "extra_field":
        bodies[0]["status"] = "completed"
    elif change == "pr_as_observation":
        bodies[1]["implementation_observations"][0]["evidence_ids"] = ["E-PR-01"]
    elif change == "dangling_id":
        bodies[1]["implementation_observations"][0]["evidence_ids"] = ["not-an-evidence"]
    elif change == "extra_location":
        bodies[1]["implementation_observations"][0]["sha"] = "0" * 40
    elif change == "missing_assumption_evidence":
        bodies[1]["assumptions"][0]["missing_evidence"] = []
    elif change == "dangling_goal":
        bodies[2]["findings"][0]["goal_ids"] = ["G2"]
    elif change == "unknown_enum":
        bodies[2]["findings"][0]["assessment"] = "approved"
    elif change == "hypothesis_conflict":
        bodies[2]["findings"][0].update(goal_ids=[], constraint_ids=[], assessment="evidence_conflict", missing_evidence=[])
    elif change == "missing_linked_evidence":
        bodies[2]["findings"][0]["evidence_ids"] = ["E-CODE-head-0001"]
    elif change == "dangling_question":
        bodies[2]["questions"][0]["related_finding_ids"] = ["B2"]
    client = MockModelClient(responses(bodies), input_tokens=100)
    task = make_task(body="the request should be processed once." if change.startswith("short_quote_") else GOAL)
    result = run_review_b_pipeline(task, client)
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert len(client.requests) == stage + 1
    assert result.usage["calls"] == stage + 1
    assert result.usage["input_tokens"] == 100 * (stage + 1)
    assert all(record.status == "not_attempted" for record in result.stages[stage + 1:])
    assert not result.findings and not result.questions


@pytest.mark.parametrize("text", [
    '{"declared_goals":[],"declared_goals":[],"declared_constraints":[],"unavailable_context":[]}',
    '{"declared_goals":NaN,"declared_constraints":[],"unavailable_context":[]}',
    "```json\n{}\n```", "[]", '{"declared_goals":true}',
])
def test_strict_json_rejects_duplicates_nonfinite_and_nonobjects(text: str) -> None:
    client = MockModelClient([ModelResponse(text, {"input_tokens": 100, "output_tokens": 50})], input_tokens=100)
    result = run_review_b_pipeline(make_task(), client)
    assert result.reason == "invalid_model_response"
    assert len(client.requests) == 1


def test_second_stage_timeout_preserves_first_call_and_unknown_usage() -> None:
    client = MockModelClient([responses()[0], TimeoutError("secret-token")], input_tokens=100)
    result = run_review_b_pipeline(make_task(), client)
    assert (result.status, result.reason) == ("failed", "model_timeout")
    assert len(client.requests) == 2
    assert result.goals is not None and result.observations is None
    assert result.usage["calls"] == 2
    assert result.usage["known_input_tokens"] == 100
    assert result.usage["input_tokens"] is None
    assert result.model_delivery["observe_changes"]["status"] == "unknown"
    assert "secret-token" not in repr(result)


def test_bounded_transient_retry_preserves_both_attempts() -> None:
    profile = ModelProfile("mock", "fixture", max_attempts=2)
    client = MockModelClient([TransientModelError("private"), *responses()], profile=profile, input_tokens=100)
    result = run_review_b_pipeline(make_task(), client)
    assert result.status == "completed"
    assert result.usage["calls"] == 4
    assert len(result.stages[0].attempts) == 2
    assert result.usage["input_tokens"] is None
    assert result.usage["known_input_tokens"] == 300


@pytest.mark.parametrize("profile,limits,reason,status,expected_calls", [
    (ModelProfile("mock", "fixture", max_input_bytes=100), None, "input_limit_exceeded", "not_run", 0),
    # Token budgets are proven post-send from response usage (no measurement call):
    # response input 100 > 99 fails closed after the single generation.
    (ModelProfile("mock", "fixture", max_input_tokens=99), None, "usage_limit_exceeded", "failed", 1),
    # Total budgets are gated pre-send via max_output_tokens > max_total_tokens (no metering).
    (ModelProfile("mock", "fixture"), IntentReviewLimits(max_total_tokens=4095), "input_limit_exceeded", "not_run", 0),
    (ModelProfile("mock", "fixture", max_attempts=0), None, "invalid_model_profile", "not_run", 0),
])
def test_preflight_limits_make_no_review_calls(profile, limits, reason: str, status: str, expected_calls: int) -> None:
    client = MockModelClient(responses(), profile=profile, input_tokens=100)
    assert not hasattr(client, "count_input_tokens")
    assert not hasattr(client, "token_count_receipt")
    result = run_review_b_pipeline(make_task(), client, limits=limits)
    assert (result.status, result.reason) == (status, reason)
    assert len(client.requests) == expected_calls
    assert result.usage["calls"] == expected_calls
    assert result.usage["token_measurements"] == 0
    assert result.usage["metering_input_tokens"] == 0
    assert result.usage["metering_output_tokens"] == 0
    assert all(record.token_measurement == "not_attempted" for record in result.stages)
    assert all(record.measured_input_tokens is None for record in result.stages)
    assert all(record.token_measurement_model_calls == 0 for record in result.stages)
    if expected_calls == 0:
        assert client.requests == []
        assert result.usage["input_tokens"] == 0
        assert result.usage["output_tokens"] == 0
    else:
        assert result.usage["input_tokens"] == 100
        assert result.usage["output_tokens"] == 50


def test_token_count_is_required_not_estimated() -> None:
    """Retired metering contract: input_tokens= is ignored; admission uses bytes+usage.

    No measurement call exists (no count_input_tokens/token_count_receipt). The mock
    validates with only complete() calls, and byte-gate refusals happen without any
    extra call. Name retained for history; assertions cover the new contract.
    """
    assert not hasattr(MockModelClient, "count_input_tokens")
    # input_tokens=None is ignored; response usage alone drives accounting.
    client = MockModelClient(responses())
    result = run_review_b_pipeline(make_task(), client)
    assert (result.status, result.reason) == ("completed", "validated")
    assert len(client.requests) == 3  # only complete() calls, zero measurement calls
    assert result.usage["calls"] == 3
    assert result.usage["input_tokens"] == 300
    assert result.usage["output_tokens"] == 150
    assert result.usage["token_measurements"] == 0
    assert result.usage["metering_input_tokens"] == 0
    assert result.usage["metering_output_tokens"] == 0
    for record in result.stages:
        assert record.token_measurement == "not_attempted"
        assert record.measured_input_tokens is None
        assert record.token_measurement_request_hash is None
        assert record.token_measurement_delivery == "not_attempted"
        assert record.token_measurement_model_calls == 0
        assert record.token_measurement_usage is None
    for record in result.model_delivery.values():
        assert record["token_measurement"] == "not_attempted"
        assert record["measured_input_tokens"] is None
        assert record["token_measurement_model_calls"] == 0
    # admission_bytes is never consulted by the declared pipeline (no extra call).
    class ExplodingAdmission(MockModelClient):
        def admission_bytes(self, request):  # type: ignore[override]
            raise AssertionError("declared pipeline must not call admission_bytes")

    exploding = ExplodingAdmission(responses())
    assert run_review_b_pipeline(make_task(), exploding).status == "completed"
    assert len(exploding.requests) == 3
    # Byte-gate refusal happens without any extra call.
    small = MockModelClient(responses(), profile=ModelProfile("mock", "fixture", max_input_bytes=100))
    refused = run_review_b_pipeline(make_task(), small)
    assert (refused.status, refused.reason) == ("not_run", "input_limit_exceeded")
    assert small.requests == []
    assert refused.usage["calls"] == 0
    assert refused.usage["token_measurements"] == 0
    assert refused.usage["metering_input_tokens"] == 0
    assert refused.usage["metering_output_tokens"] == 0


def test_stage_gate_loss_prevents_next_stage() -> None:
    client = MockModelClient(responses(), input_tokens=100)
    result = run_review_b_pipeline(make_task(), client, before_stage=lambda: len(client.requests) == 0)
    assert result.reason == "stage_gate_rejected"
    assert len(client.requests) == 1
    assert result.usage["calls"] == 1
    assert not result.findings


def test_tampered_fixed_input_is_not_sent() -> None:
    client = MockModelClient(responses(), input_tokens=100)
    task = replace(make_task(), input_hash="e" * 64)
    result = run_review_b_pipeline(task, client)
    assert result.reason == "invalid_task_input"
    assert client.requests == []


@pytest.mark.parametrize("overflow_stage", [1, 2])
def test_later_stage_overflow_keeps_actual_earlier_usage_and_full_scope(overflow_stage: int) -> None:
    bodies = stage_bodies()
    usages = [{"input_tokens": 100, "output_tokens": 50} for _ in bodies]
    usages[overflow_stage] = {"input_tokens": 200, "output_tokens": 50}
    profile = ModelProfile("mock", "fixture", max_input_tokens=150)
    # input_tokens= is ignored; per-response usage alone enforces the token budget.
    client = MockModelClient(
        [ModelResponse(json.dumps(body), usage) for body, usage in zip(bodies, usages)],
        profile=profile,
        input_tokens=9999,
    )
    assert not hasattr(client, "count_input_tokens")
    task = make_task()
    result = run_review_b_pipeline(task, client)
    assert (result.status, result.reason) == ("failed", "usage_limit_exceeded")
    assert len(client.requests) == overflow_stage + 1
    assert result.usage["calls"] == overflow_stage + 1
    assert result.usage["input_tokens"] == 100 * overflow_stage + 200
    assert result.usage["output_tokens"] == 50 * (overflow_stage + 1)
    assert result.usage["known_input_tokens"] == 100 * overflow_stage + 200
    assert result.usage["token_measurements"] == 0
    assert result.usage["metering_input_tokens"] == 0
    assert result.usage["metering_output_tokens"] == 0
    assert all(record.token_measurement == "not_attempted" for record in result.stages)
    assert all(record.measured_input_tokens is None for record in result.stages)
    assert all(record.token_measurement_model_calls == 0 for record in result.stages)
    assert result.scope["reviewed_paths"] == list(task.reviewed_paths)
    assert result.goals is not None
    assert all(record.status == "completed" for record in result.stages[:overflow_stage])
    assert result.stages[overflow_stage].status == "failed"
    assert result.stages[overflow_stage].reason == "usage_limit_exceeded"


@pytest.mark.parametrize("failure", ["oversized", "too_many", "boolean_usage", "missing_usage", "exception"])
def test_output_and_usage_fail_closed_without_raw_exception_values(failure: str) -> None:
    body = stage_bodies()[0]
    usage = {"input_tokens": 100, "output_tokens": 50}
    response = None
    if failure == "oversized":
        body["declared_goals"][0]["statement"] = "x" * 40000
    elif failure == "too_many":
        body["declared_goals"] = [{**body["declared_goals"][0], "goal_id": f"G{index}"} for index in range(21)]
    elif failure == "boolean_usage":
        usage["input_tokens"] = True
    elif failure == "missing_usage":
        usage = {}
    else:
        response = RuntimeError("SECRET PRIVATE EXCEPTION")
    client = MockModelClient([response or ModelResponse(json.dumps(body), usage)], input_tokens=100)
    result = run_review_b_pipeline(make_task(), client)
    assert result.status == "failed"
    assert len(client.requests) == 1
    assert not result.findings
    assert "SECRET PRIVATE EXCEPTION" not in repr(result)


def test_deadline_after_a_call_stops_and_keeps_that_calls_usage(monkeypatch) -> None:
    from review_bot.application import declared_intent_review

    clock = [0.0]
    monkeypatch.setattr(declared_intent_review.time, "monotonic", lambda: clock[0])

    class SlowClient(MockModelClient):
        def complete(self, request):
            response = super().complete(request)
            clock[0] = 100.0
            return response

    client = SlowClient(responses(), input_tokens=100)
    result = run_review_b_pipeline(make_task(), client, limits=IntentReviewLimits(task_timeout_seconds=10))
    assert result.reason == "task_deadline_exceeded"
    assert len(client.requests) == 1
    assert client.requests[0].timeout_seconds <= 10
    assert result.usage["input_tokens"] == 100
    assert not result.findings


@pytest.mark.parametrize("assessment", ["consistent_with_evidence", "evidence_conflict"])
def test_declared_and_observed_evidence_can_support_noninsufficient_assessments(assessment: str) -> None:
    bodies = stage_bodies()
    bodies[2]["findings"][0].update(assessment=assessment, missing_evidence=[])
    result = run_review_b_pipeline(make_task(), MockModelClient(responses(bodies), input_tokens=100))
    assert result.status == "completed"
    assert result.findings[0].assessment == assessment


def test_assessment_request_builder_revalidates_prior_results() -> None:
    from review_bot.application.declared_intent_review import build_intent_model_request, validate_intent_goals, validate_intent_observations

    task = make_task()
    bodies = stage_bodies()
    goals = validate_intent_goals(bodies[0], task)
    observations = validate_intent_observations(bodies[1], task, goals)
    forged = replace(goals, declared_goals=(replace(goals.declared_goals[0], quote="Not original text"),))
    with pytest.raises(ValueError):
        build_intent_model_request(task, "assess", ModelProfile("mock", "fixture"), goals=forged, observations=observations)


@pytest.mark.parametrize(("title", "body", "quote"), [
    ("缓存", "", "缓存"), ("Intent fixture", "不要重复", "不要重复"),
])
def test_brief_non_english_intent_is_valid_when_quote_is_the_complete_text_field(
    title: str, body: str, quote: str,
) -> None:
    task = make_task(body=body, title=title)
    bodies = stage_bodies()
    bodies[0]["declared_goals"][0].update(statement="避免重复处理", quote=quote)

    result = run_review_b_pipeline(task, MockModelClient(responses(bodies), input_tokens=100))

    assert (result.status, result.reason) == ("completed", "validated")
    assert result.goals.declared_goals[0].quote == quote


def test_extract_goals_prompt_states_quote_length_exception_and_context_expansion() -> None:
    client = MockModelClient(responses(), input_tokens=100)
    result = run_review_b_pipeline(make_task(), client)

    assert (result.status, result.reason) == ("completed", "validated")
    prompt = " ".join(client.requests[0].system_prompt.split())
    assert f"at least {MIN_QUOTE_NONSPACE_CHARS} non-whitespace Unicode characters" in prompt
    assert "entire original title or body field (after stripping leading/trailing whitespace)" in prompt
    assert "in every referenced text record" in prompt
    assert "quote surrounding original text in the same contiguous span to reach the minimum" in prompt


@pytest.mark.parametrize(("quote", "status", "calls"), [
    ("결제 중복을 막는다.", "failed", 1),
    ("배경 설명입니다. 결제 중복을 막는다.", "completed", 3),
])
def test_brief_korean_goal_embedded_in_body_requires_surrounding_exact_span(
    quote: str, status: str, calls: int,
) -> None:
    goal = "결제 중복을 막는다."
    body = "배경 설명입니다. 결제 중복을 막는다. 그 외 변경 없음."
    assert sum(not character.isspace() for character in goal) < MIN_QUOTE_NONSPACE_CHARS
    assert goal in quote and quote in body and quote != body
    task = make_task(body=body)
    bodies = stage_bodies()
    bodies[0]["declared_goals"][0].update(statement=goal, quote=quote)
    client = MockModelClient(responses(bodies), input_tokens=100)

    result = run_review_b_pipeline(task, client)

    assert (result.status, result.reason) == (status, "validated" if status == "completed" else "invalid_model_response")
    assert len(client.requests) == calls
    assert result.usage["calls"] == calls
    if status == "completed":
        assert result.goals.declared_goals[0].statement == goal
        assert result.goals.declared_goals[0].quote == quote
        assert goal not in client.requests[1].user_content
        assert all(stage.status == "completed" for stage in result.stages)
    else:
        assert all(stage.status == "not_attempted" for stage in result.stages[1:])


def test_an_existing_b_output_namespace_cannot_be_reserved_twice(tmp_path) -> None:
    from review_bot.runtime.artifacts import IntentArtifacts

    root = tmp_path / "reserved"
    IntentArtifacts(root)
    with pytest.raises(ValueError):
        IntentArtifacts(root)


def test_profile_extensions_cannot_leak_credentials_into_result() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class ExtendedProfile(ModelProfile):
        private_token: str = "SECRET NOT PROFILE METADATA"

    client = MockModelClient(responses(), profile=ExtendedProfile("mock", "fixture"), input_tokens=100)
    result = run_review_b_pipeline(make_task(), client)
    assert result.status == "completed"
    assert "private_token" not in result.model["profile"]
    assert "SECRET NOT PROFILE METADATA" not in repr(result)