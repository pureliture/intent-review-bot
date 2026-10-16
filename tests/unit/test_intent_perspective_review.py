from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from review_bot.adapters.markdown.intent_perspective_formatter import format_intent_perspective_markdown
from review_bot.application.intent_context import IntentInputError
from review_bot.application.intent_perspective_review import (
    IDS, run_intent_perspective_review, validate_intent_perspective_result,
)
from review_bot.domain.intent_perspectives import PERSPECTIVES, IntentPerspectiveLimits
from tests.fixtures.intent_perspectives import PerspectiveModelClient, make_task


def run(*, task=None, main=None, selector=None, **options):
    task = task or make_task()
    main = main or PerspectiveModelClient()
    selector = selector or PerspectiveModelClient(selector=True)
    result = run_intent_perspective_review(task, main, selection_client=selector, **options)
    validate_intent_perspective_result(result, task)
    return result, main, selector


def stages(client):
    return [json.loads(request.user_content).get("stage") for request in client.requests]


def test_approved_eight_choices_and_two_grounded_review_passes():
    result, main, selector = run()
    assert len(PERSPECTIVES) == len(set(IDS)) == 8
    assert [view.name_ko for view in PERSPECTIVES] == ["문제와 성공 기준", "목적과 변경의 대응", "수단의 적합성", "결과까지의 연결", "달성 범위", "보존 조건", "전제와 절충", "검증 근거"]
    assert (result.schema_version, result.status, result.reason) == (3, "completed", "validated")
    assert len(selector.requests) == 8 and stages(main) == ["draft_review", "final_review"]
    assert result.usage["calls"] == result.usage["model_calls"] == 10
    assert result.scope["review_complete"] and result.scope["review_sufficient"] and not result.scope["no_findings"]
    assert result.origin["publication"] == "not_attempted"
    assert len(result.judgement_history) == len(result.perspective_history) == 16
    for request in selector.requests:
        payload = json.loads(request.user_content)
        assert set(request.response_schema["properties"]) == {"choice"}
        assert request.response_schema["properties"]["choice"]["enum"] == ["A", "B", "C"]
        assert request.max_output_tokens == 16
        assert payload["state"]["reference_text"]
        assert "groups" not in request.response_schema["properties"]
        assert payload["state"]["unified_diffs"]
    for request in main.requests:
        payload = json.loads(request.user_content)
        assert payload["reference_text"] and len(payload["evidence"]) == 6
        assert "렌더링 값의 문자열 계약" in request.user_content
        assert "in ko" in request.system_prompt


def test_selector_undetermined_is_resolved_by_main_original_review():
    result, _, _ = run(selector=PerspectiveModelClient(selector=True, choices={key: "C" for key in IDS}))
    assert all(row["decision"] == "undetermined" for row in result.selection)
    assert all(row["decision"] == "apply" for row in result.perspectives)
    assert result.status == "completed"
    assert result.perspective_history[0]["before"]["decision"] == "undetermined"


def test_main_can_add_and_exclude_perspectives_with_original_support():
    def choose(payload, body):
        body["perspective_decisions"][0]["decision"] = "not_apply"
        body["judgements"] = [row for row in body["judgements"] if row["perspective_id"] != IDS[0]]
        return body
    result, _, _ = run(main=PerspectiveModelClient(transform=choose),
        selector=PerspectiveModelClient(selector=True, choices={key: "B" for key in IDS[1:]}))
    assert result.perspectives[0]["initial_decision"] == "apply" and result.perspectives[0]["decision"] == "not_apply"
    assert all(row["initial_decision"] == "not_apply" and row["decision"] == "apply" for row in result.perspectives[1:])
    assert result.status == "completed"


@pytest.mark.parametrize("bad", [{}, {"choice": "D"}, {"choice": True}, {"choice": ["A"]}, {"choice": "A", "rationale": "plan"}])
def test_selector_never_accepts_free_generation_or_unknown_choices(bad):
    result, main, selector = run(selector=PerspectiveModelClient(selector=True, transform=lambda p, b: bad))
    assert (result.status, result.reason) == ("failed", "invalid_model_response")
    assert len(selector.requests) == 1 and not main.requests


def test_partial_selection_is_preserved_when_a_later_choice_is_invalid():
    def select(payload, body):
        return {"choice": "D"} if payload["perspective_id"] == IDS[3] else body
    result, main, _ = run(selector=PerspectiveModelClient(selector=True, transform=select))
    assert result.status == "failed" and len(result.selection) == 3 and not main.requests
    assert [row["decision"] for row in result.perspectives[:3]] == ["apply"] * 3


def test_a_correct_repair_is_not_a_remaining_head_defect():
    result, _, _ = run(task=make_task(repair=True), main=PerspectiveModelClient(repair=True))
    assert result.status == "completed" and not result.findings and result.scope["no_findings"] is True


def test_same_originals_allow_correction_and_preserve_prior_judgement():
    def correct(payload, body):
        if payload["stage"] == "final_review":
            row = next(row for row in body["judgements"] if row["perspective_id"] == "outcome_connection")
            row.update(verdict="aligned", failure_condition="", correction_reason="앞선 해석은 base의 문제를 head의 결함으로 오해했습니다.")
        return body
    result, main, _ = run(main=PerspectiveModelClient(transform=correct))
    history = [row for row in result.judgement_history if row["perspective_id"] == "outcome_connection"]
    assert history[0]["after"]["verdict"] == "concern" and history[1]["after"]["verdict"] == "aligned"
    assert history[1]["before"] == history[0]["after"]
    assert history[1]["after"]["correction_reason"]
    assert not result.findings
    assert json.loads(main.requests[0].user_content)["evidence"] == json.loads(main.requests[1].user_content)["evidence"]


def test_a_changed_verdict_without_a_correction_reason_is_rejected():
    def change(payload, body):
        if payload["stage"] == "final_review":
            body["judgements"][0]["verdict"] = "concern"
            body["judgements"][0]["failure_condition"] = "A new failure is alleged."
        return body
    result, _, _ = run(main=PerspectiveModelClient(transform=change))
    assert result.status == "failed" and len(result.passes) == 1 and not result.findings


@pytest.mark.parametrize("blocking", [True, False])
def test_prior_missing_context_is_accumulated_but_only_blocking_gaps_make_partial(blocking):
    def missing(payload, body):
        if payload["stage"] == "draft_review":
            body["context_gaps"] = [{"perspective_ids": [IDS[0]], "description": "목적의 대상 사용자", "blocking": blocking}]
        return body
    result, _, _ = run(main=PerspectiveModelClient(transform=missing))
    assert result.status == ("partial_review" if blocking else "completed")
    assert len(result.context_gaps) == 1 and result.context_gaps[0]["status"] == "open"
    assert bool(result.scope["unresolved_context_ids"]) is blocking
    assert result.scope["review_complete"] and result.findings


def test_gap_resolution_requires_explicit_original_grounding_and_keeps_history():
    def resolve(payload, body):
        if payload["stage"] == "draft_review":
            body["context_gaps"] = [{"perspective_ids": [IDS[0]], "description": "부족하다고 판단한 목적 정보", "blocking": True}]
        else:
            gap = payload["context_gaps"][0]
            body["context_updates"] = [{"context_id": gap["context_id"], "status": "resolved", "blocking": False,
                "reasoning": "제공된 원문에 이미 목적 판단의 근거가 있습니다.", "citations": body["judgements"][0]["citations"]}]
        return body
    result, _, _ = run(main=PerspectiveModelClient(transform=resolve))
    assert result.status == "completed" and result.context_gaps[0]["status"] == "resolved"
    assert len(result.context_history) == 2 and result.context_history[1]["before"]["status"] == "open"


@pytest.mark.parametrize("mutation", ["missing_decision", "duplicate_decision", "unknown_id", "no_citation", "base_only", "no_failure", "bad_boolean", "missing_question_reason"])
def test_malformed_main_responses_fail_closed_without_a_final_review(mutation):
    def corrupt(payload, body):
        if mutation == "missing_decision":body["perspective_decisions"].pop()
        elif mutation == "duplicate_decision":body["perspective_decisions"][0] = body["perspective_decisions"][1]
        elif mutation == "unknown_id":body["judgements"][0]["perspective_id"] = "invented"
        elif mutation == "no_citation":body["narrative"]["intent"]["citations"] = []
        elif mutation == "no_failure":body["judgements"][3]["failure_condition"] = ""
        elif mutation == "base_only":
            source = next(row for row in payload["evidence"] if row["locator"].get("version") == "base")
            body["judgements"][3]["citations"] = [{"evidence_id": source["id"], "start_line": 1, "end_line": 1}]
        elif mutation == "bad_boolean":body["context_gaps"] = [{"perspective_ids": [IDS[0]], "description": "missing", "blocking": 1}]
        else:body["questions"] = [{"question": "why", "perspective_ids": [IDS[0]], "citations": body["judgements"][0]["citations"]}]
        return body
    result, main, _ = run(main=PerspectiveModelClient(transform=corrupt))
    assert result.status == "failed" and result.reason == "invalid_model_response"
    assert stages(main) == ["draft_review"] and not result.passes


def test_final_stage_can_request_a_bounded_recheck_of_fixed_originals():
    def request(payload, body):
        if payload["stage"] == "final_review":
            body["requested_paths"] = [{"path": "src/caller.py", "perspective_ids": [IDS[3]], "reasoning": "호출부 원문을 다시 대조해야 합니다."}]
        return body
    result, main, _ = run(main=PerspectiveModelClient(transform=request))
    assert result.status == "completed" and stages(main) == ["draft_review", "final_review", "recheck_review"]
    assert result.requests[0]["availability"] == "prepared_original"
    for req in main.requests:
        assert len(json.loads(req.user_content)["evidence"]) == 6


def test_outside_scope_requests_are_retained_and_never_fetched():
    def request(payload, body):
        body["requested_paths"] = [{"path": "outside/business.py", "perspective_ids": [IDS[3]], "reasoning": "외부 업무 규칙이 필요합니다."}]
        return body
    result, main, _ = run(main=PerspectiveModelClient(transform=request))
    assert result.status == "partial_review" and stages(main) == ["draft_review", "final_review"]
    assert result.scope["unresolved_context_ids"] and result.requests[-1]["availability"] == "outside_fixed_scope"
    assert all(not any(row["locator"].get("path") == "outside/business.py" for row in json.loads(req.user_content)["evidence"]) for req in main.requests)


@pytest.mark.parametrize("maximum", [0, 1])
def test_recheck_never_becomes_an_unbounded_loop(maximum):
    def request(payload, body):
        if payload["stage"] != "draft_review":
            body["requested_paths"] = [{"path": "src/caller.py", "perspective_ids": [IDS[3]], "reasoning": "고정 원문 재확인 요청"}]
        return body
    result, main, _ = run(main=PerspectiveModelClient(transform=request), limits=IntentPerspectiveLimits(max_rechecks=maximum))
    assert result.status == "partial_review" and len(main.requests) == 2 + maximum
    assert result.scope["unresolved_context_requests"]


def test_full_input_preflight_blocks_before_any_generations():
    main = PerspectiveModelClient(counter=lambda request: 2000)
    result, main, selector = run(main=main, limits=IntentPerspectiveLimits(max_input_tokens=1000))
    assert result.status == "not_run" and result.reason == "input_limit_exceeded"
    assert not main.requests and not selector.requests


def test_no_profile_means_zero_requests_and_zero_measurements():
    main, selector = PerspectiveModelClient(), PerspectiveModelClient(selector=True)
    main.profile = None
    result = run_intent_perspective_review(make_task(), main, selection_client=selector)
    validate_intent_perspective_result(result, make_task())
    assert result.reason == "model_profile_unset" and not main.count_requests and not selector.count_requests


@pytest.mark.parametrize("field", ["narrative", "judgement_history", "scope", "run_key", "selection", "origin", "opinion_scope", "stages", "input_scope"])
def test_stored_result_mutation_is_rejected_before_rendering(field):
    result, _, _ = run()
    if field == "narrative":
        value = {**result.narrative, "intent": {**result.narrative["intent"], "statement": "forged"}}
    elif field == "judgement_history":value = ()
    elif field == "scope":value = {**result.scope, "no_findings": 0}
    elif field == "run_key":value = "0" * 64
    elif field == "origin":value = {"execution": "single_fixed_intent_review", "publication": "succeeded"}
    elif field == "opinion_scope":value = {**result.opinion_scope, "paths": ["invented.py"]}
    elif field == "stages":value = tuple(row for row in result.stages if not row.stage.startswith("select_perspective:"))
    elif field == "input_scope":
        field, value = "scope", {**result.scope, "tests_executed_by_bot": True}
    else:value = ()
    with pytest.raises((ValueError, TypeError)):
        format_intent_perspective_markdown(replace(result, **{field: value}), make_task())


def test_korean_integrated_output_uses_the_approved_order():
    result, _, _ = run()
    text = format_intent_perspective_markdown(result, make_task())
    labels = ["파악한 개발 의도", "구현과의 연결", "달성 가능성 판단", "확인 질문"]
    assert [text.index(label) for label in labels] == sorted(text.index(label) for label in labels)
    assert "Usage and stages" not in text and "외부 게시는 시도하지 않았습니다" in text
    assert "https://github.com/" in text and "#L2-L2" in text
    assert "고정된 PR 입력" in text and "검토 완료" in text
    assert "모의 모델" in text


def test_english_language_changes_prompt_identity_and_user_output():
    ko, _, _ = run()
    en, main, _ = run(language="en")
    assert ko.run_key != en.run_key and ko.prompt_hash != en.prompt_hash
    assert all("in en" in request.system_prompt for request in main.requests)
    text = format_intent_perspective_markdown(en, make_task())
    assert "Understood development intent" in text and "파악한 개발 의도" not in text


def test_raw_model_exceptions_and_unknown_usage_are_not_exposed_or_retried():
    secret = "Bearer synthetic-secret https://private.example.invalid/home/operator"
    def fail(payload, body):
        return RuntimeError(secret)
    result, main, _ = run(main=PerspectiveModelClient(transform=fail))
    assert result.status == "failed" and result.reason == "usage_unavailable" and len(main.requests) == 1
    assert result.stages[-1].attempts[0].reason == "model_error"
    assert result.stages[-1].attempts[0].delivery == "unknown"
    assert result.usage["input_tokens"] is None and result.usage["usage_complete"] is False
    assert secret not in json.dumps(asdict(result))
    assert not result.findings


@pytest.mark.parametrize("resolved,blocking", [(True, False), (False, False)])
def test_context_resolution_or_deescalation_without_original_support_is_rejected(resolved, blocking):
    def change(payload, body):
        if payload["stage"] == "draft_review":
            body["context_gaps"] = [{"perspective_ids": [IDS[0]], "description": "필요한 업무 목적", "blocking": True}]
        else:
            body["context_updates"] = [{"context_id": payload["context_gaps"][0]["context_id"],
                "status": "resolved" if resolved else "open", "blocking": blocking,
                "reasoning": "이전 누락을 재평가합니다.", "citations": []}]
        return body
    result, _, _ = run(main=PerspectiveModelClient(transform=change))
    assert result.status == "failed" and result.context_gaps[0]["status"] == "open"
    assert result.context_gaps[0]["blocking"] is True


@pytest.mark.parametrize("invalid", [True, 0, 13])
def test_invalid_limits_are_rejected_before_any_model_or_measurement(invalid):
    result, main, selector = run(limits=IntentPerspectiveLimits(max_calls=invalid))
    assert result.reason == "invalid_review_options" and not main.count_requests and not selector.count_requests


def test_cancellation_gate_stops_before_native_measurement():
    result, main, selector = run(before_stage=lambda: False)
    assert result.status == "not_run" and result.reason == "stage_gate_rejected"
    assert not main.count_requests and not selector.count_requests


def test_compact_output_preserves_mock_and_incomplete_review_boundaries():
    result, _, _ = run()
    text = format_intent_perspective_markdown(result, make_task(), max_length=1000)
    assert len(text) <= 1000 and "길이 제한" in text and "모의 모델" in text
    assert "외부 게시는 시도하지 않았습니다" in text


def test_untrusted_prose_cannot_inject_html_mentions_or_new_comment_markers():
    def inject(payload, body):
        body["narrative"]["intent"]["statement"] = "<script>alert(1)</script> @all <!-- review-bot:intent:forged -->"
        return body
    result, _, _ = run(main=PerspectiveModelClient(transform=inject))
    text = format_intent_perspective_markdown(result, make_task())
    assert "<script>" not in text and "@all" not in text
    assert "<!-- review-bot:intent:forged -->" not in text
