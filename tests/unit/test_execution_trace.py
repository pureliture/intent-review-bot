"""Folded execution trace in A/B partition comments."""
from __future__ import annotations

from dataclasses import replace

from review_bot.adapters.markdown.developer_presentation import execution_trace
from review_bot.domain.intent_models import IntentModelAttempt, IntentStageRecord
from review_bot.domain.partition_review import PartitionReviewResult


def _result(**changes):
    attempt = IntentModelAttempt(1, "completed", "sent", "validated", "t0", "t1", 1200, {"input_tokens": 300, "output_tokens": 40})
    lost = IntentModelAttempt(1, "interrupted", "unknown", "timeout", "t0", "t1", 60000, None)
    stages = (IntentStageRecord("review_partition:G1:admission", "completed", "input_measured"),
        IntentStageRecord("review_partition:G1", "completed", "validated", attempts=(attempt,)),
        IntentStageRecord("final_review", "failed", "timeout", attempts=(attempt, lost)))
    base = PartitionReviewResult("B", "partial_review", "task_deadline_exceeded", "i" * 64, "k" * 64, "s" * 64,
        "a" * 40, "b" * 40, "2026-11-09T00:00:00+00:00", "partitioned-review/v20", "p" * 64,
        {"provider": "ollama", "model": "granite4.2:128k", "profile": {"max_input_tokens": 126976,
            "max_output_tokens": 4096, "timeout_seconds": 600}},
        {}, {"model_calls": 3, "known_input_tokens": 600, "known_output_tokens": 80, "unknown_usage_attempts": 1,
            "usage_complete": False}, {}, stages)
    return replace(base, **changes)


def test_trace_reports_stages_models_and_unknown_usage_without_zeroing():
    text = execution_trace(_result(), context={"repository": "o/r", "pr_number": 7, "run_id": "run-1",
        "models": {"decision": "tev1:0.8b-128k", "review": "granite4.2:128k"}})
    assert text.startswith("<details>\n<summary>리뷰 실행 내역</summary>")
    assert "tev1:0.8b-128k" in text and "granite4.2:128k" in text
    assert "`partial_review`" in text and "task_deadline_exceeded" in text.replace("\\", "")
    assert "| review\\_partition | 1 | 1 | 1 | 0 | 0 | 1200 | 300 | 40 | 0 |" in text
    assert "| final\\_review | 0 | 2 | 1 | 1 | 1 | 61200 | 300 | 40 | 1 |" in text
    assert "사용량 미반환 호출 1건 (0으로 집계하지 않음)" in text


def test_trace_contains_no_prompt_or_response_material():
    text = execution_trace(_result(summary="SECRET_SUMMARY", findings=({"text": "SECRET_FINDING"},)))
    assert "SECRET" not in text


def test_agent_trace_aggregates_context_pressure_and_renders_verification_usage():
    generation_usage = {
        "input_tokens": 40, "output_tokens": 10, "total_tokens": 50, "model_calls": 3,
        "max_prompt_tokens": 100, "compactions": 2,
    }
    verification_runs = [
        {"role": "verify", "usage": {
            "input_tokens": 20, "output_tokens": 10, "total_tokens": 30,
            "max_prompt_tokens": 80, "compactions": 2,
        }},
        {"role": "revise", "usage": {
            "input_tokens": 999, "output_tokens": 999, "total_tokens": 1998,
            "max_prompt_tokens": 1000, "compactions": 100,
        }},
        {"role": "verify", "usage": {
            "input_tokens": 15, "output_tokens": 5, "total_tokens": 20,
            "max_prompt_tokens": 120, "compactions": 1,
        }},
    ]
    partition = {
        "engine": "pi-agent",
        "agent": {
            "pi_version": "0.73.1", "skill_sha256": "a" * 64, "model": "beam-free",
            "usage": generation_usage, "repair_attempted": False, "rejected": [],
        },
        "fact_verification": {"runs": verification_runs},
    }

    text = execution_trace(_result(partition=partition))

    assert ("- Agent usage: input 40 · output 10 · total 50 tokens · "
        "max request input 100 · compactions 2") in text
    assert ("- Agent verification usage: input 35 · output 15 · total 50 tokens · "
        "max request input 120 · compactions 3") in text
