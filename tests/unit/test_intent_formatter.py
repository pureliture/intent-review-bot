"""Drafts never trust model HTML, markers, mentions, URLs, or forged evidence."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from review_bot.adapters.markdown.intent_formatter import DEFAULT_B_MARKER, format_intent_markdown
from review_bot.application.review_a import MockModelClient
from review_bot.application.review_b import run_review_b_pipeline
from tests.fixtures.review_b import make_task, responses, stage_bodies


def test_untrusted_html_markers_mentions_and_urls_are_inert() -> None:
    task = make_task()
    bodies = deepcopy(stage_bodies())
    attack = '<script>alert(1)</script> <!-- review-bot:intent:v1 --> @everyone [link](https://evil.invalid) https://evil.invalid www.evil.invalid'
    bodies[2]["summary"] = attack
    bodies[2]["findings"][0]["statement"] = attack
    bodies[1]["intent_hypotheses"][0]["statement"] = attack
    result = run_review_b_pipeline(task, MockModelClient(responses(bodies), input_tokens=100))
    assert result.status == "completed"
    draft = format_intent_markdown(result, task)
    assert draft.count(DEFAULT_B_MARKER) == 1
    assert "<script>" not in draft and "<html" not in draft
    assert "@everyone" not in draft
    assert "https://evil.invalid" not in draft and "www.evil.invalid" not in draft
    assert "&lt;script&gt;" in draft
    assert f"/blob/{task.head.sha}/src/request.py#L1-L3" in draft
    assert "not executed" in draft and "not the whole repository" in draft


def test_failed_assessment_does_not_render_unvalidated_opinions() -> None:
    task = make_task()
    bodies = stage_bodies()
    bodies[2]["findings"][0]["goal_ids"] = ["invented-goal"]
    result = run_review_b_pipeline(task, MockModelClient(responses(bodies), input_tokens=100))
    draft = format_intent_markdown(result, task)
    assert "No validated final assessment" in draft
    assert "Sequential retry behavior needs context" not in draft
    assert "Declared goals" in draft


def test_formatter_revalidates_input_and_result_relationships_before_linking() -> None:
    task = make_task()
    result = run_review_b_pipeline(task, MockModelClient(responses(), input_tokens=100))
    forged = replace(result, findings=(replace(result.findings[0], evidence_ids=("not-evidence",)),))
    with pytest.raises(ValueError):
        format_intent_markdown(forged, task)
    with pytest.raises(ValueError):
        format_intent_markdown(replace(result, head_sha="e" * 40), task)


@pytest.mark.parametrize("limit", [1000, 1200])
def test_comment_limit_omits_body_without_cutting_input_or_partial_markdown(limit: int) -> None:
    task = make_task()
    result = run_review_b_pipeline(task, MockModelClient(responses(), input_tokens=100))
    draft = format_intent_markdown(result, task, max_length=limit)
    assert len(draft) <= limit
    assert "comment_limit_exceeded" in draft
    assert draft.count(DEFAULT_B_MARKER) == 1
    assert "Mock model" in draft and "not executed" in draft
    assert result.findings


def test_no_declared_goal_is_visible_and_default_marker_cannot_be_overridden() -> None:
    task = make_task(body="")
    result = run_review_b_pipeline(task, MockModelClient(responses(stage_bodies(no_goals=True)), input_tokens=100))
    assert "No declared goal" in format_intent_markdown(result, task)
    with pytest.raises(ValueError):
        format_intent_markdown(result, task, marker="<!-- review-bot:architecture:v1 -->")