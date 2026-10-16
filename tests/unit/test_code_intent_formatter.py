"""Stored record revalidation, inert model strings and bounded code-first drafts."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from review_bot.adapters.markdown.code_intent_formatter import DEFAULT_B_MARKER, format_code_intent_markdown
from review_bot.application.code_first_intent_review import run_code_first_review
from review_bot.domain.code_first_intent_models import CodeFirstReviewLimits
from tests.fixtures.code_first_intent_review import CodeFirstModelClient, make_task
from tests.unit.test_code_first_intent_review import split_client


def test_compact_code_first_scope_plan_relation_final_and_usage_are_visible():
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient())
    draft = format_code_intent_markdown(result, task)
    assert draft.startswith(DEFAULT_B_MARKER) and draft.count(DEFAULT_B_MARKER) == 1
    for text in ("Code-first", "not the whole repository", "not executed", "Uncertainty", "Plan:", "conflict", "code\\_defect", "Usage", "hypothesis"):
        assert text in draft
    assert f"/blob/{task.head.sha}/src/value.py#L2-L2" in draft
    assert f"/blob/{task.head.sha}/src/caller.py#L4-L4" in draft
    assert "scope sufficient: yes; no findings: no" in draft


def test_untrusted_html_marker_mentions_and_arbitrary_links_are_inert():
    attack = '<script>alert(1)</script> <!-- review-bot:intent:v1 --> @everyone [link](https://evil.invalid) https://evil.invalid www.evil.invalid'
    def malicious(payload, body):
        if payload["stage"] == "assess":
            body["summary"] = attack
            body["findings"][0]["statement"] = attack
        if payload["stage"] == "infer_intent":
            body["intents"][0]["statement"] = attack
        return body
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient(transform=malicious))
    draft = format_code_intent_markdown(result, task)
    assert "&lt;script&gt;" in draft and "<script>" not in draft
    assert "@everyone" not in draft and "https://evil.invalid" not in draft and "www.evil.invalid" not in draft
    assert draft.count(DEFAULT_B_MARKER) == 1


@pytest.mark.parametrize("mutation", ["sha", "path", "quote", "lines", "evidence", "id", "extra_field"])
def test_stored_forged_citation_or_record_is_rejected_before_linking(mutation):
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient())
    findings = deepcopy(result.findings)
    citation = findings[0]["citations"][0]
    if mutation == "sha":
        citation["sha"] = "e" * 40
    elif mutation == "path":
        citation["path"] = "invented.py"
    elif mutation == "quote":
        citation["quote"] = "invented source"
    elif mutation == "lines":
        citation["end_line"] = 9000
    elif mutation == "evidence":
        citation["evidence_id"] = "made-up"
    elif mutation == "id":
        findings[0]["finding_id"] = "F9"
    else:
        findings[0]["url"] = "https://evil.invalid"
    with pytest.raises(ValueError):
        format_code_intent_markdown(replace(result, findings=findings), task)


def test_input_identity_and_unvalidated_final_records_are_rejected():
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient())
    with pytest.raises(ValueError):
        format_code_intent_markdown(replace(result, head_sha="e" * 40), task)
    with pytest.raises(ValueError):
        format_code_intent_markdown(replace(result, status="failed"), task)


@pytest.mark.parametrize("limit", [1000, 1200])
def test_length_limit_omits_whole_body_not_partial_markdown_and_keeps_warning(limit):
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient())
    draft = format_code_intent_markdown(result, task, max_length=limit)
    assert len(draft) <= limit and "comment_limit_exceeded" in draft
    assert draft.count(DEFAULT_B_MARKER) == 1 and "not executed" in draft and "Mock model" in draft
    assert result.findings


def test_partial_relation_review_is_not_presented_as_scope_sufficient():
    task = make_task()
    result = run_code_first_review(task, split_client(), planning_client=CodeFirstModelClient(split=True),
        limits=CodeFirstReviewLimits(max_rechecks=0))
    draft = format_code_intent_markdown(result, task)
    assert "partial_review" in draft and "scope sufficient: no" in draft and "unreviewed" in draft
    assert "Partial review: unresolved" in draft


def test_not_run_draft_has_no_final_opinion():
    task = make_task()
    result = run_code_first_review(task)
    draft = format_code_intent_markdown(result, task)
    assert "No validated final assessment" in draft and "No validated plan" in draft


@pytest.mark.parametrize("options", [{"marker": "<!-- invented -->"}, {"max_length": True}, {"max_length": 999},
    {"marker": "<!-- review-bot:intent:" + "x" * 1000 + " -->"},
    {"marker": DEFAULT_B_MARKER + "\n<!-- extra -->"}, {"max_length": 60001}, {"publication_mode": "post"}])
def test_formatter_options_cannot_create_extra_markers_or_unbounded_comments(options):
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient())
    with pytest.raises(ValueError):
        format_code_intent_markdown(result, task, **options)


@pytest.mark.parametrize("mutation", ["plan_path", "plan_extra", "group_count", "relation_quote", "intent_extra"])
def test_stored_plan_groups_relations_and_intents_have_fixed_validated_shapes(mutation):
    task = make_task()
    result = run_code_first_review(task, CodeFirstModelClient())
    if mutation == "plan_path":
        plan = deepcopy(result.plan)
        plan["groups"][0]["paths"][0] = "invented.py"
        result = replace(result, plan=plan)
    elif mutation == "plan_extra":
        result = replace(result, plan={**result.plan, "url": "https://evil.invalid"})
    elif mutation == "group_count":
        groups = deepcopy(result.group_observations)
        extra = deepcopy(groups[0])
        extra["group_id"] = "group-2"
        result = replace(result, group_observations=(*groups, extra))
    elif mutation == "relation_quote":
        relations = deepcopy(result.relations)
        relations[0]["citations"][0]["quote"] = "invented source"
        result = replace(result, relations=relations)
    else:
        intents = deepcopy(result.reconstructed_intents)
        intents[0]["declared_goal_id"] = "invented requirement"
        result = replace(result, reconstructed_intents=intents)
    with pytest.raises(ValueError):
        format_code_intent_markdown(result, task)