"""Inert code-first B drafts; only fixed input originals can produce links."""
from __future__ import annotations

from dataclasses import asdict, replace

from review_bot.adapters.markdown.intent_formatter import MAX_COMMENT_LENGTH, _escape, _link
from review_bot.application.intent_context import build_intent_evidence, decode_intent_task_input
from review_bot.application.code_first_intent_review import validate_code_first_result
from review_bot.domain.markers import DEFAULT_B_MARKER, valid_intent_marker
from review_bot.domain.intent_models import IntentTaskInput
from review_bot.domain.code_first_intent_models import CodeFirstReviewResult


def format_code_intent_markdown(
    result: CodeFirstReviewResult, task_input: IntentTaskInput, *, marker: str = DEFAULT_B_MARKER,
    max_length: int = MAX_COMMENT_LENGTH, publication_mode: str = "artifact_only",
) -> str:
    """Render a validated private draft or comment, without authorizing publication."""
    if not valid_intent_marker(marker) or len(marker) > 128 or not marker.startswith("<!-- review-bot:intent:") or type(max_length) is not int or not 1000 <= max_length <= MAX_COMMENT_LENGTH or publication_mode not in ("artifact_only", "comment"):
        raise ValueError("invalid_code_intent_formatter_options")
    task = decode_intent_task_input(asdict(task_input))
    validate_code_first_result(result, task)
    evidence = {item.id: item for item in build_intent_evidence(task)}

    def citations(rows: list[dict]) -> str:
        links = []
        for row in rows:
            original = evidence[row["evidence_id"]]
            if original.kind in ("source", "test_source"):
                original = replace(original, locator={**original.locator,
                    "begin_line": row["start_line"], "end_line": row["end_line"]})
            link = _link(original, task)
            if link not in links:
                links.append(link)
        return ", ".join(links)

    heading = "### B · Code-first intent review" + (" (private draft)" if publication_mode == "artifact_only" else "")
    notice = "Publication: not attempted (artifact_only)." if publication_mode == "artifact_only" else "Output policy: comment; publication is a separate operation."
    base = [marker, heading, "", f"- Model review: **{result.status}** · `{result.reason}`",
        f"- Base: `{task.base.sha}`; head: `{task.head.sha}`", f"- {notice}",
        "- Scope: fixed full files, not the whole repository; code-first, not declared-goal validation.",
        "- Execution: target code, builds, tests and hooks were not executed. Test source is not a test run.",
        "- Uncertainty: purpose is reconstructed, not confirmed developer intent. Validation is not semantic accuracy or approval.",
        "- Relations are not exhaustive; no guarantee of complete correctness."]
    if result.model.get("is_mock"):
        base.append("- Mock model: contract evidence only, not actual model review quality.")
    lines = [*base, "", "#### Scope and plan"]
    lines.extend(f"- {_escape(path)}" for path in task.reviewed_paths)
    if result.plan:
        lines.append(f"- Plan: {_escape(result.plan['source'])}; {_escape(result.plan['status'])}; {_escape(result.plan['reason'])}.")
        for group in result.plan["groups"]:
            lines.append(f"- {_escape(group['group_id'])}: {', '.join(_escape(path) for path in group['paths'])}")
        lines.extend(f"- Missing: {_escape(value)}" for value in result.plan["missing_context"])
    else:
        lines.append("- No validated plan.")
    lines.extend(["", "#### Observations and purpose hypotheses"])
    for group in result.group_observations:
        for row in group["observations"]:
            lines.append(f"- {_escape(row['observation_id'])}: {_escape(row['statement'])} · {citations(row['citations'])}")
    for row in result.reconstructed_intents:
        lines.append(f"- {_escape(row['intent_id'])} (hypothesis): {_escape(row['statement'])} · {citations(row['citations'])}")
        lines.append(f"  - Uncertainty: {_escape(row['uncertainty'])}; alternatives: {'; '.join(_escape(value) for value in row['alternatives'])}")
        lines.extend(f"  - Missing: {_escape(value)}" for value in row["missing_context"])
    lines.extend(["", "#### Relations"])
    for row in result.relations:
        lines.append(f"- {_escape(row['relation_id'])}: **{_escape(row['status'])}** · {_escape(row['from_path'])} → {_escape(row['to_path'])}: {_escape(row['contract'])}")
        if row["citations"]:
            lines.append(f"  - Evidence: {citations(row['citations'])}")
        lines.extend(f"  - Missing: {_escape(value)}" for value in row["missing_context"])
    if not result.relations:
        lines.append("- No validated relations recorded; this does not establish their absence.")
    lines.extend(["", "#### Final assessment"])
    if result.status in ("completed", "partial_review"):
        lines.append(_escape(result.summary))
        lines.append(f"- Assessment completed: yes; scope sufficient: {'yes' if result.scope['review_sufficient'] else 'no'}; no findings: {'yes' if result.scope['no_findings'] else 'no'}.")
        for row in result.findings:
            lines.extend([f"- {_escape(row['finding_id'])}: **{_escape(row['assessment'])}** · {_escape(row['statement'])}",
                f"  - {_escape(row['reasoning'])}", f"  - Evidence: {citations(row['citations'])}"])
            if row["failure_condition"]:
                lines.append(f"  - Failure condition: {_escape(row['failure_condition'])}")
            lines.extend(f"  - Missing: {_escape(value)}" for value in row["missing_context"])
        for row in result.questions:
            lines.extend([f"- {_escape(row['question_id'])}: {_escape(row['question'])}",
                f"  - Why: {_escape(row['why_needed'])}; effect: {_escape(row['how_answer_changes_assessment'])}",
                f"  - Evidence: {citations(row['citations'])}"])
        if result.status == "partial_review":
            lines.append("Partial review: unresolved important relations/context prevent a complete-review claim.")
    else:
        lines.append("No validated final assessment. Earlier observations are not a complete review.")
    lines.extend(f"- Missing context: {_escape(value)}" for value in result.scope.get("missing_context_requests", []))
    lines.extend(["", "#### Usage and stages",
        f"- Review calls: {_escape(str(result.usage.get('calls')))}; model calls including metering: {_escape(str(result.usage.get('model_calls')))}.",
        f"- Tokens: input {_escape(str(result.usage.get('input_tokens')))}, output {_escape(str(result.usage.get('output_tokens')))}; usage {'complete' if result.usage.get('usage_complete') else 'unknown/partial'}."])
    lines.extend(f"- {_escape(stage.stage)}: {_escape(stage.status)}; {_escape(stage.reason)}" for stage in result.stages)
    comment = "\n".join(lines) + "\n"
    if len(comment) <= max_length:
        return comment
    compact = [marker, heading, "", f"- Model review: **{result.status}** · `{result.reason}`",
        f"- Base: `{task.base.sha}`; head: `{task.head.sha}`", f"- {notice}",
        "Code-first, prepared scope only, not the whole repository. Target code, tests and builds were not executed.",
        "Purpose hypotheses remain uncertain; validation is not correctness or approval.",
        "Body omitted: comment_limit_exceeded. Complete records remain in private artifacts.",
        "No complete-review or problem-free claim is made by this shortened draft."]
    if result.model.get("is_mock"):
        compact.append("Mock model: contract evidence only, not actual model review quality.")
    return "\n".join(compact) + "\n"