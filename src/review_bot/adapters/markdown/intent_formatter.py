"""Safe B comment drafts; links are constructed only from validated input evidence."""
from __future__ import annotations

import html
import json
import re
from dataclasses import asdict
from urllib.parse import quote

from review_bot.application.intent_context import build_intent_evidence, decode_intent_task_input
from review_bot.application.declared_intent_review import validate_intent_assessment, validate_intent_goals, validate_intent_observations
from review_bot.domain.envelope_v2 import EnvelopeEvidence
from review_bot.domain.markers import DEFAULT_B_MARKER, valid_intent_marker
from review_bot.domain.intent_models import IntentTaskInput, ReviewBResult

MAX_COMMENT_LENGTH = 60000


def _escape(value: str) -> str:
    text = " ".join(str(value).split())
    text = "".join(char if char.isprintable() else " " for char in text)
    text = html.escape(text, quote=False).replace("@", "@\u200b").replace(":", ":\u200b").replace("www.", "www\u200b.")
    return re.sub(r"([\\`*_\[\]{}()|#+.!-])", r"\\\1", text)


def _wire(value):
    return json.loads(json.dumps(asdict(value), ensure_ascii=False, allow_nan=False))


def _link(item: EnvelopeEvidence, task: IntentTaskInput) -> str:
    label = _escape(f"{item.id} ({item.kind})")
    repository = task.snapshot.repository
    if item.kind in ("source", "test_source"):
        locator = item.locator
        url = f"https://github.com/{repository}/blob/{locator['sha']}/{quote(locator['path'], safe='/')}"
        if locator["begin_line"]:
            url += f"#L{locator['begin_line']}-L{locator['end_line']}"
        return f"[{label}: {_escape(locator['path'])}]({url})"
    if item.kind == "pr_text":
        return f"[{label}](https://github.com/{repository}/pull/{task.snapshot.pr_number})"
    if item.kind == "issue_text":
        return f"[{label}](https://github.com/{repository}/issues/{item.locator['issue_number']})"
    return label


def format_intent_markdown(
    result: ReviewBResult, task_input: IntentTaskInput, *, marker: str = DEFAULT_B_MARKER,
    max_length: int = MAX_COMMENT_LENGTH, publication_mode: str = "artifact_only",
) -> str:
    """Render validated output only. This function neither authorizes nor performs posting."""
    if not valid_intent_marker(marker) or type(max_length) is not int or not 1000 <= max_length <= MAX_COMMENT_LENGTH or publication_mode not in ("artifact_only", "comment"):
        raise ValueError("invalid_intent_formatter_options")
    task = decode_intent_task_input(_wire(task_input))
    if result.schema_version != 1 or result.review_type != "B" or result.input_hash != task.input_hash or (result.base_sha, result.head_sha) != (task.base.sha, task.head.sha):
        raise ValueError("intent_result_input_mismatch")
    if result.status not in ("completed", "not_run", "failed") or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", result.reason):
        raise ValueError("invalid_intent_result_status")
    goals = validate_intent_goals(_wire(result.goals), task) if result.goals is not None else None
    observations = validate_intent_observations(_wire(result.observations), task, goals) if result.observations is not None and goals is not None else None
    final = None
    if result.status == "completed":
        if result.reason != "validated" or goals is None or observations is None:
            raise ValueError("unvalidated_intent_result")
        final = validate_intent_assessment(json.loads(json.dumps({"summary": result.summary,
            "findings": [asdict(item) for item in result.findings], "questions": [asdict(item) for item in result.questions]})), task, goals, observations)
    evidence = {item.id: item for item in build_intent_evidence(task)}

    def links(refs) -> str:
        return ", ".join(_link(evidence[ref], task) for ref in refs)

    heading = "### B · PR intent review" + (" (private evaluation draft)" if publication_mode == "artifact_only" else "")
    publication_notice = (
        "- Publication: not attempted (artifact_only); no automatic approval or request-changes."
        if publication_mode == "artifact_only" else "- Output policy: comment; no automatic approval or request-changes."
    )
    lines = [marker, heading, "",
        f"- Model review: **{result.status}** · `{result.reason}`",
        f"- Comparison base: `{task.base.sha}`", f"- Head: `{task.head.sha}`",
        f"- Text captured: {_escape(task.snapshot.captured_at)}; as of: {_escape(task.snapshot.as_of)}",
        publication_notice, "",
        "**Scope:** full files at the fixed versions below, not the whole repository. Material outside these paths was not reviewed.",
        "**Execution:** target code, builds, tests and hooks were not executed by this bot. Test source is not a successful test run.",
        "**Uncertainty:** schema and citation validation do not establish semantic accuracy or complete PR correctness.", ""]
    if result.model.get("is_mock"):
        lines.extend(["> **Mock model: contract/integration evidence only, not actual model review quality.**", ""])
    for path in task.reviewed_paths:
        lines.append(f"- {_escape(path)}")
    if not task.reviewed_paths:
        lines.append("- No prepared review paths.")
    lines.extend(["", "#### Stage execution"])
    for stage in result.stages:
        lines.append(f"- {_escape(stage.stage)}: {_escape(stage.status)}; review calls: {len(stage.attempts)}; {_escape(stage.reason)}")
    if result.scope.get("missing"):
        lines.extend(["", "Prepared code scope is incomplete; no final opinion is available."])
    if result.scope.get("unavailable_context"):
        lines.extend(["", "Unavailable context:"])
        lines.extend(f"- {_escape(str(item))}" for item in result.scope["unavailable_context"])
    if result.scope.get("test_run_evidence_ids"):
        lines.extend(["", "Supplied execution records were integrity-checked, not independently attested by this bot."])
    if goals is not None:
        lines.extend(["", "#### Declared goals and constraints"])
        if not goals.declared_goals:
            lines.append("No declared goal was extracted; hypotheses are not declared requirements.")
        for item in (*goals.declared_goals, *goals.declared_constraints):
            lines.extend([f"- {_escape(item.statement)}", f"  - Original quote: {_escape(item.quote)}", f"  - Evidence: {links(item.evidence_ids)}"])
    if observations is not None:
        lines.extend(["", "#### Implementation observations"])
        for item in observations.implementation_observations:
            lines.extend([f"- {_escape(item.statement)}", f"  - Evidence: {links(item.evidence_ids)}"])
        lines.extend(["", "#### Hypotheses and assumptions (not confirmed intent)"])
        for item in (*observations.intent_hypotheses, *observations.assumptions):
            lines.append(f"- {_escape(item.statement)}")
            if item.evidence_ids:
                lines.append(f"  - Evidence: {links(item.evidence_ids)}")
            if item.missing_evidence:
                lines.append(f"  - Missing: {'; '.join(_escape(value) for value in item.missing_evidence)}")
    if final is not None:
        lines.extend(["", "#### Assessment", _escape(final.summary)])
        for finding in final.findings:
            lines.extend(["", f"- **{_escape(finding.assessment)}** · {_escape(finding.statement)}",
                f"  - {_escape(finding.reasoning)}", f"  - Evidence: {links(finding.evidence_ids)}"])
            if finding.missing_evidence:
                lines.append(f"  - Missing: {'; '.join(_escape(value) for value in finding.missing_evidence)}")
        if not final.findings:
            lines.append("No validated findings in this scope; this is not approval.")
        for question in final.questions:
            lines.extend(["", f"- Question: {_escape(question.question)}", f"  - Why: {_escape(question.why_needed)}",
                f"  - How the answer changes assessment: {_escape(question.how_answer_changes_assessment)}",
                f"  - Evidence: {links(question.evidence_ids)}"])
    else:
        lines.extend(["", "No validated final assessment. Earlier validated stages, if any, are not a complete review."])
    comment = "\n".join(lines) + "\n"
    if len(comment) > max_length:
        notice = [marker, heading, "",
            f"- Model review: **{result.status}** · `{result.reason}`",
            f"- Comparison base: `{task.base.sha}`", f"- Head: `{task.head.sha}`",
            publication_notice, "",
            "Draft body omitted: comment_limit_exceeded. Complete stage/result/scope data remains in private B/ files.",
            "Material outside the prepared paths was not reviewed; this is not a whole-repository assessment.",
            "Target code, builds, tests and hooks were not executed by this bot.",
            "Validation is not semantic accuracy, approval, or complete PR correctness."]
        if result.model.get("is_mock"):
            notice.append("Mock model: contract/integration evidence only, not actual model review quality.")
        return "\n".join(notice) + "\n"
    return comment