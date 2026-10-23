"""Validated partition artifacts and developer-facing PR comment drafts."""
from __future__ import annotations

from review_bot.adapters.markdown.intent_formatter import _escape
from review_bot.application.partition_review import validate_partition_result
from review_bot.domain.partition_review import PartitionReviewResult


def _format_private_partition_markdown(result: PartitionReviewResult, task, **_options) -> str:
    if _options.get("publication_mode", "artifact_only") != "artifact_only":
        raise ValueError("partition_evaluation_is_private_only")
    validate_partition_result(result, task)
    ko = result.language == "ko"
    name = ("의도" if ko else "Intent") if result.review_type == "B" else "PMD/CPD"
    lines = [f"### {name} 리뷰 · 분할 평가 · 비공개" if ko else f"### {name} review · partition trial · private", "",
        f"- {_escape(result.status)} / {_escape(result.reason)}",
        f"- base: {result.base_sha}; head: {result.head_sha}",
        "- 운영자가 지정한 모드의 평가입니다. 실제 PR 전체의 자동 라우팅 전환을 의미하지 않습니다." if ko else
        "- Operator-selected mode evaluation, not a replacement of PR-wide automatic routing.",
        "- 고정 원문을 정적 검토했습니다. 대상 코드·빌드·테스트 실행 및 외부 발행은 하지 않았습니다." if ko else
        "- Fixed originals only. No target code, build, test execution or external publication.",
        f"- 원문 {result.scope['originals']}개 · 구간 {result.scope['units']}개" if ko else
        f"- {result.scope['originals']} originals · {result.scope['units']} spans", ""]
    if result.model.get("is_mock"):
        lines.append("- 모의 모델을 사용한 실행 경로 검증입니다. 실제 리뷰 품질 측정 결과가 아닙니다." if ko else
            "- Mock-model contract verification, not a measurement of actual review quality.")
    if result.partition.get("shared_plan_id"):
        lines[4] = ("- PR 전체 관측 후 선택한 공통 실행 계획을 재사용한 리뷰입니다." if ko else
            "- Reuses the common execution plan selected after PR-wide observations.")
        lines.append(f"- plan: {result.partition['shared_plan_id']}")
        if result.partition.get("amended_plan_id"):
            lines.append(f"- amended_plan: {result.partition['amended_plan_id']}")
    if result.partition.get("observation_tracking_version") == 1:
        lines.append("- 관측별 유지·통합·제외 내역과 인용 근거를 추적합니다. 의미가 모두 보존됐다는 보장은 아닙니다." if ko else
            "- Observation retention/merging/exclusion and citations are tracked; semantic preservation is not guaranteed.")
        excluded = result.partition.get("excluded_observations", [])
        if excluded:
            lines.extend(["", "#### 제외·정정한 초기 관측" if ko else "#### Excluded or corrected provisional observations"])
            for row in excluded:
                lines.append(f"- {_escape(row['stage'])} · {_escape(row['observation_id'])}: {_escape(row['reason'])}")
    if result.summary:
        lines.append(_escape(result.summary))
    if result.partition.get("fact_verification"):
        verification = result.partition["fact_verification"]
        lines.extend(["", "#### 사실 검토" if ko else "#### Factual review",
            "- 코드 판단은 모델이 원문과 대조한 결과이며, 정확성을 증명하지는 않습니다." if ko else
            "- Code claims were checked by a model against originals; this is not a proof of accuracy."])
        for entry in verification["items"]:
            corrected = len(entry["attempts"]) > 1
            citation_only = (corrected and any("citation_repair" in attempt for attempt in entry["attempts"])
                and not any("rewrite" in attempt for attempt in entry["attempts"]))
            correction = ((" · 인용 보강 후 재검토" if ko else " · citations supplemented and rechecked")
                if citation_only else (" · 재작성 후 재검토" if ko else " · rewritten and rechecked") if corrected else "")
            lines.append(f"- {_escape(entry['item_id'])}: {_escape(entry['status'])}" +
                correction)
        if verification["status"] != "model_supported":
            lines.append("- 미검증 항목이 남아 있으므로 완료된 리뷰가 아닙니다." if ko else
                "- Unverified items remain; this is not a completed review.")
    for item in (*result.perspectives, *result.findings):
        label = item.get("perspective_id", "발견사항" if ko else "Finding")
        lines.extend(["", f"#### {_escape(label)}"])
        if item.get("assessments"):
            lines.append("통합 요청별 판단을 모두 보존합니다. 하나의 전역 판정으로 합치지 않았습니다." if ko else
                "Original per-request judgements are preserved; no single global verdict is asserted.")
        for assessment in item.get("assessments", [item]):
            lines.extend(["", f"**{_escape(assessment['verdict'])}**", _escape(assessment["statement"]),
                _escape(assessment["reasoning"])])
            if assessment.get("failure_condition"):
                lines.append(_escape(assessment["failure_condition"]))
            for citation in assessment["citations"]:
                location = citation["path"] or citation["evidence_id"]
                lines.append(f"- {_escape(location)} · {_escape(citation['version'])} · "
                    f"{citation['begin_line']}–{citation['end_line']}행 · "
                    f"UTF-8 [{citation['start_byte']}, {citation['end_byte']})")
    if result.questions:
        lines.extend(["", "#### 확인 질문" if ko else "#### Questions", *[f"- {_escape(q)}" for q in result.questions]])
    if result.status != "completed":
        lines.extend(["", "완료된 리뷰로 제시하지 않습니다. 검토 범위·관측·실행 기록을 함께 확인하세요." if ko else
            "No completed review is claimed. Inspect scope, observations and stage receipts."])
    return "\n".join(lines) + "\n"


def format_partition_review_markdown(result: PartitionReviewResult, task, *,
    publication_mode: str = "artifact_only", marker: str | None = None, max_length: int = 60000,
    presentation=None, comment_template: str | None = None, **options,
) -> str:
    """Render a validated artifact or a developer comment; never publish it."""
    if publication_mode == "artifact_only":
        return _format_private_partition_markdown(result, task, **options)
    if publication_mode != "comment" or result.review_type not in ("A", "B"):
        raise ValueError("invalid_partition_formatter_options")
    from review_bot.adapters.markdown.developer_presentation import (
        actionable_intent_findings, findings_section, finish, intent_lead, label, perspective_section, review_summary, walkthrough,
        selected_signals, static_analysis, status_sections,
    )
    from review_bot.domain.markers import DEFAULT_B_MARKER
    from review_bot.adapters.markdown.formatter import DEFAULT_A_MARKER

    validate_partition_result(result, task)
    architecture = result.review_type == "A"
    title = label(presentation, "title_a" if architecture else "title_b", "PMD/CPD 아키텍처 리뷰 A" if architecture else "의도 리뷰 B")
    marker = marker or (DEFAULT_A_MARKER if architecture else DEFAULT_B_MARKER)
    if result.status != "completed" or result.validation_errors:
        final = result.partition.get("final", {})
        unreviewed = list(final.get("missing_context", ())) + list(result.validation_errors)
        return finish(marker, title, status_sections(result, ko=result.language == "ko", unreviewed=unreviewed),
            scope="", max_length=max_length)
    summary = review_summary(result.summary, task, is_mock=result.model.get("is_mock"))
    sections = {"summary": summary}
    if architecture:
        target_reviews = result.partition.get("final", {}).get("target_reviews", {})
        signals = selected_signals(task, result.findings)
        walk = walkthrough(task, result.findings, result.questions, signals=signals)
        if walk:
            sections["summary"] = summary + "\n\n" + walk
        target_reviews = result.partition.get("final", {}).get("target_reviews", {})
        signals = selected_signals(task, result.findings)
        sections["static_analysis"] = static_analysis(task, signals, presentation=presentation)
        sections["llm_review"] = findings_section(result.findings, result.questions, task,
            signals=signals, target_reviews=target_reviews, presentation=presentation)
    else:
        findings = actionable_intent_findings(result.findings, result.perspectives)
        lead = intent_lead(result.perspectives)
        if lead:
            summary = lead + "\n\n" + summary
        walk = walkthrough(task, findings, result.questions)
        if walk:
            sections["summary"] = summary + "\n\n" + walk
        sections["llm_review"] = findings_section(findings, result.questions, task, presentation=presentation)
        sections["perspectives"] = perspective_section(result.perspectives, task, presentation)
    footer = label(presentation, "footer_notice", "")
    return finish(marker, title, sections, scope="", template=comment_template, max_length=max_length, footer=footer)
