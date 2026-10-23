"""A grounded, purpose-first document for one intent review."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Literal

from review_bot.adapters.markdown.intent_formatter import MAX_COMMENT_LENGTH, _escape, _link
from review_bot.application.intent_context import build_intent_evidence, decode_intent_task_input
from review_bot.application.intent_perspective_review import validate_intent_perspective_result
from review_bot.domain.intent_models import IntentTaskInput
from review_bot.domain.intent_perspectives import PERSPECTIVES, IntentPerspectiveResult
from review_bot.domain.markers import DEFAULT_B_MARKER, valid_intent_marker


def format_intent_perspective_markdown(result: IntentPerspectiveResult, task_input: IntentTaskInput, *,
    marker: str = DEFAULT_B_MARKER, max_length: int = MAX_COMMENT_LENGTH,
    publication_mode: Literal["artifact_only", "comment"] = "artifact_only",
) -> str:
    if (not valid_intent_marker(marker) or len(marker) > 128 or type(max_length) is not int
        or not 1000 <= max_length <= MAX_COMMENT_LENGTH or publication_mode not in ("artifact_only", "comment")):
        raise ValueError("invalid_intent_perspective_formatter_options")
    task = decode_intent_task_input(asdict(task_input))
    validate_intent_perspective_result(result, task)
    if publication_mode == "comment" and result.status != "completed":
        raise ValueError("incomplete_intent_perspective_comment")
    evidence = {item.id: item for item in build_intent_evidence(task)}
    ko = result.language == "ko"
    names = {item.id: item.name_ko if ko else item.name_en for item in PERSPECTIVES}
    labels = ("파악한 개발 의도", "구현과의 연결", "달성 가능성 판단", "확인 질문") if ko else (
        "Understood development intent", "Connection to implementation", "Achievability assessment", "Questions")

    def links(citations):
        values = []
        for cite in citations:
            item = evidence[cite["evidence_id"]]
            if item.kind in ("source", "test_source"):
                item = replace(item, locator={**item.locator, "begin_line": cite["start_line"], "end_line": cite["end_line"]})
            link = _link(item, task)
            if link not in values:
                values.append(link)
        return ", ".join(values)

    private = publication_mode == "artifact_only"
    title = ("### 개발 의도 리뷰 · 비공개 결과" if ko else "### Development intent review · private result") if private else (
        "### 개발 의도 리뷰" if ko else "### Development intent review")
    status = {"completed": "검토 완료", "partial_review": "부분 검토", "failed": "실행 실패", "not_run": "미실행"}.get(result.status) if ko else result.status
    scope = ("- 검토 대상은 고정된 PR 입력과 제공된 원문입니다. 대상 코드·빌드·테스트는 실행하지 않았습니다." if ko else
        "- Scope: the fixed PR input and supplied originals. Target code, builds, and tests were not executed.")
    if private:
        scope += " 외부 게시는 시도하지 않았습니다." if ko else " Publication was not attempted."
    intro = [marker, title, "", f"- {status} · {_escape(result.reason)}",
        f"- base: {task.base.sha}; head: {task.head.sha}", scope]
    if result.model.get("is_mock"):
        intro.append("- 모의 모델을 사용한 형식 검증 결과입니다. 실제 모델의 리뷰 품질을 입증하지 않습니다." if ko else
            "- Mock model: contract verification only, not evidence of actual model review quality.")
    lines = [*intro]
    if result.narrative:
        for label, key in zip(labels[:3], ("intent", "implementation", "assessment")):
            passage = result.narrative[key]
            lines.extend(["", f"#### {label}", _escape(passage["statement"]), links(passage["citations"])])
            if key == "assessment":
                for finding in result.findings:
                    lines.extend([f"- {_escape(finding['statement'])}", f"  - {_escape(finding['reasoning'])}",
                        f"  - {_escape(finding['failure_condition'])}", f"  - {links(finding['citations'])}"])
        lines.extend(["", f"#### {labels[3]}"])
        for question in result.questions:
            lines.extend([f"- {_escape(question['question'])}",
                f"  - {_escape(question['why_needed'])}",
                f"  - {_escape(question['how_answer_changes_assessment'])}",
                f"  - {links(question['citations'])}"])
        if not result.questions:
            lines.append("추가 확인 질문은 없습니다." if ko else "No additional questions were recorded.")
        blocking = set(result.scope["unresolved_context_ids"])
        for gap in result.context_gaps:
            if gap["context_id"] in blocking:
                views = ", ".join(names[key] for key in gap["perspective_ids"])
                lines.append(f"- {_escape(views)}: {_escape(gap['description'])}")
        if result.status == "partial_review":
            lines.append("필요한 정보가 부족한 관점은 판단을 유보했습니다. 확인 가능한 결과는 위에 보존했습니다." if ko else
                "Judgement is deferred where necessary information remains missing; supported results are preserved above.")
    else:
        lines.extend(["", "완료로 제시할 수 있는 최종 리뷰 결과가 없습니다. 이전에 검증한 선택·분석 기록은 비공개 산출물에 보존됩니다." if ko else
            "No final review can be presented as complete. Previously validated selection and analysis records remain in the private artifacts."])
    text = "\n".join(lines) + "\n"
    if len(text) <= max_length:
        return text
    if not private:
        raise ValueError("intent_perspective_comment_too_large")
    return "\n".join([*intro, "", "출력 길이 제한으로 본문을 생략했습니다. 전체 검토 결과는 비공개 산출물을 확인하세요." if ko else
        "Body omitted because of the output limit. The complete review remains in the private artifacts."]) + "\n"
