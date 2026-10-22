"""Developer comment sections, adapted from seulgi-server's PMD/CPD report layout.

The source template's summary table, per-file rule details, and CPD locations
are retained for selected review signals. Stored model conclusions are never
edited or replaced by metric-derived defect claims.
"""
from __future__ import annotations

import html
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass

from review_bot.domain.intent_perspectives import PERSPECTIVES


def row(value):
    return asdict(value) if is_dataclass(value) else value


def escape(value) -> str:
    text = html.escape(str(value), quote=False).replace("@", "@\u200b")
    return re.sub(r"([\\`*_\[\]{}()|#+.!-])", r"\\\1", text).replace("\r", "").replace("\n", "<br>")


def label(presentation, name: str, fallback: str) -> str:
    value = (presentation or {}).get(name, fallback)
    if not isinstance(value, str) or len(value) > 1000:
        raise ValueError("invalid_comment_presentation")
    return escape(value)


def measurement_text(measurements) -> str:
    def value(number):
        return "보고값 없음" if number is None else str(number)
    return "; ".join(f"{m.name}: {value(m.base)} → {value(m.head)}" for m in measurements)


def details(title: str, body: str) -> str:
    # Markdown escapes are literal inside an HTML summary element.
    title = re.sub(r"\\([\\`*_\[\]{}()|#+.!-])", r"\1", title)
    return f"<details>\n<summary>{title}</summary>\n\n{body}\n\n</details>"


def compose(sections: Mapping[str, str], template: str | None = None) -> str:
    """Substitute declared whole sections once; template prose is inert text."""
    if template is None:
        return "\n\n".join(value for value in sections.values() if value)
    names = re.findall(r"\{\{([a-z_]+)\}\}", template)
    remainder = re.sub(r"\{\{[a-z_]+\}\}", "", template)
    if (len(template.encode("utf-8")) > 8192 or set(names) != set(sections)
            or len(names) != len(sections) or "{" in remainder or "}" in remainder):
        raise ValueError("invalid_comment_template")
    pieces = re.split(r"(\{\{[a-z_]+\}\})", template)
    return "".join(sections[piece[2:-2]] if piece.startswith("{{") else
        "\n".join(escape(line) for line in piece.split("\n")) for piece in pieces)


def finish(marker: str, title: str, sections: Mapping[str, str], *, scope: str,
    template: str | None = None, max_length: int = 60000, footer: str = "") -> str:
    if type(max_length) is not int or not 1000 <= max_length <= 60000:
        raise ValueError("invalid_comment_length")
    intro = f"{marker}\n### {title}\n\n"
    rendered = intro + compose(sections, template) + ("\n\n" + scope if scope else "")
    if footer:
        rendered += "\n\n" + footer
    if len(rendered) + 1 > max_length:
        # Never publish a shortened subset of conclusions as a complete review.
        rendered = (intro + "> 리뷰 본문 표시 보류: 댓글 길이 한도를 초과했습니다.\n\n본문 전체를 싣지 못했습니다."
            + ("\n\n" + scope if scope else ""))
    if len(rendered) + 1 > max_length:
        raise ValueError("comment_notice_too_large")
    return rendered + "\n"


_STATUS_REASONS = {
    "factual_verification_incomplete": ("코드와 근거를 대조하는 확인이 끝나지 않았습니다.",
        "The code and supporting evidence have not been fully checked."),
    "unresolved_context": ("판단에 필요한 정보가 아직 부족합니다.", "Information needed for the assessment is still missing."),
    "context_missing": ("판단에 필요한 정보가 아직 부족합니다.", "Information needed for the assessment is still missing."),
    "call_limit_exceeded": ("검토 요청 횟수 한도에 도달해 확인을 마치지 못했습니다.",
        "The review request limit was reached before all checks finished."),
    "task_token_limit_exceeded": ("검토 처리량 한도에 도달해 확인을 마치지 못했습니다.",
        "The review capacity limit was reached before all checks finished."),
    "task_deadline_exceeded": ("검토 제한 시간이 지나 확인을 마치지 못했습니다.",
        "The review time limit was reached before all checks finished."),
    "input_limit_exceeded": ("검토 자료가 한 번에 처리할 수 있는 범위를 초과했습니다.",
        "The review material exceeds the available input capacity."),
    "model_output_limit": ("리뷰 응답이 길이 한도에 걸려 완전한 결과를 받지 못했습니다.",
        "The review response reached its length limit before a complete result was received."),
    "usage_unavailable": ("모델 응답을 끝까지 확인하지 못해 검토를 계속하지 못했습니다.",
        "The review could not continue because the model response could not be confirmed."),
    "model_cancelled": ("모델 요청이 취소되어 검토를 마치지 못했습니다.",
        "The model request was cancelled before the review finished."),
    "model_timeout": ("모델 응답을 제한 시간 안에 받지 못했습니다.",
        "The model did not respond within the time limit."),
    "model_not_configured": ("리뷰 모델이 설정되지 않아 검토를 시작하지 못했습니다.",
        "The review could not start because no review model is configured."),
    "invalid_partition_response_or_input": ("리뷰 자료를 읽는 과정에서 문제가 발견되었습니다.",
        "A problem was found while reading the review material."),
    "invalid_partition_options": ("리뷰를 시작할 조건이 맞지 않아 중단했습니다.",
        "The review stopped because the conditions to start it were not met."),
    "stage_gate_rejected": ("다음 검토 단계에 필요한 확인을 통과하지 못했습니다.",
        "The checks required for the next review stage did not pass."),
}


def status_sections(result, *, ko=True, unreviewed=()) -> dict[str, str]:
    reason = _STATUS_REASONS.get(result.reason)
    prefix = "partition_planning_failed:"
    if reason is None and isinstance(result.reason, str) and result.reason.startswith(prefix):
        known = _STATUS_REASONS.get(result.reason[len(prefix):])
        if known is not None:
            reason = ("초기 리뷰 계획을 확정하지 못했습니다. " + known[0],
                "The initial review plan could not be established. " + known[1])
    if reason is None:
        # Pipeline error text may include provider responses or internal details.
        reason = ("리뷰를 완료하지 못했습니다. 새로운 커밋이 올라오면 다시 검토합니다.",
            "The review could not be completed. It will be reviewed again when a new commit is pushed.")
    explanation = reason[0 if ko else 1]
    body = (f"> **리뷰 미완료** · {explanation}\n\n최종 판단을 게시하지 않습니다." if ko else
        f"> **Review incomplete** · {explanation}\n\nNo final assessment is available.")
    if unreviewed:
        body += ("\n\n미검토·미확정 범위: " if ko else "\n\nUnreviewed or unresolved scope: ") + "; ".join(escape(v) for v in unreviewed)
    return {"status": body}


def review_summary(summary: str, task, *, is_mock=False) -> str:
    purpose = ""
    if hasattr(task.snapshot, "pr"):
        purpose = next((e.content.splitlines()[0] for e in task.evidence
            if e.evidence_id.startswith("E-PR") and e.content.strip()), "")
    else:
        for evidence in task.snapshot.evidence:
            if evidence.kind == "pr_text" and isinstance(evidence.content, dict):
                purpose = evidence.content.get("title", "")
                break
    body = "#### 전체 판단\n\n" + escape(summary)
    if purpose:
        body = "PR 목적: " + escape(purpose) + "\n\n" + body
    if is_mock:
        body = "> 모의 모델 결과입니다. 실제 리뷰 품질을 확인한 결과가 아닙니다.\n\n" + body
    return body


def actionable_intent_findings(findings, perspectives):
    """Promote adopted concern judgements, while retaining all details below."""
    values = list(findings)
    statements = {row(value).get("statement") for value in values}
    for perspective in perspectives:
        for assessment in perspective.get("assessments", [perspective]):
            if assessment["verdict"] in ("concern", "conditional_risk", "improvement") and assessment["statement"] not in statements:
                values.append(assessment)
                statements.add(assessment["statement"])
    return values


def selected_signals(task, findings=(), target_reviews=None):
    changed = set(task.snapshot.pr.changed_files)
    linked = {identifier for finding in findings for identifier in row(finding).get("target_ids", [row(finding).get("signal_id")])}
    reviewed = set(target_reviews or {})
    selected = []
    for selected_signal in task.signals:
        signal = selected_signal.signal
        if not changed.intersection(signal.paths):
            continue
        if (signal.signal_id in linked or signal.signal_id in reviewed) and (
                str(signal.change) in ("new", "worsened", "existing") or signal.signal_id in linked):
            selected.append(signal)
    return selected


def static_analysis(task, signals, *, presentation=None) -> str:
    from review_bot.adapters.markdown.formatter import _a_link

    repository = task.snapshot.pr.repository
    changed = set(task.snapshot.pr.changed_files)
    affected = {p for signal in signals for p in signal.paths if p in changed}
    counts = {name: sum(signal.change == name for signal in signals) for name in ("new", "worsened", "existing")}
    cpd = [signal for signal in signals if signal.tool == "cpd"]
    lines = ["#### " + label(presentation, "static_section", "정적 분석 · 선택한 PMD/CPD 신호"), "",
        "| 측정 영역 | 이번 PR 검토 범위 |", "|---|---:|",
        f"| 신호가 잡힌 변경 파일 | {len(affected)} / {len(changed)}개 |",
        f"| 검토 대상으로 선택한 신호 | {len(signals)}건 |",
        f"| 신규 / 악화 / 변경과 관련된 기존 신호 | {counts['new']} / {counts['worsened']} / {counts['existing']}건 |",
        f"| 선택한 CPD 중복 블록 | {sum(len(s.head_findings) for s in cpd)}개 |", "",
        "> 복잡도·중복 측정은 코드 탐색의 출발점입니다. 수치만으로 실행 중 결함이나 공통화 필요성을 확정하지 않습니다."]
    if not task.delta.comparable:
        lines.extend(["", "> 분석 비교가 불완전하거나 조건이 다릅니다. 표시할 신호가 없더라도 분석 0건으로 확정하지 않습니다."])
    grouped = defaultdict(list)
    for signal in signals:
        if signal.tool == "cpd":
            continue
        for path in signal.paths:
            if path in changed:
                grouped[path].append(signal)
    for path, file_signals in sorted(grouped.items()):
        rows = ["| 라인 | 신호·변화 | 측정·메시지 |", "|---|---|---|"]
        for signal in file_signals:
            metrics = measurement_text(signal.measurements)
            for finding in signal.head_findings:
                if hasattr(finding, "occurrences"):
                    continue
                if finding.path == path:
                    location = _a_link(repository, task.snapshot.head.sha, path, finding.begin_line, finding.end_line)
                    anchor = f"signal-{signal.signal_id}"
                    rows.append(f'| {location} | <a id="{html.escape(anchor, quote=True)}"></a>{escape(finding.rule)} · {escape(signal.change)} | {escape(metrics)}<br>{escape(finding.message)} |')
        lines.extend(["", details(escape(path) + f" — {len(file_signals)}건", "\n".join(rows))])
    if cpd:
        rows = ["| 줄수 / 토큰 | 변화 | 중복된 위치 |", "|---:|---|---|"]
        for signal in cpd:
            for duplication in signal.head_findings:
                locations = ", ".join(_a_link(repository, task.snapshot.head.sha, loc.path, loc.begin_line, loc.end_line)
                    for loc in duplication.occurrences)
                rows.append(f'| {duplication.lines} / {duplication.tokens} | <a id="signal-{html.escape(signal.signal_id, quote=True)}"></a>{escape(signal.change)} | {locations} |')
        lines.extend(["", details("CPD · 변경과 관련된 중복 위치", "\n".join(rows))])
    return "\n".join(lines)


def citation_links(citations, task) -> str:
    from review_bot.adapters.markdown.formatter import _a_link

    repository = task.snapshot.pr.repository if hasattr(task, "snapshot") and hasattr(task.snapshot, "pr") else task.snapshot.repository
    links = []
    for raw in citations:
        cite = row(raw)
        # Diff line numbers are not source line numbers, so never fabricate blob links.
        if cite.get("path") and cite.get("version") in ("base", "head"):
            link = _a_link(repository, cite.get("sha", ""), cite["path"], cite["begin_line"], cite["end_line"])
        elif not cite.get("path") and cite.get("evidence_id", "").startswith("E-PR"):
            number = task.snapshot.pr.number if hasattr(task.snapshot, "pr") else task.snapshot.pr_number
            link = f"[PR 설명](https://github.com/{repository}/pull/{number})"
        else:
            continue
        if link not in links:
            links.append(link)
    return " · ".join(links)


def quotes(citations, task) -> str:
    blocks = []
    for raw in citations:
        cite = row(raw)
        if not cite.get("quote"):
            continue
        preview = html.escape(cite["quote"], quote=False).replace("@", "@\u200b")
        preview = preview.translate({ord(c): f"&#{ord(c)};" for c in "`*_[]{}|#+"})
        blocks.append(citation_links([cite], task) + "\n\n<pre><code>" + preview + "</code></pre>")
    return details("인용 원문", "\n\n".join(blocks)) if blocks else ""


def judgement_label(finding) -> str:
    verdict = finding.get("verdict", finding.get("judgement", ""))
    if verdict in ("concern", "confirmed_issue"):
        return "확인된 문제"
    if verdict in ("insufficient_context", "insufficient_evidence", "author_question"):
        return "작성자 확인 질문"
    if verdict == "improvement":
        return "구체적 개선 제안"
    if verdict == "conditional_risk":
        return "조건부 위험"
    if finding.get("failure_condition"):
        return "조건부 위험"
    if finding.get("basis") == "analysis_signal":
        return "구조 검토 의견"
    return escape(verdict)


def _severity_rank(finding) -> int:
    """CodeRabbit-style triage order: blocking problems first, questions last."""
    verdict = finding.get("verdict", finding.get("judgement", ""))
    if verdict in ("confirmed_issue", "concern"):
        return 0
    if verdict == "conditional_risk" or finding.get("failure_condition"):
        return 1
    if verdict == "improvement":
        return 2
    return 3


def walkthrough(task, findings, questions=(), *, signals=()) -> str:
    """Deterministic file-by-file table: which changed files drew findings.

    No model prose; counts come from validated citations and selected signals.
    Mirrors CodeRabbit's walkthrough at comment scale: one collapsed block.
    """
    if hasattr(task, "reviewed_paths") and not hasattr(task.snapshot, "pr"):
        files = list(task.reviewed_paths)
    else:
        try:
            files = list(task.snapshot.pr.changed_files)
        except AttributeError:
            files = []
    if not files:
        return ""
    counts: dict[str, int] = {path: 0 for path in files}
    for item in (*findings, *questions):
        for raw in row(item).get("citations", ()):
            path = row(raw).get("path")
            if path in counts:
                counts[path] += 1
    flagged = set()
    for signal in signals:
        for path in getattr(signal, "paths", ()):
            if path in counts:
                flagged.add(path)
    lines = ["| 파일 | 지적 | 정적 신호 |", "|---|---|---|"]
    for path in files:
        mark = "없음"
        if path in flagged:
            mark = "있음"
        lines.append(f"| {escape(path)} | {counts[path]}건 | {mark} |")
    return details("변경 파일 둘러보기", "\n".join(lines))


def findings_section(findings, questions, task, *, signals=(), target_reviews=None, presentation=None) -> str:
    from review_bot.adapters.markdown.formatter import _a_link

    lines = ["#### " + label(presentation, "llm_section", "LLM 리뷰"), ""]
    actionable = [row(item) for item in findings if row(item).get("judgement", row(item).get("verdict")) != "acceptable_with_evidence"]
    actionable.sort(key=_severity_rank)
    if not actionable and not questions:
        lines.append("이 검토 범위에서 추가로 조치할 발견사항은 없습니다.")
    by_id = {signal.signal_id: signal for signal in signals}
    for index, finding in enumerate(actionable, 1):
        statement = finding.get("title") or finding.get("statement", "")
        citations = finding.get("citations", ())
        lines.extend([f"##### {index}. {judgement_label(finding)} · {escape(statement)}", "",
            escape(finding.get("reason", finding.get("reasoning", "")))])
        if finding.get("failure_condition"):
            lines.extend(["", "성립 조건: " + escape(finding["failure_condition"])])
        if finding.get("recommendation") and finding.get("judgement") != "insufficient_evidence":
            lines.extend(["", "제안: " + escape(finding["recommendation"])])
        if finding.get("basis"):
            basis = {"criterion": "명시한 PR 기준", "explicit_criterion": "명시한 PR 기준",
                "observed_structure": "코드 구조", "analysis_signal": "정적 분석 신호", "insufficient_evidence": "근거 부족"}.get(finding["basis"], finding["basis"])
            lines.extend(["", "판단 근거: " + escape(basis)])
        targets = finding.get("target_ids", [finding.get("signal_id")])
        for target in targets:
            signal = by_id.get(target)
            if signal is None:
                continue
            rules = sorted({getattr(f, "rule", "CPD") for f in signal.head_findings})
            metrics = measurement_text(signal.measurements)
            lines.extend(["", f"관련 정적 신호: [{escape(', '.join(rules))}](#signal-{target}) · {escape(signal.change)}" +
                (" · " + escape(metrics) if metrics else "")])
            for violation in signal.head_findings:
                for loc in getattr(violation, "occurrences", (violation,)):
                    lines.append(_a_link(task.snapshot.pr.repository, task.snapshot.head.sha, loc.path, loc.begin_line, loc.end_line))
        links = citation_links(citations, task)
        if links:
            lines.extend(["", "코드·기준: " + links])
        if citations:
            lines.extend(["", quotes(citations, task)])
        lines.append("")
    if questions:
        lines.extend(["##### 작성자 확인 질문", ""])
        for question in questions:
            q = row(question)
            if isinstance(q, str):
                lines.append("- " + escape(q))
            else:
                lines.append("- " + escape(q.get("question", "")))
                if q.get("reason") or q.get("why_needed"):
                    lines.append("  - " + escape(q.get("reason", q.get("why_needed"))))
                links = citation_links(q.get("citations", ()), task)
                if links:
                    lines.append("  - " + links)
                signal = by_id.get(q.get("signal_id"))
                if signal is not None:
                    for finding in signal.head_findings:
                        for loc in getattr(finding, "occurrences", (finding,)):
                            lines.append("  - " + _a_link(task.snapshot.pr.repository, task.snapshot.head.sha,
                                loc.path, loc.begin_line, loc.end_line))
    if target_reviews:
        rows = []
        for signal in selected_signals(task, (), target_reviews):
            values = target_reviews.get(signal.signal_id, [])
            values = values if isinstance(values, list) else [values]
            for review in values:
                if not review:
                    continue
                rules = ", ".join(sorted({getattr(f, "rule", "CPD") for f in signal.head_findings}))
                state = {"reviewed": "검토함", "not_apply": "해당 없음", "insufficient_context": "추가 맥락 필요"}.get(review["state"], review["state"])
                rows.append(f"- **{escape(rules)} · {escape(state)}**: {escape(review['reason'])}\n  - {citation_links(review.get('citations', []), task)}")
        if rows:
            lines.extend(["", details(label(presentation, "details_label", "선택한 신호별 검토 근거"), "\n\n".join(rows))])
    return "\n".join(lines)


def perspective_section(perspectives, task, presentation=None) -> str:
    names = {p.id: p.name_ko for p in PERSPECTIVES}
    blocks = []
    for item in perspectives:
        title = names.get(item.get("perspective_id"), item.get("perspective_id", "관점"))
        for assessment in item.get("assessments", [item]):
            verdict = {"aligned": "목적과 부합", "concern": "문제 제기", "conditional_risk": "조건부 위험", "not_apply": "해당 없음", "insufficient_context": "추가 맥락 필요"}.get(assessment["verdict"], assessment["verdict"])
            blocks.extend(["##### " + escape(title) + " · " + escape(verdict), "",
                escape(assessment["statement"]), "", escape(assessment["reasoning"])])
            if assessment.get("failure_condition"):
                blocks.extend(["", "성립 조건: " + escape(assessment["failure_condition"])])
            blocks.extend(["", citation_links(assessment.get("citations", []), task), "", quotes(assessment.get("citations", []), task), ""])
    return details(label(presentation, "details_label", "8개 관점별 검토 근거"), "\n".join(blocks)) if blocks else ""
