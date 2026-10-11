"""Markdown review comment renderer with safe syntax-preserving truncation."""
from __future__ import annotations

import html
import re
from typing import Any
from urllib.parse import quote, urlsplit

from review_bot.domain.models_v1 import ReviewResult, TaskInput

DEFAULT_A_MARKER = "<!-- review-bot:architecture:v1 -->"
MAX_COMMENT_LENGTH = 60000


def _a_escape(value: Any) -> str:
    """Escape HTML and special Markdown characters, neutralizing @-mentions."""
    text = html.escape(str(value), quote=False).replace("@", "@\u200b")
    text = re.sub(r"([\\`*_\[\]{}()|#+.!-])", r"\\\1", text)
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def _a_link(repository: str, sha: str, path: str, begin: int, end: int) -> str:
    """Format markdown link to GitHub blob location if valid, else safe label."""
    if (
        not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository)
        or any(part in (".", "..") for part in repository.split("/"))
        or not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", sha)
        or not path
        or path.startswith("/")
        or any(part in (".", "..") for part in path.split("/"))
        or any(char in path for char in ("\\", ":"))
        or any(ord(char) < 32 for char in path)
    ):
        return _a_escape(f"{path}:{begin}-{end} ({sha})")
    url = f"https://github.com/{repository}/blob/{sha}/{quote(path, safe='/')}#L{begin}-L{end}"
    label = _a_escape(f"{path}:{begin}-{end}")
    return f"[{label}]({url})"


def _a_artifact_link(url: str | None) -> str:
    """Format safe HTTPS Actions artifact link or default notice."""
    if not url:
        return "전체 A 자료는 Actions artifact에서 확인할 수 있습니다."
    parsed = urlsplit(url)
    if (
        len(url) > 4096
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or any(ord(char) < 32 for char in url)
    ):
        raise ValueError("A artifact URL must be a credential-free HTTPS address")
    return f"[전체 A 자료]({quote(url, safe=':/?#=&%')})"


def format_review_markdown(
    result: ReviewResult,
    task_input: TaskInput,
    max_length: int = MAX_COMMENT_LENGTH,
    stale_reason: str | None = None,
    marker: str = DEFAULT_A_MARKER,
    artifact_url: str | None = None,
    artifact_notice: str | None = None,
) -> str:
    """Format complete ReviewResult and TaskInput into GitHub Markdown comment with graceful truncation."""
    mode = {"general": "일반 PR", "refactoring": "리팩토링 PR"}.get(
        str(result.review_mode), "알 수 없는 모드"
    )
    status_map = {
        "completed": "의견 검증 완료",
        "not_run": "모델 미실행",
        "failed": "모델 검토 실패",
    }
    status = status_map.get(str(result.status), str(result.status))

    # 1. Header & Badges
    lines = [marker, "### PMD/CPD 아키텍처 리뷰 A", ""]
    if stale_reason:
        lines.extend([
            "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**",
            f"> 게시 중 코드·분류·실행이 변경되었습니다: {_a_escape(stale_reason)}",
            "",
        ])

    lines.extend([
        f"- 리뷰 모드: **{mode}**",
        f"- Base SHA: `{result.base_sha}`",
        f"- Head SHA: `{result.head_sha}`",
        f"- 모델 실행: **{status}** · `{_a_escape(result.reason)}`",
        "",
    ])

    if result.model.get("is_mock"):
        lines.extend([
            "> **모의 모델 결과입니다. 계약 검증이며 실제 LLM 리뷰 품질 검증이 아닙니다.**",
            "",
        ])

    attempted = (
        result.model.get("request_attempted") is True
        or result.status in ("completed", "failed")
    )
    input_prepared = (
        result.model.get("input_prepared") is True
        or (task_input is not None and getattr(task_input, "input_hash", None) is not None)
    )
    attempts = result.model.get("attempts", 1 if attempted else 0)

    lines.extend([
        f"- 전체 입력 준비: {'완료' if input_prepared else '미완료'}",
        f"- 모델 요청 시도: {'있음' if attempted else '없음'} · 시도 횟수: {_a_escape(attempts)}",
        "",
        "#### 분석 변화",
        "",
    ])

    # 2. Analysis Delta Breakdown
    for name, value in sorted(result.analysis.analysis_status.items()):
        lines.append(f"- {_a_escape(name)}: `{_a_escape(value)}`")

    if result.analysis.comparable:
        lines.extend([
            "",
            "| existing | new | worsened | improved | resolved | unknown |",
            "|---:|---:|---:|---:|---:|---:|",
        ])
        lines.append(
            "| "
            + " | ".join(
                str(result.analysis.counts.get(k, 0))
                for k in ("existing", "new", "worsened", "improved", "resolved", "unknown")
            )
            + " |"
        )
    else:
        lines.extend([
            "",
            "> 분석 비교가 불완전하거나 조건이 다릅니다. 신호가 없더라도 분석 0건으로 확정하지 않습니다.",
        ])

    lines.append("")

    # Signals (up to 40)
    displayed = result.analysis.signals[:40]
    for signal in displayed:
        metrics = "; ".join(
            f"{metric.name}: {metric.base} → {metric.head}"
            for metric in signal.measurements
        )
        lines.append(
            f"- `{_a_escape(signal.signal_id)}` · {_a_escape(signal.tool)} · **{_a_escape(signal.change)}**"
            + (f" · {_a_escape(metrics)}" if metrics else "")
        )
        for version, findings in (("base", signal.base_findings), ("head", signal.head_findings)):
            sha = result.base_sha if version == "base" else result.head_sha
            locations = [
                location
                for finding in findings
                for location in getattr(finding, "occurrences", (finding,))
            ]
            for loc in locations[:12]:
                loc_path = getattr(loc, "path", "")
                loc_begin = getattr(loc, "begin_line", 1)
                loc_end = getattr(loc, "end_line", 1)
                lines.append(
                    f"  - {version}: {_a_link(result.pr.repository, sha, loc_path, loc_begin, loc_end)}"
                )
            if len(locations) > 12:
                lines.append(
                    f"  - 댓글 위치 표시: 12/{len(locations)}; 전체 위치·조각은 artifact에 보존됩니다."
                )

    if len(result.analysis.signals) > 40:
        lines.append(
            f"댓글 신호 표시: 40/{len(result.analysis.signals)}. 전체 분석과 모델 입력은 생략하지 않고 artifact에 보존됩니다."
        )

    lines.extend([
        "",
        "> 수치 개선은 구조 개선의 증명이 아닙니다. CPD 중복만으로 책임·목적이 같다고 판단하거나 공통화를 요구하지 않습니다.",
        "",
        "#### 검증된 의견",
        "",
    ])

    # 3. Validated Opinions
    if result.summary:
        lines.extend([_a_escape(result.summary), ""])
    if not result.findings:
        lines.append("검증된 의견 없음 — 실행 상태·검토 대상·분석 한계를 함께 확인하세요.")

    for finding in result.findings:
        lines.extend([
            f"- **{_a_escape(finding.judgement)}** · `{_a_escape(finding.signal_id)}` · {_a_escape(finding.title)}",
            f"  - 판단 근거 ({_a_escape(finding.basis)}): {_a_escape(finding.reason)}",
            f"  - 제안 또는 판단 유보: {_a_escape(finding.recommendation)}",
            f"  - 근거 ID: {', '.join('`' + _a_escape(identifier) + '`' for identifier in finding.evidence_ids)}",
        ])
        for citation in finding.citations[:8]:
            preview = html.escape(citation.quote, quote=False).replace("@", "@\u200b")
            preview = preview.translate(
                {ord(character): f"&#{ord(character)};" for character in "`*_[]{}|#+"}
            )
            lines.extend([
                f"  - {citation.version} `{citation.sha}` · {_a_link(result.pr.repository, citation.sha, citation.path, citation.begin_line, citation.end_line)}",
                "",
                "<pre><code>" + preview + "</code></pre>",
                "",
            ])
        if len(finding.citations) > 8:
            lines.append(
                f"댓글 인용 표시: 8/{len(finding.citations)}. 전체 검증된 인용은 artifact에 보존됩니다."
            )

    # 4. Questions
    if result.questions:
        lines.extend(["", "#### 정보 요청", ""])
        for question in result.questions:
            lines.extend([
                f"- `{_a_escape(question.signal_id)}`: {_a_escape(question.question)}",
                f"  - 이유: {_a_escape(question.reason)}",
                f"  - 근거 ID: {', '.join('`' + _a_escape(identifier) + '`' for identifier in question.evidence_ids)}",
            ])

    # 5. Code Scope & Opinion Scope
    lines.extend([
        "",
        "#### 코드 입력과 의견 대상",
        "",
        "| 버전 | 확보 파일 / bytes | 제공 준비 파일 / bytes | 제외 파일 | 완전성 |",
        "|---|---:|---:|---:|---|",
    ])
    for version in ("base", "head"):
        scope = result.scope.get("code", {}).get(version, {}) if result.scope else {}
        if not scope and task_input and task_input.scope:
            scope = task_input.scope.get("code", {}).get(version, {})
        acquired = scope.get("acquired", {})
        provided = scope.get("provided", {})
        excluded = scope.get("excluded", {})
        lines.append(
            f"| {version} | {_a_escape(acquired.get('file_count', 'unknown'))} / {_a_escape(acquired.get('byte_count', 'unknown'))}"
            f" | {_a_escape(provided.get('file_count', 'unknown'))} / {_a_escape(provided.get('byte_count', 'unknown'))}"
            f" | {_a_escape(excluded.get('file_count', 'unknown'))} | {'완료' if scope.get('complete') else '미확인·불완전'} |"
        )

    lines.extend([
        "",
        "제공 준비는 외부 모델 수신 확인이 아닙니다. 전체 소스·테스트·관련 설정의 확보 범위와 실제 의견 대상을 구분합니다.",
        f"- 의견 신호: {_a_escape(', '.join(result.opinion_scope.get('signal_ids', [])) or '없음')}",
        f"- 의견 코드: {_a_escape(', '.join(result.opinion_scope.get('paths', [])) or '없음')}",
        "",
        "#### 한계",
        "",
    ])

    # 6. Limitations
    limits = list(result.analysis.limitations) + list(result.validation_errors)
    for version in ("base", "head"):
        limits.extend(
            f"{version}: {value}"
            for value in result.scope.get("code", {}).get(version, {}).get("errors", [])
        )
    for limit in limits[:20]:
        lines.append(f"- {_a_escape(limit)}")
    if len(limits) > 20:
        lines.append(f"- 댓글 한계 표시: 20/{len(limits)}; 전체 한계는 artifact에 보존됩니다.")

    lines.extend([
        "- PMD/CPD 0건·검토 대상 없음·분석 실패·모델 미실행은 PR 전체 품질 승인이나 결함 없음 판정이 아닙니다.",
        "- 게시 전후 최신성을 확인하지만 GitHub 상태 조회와 댓글 갱신의 원자성을 보장하지 않습니다.",
        "",
        _a_escape(artifact_notice) if artifact_notice is not None else _a_artifact_link(artifact_url),
        "",
        "<sub>📊 전체 레포 누적: PMD/CPD 기준 분석</sub>",
        "<sub>🤖 자동 생성: review-bot architecture review A</sub>",
    ])

    # 7. Safe Syntax-Preserving Truncation (60,000 char cap)
    rendered = "\n".join(lines)
    if len(rendered) <= max_length:
        return rendered + "\n"

    # Truncate at line boundaries around ~57,000 chars while keeping footer intact
    footer_lines = lines[-5:]
    footer_text = "\n".join(footer_lines)
    omission_banner_fmt = (
        "\n\n댓글 표시 한도: {omitted}개 상세 구간은 artifact에서 확인하세요. "
        "전체 모델 입력·의견·근거는 생략 없이 저장됩니다.\n\n"
    )
    # Reserve space for footer, omission banner, and any closing tags
    safety_margin = len(footer_text) + 350
    budget = min(57000, max_length - safety_margin)

    kept: list[str] = []
    omitted = 0
    current_size = 0
    in_pre_code = False
    in_code_fence = False

    for line in lines[:-5]:
        line_len = len(line) + 1
        if current_size + line_len <= budget:
            kept.append(line)
            current_size += line_len
            if "<pre><code>" in line:
                in_pre_code = True
            if "</code></pre>" in line:
                in_pre_code = False
            if line.strip().startswith("```"):
                in_code_fence = not in_code_fence
        else:
            omitted += 1

    # Ensure unclosed HTML/Markdown blocks are cleanly terminated
    if in_pre_code:
        kept.append("</code></pre>")
    if in_code_fence:
        kept.append("```")

    omission_banner = omission_banner_fmt.format(omitted=omitted)
    truncated = "\n".join(kept) + omission_banner + footer_text + "\n"

    # If somehow still over max_length, aggressively trim kept lines from end
    while len(truncated) > max_length and len(kept) > 10:
        removed = kept.pop()
        if "<pre><code>" in removed:
            in_pre_code = False
        if "</code></pre>" in removed:
            in_pre_code = True
        omitted += 1
        closing = ""
        if in_pre_code:
            closing = "\n</code></pre>"
        omission_banner = omission_banner_fmt.format(omitted=omitted)
        truncated = "\n".join(kept) + closing + omission_banner + footer_text + "\n"

    return truncated
