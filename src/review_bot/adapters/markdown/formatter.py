"""Developer-facing A review comments with bounded, grounded sections."""
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
    task_input: TaskInput | None,
    max_length: int = MAX_COMMENT_LENGTH,
    stale_reason: str | None = None,
    marker: str = DEFAULT_A_MARKER,
    artifact_url: str | None = None,
    artifact_notice: str | None = None,
    *, presentation=None, comment_template: str | None = None,
) -> str:
    """Present selected static signals and adopted LLM opinions to developers."""
    from review_bot.adapters.markdown.developer_presentation import (
        findings_section, finish, label, review_summary, selected_signals, static_analysis, status_sections, walkthrough,
    )

    title = label(presentation, "title_a", "PMD/CPD 아키텍처 리뷰 A")
    if task_input is None:
        return finish(marker, title, status_sections(result,
            unreviewed=("검토 입력이 없어 정적 신호와 코드 근거를 다시 확인할 수 없습니다.",)),
            scope="", max_length=max_length)
    if result.status != "completed" or result.reason != "validated" or result.validation_errors or stale_reason:
        unreviewed = list(result.validation_errors)
        if stale_reason:
            unreviewed.append("최신 코드와 일치하지 않는 결과: " + stale_reason)
        return finish(marker, title, status_sections(result, unreviewed=unreviewed), scope="", max_length=max_length)
    signals = selected_signals(task_input, (*result.findings, *result.questions))
    summary = review_summary(result.summary, task_input, is_mock=result.model.get("is_mock"))
    walk = walkthrough(task_input, result.findings, result.questions, signals=signals)
    if walk:
        summary += "\n\n" + walk
    sections = {"summary": summary,
        "static_analysis": static_analysis(task_input, signals, presentation=presentation),
        "llm_review": findings_section(result.findings, result.questions, task_input, signals=signals, presentation=presentation)}
    footer = _a_escape(artifact_notice) if artifact_notice is not None else _a_artifact_link(artifact_url) if artifact_url else ""
    if presentation and presentation.get("footer_notice"):
        footer += ("\n\n" if footer else "") + label(presentation, "footer_notice", "")
    return finish(marker, title, sections, scope="", template=comment_template, max_length=max_length, footer=footer)
