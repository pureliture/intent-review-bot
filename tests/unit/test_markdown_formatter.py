"""Unit tests for Markdown Formatter adapter."""
from __future__ import annotations

import re

from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.domain.changes import Measurement
from review_bot.domain.models_v1 import (
    AnalysisDelta,
    Citation,
    Classification,
    CodeFile,
    CodeSnapshot,
    DeltaSignal,
    Evidence,
    PmdViolation,
    PrSnapshot,
    ReviewFinding,
    ReviewQuestion,
    ReviewResult,
    SelectedSignal,
    Snapshot,
    TaskInput,
)


def _build_test_fixtures(
    *,
    review_mode: str = "general",
    findings_count: int = 2,
    comparable: bool = True,
    is_mock: bool = False,
    quote_text: str = "int count = 0;",
) -> tuple[ReviewResult, TaskInput]:
    pr = PrSnapshot(
        repository="owner/repo",
        number=42,
        base_sha="a" * 40,
        head_sha="b" * 40,
        source_branch="feature/calc",
        changed_files=("src/main/java/Calc.java",),
    )

    base_files = (CodeFile(path="src/main/java/Calc.java", content="class Calc {}", content_hash="h1", kind="source"),)
    head_files = (CodeFile(path="src/main/java/Calc.java", content="class Calc { int x; }", content_hash="h2", kind="source"),)
    base_code = CodeSnapshot(sha="a" * 40, files=base_files, exclusions=(), complete=True, errors=(), inventory_hash="ih1")
    head_code = CodeSnapshot(sha="b" * 40, files=head_files, exclusions=(), complete=True, errors=(), inventory_hash="ih2")

    v1 = PmdViolation(
        path="src/main/java/Calc.java",
        raw_path="src/main/java/Calc.java",
        begin_line=10,
        end_line=20,
        begin_column=1,
        end_column=5,
        rule="CognitiveComplexity",
        ruleset="ruleset.xml",
        priority=3,
        klass="Calc",
        method="calculate",
        package="demo",
        message="High complexity",
        code_fragment="int calculate() {}",
    )
    m = Measurement(name="CognitiveComplexity", base=5, head=15)
    sig = DeltaSignal(
        signal_id="sig-1",
        tool="pmd",
        change="worsened",
        base_findings=(v1,),
        head_findings=(v1,),
        paths=("src/main/java/Calc.java",),
        matching="signature",
        measurements=(m,),
    )

    delta = AnalysisDelta(
        schema_version=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        conditions_hash="ch1",
        comparable=comparable,
        signals=(sig,),
        limitations=(),
        analysis_status={"pmd": "findings", "cpd": "clean"},
    )

    cit = Citation(
        evidence_id="code-1",
        version="head",
        sha="b" * 40,
        path="src/main/java/Calc.java",
        begin_line=10,
        end_line=20,
        quote=quote_text,
    )
    findings = []
    for i in range(findings_count):
        findings.append(
            ReviewFinding(
                signal_id=f"sig-{i + 1}",
                judgement="concern",
                change="worsened",
                basis="analysis_signal",
                title=f"Complex method logic {i + 1}",
                reason="Complexity metric exceeded threshold",
                recommendation="Extract helper methods",
                evidence_ids=(f"ev-{i + 1}",),
                citations=(cit,),
            )
        )

    questions = (
        ReviewQuestion(
            signal_id="sig-1",
            question="Can this method be refactored?",
            reason="High cognitive score",
            evidence_ids=("ev-1",),
        ),
    )

    snap = Snapshot(pr=pr, base=base_code, head=head_code)
    ev = Evidence(
        evidence_id="code-1",
        kind="code",
        version="head",
        sha="b" * 40,
        path="src/main/java/Calc.java",
        begin_line=10,
        end_line=20,
        content=quote_text,
    )
    selected = (SelectedSignal(signal=sig, role="review", reason="Worsened violation"),)
    scope = {
        "code": {
            "base": {
                "acquired": {"file_count": 1, "byte_count": 100},
                "provided": {"file_count": 1, "byte_count": 100},
                "excluded": {"file_count": 0},
                "complete": True,
            },
            "head": {
                "acquired": {"file_count": 1, "byte_count": 120},
                "provided": {"file_count": 1, "byte_count": 120},
                "excluded": {"file_count": 0},
                "complete": True,
            },
        }
    }
    task_input = TaskInput(
        schema_version=1,
        snapshot=snap,
        delta=delta,
        classification=Classification(review_mode=review_mode),
        signals=selected,
        evidence=(ev,),
        scope=scope,
        limitations=(),
        input_hash="hash-123",
    )

    result = ReviewResult(
        schema_version=1,
        review_type="A",
        run_key="rk-1",
        base_sha="a" * 40,
        head_sha="b" * 40,
        review_mode=review_mode,
        pr=pr,
        status="completed",
        reason="validated",
        summary="Review completed successfully.",
        findings=tuple(findings),
        questions=questions,
        validation_errors=(),
        scope=scope,
        opinion_scope={"signal_ids": ["sig-1"], "paths": ["src/main/java/Calc.java"]},
        analysis=delta,
        model={"is_mock": is_mock, "input_prepared": True, "request_attempted": True, "attempts": 1},
        usage={"prompt_tokens": 500, "completion_tokens": 200},
        created_at="2026-10-07T00:00:00Z",
        prompt_version="review-a/v1",
        input_hash="hash-123",
    )
    return result, task_input


def test_markdown_formatter_header_and_marker() -> None:
    result, task_input = _build_test_fixtures()
    markdown = format_review_markdown(result, task_input)

    lines = markdown.splitlines()
    assert lines[0] == "<!-- review-bot:architecture:v1 -->"
    assert lines[1] == "### PMD/CPD 아키텍처 리뷰 A"
    assert "- 리뷰 모드: **일반 PR**" in markdown
    assert f"- Base SHA: `{result.base_sha}`" in markdown
    assert f"- Head SHA: `{result.head_sha}`" in markdown
    assert "- 모델 실행: **의견 검증 완료**" in markdown


def test_markdown_formatter_refactoring_mode_badge() -> None:
    result, task_input = _build_test_fixtures(review_mode="refactoring")
    markdown = format_review_markdown(result, task_input)
    assert "- 리뷰 모드: **리팩토링 PR**" in markdown


def test_markdown_formatter_stale_warning_banner() -> None:
    result, task_input = _build_test_fixtures()
    stale_reason = "head commit moved from aaaaaaaa to bbbbbbbb"
    markdown = format_review_markdown(result, task_input, stale_reason=stale_reason)

    assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in markdown
    assert stale_reason in markdown


def test_markdown_formatter_mock_banner() -> None:
    result, task_input = _build_test_fixtures(is_mock=True)
    markdown = format_review_markdown(result, task_input)
    assert "> **모의 모델 결과입니다. 계약 검증이며 실제 LLM 리뷰 품질 검증이 아닙니다.**" in markdown


def test_markdown_formatter_delta_table_6_states() -> None:
    result, task_input = _build_test_fixtures()
    markdown = format_review_markdown(result, task_input)

    assert "| existing | new | worsened | improved | resolved | unknown |" in markdown
    assert "| 0 | 0 | 1 | 0 | 0 | 0 |" in markdown


def test_markdown_formatter_non_comparable_delta_notice() -> None:
    result, task_input = _build_test_fixtures(comparable=False)
    markdown = format_review_markdown(result, task_input)

    assert "> 분석 비교가 불완전하거나 조건이 다릅니다. 신호가 없더라도 분석 0건으로 확정하지 않습니다." in markdown
    assert "| existing | new | worsened |" not in markdown


def test_markdown_formatter_grounded_citations_and_escaping() -> None:
    unsafe_quote = "int @admin = 42; <b>bold</b> `rm -rf /`"
    result, task_input = _build_test_fixtures(quote_text=unsafe_quote)
    markdown = format_review_markdown(result, task_input)

    assert "#### 검증된 의견" in markdown
    assert "<pre><code>" in markdown
    assert "</code></pre>" in markdown
    # Check @-mention neutralized
    assert "@\u200badmin" in markdown
    # Check raw HTML tags escaped
    assert "&lt;b&gt;bold&lt;/b&gt;" in markdown


def test_markdown_formatter_questions_and_scope() -> None:
    result, task_input = _build_test_fixtures()
    markdown = format_review_markdown(result, task_input)

    assert "#### 정보 요청" in markdown
    assert "Can this method be refactored?" in markdown
    assert "#### 코드 입력과 의견 대상" in markdown
    assert "| base | 1 / 100 | 1 / 100 | 0 | 완료 |" in markdown
    assert "| head | 1 / 120 | 1 / 120 | 0 | 완료 |" in markdown
    assert "<sub>📊 전체 레포 누적: PMD/CPD 기준 분석</sub>" in markdown
    assert "<sub>🤖 자동 생성: review-bot architecture review A</sub>" in markdown


def test_markdown_formatter_truncation_under_60k() -> None:
    # Build large result with 150 findings to trigger > 60k chars
    result, task_input = _build_test_fixtures(findings_count=150, quote_text="A" * 600)
    markdown = format_review_markdown(result, task_input, max_length=60000)

    assert len(markdown) <= 60000
    # Marker must strictly remain on line 1
    assert markdown.splitlines()[0] == "<!-- review-bot:architecture:v1 -->"
    # Omission notice must be inserted
    assert "댓글 표시 한도:" in markdown
    assert "상세 구간은 artifact에서 확인하세요" in markdown
    # Metadata footer preserved
    assert "<sub>🤖 자동 생성: review-bot architecture review A</sub>" in markdown

    # Verify no broken/unclosed <pre><code> tags
    open_tags = len(re.findall(r"<pre><code>", markdown))
    close_tags = len(re.findall(r"</code></pre>", markdown))
    assert open_tags == close_tags, f"Mismatched <pre><code>: {open_tags} open vs {close_tags} close"
