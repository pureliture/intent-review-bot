"""Unit tests for Markdown Formatter adapter."""
from __future__ import annotations

import re
from dataclasses import asdict, replace

import pytest

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


def test_markdown_formatter_selected_static_template_and_llm_sections() -> None:
    result, task = _build_test_fixtures(findings_count=1)
    before = asdict(result)
    markdown = format_review_markdown(result, task)
    assert markdown.splitlines()[:2] == ["<!-- review-bot:architecture:v1 -->", "### PMD/CPD 아키텍처 리뷰 A"]
    assert "#### 전체 판단" in markdown
    assert "#### 정적 분석" in markdown
    assert "| 측정 영역 | 이번 PR 검토 범위 |" in markdown
    assert "CognitiveComplexity" in markdown
    assert "5 → 15" in markdown
    assert "#### LLM 리뷰" in markdown
    assert "관련 정적 신호:" in markdown
    assert "https://github.com/owner/repo/blob/" + "b" * 40 in markdown
    assert "#L10-L20" in markdown
    assert "<details>" in markdown
    assert "plan:" not in markdown and "Tokens:" not in markdown
    assert asdict(result) == before


def test_markdown_formatter_does_not_promote_unadopted_metric_signals() -> None:
    result, task = _build_test_fixtures(findings_count=1)
    unrelated = replace(task.signals[0].signal, signal_id="unadopted", measurements=(Measurement("hidden_metric", 1, 100),))
    task = replace(task, signals=(*task.signals, SelectedSignal(unrelated, "review", "New metric")))
    result = replace(result, analysis=replace(result.analysis, signals=(*result.analysis.signals, unrelated)))
    markdown = format_review_markdown(result, task)
    assert "hidden_metric" not in markdown
    assert "| 검토 대상으로 선택한 신호 | 1건 |" in markdown


def test_markdown_formatter_includes_change_relevant_existing_finding() -> None:
    result, task = _build_test_fixtures(findings_count=1)
    signal = replace(task.signals[0].signal, change="existing")
    task = replace(task, signals=(SelectedSignal(signal, "context", "Existing but changed path"),))
    markdown = format_review_markdown(result, task)
    assert "existing" in markdown and "| 검토 대상으로 선택한 신호 | 1건 |" in markdown


def test_markdown_formatter_non_comparable_delta_retains_analysis_limit() -> None:
    result, task = _build_test_fixtures(comparable=False)
    markdown = format_review_markdown(result, task)
    assert "분석 비교가 불완전하거나 조건이 다릅니다" in markdown
    assert "분석 0건으로 확정하지 않습니다" in markdown


@pytest.mark.parametrize("status", ["failed", "not_run", "partial_review"])
def test_markdown_formatter_incomplete_is_status_only(status) -> None:
    result, task = _build_test_fixtures()
    result = replace(result, status=status, reason="context_missing", summary="DO NOT PUBLISH PARTIAL JUDGEMENT")
    markdown = format_review_markdown(result, task)
    assert "리뷰 미완료" in markdown and "판단에 필요한 정보가 아직 부족합니다" in markdown
    assert "context\\_missing" not in markdown and status.replace("_", "\\_") not in markdown
    assert "미검토·미확정 범위" in markdown
    assert "DO NOT PUBLISH" not in markdown and "Complex method logic" not in markdown
    assert "#### LLM 리뷰" not in markdown


def test_markdown_formatter_validation_error_or_stale_is_status_only() -> None:
    result, task = _build_test_fixtures()
    for options, changed in [({"stale_reason": "head changed"}, result), ({}, replace(result, validation_errors=("citation mismatch",)))]:
        markdown = format_review_markdown(changed, task, **options)
        assert "리뷰 미완료" in markdown
        assert "Complex method logic" not in markdown


def test_markdown_formatter_grounded_citations_and_escaping() -> None:
    result, task = _build_test_fixtures(quote_text="int @admin = 42; <b>bold</b> `unsafe`")
    markdown = format_review_markdown(result, task)
    assert "<summary>인용 원문</summary>" in markdown
    assert "<pre><code>" in markdown and "</code></pre>" in markdown
    assert "@\u200badmin" in markdown and "&lt;b&gt;bold&lt;/b&gt;" in markdown
    assert markdown.count("<details>") == markdown.count("</details>")


def test_markdown_formatter_no_findings_is_scoped_and_short() -> None:
    result, task = _build_test_fixtures(findings_count=0)
    result = replace(result, questions=())
    markdown = format_review_markdown(result, task)
    assert "이 검토 범위에서 추가로 조치할 발견사항은 없습니다." in markdown
    assert "고정된 HEAD" in markdown
    assert "전체 레포 누적" not in markdown and len(markdown) < 1500


def test_markdown_formatter_cpd_retains_all_duplicate_locations() -> None:
    from review_bot.domain.models_v1 import CpdDuplication, CpdOccurrence
    result, task = _build_test_fixtures(findings_count=1)
    duplication = CpdDuplication(12, 80, (
        CpdOccurrence("src/main/java/Calc.java", "src/main/java/Calc.java", 10, 21),
        CpdOccurrence("src/main/java/Other.java", "src/main/java/Other.java", 30, 41),
    ), "duplicate")
    signal = replace(task.signals[0].signal, tool="cpd", head_findings=(duplication,), measurements=())
    task = replace(task, signals=(SelectedSignal(signal, "review", "Changed duplication"),))
    markdown = format_review_markdown(result, task)
    assert "<summary>CPD · 변경과 관련된 중복 위치</summary>" in markdown
    assert "#L10-L21" in markdown and "Other.java#L30-L41" in markdown
    assert "12 / 80" in markdown


def test_markdown_formatter_preserves_measurement_precision() -> None:
    result, task = _build_test_fixtures(findings_count=1)
    signal = replace(task.signals[0].signal, measurements=(Measurement("tcc", 0.015149999999999999, 0.01667),))
    task = replace(task, signals=(SelectedSignal(signal, "review", "Relevant structure"),))
    assert "0.015149999999999999 → 0.01667" in format_review_markdown(result, task).replace("\\.", ".")


def test_markdown_formatter_mock_result_is_visible() -> None:
    result, task = _build_test_fixtures(is_mock=True)
    assert "모의 모델 결과" in format_review_markdown(result, task)


def test_markdown_formatter_templates_reorder_whole_sections_without_execution() -> None:
    result, task = _build_test_fixtures(findings_count=1)
    markdown = format_review_markdown(result, task, comment_template="<script>@admin</script>\n{{llm_review}}\n{{summary}}\n{{static_analysis}}")
    assert markdown.splitlines()[0] == "<!-- review-bot:architecture:v1 -->"
    assert "&lt;script&gt;@\u200badmin&lt;/script&gt;" in markdown
    assert markdown.index("#### LLM 리뷰") < markdown.index("#### 전체 판단") < markdown.index("#### 정적 분석")
    for invalid in ("{{summary}}", "{{summary}}{{summary}}{{static_analysis}}{{llm_review}}", "{{summary}}{{static_analysis}}{{llm_review}}{{danger}}"):
        with pytest.raises(ValueError, match="invalid_comment_template"):
            format_review_markdown(result, task, comment_template=invalid)


def test_markdown_formatter_truncation_does_not_publish_partial_conclusions() -> None:
    result, task = _build_test_fixtures(findings_count=150, quote_text="A" * 600)
    markdown = format_review_markdown(result, task, max_length=60000)
    assert len(markdown) <= 60000
    assert markdown.splitlines()[0] == "<!-- review-bot:architecture:v1 -->"
    assert "comment_limit_exceeded" in markdown
    assert "Complex method logic" not in markdown
    assert len(re.findall(r"<pre><code>", markdown)) == len(re.findall(r"</code></pre>", markdown))
