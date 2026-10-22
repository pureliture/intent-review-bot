"""Developer-facing comments keep adopted conclusions and private traces apart."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
from review_bot.application.partition_review import run_partition_review
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient
from tests.unit.test_partition_review import _pmd_partition_task


def completed(task, mode, client=None):
    return run_partition_review(task, mode, client or PartitionModelClient(), PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=128))


def test_partition_b_completed_no_findings_has_collapsed_perspectives_and_no_internal_metadata():
    task = make_task()
    result = completed(task, "intent")
    assert result.status == "completed", result.reason
    before = deepcopy(asdict(result))
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert "### 의도 리뷰 B" in markdown
    assert "이 검토 범위에서 추가로 조치할 발견사항은 없습니다." in markdown
    assert "<summary>8개 관점별 검토 근거</summary>" in markdown
    assert "문제와 성공 기준" in markdown and "검증 근거" in markdown
    assert "https://github.com/" in markdown and "#L" in markdown
    for internal in ("shared_plan", "plan:", "UTF-8", "분할 평가", "Usage", "Tokens", "사실 검토"):
        assert internal not in markdown
    assert asdict(result) == before
    assert "비공개" in format_partition_review_markdown(result, task)


def test_partition_a_target_reviews_are_collapsed_and_metrics_are_not_findings():
    task = _pmd_partition_task()
    result = completed(task, "architecture")
    assert result.status == "completed", result.reason
    assert result.partition["final"]["target_reviews"]
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert "이 검토 범위에서 추가로 조치할 발견사항은 없습니다." in markdown
    assert "| 검토 대상으로 선택한 신호 | 0건 |" in markdown
    assert "<summary>선택한 신호별 검토 근거</summary>" in markdown
    assert "Synthetic explicit review" in markdown


def test_partition_incomplete_never_exposes_partial_summary_or_findings():
    task = make_task()
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=1))
    assert result.status != "completed"
    markdown = format_partition_review_markdown(result, task, publication_mode="comment",
        comment_template="ignored unsafe/incomplete template")
    assert "리뷰 미완료" in markdown
    assert "검토 요청 횟수 한도에 도달해 확인을 마치지 못했습니다" in markdown
    assert result.reason.replace("_", "\\_") not in markdown
    assert "#### 전체 판단" not in markdown and "#### LLM 리뷰" not in markdown
    assert "미검토·미확정 범위" not in markdown


def test_status_notice_explains_known_reason_and_hides_unknown_pipeline_error():
    from types import SimpleNamespace
    from review_bot.adapters.markdown.developer_presentation import status_sections

    result = SimpleNamespace(status="partial_review", reason="factual_verification_incomplete")
    notice = status_sections(result, unreviewed=("src/formatter.py의 경계 조건",))["status"]
    assert "코드와 근거를 대조하는 확인이 끝나지 않았습니다" in notice
    assert "src/formatter\\.py의 경계 조건" in notice
    assert "factual_verification" not in notice and "partial_review" not in notice
    english = status_sections(result, ko=False)["status"]
    assert "The code and supporting evidence have not been fully checked" in english
    for reason in ("RuntimeError: private-provider-response\nTraceback internal/path.py",
            "partition_planning_failed:usage_unavailable\nprivate-provider-response",
            "partition_planning_failed:RuntimeError: private-provider-response"):
        result.reason = reason
        unknown = status_sections(result)["status"]
        assert "새로운 커밋이 올라오면 다시 검토합니다" in unknown
        for private in ("RuntimeError", "private-provider", "Traceback", "internal/path", "usage_unavailable"):
            assert private not in unknown


@pytest.mark.parametrize("ko", [True, False])
def test_initial_plan_notice_preserves_only_the_exact_known_reason(ko):
    from types import SimpleNamespace
    from review_bot.adapters.markdown.developer_presentation import status_sections

    result = SimpleNamespace(status="not_run", reason="partition_planning_failed:usage_unavailable")
    notice = status_sections(result, ko=ko, unreviewed=("intent",))["status"]
    assert ("초기 리뷰 계획을 확정하지 못했습니다" if ko else "The initial review plan could not be established") in notice
    assert ("모델 응답을 끝까지 확인하지 못해" if ko else "model response could not be confirmed") in notice
    assert "intent" in notice and "partition_planning_failed" not in notice and "usage_unavailable" not in notice


def test_partition_template_cannot_drop_required_completed_sections():
    task = make_task()
    result = completed(task, "intent")
    with pytest.raises(ValueError, match="invalid_comment_template"):
        format_partition_review_markdown(result, task, publication_mode="comment", comment_template="{{summary}}{{perspectives}}")
    markdown = format_partition_review_markdown(result, task, publication_mode="comment",
        comment_template="{{llm_review}}\n{{summary}}\n{{perspectives}}", presentation={"title_b": "팀 의도 리뷰"})
    assert "### 팀 의도 리뷰" in markdown
    assert markdown.index("#### 리뷰 의견") < markdown.index("#### 전체 판단")


@pytest.mark.parametrize("verdict,label,other_label", [
    ("concern", "확인된 문제", "조건부 위험"),
    ("conditional_risk", "조건부 위험", "확인된 문제"),
])
def test_partition_declared_verdict_keeps_its_label_and_failure_condition(verdict, label, other_label):
    def concern(payload, body):
        if payload["stage"] == "final_review":
            body["perspectives"][0].update(verdict=verdict, statement="Missing caller guard can expose stale values.",
                failure_condition="A caller bypasses the guard while the cache is stale.")
        return body

    task = make_task()
    result = completed(task, "intent", PartitionModelClient(transform=concern))
    assert result.status == "completed", result.reason
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    main = markdown.split("<summary>8개 관점별 검토 근거</summary>")[0]
    assert label in main and other_label not in main and "Missing caller guard" in main
    assert "성립 조건:" in main
    assert "A caller bypasses the guard" in main
    assert "추가로 조치할 발견사항은 없습니다" not in main


def test_explicit_finding_verdict_precedes_its_condition_and_analysis_basis():
    from review_bot.adapters.markdown.developer_presentation import judgement_label
    for verdict, expected in (("concern", "확인된 문제"), ("confirmed_issue", "확인된 문제"),
            ("conditional_risk", "조건부 위험"), ("improvement", "구체적 개선 제안"),
            ("insufficient_evidence", "작성자 확인 질문"), ("author_question", "작성자 확인 질문")):
        assert judgement_label({"verdict": verdict, "basis": "analysis_signal", "failure_condition": "A concrete condition."}) == expected
    assert judgement_label({"judgement": "concern", "basis": "analysis_signal"}) == "확인된 문제"


def test_findings_sort_by_severity_blocking_problems_first():
    from types import SimpleNamespace
    from review_bot.adapters.markdown.developer_presentation import findings_section
    task = SimpleNamespace(snapshot=SimpleNamespace(pr=SimpleNamespace(
        repository="o/r", number=1, changed_files=("a.py",))))
    findings = [
        {"statement": "low question", "verdict": "insufficient_context", "reason": "r",
         "citations": []},
        {"statement": "top problem", "verdict": "concern", "reason": "r", "citations": []},
        {"statement": "mid risk", "verdict": "conditional_risk", "reason": "r",
         "failure_condition": "c", "citations": []},
    ]
    markdown = findings_section(findings, [], task)
    assert markdown.index("top problem") < markdown.index("mid risk") < markdown.index("low question")


def test_walkthrough_table_counts_findings_per_changed_file():
    from types import SimpleNamespace
    from review_bot.adapters.markdown.developer_presentation import walkthrough
    task = SimpleNamespace(reviewed_paths=("a.py", "b.py"),
        snapshot=SimpleNamespace())
    findings = [{"statement": "s", "verdict": "concern", "reason": "r",
        "citations": [{"path": "a.py"}, {"path": "a.py"}]}]
    table = walkthrough(task, findings, [])
    assert "<summary>변경 파일 둘러보기</summary>" in table
    assert "| a\\.py | 2건 |" in table
    assert "| b\\.py | 0건 |" in table
