"""Empirical adversarial stress-test suite for Milestone 6:
- review_bot.adapters.markdown.formatter (60k character limit, syntax-preserving truncation, code fence integrity)
- review_bot.application.publication (Guarded PR publication, marker idempotency, race conditions, freshness checks)
- review_bot.adapters.github.client (Mock and production client contract behaviors)

Adversarial Dimensions:
1. 60k Character Hard Limit & Syntax Preservation Stress Testing (100k, 200k, massive findings/questions, code fence balance)
2. Publication Race Conditions & Post-Check Freshness (Head SHA drift, PR closed, concurrent simulations)
3. Marker Idempotency & Non-Duplication (Repeated publications, coexisting human comments, pre-check freshness)
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from review_bot.adapters.github.client import MockGitHubClient
from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.application.publication import DEFAULT_A_MARKER, publish_review_comment
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


def _build_stress_fixtures(
    *,
    findings_count: int = 10,
    questions_count: int = 2,
    quote_multiplier: int = 1,
    custom_quote: str | None = None,
    extra_summary: str = "",
    limitations_count: int = 0,
) -> tuple[ReviewResult, TaskInput]:
    pr = PrSnapshot(
        repository="owner/stress-repo",
        number=777,
        base_sha="a" * 40,
        head_sha="b" * 40,
        source_branch="feature/stress-test",
        changed_files=("src/main/java/MassiveService.java",),
    )

    base_files = (CodeFile(path="src/main/java/MassiveService.java", content="class MassiveService {}", content_hash="h1", kind="source"),)
    head_files = (CodeFile(path="src/main/java/MassiveService.java", content="class MassiveService { int complex; }", content_hash="h2", kind="source"),)
    base_code = CodeSnapshot(sha="a" * 40, files=base_files, exclusions=(), complete=True, errors=(), inventory_hash="ih1")
    head_code = CodeSnapshot(sha="b" * 40, files=head_files, exclusions=(), complete=True, errors=(), inventory_hash="ih2")

    v1 = PmdViolation(
        path="src/main/java/MassiveService.java",
        raw_path="src/main/java/MassiveService.java",
        begin_line=1,
        end_line=50,
        begin_column=1,
        end_column=80,
        rule="GodClass",
        ruleset="ruleset.xml",
        priority=1,
        klass="MassiveService",
        method="executeStress",
        package="org.stress",
        message="God class pattern detected with excessive responsibilities",
        code_fragment="public class MassiveService { ... }",
    )
    sig = DeltaSignal(
        signal_id="sig-stress-1",
        tool="pmd",
        change="worsened",
        base_findings=(v1,),
        head_findings=(v1,),
        paths=("src/main/java/MassiveService.java",),
        matching="signature",
        measurements=(Measurement(name="CognitiveComplexity", base=10, head=150),),
    )

    limits = tuple(f"Limitation {i}: analysis boundary reached" for i in range(limitations_count))
    delta = AnalysisDelta(
        schema_version=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        conditions_hash="ch_stress",
        comparable=True,
        signals=(sig,),
        limitations=limits,
        analysis_status={"pmd": "findings", "cpd": "clean"},
    )

    base_quote = custom_quote if custom_quote is not None else ("public void executeStress() {\n    int a = 0;\n    // " + ("X" * 100) + "\n}\n")
    quote_text = base_quote * quote_multiplier

    cit = Citation(
        evidence_id="ev-stress-code-1",
        version="head",
        sha="b" * 40,
        path="src/main/java/MassiveService.java",
        begin_line=1,
        end_line=50,
        quote=quote_text,
    )

    findings = []
    for i in range(findings_count):
        findings.append(
            ReviewFinding(
                signal_id="sig-stress-1",
                judgement="concern",
                change="worsened",
                basis="analysis_signal",
                title=f"Excessive architectural entanglement in finding {i + 1}",
                reason=f"Detailed reason for finding {i + 1}: " + ("R" * 200),
                recommendation=f"Refactoring recommendation {i + 1}: " + ("REC" * 100),
                evidence_ids=(f"ev-stress-{i + 1}",),
                citations=(cit,),
            )
        )

    questions = []
    for i in range(questions_count):
        questions.append(
            ReviewQuestion(
                signal_id="sig-stress-1",
                question=f"Architectural question {i + 1}: " + ("Q" * 150),
                reason=f"Reason for question {i + 1}: " + ("QR" * 100),
                evidence_ids=(f"ev-q-{i + 1}",),
            )
        )

    snap = Snapshot(pr=pr, base=base_code, head=head_code)
    ev = Evidence(
        evidence_id="ev-stress-code-1",
        kind="code",
        version="head",
        sha="b" * 40,
        path="src/main/java/MassiveService.java",
        begin_line=1,
        end_line=50,
        content=quote_text,
    )
    selected = (SelectedSignal(signal=sig, role="review", reason="Stress test signal"),)
    scope = {
        "code": {
            "base": {"acquired": {"file_count": 1, "byte_count": 500}, "provided": {"file_count": 1, "byte_count": 500}, "excluded": {"file_count": 0}, "complete": True},
            "head": {"acquired": {"file_count": 1, "byte_count": 800}, "provided": {"file_count": 1, "byte_count": 800}, "excluded": {"file_count": 0}, "complete": True},
        }
    }
    task_input = TaskInput(
        schema_version=1,
        snapshot=snap,
        delta=delta,
        classification=Classification(review_mode="general"),
        signals=selected,
        evidence=(ev,),
        scope=scope,
        limitations=limits,
        input_hash="ihash-stress",
    )

    summary_text = "Review completed." + ((" " + extra_summary) if extra_summary else "")
    result = ReviewResult(
        schema_version=1,
        review_type="A",
        run_key="rk-stress",
        base_sha="a" * 40,
        head_sha="b" * 40,
        review_mode="general",
        pr=pr,
        status="completed",
        reason="validated",
        summary=summary_text,
        findings=tuple(findings),
        questions=tuple(questions),
        validation_errors=(),
        scope=scope,
        opinion_scope={"signal_ids": ["sig-stress-1"], "paths": ["src/main/java/MassiveService.java"]},
        analysis=delta,
        model={"is_mock": False, "input_prepared": True, "request_attempted": True, "attempts": 1},
        usage={"prompt_tokens": 1000, "completion_tokens": 500},
        created_at="2026-10-08T00:00:00Z",
        prompt_version="review-a/v1",
        input_hash="ihash-stress",
    )
    return result, task_input


# ==============================================================================
# Dimension 1: 60k Character Hard Limit & Syntax Preservation Stress Testing
# ==============================================================================


class TestMarkdownFormatter60kStress:
    """Empirical stress-testing of Markdown Formatter under massive input sizes."""

    def test_massive_findings_100k_chars_strictly_capped_and_valid_syntax(self) -> None:
        # 120 findings with ~900 chars each produces ~110,000 chars un-truncated
        result, task_input = _build_stress_fixtures(findings_count=120, quote_multiplier=3)
        formatted = format_review_markdown(result, task_input, max_length=60000)

        # 1. Hard character limit <= 60,000
        assert len(formatted) <= 60000, f"Exceeded 60k limit: {len(formatted)} chars"

        # 2. Marker strictly on line 1
        lines = formatted.splitlines()
        assert lines[0] == DEFAULT_A_MARKER

        # 3. Oversize reviews become a notice, never a partial set of findings.
        assert "리뷰 본문 표시 보류:" in formatted
        assert "전체 결과는 저장된 산출물을 확인하세요" in formatted
        assert "Excessive architectural entanglement" not in formatted

        # 4. The fixed code scope remains visible after the omitted body.
        assert "검토 범위: 고정된 HEAD" in formatted
        assert result.head_sha in formatted

        # 5. Code block syntax integrity: <pre><code> must be perfectly balanced
        open_pre = len(re.findall(r"<pre><code>", formatted))
        close_pre = len(re.findall(r"</code></pre>", formatted))
        assert open_pre == close_pre, f"Unbalanced <pre><code>: {open_pre} open vs {close_pre} close"

    def test_massive_findings_200k_chars_strictly_capped_and_valid_syntax(self) -> None:
        # 250 findings with large quotes produces ~230,000 chars un-truncated
        result, task_input = _build_stress_fixtures(findings_count=250, quote_multiplier=4)
        formatted = format_review_markdown(result, task_input, max_length=60000)

        assert len(formatted) <= 60000, f"Exceeded 60k limit: {len(formatted)} chars"
        lines = formatted.splitlines()
        assert lines[0] == DEFAULT_A_MARKER
        assert "리뷰 본문 표시 보류:" in formatted
        assert "검토 범위: 고정된 HEAD" in formatted
        assert "Excessive architectural entanglement" not in formatted

        open_pre = len(re.findall(r"<pre><code>", formatted))
        close_pre = len(re.findall(r"</code></pre>", formatted))
        assert open_pre == close_pre, f"Unbalanced <pre><code>: {open_pre} open vs {close_pre} close"

    def test_massive_questions_alone_trigger_truncation_cleanly(self) -> None:
        # 300 questions producing ~120,000 chars un-truncated
        result, task_input = _build_stress_fixtures(findings_count=2, questions_count=300)
        formatted = format_review_markdown(result, task_input, max_length=60000)

        assert len(formatted) <= 60000
        assert formatted.splitlines()[0] == DEFAULT_A_MARKER
        assert "리뷰 본문 표시 보류:" in formatted
        assert "검토 범위: 고정된 HEAD" in formatted
        assert "Architectural question" not in formatted

    def test_massive_private_limitations_do_not_displace_developer_review(self) -> None:
        result, task_input = _build_stress_fixtures(findings_count=2, limitations_count=200)
        formatted = format_review_markdown(result, task_input, max_length=60000)

        assert len(formatted) <= 60000
        assert formatted.splitlines()[0] == DEFAULT_A_MARKER
        assert "Excessive architectural entanglement" in formatted
        assert "analysis boundary reached" not in formatted
        assert len(result.analysis.limitations) == len(task_input.limitations) == 200
        assert "검토 범위: 고정된 HEAD" in formatted

    def test_citation_quotes_containing_markdown_code_fences_and_html(self) -> None:
        adversarial_code = (
            "```java\n"
            "public void test() {\n"
            "    String msg = \"<script>alert('xss')</script>\";\n"
            "    String email = \"user@company.com\";\n"
            "    int val = 10 * 20; // `inline code` and *italic*\n"
            "}\n"
            "```\n"
        )
        result, task_input = _build_stress_fixtures(
            findings_count=100, custom_quote=adversarial_code
        )
        formatted = format_review_markdown(result, task_input, max_length=60000)

        assert len(formatted) <= 60000
        # Check neutralization of @
        assert "@\u200bcompany" in formatted or "user@company" not in formatted
        # Check <pre><code> balanced
        open_pre = len(re.findall(r"<pre><code>", formatted))
        close_pre = len(re.findall(r"</code></pre>", formatted))
        assert open_pre == close_pre

    def test_various_custom_max_length_thresholds(self) -> None:
        result, task_input = _build_stress_fixtures(findings_count=50)
        for budget in [5000, 15000, 30000, 45000, 60000]:
            formatted = format_review_markdown(result, task_input, max_length=budget)
            assert len(formatted) <= budget, f"Failed at budget {budget}: got {len(formatted)}"
            assert formatted.splitlines()[0] == DEFAULT_A_MARKER
            assert "검토 범위: 고정된 HEAD" in formatted
            assert "리뷰 본문 표시 보류:" in formatted
            assert formatted.count("<details>") == formatted.count("</details>")

    def test_extreme_scale_500k_raw_chars_capped_under_60k(self) -> None:
        # Extreme stress test: 500 findings, massive raw payload
        result, task_input = _build_stress_fixtures(findings_count=500, quote_multiplier=4)
        formatted = format_review_markdown(result, task_input, max_length=60000)

        assert len(formatted) <= 60000, f"Exceeded 60k limit at extreme scale: {len(formatted)}"
        assert formatted.splitlines()[0] == DEFAULT_A_MARKER
        assert "리뷰 본문 표시 보류:" in formatted
        assert "검토 범위: 고정된 HEAD" in formatted
        assert "Excessive architectural entanglement" not in formatted

        open_pre = len(re.findall(r"<pre><code>", formatted))
        close_pre = len(re.findall(r"</code></pre>", formatted))
        assert open_pre == close_pre

    def test_single_giant_token_50k_chars_in_summary(self) -> None:
        # Adversarial input: giant continuous 50,000-char token without whitespace
        giant_token = "Z" * 50000
        result, task_input = _build_stress_fixtures(findings_count=1, extra_summary=giant_token)
        formatted = format_review_markdown(result, task_input, max_length=60000)

        assert len(formatted) <= 60000
        assert formatted.splitlines()[0] == DEFAULT_A_MARKER
        assert giant_token in formatted
        assert "검토 범위: 고정된 HEAD" in formatted

    def test_none_task_input_does_not_crash_formatter(self) -> None:
        result, _ = _build_stress_fixtures(findings_count=2)
        # Passing None as task_input
        formatted = format_review_markdown(result, None)  # type: ignore[arg-type]
        assert len(formatted) <= 60000
        assert formatted.splitlines()[0] == DEFAULT_A_MARKER
        assert "리뷰 미완료" in formatted
        assert "검토 입력이 없어 정적 신호와 코드 근거를 다시 확인할 수 없습니다" in formatted
        assert "Excessive architectural entanglement" not in formatted
        assert "#### LLM 리뷰" not in formatted


# ==============================================================================
# Dimension 2: Publication Race Conditions & Post-Check Freshness
# ==============================================================================


class TestGuardedPublicationRaceConditions:
    """Stress-test publication race conditions, head SHA drift, and PR state changes."""

    def test_postcheck_race_head_sha_drift_marks_receipt_stale_and_updates_banner(self) -> None:
        class DriftMockGitHubClient(MockGitHubClient):
            def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                record = super().create_comment(*args, **kwargs)
                # Commit pushed mid-flight: head SHA moves from b*40 to f*40
                self.set_pull_request(
                    pr_number=1,
                    state="open",
                    head_sha="f" * 40,
                    base_sha="a" * 40,
                )
                return record

        client = DriftMockGitHubClient()
        client.set_pull_request(pr_number=1, state="open", head_sha="b" * 40, base_sha="a" * 40)

        receipt = publish_review_comment(
            github_client=client,
            repository_id=10,
            pr_number=1,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Review comment content",
            marker=DEFAULT_A_MARKER,
        )

        # 1. Receipt must indicate stale
        assert receipt.status == "stale"
        assert receipt.reason == "stale_head_sha"
        assert receipt.comment_id is not None

        # 2. Comment body must have been updated in-place with stale warning banner
        comment = client.comments[receipt.comment_id]
        body = comment["body"]
        assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in body
        assert "head_sha moved to ffffffff" in body
        # Line 1 still has marker
        assert body.splitlines()[0] == DEFAULT_A_MARKER

    def test_postcheck_race_pr_closed_marks_receipt_stale_and_updates_banner(self) -> None:
        class ClosedMockGitHubClient(MockGitHubClient):
            def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                record = super().create_comment(*args, **kwargs)
                # PR closed while review was being posted
                self.set_pull_request(
                    pr_number=2,
                    state="closed",
                    head_sha="b" * 40,
                    base_sha="a" * 40,
                )
                return record

        client = ClosedMockGitHubClient()
        client.set_pull_request(pr_number=2, state="open", head_sha="b" * 40, base_sha="a" * 40)

        receipt = publish_review_comment(
            github_client=client,
            repository_id=10,
            pr_number=2,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Review comment content",
            marker=DEFAULT_A_MARKER,
        )

        assert receipt.status == "stale"
        assert receipt.reason == "pr_closed_during_publication"
        assert receipt.comment_id is not None

        comment = client.comments[receipt.comment_id]
        body = comment["body"]
        assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in body
        assert "pr_closed" in body
        assert body.splitlines()[0] == DEFAULT_A_MARKER

    def test_postcheck_inspection_failure_preserves_comment_and_returns_failed_receipt(self) -> None:
        class FlakyPostCheckGitHubClient(MockGitHubClient):
            def __init__(self) -> None:
                super().__init__()
                self._get_pr_calls = 0

            def get_pull_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
                self._get_pr_calls += 1
                if self._get_pr_calls > 1:
                    # Post-check network timeout or error
                    raise OSError("GitHub connection reset during post-check")
                return super().get_pull_request(*args, **kwargs)

        client = FlakyPostCheckGitHubClient()
        client.set_pull_request(pr_number=3, state="open", head_sha="b" * 40, base_sha="a" * 40)

        receipt = publish_review_comment(
            github_client=client,
            repository_id=10,
            pr_number=3,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Review comment content",
        )

        # Publication happened, but a failed freshness check must not be reported as success.
        assert receipt.status == "failed"
        assert receipt.comment_id is not None
        assert receipt.reason == "post_check_failed: GitHub connection reset during post-check"
        assert receipt.published_at is not None
        assert len(client.comments) == 1
        assert client.comments[receipt.comment_id]["body"] == f"{DEFAULT_A_MARKER}\n\nReview comment content"


# ==============================================================================
# Dimension 3: Marker Idempotency & Duplicate Prevention
# ==============================================================================


class TestMarkerIdempotencyStress:
    """Stress-test marker detection, single comment invariant, and upsert idempotency."""

    def test_multiple_sequential_runs_never_create_duplicate_comments(self) -> None:
        client = MockGitHubClient()
        client.set_pull_request(pr_number=10, state="open", head_sha="b" * 40, base_sha="a" * 40)

        # Run 1: initial publication creates comment
        r1 = publish_review_comment(
            github_client=client,
            repository_id=100,
            pr_number=10,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Run 1 body",
        )
        assert r1.status == "succeeded"
        assert len(client.comments) == 1
        cid = r1.comment_id

        # Runs 2 through 10: subsequent updates must update the SAME comment
        for i in range(2, 11):
            rx = publish_review_comment(
                github_client=client,
                repository_id=100,
                pr_number=10,
                expected_head_sha="b" * 40,
                expected_base_sha="a" * 40,
                body=f"Run {i} updated body",
            )
            assert rx.status == "succeeded"
            assert rx.comment_id == cid
            # Absolute invariant: total comments on PR remains 1!
            assert len(client.comments) == 1
            assert client.comments[cid]["body"] == f"{DEFAULT_A_MARKER}\n\nRun {i} updated body"

    def test_bot_preserves_coexisting_user_comments(self) -> None:
        client = MockGitHubClient()
        client.set_pull_request(pr_number=20, state="open", head_sha="b" * 40, base_sha="a" * 40)

        # Simulate user comments
        user_c1 = client.create_comment(20, "LGTM from user 1")
        user_c2 = client.create_comment(20, "Please fix spacing")
        assert len(client.comments) == 2

        # Bot publishes
        r1 = publish_review_comment(
            github_client=client,
            repository_id=100,
            pr_number=20,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Review Bot Output v1",
        )
        assert r1.status == "succeeded"
        bot_cid = r1.comment_id
        assert len(client.comments) == 3

        # User adds another comment
        user_c3 = client.create_comment(20, "Done fixing")
        assert len(client.comments) == 4

        # Bot publishes again
        r2 = publish_review_comment(
            github_client=client,
            repository_id=100,
            pr_number=20,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Review Bot Output v2",
        )
        assert r2.status == "succeeded"
        assert r2.comment_id == bot_cid
        assert len(client.comments) == 4  # Still 4: 3 user comments + 1 bot comment

        # Verify user comments were not modified or overwritten
        assert client.comments[user_c1["id"]]["body"] == "LGTM from user 1"
        assert client.comments[user_c2["id"]]["body"] == "Please fix spacing"
        assert client.comments[user_c3["id"]]["body"] == "Done fixing"

    def test_ambiguous_multiple_marker_comments_fails_safely_without_modification(self) -> None:
        client = MockGitHubClient()
        client.set_pull_request(pr_number=30, state="open", head_sha="b" * 40, base_sha="a" * 40)

        # Inject two conflicting comments both containing the marker
        c1 = client.create_comment(30, f"{DEFAULT_A_MARKER}\nConflicting comment 1")
        c2 = client.create_comment(30, f"{DEFAULT_A_MARKER}\nConflicting comment 2")
        assert len(client.comments) == 2

        receipt = publish_review_comment(
            github_client=client,
            repository_id=100,
            pr_number=30,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="New review",
        )

        assert receipt.status == "failed"
        assert "ambiguous_marker_comment" in receipt.reason
        assert receipt.comment_id is None
        # Conflicting comments are unmodified
        assert client.comments[c1["id"]]["body"] == f"{DEFAULT_A_MARKER}\nConflicting comment 1"
        assert client.comments[c2["id"]]["body"] == f"{DEFAULT_A_MARKER}\nConflicting comment 2"

    def test_marker_normalization_on_body_without_marker(self) -> None:
        client = MockGitHubClient()
        client.set_pull_request(pr_number=40, state="open", head_sha="b" * 40, base_sha="a" * 40)

        # Body lacking marker
        receipt = publish_review_comment(
            github_client=client,
            repository_id=100,
            pr_number=40,
            expected_head_sha="b" * 40,
            expected_base_sha="a" * 40,
            body="Just a markdown report without marker header",
        )
        assert receipt.status == "succeeded"
        body = client.comments[receipt.comment_id]["body"]
        assert body.startswith(DEFAULT_A_MARKER)
        assert "Just a markdown report" in body

class TestPublicationConcurrencyAndEdgeCases:
    """Stress-test concurrent calls, pre-check rejections, and argument variations."""

    def test_precheck_rejections_prevent_any_comment_creation(self) -> None:
        client = MockGitHubClient()
        client.set_pull_request(pr_number=50, state="open", head_sha="b" * 40, base_sha="a" * 40)

        # 1. Closed PR
        client.set_pull_request(pr_number=50, state="closed")
        r_closed = publish_review_comment(client, 10, 50, "b" * 40, "a" * 40, "body")
        assert r_closed.status == "failed"
        assert r_closed.reason == "pr_not_open"
        assert len(client.comments) == 0

        # 2. Stale Head SHA
        client.set_pull_request(pr_number=50, state="open", head_sha="x" * 40, base_sha="a" * 40)
        r_head = publish_review_comment(client, 10, 50, "b" * 40, "a" * 40, "body")
        assert r_head.status == "failed"
        assert r_head.reason == "stale_head_sha"
        assert len(client.comments) == 0

        # 3. Stale Base SHA
        client.set_pull_request(pr_number=50, state="open", head_sha="b" * 40, base_sha="y" * 40)
        r_base = publish_review_comment(client, 10, 50, "b" * 40, "a" * 40, "body")
        assert r_base.status == "failed"
        assert r_base.reason == "stale_base_sha"
        assert len(client.comments) == 0

    def test_concurrent_threads_publishing_simulation(self) -> None:
        # In a real environment, workers run sequentially per PR via SQLite lease & fencing.
        # But if concurrent threads attempt publication on the same PR:
        client = MockGitHubClient()
        client.set_pull_request(pr_number=60, state="open", head_sha="b" * 40, base_sha="a" * 40)

        # Pre-seed one marker comment
        first_receipt = publish_review_comment(
            client, 100, 60, "b" * 40, "a" * 40, "Initial comment"
        )
        assert first_receipt.status == "succeeded"
        target_comment_id = first_receipt.comment_id

        # Now launch 10 concurrent threads to publish updates
        def worker_task(thread_idx: int):
            return publish_review_comment(
                client,
                100,
                60,
                "b" * 40,
                "a" * 40,
                f"Thread {thread_idx} update content",
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            receipts = list(executor.map(worker_task, range(10)))

        # All successful receipts must target the exact same comment ID
        succeeded = [r for r in receipts if r.status == "succeeded"]
        assert len(succeeded) > 0
        for r in succeeded:
            assert r.comment_id == target_comment_id

        # Total comments on PR must remain strictly 1
        assert len(client.comments) == 1
