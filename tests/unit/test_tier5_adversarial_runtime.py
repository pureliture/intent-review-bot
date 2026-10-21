"""Tier 5 Adversarial Coverage Hardening Suite for Runtime, Publication, and CLI.

White-box empirical challenge tests covering:
1. review_bot.runtime.cli:
   - Argparse edge cases, unknown options, invalid choices (exit code 2)
   - Zero arguments / help, version, backwards compatibility subcommands (exit code 0)
   - Subcommand failure modes (missing git repo, missing commits, asymmetric/corrupted delta runs) (exit code 1)
   - Review and delta output formats (markdown, table, json, directory vs file output)
2. review_bot.runtime.webhook:
   - FastAPI streaming limits (> 10 MiB) via Content-Length header and chunked streams
   - Deceptive Content-Length headers and stream byte gating
   - Malformed JSON bodies, non-UTF-8 bytes, and schema violations
   - Header permutations, missing headers, ping event, unhandled events
   - Timing attack defenses and HMAC validation
3. review_bot.runtime.worker:
   - Worker loop lifecycle, worker_id generation, run_once / claim_and_execute_once
   - Signal handling (SIGINT / SIGTERM) and graceful termination
   - Lease expiration & renewal failure during processing -> abort without mutation
   - Generation fencing edge cases across pipeline stages
   - Pre-publication freshness check failure -> SUPERSEDED transition
   - Unhandled exception recovery -> FAILED transition without crashing daemon
4. review_bot.adapters.markdown.formatter:
   - 60,000 character boundary truncations with intact footers and omission banners
   - Syntax preservation: balancing unclosed <pre><code> tags and unclosed code fences ```
   - HTML escaping, character entities, and @-mention sanitization (@\u200b)
   - Empty sections handling (no findings, no questions, uncomparable analysis)
   - Stale warning banner injection
   - Actions artifact URL validation (https enforcement, credential stripping, length limits)
   - _a_link safe path and traversal defense
5. review_bot.application.publication:
   - Pre-check freshness verification (PR closed, stale head SHA, stale base SHA)
   - Marker discovery ambiguity (multiple marker comments) and error handling
   - Post-check race conditions: PR closed mid-write -> stale warning banner update
   - Post-check race conditions: PR head moved mid-write -> stale warning banner update
   - Multi-threaded concurrent publication race stress test
"""
from __future__ import annotations

import hashlib
import hmac
import json
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from review_bot import __version__
from review_bot.adapters.github.client import MockGitHubClient
from review_bot.adapters.github.webhook import verify_github_signature
from review_bot.adapters.markdown.formatter import (
    DEFAULT_A_MARKER,
    _a_artifact_link,
    _a_escape,
    _a_link,
    format_review_markdown,
)
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.application.publication import publish_review_comment
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import PrRequestRecord
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
from review_bot.domain.status import RunStatus
from review_bot.runtime.cli import create_parser, main
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop
from tests.conftest import SyntheticGitRepo

# ==============================================================================
# Helpers & Fixtures
# ==============================================================================

WEBHOOK_SECRET = "tier5_adversarial_test_secret_key_123"


def _sign(payload: bytes, secret: str = WEBHOOK_SECRET) -> str:
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _make_dummy_task_and_result(
    *,
    summary: str = "Analysis summary",
    findings: tuple[ReviewFinding, ...] = (),
    questions: tuple[ReviewQuestion, ...] = (),
    signals: tuple[DeltaSignal, ...] = (),
    comparable: bool = True,
) -> tuple[ReviewResult, TaskInput]:
    pr = PrSnapshot(
        repository="owner/repo",
        number=42,
        base_sha="a" * 40,
        head_sha="b" * 40,
        source_branch="feature/tier5",
        changed_files=("src/Main.java",),
    )
    delta = AnalysisDelta(
        schema_version=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        conditions_hash="ch1",
        comparable=comparable,
        signals=signals,
    )
    base_code = CodeSnapshot(
        sha="a" * 40,
        files=(CodeFile(path="src/Main.java", content="class Main {}", content_hash="h1", kind="source"),),
        exclusions=(),
        complete=True,
        errors=(),
        inventory_hash="ih1",
    )
    head_code = CodeSnapshot(
        sha="b" * 40,
        files=(CodeFile(path="src/Main.java", content="class Main { int x; }", content_hash="h2", kind="source"),),
        exclusions=(),
        complete=True,
        errors=(),
        inventory_hash="ih2",
    )
    snap = Snapshot(pr=pr, base=base_code, head=head_code)
    ev = Evidence(
        evidence_id="ev-0",
        kind="code",
        version="head",
        sha="b" * 40,
        path="src/Main.java",
        begin_line=1,
        end_line=10,
        content="class Main { int x; }",
    )
    selected = tuple(SelectedSignal(signal=s, role="review", reason="test") for s in signals)
    scope = {
        "code": {
            "base": {"acquired": {"file_count": 1, "byte_count": 50}, "provided": {"file_count": 1, "byte_count": 50}, "excluded": {"file_count": 0}, "complete": True},
            "head": {"acquired": {"file_count": 1, "byte_count": 50}, "provided": {"file_count": 1, "byte_count": 50}, "excluded": {"file_count": 0}, "complete": True},
        }
    }
    classification = Classification(review_mode="general", reasons=("default",))
    task_input = TaskInput(
        schema_version=1,
        snapshot=snap,
        delta=delta,
        classification=classification,
        signals=selected,
        evidence=(ev,),
        scope=scope,
        limitations=(),
        input_hash="ih1",
    )
    result = ReviewResult(
        schema_version=1,
        review_type="A",
        run_key="run-1",
        base_sha="a" * 40,
        head_sha="b" * 40,
        review_mode="general",
        pr=pr,
        status="completed",
        reason="validated",
        summary=summary,
        findings=findings,
        questions=questions,
        validation_errors=(),
        scope={},
        opinion_scope={"signal_ids": [s.signal_id for s in signals], "paths": ["src/Main.java"]},
        analysis=delta,
        model={"is_mock": True},
        usage={"tokens": 100},
        created_at="2026-10-08T00:00:00Z",
        prompt_version="review-a/v1",
        input_hash="ih1",
    )
    return result, task_input


# ==============================================================================
# 1. CLI White-Box Adversarial Hardening
# ==============================================================================


class TestCliAdversarialHardening:
    """Argparse edge cases, subcommand combinations, invalid flags, and exit codes 0, 1, 2."""

    def test_cli_exit_code_0_on_help_and_no_subcommand(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Calling main([]) with no arguments prints help and returns 0."""
        code = main([])
        assert code == 0
        captured = capsys.readouterr()
        assert "Standalone GitHub Architecture Review Bot CLI" in captured.out
        assert "Available subcommands" in captured.out

    def test_cli_exit_code_2_on_invalid_flags(self) -> None:
        """Argparse exits with code 2 on unknown flags or invalid options."""
        parser = create_parser()

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--bogus-option"])
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["review", "--nonexistent-flag"])
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["review", "--format", "yaml"])
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["delta", "--format", "binary"])
        assert exc_info.value.code == 2

        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["webhook", "--port", "not-a-port"])
        assert exc_info.value.code == 2

    def test_cli_version_and_compat_subcommands(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Version and backward-compatible subcommands exit with code 0."""
        assert main(["version"]) == 0
        captured = capsys.readouterr()
        assert f"review-bot {__version__}" in captured.out

        assert main(["version", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["version"] == __version__
        assert "tools_lock" in data

        # Subcommands returning 0 when initialized without action
        assert main(["offline"]) == 0
        assert "offline' initialized." in capsys.readouterr().out

        assert main(["worker"]) == 1
        assert "require --config" in capsys.readouterr().err

        assert main(["webhook"]) == 1
        assert "require --config" in capsys.readouterr().err

    def test_cli_analyze_exit_code_1_on_non_git_or_missing_commit(
        self, tmp_path: Path, synthetic_repo: SyntheticGitRepo, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Analyze exits with 1 on non-git dir and non-existent commit."""
        # Non-git repository
        not_git = tmp_path / "not_git"
        not_git.mkdir()
        code = main(["analyze", "--repo", str(not_git)])
        assert code == 1
        captured = capsys.readouterr()
        assert "is not a git repository" in captured.err

        # Missing commit in valid repository
        missing_sha = "0123456789abcdef0123456789abcdef01234567"
        code = main(["analyze", "--repo", str(synthetic_repo.repo_dir), "--commit", missing_sha])
        assert code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_cli_delta_asymmetric_arguments_exit_code_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Providing only --base-run or only --head-run exits with 1."""
        code1 = main(["delta", "--base-run", "base.json"])
        assert code1 == 1
        assert "Both --base-run and --head-run must be provided together" in capsys.readouterr().err

        code2 = main(["delta", "--head-run", "head.json"])
        assert code2 == 1
        assert "Both --base-run and --head-run must be provided together" in capsys.readouterr().err

    def test_cli_delta_corrupted_and_missing_files_exit_code_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Corrupted JSON, missing files, or bad schema in delta run files exit with code 1."""
        # Missing file
        missing1 = tmp_path / "missing1.json"
        missing2 = tmp_path / "missing2.json"
        code = main(["delta", "--base-run", str(missing1), "--head-run", str(missing2)])
        assert code == 1
        assert "Failed to read analysis run file" in capsys.readouterr().err

        # Syntax error JSON
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ unquoted string ]", encoding="utf-8")
        valid_json = tmp_path / "valid.json"
        valid_json.write_text("{}", encoding="utf-8")

        code = main(["delta", "--base-run", str(bad_json), "--head-run", str(valid_json)])
        assert code == 1
        assert "Malformed JSON in run file" in capsys.readouterr().err

        # Bad schema
        invalid_type_json = tmp_path / "invalid_type.json"
        invalid_type_json.write_text(json.dumps({"pmd": "should_be_dict_not_str"}), encoding="utf-8")
        code = main(["delta", "--base-run", str(invalid_type_json), "--head-run", str(valid_json)])
        assert code == 1
        assert "Invalid analysis run data" in capsys.readouterr().err

    def test_cli_delta_format_table_and_markdown(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Delta format 'table' and 'markdown' produce clean markdown tables."""
        run_dict = {
            "sha": "a" * 40,
            "status": "clean",
            "pmd": {"report": {"violations": []}},
            "cpd": {"report": {"duplications": []}},
        }
        f1 = tmp_path / "run1.json"
        f2 = tmp_path / "run2.json"
        f1.write_text(json.dumps(run_dict), encoding="utf-8")
        f2.write_text(json.dumps(run_dict), encoding="utf-8")

        out_file = tmp_path / "delta_table.md"
        code = main([
            "delta",
            "--base-run", str(f1),
            "--head-run", str(f2),
            "--format", "table",
            "--output", str(out_file),
        ])
        assert code == 0
        content = out_file.read_text(encoding="utf-8")
        assert "| existing | new | worsened | improved | resolved | unknown |" in content
        assert "| 0 | 0 | 0 | 0 | 0 | 0 |" in content


# ==============================================================================
# 2. Webhook Streaming Limits, Headers & Timing Attack Defenses
# ==============================================================================


class TestWebhookAdversarialHardening:
    """FastAPI streaming limits (> 10 MiB), malformed JSON bodies, header permutations, timing attacks."""

    @pytest.fixture
    def webhook_env(self, tmp_path: Path, valid_config_yaml: str) -> tuple[TestClient, SQLiteJobStore, Any]:
        store = SQLiteJobStore(tmp_path / "webhook_adv.db")
        config, _ = load_review_config_from_str(valid_config_yaml)
        app = create_webhook_app(config, store, secret=WEBHOOK_SECRET)
        return TestClient(app), store, config

    def test_streaming_limit_via_content_length_header(
        self, webhook_env: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        """Request with Content-Length > 10 MiB (10485761 bytes) is rejected with HTTP 413 immediately."""
        client, _, _ = webhook_env
        headers = {
            "Content-Length": "10485761",
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "del-huge-header",
        }
        resp = client.post("/webhook", content=b"x", headers=headers)
        assert resp.status_code == 413
        assert "Payload Too Large" in resp.text

    def test_streaming_limit_via_chunked_body_stream(
        self, webhook_env: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        """Deceptive Content-Length (small) or chunked stream exceeding 10 MiB is terminated with HTTP 413."""
        client, _, _ = webhook_env
        chunk_size = 1024 * 1024  # 1 MiB

        def chunk_generator():
            for _ in range(11):
                yield b"0" * chunk_size

        headers = {
            "Content-Length": "100",  # Deceptive header
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "del-chunked-overflow",
        }
        resp = client.post("/webhook", content=chunk_generator(), headers=headers)
        assert resp.status_code == 413
        assert "Payload Too Large" in resp.text

    def test_malformed_json_and_non_utf8_bodies(
        self, webhook_env: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        """Non-UTF-8 bytes or malformed JSON payloads return HTTP 400."""
        client, _, _ = webhook_env
        bad_bytes = b"\xff\xfe\x00\x12"
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "del-non-utf8",
            "X-Hub-Signature-256": _sign(bad_bytes),
        }
        resp = client.post("/webhook", content=bad_bytes, headers=headers)
        assert resp.status_code == 400
        assert "Malformed JSON payload" in resp.text

        bad_json = b"{'key': unquoted_value}"
        headers["X-Hub-Signature-256"] = _sign(bad_json)
        resp2 = client.post("/webhook", content=bad_json, headers=headers)
        assert resp2.status_code == 400
        assert "Malformed JSON payload" in resp2.text

    def test_header_permutations_missing_and_ping(
        self, webhook_env: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        """Missing required headers return 400; ping event returns 200 pong."""
        client, _, _ = webhook_env
        body = json.dumps({"zen": "Keep it logically awesome."}).encode("utf-8")
        sig = _sign(body)

        # Missing delivery ID
        resp1 = client.post("/webhook", content=body, headers={"X-GitHub-Event": "ping", "X-Hub-Signature-256": sig})
        assert resp1.status_code == 400
        assert "Missing required headers" in resp1.text

        # Missing event name
        resp2 = client.post("/webhook", content=body, headers={"X-GitHub-Delivery": "del-1", "X-Hub-Signature-256": sig})
        assert resp2.status_code == 400
        assert "Missing required headers" in resp2.text

        # Ping event
        resp3 = client.post(
            "/webhook",
            content=body,
            headers={"X-GitHub-Event": "ping", "X-GitHub-Delivery": "del-ping", "X-Hub-Signature-256": sig},
        )
        assert resp3.status_code == 200
        assert resp3.json() == {"status": "pong"}

    def test_hmac_timing_attack_defense_and_tampering(
        self, webhook_env: tuple[TestClient, SQLiteJobStore, Any]
    ) -> None:
        """verify_github_signature uses constant-time comparison and rejects any tampering."""
        body = b'{"action": "ping"}'
        valid_sig = _sign(body)

        assert verify_github_signature(body, valid_sig, WEBHOOK_SECRET) is True

        # Tampered byte in signature
        tampered_sig = valid_sig[:-1] + ("0" if valid_sig[-1] != "0" else "1")
        assert verify_github_signature(body, tampered_sig, WEBHOOK_SECRET) is False

        # Missing sha256= prefix
        no_prefix = valid_sig.replace("sha256=", "")
        assert verify_github_signature(body, no_prefix, WEBHOOK_SECRET) is False

        # Wrong secret
        assert verify_github_signature(body, valid_sig, "wrong_secret") is False

        # Empty secret / empty signature
        assert verify_github_signature(body, None, WEBHOOK_SECRET) is False
        assert verify_github_signature(body, valid_sig, "") is False


# ==============================================================================
# 3. Worker Loop, Lease Expiry, Generation Fencing & Signals
# ==============================================================================


class TestWorkerAdversarialHardening:
    """Worker loop lifecycle, lease expiration under simulated delays, generation fencing, SIGINT/SIGTERM."""

    @pytest.fixture
    def worker_setup(self, tmp_path: Path, valid_config_yaml: str) -> tuple[SQLiteJobStore, Any, MockGitHubClient]:
        store = SQLiteJobStore(tmp_path / "worker_adv.db")
        config, _ = load_review_config_from_str(valid_config_yaml)
        config.publish.mode = "comment"
        github = MockGitHubClient()
        return store, config, github

    def test_worker_id_and_empty_queue_claim(
        self, worker_setup: tuple[SQLiteJobStore, Any, MockGitHubClient]
    ) -> None:
        """Worker generates unique ID and returns False when queue is empty."""
        store, config, github = worker_setup
        worker = WorkerLoop(config=config, job_store=store, github_client=github)
        assert worker.worker_id is not None
        assert len(worker.worker_id) > 0
        assert worker.claim_and_execute_once() is False
        assert worker.run_once() is False

    def test_worker_signal_handlers_and_graceful_stop(
        self, worker_setup: tuple[SQLiteJobStore, Any, MockGitHubClient]
    ) -> None:
        """Worker handles SIGINT/SIGTERM and stops loop gracefully."""
        store, config, github = worker_setup
        worker = WorkerLoop(config=config, job_store=store, github_client=github)

        assert not worker._stop_event.is_set()
        worker._handle_signal(signal.SIGINT, None)
        assert worker._stop_event.is_set()

        worker2 = WorkerLoop(config=config, job_store=store, github_client=github)
        worker2.stop()
        assert worker2._stop_event.is_set()

    def test_lease_expiration_during_execution_aborts_without_mutation(
        self, worker_setup: tuple[SQLiteJobStore, Any, MockGitHubClient]
    ) -> None:
        """When lease renewal fails, worker aborts without changing durable run state."""
        store, config, github = worker_setup
        github.set_pull_request(pr_number=55, state="open", head_sha="h1" * 20, base_sha="b1" * 20)

        # Slow down PR fetch to let heartbeat fire
        def _slow_get_pr(*args: Any, **kwargs: Any):
            time.sleep(0.15)
            return {
                "number": 55,
                "state": "open",
                "head": {"sha": "h1" * 20},
                "base": {"sha": "b1" * 20},
            }
        github.get_pull_request = _slow_get_pr  # type: ignore

        req = PrRequestRecord(installation_id=1, repository_id=100, pr_number=55, desired_generation=1)
        store.enqueue_request(req)
        claimed = store.claim_next_run(worker_id="worker-test", lease_seconds=10.0)
        assert claimed is not None

        worker = WorkerLoop(
            config=config, job_store=store, github_client=github,
            worker_id="worker-test", lease_duration=0.1
        )

        # Mock renew_lease to return False, simulating lease loss
        mock_store = MagicMock(wraps=store)
        mock_store.renew_lease.return_value = False
        worker.job_store = mock_store
        conn = store._get_connection()
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (claimed.run_id,)).fetchone()

        result = worker.process_run(claimed)
        assert result is False

        mock_store.update_run.assert_not_called()
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (claimed.run_id,)).fetchone() == before

    def test_generation_fencing_aborts_stale_run(
        self, worker_setup: tuple[SQLiteJobStore, Any, MockGitHubClient]
    ) -> None:
        """Advancing desired generation triggers generation fencing and aborts run."""
        store, config, github = worker_setup
        github.set_pull_request(pr_number=66, state="open", head_sha="h1" * 20, base_sha="b1" * 20)

        req = PrRequestRecord(installation_id=1, repository_id=200, pr_number=66, desired_generation=1)
        store.enqueue_request(req)
        claimed = store.claim_next_run(worker_id="worker-fence", lease_seconds=30.0)
        assert claimed is not None

        # Advance desired generation for PR in job store to simulate a newer push
        req2 = PrRequestRecord(installation_id=1, repository_id=200, pr_number=66, desired_generation=2)
        store.enqueue_request(req2)

        worker = WorkerLoop(config=config, job_store=store, github_client=github, worker_id="worker-fence")
        result = worker.process_run(claimed)
        assert result is False

        conn = store._get_connection()
        row = conn.execute("SELECT status FROM runs WHERE run_id = ?", (claimed.run_id,)).fetchone()
        assert row is not None
        assert row[0] == "superseded"

    def test_worker_pre_publication_freshness_failure(
        self, worker_setup: tuple[SQLiteJobStore, Any, MockGitHubClient],
        worker_pipeline_ports: dict[str, Any],
    ) -> None:
        """If PR head SHA moved right before publication, worker marks run SUPERSEDED."""
        store, config, github = worker_setup
        original_head = "b" * 40
        moved_head = "9999999999999999999999999999999999999999"

        call_count = 0
        def _get_pr_mock(pr_number: int):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "number": pr_number,
                    "state": "open",
                    "head": {"sha": original_head},
                    "base": {"sha": "c" * 40},
                }
            return {
                "number": pr_number,
                "state": "open",
                "head": {"sha": moved_head},
                "base": {"sha": "c" * 40},
            }
        github.get_pull_request = _get_pr_mock  # type: ignore

        req = PrRequestRecord(installation_id=1, repository_id=300, pr_number=77, desired_generation=1)
        store.enqueue_request(req)
        claimed = store.claim_next_run(worker_id="worker-fresh", lease_seconds=30.0)
        assert claimed is not None

        worker = WorkerLoop(
            config=config, job_store=store, github_client=github, worker_id="worker-fresh",
            **worker_pipeline_ports,
        )
        result = worker.process_run(claimed)
        assert result is False

        conn = store._get_connection()
        row = conn.execute("SELECT status, error FROM runs WHERE run_id = ?", (claimed.run_id,)).fetchone()
        assert row is not None
        assert row[0] == "superseded"
        assert "Freshness check failed" in (row[1] or "")

    def test_worker_unhandled_exception_marks_failed(
        self, worker_setup: tuple[SQLiteJobStore, Any, MockGitHubClient]
    ) -> None:
        """Unexpected exception in stage marks run FAILED without terminating daemon."""
        store, config, github = worker_setup
        req = PrRequestRecord(installation_id=1, repository_id=400, pr_number=88, desired_generation=1)
        store.enqueue_request(req)
        claimed = store.claim_next_run(worker_id="worker-err", lease_seconds=30.0)
        assert claimed is not None

        github.get_pull_request = MagicMock(side_effect=RuntimeError("GitHub API crashed"))  # type: ignore

        worker = WorkerLoop(config=config, job_store=store, github_client=github, worker_id="worker-err")
        assert worker.process_run(claimed) is False

        conn = store._get_connection()
        row = conn.execute("SELECT status, error FROM runs WHERE run_id = ?", (claimed.run_id,)).fetchone()
        assert row is not None
        assert row[0] == "failed"
        assert row[1] == "worker_pipeline_error:RuntimeError"
        assert "GitHub API crashed" not in row[1]


# ==============================================================================
# 4. Markdown Formatter 60k Truncation, Fences & HTML Sanitization
# ==============================================================================


class TestMarkdownFormatterAdversarialHardening:
    """60,000 char boundary truncations, deeply nested code blocks, unclosed HTML tags, empty sections."""

    def test_formatting_within_60k_boundary_returns_intact(self) -> None:
        """Normal comment within 60,000 characters is untouched."""
        result, task_input = _make_dummy_task_and_result(summary="Clean architectural refactor")
        md = format_review_markdown(result, task_input, max_length=60000)
        assert len(md) <= 60000
        assert md.startswith("<!-- review-bot:architecture:v1 -->")
        assert "### PMD/CPD 아키텍처 리뷰 A" in md
        assert "Clean architectural refactor" in md

    def test_truncation_enforces_60k_hard_limit_on_massive_payload(self) -> None:
        """Massive payload exceeding 100,000 and 250,000 chars is safely truncated <= 60,000 chars."""
        v = PmdViolation(
            path="src/Huge.java", raw_path="src/Huge.java",
            begin_line=1, end_line=100, begin_column=1, end_column=1,
            rule="ComplexClass", ruleset="rules.xml", priority=1,
            klass="Huge", method="foo", package="pkg", message="Massive class",
        )
        sig = DeltaSignal(
            signal_id="sig-huge", tool="pmd", change="worsened",
            base_findings=(v,), head_findings=(v,), paths=("src/Huge.java",),
            matching="sig", measurements=(),
        )
        citations = tuple(
            Citation(
                evidence_id=f"ev-{i}", version="head", sha="b" * 40,
                path="src/Huge.java", begin_line=1, end_line=100,
                quote=f"public void massiveMethod_{i}() {{\n" + ("    // long comment\n" * 50) + "}",
            )
            for i in range(8)
        )
        findings = tuple(
            ReviewFinding(
                signal_id="sig-huge",
                judgement="concern",
                change="worsened",
                basis="High complexity and coupling",
                title=f"Excessive architectural responsibility {i}",
                reason=f"Detailed reason explanation for finding {i} with massive analysis",
                recommendation="Split into smaller components",
                evidence_ids=("ev-0",),
                citations=citations,
            )
            for i in range(20)
        )

        result, task_input = _make_dummy_task_and_result(
            summary="Huge review with lots of findings",
            findings=findings,
            signals=(sig,),
        )

        rendered = format_review_markdown(result, task_input, max_length=60000)
        assert len(rendered) <= 60000
        assert rendered.startswith("<!-- review-bot:architecture:v1 -->")
        assert "리뷰 본문 표시 보류:" in rendered
        assert "검토 범위: 고정된 HEAD" in rendered
        assert "Excessive architectural responsibility" not in rendered
        assert rendered.count("<details>") == rendered.count("</details>")

    def test_syntax_preservation_and_code_fence_balancing(self) -> None:
        """Truncation balancing: unclosed <pre><code> and triple backticks are safely terminated."""
        v = PmdViolation(
            path="src/A.java", raw_path="src/A.java",
            begin_line=1, end_line=10, begin_column=1, end_column=1,
            rule="R", ruleset="r", priority=1, klass="A", method="m", package="p", message="msg",
        )
        sig = DeltaSignal(
            signal_id="sig-1", tool="pmd", change="new",
            base_findings=(), head_findings=(v,), paths=("src/A.java",), matching="m", measurements=(),
        )
        citations = (
            Citation(
                evidence_id="ev-code", version="head", sha="b" * 40,
                path="src/A.java", begin_line=1, end_line=10,
                quote="```java\nclass CodeSnippet {\n" + ("    int padding = 1;\n" * 300) + "}\n```",
            ),
        )
        findings = tuple(
            ReviewFinding(
                signal_id="sig-1",
                judgement="concern",
                change="new",
                basis="basis",
                title=f"Finding {i}",
                reason="reason " * 50,
                recommendation="rec",
                evidence_ids=("ev-code",),
                citations=citations,
            )
            for i in range(15)
        )
        result, task_input = _make_dummy_task_and_result(findings=findings, signals=(sig,))

        rendered = format_review_markdown(result, task_input, max_length=15000)
        assert len(rendered) <= 15000

        pre_code_opens = rendered.count("<pre><code>")
        pre_code_closes = rendered.count("</code></pre>")
        assert pre_code_opens == pre_code_closes
        assert "리뷰 본문 표시 보류:" in rendered
        assert "Finding 0" not in rendered

        # A fitting review still contains the complete quoted block with safe syntax.
        fitting, fitting_input = _make_dummy_task_and_result(findings=(findings[0],), signals=(sig,))
        complete = format_review_markdown(fitting, fitting_input, max_length=15000)
        assert len(complete) <= 15000 and "Finding 0" in complete
        assert complete.count("<pre><code>") == complete.count("</code></pre>") == 1
        assert complete.count("<details>") == complete.count("</details>")
        assert "```" not in complete

    def test_mention_sanitization_and_html_escaping(self) -> None:
        """@-mentions are neutralized with zero-width space (@\\u200b) to avoid GitHub user pings."""
        escaped = _a_escape("@octocat @everyone @here alert!")
        assert "@\u200b" in escaped
        assert "@octocat" not in escaped
        assert "@everyone" not in escaped

    def test_a_link_safe_path_and_traversal_defense(self) -> None:
        """_a_link rejects dot-dot traversal paths and invalid SHAs, rendering safe escaped labels."""
        valid_link = _a_link("owner/repo", "a" * 40, "src/Main.java", 1, 10)
        assert "https://github.com/owner/repo/blob/" in valid_link

        # Traversal in path
        traversal_link = _a_link("owner/repo", "a" * 40, "../secret.txt", 1, 10)
        assert "https://" not in traversal_link

        # Traversal in repo
        traversal_repo = _a_link("../owner/repo", "a" * 40, "src/Main.java", 1, 10)
        assert "https://" not in traversal_repo

        # Invalid short SHA
        bad_sha = _a_link("owner/repo", "shortsha", "src/Main.java", 1, 10)
        assert "https://" not in bad_sha

    def test_stale_warning_banner_injection(self) -> None:
        """A stale result displays its freshness failure without a judgement body."""
        result, task_input = _make_dummy_task_and_result(summary="Stale summary")
        md = format_review_markdown(result, task_input, stale_reason="head_sha moved to deadbeef")
        assert "리뷰 미완료" in md
        assert "최신 코드와 일치하지 않는 결과" in md
        assert "deadbeef" in md
        assert "Stale summary" not in md and "#### LLM 리뷰" not in md

    def test_actions_artifact_url_validation_adversarial(self) -> None:
        """_a_artifact_link strictly requires credential-free HTTPS address <= 4096 chars."""
        valid_url = "https://github.com/owner/repo/actions/runs/123/artifacts/456"
        assert f"[전체 A 자료]({valid_url})" == _a_artifact_link(valid_url)

        with pytest.raises(ValueError, match="credential-free HTTPS address"):
            _a_artifact_link("http://insecure.site/artifact")

        with pytest.raises(ValueError, match="credential-free HTTPS address"):
            _a_artifact_link("https://user:password@github.com/artifact")

        huge_url = "https://github.com/" + ("a" * 4100)
        with pytest.raises(ValueError, match="credential-free HTTPS address"):
            _a_artifact_link(huge_url)

        with pytest.raises(ValueError, match="credential-free HTTPS address"):
            _a_artifact_link("https://github.com/path\x00malicious")


# ==============================================================================
# 5. Publication Concurrency, PR Closure & Race Conditions
# ==============================================================================


class TestPublicationAdversarialHardening:
    """Pre-check verification, mid-write race conditions (PR closed, head moved), and concurrency."""

    @pytest.fixture
    def github_client(self) -> MockGitHubClient:
        client = MockGitHubClient()
        client.set_pull_request(
            pr_number=10,
            state="open",
            head_sha="h" * 40,
            base_sha="b" * 40,
            repository="owner/repo",
        )
        return client

    def test_publication_pre_check_closed_pr_fails(self, github_client: MockGitHubClient) -> None:
        """Publication pre-check detects closed PR and fails immediately."""
        github_client.set_pull_request(pr_number=10, state="closed", head_sha="h" * 40, base_sha="b" * 40)
        receipt = publish_review_comment(
            github_client=github_client,
            repository_id=1,
            pr_number=10,
            expected_head_sha="h" * 40,
            expected_base_sha="b" * 40,
            body="Review comment",
        )
        assert receipt.status == "failed"
        assert receipt.reason == "pr_not_open"

    def test_publication_pre_check_stale_head_sha_fails(self, github_client: MockGitHubClient) -> None:
        """Publication pre-check detects mismatched head SHA and fails immediately."""
        receipt = publish_review_comment(
            github_client=github_client,
            repository_id=1,
            pr_number=10,
            expected_head_sha="mismatch" + "0" * 32,
            expected_base_sha="b" * 40,
            body="Review comment",
        )
        assert receipt.status == "failed"
        assert receipt.reason == "stale_head_sha"

    def test_publication_pre_check_stale_base_sha_fails(self, github_client: MockGitHubClient) -> None:
        """Publication pre-check detects mismatched base SHA and fails immediately."""
        receipt = publish_review_comment(
            github_client=github_client,
            repository_id=1,
            pr_number=10,
            expected_head_sha="h" * 40,
            expected_base_sha="mismatch" + "0" * 32,
            body="Review comment",
        )
        assert receipt.status == "failed"
        assert receipt.reason == "stale_base_sha"

    def test_publication_ambiguous_multiple_markers_fails(self, github_client: MockGitHubClient) -> None:
        """If multiple comments with marker exist, publication fails with ambiguous marker error."""
        github_client.create_comment(10, f"{DEFAULT_A_MARKER}\n\nComment 1")
        github_client.create_comment(10, f"{DEFAULT_A_MARKER}\n\nComment 2")

        receipt = publish_review_comment(
            github_client=github_client,
            repository_id=1,
            pr_number=10,
            expected_head_sha="h" * 40,
            expected_base_sha="b" * 40,
            body="New comment",
        )
        assert receipt.status == "failed"
        assert "ambiguous_marker_comment" in (receipt.reason or "")

    def test_publication_mid_write_pr_closed_injects_stale_banner(
        self, github_client: MockGitHubClient
    ) -> None:
        """PR closed during API write triggers post-check detection and injects stale warning banner."""
        call_count = 0

        def dynamic_get_pr(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "number": 10,
                    "state": "open",
                    "head": {"sha": "h" * 40},
                    "base": {"sha": "b" * 40},
                }
            return {
                "number": 10,
                "state": "closed",
                "head": {"sha": "h" * 40},
                "base": {"sha": "b" * 40},
            }

        github_client.get_pull_request = dynamic_get_pr  # type: ignore

        receipt = publish_review_comment(
            github_client=github_client,
            repository_id=1,
            pr_number=10,
            expected_head_sha="h" * 40,
            expected_base_sha="b" * 40,
            body="Review body content",
        )
        assert receipt.status == "stale"
        assert receipt.reason == "pr_closed_during_publication"
        assert receipt.comment_id is not None

        updated_comment = github_client.comments[receipt.comment_id]
        assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in updated_comment["body"]
        assert "게시 중 PR 상태가 변경되었습니다: pr_closed" in updated_comment["body"]

    def test_publication_mid_write_head_moved_injects_stale_banner(
        self, github_client: MockGitHubClient
    ) -> None:
        """Head SHA changed during API write triggers post-check detection and injects stale warning banner."""
        new_head = "newhead99999999999999999999999999999999"
        call_count = 0

        def dynamic_get_pr(*args: Any, **kwargs: Any):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {
                    "number": 10,
                    "state": "open",
                    "head": {"sha": "h" * 40},
                    "base": {"sha": "b" * 40},
                }
            return {
                "number": 10,
                "state": "open",
                "head": {"sha": new_head},
                "base": {"sha": "b" * 40},
            }

        github_client.get_pull_request = dynamic_get_pr  # type: ignore

        receipt = publish_review_comment(
            github_client=github_client,
            repository_id=1,
            pr_number=10,
            expected_head_sha="h" * 40,
            expected_base_sha="b" * 40,
            body="Review body content",
        )
        assert receipt.status == "stale"
        assert receipt.reason == "stale_head_sha"
        assert receipt.comment_id is not None

        updated_comment = github_client.comments[receipt.comment_id]
        assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in updated_comment["body"]
        assert f"head_sha moved to {new_head[:8]}" in updated_comment["body"]

    def test_concurrent_publications_stress_test(self, github_client: MockGitHubClient) -> None:
        """10 concurrent threads publishing to same PR do not crash or corrupt comments."""
        def _publish_worker(thread_idx: int):
            return publish_review_comment(
                github_client=github_client,
                repository_id=1,
                pr_number=10,
                expected_head_sha="h" * 40,
                expected_base_sha="b" * 40,
                body=f"Comment from thread {thread_idx}",
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            receipts = list(executor.map(_publish_worker, range(10)))

        for r in receipts:
            assert r.status in ("succeeded", "failed")
            assert r.marker == DEFAULT_A_MARKER
