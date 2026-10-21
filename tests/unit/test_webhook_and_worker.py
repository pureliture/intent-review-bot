"""Unit tests for FastAPI Webhook Receiver and WorkerLoop runtime."""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import tempfile
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from starlette.testclient import TestClient

from review_bot.adapters.git import GitRepositoryReader
from review_bot.adapters.github.client import MockGitHubClient as ApiMockGitHubClient
from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.application.delta import build_delta
from review_bot.application.snapshot import assemble_snapshot_bundle
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import ChangedFile, PrRequestRecord, PublicationReceipt, RunRecord
from review_bot.domain.models_v1 import ModelResponse, ReviewResult, TaskInput
from review_bot.domain.status import RunStage, RunStatus
from review_bot.ports import GitHubClient, MergeBaseResult, ModelClient, RepositoryReader, StaticAnalyzer
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop
from tests.conftest import MockGitHubClient, SyntheticGitRepo

SECRET = "test_webhook_secret_xyz"


@pytest.fixture(name="valid_config_yaml")
def comment_config_yaml(valid_config_yaml: str) -> str:
    """These publication/fencing tests explicitly exercise comment mode."""
    return valid_config_yaml.replace("mode: artifact_only", "mode: comment")


def make_sig(body: bytes, secret: str = SECRET) -> str:
    h = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={h}"


@pytest.fixture
def webhook_client(tmp_path: Path, valid_config_yaml: str) -> tuple[TestClient, SQLiteJobStore]:
    db_file = tmp_path / "webhook_test.db"
    store = SQLiteJobStore(db_file)
    config, _ = load_review_config_from_str(valid_config_yaml)
    app = create_webhook_app(config, store, secret=SECRET)
    client = TestClient(app)
    return client, store


class TestWebhookEndpoints:
    def test_health_check(self, webhook_client: tuple[TestClient, SQLiteJobStore]) -> None:
        client, _ = webhook_client
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_missing_required_headers(self, webhook_client: tuple[TestClient, SQLiteJobStore]) -> None:
        client, _ = webhook_client
        payload = json.dumps({"action": "opened"}).encode()
        # Missing X-GitHub-Event and X-GitHub-Delivery
        resp = client.post("/webhook", content=payload, headers={"Content-Type": "application/json"})
        assert resp.status_code == 400

    def test_hmac_unauthorized_if_invalid_or_missing(
        self, webhook_client: tuple[TestClient, SQLiteJobStore]
    ) -> None:
        client, _ = webhook_client
        payload = json.dumps({"action": "opened"}).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-1",
            "Content-Type": "application/json",
        }
        # No signature header -> 401
        resp = client.post("/webhook", content=payload, headers=headers)
        assert resp.status_code == 401

        # Invalid signature header -> 401
        headers["X-Hub-Signature-256"] = "sha256=invalid_hex_digits"
        resp = client.post("/webhook", content=payload, headers=headers)
        assert resp.status_code == 401

    def test_ping_event_handling(self, webhook_client: tuple[TestClient, SQLiteJobStore]) -> None:
        client, _ = webhook_client
        payload = json.dumps({"zen": "Keep it logically awesome."}).encode()
        headers = {
            "X-GitHub-Event": "ping",
            "X-GitHub-Delivery": "ping-delivery-id",
            "X-Hub-Signature-256": make_sig(payload),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=payload, headers=headers)
        assert resp.status_code == 200
        assert resp.json() == {"status": "pong"}

    def test_malformed_json_payload(self, webhook_client: tuple[TestClient, SQLiteJobStore]) -> None:
        client, _ = webhook_client
        payload = b"not a json {"
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-malformed",
            "X-Hub-Signature-256": make_sig(payload),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=payload, headers=headers)
        assert resp.status_code == 400

    def test_pull_request_opened_and_deduplication(
        self, webhook_client: tuple[TestClient, SQLiteJobStore]
    ) -> None:
        client, store = webhook_client
        pr_payload = {
            "action": "opened",
            "number": 101,
            "pull_request": {
                "number": 101,
                "state": "open",
                "head": {"sha": "head123", "ref": "feature/branch"},
                "base": {"sha": "base123", "ref": "main"},
            },
            "repository": {
                "id": 1000,
                "name": "test-repo",
                "full_name": "owner/test-repo",
            },
            "installation": {"id": 500},
        }
        body = json.dumps(pr_payload).encode()
        delivery_id = "deliv-pr-opened-101"
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": delivery_id,
            "X-Hub-Signature-256": make_sig(body),
            "Content-Type": "application/json",
        }

        # 1. First delivery accepted
        resp1 = client.post("/webhook", content=body, headers=headers)
        assert resp1.status_code == 202
        assert resp1.json() == {"status": "accepted"}

        # Verify PR request enqueued in job store
        pr_req = store.get_pr_request(1000, 101)
        assert pr_req is not None
        assert pr_req.desired_generation == 1

        # 2. Duplicate delivery deduplicated
        resp2 = client.post("/webhook", content=body, headers=headers)
        assert resp2.status_code == 202
        assert resp2.json() == {"status": "deduplicated"}

    def test_pull_request_closed_action(self, webhook_client: tuple[TestClient, SQLiteJobStore]) -> None:
        client, _ = webhook_client
        pr_payload = {
            "action": "closed",
            "number": 101,
            "installation": {"id": 42},
            "pull_request": {
                "number": 101,
                "state": "closed",
                "head": {"sha": "head123", "ref": "feature/branch"},
                "base": {"sha": "base123", "ref": "main"},
            },
            "repository": {
                "id": 1000,
                "name": "test-repo",
                "full_name": "owner/test-repo",
            },
        }
        body = json.dumps(pr_payload).encode()
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-closed-1",
            "X-Hub-Signature-256": make_sig(body),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=body, headers=headers)
        assert resp.status_code == 202
        assert resp.json() == {"status": "cancelled", "reason": "closed"}

    def test_unhandled_action_and_event(self, webhook_client: tuple[TestClient, SQLiteJobStore]) -> None:
        client, _ = webhook_client
        # Non-PR event
        body_push = json.dumps({"ref": "refs/heads/main"}).encode()
        resp_push = client.post(
            "/webhook",
            content=body_push,
            headers={
                "X-GitHub-Event": "push",
                "X-GitHub-Delivery": "deliv-push-1",
                "X-Hub-Signature-256": make_sig(body_push),
            },
        )
        assert resp_push.status_code == 202
        assert resp_push.json()["status"] == "ignored"

        # Unhandled PR action
        body_assigned = json.dumps({
            "action": "assigned",
            "number": 10,
            "pull_request": {
                "number": 10,
                "head": {"sha": "h", "ref": "r"},
                "base": {"sha": "b", "ref": "m"},
            },
            "repository": {"id": 1, "name": "r", "full_name": "o/r"},
        }).encode()
        resp_assigned = client.post(
            "/webhook",
            content=body_assigned,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "deliv-assigned-1",
                "X-Hub-Signature-256": make_sig(body_assigned),
            },
        )
        assert resp_assigned.status_code == 202
        assert resp_assigned.json()["status"] == "ignored"

    def test_allowlist_filtering(self, tmp_path: Path, valid_config_yaml: str) -> None:
        db_file = tmp_path / "allowlist_test.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        # Set allowlists
        config.service.repository_allowlist = ["allowed-org/allowed-repo"]
        config.service.installation_allowlist = [999]

        app = create_webhook_app(config, store, secret=SECRET)
        client = TestClient(app)

        # Unlisted repository
        body_unlisted_repo = json.dumps({
            "action": "opened",
            "number": 1,
            "pull_request": {
                "number": 1,
                "head": {"sha": "h", "ref": "r"},
                "base": {"sha": "b", "ref": "m"},
            },
            "repository": {"id": 1, "name": "repo", "full_name": "other-org/other-repo"},
            "installation": {"id": 999},
        }).encode()
        resp = client.post(
            "/webhook",
            content=body_unlisted_repo,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "deliv-unlisted-1",
                "X-Hub-Signature-256": make_sig(body_unlisted_repo),
            },
        )
        assert resp.status_code == 202
        assert resp.json() == {"status": "ignored", "reason": "unlisted"}

        # Unlisted installation
        body_unlisted_inst = json.dumps({
            "action": "opened",
            "number": 1,
            "pull_request": {
                "number": 1,
                "head": {"sha": "h", "ref": "r"},
                "base": {"sha": "b", "ref": "m"},
            },
            "repository": {"id": 1, "name": "repo", "full_name": "allowed-org/allowed-repo"},
            "installation": {"id": 111},
        }).encode()
        resp2 = client.post(
            "/webhook",
            content=body_unlisted_inst,
            headers={
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": "deliv-unlisted-2",
                "X-Hub-Signature-256": make_sig(body_unlisted_inst),
            },
        )
        assert resp2.status_code == 202
        assert resp2.json() == {"status": "ignored", "reason": "unlisted"}

        store.close()

    def test_oversized_payload_rejected_with_413(
        self, webhook_client: tuple[TestClient, SQLiteJobStore]
    ) -> None:
        client, _ = webhook_client
        # Send Content-Length exceeding 10 MiB
        headers = {
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "deliv-huge",
            "Content-Length": str(10485760 + 100),
            "Content-Type": "application/json",
        }
        resp = client.post("/webhook", content=b"a", headers=headers)
        assert resp.status_code == 413


class TestWorkerLoopRuntime:
    def test_worker_claim_and_execute_empty_queue(self, tmp_path: Path, valid_config_yaml: str) -> None:
        db_file = tmp_path / "worker_test.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        worker = WorkerLoop(config, store, worker_id="test-worker", lease_duration=5.0)

        assert worker.claim_and_execute_once() is False
        store.close()

    def test_worker_generation_fencing_aborts_stale_run(self, tmp_path: Path, valid_config_yaml: str) -> None:
        db_file = tmp_path / "fencing_worker.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        mock_gh = MockGitHubClient()

        worker = WorkerLoop(
            config=config,
            job_store=store,
            github_client=mock_gh,
            worker_id="test-worker",
            lease_duration=10.0,
        )

        # Enqueue run generation 1
        req = PrRequestRecord(installation_id=1, repository_id=50, pr_number=1, desired_generation=1)
        store.enqueue_request(req)

        # Claim the run
        run = store.claim_next_run("test-worker", lease_seconds=10.0)
        assert run is not None
        assert run.generation == 1

        # Advance desired generation to 2 in pr_requests
        store.enqueue_request(req)

        # Process the claimed generation 1 run
        # It should detect generation 1 < 2 at fencing checkpoint and abort as SUPERSEDED!
        processed = worker.process_run(run)
        assert processed is False

        # Status must be superseded
        conn = store._get_connection()
        status = conn.execute("SELECT status FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()[0]
        assert status == "superseded"
        store.close()

    def test_worker_heartbeat_renewal(self, tmp_path: Path, valid_config_yaml: str) -> None:
        db_file = tmp_path / "heartbeat_worker.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)

        worker = WorkerLoop(
            config=config,
            job_store=store,
            worker_id="heartbeat-worker",
            lease_duration=0.3,
        )

        req = PrRequestRecord(installation_id=1, repository_id=60, pr_number=2, desired_generation=1)
        store.enqueue_request(req)
        run = store.claim_next_run("heartbeat-worker", lease_seconds=0.3)
        assert run is not None

        # Start heartbeat
        heartbeat_stop, heartbeat_thread = worker._start_heartbeat(run.run_id)
        time.sleep(0.35)

        # Lease should have been renewed by heartbeat!
        conn = store._get_connection()
        lease_exp = conn.execute("SELECT lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()[0]
        assert lease_exp > time.time()

        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        store.close()

    def test_worker_stop_gracefully(self, tmp_path: Path, valid_config_yaml: str) -> None:
        db_file = tmp_path / "stop_worker.db"
        store = SQLiteJobStore(db_file)
        config, _ = load_review_config_from_str(valid_config_yaml)
        worker = WorkerLoop(config, store, worker_id="stopping-worker", poll_interval=0.05)

        worker.stop()
        assert worker._stop_event.is_set()
        # run_forever exits immediately when stop event is set
        worker.run_forever()
        store.close()

    @pytest.mark.parametrize("topology", ["orphan", "criss_cross"])
    def test_worker_halts_incomparable_comparison(
        self,
        synthetic_repo: SyntheticGitRepo,
        tmp_path: Path,
        valid_config_yaml: str,
        monkeypatch: pytest.MonkeyPatch,
        topology: str,
    ) -> None:
        if topology == "orphan":
            target_base_sha, head_sha = synthetic_repo.setup_orphan_branches()
            resolution_status = "unavailable"
        else:
            topo = synthetic_repo.setup_criss_cross_merge()
            target_base_sha, head_sha = topo["sha_d"], topo["sha_e"]
            resolution_status = "ambiguous"
        reader = GitRepositoryReader(synthetic_repo.repo_dir)
        changed_files = Mock(wraps=reader.get_changed_files)
        rename_map = Mock(wraps=reader.get_rename_map)
        code_snapshot = Mock(wraps=reader.collect_code_snapshot)
        monkeypatch.setattr(reader, "get_changed_files", changed_files)
        monkeypatch.setattr(reader, "get_rename_map", rename_map)
        monkeypatch.setattr(reader, "collect_code_snapshot", code_snapshot)
        analyzer = Mock(spec=StaticAnalyzer)
        analyzer.run_analysis.side_effect = AssertionError("Incomparable revisions must not be analyzed")
        model = Mock(spec=ModelClient)
        delta_builder = Mock(wraps=build_delta)
        review_pipeline = Mock()
        monkeypatch.setattr("review_bot.runtime.worker.build_delta", delta_builder)
        monkeypatch.setattr("review_bot.runtime.worker.run_review_a_pipeline", review_pipeline)
        github = ApiMockGitHubClient()
        github.set_pull_request(1, head_sha=head_sha, base_sha=target_base_sha)
        config, _ = load_review_config_from_str(valid_config_yaml)
        store = SQLiteJobStore(tmp_path / "incomparable_worker.sqlite")
        try:
            worker = WorkerLoop(
                config, store, repo_reader=reader, analyzer=analyzer,
                model_client=model, github_client=github, worker_id="test-worker",
            )
            store.enqueue_request(PrRequestRecord(1, 50, 1, 1))
            run = store.claim_next_run("test-worker", lease_seconds=30.0)
            assert run is not None

            assert worker.process_run(run) is False

            changed_files.assert_not_called()
            rename_map.assert_not_called()
            analyzer.run_analysis.assert_not_called()
            delta_builder.assert_not_called()
            code_snapshot.assert_not_called()
            review_pipeline.assert_not_called()
            model.complete.assert_not_called()
            assert github.posted_comments == []
            assert github.updated_comments == []
            conn = store._get_connection()
            status, stage, error, completed_at = conn.execute(
                "SELECT status, stage, error, completed_at FROM runs WHERE run_id = ?;",
                (run.run_id,),
            ).fetchone()
            assert status == "failed"
            assert stage == "select"
            assert "comparison_status=incomparable" in error
            assert resolution_status in error
            assert completed_at is not None
            assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
        finally:
            store.close()


@pytest.fixture
def fast_worker(
    tmp_path: Path, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch,
    worker_review_inputs: tuple[TaskInput, ReviewResult],
) -> Iterator[tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]]]:
    config, _ = load_review_config_from_str(
        valid_config_yaml.replace("selector_uses_model: false", "selector_uses_model: true"),
    )
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult("c" * 40, "a" * 40, "comparable")
    reader.get_changed_files.return_value = (ChangedFile("README.md"), ChangedFile("docs/guide.md"))
    reader.get_rename_map.return_value = {}
    github = Mock(spec=GitHubClient)
    github.get_pull_request.return_value = {
        "state": "open",
        "head": {"sha": "b" * 40, "ref": "docs/typos"},
        "base": {"sha": "c" * 40, "repo": {"full_name": "org/repo"}},
        "title": "Fix documentation typos",
        "body": "Correct spelling in the user guide",
    }
    github.find_marker_comment.return_value = None
    github.post_comment.return_value = {"id": 17}
    task, result = worker_review_inputs
    code = {"a" * 40: task.snapshot.base, "b" * 40: task.snapshot.head}
    reader.read_tree_blobs.side_effect = lambda sha: {
        file.path: file.content.encode("utf-8") for file in code[sha].files
    }
    calls = {
        "analyzer": Mock(spec=StaticAnalyzer),
        "model": Mock(spec=ModelClient),
        "delta": Mock(return_value=task.delta),
        "context": Mock(return_value=task),
        "review": Mock(return_value=result),
    }
    monkeypatch.setattr("review_bot.runtime.worker.build_delta", calls["delta"])
    monkeypatch.setattr("review_bot.runtime.worker.build_task_input", calls["context"])
    monkeypatch.setattr("review_bot.runtime.worker.run_review_a_pipeline", calls["review"])
    store = SQLiteJobStore(tmp_path / "fast_worker.sqlite")
    calls["publication"] = Mock(wraps=store.record_publication)
    monkeypatch.setattr(store, "record_publication", calls["publication"])
    worker = WorkerLoop(
        config, store, repo_reader=reader, analyzer=calls["analyzer"],
        model_client=calls["model"], github_client=github, worker_id="worker-a",
    )
    store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
    run = store.claim_next_run("worker-a", lease_seconds=30.0)
    assert run is not None
    try:
        yield store, worker, run, calls
    finally:
        store.close()


@pytest.mark.parametrize("paths", [("README.md", "docs/guide.md"), ("assets/logo.png",), ()])
def test_fast_worker_skips_without_downstream_compute(
    fast_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]],
    paths: tuple[str, ...],
) -> None:
    store, worker, run, calls = fast_worker
    worker.repo_reader.get_changed_files.return_value = tuple(ChangedFile(path) for path in paths)
    # GitHub bodies may be null, including on docs-only PRs.
    worker.github_client.get_pull_request.return_value["body"] = None
    started = time.monotonic()

    assert worker.process_run(run) is True

    assert time.monotonic() - started < 1.0
    status, stage, error, completed_at, lease = store._get_connection().execute(
        "SELECT status, stage, error, completed_at, lease_expires_at FROM runs WHERE run_id = ?;",
        (run.run_id,),
    ).fetchone()
    assert (status, stage) == ("skipped", "select")
    assert store._get_connection().execute(
        "SELECT reason FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()[0] == "not_applicable"
    assert error is None
    assert completed_at is not None and lease is None
    calls["analyzer"].run_analysis.assert_not_called()
    calls["model"].complete.assert_not_called()
    for name in ("delta", "context", "review", "publication"):
        calls[name].assert_not_called()
    worker.repo_reader.read_tree_blobs.assert_not_called()
    worker.repo_reader.collect_code_snapshot.assert_not_called()
    worker.github_client.post_comment.assert_not_called()
    worker.github_client.update_comment.assert_not_called()
    assert store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
    assert store.claim_next_run("worker-b", lease_seconds=30.0) is None
    assert worker._current_run_id is None


@pytest.mark.parametrize("path", ["config/appearance.settings", "config/runtime.txt"])
def test_fast_worker_skips_non_java_uncertain_paths_without_model_requests(
    fast_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]], path: str,
) -> None:
    store, worker, run, calls = fast_worker
    worker.repo_reader.get_changed_files.return_value = (ChangedFile(path),)
    calls["model"].complete.return_value = ModelResponse('{"selected_modes": []}')

    assert worker.process_run(run) is True

    calls["model"].complete.assert_not_called()
    calls["analyzer"].run_analysis.assert_not_called()
    calls["delta"].assert_not_called()
    calls["context"].assert_not_called()
    calls["review"].assert_not_called()
    assert store._get_connection().execute(
        "SELECT status, reason FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone() == ("skipped", "not_applicable")


def test_fast_worker_calls_selector_model_without_review_pipeline(
    fast_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]],
) -> None:
    from review_bot.application.selection import ModelModeSelector

    store, original_worker, run, calls = fast_worker
    original_worker.repo_reader.get_changed_files.return_value = (ChangedFile("config/appearance.settings"),)
    selector_client = Mock(spec=ModelClient)
    selector_client.complete.return_value = ModelResponse('{"selected_modes": []}')
    worker = WorkerLoop(
        original_worker.config, store, repo_reader=original_worker.repo_reader,
        analyzer=calls["analyzer"], model_client=calls["model"],
        github_client=original_worker.github_client, worker_id="worker-a",
        mode_selector=ModelModeSelector(selector_client, model="selector-fast-v1"),
    )

    assert worker.process_run(run) is True

    selector_client.complete.assert_called_once()
    request = selector_client.complete.call_args.args[0]
    assert request.model == "selector-fast-v1"
    metadata = json.loads(request.user_content)
    assert metadata["title"] == "Fix documentation typos"
    assert metadata["body"] == "Correct spelling in the user guide"
    assert metadata["changed_files"] == ["config/appearance.settings"]
    calls["model"].complete.assert_not_called()
    calls["analyzer"].run_analysis.assert_not_called()
    calls["review"].assert_not_called()
    calls["publication"].assert_not_called()
    assert store._get_connection().execute(
        "SELECT status, stage, reason FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone() == ("skipped", "select", "not_applicable")


def test_fast_worker_accepts_injected_mode_selector(
    fast_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]],
) -> None:
    from review_bot.application.selection import select_modes
    from review_bot.ports import ModeSelectorPort

    store, original_worker, run, calls = fast_worker
    original_worker.repo_reader.get_changed_files.return_value = (ChangedFile("custom.settings"),)
    selector = Mock(spec=ModeSelectorPort)
    selector.select_modes.side_effect = lambda snapshot, pr, modes: replace(
        select_modes(snapshot.snapshot_id, modes), selected_modes=(),
    )
    worker = WorkerLoop(
        original_worker.config, store, repo_reader=original_worker.repo_reader,
        analyzer=calls["analyzer"], model_client=calls["model"],
        github_client=original_worker.github_client, worker_id="worker-a", mode_selector=selector,
    )

    assert worker.process_run(run) is True

    selector.select_modes.assert_called_once()
    snapshot, pr, modes = selector.select_modes.call_args.args
    assert snapshot.evidence[0].content == {
        "title": "Fix documentation typos", "body": "Correct spelling in the user guide",
    }
    assert pr.changed_files == ("custom.settings",)
    assert modes == ("architecture",)
    calls["model"].complete.assert_not_called()
    calls["analyzer"].run_analysis.assert_not_called()
    calls["review"].assert_not_called()
    calls["publication"].assert_not_called()
    assert store._get_connection().execute(
        "SELECT status, reason FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone() == ("skipped", "not_applicable")


@pytest.mark.parametrize("race", ["lease_transfer", "generation_advance"])
def test_fast_worker_skip_is_fenced(
    fast_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]],
    monkeypatch: pytest.MonkeyPatch, race: str,
) -> None:
    store, worker, run, calls = fast_worker
    conn = store._get_connection()
    update_run = store.update_run
    before: tuple[Any, ...] | None = None

    def racing_skip(*args: Any, **kwargs: Any) -> bool:
        nonlocal before
        if kwargs.get("status") == "skipped":
            if race == "lease_transfer":
                conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
                assert store.claim_next_run("worker-b", lease_seconds=30.0) is not None
            else:
                store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
            before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        return update_run(*args, **kwargs)

    monkeypatch.setattr(store, "update_run", racing_skip)

    assert worker.process_run(run) is False

    assert before is not None
    if race == "lease_transfer":
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    else:
        assert conn.execute(
            "SELECT status, stage, reason FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone() == ("superseded", "select", None)
    calls["analyzer"].run_analysis.assert_not_called()
    calls["review"].assert_not_called()
    calls["publication"].assert_not_called()


@pytest.mark.parametrize("status", ["unavailable", "ambiguous"])
def test_fast_worker_does_not_skip_incomparable_merge_base(
    fast_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord, dict[str, Mock]], status: str,
) -> None:
    store, worker, run, calls = fast_worker
    worker.repo_reader.resolve_merge_base.return_value = MergeBaseResult("base", None, status)

    assert worker.process_run(run) is False

    row = store._get_connection().execute(
        "SELECT status, stage, error FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    assert row[:2] == ("failed", "select")
    assert "comparison_status=incomparable" in row[2]
    worker.repo_reader.get_changed_files.assert_not_called()
    worker.repo_reader.read_tree_blobs.assert_not_called()
    calls["model"].complete.assert_not_called()
    calls["analyzer"].run_analysis.assert_not_called()
    calls["review"].assert_not_called()
    calls["publication"].assert_not_called()


@pytest.fixture
def claimed_worker(
    tmp_path: Path, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch,
    worker_review_inputs: tuple[TaskInput, ReviewResult],
) -> Iterator[tuple[SQLiteJobStore, WorkerLoop, RunRecord]]:
    store = SQLiteJobStore(tmp_path / "worker_cas.db")
    config, _ = load_review_config_from_str(valid_config_yaml)
    github = ApiMockGitHubClient()
    github.set_pull_request(1, head_sha="b" * 40, base_sha="c" * 40)
    task, result = worker_review_inputs
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult("c" * 40, "a" * 40, "comparable")
    reader.get_changed_files.return_value = (ChangedFile("src/Service.java"),)
    reader.get_rename_map.return_value = {}
    reader.collect_code_snapshot.side_effect = [task.snapshot.base, task.snapshot.head]
    def read_blobs(sha, paths=None):
        blobs = {"build.gradle": b"plugins { id 'java' }\n", **{
            file.path: file.content.encode("utf-8")
            for file in (task.snapshot.base if sha == task.snapshot.base.sha else task.snapshot.head).files}}
        return blobs if paths is None else {path: data for path, data in blobs.items() if path in paths}

    reader.read_tree_blobs.side_effect = read_blobs

    def analyze(sha: str, source_root: Path, *args: Any, **kwargs: Any) -> Mock:
        assert isinstance(source_root, Path) and source_root.is_dir()
        code = task.snapshot.base if sha == task.snapshot.base.sha else task.snapshot.head
        for file in code.files:
            assert (source_root / file.path).read_text(encoding="utf-8") == file.content
        return Mock()

    analyzer = Mock(spec=StaticAnalyzer)
    analyzer.run_analysis.side_effect = analyze
    monkeypatch.setattr("review_bot.runtime.worker.build_delta", Mock(return_value=task.delta))
    monkeypatch.setattr("review_bot.runtime.worker.build_task_input", Mock(return_value=task))
    monkeypatch.setattr("review_bot.runtime.worker.run_review_a_pipeline", Mock(return_value=result))
    worker = WorkerLoop(
        config, store, repo_reader=reader, analyzer=analyzer,
        github_client=github, worker_id="worker-a", lease_duration=60.0,
    )
    store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
    run = store.claim_next_run("worker-a", lease_seconds=60.0)
    assert run is not None
    try:
        yield store, worker, run
    finally:
        store.close()


@pytest.fixture
def worker_tree_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    roots: list[Path] = []
    temporary_directory = tempfile.TemporaryDirectory

    def create_directory(**kwargs: Any) -> tempfile.TemporaryDirectory:
        directory = temporary_directory(dir=tmp_path, **kwargs)
        roots.append(Path(directory.name).resolve())
        return directory

    monkeypatch.setattr("review_bot.runtime.worker.tempfile.TemporaryDirectory", create_directory)
    return roots


@pytest.mark.parametrize("revision", ["base", "head"])
@pytest.mark.parametrize("failure", ["read", "write", "analyze"])
def test_worker_tree_cleanup_on_failure(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    worker_tree_roots: list[Path], monkeypatch: pytest.MonkeyPatch,
    revision: str, failure: str,
) -> None:
    store, worker, run = claimed_worker
    failed_sha = ("a" if revision == "base" else "b") * 40
    read_blobs = worker.repo_reader.read_tree_blobs.side_effect
    analyze = worker.analyzer.run_analysis.side_effect
    write_bytes = Path.write_bytes

    def read(sha: str, paths=None) -> dict[str, bytes]:
        if paths is not None:
            return read_blobs(sha, paths=paths)
        if sha == failed_sha and failure == "read":
            raise OSError("tree read failed")
        return read_blobs(sha)

    def write(path: Path, content: bytes) -> int:
        expected_roots = 1 if revision == "base" else 2
        if failure == "write" and len(worker_tree_roots) == expected_roots:
            raise OSError("tree write failed")
        return write_bytes(path, content)

    def analyze_tree(sha: str, *args: Any, **kwargs: Any) -> Any:
        if sha == failed_sha and failure == "analyze":
            raise RuntimeError("tree analyze failed")
        return analyze(sha, *args, **kwargs)

    worker.repo_reader.read_tree_blobs.side_effect = read
    worker.analyzer.run_analysis.side_effect = analyze_tree
    monkeypatch.setattr(Path, "write_bytes", write)

    assert worker.process_run(run) is False

    assert len(worker_tree_roots) == (1 if revision == "base" else 2)
    assert all(not root.exists() for root in worker_tree_roots)
    status, stage, error, lease = store._get_connection().execute(
        "SELECT status, stage, error, lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    error_type = "RuntimeError" if failure == "analyze" else "OSError"
    notice_posted = failure != "read"
    assert (status, stage, error, lease) == ("failed", "publish" if notice_posted else "analyze", f"worker_pipeline_error:{error_type}", None)
    assert f"tree {failure} failed" not in error
    assert store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == int(notice_posted)
    assert len(worker.github_client.comments) == int(notice_posted)
    if notice_posted:
        assert "리뷰 미완료" in next(iter(worker.github_client.comments.values()))["body"]
    assert worker._current_run_id is None


@pytest.mark.parametrize("path_kind", ["parent", "nested_parent", "absolute"])
def test_worker_tree_cannot_escape_temporary_root(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    worker_tree_roots: list[Path], tmp_path: Path, path_kind: str,
) -> None:
    store, worker, run = claimed_worker
    outside = tmp_path / "escaped.java"
    outside.write_bytes(b"unchanged")
    unsafe_path = {
        "parent": "../escaped.java",
        "nested_parent": "src/../../escaped.java",
        "absolute": str(outside),
    }[path_kind]
    worker.repo_reader.read_tree_blobs.side_effect = None
    worker.repo_reader.read_tree_blobs.return_value = {
        "build.gradle": b"plugins { id 'java' }\n",
        "src/main/java/Safe.java": b"class Safe {}", unsafe_path: b"overwritten",
    }

    assert worker.process_run(run) is False

    assert outside.read_bytes() == b"unchanged"
    assert len(worker_tree_roots) == 1 and not worker_tree_roots[0].exists()
    worker.analyzer.run_analysis.assert_not_called()
    status, stage, error = store._get_connection().execute(
        "SELECT status, stage, error FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    assert (status, stage) == ("failed", "publish")
    assert error == "worker_pipeline_error:ValueError"
    assert unsafe_path not in error
    assert store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 1
    assert "리뷰 미완료" in next(iter(worker.github_client.comments.values()))["body"]
    assert unsafe_path not in next(iter(worker.github_client.comments.values()))["body"]


@pytest.mark.parametrize("boundary", ["before_read", "after_read", "after_write", "between_analyses"])
@pytest.mark.parametrize("race", ["expired", "lease_transfer", "generation_advance"])
def test_worker_tree_materialization_is_fenced(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    worker_tree_roots: list[Path], monkeypatch: pytest.MonkeyPatch, boundary: str, race: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    update_run = store.update_run
    read_blobs = worker.repo_reader.read_tree_blobs.side_effect
    analyze = worker.analyzer.run_analysis.side_effect
    write_bytes = Path.write_bytes
    before: tuple[Any, ...] | None = None

    def lose_fence() -> None:
        nonlocal before
        if race == "generation_advance":
            store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
        else:
            conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
            if race == "lease_transfer":
                successor = store.claim_next_run("worker-b", lease_seconds=60.0)
                assert successor is not None and successor.run_id == run.run_id
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()

    def update(*args: Any, **kwargs: Any) -> bool:
        applied = update_run(*args, **kwargs)
        if kwargs.get("stage") == RunStage.ANALYZE and boundary == "before_read":
            lose_fence()
        return applied

    def read(sha: str, paths=None) -> dict[str, bytes]:
        if paths is not None:
            return read_blobs(sha, paths=paths)
        blobs = read_blobs(sha)
        if boundary == "after_read":
            lose_fence()
        return blobs

    def write(path: Path, content: bytes) -> int:
        written = write_bytes(path, content)
        if boundary == "after_write":
            lose_fence()
        return written

    def analyze_tree(*args: Any, **kwargs: Any) -> Any:
        result = analyze(*args, **kwargs)
        if boundary == "between_analyses":
            lose_fence()
        return result

    monkeypatch.setattr(store, "update_run", update)
    monkeypatch.setattr(Path, "write_bytes", write)
    worker.repo_reader.read_tree_blobs.side_effect = read
    worker.analyzer.run_analysis.side_effect = analyze_tree

    assert worker.process_run(run) is False

    assert before is not None
    expected_reads = int(boundary != "before_read")
    assert sum(call.kwargs.get("paths") is None
        for call in worker.repo_reader.read_tree_blobs.call_args_list) == expected_reads
    assert len(worker_tree_roots) == expected_reads
    assert all(not root.exists() for root in worker_tree_roots)
    assert worker.analyzer.run_analysis.call_count == int(boundary == "between_analyses")
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
    assert worker.github_client.comments == {}
    if race == "generation_advance":
        assert conn.execute(
            "SELECT status, stage FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone() == ("superseded", "analyze")
    else:
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    assert worker._current_run_id is None


@pytest.mark.parametrize("existing, marker", [
    (False, "<!-- review-bot:architecture:v1 -->"),
    (True, "<!-- review-bot:architecture:custom -->"),
])
def test_worker_publishes_formatted_review_with_real_receipt(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    worker_review_inputs: tuple[TaskInput, ReviewResult], monkeypatch: pytest.MonkeyPatch,
    existing: bool, marker: str,
) -> None:
    store, worker, run = claimed_worker
    task, result = worker_review_inputs
    github = worker.github_client
    worker.config = worker.config.model_copy(update={
        "publish": worker.config.publish.model_copy(update={"a_marker": marker}),
    })
    previous_id = github.create_comment(1, f"{marker}\nOld review")["id"] if existing else None
    record_publication = Mock(wraps=store.record_publication)
    monkeypatch.setattr(store, "record_publication", record_publication)

    assert worker.process_run(run) is True

    assert len(github.comments) == 1
    comment = next(iter(github.comments.values()))
    assert comment["body"] == format_review_markdown(result, task, marker=marker)
    assert r"Worker architecture review completed\." in comment["body"]
    assert f"HEAD `{'b' * 40}`" in comment["body"]
    assert "전체 판단" in comment["body"]
    if existing:
        assert comment["id"] == previous_id
        assert len(github.posted_comments) == 1
        assert len(github.updated_comments) == 1
    else:
        assert len(github.posted_comments) == 1
        assert github.updated_comments == []
    record_publication.assert_called_once()
    receipt = record_publication.call_args.args[0]
    assert receipt.status == "succeeded"
    assert type(receipt.comment_id) is int and receipt.comment_id > 0
    assert receipt.comment_id == comment["id"]
    assert receipt.marker == marker and receipt.published_at is not None
    assert len(receipt.fingerprint) == 64
    assert record_publication.call_args.kwargs == {
        "run_id": run.run_id, "worker_id": worker.worker_id, "generation": run.generation,
    }
    conn = store._get_connection()
    assert conn.execute(
        "SELECT comment_id, published_fingerprint, published_at FROM publication_receipts;",
    ).fetchone() == (receipt.comment_id, receipt.fingerprint, receipt.published_at)
    status, stage, completed_at, lease = conn.execute(
        "SELECT status, stage, completed_at, lease_expires_at FROM runs WHERE run_id = ?;",
        (run.run_id,),
    ).fetchone()
    assert (status, stage) == ("succeeded", "publish")
    assert completed_at is not None and lease is None


@pytest.mark.parametrize("missing", ["github_client", "review_result", "task_input"])
def test_worker_missing_publication_input_never_records_success(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    store, worker, run = claimed_worker
    github = worker.github_client
    if missing == "github_client":
        worker.github_client = None
    elif missing == "review_result":
        monkeypatch.setattr("review_bot.runtime.worker.run_review_a_pipeline", Mock(return_value=None))
    else:
        monkeypatch.setattr("review_bot.runtime.worker.build_task_input", Mock(return_value=None))

    assert worker.process_run(run) is False

    status, stage, error = store._get_connection().execute(
        "SELECT status, stage, error FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    assert (status, stage) == ("failed", "publish")
    assert ("github_client_unavailable" if missing == "github_client" else "review_result_unavailable") in error
    notices = [] if missing == "github_client" else github.posted_comments
    assert len(notices) == (0 if missing == "github_client" else 1)
    assert store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == len(notices)
    assert store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts WHERE input_fingerprint IS NOT NULL;").fetchone()[0] == 0
    assert github.updated_comments == []
    if notices:
        assert "리뷰 미완료" in notices[0]["body"]
    assert worker._current_run_id is None


@pytest.mark.parametrize("race", ["expired", "lease_transfer", "generation_advance"])
@pytest.mark.parametrize("existing", [False, True])
def test_worker_comment_write_is_fenced_after_marker_discovery(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    monkeypatch: pytest.MonkeyPatch, race: str, existing: bool,
) -> None:
    store, worker, run = claimed_worker
    github = worker.github_client
    marker = worker.config.publish.a_marker
    if existing:
        github.create_comment(1, f"{marker}\nPrevious review")
    comments_before = {key: dict(value) for key, value in github.comments.items()}
    find_marker_comment = github.find_marker_comment
    conn = store._get_connection()
    before: tuple[Any, ...] | None = None

    def lose_fence_during_discovery(*args: Any, **kwargs: Any) -> Any:
        nonlocal before
        found = find_marker_comment(*args, **kwargs)
        if race == "generation_advance":
            store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
        else:
            conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
            if race == "lease_transfer":
                successor = store.claim_next_run("worker-b", lease_seconds=60.0)
                assert successor is not None and successor.run_id == run.run_id
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        return found

    monkeypatch.setattr(github, "find_marker_comment", lose_fence_during_discovery)

    assert worker.process_run(run) is False

    assert before is not None
    assert github.comments == comments_before
    assert len(github.posted_comments) == int(existing)
    assert github.updated_comments == []
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
    if race == "generation_advance":
        assert conn.execute(
            "SELECT status, stage FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone() == ("superseded", "publish")
    else:
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    assert worker._current_run_id is None


@pytest.mark.parametrize("race", ["lease_transfer", "generation_advance"])
def test_worker_stale_warning_is_fenced_after_comment_post(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    monkeypatch: pytest.MonkeyPatch, race: str,
) -> None:
    store, worker, run = claimed_worker
    github = worker.github_client
    create_comment = github.create_comment
    conn = store._get_connection()
    successor_body = f"{worker.config.publish.a_marker}\nSuccessor worker review"
    before: tuple[Any, ...] | None = None

    def successor_after_post(*args: Any, **kwargs: Any) -> Any:
        nonlocal before
        created = create_comment(*args, **kwargs)
        if race == "lease_transfer":
            conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
            assert store.claim_next_run("worker-b", lease_seconds=60.0) is not None
        else:
            store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
        github.set_pull_request(1, head_sha="d" * 40, base_sha="c" * 40)
        github.update_comment(created["id"], successor_body)
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        return created

    monkeypatch.setattr(github, "create_comment", successor_after_post)

    assert worker.process_run(run) is False

    assert before is not None
    assert len(github.comments) == 1
    assert next(iter(github.comments.values()))["body"] == successor_body
    assert len(github.updated_comments) == 1
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
    if race == "lease_transfer":
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    else:
        assert conn.execute(
            "SELECT status FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone()[0] == "superseded"


@pytest.mark.parametrize("timing, field, reason", [
    ("pre", "head", "stale_head_sha"),
    ("pre", "base", "stale_base_sha"),
    ("pre", "state", "pr_not_open"),
    ("post", "head", "stale_head_sha"),
    ("post", "base", "stale_base_sha"),
    ("post", "state", "pr_closed_during_publication"),
])
def test_worker_publication_freshness_outcome(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    worker_review_inputs: tuple[TaskInput, ReviewResult], monkeypatch: pytest.MonkeyPatch,
    timing: str, field: str, reason: str,
) -> None:
    store, worker, run = claimed_worker
    github = worker.github_client
    _, result = worker_review_inputs

    def change_pr() -> None:
        pr = github.pull_requests[1]
        if field == "state":
            pr["state"] = "closed"
        else:
            pr[field]["sha"] = "d" * 40

    if timing == "pre":
        def change_before_publication(*args: Any) -> ReviewResult:
            change_pr()
            return result

        monkeypatch.setattr("review_bot.runtime.worker.run_review_a_pipeline", change_before_publication)
    else:
        create_comment = github.create_comment

        def change_after_publication(*args: Any, **kwargs: Any) -> Any:
            created = create_comment(*args, **kwargs)
            change_pr()
            return created

        monkeypatch.setattr(github, "create_comment", change_after_publication)
    record_publication = Mock(wraps=store.record_publication)
    monkeypatch.setattr(store, "record_publication", record_publication)

    assert worker.process_run(run) is False

    conn = store._get_connection()
    status, stage, error, lease = conn.execute(
        "SELECT status, stage, error, lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    assert (status, stage, lease) == ("superseded", "publish", None)
    assert reason in error
    if timing == "post":
        receipt = record_publication.call_args.args[0]
        assert receipt.status == "stale" and receipt.reason == reason
        assert receipt.comment_id in github.comments
        assert "오래된 결과" in github.comments[receipt.comment_id]["body"]
        assert conn.execute("SELECT comment_id FROM publication_receipts;").fetchone()[0] == receipt.comment_id
    else:
        record_publication.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
        assert github.posted_comments == [] and github.updated_comments == []


@pytest.mark.parametrize("operation, reason", [
    ("find_marker_comment", "find_marker_failed"),
    ("create_comment", "create_comment_failed"),
    ("update_comment", "update_comment_failed"),
    ("post_check", "post_check_failed"),
])
def test_worker_publication_http_failure_never_succeeds(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord],
    monkeypatch: pytest.MonkeyPatch, operation: str, reason: str,
) -> None:
    store, worker, run = claimed_worker
    github = worker.github_client
    error = httpx.ReadTimeout("GitHub request timed out")
    if operation == "update_comment":
        github.create_comment(1, f"{worker.config.publish.a_marker}\nPrevious review")
    if operation == "post_check":
        pr = github.get_pull_request(1)
        monkeypatch.setattr(github, "get_pull_request", Mock(side_effect=[pr, pr, error]))
    else:
        monkeypatch.setattr(github, operation, Mock(side_effect=error))
    record_publication = Mock(wraps=store.record_publication)
    monkeypatch.setattr(store, "record_publication", record_publication)

    assert worker.process_run(run) is False

    conn = store._get_connection()
    status, stage, run_error, lease = conn.execute(
        "SELECT status, stage, error, lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    assert (status, stage, lease) == ("failed", "publish", None)
    assert reason in run_error
    if operation in ("create_comment", "update_comment", "post_check"):
        receipt = record_publication.call_args.args[0]
        assert receipt.status == ("failed" if operation == "post_check" else "unknown")
        assert receipt.reason.startswith(f"{reason}:")
        if operation == "create_comment":
            assert receipt.comment_id is None
        else:
            assert receipt.comment_id in github.comments
        assert conn.execute("SELECT comment_id FROM publication_receipts;").fetchone()[0] == receipt.comment_id
    else:
        record_publication.assert_not_called()
        assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
    assert worker._current_run_id is None


def test_worker_rejects_success_receipt_without_comment_id(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, worker, run = claimed_worker
    monkeypatch.setattr("review_bot.runtime.worker.publish_review_comment", Mock(return_value=PublicationReceipt(
        comment_id=None, status="succeeded", fingerprint="unverified",
    )))

    assert worker.process_run(run) is False

    status, error = store._get_connection().execute(
        "SELECT status, error FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    assert status == "failed" and "invalid_comment_id" in error
    assert store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0


@pytest.mark.parametrize("loss", ["heartbeat", "expired", "transferred"])
def test_worker_lease_loss_aborts_without_mutation(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], loss: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    if loss == "heartbeat":
        worker._lease_lost = True
    else:
        conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
        if loss == "transferred":
            successor = store.claim_next_run("worker-b", lease_seconds=60.0)
            assert successor is not None and successor.run_id == run.run_id
    before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()

    assert worker._check_fencing_and_lease(run.run_id, run.generation) is False

    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before


@pytest.mark.parametrize(
    "stage", [RunStage.PREPARE, RunStage.SELECT, RunStage.ANALYZE, RunStage.COMPARE,
              RunStage.CONTEXT, RunStage.REVIEW, RunStage.VALIDATE, RunStage.PUBLISH],
)
@pytest.mark.parametrize("race", ["lease_transfer", "generation_advance"])
def test_worker_stage_transition_is_atomic_with_fence(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], monkeypatch: pytest.MonkeyPatch,
    stage: RunStage, race: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    update_run = store.update_run
    before: tuple[Any, ...] | None = None

    def racing_update(run_id: str, status: Any, stage: Any = None, **kwargs: Any) -> Any:
        nonlocal before
        if before is None and status == RunStatus.RUNNING and stage == target_stage:
            if race == "lease_transfer":
                conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run_id,))
                successor = store.claim_next_run("worker-b", lease_seconds=60.0)
                assert successor is not None and successor.run_id == run_id
            else:
                store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
            before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run_id,)).fetchone()
        return update_run(run_id, status, stage, **kwargs)

    target_stage = stage
    monkeypatch.setattr(store, "update_run", racing_update)

    assert worker.process_run(run) is False

    assert before is not None
    if race == "lease_transfer":
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    else:
        status, final_stage, error = conn.execute(
            "SELECT status, stage, error FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone()
        assert status == "superseded"
        assert final_stage == before[8]
        assert "Fenced" in error
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0
    assert worker._current_run_id is None


@pytest.mark.parametrize("status", [RunStatus.SUCCEEDED, RunStatus.SKIPPED, RunStatus.FAILED, RunStatus.SUPERSEDED, RunStatus.CANCELLED])
def test_worker_does_not_reopen_terminal_run(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], status: RunStatus,
) -> None:
    store, worker, run = claimed_worker
    store.update_run(run.run_id, status)
    conn = store._get_connection()
    before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()

    assert worker.process_run(run) is False

    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0


@pytest.mark.parametrize("race", ["lease_transfer", "generation_advance"])
def test_worker_exception_after_fence_loss_does_not_mark_failed(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], monkeypatch: pytest.MonkeyPatch, race: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    before: tuple[Any, ...] | None = None

    def failing_fetch(pr_number: int) -> None:
        nonlocal before
        if race == "lease_transfer":
            conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
            assert store.claim_next_run("worker-b", lease_seconds=60.0) is not None
        else:
            store.enqueue_request(PrRequestRecord(1, 100, pr_number, 1))
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        raise RuntimeError("late API failure")

    monkeypatch.setattr(worker.github_client, "get_pull_request", failing_fetch)

    assert worker.process_run(run) is False

    assert before is not None
    row = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
    if race == "lease_transfer":
        assert row == before
    else:
        status, error = conn.execute(
            "SELECT status, error FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone()
        assert status == "superseded"
        assert "Fenced" in error
    assert worker._current_run_id is None


@pytest.mark.parametrize("race", ["lease_transfer", "generation_advance"])
def test_worker_receipt_write_is_atomic_with_fence(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    record_publication = store.record_publication
    before: tuple[Any, ...] | None = None

    def racing_publication(*args: Any, **kwargs: Any) -> Any:
        nonlocal before
        if race == "lease_transfer":
            conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
            assert store.claim_next_run("worker-b", lease_seconds=60.0) is not None
        else:
            store.enqueue_request(PrRequestRecord(1, 100, 1, 1))
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
        return record_publication(*args, **kwargs)

    monkeypatch.setattr(store, "record_publication", racing_publication)

    assert worker.process_run(run) is False

    assert before is not None
    if race == "lease_transfer":
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    else:
        assert conn.execute(
            "SELECT status, stage FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone() == ("superseded", "publish")
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0


@pytest.mark.parametrize("operation", ["renew_lease", "check_generation_fencing"])
def test_worker_fence_validation_error_aborts_without_terminal_write(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    before = conn.execute(
        "SELECT status, stage, completed_at, error FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone()
    monkeypatch.setattr(store, operation, Mock(side_effect=sqlite3.OperationalError("database is locked")))

    assert worker.process_run(run) is False

    assert worker._lease_lost is True
    assert worker._current_run_id is None
    assert conn.execute(
        "SELECT status, stage, completed_at, error FROM runs WHERE run_id = ?;", (run.run_id,),
    ).fetchone() == before
    assert conn.execute("SELECT COUNT(*) FROM publication_receipts;").fetchone()[0] == 0


@pytest.mark.parametrize("terminal_path", ["success", "freshness", "incomparable", "not_selected"])
def test_worker_terminal_write_cannot_clobber_successor(
    claimed_worker: tuple[SQLiteJobStore, WorkerLoop, RunRecord], monkeypatch: pytest.MonkeyPatch,
    synthetic_repo: SyntheticGitRepo, terminal_path: str,
) -> None:
    store, worker, run = claimed_worker
    conn = store._get_connection()
    update_run = store.update_run
    before: tuple[Any, ...] | None = None
    receipts_before: list[tuple[Any, ...]] | None = None
    target_status = RunStatus.SUCCEEDED
    if terminal_path in ("incomparable", "not_selected"):
        worker.repo_reader = GitRepositoryReader(synthetic_repo.repo_dir)
        if terminal_path == "incomparable":
            base_sha, head_sha = synthetic_repo.setup_orphan_branches()
            target_status = RunStatus.FAILED
        else:
            base_sha = head_sha = synthetic_repo.get_head_sha()
            target_status = RunStatus.SKIPPED

            def select_no_modes(**kwargs: Any) -> Any:
                snapshot, pr, classification, selection = assemble_snapshot_bundle(**kwargs)
                return snapshot, pr, classification, replace(selection, selected_modes=())

            monkeypatch.setattr("review_bot.runtime.worker.assemble_snapshot_bundle", select_no_modes)
        worker.github_client.set_pull_request(1, head_sha=head_sha, base_sha=base_sha)
    elif terminal_path == "freshness":
        target_status = RunStatus.SUPERSEDED
        get_pull_request = worker.github_client.get_pull_request
        fetch_count = 0

        def changed_head(pr_number: int) -> dict[str, Any]:
            nonlocal fetch_count
            fetch_count += 1
            pr = get_pull_request(pr_number)
            if fetch_count == 2:
                pr["head"]["sha"] = "force-pushed-head"
            return pr

        monkeypatch.setattr(worker.github_client, "get_pull_request", changed_head)

    def racing_terminal_update(run_id: str, status: Any, stage: Any = None, **kwargs: Any) -> Any:
        nonlocal before, receipts_before
        if status == target_status:
            conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run_id,))
            successor = store.claim_next_run("worker-b", lease_seconds=60.0)
            assert successor is not None and successor.run_id == run_id
            before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run_id,)).fetchone()
            receipts_before = conn.execute("SELECT * FROM publication_receipts;").fetchall()
        return update_run(run_id, status, stage, **kwargs)

    monkeypatch.setattr(store, "update_run", racing_terminal_update)

    assert worker.process_run(run) is False

    assert before is not None
    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    assert conn.execute("SELECT * FROM publication_receipts;").fetchall() == receipts_before
