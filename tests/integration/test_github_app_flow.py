"""Real signed intake, SQLite, Git, PMD/CPD and A pipeline; only network is replaced."""
from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from review_bot.adapters.github.auth import GitHubAppTokenProvider
from review_bot.adapters.pmd.runner import PmdRunner
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import load_review_config_from_str
from review_bot.runtime.github_app import GitHubAppAdapters
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop
from tests.conftest import SyntheticGitRepo
from tests.fixtures.partition_review import PartitionModelClient

SECRET = "synthetic-webhook-secret"
TOKEN = "synthetic-installation-token"


@pytest.fixture
def app_flow(tmp_path: Path, synthetic_repo: SyntheticGitRepo, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch):
    topology = synthetic_repo.setup_standard_feature_pr(advance_base=False)
    synthetic_repo.write_file("src/main/java/PaymentService.java", """public class PaymentService {
    public void nested(int a, int b, int c, int d, int e) {
        if (a > 0) { for (int i = 0; i < 10; i++) { if (b > 0) {
            while (c > 0) { if (d > 0) { if (e > 0) { System.out.println(1); }
            else { System.out.println(2); } } }
        } } }
    }
}
""")
    head = synthetic_repo.commit("Synthetic complexity change")
    synthetic_repo._git("update-ref", "refs/pull/7/head", head)
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["mode"] = "github_app"
    raw["publish"].pop("mode", None)
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    private_key = tmp_path / "synthetic-app.pem"
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_key.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    ))
    private_key.chmod(0o600)
    pr = {
        "number": 7, "state": "open", "draft": False, "title": "Refactor payment", "body": "Synthetic PR",
        "head": {"sha": head, "ref": "feature/payment"},
        "base": {"sha": topology["target_base_sha"], "ref": "main", "repo": {"id": 99, "full_name": "org/repo"}},
        "labels": [],
    }
    comments = []
    calls = []

    def http(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/app":
            assert request.method == "GET"
            return httpx.Response(200, json={"id": 1001, "slug": "review-app"})
        if request.url.path == "/app/installations/44/access_tokens":
            assert request.method == "POST"
            body = json.loads(request.content)
            assert body["repository_ids"] == [99]
            assert body["permissions"] == {"contents": "read", "pull_requests": "write"}
            return httpx.Response(201, json={
                "token": TOKEN, "expires_at": (datetime.now(UTC) + timedelta(hours=1)).isoformat(),
                "permissions": body["permissions"], "repository_selection": "selected",
                "repositories": [{"id": 99}],
            })
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        if request.url.path == "/repositories/99":
            return httpx.Response(200, json={"id": 99, "full_name": "org/repo"})
        if request.url.path == "/repos/org/repo/pulls/7":
            return httpx.Response(200, json=pr)
        if request.url.path == "/repos/org/repo/issues/7/comments":
            if request.method == "GET":
                return httpx.Response(200, json=comments)
            comment = {"id": 701, "body": json.loads(request.content)["body"], "user": {"login": "review-app[bot]", "type": "Bot"}}
            comments.append(comment)
            return httpx.Response(201, json=comment)
        if request.url.path == "/repos/org/repo/issues/comments/701":
            comments[0]["body"] = json.loads(request.content)["body"]
            return httpx.Response(200, json=comments[0])
        raise AssertionError(f"Unexpected HTTP boundary: {request.method} {request.url.path}")

    transport = httpx.MockTransport(http)
    provider = GitHubAppTokenProvider("1001", private_key, transport=transport)
    adapters = GitHubAppAdapters(provider, state / "repositories", config.service, transport=transport)
    execute = subprocess.run

    def synthetic_network(command, **kwargs):
        if command[0] == "git" and "fetch" in command and "https://github.com/org/repo.git" in command:
            command = [str(synthetic_repo.repo_dir) if value == "https://github.com/org/repo.git" else value for value in command]
            kwargs["env"] = {**kwargs["env"], "GIT_ALLOW_PROTOCOL": "file"}
        return execute(command, **kwargs)

    monkeypatch.setattr("review_bot.adapters.github.repository.subprocess.run", synthetic_network)
    store = SQLiteJobStore(state / "jobs.sqlite")
    analyzer = PmdRunner(output_root=state / "tools")
    worker = WorkerLoop(
        config, store, analyzer=analyzer, worker_id="installation-worker",
        model_client=PartitionModelClient(),
        planning_client=PartitionModelClient(name="tev", group_all=True),
        github_client_factory=adapters.github_client,
        repo_reader_factory=adapters.repository_cache.prepare, artifact_root=state,
    )
    with TestClient(create_webhook_app(config, store, secret=SECRET)) as client:
        try:
            yield SimpleNamespace(
                config=config, state=state, store=store, worker=worker, client=client,
                pr=pr, calls=calls, comments=comments, repo=synthetic_repo,
            )
        finally:
            store.close()


def send_event(flow, action: str = "opened", delivery_id: str = "first"):
    payload = {
        "action": action, "number": 7, "installation": {"id": 44},
        "repository": {"id": 99, "name": "repo", "full_name": "org/repo"}, "pull_request": flow.pr,
    }
    body = json.dumps(payload).encode()
    return flow.client.post("/webhook", content=body, headers={
        "X-GitHub-Event": "pull_request", "X-GitHub-Delivery": delivery_id,
        "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
    })


def test_installed_first_pr_runs_real_analysis_and_creates_one_comment(app_flow) -> None:
    flow = app_flow
    assert flow.config.publish.mode == "comment"
    assert flow.config.service.installation_allowlist == []
    assert flow.config.service.repository_allowlist == []
    assert send_event(flow).json() == {"status": "accepted"}
    assert flow.worker.run_once() is True
    assert len(flow.comments) == 1
    assert flow.calls.count(("POST", "/repos/org/repo/issues/7/comments")) == 1
    assert "<!-- review-bot:architecture:v1 -->" in flow.comments[0]["body"]
    assert "#### 전체 판단" in flow.comments[0]["body"]
    assert "#### 리뷰 의견" in flow.comments[0]["body"]
    assert "모델 미실행" not in flow.comments[0]["body"]
    assert "Actions artifact" not in flow.comments[0]["body"]
    run_id, status = flow.store._get_connection().execute("SELECT run_id, status FROM runs;").fetchone()
    assert status == "succeeded"
    output = flow.state / "runs" / "99" / "7" / run_id
    manifest = json.loads((output / "manifest.json").read_text())
    result = json.loads((output / "A/result.json").read_text())
    delta = json.loads((output / "A/delta.json").read_text())
    analysis = json.loads((output / "A/analysis.json").read_text())
    assert manifest["publication"]["comment_id"] == 701
    assert manifest["status"] == "succeeded"
    assert result["status"] == "completed" and result["schema_version"] == 13
    assert analysis["base"]["status"] == "clean"
    assert analysis["head"]["pmd"]["status"] == "findings"
    assert any(signal["tool"] == "pmd" for signal in delta["signals"])
    for path in ("manifest.json", "snapshot.json", "selection.json", "A/input.json", "A/result.json", "A/comment.md"):
        content = (output / path).read_text()
        assert TOKEN not in content and "PRIVATE KEY" not in content
        assert (output / path).stat().st_mode & 0o077 == 0
    assert flow.store._get_connection().execute("SELECT comment_id FROM publication_receipts;").fetchone()[0] == 701
    assert flow.worker.run_once() is False


@pytest.mark.parametrize("author", [
    {"login": "person", "type": "User"},
    {"login": "other-app[bot]", "type": "Bot"},
    None,
])
def test_foreign_marker_never_updates_or_creates_a_duplicate(app_flow, author) -> None:
    flow = app_flow
    original = {"id": 701, "body": "<!-- review-bot:architecture:v1 -->\nExisting unrelated comment"}
    if author is not None:
        original["user"] = author
    flow.comments.append(original.copy())
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    assert flow.comments == [original]
    assert ("POST", "/repos/org/repo/issues/7/comments") not in flow.calls
    assert ("PATCH", "/repos/org/repo/issues/comments/701") not in flow.calls
    assert flow.store._get_connection().execute("SELECT status FROM runs;").fetchone()[0] == "failed"


def test_explicit_artifact_only_writes_files_but_never_a_comment(app_flow) -> None:
    flow = app_flow
    flow.config.publish.mode = "artifact_only"
    assert send_event(flow).json() == {"status": "accepted"}
    assert flow.worker.run_once() is True
    assert flow.comments == []
    assert not any(method in ("POST", "PATCH") and "/comments" in path for method, path in flow.calls)
    run_id, status = flow.store._get_connection().execute("SELECT run_id, status FROM runs;").fetchone()
    assert status == "succeeded"
    manifest = json.loads((flow.state / "runs/99/7" / run_id / "manifest.json").read_text())
    assert manifest["publication"]["status"] == "not_attempted"
    assert manifest["publication"]["comment_id"] is None
    assert manifest["publication"]["published_at"] is None


def test_new_head_updates_the_same_owned_comment(app_flow) -> None:
    flow = app_flow
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is True
    first_head = flow.pr["head"]["sha"]
    source = flow.repo.repo_dir / "src/main/java/PaymentService.java"
    flow.repo.write_file("src/main/java/PaymentService.java", source.read_text() + "\n// Second synthetic revision\n")
    second_head = flow.repo.commit("Second synthetic revision")
    flow.repo._git("update-ref", "refs/pull/7/head", second_head)
    flow.pr["head"]["sha"] = second_head
    assert send_event(flow, "synchronize", "updated").status_code == 202
    assert flow.worker.run_once() is True
    assert len(flow.comments) == 1 and flow.comments[0]["id"] == 701
    assert second_head in flow.comments[0]["body"] and first_head not in flow.comments[0]["body"]
    assert flow.calls.count(("POST", "/repos/org/repo/issues/7/comments")) == 1
    assert flow.calls.count(("PATCH", "/repos/org/repo/issues/comments/701")) == 1


@pytest.mark.parametrize("action", ["edited", "labeled", "unlabeled", "ready_for_review"])
def test_reselection_action_is_durably_accepted(app_flow, action: str) -> None:
    assert send_event(app_flow, action, action).json() == {"status": "accepted"}
    assert app_flow.store.get_pr_request(99, 7) is not None


def test_closed_cancels_pending_work_before_any_analysis_or_http(app_flow) -> None:
    flow = app_flow
    assert send_event(flow).status_code == 202
    flow.pr["state"] = "closed"
    assert send_event(flow, "closed", "closed").json() == {"status": "cancelled", "reason": "closed"}
    assert flow.worker.run_once() is False
    assert flow.calls == [] and flow.comments == []
    assert flow.store.is_delivery_processed("closed")
    assert flow.store._get_connection().execute("SELECT status FROM runs;").fetchone()[0] == "cancelled"


def test_irrelevant_label_reuses_current_input_without_analysis_or_write(app_flow, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = app_flow
    analysis = Mock(wraps=flow.worker.analyzer.run_analysis)
    monkeypatch.setattr(flow.worker.analyzer, "run_analysis", analysis)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is True
    assert analysis.call_count == 2
    first_body = flow.comments[0]["body"]
    flow.pr["labels"].append({"name": "documentation"})
    assert send_event(flow, "labeled", "irrelevant").status_code == 202
    assert flow.worker.run_once() is True
    assert analysis.call_count == 2
    assert flow.comments[0]["body"] == first_body
    assert flow.calls.count(("POST", "/repos/org/repo/issues/7/comments")) == 1
    assert ("PATCH", "/repos/org/repo/issues/comments/701") not in flow.calls
    assert flow.store._get_connection().execute("SELECT status, reason FROM runs ORDER BY rowid DESC LIMIT 1;").fetchone() == ("skipped", "unchanged_input")


def test_close_during_real_analysis_fences_every_comment_write(app_flow, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = app_flow
    execute = flow.worker.analyzer.run_analysis
    calls = []

    def closing_analysis(*args, **kwargs):
        result = execute(*args, **kwargs)
        calls.append(args[0])
        flow.pr["state"] = "closed"
        assert send_event(flow, "closed", "mid-analysis-close").status_code == 202
        return result

    monkeypatch.setattr(flow.worker.analyzer, "run_analysis", closing_analysis)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    assert len(calls) == 1
    assert flow.comments == []
    assert not any(method in ("POST", "PATCH") and "/comments" in path for method, path in flow.calls)
    assert flow.store._get_connection().execute("SELECT status FROM runs;").fetchone()[0] == "cancelled"


def test_pr_text_changed_without_delivery_never_publishes_old_context(app_flow, monkeypatch: pytest.MonkeyPatch) -> None:
    flow = app_flow
    execute = flow.worker.analyzer.run_analysis

    def changed_text(*args, **kwargs):
        result = execute(*args, **kwargs)
        flow.pr["body"] = "Changed PR context during analysis"
        return result

    monkeypatch.setattr(flow.worker.analyzer, "run_analysis", changed_text)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    assert flow.comments == []
    assert not any(method in ("POST", "PATCH") and "/comments" in path for method, path in flow.calls)
    assert flow.store._get_connection().execute("SELECT status FROM runs;").fetchone()[0] == "superseded"


def test_configured_draft_opt_out_stores_results_without_writes(app_flow) -> None:
    flow = app_flow
    assert flow.config.publish.draft_comments is False
    flow.pr["draft"] = True
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is True
    assert flow.comments == []
    run_id = flow.store._get_connection().execute("SELECT run_id FROM runs;").fetchone()[0]
    manifest = json.loads((flow.state / "runs/99/7" / run_id / "manifest.json").read_text())
    assert manifest["publication"]["status"] == "not_attempted"
    assert manifest["publication"]["reason"] == "draft_comments_disabled"
