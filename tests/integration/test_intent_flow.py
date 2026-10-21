"""Signed intake → real SQLite/Git → real B core; only HTTP/model calls are fixtures."""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from review_bot.adapters.git import GitRepositoryReader
from review_bot.adapters.github.client import GitHubApiClient
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.application.architecture_review import MockModelClient
from review_bot.config import load_review_config_from_str
from review_bot.domain.models_v1 import ModelResponse
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop
from tests.conftest import SyntheticGitRepo
from tests.unit.test_delta_adversarial import make_test_run
from tests.fixtures.partition_review import PartitionModelClient

SECRET = "synthetic-intent-webhook-secret"
GOAL = "Serialize request handling"


class IntentFixtureModel(PartitionModelClient):
    def __init__(self) -> None:
        super().__init__()

    def complete(self, request):
        data = json.loads(request.user_content)
        if data["stage"] not in ("extract_goals", "observe_changes", "assess"):
            return super().complete(request)
        # Explicit legacy/offline HTTP fixtures still import this response helper.
        self.requests.append(request)
        if data["stage"] == "extract_goals":
            body = {"declared_goals": [{"goal_id": "G1", "statement": GOAL,
                "quote": GOAL, "evidence_ids": [data["evidence"][0]["id"]]}],
                "declared_constraints": [], "unavailable_context": data["unavailable_context"]}
        elif data["stage"] == "observe_changes":
            ref = next(item["id"] for item in data["evidence"] if item["kind"] == "source" and item["locator"]["version"] == "head")
            body = {"implementation_observations": [{"observation_id": "O1",
                "statement": "The prepared source uses a lock.", "evidence_ids": [ref]}],
                "intent_hypotheses": [], "assumptions": []}
        else:
            refs = [data["goals"]["declared_goals"][0]["evidence_ids"][0],
                data["observations"]["implementation_observations"][0]["evidence_ids"][0]]
            body = {"summary": "Runtime behavior was not verified.", "findings": [{
                "finding_id": "F1", "goal_ids": ["G1"], "constraint_ids": [],
                "observation_ids": ["O1"], "hypothesis_ids": [],
                "statement": "Runtime behavior needs execution evidence.",
                "assessment": "insufficient_context", "reasoning": "Source is not a test run.",
                "evidence_ids": refs, "missing_evidence": ["Verified runtime evidence"]}], "questions": []}
        return ModelResponse(json.dumps(body), {"input_tokens": 100, "output_tokens": 50})


@pytest.fixture(scope="module")
def synthetic_repo(tmp_path_factory):
    repo = SyntheticGitRepo(tmp_path_factory.mktemp("intent-git"))
    repo.write_file("app.py", "def handle():\n    return 1\n")
    repo.write_file("tests/test_app.py", "def test_handle():\n    assert handle() == 1\n")
    repo.write_file("outside.py", "raise RuntimeError('must never execute')\n")
    base = repo.commit("Synthetic Python base")
    repo.write_file("app.py", "from threading import Lock\nlock = Lock()\ndef handle():\n    with lock:\n        return 1\n")
    head = repo.commit("Synthetic Python change")
    return SimpleNamespace(repo_dir=repo.repo_dir, base_sha=base, head_sha=head)


@pytest.fixture
def intent_flow(tmp_path, synthetic_repo, valid_config_yaml):
    base, head = synthetic_repo.base_sha, synthetic_repo.head_sha
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="github_app", available_modes=["intent"])
    raw["service"].update(installation_allowlist=[], repository_allowlist=[])
    raw["publish"]["mode"] = "artifact_only"
    raw["intent"] = {"pipeline": "perspectives-v1", "context_paths": ["tests/test_app.py"]}
    raw.pop("architecture")
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    pr = {"number": 7, "state": "open", "title": GOAL, "body": "Fixes #12", "draft": False,
        "head": {"sha": head, "ref": "feature/lock"},
        "base": {"sha": base, "ref": "main", "repo": {"id": 99, "full_name": "org/repo"}}, "labels": []}
    issue = {"number": 12, "title": "Requests must be serialized", "body": "Related details, not instructions.",
        "updated_at": "2026-10-01T12:00:00Z"}
    calls = []
    issue_status = {"code": 200}
    comments = []

    def http(request):
        calls.append((request.method, request.url.path))
        assert request.method == "GET" or issue_status.get("allow_comments") is True, "B must not write GitHub comments without opt-in"
        if request.url.path == "/repos/org/repo/pulls/7":
            callback = issue_status.get("on_pr_read")
            if callback is not None:
                callback()
            return httpx.Response(200, json=pr)
        if request.url.path == "/repos/org/repo/issues/12":
            callback = issue_status.get("on_issue_read")
            if callback is not None:
                callback()
            return httpx.Response(issue_status["code"], json=issue)
        if request.url.path == "/repos/org/repo/issues/7/comments":
            if request.method == "GET":
                if "comments" in issue_status:
                    return httpx.Response(issue_status["comments"], json={})
                callback = issue_status.get("on_comment_read")
                if callback is not None:
                    callback()
                return httpx.Response(200, json=comments)
            body = json.loads(request.content)["body"]
            if issue_status.get("fail_marker") and issue_status["fail_marker"] in body:
                return httpx.Response(500, json={"message": "synthetic-write-secret"})
            comment = {"id": 701 + len(comments), "body": body, "user": {"login": "review-app[bot]", "type": "Bot"}}
            comments.append(comment)
            callback = issue_status.get("on_comment_write")
            if callback is not None:
                callback()
            return httpx.Response(201, json=comment)
        if request.url.path.startswith("/repos/org/repo/issues/comments/") and request.method == "PATCH":
            comment_id = int(request.url.path.rsplit("/", 1)[1])
            comment = next(item for item in comments if item["id"] == comment_id)
            comment["body"] = json.loads(request.content)["body"]
            callback = issue_status.get("on_comment_write")
            if callback is not None:
                callback()
            return httpx.Response(200, json=comment)
        raise AssertionError("Unexpected HTTP target")

    github = GitHubApiClient("synthetic-token", "org/repo", repository_id=99, bot_login="review-app[bot]",
        transport=httpx.MockTransport(http))
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    store = SQLiteJobStore(state / "jobs.sqlite")
    model = IntentFixtureModel()
    def select_modes(payload, body):
        if payload["stage"] == "select_review_scope":
            modes = flow.mode_selection
            if modes is None:
                modes = payload["state"]["available_modes"]
            key = "review:" + (",".join(modes) if modes else "none")
            body["choice"] = next(option["label"] for option in payload["options"] if option["key"] == key)
        return body

    planning_model = PartitionModelClient(name="tev", group_all=True, transform=select_modes)
    reader = GitRepositoryReader(synthetic_repo.repo_dir)
    analyzer = Mock()
    analyzer.run_analysis.side_effect = AssertionError("B must not initialize/use Java analysis")
    worker = WorkerLoop(config, store, analyzer=analyzer, model_client=model, planning_client=planning_model, worker_id="intent-worker",
        github_client_factory=lambda run: github,
        repo_reader_factory=lambda run, pr_data, client: reader, artifact_root=state)
    with TestClient(create_webhook_app(config, store, secret=SECRET)) as client:
        try:
            flow = SimpleNamespace(config=config, state=state, store=store, model=model, worker=worker,
                client=client, pr=pr, issue=issue, calls=calls, reader=reader, analyzer=analyzer,
                issue_status=issue_status, comments=comments, github=github,
                planning_model=planning_model, mode_selection=None)
            yield flow
        finally:
            store.close()


def send_event(flow, delivery="intent-first", action="opened"):
    payload = {"action": action, "number": 7, "installation": {"id": 44},
        "repository": {"id": 99, "full_name": "org/repo", "name": "repo"}, "pull_request": flow.pr}
    body = json.dumps(payload).encode()
    return flow.client.post("/webhook", content=body, headers={
        "X-GitHub-Event": "pull_request", "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()})


def output(flow, requested_run_id=None):
    connection = flow.store._get_connection()
    if requested_run_id is None:
        run_id, status = connection.execute("SELECT run_id, status FROM runs ORDER BY rowid DESC LIMIT 1").fetchone()
    else:
        run_id, status = connection.execute("SELECT run_id, status FROM runs WHERE run_id = ?", (requested_run_id,)).fetchone()
    return flow.state / "runs/99/7" / run_id, status


def test_signed_python_pr_creates_private_validated_b_without_java(intent_flow):
    flow = intent_flow
    response = send_event(flow)
    assert response.status_code == 202, response.text
    assert response.json() == {"status": "accepted"}
    assert flow.worker.run_once() is True, flow.store._get_connection().execute("SELECT status, error FROM runs").fetchall()
    root, status = output(flow)
    result = json.loads((root / "B/result.json").read_text())
    task = json.loads((root / "B/input.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    receipt = json.loads((root / "B/receipt.json").read_text())
    assert status == manifest["status"] == "succeeded"
    assert (result["status"], result["reason"]) == ("completed", "validated")
    assert len(flow.model.requests) == result["usage"]["calls"] == 3
    assert {item["kind"] for item in task["snapshot"]["evidence"]} == {"pr_text", "issue_text"}
    assert task["reviewed_paths"] == ["app.py", "tests/test_app.py"]
    assert "outside.py" not in json.dumps(task)
    assert result["scope"]["tests_executed_by_bot"] is False
    assert receipt["b_result"] == result and receipt["a_result"] is None
    assert manifest["modes"]["intent"]["publication"]["reason"] == "artifact_only"
    flow.analyzer.run_analysis.assert_not_called()
    assert not (root / "A/result.json").exists()
    for name in ("input.json", "partition.json", "perspectives.json", "fact-verification.json",
        "result.json", "scope.json", "usage.json", "validation.json", "comment.md", "receipt.json"):
        assert (root / "B" / name).stat().st_mode & 0o777 == 0o600
    assert (root / "B").stat().st_mode & 0o777 == 0o700


def use_modes(flow, modes, *, context_paths=None, task_timeout=900):
    if "architecture" in modes and not getattr(flow, "gradle_java_change", False):
        # Mixed-mode fixtures must contain a real supported Java/Gradle change.
        # Preserve the original B sources so its independent assertions still apply.
        source = flow.reader
        repo_dir = flow.state / "mixed-gradle-java"
        repo_dir.mkdir()
        repo = SyntheticGitRepo(repo_dir)
        java = "src/main/java/demo/ModeFixture.java"
        for version in ("base", "head"):
            for path, content in source.read_tree_blobs(flow.pr[version]["sha"]).items():
                repo.write_file(path, content)
            repo.write_file("build.gradle", "plugins { id 'java' }\n")
            repo.write_file(java, "package demo; class ModeFixture { int value = " +
                ("0" if version == "base" else "1") + "; }\n")
            flow.pr[version]["sha"] = repo.commit("Supported mixed " + version)
        flow.reader = GitRepositoryReader(repo_dir)
        flow.worker.repo_reader_factory = lambda run, pr_data, client: flow.reader
        flow.gradle_java_change = True
    raw = flow.config.model_dump()
    raw["execution"].update(available_modes=list(modes), task_timeout_seconds=task_timeout)
    if context_paths is not None:
        raw["intent"]["context_paths"] = context_paths
    flow.config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    flow.worker.config = flow.config


def enable_a_fixture(flow):
    flow.analyzer.run_analysis.side_effect = lambda sha, *args: make_test_run(sha,
        source_files=("src/main/java/demo/ModeFixture.java",))


@pytest.mark.parametrize("modes", [("architecture",), ("architecture", "intent"), ("intent", "architecture")])
def test_selected_modes_keep_partition_a_and_b_results_separate(intent_flow, modes):
    flow = intent_flow
    use_modes(flow, modes)
    enable_a_fixture(flow)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is True
    root, status = output(flow)
    manifest = json.loads((root / "manifest.json").read_text())
    a = json.loads((root / "A/result.json").read_text())
    assert status == "succeeded" and a["schema_version"] == 13 and a["review_type"] == "A"
    assert a["status"] == "not_run", "Empty A static signals must not prevent B execution"
    assert flow.analyzer.run_analysis.call_count == 2
    assert set(manifest["modes"]) == set(modes)
    if "intent" in modes:
        b = json.loads((root / "B/result.json").read_text())
        assert b["status"] == "completed" and len(flow.model.requests) == 3
        assert manifest["modes"]["architecture"]["input_fingerprint"] != manifest["modes"]["intent"]["input_fingerprint"]
        assert manifest["modes"]["architecture"]["usage"] == a["usage"]
        assert manifest["modes"]["intent"]["usage"] == b["usage"]
    else:
        assert not (root / "B").exists() and flow.model.requests == []
        assert any(path == "/repos/org/repo/issues/12" for _, path in flow.calls)
        assert flow.planning_model.requests, "A selection uses the same frozen PR purpose and issue evidence"


@pytest.mark.parametrize("a_failure", ["analysis", "preparation", "review", "artifact", "publication"])
def test_a_local_failure_does_not_discard_valid_b(intent_flow, monkeypatch, a_failure):
    flow = intent_flow
    use_modes(flow, ("architecture", "intent"))
    if a_failure == "preparation":
        from review_bot.runtime.service import _architecture_analyzer

        flow.worker.analyzer = None
        factory = Mock(side_effect=lambda: _architecture_analyzer(flow.config, flow.state))
        flow.worker.analyzer_factory = factory
    elif a_failure == "review":
        enable_a_fixture(flow)
        original_review = flow.worker._review_partitioned

        def review_mode(task, mode, *args, **kwargs):
            if mode == "architecture":
                raise RuntimeError("synthetic-secret")
            return original_review(task, mode, *args, **kwargs)

        monkeypatch.setattr(flow.worker, "_review_partitioned", review_mode)
    elif a_failure == "artifact":
        enable_a_fixture(flow)
        from review_bot.runtime.artifacts import RunArtifacts

        original = RunArtifacts._write

        def write(artifacts, path, content):
            if path.startswith("A/"):
                raise OSError("synthetic-secret")
            return original(artifacts, path, content)

        monkeypatch.setattr(RunArtifacts, "_write", write)
    elif a_failure == "publication":
        enable_a_fixture(flow)
        flow.worker.config.publish.mode = "comment"
        flow.issue_status["comments"] = 403
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow)
    manifest = json.loads((root / "manifest.json").read_text())
    b = json.loads((root / "B/result.json").read_text())
    assert status == manifest["status"] == "failed"
    assert b["status"] == "completed" and b["usage"]["calls"] == len(flow.model.requests) == 3
    assert manifest["modes"]["architecture"]["execution_status"] == "failed"
    assert manifest["modes"]["intent"]["execution_status"] == "completed"
    assert "synthetic-secret" not in (root / "manifest.json").read_text()
    if a_failure == "preparation":
        factory.assert_called_once()
    if a_failure != "artifact":
        assert (root / "A/receipt.json").is_file()


def test_b_invalid_response_retains_called_stage_and_does_not_suppress_a(intent_flow):
    flow = intent_flow
    use_modes(flow, ("intent", "architecture"))
    enable_a_fixture(flow)
    failed_model = MockModelClient([ModelResponse("{}", {"input_tokens": 100, "output_tokens": 10})], input_tokens=100)
    flow.worker.model_client = failed_model
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow)
    b = json.loads((root / "B/result.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    assert status == "failed" and b["status"] == "failed"
    assert b["usage"]["calls"] == len(failed_model.requests) == 1
    assert b["usage"]["input_tokens"] == 100
    assert b["stages"][1]["attempts"] == [] and b["stages"][2]["attempts"] == []
    assert (root / "A/result.json").is_file() and flow.analyzer.run_analysis.call_count == 2
    assert manifest["modes"]["intent"]["execution_status"] == "failed"
    assert manifest["modes"]["architecture"]["execution_status"] == "not_run"


@pytest.mark.parametrize("case", ["model_unset", "missing_path", "overflow", "excluded_changed_path"])
def test_b_not_run_is_explicit_and_does_not_shrink_scope(intent_flow, case, monkeypatch):
    flow = intent_flow
    if case == "model_unset":
        flow.worker.model_client = None
        expected = "partition_model_profile_unset"
    elif case == "missing_path":
        use_modes(flow, ("intent",), context_paths=["tests/test_app.py", "tests/missing.py"])
        expected = "incomplete_code_input"
    elif case == "overflow":
        flow.model.profile = flow.model.profile.__class__("mock", "small-profile", 100, is_mock=True)
        expected = "fixed_request_exceeds_input_budget"
    else:
        from review_bot.domain.envelope_v2 import ChangedFile

        original = flow.reader.get_changed_files
        monkeypatch.setattr(flow.reader, "get_changed_files", lambda *args: (*original(*args), ChangedFile(".env")))
        expected = "incomplete_code_input"
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow)
    plan = json.loads((root / "execution-plan.json").read_text())
    task = json.loads((root / "planning-input.json").read_text())
    assert status == "failed" and (plan["status"], plan["reason"]) == ("not_run", expected)
    assert not (root / "B/result.json").exists()
    assert "app.py" in task["reviewed_paths"] and "tests/test_app.py" in task["reviewed_paths"]
    if case == "missing_path":
        assert "tests/missing.py" in task["reviewed_paths"]
    assert plan["usage"]["calls"] == 0 and flow.model.requests == []
    assert not flow.planning_model.requests


def test_issue_text_and_missing_state_change_frozen_snapshot_and_input_hash(intent_flow):
    flow = intent_flow
    snapshots = []
    inputs = []
    for index in range(3):
        if index == 1:
            flow.issue["body"] = "Changed requirement text"
        elif index == 2:
            flow.issue_status["code"] = 403
        assert send_event(flow, f"source-change-{index}", "edited").status_code == 202
        assert flow.worker.run_once() is True
        root, _ = output(flow)
        snapshots.append(json.loads((root / "snapshot.json").read_text())["snapshot_id"])
        inputs.append(json.loads((root / "B/input.json").read_text())["input_hash"])
    assert len(set(snapshots)) == len(set(inputs)) == 3
    task = json.loads((root / "B/input.json").read_text())
    assert task["snapshot"]["unavailable_sources"] == [{"source": "issue:12", "reason": "forbidden"}]


def test_scope_and_text_are_frozen_before_selection_and_analyzer_is_lazy(intent_flow):
    from review_bot.ports import ModeSelectorPort

    flow = intent_flow
    use_modes(flow, ("architecture", "intent"))
    flow.worker.config.execution.selector_uses_model = True
    flow.worker.analyzer = None
    factory = Mock(side_effect=AssertionError("Unselected A must not initialize"))
    flow.worker.analyzer_factory = factory
    collected = Mock(wraps=flow.reader.read_tree_blobs)
    flow.reader.read_tree_blobs = collected
    old_body = flow.issue["body"]

    flow.mode_selection = ("intent",)
    original_choice = flow.planning_model.transform
    def select(payload, body):
        if payload["stage"] == "select_review_scope":
            assert collected.call_count >= 2
            flow.issue["body"] = "Newer text must not mutate the selected snapshot"
        return original_choice(payload, body)
    flow.planning_model.transform = select
    flow.worker.mode_selector = Mock(spec=ModeSelectorPort)
    flow.worker.mode_selector.select_modes.side_effect = AssertionError("Legacy selection must not replace TEV")
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    root, _ = output(flow)
    task = json.loads((root / "B/input.json").read_text())
    assert task["snapshot"]["evidence"][1]["content"]["body"] == old_body
    assert any(set(call.kwargs["paths"]) >= {"app.py", "src/main/java/demo/ModeFixture.java", "tests/test_app.py"}
        for call in collected.call_args_list)
    assert all(call.kwargs.get("paths") is not None and "outside.py" not in call.kwargs["paths"]
        for call in collected.call_args_list)
    flow.worker.mode_selector.select_modes.assert_not_called()
    factory.assert_not_called()
    assert not (root / "A").exists()


@pytest.mark.parametrize("failure", ["authorization", "git_preparation", "git_a_collection"])
def test_common_collection_failure_aborts_both_modes(intent_flow, failure, monkeypatch):
    flow = intent_flow
    use_modes(flow, ("architecture", "intent"))
    enable_a_fixture(flow)
    if failure == "authorization":
        flow.issue_status["code"] = 401
    elif failure == "git_preparation":
        monkeypatch.setattr(flow.reader, "resolve_merge_base", Mock(side_effect=OSError("synthetic-secret")))
    else:
        original = flow.reader.read_tree_blobs

        def read(sha, *, paths=None):
            if paths is None:
                raise OSError("synthetic-secret")
            return original(sha, paths=paths)

        monkeypatch.setattr(flow.reader, "read_tree_blobs", read)
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow)
    assert status == "failed" and flow.model.requests == []
    assert not (root / "B/result.json").exists()
    error = flow.store._get_connection().execute("SELECT error FROM runs").fetchone()[0]
    assert "synthetic-secret" not in error


def test_publication_authorization_loss_stops_pending_b(intent_flow):
    flow = intent_flow
    use_modes(flow, ("architecture", "intent"))
    enable_a_fixture(flow)
    flow.worker.config.publish.mode = "comment"
    flow.issue_status["comments"] = 401
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow)
    assert status == "failed" and flow.model.requests == []
    assert (root / "A/result.json").is_file() and not (root / "B/result.json").exists()
    assert all(method == "GET" for method, _ in flow.calls)


@pytest.mark.parametrize("abort", ["stop", "closed", "generation", "lease", "deadline"])
def test_common_gates_stop_next_b_call_and_never_publish(intent_flow, monkeypatch, abort):
    flow = intent_flow
    use_modes(flow, ("intent",), task_timeout=2)
    clock = [100.0]
    for module in ("review_bot.runtime.worker", "review_bot.application.partition_review",
        "review_bot.application.model_execution"):
        monkeypatch.setattr(module + ".time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = flow.model.complete
    observed = {}

    def complete(request):
        response = original(request)
        observed["run_id"] = flow.worker._current_run_id
        assert json.loads(request.user_content)["stage"] == "review_partition"
        if abort == "stop":
            flow.worker.stop()
        elif abort == "closed":
            flow.pr["state"] = "closed"
            assert send_event(flow, "closed-after-goals", "closed").status_code == 202
        elif abort == "generation":
            assert send_event(flow, "new-generation", "edited").status_code == 202
        elif abort == "lease":
            flow.store._get_connection().execute("UPDATE runs SET lease_expires_at = 0")
            observed["recovered"] = flow.store.claim_next_run("replacement", 30)
            assert observed["recovered"] is not None
        else:
            clock[0] += 3
        return response

    flow.model.complete = complete
    assert send_event(flow).status_code == 202
    assert flow.worker.run_once() is False
    root, status = output(flow, observed["run_id"])
    assert len(flow.model.requests) == 1
    assert not (root / "B/result.json").exists()
    assert flow.store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0] == 0
    if abort in ("stop", "closed"):
        assert status == "cancelled"
    elif abort == "generation":
        assert status == "superseded"
    elif abort == "deadline":
        assert status == "failed"
        assert flow.store._get_connection().execute("SELECT reason FROM runs").fetchone()[0] == "task_deadline_exceeded"
    else:
        assert status == "running"
        assert observed["recovered"].run_id == root.name


def test_comment_configuration_keeps_b_automatic_posting_disabled(intent_flow):
    flow = intent_flow
    flow.worker.config.publish.mode = "comment"
    assert send_event(flow).status_code == 202 and flow.worker.run_once() is True
    root, status = output(flow)
    receipt = json.loads((root / "B/receipt.json").read_text())
    assert status == "succeeded" and receipt["publication"]["reason"] == "b_comments_disabled"
    assert all(method == "GET" for method, _ in flow.calls)
