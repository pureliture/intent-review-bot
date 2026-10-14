"""Two real Workers share signed intake, SQLite, fixed Git and admitted fixture HTTP."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import signal
import sqlite3
import stat
import threading
from collections import Counter
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from review_bot.adapters import llamacpp
from review_bot.adapters.git import GitRepositoryReader
from review_bot.adapters.github.client import GitHubApiClient
from review_bot.adapters.llamacpp import LLAMACPP_PROFILE, REVIEW_MODEL, LlamaCppModelClient
from review_bot.adapters.sqlite.model_admission import SQLiteModelAdmission
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import load_review_config_from_str
from review_bot.domain.models_v1 import ModelRequest
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop
from tests.integration.test_intent_flow import GOAL, SECRET, IntentFixtureModel, SyntheticGitRepo

WAIT_SECONDS = 10
MODEL_URL = "http://localhost:8080/v1"
STAGES = ("extract_goals", "observe_changes", "assess")
HEAD_SOURCE = "from threading import Lock\nlock = Lock()\ndef handle():\n    with lock:\n        return 1\n"


class ControlledLlamaHTTP:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.releases = [threading.Event() for _ in range(4)]
        self.hold_stage = STAGES[0]
        self.hold_limit = 4
        self.held = []
        self.requests = []
        self.calls = []
        self.prompts = set()
        self.tokenized = set()
        self.active = 0
        self.peak = 0
        self.errors = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        with self.condition:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((request.method, request.url.path))
        try:
            assert request.url.host == "localhost" and request.url.port == 8080
            assert self.active <= 2, "The shared HTTP admission limit was exceeded"
            return self._respond(request)
        except BaseException as exc:
            with self.condition:
                self.errors.append(exc)
                self.condition.notify_all()
            raise
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()

    def _respond(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in ("/v1/models", "/props"):
            assert request.method == "GET" and request.content == b""
            body = ({"object": "list", "data": [{"id": REVIEW_MODEL}]} if path == "/v1/models" else {
                "chat_template": "synthetic-system-user-template",
                "default_generation_settings": {"n_ctx": 262144}, "total_slots": 2,
            })
            return httpx.Response(200, json=body)
        assert request.method == "POST"
        payload = json.loads(request.content)
        if path == "/tokenize":
            assert set(payload) == {"content", "add_special", "parse_special", "with_pieces"}
            assert payload["add_special"] is True and payload["parse_special"] is True
            assert payload["with_pieces"] is False
            with self.condition:
                assert payload["content"] in self.prompts
                self.tokenized.add(payload["content"])
            return httpx.Response(200, json={"tokens": list(range(100))})
        messages = payload["messages"]
        assert len(messages) == 2 and [item["role"] for item in messages] == ["system", "user"]
        assert all(set(item) == {"role", "content"} and isinstance(item["content"], str) for item in messages)
        prompt = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        if path == "/apply-template":
            assert set(payload) == {"messages"}
            with self.condition:
                self.prompts.add(prompt)
            return httpx.Response(200, json={"prompt": prompt})
        assert path == "/v1/chat/completions"
        assert payload["model"] == REVIEW_MODEL
        assert payload["stream"] is False and payload["temperature"] == payload["seed"] == 0
        assert payload["n"] == 1 and 0 < payload["max_tokens"] <= 4096
        assert payload["stop"] == [] and payload["cache_prompt"] is False and payload["n_keep"] == -1
        assert payload["response_format"]["type"] == "json_schema"
        schema = payload["response_format"]["schema"]
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        model_request = ModelRequest(1, REVIEW_MODEL, messages[0]["content"], messages[1]["content"],
            30, payload["max_tokens"], schema)
        data = json.loads(model_request.user_content)
        assert data["stage"] in STAGES
        with self.condition:
            assert prompt in self.tokenized, "Generation must use the measured system/user messages"
            self.requests.append(model_request)
            gate = None
            if data["stage"] == self.hold_stage and len(self.held) < self.hold_limit:
                gate = self.releases[len(self.held)]
                self.held.append(model_request)
            self.condition.notify_all()
        if gate is not None:
            assert gate.wait(WAIT_SECONDS), "The test did not release its held generation"
        response = IntentFixtureModel().complete(model_request)
        assert set(json.loads(response.text)) == set(schema["required"])
        assert response.usage == {"input_tokens": 100, "output_tokens": 50}
        return httpx.Response(200, json={
            "model": REVIEW_MODEL,
            "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": response.text}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        })

    def wait_for_held(self, count: int) -> None:
        with self.condition:
            ready = self.condition.wait_for(lambda: len(self.held) >= count or self.errors, WAIT_SECONDS)
            self.check_errors()
            assert ready and len(self.held) >= count, "Workers did not reach the expected generation boundary"

    def check_errors(self) -> None:
        if self.errors:
            raise self.errors[0]

    def release_all(self) -> None:
        for event in self.releases:
            event.set()


class GitHubFixture:
    def __init__(self, prs) -> None:
        self.prs = prs
        self.calls = []
        self.comments = []
        self.allow_comments = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append((request.method, path))
        assert request.url.host == "api.github.com"
        if path.startswith("/repos/org/repo/pulls/"):
            assert request.method == "GET"
            return httpx.Response(200, json=deepcopy(self.prs[int(path.rsplit("/", 1)[1])]))
        if path == "/repos/org/repo/issues/12":
            assert request.method == "GET"
            return httpx.Response(200, json={"number": 12, "title": "Requests must be serialized",
                "body": "Related synthetic details, not instructions.", "updated_at": "2026-10-01T12:00:00Z"})
        if path.endswith("/comments"):
            assert self.allow_comments, "Artifact-only reviews must not reach comment HTTP"
            if request.method == "GET":
                return httpx.Response(200, json=self.comments)
            assert request.method == "POST"
            comment = {"id": 701, "body": json.loads(request.content)["body"],
                "user": {"login": "review-app[bot]", "type": "Bot"}}
            self.comments.append(comment)
            return httpx.Response(201, json=comment)
        raise AssertionError("Unexpected synthetic GitHub boundary")


class WorkerExecution:
    def __init__(self, worker: WorkerLoop, count: int, stop: threading.Event) -> None:
        self.worker = worker
        self.count = count
        self.stop = stop
        self.results = []
        self.error = None
        self.thread = threading.Thread(target=self._run, name=worker.worker_id, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.worker.prepare_models()
            for _ in range(self.count):
                if self.stop.is_set():
                    break
                self.results.append(self.worker.claim_and_execute_once())
        except BaseException as exc:
            self.error = exc
        finally:
            self.worker.job_store.close()

    def join(self) -> None:
        self.thread.join(WAIT_SECONDS)
        assert not self.thread.is_alive(), "Worker did not finish within the bounded test budget"
        if self.error is not None:
            raise self.error


@pytest.fixture
def concurrency_flow(tmp_path: Path, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch):
    repo_path = tmp_path / "git"
    repo_path.mkdir(mode=0o700)
    repo = SyntheticGitRepo(repo_path)
    repo.write_file("app.py", "def handle():\n    return 1\n")
    repo.write_file("tests/test_app.py", "raise RuntimeError('target tests must never execute')\n")
    repo.write_file("outside.py", "raise RuntimeError('target source must never execute')\n")
    base = repo.commit("Synthetic base")
    repo.write_file("app.py", HEAD_SOURCE)
    head = repo.commit("Synthetic locked source")
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="github_app", available_modes=["intent"], selector_uses_model=False,
        selector_policy="all_available", worker_concurrency=2, task_timeout_seconds=40)
    raw["service"].update(installation_allowlist=[], repository_allowlist=[])
    raw["publish"]["mode"] = "artifact_only"
    raw["intent"] = {"context_paths": ["tests/test_app.py"]}
    raw["model"] = {"profile": LLAMACPP_PROFILE}
    raw.pop("architecture")
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    prs = {number: {"number": number, "state": "open", "title": GOAL, "body": "Fixes #12", "draft": False,
        "head": {"sha": head, "ref": "feature/lock"},
        "base": {"sha": base, "ref": "main", "repo": {"id": 99, "full_name": "org/repo"}}, "labels": []}
        for number in range(7, 11)}
    http = ControlledLlamaHTTP()
    github_http = GitHubFixture(prs)
    client_factory = httpx.Client

    def model_http_client(**kwargs):
        assert kwargs["trust_env"] is False and kwargs["follow_redirects"] is False
        return client_factory(transport=httpx.MockTransport(http), **kwargs)

    def no_network(*args, **kwargs):
        raise AssertionError("Real network transport is forbidden in this integration test")

    monkeypatch.setattr(llamacpp, "httpx", SimpleNamespace(**{**vars(httpx), "Client": model_http_client}))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    database = state / "jobs.sqlite"
    database.touch(mode=0o600, exist_ok=False)
    model_root = state / "model"
    model_root.mkdir(mode=0o700)
    model_database = model_root / "requests.sqlite"
    stores = [SQLiteJobStore(database) for _ in range(4)]
    stop = threading.Event()
    analyzer_factory = Mock(side_effect=AssertionError("B must not initialize PMD or run target code"))
    readers = []
    workers = []
    executions = []
    saved_signals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for index in range(2):
        github = GitHubApiClient("synthetic-token", "org/repo", repository_id=99, bot_login="review-app[bot]",
            transport=httpx.MockTransport(github_http))
        model = LlamaCppModelClient(api_url=MODEL_URL, timeout_seconds=30)
        model.request_admission = SQLiteModelAdmission(model_database, max_concurrency=2, poll_interval=0.01)

        def reader_factory(run, pr_data, client):
            readers.append((run.run_id, run.pr_number, pr_data["base"]["sha"], pr_data["head"]["sha"]))
            return GitRepositoryReader(repo.repo_dir)

        workers.append(WorkerLoop(config, stores[index + 1], model_client=model, worker_id=f"intent-worker-{index}",
            github_client_factory=lambda run, client=github: client, repo_reader_factory=reader_factory,
            artifact_root=state, analyzer_factory=analyzer_factory, lease_duration=30, stop_event=stop))

    def start(index, count):
        execution = WorkerExecution(workers[index], count, stop)
        executions.append(execution)
        return execution

    with TestClient(create_webhook_app(config, stores[0], secret=SECRET)) as client:
        try:
            yield SimpleNamespace(config=config, state=state, model_database=model_database, store=stores[3],
                client=client, prs=prs, http=http, github_http=github_http, workers=workers, start=start,
                analyzer_factory=analyzer_factory, readers=readers, base=base, head=head)
        finally:
            stop.set()
            http.release_all()
            try:
                for execution in executions:
                    execution.join()
            finally:
                for store in stores:
                    store.close()
                for sig, handler in saved_signals.items():
                    signal.signal(sig, handler)


def send_signed(flow, number: int, delivery: str, action: str = "opened"):
    payload = {"action": action, "number": number, "installation": {"id": 44},
        "repository": {"id": 99, "full_name": "org/repo", "name": "repo"}, "pull_request": flow.prs[number]}
    body = json.dumps(payload).encode()
    response = flow.client.post("/webhook", content=body, headers={
        "X-GitHub-Event": "pull_request", "X-GitHub-Delivery": delivery,
        "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
    })
    assert response.status_code == 202 and response.json() == {"status": "accepted"}, response.text


def runs(flow):
    columns = ("run_id", "pr_number", "generation", "status", "worker_id", "attempts")
    return [dict(zip(columns, row)) for row in flow.store._get_connection().execute(
        "SELECT run_id, pr_number, generation, status, worker_id, attempts FROM runs ORDER BY rowid").fetchall()]


def admission_states(flow):
    with closing(sqlite3.connect(flow.model_database)) as connection:
        return connection.execute("SELECT state, caller_limit FROM model_admission").fetchall()


def assert_private_result(flow, run, *, publication="artifact_only"):
    root = flow.state / "runs" / "99" / str(run["pr_number"]) / run["run_id"]
    manifest = json.loads((root / "manifest.json").read_text())
    result = json.loads((root / "B/result.json").read_text())
    task = json.loads((root / "B/input.json").read_text())
    receipt = json.loads((root / "B/receipt.json").read_text())
    assert run["status"] == manifest["status"] == "succeeded"
    assert set(manifest["modes"]) == {"intent"}
    mode = manifest["modes"]["intent"]
    for key in ("execution_status", "error_code", "publication", "input_fingerprint", "reused", "usage", "model_delivery"):
        assert mode[key] == receipt[key]
    for key in ("status", "reason", "input_hash", "prompt_version", "prompt_hash"):
        assert mode[key] == result[key]
    assert mode["core_fingerprint"] == receipt["fingerprint"] == result["run_key"]
    assert receipt["coverage"] == result["scope"] and receipt["model_delivery"] == result["model_delivery"]
    assert receipt["b_result"] == result and receipt["a_result"] is None
    assert receipt["generation"] == run["generation"] and receipt["execution_status"] == "completed"
    assert json.loads((root / "selection.json").read_text())["selected_modes"] == ["intent"]
    assert result["review_type"] == "B" and (result["status"], result["reason"]) == ("completed", "validated")
    assert result["validation_errors"] == [] and result["scope"]["tests_executed_by_bot"] is False
    assert result["base_sha"] == task["base"]["sha"] == flow.base
    assert result["head_sha"] == task["head"]["sha"] == flow.head
    assert task["snapshot"]["pr_number"] == run["pr_number"]
    assert task["reviewed_paths"] == ["app.py", "tests/test_app.py"] and "outside.py" not in json.dumps(task)
    assert next(item["content"] for item in task["head"]["files"] if item["path"] == "app.py") == HEAD_SOURCE
    assert result["goals"]["declared_goals"][0]["quote"] == GOAL
    assert {key: result["usage"][key] for key in ("calls", "input_tokens", "output_tokens")} == {
        "calls": 3, "input_tokens": 300, "output_tokens": 150,
    }
    assert [stage["stage"] for stage in result["stages"]] == list(STAGES)
    for stage in result["stages"]:
        assert stage["measured_input_tokens"] == 100 and len(stage["attempts"]) == 1
        assert stage["token_measurement_delivery"] == "sent" and stage["token_measurement_model_calls"] == 0
        assert stage["token_measurement_usage"] == {"input_tokens": 0, "output_tokens": 0}
        assert stage["attempts"][0]["usage"] == {"input_tokens": 100, "output_tokens": 50}
    if publication == "artifact_only":
        assert receipt["publication"]["reason"] == publication and receipt["publication"]["status"] == "not_attempted"
    else:
        assert receipt["publication"]["status"] == publication
    assert list(root.rglob("result.json")) == [root / "B/result.json"] and not (root / "A").exists()
    for name in ("input.json", "goals.json", "observations.json", "result.json", "scope.json", "usage.json",
        "validation.json", "comment.md", "receipt.json"):
        assert (root / "B" / name).is_file()
    for path in (root, *root.rglob("*")):
        metadata = path.lstat()
        assert metadata.st_uid == os.geteuid()
        assert stat.S_IMODE(metadata.st_mode) == (0o700 if path.is_dir() else 0o600)
        assert path.is_dir() or metadata.st_nlink == 1
    return result


def test_four_signed_prs_run_on_two_workers_with_private_b_artifacts(concurrency_flow):
    flow = concurrency_flow
    for number in flow.prs:
        send_signed(flow, number, f"synthetic-pr-{number}")
    assert len(runs(flow)) == 4 and {run["status"] for run in runs(flow)} == {"queued"}
    executions = [flow.start(index, 2) for index in range(2)]
    flow.http.wait_for_held(2)
    initial = runs(flow)
    assert Counter(run["status"] for run in initial) == {"running": 2, "queued": 2}
    assert {run["worker_id"] for run in initial if run["status"] == "running"} == {worker.worker_id for worker in flow.workers}
    assert flow.store.claim_next_run("synthetic-third-claim", 30) is None
    assert runs(flow) == initial, "A refused claim must not consume either remaining PR"
    assert admission_states(flow) == [("sent", 2), ("sent", 2)]
    assert Counter(json.loads(request.user_content)["stage"] for request in flow.http.requests) == {STAGES[0]: 2}
    flow.http.releases[0].set()
    flow.http.wait_for_held(3)
    progressed = runs(flow)
    assert Counter(run["status"] for run in progressed) == {"succeeded": 1, "running": 2, "queued": 1}
    initial_prs = {run["pr_number"] for run in initial if run["status"] == "running"}
    assert any(run["status"] == "running" and run["pr_number"] not in initial_prs for run in progressed)
    assert not flow.http.releases[1].is_set(), "The other original request must remain held"
    assert flow.store.claim_next_run("synthetic-third-claim", 30) is None
    assert admission_states(flow) == [("sent", 2), ("sent", 2)]
    flow.http.release_all()
    for execution in executions:
        execution.join()
        assert execution.results == [True, True], runs(flow)
    flow.http.check_errors()
    completed = runs(flow)
    assert len(completed) == 4 and {run["status"] for run in completed} == {"succeeded"}
    assert {run["pr_number"] for run in completed} == set(flow.prs)
    assert len({run["run_id"] for run in completed}) == 4 and all(run["attempts"] == 1 for run in completed)
    results = [assert_private_result(flow, run) for run in completed]
    assert len({result["run_key"] for result in results}) == 4
    assert len(flow.readers) == 4 and {item[0] for item in flow.readers} == {run["run_id"] for run in completed}
    assert all(item[2:] == (flow.base, flow.head) for item in flow.readers)
    assert Counter(json.loads(request.user_content)["stage"] for request in flow.http.requests) == {stage: 4 for stage in STAGES}
    assert {path for _, path in flow.http.calls} == {"/v1/models", "/props", "/apply-template", "/tokenize", "/v1/chat/completions"}
    assert flow.http.peak == 2 and flow.http.active == 0 and admission_states(flow) == []
    assert flow.store.claim_next_run("synthetic-drained-claim", 30) is None
    deliveries = flow.store._get_connection().execute(
        "SELECT delivery_id, action, installation_id, repository_id, pr_number, status FROM deliveries ORDER BY pr_number").fetchall()
    assert deliveries == [(f"synthetic-pr-{number}", "opened", 44, 99, number, "received") for number in flow.prs]
    assert all(method == "GET" for method, _ in flow.github_http.calls) and flow.github_http.comments == []
    flow.analyzer_factory.assert_not_called()


def test_new_generation_fences_held_old_response_and_only_publishes_new_result(concurrency_flow):
    flow = concurrency_flow
    flow.config.publish.mode = "comment"
    flow.config.publish.b_comments_enabled = True
    flow.github_http.allow_comments = True
    flow.http.hold_stage = STAGES[-1]
    flow.http.hold_limit = 1
    send_signed(flow, 7, "synthetic-old-generation")
    old_execution = flow.start(0, 1)
    flow.http.wait_for_held(1)
    old = runs(flow)[0]
    assert old["status"] == "running" and old["generation"] == 1
    flow.prs[7]["body"] += "\nClarified synthetic second generation."
    send_signed(flow, 7, "synthetic-new-generation", "edited")
    pending = runs(flow)
    assert [(run["generation"], run["status"]) for run in pending] == [(1, "running"), (2, "queued")]
    assert flow.store.claim_next_run("synthetic-same-pr-claim", 30) is None
    assert flow.store.get_pr_request(99, 7).desired_generation == 2
    flow.http.release_all()
    old_execution.join()
    flow.http.check_errors()
    assert old_execution.results == [False]
    fenced = runs(flow)
    assert [(run["generation"], run["status"]) for run in fenced] == [(1, "superseded"), (2, "queued")]
    old_root = flow.state / "runs/99/7" / old["run_id"]
    assert not (old_root / "B/result.json").exists()
    assert flow.github_http.comments == [] and all(method == "GET" for method, _ in flow.github_http.calls)
    assert flow.store._get_connection().execute("SELECT COUNT(*) FROM publication_receipts").fetchone()[0] == 0
    assert admission_states(flow) == []
    new_execution = flow.start(1, 1)
    new_execution.join()
    flow.http.check_errors()
    assert new_execution.results == [True], runs(flow)
    completed = runs(flow)
    assert [(run["generation"], run["status"]) for run in completed] == [(1, "superseded"), (2, "succeeded")]
    assert completed[0]["run_id"] != completed[1]["run_id"]
    assert_private_result(flow, completed[1], publication="succeeded")
    assert len(flow.github_http.comments) == 1
    assert [(method, path) for method, path in flow.github_http.calls if method != "GET"] == [
        ("POST", "/repos/org/repo/issues/7/comments"),
    ]
    receipt = flow.store._get_connection().execute("SELECT run_id, status, comment_id FROM publication_receipts").fetchone()
    assert receipt == (completed[1]["run_id"], "succeeded", 701)
    assert len(flow.http.requests) == 6 and admission_states(flow) == []
    assert flow.store.claim_next_run("synthetic-drained-claim", 30) is None
    flow.analyzer_factory.assert_not_called()