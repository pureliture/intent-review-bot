"""HTTP intake, durable SQLite handoff, leased worker and private schema-3 results."""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import PrRequestRecord
from review_bot.runtime.intent_api import create_intent_app
from review_bot.runtime.mode_fingerprints import intent_input_fingerprint
from review_bot.runtime.worker import WorkerLoop
from tests.fixtures.intent_perspectives import PerspectiveModelClient, make_task


@pytest.fixture
def api_flow(tmp_path, valid_config_yaml, monkeypatch):
    monkeypatch.setenv("REVIEW_BOT_API_TOKEN", "synthetic-private-api-token")
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="offline", available_modes=["intent"])
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    previous = os.umask(0o077)
    store = SQLiteJobStore(root / "jobs.sqlite")
    model, selector = PerspectiveModelClient(), PerspectiveModelClient(selector=True)

    def forbidden(*args):
        raise AssertionError("A fixed input must not collect GitHub or repository data")

    flow = SimpleNamespace(root=root, store=store, config=config, model=model, selector=selector,
        forbidden=forbidden, task=make_task(), headers={"Authorization": "Bearer synthetic-private-api-token"})
    try:
        yield flow
    finally:
        store.close()
        os.umask(previous)


def app(flow):
    return create_intent_app(flow.config, flow.store, flow.root / "artifacts")


def worker(flow, **kwargs):
    return WorkerLoop(flow.config, flow.store, model_client=flow.model,
        intent_selection_client=flow.selector, github_client_factory=flow.forbidden,
        repo_reader_factory=flow.forbidden, analyzer_factory=flow.forbidden,
        artifact_root=flow.root / "artifacts", **kwargs)


def submit(client, flow, language="ko"):
    return client.post("/reviews/intent?language=" + language, json=asdict(flow.task), headers=flow.headers)


def test_api_survives_receiver_restart_and_worker_saves_queryable_result(api_flow):
    f = api_flow
    with TestClient(app(f)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = submit(client, f)
        assert response.status_code == 202, response.text
        urls = response.json()
        assert response.headers["location"] == urls["status_url"]
        assert not f.model.requests and not f.selector.requests
        assert client.get(urls["result_url"], headers=f.headers).status_code == 409
    f.store.close()
    f.store = SQLiteJobStore(f.root / "jobs.sqlite")
    assert worker(f).run_once() is True
    with TestClient(app(f)) as client:
        state = client.get(urls["status_url"], headers=f.headers).json()
        assert state["status"] == "succeeded" and state["attempts"] == 1
        result = client.get(urls["result_url"], headers=f.headers)
        assert result.status_code == 200
        body = result.json()
    assert body["schema_version"] == 3 and body["status"] == "completed"
    assert len(f.selector.requests) == 8 and len(f.model.requests) == 2
    assert body["scope"]["tests_executed_by_bot"] is False
    directory = f.root / "artifacts/fixed" / urls["review_id"] / "attempt-1"
    assert json.loads((directory / "manifest.json").read_text())["publication"]["reason"] == "artifact_only"
    assert json.loads((directory / "B/input.json").read_text()) == asdict_json(f.task)
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in directory.rglob("*") if p.is_file())
    assert worker(f).run_once() is False


def asdict_json(task):
    return json.loads(json.dumps(asdict(task)))


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}])
def test_api_authentication_precedes_input_and_result_access(api_flow, headers):
    f = api_flow
    with TestClient(app(f)) as client:
        assert client.post("/reviews/intent", json=asdict(f.task), headers=headers).status_code == 401
        assert client.get("/reviews/intent/run-" + "a" * 32, headers=headers).status_code == 401
        assert client.get("/reviews/intent/run-" + "a" * 32 + "/result", headers=headers).status_code == 401
    assert not f.store._get_connection().execute("SELECT run_id FROM runs").fetchall()


def test_api_rejects_invalid_oversized_inputs_and_bad_language(api_flow):
    f = api_flow
    with TestClient(app(f)) as client:
        assert client.post("/reviews/intent", json={}, headers=f.headers).status_code == 422
        bad = asdict(f.task)
        bad["input_hash"] = "a" * 64
        assert client.post("/reviews/intent", json=bad, headers=f.headers).status_code == 422
        assert submit(client, f, "invalid").status_code == 422
        assert client.get("/reviews/intent/run-" + "a" * 32, headers=f.headers).status_code == 404
    f.config.service.webhook_max_bytes = 10
    with TestClient(app(f)) as client:
        assert submit(client, f).status_code == 413
    assert not f.store._get_connection().execute("SELECT run_id FROM runs").fetchall()


def test_api_fails_closed_without_token(api_flow, monkeypatch):
    monkeypatch.delenv("REVIEW_BOT_API_TOKEN")
    with pytest.raises(ValueError, match="requires_token"):
        app(api_flow)


def test_fixed_queue_keeps_webhook_coalescing_and_generation_separate(api_flow):
    f = api_flow
    rid, number = f.task.snapshot.repository_id, f.task.snapshot.pr_number
    fixed = f.store.enqueue_fixed_intent(json.dumps(asdict(f.task)), rid, number, "en")
    f.store.enqueue_request(PrRequestRecord(77, rid, number, 1))
    f.store.enqueue_request(PrRequestRecord(77, rid, number, 1))
    first = f.store.claim_next_run("fixed", 30)
    assert first.run_id == fixed and first.kind == "fixed_intent" and first.generation == 1
    assert f.store.check_generation_fencing(first.run_id, worker_id="fixed", generation=1)
    second = f.store.claim_next_run("webhook", 30)
    assert second.kind == "github_pr" and second.generation == 2
    assert f.store.claim_next_run("third", 30) is None
    assert f.store.update_run(first.run_id, "succeeded", worker_id="fixed", generation=1)


def test_fixed_queue_retains_lease_fence_and_recovers_expired_attempt(api_flow):
    f = api_flow
    rid = f.store.enqueue_fixed_intent(json.dumps(asdict(f.task)), 99, 7, "ko")
    first = f.store.claim_next_run("old", 30)
    f.store._get_connection().execute("UPDATE runs SET lease_expires_at=0 WHERE run_id=?", (rid,))
    second = f.store.claim_next_run("new", 30)
    assert second.run_id == first.run_id and second.attempts == 2
    assert not f.store.update_run(rid, "succeeded", worker_id="old", generation=1)
    assert f.store.update_run(rid, "failed", worker_id="new", generation=1)


def test_partial_or_unconfigured_review_is_failed_and_remains_queryable(api_flow):
    f = api_flow
    with TestClient(app(f)) as client:
        urls = submit(client, f).json()
        f.model = None
        assert worker(f).run_once() is False
        assert client.get(urls["status_url"], headers=f.headers).json()["status"] == "failed"
        result = client.get(urls["result_url"], headers=f.headers).json()
        assert result["status"] != "completed"


def test_perspective_fingerprint_tracks_role_and_language_without_changing_legacy(api_flow):
    f = api_flow
    first = intent_input_fingerprint(f.task, f.config, f.model, selection_client=f.selector)
    assert first
    f.config.output.language = "en"
    assert intent_input_fingerprint(f.task, f.config, f.model, selection_client=f.selector) != first
    f.config.output.language = "ko"
    f.selector.api_url = "http://different-selector.invalid/v1"
    assert intent_input_fingerprint(f.task, f.config, f.model, selection_client=f.selector) != first
