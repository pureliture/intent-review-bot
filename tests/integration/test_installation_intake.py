"""Signed installation-based intake using the real durable SQLite store."""
from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import ConfigValidationError, ReviewBotConfig, load_review_config_from_str
from review_bot.domain.envelope_v2 import PrRequestRecord
from review_bot.runtime.webhook import create_webhook_app

SECRET = "synthetic-installation-secret"


def app_config(valid_config_yaml: str) -> ReviewBotConfig:
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["mode"] = "github_app"
    raw["publish"].pop("mode", None)
    return load_review_config_from_str(yaml.safe_dump(raw))[0]


def opened_payload() -> dict:
    return {
        "action": "opened", "number": 7,
        "installation": {"id": 44},
        "repository": {"id": 99, "name": "repo", "full_name": "org/repo"},
        "pull_request": {
            "number": 7, "state": "open",
            "head": {"sha": "b" * 40, "ref": "feature/test"},
            "base": {"sha": "a" * 40, "ref": "main"},
        },
    }


def deliver(client: TestClient, payload: dict, delivery_id: str = "first", event_name: str = "pull_request"):
    body = json.dumps(payload).encode()
    return client.post("/webhook", content=body, headers={
        "X-GitHub-Event": event_name,
        "X-GitHub-Delivery": delivery_id,
        "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
    })


def test_installation_without_allowlist_is_durable_and_deduplicated(
    tmp_path: Path, valid_config_yaml: str,
) -> None:
    store = SQLiteJobStore(tmp_path / "intake.db")
    try:
        with TestClient(create_webhook_app(app_config(valid_config_yaml), store, secret=SECRET)) as client:
            assert deliver(client, opened_payload()).json() == {"status": "accepted"}
            assert deliver(client, opened_payload()).json() == {"status": "deduplicated"}
        store.close()
        restored = SQLiteJobStore(tmp_path / "intake.db")
        try:
            request = restored.get_pr_request(99, 7)
            assert request is not None and request.installation_id == 44
            assert request.desired_generation == 1
            run = restored.claim_next_run("restored-worker", 30)
            assert run is not None and run.repository_id == 99 and run.installation_id == 44
            assert restored.claim_next_run("second-worker", 30) is None
        finally:
            restored.close()
    finally:
        store.close()


@pytest.mark.parametrize("restriction", ["none", "installation", "repository"])
def test_missing_installation_is_never_eligible(
    tmp_path: Path, valid_config_yaml: str, restriction: str,
) -> None:
    config = app_config(valid_config_yaml)
    if restriction == "installation":
        config.service.installation_allowlist = [44]
    if restriction == "repository":
        config.service.repository_allowlist = ["org/repo"]
    store = SQLiteJobStore(tmp_path / "missing.db")
    try:
        payload = opened_payload()
        payload.pop("installation")
        with TestClient(create_webhook_app(config, store, secret=SECRET)) as client:
            response = deliver(client, payload)
            assert response.status_code == 202
            assert response.json() == {"status": "ignored", "reason": "unlisted"}
        assert store.is_delivery_processed("first")
        assert store.get_pr_request(99, 7) is None
        assert store.claim_next_run("worker", 30) is None
    finally:
        store.close()


@pytest.mark.parametrize("restriction", ["installation", "repository"])
def test_each_optional_restriction_is_applied_independently(
    tmp_path: Path, valid_config_yaml: str, restriction: str,
) -> None:
    config = app_config(valid_config_yaml)
    if restriction == "installation":
        config.service.installation_allowlist = [45]
    else:
        config.service.repository_allowlist = ["org/another"]
    store = SQLiteJobStore(tmp_path / "unlisted.db")
    try:
        with TestClient(create_webhook_app(config, store, secret=SECRET)) as client:
            assert deliver(client, opened_payload()).json()["reason"] == "unlisted"
        assert store.get_pr_request(99, 7) is None
    finally:
        store.close()


@pytest.mark.parametrize("failure", [sqlite3.OperationalError, sqlite3.IntegrityError])
def test_partial_intake_failure_rolls_back_and_same_delivery_can_retry(
    tmp_path: Path, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch, failure: type[Exception],
) -> None:
    store = SQLiteJobStore(tmp_path / "retry.db")
    original_enqueue = store.enqueue_request

    def fail_after_enqueue(request):
        original_enqueue(request)
        raise failure("synthetic storage failure")

    try:
        monkeypatch.setattr(store, "enqueue_request", fail_after_enqueue)
        with TestClient(create_webhook_app(app_config(valid_config_yaml), store, secret=SECRET)) as client:
            assert deliver(client, opened_payload()).status_code == 503
            assert not store.is_delivery_processed("first")
            assert store.get_pr_request(99, 7) is None
            assert store.claim_next_run("worker", 30) is None
            monkeypatch.setattr(store, "enqueue_request", original_enqueue)
            assert deliver(client, opened_payload()).json() == {"status": "accepted"}
        assert store.get_pr_request(99, 7).desired_generation == 1
    finally:
        store.close()


def test_bad_signature_never_records_delivery(tmp_path: Path, valid_config_yaml: str) -> None:
    store = SQLiteJobStore(tmp_path / "unsigned.db")
    try:
        with TestClient(create_webhook_app(app_config(valid_config_yaml), store, secret=SECRET)) as client:
            response = client.post("/webhook", json=opened_payload(), headers={
                "X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "unsigned",
                "X-Hub-Signature-256": "sha256=bad",
            })
        assert response.status_code == 401
        assert not store.is_delivery_processed("unsigned")
        assert store.claim_next_run("worker", 30) is None
    finally:
        store.close()


def test_active_receiver_fails_closed_without_secret(
    valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = app_config(valid_config_yaml)
    monkeypatch.delenv(config.service.webhook_secret_env, raising=False)
    store = SQLiteJobStore()
    try:
        with pytest.raises(ConfigValidationError, match="webhook secret"):
            create_webhook_app(config, store)
    finally:
        store.close()


@pytest.mark.parametrize("event_name,action", [
    ("installation", "deleted"), ("installation", "suspend"), ("installation_repositories", "removed"),
])
def test_signed_revocation_stops_only_matching_installation_repositories(
    tmp_path: Path, valid_config_yaml: str, event_name: str, action: str,
) -> None:
    store = SQLiteJobStore(tmp_path / "revoked.db")
    try:
        for installation, repository, number in ((44, 99, 7), (44, 100, 8), (45, 101, 9)):
            store.enqueue_request(PrRequestRecord(installation, repository, number, 1))
        payload = {"action": action, "installation": {"id": 44}, "repositories_removed": [{"id": 99}]}
        with TestClient(create_webhook_app(app_config(valid_config_yaml), store, secret=SECRET)) as client:
            assert deliver(client, payload, "revoked", event_name).json() == {"status": "cancelled", "reason": "installation_revoked"}
            assert deliver(client, payload, "revoked", event_name).json() == {"status": "deduplicated"}
        statuses = dict(store._get_connection().execute("SELECT repository_id, status FROM runs;").fetchall())
        assert statuses[99] == "cancelled"
        assert statuses[100] == ("queued" if action == "removed" else "cancelled")
        assert statuses[101] == "queued"
        assert store.get_pr_request(99, 7).desired_generation == 2
    finally:
        store.close()


def test_cancellation_failure_rolls_back_delivery_and_queued_work(
    tmp_path: Path, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteJobStore(tmp_path / "cancel-retry.db")
    store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
    cancel = store._cancel_request

    def failing_cancel(*args, **kwargs):
        cancel(*args, **kwargs)
        raise sqlite3.OperationalError("synthetic cancellation failure")

    try:
        monkeypatch.setattr(store, "_cancel_request", failing_cancel)
        payload = opened_payload()
        payload["action"] = "closed"
        payload["pull_request"]["state"] = "closed"
        with TestClient(create_webhook_app(app_config(valid_config_yaml), store, secret=SECRET)) as client:
            assert deliver(client, payload, "close").status_code == 503
            assert not store.is_delivery_processed("close")
            assert store.get_pr_request(99, 7).desired_generation == 1
            assert store._get_connection().execute("SELECT status FROM runs;").fetchone()[0] == "queued"
            monkeypatch.setattr(store, "_cancel_request", cancel)
            assert deliver(client, payload, "close").json()["status"] == "cancelled"
    finally:
        store.close()