"""Mode receipts retain partial success without changing comment-body fingerprints."""
from __future__ import annotations

import sqlite3
from dataclasses import replace

import pytest
import yaml

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import ConfigValidationError, PublishConfig, load_review_config_from_str
from review_bot.domain.envelope_v2 import PrRequestRecord, PublicationReceipt
from review_bot.domain.status import RunStatus

A_MARKER = "<!-- review-bot:architecture:v1 -->"
B_MARKER = "<!-- review-bot:intent:v1 -->"


def test_b_publication_defaults_do_not_opt_in_or_change_a_hash(valid_config_yaml):
    config, digest = load_review_config_from_str(valid_config_yaml)
    assert config.publish.b_marker == B_MARKER
    assert config.publish.b_comments_enabled is False
    assert digest == "9e26359acd9f85ec286c422413611353ccb6d052fb6c6c652cd68d6d60252656"


@pytest.mark.parametrize("marker", [A_MARKER, "", "<!-- unsafe -->\n<script>", "<!-- a --> <!-- b -->", "<!-- @all -->"])
def test_b_marker_cannot_collide_or_inject_html(valid_config_yaml, marker):
    raw = yaml.safe_load(valid_config_yaml)
    raw["publish"]["b_marker"] = marker
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(yaml.safe_dump(raw))


def test_custom_b_marker_is_operator_owned_and_distinct():
    marker = "<!-- review-bot:intent:team-v1 -->"
    assert PublishConfig(b_marker=marker, b_comments_enabled=True).b_marker == marker
    with pytest.raises(ValueError):
        PublishConfig(a_marker=marker, b_marker=marker)
    with pytest.raises(ValueError):
        PublishConfig(b_comments_enabled="true")


@pytest.mark.parametrize("prior_status_columns", [False, True])
def test_receipt_migration_preserves_legacy_body_hash_and_never_infers_b_success(tmp_path, prior_status_columns):
    path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE publication_receipts (
                repository_id INTEGER NOT NULL, pr_number INTEGER NOT NULL,
                marker TEXT NOT NULL, comment_id INTEGER,
                published_fingerprint TEXT NOT NULL, published_at TEXT NOT NULL,
                PRIMARY KEY (repository_id, pr_number, marker)
            );
        """)
        connection.execute("INSERT INTO publication_receipts VALUES (99, 7, ?, 701, 'old-body-hash', 'old-time')", (B_MARKER,))
        if prior_status_columns:
            connection.execute("ALTER TABLE publication_receipts ADD COLUMN status TEXT NOT NULL DEFAULT 'unknown'")
            connection.execute("ALTER TABLE publication_receipts ADD COLUMN run_id TEXT")
            connection.execute("UPDATE publication_receipts SET status = 'succeeded', run_id = 'old-run'")
    for _ in range(2):
        store = SQLiteJobStore(path)
        try:
            row = store._get_connection().execute("SELECT comment_id, published_fingerprint, published_at, status, run_id, input_fingerprint FROM publication_receipts").fetchone()
            expected_status, expected_run = ("succeeded", "old-run") if prior_status_columns else ("unknown", None)
            assert row == (701, "old-body-hash", "old-time", expected_status, expected_run, None)
            store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
            run = store.claim_next_run("worker", 60)
            assert store.get_current_publication(run, B_MARKER, input_fingerprint="b" * 64) is None
            assert store.update_run(run.run_id, RunStatus.SUCCEEDED, worker_id="worker", generation=run.generation)
        finally:
            store.close()


@pytest.mark.parametrize("terminal", [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.SUPERSEDED])
def test_mode_cache_uses_individual_success_and_installation_boundary(tmp_path, terminal):
    store = SQLiteJobStore(tmp_path / "modes.sqlite")
    try:
        request = PrRequestRecord(44, 99, 7, 1)
        store.enqueue_request(request)
        first = store.claim_next_run("worker", 60)
        a = PublicationReceipt(701, "succeeded", "a-body-hash", marker=A_MARKER, input_fingerprint="a" * 64)
        b = PublicationReceipt(702, "failed", "b-body-hash", marker=B_MARKER, input_fingerprint="b" * 64)
        for receipt in (a, b):
            assert store.record_publication(receipt, 99, 7, run_id=first.run_id, worker_id="worker", generation=first.generation)
        assert store.update_run(first.run_id, terminal, worker_id="worker", generation=first.generation)
        store.enqueue_request(request)
        second = store.claim_next_run("worker", 60)
        second = replace(second, fingerprint="different-whole-run-fingerprint")
        cached = store.get_current_publication(second, A_MARKER, input_fingerprint="a" * 64)
        if terminal in (RunStatus.SUCCEEDED, RunStatus.FAILED):
            assert cached.comment_id == 701 and cached.fingerprint == "a-body-hash"
            assert cached.input_fingerprint == "a" * 64
        else:
            assert cached is None
        assert store.get_current_publication(second, B_MARKER, input_fingerprint="b" * 64) is None
        assert store.get_current_publication(second, A_MARKER, input_fingerprint="c" * 64) is None
        assert store.get_current_publication(replace(second, installation_id=45), A_MARKER, input_fingerprint="a" * 64) is None
        assert store.get_current_publication(second, A_MARKER) is None
    finally:
        store.close()