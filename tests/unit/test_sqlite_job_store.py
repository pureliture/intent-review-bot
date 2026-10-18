"""Unit tests for SQLite WAL JobStore implementation."""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.domain.envelope_v2 import (
    DeliveryRecord,
    PrRequestRecord,
    PublicationReceipt,
    RunRecord,
)
from review_bot.domain.status import RunStage, RunStatus
from review_bot.ports import JobStore


def test_sqlite_job_store_implements_protocol() -> None:
    store = SQLiteJobStore(":memory:")
    assert isinstance(store, JobStore)
    store.close()


def test_sqlite_job_store_pragmas(tmp_path: Path) -> None:
    db_file = tmp_path / "test_pragmas.db"
    store = SQLiteJobStore(db_file)
    conn = store._get_connection()

    mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert mode.lower() == "wal"

    busy_timeout = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
    assert busy_timeout == 5000

    fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    assert fk == 1

    sync = conn.execute("PRAGMA synchronous;").fetchone()[0]
    # synchronous = NORMAL is 1 in SQLite
    assert sync == 1
    store.close()


def test_delivery_deduplication(tmp_path: Path) -> None:
    db_file = tmp_path / "test_dedup.db"
    store = SQLiteJobStore(db_file)

    delivery = DeliveryRecord(
        delivery_id="deliv-unique-1",
        payload_digest="sha256-abc",
        received_at="2026-10-07T00:00:00Z",
        action="opened",
        installation_id=10,
        repository_id=20,
        pr_number=30,
        processed=False,
    )

    assert store.is_delivery_processed("deliv-unique-1") is False
    store.record_delivery(delivery)
    assert store.is_delivery_processed("deliv-unique-1") is True

    # Duplicate delivery must raise sqlite3.IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        store.record_delivery(delivery)
    store.close()


def test_enqueue_request_coalesces_generation(tmp_path: Path) -> None:
    db_file = tmp_path / "test_coalesce.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=100,
        pr_number=42,
        desired_generation=1,
    )

    # 1. First event: PR opened
    store.enqueue_request(req)
    pr_req = store.get_pr_request(100, 42)
    assert pr_req is not None
    assert pr_req.desired_generation == 1

    # 2. Second event: PR synchronize (push)
    store.enqueue_request(req)
    pr_req2 = store.get_pr_request(100, 42)
    assert pr_req2 is not None
    assert pr_req2.desired_generation == 2

    # 3. Third event: PR synchronize (push)
    store.enqueue_request(req)
    pr_req3 = store.get_pr_request(100, 42)
    assert pr_req3 is not None
    assert pr_req3.desired_generation == 3

    # The pending run should have coalesced directly to generation 3!
    run = store.claim_next_run("worker-1", lease_seconds=10.0)
    assert run is not None
    assert run.generation == 3
    assert run.repository_id == 100
    assert run.pr_number == 42
    assert run.status == "running"
    store.close()


def test_worker_lease_claiming_and_renewal(tmp_path: Path) -> None:
    db_file = tmp_path / "test_lease.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=200,
        pr_number=15,
        desired_generation=1,
    )
    store.enqueue_request(req)

    # Worker 1 claims with 0.2s lease
    run1 = store.claim_next_run("worker-1", lease_seconds=0.2)
    assert run1 is not None
    assert run1.status == "running"

    # Worker 2 attempts to claim while lease active -> None
    run2 = store.claim_next_run("worker-2", lease_seconds=1.0)
    assert run2 is None

    # Worker 1 renews lease
    renewed = store.renew_lease(run1.run_id, "worker-1", lease_seconds=0.3)
    assert renewed is True

    # Unknown worker cannot renew
    renewed_fake = store.renew_lease(run1.run_id, "fake-worker", lease_seconds=1.0)
    assert renewed_fake is False

    # Wait for lease to expire
    time.sleep(0.35)

    # Worker 2 reclaims expired lease
    run3 = store.claim_next_run("worker-2", lease_seconds=10.0)
    assert run3 is not None
    assert run3.run_id == run1.run_id
    assert run3.attempts == 2

    store.close()


def test_generation_fencing_marks_superseded(tmp_path: Path) -> None:
    db_file = tmp_path / "test_fencing.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=300,
        pr_number=99,
        desired_generation=1,
    )
    store.enqueue_request(req)

    # Worker claims generation 1
    run = store.claim_next_run("worker-a", lease_seconds=60.0)
    assert run is not None
    assert run.generation == 1

    # Check fencing initially -> True
    assert store.check_generation_fencing(run.run_id) is True

    # A newer commit arrives for the same PR (advancing desired_generation to 2)
    store.enqueue_request(req)
    pr_req = store.get_pr_request(300, 99)
    assert pr_req is not None
    assert pr_req.desired_generation == 2

    # Worker checks fencing again -> False because run.generation (1) < desired_generation (2)
    assert store.check_generation_fencing(run.run_id) is False

    # Run should now be marked superseded in database
    conn = store._get_connection()
    status = conn.execute(
        "SELECT status FROM runs WHERE run_id = ?;", (run.run_id,)
    ).fetchone()[0]
    assert status == "superseded"

    store.close()


def test_update_run_lifecycle_and_completion(tmp_path: Path) -> None:
    db_file = tmp_path / "test_lifecycle.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=400,
        pr_number=5,
        desired_generation=1,
    )
    store.enqueue_request(req)

    run = store.claim_next_run("worker-1", lease_seconds=60.0)
    assert run is not None

    store.update_run(run.run_id, status=RunStatus.RUNNING, stage=RunStage.ANALYZE)
    conn = store._get_connection()
    row = conn.execute(
        "SELECT stage, status FROM runs WHERE run_id = ?;", (run.run_id,)
    ).fetchone()
    assert row[0] == "analyze"
    assert row[1] == "running"

    # Complete run
    store.complete_run(run.run_id)
    row = conn.execute(
        "SELECT status, completed_at, lease_expires_at FROM runs WHERE run_id = ?;",
        (run.run_id,),
    ).fetchone()
    assert row[0] == "succeeded"
    assert row[1] is not None
    assert row[2] is None  # Lease cleared on completion

    store.close()


def test_record_publication_receipt(tmp_path: Path) -> None:
    db_file = tmp_path / "test_pub.db"
    store = SQLiteJobStore(db_file)

    receipt = PublicationReceipt(
        comment_id=12345,
        status="succeeded",
        fingerprint="fp-xyz",
        marker="<!-- review-bot:architecture:v1 -->",
        published_at="2026-10-07T01:00:00Z",
    )
    store.record_publication(receipt, repository_id=500, pr_number=10)

    conn = store._get_connection()
    row = conn.execute(
        "SELECT comment_id, published_fingerprint, marker FROM publication_receipts WHERE repository_id = 500 AND pr_number = 10;",
    ).fetchone()
    assert row[0] == 12345
    assert row[1] == "fp-xyz"
    assert row[2] == "<!-- review-bot:architecture:v1 -->"

    # Update publication receipt on new run
    updated_receipt = PublicationReceipt(
        comment_id=12345,
        status="succeeded",
        fingerprint="fp-new-hash",
        marker="<!-- review-bot:architecture:v1 -->",
        published_at="2026-10-07T02:00:00Z",
    )
    store.record_publication(updated_receipt, repository_id=500, pr_number=10)
    row = conn.execute(
        "SELECT comment_id, published_fingerprint FROM publication_receipts WHERE repository_id = 500 AND pr_number = 10;",
    ).fetchone()
    assert row[0] == 12345
    assert row[1] == "fp-new-hash"

    store.close()


def test_concurrent_worker_claiming_thread_safety(tmp_path: Path) -> None:
    db_file = tmp_path / "test_concurrent.db"
    store = SQLiteJobStore(db_file)

    # Enqueue 10 separate PR requests
    for i in range(10):
        req = PrRequestRecord(
            installation_id=1,
            repository_id=600,
            pr_number=i + 1,
            desired_generation=1,
        )
        store.enqueue_request(req)

    claimed_run_ids: list[str] = []
    lock = threading.Lock()

    def worker_thread(worker_num: int) -> None:
        thread_store = SQLiteJobStore(db_file)
        for _ in range(5):
            run = thread_store.claim_next_run(f"worker-{worker_num}", lease_seconds=10.0)
            if run:
                with lock:
                    claimed_run_ids.append(run.run_id)
                assert thread_store.update_run(
                    run.run_id, RunStatus.SUCCEEDED, worker_id=f"worker-{worker_num}", generation=run.generation,
                )
            time.sleep(0.01)
        thread_store.close()

    threads = [threading.Thread(target=worker_thread, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All claimed runs must be unique — no double claims!
    assert len(claimed_run_ids) == 10
    assert len(set(claimed_run_ids)) == 10
    store.close()


def test_existing_database_migrates_skip_reason_without_losing_runs(tmp_path: Path) -> None:
    db_file = tmp_path / "legacy.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                installation_id INTEGER NOT NULL DEFAULT 0,
                repository_id INTEGER NOT NULL,
                repo_full_name TEXT NOT NULL DEFAULT '',
                pr_number INTEGER NOT NULL,
                generation INTEGER NOT NULL DEFAULT 1,
                fingerprint TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT 'fetch',
                worker_id TEXT,
                lease_expires_at REAL,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error TEXT
            );
        """)
        conn.execute("""
            INSERT INTO runs (run_id, repository_id, pr_number, status, created_at, error)
            VALUES ('legacy-run', 100, 42, 'queued', '2026-10-08', 'previous diagnostic');
        """)
    store = SQLiteJobStore(db_file)
    try:
        run = store.claim_next_run("worker-a", lease_seconds=30.0)
        assert run is not None and run.run_id == "legacy-run"
        assert run.reason is None
        assert run.error == "previous diagnostic"
        assert store.update_run(
            run.run_id, "skipped", RunStage.SELECT, reason="not_applicable",
            worker_id="worker-a", generation=run.generation,
        ) is True
        assert store._get_connection().execute(
            "SELECT status, reason, error, lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,),
        ).fetchone() == ("skipped", "not_applicable", "previous diagnostic", None)
        assert store.renew_lease(run.run_id, "worker-a", 30.0) is False
        assert store.update_run(run.run_id, RunStatus.RUNNING, worker_id="worker-a", generation=run.generation) is False
    finally:
        store.close()
    reopened = SQLiteJobStore(db_file)
    try:
        assert reopened._get_connection().execute("SELECT reason FROM runs;").fetchone() == ("not_applicable",)
    finally:
        reopened.close()


@pytest.fixture
def claimed_store(tmp_path: Path) -> Iterator[tuple[SQLiteJobStore, RunRecord, PrRequestRecord]]:
    store = SQLiteJobStore(tmp_path / "cas.db")
    request = PrRequestRecord(1, 100, 42, 1)
    store.enqueue_request(request)
    run = store.claim_next_run("worker-a", lease_seconds=60.0)
    assert run is not None
    try:
        yield store, run, request
    finally:
        store.close()


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.SUCCEEDED])
@pytest.mark.parametrize(
    "invalid_fence",
    ["transferred", "expired", "no_lease", "unclaimed", "generation", "new_request", "terminal"],
)
def test_worker_update_rejects_invalid_fence_without_changing_row(
    claimed_store: tuple[SQLiteJobStore, RunRecord, PrRequestRecord],
    status: RunStatus,
    invalid_fence: str,
) -> None:
    store, run, request = claimed_store
    conn = store._get_connection()
    generation = run.generation
    if invalid_fence == "transferred":
        conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
        successor = store.claim_next_run("worker-b", lease_seconds=60.0)
        assert successor is not None and successor.run_id == run.run_id
    elif invalid_fence == "expired":
        conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
    elif invalid_fence == "no_lease":
        conn.execute("UPDATE runs SET lease_expires_at = NULL WHERE run_id = ?;", (run.run_id,))
    elif invalid_fence == "unclaimed":
        conn.execute(
            "UPDATE runs SET status = 'queued', worker_id = NULL, lease_expires_at = NULL WHERE run_id = ?;",
            (run.run_id,),
        )
    elif invalid_fence == "generation":
        generation += 1
    elif invalid_fence == "new_request":
        store.enqueue_request(request)
    else:
        # Even an erroneously retained lease cannot authorize reopening a terminal run.
        conn.execute("UPDATE runs SET status = 'succeeded' WHERE run_id = ?;", (run.run_id,))
    before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()

    assert store.update_run(
        run.run_id, status=status, stage=RunStage.PUBLISH, error="stale write",
        worker_id="worker-a", generation=generation,
    ) is False

    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before


def test_worker_update_accepts_current_owner_and_completes_once(
    claimed_store: tuple[SQLiteJobStore, RunRecord, PrRequestRecord],
) -> None:
    store, run, _ = claimed_store
    assert store.update_run(
        run.run_id, RunStatus.RUNNING, RunStage.ANALYZE,
        worker_id="worker-a", generation=run.generation,
    ) is True
    assert store.update_run(
        run.run_id, RunStatus.SUCCEEDED, RunStage.PUBLISH,
        worker_id="worker-a", generation=run.generation,
    ) is True
    conn = store._get_connection()
    before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
    status, stage, completed_at, lease = conn.execute(
        "SELECT status, stage, completed_at, lease_expires_at FROM runs WHERE run_id = ?;",
        (run.run_id,),
    ).fetchone()
    assert (status, stage) == ("succeeded", "publish")
    assert completed_at is not None and lease is None
    assert store.update_run(
        run.run_id, RunStatus.FAILED, error="late exception",
        worker_id="worker-a", generation=run.generation,
    ) is False
    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before


def test_generation_fence_cannot_supersede_transferred_or_terminal_run(
    claimed_store: tuple[SQLiteJobStore, RunRecord, PrRequestRecord],
) -> None:
    store, run, request = claimed_store
    conn = store._get_connection()
    conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
    successor = store.claim_next_run("worker-b", lease_seconds=60.0)
    assert successor is not None and successor.run_id == run.run_id
    store.enqueue_request(request)
    before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
    assert store.check_generation_fencing(
        run.run_id, worker_id="worker-a", generation=run.generation,
    ) is False
    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before

    store.complete_run(run.run_id)
    before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
    assert store.check_generation_fencing(run.run_id) is False
    assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before


@pytest.mark.parametrize("operation", ["update", "renew", "claim"])
def test_lease_time_is_checked_after_acquiring_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "review_bot.adapters.sqlite.store.time", SimpleNamespace(time=lambda: clock.now),
    )
    store = SQLiteJobStore(tmp_path / "lock_wait.db")
    try:
        store.enqueue_request(PrRequestRecord(1, 100, 42, 1))
        run = store.claim_next_run("worker-a", lease_seconds=1.0)
        assert run is not None
        transaction = store.transaction

        @contextmanager
        def delayed_transaction(immediate: bool = True) -> Iterator[sqlite3.Connection]:
            with transaction(immediate) as conn:
                # Model time spent waiting for SQLite's RESERVED lock, not execution time.
                clock.now = 102.0
                yield conn

        monkeypatch.setattr(store, "transaction", delayed_transaction)
        if operation == "update":
            assert store.update_run(
                run.run_id, RunStatus.SUCCEEDED, worker_id="worker-a", generation=run.generation,
            ) is False
        elif operation == "renew":
            assert store.renew_lease(run.run_id, "worker-a", 60.0) is False
        else:
            successor = store.claim_next_run("worker-b", lease_seconds=60.0)
            assert successor is not None and successor.run_id == run.run_id
            assert store._get_connection().execute(
                "SELECT lease_expires_at FROM runs WHERE run_id = ?;", (run.run_id,),
            ).fetchone()[0] == 162.0
    finally:
        store.close()


def test_generation_fence_rechecks_lease_before_superseding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        "review_bot.adapters.sqlite.store.time", SimpleNamespace(time=lambda: clock.now),
    )
    store = SQLiteJobStore(tmp_path / "generation_expiry.db")
    try:
        request = PrRequestRecord(1, 100, 42, 1)
        store.enqueue_request(request)
        run = store.claim_next_run("worker-a", lease_seconds=1.0)
        assert run is not None
        store.enqueue_request(request)
        conn = store._get_connection()
        before = conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()

        def expire_during_read(sql: str) -> None:
            if sql.lstrip().startswith("SELECT generation, COALESCE"):
                clock.now = 102.0

        conn.set_trace_callback(expire_during_read)
        assert store.check_generation_fencing(
            run.run_id, worker_id="worker-a", generation=run.generation,
        ) is False
        assert conn.execute("SELECT * FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone() == before
    finally:
        store.close()


@pytest.mark.parametrize("invalid_fence", ["transferred", "expired", "new_request", "wrong_pr"])
def test_publication_receipt_upsert_rejects_stale_worker(
    claimed_store: tuple[SQLiteJobStore, RunRecord, PrRequestRecord], invalid_fence: str,
) -> None:
    store, run, request = claimed_store
    current = PublicationReceipt(120, "succeeded", "successor-fingerprint")
    assert store.record_publication(
        current, run.repository_id, run.pr_number,
        run_id=run.run_id, worker_id="worker-a", generation=run.generation,
    ) is True
    conn = store._get_connection()
    if invalid_fence in ("transferred", "expired"):
        conn.execute("UPDATE runs SET lease_expires_at = 0 WHERE run_id = ?;", (run.run_id,))
        if invalid_fence == "transferred":
            assert store.claim_next_run("worker-b", lease_seconds=60.0) is not None
    elif invalid_fence == "new_request":
        store.enqueue_request(request)
    before = conn.execute("SELECT * FROM publication_receipts;").fetchall()
    pr_number = run.pr_number + 1 if invalid_fence == "wrong_pr" else run.pr_number

    assert store.record_publication(
        PublicationReceipt(1, "succeeded", "stale-fingerprint"), run.repository_id, pr_number,
        run_id=run.run_id, worker_id="worker-a", generation=run.generation,
    ) is False

    assert conn.execute("SELECT * FROM publication_receipts;").fetchall() == before


def test_concurrent_stale_worker_writes_cannot_clobber_successor(tmp_path: Path) -> None:
    db_file = tmp_path / "cas_stress.db"
    store = SQLiteJobStore(db_file)
    try:
        store.enqueue_request(PrRequestRecord(1, 100, 42, 1))
        stale = store.claim_next_run("worker-a", lease_seconds=-1.0)
        assert stale is not None
        successor = store.claim_next_run("worker-b", lease_seconds=60.0)
        assert successor is not None and successor.run_id == stale.run_id
        barrier = threading.Barrier(5)

        def writer(worker_id: str) -> None:
            thread_store = SQLiteJobStore(db_file)
            try:
                barrier.wait(timeout=5.0)
                for _ in range(25):
                    applied = thread_store.update_run(
                        stale.run_id, RunStatus.RUNNING, RunStage.CONTEXT, error=worker_id,
                        worker_id=worker_id, generation=stale.generation,
                    )
                    assert applied is (worker_id == "worker-b")
                    recorded = thread_store.record_publication(
                        PublicationReceipt(120 if worker_id == "worker-b" else 1, "succeeded", worker_id),
                        stale.repository_id, stale.pr_number,
                        run_id=stale.run_id, worker_id=worker_id, generation=stale.generation,
                    )
                    assert recorded is (worker_id == "worker-b")
            finally:
                thread_store.close()

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(writer, "worker-a") for _ in range(4)]
            futures.append(executor.submit(writer, "worker-b"))
            for future in futures:
                future.result(timeout=10.0)
        conn = store._get_connection()
        assert conn.execute(
            "SELECT status, stage, worker_id, error FROM runs WHERE run_id = ?;", (stale.run_id,),
        ).fetchone() == ("running", "context", "worker-b", "worker-b")
        assert conn.execute(
            "SELECT comment_id, published_fingerprint FROM publication_receipts;",
        ).fetchone() == (120, "worker-b")
    finally:
        store.close()
