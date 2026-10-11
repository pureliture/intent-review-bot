"""Empirical adversarial stress test suite for Milestone 5 SQLite WAL JobStore.

Covers:
1. Concurrent worker claims under race conditions (multi-thread & multi-process).
2. Expired lease reclamation & stale worker renewal rejection.
3. Delivery deduplication under high concurrent pressure.
4. Rapid PR updates and generation coalescing (sequential and concurrent).
5. Generation fencing detection, automatic supersession, and worker abort.
6. Boundary conditions: zero/negative leases, transaction rollbacks, SQL character safety.
"""
from __future__ import annotations

import multiprocessing
import sqlite3
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.domain.envelope_v2 import (
    DeliveryRecord,
    PrRequestRecord,
    PublicationReceipt,
    RunRecord,
)
from review_bot.domain.status import RunStatus
from review_bot.runtime.worker import WorkerLoop


# Helper for multiprocessing test
def _worker_process_claim(db_path_str: str, worker_id: str, max_claims: int) -> list[str]:
    store = SQLiteJobStore(Path(db_path_str), timeout=10.0)
    claimed: list[str] = []
    for _ in range(max_claims):
        run = store.claim_next_run(worker_id, lease_seconds=30.0)
        if run:
            claimed.append(run.run_id)
        time.sleep(0.01)
    store.close()
    return claimed


# ==============================================================================
# 1. Concurrent Worker Claims (Race Condition Avoidance)
# ==============================================================================


def test_concurrent_threads_claim_single_pending_run(tmp_path: Path) -> None:
    """Stress-test 40 threads competing simultaneously for the exact same single pending run.

    Only 1 thread must claim it; 39 must receive None. Zero double-claims or locked errors.
    """
    db_file = tmp_path / "test_single_claim.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=101,
        pr_number=1,
        desired_generation=1,
    )
    store.enqueue_request(req)

    thread_count = 40
    barrier = threading.Barrier(thread_count)
    results: list[RunRecord | None] = []
    lock = threading.Lock()
    exceptions: list[Exception] = []

    def claim_worker(worker_idx: int) -> None:
        t_store = SQLiteJobStore(db_file, timeout=10.0)
        try:
            barrier.wait()  # Align all 40 threads to strike at the exact microsecond
            run = t_store.claim_next_run(f"worker-{worker_idx}", lease_seconds=10.0)
            with lock:
                results.append(run)
        except Exception as e:  # noqa: BLE001
            with lock:
                exceptions.append(e)
        finally:
            t_store.close()

    threads = [threading.Thread(target=claim_worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not exceptions, f"Unexpected exceptions during concurrent claiming: {exceptions}"
    successful_claims = [r for r in results if r is not None]
    assert len(successful_claims) == 1, f"Expected exactly 1 claim, got {len(successful_claims)}"

    # Verify database state
    conn = store._get_connection()
    row = conn.execute("SELECT status, attempts, worker_id FROM runs;").fetchone()
    assert row[0] == "running"
    assert row[1] == 1
    assert row[2] is not None and row[2].startswith("worker-")
    store.close()


def test_multiprocess_concurrent_worker_claims(tmp_path: Path) -> None:
    """Stress-test concurrent worker claims across distinct OS processes on SQLite WAL."""
    db_file = tmp_path / "test_multiprocess.db"
    store = SQLiteJobStore(db_file, timeout=10.0)

    total_jobs = 20
    for i in range(total_jobs):
        req = PrRequestRecord(
            installation_id=1,
            repository_id=200,
            pr_number=i + 1,
            desired_generation=1,
        )
        store.enqueue_request(req)
    store.close()

    process_count = 4
    claims_per_process = total_jobs
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=process_count, mp_context=ctx) as executor:
        futures = [
            executor.submit(_worker_process_claim, str(db_file), f"proc-{p}", claims_per_process)
            for p in range(process_count)
        ]
        all_claimed: list[str] = []
        for f in futures:
            all_claimed.extend(f.result())

    # All jobs must be claimed exactly once across all OS processes
    assert len(all_claimed) == total_jobs, f"Expected {total_jobs} claims, got {len(all_claimed)}"
    assert len(set(all_claimed)) == total_jobs, "Duplicate claim detected across processes!"

    # Verify all jobs in DB are marked running with attempts == 1
    verify_store = SQLiteJobStore(db_file)
    conn = verify_store._get_connection()
    rows = conn.execute("SELECT status, attempts FROM runs;").fetchall()
    assert len(rows) == total_jobs
    for status, attempts in rows:
        assert status == "running"
        assert attempts == 1
    verify_store.close()


def test_concurrent_claim_with_active_and_pending_jobs(tmp_path: Path) -> None:
    """Ensure active leases are protected when threads compete for available pending runs."""
    db_file = tmp_path / "test_mixed_runs.db"
    store = SQLiteJobStore(db_file)

    # 1. Create 5 pending runs
    for i in range(1, 6):
        store.enqueue_request(
            PrRequestRecord(installation_id=1, repository_id=300, pr_number=i, desired_generation=1)
        )

    # 2. Claim first 2 with 60s leases (active running)
    claimed_1 = store.claim_next_run("active-worker-1", lease_seconds=60.0)
    claimed_2 = store.claim_next_run("active-worker-2", lease_seconds=60.0)
    assert claimed_1 is not None and claimed_2 is not None
    active_ids = {claimed_1.run_id, claimed_2.run_id}

    # 3. 20 threads compete for remaining runs
    claimed_by_threads: list[str] = []
    lock = threading.Lock()

    def worker_claim(idx: int) -> None:
        t_store = SQLiteJobStore(db_file, timeout=10.0)
        run = t_store.claim_next_run(f"pool-{idx}", lease_seconds=10.0)
        if run:
            with lock:
                claimed_by_threads.append(run.run_id)
        t_store.close()

    threads = [threading.Thread(target=worker_claim, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly 3 remaining runs must be claimed, and none of active_ids may be re-claimed
    assert len(claimed_by_threads) == 3
    assert len(set(claimed_by_threads)) == 3
    for cid in claimed_by_threads:
        assert cid not in active_ids, f"Active run {cid} was unlawfully claimed!"
    store.close()


# ==============================================================================
# 2. Expired Lease Reclamation & Stale Renewal Rejection
# ==============================================================================


def test_expired_lease_reclamation_and_stale_renewal_rejection(tmp_path: Path) -> None:
    """Verify crashed worker's expired run is reclaimed, and stale worker's renewal is rejected."""
    db_file = tmp_path / "test_reclaim.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=400,
        pr_number=10,
        desired_generation=1,
    )
    store.enqueue_request(req)

    # Worker A claims run with short lease of 0.15s
    worker_a = "worker-alpha"
    run_a = store.claim_next_run(worker_a, lease_seconds=0.15)
    assert run_a is not None
    assert run_a.attempts == 1

    # Worker B tries to claim immediately before lease expires -> must return None
    worker_b = "worker-beta"
    assert store.claim_next_run(worker_b, lease_seconds=5.0) is None

    # Wait for lease to expire
    time.sleep(0.20)

    # Worker B claims expired run -> succeeds, attempts incremented to 2
    run_b = store.claim_next_run(worker_b, lease_seconds=10.0)
    assert run_b is not None
    assert run_b.run_id == run_a.run_id
    assert run_b.attempts == 2

    # Stale Worker A wakes up and attempts renewal -> must be rejected
    renew_a = store.renew_lease(run_a.run_id, worker_a, lease_seconds=10.0)
    assert renew_a is False, "Stale worker renewal should have been rejected!"

    # Current owner Worker B renews lease -> must succeed
    renew_b = store.renew_lease(run_b.run_id, worker_b, lease_seconds=15.0)
    assert renew_b is True, "Active worker renewal should succeed"

    # Worker B marks run complete
    store.complete_run(run_b.run_id)

    # Neither worker can renew a completed run
    assert store.renew_lease(run_b.run_id, worker_b, lease_seconds=10.0) is False
    assert store.renew_lease(run_b.run_id, worker_a, lease_seconds=10.0) is False
    store.close()


def test_multiple_expired_runs_fifo_reclaim_order(tmp_path: Path) -> None:
    """Verify expired runs are reclaimed in FIFO order based on created_at."""
    db_file = tmp_path / "test_fifo.db"
    store = SQLiteJobStore(db_file)

    # Manually insert 3 runs with staggered created_at and already-expired leases
    now = time.time()
    conn = store._get_connection()
    runs_data = [
        ("run-early", "2026-10-07T01:00:00Z", now - 100),
        ("run-mid", "2026-10-07T02:00:00Z", now - 50),
        ("run-late", "2026-10-07T03:00:00Z", now - 10),
    ]
    for r_id, created, exp in runs_data:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, installation_id, repository_id, pr_number, generation,
                status, stage, worker_id, lease_expires_at, created_at
            ) VALUES (?, 1, 500, 1, 1, 'running', 'analyze', 'old-worker', ?, ?);
            """,
            (r_id, exp, created),
        )

    # Reclaiming worker must get them strictly in created_at ASC order
    c1 = store.claim_next_run("new-worker", lease_seconds=10.0)
    assert c1 is not None and c1.run_id == "run-early"

    c2 = store.claim_next_run("new-worker", lease_seconds=10.0)
    assert c2 is not None and c2.run_id == "run-mid"

    c3 = store.claim_next_run("new-worker", lease_seconds=10.0)
    assert c3 is not None and c3.run_id == "run-late"

    c4 = store.claim_next_run("new-worker", lease_seconds=10.0)
    assert c4 is None
    store.close()


# ==============================================================================
# 3. Delivery Deduplication Under Concurrent Pressure
# ==============================================================================


def test_massive_concurrent_duplicate_delivery_insertions(tmp_path: Path) -> None:
    """Stress-test 50 concurrent threads trying to record the exact same delivery_id simultaneously."""
    db_file = tmp_path / "test_dedup_stress.db"
    store = SQLiteJobStore(db_file)

    delivery = DeliveryRecord(
        delivery_id="deliv-stress-exact-dup",
        payload_digest="sha256-deadbeef",
        received_at="2026-10-07T05:00:00Z",
        action="synchronize",
        installation_id=1,
        repository_id=600,
        pr_number=7,
        processed=False,
    )

    thread_count = 50
    barrier = threading.Barrier(thread_count)
    successes: list[int] = []
    integrity_errors: list[sqlite3.IntegrityError] = []
    other_errors: list[Exception] = []
    lock = threading.Lock()

    def inserter(idx: int) -> None:
        t_store = SQLiteJobStore(db_file, timeout=10.0)
        try:
            barrier.wait()
            t_store.record_delivery(delivery)
            with lock:
                successes.append(idx)
        except sqlite3.IntegrityError as ie:
            with lock:
                integrity_errors.append(ie)
        except Exception as e:  # noqa: BLE001
            with lock:
                other_errors.append(e)
        finally:
            t_store.close()

    threads = [threading.Thread(target=inserter, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not other_errors, f"Unexpected errors: {other_errors}"
    assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
    assert len(integrity_errors) == 49, f"Expected 49 IntegrityErrors, got {len(integrity_errors)}"

    # Exactly 1 row in database
    conn = store._get_connection()
    count = conn.execute("SELECT COUNT(*) FROM deliveries WHERE delivery_id = 'deliv-stress-exact-dup';").fetchone()[0]
    assert count == 1
    assert store.is_delivery_processed("deliv-stress-exact-dup") is True
    store.close()


def test_mixed_deliveries_concurrent_pressure(tmp_path: Path) -> None:
    """Test 10 distinct deliveries submitted by 4 threads each (40 threads total) concurrently."""
    db_file = tmp_path / "test_mixed_dedup.db"
    store = SQLiteJobStore(db_file)

    unique_keys = [f"deliv-mixed-{i}" for i in range(10)]
    tasks = unique_keys * 4  # 40 items total
    barrier = threading.Barrier(len(tasks))
    success_keys: list[str] = []
    integrity_errors: list[Exception] = []
    lock = threading.Lock()

    def inserter(d_id: str) -> None:
        t_store = SQLiteJobStore(db_file, timeout=10.0)
        rec = DeliveryRecord(
            delivery_id=d_id,
            payload_digest="sha256-hash",
            received_at="2026-10-07T05:00:00Z",
            action="opened",
            installation_id=1,
            repository_id=700,
            pr_number=1,
            processed=False,
        )
        try:
            barrier.wait()
            t_store.record_delivery(rec)
            with lock:
                success_keys.append(d_id)
        except sqlite3.IntegrityError as ie:
            with lock:
                integrity_errors.append(ie)
        finally:
            t_store.close()

    threads = [threading.Thread(target=inserter, args=(k,)) for k in tasks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(success_keys) == 10
    assert set(success_keys) == set(unique_keys)
    assert len(integrity_errors) == 30

    conn = store._get_connection()
    count = conn.execute("SELECT COUNT(*) FROM deliveries;").fetchone()[0]
    assert count == 10
    store.close()


# ==============================================================================
# 4. Rapid PR Updates & Generation Coalescing
# ==============================================================================


def test_rapid_pr_updates_coalesce_generation_burst(tmp_path: Path) -> None:
    """Sequential burst of 25 rapid updates to a single PR coalesces to gen 25 in 1 run."""
    db_file = tmp_path / "test_burst_coalesce.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=800,
        pr_number=55,
        desired_generation=1,
    )

    for _ in range(25):
        store.enqueue_request(req)

    pr_rec = store.get_pr_request(800, 55)
    assert pr_rec is not None
    assert pr_rec.desired_generation == 25

    # Exactly 1 run must exist, coalesced to generation 25
    conn = store._get_connection()
    runs = conn.execute(
        "SELECT run_id, generation, status FROM runs WHERE repository_id = 800 AND pr_number = 55;"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0][1] == 25
    assert runs[0][2] == "queued"

    # Worker claims directly at generation 25
    claimed = store.claim_next_run("worker-coalesce", lease_seconds=10.0)
    assert claimed is not None
    assert claimed.generation == 25
    store.close()


def test_concurrent_rapid_pr_updates_from_multiple_threads(tmp_path: Path) -> None:
    """20 threads concurrently calling enqueue_request for the same PR.

    Desired generation must increment accurately to 20 without lost updates.
    """
    db_file = tmp_path / "test_concurrent_enqueue.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=900,
        pr_number=77,
        desired_generation=1,
    )

    thread_count = 20
    barrier = threading.Barrier(thread_count)
    errors: list[Exception] = []
    lock = threading.Lock()

    def updater() -> None:
        t_store = SQLiteJobStore(db_file, timeout=10.0)
        try:
            barrier.wait()
            t_store.enqueue_request(req)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(e)
        finally:
            t_store.close()

    threads = [threading.Thread(target=updater) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent enqueue errors: {errors}"

    pr_rec = store.get_pr_request(900, 77)
    assert pr_rec is not None
    assert pr_rec.desired_generation == 20

    # Pending run coalesced to generation 20
    claimed = store.claim_next_run("worker-conc", lease_seconds=10.0)
    assert claimed is not None
    assert claimed.generation == 20
    store.close()


def test_coalescing_with_active_running_job(tmp_path: Path) -> None:
    """When a job is currently RUNNING, new PR updates create a new run and coalesce it."""
    db_file = tmp_path / "test_running_coalesce.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=1000,
        pr_number=88,
        desired_generation=1,
    )

    # 1. Enqueue gen 1 and claim it
    store.enqueue_request(req)
    run_1 = store.claim_next_run("worker-1", lease_seconds=60.0)
    assert run_1 is not None
    assert run_1.generation == 1
    assert run_1.status == "running"

    # 2. PR push happens (update 2, 3, 4)
    store.enqueue_request(req)
    store.enqueue_request(req)
    store.enqueue_request(req)

    pr_rec = store.get_pr_request(1000, 88)
    assert pr_rec is not None
    assert pr_rec.desired_generation == 4

    # 3. Runs table must contain:
    # - run_1 (status='running', generation=1)
    # - run_2 (status='queued', generation=4)
    conn = store._get_connection()
    rows = conn.execute(
        "SELECT run_id, generation, status FROM runs WHERE repository_id = 1000 AND pr_number = 88 ORDER BY generation ASC;"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == run_1.run_id
    assert rows[0][1] == 1
    assert rows[0][2] == "running"

    assert rows[1][1] == 4
    assert rows[1][2] == "queued"

    # 4. Worker 1 fencing check detects staleness
    assert store.check_generation_fencing(run_1.run_id) is False

    # Run 1 is now SUPERSEDED
    stat = conn.execute("SELECT status FROM runs WHERE run_id = ?;", (run_1.run_id,)).fetchone()[0]
    assert stat == "superseded"

    # 5. Worker 2 claims the coalesced queued run at gen 4
    run_2 = store.claim_next_run("worker-2", lease_seconds=60.0)
    assert run_2 is not None
    assert run_2.generation == 4
    assert store.check_generation_fencing(run_2.run_id) is True
    store.close()


# ==============================================================================
# 5. Generation Fencing Detection & Stale Abort
# ==============================================================================


def test_generation_fencing_detection_and_superseded_transition(tmp_path: Path) -> None:
    """check_generation_fencing detects stale generations and marks run superseded."""
    db_file = tmp_path / "test_fencing_edge.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=1100,
        pr_number=12,
        desired_generation=1,
    )
    store.enqueue_request(req)

    run = store.claim_next_run("worker-fence", lease_seconds=30.0)
    assert run is not None

    # Current gen matches desired gen -> fresh
    assert store.check_generation_fencing(run.run_id) is True

    # Desired generation increments to 2
    store.enqueue_request(req)

    # Next check detects stale generation -> False
    assert store.check_generation_fencing(run.run_id) is False

    # Verify run is marked superseded in DB, completed_at set, lease cleared
    conn = store._get_connection()
    row = conn.execute(
        "SELECT status, completed_at, lease_expires_at FROM runs WHERE run_id = ?;",
        (run.run_id,),
    ).fetchone()
    assert row[0] == "superseded"
    assert row[1] is not None
    assert row[2] is None

    # Repeated calls remain False (idempotent)
    assert store.check_generation_fencing(run.run_id) is False
    store.close()


def test_worker_loop_fencing_aborts_pipeline_cleanly(tmp_path: Path) -> None:
    """WorkerLoop aborts immediately upon encountering stale generation mid-pipeline."""
    db_file = tmp_path / "test_worker_fencing.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=1200,
        pr_number=34,
        desired_generation=1,
    )
    store.enqueue_request(req)

    claimed = store.claim_next_run("test-worker", lease_seconds=30.0)
    assert claimed is not None

    # Before worker processes stages, advance generation
    store.enqueue_request(req)

    class DummyConfig:
        execution = None
        publish = None

    worker = WorkerLoop(config=DummyConfig(), job_store=store, worker_id="test-worker")

    # process_run must detect fencing at PREPARE stage and return False
    success = worker.process_run(claimed)
    assert success is False

    conn = store._get_connection()
    row = conn.execute("SELECT status, error FROM runs WHERE run_id = ?;", (claimed.run_id,)).fetchone()
    assert row[0] == "superseded"
    assert "Fenced" in (row[1] or "")
    store.close()


# ==============================================================================
# 6. Boundary Conditions & Stress Corner Cases
# ==============================================================================


def test_zero_and_negative_lease_seconds(tmp_path: Path) -> None:
    """Claiming with zero or negative lease expires immediately and is reclaimable."""
    db_file = tmp_path / "test_zero_lease.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=1300,
        pr_number=1,
        desired_generation=1,
    )
    store.enqueue_request(req)

    # Claim with lease of -1.0s (already expired)
    run_1 = store.claim_next_run("worker-instant-expire", lease_seconds=-1.0)
    assert run_1 is not None

    # Immediate second claim succeeds because lease is in the past
    run_2 = store.claim_next_run("worker-reclaimer", lease_seconds=10.0)
    assert run_2 is not None
    assert run_2.run_id == run_1.run_id
    assert run_2.attempts == 2
    store.close()


def test_immediate_transaction_rollback_on_failure(tmp_path: Path) -> None:
    """Verify BEGIN IMMEDIATE transaction rolls back cleanly on exception."""
    db_file = tmp_path / "test_tx_rollback.db"
    store = SQLiteJobStore(db_file)

    with pytest.raises(RuntimeError), store.transaction(immediate=True) as conn:
        conn.execute(
            "INSERT INTO deliveries (delivery_id, payload_digest, received_at) VALUES ('d-test', 'hash', datetime('now'));"
        )
        raise RuntimeError("Forced transaction failure")

    # The delivery row must NOT exist
    assert store.is_delivery_processed("d-test") is False

    # Store must remain fully usable after rollback
    delivery = DeliveryRecord(
        delivery_id="d-test-recovered",
        payload_digest="hash",
        received_at="2026-10-07T05:00:00Z",
        action="opened",
        installation_id=1,
        repository_id=1,
        pr_number=1,
        processed=False,
    )
    store.record_delivery(delivery)
    assert store.is_delivery_processed("d-test-recovered") is True
    store.close()


def test_sql_character_safety_in_run_error_and_receipts(tmp_path: Path) -> None:
    """Verify special characters, quotes, and Unicode in errors and publication receipts."""
    db_file = tmp_path / "test_special_chars.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=1400,
        pr_number=1,
        desired_generation=1,
    )
    store.enqueue_request(req)
    run = store.claim_next_run("worker-special", lease_seconds=10.0)
    assert run is not None

    malicious_error = "Error: O'Reilly \"-- DROP TABLE runs; -- '\n\tUnicode: 한글 🚀"
    store.fail_run(run.run_id, error=malicious_error)

    conn = store._get_connection()
    row = conn.execute("SELECT status, error FROM runs WHERE run_id = ?;", (run.run_id,)).fetchone()
    assert row[0] == "failed"
    assert row[1] == malicious_error

    # Publication receipt with quotes
    receipt = PublicationReceipt(
        comment_id=999,
        status="succeeded",
        fingerprint="fp-'\"-special",
        marker="<!-- review-bot:architecture:v1 -->",
        published_at="2026-10-07T05:00:00Z",
    )
    store.record_publication(receipt, 1400, 1)

    rec_row = conn.execute(
        "SELECT published_fingerprint FROM publication_receipts WHERE repository_id = 1400 AND pr_number = 1;"
    ).fetchone()
    assert rec_row[0] == "fp-'\"-special"
    store.close()


def test_stale_worker_lease_loss_cannot_overwrite_active_successor(tmp_path: Path) -> None:
    """Worker A's stale terminal write cannot clobber Worker B's active run or lease."""
    db_file = tmp_path / "test_stale_successor_clobber.db"
    store = SQLiteJobStore(db_file)

    req = PrRequestRecord(
        installation_id=1,
        repository_id=1500,
        pr_number=999,
        desired_generation=1,
    )
    store.enqueue_request(req)

    # 1. Worker A claims with a short lease (0.05s)
    run_a = store.claim_next_run("worker-a", lease_seconds=0.05)
    assert run_a is not None
    assert run_a.attempts == 1

    # 2. Worker A sleeps past lease expiry
    time.sleep(0.10)

    # 3. Worker B reclaims the expired run with a 30s lease
    run_b = store.claim_next_run("worker-b", lease_seconds=30.0)
    assert run_b is not None
    assert run_b.run_id == run_a.run_id
    assert run_b.attempts == 2

    conn = store._get_connection()
    row_b = conn.execute("SELECT status, worker_id, attempts FROM runs WHERE run_id = ?;", (run_a.run_id,)).fetchone()
    assert row_b[0] == "running"
    assert row_b[1] == "worker-b"

    # 4. Worker A wakes up and its heartbeat detects lease lost
    assert store.renew_lease(run_a.run_id, "worker-a", lease_seconds=10.0) is False

    before = conn.execute(
        "SELECT status, worker_id, lease_expires_at, error FROM runs WHERE run_id = ?;",
        (run_a.run_id,),
    ).fetchone()
    assert store.update_run(
        run_id=run_a.run_id, status=RunStatus.SUPERSEDED, error="Lease lost during execution",
        worker_id="worker-a", generation=run_a.generation,
    ) is False
    assert conn.execute(
        "SELECT status, worker_id, lease_expires_at, error FROM runs WHERE run_id = ?;",
        (run_a.run_id,),
    ).fetchone() == before

    # Worker B retains its active lease and can continue renewing it.
    worker_b_renew = store.renew_lease(run_b.run_id, "worker-b", lease_seconds=30.0)
    assert worker_b_renew is True

    store.close()

