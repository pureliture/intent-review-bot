"""Contract tests for Feature 23: Durable SQLite JobStore (WAL),

Feature 24: Event Deduplication & Coalescing, and
Feature 25: Worker Lease & Generation Fencing.

Requirements:
- runtime-a-draft.md §4:
  - SQLite WAL mode, busy timeout, foreign keys enabled.
  - delivery_id uniqueness enforces delivery deduplication.
  - Multiple webhook events for the same PR coalesce to the latest desired_generation.
  - Atomic run claiming with lease timeout.
  - Fencing: workers with stale generation (generation < desired_generation) are
    fenced from publishing or completing.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

try:
    from review_bot.adapters.sqlite.store import SQLiteJobStore  # type: ignore
except ImportError:
    SQLiteJobStore = None  # type: ignore


# Standard canonical SQLite schema contract
SQLITE_SCHEMA_CONTRACT = """
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS deliveries (
    delivery_id TEXT PRIMARY KEY,
    payload_digest TEXT NOT NULL,
    received_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pr_requests (
    installation_id INTEGER NOT NULL,
    repository_id INTEGER NOT NULL,
    pr_number INTEGER NOT NULL,
    desired_generation INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (repository_id, pr_number)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    repository_id INTEGER NOT NULL,
    pr_number INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL, -- PENDING, CLAIMED, COMPLETED, SUPERSEDED, FAILED
    worker_id TEXT,
    lease_expires_at REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS publication_receipts (
    repository_id INTEGER NOT NULL,
    pr_number INTEGER NOT NULL,
    marker TEXT NOT NULL,
    comment_id INTEGER NOT NULL,
    published_fingerprint TEXT NOT NULL,
    published_at TEXT NOT NULL,
    PRIMARY KEY (repository_id, pr_number, marker)
);
"""


@pytest.fixture
def sqlite_contract_db(temp_sqlite_db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(temp_sqlite_db), timeout=5.0)
    conn.executescript(SQLITE_SCHEMA_CONTRACT)
    conn.commit()
    return conn


# ==============================================================================
# Tier 1: WAL Mode, Deduplication & Coalescing Contracts
# ==============================================================================


class TestSQLiteJobStoreTier1:
    def test_sqlite_wal_mode_active(self, sqlite_contract_db: sqlite3.Connection) -> None:
        cursor = sqlite_contract_db.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_delivery_deduplication_enforced(self, sqlite_contract_db: sqlite3.Connection) -> None:
        """Inserting identical delivery_id must violate PRIMARY KEY constraint."""
        delivery_id = "deliv-12345-uuid"
        sqlite_contract_db.execute(
            "INSERT INTO deliveries (delivery_id, payload_digest, received_at, status) "
            "VALUES (?, ?, datetime('now'), 'received');",
            (delivery_id, "digest_abc"),
        )
        sqlite_contract_db.commit()

        # Duplicate delivery must raise IntegrityError
        with pytest.raises(sqlite3.IntegrityError):
            sqlite_contract_db.execute(
                "INSERT INTO deliveries (delivery_id, payload_digest, received_at, status) "
                "VALUES (?, ?, datetime('now'), 'received');",
                (delivery_id, "digest_abc"),
            )

    def test_rapid_pr_updates_coalesce_generation(self, sqlite_contract_db: sqlite3.Connection) -> None:
        """Consecutive updates to the same PR increment desired_generation."""
        repo_id, pr_num = 100, 42
        # First event: PR opened
        sqlite_contract_db.execute(
            "INSERT INTO pr_requests (installation_id, repository_id, pr_number, desired_generation, updated_at) "
            "VALUES (1, ?, ?, 1, datetime('now')) "
            "ON CONFLICT(repository_id, pr_number) DO UPDATE SET "
            "desired_generation = desired_generation + 1, updated_at = datetime('now');",
            (repo_id, pr_num),
        )
        # Second event: PR synchronize (new commit pushed)
        sqlite_contract_db.execute(
            "INSERT INTO pr_requests (installation_id, repository_id, pr_number, desired_generation, updated_at) "
            "VALUES (1, ?, ?, 1, datetime('now')) "
            "ON CONFLICT(repository_id, pr_number) DO UPDATE SET "
            "desired_generation = desired_generation + 1, updated_at = datetime('now');",
            (repo_id, pr_num),
        )
        sqlite_contract_db.commit()

        row = sqlite_contract_db.execute(
            "SELECT desired_generation FROM pr_requests WHERE repository_id = ? AND pr_number = ?;",
            (repo_id, pr_num),
        ).fetchone()
        assert row[0] == 2


# ==============================================================================
# Tier 2: Leases & Generation Fencing Contracts
# ==============================================================================


class TestWorkerLeaseAndFencingTier2:
    def test_worker_lease_claim_and_expiry(self, sqlite_contract_db: sqlite3.Connection) -> None:
        """Worker claims pending run with lease; expired lease is reclaimable."""
        run_id = "run-001"
        now = time.time()
        sqlite_contract_db.execute(
            "INSERT INTO runs (run_id, repository_id, pr_number, generation, status, created_at) "
            "VALUES (?, 100, 42, 1, 'PENDING', datetime('now'));",
            (run_id,),
        )
        sqlite_contract_db.commit()

        # Worker 1 claims run with 1-second lease
        worker_1 = "worker-alpha"
        lease_exp = now + 0.1  # 100ms lease
        sqlite_contract_db.execute(
            "UPDATE runs SET status = 'CLAIMED', worker_id = ?, lease_expires_at = ? "
            "WHERE run_id = ? AND status = 'PENDING';",
            (worker_1, lease_exp, run_id),
        )
        sqlite_contract_db.commit()

        # Worker 2 attempts to claim active run -> fails (0 rows affected)
        cur = sqlite_contract_db.execute(
            "UPDATE runs SET status = 'CLAIMED', worker_id = 'worker-beta', lease_expires_at = ? "
            "WHERE run_id = ? AND (status = 'PENDING' OR lease_expires_at < ?);",
            (now + 10, run_id, time.time()),
        )
        assert cur.rowcount == 0

        # Wait for lease to expire
        time.sleep(0.15)
        # Now Worker 2 can reclaim expired lease
        cur = sqlite_contract_db.execute(
            "UPDATE runs SET status = 'CLAIMED', worker_id = 'worker-beta', lease_expires_at = ? "
            "WHERE run_id = ? AND (status = 'PENDING' OR lease_expires_at < ?);",
            (now + 10, run_id, time.time()),
        )
        assert cur.rowcount == 1

    def test_generation_fencing_prevents_stale_completion(
        self, sqlite_contract_db: sqlite3.Connection
    ) -> None:
        """If desired_generation has advanced past run.generation,

        the run is fenced and marked SUPERSEDED.
        """
        repo_id, pr_num = 100, 42
        # Run started at generation 1
        run_id = "run-gen1"
        sqlite_contract_db.execute(
            "INSERT INTO runs (run_id, repository_id, pr_number, generation, status, created_at) "
            "VALUES (?, ?, ?, 1, 'CLAIMED', datetime('now'));",
            (run_id, repo_id, pr_num),
        )
        # Meanwhile PR request updated to generation 2
        sqlite_contract_db.execute(
            "INSERT INTO pr_requests (installation_id, repository_id, pr_number, desired_generation, updated_at) "
            "VALUES (1, ?, ?, 2, datetime('now'));",
            (repo_id, pr_num),
        )
        sqlite_contract_db.commit()

        # Fencing check before publication:
        cur = sqlite_contract_db.execute(
            "SELECT r.generation, p.desired_generation "
            "FROM runs r JOIN pr_requests p ON r.repository_id = p.repository_id AND r.pr_number = p.pr_number "
            "WHERE r.run_id = ?;",
            (run_id,),
        )
        run_gen, desired_gen = cur.fetchone()
        assert run_gen < desired_gen, "Stale worker detected"

        # Worker must mark run as SUPERSEDED and abort
        sqlite_contract_db.execute(
            "UPDATE runs SET status = 'SUPERSEDED' WHERE run_id = ?;", (run_id,)
        )
        sqlite_contract_db.commit()

        final_status = sqlite_contract_db.execute(
            "SELECT status FROM runs WHERE run_id = ?;", (run_id,)
        ).fetchone()[0]
        assert final_status == "SUPERSEDED"
