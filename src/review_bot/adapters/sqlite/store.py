"""Durable SQLite WAL JobStore implementation."""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from review_bot.adapters.sqlite.schema import (
    CREATE_INDEXES_SQL,
    CREATE_TABLES_SQL,
)
from review_bot.domain.envelope_v2 import (
    DeliveryRecord,
    PrRequestRecord,
    PublicationReceipt,
    RunRecord,
)
from review_bot.domain.status import RunStageKind, RunStatus, RunStatusKind


_WORKER_FENCE_SQL = """
    AND worker_id = ?
    AND lease_expires_at > ?
    AND LOWER(status) IN ('running', 'claimed')
    AND generation = COALESCE(?, generation)
"""

_GENERATION_FENCE_SQL = """
    AND NOT EXISTS (
        SELECT 1 FROM pr_requests
        WHERE repository_id = runs.repository_id AND pr_number = runs.pr_number
          AND desired_generation > runs.generation
    )
"""


class SQLiteJobStore:
    """SQLite WAL implementation of the JobStore port with deduplication, coalescing, and leasing."""

    def __init__(self, db_path: Path | str = ":memory:", timeout: float = 5.0) -> None:
        self.db_path = str(db_path)
        self.timeout = timeout
        self._local = threading.local()
        self._init_db()

    def _utc_now_iso(self) -> str:
        return datetime.now(UTC).isoformat()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.timeout,
                isolation_level=None,  # Autocommit mode for explicit transaction control
                check_same_thread=False,
            )
            # Apply connection pragmas
            if self.db_path != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)};")
            conn.execute("PRAGMA foreign_keys = ON;")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.executescript(CREATE_TABLES_SQL)
        conn.executescript(CREATE_INDEXES_SQL)
        with self.transaction(immediate=True) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runs);")}
            if "reason" not in columns:
                conn.execute("ALTER TABLE runs ADD COLUMN reason TEXT;")
            receipt_columns = {row[1] for row in conn.execute("PRAGMA table_info(publication_receipts);")}
            if "status" not in receipt_columns:
                conn.execute("ALTER TABLE publication_receipts ADD COLUMN status TEXT NOT NULL DEFAULT 'unknown';")
            if "run_id" not in receipt_columns:
                conn.execute("ALTER TABLE publication_receipts ADD COLUMN run_id TEXT;")
            if "input_fingerprint" not in receipt_columns:
                conn.execute("ALTER TABLE publication_receipts ADD COLUMN input_fingerprint TEXT;")

    @contextmanager
    def transaction(self, immediate: bool = True) -> Generator[sqlite3.Connection, None, None]:
        """Atomic short transaction helper acquiring RESERVED lock immediately."""
        conn = self._get_connection()
        if conn.in_transaction:
            savepoint = f"sp_{uuid4().hex}"
            conn.execute(f"SAVEPOINT {savepoint};")
            try:
                yield conn
                conn.execute(f"RELEASE SAVEPOINT {savepoint};")
            except Exception:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint};")
                conn.execute(f"RELEASE SAVEPOINT {savepoint};")
                raise
            return
        cmd = "BEGIN IMMEDIATE;" if immediate else "BEGIN;"
        conn.execute(cmd)
        try:
            yield conn
            conn.execute("COMMIT;")
        except Exception:
            conn.execute("ROLLBACK;")
            raise

    # --------------------------------------------------------------------------
    # JobStore Protocol Implementation
    # --------------------------------------------------------------------------

    def accept_delivery(self, delivery: DeliveryRecord, request: PrRequestRecord) -> None:
        """Commit the delivery and runnable request together, or neither on failure."""
        with self.transaction(immediate=True):
            self.record_delivery(delivery)
            self.enqueue_request(request)

    def _cancel_request(
        self, conn: sqlite3.Connection, installation_id: int, repository_id: int,
        pr_number: int, reason: str,
    ) -> None:
        conn.execute(
            "UPDATE pr_requests SET desired_generation = desired_generation + 1, updated_at = datetime('now') "
            "WHERE installation_id = ? AND repository_id = ? AND pr_number = ?;",
            (installation_id, repository_id, pr_number),
        )
        conn.execute(
            "UPDATE runs SET status = 'cancelled', reason = ?, completed_at = datetime('now'), lease_expires_at = NULL "
            "WHERE installation_id = ? AND repository_id = ? AND pr_number = ? "
            "AND LOWER(status) IN ('pending', 'queued', 'running', 'claimed');",
            (reason, installation_id, repository_id, pr_number),
        )

    def accept_cancellation(self, delivery: DeliveryRecord) -> None:
        """Acknowledge a closed PR only after the delivery and cancellation are durable."""
        with self.transaction(immediate=True) as conn:
            self.record_delivery(delivery)
            self._cancel_request(conn, delivery.installation_id, delivery.repository_id, delivery.pr_number, "closed")

    def accept_revocation(self, delivery: DeliveryRecord, repository_ids: tuple[int, ...] | None = None) -> None:
        """Cancel only this installation's work; never delete private retained data."""
        with self.transaction(immediate=True) as conn:
            self.record_delivery(delivery)
            targets = conn.execute(
                "SELECT repository_id, pr_number FROM pr_requests WHERE installation_id = ?;",
                (delivery.installation_id,),
            ).fetchall()
            for repository_id, pr_number in targets:
                if repository_ids is None or repository_id in repository_ids:
                    self._cancel_request(conn, delivery.installation_id, repository_id, pr_number, "installation_revoked")

    def record_delivery(self, delivery: DeliveryRecord) -> None:
        """Record raw webhook delivery idempotently. Duplicate delivery_id raises sqlite3.IntegrityError."""
        with self.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO deliveries (
                    delivery_id, payload_digest, received_at, action,
                    installation_id, repository_id, pr_number, processed, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    delivery.delivery_id,
                    delivery.payload_digest,
                    delivery.received_at or self._utc_now_iso(),
                    delivery.action,
                    delivery.installation_id,
                    delivery.repository_id,
                    delivery.pr_number,
                    1 if delivery.processed else 0,
                    "received",
                ),
            )

    def is_delivery_processed(self, delivery_id: str) -> bool:
        """Check if delivery_id has already been recorded."""
        conn = self._get_connection()
        cur = conn.execute("SELECT 1 FROM deliveries WHERE delivery_id = ?;", (delivery_id,))
        return cur.fetchone() is not None

    def enqueue_request(self, request: PrRequestRecord) -> None:
        """Upsert PR request, coalesce desired_generation, and ensure a runnable run exists."""
        with self.transaction(immediate=True) as conn:
            # 1. Upsert pr_requests and increment desired_generation
            conn.execute(
                """
                INSERT INTO pr_requests (
                    installation_id, repository_id, pr_number, desired_generation, updated_at
                ) VALUES (?, ?, ?, ?, datetime('now'))
                ON CONFLICT(repository_id, pr_number) DO UPDATE SET
                    desired_generation = desired_generation + 1,
                    installation_id = excluded.installation_id,
                    updated_at = datetime('now');
                """,
                (
                    request.installation_id,
                    request.repository_id,
                    request.pr_number,
                    max(1, request.desired_generation),
                ),
            )

            # Fetch updated desired_generation
            cur = conn.execute(
                "SELECT desired_generation FROM pr_requests WHERE repository_id = ? AND pr_number = ?;",
                (request.repository_id, request.pr_number),
            )
            row = cur.fetchone()
            latest_gen = row[0] if row else 1

            # 2. Check if there is already a pending run to coalesce
            cur = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE repository_id = ? AND pr_number = ? AND LOWER(status) IN ('pending', 'queued');
                """,
                (request.repository_id, request.pr_number),
            )
            pending_row = cur.fetchone()
            if pending_row:
                # Coalesce generation of the pending run
                conn.execute(
                    "UPDATE runs SET generation = ?, installation_id = ? WHERE run_id = ?;",
                    (latest_gen, request.installation_id, pending_row[0]),
                )
            else:
                # Create new pending run
                new_run_id = f"run-{uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, installation_id, repository_id, pr_number,
                        generation, fingerprint, status, stage, created_at
                    ) VALUES (?, ?, ?, ?, ?, '', 'queued', 'fetch', datetime('now'));
                    """,
                    (
                        new_run_id,
                        request.installation_id,
                        request.repository_id,
                        request.pr_number,
                        latest_gen,
                    ),
                )

    def claim_next_run(self, worker_id: str, lease_seconds: float) -> RunRecord | None:
        """Atomically claim the next runnable execution with an expiring lease."""
        with self.transaction(immediate=True) as conn:
            now = time.time()
            lease_expires_at = now + lease_seconds
            # Find next runnable job: pending OR expired lease
            cur = conn.execute(
                """
                SELECT run_id, installation_id, repository_id, pr_number, generation,
                       fingerprint, status, stage, attempts, started_at, completed_at, error, reason
                FROM runs
                WHERE (
                    LOWER(status) IN ('pending', 'queued')
                    OR (
                        LOWER(status) IN ('running', 'claimed')
                        AND lease_expires_at IS NOT NULL
                        AND lease_expires_at <= ?
                    )
                )
                ORDER BY created_at ASC
                LIMIT 1;
                """,
                (now,),
            )
            row = cur.fetchone()
            if not row:
                return None

            run_id = row[0]
            new_attempts = (row[8] or 0) + 1

            # Atomic CAS update
            cur = conn.execute(
                """
                UPDATE runs
                SET status = 'running',
                    worker_id = ?,
                    lease_expires_at = ?,
                    attempts = ?,
                    started_at = COALESCE(started_at, datetime('now'))
                WHERE run_id = ?
                  AND (
                      LOWER(status) IN ('pending', 'queued')
                      OR (
                          LOWER(status) IN ('running', 'claimed')
                          AND lease_expires_at IS NOT NULL
                          AND lease_expires_at <= ?
                      )
                  );
                """,
                (worker_id, lease_expires_at, new_attempts, run_id, now),
            )

            if cur.rowcount == 0:
                return None

            return RunRecord(
                run_id=row[0],
                installation_id=row[1] or 0,
                repository_id=row[2],
                pr_number=row[3],
                generation=row[4],
                fingerprint=row[5] or "",
                status="running",
                stage=row[7] or "fetch",
                attempts=new_attempts,
                started_at=row[9] or self._utc_now_iso(),
                completed_at=row[10],
                error=row[11],
                reason=row[12],
            )

    def renew_lease(self, run_id: str, worker_id: str, lease_seconds: float) -> bool:
        """Extend active worker lease on a claimed run."""
        with self.transaction(immediate=True) as conn:
            now = time.time()
            new_expires_at = now + lease_seconds
            cur = conn.execute(
                """
                UPDATE runs
                SET lease_expires_at = ?
                WHERE run_id = ?
                  AND worker_id = ?
                  AND LOWER(status) IN ('running', 'claimed')
                  AND lease_expires_at > ?;
                """,
                (new_expires_at, run_id, worker_id, now),
            )
            return cur.rowcount > 0

    def update_run(
        self,
        run_id: str,
        status: RunStatusKind | str,
        stage: RunStageKind | str | None = None,
        error: str | None = None,
        *,
        worker_id: str | None = None,
        generation: int | None = None,
        reason: str | None = None,
        fingerprint: str | None = None,
    ) -> bool:
        """Update a run; worker calls are lease/generation fenced, omitted worker_id is administrative.

        Return False when the run is missing or a worker fence rejects the mutation.
        """
        if generation is not None and worker_id is None:
            raise ValueError("generation fencing requires worker_id")
        status_val = str(status.value if hasattr(status, "value") else status).lower()
        stage_val = str(stage.value if hasattr(stage, "value") else stage).lower() if stage is not None else None
        is_terminal = status_val in ("succeeded", "completed", "skipped", "failed", "superseded", "cancelled")

        with self.transaction(immediate=True) as conn:
            fence = ""
            params: tuple[object, ...] = (status_val, stage_val, error, reason, fingerprint, run_id)
            if worker_id is not None:
                fence = _WORKER_FENCE_SQL + _GENERATION_FENCE_SQL
                params += (worker_id, time.time(), generation)
            if is_terminal:
                cur = conn.execute(
                    f"""
                    UPDATE runs
                    SET status = ?,
                        stage = COALESCE(?, stage),
                        error = COALESCE(?, error),
                        reason = COALESCE(?, reason),
                        fingerprint = COALESCE(?, fingerprint),
                        completed_at = COALESCE(completed_at, datetime('now')),
                        lease_expires_at = NULL
                    WHERE run_id = ? {fence};
                    """,
                    params,
                )
            else:
                cur = conn.execute(
                    f"""
                    UPDATE runs
                    SET status = ?,
                        stage = COALESCE(?, stage),
                        error = COALESCE(?, error),
                        reason = COALESCE(?, reason),
                        fingerprint = COALESCE(?, fingerprint)
                    WHERE run_id = ? {fence};
                    """,
                    params,
                )
            return cur.rowcount > 0

    def complete_run(self, run_id: str) -> None:
        """Administrative completion; workers must use update_run with their lease fence."""
        self.update_run(run_id, status=RunStatus.SUCCEEDED)

    def fail_run(self, run_id: str, error: str) -> None:
        """Administrative failure; workers must use update_run with their lease fence."""
        self.update_run(run_id, status=RunStatus.FAILED, error=error)

    def record_publication(
        self, receipt: PublicationReceipt, repository_id: int, pr_number: int,
        *,
        run_id: str | None = None,
        worker_id: str | None = None,
        generation: int | None = None,
    ) -> bool:
        """Persist a receipt with an atomic run fence for workers; unfenced calls are administrative."""
        if worker_id is None and (run_id is not None or generation is not None):
            raise ValueError("publication fencing requires worker_id")
        if worker_id is not None and run_id is None:
            raise ValueError("publication fencing requires run_id")
        published_at = "" if receipt.status == "not_attempted" else receipt.published_at or self._utc_now_iso()
        with self.transaction(immediate=True) as conn:
            source = "WHERE 1"
            params: tuple[object, ...] = (
                repository_id, pr_number, receipt.marker, receipt.comment_id,
                receipt.fingerprint, published_at, receipt.status, run_id, receipt.input_fingerprint,
            )
            if worker_id is not None:
                source = (
                    "FROM runs WHERE run_id = ? AND repository_id = ? AND pr_number = ? "
                    + _WORKER_FENCE_SQL + _GENERATION_FENCE_SQL
                )
                params += (run_id, repository_id, pr_number, worker_id, time.time(), generation)
            cur = conn.execute(
                f"""
                INSERT INTO publication_receipts (
                    repository_id, pr_number, marker, comment_id, published_fingerprint, published_at, status, run_id, input_fingerprint
                ) SELECT ?, ?, ?, ?, ?, ?, ?, ?, ? {source}
                ON CONFLICT(repository_id, pr_number, marker) DO UPDATE SET
                    comment_id = excluded.comment_id,
                    published_fingerprint = excluded.published_fingerprint,
                    published_at = excluded.published_at,
                    status = excluded.status,
                    run_id = excluded.run_id,
                    input_fingerprint = excluded.input_fingerprint;
                """,
                params,
            )
            return cur.rowcount > 0

    def get_current_publication(
        self, run: RunRecord, marker: str, *, input_fingerprint: str | None = None,
    ) -> PublicationReceipt | None:
        """Reuse individual success, or the legacy A whole-run success lookup."""
        if input_fingerprint is not None and (
            type(input_fingerprint) is not str or len(input_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in input_fingerprint)
        ):
            return None
        match = (
            "p.input_fingerprint = ? AND r.status IN ('succeeded', 'failed', 'skipped')"
            if input_fingerprint is not None else "r.fingerprint = ? AND r.status = 'succeeded'"
        )
        row = self._get_connection().execute(
            "SELECT p.comment_id, p.published_fingerprint, p.published_at, p.status, p.input_fingerprint "
            "FROM publication_receipts p JOIN runs r ON r.run_id = p.run_id "
            "WHERE p.repository_id = ? AND p.pr_number = ? AND p.marker = ? "
            f"AND r.installation_id = ? AND {match} "
            "AND p.status IN ('fresh', 'succeeded');",
            (run.repository_id, run.pr_number, marker, run.installation_id,
                input_fingerprint if input_fingerprint is not None else run.fingerprint),
        ).fetchone()
        if row is None:
            return None
        return PublicationReceipt(row[0], row[3], row[1], marker, row[2] or None, input_fingerprint=row[4])

    def get_pr_request(
        self, repository_id: int, pr_number: int
    ) -> PrRequestRecord | None:
        """Fetch current desired_generation and record for a repository PR."""
        conn = self._get_connection()
        cur = conn.execute(
            """
            SELECT installation_id, repository_id, pr_number, desired_generation
            FROM pr_requests
            WHERE repository_id = ? AND pr_number = ?;
            """,
            (repository_id, pr_number),
        )
        row = cur.fetchone()
        if not row:
            return None
        return PrRequestRecord(
            installation_id=row[0],
            repository_id=row[1],
            pr_number=row[2],
            desired_generation=row[3],
        )

    def check_generation_fencing(
        self, run_id: str, *, worker_id: str | None = None, generation: int | None = None,
    ) -> bool:
        """Check generation atomically and supersede stale nonterminal runs, fenced for worker calls."""
        if generation is not None and worker_id is None:
            raise ValueError("generation fencing requires worker_id")
        with self.transaction(immediate=True) as conn:
            fence = ""
            params: tuple[object, ...] = (run_id,)
            if worker_id is not None:
                fence = _WORKER_FENCE_SQL
                params += (worker_id, time.time(), generation)
            cur = conn.execute(
                f"""
                SELECT generation, COALESCE((
                    SELECT desired_generation FROM pr_requests
                    WHERE repository_id = runs.repository_id AND pr_number = runs.pr_number
                ), generation)
                FROM runs WHERE run_id = ? {fence};
                """,
                params,
            )
            row = cur.fetchone()
            if row is None:
                return False
            is_fresh = row[0] >= row[1]
            if not is_fresh:
                if worker_id is not None:
                    params = (run_id, worker_id, time.time(), generation)
                conn.execute(
                    f"""
                    UPDATE runs
                    SET status = 'superseded',
                        completed_at = COALESCE(completed_at, datetime('now')),
                        lease_expires_at = NULL,
                        error = 'Fenced: desired generation has advanced'
                    WHERE run_id = ? {fence}
                      AND LOWER(status) IN ('pending', 'queued', 'running', 'claimed');
                    """,
                    params,
                )
            return is_fresh

    def close(self) -> None:
        """Close connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except sqlite3.Error:
                pass
            self._local.conn = None
