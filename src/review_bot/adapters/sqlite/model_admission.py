"""Private SQLite admission for the shared, at-most-two model HTTP slots."""
from __future__ import annotations

import math
import os
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar
from uuid import UUID, uuid4


_CLEANUP_TIMEOUT_SECONDS = 1.0
_T = TypeVar("_T")
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS model_admission (
    token TEXT PRIMARY KEY NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('reserved', 'sent', 'uncertain', 'metadata')),
    expires_at REAL NOT NULL,
    caller_limit INTEGER NOT NULL CHECK (caller_limit IN (1, 2))
);
"""


class ModelAdmissionError(RuntimeError):
    """Admission cannot safely proceed; messages contain no caller or database data."""


class ModelAdmissionCancelled(ModelAdmissionError):
    """The caller cancelled before admission or before the HTTP send boundary."""


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Model admission requires finite numeric budgets")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("Model admission requires finite numeric budgets") from None
    if not math.isfinite(result):
        raise ValueError("Model admission requires finite numeric budgets")
    return result


def _private_file_metadata(metadata: os.stat_result) -> None:
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise ModelAdmissionError("Model admission files must be private, owned and not linked")


class ModelAdmissionTicket:
    """One reservation. Only a confirmed complete response or non-send permits finish."""

    def __init__(
        self,
        admission: SQLiteModelAdmission,
        token: str,
        deadline: float,
        generates: bool,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        self._identity_token = token
        self._admission = admission
        self._deadline = deadline
        self._generates = generates
        self._cancelled = cancelled
        self._lock = threading.Lock()
        self._sent = False
        self._closed = False
        self._finished = False

    @property
    def token(self) -> str:
        return self._identity_token

    def mark_sent(self) -> None:
        """CAS a live reservation immediately before starting HTTP, exactly once."""
        with self._lock:
            if self._sent or self._closed or self._finished:
                raise ModelAdmissionError("Model admission reservation is no longer valid")
            self._admission._mark_sent(self.token, self._deadline, self._generates, self._cancelled)
            self._sent = True

    def finish(self) -> None:
        """Confirm server completion or definite non-send and release only this token."""
        with self._lock:
            if self._finished:
                return
            self._admission._finish(self.token)
            self._finished = True
            self._closed = True

    def _cleanup(self) -> None:
        with self._lock:
            self._closed = True
            if not self._finished:
                self._finished = self._admission._cleanup(self.token, self._generates)


class SQLiteModelAdmission:
    """A shared hard cap of two; uncertain generating sends never expire automatically."""

    def __init__(
        self,
        db_path: Path,
        max_concurrency: int = 2,
        *,
        poll_interval: float = 0.05,
    ) -> None:
        if (
            not isinstance(db_path, Path)
            or not db_path.is_absolute()
            or db_path == Path("/")
            or ".." in db_path.parts
            or any(ord(character) < 32 or ord(character) == 127 for character in os.fspath(db_path))
        ):
            raise ValueError("Model admission requires a safe absolute database path")
        if type(max_concurrency) is not int or max_concurrency not in (1, 2):
            raise ValueError("Model admission concurrency must be one or two")
        interval = _finite_number(poll_interval)
        if interval <= 0:
            raise ValueError("Model admission poll interval must be positive")
        self.db_path = db_path
        self.max_concurrency = max_concurrency
        self.poll_interval = interval

    @contextmanager
    def request(
        self,
        *,
        deadline: float,
        generates: bool,
        cancelled: Callable[[], bool] | None = None,
    ) -> Generator[ModelAdmissionTicket, None, None]:
        """Wait within the caller's monotonic budget, without holding a transaction."""
        deadline = _finite_number(deadline)
        if type(generates) is not bool:
            raise ValueError("Model admission generates must be boolean")
        if cancelled is not None and not callable(cancelled):
            raise ValueError("Model admission cancellation must be callable")
        token = uuid4().hex

        def reserve(connection: sqlite3.Connection) -> bool:
            remaining = self._check_budget(deadline, None)
            now = time.time()
            expires_at = _finite_number(now + remaining)
            connection.execute(
                "DELETE FROM model_admission WHERE state IN ('reserved', 'metadata') AND expires_at <= ?",
                (now,),
            )
            limits = [row[0] for row in connection.execute("SELECT caller_limit FROM model_admission")]
            if any(type(limit) is not int or limit not in (1, 2) for limit in limits):
                raise ModelAdmissionError("Model admission state is unavailable")
            effective_limit = min(2, self.max_concurrency, *limits)
            if len(limits) >= effective_limit:
                return False
            connection.execute(
                "INSERT INTO model_admission (token, state, expires_at, caller_limit) VALUES (?, 'reserved', ?, ?)",
                (token, expires_at, self.max_concurrency),
            )
            return True

        while not self._run(reserve, deadline=deadline, cancelled=cancelled, create=True):
            self._wait(deadline, cancelled)
        ticket = ModelAdmissionTicket(self, token, deadline, generates, cancelled)
        try:
            self._check_budget(deadline, cancelled)
            yield ticket
        except BaseException:
            try:
                ticket._cleanup()
            except (ModelAdmissionError, TimeoutError):
                # Preserve the caller's failure; failed cleanup leaves a conservative DB slot.
                pass
            raise
        else:
            ticket._cleanup()

    def resolve_uncertain(self, token: str, *, confirmed_idle: bool) -> bool:
        """Explicit operator confirmation only; also resolves sent records left by a crash."""
        if confirmed_idle is not True:
            raise ValueError("Model admission resolution requires confirmed idle")
        try:
            valid = isinstance(token, str) and UUID(token).version == 4 and UUID(token).hex == token
        except (ValueError, AttributeError):
            valid = False
        if not valid:
            raise ValueError("Model admission resolution requires a valid ticket token")

        def resolve(connection: sqlite3.Connection) -> bool:
            return connection.execute(
                "DELETE FROM model_admission WHERE token = ? AND state IN ('sent', 'uncertain')",
                (token,),
            ).rowcount == 1

        return bool(self._run(resolve, deadline=time.monotonic() + _CLEANUP_TIMEOUT_SECONDS, missing_ok=True))

    def _mark_sent(
        self,
        token: str,
        deadline: float,
        generates: bool,
        cancelled: Callable[[], bool] | None,
    ) -> None:
        def mark(connection: sqlite3.Connection) -> None:
            updated = connection.execute(
                "UPDATE model_admission SET state = ? WHERE token = ? AND state = 'reserved' AND expires_at > ?",
                ("sent" if generates else "metadata", token, time.time()),
            ).rowcount
            if updated != 1:
                raise ModelAdmissionError("Model admission reservation is no longer valid")

        self._run(mark, deadline=deadline, cancelled=cancelled)

    def _finish(self, token: str) -> None:
        def finish(connection: sqlite3.Connection) -> None:
            connection.execute("DELETE FROM model_admission WHERE token = ?", (token,))

        self._run(finish, deadline=time.monotonic() + _CLEANUP_TIMEOUT_SECONDS)

    def _cleanup(self, token: str, generates: bool) -> bool:
        def cleanup(connection: sqlite3.Connection) -> bool:
            if generates:
                connection.execute(
                    "UPDATE model_admission SET state = 'uncertain' WHERE token = ? AND state = 'sent'", (token,)
                )
                connection.execute("DELETE FROM model_admission WHERE token = ? AND state = 'reserved'", (token,))
            else:
                connection.execute(
                    "DELETE FROM model_admission WHERE token = ? AND state IN ('reserved', 'metadata')", (token,)
                )
            return connection.execute("SELECT 1 FROM model_admission WHERE token = ?", (token,)).fetchone() is None

        return bool(self._run(cleanup, deadline=time.monotonic() + _CLEANUP_TIMEOUT_SECONDS))

    def _check_budget(self, deadline: float, cancelled: Callable[[], bool] | None) -> float:
        if cancelled is not None:
            try:
                cancellation = cancelled()
            except Exception:
                raise ModelAdmissionError("Model admission cancellation check failed") from None
            if type(cancellation) is not bool:
                raise ModelAdmissionError("Model admission cancellation check failed")
            if cancellation:
                raise ModelAdmissionCancelled("Model admission cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Model admission deadline exceeded")
        return remaining

    def _wait(self, deadline: float, cancelled: Callable[[], bool] | None) -> None:
        time.sleep(min(self.poll_interval, self._check_budget(deadline, cancelled)))

    def _run(
        self,
        operation: Callable[[sqlite3.Connection], _T],
        *,
        deadline: float,
        cancelled: Callable[[], bool] | None = None,
        create: bool = False,
        missing_ok: bool = False,
    ) -> _T | None:
        while True:
            self._check_budget(deadline, cancelled)
            try:
                with self._connection(create=create) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        self._check_budget(deadline, None)
                        connection.execute(_CREATE_TABLE_SQL)
                        result = operation(connection)
                        self._check_budget(deadline, None)
                        connection.execute("COMMIT")
                    except BaseException:
                        if connection.in_transaction:
                            connection.execute("ROLLBACK")
                        raise
                self._check_budget(deadline, cancelled)
                return result
            except FileNotFoundError:
                if missing_ok:
                    return None
                raise ModelAdmissionError("Model admission state is unavailable") from None
            except sqlite3.Error as error:
                code = getattr(error, "sqlite_errorcode", 0)
                if code & 0xFF not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                    raise ModelAdmissionError("Model admission state is unavailable") from None
            except TimeoutError:
                raise TimeoutError("Model admission deadline exceeded") from None
            except OSError:
                raise ModelAdmissionError("Model admission private state is unavailable") from None
            self._wait(deadline, cancelled)

    @contextmanager
    def _private_directory(self, *, create: bool) -> Generator[int, None, None]:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open("/", flags)
        try:
            for part in self.db_path.parent.parts[1:]:
                try:
                    child = os.open(part, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                    child = os.open(part, flags, dir_fd=descriptor)
                try:
                    metadata = os.fstat(child)
                    sticky_root = metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX)
                    if (
                        not stat.S_ISDIR(metadata.st_mode)
                        or metadata.st_uid not in (0, os.geteuid())
                        or (metadata.st_mode & 0o022 and not sticky_root)
                    ):
                        raise ModelAdmissionError("Model admission paths require trusted non-linked parents")
                except BaseException:
                    os.close(child)
                    raise
                os.close(descriptor)
                descriptor = child
            metadata = os.fstat(descriptor)
            if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise ModelAdmissionError("Model admission directory must be private and owned")
            yield descriptor
        finally:
            os.close(descriptor)

    def _check_files(self, directory: int) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            try:
                metadata = os.stat(self.db_path.name + suffix, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                if suffix:
                    continue
                raise
            # A concurrent SQLite connection may unlink a sidecar after pathname
            # lookup but before stat finishes (observed on macOS as nlink=0).
            # Recheck that pathname; never relax checks on a live file or the DB.
            if suffix and metadata.st_nlink == 0:
                try:
                    metadata = os.stat(self.db_path.name + suffix, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    continue
            _private_file_metadata(metadata)

    def _open_database(self, directory: int, *, create: bool) -> int:
        flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            metadata = os.stat(self.db_path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            if not create:
                raise
            try:
                descriptor = os.open(
                    self.db_path.name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory
                )
            except FileExistsError:
                descriptor = os.open(self.db_path.name, flags, dir_fd=directory)
            else:
                try:
                    os.fchmod(descriptor, 0o600)
                except BaseException:
                    os.close(descriptor)
                    raise
        else:
            _private_file_metadata(metadata)
            descriptor = os.open(self.db_path.name, flags, dir_fd=directory)
        try:
            _private_file_metadata(os.fstat(descriptor))
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _check_identity(self, directory: int, descriptor: int) -> None:
        expected = os.fstat(descriptor)
        actual = os.stat(self.db_path.name, dir_fd=directory, follow_symlinks=False)
        _private_file_metadata(expected)
        _private_file_metadata(actual)
        if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
            raise ModelAdmissionError("Model admission private state changed")
        with self._private_directory(create=False) as current:
            original_parent, current_parent = os.fstat(directory), os.fstat(current)
            if (original_parent.st_dev, original_parent.st_ino) != (current_parent.st_dev, current_parent.st_ino):
                raise ModelAdmissionError("Model admission private state changed")

    @contextmanager
    def _connection(self, *, create: bool) -> Generator[sqlite3.Connection, None, None]:
        with self._private_directory(create=create) as directory:
            descriptor = self._open_database(directory, create=create)
            try:
                self._check_identity(directory, descriptor)
                self._check_files(directory)
                connection = sqlite3.connect(
                    self.db_path.as_uri() + "?mode=rw", uri=True, timeout=0.0, isolation_level=None
                )
                try:
                    self._check_identity(directory, descriptor)
                    self._check_files(directory)
                    yield connection
                    self._check_identity(directory, descriptor)
                    self._check_files(directory)
                finally:
                    connection.close()
            finally:
                os.close(descriptor)
