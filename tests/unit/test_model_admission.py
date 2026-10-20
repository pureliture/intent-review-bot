"""Cross-process admission and fail-closed HTTP ticket lifetime tests."""
from __future__ import annotations

import multiprocessing
import os
import queue
import sqlite3
import stat
import threading
import time
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from review_bot.adapters.sqlite import model_admission as admission_module
from review_bot.adapters.sqlite.model_admission import (
    ModelAdmissionCancelled,
    ModelAdmissionError,
    SQLiteModelAdmission,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "private-state" / "admission.sqlite"


def _deadline(seconds: float = 3.0) -> float:
    return time.monotonic() + seconds


def _rows(db_path: Path) -> list[tuple]:
    with closing(sqlite3.connect(db_path)) as connection:
        return connection.execute(
            "SELECT token, state, expires_at, caller_limit FROM model_admission ORDER BY token"
        ).fetchall()


def _expire(db_path: Path, token: str) -> None:
    with closing(sqlite3.connect(db_path, isolation_level=None)) as connection:
        connection.execute(
            "UPDATE model_admission SET expires_at = ? WHERE token = ?",
            (time.time() - 1.0, token),
        )


def _join(thread: threading.Thread, errors: queue.Queue) -> None:
    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert errors.empty(), list(errors.queue)


def _process_hold(db_path, index, barrier, entered, release, attempted, errors):
    try:
        admission = SQLiteModelAdmission(Path(db_path), poll_interval=0.005)
        barrier.wait(timeout=5.0)
        with admission.request(
            deadline=_deadline(5.0), generates=True, cancelled=lambda: attempted.set() or False
        ) as ticket:
            ticket.mark_sent()
            entered.put((index, ticket.token))
            assert release.wait(timeout=5.0)
            ticket.finish()
    except BaseException as error:
        errors.put(type(error).__name__)


def _process_crash(db_path, generates, mark_sent, sender):
    admission = SQLiteModelAdmission(Path(db_path), max_concurrency=1)
    with admission.request(deadline=_deadline(10.0), generates=generates) as ticket:
        if mark_sent:
            ticket.mark_sent()
        sender.send(ticket.token)
        sender.close()
        os._exit(17)


def test_construction_is_offline_and_umask_is_never_changed(db_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected construction I/O or umask mutation")

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", forbidden)
        patch.setattr(os, "open", forbidden)
        patch.setattr(Path, "lstat", forbidden)
        patch.setattr(Path, "mkdir", forbidden)
        patch.setattr(os, "umask", forbidden)
        admission = SQLiteModelAdmission(db_path)
    assert not db_path.parent.exists()
    monkeypatch.setattr(os, "umask", forbidden)
    with admission.request(deadline=_deadline(), generates=True) as ticket:
        ticket.mark_sent()
        ticket.finish()
    assert db_path.exists()


@pytest.mark.parametrize("limit", [True, False, 0, -1, 3, 1.0, float("nan"), float("inf"), "2"])
def test_invalid_concurrency_is_rejected_without_io(db_path, limit):
    with pytest.raises(ValueError):
        SQLiteModelAdmission(db_path, max_concurrency=limit)
    assert not db_path.parent.exists()


@pytest.mark.parametrize("interval", [True, False, 0, -0.1, float("nan"), float("inf"), "0.05"])
def test_invalid_poll_interval_is_rejected_without_io(db_path, interval):
    with pytest.raises(ValueError):
        SQLiteModelAdmission(db_path, poll_interval=interval)
    assert not db_path.parent.exists()


@pytest.mark.parametrize("deadline", [True, False, float("nan"), float("inf"), "later"])
def test_invalid_deadline_is_rejected_without_io(db_path, deadline):
    with pytest.raises(ValueError):
        with SQLiteModelAdmission(db_path).request(deadline=deadline, generates=True):
            pytest.fail("invalid deadline admitted")
    assert not db_path.parent.exists()


@pytest.mark.parametrize("generates", [0, 1, None, "true"])
def test_generates_must_be_boolean(db_path, generates):
    with pytest.raises(ValueError):
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=generates):
            pytest.fail("invalid request admitted")
    assert not db_path.parent.exists()


def test_cancelled_must_be_callable(db_path):
    with pytest.raises(ValueError):
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True, cancelled=True):
            pytest.fail("invalid cancellation admitted")
    assert not db_path.parent.exists()


def test_expired_deadline_and_initial_cancellation_do_not_create_files(db_path):
    admission = SQLiteModelAdmission(db_path)
    with pytest.raises(TimeoutError):
        with admission.request(deadline=time.monotonic() - 1.0, generates=True):
            pytest.fail("expired request admitted")
    with pytest.raises(ModelAdmissionCancelled):
        with admission.request(deadline=_deadline(), generates=True, cancelled=lambda: True):
            pytest.fail("cancelled request admitted")
    assert not db_path.parent.exists()


def test_persisted_expiry_is_wall_clock_not_monotonic(db_path, monkeypatch):
    clock = SimpleNamespace(monotonic=lambda: 100.0, time=lambda: 2_000_000_000.0)
    monkeypatch.setattr(admission_module, "time", clock)
    with SQLiteModelAdmission(db_path).request(deadline=125.0, generates=True) as ticket:
        assert UUID(ticket.token).version == 4
        assert _rows(db_path) == [(ticket.token, "reserved", 2_000_000_025.0, 2)]
    assert _rows(db_path) == []


def test_only_ticket_fields_are_persisted(db_path):
    with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True):
        with closing(sqlite3.connect(db_path)) as connection:
            columns = [row[1] for row in connection.execute("PRAGMA table_info(model_admission)")]
        assert columns == ["token", "state", "expires_at", "caller_limit"]


def test_two_tickets_enter_and_third_thread_waits_until_finish(db_path):
    admission = SQLiteModelAdmission(db_path, poll_interval=0.005)
    attempted, entered, release = threading.Event(), threading.Event(), threading.Event()
    errors = queue.Queue()

    def third_request():
        try:
            with SQLiteModelAdmission(db_path, poll_interval=0.005).request(
                deadline=_deadline(), generates=True, cancelled=lambda: attempted.set() or False
            ) as ticket:
                ticket.mark_sent()
                entered.set()
                assert release.wait(timeout=2.0)
                ticket.finish()
        except BaseException as error:
            errors.put(error)

    with ExitStack() as stack:
        first = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        second = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        first.mark_sent()
        second.mark_sent()
        assert len(_rows(db_path)) == 2
        thread = threading.Thread(target=third_request)
        thread.start()
        try:
            assert attempted.wait(timeout=2.0)
            assert not entered.wait(timeout=0.03)
            first.finish()
            assert entered.wait(timeout=2.0)
            assert len(_rows(db_path)) == 2
        finally:
            release.set()
            _join(thread, errors)
            second.finish()
    assert _rows(db_path) == []


def test_three_processes_share_exactly_two_slots_and_race_schema_initialization(db_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(4)
    entered, errors = context.Queue(), context.Queue()
    releases = [context.Event() for _ in range(3)]
    attempted = [context.Event() for _ in range(3)]
    processes = [
        context.Process(
            target=_process_hold,
            args=(str(db_path), index, barrier, entered, releases[index], attempted[index], errors),
        )
        for index in range(3)
    ]
    for process in processes:
        process.start()
    try:
        barrier.wait(timeout=5.0)
        first = entered.get(timeout=5.0)
        second = entered.get(timeout=5.0)
        assert first[0] != second[0]
        assert all(event.wait(timeout=2.0) for event in attempted)
        with pytest.raises(queue.Empty):
            entered.get(timeout=0.03)
        assert len(_rows(db_path)) == 2
        releases[first[0]].set()
        third = entered.get(timeout=5.0)
        assert third[0] not in (first[0], second[0])
        assert len(_rows(db_path)) == 2
    finally:
        for release in releases:
            release.set()
        for process in processes:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2.0)
        assert all(process.exitcode == 0 for process in processes)
    with pytest.raises(queue.Empty):
        errors.get(timeout=0.03)
    assert _rows(db_path) == []


@pytest.mark.parametrize("active_limit,new_limit", [(1, 2), (2, 1), (1, 1)])
def test_minimum_of_active_and_new_caller_limits_is_enforced(db_path, active_limit, new_limit):
    active = SQLiteModelAdmission(db_path, max_concurrency=active_limit)
    newcomer = SQLiteModelAdmission(db_path, max_concurrency=new_limit, poll_interval=0.005)
    with active.request(deadline=_deadline(), generates=True) as ticket:
        ticket.mark_sent()
        with pytest.raises(TimeoutError):
            with newcomer.request(deadline=_deadline(0.03), generates=True):
                pytest.fail("single-slot limit bypassed")
        assert [row[0] for row in _rows(db_path)] == [ticket.token]
        ticket.finish()
        with newcomer.request(deadline=_deadline(), generates=True) as replacement:
            assert replacement.token != ticket.token
    assert _rows(db_path) == []


def test_cancellation_while_waiting_preserves_other_tickets(db_path):
    admission = SQLiteModelAdmission(db_path, poll_interval=0.005)
    attempted, cancel = threading.Event(), threading.Event()
    result = queue.Queue()

    def cancelled():
        attempted.set()
        return cancel.is_set()

    def wait_for_slot():
        try:
            with admission.request(deadline=_deadline(), generates=True, cancelled=cancelled):
                result.put("unexpected admission")
        except BaseException as error:
            result.put(error)

    with ExitStack() as stack:
        tickets = [stack.enter_context(admission.request(deadline=_deadline(), generates=True)) for _ in range(2)]
        thread = threading.Thread(target=wait_for_slot)
        thread.start()
        try:
            assert attempted.wait(timeout=2.0)
            cancel.set()
            outcome = result.get(timeout=2.0)
            assert isinstance(outcome, ModelAdmissionCancelled)
            assert {row[0] for row in _rows(db_path)} == {ticket.token for ticket in tickets}
        finally:
            cancel.set()
            thread.join(timeout=3.0)
            assert not thread.is_alive()


def test_cancellation_callback_failure_is_sanitized(db_path):
    def cancelled():
        raise RuntimeError("synthetic-private-input")

    with pytest.raises(ModelAdmissionError) as raised:
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True, cancelled=cancelled):
            pytest.fail("failed cancellation check admitted")
    assert "synthetic-private-input" not in str(raised.value)
    assert raised.value.__suppress_context__
    assert not db_path.parent.exists()


def test_cancellation_callback_runs_outside_sqlite_write_transactions(db_path):
    def cancelled():
        if db_path.exists():
            with closing(sqlite3.connect(db_path, isolation_level=None, timeout=0)) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("ROLLBACK")
        return False

    with SQLiteModelAdmission(db_path).request(
        deadline=_deadline(), generates=True, cancelled=cancelled
    ) as ticket:
        ticket.mark_sent()
        ticket.finish()
    assert _rows(db_path) == []


def test_cancellation_after_reservation_prevents_mark_sent(db_path):
    cancel = threading.Event()
    with SQLiteModelAdmission(db_path).request(
        deadline=_deadline(), generates=True, cancelled=cancel.is_set
    ) as ticket:
        cancel.set()
        with pytest.raises(ModelAdmissionCancelled):
            ticket.mark_sent()
        assert _rows(db_path)[0][1] == "reserved"
    assert _rows(db_path) == []


def test_reclaimed_reserved_ticket_cannot_send_or_delete_replacement(db_path):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=True) as stale:
        _expire(db_path, stale.token)
        with admission.request(deadline=_deadline(), generates=True) as replacement:
            with pytest.raises(ModelAdmissionError):
                stale.mark_sent()
            stale.finish()
            stale.finish()
            assert [row[0] for row in _rows(db_path)] == [replacement.token]
            replacement.mark_sent()
            replacement.finish()
    assert _rows(db_path) == []


def test_mark_sent_checks_monotonic_deadline_even_if_wall_expiry_is_future(db_path, monkeypatch):
    clock = SimpleNamespace(monotonic=lambda: 100.0, time=lambda: 2_000_000_000.0)
    monkeypatch.setattr(admission_module, "time", clock)
    with SQLiteModelAdmission(db_path).request(deadline=101.0, generates=True) as ticket:
        clock.monotonic = lambda: 102.0
        with pytest.raises(TimeoutError):
            ticket.mark_sent()
    assert _rows(db_path) == []


def test_mark_sent_is_one_shot_and_closed_ticket_cannot_send(db_path):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=True) as ticket:
        ticket.mark_sent()
        with pytest.raises(ModelAdmissionError):
            ticket.mark_sent()
        ticket.finish()
        with pytest.raises(ModelAdmissionError):
            ticket.mark_sent()
    with admission.request(deadline=_deadline(), generates=False) as unused:
        pass
    with pytest.raises(ModelAdmissionError):
        unused.mark_sent()


def test_ticket_token_cannot_be_replaced_to_finish_another_ticket(db_path):
    admission = SQLiteModelAdmission(db_path)
    with ExitStack() as stack:
        first = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        second = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        with pytest.raises(AttributeError):
            first.token = second.token
        first.finish()
        assert [row[0] for row in _rows(db_path)] == [second.token]


def test_concurrent_mark_sent_has_exactly_one_winner(db_path):
    barrier = threading.Barrier(3)
    results = queue.Queue()
    with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True) as ticket:
        def mark():
            try:
                barrier.wait(timeout=2.0)
                ticket.mark_sent()
                results.put("sent")
            except BaseException as error:
                results.put(error)

        threads = [threading.Thread(target=mark) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2.0)
        for thread in threads:
            thread.join(timeout=3.0)
            assert not thread.is_alive()
        outcomes = [results.get(timeout=1.0) for _ in threads]
        assert sum(outcome == "sent" for outcome in outcomes) == 1
        assert sum(isinstance(outcome, ModelAdmissionError) for outcome in outcomes) == 1
        assert _rows(db_path)[0][:2] == (ticket.token, "sent")
        ticket.finish()


@pytest.mark.parametrize("generates", [False, True])
def test_finish_is_idempotent_and_releases_only_its_own_token(db_path, generates):
    admission = SQLiteModelAdmission(db_path)
    with ExitStack() as stack:
        first = stack.enter_context(admission.request(deadline=_deadline(), generates=generates))
        second = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        first.mark_sent()
        second.mark_sent()
        first.finish()
        first.finish()
        assert [row[0] for row in _rows(db_path)] == [second.token]
        second.finish()
    first.finish()
    assert _rows(db_path) == []


def test_concurrent_finish_is_idempotent_and_preserves_another_ticket(db_path):
    admission = SQLiteModelAdmission(db_path)
    barrier = threading.Barrier(4)
    errors = queue.Queue()
    with ExitStack() as stack:
        first = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        second = stack.enter_context(admission.request(deadline=_deadline(), generates=True))
        first.mark_sent()
        second.mark_sent()

        def finish():
            try:
                barrier.wait(timeout=2.0)
                first.finish()
            except BaseException as error:
                errors.put(error)

        threads = [threading.Thread(target=finish) for _ in range(3)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2.0)
        for thread in threads:
            _join(thread, errors)
        assert _rows(db_path)[0][:2] == (second.token, "sent")
        second.finish()


@pytest.mark.parametrize("generates", [False, True])
@pytest.mark.parametrize("exception", [None, TimeoutError, ConnectionError])
def test_sent_generation_is_preserved_but_metadata_is_released(db_path, generates, exception):
    admission = SQLiteModelAdmission(db_path)
    token = None

    def request():
        nonlocal token
        with admission.request(deadline=_deadline(), generates=generates) as ticket:
            token = ticket.token
            ticket.mark_sent()
            if exception is not None:
                raise exception("synthetic transport failure")

    if exception is None:
        request()
    else:
        with pytest.raises(exception):
            request()
    if generates:
        assert _rows(db_path)[0][:2] == (token, "uncertain")
        _expire(db_path, token)
        with admission.request(deadline=_deadline(), generates=False):
            assert len(_rows(db_path)) == 2
        assert _rows(db_path)[0][:2] == (token, "uncertain")
    else:
        assert _rows(db_path) == []


def test_full_response_finish_releases_before_caller_validation_failure(db_path):
    with pytest.raises(ValueError, match="synthetic validation failure"):
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True) as ticket:
            ticket.mark_sent()
            ticket.finish()
            raise ValueError("synthetic validation failure")
    assert _rows(db_path) == []


def test_unsent_exception_releases_reservation(db_path):
    with pytest.raises(RuntimeError, match="synthetic local failure"):
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True):
            raise RuntimeError("synthetic local failure")
    assert _rows(db_path) == []


def test_finish_can_confirm_response_after_uncertain_context_exit(db_path):
    with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True) as ticket:
        ticket.mark_sent()
    assert _rows(db_path)[0][1] == "uncertain"
    ticket.finish()
    ticket.finish()
    assert _rows(db_path) == []


def test_two_expired_uncertain_generations_still_block_new_admission(db_path):
    admission = SQLiteModelAdmission(db_path, poll_interval=0.005)
    tokens = []
    for _ in range(2):
        with admission.request(deadline=_deadline(), generates=True) as ticket:
            ticket.mark_sent()
            tokens.append(ticket.token)
        _expire(db_path, ticket.token)
    with pytest.raises(TimeoutError):
        with admission.request(deadline=_deadline(0.03), generates=True):
            pytest.fail("uncertain generation expired automatically")
    assert {row[0] for row in _rows(db_path)} == set(tokens)
    assert admission.resolve_uncertain(tokens[0], confirmed_idle=True)
    with admission.request(deadline=_deadline(), generates=True) as replacement:
        assert {row[0] for row in _rows(db_path)} == {tokens[1], replacement.token}
    assert admission.resolve_uncertain(tokens[1], confirmed_idle=True)


@pytest.mark.parametrize("generates,mark_sent,state", [(True, False, "reserved"), (True, True, "sent"), (False, True, "metadata")])
def test_process_crash_only_allows_reclaim_of_non_generating_work(db_path, generates, mark_sent, state):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_process_crash, args=(str(db_path), generates, mark_sent, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(timeout=5.0)
        token = receiver.recv()
        process.join(timeout=5.0)
        assert process.exitcode == 17
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
    assert _rows(db_path)[0][:2] == (token, state)
    _expire(db_path, token)
    admission = SQLiteModelAdmission(db_path, poll_interval=0.005)
    if state == "sent":
        with pytest.raises(TimeoutError):
            with admission.request(deadline=_deadline(0.03), generates=True):
                pytest.fail("crashed generation was reclaimed")
        assert _rows(db_path)[0][:2] == (token, "sent")
        assert admission.resolve_uncertain(token, confirmed_idle=True) is True
    with admission.request(deadline=_deadline(), generates=True) as replacement:
        assert [row[0] for row in _rows(db_path)] == [replacement.token]
    assert _rows(db_path) == []


@pytest.mark.parametrize("confirmation", [False, 1, "yes", None])
def test_uncertain_resolution_requires_literal_true_and_preserves_other_slots(db_path, confirmation):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=True) as first:
        first.mark_sent()
    with admission.request(deadline=_deadline(), generates=True) as second:
        with pytest.raises(ValueError):
            admission.resolve_uncertain(first.token, confirmed_idle=confirmation)
        assert {row[0] for row in _rows(db_path)} == {first.token, second.token}
        assert admission.resolve_uncertain(second.token, confirmed_idle=True) is False
        assert admission.resolve_uncertain(first.token, confirmed_idle=True) is True
        assert admission.resolve_uncertain(first.token, confirmed_idle=True) is False
        assert [row[0] for row in _rows(db_path)] == [second.token]


def test_resolve_before_first_request_never_creates_files(db_path):
    assert SQLiteModelAdmission(db_path).resolve_uncertain(uuid4().hex, confirmed_idle=True) is False
    assert not db_path.parent.exists()


@pytest.mark.parametrize("token", ["", "synthetic-token", "' OR 1=1 --", None, 17])
def test_invalid_resolution_token_is_rejected_without_io(db_path, token):
    with pytest.raises(ValueError):
        SQLiteModelAdmission(db_path).resolve_uncertain(token, confirmed_idle=True)
    assert not db_path.parent.exists()


def test_transient_database_lock_is_retried_without_deleting_other_slots(db_path):
    admission = SQLiteModelAdmission(db_path, poll_interval=0.005)
    with admission.request(deadline=_deadline(), generates=True) as first:
        first.mark_sent()
        with closing(sqlite3.connect(db_path, isolation_level=None, timeout=0)) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            attempted, entered = threading.Event(), threading.Event()
            errors = queue.Queue()

            def wait_for_lock():
                try:
                    with admission.request(
                        deadline=_deadline(), generates=False, cancelled=lambda: attempted.set() or False
                    ) as ticket:
                        entered.set()
                        ticket.finish()
                except BaseException as error:
                    errors.put(error)

            thread = threading.Thread(target=wait_for_lock)
            thread.start()
            try:
                assert attempted.wait(timeout=2.0)
                assert not entered.wait(timeout=0.03)
                assert _rows(db_path)[0][0] == first.token
                blocker.execute("ROLLBACK")
                assert entered.wait(timeout=2.0)
            finally:
                if blocker.in_transaction:
                    blocker.execute("ROLLBACK")
                _join(thread, errors)
        assert _rows(db_path)[0][:2] == (first.token, "sent")
        first.finish()


def test_lock_deadline_and_failed_finish_do_not_delete_any_slot(db_path, monkeypatch):
    admission = SQLiteModelAdmission(db_path, poll_interval=0.005)
    monkeypatch.setattr(admission_module, "_CLEANUP_TIMEOUT_SECONDS", 0.03)
    with ExitStack() as stack:
        tickets = [stack.enter_context(admission.request(deadline=_deadline(), generates=True)) for _ in range(2)]
        for ticket in tickets:
            ticket.mark_sent()
        with closing(sqlite3.connect(db_path, isolation_level=None, timeout=0)) as blocker:
            blocker.execute("BEGIN IMMEDIATE")
            try:
                with pytest.raises(TimeoutError):
                    with admission.request(deadline=_deadline(0.03), generates=True):
                        pytest.fail("locked database admitted")
                with pytest.raises(TimeoutError):
                    tickets[0].finish()
                assert {row[0] for row in _rows(db_path)} == {ticket.token for ticket in tickets}
            finally:
                blocker.execute("ROLLBACK")
        for ticket in tickets:
            ticket.finish()
    assert _rows(db_path) == []


def test_sqlite_errors_are_safe_and_do_not_include_path_or_raw_error(db_path, monkeypatch):
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic-private-input")

    monkeypatch.setattr(sqlite3, "connect", fail)
    with pytest.raises(ModelAdmissionError) as raised:
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True):
            pytest.fail("unavailable database admitted")
    assert "synthetic-private-input" not in str(raised.value)
    assert str(db_path) not in str(raised.value)
    assert raised.value.__suppress_context__


def test_deadline_expiring_during_connect_remains_timeout_and_rolls_back(db_path, monkeypatch):
    clock = SimpleNamespace(now=100.0, time=lambda: 2_000_000_000.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(admission_module, "time", clock)
    original = sqlite3.connect

    def slow_connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        clock.now = 102.0
        return connection

    monkeypatch.setattr(sqlite3, "connect", slow_connect)
    with pytest.raises(TimeoutError, match="deadline exceeded"):
        with SQLiteModelAdmission(db_path).request(deadline=101.0, generates=True):
            pytest.fail("over-budget connection admitted")
    with closing(original(db_path)) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE name = 'model_admission'").fetchall() == []


def test_mark_sent_rechecks_deadline_after_commit_and_connection_close(db_path, monkeypatch):
    clock = SimpleNamespace(now=100.0, time=lambda: 2_000_000_000.0)
    clock.monotonic = lambda: clock.now
    monkeypatch.setattr(admission_module, "time", clock)
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=105.0, generates=True) as ticket:
        original = admission._connection

        @contextmanager
        def late_close(*, create):
            with original(create=create) as connection:
                yield connection
            clock.now = 106.0

        with monkeypatch.context() as patch:
            patch.setattr(admission, "_connection", late_close)
            with pytest.raises(TimeoutError):
                ticket.mark_sent()
    assert _rows(db_path)[0][:2] == (ticket.token, "uncertain")
    ticket.finish()
    assert _rows(db_path) == []


def test_failed_cleanup_preserves_sent_slot_and_original_timeout(db_path, monkeypatch):
    admission = SQLiteModelAdmission(db_path)
    monkeypatch.setattr(admission_module, "_CLEANUP_TIMEOUT_SECONDS", 0.03)
    blocker = None
    try:
        with pytest.raises(TimeoutError, match="synthetic transport timeout"):
            with admission.request(deadline=_deadline(), generates=True) as ticket:
                ticket.mark_sent()
                blocker = sqlite3.connect(db_path, isolation_level=None, timeout=0)
                blocker.execute("BEGIN IMMEDIATE")
                raise TimeoutError("synthetic transport timeout")
        assert _rows(db_path)[0][:2] == (ticket.token, "sent")
    finally:
        if blocker is not None:
            blocker.execute("ROLLBACK")
            blocker.close()
    assert admission.resolve_uncertain(ticket.token, confirmed_idle=True)


def test_new_directories_and_database_have_exact_private_permissions(tmp_path):
    db_path = tmp_path / "first" / "second" / "admission.sqlite"
    with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True):
        for directory in (db_path.parent, db_path.parent.parent):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            assert directory.stat().st_uid == os.geteuid()
        assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        assert db_path.stat().st_nlink == 1


def test_wal_and_shm_inherit_database_permissions(db_path):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=False):
        pass
    with closing(sqlite3.connect(db_path, isolation_level=None)) as connection:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        connection.execute("BEGIN")
        connection.execute("SELECT COUNT(*) FROM model_admission").fetchone()
        with admission.request(deadline=_deadline(), generates=True):
            for suffix in ("", "-wal", "-shm"):
                metadata = Path(str(db_path) + suffix).lstat()
                assert stat.S_IMODE(metadata.st_mode) == 0o600
                assert metadata.st_uid == os.geteuid()
                assert metadata.st_nlink == 1
        connection.execute("ROLLBACK")


@pytest.mark.parametrize("path", [Path("relative.sqlite"), Path("/"), Path("/synthetic/../admission.sqlite"), Path("/synthetic/admission\n.sqlite")])
def test_unsafe_path_shape_is_rejected_without_io(path):
    with pytest.raises(ValueError):
        SQLiteModelAdmission(path)


@pytest.mark.parametrize("mode", [0o755, 0o770, 0o777])
def test_existing_parent_must_be_private_and_is_never_chmodded(db_path, mode):
    db_path.parent.mkdir(mode=0o700)
    db_path.parent.chmod(mode)
    with pytest.raises(ModelAdmissionError):
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True):
            pytest.fail("unsafe parent admitted")
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == mode
    assert not db_path.exists()


def test_writable_ancestor_is_rejected_even_if_leaf_is_private(tmp_path):
    ancestor = tmp_path / "unsafe-ancestor"
    ancestor.mkdir(mode=0o700)
    ancestor.chmod(0o770)
    private = ancestor / "private-leaf"
    private.mkdir(mode=0o700)
    with pytest.raises(ModelAdmissionError):
        with SQLiteModelAdmission(private / "admission.sqlite").request(deadline=_deadline(), generates=True):
            pytest.fail("writable ancestor admitted")
    assert not (private / "admission.sqlite").exists()


def test_symlink_in_directory_chain_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ModelAdmissionError):
        with SQLiteModelAdmission(alias / "nested" / "admission.sqlite").request(deadline=_deadline(), generates=True):
            pytest.fail("linked ancestor admitted")
    assert not (actual / "nested").exists()


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
@pytest.mark.parametrize("linked", ["symlink", "hardlink", "loose-mode"])
def test_database_and_sidecar_private_boundaries_are_checked_before_sqlite(db_path, tmp_path, suffix, linked):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=False):
        pass
    candidate = Path(str(db_path) + suffix)
    if suffix:
        candidate.touch(mode=0o600)
    if linked == "symlink":
        target = tmp_path / "synthetic-target"
        target.write_bytes(candidate.read_bytes())
        target.chmod(0o600)
        candidate.unlink()
        candidate.symlink_to(target)
    elif linked == "hardlink":
        os.link(candidate, tmp_path / "synthetic-link")
    else:
        candidate.chmod(0o660)
    with pytest.raises(ModelAdmissionError):
        with admission.request(deadline=_deadline(), generates=True):
            pytest.fail("unsafe database file admitted")


def test_foreign_owned_file_is_rejected(db_path, monkeypatch):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=False):
        pass
    inode = db_path.stat().st_ino
    original = os.fstat

    def foreign_metadata(descriptor):
        metadata = original(descriptor)
        if metadata.st_ino == inode:
            values = list(metadata)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(os, "fstat", foreign_metadata)
    with pytest.raises(ModelAdmissionError):
        with admission.request(deadline=_deadline(), generates=True):
            pytest.fail("foreign database admitted")


def test_directory_instead_of_database_is_rejected_without_sqlite_io(db_path, monkeypatch):
    db_path.parent.mkdir(mode=0o700)
    db_path.mkdir(mode=0o700)

    def forbidden(*args, **kwargs):
        pytest.fail("unsafe path reached SQLite")

    monkeypatch.setattr(sqlite3, "connect", forbidden)
    with pytest.raises(ModelAdmissionError):
        with SQLiteModelAdmission(db_path).request(deadline=_deadline(), generates=True):
            pytest.fail("directory admitted as database")

@pytest.mark.parametrize("replacement_mode", [None, 0o600, 0o660])
def test_disappearing_sidecar_is_rechecked_without_accepting_unsafe_replacement(db_path, monkeypatch, replacement_mode):
    admission = SQLiteModelAdmission(db_path)
    with admission.request(deadline=_deadline(), generates=False):
        pass
    journal = Path(str(db_path) + "-journal")
    journal.write_bytes(b"synthetic journal")
    journal.chmod(0o600)
    original_stat = os.stat
    intercepted = []
    def disappearing(path, *args, **kwargs):
        metadata = original_stat(path, *args, **kwargs)
        if path == journal.name and kwargs.get("dir_fd") is not None and not intercepted:
            intercepted.append(True)
            journal.unlink()
            if replacement_mode is not None:
                journal.write_bytes(b"synthetic replacement")
                journal.chmod(replacement_mode)
            fields = list(metadata)
            fields[3] = 0  # inode unlinked between path lookup and stat completion.
            return os.stat_result(fields)
        return metadata
    monkeypatch.setattr(os, "stat", disappearing)
    with admission._private_directory(create=False) as directory:
        if replacement_mode == 0o660:
            with pytest.raises(ModelAdmissionError, match="private, owned and not linked"):
                admission._check_files(directory)
        else:
            admission._check_files(directory)
    assert intercepted
