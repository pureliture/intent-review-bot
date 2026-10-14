"""Two leased tasks, with durable waiting and per-PR exclusion across workers."""
from __future__ import annotations

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from review_bot.adapters.sqlite import store as store_module
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import ExecutionConfig, ModelConfig
from review_bot.domain.envelope_v2 import PrRequestRecord
from review_bot.domain.status import RunStatus


def _claim_in_process(path, name, results):
    store = SQLiteJobStore(path)
    try:
        run = store.claim_next_run(name, 60)
        results.put(None if run is None else run.run_id)
    finally:
        store.close()


def _seed(store):
    for pr_number in range(1, 4):
        store.enqueue_request(PrRequestRecord(44, 99, pr_number, 1))


def test_three_threads_claim_only_two_then_release_to_waiting_pr(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = SQLiteJobStore(path)
    _seed(store)
    barrier = threading.Barrier(3)

    def claim(index):
        local = SQLiteJobStore(path)
        try:
            barrier.wait(timeout=5)
            return local.claim_next_run(f"worker-{index}", 60)
        finally:
            local.close()

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            claimed = list(executor.map(claim, range(3)))
        active = [run for run in claimed if run is not None]
        assert len(active) == 2
        assert len({run.run_id for run in active}) == 2
        assert store.claim_next_run("fourth-worker", 60) is None
        assert store.update_run(active[0].run_id, RunStatus.SUCCEEDED)
        next_run = store.claim_next_run("fourth-worker", 60)
        assert next_run is not None
        assert next_run.pr_number not in {run.pr_number for run in active}
    finally:
        store.close()


def test_task_limit_is_shared_by_independent_processes(tmp_path):
    path = tmp_path / "jobs.sqlite"
    store = SQLiteJobStore(path)
    _seed(store)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    processes = [context.Process(target=_claim_in_process, args=(path, f"child-{i}", results)) for i in range(3)]
    try:
        for process in processes:
            process.start()
        claimed = [results.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert len([run for run in claimed if run is not None]) == 2
        assert store.claim_next_run("parent", 60) is None
    finally:
        for process in processes:
            process.join(timeout=1)
        results.close()
        results.join_thread()
        store.close()


def test_new_generation_of_same_pr_waits_for_current_owner(tmp_path):
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    try:
        store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        old = store.claim_next_run("old-owner", 60)
        store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        store.enqueue_request(PrRequestRecord(44, 99, 8, 1))
        unrelated = store.claim_next_run("new-owner", 60)
        assert unrelated.pr_number == 8
        assert not store.check_generation_fencing(old.run_id, worker_id="old-owner", generation=old.generation)
        assert store.update_run(old.run_id, RunStatus.SUPERSEDED)
        current = store.claim_next_run("new-owner-2", 60)
        assert current.pr_number == 7
        assert current.generation == old.generation + 1
    finally:
        store.close()


def test_expired_task_recovers_without_giving_old_owner_write_rights(tmp_path, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(store_module.time, "time", lambda: now[0])
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    try:
        _seed(store)
        first = store.claim_next_run("old", 30)
        second = store.claim_next_run("other", 30)
        assert first and second and store.claim_next_run("waiting", 30) is None
        now[0] = 1031.0
        recovered = store.claim_next_run("replacement", 30)
        assert recovered.run_id == first.run_id
        assert recovered.attempts == first.attempts + 1
        assert not store.renew_lease(first.run_id, "old", 30)
        assert not store.update_run(first.run_id, RunStatus.SUCCEEDED, worker_id="old", generation=first.generation)
    finally:
        store.close()


@pytest.mark.parametrize("value", [0, 3, True, 1.5])
def test_both_configured_limits_reject_values_outside_one_or_two(value):
    with pytest.raises(ValueError):
        ExecutionConfig(available_modes=["intent"], worker_concurrency=value)
    with pytest.raises(ValueError):
        ModelConfig(max_concurrency=value)


@pytest.mark.parametrize("value", [1, 2])
def test_both_configured_limits_accept_one_or_two(value):
    assert ExecutionConfig(available_modes=["intent"], worker_concurrency=value).worker_concurrency == value
    assert ModelConfig(max_concurrency=value).max_concurrency == value


def test_late_selector_context_cannot_mutate_the_next_run(tmp_path, monkeypatch, valid_config_yaml):
    from review_bot.config import load_review_config_from_str
    from review_bot.runtime import worker as worker_module

    monkeypatch.setattr(worker_module.WorkerLoop, "_setup_signals", lambda self: None)
    config, _ = load_review_config_from_str(valid_config_yaml)
    store = SQLiteJobStore(tmp_path / "jobs.sqlite")
    callbacks = []
    original = worker_module.set_model_request_context

    def capture(cancelled, deadline):
        callbacks.append(cancelled)
        return original(cancelled, deadline)

    def fail_before_collection(*args, **kwargs):
        raise RuntimeError("controlled prepare failure")

    monkeypatch.setattr(worker_module, "set_model_request_context", capture)
    worker = worker_module.WorkerLoop(config, store, worker_id="worker")
    monkeypatch.setattr(worker, "_stage", fail_before_collection)
    try:
        store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        run = store.claim_next_run("worker", 60)
        assert worker.process_run(run) is False
        assert len(callbacks) == 1
        worker._current_run_id = "next-run"
        worker._lease_lost = False
        assert callbacks[0]() is True
        assert worker._lease_lost is False
    finally:
        store.close()