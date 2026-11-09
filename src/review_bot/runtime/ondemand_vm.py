"""One-shot VM side of the on-demand contract: claim, heartbeat, run, persist trace, complete, terminate."""
from __future__ import annotations

import threading
from collections.abc import Callable

from review_bot.application.ondemand import OnDemandCoordinator, ReviewJob, StaleLease

# execute(job, is_current) -> (mode_results, trace). ``is_current`` must gate every external write.
Execute = Callable[[ReviewJob, Callable[[], bool]], tuple[dict[str, str], dict]]


def run_job(coordinator: OnDemandCoordinator, job_id: str, worker_id: str, execute: Execute, *,
    heartbeat_seconds: float | None = None) -> ReviewJob | None:
    job = coordinator.claim(job_id, worker_id)
    if job is None:
        return None
    lost = threading.Event()
    stop = threading.Event()
    interval = heartbeat_seconds if heartbeat_seconds is not None else max(1.0, coordinator.lease_seconds / 3)

    def beat() -> None:
        while not stop.wait(interval):
            try:
                coordinator.heartbeat(job_id, worker_id)
            except StaleLease:
                lost.set()
                return

    thread = threading.Thread(target=beat, name=f"lease-{job_id[:8]}", daemon=True)
    thread.start()
    finished = None
    try:
        def is_current() -> bool:
            return not lost.is_set() and coordinator.is_current(job_id, worker_id)

        try:
            mode_results, trace = execute(job, is_current)
        except Exception as exc:
            mode_results, trace = {"review": "failed"}, {"error": type(exc).__name__}
        stop.set()
        thread.join(timeout=interval + 1)
        try:
            coordinator.record_trace(job_id, worker_id, trace)
            finished = coordinator.complete(job_id, worker_id, mode_results)
        except StaleLease:
            finished = None
        return finished
    finally:
        stop.set()
        current = coordinator.store.get(job_id)
        if current is not None and current.vm_id:
            coordinator.release_vm(current)
