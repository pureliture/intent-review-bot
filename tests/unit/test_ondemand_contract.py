"""Lambda→one-shot VM job contract: idempotent intake, lease, retry, recovery, trace durability."""
from __future__ import annotations

import pytest

from review_bot.application.ondemand import (
    InMemoryJobStore, IntakeEvent, OnDemandCoordinator, StaleLease, job_key,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Launcher:
    def __init__(self) -> None:
        self.launched: list[str] = []
        self.terminated: list[str] = []
        self.fail = False

    def launch(self, job_id: str, *, max_lifetime_seconds: int, launch_token: str) -> str:
        assert max_lifetime_seconds > 0 and launch_token.startswith(job_id[:40])
        if self.fail:
            raise RuntimeError("capacity")
        self.launched.append(job_id)
        return f"i-{len(self.launched)}"

    def terminate(self, vm_id: str) -> None:
        self.terminated.append(vm_id)


def event(delivery="d1", action="synchronize", head="b" * 40, base="a" * 40, digest="x"):
    return IntakeEvent(delivery, digest, action, 1, 2, 3, base, head)


@pytest.fixture
def env():
    clock, store, launcher = Clock(), InMemoryJobStore(), Launcher()
    return clock, store, launcher, OnDemandCoordinator(store, launcher, clock=clock, max_attempts=2, lease_seconds=60)


def test_duplicate_delivery_and_same_head_create_one_job_and_one_vm(env):
    _, store, launcher, c = env
    first = c.accept(event())
    assert first.outcome == "accepted" and first.launch
    assert c.launch(first.job_id).vm_id == "i-1"
    replay = c.accept(event())
    other_delivery = c.accept(event(delivery="d2"))
    assert (replay.outcome, replay.launch) == ("duplicate_delivery", False)
    assert (other_delivery.outcome, other_delivery.launch) == ("duplicate_job", False)
    assert c.launch(first.job_id) is None
    assert len(store.jobs) == 1 and launcher.launched == [first.job_id]


def test_replayed_delivery_id_with_different_payload_is_rejected(env):
    _, store, _, c = env
    c.accept(event())
    assert c.accept(event(digest="tampered", head="c" * 40)).outcome == "replay_mismatch"
    assert len(store.jobs) == 1


def test_new_head_supersedes_running_job_and_fences_its_publication(env):
    _, store, _, c = env
    old = c.accept(event()).job_id
    c.launch(old)
    assert c.claim(old, "vm-old")
    new = c.accept(event(delivery="d2", head="c" * 40))
    assert new.superseded == (old,) and store.get(old).state == "superseded"
    assert not c.is_current(old, "vm-old")
    with pytest.raises(StaleLease):
        c.record_trace(old, "vm-old", {"stages": []})


def test_closed_pr_cancels_active_jobs_and_ignores_unrelated_actions(env):
    _, store, _, c = env
    job = c.accept(event()).job_id
    assert c.accept(event(delivery="d2", action="assigned")).outcome == "ignored"
    closed = c.accept(event(delivery="d3", action="closed"))
    assert closed.outcome == "cancelled" and store.get(job).state == "cancelled"
    assert c.claim(job, "vm") is None


def test_single_owner_lease_and_heartbeat(env):
    clock, _, _, c = env
    job = c.accept(event()).job_id
    c.launch(job)
    assert c.claim(job, "vm-a").attempts == 1
    assert c.claim(job, "vm-b") is None
    clock.now += 50
    c.heartbeat(job, "vm-a")
    clock.now += 50
    assert c.claim(job, "vm-b") is None, "renewed lease must not be stolen"
    with pytest.raises(StaleLease):
        c.heartbeat(job, "vm-b")


def test_completion_requires_persisted_trace_and_then_terminates_vm(env):
    _, store, launcher, c = env
    job = c.accept(event()).job_id
    c.launch(job)
    c.claim(job, "vm")
    with pytest.raises(ValueError, match="trace_not_persisted"):
        c.complete(job, "vm", {"architecture": "completed", "intent": "completed"})
    c.record_trace(job, "vm", {"stages": [{"stage": "final_review", "calls": 1}]})
    done = c.complete(job, "vm", {"architecture": "completed", "intent": "partial_review"})
    c.release_vm(done)
    stored = store.get(job)
    assert stored.state == "partial" and stored.mode_results == {"architecture": "completed", "intent": "partial_review"}
    assert stored.trace["stages"][0]["stage"] == "final_review", "trace survives VM termination"
    assert launcher.terminated == ["i-1"]


@pytest.mark.parametrize(("results", "state"), [
    ({"architecture": "completed", "intent": "completed"}, "succeeded"),
    ({"architecture": "failed", "intent": "completed"}, "partial"),
    ({"architecture": "publication_failed", "intent": "failed"}, "failed"),
    ({"architecture": "not_selected", "intent": "completed"}, "succeeded"),
])
def test_ab_results_stay_independent_in_job_state(env, results, state):
    _, _, _, c = env
    job = c.accept(event()).job_id
    c.claim(job, "vm")
    c.record_trace(job, "vm", {})
    assert c.complete(job, "vm", results).state == state


def test_vm_loss_requeues_then_dead_letters_after_max_attempts(env):
    clock, store, launcher, c = env
    job = c.accept(event()).job_id
    c.launch(job)
    c.claim(job, "vm-1")
    clock.now += 61
    assert c.recover() == [(job, "requeued")]
    assert store.get(job).state == "queued" and launcher.terminated == ["i-1"]
    assert c.launch(job).vm_id == "i-2"
    assert c.claim(job, "vm-2").attempts == 2
    clock.now += 61
    assert c.recover() == [(job, "dead_letter")]
    assert store.get(job).state == "dead_letter"
    assert c.claim(job, "vm-3") is None


def test_expired_lease_owner_cannot_complete_after_takeover(env):
    clock, _, _, c = env
    job = c.accept(event()).job_id
    c.claim(job, "vm-1")
    clock.now += 61
    assert c.claim(job, "vm-2").lease_owner == "vm-2"
    with pytest.raises(StaleLease):
        c.record_trace(job, "vm-1", {})


def test_launch_failure_leaves_job_queued_for_retry(env):
    _, store, launcher, c = env
    job = c.accept(event()).job_id
    launcher.fail = True
    assert c.launch(job) is None
    assert store.get(job).state == "queued" and store.get(job).reason.startswith("launch_failed")
    launcher.fail = False
    retry = c.accept(event(delivery="d2"))
    assert retry.launch, "a redelivery may relaunch a queued job without a VM"
    assert c.launch(job).vm_id == "i-1"


def test_stuck_launch_is_recovered(env):
    clock, store, _, c = env
    job = c.accept(event()).job_id
    c.launch(job)
    clock.now += 601
    assert c.recover() == [(job, "requeued")]
    assert store.get(job).reason == "launch_timeout"


def test_job_key_binds_base_and_head():
    assert job_key(1, 2, 3, "a", "b") != job_key(1, 2, 3, "a", "c") != job_key(1, 2, 3, "z", "b")
