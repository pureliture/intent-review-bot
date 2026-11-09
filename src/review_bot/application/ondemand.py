"""Cloud-neutral contract for handing a PR review from serverless intake to a one-shot VM.

Intake (e.g. AWS Lambda) verifies the webhook, records the delivery, creates at most one
job per (installation, repository, PR, base, head) and asks a launcher for a VM. The VM
claims the job under a lease, renews it, persists its trace, completes and terminates.
A scheduled recovery pass re-queues jobs whose VM vanished. Every transition is a
conditional write on ``version`` so no shared SQLite or single host is assumed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from typing import Protocol

ACTIVE_STATES = ("queued", "launching", "running")
FINAL_STATES = ("succeeded", "partial", "failed", "superseded", "cancelled", "dead_letter")
REVIEW_ACTIONS = ("opened", "synchronize", "reopened", "edited", "labeled", "unlabeled", "ready_for_review")


def job_key(installation_id: int, repository_id: int, pr_number: int, base_sha: str, head_sha: str) -> str:
    material = f"v1:{installation_id}:{repository_id}:{pr_number}:{base_sha}:{head_sha}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IntakeEvent:
    delivery_id: str
    payload_digest: str
    action: str
    installation_id: int
    repository_id: int
    pr_number: int
    base_sha: str
    head_sha: str


@dataclass(frozen=True)
class ReviewJob:
    job_id: str
    installation_id: int
    repository_id: int
    pr_number: int
    base_sha: str
    head_sha: str
    state: str
    created_at: float
    updated_at: float
    version: int = 0
    attempts: int = 0
    max_attempts: int = 3
    lease_owner: str | None = None
    lease_expires_at: float | None = None
    vm_id: str | None = None
    reason: str = ""
    delivery_id: str = ""
    trace: dict | None = None
    mode_results: dict = field(default_factory=dict)


@dataclass(frozen=True)
class IntakeDecision:
    outcome: str
    job_id: str | None = None
    launch: bool = False
    superseded: tuple[str, ...] = ()


class JobStorePort(Protocol):
    """Durable store with conditional writes (DynamoDB, Firestore, Cloud SQL, ...)."""

    def record_delivery(self, delivery_id: str, payload_digest: str, received_at: float) -> str | None:
        """Insert if absent. Returns None when new, else the previously stored digest."""

    def get(self, job_id: str) -> ReviewJob | None: ...

    def put_new(self, job: ReviewJob) -> bool:
        """Insert only if ``job_id`` is absent."""

    def replace_if_version(self, job: ReviewJob, expected_version: int) -> bool:
        """Write ``job`` (with version expected_version+1) only if the stored version matches."""

    def list_for_pr(self, repository_id: int, pr_number: int) -> list[ReviewJob]: ...

    def list_active(self) -> list[ReviewJob]: ...


class VmLauncherPort(Protocol):
    def launch(self, job_id: str, *, max_lifetime_seconds: int) -> str:
        """Start a one-shot VM for ``job_id``; returns the provider instance id."""

    def terminate(self, vm_id: str) -> None: ...


class StaleLease(RuntimeError):
    pass


class OnDemandCoordinator:
    def __init__(self, store: JobStorePort, launcher: VmLauncherPort, *, clock, max_attempts: int = 3,
        lease_seconds: int = 300, launch_timeout_seconds: int = 600, max_vm_lifetime_seconds: int = 3600) -> None:
        self.store, self.launcher, self.clock = store, launcher, clock
        self.max_attempts, self.lease_seconds = max_attempts, lease_seconds
        self.launch_timeout_seconds, self.max_vm_lifetime_seconds = launch_timeout_seconds, max_vm_lifetime_seconds

    def _cas(self, job: ReviewJob, **changes) -> ReviewJob | None:
        updated = replace(job, **changes, version=job.version + 1, updated_at=self.clock())
        return updated if self.store.replace_if_version(updated, job.version) else None

    def _settle(self, job_id: str, change, attempts: int = 8) -> ReviewJob | None:
        """Retry a read-modify-CAS until it applies or ``change`` declines (returns None)."""
        for _ in range(attempts):
            job = self.store.get(job_id)
            if job is None:
                return None
            changes = change(job)
            if changes is None:
                return None
            updated = self._cas(job, **changes)
            if updated is not None:
                return updated
        raise RuntimeError("job_update_contention")

    # Intake (serverless, idempotent; safe to retry on the same delivery)
    def accept(self, event: IntakeEvent) -> IntakeDecision:
        now = self.clock()
        previous = self.store.record_delivery(event.delivery_id, event.payload_digest, now)
        if previous is not None and previous != event.payload_digest:
            return IntakeDecision("replay_mismatch")
        if event.action == "closed":
            cancelled = self._finish_pr_jobs(event, "cancelled", "pr_closed", keep=None)
            return IntakeDecision("cancelled", superseded=cancelled)
        if event.action not in REVIEW_ACTIONS:
            return IntakeDecision("ignored")
        job_id = job_key(event.installation_id, event.repository_id, event.pr_number, event.base_sha, event.head_sha)
        superseded = self._finish_pr_jobs(event, "superseded", "newer_head_or_base", keep=job_id)
        job = ReviewJob(job_id, event.installation_id, event.repository_id, event.pr_number, event.base_sha,
            event.head_sha, "queued", now, now, max_attempts=self.max_attempts, delivery_id=event.delivery_id)
        if self.store.put_new(job):
            return IntakeDecision("accepted", job_id, launch=True, superseded=superseded)
        existing = self.store.get(job_id)
        relaunch = existing is not None and existing.state == "queued" and existing.vm_id is None
        return IntakeDecision("duplicate_delivery" if previous is not None else "duplicate_job", job_id,
            launch=relaunch, superseded=superseded)

    def _finish_pr_jobs(self, event: IntakeEvent, state: str, reason: str, keep: str | None) -> tuple[str, ...]:
        changed = []
        for job in self.store.list_for_pr(event.repository_id, event.pr_number):
            if job.job_id == keep or job.state not in ACTIVE_STATES:
                continue
            if self._settle(job.job_id, lambda j: {"state": state, "reason": reason} if j.state in ACTIVE_STATES else None):
                changed.append(job.job_id)
        return tuple(changed)

    def launch(self, job_id: str) -> ReviewJob | None:
        job = self._settle(job_id, lambda j: {"state": "launching", "reason": "vm_requested"}
            if j.state == "queued" and j.vm_id is None else None)
        if job is None:
            return None
        try:
            vm_id = self.launcher.launch(job_id, max_lifetime_seconds=self.max_vm_lifetime_seconds)
        except Exception as exc:
            self._settle(job_id, lambda j: {"state": "queued", "reason": f"launch_failed:{type(exc).__name__}"}
                if j.state == "launching" and j.vm_id is None else None)
            return None
        return self._settle(job_id, lambda j: {"vm_id": vm_id} if j.state in ACTIVE_STATES and j.vm_id is None else None)

    # VM side
    def claim(self, job_id: str, worker_id: str) -> ReviewJob | None:
        now = self.clock()

        def change(j: ReviewJob):
            claimable = j.state in ("queued", "launching") or (
                j.state == "running" and j.lease_expires_at is not None and j.lease_expires_at <= now)
            if not claimable:
                return None
            if j.attempts >= j.max_attempts:
                return {"state": "dead_letter", "reason": "max_attempts_exceeded", "lease_owner": None}
            return {"state": "running", "attempts": j.attempts + 1, "lease_owner": worker_id,
                "lease_expires_at": now + self.lease_seconds, "reason": "claimed"}

        job = self._settle(job_id, change)
        return job if job is not None and job.state == "running" and job.lease_owner == worker_id else None

    def _owned(self, job_id: str, worker_id: str, change) -> ReviewJob:
        now = self.clock()

        def guarded(j: ReviewJob):
            if j.state != "running" or j.lease_owner != worker_id or (j.lease_expires_at or 0) <= now:
                return None
            return change(j)

        job = self._settle(job_id, guarded)
        if job is None:
            raise StaleLease(job_id)
        return job

    def heartbeat(self, job_id: str, worker_id: str) -> ReviewJob:
        return self._owned(job_id, worker_id, lambda j: {"lease_expires_at": self.clock() + self.lease_seconds})

    def is_current(self, job_id: str, worker_id: str) -> bool:
        job = self.store.get(job_id)
        return (job is not None and job.state == "running" and job.lease_owner == worker_id
            and (job.lease_expires_at or 0) > self.clock())

    def record_trace(self, job_id: str, worker_id: str, trace: dict) -> ReviewJob:
        """Persist the execution trace before publication and before the VM is torn down."""
        return self._owned(job_id, worker_id, lambda j: {"trace": dict(trace)})

    def complete(self, job_id: str, worker_id: str, mode_results: dict[str, str]) -> ReviewJob:
        if not mode_results or any(v not in ("completed", "partial_review", "failed", "publication_failed", "not_selected")
                for v in mode_results.values()):
            raise ValueError("invalid_mode_results")
        outcomes = set(mode_results.values()) - {"not_selected"}
        state = ("succeeded" if outcomes == {"completed"} else
            "failed" if outcomes and outcomes <= {"failed", "publication_failed"} else "partial")

        def change(j: ReviewJob):
            if j.trace is None:
                raise ValueError("trace_not_persisted")
            return {"state": state, "mode_results": dict(mode_results), "lease_owner": None,
                "lease_expires_at": None, "reason": "completed"}

        return self._owned(job_id, worker_id, change)

    def release_vm(self, job: ReviewJob) -> None:
        if job.vm_id:
            self.launcher.terminate(job.vm_id)

    # Scheduled recovery (serverless)
    def recover(self) -> list[tuple[str, str]]:
        """Re-queue jobs whose VM vanished; dead-letter after ``max_attempts``."""
        now = self.clock()
        actions = []
        for job in self.store.list_active():
            lost_vm = job.state == "running" and (job.lease_expires_at or 0) <= now
            stuck_launch = job.state == "launching" and job.updated_at + self.launch_timeout_seconds <= now
            if not (lost_vm or stuck_launch):
                continue
            exhausted = job.attempts >= job.max_attempts
            changes = ({"state": "dead_letter", "reason": "max_attempts_exceeded", "lease_owner": None}
                if exhausted else {"state": "queued", "vm_id": None, "lease_owner": None, "lease_expires_at": None,
                    "reason": "vm_lost" if lost_vm else "launch_timeout"})
            if self._cas(job, **changes) is None:
                continue
            if job.vm_id:
                try:
                    self.launcher.terminate(job.vm_id)
                except Exception:
                    pass
            actions.append((job.job_id, "dead_letter" if exhausted else "requeued"))
        return actions


class InMemoryJobStore:
    """Reference JobStorePort with the same conditional-write semantics as the cloud adapters."""

    def __init__(self) -> None:
        self.jobs: dict[str, ReviewJob] = {}
        self.deliveries: dict[str, str] = {}

    def record_delivery(self, delivery_id: str, payload_digest: str, received_at: float) -> str | None:
        if delivery_id in self.deliveries:
            return self.deliveries[delivery_id]
        self.deliveries[delivery_id] = payload_digest
        return None

    def get(self, job_id: str) -> ReviewJob | None:
        return self.jobs.get(job_id)

    def put_new(self, job: ReviewJob) -> bool:
        if job.job_id in self.jobs:
            return False
        self.jobs[job.job_id] = job
        return True

    def replace_if_version(self, job: ReviewJob, expected_version: int) -> bool:
        current = self.jobs.get(job.job_id)
        if current is None or current.version != expected_version or job.version != expected_version + 1:
            return False
        self.jobs[job.job_id] = job
        return True

    def list_for_pr(self, repository_id: int, pr_number: int) -> list[ReviewJob]:
        return [j for j in self.jobs.values() if j.repository_id == repository_id and j.pr_number == pr_number]

    def list_active(self) -> list[ReviewJob]:
        return [j for j in self.jobs.values() if j.state in ACTIVE_STATES]
