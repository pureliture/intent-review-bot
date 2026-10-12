"""Abstract Protocols and port types for github-review-bot Hexagonal Architecture."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from review_bot.domain.envelope_v2 import (
    ChangedFile,
    DeliveryRecord,
    ModeSelection,
    PrRequestRecord,
    PublicationReceipt,
    ReviewSnapshot,
    RunRecord,
)
from review_bot.domain.models_v1 import (
    AnalysisRun,
    CodeSnapshot,
    ModelRequest,
    ModelResponse,
    PrSnapshot,
)
from review_bot.domain.status import (
    ComparisonStatusKind,
    RunStageKind,
    RunStatusKind,
)


@dataclass(frozen=True)
class MergeBaseResult:
    """Outcome of merge-base calculation separating target base and comparison base."""

    target_base_sha: str
    comparison_base_sha: str | None
    comparison_status: ComparisonStatusKind | str  # 'comparable', 'unavailable', 'ambiguous'
    candidates: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """Alias for comparison_status for backwards compatibility with tests."""
        return str(self.comparison_status)

    @property
    def is_comparable(self) -> bool:
        """True if and only if a unique comparison base commit was identified."""
        return self.comparison_status == "comparable" and self.comparison_base_sha is not None


@dataclass(frozen=True)
class Commit:
    """Git commit object representation."""

    sha: str
    tree_sha: str
    parents: tuple[str, ...]
    author: str
    committed_at: str
    message: str


@runtime_checkable
class RepositoryReader(Protocol):
    """Port for interacting with a Git repository directly via safe object reads."""

    def get_commit(self, sha: str) -> Commit:
        """Retrieve commit details by revision or SHA."""
        ...

    def read_tree_blobs(
        self, commit_sha: str, paths: tuple[str, ...] | list[str] | None = None
    ) -> dict[str, bytes]:
        """Read tracked blobs for a commit tree. If paths is specified, reads only those paths."""
        ...

    def compute_merge_base(
        self, base_sha: str, head_sha: str, all_bases: bool = False
    ) -> tuple[str, ...]:
        """Compute common ancestor commit SHA(s) using git merge-base."""
        ...

    def resolve_merge_base(
        self, target_base_sha: str, head_sha: str
    ) -> MergeBaseResult:
        """Determine three-dot comparison base separating target_base_sha from comparison_base_sha."""
        ...

    def get_changed_files(
        self, base_sha: str, head_sha: str
    ) -> tuple[ChangedFile, ...]:
        """Detect file additions, modifications, deletions, and renames between revisions."""
        ...

    def get_rename_map(
        self, base_sha: str, head_sha: str
    ) -> dict[str, str]:
        """Extract rename mappings {previous_path: current_path}."""
        ...

    def collect_code_snapshot(
        self, sha: str
    ) -> CodeSnapshot:
        """Gather full code inventory (source, test, build) at sha into immutable CodeSnapshot."""
        ...


@runtime_checkable
class StaticAnalyzer(Protocol):
    """Port for bot-owned static analysis execution (PMD 7.3.0 & CPD 100 tokens)."""

    def run_analysis(
        self,
        commit_sha: str,
        source_root: Path,
        ruleset_resource: str,
        profile: str = "java21-jdk-only-v1",
    ) -> AnalysisRun:
        """Execute static analyzer against isolated source tree and parse strict XML reports."""
        ...


@runtime_checkable
class ModeSelectorPort(Protocol):
    """Port for sub-second review mode selection from PR metadata and changed paths."""

    def select_modes(
        self,
        snapshot: ReviewSnapshot,
        pr: PrSnapshot,
        available_modes: Sequence[str],
    ) -> ModeSelection:
        """Select registered modes; PR title/body are carried in snapshot PR-text evidence."""
        ...


@runtime_checkable
class ModelClient(Protocol):
    """Provider-neutral LLM invocation port with strict JSON response handling."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send formatted prompt to model client and return response text with usage statistics."""
        ...


@runtime_checkable
class JobStore(Protocol):
    """Port for durable job queue, webhook deduplication, PR generation coalescing, and leasing."""

    def accept_delivery(self, delivery: DeliveryRecord, request: PrRequestRecord) -> None:
        """Atomically preserve a webhook delivery and its runnable request."""
        ...

    def accept_cancellation(self, delivery: DeliveryRecord) -> None:
        """Atomically record the delivery and cancel this installation's PR work."""
        ...

    def accept_revocation(self, delivery: DeliveryRecord, repository_ids: tuple[int, ...] | None = None) -> None:
        """Atomically stop this installation's work for all or removed repositories."""
        ...

    def record_delivery(self, delivery: DeliveryRecord) -> None:
        """Record raw webhook delivery idempotently. Duplicate delivery_id raises error."""
        ...

    def is_delivery_processed(self, delivery_id: str) -> bool:
        """Check whether the delivery has already been preserved."""
        ...

    def enqueue_request(self, request: PrRequestRecord) -> None:
        """Upsert PR request and coalesce desired_generation."""
        ...

    def claim_next_run(self, worker_id: str, lease_seconds: float) -> RunRecord | None:
        """Atomically claim the next runnable execution with an expiring lease."""
        ...

    def renew_lease(self, run_id: str, worker_id: str, lease_seconds: float) -> bool:
        """Extend active worker lease on a claimed run."""
        ...

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
        """Update lifecycle state; worker calls require a live lease and current generation.

        Return False on a rejected fence. Omitted worker_id preserves administrative updates.
        """
        ...

    def record_publication(
        self, receipt: PublicationReceipt, repository_id: int, pr_number: int,
        *,
        run_id: str | None = None,
        worker_id: str | None = None,
        generation: int | None = None,
    ) -> bool:
        """Persist a receipt atomically fenced to the worker's live run; False on rejection."""
        ...

    def get_current_publication(
        self, run: RunRecord, marker: str, *, input_fingerprint: str | None = None,
    ) -> PublicationReceipt | None:
        """Get successful mode output, or legacy whole-run output, in the same installation."""
        ...

    def get_pr_request(
        self, repository_id: int, pr_number: int
    ) -> PrRequestRecord | None:
        """Fetch current desired_generation and record for a repository PR."""
        ...

    def check_generation_fencing(
        self, run_id: str, *, worker_id: str | None = None, generation: int | None = None,
    ) -> bool:
        """Check current generation and supersede stale runs only under the supplied worker fence."""
        ...


class GitHubAuthorizationError(RuntimeError):
    """Authorization loss is common failure, not a mode-local API response error."""


@runtime_checkable
class GitHubClient(Protocol):
    """Port for interacting with the GitHub API (PR retrieval and comment management)."""

    def get_pull_request(
        self, pr_number: int, *, max_response_bytes: int | None = None,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Fetch PR state, head SHA, base SHA, branch name, and label names."""
        ...

    def get_issue(
        self, issue_number: int, *, max_response_bytes: int = 524288,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Read one issue within this client's bound repository, with bounded bytes/time."""
        ...

    def find_marker_comment(
        self, pr_number: int, marker: str = "<!-- review-bot:architecture:v1 -->", *,
        before_request: Callable[[], bool] | None = None, timeout_seconds: float | None = None,
        max_response_bytes: int | None = None,
    ) -> Any | None:
        """Find existing bot review comment containing the designated marker."""
        ...

    def post_comment(self, pr_number: int, body: str) -> Any:
        """Post new review comment."""
        ...

    def update_comment(self, comment_id: int, body: str) -> Any:
        """Update existing review comment body."""
        ...


@runtime_checkable
class ExecutionIsolation(Protocol):
    """Port for running commands under rootless container or subprocess isolation."""

    def run_command(
        self,
        command: tuple[str, ...],
        cwd: Path,
        timeout_seconds: int = 120,
        read_only_root: bool = True,
        network: bool = False,
    ) -> tuple[int, bytes, bytes]:
        """Execute command in isolated environment, returning (exit_code, stdout, stderr)."""
        ...
