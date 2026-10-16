"""Service Envelope v2 records for PR reviews, mode selections, and receipts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from .models_v1 import ReviewResult
from .status import (
    ComparisonStatusKind,
    ExecutionStatusKind,
    PublicationStatusKind,
    SelectionStatusKind,
)

if TYPE_CHECKING:
    from .intent_models import ReviewBResult
    from .code_first_intent_models import CodeFirstReviewResult


@dataclass(frozen=True)
class ChangedFile:
    path: str
    previous_path: str | None = None
    change_type: str = "modified"
    has_diff: bool = True


@dataclass(frozen=True)
class UnavailableSource:
    source: str
    reason: str


@dataclass(frozen=True)
class EnvelopeEvidence:
    id: str
    kind: (
        Literal[
            "pr_text",
            "issue_text",
            "source",
            "test_source",
            "test_run",
            "static_signal",
            "architecture_rule",
        ]
        | str
    )
    version: str
    locator: dict[str, object]
    content: str | dict[str, object]
    digest: str


@dataclass(frozen=True)
class ReviewSnapshot:
    schema_version: int
    snapshot_id: str
    repository: str
    pr_number: int
    repository_id: int
    installation_id: int
    target_base_sha: str
    head_sha: str
    comparison_base_sha: str | None
    comparison_status: ComparisonStatusKind | str
    captured_at: str
    as_of: str
    pr_text_digest: str
    context_digest: str
    changed_files: tuple[ChangedFile, ...] = ()
    evidence: tuple[EnvelopeEvidence, ...] = ()
    unavailable_sources: tuple[UnavailableSource, ...] = ()


@dataclass(frozen=True)
class ModeSelection:
    selection_id: str
    snapshot_id: str
    policy_version: str
    selection_digest: str
    available_modes: tuple[str, ...]
    selected_modes: tuple[str, ...]
    selection_status: SelectionStatusKind | str = "completed"


@dataclass(frozen=True)
class PublicationReceipt:
    comment_id: int | None
    status: PublicationStatusKind | str
    fingerprint: str
    marker: str = "<!-- review-bot:architecture:v1 -->"
    published_at: str | None = None
    reason: str = ""
    input_fingerprint: str | None = None


@dataclass(frozen=True)
class ExecutionReceipt:
    schema_version: int
    mode: str
    selection_id: str
    snapshot_id: str
    generation: int
    fingerprint: str
    execution_status: ExecutionStatusKind | str
    a_result: ReviewResult | None = None
    coverage: dict[str, object] = field(default_factory=dict)
    model_delivery: dict[str, object] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    error_code: str | None = None
    publication: PublicationReceipt | None = None
    b_result: ReviewBResult | CodeFirstReviewResult | None = None
    usage: dict[str, object] | None = None
    input_fingerprint: str | None = None
    reused: bool = False


# Aliased for contract alignment
ServiceReviewResult = ExecutionReceipt


@dataclass(frozen=True)
class DeliveryRecord:
    delivery_id: str
    payload_digest: str
    received_at: str
    action: str
    installation_id: int
    repository_id: int
    pr_number: int
    processed: bool = False


@dataclass(frozen=True)
class PrRequestRecord:
    installation_id: int
    repository_id: int
    pr_number: int
    desired_generation: int


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    installation_id: int
    repository_id: int
    pr_number: int
    generation: int
    fingerprint: str
    status: str
    stage: str
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    reason: str | None = None
