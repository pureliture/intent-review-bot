"""Private model-backed partition evaluation; not a publication approval."""
from __future__ import annotations

from dataclasses import dataclass, field

from .intent_models import IntentReviewLimits, IntentStageRecord


@dataclass(frozen=True)
class PartitionReviewLimits(IntentReviewLimits):
    max_input_tokens: int | None = None
    max_calls: int = 0
    input_target_tokens: int | None = None
    observation_output_tokens: int | None = None
    measure_only: bool = False
    # Each integration page has one bounded correction; run budgets remain shared.
    max_rechecks: int = 1


@dataclass(frozen=True)
class PartitionReviewResult:
    review_type: str
    status: str
    reason: str
    input_hash: str
    run_key: str
    snapshot_id: str
    base_sha: str
    head_sha: str
    created_at: str
    prompt_version: str
    prompt_hash: str
    model: dict
    scope: dict
    usage: dict
    model_delivery: dict
    stages: tuple[IntentStageRecord, ...]
    schema_version: int = 13
    language: str = "ko"
    summary: str = ""
    findings: tuple[dict, ...] = ()
    questions: tuple[str, ...] = ()
    perspectives: tuple[dict, ...] = ()
    partition: dict = field(default_factory=dict)
    validation_errors: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    origin: dict = field(default_factory=lambda: {
        "execution": "partitioned_mode_evaluation", "publication": "not_attempted"})
