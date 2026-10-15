"""Code-first B v2 records, independent of declared-goal B v1 results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .models_b import IntentReviewLimits, IntentStageRecord


@dataclass(frozen=True)
class CodeFirstReviewLimits(IntentReviewLimits):
    max_calls: int = 12
    max_groups: int = 8
    max_rechecks: int = 1


@dataclass(frozen=True)
class CodeFirstReviewResult:
    schema_version: int
    review_type: Literal["B"]
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
    prompt_hashes: dict[str, str]
    model: dict[str, object]
    scope: dict[str, object]
    opinion_scope: dict[str, object]
    usage: dict[str, object]
    model_delivery: dict[str, object]
    stages: tuple[IntentStageRecord, ...]
    summary: str = ""
    findings: tuple[dict, ...] = ()
    questions: tuple[dict, ...] = ()
    validation_errors: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    origin: dict[str, object] = field(default_factory=dict)
    plan: dict | None = None
    group_observations: tuple[dict, ...] = ()
    reconstructed_intents: tuple[dict, ...] = ()
    relations: tuple[dict, ...] = ()