"""Independent B v1 records; no A analysis or publication is required."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .envelope_v2 import ReviewSnapshot
from .models_v1 import CodeSnapshot


@dataclass(frozen=True)
class IntentTaskInput:
    schema_version: int
    snapshot: ReviewSnapshot
    base: CodeSnapshot
    head: CodeSnapshot
    reviewed_paths: tuple[str, ...]
    input_hash: str


@dataclass(frozen=True)
class DeclaredGoal:
    goal_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class DeclaredConstraint:
    constraint_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    quote: str


@dataclass(frozen=True)
class IntentGoals:
    declared_goals: tuple[DeclaredGoal, ...]
    declared_constraints: tuple[DeclaredConstraint, ...]
    unavailable_context: tuple[str, ...]


@dataclass(frozen=True)
class ImplementationObservation:
    observation_id: str
    statement: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class IntentHypothesis:
    hypothesis_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True)
class IntentAssumption:
    assumption_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True)
class IntentObservations:
    implementation_observations: tuple[ImplementationObservation, ...]
    intent_hypotheses: tuple[IntentHypothesis, ...]
    assumptions: tuple[IntentAssumption, ...]


@dataclass(frozen=True)
class IntentFinding:
    finding_id: str
    goal_ids: tuple[str, ...]
    constraint_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    statement: str
    assessment: Literal["consistent_with_evidence", "evidence_conflict", "insufficient_context"] | str
    reasoning: str
    evidence_ids: tuple[str, ...]
    missing_evidence: tuple[str, ...]


@dataclass(frozen=True)
class IntentQuestion:
    question_id: str
    question: str
    related_finding_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    why_needed: str
    how_answer_changes_assessment: str


@dataclass(frozen=True)
class IntentAssessment:
    summary: str
    findings: tuple[IntentFinding, ...]
    questions: tuple[IntentQuestion, ...]


@dataclass(frozen=True)
class IntentReviewLimits:
    max_input_tokens: int = 131072
    max_total_tokens: int = 131072
    task_timeout_seconds: int = 900


@dataclass(frozen=True)
class IntentModelAttempt:
    attempt: int
    status: str
    delivery: str
    reason: str
    started_at: str
    completed_at: str
    duration_ms: int
    usage: dict[str, int] | None = None
    request_hash: str = ""


@dataclass(frozen=True)
class IntentStageRecord:
    stage: str
    status: str = "not_attempted"
    reason: str = ""
    prompt_hash: str = ""
    request_hash: str | None = None
    request_bytes: int | None = None
    measured_input_tokens: int | None = None
    token_measurement: str = "not_attempted"
    attempts: tuple[IntentModelAttempt, ...] = ()
    token_measurement_request_hash: str | None = None
    token_measurement_delivery: str = "not_attempted"
    token_measurement_usage: dict[str, int] | None = None
    token_measurement_model_calls: int | None = 0


@dataclass(frozen=True)
class ReviewBResult:
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
    goals: IntentGoals | None = None
    observations: IntentObservations | None = None
    summary: str = ""
    findings: tuple[IntentFinding, ...] = ()
    questions: tuple[IntentQuestion, ...] = ()
    validation_errors: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    origin: dict[str, object] = field(default_factory=dict)