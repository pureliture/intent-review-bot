"""Analysis, execution, and review status hierarchies."""
from __future__ import annotations

from enum import Enum
from typing import Literal


# --- Tool Identifiers ---
class Tool(str, Enum):
    PMD = "pmd"
    CPD = "cpd"


ToolKind = Literal["pmd", "cpd"]

# --- Core v1 Statuses ---
class ParseStatus(str, Enum):
    VALID = "valid"
    MISSING = "missing"
    INVALID = "invalid"


ParseStatusKind = Literal["valid", "missing", "invalid"]


class AnalysisStatus(str, Enum):
    CLEAN = "clean"
    FINDINGS = "findings"
    INCOMPLETE = "incomplete"
    FAILED = "failed"
    MISSING_REPORT = "missing_report"
    NO_SOURCES = "no_sources"


AnalysisStatusKind = Literal["clean", "findings", "incomplete", "failed", "missing_report", "no_sources"]


class ExecutionState(str, Enum):
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    NOT_RUN = "not_run"


ExecutionStateKind = Literal["completed", "timed_out", "failed", "not_run"]


class SourceState(str, Enum):
    COMPLETE = "complete"
    NO_JAVA_SOURCES = "no_java_sources"
    FAILED = "failed"


SourceStateKind = Literal["complete", "no_java_sources", "failed"]


class ReviewMode(str, Enum):
    REFACTORING = "refactoring"
    GENERAL = "general"


ReviewModeKind = Literal["refactoring", "general"]


class ReviewStatus(str, Enum):
    COMPLETED = "completed"
    NOT_RUN = "not_run"
    FAILED = "failed"


ReviewStatusKind = Literal["completed", "not_run", "failed"]

# --- Finding Judgements & Basis ---
class FindingJudgement(str, Enum):
    CONCERN = "concern"
    ACCEPTABLE_WITH_EVIDENCE = "acceptable_with_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


FindingJudgementKind = Literal["concern", "acceptable_with_evidence", "insufficient_evidence"]


class FindingBasis(str, Enum):
    ANALYSIS_SIGNAL = "analysis_signal"
    EXPLICIT_CRITERION = "explicit_criterion"
    OBSERVED_STRUCTURE = "observed_structure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


FindingBasisKind = Literal["analysis_signal", "explicit_criterion", "observed_structure", "insufficient_evidence"]

# --- Service Envelope v2 Statuses ---
class ComparisonStatus(str, Enum):
    COMPARABLE = "comparable"
    UNAVAILABLE = "unavailable"
    AMBIGUOUS = "ambiguous"


ComparisonStatusKind = Literal["comparable", "unavailable", "ambiguous"]


class SelectionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


SelectionStatusKind = Literal["completed", "failed", "superseded"]


class ExecutionStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"
    SUPERSEDED = "superseded"


ExecutionStatusKind = Literal["completed", "failed", "not_run", "not_applicable", "superseded"]


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUPERSEDED = "superseded"


RunStatusKind = Literal["queued", "running", "succeeded", "skipped", "failed", "cancelled", "superseded"]


class RunStage(str, Enum):
    FETCH = "fetch"
    SELECT = "select"
    PREPARE = "prepare"
    ANALYZE = "analyze"
    COMPARE = "compare"
    CONTEXT = "context"
    REVIEW = "review"
    VALIDATE = "validate"
    PUBLISH = "publish"


RunStageKind = Literal["fetch", "select", "prepare", "analyze", "compare", "context", "review", "validate", "publish"]


class PublicationStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


PublicationStatusKind = Literal["not_attempted", "succeeded", "failed", "unknown"]


class ModelDeliveryStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    SENT = "sent"
    UNKNOWN = "unknown"


ModelDeliveryStatusKind = Literal["not_attempted", "sent", "unknown"]
