"""Versioned contracts for Core v1 static analysis and review."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .changes import Change, Measurement
from .status import (
    AnalysisStatusKind,
    ExecutionStateKind,
    FindingBasisKind,
    FindingJudgementKind,
    ParseStatusKind,
    ReviewModeKind,
    ReviewStatusKind,
    SourceStateKind,
    ToolKind,
)


@dataclass(frozen=True)
class AnalysisError:
    kind: str
    message: str
    path: str = ""
    detail: str = ""


@dataclass(frozen=True)
class PmdViolation:
    path: str
    raw_path: str
    begin_line: int
    end_line: int
    begin_column: int
    end_column: int
    rule: str
    ruleset: str
    priority: int
    klass: str
    method: str
    package: str
    message: str
    code_fragment: str | None = None


@dataclass(frozen=True)
class CpdOccurrence:
    path: str
    raw_path: str
    begin_line: int
    end_line: int
    begin_column: int | None = None
    end_column: int | None = None
    end_line_inferred: bool = False
    code_fragment: str | None = None


@dataclass(frozen=True)
class CpdDuplication:
    lines: int
    tokens: int
    occurrences: tuple[CpdOccurrence, ...]
    code_fragment: str


Finding = PmdViolation | CpdDuplication


@dataclass(frozen=True)
class ParsedReport:
    tool: ToolKind | str
    path: str
    exists: bool
    parse_status: ParseStatusKind | str
    tool_version: str | None = None
    content_hash: str | None = None
    violations: tuple[PmdViolation, ...] = ()
    duplications: tuple[CpdDuplication, ...] = ()
    errors: tuple[AnalysisError, ...] = ()
    suppressed_count: int = 0

    @property
    def finding_count(self) -> int:
        return len(self.violations) if self.tool == "pmd" else len(self.duplications)


@dataclass(frozen=True)
class ExecutionRecord:
    schema_version: int
    tool: ToolKind | str
    exit_code: int | None
    tool_version: str
    java_version: str
    java_vendor: str
    classpath_hash: str
    profile_hash: str
    sha: str
    run_id: str
    source_hash: str | None
    source_root: str
    source_roots: tuple[str, ...]
    source_count: int
    ruleset_hash: str | None
    minimum_tokens: int | None
    language_version: str
    encoding: str
    type_resolution: str
    started_at: str
    completed_at: str
    report_exists: bool
    report_hash: str | None
    command: tuple[str, ...]
    state: ExecutionStateKind | str = "completed"


@dataclass(frozen=True)
class ToolAnalysis:
    tool: ToolKind | str
    status: AnalysisStatusKind | str
    exit_code: int | None
    report: ParsedReport
    execution: ExecutionRecord | None
    errors: tuple[AnalysisError, ...] = ()


@dataclass(frozen=True)
class SourceExclusion:
    path: str
    reason: str


@dataclass(frozen=True)
class AnalysisRun:
    schema_version: int
    run_id: str
    sha: str
    profile_hash: str
    conditions_hash: str
    conditions: dict[str, object]
    output_dir: str
    started_at: str
    completed_at: str
    status: AnalysisStatusKind | str
    source_state: SourceStateKind | str
    source_files: tuple[str, ...]
    source_hash: str | None
    exclusions: tuple[SourceExclusion, ...]
    pmd: ToolAnalysis
    cpd: ToolAnalysis
    build_exit_code: int | None
    errors: tuple[AnalysisError, ...] = ()
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeltaSignal:
    signal_id: str
    tool: ToolKind | str
    change: Change
    base_findings: tuple[Finding, ...]
    head_findings: tuple[Finding, ...]
    paths: tuple[str, ...]
    matching: str
    measurements: tuple[Measurement, ...] = ()
    limitations: tuple[str, ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass(frozen=True)
class AnalysisDelta:
    schema_version: int
    base_sha: str
    head_sha: str
    conditions_hash: str | None
    comparable: bool
    signals: tuple[DeltaSignal, ...]
    limitations: tuple[str, ...] = ()
    analysis_status: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    @property
    def counts(self) -> dict[str, int]:
        counts = {
            "existing": 0,
            "new": 0,
            "worsened": 0,
            "improved": 0,
            "resolved": 0,
            "unknown": 0,
        }
        for signal in self.signals:
            if signal.change in counts:
                counts[signal.change] += 1
            else:
                counts["unknown"] += 1
        return counts


@dataclass(frozen=True)
class CodeFile:
    path: str
    content: str
    content_hash: str
    kind: Literal["source", "test", "build"] | str


@dataclass(frozen=True)
class CodeSnapshot:
    sha: str
    files: tuple[CodeFile, ...]
    exclusions: tuple[SourceExclusion, ...] = ()
    inventory_hash: str = ""
    complete: bool = True
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrSnapshot:
    repository: str
    number: int
    base_sha: str
    head_sha: str
    source_branch: str
    labels: tuple[str, ...] = ()
    changed_files: tuple[str, ...] = ()
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Snapshot:
    pr: PrSnapshot
    base: CodeSnapshot
    head: CodeSnapshot
    explicit_criteria: tuple[dict[str, object], ...] = ()
    observations: tuple[dict[str, object], ...] = ()


@dataclass(frozen=True)
class Classification:
    review_mode: ReviewModeKind | str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    kind: Literal["signal", "code", "criterion", "observation"] | str
    version: Literal["base", "head", "both"] | str
    sha: str
    path: str
    begin_line: int
    end_line: int
    content: str
    signal_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SelectedSignal:
    signal: DeltaSignal
    role: Literal["review", "context", "limitation"] | str
    reason: str


@dataclass(frozen=True)
class TaskInput:
    schema_version: int
    snapshot: Snapshot
    delta: AnalysisDelta
    classification: Classification
    signals: tuple[SelectedSignal, ...]
    evidence: tuple[Evidence, ...]
    scope: dict[str, object]
    limitations: tuple[str, ...]
    input_hash: str

    @property
    def review_mode(self) -> str:
        return str(self.classification.review_mode)


@dataclass(frozen=True)
class Citation:
    evidence_id: str
    version: Literal["base", "head"] | str
    sha: str
    path: str
    begin_line: int
    end_line: int
    quote: str


@dataclass(frozen=True)
class ReviewFinding:
    signal_id: str
    judgement: FindingJudgementKind | str
    change: Change
    basis: FindingBasisKind | str
    title: str
    reason: str
    recommendation: str
    evidence_ids: tuple[str, ...]
    citations: tuple[Citation, ...]
    criterion_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    measurements: tuple[Measurement, ...] = ()


@dataclass(frozen=True)
class ReviewQuestion:
    signal_id: str
    question: str
    reason: str
    evidence_ids: tuple[str, ...]
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class ReviewResult:
    schema_version: int
    review_type: Literal["A"] | str
    run_key: str
    base_sha: str
    head_sha: str
    review_mode: ReviewModeKind | str
    pr: PrSnapshot
    status: ReviewStatusKind | str
    reason: str
    summary: str
    findings: tuple[ReviewFinding, ...]
    questions: tuple[ReviewQuestion, ...]
    validation_errors: tuple[str, ...]
    scope: dict[str, object]
    opinion_scope: dict[str, object]
    analysis: AnalysisDelta
    model: dict[str, object]
    usage: dict[str, int]
    created_at: str
    prompt_version: str
    input_hash: str
    origin: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisBundle:
    schema_version: int
    pr: PrSnapshot
    base: AnalysisRun
    head: AnalysisRun
    task_input: TaskInput


@dataclass(frozen=True)
class ModelRequest:
    schema_version: int
    model: str
    system_prompt: str
    user_content: str
    timeout_seconds: int
    max_output_tokens: int | None
    response_schema: dict[str, object]
    reasoning_budget_tokens: int | None = None


@dataclass(frozen=True)
class ModelResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
