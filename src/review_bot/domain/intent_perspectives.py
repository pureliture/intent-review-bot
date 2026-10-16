"""Approved intent-review questions and private, single-run result records."""
from __future__ import annotations

from dataclasses import dataclass, field

from .intent_models import IntentReviewLimits, IntentStageRecord

CATALOG_VERSION = "intent-perspectives/v1"


@dataclass(frozen=True)
class IntentPerspective:
    id: str
    name_ko: str
    name_en: str
    question_ko: str
    question_en: str


PERSPECTIVES = (
    IntentPerspective("problem_success", "문제와 성공 기준", "Problem and success criteria",
        "누구의 어떤 상황이 문제이며, 무엇이 달라지면 해결인가?",
        "Whose situation is a problem, and what observable change would count as success?"),
    IntentPerspective("intent_change", "목적과 변경의 대응", "Intent and change correspondence",
        "목적마다 구현이 있고 주요 변경마다 설명되는 이유가 있는가?",
        "Does each purpose have an implementation, and does each material change have a reason?"),
    IntentPerspective("means_fit", "수단의 적합성", "Fitness of the means",
        "구현이 의도한 문제에 작용하며 원하는 해결 수준에 맞는가?",
        "Does the mechanism address the intended problem at the intended level of resolution?"),
    IntentPerspective("outcome_connection", "결과까지의 연결", "Connection to the outcome",
        "구현의 효과가 실제 사용자·호출자·후속 처리에 도달하는가?",
        "Does the implementation's effect reach the intended user, caller, or downstream process?"),
    IntentPerspective("achievement_scope", "달성 범위", "Scope of achievement",
        "의도한 대상과 조건에서 빠짐없이, 과하지 않게 적용되는가?",
        "Does the change cover the intended subjects and conditions without overextending them?"),
    IntentPerspective("preservation", "보존 조건", "Preservation conditions",
        "유지하려던 결과와 바꾸지 않겠다는 경계를 지키는가?",
        "Does the change preserve the results and boundaries that should remain intact?"),
    IntentPerspective("assumptions_tradeoffs", "전제와 절충", "Assumptions and trade-offs",
        "필요한 가정과 감수하는 손실이 목적에 맞는가?",
        "Are the necessary assumptions and accepted costs compatible with the purpose?"),
    IntentPerspective("supporting_evidence", "검증 근거", "Supporting evidence",
        "코드·테스트가 목적 달성에 대한 주장을 뒷받침하는가?",
        "Do the supplied code and tests support the claims about achieving the purpose?"),
)


@dataclass(frozen=True)
class IntentPerspectiveLimits(IntentReviewLimits):
    max_calls: int = 12
    max_rechecks: int = 1


@dataclass(frozen=True)
class IntentPerspectiveResult:
    schema_version: int
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
    prompt_hashes: dict
    model: dict
    scope: dict
    opinion_scope: dict
    usage: dict
    model_delivery: dict
    stages: tuple[IntentStageRecord, ...]
    language: str = "ko"
    limits: dict = field(default_factory=dict)
    catalog_version: str = CATALOG_VERSION
    catalog_hash: str = ""
    selection: tuple[dict, ...] = ()
    perspectives: tuple[dict, ...] = ()
    perspective_history: tuple[dict, ...] = ()
    narrative: dict = field(default_factory=dict)
    judgements: tuple[dict, ...] = ()
    judgement_history: tuple[dict, ...] = ()
    context_gaps: tuple[dict, ...] = ()
    context_history: tuple[dict, ...] = ()
    requests: tuple[dict, ...] = ()
    passes: tuple[dict, ...] = ()
    summary: str = ""
    findings: tuple[dict, ...] = ()
    questions: tuple[dict, ...] = ()
    validation_errors: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    origin: dict = field(default_factory=dict)
