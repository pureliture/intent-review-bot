"""Fixed, synthetic B material; no target repository or live model is accessed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from review_bot.application.intent_context import build_intent_task_input
from review_bot.domain.envelope_v2 import ChangedFile, EnvelopeEvidence, ReviewSnapshot, UnavailableSource
from review_bot.domain.models_v1 import CodeFile, CodeSnapshot, ModelResponse

GOAL = "The same request must be processed once, including sequential retries."
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
CAPTURED_AT = "2026-10-01T12:00:00+00:00"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_task(language: str = "python", *, body: str = GOAL, unavailable: bool = False, title: str = "Intent fixture"):
    case = json.loads(Path(__file__).with_name("cases.json").read_text())[language]
    text_digest = digest(f"{title}\n{body}")
    snapshot = ReviewSnapshot(
        schema_version=2, snapshot_id="c" * 64, repository="example/project", pr_number=36,
        repository_id=1, installation_id=1, target_base_sha=BASE_SHA, head_sha=HEAD_SHA,
        comparison_base_sha=BASE_SHA, comparison_status="comparable", captured_at=CAPTURED_AT,
        as_of=CAPTURED_AT, pr_text_digest=text_digest, context_digest="d" * 64,
        changed_files=(ChangedFile(case["source_path"]),),
        evidence=(EnvelopeEvidence(
            "E-PR-01", "pr_text", text_digest,
            {"repository": "example/project", "pr_number": 36},
            {"title": title, "body": body}, text_digest,
        ),),
        unavailable_sources=(UnavailableSource("issue:12", "forbidden"),) if unavailable else (),
    )
    base = CodeSnapshot(BASE_SHA, (CodeFile(case["source_path"], case["base"], digest(case["base"]), "source"),))
    head = CodeSnapshot(HEAD_SHA, (
        CodeFile(case["source_path"], case["head"], digest(case["head"]), "source"),
        CodeFile(case["test_path"], case["test"], digest(case["test"]), "test"),
    ))
    return build_intent_task_input(snapshot, base, head, related_paths=(case["test_path"],))


def stage_bodies(*, no_goals: bool = False, unavailable: bool = False):
    goals = {
        "declared_goals": [] if no_goals else [{
            "goal_id": "G1", "statement": GOAL, "evidence_ids": ["E-PR-01"], "quote": GOAL,
        }],
        "declared_constraints": [],
        "unavailable_context": ["issue:12:forbidden"] if unavailable else [],
    }
    observations = {
        "implementation_observations": [{
            "observation_id": "O1", "statement": "The implementation adds mutual exclusion.",
            "evidence_ids": ["E-CODE-head-0001"],
        }, {
            "observation_id": "O2", "statement": "The supplied test contains an expected return value, not an execution result.",
            "evidence_ids": ["E-CODE-head-0002"],
        }],
        "intent_hypotheses": [{
            "hypothesis_id": "H1", "statement": "The change may aim to limit concurrent execution.",
            "evidence_ids": ["E-CODE-head-0001"], "missing_evidence": ["Relationship to sequential retries"],
        }],
        "assumptions": [{
            "assumption_id": "S1", "statement": "An existing deduplication mechanism may handle sequential retries.",
            "evidence_ids": [], "missing_evidence": ["Deduplication implementation or a recorded test run"],
        }],
    }
    refs = ["E-CODE-head-0001"] if no_goals else ["E-PR-01", "E-CODE-head-0001"]
    assessment = {
        "summary": "Mutual exclusion is visible; sequential retry behavior remains unverified.",
        "findings": [{
            "finding_id": "B1", "goal_ids": [] if no_goals else ["G1"], "constraint_ids": [],
            "observation_ids": ["O1"], "hypothesis_ids": ["H1"],
            "statement": "Sequential retry behavior needs context.", "assessment": "insufficient_context",
            "reasoning": "A lock alone is not evidence about sequential retries or the whole repository.",
            "evidence_ids": refs, "missing_evidence": ["Existing deduplication code or a verified retry run"],
        }],
        "questions": [{
            "question_id": "Q1", "question": "Where is sequential retry deduplication implemented?",
            "related_finding_ids": ["B1"], "evidence_ids": refs,
            "why_needed": "The supplied scope does not establish retry behavior.",
            "how_answer_changes_assessment": "A recorded implementation or execution result could resolve the uncertainty.",
        }],
    }
    return [goals, observations, assessment]


def responses(bodies=None):
    return [ModelResponse(json.dumps(body), {"input_tokens": 100, "output_tokens": 50})
            for body in (stage_bodies() if bodies is None else bodies)]