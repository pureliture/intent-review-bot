"""Contract tests for Feature 14: Strict A Model Request Builder,

Feature 15: Strict Model Echo Verification,
Feature 16: Model Output Schema Enforcement,
Feature 17: Grounded Citation Verification, and
Feature 18: CPD Structural Rule Defense.

Requirements:
- contracts.md §7:
  - Strict model echo: returned base_sha, head_sha, review_mode must match input exactly.
  - Output schema: summary (1-4000 chars), max 20 findings, max 20 questions.
  - Citations: must match existing Evidence IDs and verbatim quotes.
  - CPD defense: cannot claim architecture concern based solely on duplicate count.
"""

from __future__ import annotations

from typing import Any

try:
    from review_bot.application.review_a import validate_model_response  # type: ignore
except ImportError:
    validate_model_response = None  # type: ignore


# Reference oracle for validating model response contracts
def oracle_validate_model_response(
    response: dict[str, Any],
    expected_base_sha: str,
    expected_head_sha: str,
    expected_review_mode: str,
    known_evidence: dict[str, str],  # evidence_id -> content
) -> list[str]:
    errors = []

    # 1. Echo verification (Feature 15)
    if response.get("base_sha") != expected_base_sha:
        errors.append(f"Mismatched base_sha: expected {expected_base_sha}, got {response.get('base_sha')}")
    if response.get("head_sha") != expected_head_sha:
        errors.append(f"Mismatched head_sha: expected {expected_head_sha}, got {response.get('head_sha')}")
    if response.get("review_mode") != expected_review_mode:
        errors.append(f"Mismatched review_mode: expected {expected_review_mode}, got {response.get('review_mode')}")

    # 2. Schema enforcement (Feature 16)
    summary = response.get("summary", "")
    if not isinstance(summary, str) or len(summary) < 1 or len(summary) > 4000:
        errors.append("Summary must be between 1 and 4000 characters")

    findings = response.get("findings", [])
    if not isinstance(findings, list) or len(findings) > 20:
        errors.append("Findings must be a list with at most 20 items")

    questions = response.get("questions", [])
    if not isinstance(questions, list) or len(questions) > 20:
        errors.append("Questions must be a list with at most 20 items")

    # 3. Grounded Citation verification (Feature 17)
    for finding in findings:
        citations = finding.get("citations", [])
        for cit in citations:
            ev_id = cit.get("evidence_id")
            quote = cit.get("quote", "")
            if ev_id not in known_evidence:
                errors.append(f"Unknown evidence_id referenced: {ev_id}")
            elif quote and quote not in known_evidence[ev_id]:
                errors.append(f"Citation quote '{quote}' not found verbatim in evidence {ev_id}")

    # 4. CPD Structural Rule Defense (Feature 18)
    for finding in findings:
        category = finding.get("category", "")
        basis = finding.get("basis", "")
        # Cannot declare architecture concern solely on CPD duplicate signal
        if category == "architecture_concern" and "cpd_count_only" in basis:
            errors.append("CPD count alone cannot justify an architecture concern")

    return errors


# ==============================================================================
# Tier 1: Model Echo & Schema Contracts
# ==============================================================================


class TestModelEchoAndSchemaTier1:
    def test_valid_model_response_passes(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "refactoring"
        evidence = {"E-SRC-01": "public class OrderService { void process() {} }"}

        valid_payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Refactoring review: reduced complexity in OrderService.",
            "findings": [
                {
                    "id": "F-01",
                    "category": "improvement",
                    "basis": "cognitive_complexity_reduction",
                    "citations": [
                        {"evidence_id": "E-SRC-01", "quote": "public class OrderService"}
                    ],
                }
            ],
            "questions": [],
        }

        errors = oracle_validate_model_response(
            valid_payload, base_sha, head_sha, mode, evidence
        )
        assert errors == []

    def test_mismatched_echo_rejected(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {}

        payload = {
            "base_sha": "wrong_base" + "0" * 30,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [],
            "questions": [],
        }

        errors = oracle_validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("Mismatched base_sha" in e for e in errors)


# ==============================================================================
# Tier 2: Citations & CPD Defense Contracts
# ==============================================================================


class TestModelCitationsAndCpdDefenseTier2:
    def test_hallucinated_evidence_id_rejected(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {"E-SRC-01": "class Valid {}"}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [
                {
                    "id": "F-01",
                    "citations": [{"evidence_id": "E-NON-EXISTENT", "quote": "class Valid"}],
                }
            ],
            "questions": [],
        }

        errors = oracle_validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("Unknown evidence_id" in e for e in errors)

    def test_non_verbatim_quote_rejected(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {"E-SRC-01": "class OrderService {}"}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [
                {
                    "id": "F-01",
                    "citations": [{"evidence_id": "E-SRC-01", "quote": "class FakeService"}],
                }
            ],
            "questions": [],
        }

        errors = oracle_validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("not found verbatim" in e for e in errors)

    def test_cpd_duplicate_count_alone_cannot_trigger_architecture_concern(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [
                {
                    "id": "F-01",
                    "category": "architecture_concern",
                    "basis": "cpd_count_only",
                    "citations": [],
                }
            ],
            "questions": [],
        }

        errors = oracle_validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("CPD count alone" in e for e in errors)

    def test_findings_limit_exceeded_rejected(self) -> None:
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [{"id": f"F-{i}"} for i in range(25)],  # Limit is 20
            "questions": [],
        }

        errors = oracle_validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("at most 20 items" in e for e in errors)


# ==============================================================================
# Tier 3: Direct Application Implementation Verification
# ==============================================================================


class TestValidateModelResponseImplementation:
    def test_impl_valid_model_response_passes(self) -> None:
        assert validate_model_response is not None
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "refactoring"
        evidence = {"E-SRC-01": "public class OrderService { void process() {} }"}

        valid_payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Refactoring review: reduced complexity in OrderService.",
            "findings": [
                {
                    "id": "F-01",
                    "category": "improvement",
                    "basis": "cognitive_complexity_reduction",
                    "citations": [
                        {"evidence_id": "E-SRC-01", "quote": "public class OrderService"}
                    ],
                }
            ],
            "questions": [],
        }

        errors = validate_model_response(
            valid_payload, base_sha, head_sha, mode, evidence
        )
        assert errors == []

    def test_impl_mismatched_echo_rejected(self) -> None:
        assert validate_model_response is not None
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {}

        payload = {
            "base_sha": "wrong_base" + "0" * 30,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [],
            "questions": [],
        }

        errors = validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("Mismatched base_sha" in e for e in errors)

    def test_impl_hallucinated_evidence_id_rejected(self) -> None:
        assert validate_model_response is not None
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {"E-SRC-01": "class Valid {}"}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [
                {
                    "id": "F-01",
                    "citations": [{"evidence_id": "E-NON-EXISTENT", "quote": "class Valid"}],
                }
            ],
            "questions": [],
        }

        errors = validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("Unknown evidence_id" in e for e in errors)

    def test_impl_non_verbatim_quote_rejected(self) -> None:
        assert validate_model_response is not None
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {"E-SRC-01": "class OrderService {}"}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [
                {
                    "id": "F-01",
                    "citations": [{"evidence_id": "E-SRC-01", "quote": "class FakeService"}],
                }
            ],
            "questions": [],
        }

        errors = validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("not found verbatim" in e for e in errors)

    def test_impl_cpd_duplicate_count_alone_cannot_trigger_architecture_concern(self) -> None:
        assert validate_model_response is not None
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [
                {
                    "id": "F-01",
                    "category": "architecture_concern",
                    "basis": "cpd_count_only",
                    "citations": [],
                }
            ],
            "questions": [],
        }

        errors = validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("CPD count alone" in e for e in errors)

    def test_impl_findings_limit_exceeded_rejected(self) -> None:
        assert validate_model_response is not None
        base_sha = "a" * 40
        head_sha = "b" * 40
        mode = "general"
        evidence = {}

        payload = {
            "base_sha": base_sha,
            "head_sha": head_sha,
            "review_mode": mode,
            "summary": "Review summary",
            "findings": [{"id": f"F-{i}"} for i in range(25)],  # Limit is 20
            "questions": [],
        }

        errors = validate_model_response(
            payload, base_sha, head_sha, mode, evidence
        )
        assert any("at most 20 items" in e for e in errors)

