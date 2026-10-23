"""Closed choices route to the Decisions API when a decision client is set.

No measurement calls exist on either path: the planning chat client is
never consulted for choices, and the decision client performs exactly
one decisions POST per choice with zero generation calls.
"""
from __future__ import annotations

import json

import httpx
import pytest

from review_bot.adapters.model_http import reset_model_request_context, set_model_request_context
from review_bot.adapters.openrouter_decisions import DECISION_MODEL, OpenRouterDecisionClient
from review_bot.application.architecture_review import MockModelClient
from review_bot.application.partition_review import run_partition_review, validate_partition_result
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


def limits(**kwargs):
    return PartitionReviewLimits(max_total_tokens=1000000, max_calls=512, **kwargs)


class DecisionTransport:
    """Mock OpenRouter Decisions endpoint answering every choice with options[0]."""

    def __init__(self, model: str = DECISION_MODEL, context: int = 262144):
        self.calls = []
        self.model = model
        self.context = context
        self.fail_choice = None

    def __call__(self, request):
        import time
        self.calls.append((request.method, request.url.path))
        if request.url.path == "/api/v1/models":
            return httpx.Response(200, json={"data": [{
                "id": self.model, "context_length": self.context,
                "architecture": {"output_modalities": ["decisions"]}}]})
        assert request.url.path == "/api/alpha/decisions"
        payload = json.loads(request.content)
        assert payload["model"] == self.model
        answers = {}
        for key, question in payload["questions"].items():
            assert question["type"] == "choice" and question["criteria"]
            if self.fail_choice == key:
                answers[key] = {"type": "choice", "choice": "no-such-option",
                    "probabilities": {name: 1.0 if i == 0 else 0.0 for i, name in enumerate(question["criteria"])},
                    "confidence": 1.0}
                continue
            first = next(iter(question["criteria"]))
            answers[key] = {"type": "choice", "choice": first,
                "probabilities": {name: 1.0 if name == first else 0.0 for name in question["criteria"]},
                "confidence": 1.0}
        return httpx.Response(200, json={"id": "gen-dec-test", "model": self.model,
            "answers": answers, "usage": {"input_tokens": 40, "output_tokens": 1}})


@pytest.fixture
def decision_pair(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(tmp_path / "admission"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    transport = DecisionTransport()
    http_client = httpx.Client

    def client(**options):
        return http_client(transport=httpx.MockTransport(transport), **options)

    monkeypatch.setattr(httpx, "Client", client)
    import time
    token = set_model_request_context(lambda: False, time.monotonic() + 300)
    yield transport, OpenRouterDecisionClient(api_key="test-key")
    reset_model_request_context(token)


class ExplodingPlanningClient(PartitionModelClient):
    def complete(self, request):
        payload = json.loads(request.user_content)
        assert payload["stage"] == "select_perspectives", "single choices must not reach planning chat"
        return super().complete(request)


def test_choices_use_decisions_transport_not_planning_chat(decision_pair):
    transport, decider = decision_pair
    review = PartitionModelClient()
    planning = ExplodingPlanningClient(group_all=True)
    result = run_partition_review(make_task(), "planning", review, planning,
        limits=limits(), available_modes=("intent",), decision_client=decider)
    assert result.status == "planned", result.reason
    assert transport.calls, "no decision call was made"
    assert all(path == "/api/v1/models" or path == "/api/alpha/decisions"
        for _, path in transport.calls), transport.calls
    decision_posts = [c for c in transport.calls if c[1] == "/api/alpha/decisions"]
    assert len(decision_posts) >= 2
    # Admission probes may consult the planning client (zero HTTP each);
    # Single choices stay on Decisions; the one subset selection uses planning chat.
    assert [json.loads(r.user_content)["stage"] for r in planning.requests] == ["select_perspectives"]
    assert result.partition["decisions"], "no choice records"
    assert all(record["method"] == "model" for record in result.partition["decisions"])
    validate_partition_result(result, make_task())


def test_choices_fall_back_to_planning_client_without_decider():
    review = PartitionModelClient()
    planning = PartitionModelClient(group_all=True)
    result = run_partition_review(make_task(), "planning", review, planning,
        limits=limits(), available_modes=("intent",))
    assert result.status == "planned", result.reason
    assert planning.requests, "planning chat client saw no choice"
    validate_partition_result(result, make_task())


def test_decision_choice_contract_rejects_unknown_options(decision_pair):
    transport, decider = decision_pair
    review = PartitionModelClient()
    planning = ExplodingPlanningClient(group_all=True)
    transport.fail_choice = "choice"
    result = run_partition_review(make_task(), "planning", review, planning,
        limits=limits(), available_modes=("intent",), decision_client=decider)
    # On the Decisions transport a malformed answer is a model error, not a
    # choice mismatch: the server cannot return an off-list label through a
    # validated answer, so there is nothing to record as a rejected choice.
    assert result.status == "failed" and result.reason == "usage_unavailable"
    assert not result.partition.get("rejected_choices")
    assert planning.requests == []
    assert any(c[1] == "/api/alpha/decisions" for c in transport.calls)


def test_review_generations_stay_on_review_client(decision_pair):
    transport, decider = decision_pair
    review = PartitionModelClient()
    planning = ExplodingPlanningClient(group_all=True)
    result = run_partition_review(make_task(), "intent", review, PartitionModelClient(group_all=True),
        limits=limits(), decision_client=decider)
    assert result.status in ("completed", "partial_review"), result.reason
    assert review.requests, "no review generation happened"
    decisions = [c for c in transport.calls if c[1] == "/api/alpha/decisions"]
    assert decisions, "no routing choice went to decisions"
    validate_partition_result(result, make_task())
