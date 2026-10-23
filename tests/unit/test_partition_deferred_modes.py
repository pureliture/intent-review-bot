"""A completed global mode choice survives failure of B's later perspective choice."""
from __future__ import annotations

from copy import deepcopy

import pytest

from review_bot.application.context import hash_json
from review_bot.application.partition_review import (
    run_partition_review, validate_execution_plan, validate_partition_result,
)
from review_bot.application.source_partition import PartitionError
from review_bot.domain.intent_perspectives import PERSPECTIVES
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


@pytest.mark.parametrize("failed_stage", ["select_review_scope", "select_perspectives"])
def test_only_post_global_perspective_failure_defers_b_and_preserves_a(failed_stage):
    task = make_task()

    def fail_closed_choice(payload, body):
        if payload["stage"] == failed_stage:
            if failed_stage == "select_review_scope":
                body["choice"] = "outside-declared-options"
            else:
                body["selected_perspective_ids"] = ["outside-declared-options"]
        return body

    review = PartitionModelClient()
    tev = PartitionModelClient(group_all=True, transform=fail_closed_choice)
    plan = run_partition_review(task, "planning", review, tev,
        available_modes=("architecture", "intent"),
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=128))
    assert not review.requests
    validate_partition_result(plan, task)
    if failed_stage == "select_review_scope":
        assert plan.status == "failed" and plan.reason == "invalid_closed_choice"
        assert not plan.partition.get("execution_plan")
        assert not plan.partition.get("deferred_modes")
        return

    assert plan.status == "planned", plan.reason
    validate_execution_plan(plan)  # A can bind the validated shared plan.
    assert plan.partition["selected_modes"] == ["architecture", "intent"]
    assert plan.partition["deferred_modes"] == {"intent": "invalid_perspective_subset"}
    assert plan.partition["execution_plan"]["deferred_modes"] == plan.partition["deferred_modes"]
    assert all(row["decision"] == "undetermined" for row in plan.partition["scope_selection"]["perspectives"])
    selection = plan.partition["scope_selection"]["perspective_selection"]
    assert selection["status"] == "failed" and selection["response"] == {"selected_perspective_ids": ["outside-declared-options"]}
    assert sum(stage.stage == "select_perspectives" for stage in plan.stages) == 1
    assert plan.usage["calls"] == len(tev.requests)
    previous_calls = (len(review.requests), len(tev.requests))
    deferred = run_partition_review(task, "intent", review, tev, prepared_plan=plan,
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=128))
    assert deferred.status == "not_run" and deferred.reason == "mode_deferred_in_shared_plan"
    assert (len(review.requests), len(tev.requests)) == previous_calls
    validate_partition_result(deferred, task)

    # Recomputing the container hash cannot turn this failure into a B exclusion.
    changed = deepcopy(plan)
    changed.partition["selected_modes"] = ["architecture"]
    changed.partition["scope_selection"]["selected_modes"] = ["architecture"]
    changed.partition["execution_plan"]["selected_modes"] = ["architecture"]
    changed.partition["execution_plan"]["selection"]["selected_modes"] = ["architecture"]
    changed.partition["deferred_modes"].clear()
    changed.partition["deferred_mode_failures"].clear()
    changed.partition["execution_plan"]["deferred_modes"].clear()
    changed.partition["execution_plan"]["deferred_mode_failures"].clear()
    changed.partition["plan_id"] = hash_json(changed.partition["execution_plan"])
    with pytest.raises(PartitionError, match="stored_global_modes_mismatch|stored_perspective_subset_failure_changed"):
        validate_partition_result(changed, task)

    changed = deepcopy(plan)
    changed.partition["deferred_modes"]["intent"] = "invented_failure"
    changed.partition["deferred_mode_failures"]["intent"]["reason"] = "invented_failure"
    changed.partition["execution_plan"]["deferred_modes"]["intent"] = "invented_failure"
    changed.partition["execution_plan"]["deferred_mode_failures"]["intent"]["reason"] = "invented_failure"
    changed.partition["plan_id"] = hash_json(changed.partition["execution_plan"])
    with pytest.raises(PartitionError, match="stored_perspective_subset_failure_changed"):
        validate_partition_result(changed, task)
