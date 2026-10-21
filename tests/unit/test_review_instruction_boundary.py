"""Repository prose guidance changes review prompts, preserving fixed controls."""
from dataclasses import replace
import json

from review_bot.application.partition_review import run_partition_review, validate_partition_result
from review_bot.domain.partition_review import PartitionReviewLimits
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


def test_review_guidance_reaches_multi_page_review_but_not_tev_or_factual_controls():
    task = make_task()
    guidance = "custom review guidance"

    def run(instructions):
        rewritten = False

        def count(request):
            payload = json.loads(request.user_content)
            if payload["stage"] == "final_review":
                return 200 + 100 * len(payload["observations"])
            if payload["stage"] == "global_review":
                return 1000 if "complete_originals" in payload else 100
            return 100

        def force_one_factual_correction(payload, body):
            nonlocal rewritten
            if payload["stage"] == "fact_check" and not rewritten and "perspectives:2" in body["items"]:
                body["items"]["perspectives:2"]["reasoning"].update(
                    state="unverifiable", reason="Synthetic factual control requires one correction.")
            if payload["stage"] == "fact_rewrite":
                body["replacement"]["reasoning"] = "The corrected explanation is limited to the fixed source."
                rewritten = True
            return body

        review = PartitionModelClient(counter=count, transform=force_one_factual_correction)
        review.profile = replace(review.profile, max_input_tokens=350, max_output_tokens=512)
        tev = PartitionModelClient()
        result = run_partition_review(task, "intent", review, tev,
            limits=PartitionReviewLimits(max_calls=128, max_total_tokens=1000000),
            review_instructions=instructions)
        assert result.status == "completed", result.reason
        assert len(result.partition["integration_pages"]) > 1
        assert result.partition["global_review"]["attempts"]
        assert rewritten
        validate_partition_result(result, task)
        return review, tev

    baseline_review, baseline_tev = run("")
    review, tev = run(guidance)
    stages = {json.loads(request.user_content)["stage"] for request in review.requests}
    assert {"review_partition", "final_review", "global_review", "fact_check", "fact_rewrite"} <= stages
    assert len(review.requests) == len(baseline_review.requests)
    for original, request in zip(baseline_review.requests, review.requests):
        stage = json.loads(request.user_content)["stage"]
        assert stage == json.loads(original.user_content)["stage"]
        assert request.response_schema == original.response_schema
        if stage in ("review_partition", "final_review", "global_review"):
            assert guidance in request.system_prompt
            assert request.system_prompt != original.system_prompt
        else:
            assert stage in ("fact_check", "fact_rewrite")
            assert guidance not in request.system_prompt and guidance not in request.user_content
            assert request.system_prompt == original.system_prompt
    assert tev.requests and len(tev.requests) == len(baseline_tev.requests)
    for original, request in zip(baseline_tev.requests, tev.requests):
        assert guidance not in request.system_prompt and guidance not in request.user_content
        assert request.system_prompt == original.system_prompt
        assert request.response_schema == original.response_schema
        assert set(request.response_schema["properties"]) == {"choice"}
