"""Explicit code-first opt-in preserves the existing A/B configuration boundary."""
from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from review_bot.config import ModelConfig, ReviewBotConfig, compute_config_hash, load_review_config_from_str


def test_code_first_is_explicit_and_old_configuration_hash_is_preserved(valid_config_yaml):
    config, old_hash = load_review_config_from_str(valid_config_yaml)
    assert config.intent.pipeline == "declared-v1"
    old = config.model_dump(mode="json")
    old["intent"].pop("pipeline")
    assert compute_config_hash(old) == old_hash
    configured = config.model_dump(mode="json")
    configured["intent"]["pipeline"] = "code-first-v2"
    new = ReviewBotConfig.model_validate(configured)
    assert new.intent.pipeline == "code-first-v2"
    assert new.config_hash != old_hash
    assert new.publish.b_comments_enabled is False


@pytest.mark.parametrize("value", [None, True, 2, "code-first", "unknown"])
def test_unknown_pipeline_cannot_silently_select_a_different_review(valid_config_yaml, value):
    config, _ = load_review_config_from_str(valid_config_yaml)
    configured = config.model_dump(mode="json")
    configured["intent"]["pipeline"] = value
    with pytest.raises(ValidationError):
        ReviewBotConfig.model_validate(configured)


@pytest.fixture
def offline_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv("MACBOOK_LFM_BASE_URL", "http://127.0.0.1:18135/v1")
    monkeypatch.setenv("MACBOOK_TEV1_BASE_URL", "http://127.0.0.1:18137/v1")
    monkeypatch.setenv("ME707_TEV1_BASE_URL", "http://ubuntu-ms-7e07:11437/v1")
    monkeypatch.setenv("REVIEW_BOT_MACBOOK_MODEL_STATE_DIR", str(tmp_path / "macbook-admission"))
    monkeypatch.setenv("REVIEW_BOT_MODEL_STATE_DIR", str(tmp_path / "ubuntu-admission"))
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("Composition must not make an HTTP request"))


@pytest.mark.parametrize("profile,model,context", [
    ("macbook-llamacpp", "tev1-0.8b-q8_0", 32768),
    ("me707-llamacpp", "tev1-4b-q8_0", 131072),
])
def test_explicit_planning_role_does_not_widen_existing_selector(offline_profiles, profile, model, context):
    from review_bot.runtime.model_clients import build_planning_client, build_selector_client

    plan = build_planning_client(profile, ModelConfig(call_timeout_seconds=17, max_concurrency=2))
    selector = build_selector_client(profile)
    assert plan.model == selector.model == model
    assert plan.profile.max_input_tokens == context - 4096
    assert plan.profile.max_output_tokens == 4096
    assert plan.timeout_seconds == 17
    assert type(plan) is not type(selector)


@pytest.mark.parametrize("profile,variable", [
    ("macbook-llamacpp", "MACBOOK_TEV1_BASE_URL"), ("me707-llamacpp", "ME707_TEV1_BASE_URL"),
])
def test_unconfigured_planner_is_absent_without_fallback(offline_profiles, monkeypatch, profile, variable):
    from review_bot.runtime.model_clients import build_planning_client

    monkeypatch.delenv(variable)
    assert build_planning_client(profile) is None
    assert build_planning_client(None) is None
    assert build_planning_client("ollama-local-128k") is None
    with pytest.raises(ValueError, match="Unsupported"):
        build_planning_client("automatic")


def test_cli_preserves_v1_default_and_requires_explicit_code_first_choice():
    from review_bot.runtime.cli import create_parser

    args = ["intent", "--input", "fixed.json", "--output", "private-output"]
    assert create_parser().parse_args(args).pipeline == "declared-v1"
    selected = create_parser().parse_args([*args, "--pipeline", "code-first-v2"])
    assert selected.pipeline == "code-first-v2" and selected.model_profile is None
    assert selected.call_timeout_seconds is None
    assert create_parser().parse_args([*args, "--call-timeout-seconds", "120"]).call_timeout_seconds == 120


def test_code_first_cache_cannot_reuse_v1_or_a_different_planning_role(valid_config_yaml, offline_profiles):
    from review_bot.runtime.mode_fingerprints import intent_input_fingerprint
    from review_bot.runtime.model_clients import build_planning_client, build_review_client
    from tests.fixtures.intent_review import make_task

    config, _ = load_review_config_from_str(valid_config_yaml)
    task = make_task()
    client = build_review_client("macbook-llamacpp")
    planner = build_planning_client("macbook-llamacpp")
    legacy = intent_input_fingerprint(task, config, client)
    assert legacy is not None
    assert intent_input_fingerprint(task, config, client, planning_client=planner) == legacy
    config.intent.pipeline = "code-first-v2"
    planned = intent_input_fingerprint(task, config, client, planning_client=planner)
    without_plan = intent_input_fingerprint(task, config, client)
    assert planned is not None and without_plan is not None
    assert len({legacy, planned, without_plan}) == 3


def test_full_input_budget_does_not_reserve_profile_output_space_twice():
    from review_bot.application.architecture_review import ModelProfile
    from review_bot.application.code_first_intent_review import run_code_first_review
    from tests.fixtures.code_first_intent_review import CodeFirstModelClient, make_task

    profile = ModelProfile("mock", "input-budget-fixture", max_input_tokens=128, max_output_tokens=64, is_mock=True)
    client = CodeFirstModelClient(profile=profile)
    result = run_code_first_review(make_task(), client)
    assert (result.status, result.reason) == ("completed", "validated")
    assert result.plan["source"] == "program_joint"
    assert len(client.requests) == 3
    # Zero-measurement contract: no metering reserve in ModelExecutionBudget.measure().
    # The mock has no admission_bytes, so admission admits with bound None;
    # accounting is generation-only (response usage), never double-reserved.
    assert client.count_requests == []
    assert all(record.measured_input_tokens is None for record in result.stages)
    assert all(record.token_measurement == "not_attempted" for record in result.stages)
    assert all(record.token_measurement_model_calls == 0 for record in result.stages)
    assert all(record.token_measurement_usage is None for record in result.stages)
    assert result.usage["calls"] == 3
    assert result.usage["token_measurements"] == 0
    assert result.usage["metering_input_tokens"] == 0
    assert result.usage["metering_output_tokens"] == 0
    assert result.usage["input_tokens"] == 300
    assert result.usage["output_tokens"] == 150
    assert result.usage["known_input_tokens"] == 300
    assert result.usage["known_output_tokens"] == 150


def test_planner_receives_measured_full_fit_and_schema_requires_every_prepared_path():
    import json

    from review_bot.application.code_first_intent_review import run_code_first_review
    from tests.fixtures.code_first_intent_review import CodeFirstModelClient, make_task

    task = make_task()
    review, planner = CodeFirstModelClient(), CodeFirstModelClient()
    assert run_code_first_review(task, review, planning_client=planner).status == "completed"
    request = planner.requests[0]
    assert json.loads(request.user_content)["full_input_fits"] is True
    groups = request.response_schema["properties"]["groups"]
    assert groups["minItems"] == groups["maxItems"] == 1
    paths = groups["items"]["properties"]["paths"]
    assert paths["minItems"] == paths["maxItems"] == len(task.reviewed_paths)
    assert set(paths["items"]["enum"]) == set(task.reviewed_paths)
    intent_fields = review.requests[1].response_schema["properties"]["intents"]["items"]["properties"]
    assert intent_fields["alternatives"]["minItems"] == 1
    assert intent_fields["observation_ids"]["minItems"] == 1
    assert intent_fields["citations"]["minItems"] == 1
    variants = review.requests[-1].response_schema["properties"]["findings"]["items"]["oneOf"]
    finding_fields = {item["properties"]["assessment"]["enum"][0]: item["properties"] for item in variants}
    assert finding_fields["insufficient_context"]["missing_context"]["minItems"] == 1
    assert finding_fields["code_defect"]["failure_condition"]["minLength"] == 1


def test_request_for_the_whole_original_already_in_the_group_does_not_trigger_a_no_progress_recheck():
    from review_bot.application.code_first_intent_review import run_code_first_review
    from tests.fixtures.code_first_intent_review import CodeFirstModelClient, make_task

    def already_delivered(payload, body):
        if payload["stage"] == "observe_group":
            return {**body, "requested_paths": list(payload["paths"])}
        return body

    task = make_task()
    client = CodeFirstModelClient(transform=already_delivered)
    result = run_code_first_review(task, client, planning_client=CodeFirstModelClient())
    assert result.status == "completed" and result.relations[0]["status"] == "conflict"
    assert result.group_observations[0]["requested_paths"] == list(task.reviewed_paths)
    assert len(client.requests) == 3
    assert all(record.stage != "recheck_relations" for record in result.stages)
    assert {cite["path"] for cite in result.findings[0]["citations"]} == {"src/value.py", "src/caller.py"}