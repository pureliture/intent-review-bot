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


def test_code_first_cache_cannot_reuse_v1_or_a_different_planning_role(valid_config_yaml, offline_profiles):
    from review_bot.runtime.mode_fingerprints import intent_input_fingerprint
    from review_bot.runtime.model_clients import build_planning_client, build_review_client
    from tests.fixtures.review_b import make_task

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
    from review_bot.application.review_a import ModelProfile
    from review_bot.application.review_b_code import run_code_first_review
    from tests.fixtures.review_b_code import CodeFirstModelClient, make_task

    profile = ModelProfile("mock", "input-budget-fixture", max_input_tokens=128, max_output_tokens=64, is_mock=True)
    client = CodeFirstModelClient(profile=profile)
    result = run_code_first_review(make_task(), client)
    assert (result.status, result.reason) == ("completed", "validated")
    assert result.plan["source"] == "program_joint"
    assert len(client.requests) == 3
    assert all(record.measured_input_tokens == 100 for record in result.stages)