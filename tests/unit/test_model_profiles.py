"""Explicit profile composition; construction never contacts a model server."""
from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest

from review_bot.application.selection import LightweightModeSelector, ModelModeSelector
from review_bot.config import ModelConfig, load_review_config_from_str
from review_bot.ports import JobStore
from review_bot.runtime import cli, model_clients
from review_bot.runtime.worker import WorkerLoop


@pytest.fixture(autouse=True)
def offline_profiles(monkeypatch):
    monkeypatch.setenv("ME707_BONSAI_BASE_URL", "http://localhost:11435/v1")
    monkeypatch.setenv("ME707_TEV1_BASE_URL", "http://localhost:11437/v1")
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("Construction must remain offline"))
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)


def test_no_profile_does_not_inherit_remote_environment(monkeypatch):
    monkeypatch.setenv("REVIEW_MODEL_PROFILE", "me707-llamacpp")
    assert model_clients.build_review_client(None) is None
    assert model_clients.build_selector_client(None) is None


def test_me707_review_and_selector_use_distinct_explicit_roles():
    settings = ModelConfig(call_timeout_seconds=17, max_output_bytes=12000, max_attempts=2)
    review = model_clients.build_review_client("me707-llamacpp", settings)
    selector = model_clients.build_selector_client("me707-llamacpp", settings)
    assert review.model == "bonsai2-27b"
    assert review.api_url == "http://localhost:11435/v1"
    assert review.profile.provider == "llamacpp"
    assert review.profile.timeout_seconds == 17
    assert review.profile.max_output_bytes == 12000
    assert review.profile.max_attempts == 2
    assert selector.model == "tev1-4b-q8_0"
    assert selector.api_url == "http://localhost:11437/v1"
    assert selector.timeout_seconds == 17


@pytest.mark.parametrize("role,variable", [
    ("review", "ME707_BONSAI_BASE_URL"), ("selector", "ME707_TEV1_BASE_URL"),
])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_remote_profile_requires_each_explicit_endpoint(monkeypatch, role, variable, value):
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    factory = getattr(model_clients, f"build_{role}_client")
    with pytest.raises(ValueError, match=variable):
        factory("me707-llamacpp")


@pytest.mark.parametrize("role", ["review", "selector"])
def test_unsupported_profile_is_not_a_provider_fallback(role):
    with pytest.raises(ValueError, match="Unsupported model profile"):
        getattr(model_clients, f"build_{role}_client")("missing-profile")


@pytest.mark.parametrize("command", ["review", "offline", "intent"])
def test_cli_explicitly_accepts_me707_profile(command):
    arguments = [command, "--model-profile", "me707-llamacpp"]
    if command == "intent":
        arguments += ["--input", "fixed.json", "--output", "private-output"]
    args = cli.create_parser().parse_args(arguments)
    assert args.model_profile == "me707-llamacpp"


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_uses_distinct_selector_only_when_enabled(valid_config_yaml, enabled):
    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: me707-llamacpp",
    ).replace("selector_uses_model: false", f"selector_uses_model: {str(enabled).lower()}"))
    worker = WorkerLoop(config, Mock(spec=JobStore))
    assert worker.model_client.profile.model == "bonsai2-27b"
    if enabled:
        assert isinstance(worker.mode_selector, ModelModeSelector)
        assert worker.mode_selector.selector.client.model == "tev1-4b-q8_0"
    else:
        assert isinstance(worker.mode_selector.selector, LightweightModeSelector)


def test_explicit_selector_injection_does_not_require_tev_endpoint(valid_config_yaml, monkeypatch):
    monkeypatch.delenv("ME707_TEV1_BASE_URL")
    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: me707-llamacpp",
    ).replace("selector_uses_model: false", "selector_uses_model: true"))
    selector = LightweightModeSelector()
    worker = WorkerLoop(config, Mock(spec=JobStore), mode_selector=selector)
    assert worker.mode_selector.selector is selector