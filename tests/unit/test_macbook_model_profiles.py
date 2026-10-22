"""MacBook profile composition remains offline and separate from me707/Ollama."""
from __future__ import annotations

from unittest.mock import Mock

import httpx
import pytest

from review_bot.adapters.openrouter_chat import OpenRouterReviewClient
from review_bot.application.selection import LightweightModeSelector, ModelModeSelector
from review_bot.config import ModelConfig, load_review_config_from_str
from review_bot.ports import JobStore
from review_bot.runtime import cli, model_clients
from review_bot.runtime.mode_fingerprints import intent_input_fingerprint
from review_bot.runtime.worker import WorkerLoop
from tests.fixtures.intent_review import make_task


@pytest.fixture(autouse=True)
def offline_macbook(monkeypatch, tmp_path):
    monkeypatch.delenv("MACBOOK_BEAM_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("MACBOOK_BEAM_BASE_URL", raising=False)
    monkeypatch.delenv("MACBOOK_TEV4B_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)
    monkeypatch.setenv("MACBOOK_LFM_BASE_URL", "http://127.0.0.1:18135/v1")
    monkeypatch.setenv("MACBOOK_TEV1_BASE_URL", "http://127.0.0.1:18137/v1")
    monkeypatch.setenv("REVIEW_BOT_MACBOOK_MODEL_STATE_DIR", str(tmp_path / "mac-admission"))
    monkeypatch.setenv("ME707_BONSAI_BASE_URL", "http://ubuntu-ms-7e07:11435/v1")
    monkeypatch.setenv("ME707_TEV1_BASE_URL", "http://ubuntu-ms-7e07:11437/v1")
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("Construction must remain offline"))
    monkeypatch.setattr(WorkerLoop, "_setup_signals", lambda self: None)


def test_macbook_profile_has_distinct_local_roles_and_limits():
    settings = ModelConfig(call_timeout_seconds=17, max_output_bytes=12000, max_attempts=2, max_concurrency=2)
    review = model_clients.build_review_client("macbook-llamacpp", settings)
    selector = model_clients.build_selector_client("macbook-llamacpp", settings)
    assert review.model == "lfm2.5-2.6b-q8_0"
    assert review.api_url == "http://127.0.0.1:18135/v1"
    assert review.profile.provider == "llamacpp"
    assert review.profile.max_input_tokens == 32768 - 4096
    assert review.profile.max_output_tokens == 4096
    assert review.profile.timeout_seconds == 17
    assert review.profile.max_output_bytes == 12000
    assert review.profile.max_attempts == 2
    assert selector.model == "tev1-0.8b-q8_0"
    assert selector.api_url == "http://127.0.0.1:18137/v1"
    assert selector.timeout_seconds == 17


def test_explicit_granite_profile_keeps_local_admission_and_existing_tev(monkeypatch, tmp_path):
    monkeypatch.setenv("MACBOOK_GRANITE_BASE_URL", "http://127.0.0.1:18139/v1")
    review = model_clients.build_review_client("macbook-granite-llamacpp", ModelConfig(max_output_bytes=65536))
    planning = model_clients.build_planning_client("macbook-granite-llamacpp")
    assert review.model == "granite4.2"
    assert (review.profile.max_input_tokens, review.profile.max_output_tokens) == (49152, 16384)
    assert review._context_tokens == 65536 and review._server_generation_slots == 1
    assert review.request_admission.db_path == tmp_path / "mac-admission/requests.sqlite"
    assert planning.model == "tev1-0.8b-q8_0"
    assert planning.profile.max_input_tokens == 28672
    monkeypatch.delenv("MACBOOK_GRANITE_BASE_URL")
    with pytest.raises(ValueError, match="MACBOOK_GRANITE_BASE_URL"):
        model_clients.build_review_client("macbook-granite-llamacpp")


@pytest.mark.parametrize("role,variable", [
    ("review", "MACBOOK_LFM_BASE_URL"), ("selector", "MACBOOK_TEV1_BASE_URL"),
])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_macbook_requires_own_endpoints_without_remote_fallback(monkeypatch, role, variable, value):
    if value is None:
        monkeypatch.delenv(variable)
    else:
        monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        getattr(model_clients, f"build_{role}_client")("macbook-llamacpp")


@pytest.mark.parametrize("role,variable", [
    ("review", "MACBOOK_LFM_BASE_URL"), ("selector", "MACBOOK_TEV1_BASE_URL"),
])
@pytest.mark.parametrize("url", ["http://ubuntu-ms-7e07:11435/v1", "http://example.com/v1", "http://192.0.2.1/v1"])
def test_macbook_rejects_remote_endpoints(monkeypatch, role, variable, url):
    monkeypatch.setenv(variable, url)
    with pytest.raises(ValueError):
        getattr(model_clients, f"build_{role}_client")("macbook-llamacpp")


def test_macbook_environment_cannot_change_me707_roles(monkeypatch):
    monkeypatch.setenv("MACBOOK_LFM_BASE_URL", "not-a-url")
    monkeypatch.setenv("MACBOOK_TEV1_BASE_URL", "not-a-url")
    assert model_clients.build_review_client("me707-llamacpp").model == "bonsai2-27b"
    assert model_clients.build_selector_client("me707-llamacpp").model == "tev1-4b-q8_0"
    assert model_clients.build_review_client("ollama-local-128k").model == "granite4.2:128k"


@pytest.mark.parametrize("command", ["review", "offline", "intent"])
def test_cli_accepts_explicit_macbook_profile(command):
    arguments = [command, "--model-profile", "macbook-llamacpp"]
    if command == "intent":
        arguments += ["--input", "fixed.json", "--output", "private-output"]
    assert cli.create_parser().parse_args(arguments).model_profile == "macbook-llamacpp"


@pytest.mark.parametrize("enabled", [False, True])
def test_worker_composes_macbook_without_changing_review_engine(valid_config_yaml, enabled):
    config, _ = load_review_config_from_str(valid_config_yaml.replace(
        "profile: null", "profile: macbook-llamacpp",
    ).replace("selector_uses_model: false", f"selector_uses_model: {str(enabled).lower()}"))
    worker = WorkerLoop(config, Mock(spec=JobStore))
    assert worker.model_client.model == "lfm2.5-2.6b-q8_0"
    assert worker.model_client.request_cancelled == worker._stop_event.is_set
    if enabled:
        assert isinstance(worker.mode_selector, ModelModeSelector)
        assert worker.mode_selector.selector.client.model == "tev1-0.8b-q8_0"
    else:
        assert isinstance(worker.mode_selector.selector, LightweightModeSelector)


def test_macbook_and_me707_cannot_reuse_each_others_intent_fingerprint(valid_config_yaml):
    config, _ = load_review_config_from_str(valid_config_yaml)
    task = make_task()
    local = intent_input_fingerprint(task, config, model_clients.build_review_client("macbook-llamacpp"))
    remote = intent_input_fingerprint(task, config, model_clients.build_review_client("me707-llamacpp"))
    assert local is not None and remote is not None and local != remote


def test_no_profile_does_not_enable_macbook_from_environment(monkeypatch):
    monkeypatch.setenv("REVIEW_MODEL_PROFILE", "macbook-llamacpp")
    assert model_clients.build_review_client(None) is None
    assert model_clients.build_selector_client(None) is None


def test_explicit_beam_tev4b_profile_composes_distinct_models_without_http(monkeypatch, tmp_path, valid_config_yaml):
    # Beam bridge is retired: review is OpenRouter (no HTTP at construction),
    # tev4b roles stay local loopback. MACBOOK_BEAM_BASE_URL is not consulted.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MACBOOK_TEV4B_BASE_URL", "http://127.0.0.1:18139/v1")
    monkeypatch.delenv("MACBOOK_BEAM_BASE_URL", raising=False)
    monkeypatch.delenv("MACBOOK_BEAM_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)
    settings = ModelConfig(call_timeout_seconds=17, max_output_bytes=1048576)
    profile = model_clients.BEAM_CLOUD_TEV4B_PROFILE
    review = model_clients.build_review_client(profile, settings)
    planning = model_clients.build_planning_client(profile, settings)
    selector = model_clients.build_selector_client(profile, settings)
    perspectives = model_clients.build_intent_selection_client(profile, settings)
    assert isinstance(review, OpenRouterReviewClient)
    assert review.model == "google/gemma-4-31b-it:free"
    assert review.profile.provider == "openrouter"
    assert review.profile.model == "google/gemma-4-31b-it:free"
    assert (review.profile.max_input_tokens, review.profile.max_output_tokens) == (98394, 32678)
    assert review.profile.timeout_seconds == 17
    assert review.profile.max_output_bytes == 1048576
    assert review.profile.max_attempts == 1
    assert review.timeout_seconds == 17
    assert not hasattr(review, "supports_native_input_measurement")
    assert not hasattr(review, "count_input_tokens")
    # OPENROUTER_REVIEW_MODEL override composes without HTTP.
    monkeypatch.setenv("OPENROUTER_REVIEW_MODEL", "custom/override-model:free")
    overridden = model_clients.build_review_client(profile, settings)
    assert overridden.model == "custom/override-model:free"
    assert overridden.profile.model == "custom/override-model:free"
    assert (overridden.profile.max_input_tokens, overridden.profile.max_output_tokens) == (98394, 32678)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL")
    # Custom ModelConfig app budget propagates to the OpenRouter review client.
    budgeted = model_clients.build_review_client(
        profile, ModelConfig(review_input_tokens=50000, review_output_tokens=10000))
    assert (budgeted.profile.max_input_tokens, budgeted.profile.max_output_tokens) == (50000, 10000)
    assert budgeted.model == "google/gemma-4-31b-it:free"
    for client in (planning, selector, perspectives):
        assert client.model == "tev1-4b-q8_0"
        assert client._context_tokens == 131072 and client._server_generation_slots == 1
        assert client.api_url == "http://127.0.0.1:18139/v1"
    assert planning.profile.max_input_tokens == 126976
    assert planning.profile.max_output_tokens == 4096
    for client in (review, planning, selector, perspectives):
        assert client.request_admission.db_path == tmp_path / "mac-admission/requests.sqlite"
    config, _ = load_review_config_from_str(valid_config_yaml.replace("profile: null", "profile: " + profile))
    worker = WorkerLoop(config, Mock(spec=JobStore))
    assert isinstance(worker.model_client, OpenRouterReviewClient)
    assert worker.model_client.model == "google/gemma-4-31b-it:free"
    assert worker.model_client.profile.max_input_tokens == 98394
    assert worker.model_client.profile.max_output_tokens == 32678
    assert worker.model_client.request_cancelled == worker._stop_event.is_set
    # Retired Beam context cap never silently builds; the legacy base URL is ignored.
    monkeypatch.setenv("MACBOOK_BEAM_BASE_URL", "http://127.0.0.1:11434")
    assert model_clients.build_review_client(profile, settings).model == "google/gemma-4-31b-it:free"
    monkeypatch.delenv("MACBOOK_BEAM_BASE_URL")
    monkeypatch.setenv("MACBOOK_BEAM_CONTEXT_TOKENS", "131072")
    with pytest.raises(ValueError, match="MACBOOK_BEAM_CONTEXT_TOKENS"):
        model_clients.build_review_client(profile)
    monkeypatch.delenv("MACBOOK_BEAM_CONTEXT_TOKENS")
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        model_clients.build_review_client(profile)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("MACBOOK_TEV4B_BASE_URL")
    assert model_clients.build_planning_client(profile) is None
    with pytest.raises(ValueError, match="MACBOOK_TEV4B_BASE_URL"):
        model_clients.build_selector_client(profile)


def test_beam_operating_context_cap_reaches_worker_and_fingerprint_without_lowering_tev(monkeypatch, valid_config_yaml):
    # Retired Beam cap is replaced by the ModelConfig app budget (no HTTP at construction).
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("MACBOOK_TEV4B_BASE_URL", "http://127.0.0.1:18139/v1")
    monkeypatch.delenv("MACBOOK_BEAM_BASE_URL", raising=False)
    monkeypatch.delenv("MACBOOK_BEAM_CONTEXT_TOKENS", raising=False)
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)
    profile = model_clients.BEAM_CLOUD_TEV4B_PROFILE
    default = model_clients.build_review_client(profile)
    assert isinstance(default, OpenRouterReviewClient)
    assert default.model == "google/gemma-4-31b-it:free"
    assert default.profile.provider == "openrouter"
    assert default.profile.max_input_tokens == 98394
    assert default.profile.max_output_tokens == 32678
    assert not hasattr(default, "count_input_tokens")
    assert not hasattr(default, "supports_native_input_measurement")
    custom = model_clients.build_review_client(
        profile, ModelConfig(review_input_tokens=50000, review_output_tokens=10000))
    assert custom.profile.max_input_tokens == 50000
    assert custom.profile.max_output_tokens == 10000
    planner = model_clients.build_planning_client(profile)
    assert planner.profile.max_input_tokens == 126976
    assert planner.profile.max_output_tokens == 4096
    config, _ = load_review_config_from_str(valid_config_yaml.replace("profile: null", "profile: " + profile))
    worker = WorkerLoop(config, Mock(spec=JobStore))
    assert worker.model_client.profile == default.profile
    assert worker.model_client.model == "google/gemma-4-31b-it:free"
    task = make_task()
    default_key = intent_input_fingerprint(task, config, default, planning_client=planner)
    custom_key = intent_input_fingerprint(task, config, custom, planning_client=planner)
    assert default_key is not None and custom_key is not None and default_key != custom_key
    # Any retired cap value never silently falls back to a native capacity.
    monkeypatch.setenv("MACBOOK_BEAM_CONTEXT_TOKENS", "131072")
    with pytest.raises(ValueError, match="MACBOOK_BEAM_CONTEXT_TOKENS"):
        model_clients.build_review_client(profile)


@pytest.mark.parametrize("value", ["", "not-an-integer", "32678", "1048577"])
def test_invalid_beam_operating_context_never_silently_uses_native_capacity(monkeypatch, value):
    # MACBOOK_BEAM_CONTEXT_TOKENS is retired: any set value must fail closed.
    # Empty string counts as unset and keeps the default OpenRouter app budget.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_REVIEW_MODEL", raising=False)
    monkeypatch.setenv("MACBOOK_BEAM_CONTEXT_TOKENS", value)
    if value == "":
        review = model_clients.build_review_client(model_clients.BEAM_CLOUD_TEV4B_PROFILE)
        assert isinstance(review, OpenRouterReviewClient)
        assert (review.profile.max_input_tokens, review.profile.max_output_tokens) == (98394, 32678)
        return
    with pytest.raises(ValueError, match="MACBOOK_BEAM_CONTEXT_TOKENS"):
        model_clients.build_review_client(model_clients.BEAM_CLOUD_TEV4B_PROFILE)
