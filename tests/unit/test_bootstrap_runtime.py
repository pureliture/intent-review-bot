from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from review_bot.config import load_review_config_from_str


def load_bootstrap_runtime():
    path = Path(__file__).resolve().parents[2] / "deploy/packer/bootstrap_runtime.py"
    spec = importlib.util.spec_from_file_location("review_bot_bootstrap_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_operator_config_uses_current_timeouts_and_provider_settings():
    module = load_bootstrap_runtime()
    rendered = module.render_operator_config(
        "sha256:" + "a" * 64,
        provider="opencode-go",
        model_id="openai/gpt-4.1-mini",
        context_window_tokens=8192,
    )
    config, _ = load_review_config_from_str(rendered)
    assert config.execution.task_timeout_seconds == 3000
    assert config.agent_engine.timeout_seconds is None
    assert config.agent_engine.max_output_tokens == 4096
    assert config.agent_engine.profile == "fast"
    assert config.agent_engine.max_proxy_calls == 16
    assert config.agent_engine.max_revisions == 0
    assert config.agent_engine.provider == "opencode-go"
    assert config.agent_engine.model_id == "openai/gpt-4.1-mini"
    assert config.agent_engine.context_window_tokens == 8192
    assert "timeout_seconds: 900" not in rendered
    assert "max_output_tokens: 4096" not in rendered


def test_render_operator_config_supports_opt_in_deep_profile():
    module = load_bootstrap_runtime()
    config, _ = load_review_config_from_str(module.render_operator_config("sha256:" + "a" * 64,
        profile="deep"))
    assert config.agent_engine.profile == "deep"
    assert config.agent_engine.max_proxy_calls == 40
    assert config.agent_engine.max_output_tokens is None
    assert config.agent_engine.max_revisions == 1
    with pytest.raises(ValueError, match="invalid_agent_profile"):
        module.render_operator_config("sha256:" + "a" * 64, profile="other")


def test_render_operator_config_defaults_agent_provider_and_omits_context_window():
    module = load_bootstrap_runtime()
    rendered = module.render_operator_config("sha256:" + "b" * 64)
    config, _ = load_review_config_from_str(rendered)
    assert config.agent_engine.provider == "openrouter"
    assert config.agent_engine.model_id == "stealth/crouton-alpha"
    assert config.agent_engine.context_window_tokens is None
    assert "context_window_tokens:" not in rendered


@pytest.mark.parametrize(
    "environ",
    [
        {"REVIEW_BOT_AGENT_PROVIDER": "unsupported"},
        {"REVIEW_BOT_AGENT_MODEL_ID": "../bad"},
        {"REVIEW_BOT_AGENT_MODEL_ID": "a" * 129},
        {"REVIEW_BOT_AGENT_CONTEXT_WINDOW_TOKENS": "not-a-number"},
        {"REVIEW_BOT_AGENT_CONTEXT_WINDOW_TOKENS": "1023"},
        {"REVIEW_BOT_AGENT_CONTEXT_WINDOW_TOKENS": "10000001"},
    ],
)
def test_agent_environment_validation_rejects_invalid_values(environ):
    module = load_bootstrap_runtime()
    with pytest.raises(ValueError):
        module.agent_settings_from_environment(environ)


def test_agent_environment_validation_uses_defaults_for_empty_values():
    module = load_bootstrap_runtime()
    assert module.agent_settings_from_environment({
        "REVIEW_BOT_AGENT_PROVIDER": "",
        "REVIEW_BOT_AGENT_MODEL_ID": "",
        "REVIEW_BOT_AGENT_CONTEXT_WINDOW_TOKENS": "",
    }) == ("openrouter", "stealth/crouton-alpha", None)
