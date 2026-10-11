"""Installation eligibility and execution-specific publication defaults."""
from __future__ import annotations

import pytest
import yaml

from review_bot.config import (
    ConfigValidationError,
    ReviewBotConfig,
    load_review_config_from_str,
    validate_active_environment,
)


@pytest.mark.parametrize("mode", ["offline", "github_app"])
@pytest.mark.parametrize("publish", [None, {}, {"a_marker": "<!-- custom:A -->"}])
def test_omitted_publish_mode_uses_execution_default(
    valid_config_yaml: str, mode: str, publish: dict[str, str] | None,
) -> None:
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["mode"] = mode
    if publish is None:
        raw.pop("publish")
    else:
        raw["publish"] = publish

    config, digest = load_review_config_from_str(yaml.safe_dump(raw))

    expected = "comment" if mode == "github_app" else "artifact_only"
    assert config.publish.mode == expected
    assert config.service.installation_allowlist == []
    assert config.service.repository_allowlist == []
    assert ReviewBotConfig.model_validate(config.model_dump()).config_hash == digest


@pytest.mark.parametrize("mode", ["offline", "github_app"])
@pytest.mark.parametrize("publish_mode", ["comment", "artifact_only"])
def test_explicit_publish_mode_wins(
    valid_config_yaml: str, mode: str, publish_mode: str,
) -> None:
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["mode"] = mode
    raw["publish"]["mode"] = publish_mode
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))

    assert config.publish.mode == publish_mode


def test_optional_allowlists_do_not_remove_secret_requirements(valid_config_yaml: str) -> None:
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["mode"] = "github_app"
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))

    with pytest.raises(ConfigValidationError, match="Missing required environment"):
        validate_active_environment(config, env={})

    validate_active_environment(config, env={
        config.service.app_id_env: "1001",
        config.service.app_private_key_file_env: "/private/app.pem",
        config.service.webhook_secret_env: "test-secret",
    })