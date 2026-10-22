"""Unit tests for strict YAML loader, Pydantic schemas, and canonical hashing."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from review_bot.config import (
    ConfigAliasError,
    ConfigDuplicateKeyError,
    ConfigSecurityError,
    ConfigSyntaxError,
    ConfigValidationError,
    compute_config_hash,
    load_review_config,
    load_review_config_from_str,
    load_target_repository_config,
    validate_active_environment,
)

EXPECTED_EXAMPLE_HASH = "b0065ca42fc04dd13933fe684ecbf30183d06366b4751adc949a921a2a088bc5"


def test_load_example_config(valid_config_yaml: str) -> None:
    config, config_hash = load_review_config_from_str(valid_config_yaml)
    assert config.schema_version == 2
    assert config.codebase.product == "github-review-bot"
    assert config.codebase.python_package == "review_bot"
    assert config.execution.mode == "offline"
    assert config.architecture.cpd_minimum_tokens == 100
    assert config.architecture.run_target_build_scripts is False
    assert config.artifacts.publish_private_data_to_public_ci is False
    assert config_hash == EXPECTED_EXAMPLE_HASH


def test_load_example_config_from_file() -> None:
    specs_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "specs"
        / "20261002-github-review-bot"
        / "review-config.example.yaml"
    )
    if specs_path.exists():
        config, config_hash = load_review_config(specs_path)
        assert compute_config_hash(config) == config_hash == EXPECTED_EXAMPLE_HASH


def test_duplicate_key_rejection() -> None:
    # Root duplicate key
    root_dup = "schema_version: 2\nschema_version: 2\n"
    with pytest.raises(ConfigDuplicateKeyError) as exc_info:
        load_review_config_from_str(root_dup)
    assert "schema_version" in str(exc_info.value)

    # Nested duplicate key
    nested_dup = """schema_version: 2
codebase:
  product: github-review-bot
  product: duplicate-bot
"""
    with pytest.raises(ConfigDuplicateKeyError) as exc_info:
        load_review_config_from_str(nested_dup)
    assert "product" in str(exc_info.value)


def test_alias_rejection() -> None:
    alias_yaml = """
schema_version: 2
common: &common_anchor
  product: github-review-bot
codebase: *common_anchor
"""
    with pytest.raises(ConfigAliasError):
        load_review_config_from_str(alias_yaml)


def test_non_dict_root_rejection() -> None:
    with pytest.raises(ConfigSyntaxError):
        load_review_config_from_str("just a scalar string")

    with pytest.raises(ConfigSyntaxError):
        load_review_config_from_str("- item 1\n- item 2\n")


def test_schema_version_validation(valid_config_yaml: str) -> None:
    bad_version = valid_config_yaml.replace("schema_version: 2", "schema_version: 1")
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_version)

    string_version = valid_config_yaml.replace("schema_version: 2", 'schema_version: "2"')
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(string_version)


def test_extra_key_rejection(valid_config_yaml: str) -> None:
    root_extra = valid_config_yaml + "\nunexpected_root_key: 123\n"
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(root_extra)

    nested_extra = valid_config_yaml.replace(
        "layout: src", "layout: src\n  unexpected_nested: true"
    )
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(nested_extra)


def test_range_and_limit_validation(valid_config_yaml: str) -> None:
    bad_concurrency = valid_config_yaml.replace(
        "worker_concurrency: 1", "worker_concurrency: 0"
    )
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_concurrency)

    bad_timeout = valid_config_yaml.replace(
        "task_timeout_seconds: 900", "task_timeout_seconds: -1"
    )
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_timeout)

    bad_memory = valid_config_yaml.replace("memory_mib: 2048", "memory_mib: 64")
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_memory)

    bad_cpd = valid_config_yaml.replace(
        "cpd_minimum_tokens: 100", "cpd_minimum_tokens: 5"
    )
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_cpd)


def test_security_invariants(valid_config_yaml: str) -> None:
    # Target build scripts must NOT be enabled
    script_exploit = valid_config_yaml.replace(
        "run_target_build_scripts: false", "run_target_build_scripts: true"
    )
    with pytest.raises(ConfigSecurityError) as exc_info:
        load_review_config_from_str(script_exploit)
    assert "run_target_build_scripts MUST be false for security" in str(exc_info.value)

    # Public CI publishing must NOT be enabled
    leak_exploit = valid_config_yaml.replace(
        "publish_private_data_to_public_ci: false",
        "publish_private_data_to_public_ci: true",
    )
    with pytest.raises(ConfigSecurityError) as exc_info:
        load_review_config_from_str(leak_exploit)
    assert "publish_private_data_to_public_ci MUST be false for security" in str(
        exc_info.value
    )


def test_env_var_pattern_validation(valid_config_yaml: str) -> None:
    bad_env_hyphen = valid_config_yaml.replace(
        "app_id_env: REVIEW_BOT_APP_ID", "app_id_env: REVIEW-BOT-APP-ID"
    )
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_env_hyphen)

    bad_env_digit = valid_config_yaml.replace(
        "app_id_env: REVIEW_BOT_APP_ID", "app_id_env: 123_APP_ID"
    )
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(bad_env_digit)


def test_target_repository_config_subset() -> None:
    valid_target = """
analysis_profile: java21-jdk-only-v1
source_roots:
  - src/main/java
  - src/custom/java
architecture_rule_paths:
  - rules/arch.xml
"""
    target_cfg = load_target_repository_config(valid_target)
    assert len(target_cfg.source_roots) == 2
    assert target_cfg.architecture_rule_paths == ["rules/arch.xml"]

    # Target config attempting to inject runtime or credentials must be rejected
    malicious_target = """
analysis_profile: java21-jdk-only-v1
isolation:
  runtime: host
"""
    with pytest.raises(ConfigValidationError):
        load_target_repository_config(malicious_target)


def test_config_hash_invariance(valid_config_yaml: str) -> None:
    _cfg1, hash1 = load_review_config_from_str(valid_config_yaml)
    # Parse YAML to dict, invert dict keys order
    raw_dict = yaml.safe_load(valid_config_yaml)
    reversed_dict = dict(reversed(list(raw_dict.items())))
    reversed_yaml = yaml.dump(reversed_dict)
    _cfg2, hash2 = load_review_config_from_str(reversed_yaml)

    assert hash1 == hash2 == EXPECTED_EXAMPLE_HASH


def test_active_environment_validation(valid_config_yaml: str) -> None:
    # offline mode passes with empty env
    cfg, _ = load_review_config_from_str(valid_config_yaml)
    validate_active_environment(cfg, env={})

    # Installation-based eligibility permits empty operator restrictions.
    gh_app_yaml = valid_config_yaml.replace("mode: offline", "mode: github_app")
    unrestricted, _ = load_review_config_from_str(gh_app_yaml)
    assert unrestricted.service.installation_allowlist == []
    assert unrestricted.service.repository_allowlist == []
    with pytest.raises(ConfigValidationError, match="Missing required environment"):
        validate_active_environment(unrestricted, env={})

    # github_app mode with allowlists but missing env vars fails validate_active_environment
    gh_app_with_allowlist = gh_app_yaml.replace(
        "installation_allowlist: []", "installation_allowlist: [123]"
    ).replace("repository_allowlist: []", "repository_allowlist: ['org/repo']")
    cfg_app, _ = load_review_config_from_str(gh_app_with_allowlist)

    with pytest.raises(ConfigValidationError) as exc:
        validate_active_environment(cfg_app, env={})
    assert "Missing required environment variables" in str(exc.value)

    # with all env vars populated passes
    env_vars = {
        "REVIEW_BOT_APP_ID": "1001",
        "REVIEW_BOT_APP_PRIVATE_KEY_FILE": "/path/to/key.pem",
        "REVIEW_BOT_WEBHOOK_SECRET": "secret123",
    }
    validate_active_environment(cfg_app, env=env_vars)
