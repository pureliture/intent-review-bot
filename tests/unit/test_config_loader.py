"""Unit tests for Feature 27: Strict YAML Configuration Loader.

Requirements:
- ORIGINAL_REQUEST §R1 & spec_miner_m1_3:
  - Safe YAML loader (yaml.SafeLoader) customized to forbid duplicate dictionary keys.
  - Full schema validation for review-config.example.yaml (schema_version 2).
  - Reject unknown keys, invalid data types, negative numbers, or invalid enum values.
  - Compute canonical JSON digest (config_hash).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

try:
    from review_bot.config import ConfigValidationError, load_config  # type: ignore
except ImportError:
    load_config = None  # type: ignore
    ConfigValidationError = Exception  # type: ignore


@pytest.fixture
def config_loader() -> Callable[[str], Any]:
    if load_config is None:
        pytest.skip("Configuration loader review_bot.config not yet implemented by Worker")
    return load_config


# ==============================================================================
# Tier 1: Happy Path Tests
# ==============================================================================


class TestConfigLoaderTier1:
    def test_load_authoritative_example_config(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        cfg = config_loader(valid_config_yaml)
        assert cfg is not None
        assert getattr(cfg, "schema_version", None) == 2 or cfg["schema_version"] == 2

    def test_canonical_config_hash_computed(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        cfg = config_loader(valid_config_yaml)
        config_hash = getattr(cfg, "config_hash", None) or (
            cfg.get("config_hash") if isinstance(cfg, dict) else None
        )
        assert config_hash is not None
        assert isinstance(config_hash, str)
        assert len(config_hash) == 64  # SHA-256 hex string

    def test_execution_settings_parsed(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        cfg = config_loader(valid_config_yaml)
        execution = getattr(cfg, "execution", None) or cfg["execution"]
        mode = getattr(execution, "mode", None) or execution["mode"]
        assert mode == "offline"

    def test_architecture_settings_parsed(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        cfg = config_loader(valid_config_yaml)
        arch = getattr(cfg, "architecture", None) or cfg["architecture"]
        tool_ver = getattr(arch, "tool_version", None) or arch["tool_version"]
        assert tool_ver == "7.3.0"
        cpd_tokens = getattr(arch, "cpd_minimum_tokens", None) or arch["cpd_minimum_tokens"]
        assert cpd_tokens == 100

    def test_publish_marker_parsed(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        cfg = config_loader(valid_config_yaml)
        pub = getattr(cfg, "publish", None) or cfg["publish"]
        marker = getattr(pub, "a_marker", None) or pub["a_marker"]
        assert marker == "<!-- review-bot:architecture:v1 -->"


# ==============================================================================
# Tier 2: Validation, Strictness & Error Handling
# ==============================================================================


class TestConfigLoaderTier2:
    def test_rejects_duplicate_dictionary_keys(
        self, config_loader: Callable, duplicate_keys_yaml: str
    ) -> None:
        with pytest.raises((ValueError, ConfigValidationError, KeyError)):
            config_loader(duplicate_keys_yaml)

    def test_rejects_unknown_top_level_keys(
        self, config_loader: Callable, unknown_keys_yaml: str
    ) -> None:
        with pytest.raises((ValueError, ConfigValidationError)):
            config_loader(unknown_keys_yaml)

    def test_rejects_invalid_schema_version(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        invalid_ver = valid_config_yaml.replace("schema_version: 2", "schema_version: 99")
        with pytest.raises((ValueError, ConfigValidationError)):
            config_loader(invalid_ver)

    def test_rejects_negative_timeouts(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        neg_timeout = valid_config_yaml.replace(
            "task_timeout_seconds: 900", "task_timeout_seconds: -10"
        )
        with pytest.raises((ValueError, ConfigValidationError)):
            config_loader(neg_timeout)

    def test_rejects_invalid_enum_values(
        self, config_loader: Callable, valid_config_yaml: str
    ) -> None:
        bad_enum = valid_config_yaml.replace(
            "mode: offline", "mode: unsupported_remote_mode"
        )
        with pytest.raises((ValueError, ConfigValidationError)):
            config_loader(bad_enum)

    def test_empty_yaml_rejected(self, config_loader: Callable) -> None:
        with pytest.raises((ValueError, ConfigValidationError)):
            config_loader("")
