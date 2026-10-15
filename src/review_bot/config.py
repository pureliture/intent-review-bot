"""Strict YAML configuration loader, Pydantic schemas, and canonical hashing."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TextIO

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from review_bot.domain.markers import DEFAULT_B_MARKER, valid_intent_marker

ENV_VAR_PATTERN = r"^[A-Z_][A-Z0-9_]*$"


# ==============================================================================
# Exceptions
# ==============================================================================


class ConfigError(ValueError):
    """Base exception for configuration errors."""


class ConfigSyntaxError(ConfigError):
    """Raised when configuration contains invalid syntax or non-mapping root."""


class ConfigDuplicateKeyError(ConfigError):
    """Raised when duplicate mapping keys are encountered in YAML."""


class ConfigAliasError(ConfigError):
    """Raised when YAML aliases or anchors are encountered."""


class ConfigValidationError(ConfigError):
    """Raised when configuration fails schema validation."""


class ConfigSecurityError(ConfigError):
    """Raised when security invariants are violated."""


# ==============================================================================
# StrictSafeLoader
# ==============================================================================


class StrictSafeLoader(yaml.SafeLoader):
    """PyYAML SafeLoader subclass intercepting mapping construction and aliases."""

    def compose_node(self, parent: Any, index: Any) -> yaml.Node:
        if self.check_event(yaml.events.AliasEvent):
            event = self.get_event()
            line = event.start_mark.line + 1 if event.start_mark else 0
            col = event.start_mark.column + 1 if event.start_mark else 0
            raise ConfigAliasError(
                f"YAML aliases are forbidden in strict configuration: alias '*{event.anchor}' at line {line}, column {col}"
            )
        event = self.peek_event()
        if getattr(event, "anchor", None) is not None:
            line = event.start_mark.line + 1 if event.start_mark else 0
            col = event.start_mark.column + 1 if event.start_mark else 0
            raise ConfigAliasError(
                f"YAML anchors are forbidden in strict configuration: anchor '&{event.anchor}' at line {line}, column {col}"
            )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise ConfigSyntaxError(
                f"Expected mapping node, but found {node.id} at line {node.start_mark.line + 1}"
            )
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                raise ConfigSyntaxError(
                    f"Found unhashable key at line {key_node.start_mark.line + 1}"
                ) from exc
            if key in mapping:
                line = key_node.start_mark.line + 1
                col = key_node.start_mark.column + 1
                raise ConfigDuplicateKeyError(
                    f"Duplicate key '{key}' found at line {line}, column {col}"
                )
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value
        return mapping


# ==============================================================================
# Pydantic Schemas
# ==============================================================================


class CodebaseConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    product: str = Field(min_length=1)
    python_package: str = Field(min_length=1)
    python_min_version: str = Field(min_length=1)
    layout: Literal["src"] = "src"


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    mode: Literal["offline", "github_app"] = "offline"
    backend: Literal["github_app", "github_api", "mock"] = "github_app"
    available_modes: list[str] = Field(min_length=1)
    select_modes_for_all_prs: bool = True
    run_only_selected_modes: bool = True
    selector_policy: Literal["all_available"] = "all_available"
    selector_uses_model: bool = False
    worker_concurrency: int = Field(ge=1, le=2, default=1)
    task_timeout_seconds: int = Field(ge=1, default=900)
    pr_actions: list[str] = Field(
        default_factory=lambda: [
            "opened",
            "synchronize",
            "reopened",
            "edited",
            "labeled",
            "unlabeled",
            "ready_for_review",
            "closed",
        ]
    )


class ServiceConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    receiver: Literal["fastapi_uvicorn"] = "fastapi_uvicorn"
    worker: Literal["synchronous", "asynchronous"] = "synchronous"
    job_store: Literal["sqlite"] = "sqlite"
    sqlite_wal: bool = True
    lease_and_generation_fencing: bool = True
    webhook_max_bytes: int = Field(gt=0, default=10485760)
    state_root_env: str = Field(pattern=ENV_VAR_PATTERN, default="REVIEW_BOT_STATE_DIR")
    app_id_env: str = Field(pattern=ENV_VAR_PATTERN, default="REVIEW_BOT_APP_ID")
    app_private_key_file_env: str = Field(
        pattern=ENV_VAR_PATTERN, default="REVIEW_BOT_APP_PRIVATE_KEY_FILE"
    )
    webhook_secret_env: str = Field(pattern=ENV_VAR_PATTERN, default="REVIEW_BOT_WEBHOOK_SECRET")
    installation_allowlist: list[int] = Field(default_factory=list)
    repository_allowlist: list[str] = Field(default_factory=list)


class LegacyMigrationConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    enabled: bool = False
    marker: str = "<!-- seulgi-review:A -->"
    exact_title: str = "## 리팩토링 신호 — 이번 PR 변경 파일 기준"


class PublishConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    mode: Literal["artifact_only", "comment"] = "artifact_only"
    separate_comments_per_task: bool = True
    a_marker: str = "<!-- review-bot:architecture:v1 -->"
    b_marker: str = DEFAULT_B_MARKER
    b_comments_enabled: bool = False
    draft_comments: bool = False
    automatic_approval: bool = False
    automatic_request_changes: bool = False
    recheck_before_and_after_write: bool = True
    legacy_migration: LegacyMigrationConfig | None = None

    @model_validator(mode="after")
    def validate_b_marker(self) -> PublishConfig:
        if not valid_intent_marker(self.b_marker):
            raise ValueError("invalid B publication marker")
        if self.a_marker in self.b_marker or self.b_marker in self.a_marker:
            raise ValueError("A and B publication markers must be distinct")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    profile: str | None = None
    profile_env: str = Field(pattern=ENV_VAR_PATTERN, default="REVIEW_MODEL_PROFILE")
    unconfigured_action: Literal["not_run", "fail"] = "not_run"
    input_policy: Literal["full_or_not_run"] = "full_or_not_run"
    overflow_action: Literal["not_run", "fail"] = "not_run"
    require_token_count: bool = True
    allow_summarization: bool = False
    allow_truncation: bool = False
    allow_automatic_splitting: bool = False
    max_concurrency: int = Field(ge=1, le=2, default=1)
    call_timeout_seconds: int = Field(ge=1, default=60)
    max_attempts: int = Field(ge=1, default=1)
    max_output_bytes: int = Field(ge=1, default=32768)
    required_profile_limits: list[str] = Field(
        default_factory=lambda: [
            "max_input_bytes",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
        ]
    )


class IsolationConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    runtime: Literal["rootless_docker", "subprocess"] = "rootless_docker"
    network: Literal["none"] = "none"
    read_only_root: bool = True
    no_new_privileges: bool = True
    cpus: int = Field(ge=1, default=2)
    memory_mib: int = Field(ge=128, default=2048)
    pids: int = Field(ge=16, default=128)


class ArchitectureConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    review_type: Literal["A"] = "A"
    analysis_profile: str = "java21-jdk-only-v1"
    tool_version: str = "7.3.0"
    runtime_java_version: str = "21"
    language_version: str = "21"
    cpd_minimum_tokens: int = Field(ge=10, default=100)
    encoding: str = "UTF-8"
    type_resolution: Literal["jdk_only"] = "jdk_only"
    ruleset_resource: str = "pmd/ruleset.xml"
    tools_lock_path: str = "tooling/tools.lock.json"
    tool_image_digest: str | None = None
    run_target_build_scripts: bool = False
    tool_timeout_seconds: int = Field(ge=1, default=120)
    isolation: IsolationConfig = Field(default_factory=IsolationConfig)

    @field_validator("run_target_build_scripts")
    @classmethod
    def validate_no_target_scripts(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("run_target_build_scripts MUST be false for security")
        return v


class RepositorySettingsConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    analysis_profile: str = "java21-jdk-only-v1"
    source_roots: list[str] = Field(default_factory=lambda: ["src/main/java"])
    architecture_rule_paths: list[str] = Field(default_factory=list)


class IntentConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    pipeline: Literal["declared-v1", "code-first-v2"] = "declared-v1"
    context_paths: list[str] = Field(default_factory=list, max_length=5000)

    @field_validator("context_paths")
    @classmethod
    def validate_context_paths(cls, paths: list[str]) -> list[str]:
        from review_bot.application.intent_context import validate_intent_context_paths

        validate_intent_context_paths(tuple(paths))
        return paths


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    base_strategy: Literal["unique_merge_base"] = "unique_merge_base"
    record_target_base: bool = True
    ambiguous_base_action: Literal["incomparable"] = "incomparable"


class CodeInputConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    versions: list[str] = Field(default_factory=lambda: ["comparison_base", "head"])
    include_all_source_tests_and_related_config: bool = True
    exclusion_policy: Literal["trusted_builtin"] = "trusted_builtin"
    incomplete_required_input_action: Literal["not_run"] = "not_run"


class ArtifactsConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    visibility: Literal["private"] = "private"
    retention_days: int = Field(ge=1, default=30)
    max_archive_bytes: int = Field(gt=0, default=104857600)
    max_unpacked_bytes: int = Field(gt=0, default=402653184)
    max_file_count: int = Field(gt=0, default=20000)
    publish_private_data_to_public_ci: bool = False

    @field_validator("publish_private_data_to_public_ci")
    @classmethod
    def validate_no_ci_publish(cls, v: bool) -> bool:
        if v is not False:
            raise ValueError("publish_private_data_to_public_ci MUST be false for security")
        return v


class OutputConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    language: Literal["ko", "en"] = "ko"
    max_findings_per_task: int = Field(ge=1, le=100, default=20)
    max_questions_per_task: int = Field(ge=1, le=100, default=20)
    max_comment_characters: int = Field(ge=1000, le=100000, default=60000)
    include_coverage: bool = True
    distinguish_prepared_sent_and_opinion_scope: bool = True
    evidence_ids_required: bool = True


class ReviewBotConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    schema_version: Literal[2] = 2
    codebase: CodebaseConfig
    execution: ExecutionConfig
    service: ServiceConfig
    publish: PublishConfig = Field(default_factory=PublishConfig)
    model: ModelConfig
    architecture: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    intent: IntentConfig = Field(default_factory=IntentConfig)
    repository_settings: RepositorySettingsConfig
    comparison: ComparisonConfig
    code_input: CodeInputConfig
    artifacts: ArtifactsConfig
    output: OutputConfig

    @model_validator(mode="after")
    def validate_github_app_mode(self) -> ReviewBotConfig:
        if (
            "architecture" in self.execution.available_modes and "intent" not in self.execution.available_modes
            and "architecture" not in self.model_fields_set
        ):
            raise ValueError("architecture configuration is required for A-only execution")
        if self.execution.mode == "github_app" and "mode" not in self.publish.model_fields_set:
            self.publish.mode = "comment"
        return self

    @property
    def config_hash(self) -> str:
        return compute_config_hash(self)


class TargetRepositoryConfig(BaseModel):
    """Restricted configuration model allowed in untrusted target repositories (.review-bot.yml)."""

    model_config = ConfigDict(strict=True, extra="forbid")
    analysis_profile: str = "java21-jdk-only-v1"
    source_roots: list[str] = Field(default_factory=lambda: ["src/main/java"])
    architecture_rule_paths: list[str] = Field(default_factory=list)


# ==============================================================================
# Public API Functions
# ==============================================================================


def compute_config_hash(data: dict[str, Any] | BaseModel) -> str:
    """Computes a deterministic SHA-256 digest over normalized canonical JSON."""
    raw_dict = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    if isinstance(raw_dict.get("intent"), dict) and raw_dict["intent"].get("pipeline") == "declared-v1":
        raw_dict = {**raw_dict, "intent": {key: value for key, value in raw_dict["intent"].items() if key != "pipeline"}}
    execution = raw_dict.get("execution")
    if (
        raw_dict.get("schema_version") == 2 and isinstance(execution, dict)
        and "intent" not in execution.get("available_modes", ["architecture"])
        and isinstance(raw_dict.get("publish"), dict)
    ):
        publish = dict(raw_dict["publish"])
        for key, default in (("b_marker", DEFAULT_B_MARKER), ("b_comments_enabled", False)):
            if publish.get(key) == default:
                publish.pop(key, None)
        raw_dict = {**raw_dict, "publish": publish}
    if (
        raw_dict.get("schema_version") == 2 and isinstance(execution, dict)
        and isinstance(execution.get("available_modes", ["architecture"]), list)
        and "intent" not in execution.get("available_modes", ["architecture"])
        and raw_dict.get("intent") == {"context_paths": []}
    ):
        raw_dict = {key: value for key, value in raw_dict.items() if key != "intent"}
    canonical_json = json.dumps(
        raw_dict,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def parse_yaml_strict(stream_or_str: str | TextIO) -> dict[str, Any]:
    """Parses YAML text or stream with StrictSafeLoader, enforcing duplicate key and alias prohibition."""
    try:
        data = yaml.load(stream_or_str, Loader=StrictSafeLoader)
    except (ConfigDuplicateKeyError, ConfigAliasError, ConfigSecurityError):
        raise
    except (yaml.YAMLError, Exception) as exc:
        raise ConfigSyntaxError(f"YAML parsing error: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigSyntaxError(
            f"Configuration root must be a YAML dictionary mapping, got {type(data).__name__}"
        )
    return data


def load_review_config_from_str(content: str) -> tuple[ReviewBotConfig, str]:
    """Parses and validates operator YAML config text, returning (ReviewBotConfig, config_hash)."""
    raw = parse_yaml_strict(content)
    try:
        config = ReviewBotConfig.model_validate(raw)
    except ValidationError as exc:
        for err in exc.errors():
            msg = err.get("msg", "")
            if "MUST be false for security" in msg:
                raise ConfigSecurityError(f"Security invariant violated: {msg}") from exc
        raise ConfigValidationError(f"Configuration schema validation failed: {exc}") from exc
    config_hash = compute_config_hash(config)
    return config, config_hash


def load_review_config(path: str | Path | TextIO) -> tuple[ReviewBotConfig, str]:
    """Loads and validates operator YAML configuration from file path or stream."""
    if isinstance(path, (str, Path)):
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        content = file_path.read_text(encoding="utf-8")
    elif hasattr(path, "read"):
        content = path.read()
    else:
        raise TypeError(f"Expected path or file-like object, got {type(path)}")
    return load_review_config_from_str(content)


def load_target_repository_config(path_or_str: str | Path | TextIO) -> TargetRepositoryConfig:
    """Loads and validates untrusted target repository config (.review-bot.yml) strictly restricting keys."""
    if isinstance(path_or_str, (str, Path)):
        p = Path(path_or_str)
        if p.is_file():
            content = p.read_text(encoding="utf-8")
        else:
            content = str(path_or_str)
    elif hasattr(path_or_str, "read"):
        content = path_or_str.read()
    else:
        raise TypeError(f"Expected path, str, or file-like object, got {type(path_or_str)}")

    raw = parse_yaml_strict(content)
    try:
        return TargetRepositoryConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigValidationError(f"Target repository config validation failed: {exc}") from exc


def validate_active_environment(
    config: ReviewBotConfig,
    env: Mapping[str, str] = os.environ,
) -> None:
    """Validates required environment secrets when running in active GitHub App mode."""
    if config.execution.mode == "github_app":
        missing_vars: list[str] = []
        for var_name in (
            config.service.app_id_env,
            config.service.app_private_key_file_env,
            config.service.webhook_secret_env,
        ):
            if var_name not in env or not env[var_name].strip():
                missing_vars.append(var_name)
        if missing_vars:
            raise ConfigValidationError(
                f"Missing required environment variables for github_app mode: {', '.join(missing_vars)}"
            )


def load_config(source: str | Path | TextIO) -> ReviewBotConfig:
    """Convenience loader for ReviewBotConfig from file path, stream, or YAML string."""
    if isinstance(source, str):
        if not source.strip():
            raise ConfigValidationError("Configuration content cannot be empty")
        if "\n" not in source and source.endswith((".yaml", ".yml")):
            p = Path(source)
            if p.is_file():
                cfg, _ = load_review_config(p)
                return cfg
        cfg, _ = load_review_config_from_str(source)
        return cfg
    cfg, _ = load_review_config(source)
    return cfg
