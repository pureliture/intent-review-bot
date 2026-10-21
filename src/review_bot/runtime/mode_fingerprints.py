"""Semantic per-mode publication identities, distinct from full audit/run hashes."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.markdown.intent_formatter import format_intent_markdown
from review_bot.adapters.llamacpp import LlamaCppModelClient
from review_bot.adapters.llamacpp_macbook import MacBookLlamaCppModelClient
from review_bot.adapters.intent_planner import LlamaCppPlanClient
from review_bot.application.context import build_task_input, hash_json
from review_bot.application.delta import build_delta
from review_bot.application.intent_context import build_intent_task_input
from review_bot.application.architecture_review import ModelProfile, build_system_prompt, run_review_a_pipeline
from review_bot.application.model_execution import ModelExecutionBudget
from review_bot.application.model_wire_contract import beam_wire_payload
from review_bot.application.partition_choices import plan_candidates
from review_bot.application.partition_integration import native_integration_pages
from review_bot.application.review_fact_check import run as run_fact_verification
from review_bot.application.selection import is_gradle_java_path
from review_bot.application.source_citations import citation_spans
from review_bot.application.declared_intent_review import PROMPT_VERSION, SYSTEM_PROMPTS, response_schema, run_review_b_pipeline
from review_bot.domain.intent_models import IntentReviewLimits
from review_bot.config import shared_plan_enabled


def _implementation_hash(*functions) -> str:
    modules = {inspect.getmodule(function) for function in functions}
    return hash_json({module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in modules})


def _profile(client):
    profile = getattr(client, "profile", None)
    if profile is not None and not isinstance(profile, ModelProfile):
        raise ValueError("invalid_model_profile")
    return {"profile": asdict(profile) if profile is not None else None,
        "endpoint": getattr(client, "api_url", None)}


def architecture_input_fingerprint(snapshot, classification, config, client, *, title: str, body: str) -> str | None:
    """Fixed Git objects and trusted analysis recipe identify A before tool initialization."""
    try:
        if shared_plan_enabled(config):
            from review_bot.application.partition_review import run_partition_review
            from review_bot.application.source_partition import partition_original
            return hash_json({"schema": "partition-evaluation/architecture", "snapshot": asdict(snapshot),
                "classification": asdict(classification), "config": config.config_hash, "model": _profile(client),
                "implementation": _implementation_hash(run_partition_review, partition_original, ModelExecutionBudget, plan_candidates, citation_spans, native_integration_pages, run_fact_verification, is_gradle_java_path,
                    LlamaCppModelClient, MacBookLlamaCppModelClient, LlamaCppPlanClient, beam_wire_payload)})
        architecture = config.architecture.model_dump(mode="json")
        ruleset = Path(architecture["ruleset_resource"])
        rules = ruleset.read_bytes() if ruleset.is_file() else resources.files("review_bot.resources").joinpath(architecture["ruleset_resource"]).read_bytes()
        lock = Path(architecture["tools_lock_path"]).read_bytes()
        return hash_json({"schema": "mode-publication/v1", "mode": "architecture",
            "repository": snapshot.repository, "repository_id": snapshot.repository_id,
            "installation_id": snapshot.installation_id, "pr_number": snapshot.pr_number,
            "target_base_sha": snapshot.target_base_sha, "comparison_base_sha": snapshot.comparison_base_sha,
            "head_sha": snapshot.head_sha, "changed_files": [asdict(item) for item in snapshot.changed_files],
            "perspective": classification.review_mode, "title": title, "body": body,
            "analysis": architecture, "ruleset_hash": hashlib.sha256(rules).hexdigest(),
            "tools_lock_hash": hashlib.sha256(lock).hexdigest(),
            "model": _profile(client), "prompt": build_system_prompt(classification.review_mode),
            "implementation": _implementation_hash(build_task_input, build_delta, run_review_a_pipeline, format_review_markdown),
            "code_input": config.code_input.model_dump(mode="json"),
            "repository_settings": config.repository_settings.model_dump(mode="json"),
            "output": config.output.model_dump(mode="json"), "marker": config.publish.a_marker})
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def intent_input_fingerprint(task, config, client, *, planning_client=None, selection_client=None) -> str | None:
    """Hash full effective material/profile/prompts, not retrieval clocks or remaining deadline."""
    try:
        material = asdict(task)
        material.pop("input_hash")
        material["snapshot"].pop("captured_at")
        material["snapshot"].pop("as_of")
        material["snapshot"].pop("snapshot_id")
        material["snapshot"].pop("context_digest")
        for item in material["snapshot"]["evidence"]:
            item["locator"].pop("retrieved_at", None)
            if item["kind"] == "pr_text":
                item["locator"].pop("updated_at", None)
        if shared_plan_enabled(config):
            from review_bot.application.partition_review import run_partition_review
            from review_bot.application.source_partition import partition_original
            return hash_json({"schema": "partition-evaluation/intent", "material": material,
                "model": _profile(client), "planner": _profile(planning_client),
                "trial": config.partition_evaluation.model_dump(mode="json"),
                "language": config.output.language,
                "implementation": _implementation_hash(run_partition_review, partition_original, ModelExecutionBudget, plan_candidates, citation_spans, native_integration_pages, run_fact_verification, is_gradle_java_path,
                    LlamaCppModelClient, MacBookLlamaCppModelClient, LlamaCppPlanClient, beam_wire_payload)})
        if config.intent.pipeline == "perspectives-v1":
            from review_bot.adapters.markdown.intent_perspective_formatter import format_intent_perspective_markdown
            from review_bot.application.intent_perspective_review import (
                CATALOG_HASH, PROMPT_VERSION as PERSPECTIVE_PROMPT_VERSION,
                perspective_prompt_hashes, run_intent_perspective_review,
            )
            from review_bot.domain.intent_perspectives import IntentPerspectiveLimits

            return hash_json({"schema": "mode-publication/intent-perspectives-v1", "mode": "intent", "material": material,
                "model": _profile(client), "selector": _profile(selection_client), "catalog_hash": CATALOG_HASH,
                "prompt_version": PERSPECTIVE_PROMPT_VERSION, "prompts": perspective_prompt_hashes(config.output.language),
                "implementation": _implementation_hash(build_intent_task_input, run_intent_perspective_review,
                    format_intent_perspective_markdown, ModelExecutionBudget),
                "limits": asdict(IntentPerspectiveLimits(task_timeout_seconds=config.execution.task_timeout_seconds)),
                "output": config.output.model_dump(mode="json"), "marker": config.publish.b_marker})
        if config.intent.pipeline == "code-first-v2":
            from review_bot.adapters.markdown.code_intent_formatter import format_code_intent_markdown
            from review_bot.application.code_first_intent_review import code_first_prompt_hashes, run_code_first_review
            from review_bot.domain.code_first_intent_models import CodeFirstReviewLimits

            return hash_json({"schema": "mode-publication/code-first-v2", "mode": "intent", "material": material,
                "model": _profile(client), "planner": _profile(planning_client), "prompts": code_first_prompt_hashes(),
                "implementation": _implementation_hash(build_intent_task_input, run_code_first_review,
                    run_review_b_pipeline, format_code_intent_markdown, ModelExecutionBudget),
                "limits": asdict(CodeFirstReviewLimits(task_timeout_seconds=config.execution.task_timeout_seconds)),
                "intent": config.intent.model_dump(mode="json"),
                "output": config.output.model_dump(mode="json"), "marker": config.publish.b_marker})
        return hash_json({"schema": "mode-publication/v1", "mode": "intent", "material": material,
            "model": _profile(client), "prompt_version": PROMPT_VERSION,
            "prompts": SYSTEM_PROMPTS, "schemas": {stage: response_schema(stage) for stage in SYSTEM_PROMPTS},
            "implementation": _implementation_hash(build_intent_task_input, run_review_b_pipeline, format_intent_markdown),
            "limits": asdict(IntentReviewLimits(task_timeout_seconds=config.execution.task_timeout_seconds)),
            "output": config.output.model_dump(mode="json"), "marker": config.publish.b_marker})
    except (OSError, ValueError, TypeError, AttributeError):
        return None
