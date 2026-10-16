"""Semantic per-mode publication identities, distinct from full audit/run hashes."""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import asdict
from importlib import resources
from pathlib import Path

from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.markdown.intent_formatter import format_intent_markdown
from review_bot.application.context import build_task_input, hash_json
from review_bot.application.delta import build_delta
from review_bot.application.intent_context import build_intent_task_input
from review_bot.application.architecture_review import ModelProfile, build_system_prompt, run_review_a_pipeline
from review_bot.application.declared_intent_review import PROMPT_VERSION, SYSTEM_PROMPTS, response_schema, run_review_b_pipeline
from review_bot.domain.intent_models import IntentReviewLimits


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


def intent_input_fingerprint(task, config, client, *, planning_client=None) -> str | None:
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
        if config.intent.pipeline == "code-first-v2":
            from review_bot.adapters.markdown.code_intent_formatter import format_code_intent_markdown
            from review_bot.application.code_first_intent_review import code_first_prompt_hashes, run_code_first_review
            from review_bot.domain.code_first_intent_models import CodeFirstReviewLimits

            return hash_json({"schema": "mode-publication/code-first-v2", "mode": "intent", "material": material,
                "model": _profile(client), "planner": _profile(planning_client), "prompts": code_first_prompt_hashes(),
                "implementation": _implementation_hash(build_intent_task_input, run_code_first_review,
                    run_review_b_pipeline, format_code_intent_markdown),
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