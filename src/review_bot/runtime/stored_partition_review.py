"""Decode existing domain records before replay validation; no model generation."""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib import resources
from pathlib import Path

from pydantic import TypeAdapter

from review_bot.application.context import hash_json
from review_bot.application.intent_context import decode_intent_task_input
from review_bot.application.partition_review import validate_partition_result
from review_bot.domain.models_v1 import TaskInput
from review_bot.domain.partition_review import PartitionReviewResult


_GENERATION_IMPLEMENTATION = (
    "application/partition_review.py", "application/source_partition.py", "application/partition_choices.py",
    "application/partition_integration.py", "application/partition_global_review.py", "application/review_fact_check.py",
    "application/source_citations.py", "application/source_quote_citations.py", "application/model_execution.py", "application/model_wire_contract.py", "application/selection.py",
    "application/review_output_contract.py", "application/context.py", "application/intent_context.py",
    "application/delta.py", "application/architecture_review.py",
    "adapters/llamacpp.py", "adapters/llamacpp_macbook.py", "adapters/intent_planner.py",
    "adapters/openrouter_chat.py", "adapters/openrouter_decisions.py", "adapters/openrouter_common.py", "adapters/tev4b.py",
    "adapters/intent_selection.py", "adapters/model_http.py", "adapters/ollama.py", "runtime/model_clients.py",
    "domain/intent_perspectives.py", "domain/partition_review.py", "domain/intent_models.py",
    "domain/models_v1.py", "domain/envelope_v2.py", "domain/changes.py", "domain/status.py",
)
_ANALYSIS_IMPLEMENTATION = tuple("adapters/pmd/" + name for name in (
    "container.py", "runner.py", "parsers.py", "provisioning.py"))


def _analysis_recipe(config, package):
    """Fingerprint actual recipe bytes, including installed-package defaults."""
    def ruleset():
        configured = config.architecture.ruleset_resource
        for candidate in (Path(configured), package.parents[1] / configured):
            if candidate.is_file():
                return candidate.read_bytes()
        return resources.files("review_bot.resources").joinpath(configured).read_bytes()

    def tools_lock():
        configured = Path(config.architecture.tools_lock_path)
        if configured.is_file():
            return configured.read_bytes()
        if str(configured) == "tooling/tools.lock.json":
            # Provisioning uses this bundled lock in installed wheels and the
            # project tooling lock in a source checkout.
            packaged = resources.files("review_bot.resources").joinpath("pmd/tools.lock.json")
            if packaged.is_file():
                return packaged.read_bytes()
            return (package.parents[1] / configured).read_bytes()
        return configured.read_bytes()

    recipe = {}
    for name, read in (("ruleset", ruleset), ("tools_lock", tools_lock)):
        try:
            recipe[name] = {"state": "available", "sha256": sha256(read()).hexdigest()}
        except (OSError, ValueError, TypeError, ModuleNotFoundError) as error:
            # Never let an unavailable recipe match a previously available one.
            # Actual analysis owns the diagnostic and execution failure.
            recipe[name] = {"state": "unavailable", "error": type(error).__name__}
    return recipe


def decode_stored_partition_review(packet):
    if not isinstance(packet, dict) or set(packet) != {"task", "result"}:
        raise ValueError("invalid_stored_review_packet")
    kind = packet["result"].get("review_type")
    task = (TypeAdapter(TaskInput).validate_python(packet["task"]) if kind == "A"
        else decode_intent_task_input(packet["task"]))
    result = TypeAdapter(PartitionReviewResult).validate_python(packet["result"])
    validate_partition_result(result, task)
    return task, result


def shared_review_generation_key(task, config, review_client, planning_client, bundle, mode,
    decision_client=None):
    """Presentation and retrieval clocks never invalidate unchanged model work."""
    value = asdict(task)
    value.pop("input_hash", None)
    snapshot = value.get("snapshot", {})
    for key in ("captured_at", "as_of", "snapshot_id", "context_digest"):
        snapshot.pop(key, None)
    if (snapshot.get("comparison_status") == "comparable"
            and snapshot.get("comparison_base_sha") == value.get("base", {}).get("sha")
            and snapshot.get("head_sha") == value.get("head", {}).get("sha")
            and snapshot.get("comparison_base_sha") and snapshot.get("head_sha")):
        # Publication still fences the current target ref. The review's actual
        # comparison/head commits and complete material below remain in the key.
        snapshot.pop("target_base_sha", None)
    for evidence in snapshot.get("evidence", []):
        locator = evidence.get("locator", {})
        locator.pop("retrieved_at", None)
        if evidence.get("kind") == "pr_text":
            locator.pop("updated_at", None)
    package = Path(__file__).parents[1]
    analysis_needed = mode == "architecture" or mode == "planning" and "architecture" in config.execution.available_modes
    sources = _GENERATION_IMPLEMENTATION + (_ANALYSIS_IMPLEMENTATION if analysis_needed else ())
    implementation = {name: sha256((package / name).read_bytes()).hexdigest() for name in sources}
    def profile(client):
        current = getattr(client, "profile", None)
        return {"profile": asdict(current) if current is not None else None,
            "endpoint": getattr(client, "api_url", None)}
    return hash_json({"version": "shared-review-generation/v2", "mode": mode,
        "material": value, "available_modes": config.execution.available_modes,
        "review_instructions": bundle.instructions_for(mode) if mode != "planning" else "",
        "model": profile(review_client), "planner": profile(planning_client),
        "decider": profile(decision_client),
        "language": config.output.language, "analysis": config.architecture.model_dump(mode="json"),
        "limits": {"calls": config.partition_evaluation.max_calls or config.execution.max_model_calls,
            "tokens": config.partition_evaluation.max_total_tokens or config.execution.max_total_tokens,
            "input_target": config.partition_evaluation.input_target_tokens,
            "observation_output": config.partition_evaluation.observation_output_tokens},
        "implementation": implementation,
        "analysis_recipe": _analysis_recipe(config, package) if analysis_needed else None})
