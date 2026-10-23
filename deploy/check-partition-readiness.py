#!/usr/bin/env python3
"""Explicit, bounded Linux readiness probe; never submits PRs or changes admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path

from review_bot.application.intent_context import strict_json_loads
from review_bot.config import load_review_config
from review_bot.domain.models_v1 import ModelRequest
from review_bot.runtime.artifacts import _intent_directory
from review_bot.runtime.model_clients import build_planning_client, build_review_client
from review_bot.runtime.service import _architecture_analyzer


class ReadinessError(ValueError):
    """Only fixed, non-sensitive reason codes are exposed to the receipt."""


def check(config, result):
    if platform.system() != "Linux":
        raise ReadinessError("linux_node_required")
    if not config.partition_evaluation.enabled or config.publish.mode != "artifact_only":
        raise ReadinessError("private_partition_configuration_required")
    root_text = os.environ.get(config.service.state_root_env, "")
    if not root_text or not Path(root_text).is_absolute():
        raise ReadinessError("private_state_root_required")
    root = _intent_directory(Path(root_text))
    if "architecture" in config.execution.available_modes:
        result["phase"] = "rootless_analysis"
        analyzer = _architecture_analyzer(config, root)
        with tempfile.TemporaryDirectory(prefix="partition-probe-", dir=root) as folder:
            source = Path(folder)
            (source / "build.gradle").write_text("plugins { id 'java' }\n")
            (source / "Probe.java").write_text("class Probe { void run() { int unused = 1; } }\n")
            analysis = analyzer.run_analysis("0" * 40, source, config.architecture.ruleset_resource,
                config.architecture.analysis_profile)
        if analysis.pmd.status not in ("clean", "findings") or analysis.cpd.status not in ("clean", "findings"):
            raise ReadinessError("isolated_analysis_probe_failed")
        result["analysis"] = {"image": config.architecture.tool_image_digest,
            "pmd": analysis.pmd.status, "cpd": analysis.cpd.status,
            "pmd_findings": analysis.pmd.report.finding_count, "cpd_findings": analysis.cpd.report.finding_count}
    for role, builder in (("review", build_review_client), ("planning", build_planning_client)):
        result["phase"] = role
        client = builder(config.model.profile, config.model)
        if client.profile.provider != "llamacpp" or client.profile.is_mock:
            raise ReadinessError("native_model_profile_required")
        client.warmup()
        if role == "planning":
            system = "Select exactly one listed option from the structured decision. Return only JSON with its letter in choice."
            content = json.dumps({"state": {"value": 3}, "question": "Is the supplied value positive?",
                "options": [{"label": "A", "key": "positive", "description": "The value is positive."},
                    {"label": "B", "key": "not_positive", "description": "The value is zero or negative."}]})
            schema = {"type": "object", "properties": {"choice": {"type": "string", "enum": ["A", "B"]}},
                "required": ["choice"], "additionalProperties": False}
            expected, ceiling = {"choice": "A"}, 16
        else:
            system, content = 'Return exactly {"ok":true}. No prose or tools.', "Readiness probe only."
            schema = {"type": "object", "properties": {"ok": {"type": "boolean", "const": True}},
                "required": ["ok"], "additionalProperties": False}
            expected, ceiling = {"ok": True}, 128
        request = ModelRequest(1, client.profile.model, system, content,
            min(120, config.model.call_timeout_seconds), ceiling, schema)
        bound = client.admission_bytes(request) if hasattr(client, "admission_bytes") else None
        response = client.complete(request)
        if type(bound) is not int or bound <= 0 or strict_json_loads(response.text) != expected:
            raise ReadinessError("model_contract_probe_failed")
        if not isinstance(response.usage, dict) or not {"input_tokens", "output_tokens"} <= response.usage.keys():
            raise ReadinessError("model_probe_usage_unavailable")
        result["models"][role] = {"model": client.profile.model, "admission_input_tokens": bound,
            "usage": response.usage, "request_sha256": hashlib.sha256(request.user_content.encode()).hexdigest()}
    result["status"] = "ready_for_pr_tests"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = _intent_directory(args.output.absolute().parent) / args.output.name
    # Claim an exclusive private receipt before making any model call.
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        result = {"status": "checking", "phase": "configuration", "models": {}, "analysis": None,
            "pr_review_executed": False, "publication": "not_attempted"}
        try:
            config, _ = load_review_config(args.config)
            result = check(config, result)
        except Exception as error:
            # Do not serialize credentials, source, endpoints or exception messages.
            result.update(status="blocked", error_type=type(error).__name__,
                reason=str(error) if isinstance(error, ReadinessError) else "dependency_failure")
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "receipt": str(output)}))
    return 0 if result["status"] == "ready_for_pr_tests" else 1


if __name__ == "__main__":
    raise SystemExit(main())
