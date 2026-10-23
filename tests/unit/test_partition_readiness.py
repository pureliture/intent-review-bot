"""Bounded deployment probes use existing clients and rootless analysis only."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from review_bot.application.architecture_review import ModelProfile
from review_bot.config import load_review_config_from_str
from review_bot.domain.models_v1 import ModelResponse


def load_probe():
    path = Path(__file__).resolve().parents[2] / "deploy/check-partition-readiness.py"
    spec = importlib.util.spec_from_file_location("partition_readiness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pmd_status,cpd_status", [("clean", "clean"), ("findings", "clean")])
def test_readiness_checks_rootless_analysis_and_two_bounded_model_requests(tmp_path, valid_config_yaml, monkeypatch, pmd_status, cpd_status):
    module = load_probe()
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["available_modes"] = ["architecture", "intent"]
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True, "max_total_tokens": 524288, "max_calls": 64}
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    monkeypatch.setenv(config.service.state_root_env, str(state))
    monkeypatch.setattr(module.platform, "system", lambda: "Linux")
    report = SimpleNamespace(finding_count=1)
    analyzer = Mock()
    analyzer.run_analysis.return_value = SimpleNamespace(pmd=SimpleNamespace(status=pmd_status, report=report),
        cpd=SimpleNamespace(status=cpd_status, report=report))
    factory = Mock(return_value=analyzer)
    monkeypatch.setattr(module, "_architecture_analyzer", factory)
    clients = []
    def client_factory(*args):
        client = Mock()
        client.profile = ModelProfile("llamacpp", "bounded-probe", max_output_tokens=32678)
        client.admission_bytes.return_value = 20
        client.complete.side_effect = lambda request: ModelResponse(
            '{"choice":"A"}' if "choice" in request.response_schema["properties"] else '{"ok":true}',
            {"input_tokens": 20, "output_tokens": 4})
        clients.append(client)
        return client
    monkeypatch.setattr(module, "build_review_client", client_factory)
    monkeypatch.setattr(module, "build_planning_client", client_factory)
    result = module.check(config, {"models": {}, "analysis": None, "pr_review_executed": False})
    assert result["status"] == "ready_for_pr_tests" and result["pr_review_executed"] is False
    assert set(result["models"]) == {"review", "planning"}
    factory.assert_called_once()
    analyzer.run_analysis.assert_called_once()
    assert len(clients) == 2
    assert [c.complete.call_args.args[0].max_output_tokens for c in clients] == [128, 16]


def test_readiness_rejects_nonlinux_before_accessing_runtime(monkeypatch):
    module = load_probe()
    monkeypatch.setattr(module.platform, "system", lambda: "Darwin")
    with pytest.raises(module.ReadinessError, match="linux_node_required"):
        module.check(None, {})
