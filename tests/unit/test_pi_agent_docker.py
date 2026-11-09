from __future__ import annotations

import json
import subprocess

import pytest

from review_bot.adapters.pi_agent_docker import DockerPiAgentRunner
from review_bot.config import AgentEngineConfig
from review_bot.ports import AgentRunError, AgentRunRequest


def test_docker_runner_hardens_agent_and_tears_down_after_failure(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "review-agent-" in " ".join(argv) and "review-bot-pi-agent:0.73.1" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "agent failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig(session_store_dir=str(tmp_path / "sessions")))
    with pytest.raises(AgentRunError, match="agent_crashed"):
        runner.run(AgentRunRequest("architecture", workspace, "review-architecture",
            "stealth/crouton-alpha", 10))

    agent = next(argv for argv in calls if "review-bot-pi-agent:0.73.1" in argv)
    assert "--network" in agent and any(arg.startswith("review-agent-net-") for arg in agent)
    for flag in ("--read-only", "--cap-drop", "--security-opt", "--pids-limit", "--memory",
            "--tmpfs", "--user"):
        assert flag in agent
    assert agent[agent.index("--cap-drop") + 1] == "ALL"
    assert agent[agent.index("--security-opt") + 1] == "no-new-privileges"
    assert agent[agent.index("--pids-limit") + 1] == "256"
    assert agent[agent.index("--memory") + 1] == "2g"
    assert "/tmp:rw,exec" in agent[agent.index("--tmpfs") + 1]
    assert agent[agent.index("--user") + 1] != "0:0"
    assert any(":/workspace:ro" in arg for arg in agent if arg.startswith(str(workspace)))
    assert any(":/skill/SKILL.md:ro" in arg for arg in agent)
    assert any(":/agent:ro" in arg for arg in agent)
    assert any(":/sessions:rw" in arg for arg in agent)
    assert "PI_CODING_AGENT_DIR=/agent" in agent
    assert "OPENROUTER_API_KEY" not in agent and "test-only-secret" not in json.dumps(agent)
    assert not any("docker.sock" in argument for argument in agent)
    assert any("OPENROUTER_API_KEY" in argv for argv in calls if "review-bot-agent-proxy:1" in argv)
    assert any(argv[1:4] == ["rm", "-f", next(arg for arg in argv if arg.startswith("review-agent-"))]
        for argv in calls if len(argv) > 3 and argv[1] == "rm")
    assert any(argv[1:3] == ["network", "rm"] for argv in calls)


def test_docker_runner_maps_timeout_to_agent_error_and_cleans_up(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "review-bot-pi-agent:0.73.1" in argv:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig())
    with pytest.raises(AgentRunError, match="timeout"):
        runner.run(AgentRunRequest("intent", workspace, "review-intent", "stealth/crouton-alpha", 1))
    assert any(argv[1:3] == ["network", "rm"] for argv in calls)


def test_docker_runner_rejects_skill_mode_mismatch(tmp_path):
    runner = DockerPiAgentRunner(AgentEngineConfig())
    with pytest.raises(AgentRunError, match="invalid_request"):
        runner.run(AgentRunRequest("architecture", tmp_path, "review-intent", "model", 1))
