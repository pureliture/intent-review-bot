from __future__ import annotations

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from review_bot.adapters import pi_agent_docker
from review_bot.adapters.pi_agent_docker import DockerPiAgentRunner, proxy_image_tag
from review_bot.application.agent_review import _SEARCH_SCOPE_RULE, _initial_agent_prompt
from review_bot.config import AgentEngineConfig
from review_bot.ports import AgentRunError, AgentRunRequest, AgentRunResponse


def test_docker_runner_resolves_relative_session_store_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = DockerPiAgentRunner(AgentEngineConfig(session_store_dir="relative-ledger"))
    assert runner.session_store_dir == (tmp_path / "relative-ledger").resolve()
    assert runner.session_store_dir.is_absolute()


def test_proxy_image_tag_is_stable():
    tag = proxy_image_tag()
    digest = tag.removeprefix("review-bot-agent-proxy:src-")
    assert len(digest) == 12
    assert all(character in "0123456789abcdef" for character in digest)
    assert proxy_image_tag() == tag


def test_proxy_image_tag_changes_when_llm_proxy_changes(tmp_path, monkeypatch):
    inputs = []
    for name, source in pi_agent_docker._proxy_build_inputs():
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        inputs.append((name, destination))
    monkeypatch.setattr(pi_agent_docker, "_proxy_build_inputs", lambda: tuple(inputs))

    initial_tag = proxy_image_tag()
    assert proxy_image_tag() == initial_tag
    proxy_source = tmp_path / "llm_proxy.py"
    proxy_source.write_bytes(proxy_source.read_bytes() + b"\n")
    assert proxy_image_tag() != initial_tag


def test_missing_proxy_image_fails_before_starting_containers(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, "", "image not found")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig())
    request = AgentRunRequest("architecture", tmp_path, "review-architecture", "model", 10)

    with pytest.raises(AgentRunError, match="proxy_image_missing"):
        runner.run(request)

    assert calls == [["docker", "image", "inspect", proxy_image_tag()]]


def test_docker_runner_tracks_concurrent_sessions_safely(tmp_path, monkeypatch):
    runner = DockerPiAgentRunner(AgentEngineConfig())
    barrier = threading.Barrier(2, timeout=5)

    class Temporary:
        def cleanup(self):
            pass

    def start(request):
        temporary = Temporary()
        run = SimpleNamespace(
            request=request,
            network=f"network-{request.mode}",
            proxy_name=f"proxy-{request.mode}",
            agent_name=f"agent-{request.mode}",
            config_temporary=temporary,
            session_temporary=temporary,
            skills_temporary=temporary,
            output_temporary=temporary,
            started=time.monotonic(),
            session_jsonl=b"session",
            ledger_path=tmp_path / f"{request.mode}.jsonl",
        )
        return None, run

    def invoke(run, _prompt, _session_id):
        barrier.wait()
        return AgentRunResponse("ok", b"session", {}, "0.73.1", 0, f"session-{run.request.mode}")

    monkeypatch.setattr(runner, "_start", start)
    monkeypatch.setattr(runner, "_invoke", invoke)
    monkeypatch.setattr(runner, "_cleanup_names", lambda *_args: None)
    monkeypatch.setattr(runner, "_proxy_usage", lambda _path, _session: {"calls": 0})
    monkeypatch.setattr(runner, "_agent_reported_usage", lambda _session: {})

    requests = [
        AgentRunRequest(mode, tmp_path, "skill", "model", 10)
        for mode in ("architecture", "intent")
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(runner.run, requests))

    assert {response.session_id for response in responses} == {
        "session-architecture", "session-intent",
    }
    assert set(runner._active_runs) == {"session-architecture", "session-intent"}

    with ThreadPoolExecutor(max_workers=2) as executor:
        finalizations = list(executor.map(runner.finish, sorted(runner._active_runs)))

    assert all(finalization.session_jsonl == b"session" for finalization in finalizations)
    assert runner._active_runs == {}


def test_docker_runner_hardens_agent_and_tears_down_after_failure(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "review-agent-" in " ".join(argv) and "review-bot-pi-agent:0.73.1" in argv:
            skills_mount = next(item for item in argv if item.endswith(":/skills:ro"))
            skills_dir = Path(skills_mount.split(":/skills:ro")[0])
            assert (skills_dir / "review-architecture" / "SKILL.md").is_file()
            assert (skills_dir / "review-intent" / "SKILL.md").is_file()
            assert (skills_dir / "check_output.mjs").is_file()
            settings_mount = next(item for item in argv
                if item.endswith(":/agent/settings.json:ro"))
            config_dir = Path(settings_mount.split(":/agent/settings.json:ro")[0]).parent
            models = json.loads((config_dir / "models.json").read_text())
            model = models["providers"]["llmproxy"]["models"][0]
            assert model["id"] == "stealth/crouton-alpha"
            assert "contextWindow" not in model
            assert "maxTokens" not in model
            assert any(item.endswith(":/agent/models.json:ro") for item in argv)
            assert any(item.startswith("/agent:rw,") for index, item in enumerate(argv)
                if index > 0 and argv[index - 1] == "--tmpfs")
            settings_path = config_dir / "settings.json"
            assert json.loads(settings_path.read_text()) == {"shellCommandPrefix": "ulimit -t 120"}
            assert settings_path.stat().st_mode & 0o777 == 0o444
            prompt = argv[argv.index("--print") + 1]
            assert _SEARCH_SCOPE_RULE in prompt
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
    assert any(":/skills:ro" in arg for arg in agent)
    assert not any(":/skills/check_output.mjs:ro" in arg for arg in agent)
    assert any(":/out:rw" in arg for arg in agent)
    assert any(arg.endswith(":/agent/models.json:ro") for arg in agent)
    assert any(arg.endswith(":/agent/settings.json:ro") for arg in agent)
    assert any(":/sessions:rw" in arg for arg in agent)
    assert agent[agent.index("--skill") + 1] == "/skills/review-architecture/SKILL.md"
    assert "bash" in agent[agent.index("--tools") + 1].split(",")
    writable_mounts = [arg for arg in agent if arg.startswith("/") and arg.endswith(":rw")]
    assert len(writable_mounts) == 2
    assert any(":/out:rw" in arg for arg in writable_mounts)
    assert any(":/sessions:rw" in arg for arg in writable_mounts)
    assert "PI_CODING_AGENT_DIR=/agent" in agent
    assert "OPENROUTER_API_KEY" not in agent and "test-only-secret" not in json.dumps(agent)
    assert not any("docker.sock" in argument for argument in agent)
    proxy = next(argv for argv in calls if argv[1] == "run" and proxy_image_tag() in argv)
    assert "UPSTREAM_API_KEY" in proxy
    assert not any(arg.startswith("MAX_OUTPUT_TOKENS=") for arg in proxy)
    assert not any(arg.startswith("UPSTREAM_API_KEY=") for arg in proxy)
    assert "PROVIDER=openrouter" in proxy
    assert not any("OPENROUTER_API_KEY" in argument for argument in proxy)
    assert all("test-only-secret" not in argument for argv in calls for argument in argv)
    assert any(argv[1:4] == ["rm", "-f", next(arg for arg in argv if arg.startswith("review-agent-"))]
        for argv in calls if len(argv) > 3 and argv[1] == "rm")
    assert any(argv[1:3] == ["network", "rm"] for argv in calls)


@pytest.mark.parametrize(("mode", "skill_id"), [
    ("architecture", "review-architecture"),
    ("verify", "verify-claims"),
])
def test_initial_agent_prompt_scopes_searches(tmp_path, monkeypatch, mode, skill_id):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    prompts = []

    def fake_run(argv, **kwargs):
        if "review-bot-pi-agent:0.73.1" in argv:
            prompts.append(argv[argv.index("--print") + 1])
            return subprocess.CompletedProcess(argv, 1, "", "agent failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig())
    with pytest.raises(AgentRunError, match="agent_crashed"):
        runner.run(AgentRunRequest(mode, workspace, skill_id, "model-id", 10))

    assert prompts == [_initial_agent_prompt(mode)]
    assert _SEARCH_SCOPE_RULE in prompts[0]


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


def test_docker_repair_reuses_proxy_session_and_reads_usage_ledger_after_teardown(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENCODE_API_KEY", "test-only-secret")
    calls = []
    agent_timeouts = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "run" and proxy_image_tag() in argv:
            mount = next(item for item in argv if ":/ledger:rw" in item)
            ledger_dir = Path(mount.split(":/ledger:rw")[0])
            (ledger_dir / "usage.jsonl").write_text(
                '{"call":1,"status":200,"prompt_tokens":23,"completion_tokens":7,'
                '"total_tokens":30,"cost":{"total":0.0025}}\n', encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "proxy-id", "")
        if "review-bot-pi-agent:0.73.1" in argv:
            agent_timeouts.append(kwargs["timeout"])
            skills_mount = next(item for item in argv if item.endswith(":/skills:ro"))
            skills_dir = Path(skills_mount.split(":/skills:ro")[0])
            assert (skills_dir / "review-architecture" / "SKILL.md").is_file()
            assert (skills_dir / "check_output.mjs").is_file()
            mount = next(item for item in argv if ":/sessions:rw" in item)
            session_dir = Path(mount.split(":/sessions:rw")[0])
            is_verifier = any("verify-claims" in item for item in argv)
            session_jsonl = ('{"type":"session","id":"session-verify"}\n' if is_verifier else
                '{"type":"session","id":"session-123"}\n{"type":"compaction","tokensBefore":42}\n')
            (session_dir / "session.jsonl").write_text(session_jsonl, encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, '{"summary":"ok"}', "")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig(provider="opencode-go",
        session_store_dir=str(tmp_path / "sessions")))
    request = AgentRunRequest("architecture", workspace, "review-architecture",
        "stealth/crouton-alpha", 20)
    first = runner.run(request)
    repaired = runner.repair(request, first.session_id, "item_index=0 reason=agent_output_invalid",
        timeout_seconds=7)
    result = runner.finish(first.session_id)
    verifier_finalization = None
    verifier_request = AgentRunRequest("verify", workspace, "verify-claims",
        "stealth/crouton-alpha", 20)
    verifier = runner.run(verifier_request)
    verifier_finalization = runner.finish(verifier.session_id)

    agents = [argv for argv in calls if "review-bot-pi-agent:0.73.1" in argv]
    proxies = [argv for argv in calls if argv[1] == "run" and proxy_image_tag() in argv]
    assert first.session_id == repaired.session_id == "session-123"
    assert verifier.session_id == "session-verify"
    assert len(agents) == 3 and len(proxies) == 2
    assert agent_timeouts[1] == 7
    assert agents[1][agents[1].index("--session") + 1] == "session-123"
    assert agents[0][agents[0].index("--network") + 1] == agents[1][agents[1].index("--network") + 1]
    assert "OPENROUTER_API_KEY" not in agents[0] and "test-only-secret" not in json.dumps(agents)
    assert all("UPSTREAM_API_KEY" in argv and "PROVIDER=opencode-go" in argv
        and not any(arg.startswith("UPSTREAM_API_KEY=") for arg in argv) for argv in proxies)
    assert all("test-only-secret" not in argument for argv in calls for argument in argv)
    proxy_sessions = [next(arg.split("=", 1)[1] for arg in argv
        if arg.startswith("UPSTREAM_SESSION_ID=")) for argv in proxies]
    assert all(session.startswith("ses_") and len(session) == 36 for session in proxy_sessions)
    assert proxy_sessions[0] != proxy_sessions[1]
    assert result.usage["input_tokens"] == 23 and result.usage["output_tokens"] == 7
    assert result.usage["cost"]["total"] == 0.0025
    assert first.usage["max_prompt_tokens"] == repaired.usage["max_prompt_tokens"] == 23
    assert first.usage["compactions"] == repaired.usage["compactions"] == result.usage["compactions"] == 1
    assert result.usage["max_prompt_tokens"] == 23
    assert verifier_finalization.usage["max_prompt_tokens"] == 23
    assert verifier_finalization.usage["compactions"] == 0
    assert result.proxy_ledger_path.endswith("usage.jsonl")
    out_mounts = [next(arg for arg in argv if ":/out:rw" in arg) for argv in agents]
    assert out_mounts[0] == out_mounts[1]
    skills_mounts = [next(arg for arg in argv if arg.endswith(":/skills:ro")) for argv in agents]
    assert skills_mounts[0] == skills_mounts[1]
    assert agents[0][agents[0].index("--skill") + 1] == "/skills/review-architecture/SKILL.md"
    assert any(argv[1:3] == ["network", "rm"] for argv in calls)


def test_docker_runner_reads_key_file_and_exposes_key_only_to_proxy_process(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_file = tmp_path / "openrouter-key"
    key_file.write_text("file-only-secret\n", encoding="utf-8")
    key_file.chmod(0o600)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("REVIEW_BOT_AGENT_DOCKER_HOST", "unix:///var/run/docker.sock")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "review-bot-pi-agent:0.73.1" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "agent failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig(session_store_dir=str(tmp_path / "sessions")))
    with pytest.raises(AgentRunError, match="agent_crashed"):
        runner.run(AgentRunRequest(
            "architecture", workspace, "review-architecture", "stealth/crouton-alpha", 10
        ))

    proxy = next((argv, kwargs) for argv, kwargs in calls if "review-bot-agent-proxy:1" in argv)
    assert proxy[1]["env"]["UPSTREAM_API_KEY"] == "file-only-secret"
    agent = next((argv, kwargs) for argv, kwargs in calls if "review-bot-pi-agent:0.73.1" in argv)
    assert "UPSTREAM_API_KEY" not in agent[1].get("env", {})
    assert agent[1]["env"]["DOCKER_HOST"] == "unix:///var/run/docker.sock"
    assert "file-only-secret" not in json.dumps(agent[0])


def test_proxy_usage_tracks_max_prompt_tokens_and_none_when_unreported(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    ledger.write_text(
        '{"prompt_tokens":42,"completion_tokens":3}\n{"completion_tokens":8}\n',
        encoding="utf-8",
    )
    assert DockerPiAgentRunner._proxy_usage(ledger)["max_prompt_tokens"] == 42

    ledger.write_text('{"completion_tokens":8}\n', encoding="utf-8")
    assert DockerPiAgentRunner._proxy_usage(ledger)["max_prompt_tokens"] is None


def test_session_compaction_count_uses_persisted_compaction_entry_type():
    session = (
        b'{"type":"session","id":"session-123"}\n'
        b'{"type":"compaction","summary":"summary","tokensBefore":42}\n'
        b'{"type":"message","message":{"role":"user","content":"hello"}}\n'
    )
    assert DockerPiAgentRunner._session_compactions(session) == 1


@pytest.mark.parametrize(("provider", "key_file_variable", "key_variable", "credential_source"), [
    ("openrouter", "OPENROUTER_API_KEY_FILE", "OPENROUTER_API_KEY", "file"),
    ("opencode-go", "OPENCODE_API_KEY_FILE", "OPENCODE_API_KEY", "file"),
    ("openrouter", "OPENROUTER_API_KEY_FILE", "OPENROUTER_API_KEY", "environment"),
    ("opencode-go", "OPENCODE_API_KEY_FILE", "OPENCODE_API_KEY", "environment"),
])
def test_provider_key_file_is_sent_only_to_proxy(tmp_path, monkeypatch,
    provider, key_file_variable, key_variable, credential_source, capsys):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    selected_key = f"selected-{provider}-secret"
    fallback_key = f"fallback-{provider}-secret"
    key_file = tmp_path / f"{provider}.key"
    key_file.write_text(selected_key, encoding="utf-8")
    if credential_source == "file":
        monkeypatch.setenv(key_file_variable, str(key_file))
        monkeypatch.setenv(key_variable, fallback_key)
    else:
        monkeypatch.delenv(key_file_variable, raising=False)
        monkeypatch.setenv(key_variable, selected_key)
    other_key = "unused-other-provider-secret"
    other_key_variable = "OPENCODE_API_KEY" if provider == "openrouter" else "OPENROUTER_API_KEY"
    monkeypatch.setenv(other_key_variable, other_key)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "review-bot-pi-agent:0.73.1" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "agent failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig(provider=provider,
        session_store_dir=str(tmp_path / "sessions")))
    with pytest.raises(AgentRunError, match="agent_crashed") as error:
        runner.run(AgentRunRequest("architecture", workspace, "review-architecture", "model-id", 10))

    proxy_call = next((argv, kwargs) for argv, kwargs in calls
        if argv[1] == "run" and proxy_image_tag() in argv)
    proxy, proxy_kwargs = proxy_call
    agent = next(argv for argv, _ in calls if "review-bot-pi-agent:0.73.1" in argv)
    assert "UPSTREAM_API_KEY" in proxy
    assert not any(arg.startswith("UPSTREAM_API_KEY=") for arg in proxy)
    assert f"PROVIDER={provider}" in proxy
    assert proxy_kwargs["env"]["UPSTREAM_API_KEY"] == selected_key
    assert all("env" not in kwargs for argv, kwargs in calls
        if not (argv[1] == "run" and proxy_image_tag() in argv))
    assert all(selected_key not in argument for argv, _ in calls for argument in argv)
    assert selected_key not in json.dumps(agent)
    assert fallback_key not in json.dumps(agent)
    assert other_key not in json.dumps(agent)
    assert selected_key not in str(error.value)
    captured = capsys.readouterr()
    assert selected_key not in captured.out + captured.err


def test_pi_models_json_uses_configured_context_window(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")

    def fake_run(argv, **kwargs):
        if "review-bot-pi-agent:0.73.1" in argv:
            config_mount = next(item for item in argv if item.endswith(":/agent/models.json:ro"))
            models = json.loads(Path(config_mount.split(":/agent/models.json:ro")[0]).read_text())
            assert models["providers"]["llmproxy"]["models"][0]["contextWindow"] == 262144
            return subprocess.CompletedProcess(argv, 1, "", "agent failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig(context_window_tokens=262144,
        session_store_dir=str(tmp_path / "sessions")))
    with pytest.raises(AgentRunError, match="agent_crashed"):
        runner.run(AgentRunRequest("architecture", workspace, "review-architecture", "model-id", 10))


def test_pi_and_proxy_receive_explicit_output_token_cap(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "review-bot-pi-agent:0.73.1" in argv:
            config_mount = next(item for item in argv if item.endswith(":/agent/models.json:ro"))
            models = json.loads(Path(config_mount.split(":/agent/models.json:ro")[0]).read_text())
            assert models["providers"]["llmproxy"]["models"][0]["maxTokens"] == 8192
            return subprocess.CompletedProcess(argv, 1, "", "agent failed")
        return subprocess.CompletedProcess(argv, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    runner = DockerPiAgentRunner(AgentEngineConfig(max_output_tokens=8192,
        session_store_dir=str(tmp_path / "sessions")))
    with pytest.raises(AgentRunError, match="agent_crashed"):
        runner.run(AgentRunRequest("architecture", workspace, "review-architecture", "model-id", 10))

    proxy = next(argv for argv in calls if argv[1] == "run" and proxy_image_tag() in argv)
    assert "MAX_OUTPUT_TOKENS=8192" in proxy


@pytest.mark.parametrize(("provider", "error_code"), [
    ("openrouter", "openrouter_api_key_missing"),
    ("opencode-go", "opencode_api_key_missing"),
])
def test_provider_specific_missing_key_error(tmp_path, monkeypatch, provider, error_code):
    for variable in ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_FILE",
        "OPENCODE_API_KEY", "OPENCODE_API_KEY_FILE"):
        monkeypatch.delenv(variable, raising=False)
    runner = DockerPiAgentRunner(AgentEngineConfig(provider=provider))
    with pytest.raises(AgentRunError, match=error_code):
        runner.run(AgentRunRequest("architecture", tmp_path, "review-architecture", "model-id", 10))
