"""Docker-isolated pi-agent runner with a credential-holding, bounded proxy."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from review_bot.application.agent_review import (
    _initial_agent_prompt,
    agent_checker_path,
    agent_skill_path,
    agent_skills_dir,
)
from review_bot.config import AgentEngineConfig
from review_bot.ports import (
    AgentRunError,
    AgentRunFinalization,
    AgentRunRequest,
    AgentRunResponse,
)

PROXY_IMAGE = "review-bot-agent-proxy:1"
PI_VERSION = "0.73.1"
_SHELL_CPU_SECONDS = 120


@dataclass
class _ActiveRun:
    network: str
    proxy_name: str
    agent_name: str
    config_dir: Path
    session_dir: Path
    ledger_path: Path
    ledger_dir: Path
    request: AgentRunRequest
    workspace: Path
    skills_dir: Path
    output_dir: Path
    config_temporary: tempfile.TemporaryDirectory
    session_temporary: tempfile.TemporaryDirectory
    skills_temporary: tempfile.TemporaryDirectory
    output_temporary: tempfile.TemporaryDirectory
    started: float
    deadline: float
    session_jsonl: bytes = b""


class DockerPiAgentRunner:
    def __init__(self, config: AgentEngineConfig, *, repo_root: Path | None = None) -> None:
        self.config = config
        self.model_id = config.model_id
        self.timeout_seconds = config.timeout_seconds
        self.session_store_dir = Path(config.session_store_dir).expanduser().resolve()
        self.repo_root = repo_root or Path(__file__).resolve().parents[3]
        self._active_runs: dict[str, _ActiveRun] = {}
        self._active_runs_lock = threading.Lock()

    def _docker(self, args: list[str], *, timeout: int = 30,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
        options: dict[str, Any] = {"capture_output": True, "text": True, "timeout": timeout, "check": False}
        agent_docker_host = os.environ.get("REVIEW_BOT_AGENT_DOCKER_HOST")
        if env is not None or agent_docker_host:
            docker_env = dict(env if env is not None else os.environ)
            if agent_docker_host:
                docker_env["DOCKER_HOST"] = agent_docker_host
                docker_env.pop("DOCKER_CONTEXT", None)
            options["env"] = docker_env
        return subprocess.run(["docker", *args], **options)

    @staticmethod
    def _runtime_uid_gid() -> tuple[int, int]:
        uid = os.getuid() if hasattr(os, "getuid") else 10001
        gid = os.getgid() if hasattr(os, "getgid") else 10001
        return (uid, gid) if uid != 0 else (10001, 10001)

    def _agent_argv(
        self,
        run: _ActiveRun,
        name: str,
        prompt: str,
        session_id: str | None = None,
    ) -> list[str]:
        uid, gid = self._runtime_uid_gid()
        argv = [
            "run", "--name", name, "--network", run.network, "--read-only",
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=512m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--memory", "2g", "--user", f"{uid}:{gid}",
            "--workdir", "/workspace",
            "--volume", f"{run.workspace}:/workspace:ro",
            "--volume", f"{run.skills_dir}:/skills:ro",
            "--volume", f"{run.output_dir}:/out:rw",
            "--tmpfs", f"/agent:rw,nosuid,nodev,size=1m,uid={uid},gid={gid},mode=700",
            "--volume", f"{run.config_dir / 'models.json'}:/agent/models.json:ro",
            "--volume", f"{run.config_dir / 'settings.json'}:/agent/settings.json:ro",
            "--volume", f"{run.session_dir}:/sessions:rw",
            "--env", "PI_CODING_AGENT_DIR=/agent",
            "--env", "PI_CODING_AGENT_SESSION_DIR=/sessions",
            "--env", "HOME=/tmp",
            "--env", "PI_TELEMETRY=0",
            self.config.image_tag,
            "--model", f"llmproxy/{run.request.model}",
            "--skill", f"/skills/{run.request.skill_id}/SKILL.md",
            "--session-dir", "/sessions",
            "--tools", "read,grep,find,ls,bash",
            "--no-extensions", "--no-context-files", "--no-prompt-templates", "--no-themes",
            "--thinking", "medium",
        ]
        if session_id is not None:
            argv.extend(["--session", session_id])
        argv.extend(["--print", prompt])
        return argv

    def _proxy_argv(self, network: str, name: str, model: str, ledger_dir: Path,
        upstream_session_id: str) -> list[str]:
        uid, gid = self._runtime_uid_gid()
        argv = [
            "run", "-d", "--name", name, "--network", "bridge", "--read-only",
            "--tmpfs", "/tmp:rw,exec,nosuid,nodev,size=32m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "256m", "--user", f"{uid}:{gid}",
            "--volume", f"{ledger_dir}:/ledger:rw",
            "--env", "UPSTREAM_API_KEY",
            "--env", f"PROVIDER={self.config.provider}",
            "--env", f"UPSTREAM_SESSION_ID={upstream_session_id}",
            "--env", f"MODEL_ID={model}",
            "--env", f"MAX_PROXY_CALLS={self.config.max_proxy_calls}",
        ]
        if self.config.max_output_tokens is not None:
            argv.extend(["--env", f"MAX_OUTPUT_TOKENS={self.config.max_output_tokens}"])
        argv.extend(["--env", "USAGE_LEDGER_PATH=/ledger/usage.jsonl", PROXY_IMAGE])
        return argv

    def _upstream_api_key(self) -> str:
        variables = {
            "openrouter": ("OPENROUTER_API_KEY_FILE", "OPENROUTER_API_KEY"),
            "opencode-go": ("OPENCODE_API_KEY_FILE", "OPENCODE_API_KEY"),
        }
        file_variable, key_variable = variables[self.config.provider]
        key_file = os.environ.get(file_variable, "").strip()
        if key_file:
            try:
                value = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                value = ""
            if value:
                return value
        return os.environ.get(key_variable, "").strip()

    @staticmethod
    def _session_id(session: bytes) -> str | None:
        for line in session.splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict) and record.get("type") == "session":
                value = record.get("id")
                return value if isinstance(value, str) and value else None
        return None

    @staticmethod
    def _agent_reported_usage(session: bytes) -> dict[str, Any]:
        input_tokens = output_tokens = total_tokens = model_calls = 0
        cost: dict[str, float] = {}
        for line in session.splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            message = record.get("message", {}) if isinstance(record, dict) else {}
            usage = message.get("usage", {}) if isinstance(message, dict) else {}
            if not isinstance(usage, dict) or not usage:
                continue
            model_calls += 1
            input_tokens += int(usage.get("input", usage.get("input_tokens", 0)) or 0)
            output_tokens += int(usage.get("output", usage.get("output_tokens", 0)) or 0)
            total_tokens += int(usage.get("totalTokens", usage.get("total_tokens", 0)) or 0)
            nested = usage.get("cost")
            if isinstance(nested, dict):
                for key, value in nested.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        cost[key] = cost.get(key, 0.0) + float(value)
            elif isinstance(nested, (int, float)) and not isinstance(nested, bool):
                cost["total"] = cost.get("total", 0.0) + float(nested)
        result: dict[str, Any] = {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens, "model_calls": model_calls,
        }
        if cost:
            result["cost"] = cost
        return result

    @staticmethod
    def _session_compactions(session: bytes) -> int:
        compactions = 0
        for line in session.splitlines():
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict) and record.get("type") == "compaction":
                compactions += 1
        return compactions

    @staticmethod
    def _proxy_usage(ledger_path: Path, session: bytes = b"") -> dict[str, Any]:
        calls = []
        if ledger_path.is_file():
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    calls.append(row)
        input_tokens = output_tokens = total_tokens = unknown = 0
        cost: dict[str, float] = {}
        for row in calls:
            prompt = row.get("prompt_tokens")
            completion = row.get("completion_tokens")
            if type(prompt) is int and type(completion) is int:
                input_tokens += prompt
                output_tokens += completion
                total_tokens += row.get("total_tokens", prompt + completion)
            else:
                unknown += 1
            for key, value in row.get("cost", {}).items() if isinstance(row.get("cost"), dict) else ():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    cost[key] = cost.get(key, 0.0) + float(value)
        max_prompt_tokens = max(
            (row["prompt_tokens"] for row in calls if type(row.get("prompt_tokens")) is int),
            default=None,
        )
        result: dict[str, Any] = {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens, "model_calls": len(calls),
            "known_input_tokens": input_tokens, "known_output_tokens": output_tokens,
            "usage_complete": unknown == 0, "unknown_usage_attempts": unknown,
            "calls": calls, "max_prompt_tokens": max_prompt_tokens,
            "compactions": DockerPiAgentRunner._session_compactions(session),
        }
        if cost:
            result["cost"] = cost
        return result

    def _start(self, request: AgentRunRequest) -> tuple[str, _ActiveRun]:
        allowed_skills = {
            "architecture": "review-architecture",
            "intent": "review-intent",
            "verify": "verify-claims",
        }
        if request.mode not in allowed_skills or request.skill_id != allowed_skills[request.mode]:
            raise AgentRunError("invalid_request")
        upstream_api_key = self._upstream_api_key()
        if not upstream_api_key:
            missing_key_code = ("openrouter_api_key_missing" if self.config.provider == "openrouter"
                else "opencode_api_key_missing")
            raise AgentRunError(missing_key_code)
        skill = agent_skill_path(request.mode)
        source_skills_dir = agent_skills_dir()
        checker_path = agent_checker_path()
        if not skill.is_file() or not source_skills_dir.is_dir() or not checker_path.is_file():
            raise AgentRunError("agent_skill_unavailable")
        suffix = uuid.uuid4().hex[:16]
        network = f"review-agent-net-{suffix}"
        proxy_name = f"review-agent-proxy-{suffix}"
        agent_name = f"review-agent-{suffix}"
        upstream_session_id = f"ses_{uuid.uuid4().hex}"
        self.session_store_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        ledger_root = self.session_store_dir / "proxy-ledgers"
        ledger_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        ledger_dir = ledger_root / suffix
        ledger_dir.mkdir(mode=0o700)
        uid, gid = self._runtime_uid_gid()
        if os.getuid() == 0 and hasattr(os, "chown"):
            os.chown(ledger_dir, uid, gid)
        ledger_path = ledger_dir / "usage.jsonl"
        config_temporary = tempfile.TemporaryDirectory(prefix="review-agent-config-")
        session_temporary = tempfile.TemporaryDirectory(prefix="review-agent-session-")
        skills_temporary = tempfile.TemporaryDirectory(prefix="review-agent-skills-")
        output_temporary = tempfile.TemporaryDirectory(prefix="review-agent-output-")
        config_dir, session_dir = Path(config_temporary.name), Path(session_temporary.name)
        skills_dir = Path(skills_temporary.name)
        output_dir = Path(output_temporary.name)
        try:
            shutil.copytree(source_skills_dir, skills_dir, dirs_exist_ok=True)
            shutil.copyfile(checker_path, skills_dir / "check_output.mjs")
            for directory, subdirectories, files in os.walk(skills_dir):
                os.chmod(directory, 0o755)
                for filename in files:
                    os.chmod(Path(directory) / filename, 0o644)
            model_config = {
                "id": request.model, "name": request.model, "reasoning": True,
            }
            if self.config.max_output_tokens is not None:
                model_config["maxTokens"] = self.config.max_output_tokens
            if self.config.context_window_tokens is not None:
                model_config["contextWindow"] = self.config.context_window_tokens
            models = {"providers": {"llmproxy": {
                "baseUrl": "http://llm-proxy:8080/v1", "apiKey": "DUMMY_NOT_A_KEY",
                "api": "openai-completions", "models": [model_config],
            }}}
            (config_dir / "models.json").write_text(json.dumps(models), encoding="utf-8")
            os.chmod(config_dir / "models.json", 0o444)
            settings_path = config_dir / "settings.json"
            settings_path.write_text(json.dumps({
                "shellCommandPrefix": f"ulimit -t {_SHELL_CPU_SECONDS}",
            }), encoding="utf-8")
            os.chmod(settings_path, 0o444)
            os.chmod(config_dir, 0o755)
            if os.getuid() == 0 and hasattr(os, "chown"):
                os.chown(session_dir, uid, gid)
                os.chown(output_dir, uid, gid)
            os.chmod(session_dir, 0o700)
            os.chmod(output_dir, 0o700)
            created = self._docker(["network", "create", "--internal", "--driver", "bridge", network])
            if created.returncode != 0:
                raise AgentRunError("docker_network_failed")
            proxy_env = os.environ.copy()
            proxy_env["UPSTREAM_API_KEY"] = upstream_api_key
            proxy = self._docker(self._proxy_argv(network, proxy_name, request.model, ledger_dir,
                upstream_session_id), env=proxy_env)
            if proxy.returncode != 0:
                raise AgentRunError("proxy_start_failed")
            connected = self._docker(["network", "connect", "--alias", "llm-proxy", network, proxy_name])
            if connected.returncode != 0:
                raise AgentRunError("proxy_network_failed")
            now = time.monotonic()
            run = _ActiveRun(
                network, proxy_name, agent_name, config_dir, session_dir, ledger_path, ledger_dir,
                request, request.workspace_dir.resolve(), skills_dir.resolve(), output_dir,
                config_temporary, session_temporary, skills_temporary, output_temporary,
                now, now + max(1, request.timeout_seconds),
            )
            return suffix, run
        except Exception:
            try:
                self._cleanup_names(agent_name, proxy_name, network)
            finally:
                config_temporary.cleanup()
                session_temporary.cleanup()
                skills_temporary.cleanup()
                output_temporary.cleanup()
            raise

    def _invoke(self, run: _ActiveRun, prompt: str, session_id: str | None) -> AgentRunResponse:
        remaining = run.deadline - time.monotonic()
        if remaining <= 0:
            raise AgentRunError("timeout")
        try:
            completed = self._docker(self._agent_argv(run, run.agent_name, prompt, session_id),
                timeout=max(1, math.ceil(remaining)))
        except subprocess.TimeoutExpired:
            raise AgentRunError("timeout") from None
        except OSError:
            raise AgentRunError("docker_runner_failed") from None
        session_files = sorted(run.session_dir.rglob("*.jsonl"), key=lambda path: path.stat().st_mtime)
        if session_files:
            run.session_jsonl = session_files[-1].read_bytes()
        if completed.returncode != 0:
            raise AgentRunError("agent_crashed")
        try:
            logs = self._docker(["logs", run.proxy_name], timeout=10)
        except (OSError, subprocess.SubprocessError):
            raise AgentRunError("docker_runner_failed") from None
        if "proxy_budget_exhausted" in (logs.stdout or ""):
            raise AgentRunError("proxy_budget_exhausted")
        resolved_session = self._session_id(run.session_jsonl)
        if resolved_session is None:
            raise AgentRunError("agent_session_unavailable")
        with suppress(Exception):
            self._docker(["rm", "-f", run.agent_name], timeout=15)
        usage = self._agent_reported_usage(run.session_jsonl)
        try:
            proxy_usage = self._proxy_usage(run.ledger_path, run.session_jsonl)
        except OSError:
            proxy_usage = {"max_prompt_tokens": None,
                "compactions": self._session_compactions(run.session_jsonl)}
        usage.update(max_prompt_tokens=proxy_usage["max_prompt_tokens"],
            compactions=proxy_usage["compactions"])
        return AgentRunResponse(completed.stdout, run.session_jsonl,
            usage, PI_VERSION, completed.returncode, resolved_session,
            run.output_dir / "result.json")

    def run(self, request: AgentRunRequest) -> AgentRunResponse:
        try:
            _, run = self._start(request)
            response = self._invoke(run, _initial_agent_prompt(request.mode), None)
            with self._active_runs_lock:
                self._active_runs[response.session_id] = run
            return response
        except AgentRunError:
            if "run" in locals():
                try:
                    self._cleanup_names(run.agent_name, run.proxy_name, run.network)
                finally:
                    run.config_temporary.cleanup()
                    run.session_temporary.cleanup()
                    run.skills_temporary.cleanup()
                    run.output_temporary.cleanup()
            raise
        except (OSError, subprocess.SubprocessError):
            if "run" in locals():
                try:
                    self._cleanup_names(run.agent_name, run.proxy_name, run.network)
                finally:
                    run.config_temporary.cleanup()
                    run.session_temporary.cleanup()
                    run.skills_temporary.cleanup()
                    run.output_temporary.cleanup()
            raise AgentRunError("docker_runner_failed") from None

    def repair(self, request: AgentRunRequest, session_id: str, message: str, *,
        timeout_seconds: int | None = None) -> AgentRunResponse:
        with self._active_runs_lock:
            run = self._active_runs.get(session_id)
        if run is None or run.request != request:
            raise AgentRunError("agent_session_unknown")
        if timeout_seconds is not None:
            run.deadline = time.monotonic() + timeout_seconds
        return self._invoke(run, message, session_id)

    def finish(self, session_id: str) -> AgentRunFinalization:
        with self._active_runs_lock:
            run = self._active_runs.pop(session_id, None)
        if run is None:
            raise AgentRunError("agent_session_unknown")
        duration_ms = int((time.monotonic() - run.started) * 1000)
        session = run.session_jsonl
        try:
            self._cleanup_names(run.agent_name, run.proxy_name, run.network)
            try:
                usage = self._proxy_usage(run.ledger_path, session)
            except OSError:
                raise AgentRunError("usage_ledger_unavailable") from None
            return AgentRunFinalization(session, usage, self._agent_reported_usage(session),
                str(run.ledger_path), duration_ms)
        finally:
            run.config_temporary.cleanup()
            run.session_temporary.cleanup()
            run.skills_temporary.cleanup()
            run.output_temporary.cleanup()

    def _cleanup_names(self, agent_name: str, proxy_name: str, network: str) -> None:
        failures = []
        for name in (agent_name, proxy_name):
            removed = False
            for _attempt in range(2):
                try:
                    result = self._docker(["rm", "-f", name], timeout=15)
                    if result.returncode == 0 or "no such container" in result.stderr.lower():
                        removed = True
                        break
                except (OSError, subprocess.SubprocessError):
                    continue
            if not removed:
                failures.append(name)
        removed = False
        for _attempt in range(2):
            try:
                result = self._docker(["network", "rm", network], timeout=15)
                if (result.returncode == 0 or "no such network" in result.stderr.lower()
                        or "not found" in result.stderr.lower()):
                    removed = True
                    break
            except (OSError, subprocess.SubprocessError):
                continue
        if not removed:
            failures.append(network)
        if failures:
            raise AgentRunError("docker_cleanup_failed")
