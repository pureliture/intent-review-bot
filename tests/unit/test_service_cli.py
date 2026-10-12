"""Service CLI orchestration without a server, network, daemon or model calls."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.domain.envelope_v2 import PrRequestRecord
from review_bot.runtime.cli import main

SECRET = "synthetic-step7-secret"
IMAGE = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def no_external_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("External runtime execution is forbidden in service CLI tests")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    monkeypatch.setattr(uvicorn, "run", Mock())
    monkeypatch.setattr("review_bot.runtime.worker.WorkerLoop._setup_signals", lambda self: None)


@pytest.fixture
def service_config(tmp_path: Path, valid_config_yaml: str, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path.resolve()
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="github_app", backend="github_app", worker_concurrency=1)
    raw["publish"].pop("mode", None)
    raw["model"]["profile"] = None
    raw["service"].update(
        state_root_env="STEP7_STATE_ROOT", app_id_env="STEP7_APP_ID",
        app_private_key_file_env="STEP7_KEY_FILE", webhook_secret_env="STEP7_SECRET",
        installation_allowlist=[], repository_allowlist=[],
    )
    raw["architecture"]["tool_image_digest"] = IMAGE
    raw["architecture"]["tool_timeout_seconds"] = 17
    raw["architecture"]["isolation"].update(memory_mib=3072, pids=64, cpus=1)
    path = root / "operator.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    key = root / "synthetic-key.pem"
    key.write_text("not a real private key", encoding="utf-8")
    key.chmod(0o600)
    state = root / "state"
    monkeypatch.setenv("STEP7_STATE_ROOT", str(state))
    monkeypatch.setenv("STEP7_APP_ID", "1001")
    monkeypatch.setenv("STEP7_KEY_FILE", str(key))
    monkeypatch.setenv("STEP7_SECRET", SECRET)
    return SimpleNamespace(path=path, state=state, key=key, raw=raw)


def current_umask() -> int:
    previous = os.umask(0o077)
    os.umask(previous)
    return previous


def test_webhook_runs_signed_app_with_private_sqlite(
    service_config, monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = current_umask()
    served = []

    def serve(app, **kwargs):
        assert isinstance(app, FastAPI)
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8765
        assert current_umask() == 0o077
        with TestClient(app) as client:
            assert client.get("/health").json() == {"status": "ok"}
            body = json.dumps({
                "action": "opened", "number": 7, "installation": {"id": 44},
                "repository": {"id": 99, "name": "repo", "full_name": "org/repo"},
                "pull_request": {
                    "number": 7, "state": "open",
                    "head": {"sha": "b" * 40, "ref": "feature/test"},
                    "base": {"sha": "a" * 40, "ref": "main"},
                },
            }).encode()
            headers = {
                "X-GitHub-Event": "pull_request", "X-GitHub-Delivery": "cli-delivery",
                "X-Hub-Signature-256": "sha256=" + hmac.new(
                    SECRET.encode(), body, hashlib.sha256,
                ).hexdigest(),
            }
            assert client.post("/webhook", content=body, headers=headers).json() == {"status": "accepted"}
        served.append(app)

    monkeypatch.setattr(uvicorn, "run", serve)
    assert main(["webhook", "--config", str(service_config.path), "--port", "8765"]) == 0
    assert len(served) == 1
    assert service_config.state.stat().st_mode & 0o777 == 0o700
    assert (service_config.state / "jobs.sqlite").stat().st_mode & 0o777 == 0o600
    assert current_umask() == before


@pytest.mark.parametrize("command", [["webhook"], ["worker", "--once"]])
def test_service_commands_require_operator_config(command, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(command) == 1
    captured = capsys.readouterr()
    assert "--config" in captured.err
    assert "initialized" not in captured.out


@pytest.fixture
def dependencies(monkeypatch: pytest.MonkeyPatch):
    from review_bot.runtime import service

    opened = []

    def open_store(path):
        job_store = SQLiteJobStore(path)
        job_store.close = Mock(wraps=job_store.close)
        opened.append(job_store)
        return job_store

    loop = Mock(worker_id="cli-test", lease_duration=30.0)
    loop.process_run.return_value = True
    analyzer = Mock()
    provider = Mock()
    adapters = Mock()
    store_factory = Mock(side_effect=open_store)
    analyzer_factory = Mock(return_value=analyzer)
    worker_factory = Mock(return_value=loop)
    provider_factory = Mock(return_value=provider)
    adapters_factory = Mock(return_value=adapters)
    monkeypatch.setattr(service, "SQLiteJobStore", store_factory)
    monkeypatch.setattr(service, "RootlessPmdAnalyzer", analyzer_factory)
    monkeypatch.setattr(service, "WorkerLoop", worker_factory)
    monkeypatch.setattr(service, "GitHubAppTokenProvider", provider_factory)
    monkeypatch.setattr(service, "GitHubAppAdapters", adapters_factory)
    return SimpleNamespace(
        opened=opened, loop=loop, analyzer=analyzer, provider=provider, adapters=adapters,
        store_factory=store_factory, analyzer_factory=analyzer_factory,
        worker_factory=worker_factory, provider_factory=provider_factory,
        adapters_factory=adapters_factory,
    )


def seed_run(service_config) -> None:
    previous = os.umask(0o077)
    try:
        service_config.state.mkdir(mode=0o700, exist_ok=True)
        job_store = SQLiteJobStore(service_config.state / "jobs.sqlite")
        try:
            job_store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        finally:
            job_store.close()
    finally:
        os.umask(previous)


def write_config(service_config) -> None:
    service_config.path.write_text(yaml.safe_dump(service_config.raw), encoding="utf-8")


def test_worker_once_empty_composes_ports_without_reading_key(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = current_umask()
    monkeypatch.setenv("STEP7_KEY_FILE", str(service_config.key.parent / "absent-key.pem"))
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 0
    config, job_store = dependencies.worker_factory.call_args.args
    ports = dependencies.worker_factory.call_args.kwargs
    assert config.publish.mode == "comment"
    assert config.service.installation_allowlist == []
    assert config.service.repository_allowlist == []
    dependencies.analyzer_factory.assert_not_called()
    assert ports["analyzer_factory"]() is dependencies.analyzer
    assert callable(ports["github_client_factory"])
    assert callable(ports["repo_reader_factory"])
    assert ports["artifact_root"] == service_config.state / "artifacts"
    dependencies.analyzer_factory.assert_called_once_with(
        IMAGE, service_config.state / "analysis", memory_mib=3072, pids=64, cpus=1,
        timeout_seconds=17,
    )
    dependencies.loop.prepare_models.assert_called_once_with()
    dependencies.loop.process_run.assert_not_called()
    dependencies.loop.run_forever.assert_not_called()
    dependencies.provider_factory.assert_not_called()
    dependencies.adapters_factory.assert_not_called()
    job_store.close.assert_called_once_with()
    assert current_umask() == before


@pytest.mark.parametrize("success, expected", [(True, 0), (False, 1)])
def test_worker_once_distinguishes_claimed_failure_from_empty(
    service_config, dependencies, success: bool, expected: int,
) -> None:
    seed_run(service_config)

    def process(run):
        config, job_store = dependencies.worker_factory.call_args.args
        ports = dependencies.worker_factory.call_args.kwargs
        assert dependencies.loop.prepare_models.call_count == 1
        with job_store.transaction(immediate=False) as connection:
            assert connection.execute("SELECT status FROM runs").fetchone()[0] == "running"
        dependencies.provider_factory.assert_not_called()
        client = ports["github_client_factory"](run)
        pr_data = {"number": 7}
        assert ports["repo_reader_factory"](run, pr_data, client) is dependencies.adapters.repository_cache.prepare.return_value
        dependencies.adapters.repository_cache.prepare.assert_called_once_with(run, pr_data, client)
        dependencies.provider_factory.assert_called_once_with("1001", service_config.key)
        dependencies.adapters_factory.assert_called_once_with(
            dependencies.provider, service_config.state / "repositories", config.service,
        )
        return success

    dependencies.loop.process_run.side_effect = process
    assert main(["worker", "--config", str(service_config.path), "--once"]) == expected
    dependencies.loop.process_run.assert_called_once()
    dependencies.opened[0].close.assert_called_once_with()


def test_worker_forever_uses_existing_loop_and_closes_store(service_config, dependencies) -> None:
    assert main(["worker", "--config", str(service_config.path)]) == 0
    dependencies.loop.run_forever.assert_called_once_with()
    dependencies.loop.process_run.assert_not_called()
    dependencies.loop.prepare_models.assert_not_called()
    dependencies.provider_factory.assert_not_called()
    dependencies.opened[0].close.assert_called_once_with()


@pytest.mark.parametrize("command", ["webhook", "worker"])
@pytest.mark.parametrize("missing", ["STEP7_APP_ID", "STEP7_KEY_FILE", "STEP7_SECRET", "STEP7_STATE_ROOT"])
def test_service_requires_active_environment(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch,
    command: str, missing: str,
) -> None:
    monkeypatch.delenv(missing)
    assert main([command, "--config", str(service_config.path)]) == 1
    dependencies.store_factory.assert_not_called()
    dependencies.analyzer_factory.assert_not_called()
    uvicorn.run.assert_not_called()


@pytest.mark.parametrize("section, field, value", [
    ("execution", "mode", "offline"),
    ("execution", "backend", "mock"),
    ("execution", "worker_concurrency", 2),
    ("service", "worker", "asynchronous"),
    ("service", "sqlite_wal", False),
    ("service", "lease_and_generation_fencing", False),
    ("architecture", "tool_image_digest", None),
    ("architecture", "tool_image_digest", "trusted/analyzer:latest"),
    ("architecture", "tool_image_digest", "trusted/analyzer@sha256:abcd"),
])
def test_service_rejects_unsupported_configuration(
    service_config, dependencies, section: str, field: str, value,
    capsys: pytest.CaptureFixture[str],
) -> None:
    service_config.raw[section][field] = value
    write_config(service_config)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()
    dependencies.analyzer_factory.assert_not_called()
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("field, value", [
    ("runtime", "subprocess"), ("read_only_root", False), ("no_new_privileges", False),
])
def test_service_rejects_weakened_rootless_isolation(
    service_config, dependencies, field: str, value,
) -> None:
    service_config.raw["architecture"]["isolation"][field] = value
    write_config(service_config)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()
    dependencies.analyzer_factory.assert_not_called()


@pytest.mark.parametrize("permissions", [0o755, 0o770, 0o777])
def test_service_rejects_shared_state_without_chmod(
    service_config, dependencies, permissions: int,
) -> None:
    service_config.state.mkdir(mode=permissions)
    service_config.state.chmod(permissions)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    assert service_config.state.stat().st_mode & 0o777 == permissions
    dependencies.store_factory.assert_not_called()


@pytest.mark.parametrize("unsafe", ["relative", "traversal", "symlink", "parent_symlink", "shared_parent", "root", "home", "double_slash"])
def test_service_rejects_unsafe_state_paths(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch, unsafe: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = service_config.state
    if unsafe == "relative":
        path = Path("relative-state")
    elif unsafe == "traversal":
        path = path.parent / ".." / "state"
    elif unsafe == "symlink":
        target = path.parent / "other-state"
        target.mkdir(mode=0o700)
        path.symlink_to(target, target_is_directory=True)
    elif unsafe == "parent_symlink":
        target = path.parent / "other-state"
        target.mkdir(mode=0o700)
        path.symlink_to(target, target_is_directory=True)
        path = path / "child"
    elif unsafe == "shared_parent":
        path.mkdir(mode=0o700)
        path.chmod(0o777)
        path = path / "child"
    elif unsafe == "home":
        path = Path.home()
    elif unsafe == "double_slash":
        path = Path("/" + str(Path.home()))
    else:
        path = Path("/")
    monkeypatch.setenv("STEP7_STATE_ROOT", str(path))
    before = current_umask()
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()
    assert current_umask() == before
    captured = capsys.readouterr()
    if str(path) != "/":
        assert str(path) not in captured.out + captured.err


@pytest.mark.parametrize("name", ["jobs.sqlite", "jobs.sqlite-wal", "jobs.sqlite-shm", "jobs.sqlite-journal", "repositories", "analysis", "artifacts"])
def test_service_rejects_symlinked_managed_paths(service_config, dependencies, name: str) -> None:
    service_config.state.mkdir(mode=0o700)
    target = service_config.key if name.startswith("jobs.sqlite") else service_config.key.parent
    (service_config.state / name).symlink_to(target)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()


@pytest.mark.parametrize("unsafe", ["shared", "hardlink", "foreign_owner"])
def test_service_rejects_unsafe_database(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch, unsafe: str,
) -> None:
    service_config.state.mkdir(mode=0o700)
    database = service_config.state / "jobs.sqlite"
    database.write_bytes(b"")
    database.chmod(0o600)
    if unsafe == "shared":
        database.chmod(0o644)
    elif unsafe == "hardlink":
        (service_config.state / "database-copy").hardlink_to(database)
    else:
        original = Path.lstat

        def foreign(path):
            metadata = original(path)
            if path == database:
                values = list(metadata)
                values[4] = os.geteuid() + 1
                return os.stat_result(values)
            return metadata

        monkeypatch.setattr(Path, "lstat", foreign)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()


@pytest.mark.parametrize("failure", ["analyzer", "worker", "loop", "uvicorn", "app"])
def test_service_failures_close_store_restore_umask_and_redact_values(
    service_config, dependencies, worker_pipeline_ports, monkeypatch: pytest.MonkeyPatch,
    failure: str, capsys: pytest.CaptureFixture[str],
) -> None:
    from review_bot.runtime import service

    message = SECRET + str(service_config.key) + str(service_config.state)
    error = RuntimeError(message)
    command = "worker"
    if failure == "analyzer":
        from review_bot.adapters.github.client import MockGitHubClient
        from review_bot.runtime.worker import WorkerLoop

        seed_run(service_config)
        github = MockGitHubClient()
        github.set_pull_request(7, repository="org/repo")
        github.pull_requests[7]["base"]["repo"] = {"id": 99, "full_name": "org/repo"}
        dependencies.adapters.github_client.return_value = github
        dependencies.adapters.repository_cache.prepare.return_value = worker_pipeline_ports["repo_reader"]
        monkeypatch.setattr(service, "WorkerLoop", WorkerLoop)
        dependencies.analyzer_factory.side_effect = error
    elif failure == "worker":
        dependencies.worker_factory.side_effect = error
    elif failure == "loop":
        dependencies.loop.run_forever.side_effect = error
    elif failure == "uvicorn":
        command = "webhook"
        uvicorn.run.side_effect = error
    else:
        command = "webhook"
        monkeypatch.setattr(service, "create_webhook_app", Mock(side_effect=error))
    before = current_umask()
    args = [command, "--config", str(service_config.path)]
    if failure == "analyzer":
        args.append("--once")
    assert main(args) == 1
    assert current_umask() == before
    dependencies.opened[0].close.assert_called_once_with()
    if failure == "analyzer":
        dependencies.analyzer_factory.assert_called_once()
        connection = sqlite3.connect(service_config.state / "jobs.sqlite")
        try:
            assert connection.execute("SELECT status, stage, error FROM runs").fetchone() == (
                "failed", "analyze", "worker_pipeline_error:RuntimeError")
        finally:
            connection.close()
    else:
        dependencies.provider_factory.assert_not_called()
    captured = capsys.readouterr()
    for value in (SECRET, str(service_config.key), str(service_config.state)):
        assert value not in captured.out + captured.err


def test_invalid_config_does_not_echo_raw_values(
    service_config, dependencies, capsys: pytest.CaptureFixture[str],
) -> None:
    service_config.raw["execution"]["mode"] = SECRET
    write_config(service_config)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    dependencies.store_factory.assert_not_called()


def test_webhook_honors_host_port_and_closes_store(service_config, dependencies) -> None:
    assert main([
        "webhook", "--config", str(service_config.path), "--host", "0.0.0.0", "--port", "9012",
    ]) == 0
    app = uvicorn.run.call_args.args[0]
    assert isinstance(app, FastAPI)
    assert uvicorn.run.call_args.kwargs == {"host": "0.0.0.0", "port": 9012}
    dependencies.opened[0].close.assert_called_once_with()
    dependencies.provider_factory.assert_not_called()
    dependencies.analyzer_factory.assert_not_called()


@pytest.mark.parametrize("port", ["0", "65536"])
def test_webhook_rejects_invalid_port_before_opening_state(service_config, dependencies, port: str) -> None:
    assert main(["webhook", "--config", str(service_config.path), "--port", port]) == 1
    dependencies.store_factory.assert_not_called()
    uvicorn.run.assert_not_called()


@pytest.mark.parametrize("name", ["STEP7_APP_ID", "STEP7_KEY_FILE", "STEP7_SECRET", "STEP7_STATE_ROOT"])
def test_service_rejects_blank_environment_references(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    monkeypatch.setenv(name, " \t ")
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()


def test_service_calls_canonical_environment_validator(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from review_bot.runtime import service

    validate = Mock(wraps=service.validate_active_environment)
    monkeypatch.setattr(service, "validate_active_environment", validate)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 0
    validate.assert_called_once_with(dependencies.worker_factory.call_args.args[0])


@pytest.mark.parametrize("name", ["state", "repositories", "analysis", "artifacts"])
def test_service_rejects_foreign_owned_state_directories(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    service_config.state.mkdir(mode=0o700)
    path = service_config.state if name == "state" else service_config.state / name
    path.mkdir(mode=0o700, exist_ok=True)
    original = Path.lstat

    def foreign(target):
        metadata = original(target)
        if target == path:
            values = list(metadata)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(Path, "lstat", foreign)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    dependencies.store_factory.assert_not_called()


@pytest.mark.parametrize("unsafe", ["shared", "symlink", "hardlink", "missing"])
def test_claimed_worker_rejects_unsafe_key_and_records_failure(
    service_config, dependencies, monkeypatch: pytest.MonkeyPatch, unsafe: str,
    capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    from review_bot.runtime import service
    from review_bot.runtime.worker import WorkerLoop

    seed_run(service_config)
    if unsafe == "shared":
        service_config.key.chmod(0o644)
    elif unsafe == "symlink":
        link = service_config.key.parent / "key-link.pem"
        link.symlink_to(service_config.key)
        monkeypatch.setenv("STEP7_KEY_FILE", str(link))
    elif unsafe == "hardlink":
        (service_config.key.parent / "key-copy.pem").hardlink_to(service_config.key)
    else:
        monkeypatch.setenv("STEP7_KEY_FILE", str(service_config.key.parent / "absent.pem"))
    monkeypatch.setattr(service, "WorkerLoop", WorkerLoop)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    connection = sqlite3.connect(service_config.state / "jobs.sqlite")
    try:
        status, error = connection.execute("SELECT status, error FROM runs").fetchone()
        assert status == "failed"
        assert error == "worker_pipeline_error:ServiceConfigurationError"
    finally:
        connection.close()
    dependencies.provider_factory.assert_not_called()
    dependencies.adapters_factory.assert_not_called()
    dependencies.opened[0].close.assert_called_once_with()
    captured = capsys.readouterr()
    assert str(service_config.key) not in captured.out + captured.err + caplog.text


def test_rootless_preflight_failure_never_falls_back(
    service_config, dependencies, worker_pipeline_ports, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from review_bot.adapters.pmd.container import RootlessPmdAnalyzer
    from review_bot.adapters.github.client import MockGitHubClient
    from review_bot.runtime import service
    from review_bot.runtime.worker import WorkerLoop

    seed_run(service_config)
    github = MockGitHubClient()
    github.set_pull_request(7, repository="org/repo")
    github.pull_requests[7]["base"]["repo"] = {"id": 99, "full_name": "org/repo"}
    dependencies.adapters.github_client.return_value = github
    dependencies.adapters.repository_cache.prepare.return_value = worker_pipeline_ports["repo_reader"]
    monkeypatch.setattr(service, "WorkerLoop", WorkerLoop)

    docker = Mock(side_effect=OSError(SECRET + str(service_config.state)))
    monkeypatch.setattr(RootlessPmdAnalyzer, "_docker", docker)
    monkeypatch.setattr(service, "RootlessPmdAnalyzer", RootlessPmdAnalyzer)
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 1
    docker.assert_called_once_with(["info", "--format", "{{json .}}"], timeout=10)
    dependencies.analyzer_factory.assert_not_called()
    connection = sqlite3.connect(service_config.state / "jobs.sqlite")
    try:
        status, stage, error = connection.execute("SELECT status, stage, error FROM runs").fetchone()
        assert (status, stage) == ("failed", "analyze")
        assert error == "worker_pipeline_error:AnalysisSetupError"
    finally:
        connection.close()
    dependencies.opened[0].close.assert_called_once_with()
    assert SECRET not in capsys.readouterr().err


@pytest.mark.parametrize("command", ["worker", "webhook"])
def test_intent_only_service_has_no_architecture_or_java_prerequisite(service_config, dependencies, command) -> None:
    service_config.raw["execution"]["available_modes"] = ["intent"]
    service_config.raw["publish"]["mode"] = "artifact_only"
    service_config.raw.pop("architecture")
    write_config(service_config)
    args = [command, "--config", str(service_config.path)]
    if command == "worker":
        args.append("--once")
    assert main(args) == 0
    dependencies.analyzer_factory.assert_not_called()
    assert not (service_config.state / "analysis").exists()
    if command == "worker":
        assert dependencies.worker_factory.call_args.kwargs["analyzer_factory"] is None


@pytest.mark.parametrize("operation", ["create", "update", "artifact_only"])
def test_signed_cli_intake_to_real_worker_with_mock_github(
    service_config, dependencies, worker_pipeline_ports,
    monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    from review_bot.adapters.github.client import MockGitHubClient
    from review_bot.runtime import service
    from review_bot.runtime.worker import WorkerLoop

    github = MockGitHubClient()
    github.set_pull_request(7, repository="org/repo")
    github.pull_requests[7]["base"]["repo"] = {"id": 99, "full_name": "org/repo"}
    github.pull_requests[7].update(title="Synthetic architecture change", body="Synthetic PR", draft=False)
    marker = service_config.raw["publish"]["a_marker"]
    if operation == "update":
        github.create_comment(99, 7, marker + "\nOld owned A comment")
    elif operation == "artifact_only":
        service_config.raw["publish"]["mode"] = "artifact_only"
        write_config(service_config)
    dependencies.analyzer_factory.return_value = worker_pipeline_ports["analyzer"]
    dependencies.adapters.github_client.return_value = github
    dependencies.adapters.repository_cache.prepare.return_value = worker_pipeline_ports["repo_reader"]
    monkeypatch.setattr(service, "WorkerLoop", WorkerLoop)
    accepted = []

    def serve(app, **kwargs):
        body = json.dumps({
            "action": "opened", "number": 7, "installation": {"id": 44},
            "repository": {"id": 99, "name": "repo", "full_name": "org/repo"},
            "pull_request": github.pull_requests[7],
        }).encode()
        with TestClient(app) as client:
            response = client.post("/webhook", content=body, headers={
                "X-GitHub-Event": "pull_request", "X-GitHub-Delivery": f"cli-flow-{len(accepted)}",
                "X-Hub-Signature-256": "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest(),
            })
            assert response.status_code == 202
            assert response.json() == {"status": "accepted"}
        accepted.append(True)

    monkeypatch.setattr(uvicorn, "run", serve)
    assert main(["webhook", "--config", str(service_config.path)]) == 0
    dependencies.provider_factory.assert_not_called()
    assert main(["worker", "--config", str(service_config.path), "--once"]) == 0
    connection = sqlite3.connect(service_config.state / "jobs.sqlite")
    try:
        run_id, status = connection.execute("SELECT run_id, status FROM runs").fetchone()
        assert status == "succeeded"
        publication = connection.execute("SELECT comment_id, status FROM publication_receipts").fetchone()
    finally:
        connection.close()
    artifacts = service_config.state / "artifacts" / "runs" / "99" / "7" / run_id
    result = json.loads((artifacts / "A" / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "not_run"
    assert (artifacts / "A" / "result.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))["status"] == "succeeded"
    assert worker_pipeline_ports["analyzer"].run_analysis.call_count == 2
    assert [call.args[0] for call in worker_pipeline_ports["analyzer"].run_analysis.call_args_list] == ["a" * 40, "b" * 40]
    for call in worker_pipeline_ports["analyzer"].run_analysis.call_args_list:
        assert call.args[2:] == ("pmd/ruleset.xml", "java21-jdk-only-v1")
    if operation == "artifact_only":
        assert github.comments == {}
        assert publication == (None, "not_attempted")
    else:
        assert len(github.comments) == 1
        assert len(github.posted_comments) == 1
        assert len(github.updated_comments) == (1 if operation == "update" else 0)
        assert publication == (1, "succeeded")
        assert marker in github.comments[1]["body"]
        assert main(["webhook", "--config", str(service_config.path)]) == 0
        assert main(["worker", "--config", str(service_config.path), "--once"]) == 0
        assert len(github.comments) == 1
        assert len(github.posted_comments) == 1
        assert len(github.updated_comments) == (1 if operation == "update" else 0)
    for job_store in dependencies.opened:
        job_store.close.assert_called_once_with()