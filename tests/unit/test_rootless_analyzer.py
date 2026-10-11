"""Rootless analyzer contract tests; no Docker daemon or analyzer process is started."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest

from review_bot.adapters.pmd import container
from review_bot.adapters.pmd.container import RootlessPmdAnalyzer
from review_bot.adapters.pmd.runner import (
    AnalysisSetupError,
    AnalysisTimeoutError,
    scan_java_sources,
)
from review_bot.domain.models_v1 import (
    AnalysisError,
    AnalysisRun,
    CpdDuplication,
    CpdOccurrence,
    ExecutionRecord,
    ParsedReport,
    PmdViolation,
    ToolAnalysis,
)
from review_bot.ports import StaticAnalyzer

SHA = "a" * 40
IMAGE = "review-bot/analyzer@sha256:" + "b" * 64
INNER_ID = "c" * 32
SECRET = "private-key-and-model-token-never-log-this"


def _hash_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _mount(argv: list[str], target: str) -> Path:
    for index, arg in enumerate(argv):
        if arg == "--mount":
            options = dict(
                item.split("=", 1) for item in argv[index + 1].split(",") if "=" in item
            )
            if options.get("target") == target:
                return Path(options["source"])
    raise AssertionError(f"Missing mount: {target}")


def _write_result(source: Path, output: Path, sha: str) -> dict:
    files, source_hash, _ = scan_java_sources(source)
    conditions = {
        "schema_version": 1,
        "pmd_version": "7.3.0",
        "cpd_version": "7.3.0",
        "ruleset_hash": "d" * 64,
        "minimum_tokens": 100,
        "source_roots": ["src/main/java"],
        "language": "java",
        "language_version": "21",
        "encoding": "UTF-8",
        "type_resolution": "jdk_only",
        "java_version": "21.0.6",
        "java_vendor": "Amazon.com Inc.",
        "java_release_hash": "d" * 64,
        "java_executable_hash": "d" * 64,
        "tool_classpath": [["pmd.jar", "d" * 64]],
        "classpath_hash": "d" * 64,
        "runtime_classpath_hashes": {"pmd": "d" * 64, "cpd": "d" * 64},
    }
    profile_hash = _hash_json({k: v for k, v in conditions.items() if k != "runtime_classpath_hashes"})
    run_dir = output / sha / INNER_ID
    run_dir.mkdir(parents=True)
    container_dir = f"/output/{sha}/{INNER_ID}"
    if files:
        shutil.copytree(source, run_dir / "sources")
    tools = {}
    for tool in ("pmd", "cpd"):
        report_bytes = b"<pmd/>" if tool == "pmd" else b"<pmd-cpd/>"
        if files:
            (run_dir / f"{tool}.xml").write_bytes(report_bytes)
        execution = ExecutionRecord(
            schema_version=1,
            tool=tool,
            exit_code=0 if files else None,
            tool_version="7.3.0",
            java_version="21.0.6",
            java_vendor="Amazon.com Inc.",
            classpath_hash=conditions["classpath_hash"],
            profile_hash=profile_hash,
            sha=sha,
            run_id=INNER_ID,
            source_hash=source_hash,
            source_root=f"{container_dir}/sources" if files else "/source",
            source_roots=("src/main/java",),
            source_count=len(files),
            ruleset_hash=conditions["ruleset_hash"] if tool == "pmd" else None,
            minimum_tokens=100 if tool == "cpd" else None,
            language_version="21",
            encoding="UTF-8",
            type_resolution="jdk_only",
            started_at="2026-10-11T00:00:00+00:00",
            completed_at="2026-10-11T00:00:01+00:00",
            report_exists=bool(files),
            report_hash=hashlib.sha256(report_bytes).hexdigest() if files else None,
            command=("/trusted/java", "net.sourceforge.pmd.cli.PmdCli") if files else (),
            state="completed" if files else "not_run",
        )
        report = ParsedReport(
            tool=tool,
            path=f"{container_dir}/{tool}.xml" if files else "",
            exists=bool(files),
            parse_status="valid" if files else "missing",
        )
        tools[tool] = ToolAnalysis(
            tool=tool,
            status="clean" if files else "no_sources",
            exit_code=execution.exit_code,
            report=report,
            execution=execution,
        )
    run = AnalysisRun(
        schema_version=1,
        run_id=INNER_ID,
        sha=sha,
        profile_hash=profile_hash,
        conditions_hash=_hash_json(conditions),
        conditions=conditions,
        output_dir=container_dir,
        started_at="2026-10-11T00:00:00+00:00",
        completed_at="2026-10-11T00:00:01+00:00",
        status="clean" if files else "no_sources",
        source_state="complete" if files else "no_java_sources",
        source_files=files,
        source_hash=source_hash,
        exclusions=(),
        pmd=tools["pmd"],
        cpd=tools["cpd"],
        build_exit_code=None,
        errors=(),
        limitations=(),
    )
    return asdict(run)


class FakeDocker:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict]] = []
        self.info = {
            "OSType": "linux",
            "OperatingSystem": "Ubuntu 24.04 LTS",
            "Name": "rootless-worker",
            "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"],
            "CgroupVersion": "2",
            "CgroupDriver": "systemd",
            "MemoryLimit": True,
            "SwapLimit": True,
            "CpuCfsPeriod": True,
            "CpuCfsQuota": True,
            "PidsLimit": True,
        }
        self.info_output: str | None = None
        self.info_error: Exception | None = None
        self.info_code = 0
        self.run_error: Exception | None = None
        self.run_code = 0
        self.cleanup_error: Exception | None = None
        self.cleanup_code = 0
        self.cleanup_absent = False
        self.mutate = lambda data, output: None

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, kwargs))
        assert argv[0] == "docker", "Only the Docker CLI may run on the host"
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["cwd"] == "/"
        assert kwargs["timeout"] > 0
        assert not kwargs.get("shell", False)
        if argv[1] == "info":
            if self.info_error:
                raise self.info_error
            text = json.dumps(self.info) if self.info_output is None else self.info_output
            return subprocess.CompletedProcess(argv, self.info_code, text, SECRET)
        if argv[1] == "rm":
            if self.cleanup_error:
                raise self.cleanup_error
            name = argv[-1]
            stderr = f"Error response from daemon: No such container: {name}\n" if self.cleanup_absent else SECRET
            return subprocess.CompletedProcess(argv, self.cleanup_code, "", stderr)
        assert argv[1] == "run", "No build, pull, provision, shell, or host analyzer is allowed"
        if self.run_error:
            raise self.run_error
        if self.run_code:
            return subprocess.CompletedProcess(argv, self.run_code, SECRET, SECRET)
        output = _mount(argv, "/output")
        data = _write_result(_mount(argv, "/source"), output, argv[-2])
        self.mutate(data, output)
        if data is not None:
            (output / "analysis-run.json").write_text(json.dumps(data), encoding="utf-8")
            if isinstance(data, dict) and "run_id" in data:
                run_dir = output / SHA / INNER_ID
                (run_dir / "analysis-run.json").write_text(json.dumps(data), encoding="utf-8")
                for tool in ("pmd", "cpd"):
                    if isinstance(data.get(tool), dict) and "execution" in data[tool]:
                        (run_dir / f"{tool}-execution.json").write_text(
                            json.dumps(data[tool]["execution"]), encoding="utf-8"
                        )
        return subprocess.CompletedProcess(argv, 0, "", "")


@pytest.fixture(autouse=True)
def forbid_real_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Real subprocess execution is forbidden in this test module")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "XDG_RUNTIME_DIR"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(container.subprocess, "run", fake)
    return fake


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "export"
    java = root / "src/main/java/Example.java"
    java.parent.mkdir(parents=True)
    java.write_text("class Example {}\n", encoding="utf-8")
    return root


def _analyzer(tmp_path, **kwargs):
    return RootlessPmdAnalyzer(IMAGE, tmp_path / "outputs", **kwargs)


def test_command_contract_and_host_paths(docker, source, tmp_path):
    analyzer = _analyzer(tmp_path)
    assert isinstance(analyzer, StaticAnalyzer)
    run = analyzer.run_analysis(SHA, source, "pmd/ruleset.xml")
    assert isinstance(run, AnalysisRun)
    assert run.sha == SHA
    assert run.status == "clean"
    assert run.pmd.execution is not None
    assert run.cpd.execution is not None
    assert Path(run.output_dir).is_relative_to(tmp_path / "outputs")
    assert Path(run.pmd.report.path).is_file()
    assert Path(run.cpd.report.path).is_file()
    assert Path(run.pmd.execution.source_root).is_dir()
    assert (Path(run.output_dir) / "sources/src/main/java/Example.java").is_file()
    argv, kwargs = next(call for call in docker.calls if call[0][1] == "run")
    required = {
        "--rm", "--pull=never", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--memory=2048m", "--memory-swap=2048m",
        "--pids-limit=128", "--cpus=2", "--user=0:0", "--workdir=/tmp",
        "--entrypoint=/usr/local/bin/python3",
    }
    assert required.issubset(argv)
    assert "--detach" not in argv and "-d" not in argv
    tmpfs = next(arg for arg in argv if arg.startswith("--tmpfs="))
    assert all(option in tmpfs for option in ("noexec", "nosuid", "nodev", "size=", "mode=1777"))
    mounts = [argv[index + 1] for index, arg in enumerate(argv) if arg == "--mount"]
    assert len(mounts) == 2
    assert "readonly" in mounts[0]
    assert all("bind-recursive=disabled" in mount for mount in mounts)
    assert str(source) not in " ".join(mounts)
    assert str(tmp_path / "outputs") != str(_mount(argv, "/output"))
    assert _mount(argv, "/output").stat().st_mode & 0o777 == 0o700
    assert kwargs["timeout"] == 120
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL
    image_index = argv.index(IMAGE)
    assert argv[image_index + 1:image_index + 3] == ["-I", "-c"]
    ast.parse(argv[image_index + 3])
    assert argv[-2:] == [SHA, "120"]
    name = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--name="))
    assert name.startswith("review-bot-pmd-")
    assert docker.calls[-1][0] == ["docker", "rm", "--force", name]


def test_only_java_is_exported_not_scripts_config_or_credentials(docker, source, tmp_path):
    for name in ("gradlew", "build.gradle", "run.py", ".env", "private.pem", "pmd/ruleset.xml", ".git-hooks/pre-commit"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(SECRET, encoding="utf-8")

    def inspect_input(data, output):
        argv = next(argv for argv, _ in docker.calls if argv[1] == "run")
        staged = _mount(argv, "/source")
        assert [p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()] == [
            "src/main/java/Example.java"
        ]
        assert not (staged / "pmd/ruleset.xml").exists()

    docker.mutate = inspect_input
    assert _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml").status == "clean"


def test_minimal_cli_environment_and_no_container_secrets(docker, source, tmp_path, monkeypatch):
    original_home = os.environ.get("HOME")
    keys = (
        "GITHUB_TOKEN", "GITHUB_APP_PRIVATE_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
        "MODEL_API_KEY", "DOCKER_AUTH_CONFIG", "DOCKER_CONFIG", "JAVA_TOOL_OPTIONS", "PYTHONPATH",
        "HTTPS_PROXY", "AWS_SECRET_ACCESS_KEY", "UNEXPECTED_NEW_SECRET",
    )
    for key in keys:
        monkeypatch.setenv(key, SECRET)
    allowed = {"PATH": os.environ["PATH"], "DOCKER_HOST": "unix:///run/user/1000/docker.sock", "DOCKER_CONTEXT": "rootless", "XDG_RUNTIME_DIR": "/run/user/1000"}
    for key, value in allowed.items():
        monkeypatch.setenv(key, value)
    _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert os.environ.get("HOME") == original_home
    for argv, kwargs in docker.calls:
        assert kwargs["env"] == allowed
        assert SECRET not in " ".join(argv)
        assert not any(arg in ("-e", "--env", "--env-file") or arg.startswith("--env=") for arg in argv)


@pytest.mark.parametrize("image", ["sha256:" + "b" * 64, IMAGE, "localhost:5000/tools/pmd@sha256:" + "b" * 64])
def test_exact_digest_pins_are_accepted(docker, tmp_path, image):
    RootlessPmdAnalyzer(image, tmp_path / "outputs")


@pytest.mark.parametrize("image", [
    "review-bot/analyzer:latest", "review-bot/analyzer", "sha256:" + "b" * 63,
    "sha256:" + "B" * 64, IMAGE + "\n", "--privileged", IMAGE + ",evil",
    "review-bot/analyzer:latest@sha256:" + "b" * 64,
])
def test_mutable_or_malformed_image_is_rejected_before_docker(docker, tmp_path, image):
    with pytest.raises(AnalysisSetupError):
        RootlessPmdAnalyzer(image, tmp_path / "outputs")
    assert not docker.calls


@pytest.mark.parametrize("change", [
    {"SecurityOptions": []}, {"SecurityOptions": ["not-name=rootless"]},
    {"SecurityOptions": "name=rootless"}, {"OSType": "windows"},
    {"OperatingSystem": "Docker Desktop"}, {"Name": "docker-desktop"},
    {"CgroupVersion": "1"}, {"CgroupDriver": "none"}, {"CgroupDriver": "cgroupfs"},
    {"MemoryLimit": False}, {"SwapLimit": False}, {"CpuCfsPeriod": False},
    {"CpuCfsQuota": False}, {"PidsLimit": False}, {"MemoryLimit": "true"},
])
def test_preflight_fails_closed_for_rootful_desktop_or_missing_limits(docker, tmp_path, change):
    docker.info.update(change)
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path)
    assert all(argv[1] == "info" for argv, _ in docker.calls)


@pytest.mark.parametrize("text", ["", "not-json " + SECRET, "[]", "null", "{}"])
def test_malformed_preflight_is_redacted(docker, tmp_path, text):
    docker.info_output = text
    with pytest.raises(AnalysisSetupError) as error:
        _analyzer(tmp_path)
    assert SECRET not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("error", [FileNotFoundError(SECRET), subprocess.TimeoutExpired([SECRET], 10, output=SECRET, stderr=SECRET)])
def test_preflight_subprocess_failure_has_no_fallback(docker, tmp_path, error):
    docker.info_error = error
    with pytest.raises(AnalysisSetupError) as caught:
        _analyzer(tmp_path)
    assert SECRET not in str(caught.value)
    assert len(docker.calls) == 1


def test_preflight_is_rechecked_before_every_run(docker, source, tmp_path):
    analyzer = _analyzer(tmp_path)
    docker.info["SecurityOptions"] = []
    with pytest.raises(AnalysisSetupError):
        analyzer.run_analysis(SHA, source, "pmd/ruleset.xml")
    assert all(argv[1] == "info" for argv, _ in docker.calls)


@pytest.mark.parametrize("settings", [
    {"memory_mib": 0}, {"memory_mib": True}, {"memory_mib": "2048"},
    {"pids": 0}, {"pids": -1}, {"pids": True}, {"cpus": 0}, {"cpus": -1},
    {"cpus": float("nan")}, {"cpus": float("inf")}, {"cpus": True},
    {"timeout_seconds": 0}, {"timeout_seconds": True}, {"timeout_seconds": 1.5},
])
def test_invalid_resource_limits_fail_before_docker(docker, tmp_path, settings):
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path, **settings)
    assert not docker.calls


def test_configured_limits_reach_docker(docker, source, tmp_path):
    _analyzer(tmp_path, memory_mib=512, pids=32, cpus=0.5, timeout_seconds=7).run_analysis(
        SHA, source, "pmd/ruleset.xml"
    )
    argv, kwargs = next(call for call in docker.calls if call[0][1] == "run")
    assert {"--memory=512m", "--memory-swap=512m", "--pids-limit=32", "--cpus=0.5"}.issubset(argv)
    assert kwargs["timeout"] == 7
    assert argv[-1] == "7"


@pytest.mark.parametrize("sha", ["HEAD", "--help", "../escape", "a" * 39, SHA + "\n", SHA + ",rw", "A" * 40])
def test_unsafe_sha_never_reaches_docker_run(docker, source, tmp_path, sha):
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(sha, source, "pmd/ruleset.xml")
    assert not any(argv[1] == "run" for argv, _ in docker.calls)


@pytest.mark.parametrize("ruleset,profile", [
    ("../pmd/ruleset.xml", "java21-jdk-only-v1"), ("/source/pmd/ruleset.xml", "java21-jdk-only-v1"),
    ("https://target/rules.xml", "java21-jdk-only-v1"), ("pmd/ruleset.xml", "custom"),
    ("pmd/ruleset.xml\n", "java21-jdk-only-v1"),
])
def test_target_ruleset_or_profile_cannot_select_executable_input(docker, source, tmp_path, ruleset, profile):
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, ruleset, profile)
    assert not any(argv[1] == "run" for argv, _ in docker.calls)


@pytest.mark.parametrize("name", ["bad,readonly", "bad\npath", 'bad"path'])
def test_mount_control_characters_are_rejected(docker, tmp_path, name):
    root = tmp_path / name
    root.mkdir()
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, root, "pmd/ruleset.xml")
    assert not any(argv[1] == "run" for argv, _ in docker.calls)


def test_repository_home_root_and_output_root_are_not_source_mounts(docker, source, tmp_path):
    (source / ".git").mkdir()
    analyzer = _analyzer(tmp_path)
    for path in (source, Path("/"), Path.home(), tmp_path / "outputs"):
        with pytest.raises(AnalysisSetupError):
            analyzer.run_analysis(SHA, path, "pmd/ruleset.xml")
    assert not any(argv[1] == "run" for argv, _ in docker.calls)


@pytest.mark.parametrize("kind", ["file", "directory", "root", "hardlink", "fifo"])
def test_source_symlinks_hardlinks_and_special_files_are_rejected(docker, source, tmp_path, kind):
    outside = tmp_path / "outside.java"
    outside.write_text(SECRET, encoding="utf-8")
    java = source / "src/main/java/Example.java"
    if kind == "file":
        java.unlink()
        java.symlink_to(outside)
    elif kind == "directory":
        (java.parent / "escape").symlink_to(tmp_path, target_is_directory=True)
    elif kind == "root":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        source = alias
    elif kind == "hardlink":
        java.unlink()
        os.link(outside, java)
    else:
        java.unlink()
        os.mkfifo(java)
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert not any(argv[1] == "run" for argv, _ in docker.calls)


def test_output_root_symlink_or_insecure_permissions_fail_closed(docker, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(AnalysisSetupError):
        RootlessPmdAnalyzer(IMAGE, alias)
    outside.chmod(0o777)
    with pytest.raises(AnalysisSetupError):
        RootlessPmdAnalyzer(IMAGE, outside)


def test_no_sources_is_explicit_not_empty_clean(docker, tmp_path):
    source = tmp_path / "empty-export"
    source.mkdir()
    run = _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert run.status == "no_sources"
    assert run.source_state == "no_java_sources"
    assert run.pmd.status == run.cpd.status == "no_sources"
    assert not run.pmd.report.exists
    assert run.pmd.execution.state == "not_run"


def test_findings_and_duplications_survive_deserialization(docker, source, tmp_path):
    def findings(data, output):
        path = "src/main/java/Example.java"
        raw_path = f"/output/{SHA}/{INNER_ID}/sources/{path}"
        data["pmd"]["report"]["violations"] = [asdict(PmdViolation(
            path=path, raw_path=raw_path, begin_line=1, end_line=1,
            begin_column=1, end_column=5, rule="ExampleRule", ruleset="Code Style",
            priority=3, klass="Example", method="", package="", message="Example finding",
        ))]
        occurrence = CpdOccurrence(
            path=path, raw_path=raw_path, begin_line=1, end_line=1,
            begin_column=None, end_column=None, end_line_inferred=True,
        )
        data["cpd"]["report"]["duplications"] = [asdict(CpdDuplication(
            lines=1, tokens=100, occurrences=(occurrence, occurrence), code_fragment="class Example {}",
        ))]
        for tool in ("pmd", "cpd"):
            data[tool]["status"] = "findings"
            data[tool]["exit_code"] = data[tool]["execution"]["exit_code"] = 4
        data["status"] = "findings"

    docker.mutate = findings
    run = _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert run.status == "findings"
    assert run.pmd.report.violations[0].rule == "ExampleRule"
    assert run.cpd.report.duplications[0].tokens == 100
    assert run.cpd.report.duplications[0].occurrences[0].begin_column is None
    assert run.cpd.report.duplications[0].occurrences[0].end_line_inferred
    assert run.pmd.execution.exit_code == run.cpd.execution.exit_code == 4


def test_valid_analysis_failure_and_errors_are_not_lost(docker, source, tmp_path):
    def failure(data, output):
        error = asdict(AnalysisError("execution_timeout", "PMD exceeded its tool timeout"))
        data["pmd"]["status"] = "failed"
        data["pmd"]["exit_code"] = data["pmd"]["execution"]["exit_code"] = None
        data["pmd"]["execution"]["state"] = "timed_out"
        data["pmd"]["errors"] = data["errors"] = [error]
        data["status"] = "failed"
        data["limitations"] = ["Partial PMD analysis"]

    docker.mutate = failure
    run = _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert run.status == run.pmd.status == "failed"
    assert run.errors[0] == run.pmd.errors[0] == AnalysisError("execution_timeout", "PMD exceeded its tool timeout")
    assert run.pmd.execution.state == "timed_out"
    assert run.limitations == ("Partial PMD analysis",)


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True), ("sha", "e" * 40),
    ("run_id", "../escape"), ("output_dir", "/etc"), ("output_dir", "/output/../escape"),
    ("source_files", []), ("source_hash", None), ("conditions_hash", "e" * 64),
    ("profile_hash", "e" * 64), ("build_exit_code", 0), ("status", "no_sources"),
])
def test_wrong_schema_sha_provenance_or_escape_is_rejected(docker, source, tmp_path, field, value):
    docker.mutate = lambda data, output: data.__setitem__(field, value)
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")


@pytest.mark.parametrize("field", ["pmd", "cpd", "source_files", "conditions", "status", "schema_version", "errors"])
def test_missing_manifest_fields_cannot_be_defaulted_to_clean(docker, source, tmp_path, field):
    docker.mutate = lambda data, output: data.pop(field)
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")


@pytest.mark.parametrize("change", [
    lambda data: data["pmd"].pop("report"),
    lambda data: data["pmd"].pop("execution"),
    lambda data: data["cpd"]["report"].pop("exists"),
    lambda data: data["pmd"]["report"].pop("violations"),
    lambda data: data["pmd"]["report"].update(path="/etc/passwd"),
    lambda data: data["pmd"]["report"].update(path="/output/../escape.xml"),
    lambda data: data["pmd"]["report"].update(exists=False),
    lambda data: data["pmd"]["report"].update(parse_status="missing"),
    lambda data: data["pmd"]["execution"].update(sha="e" * 40),
    lambda data: data["pmd"]["execution"].update(report_hash="e" * 64),
    lambda data: data["pmd"]["execution"].update(source_root="/etc"),
    lambda data: data["pmd"].update(status="findings"),
    lambda data: data["pmd"].update(exit_code=4),
])
def test_incomplete_tool_evidence_cannot_become_clean(docker, source, tmp_path, change):
    docker.mutate = lambda data, output: change(data)
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")


@pytest.mark.parametrize("filename", ["pmd.xml", "cpd.xml", "sources/src/main/java/Example.java"])
def test_missing_output_artifact_is_not_normal_success(docker, source, tmp_path, filename):
    docker.mutate = lambda data, output: (output / SHA / INNER_ID / filename).unlink()
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")


@pytest.mark.parametrize("kind", ["report", "source", "extra", "parent", "hardlink", "fifo"])
def test_any_output_symlink_escape_or_special_file_is_rejected(docker, source, tmp_path, kind):
    outside = tmp_path / "private-host-file"
    outside.write_text(SECRET, encoding="utf-8")

    def mutate(data, output):
        run_dir = output / SHA / INNER_ID
        if kind == "parent":
            nested = run_dir / "escape"
            nested.symlink_to(tmp_path, target_is_directory=True)
            return
        name = "pmd.xml" if kind in ("report", "hardlink", "fifo") else "extra"
        if kind == "source":
            name = "sources/src/main/java/Example.java"
        target = run_dir / name
        target.unlink(missing_ok=True)
        if kind == "hardlink":
            os.link(outside, target)
        elif kind == "fifo":
            os.mkfifo(target)
        else:
            target.symlink_to(outside)

    docker.mutate = mutate
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert outside.read_text() == SECRET


def test_malformed_json_and_manifest_symlink_are_rejected(docker, source, tmp_path):
    original = docker.__class__.__call__

    def corrupt(argv, **kwargs):
        result = original(docker, argv, **kwargs)
        if argv[1] == "run":
            (_mount(argv, "/output") / "analysis-run.json").write_text("not-json " + SECRET)
        return result

    container.subprocess.run = corrupt
    with pytest.raises(AnalysisSetupError) as caught:
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert SECRET not in str(caught.value)

    def link_manifest(argv, **kwargs):
        result = original(docker, argv, **kwargs)
        if argv[1] == "run":
            manifest = _mount(argv, "/output") / "analysis-run.json"
            manifest.unlink()
            manifest.symlink_to(tmp_path / "host-secret")
        return result

    container.subprocess.run = link_manifest
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")


def test_manifest_cannot_claim_success_without_any_output(docker, source, tmp_path, monkeypatch):
    original = docker.__class__.__call__

    def no_output(argv, **kwargs):
        result = original(docker, argv, **kwargs)
        if argv[1] == "run":
            (_mount(argv, "/output") / "analysis-run.json").unlink()
        return result

    monkeypatch.setattr(container.subprocess, "run", no_output)
    with pytest.raises(AnalysisSetupError):
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")


@pytest.mark.parametrize("error", [subprocess.TimeoutExpired([SECRET], 120, output=SECRET, stderr=SECRET), OSError(SECRET)])
def test_run_timeout_or_cli_failure_removes_only_its_named_container(docker, source, tmp_path, error):
    docker.run_error = error
    expected = AnalysisTimeoutError if isinstance(error, subprocess.TimeoutExpired) else AnalysisSetupError
    with pytest.raises(expected) as caught:
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert SECRET not in str(caught.value)
    run_argv = next(argv for argv, _ in docker.calls if argv[1] == "run")
    name = next(arg.split("=", 1)[1] for arg in run_argv if arg.startswith("--name="))
    assert docker.calls[-1][0] == ["docker", "rm", "--force", name]
    assert len([argv for argv, _ in docker.calls if argv[1] == "run"]) == 1


def test_missing_local_image_has_no_pull_or_host_fallback(docker, source, tmp_path):
    docker.run_code = 125
    with pytest.raises(AnalysisSetupError) as caught:
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert SECRET not in str(caught.value)
    assert {argv[1] for argv, _ in docker.calls} == {"info", "run", "rm"}


def test_already_auto_removed_container_is_safe(docker, source, tmp_path):
    docker.cleanup_code = 1
    docker.cleanup_absent = True
    assert _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml").status == "clean"


@pytest.mark.parametrize("error", [None, OSError(SECRET), subprocess.TimeoutExpired([SECRET], 10, stderr=SECRET)])
def test_cleanup_failure_is_not_reported_as_success(docker, source, tmp_path, error):
    docker.cleanup_error = error
    docker.cleanup_code = 1
    with pytest.raises(AnalysisSetupError, match="cleanup") as caught:
        _analyzer(tmp_path).run_analysis(SHA, source, "pmd/ruleset.xml")
    assert SECRET not in str(caught.value)


def test_each_run_has_a_distinct_private_output_and_container(docker, source, tmp_path):
    analyzer = _analyzer(tmp_path)
    analyzer.run_analysis(SHA, source, "pmd/ruleset.xml")
    analyzer.run_analysis(SHA, source, "pmd/ruleset.xml")
    runs = [argv for argv, _ in docker.calls if argv[1] == "run"]
    assert _mount(runs[0], "/output") != _mount(runs[1], "/output")
    assert next(arg for arg in runs[0] if arg.startswith("--name=")) != next(
        arg for arg in runs[1] if arg.startswith("--name=")
    )