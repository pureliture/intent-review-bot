"""Fail-closed PMD/CPD execution in a prebuilt, digest-pinned rootless Linux image.

The worker and the local rootless daemon must belong to the same host UID. Container
UID 0 maps to that daemon's unprivileged UID, not host root; using the host UID as
the container UID would instead select a subordinate UID and break private bind
mount ownership. Outputs stay worker-owned and private; permissions are never
relaxed to compensate for a different daemon user. This mapping needs an actual
rootless Ubuntu deployment check; unit tests do not establish it.

The existing repository-oriented CLI analyze command requires .git. Its trusted
PmdRunner/AnalysisRun operation is invoked by a fixed isolated Python entrypoint
inside the container instead: neither .git nor target scripts are exposed. The
image must already contain /usr/local/bin/python3, review-bot and verified PMD/JDK
artifacts. This adapter never builds, provisions, pulls, or runs a host analyzer.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from review_bot.adapters.pmd.runner import (
    AnalysisSetupError,
    AnalysisTimeoutError,
    check_exit_finding_concordance,
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

_DIGEST = r"sha256:[0-9a-f]{64}"
_REPO_COMPONENT = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_REPOSITORY = rf"(?:[a-z0-9]+(?:[.-][a-z0-9]+)*(?::[0-9]+)?/)?{_REPO_COMPONENT}(?:/{_REPO_COMPONENT})*"
_IMAGE_PIN = re.compile(rf"(?:{_DIGEST}|{_REPOSITORY}@{_DIGEST})")
_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_RUN_ID = re.compile(r"[0-9a-f]{32}")
_RULESET = "pmd/ruleset.xml"
_PROFILE = "java21-jdk-only-v1"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_EXCLUDED_DIRS = {"build", "target", "out", "dist", "__pycache__"}
_CONTAINER_ANALYZE = """import json, sys
from dataclasses import asdict
from pathlib import Path
from review_bot.adapters.pmd.runner import PmdRunner
run = PmdRunner(output_root=Path('/output'), timeout_seconds=int(sys.argv[2])).run_analysis(
    sys.argv[1], Path('/source'), 'pmd/ruleset.xml', 'java21-jdk-only-v1')
Path('/output/analysis-run.json').write_text(json.dumps(asdict(run)), encoding='utf-8')
"""


def _require(condition: bool, message: str = "Invalid rootless analyzer output") -> None:
    if not condition:
        raise AnalysisSetupError(message)


def _checked_path(path: Path) -> Path:
    text = os.fspath(path)
    _require(
        not any(ord(char) < 32 or ord(char) == 127 or char in ',"' for char in text)
        and ".." not in path.parts,
        "Unsafe analyzer filesystem path",
    )
    absolute = path.absolute()
    _require(
        absolute != Path("/") and not Path.home().absolute().is_relative_to(absolute),
        "Analyzer paths must not expose root or HOME",
    )
    _require(
        not any(part.is_symlink() for part in (absolute, *absolute.parents)),
        "Analyzer filesystem symlinks are forbidden",
    )
    return absolute


def _private_directory(path: Path, *, exclusive: bool = False) -> Path:
    path = _checked_path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=not exclusive)
    metadata = path.lstat()
    _require(
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_mode & 0o022 == 0,
        "Analyzer output directory must be private and worker-owned",
    )
    return path


@contextmanager
def _regular_file(root: Path, path: Path, *, worker_owned: bool = False):
    relative = path.relative_to(root)
    _require(bool(relative.parts) and ".." not in relative.parts)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        _require(stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1)
        _require(not worker_owned or metadata.st_uid == os.geteuid())
        yield stream


def _raise_walk_error(error: OSError) -> None:
    raise error


def _stage_sources(source: Path, staged: Path) -> tuple[tuple[str, ...], str | None]:
    candidate = source / "src/main/java"
    active = _checked_path(candidate) if candidate.is_dir() else source
    if active != source:
        (staged / "src/main/java").mkdir(parents=True, mode=0o700)
    for parent, directories, names in os.walk(active, followlinks=False, onerror=_raise_walk_error):
        directories[:] = [
            name for name in directories if not name.startswith(".") and name not in _EXCLUDED_DIRS
        ]
        for name in directories:
            _require(not (Path(parent) / name).is_symlink(), "Source directory symlinks are forbidden")
        for name in names:
            if not name.endswith(".java"):
                continue
            original = Path(parent) / name
            destination = staged / original.relative_to(source)
            destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            with _regular_file(source, original) as stream, destination.open("xb") as target:
                shutil.copyfileobj(stream, target)
            destination.chmod(0o444)
    source_files, source_hash, _ = scan_java_sources(staged)
    return source_files, source_hash


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON field")
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


def _json(text: str | bytes) -> Any:
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)


def _read_json(root: Path, path: Path) -> Any:
    with _regular_file(root, path, worker_owned=True) as stream:
        _require(os.fstat(stream.fileno()).st_size <= _MAX_MANIFEST_BYTES)
        return _json(stream.read(_MAX_MANIFEST_BYTES + 1))


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_file(root: Path, path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(root, path, worker_owned=True) as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(data: Any, model: type) -> None:
    _require(isinstance(data, dict) and {field.name for field in fields(model)} <= data.keys())


def _errors(data: Any) -> tuple[AnalysisError, ...]:
    _require(isinstance(data, list))
    for error in data:
        _record(error, AnalysisError)
        _require(all(isinstance(error[key], str) for key in ("kind", "message", "path", "detail")))
    return tuple(AnalysisError(**error) for error in data)


def _check_output_tree(root: Path) -> None:
    _private_directory(root)
    for parent, directories, names in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        for name in directories + names:
            metadata = (Path(parent) / name).lstat()
            _require(metadata.st_uid == os.geteuid() and metadata.st_mode & 0o022 == 0)
            _require(
                stat.S_ISDIR(metadata.st_mode)
                or (stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1)
            )


def _check_findings(report: dict, source_files: tuple[str, ...]) -> None:
    for key in ("violations", "duplications"):
        _require(isinstance(report[key], list))
    for violation in report["violations"]:
        _record(violation, PmdViolation)
        _require(violation["path"] in source_files)
        _require(type(violation["priority"]) is int and 1 <= violation["priority"] <= 5)
        for key in ("begin_line", "end_line", "begin_column", "end_column"):
            _require(type(violation[key]) is int and violation[key] >= 1)
    for duplication in report["duplications"]:
        _record(duplication, CpdDuplication)
        _require(isinstance(duplication["occurrences"], list) and len(duplication["occurrences"]) >= 2)
        for key in ("lines", "tokens"):
            _require(type(duplication[key]) is int and duplication[key] > 0)
        for occurrence in duplication["occurrences"]:
            _record(occurrence, CpdOccurrence)
            _require(occurrence["path"] in source_files)
            for key in ("begin_line", "end_line"):
                _require(type(occurrence[key]) is int and occurrence[key] >= 1)
            for key in ("begin_column", "end_column"):
                _require(occurrence[key] is None or (type(occurrence[key]) is int and occurrence[key] >= 1))
            _require(type(occurrence["end_line_inferred"]) is bool)


class RootlessPmdAnalyzer(StaticAnalyzer):
    """StaticAnalyzer with no rootful, Desktop, pull, build, or host-execution fallback."""

    def __init__(
        self,
        image: str,
        output_root: Path,
        *,
        memory_mib: int = 2048,
        pids: int = 128,
        cpus: float = 2,
        timeout_seconds: int = 120,
    ) -> None:
        _require(isinstance(image, str) and _IMAGE_PIN.fullmatch(image) is not None, "A precise image digest pin is required")
        _require(type(memory_mib) is int and memory_mib >= 128, "Invalid memory limit")
        _require(type(pids) is int and pids >= 16, "Invalid PID limit")
        _require(type(cpus) in (int, float) and math.isfinite(cpus) and cpus > 0, "Invalid CPU limit")
        _require(type(timeout_seconds) is int and timeout_seconds > 0, "Invalid analysis timeout")
        self.image = image
        self.memory_mib = memory_mib
        self.pids = pids
        self.cpus = cpus
        self.timeout_seconds = timeout_seconds
        self._env = {"PATH": os.environ.get("PATH", os.defpath)}
        self._env.update({
            key: os.environ[key]
            for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "XDG_RUNTIME_DIR") if key in os.environ
        })
        try:
            self.output_root = _private_directory(output_root)
            self._preflight()
        except (OSError, ValueError):
            raise AnalysisSetupError("Rootless analyzer startup failed") from None

    def _docker(self, arguments: list[str], *, timeout: int, capture: bool = True):
        return subprocess.run(
            ["docker", *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=dict(self._env),
            cwd="/",
            timeout=timeout,
            check=False,
        )

    def _preflight(self) -> None:
        try:
            result = self._docker(["info", "--format", "{{json .}}"], timeout=10)
            _require(result.returncode == 0, "Rootless Docker preflight failed")
            info = _json(result.stdout)
            _require(isinstance(info, dict), "Invalid Docker preflight response")
            options = info.get("SecurityOptions")
            _require(isinstance(options, list) and "name=rootless" in options, "Rootless Docker is required")
            _require(
                info.get("OSType") == "linux"
                and isinstance(info.get("OperatingSystem"), str)
                and isinstance(info.get("Name"), str)
                and "desktop" not in (info["OperatingSystem"] + info["Name"]).lower(),
                "A rootless Linux daemon, not Docker Desktop, is required",
            )
            _require(
                info.get("CgroupVersion") == "2" and info.get("CgroupDriver") == "systemd"
                and all(info.get(key) is True for key in (
                    "MemoryLimit", "SwapLimit", "CpuCfsPeriod", "CpuCfsQuota", "PidsLimit"
                )),
                "Rootless Docker must support delegated cgroup v2 memory, swap, CPU and PID limits",
            )
        except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
            raise AnalysisSetupError("Rootless Docker preflight failed") from None

    def _cleanup(self, name: str) -> None:
        try:
            result = self._docker(["rm", "--force", name], timeout=10)
            absent = f"Error response from daemon: No such container: {name}"
            _require(
                result.returncode == 0 or (result.returncode == 1 and result.stderr.strip() == absent),
                "Rootless analyzer container cleanup could not be confirmed",
            )
        except (OSError, subprocess.TimeoutExpired):
            raise AnalysisSetupError("Rootless analyzer container cleanup could not be confirmed") from None

    def run_analysis(
        self,
        commit_sha: str,
        source_root: Path,
        ruleset_resource: str,
        profile: str = "java21-jdk-only-v1",
    ) -> AnalysisRun:
        _require(isinstance(commit_sha, str) and _SHA.fullmatch(commit_sha) is not None, "A full commit SHA is required")
        _require(ruleset_resource == _RULESET and profile == _PROFILE, "Only the trusted ruleset and analysis profile are permitted")
        try:
            source = _checked_path(source_root)
            _require(source.is_dir() and not (source / ".git").exists(), "A materialized source export, not a Git repository, is required")
            _require(
                not source.is_relative_to(self.output_root) and not self.output_root.is_relative_to(source),
                "Source and analyzer state must not overlap",
            )
            self._preflight()
            _private_directory(self.output_root)
            identifier = uuid.uuid4().hex
            name = f"review-bot-pmd-{identifier}"
            output = _private_directory(self.output_root / commit_sha / identifier, exclusive=True)
            with tempfile.TemporaryDirectory(prefix=".pmd-source-", dir=self.output_root) as temporary:
                staged = _checked_path(Path(temporary))
                source_files, source_hash = _stage_sources(source, staged)
                command = [
                    "run", "--rm", "--pull=never", f"--name={name}", "--network=none", "--read-only",
                    "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=0:0",
                    f"--memory={self.memory_mib}m", f"--memory-swap={self.memory_mib}m",
                    f"--pids-limit={self.pids}", f"--cpus={self.cpus:g}",
                    f"--tmpfs=/tmp:rw,noexec,nosuid,nodev,size={min(128, self.memory_mib // 2)}m,mode=1777",
                    "--workdir=/tmp", "--entrypoint=/usr/local/bin/python3",
                    "--mount", f"type=bind,source={staged},target=/source,readonly,bind-recursive=disabled",
                    "--mount", f"type=bind,source={output},target=/output,bind-recursive=disabled",
                    self.image, "-I", "-c", _CONTAINER_ANALYZE, commit_sha, str(self.timeout_seconds),
                ]
                try:
                    try:
                        result = self._docker(command, timeout=self.timeout_seconds, capture=False)
                        _require(result.returncode == 0, "Rootless analyzer container failed")
                    except subprocess.TimeoutExpired:
                        raise AnalysisTimeoutError() from None
                    except OSError:
                        raise AnalysisSetupError("Rootless analyzer container could not start") from None
                finally:
                    self._cleanup(name)
                return self._read_result(output, source, commit_sha, source_files, source_hash)
        except AnalysisTimeoutError:
            raise
        except (OSError, ValueError, TypeError, KeyError, AttributeError, RecursionError):
            raise AnalysisSetupError("Invalid rootless analyzer input or output") from None

    def _read_result(
        self, output: Path, source: Path, sha: str, source_files: tuple[str, ...], source_hash: str | None
    ) -> AnalysisRun:
        _check_output_tree(output)
        data = _read_json(output, output / "analysis-run.json")
        _record(data, AnalysisRun)
        _require(type(data["schema_version"]) is int and data["schema_version"] == 1 and data["sha"] == sha)
        _require(isinstance(data["run_id"], str) and _RUN_ID.fullmatch(data["run_id"]) is not None)
        container_dir = f"/output/{sha}/{data['run_id']}"
        run_dir = output / sha / data["run_id"]
        _require(data["output_dir"] == container_dir and run_dir.is_dir())
        _require(_read_json(output, run_dir / "analysis-run.json") == data)
        _require(data["source_files"] == list(source_files) and data["source_hash"] == source_hash)
        _require(data["build_exit_code"] is None and data["exclusions"] == [])
        _require(isinstance(data["limitations"], list) and all(isinstance(item, str) for item in data["limitations"]))
        _require(all(isinstance(data[key], str) and data[key] for key in ("started_at", "completed_at")))
        conditions = data["conditions"]
        _require(isinstance(conditions, dict))
        approved = {
            "schema_version": 1, "pmd_version": "7.3.0", "cpd_version": "7.3.0",
            "minimum_tokens": 100, "source_roots": ["src/main/java"], "language": "java",
            "language_version": "21", "encoding": "UTF-8", "type_resolution": "jdk_only",
        }
        _require(all(conditions.get(key) == value for key, value in approved.items()))
        _require(type(conditions["schema_version"]) is int and type(conditions["minimum_tokens"]) is int)
        _require(data["conditions_hash"] == _hash_json(conditions))
        _require(data["profile_hash"] == _hash_json({k: v for k, v in conditions.items() if k != "runtime_classpath_hashes"}))
        errors = _errors(data["errors"])
        if source_files:
            _require(data["source_state"] == "complete" and data["status"] != "no_sources")
            snapshot_files, snapshot_hash, _ = scan_java_sources(run_dir / "sources")
            _require(snapshot_files == source_files and snapshot_hash == source_hash)
        else:
            _require(data["source_state"] == "no_java_sources" and data["status"] == "no_sources" and not errors)

        executions = {}
        for tool in ("pmd", "cpd"):
            tool_data = data[tool]
            _record(tool_data, ToolAnalysis)
            _require(tool_data["tool"] == tool)
            report = tool_data["report"]
            _record(report, ParsedReport)
            _require(report["tool"] == tool and type(report["exists"]) is bool)
            _require(type(report["suppressed_count"]) is int and report["suppressed_count"] >= 0)
            _check_findings(report, source_files)
            _require(not report["duplications"] if tool == "pmd" else not report["violations"])
            execution = tool_data["execution"]
            _record(execution, ExecutionRecord)
            _require(_read_json(output, run_dir / f"{tool}-execution.json") == execution)
            _require(
                type(execution["schema_version"]) is int and execution["schema_version"] == 1
                and execution["tool"] == tool and execution["sha"] == sha
                and execution["run_id"] == data["run_id"] and execution["profile_hash"] == data["profile_hash"]
                and execution["source_hash"] == source_hash and execution["source_roots"] == ["src/main/java"]
                and type(execution["source_count"]) is int and execution["source_count"] == len(source_files)
                and type(execution["report_exists"]) is bool and execution["report_exists"] == report["exists"]
                and execution["exit_code"] == tool_data["exit_code"]
            )
            _require(
                (tool_data["exit_code"] is None or type(tool_data["exit_code"]) is int)
                and type(execution["exit_code"]) is type(tool_data["exit_code"])
            )
            _require(isinstance(execution["command"], list) and all(isinstance(arg, str) for arg in execution["command"]))
            report_path = run_dir / f"{tool}.xml"
            if not source_files:
                _require(
                    tool_data["status"] == "no_sources" and tool_data["exit_code"] is None
                    and report["path"] == "" and not report["exists"] and report["parse_status"] == "missing"
                    and execution["state"] == "not_run" and execution["source_root"] == "/source"
                    and execution["report_hash"] is None and execution["command"] == []
                )
                host_source = source
            else:
                _require(report["path"] == f"{container_dir}/{tool}.xml")
                _require(execution["source_root"] == f"{container_dir}/sources")
                _require(report_path.is_file() == report["exists"])
                _require(report["parse_status"] in ("valid", "invalid", "missing"))
                _require(execution["state"] in ("completed", "failed", "timed_out"))
                _require(execution["report_hash"] == (_hash_file(output, report_path) if report["exists"] else None))
                _require(tool_data["status"] in ("clean", "findings", "failed", "incomplete", "missing_report"))
                if tool_data["status"] in ("clean", "findings"):
                    finding_count = len(report["violations"] if tool == "pmd" else report["duplications"])
                    concordance, _ = check_exit_finding_concordance(tool_data["exit_code"], finding_count, tool)
                    _require(
                        report["exists"] and report["parse_status"] == "valid"
                        and execution["state"] == "completed" and concordance == tool_data["status"]
                        and not tool_data["errors"] and not report["errors"]
                    )
                report["path"] = str(report_path)
                host_source = run_dir / "sources"
            execution = dict(execution)
            execution.update(source_root=str(host_source), source_roots=tuple(execution["source_roots"]), command=tuple(execution["command"]))
            executions[tool] = ExecutionRecord(**execution)

        if source_files:
            statuses = {data["pmd"]["status"], data["cpd"]["status"]}
            expected_status = "clean"
            for status in ("findings", "incomplete", "missing_report", "failed"):
                if status in statuses:
                    expected_status = status
            _require(data["status"] == ("failed" if errors else expected_status))

        from review_bot.runtime.cli import _analysis_run_from_dict

        data["output_dir"] = str(run_dir)
        run = _analysis_run_from_dict(data)
        hydrated_tools = {}
        for tool in ("pmd", "cpd"):
            tool_run = getattr(run, tool)
            duplications = tuple(
                replace(duplication, occurrences=tuple(
                    replace(occurrence, end_line_inferred=raw["end_line_inferred"])
                    for occurrence, raw in zip(duplication.occurrences, original["occurrences"], strict=True)
                ))
                for duplication, original in zip(tool_run.report.duplications, data[tool]["report"]["duplications"], strict=True)
            )
            hydrated_tools[tool] = replace(
                tool_run,
                execution=executions[tool],
                errors=_errors(data[tool]["errors"]),
                report=replace(tool_run.report, errors=_errors(data[tool]["report"]["errors"]), duplications=duplications),
            )
        return replace(run, **hydrated_tools, errors=errors, limitations=tuple(data["limitations"]))