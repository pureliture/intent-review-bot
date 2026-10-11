"""PMD 7.3.0 and CPD execution runner with process group isolation and concordance validation."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from review_bot.adapters.pmd.parsers import parse_cpd_xml, parse_pmd_xml
from review_bot.domain.models_v1 import (
    AnalysisError,
    AnalysisRun,
    ExecutionRecord,
    ParsedReport,
    ToolAnalysis,
)
from review_bot.domain.status import (
    AnalysisStatusKind,
    ExecutionStateKind,
)
from review_bot.ports import ExecutionIsolation, StaticAnalyzer

# Default search paths for Amazon Corretto JDK 21 on macOS and Linux
KNOWN_JDK21_SEARCH_PATHS = (
    Path.home() / "Library/Java/JavaVirtualMachines/corretto-21.0.6/Contents/Home",
    Path("/Library/Java/JavaVirtualMachines/corretto-21.jdk/Contents/Home"),
    Path("/usr/lib/jvm/java-21-amazon-corretto"),
    Path("/usr/lib/jvm/java-21-openjdk-amd64"),
    Path("/usr/lib/jvm/java-21-openjdk"),
    Path("/opt/java/corretto-21"),
)


class AnalysisSetupError(RuntimeError):
    """Raised when JDK 21 or tool dependencies are missing, invalid, or corrupted."""


class AnalysisTimeoutError(TimeoutError):
    """Raised when analysis process exceeds configured timeout."""

    def __init__(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code
        super().__init__(f"Analysis process exceeded timeout (returncode={exit_code})")


@dataclass(frozen=True)
class JdkInfo:
    """Discovered and verified JDK 21 runtime metadata."""

    home: Path
    executable: Path
    version: str
    vendor: str
    release_hash: str | None
    executable_hash: str


def _now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def _file_hash(path: Path) -> str | None:
    """Calculate SHA-256 digest of a regular file."""
    if not path.is_file() or path.is_symlink():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    """Deterministic JSON SHA-256 hash."""
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sanitize_environment(java_home: Path | None = None) -> dict[str, str]:
    """Create a pristine subprocess execution environment stripping JVM injection variables and secrets."""
    hazardous_keys = {
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "_JAVA_OPTIONS",
        "JAVA_OPTS",
        "GRADLE_OPTS",
        "CLASSPATH",
        "GITHUB_TOKEN",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_WEBHOOK_SECRET",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODEL_API_KEY",
    }
    env = {k: v for k, v in os.environ.items() if k not in hazardous_keys}
    if java_home is not None:
        env["JAVA_HOME"] = str(java_home)
    env["LC_ALL"] = "C.UTF-8"
    env["LANG"] = "C.UTF-8"
    return env


def discover_jdk_21(configured_home: Path | str | None = None) -> JdkInfo:
    """Discover and verify Amazon Corretto JDK 21 or compatible JDK 21."""
    candidates: list[Path] = []
    if configured_home is not None:
        candidates.append(Path(configured_home).resolve())
    if os.environ.get("JAVA_HOME"):
        candidates.append(Path(os.environ["JAVA_HOME"]).resolve())
    candidates.extend(p.resolve() for p in KNOWN_JDK21_SEARCH_PATHS if p.exists())

    which_java = shutil.which("java")
    if which_java:
        which_path = Path(which_java).resolve()
        home_candidate = which_path.parent.parent
        if (home_candidate / "bin" / "java").is_file():
            candidates.append(home_candidate)

    if not candidates:
        raise AnalysisSetupError("JDK 21 is required but could not be located")

    clean_env = sanitize_environment()
    for candidate in candidates:
        bin_java = candidate / "bin" / "java"
        if not bin_java.is_file():
            continue
        try:
            probe = subprocess.run(
                [str(bin_java), "-XshowSettings:properties", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=clean_env,
                check=False,
            )
            properties = dict(
                re.findall(r"^\s*(java\.(?:home|version|vendor))\s*=\s*(.*?)\s*$", probe.stderr, re.MULTILINE)
            )
            version = properties.get("java.version", "")
            if not version:
                m = re.search(r'version "(\d+)(?:\.|\+|-)', probe.stderr + "\n" + probe.stdout)
                if m:
                    version = m.group(1)
            major = version.split(".")[0] if version else ""
            if probe.returncode != 0 or major != "21":
                continue

            runtime_home = Path(properties.get("java.home", str(candidate))).resolve()
            release_hash = _file_hash(runtime_home / "release")
            executable_hash = _file_hash(bin_java)
            if executable_hash is None:
                continue

            return JdkInfo(
                home=runtime_home,
                executable=bin_java,
                version=version,
                vendor=properties.get("java.vendor", "Unknown"),
                release_hash=release_hash,
                executable_hash=executable_hash,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

    raise AnalysisSetupError("Static analysis requires JDK 21; none of the candidate runtimes matched version 21")


def build_pmd_check_command(
    java_executable: Path,
    classpath: Sequence[Path],
    source_dir: Path,
    ruleset_path: Path,
    report_file: Path,
    language_version: str = "java-21",
    encoding: str = "UTF-8",
) -> tuple[str, ...]:
    """Construct command tuple for PMD check via PmdCli."""
    return (
        str(java_executable),
        "-Dfile.encoding=UTF-8",
        "-cp",
        os.pathsep.join(str(p) for p in classpath),
        "net.sourceforge.pmd.cli.PmdCli",
        "check",
        "--dir",
        str(source_dir),
        "--rulesets",
        str(ruleset_path),
        "--format",
        "xml",
        "--report-file",
        str(report_file),
        "--use-version",
        language_version,
        "--encoding",
        encoding,
        "--no-cache",
    )


def build_cpd_command(
    java_executable: Path,
    classpath: Sequence[Path],
    source_dir: Path,
    minimum_tokens: int = 100,
    language: str = "java",
    encoding: str = "UTF-8",
) -> tuple[str, ...]:
    """Construct command tuple for CPD token duplication via PmdCli."""
    return (
        str(java_executable),
        "-Dfile.encoding=UTF-8",
        "-cp",
        os.pathsep.join(str(p) for p in classpath),
        "net.sourceforge.pmd.cli.PmdCli",
        "cpd",
        "--minimum-tokens",
        str(minimum_tokens),
        "--language",
        language,
        "--dir",
        str(source_dir),
        "--format",
        "xml",
        "--encoding",
        encoding,
    )


def execute_java_process(
    command: tuple[str, ...],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: int = 120,
    stdout_path: Path | None = None,
) -> tuple[int, ExecutionStateKind]:
    """Execute Java subprocess with session group isolation and two-stage timeout."""
    stream = stdout_path.open("xb") if stdout_path is not None else None
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream if stream is not None else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout_seconds)
            return (code, "completed")
        except subprocess.TimeoutExpired:
            # Two-stage escalation: SIGTERM (2s grace) -> SIGKILL (2s grace)
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise AnalysisTimeoutError(process.returncode) from None
    finally:
        if stream is not None:
            stream.close()


def scan_java_sources(
    source_root: Path,
    source_roots: tuple[str, ...] = ("src/main/java",),
) -> tuple[tuple[str, ...], str | None, Path]:
    """Scan and inventory regular Java source files, returning relative paths, digest, and active directory."""
    candidate_dir = source_root / "src/main/java"
    active_dir = candidate_dir if candidate_dir.is_dir() else source_root

    paths: list[str] = []
    for parent, _dirs, files in os.walk(active_dir, followlinks=False):
        _dirs[:] = [
            d for d in _dirs
            if not d.startswith(".") and d not in ("build", "target", "out", "dist", "__pycache__")
        ]
        for f in files:
            if f.endswith(".java"):
                p = Path(parent) / f
                if p.is_file() and not p.is_symlink():
                    paths.append(p.relative_to(source_root).as_posix())

    paths.sort()
    if not paths:
        return ((), None, active_dir)

    digest = hashlib.sha256()
    for rel_path in paths:
        file_hash = _file_hash(source_root / rel_path) or ""
        digest.update(rel_path.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")

    return (tuple(paths), digest.hexdigest(), active_dir)


def check_exit_finding_concordance(
    exit_code: int | None,
    finding_count: int,
    tool: str,
) -> tuple[AnalysisStatusKind, tuple[AnalysisError, ...]]:
    """Verify strict concordance between tool exit code and parsed finding count."""
    errors: list[AnalysisError] = []
    if exit_code is None or exit_code not in (0, 4, 5):
        errors.append(AnalysisError("execution_failed", f"{tool.upper()} execution failed (exit {exit_code})"))
        return ("failed", tuple(errors))

    if exit_code == 5:
        errors.append(
            AnalysisError("incomplete_analysis", f"{tool.upper()} reported recoverable analysis errors (exit 5)")
        )
        return ("incomplete", tuple(errors))

    if (exit_code == 0 and finding_count > 0) or (exit_code == 4 and finding_count == 0):
        errors.append(
            AnalysisError(
                "exit_report_mismatch",
                f"{tool.upper()} exit {exit_code} and report finding count {finding_count} disagree",
            )
        )
        return ("failed", tuple(errors))

    status: AnalysisStatusKind = "findings" if finding_count > 0 else "clean"
    return (status, ())


class SubprocessIsolation(ExecutionIsolation):
    """Subprocess-based implementation of ExecutionIsolation protocol with process group session."""

    def __init__(self, java_home: Path | None = None) -> None:
        self.java_home = java_home

    def run_command(
        self,
        command: tuple[str, ...],
        cwd: Path,
        timeout_seconds: int = 120,
        read_only_root: bool = True,
        network: bool = False,
    ) -> tuple[int, bytes, bytes]:
        env = sanitize_environment(self.java_home)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            out, err = process.communicate(timeout=timeout_seconds)
            return (process.returncode, out, err)
        except subprocess.TimeoutExpired:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise AnalysisTimeoutError(process.returncode) from None


class PmdRunner(StaticAnalyzer):
    """Bot-owned execution adapter for PMD 7.3.0 and CPD implementing StaticAnalyzer port."""

    def __init__(
        self,
        java_home: Path | str | None = None,
        tool_classpath: tuple[Path, ...] | None = None,
        output_root: Path | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        self.jdk_info = discover_jdk_21(java_home)
        self.output_root = (output_root or Path("build/review-a")).resolve()
        self.timeout_seconds = timeout_seconds

        if tool_classpath is not None:
            self.tool_classpath = tuple(p.resolve() for p in tool_classpath)
        else:
            from review_bot.adapters.pmd.provisioning import verify_tool_installation

            resolved_cp, _, _ = verify_tool_installation(java_home=self.jdk_info.home)
            self.tool_classpath = resolved_cp

    def run_analysis(
        self,
        commit_sha: str,
        source_root: Path,
        ruleset_resource: str,
        profile: str = "java21-jdk-only-v1",
    ) -> AnalysisRun:
        """Execute static analyzer against isolated source tree and parse strict XML reports."""
        run_id = uuid.uuid4().hex
        started_at = _now_iso()
        source_root = source_root.resolve()

        output_dir = self.output_root / commit_sha / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Locate ruleset resource content
        ruleset_path = output_dir / "ruleset.xml"
        ruleset_file_candidate = Path(ruleset_resource)
        if ruleset_file_candidate.is_file():
            ruleset_content = ruleset_file_candidate.read_bytes()
        else:
            project_root = Path(__file__).resolve().parents[4]
            direct_candidate = project_root / ruleset_resource
            if direct_candidate.is_file():
                ruleset_content = direct_candidate.read_bytes()
            else:
                try:
                    ruleset_content = resources.files("review_bot.resources").joinpath(ruleset_resource).read_bytes()
                except Exception as exc:
                    raise AnalysisSetupError(
                        f"Could not locate ruleset resource '{ruleset_resource}': {exc}"
                    ) from exc

        ruleset_path.write_bytes(ruleset_content)
        ruleset_hash = hashlib.sha256(ruleset_content).hexdigest()

        # Classpath inventory & hash
        classpath_files = tuple((p.name, _file_hash(p) or "") for p in self.tool_classpath)
        classpath_hash = _hash_json(classpath_files)

        # Condition dictionary
        conditions: dict[str, object] = {
            "schema_version": 1,
            "pmd_version": "7.3.0",
            "cpd_version": "7.3.0",
            "ruleset_hash": ruleset_hash,
            "minimum_tokens": 100,
            "source_roots": ["src/main/java"],
            "language": "java",
            "language_version": "21",
            "encoding": "UTF-8",
            "type_resolution": "jdk_only",
            "java_version": self.jdk_info.version,
            "java_vendor": self.jdk_info.vendor,
            "java_release_hash": self.jdk_info.release_hash,
            "java_executable_hash": self.jdk_info.executable_hash,
            "tool_classpath": list(classpath_files),
            "classpath_hash": classpath_hash,
            "runtime_classpath_hashes": {"pmd": classpath_hash, "cpd": classpath_hash},
        }
        conditions_hash = _hash_json(conditions)
        profile_hash = _hash_json({
            key: value for key, value in conditions.items() if key != "runtime_classpath_hashes"
        })

        # Inventory Java sources
        source_files, source_hash, active_source_dir = scan_java_sources(source_root)
        source_count = len(source_files)

        # Optimization: Zero Java sources short-circuit
        if source_count == 0:
            pmd_report = ParsedReport(tool="pmd", path="", exists=False, parse_status="missing")
            cpd_report = ParsedReport(tool="cpd", path="", exists=False, parse_status="missing")
            pmd_exec = ExecutionRecord(
                schema_version=1,
                tool="pmd",
                exit_code=None,
                tool_version="7.3.0",
                java_version=self.jdk_info.version,
                java_vendor=self.jdk_info.vendor,
                classpath_hash=classpath_hash,
                profile_hash=profile_hash,
                sha=commit_sha,
                run_id=run_id,
                source_hash=None,
                source_root=str(source_root),
                source_roots=("src/main/java",),
                source_count=0,
                ruleset_hash=ruleset_hash,
                minimum_tokens=None,
                language_version="21",
                encoding="UTF-8",
                type_resolution="jdk_only",
                started_at=started_at,
                completed_at=_now_iso(),
                report_exists=False,
                report_hash=None,
                command=(),
                state="not_run",
            )
            cpd_exec = ExecutionRecord(
                schema_version=1,
                tool="cpd",
                exit_code=None,
                tool_version="7.3.0",
                java_version=self.jdk_info.version,
                java_vendor=self.jdk_info.vendor,
                classpath_hash=classpath_hash,
                profile_hash=profile_hash,
                sha=commit_sha,
                run_id=run_id,
                source_hash=None,
                source_root=str(source_root),
                source_roots=("src/main/java",),
                source_count=0,
                ruleset_hash=None,
                minimum_tokens=100,
                language_version="21",
                encoding="UTF-8",
                type_resolution="jdk_only",
                started_at=started_at,
                completed_at=_now_iso(),
                report_exists=False,
                report_hash=None,
                command=(),
                state="not_run",
            )
            self._write_execution_json(output_dir / "pmd-execution.json", pmd_exec)
            self._write_execution_json(output_dir / "cpd-execution.json", cpd_exec)

            pmd_analysis = ToolAnalysis(
                tool="pmd", status="no_sources", exit_code=None, report=pmd_report, execution=pmd_exec
            )
            cpd_analysis = ToolAnalysis(
                tool="cpd", status="no_sources", exit_code=None, report=cpd_report, execution=cpd_exec
            )

            run = AnalysisRun(
                schema_version=1,
                run_id=run_id,
                sha=commit_sha,
                profile_hash=profile_hash,
                conditions_hash=conditions_hash,
                conditions=conditions,
                output_dir=str(output_dir),
                started_at=started_at,
                completed_at=_now_iso(),
                status="no_sources",
                source_state="no_java_sources",
                source_files=(),
                source_hash=None,
                exclusions=(),
                pmd=pmd_analysis,
                cpd=cpd_analysis,
                build_exit_code=None,
                errors=(),
                limitations=(),
            )
            self._write_analysis_run_manifest(output_dir / "analysis-run.json", run)
            return run

        # Retain the executed inventory with the provenance consumed by the delta engine.
        snapshot_root = output_dir / "sources"
        for path in source_files:
            dest = snapshot_root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((source_root / path).read_bytes())
        source_root = snapshot_root
        source_files, source_hash, active_source_dir = scan_java_sources(source_root)

        # Run PMD check
        pmd_cmd = build_pmd_check_command(
            self.jdk_info.executable,
            self.tool_classpath,
            active_source_dir,
            ruleset_path,
            output_dir / "pmd.xml",
        )
        pmd_exec, pmd_report, pmd_status, pmd_errs = self._run_single_tool(
            tool="pmd",
            command=pmd_cmd,
            cwd=active_source_dir,
            output_dir=output_dir,
            commit_sha=commit_sha,
            run_id=run_id,
            source_root=source_root,
            source_count=source_count,
            source_hash=source_hash,
            ruleset_hash=ruleset_hash,
            minimum_tokens=None,
            classpath_hash=classpath_hash,
            profile_hash=profile_hash,
        )

        # Run CPD duplication
        cpd_cmd = build_cpd_command(
            self.jdk_info.executable,
            self.tool_classpath,
            active_source_dir,
            minimum_tokens=100,
        )
        cpd_exec, cpd_report, cpd_status, cpd_errs = self._run_single_tool(
            tool="cpd",
            command=cpd_cmd,
            cwd=active_source_dir,
            output_dir=output_dir,
            commit_sha=commit_sha,
            run_id=run_id,
            source_root=source_root,
            source_count=source_count,
            source_hash=source_hash,
            ruleset_hash=None,
            minimum_tokens=100,
            classpath_hash=classpath_hash,
            profile_hash=profile_hash,
            stdout_path=output_dir / "cpd.xml",
        )

        pmd_analysis = ToolAnalysis(
            tool="pmd",
            status=pmd_status,
            exit_code=pmd_exec.exit_code,
            report=pmd_report,
            execution=pmd_exec,
            errors=pmd_errs,
        )
        cpd_analysis = ToolAnalysis(
            tool="cpd",
            status=cpd_status,
            exit_code=cpd_exec.exit_code,
            report=cpd_report,
            execution=cpd_exec,
            errors=cpd_errs,
        )

        # Evaluate aggregate run status
        all_errors = list(pmd_errs + cpd_errs)
        statuses = {pmd_status, cpd_status}
        if all_errors or "failed" in statuses:
            overall_status: AnalysisStatusKind = "failed"
        elif "missing_report" in statuses:
            overall_status = "missing_report"
        elif "incomplete" in statuses:
            overall_status = "incomplete"
        elif "findings" in statuses:
            overall_status = "findings"
        else:
            overall_status = "clean"

        run = AnalysisRun(
            schema_version=1,
            run_id=run_id,
            sha=commit_sha,
            profile_hash=profile_hash,
            conditions_hash=conditions_hash,
            conditions=conditions,
            output_dir=str(output_dir),
            started_at=started_at,
            completed_at=_now_iso(),
            status=overall_status,
            source_state="complete",
            source_files=source_files,
            source_hash=source_hash,
            exclusions=(),
            pmd=pmd_analysis,
            cpd=cpd_analysis,
            build_exit_code=None,
            errors=tuple(all_errors),
            limitations=(),
        )

        # Write manifest
        self._write_analysis_run_manifest(output_dir / "analysis-run.json", run)
        return run

    def _run_single_tool(
        self,
        tool: str,
        command: tuple[str, ...],
        cwd: Path,
        output_dir: Path,
        commit_sha: str,
        run_id: str,
        source_root: Path,
        source_count: int,
        source_hash: str | None,
        ruleset_hash: str | None,
        minimum_tokens: int | None,
        classpath_hash: str,
        profile_hash: str,
        stdout_path: Path | None = None,
    ) -> tuple[ExecutionRecord, ParsedReport, AnalysisStatusKind, tuple[AnalysisError, ...]]:
        started_at = _now_iso()
        code: int | None = None
        state: ExecutionStateKind = "failed"
        errors: list[AnalysisError] = []

        clean_env = sanitize_environment(self.jdk_info.home)
        try:
            code, state = execute_java_process(
                command,
                cwd=cwd,
                env=clean_env,
                timeout_seconds=self.timeout_seconds,
                stdout_path=stdout_path,
            )
        except AnalysisTimeoutError as exc:
            code, state = exc.exit_code, "timed_out"
            errors.append(
                AnalysisError("execution_timeout", f"{tool.upper()} exceeded timeout of {self.timeout_seconds}s")
            )
        except OSError as exc:
            state = "failed"
            errors.append(AnalysisError("execution_failed", f"{tool.upper()} could not start: {exc}"))

        report_file = output_dir / f"{tool}.xml"
        report_exists = report_file.is_file()
        report_hash = _file_hash(report_file) if report_exists else None

        # Parse report
        parser = parse_pmd_xml if tool == "pmd" else parse_cpd_xml
        if report_exists:
            try:
                report = parser(report_file, source_root=source_root)
            except (ValueError, OSError, RuntimeError) as exc:
                report = ParsedReport(
                    tool=tool,
                    path=str(report_file),
                    exists=True,
                    parse_status="invalid",
                    errors=(AnalysisError("parse_error", str(exc)),),
                )
                errors.append(AnalysisError("parse_error", f"{tool.upper()} XML report could not be parsed: {exc}"))
        else:
            report = ParsedReport(tool=tool, path=str(report_file), exists=False, parse_status="missing")
            if state == "completed":
                errors.append(AnalysisError("missing_report", f"{tool.upper()} did not produce report XML"))

        # Concordance check
        concordance_status, concordance_errs = check_exit_finding_concordance(code, report.finding_count, tool)
        errors.extend(concordance_errs)

        exec_record = ExecutionRecord(
            schema_version=1,
            tool=tool,
            exit_code=code,
            tool_version="7.3.0",
            java_version=self.jdk_info.version,
            java_vendor=self.jdk_info.vendor,
            classpath_hash=classpath_hash,
            profile_hash=profile_hash,
            sha=commit_sha,
            run_id=run_id,
            source_hash=source_hash,
            source_root=str(source_root),
            source_roots=("src/main/java",),
            source_count=source_count,
            ruleset_hash=ruleset_hash,
            minimum_tokens=minimum_tokens,
            language_version="21",
            encoding="UTF-8",
            type_resolution="jdk_only",
            started_at=started_at,
            completed_at=_now_iso(),
            report_exists=report_exists,
            report_hash=report_hash,
            command=command,
            state=state,
        )
        self._write_execution_json(output_dir / f"{tool}-execution.json", exec_record)

        final_status: AnalysisStatusKind = concordance_status if not errors else "failed"
        return (exec_record, report, final_status, tuple(errors))

    def _write_execution_json(self, path: Path, record: ExecutionRecord) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(asdict(record), stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    def _write_analysis_run_manifest(self, path: Path, run: AnalysisRun) -> None:
        with path.open("w", encoding="utf-8") as stream:
            json.dump(asdict(run), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
