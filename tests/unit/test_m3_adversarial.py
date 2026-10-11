"""Empirical and adversarial stress test suite for Milestone 3:
- review_bot.adapters.pmd.provisioning (tool provisioning & checksum locking)
- review_bot.adapters.pmd.runner (process isolation, execution & concordance)
- tooling.verify_tools (standalone lockfile verification CLI)

Adversarial Dimensions:
1. Supply Chain & Lockfile Tampering Attacks
2. Process Isolation & Subprocess Sanitization
3. Timeout & Process Group Termination (os.killpg)
4. Exit Code & Finding Concordance Attacks
5. Zero Java Sources Edge Cases
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "tooling") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tooling"))

import verify_tools as cli_verify_tools

from review_bot.adapters.pmd.provisioning import (
    ToolVerificationError,
    find_pmd_jars,
    get_clean_env,
    verify_jdk,
    verify_tool_installation,
)
from review_bot.adapters.pmd.runner import (
    AnalysisTimeoutError,
    PmdRunner,
    SubprocessIsolation,
    check_exit_finding_concordance,
    execute_java_process,
    sanitize_environment,
    scan_java_sources,
)
from review_bot.domain.models_v1 import AnalysisRun

# ==============================================================================
# Helper Fixtures & Utilities
# ==============================================================================


@pytest.fixture
def locked_tools_sandbox(tmp_path: Path) -> dict[str, Path]:
    """Create a temporary sandbox containing copies of tools.lock.json, ruleset.xml, and lib/."""
    sandbox_root = tmp_path / "sandbox"
    tooling_dir = sandbox_root / "tooling"
    lib_dir = tooling_dir / "lib"
    resources_dir = sandbox_root / "src" / "review_bot" / "resources" / "pmd"
    lib_dir.mkdir(parents=True)
    resources_dir.mkdir(parents=True)

    real_tooling = PROJECT_ROOT / "tooling"
    real_lock = real_tooling / "tools.lock.json"
    real_lib = real_tooling / "lib"
    real_ruleset = PROJECT_ROOT / "src" / "review_bot" / "resources" / "pmd" / "ruleset.xml"

    sandbox_lock = tooling_dir / "tools.lock.json"
    shutil.copy2(real_lock, sandbox_lock)

    sandbox_ruleset = resources_dir / "ruleset.xml"
    shutil.copy2(real_ruleset, sandbox_ruleset)

    # Copy jars
    for jar in real_lib.glob("*.jar"):
        shutil.copy2(jar, lib_dir / jar.name)

    return {
        "root": sandbox_root,
        "lockfile": sandbox_lock,
        "lib_dir": lib_dir,
        "ruleset": sandbox_ruleset,
    }


# ==============================================================================
# 1. Supply Chain & Lockfile Tampering Attacks
# ==============================================================================


class TestSupplyChainLockfileTamperingAdversarial:
    def test_bit_flip_in_jar_detected_by_cli(self, locked_tools_sandbox: dict[str, Path]) -> None:
        """Mutating a single byte in a JAR file must cause CLI verification to fail."""
        target_jar = locked_tools_sandbox["lib_dir"] / "pmd-cli-7.3.0.jar"
        assert target_jar.is_file()

        # Flip the first byte
        data = bytearray(target_jar.read_bytes())
        data[0] ^= 0xFF
        target_jar.write_bytes(bytes(data))

        # CLI via python function
        res = cli_verify_tools.verify_tools(
            lockfile_path=locked_tools_sandbox["lockfile"],
            lib_dir=locked_tools_sandbox["lib_dir"],
            ruleset_path=locked_tools_sandbox["ruleset"],
            skip_java=True,
        )
        assert res == 1

        # CLI via subprocess
        proc = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "tooling" / "verify_tools.py"),
                "--lockfile",
                str(locked_tools_sandbox["lockfile"]),
                "--lib-dir",
                str(locked_tools_sandbox["lib_dir"]),
                "--ruleset",
                str(locked_tools_sandbox["ruleset"]),
                "--skip-java",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 1
        assert "SHA-256 mismatch on pmd-cli-7.3.0.jar" in proc.stderr

    def test_bit_flip_in_jar_rejected_by_provisioning_adapter(
        self, locked_tools_sandbox: dict[str, Path]
    ) -> None:
        """Mutating a single byte in a JAR must raise ToolVerificationError(checksum_mismatch)."""
        target_jar = locked_tools_sandbox["lib_dir"] / "Saxon-HE-12.4.jar"
        data = bytearray(target_jar.read_bytes())
        data[len(data) // 2] ^= 0x01
        target_jar.write_bytes(bytes(data))

        with pytest.raises(ToolVerificationError) as exc_info:
            verify_tool_installation(
                tools_lock_path=locked_tools_sandbox["lockfile"],
                lib_dir=locked_tools_sandbox["lib_dir"],
                ruleset_path=locked_tools_sandbox["ruleset"],
            )
        assert exc_info.value.reason == "checksum_mismatch"
        assert "Saxon-HE-12.4.jar" in exc_info.value.details

    def test_ruleset_byte_mutation_detected(self, locked_tools_sandbox: dict[str, Path]) -> None:
        """Mutating bytes in ruleset.xml without changing file length triggers checksum mismatch."""
        ruleset = locked_tools_sandbox["ruleset"]
        raw = bytearray(ruleset.read_bytes())
        # Flip a bit in the XML content keeping identical byte length
        raw[20] ^= 0x01
        ruleset.write_bytes(bytes(raw))

        # CLI check
        res = cli_verify_tools.verify_tools(
            lockfile_path=locked_tools_sandbox["lockfile"],
            lib_dir=locked_tools_sandbox["lib_dir"],
            ruleset_path=ruleset,
            skip_java=True,
        )
        assert res == 1

        # Provisioning check
        with pytest.raises(ToolVerificationError) as exc_info:
            verify_tool_installation(
                tools_lock_path=locked_tools_sandbox["lockfile"],
                lib_dir=locked_tools_sandbox["lib_dir"],
                ruleset_path=ruleset,
            )
        assert exc_info.value.reason == "checksum_mismatch"

    def test_ruleset_size_truncation_detected(self, locked_tools_sandbox: dict[str, Path]) -> None:
        """Truncating ruleset.xml triggers size_mismatch error."""
        ruleset = locked_tools_sandbox["ruleset"]
        ruleset.write_bytes(b"<ruleset/>")

        with pytest.raises(ToolVerificationError) as exc_info:
            verify_tool_installation(
                tools_lock_path=locked_tools_sandbox["lockfile"],
                lib_dir=locked_tools_sandbox["lib_dir"],
                ruleset_path=ruleset,
            )
        assert exc_info.value.reason == "size_mismatch"

    def test_altered_sha256_in_lockfile_rejected(self, locked_tools_sandbox: dict[str, Path]) -> None:
        """Altering a digest in tools.lock.json must be rejected."""
        lockfile = locked_tools_sandbox["lockfile"]
        data = json.loads(lockfile.read_text(encoding="utf-8"))
        # Tamper pmd-java digest
        real_hash = data["artifacts"]["pmd-java-7.3.0.jar"]["sha256"]
        tampered_hash = ("0" if real_hash[0] != "0" else "1") + real_hash[1:]
        data["artifacts"]["pmd-java-7.3.0.jar"]["sha256"] = tampered_hash
        lockfile.write_text(json.dumps(data, indent=2), encoding="utf-8")

        with pytest.raises(ToolVerificationError) as exc_info:
            verify_tool_installation(
                tools_lock_path=lockfile,
                lib_dir=locked_tools_sandbox["lib_dir"],
                ruleset_path=locked_tools_sandbox["ruleset"],
            )
        assert exc_info.value.reason == "checksum_mismatch"
        assert "pmd-java-7.3.0.jar" in exc_info.value.details

    def test_symlink_jar_injection_rejected_by_cli(self, locked_tools_sandbox: dict[str, Path]) -> None:
        """Replacing a regular JAR with a symlink must be strictly rejected by verify_tools."""
        target_jar = locked_tools_sandbox["lib_dir"] / "pmd-cli-7.3.0.jar"
        backup = locked_tools_sandbox["root"] / "pmd-cli-backup.jar"
        shutil.move(str(target_jar), str(backup))

        # Create symlink pointing to backup
        target_jar.symlink_to(backup)
        assert target_jar.is_symlink()

        # CLI rejection
        res = cli_verify_tools.verify_tools(
            lockfile_path=locked_tools_sandbox["lockfile"],
            lib_dir=locked_tools_sandbox["lib_dir"],
            ruleset_path=locked_tools_sandbox["ruleset"],
            skip_java=True,
        )
        assert res == 1

    def test_symlink_jar_ignored_or_rejected_in_provisioning(
        self, locked_tools_sandbox: dict[str, Path]
    ) -> None:
        """A symlinked JAR is filtered out during discovery and never accepted from the symlink path."""
        isolated_lib = locked_tools_sandbox["root"] / "isolated_lib"
        isolated_lib.mkdir()
        for jar in locked_tools_sandbox["lib_dir"].glob("*.jar"):
            if jar.name != "pmd-cli-7.3.0.jar":
                shutil.copy2(jar, isolated_lib / jar.name)
            else:
                real_file = locked_tools_sandbox["root"] / "real_cli.jar"
                shutil.copy2(jar, real_file)
                symlink_file = isolated_lib / "pmd-cli-7.3.0.jar"
                symlink_file.symlink_to(real_file)

        # In find_pmd_jars with search_dirs, symlink JAR artifact must raise ToolVerificationError
        with pytest.raises(ToolVerificationError) as exc_info:
            find_pmd_jars(search_dirs=[isolated_lib], project_root=locked_tools_sandbox["root"])
        assert exc_info.value.reason == "symlink_rejected"
        assert "pmd-cli-7.3.0.jar" in exc_info.value.details

        # verify_tool_installation also rejects symlinked JAR
        with pytest.raises(ToolVerificationError) as exc_info_install:
            verify_tool_installation(
                tools_lock_path=locked_tools_sandbox["lockfile"],
                lib_dir=isolated_lib,
                ruleset_path=locked_tools_sandbox["ruleset"],
            )
        assert exc_info_install.value.reason == "symlink_rejected"

    def test_symlink_ruleset_injection_rejected(self, locked_tools_sandbox: dict[str, Path]) -> None:
        """Replacing ruleset.xml with a symlink must be rejected."""
        ruleset = locked_tools_sandbox["ruleset"]
        backup = locked_tools_sandbox["root"] / "ruleset-backup.xml"
        shutil.move(str(ruleset), str(backup))

        ruleset.symlink_to(backup)
        assert ruleset.is_symlink()

        with pytest.raises(ToolVerificationError) as exc_info:
            verify_tool_installation(
                tools_lock_path=locked_tools_sandbox["lockfile"],
                lib_dir=locked_tools_sandbox["lib_dir"],
                ruleset_path=ruleset,
            )
        assert exc_info.value.reason == "symlink_rejected"

    def test_strict_mode_rejects_unauthorized_extra_files(
        self, locked_tools_sandbox: dict[str, Path]
    ) -> None:
        """In strict mode, verify_tools must fail if unauthorized files exist in lib_dir."""
        hostile_file = locked_tools_sandbox["lib_dir"] / "evil-payload.jar"
        hostile_file.write_bytes(b"malicious_payload")

        # Non-strict mode passes
        res_non_strict = cli_verify_tools.verify_tools(
            lockfile_path=locked_tools_sandbox["lockfile"],
            lib_dir=locked_tools_sandbox["lib_dir"],
            ruleset_path=locked_tools_sandbox["ruleset"],
            skip_java=True,
            strict=False,
        )
        assert res_non_strict == 0

        # Strict mode fails
        res_strict = cli_verify_tools.verify_tools(
            lockfile_path=locked_tools_sandbox["lockfile"],
            lib_dir=locked_tools_sandbox["lib_dir"],
            ruleset_path=locked_tools_sandbox["ruleset"],
            skip_java=True,
            strict=True,
        )
        assert res_strict == 1

    @pytest.mark.parametrize(
        "mock_version_output,expected_reason",
        [
            ('openjdk version "17.0.2" 2022-01-18\njava.version = 17.0.2', "jdk_version_mismatch"),
            ('openjdk version "25-ea" 2025-09-16\njava.version = 25-ea', "jdk_version_mismatch"),
            ('java version "1.8.0_312"\njava.version = 1.8.0_312', "jdk_version_mismatch"),
            ('unknown format output', "jdk_probe_failed"),
        ],
    )
    def test_non_21_java_versions_rejected(
        self,
        tmp_path: Path,
        mock_version_output: str,
        expected_reason: str,
    ) -> None:
        """Any JVM reporting a non-21 major version must be rejected immediately."""
        fake_bin = tmp_path / "bin" / "java"
        fake_bin.parent.mkdir(parents=True)
        # Create executable shell script mimicking java
        fake_bin.write_text(
            f"#!/bin/sh\ncat <<'EOF'\n{mock_version_output}\nEOF\n",
            encoding="utf-8",
        )
        fake_bin.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        # CLI verify_java_version check
        ok, msg = cli_verify_tools.verify_java_version(fake_bin)
        assert not ok
        assert "expected 21" in msg or "Could not determine" in msg

        # Provisioning verify_jdk check
        with pytest.raises(ToolVerificationError) as exc_info:
            verify_jdk(java_executable=fake_bin)
        assert exc_info.value.reason == expected_reason

    def test_crashing_java_executable_rejected(self, tmp_path: Path) -> None:
        """A crashing Java executable (exit code 1 or segfault) must raise ToolVerificationError."""
        fake_bin = tmp_path / "bin" / "java"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake_bin.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        with pytest.raises(ToolVerificationError):
            verify_jdk(java_executable=fake_bin)


# ==============================================================================
# 2. Process Isolation & Subprocess Sanitization
# ==============================================================================


class TestProcessIsolationAndSanitizationAdversarial:
    HOSTILE_ENV_KEYS: ClassVar[dict[str, str]] = {
        "JAVA_TOOL_OPTIONS": "-agentlib:jdwp=transport=dt_socket,server=y,suspend=y",
        "_JAVA_OPTIONS": "-Dmalicious=true",
        "JDK_JAVA_OPTIONS": "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "JAVA_OPTS": "-Xmx999g",
        "GRADLE_OPTS": "--init-script=/tmp/evil.gradle",
        "CLASSPATH": "/tmp/evil.jar:/evil/classes",
        "GITHUB_TOKEN": "ghp_super_secret_adversarial_token",
        "GITHUB_APP_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...",
        "GITHUB_WEBHOOK_SECRET": "whsec_adversarial_secret",
        "OPENAI_API_KEY": "sk-adversarial-model-key",
        "ANTHROPIC_API_KEY": "sk-ant-adversarial-key",
        "MODEL_API_KEY": "model_key_secret",
    }

    def test_environment_sanitizer_removes_all_hostile_keys(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sanitize_environment and get_clean_env must scrub all hostile variables and secrets."""
        for k, v in self.HOSTILE_ENV_KEYS.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("BENIGN_ENV_KEY", "should_survive")

        cleaned_runner = sanitize_environment(java_home=Path("/opt/jdk21"))
        cleaned_prov = get_clean_env(java_home=Path("/opt/jdk21"))

        # runner.py scrubs all hostile keys
        for k in self.HOSTILE_ENV_KEYS:
            assert k not in cleaned_runner, f"Runner environment leaked {k}"

        # provisioning.py scrubs all hostile keys including ANTHROPIC_API_KEY
        for k in self.HOSTILE_ENV_KEYS:
            assert k not in cleaned_prov, f"Provisioning environment leaked {k}"

        assert cleaned_runner["BENIGN_ENV_KEY"] == "should_survive"
        assert cleaned_runner["JAVA_HOME"] == "/opt/jdk21"
        assert cleaned_runner["LC_ALL"] == "C.UTF-8"
        assert cleaned_runner["LANG"] == "C.UTF-8"

    def test_subprocess_isolation_execution_does_not_leak_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When executing via SubprocessIsolation, child processes must see 0 leaked secrets."""
        for k, v in self.HOSTILE_ENV_KEYS.items():
            monkeypatch.setenv(k, v)

        iso = SubprocessIsolation()
        # Probe os.environ inside the spawned child
        probe_cmd = (
            sys.executable,
            "-c",
            "import os, json; print(json.dumps(dict(os.environ)))",
        )
        code, out, _err = iso.run_command(probe_cmd, cwd=tmp_path)
        assert code == 0

        child_env = json.loads(out.decode("utf-8"))
        for hostile_key in self.HOSTILE_ENV_KEYS:
            assert hostile_key not in child_env, f"Subprocess child environment leaked {hostile_key}"


# ==============================================================================
# 3. Timeout & Process Group Termination (os.killpg)
# ==============================================================================


class TestTimeoutAndProcessGroupTerminationAdversarial:
    def test_simple_timeout_raises_analysis_timeout_error(self, tmp_path: Path) -> None:
        """A hanging command must timeout and raise AnalysisTimeoutError."""
        iso = SubprocessIsolation()
        with pytest.raises(AnalysisTimeoutError):
            iso.run_command(
                (sys.executable, "-c", "import time; time.sleep(10)"),
                cwd=tmp_path,
                timeout_seconds=1,
            )

    def test_process_group_kills_entire_tree_without_orphans(self, tmp_path: Path) -> None:
        """os.killpg must terminate child and background grandchild processes."""
        pid_file = tmp_path / "grandchild.pid"

        # Python script that spawns a background grandchild process that loops forever
        script = f"""
import subprocess, sys, time, os
from pathlib import Path

# Grandchild writes its own PID to file immediately
child_code = "import os, time; from pathlib import Path; Path('{pid_file}').write_text(str(os.getpid())); time.sleep(100)"
proc = subprocess.Popen([sys.executable, "-c", child_code])

# Wait until pid_file is written
for _ in range(100):
    if Path('{pid_file}').is_file():
        break
    time.sleep(0.05)

# Parent sleeps indefinitely
time.sleep(100)
"""
        script_file = tmp_path / "fork_tree.py"
        script_file.write_text(script, encoding="utf-8")

        iso = SubprocessIsolation()
        with pytest.raises(AnalysisTimeoutError):
            iso.run_command(
                (sys.executable, str(script_file)),
                cwd=tmp_path,
                timeout_seconds=2,
            )

        # Verify grandchild PID exists and was killed
        assert pid_file.is_file(), "Grandchild failed to start"
        grandchild_pid = int(pid_file.read_text().strip())

        # Give small grace period for kernel cleanup
        time.sleep(0.5)

        # Check if grandchild is alive via os.kill(pid, 0)
        is_alive = True
        try:
            os.kill(grandchild_pid, 0)
        except (ProcessLookupError, OSError):
            is_alive = False

        assert not is_alive, f"Grandchild process {grandchild_pid} was orphaned and survived!"

    def test_sigterm_ignored_escalates_to_sigkill(self, tmp_path: Path) -> None:
        """Processes ignoring SIGTERM must be killed by escalated SIGKILL."""
        script = """
import signal, time

# Explicitly ignore SIGTERM
signal.signal(signal.SIGTERM, signal.SIG_IGN)

while True:
    time.sleep(0.1)
"""
        script_file = tmp_path / "ignore_sigterm.py"
        script_file.write_text(script, encoding="utf-8")

        clean_env = sanitize_environment()
        with pytest.raises(AnalysisTimeoutError) as exc_info:
            execute_java_process(
                (sys.executable, str(script_file)),
                cwd=tmp_path,
                env=clean_env,
                timeout_seconds=1,
            )
        # Should have returncode -9 (SIGKILL)
        assert exc_info.value.exit_code in (-signal.SIGKILL, -9, 137, None)


# ==============================================================================
# 4. Exit Code & Finding Concordance Attacks
# ==============================================================================


class TestConcordanceAdversarialAttacks:
    @pytest.mark.parametrize(
        "exit_code,finding_count,expected_status,expected_kind",
        [
            (0, 5, "failed", "exit_report_mismatch"),       # Exit 0 but has findings
            (4, 0, "failed", "exit_report_mismatch"),       # Exit 4 but has zero findings
            (0, 100, "failed", "exit_report_mismatch"),     # Exit 0 with 100 findings
            (5, 0, "incomplete", "incomplete_analysis"),    # Exit 5 recoverable lexer error
            (5, 3, "incomplete", "incomplete_analysis"),    # Exit 5 with findings
            (1, 0, "failed", "execution_failed"),           # Exit 1 fatal error
            (2, 0, "failed", "execution_failed"),           # Exit 2 argument syntax error
            (137, 0, "failed", "execution_failed"),         # OOM killed (SIGKILL)
            (-9, 0, "failed", "execution_failed"),          # Process terminated by signal
            (None, 0, "failed", "execution_failed"),        # Process crashed without exit code
        ],
    )
    def test_concordance_matrix(
        self,
        exit_code: int | None,
        finding_count: int,
        expected_status: str,
        expected_kind: str,
    ) -> None:
        status, errors = check_exit_finding_concordance(exit_code, finding_count, "pmd")
        assert status == expected_status
        assert len(errors) >= 1
        assert errors[0].kind == expected_kind

    def test_runner_concordance_mismatch_exit_0_with_violations(
        self, tmp_path: Path
    ) -> None:
        """When tool exits 0 but report has violations, AnalysisRun must fail with exit_report_mismatch."""
        runner = PmdRunner()

        src = tmp_path / "src/main/java/demo"
        src.mkdir(parents=True)
        (src / "Test.java").write_text("package demo;\npublic class Test {}\n", encoding="utf-8")

        def fake_exec(cmd: tuple[str, ...], **kwargs: Any) -> tuple[int, str]:
            out_xml = kwargs.get("stdout_path")
            if "--report-file" in cmd:
                idx = cmd.index("--report-file")
                rep_path = Path(cmd[idx + 1])
                rep_path.write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<pmd version="7.3.0">\n'
                    '  <file name="src/main/java/demo/Test.java">\n'
                    '    <violation beginline="1" endline="1" begincolumn="1" endcolumn="1" '
                    'rule="CognitiveComplexity" ruleset="Design" priority="3" class="Test" '
                    'method="test" package="demo">Complexity too high</violation>\n'
                    '  </file>\n'
                    '</pmd>',
                    encoding="utf-8",
                )
            elif out_xml is not None:
                Path(out_xml).write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n<pmd-cpd version="1.0.0"/>\n',
                    encoding="utf-8",
                )
            # Adversarial attack: exit 0 despite violation in report
            return (0, "completed")

        with patch("review_bot.adapters.pmd.runner.execute_java_process", side_effect=fake_exec):
            run = runner.run_analysis(
                commit_sha="a" * 40,
                source_root=tmp_path,
                ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
            )

            assert isinstance(run, AnalysisRun)
            assert run.status == "failed"
            assert run.pmd.status == "failed"
            error_kinds = [e.kind for e in run.errors]
            assert "exit_report_mismatch" in error_kinds

    def test_runner_concordance_mismatch_exit_4_clean_report(
        self, tmp_path: Path
    ) -> None:
        """When tool exits 4 but report has 0 violations, AnalysisRun must fail with exit_report_mismatch."""
        runner = PmdRunner()

        src = tmp_path / "src/main/java/demo"
        src.mkdir(parents=True)
        (src / "Test.java").write_text("package demo;\npublic class Test {}\n", encoding="utf-8")

        def fake_exec(cmd: tuple[str, ...], **kwargs: Any) -> tuple[int, str]:
            out_xml = kwargs.get("stdout_path")
            if "--report-file" in cmd:
                idx = cmd.index("--report-file")
                rep_path = Path(cmd[idx + 1])
                # Empty report with 0 violations
                rep_path.write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n<pmd version="7.3.0"/>\n',
                    encoding="utf-8",
                )
            elif out_xml is not None:
                Path(out_xml).write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n<pmd-cpd version="1.0.0"/>\n',
                    encoding="utf-8",
                )
            # Adversarial attack: exit 4 despite 0 violations
            return (4, "completed")

        with patch("review_bot.adapters.pmd.runner.execute_java_process", side_effect=fake_exec):
            run = runner.run_analysis(
                commit_sha="e" * 40,
                source_root=tmp_path,
                ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
            )

            assert isinstance(run, AnalysisRun)
            assert run.status == "failed"
            assert run.pmd.status == "failed"
            error_kinds = [e.kind for e in run.errors]
            assert "exit_report_mismatch" in error_kinds


# ==============================================================================
# 5. Zero Java Sources Edge Cases
# ==============================================================================


class TestZeroJavaSourcesAdversarialEdgeCases:
    def test_completely_empty_directory(self, tmp_path: Path) -> None:
        """An empty repository short-circuits to no_sources without invoking JVM."""
        runner = PmdRunner()
        run = runner.run_analysis(
            commit_sha="b" * 40,
            source_root=tmp_path,
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert run.status == "no_sources"
        assert run.source_state == "no_java_sources"
        assert len(run.source_files) == 0
        assert run.pmd.status == "no_sources"
        assert run.cpd.status == "no_sources"
        assert run.pmd.execution is not None
        assert run.pmd.execution.state == "not_run"
        assert run.cpd.execution is not None
        assert run.cpd.execution.state == "not_run"

        # Manifests check
        out_dir = Path(run.output_dir)
        assert (out_dir / "analysis-run.json").is_file()
        assert (out_dir / "pmd-execution.json").is_file()
        assert (out_dir / "cpd-execution.json").is_file()

    def test_only_non_java_files_present(self, tmp_path: Path) -> None:
        """Repositories with markdown, python, txt, or json files short-circuit."""
        (tmp_path / "README.md").write_text("# Hello", encoding="utf-8")
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
        (tmp_path / "config.yaml").write_text("key: value", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")

        runner = PmdRunner()
        run = runner.run_analysis(
            commit_sha="c" * 40,
            source_root=tmp_path,
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert run.status == "no_sources"
        assert run.source_state == "no_java_sources"
        assert len(run.source_files) == 0
        assert run.pmd.execution.state == "not_run"

    def test_only_symlinked_java_files_short_circuit(self, tmp_path: Path) -> None:
        """Symlinked .java files must be ignored, short-circuiting to no_sources."""
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        real_java = outside_dir / "Real.java"
        real_java.write_text("public class Real {}", encoding="utf-8")

        inside_dir = tmp_path / "repo" / "src" / "main" / "java"
        inside_dir.mkdir(parents=True)
        symlink_java = inside_dir / "Symlink.java"
        symlink_java.symlink_to(real_java)
        assert symlink_java.is_symlink()

        files, digest, _active = scan_java_sources(tmp_path / "repo")
        assert len(files) == 0
        assert digest is None

        runner = PmdRunner()
        run = runner.run_analysis(
            commit_sha="d" * 40,
            source_root=tmp_path / "repo",
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert run.status == "no_sources"
        assert run.source_state == "no_java_sources"
        assert run.pmd.execution.state == "not_run"

    def test_broken_symlinks_do_not_crash_scanner(self, tmp_path: Path) -> None:
        """Broken symlinks in source tree are safely ignored."""
        inside_dir = tmp_path / "src" / "main" / "java"
        inside_dir.mkdir(parents=True)
        broken = inside_dir / "Broken.java"
        broken.symlink_to(tmp_path / "nonexistent.java")

        files, digest, _active = scan_java_sources(tmp_path)
        assert len(files) == 0
        assert digest is None
