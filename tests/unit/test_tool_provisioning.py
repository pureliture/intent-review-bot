"""Unit tests for static analysis tool provisioning, checksum lock, and verifier CLI."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from review_bot.adapters.pmd.provisioning import (
    ArtifactLock,
    JdkIdentity,
    ToolsLock,
    ToolVerificationError,
    compute_classpath_hash,
    find_pmd_jars,
    get_clean_env,
    load_tools_lock,
    verify_jdk,
    verify_tool_installation,
)


class TestToolsLockLoading:
    def test_load_default_tools_lock(self) -> None:
        lock = load_tools_lock()
        assert isinstance(lock, ToolsLock)
        assert lock.pmd_version == "7.3.0"
        assert lock.java_version == "21"
        assert len(lock.artifacts) == 25
        assert "pmd-cli-7.3.0.jar" in lock.artifacts
        assert "ruleset.xml" in lock.artifacts

        ruleset_art = lock.artifacts["ruleset.xml"]
        assert isinstance(ruleset_art, ArtifactLock)
        assert ruleset_art.category == "ruleset"
        assert ruleset_art.size_bytes == 1987
        assert ruleset_art.sha256 == "c3e9649c6e69a94a22644db666c098d05a7b7f4ea364e496d320cdd24dece2f2"

    def test_missing_lockfile_raises_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing.lock.json"
        with pytest.raises(ToolVerificationError, match="missing_lockfile"):
            load_tools_lock(missing)

    def test_invalid_json_raises_error(self, tmp_path: Path) -> None:
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ToolVerificationError, match="invalid_lockfile_json"):
            load_tools_lock(bad_json)

    def test_schema_violation_raises_error(self, tmp_path: Path) -> None:
        bad_schema = tmp_path / "incomplete.json"
        bad_schema.write_text(json.dumps({"pmd_version": "7.3.0"}), encoding="utf-8")
        with pytest.raises(ToolVerificationError, match="schema_contract_violation"):
            load_tools_lock(bad_schema)

    def test_unsupported_version_raises_error(self, tmp_path: Path) -> None:
        bad_version = tmp_path / "version.json"
        bad_version.write_text(
            json.dumps({"pmd_version": "6.55.0", "java_version": "21", "artifacts": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ToolVerificationError, match="version_mismatch"):
            load_tools_lock(bad_version)


class TestEnvironmentSanitization:
    def test_strips_hazardous_vars_and_secrets(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("JAVA_TOOL_OPTIONS", "-javaagent:malicious.jar")
        monkeypatch.setenv("_JAVA_OPTIONS", "-Dfoo=bar")
        monkeypatch.setenv("CLASSPATH", "/bad/path")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("SAFE_VAR", "keep_me")

        clean = get_clean_env(java_home=Path("/opt/jdk21"))
        assert "JAVA_TOOL_OPTIONS" not in clean
        assert "_JAVA_OPTIONS" not in clean
        assert "CLASSPATH" not in clean
        assert "GITHUB_TOKEN" not in clean
        assert "OPENAI_API_KEY" not in clean
        assert "ANTHROPIC_API_KEY" not in clean
        assert clean["SAFE_VAR"] == "keep_me"
        assert clean["JAVA_HOME"] == "/opt/jdk21"
        assert clean["LC_ALL"] == "C.UTF-8"
        assert clean["LANG"] == "C.UTF-8"


class TestJdkVerification:
    def test_verify_host_corretto_21(self) -> None:
        jdk = verify_jdk()
        assert isinstance(jdk, JdkIdentity)
        assert jdk.major_version == 21
        assert "21." in jdk.version
        assert jdk.executable.is_file()
        assert len(jdk.executable_hash) == 64

    def test_nonexistent_executable_raises_error(self, tmp_path: Path) -> None:
        bad_java = tmp_path / "bin" / "java"
        with pytest.raises(ToolVerificationError, match="jdk_not_found"):
            verify_jdk(java_executable=bad_java)


class TestClasspathAndJarProvisioning:
    def test_find_pmd_jars_discovers_24_jars(self) -> None:
        jars = find_pmd_jars()
        assert "pmd-cli-7.3.0.jar" in jars
        assert "pmd-java-7.3.0.jar" in jars
        assert "pmd-core-7.3.0.jar" in jars
        assert len(jars) >= 24

    def test_compute_classpath_hash_is_deterministic(self) -> None:
        jars = find_pmd_jars()
        cp = (jars["pmd-cli-7.3.0.jar"], jars["pmd-core-7.3.0.jar"])
        h1 = compute_classpath_hash(cp)
        # Reversed order should still produce identical hash because it sorts by name
        h2 = compute_classpath_hash(tuple(reversed(cp)))
        assert h1 == h2
        assert len(h1) == 64

    def test_verify_tool_installation_succeeds(self) -> None:
        classpath, jdk, ruleset = verify_tool_installation()
        assert len(classpath) == 24
        # Deterministic ordering checks: pmd-cli is first, then pmd-java, pmd-core
        assert classpath[0].name == "pmd-cli-7.3.0.jar"
        assert classpath[1].name == "pmd-java-7.3.0.jar"
        assert classpath[2].name == "pmd-core-7.3.0.jar"
        assert jdk.major_version == 21
        assert ruleset.is_file()
        assert ruleset.name == "ruleset.xml"

    def test_tampered_ruleset_detected(self, tmp_path: Path) -> None:
        tampered_ruleset = tmp_path / "ruleset.xml"
        tampered_ruleset.write_text("<ruleset>tampered</ruleset>", encoding="utf-8")
        with pytest.raises(ToolVerificationError, match="size_mismatch|checksum_mismatch"):
            verify_tool_installation(ruleset_path=tampered_ruleset)

    def test_symlink_ruleset_rejected(self, tmp_path: Path) -> None:
        real_ruleset = Path("src/review_bot/resources/pmd/ruleset.xml").resolve()
        symlink_ruleset = tmp_path / "ruleset_symlink.xml"
        symlink_ruleset.symlink_to(real_ruleset)
        with pytest.raises(ToolVerificationError, match="symlink_rejected"):
            verify_tool_installation(ruleset_path=symlink_ruleset)

    def test_symlink_jar_rejected(self, tmp_path: Path) -> None:
        lib_dir = tmp_path / "lib"
        lib_dir.mkdir()
        fake_jar = tmp_path / "fake.jar"
        fake_jar.write_text("fake jar content", encoding="utf-8")
        symlink_jar = lib_dir / "pmd-cli-7.3.0.jar"
        symlink_jar.symlink_to(fake_jar)
        with pytest.raises(ToolVerificationError, match="symlink_rejected"):
            verify_tool_installation(lib_dir=lib_dir)


class TestVerifyToolsCli:
    def test_standalone_cli_success(self) -> None:
        project_root = Path(__file__).resolve().parents[2]
        cli = project_root / "tooling" / "verify_tools.py"
        res = subprocess.run(
            [sys.executable, str(cli), "--verbose"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0
        assert "SUCCESS: Verified 25 locked artifacts and JDK 21 environment." in res.stdout

    def test_standalone_cli_failure_on_corrupt_lockfile(self, tmp_path: Path) -> None:
        project_root = Path(__file__).resolve().parents[2]
        cli = project_root / "tooling" / "verify_tools.py"
        corrupt_lock = tmp_path / "corrupt.json"
        corrupt_lock.write_text("{}", encoding="utf-8")
        res = subprocess.run(
            [sys.executable, str(cli), "--lockfile", str(corrupt_lock)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 1
        assert "Missing required lockfile key" in res.stderr
