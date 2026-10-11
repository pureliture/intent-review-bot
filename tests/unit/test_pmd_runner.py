"""Unit tests for PMD/CPD execution runner, process isolation, and concordance."""
from __future__ import annotations

from pathlib import Path

import pytest

from review_bot.adapters.pmd.runner import (
    AnalysisTimeoutError,
    PmdRunner,
    SubprocessIsolation,
    build_cpd_command,
    build_pmd_check_command,
    check_exit_finding_concordance,
    scan_java_sources,
)
from review_bot.domain.models_v1 import AnalysisRun
from review_bot.ports import ExecutionIsolation, StaticAnalyzer


class TestCommandConstruction:
    def test_build_pmd_check_command(self) -> None:
        cmd = build_pmd_check_command(
            java_executable=Path("/opt/jdk/bin/java"),
            classpath=(Path("/lib/pmd.jar"), Path("/lib/core.jar")),
            source_dir=Path("/src"),
            ruleset_path=Path("/rules/ruleset.xml"),
            report_file=Path("/out/pmd.xml"),
            language_version="java-21",
            encoding="UTF-8",
        )
        assert cmd[0] == "/opt/jdk/bin/java"
        assert "-Dfile.encoding=UTF-8" in cmd
        assert "net.sourceforge.pmd.cli.PmdCli" in cmd
        assert "check" in cmd
        assert "--rulesets" in cmd
        assert "/rules/ruleset.xml" in cmd
        assert "--report-file" in cmd
        assert "/out/pmd.xml" in cmd
        assert "--use-version" in cmd
        assert "java-21" in cmd
        assert "--encoding" in cmd
        assert "UTF-8" in cmd
        assert "--no-cache" in cmd

    def test_build_cpd_command(self) -> None:
        cmd = build_cpd_command(
            java_executable=Path("/opt/jdk/bin/java"),
            classpath=(Path("/lib/pmd.jar"),),
            source_dir=Path("/src"),
            minimum_tokens=100,
            language="java",
            encoding="UTF-8",
        )
        assert cmd[0] == "/opt/jdk/bin/java"
        assert "net.sourceforge.pmd.cli.PmdCli" in cmd
        assert "cpd" in cmd
        assert "--minimum-tokens" in cmd
        assert "100" in cmd
        assert "--language" in cmd
        assert "java" in cmd
        assert "--format" in cmd
        assert "xml" in cmd
        assert "--encoding" in cmd
        assert "UTF-8" in cmd


class TestConcordanceVerification:
    def test_exit_0_zero_findings_is_clean(self) -> None:
        status, errors = check_exit_finding_concordance(0, 0, "pmd")
        assert status == "clean"
        assert len(errors) == 0

    def test_exit_4_positive_findings_is_findings(self) -> None:
        status, errors = check_exit_finding_concordance(4, 2, "pmd")
        assert status == "findings"
        assert len(errors) == 0

    def test_exit_5_is_incomplete(self) -> None:
        status, errors = check_exit_finding_concordance(5, 0, "pmd")
        assert status == "incomplete"
        assert len(errors) == 1
        assert errors[0].kind == "incomplete_analysis"

    def test_exit_0_with_findings_is_mismatch_failure(self) -> None:
        status, errors = check_exit_finding_concordance(0, 3, "pmd")
        assert status == "failed"
        assert len(errors) == 1
        assert errors[0].kind == "exit_report_mismatch"

    def test_exit_4_with_zero_findings_is_mismatch_failure(self) -> None:
        status, errors = check_exit_finding_concordance(4, 0, "pmd")
        assert status == "failed"
        assert len(errors) == 1
        assert errors[0].kind == "exit_report_mismatch"

    def test_exit_1_is_execution_failed(self) -> None:
        status, errors = check_exit_finding_concordance(1, 0, "pmd")
        assert status == "failed"
        assert len(errors) == 1
        assert errors[0].kind == "execution_failed"

    def test_exit_none_is_execution_failed(self) -> None:
        status, errors = check_exit_finding_concordance(None, 0, "pmd")
        assert status == "failed"
        assert len(errors) == 1
        assert errors[0].kind == "execution_failed"


class TestSourceScanner:
    def test_scan_with_java_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src/main/java/demo"
        src.mkdir(parents=True)
        (src / "Foo.java").write_text("class Foo {}", encoding="utf-8")
        (src / "Bar.java").write_text("class Bar {}", encoding="utf-8")
        (src / "readme.txt").write_text("notes", encoding="utf-8")

        files, digest, _active = scan_java_sources(tmp_path)
        assert len(files) == 2
        assert "src/main/java/demo/Bar.java" in files
        assert "src/main/java/demo/Foo.java" in files
        assert digest is not None
        assert len(digest) == 64

    def test_scan_zero_java_files(self, tmp_path: Path) -> None:
        src = tmp_path / "docs"
        src.mkdir(parents=True)
        (src / "README.md").write_text("hello", encoding="utf-8")

        files, digest, _active = scan_java_sources(tmp_path)
        assert len(files) == 0
        assert digest is None

    def test_scan_ignores_pruned_directories(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "Valid.java").write_text("class Valid {}", encoding="utf-8")

        for ignored_dir in ("build", "target", "out", "dist", "__pycache__", ".git", ".gradle"):
            d = tmp_path / ignored_dir
            d.mkdir()
            (d / "Ignored.java").write_text("class Ignored {}", encoding="utf-8")

        files, _digest, _active = scan_java_sources(tmp_path)
        assert len(files) == 1
        assert files[0] == "src/Valid.java"


class TestSubprocessIsolation:
    def test_satisfies_protocol(self) -> None:
        iso = SubprocessIsolation()
        assert isinstance(iso, ExecutionIsolation)

    def test_run_command_success(self, tmp_path: Path) -> None:
        iso = SubprocessIsolation()
        code, out, _err = iso.run_command(("echo", "isolated_execution"), cwd=tmp_path)
        assert code == 0
        assert b"isolated_execution" in out

    def test_run_command_timeout(self, tmp_path: Path) -> None:
        iso = SubprocessIsolation()
        with pytest.raises(AnalysisTimeoutError):
            iso.run_command(("sleep", "5"), cwd=tmp_path, timeout_seconds=1)


class TestPmdRunnerEndToEnd:
    @pytest.fixture
    def runner(self) -> PmdRunner:
        return PmdRunner()

    def test_satisfies_protocol(self, runner: PmdRunner) -> None:
        assert isinstance(runner, StaticAnalyzer)

    def test_zero_java_sources_short_circuit(self, runner: PmdRunner, tmp_path: Path) -> None:
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "index.md").write_text("# Docs", encoding="utf-8")

        sha = "1" * 40
        run = runner.run_analysis(
            commit_sha=sha,
            source_root=tmp_path,
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert isinstance(run, AnalysisRun)
        assert run.status == "no_sources"
        assert run.source_state == "no_java_sources"
        assert len(run.source_files) == 0
        assert run.pmd.status == "no_sources"
        assert run.cpd.status == "no_sources"
        assert run.pmd.execution is not None
        assert run.pmd.execution.state == "not_run"
        assert run.cpd.execution is not None
        assert run.cpd.execution.state == "not_run"

        # Manifests were written
        out_dir = Path(run.output_dir)
        assert (out_dir / "pmd-execution.json").is_file()
        assert (out_dir / "cpd-execution.json").is_file()
        assert (out_dir / "analysis-run.json").is_file()

    def test_clean_java_source_execution(self, runner: PmdRunner, tmp_path: Path) -> None:
        src = tmp_path / "src/main/java/demo"
        src.mkdir(parents=True)
        (src / "Clean.java").write_text(
            "package demo;\npublic class Clean {\n  public int getFive() {\n    return 5;\n  }\n}\n",
            encoding="utf-8",
        )
        sha = "2" * 40
        run = runner.run_analysis(
            commit_sha=sha,
            source_root=tmp_path,
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert run.status == "clean"
        assert run.source_state == "complete"
        assert run.pmd.status == "clean"
        assert run.cpd.status == "clean"
        assert len(run.pmd.report.violations) == 0
        assert len(run.cpd.report.duplications) == 0

    def test_cognitive_complexity_violation_detected(self, runner: PmdRunner, tmp_path: Path) -> None:
        src = tmp_path / "src/main/java/demo"
        src.mkdir(parents=True)
        # Deeply nested logic triggers CognitiveComplexity
        code = """package demo;
public class HighComplexity {
    public void deepNest(int a, int b, int c, int d, int e) {
        if (a > 0) {
            for (int i = 0; i < 10; i++) {
                if (b > 0) {
                    while (c > 0) {
                        if (d > 0) {
                            if (e > 0) {
                                System.out.println(1);
                            } else {
                                System.out.println(2);
                            }
                        }
                    }
                }
            }
        }
    }
}
"""
        (src / "HighComplexity.java").write_text(code, encoding="utf-8")
        sha = "3" * 40
        run = runner.run_analysis(
            commit_sha=sha,
            source_root=tmp_path,
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert run.status == "findings"
        assert run.pmd.status == "findings"
        assert run.pmd.exit_code == 4
        assert len(run.pmd.report.violations) >= 1
        rule_names = [v.rule for v in run.pmd.report.violations]
        assert "CognitiveComplexity" in rule_names

    def test_ruleset_resolution_with_relative_repo_path(self, runner: PmdRunner, tmp_path: Path) -> None:
        run = runner.run_analysis(
            commit_sha="f" * 40,
            source_root=tmp_path,
            ruleset_resource="src/review_bot/resources/pmd/ruleset.xml",
        )
        assert run.status == "no_sources"
        assert run.pmd.status == "no_sources"
