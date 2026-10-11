"""Empirical adversarial stress test suite for Milestone 4:
`src/review_bot/application/delta.py` (6-state Delta Engine).

Tests:
1. 6 Change States with Extreme Boundary Conditions
   (existing, new, worsened, improved, resolved, unknown)
2. Rename Tracking Edge Cases
   (circular renames, many-to-one collisions, case-only renames, nested paths, unsafe paths, CPD renames)
3. Structural Metric Direction and Polarity
   (GodClass TCC polarity inversion, WMC/ATFD metrics, mixed metric movements, rewritten code with identical scores)
4. Incomparable Conditions and Error Reporting
   (mismatched tool versions, rulesets, minimum tokens, classpaths, concordance failures, suppressed counts, no-sources)
"""
from __future__ import annotations

import hashlib

import pytest

from review_bot.application.delta import (
    _metrics,
    _rename_errors,
    build_delta,
    compute_delta,
    hash_json,
)
from review_bot.domain.models_v1 import (
    AnalysisRun,
    CpdDuplication,
    CpdOccurrence,
    ExecutionRecord,
    ParsedReport,
    PmdViolation,
    ToolAnalysis,
)


def make_test_run(
    sha: str,
    ruleset_hash: str = "a" * 64,
    classpath_hash: str = "b" * 64,
    source_hash: str = "c" * 64,
    pmd_version: str = "7.3.0",
    cpd_version: str = "7.3.0",
    minimum_tokens: int = 100,
    pmd_violations: tuple[PmdViolation, ...] = (),
    cpd_duplications: tuple[CpdDuplication, ...] = (),
    suppressed_count: int = 0,
    source_files: tuple[str, ...] = ("src/main/java/demo/Foo.java",),
    source_state: str = "complete",
    pmd_exit_code: int | None = None,
    cpd_exit_code: int | None = None,
    pmd_status: str | None = None,
    cpd_status: str | None = None,
    run_status: str | None = None,
    runtime_classpath_hashes: dict[str, str] | None = None,
) -> AnalysisRun:
    """Helper to build a structurally valid AnalysisRun for empirical testing."""
    now = "2026-10-07T00:00:00+00:00"
    end = "2026-10-07T00:01:00+00:00"
    report_hash = hashlib.sha256(b"pmd_report").hexdigest()
    cpd_report_hash = hashlib.sha256(b"cpd_report").hexdigest()

    if runtime_classpath_hashes is None:
        runtime_classpath_hashes = {"pmd": classpath_hash, "cpd": classpath_hash}

    conds = {
        "schema_version": 1,
        "pmd_version": pmd_version,
        "cpd_version": cpd_version,
        "ruleset_hash": ruleset_hash,
        "minimum_tokens": minimum_tokens,
        "source_roots": ["src/main/java"],
        "language": "java",
        "language_version": "21",
        "encoding": "UTF-8",
        "type_resolution": "auxclasspath",
        "java_version": "21",
        "java_vendor": "Eclipse Adoptium",
        "classpath_hash": classpath_hash,
        "runtime_classpath_hashes": runtime_classpath_hashes,
    }
    profile = {k: v for k, v in conds.items() if k != "runtime_classpath_hashes"}
    conds_hash = hash_json(conds)
    prof_hash = hash_json(profile)

    actual_pmd_exit = (
        pmd_exit_code if pmd_exit_code is not None else (4 if pmd_violations else 0)
    )
    actual_cpd_exit = (
        cpd_exit_code if cpd_exit_code is not None else (4 if cpd_duplications else 0)
    )

    actual_pmd_status = (
        pmd_status
        if pmd_status is not None
        else ("findings" if pmd_violations else "clean")
    )
    actual_cpd_status = (
        cpd_status
        if cpd_status is not None
        else ("findings" if cpd_duplications else "clean")
    )

    if source_state == "no_java_sources":
        actual_pmd_exit = None
        actual_cpd_exit = None
        actual_pmd_status = "no_sources"
        actual_cpd_status = "no_sources"
        source_files = ()
        source_hash = hashlib.sha256(b"").hexdigest()

    pmd_exec = ExecutionRecord(
        schema_version=1,
        tool="pmd",
        exit_code=actual_pmd_exit,
        tool_version=pmd_version,
        java_version="21",
        java_vendor="Eclipse Adoptium",
        classpath_hash=runtime_classpath_hashes.get("pmd", classpath_hash),
        profile_hash=prof_hash,
        sha=sha,
        run_id="run-1",
        source_hash=source_hash,
        source_root="output/sources",
        source_roots=("src/main/java",),
        source_count=len(source_files),
        ruleset_hash=ruleset_hash,
        minimum_tokens=None,
        language_version="21",
        encoding="UTF-8",
        type_resolution="auxclasspath",
        started_at=now,
        completed_at=end,
        report_exists=source_state != "no_java_sources",
        report_hash=report_hash if source_state != "no_java_sources" else None,
        command=(
            "java",
            "-cp",
            "...",
            "net.sourceforge.pmd.cli.PmdCli",
            "check",
            "--rulesets",
            "...",
        ),
        state="not_run" if source_state == "no_java_sources" else "completed",
    )
    pmd_rep = ParsedReport(
        tool="pmd",
        path="output/pmd.xml",
        exists=source_state != "no_java_sources",
        parse_status="valid" if source_state != "no_java_sources" else "missing",
        tool_version=pmd_version if source_state != "no_java_sources" else None,
        content_hash=report_hash if source_state != "no_java_sources" else None,
        violations=pmd_violations,
        suppressed_count=suppressed_count,
    )
    pmd_analysis = ToolAnalysis(
        tool="pmd",
        status=actual_pmd_status,
        exit_code=actual_pmd_exit,
        report=pmd_rep,
        execution=pmd_exec,
    )

    cpd_exec = ExecutionRecord(
        schema_version=1,
        tool="cpd",
        exit_code=actual_cpd_exit,
        tool_version=cpd_version,
        java_version="21",
        java_vendor="Eclipse Adoptium",
        classpath_hash=runtime_classpath_hashes.get("cpd", classpath_hash),
        profile_hash=prof_hash,
        sha=sha,
        run_id="run-1",
        source_hash=source_hash,
        source_root="output/sources",
        source_roots=("src/main/java",),
        source_count=len(source_files),
        ruleset_hash=None,
        minimum_tokens=minimum_tokens,
        language_version="21",
        encoding="UTF-8",
        type_resolution="auxclasspath",
        started_at=now,
        completed_at=end,
        report_exists=source_state != "no_java_sources",
        report_hash=cpd_report_hash if source_state != "no_java_sources" else None,
        command=(
            "java",
            "-cp",
            "...",
            "net.sourceforge.pmd.cli.PmdCli",
            "cpd",
            "--minimum-tokens",
            str(minimum_tokens),
        ),
        state="not_run" if source_state == "no_java_sources" else "completed",
    )
    cpd_rep = ParsedReport(
        tool="cpd",
        path="output/cpd.xml",
        exists=source_state != "no_java_sources",
        parse_status="valid" if source_state != "no_java_sources" else "missing",
        tool_version=cpd_version if source_state != "no_java_sources" else None,
        content_hash=cpd_report_hash if source_state != "no_java_sources" else None,
        duplications=cpd_duplications,
        suppressed_count=0,
    )
    cpd_analysis = ToolAnalysis(
        tool="cpd",
        status=actual_cpd_status,
        exit_code=actual_cpd_exit,
        report=cpd_rep,
        execution=cpd_exec,
    )

    if run_status is None:
        if source_state == "no_java_sources":
            run_status = "no_sources"
        elif "findings" in (actual_pmd_status, actual_cpd_status):
            run_status = "findings"
        else:
            run_status = "clean"

    return AnalysisRun(
        schema_version=1,
        run_id="run-1",
        sha=sha,
        profile_hash=prof_hash,
        conditions_hash=conds_hash,
        conditions=conds,
        output_dir="output",
        started_at=now,
        completed_at=end,
        status=run_status,
        source_state=source_state,
        source_files=tuple(source_files),
        source_hash=source_hash,
        exclusions=(),
        pmd=pmd_analysis,
        cpd=cpd_analysis,
        build_exit_code=0,
    )


def make_pmd_violation(
    path: str = "src/main/java/demo/Foo.java",
    begin_line: int = 10,
    end_line: int = 25,
    rule: str = "CognitiveComplexity",
    ruleset: str = "Design",
    priority: int = 3,
    klass: str = "Foo",
    method: str = "calculate",
    package: str = "demo",
    message: str = "The method 'calculate' has a cognitive complexity of 20, current threshold is 15.",
    code_fragment: str = "int calculate(int x) { if (x > 0) return x; return -x; }",
) -> PmdViolation:
    return PmdViolation(
        path=path,
        raw_path=path,
        begin_line=begin_line,
        end_line=end_line,
        begin_column=1,
        end_column=50,
        rule=rule,
        ruleset=ruleset,
        priority=priority,
        klass=klass,
        method=method,
        package=package,
        message=message,
        code_fragment=code_fragment,
    )


def make_cpd_duplication(
    lines: int = 15,
    tokens: int = 120,
    occurrences: tuple[tuple[str, int, int], ...] = (
        ("src/main/java/demo/Foo.java", 10, 25),
        ("src/main/java/demo/Bar.java", 15, 30),
    ),
    code_fragment: str = "for (int i = 0; i < n; i++) { sum += items.get(i); }",
) -> CpdDuplication:
    occs = tuple(
        CpdOccurrence(
            path=path,
            raw_path=path,
            begin_line=b_line,
            end_line=e_line,
            begin_column=1,
            end_column=50,
            end_line_inferred=False,
            code_fragment=code_fragment,
        )
        for path, b_line, e_line in occurrences
    )
    return CpdDuplication(
        lines=lines,
        tokens=tokens,
        occurrences=occs,
        code_fragment=code_fragment,
    )


# ==============================================================================
# Dimension 1: 6 Change States with Extreme Boundary Conditions
# ==============================================================================


class Test6ChangeStatesExtremeBoundaries:
    def test_existing_finding_with_line_number_drift(self) -> None:
        """Moving lines due to edits above must match as 'existing' (lines != identity)."""
        v_base = make_pmd_violation(begin_line=10, end_line=20)
        v_head = make_pmd_violation(begin_line=250, end_line=260)  # Drifted 240 lines down
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert delta.comparable is True
        assert len(delta.signals) == 1
        signal = delta.signals[0]
        assert signal.change == "existing"
        assert "rename-canonical path and package/class/method with exact normalized code" in signal.matching

    def test_existing_finding_with_comments_and_whitespace_variance(self) -> None:
        """Java comments (// and /* */) and whitespace variance in code must still match as existing."""
        code_base = "int calculate(int x) {\n    // single line comment\n    if (x > 0) return x;\n    return -x;\n}"
        code_head = "int calculate(int x) {\n    /* multi\n       line */\n  if (x > 0)   return x;\n return -x;\n}"
        v_base = make_pmd_violation(code_fragment=code_base)
        v_head = make_pmd_violation(code_fragment=code_head)
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "existing"

    def test_existing_cpd_duplication(self) -> None:
        """Identical CPD duplicate block across base and head must match as existing."""
        d_base = make_cpd_duplication()
        d_head = make_cpd_duplication()
        sources = ("src/main/java/demo/Foo.java", "src/main/java/demo/Bar.java")
        run_base = make_test_run("a" * 40, cpd_duplications=(d_base,), source_files=sources)
        run_head = make_test_run("b" * 40, cpd_duplications=(d_head,), source_files=sources)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "existing"
        assert delta.signals[0].tool == "cpd"

    def test_new_finding_in_head(self) -> None:
        """Finding present in head only with valid code fragment must emit 'new'."""
        v_head = make_pmd_violation()
        run_base = make_test_run("a" * 40, pmd_violations=())
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "new"

    def test_new_cpd_duplication(self) -> None:
        """CPD duplication introduced in head only must emit 'new'."""
        d_head = make_cpd_duplication()
        sources = ("src/main/java/demo/Foo.java", "src/main/java/demo/Bar.java")
        run_base = make_test_run("a" * 40, source_files=sources)
        run_head = make_test_run("b" * 40, cpd_duplications=(d_head,), source_files=sources)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "new"
        assert delta.signals[0].tool == "cpd"

    def test_new_finding_with_empty_or_unparseable_code_fragment(self) -> None:
        """Unmatched finding without valid code tokens cannot be verified as 'new'; becomes 'unknown'."""
        v_head = make_pmd_violation(code_fragment="")  # Empty code fragment
        run_base = make_test_run("a" * 40)
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "unknown"
        assert "lacks unique code evidence" in delta.signals[0].matching

    def test_resolved_finding_in_base(self) -> None:
        """Finding present in base and absent in head must emit 'resolved'."""
        v_base = make_pmd_violation()
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "resolved"

    def test_resolved_cpd_duplication(self) -> None:
        """CPD duplication eliminated in head must emit 'resolved'."""
        d_base = make_cpd_duplication()
        sources = ("src/main/java/demo/Foo.java", "src/main/java/demo/Bar.java")
        run_base = make_test_run("a" * 40, cpd_duplications=(d_base,), source_files=sources)
        run_head = make_test_run("b" * 40, source_files=sources)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "resolved"

    def test_resolved_finding_with_empty_code_fragment(self) -> None:
        """Unmatched finding without code fragment in base cannot be claimed as resolved; becomes 'unknown'."""
        v_base = make_pmd_violation(code_fragment="")
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "unknown"

    def test_unmatched_finding_with_suppressed_count_in_report(self) -> None:
        """When PMD report has suppressed violations, unmatched observations cannot establish resolved/new."""
        v_base = make_pmd_violation()
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,), suppressed_count=2)
        run_head = make_test_run("b" * 40, suppressed_count=0)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "unknown"
        assert any("Suppressed observations" in lim for lim in delta.signals[0].limitations)

    def test_worsened_cognitive_complexity_increase(self) -> None:
        """CognitiveComplexity increase (18 -> 32) must emit 'worsened'."""
        v_base = make_pmd_violation(
            message="The method 'calculate' has a cognitive complexity of 18, current threshold is 15."
        )
        v_head = make_pmd_violation(
            message="The method 'calculate' has a cognitive complexity of 32, current threshold is 15."
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "worsened"

    def test_improved_cognitive_complexity_decrease(self) -> None:
        """CognitiveComplexity decrease (32 -> 18) must emit 'improved'."""
        v_base = make_pmd_violation(
            message="The method 'calculate' has a cognitive complexity of 32, current threshold is 15."
        )
        v_head = make_pmd_violation(
            message="The method 'calculate' has a cognitive complexity of 18, current threshold is 15."
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "improved"

    def test_worsened_cpd_occurrence_count_increase(self) -> None:
        """CPD duplication spreading to an additional file must emit 'worsened'."""
        sources = (
            "src/main/java/demo/Foo.java",
            "src/main/java/demo/Bar.java",
            "src/main/java/demo/Baz.java",
        )
        d_base = make_cpd_duplication(
            occurrences=(
                ("src/main/java/demo/Foo.java", 10, 25),
                ("src/main/java/demo/Bar.java", 15, 30),
            )
        )
        d_head = make_cpd_duplication(
            occurrences=(
                ("src/main/java/demo/Foo.java", 10, 25),
                ("src/main/java/demo/Bar.java", 15, 30),
                ("src/main/java/demo/Baz.java", 20, 35),
            )
        )
        run_base = make_test_run("a" * 40, cpd_duplications=(d_base,), source_files=sources)
        run_head = make_test_run("b" * 40, cpd_duplications=(d_head,), source_files=sources)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "worsened"

    def test_improved_cpd_occurrence_count_decrease(self) -> None:
        """CPD duplication occurrences reduced from 3 to 2 must emit 'improved'."""
        sources = (
            "src/main/java/demo/Foo.java",
            "src/main/java/demo/Bar.java",
            "src/main/java/demo/Baz.java",
        )
        d_base = make_cpd_duplication(
            occurrences=(
                ("src/main/java/demo/Foo.java", 10, 25),
                ("src/main/java/demo/Bar.java", 15, 30),
                ("src/main/java/demo/Baz.java", 20, 35),
            )
        )
        d_head = make_cpd_duplication(
            occurrences=(
                ("src/main/java/demo/Foo.java", 10, 25),
                ("src/main/java/demo/Bar.java", 15, 30),
            )
        )
        run_base = make_test_run("a" * 40, cpd_duplications=(d_base,), source_files=sources)
        run_head = make_test_run("b" * 40, cpd_duplications=(d_head,), source_files=sources)

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "improved"

    def test_unknown_for_ambiguous_candidate_component(self) -> None:
        """When multiple identical findings in different overloaded methods cannot be disambiguated, status is unknown."""
        v_base_1 = make_pmd_violation(begin_line=10, klass="Foo", method="calc")
        v_base_2 = make_pmd_violation(begin_line=40, klass="Foo", method="calc")
        v_head_1 = make_pmd_violation(begin_line=15, klass="Foo", method="calc")
        v_head_2 = make_pmd_violation(begin_line=45, klass="Foo", method="calc")

        run_base = make_test_run("a" * 40, pmd_violations=(v_base_1, v_base_2))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head_1, v_head_2))

        delta = build_delta(run_base, run_head)
        changes = [s.change for s in delta.signals]
        assert "unknown" in changes


# ==============================================================================
# Dimension 2: Rename Tracking Edge Cases
# ==============================================================================


class TestRenameTrackingAdversarial:
    def test_2_way_circular_rename_rejected(self) -> None:
        """Circular rename (A -> B, B -> A) must be rejected with 'rename map contains a cycle'."""
        sources = ("src/main/java/demo/A.java", "src/main/java/demo/B.java")
        run_base = make_test_run("a" * 40, source_files=sources)
        run_head = make_test_run("b" * 40, source_files=sources)
        cycle_map = {
            "src/main/java/demo/A.java": "src/main/java/demo/B.java",
            "src/main/java/demo/B.java": "src/main/java/demo/A.java",
        }

        delta = build_delta(run_base, run_head, rename_map=cycle_map)
        assert delta.comparable is False
        assert "rename map contains a cycle" in delta.limitations

    def test_3_way_circular_rename_rejected(self) -> None:
        """3-way circular rename (A -> B -> C -> A) must be rejected with 'rename map contains a cycle'."""
        cycle_map = {
            "src/main/java/demo/A.java": "src/main/java/demo/B.java",
            "src/main/java/demo/B.java": "src/main/java/demo/C.java",
            "src/main/java/demo/C.java": "src/main/java/demo/A.java",
        }
        errors = _rename_errors(cycle_map)
        assert errors == ("rename map contains a cycle",)

    def test_many_to_one_collision_attempt_rejected(self) -> None:
        """Many-to-one collision attempts (A -> C, B -> C) must be rejected as ambiguous."""
        collision_map = {
            "src/main/java/demo/A.java": "src/main/java/demo/C.java",
            "src/main/java/demo/B.java": "src/main/java/demo/C.java",
        }
        errors = _rename_errors(collision_map)
        assert errors == ("rename map has ambiguous many-to-one destinations",)

        sources = ("src/main/java/demo/A.java", "src/main/java/demo/B.java")
        run_base = make_test_run("a" * 40, source_files=sources)
        run_head = make_test_run("b" * 40, source_files=("src/main/java/demo/C.java",))
        delta = build_delta(run_base, run_head, rename_map=collision_map)
        assert delta.comparable is False
        assert "rename map has ambiguous many-to-one destinations" in delta.limitations

    def test_case_only_rename_handled_safely(self) -> None:
        """Case-only rename (FooBar.java -> Foobar.java) is a valid 1-to-1 mapping."""
        case_map = {
            "src/main/java/demo/FooBar.java": "src/main/java/demo/Foobar.java"
        }
        assert _rename_errors(case_map) == ()

        v_base = make_pmd_violation(path="src/main/java/demo/FooBar.java")
        v_head = make_pmd_violation(path="src/main/java/demo/Foobar.java")
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,), source_files=("src/main/java/demo/FooBar.java",))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,), source_files=("src/main/java/demo/Foobar.java",))

        delta = build_delta(run_base, run_head, rename_map=case_map)
        assert delta.comparable is True
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "existing"

    def test_nested_path_rename_tracks_as_existing(self) -> None:
        """File moved across deep directory hierarchy must match as existing with rename_map."""
        old_path = "src/main/java/com/example/legacy/OldService.java"
        new_path = "src/main/java/com/example/modern/service/NewService.java"
        rename_map = {old_path: new_path}
        assert _rename_errors(rename_map) == ()

        v_base = make_pmd_violation(path=old_path)
        v_head = make_pmd_violation(path=new_path)
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,), source_files=(old_path,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,), source_files=(new_path,))

        delta = build_delta(run_base, run_head, rename_map=rename_map)
        assert delta.comparable is True
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "existing"

    @pytest.mark.parametrize(
        "bad_map",
        [
            {"../outside.java": "src/main/java/demo/Foo.java"},
            {"src/main/java/demo/Foo.java": "../outside.java"},
            {"/absolute/Foo.java": "src/main/java/demo/Foo.java"},
            {"src/main/java/demo/Foo.java": "C:\\Windows\\Foo.java"},
            {"": "src/main/java/demo/Foo.java"},
            {"src/main/java/demo/Foo.java": ""},
            {"src/main/java/./Foo.java": "src/main/java/demo/Foo.java"},
        ],
    )
    def test_unsafe_paths_in_rename_map_rejected(self, bad_map: dict[str, str]) -> None:
        """Any unsafe, non-canonical, or escaping paths in rename_map must be rejected."""
        errors = _rename_errors(bad_map)
        assert errors == ("rename map contains unsafe paths",)

    def test_cpd_duplication_tracking_across_renamed_file(self) -> None:
        """CPD duplication where one constituent file was renamed must match as existing."""
        old_path = "src/main/java/demo/OldFile.java"
        new_path = "src/main/java/demo/NewFile.java"
        other_path = "src/main/java/demo/Other.java"

        d_base = make_cpd_duplication(
            occurrences=((old_path, 10, 25), (other_path, 10, 25))
        )
        d_head = make_cpd_duplication(
            occurrences=((new_path, 10, 25), (other_path, 10, 25))
        )
        sources_base = (old_path, other_path)
        sources_head = (new_path, other_path)

        run_base = make_test_run("a" * 40, cpd_duplications=(d_base,), source_files=sources_base)
        run_head = make_test_run("b" * 40, cpd_duplications=(d_head,), source_files=sources_head)

        delta = build_delta(run_base, run_head, rename_map={old_path: new_path})
        assert delta.comparable is True
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "existing"


# ==============================================================================
# Dimension 3: Structural Metric Direction Logic & Polarity Inversion
# ==============================================================================


class TestStructuralMetricDirectionLogic:
    def test_god_class_tcc_increase_is_improved(self) -> None:
        """TCC measures class cohesion: higher cohesion is IMPROVED (polarity inversion)."""
        v_base = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=20%)",
        )
        v_head = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=45%)",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "improved"

    def test_god_class_tcc_decrease_is_worsened(self) -> None:
        """TCC decrease means loss of cohesion: lower cohesion is WORSENED (polarity inversion)."""
        v_base = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=45%)",
        )
        v_head = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=20%)",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "worsened"

    def test_god_class_wmc_increase_is_worsened(self) -> None:
        """WMC measures Weighted Methods per Class: higher complexity is WORSENED."""
        v_base = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=30%)",
        )
        v_head = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=85, ATFD=5, TCC=30%)",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "worsened"

    def test_god_class_atfd_increase_is_worsened(self) -> None:
        """ATFD measures Access To Foreign Data: higher coupling is WORSENED."""
        v_base = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=30%)",
        )
        v_head = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=15, TCC=30%)",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "worsened"

    def test_god_class_mixed_tcc_improved_vs_wmc_worsened(self) -> None:
        """When TCC improves (cohesion up) but WMC worsens (complexity up), mixed movements yield 'unknown'."""
        v_base = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=50, ATFD=5, TCC=20%)",
        )
        v_head = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=90, ATFD=5, TCC=45%)",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        signal = delta.signals[0]
        assert signal.change == "unknown"
        assert any("mixed observed metrics" in lim for lim in signal.limitations)

    def test_god_class_both_tcc_and_wmc_improved(self) -> None:
        """When both TCC improves (cohesion up) and WMC improves (complexity down), status is 'improved'."""
        v_base = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=80, ATFD=10, TCC=20%)",
        )
        v_head = make_pmd_violation(
            rule="GodClass",
            message="Possible God Class (WMC=40, ATFD=5, TCC=40%)",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "improved"

    def test_identical_metric_scores_on_rewritten_code_with_substantive_overlap(self) -> None:
        """When code has substantive overlap (>0.6 ratio) but different tokens, and identical metric, change is 'unknown'."""
        code_base = (
            "void processData(List<String> items, Map<String, Integer> counts) {\n"
            "    for (String item : items) {\n"
            "        if (item != null && item.length() > 0) {\n"
            "            counts.put(item, counts.getOrDefault(item, 0) + 1);\n"
            "        }\n"
            "    }\n"
            "}"
        )
        code_head = (
            "void processData(List<String> items, Map<String, Integer> counts) {\n"
            "    for (String item : items) {\n"
            "        if (item != null && !item.isEmpty()) {\n"
            "            counts.put(item, counts.getOrDefault(item, 0) + 1);\n"
            "        }\n"
            "    }\n"
            "}"
        )
        v_base = make_pmd_violation(
            rule="CognitiveComplexity",
            message="The method 'processData' has a cognitive complexity of 16, current threshold is 15.",
            code_fragment=code_base,
            method="processData",
        )
        v_head = make_pmd_violation(
            rule="CognitiveComplexity",
            message="The method 'processData' has a cognitive complexity of 16, current threshold is 15.",
            code_fragment=code_head,
            method="processData",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        signal = delta.signals[0]
        assert signal.change == "unknown"
        assert "substantive code overlap" in signal.matching
        assert any("equal or mixed observed metrics" in lim for lim in signal.limitations)

    def test_identical_metric_scores_on_rewritten_code_insufficient_overlap(self) -> None:
        """When code is rewritten with insufficient overlap (<0.6 ratio), correspondence is not established -> 'unknown'."""
        code_base = "int calc(int x) { if (x > 0) return 1; else return 2; }"
        code_head = "int calc(int x) { return (x > 0) ? 1 : 2; }"
        v_base = make_pmd_violation(
            rule="CyclomaticComplexity",
            message="The method 'calc' has a cyclomatic complexity of 2.",
            code_fragment=code_base,
            method="calc",
        )
        v_head = make_pmd_violation(
            rule="CyclomaticComplexity",
            message="The method 'calc' has a cyclomatic complexity of 2.",
            code_fragment=code_head,
            method="calc",
        )
        run_base = make_test_run("a" * 40, pmd_violations=(v_base,))
        run_head = make_test_run("b" * 40, pmd_violations=(v_head,))

        delta = build_delta(run_base, run_head)
        assert len(delta.signals) == 1
        signal = delta.signals[0]
        assert signal.change == "unknown"
        assert "correspondence is not established" in signal.matching
        assert any("insufficient code correspondence" in lim for lim in signal.limitations)

    def test_tcc_parsing_formats(self) -> None:
        """Test percentage parsing (25%), decimal ratio (0.25), and percentage with decimals (25.5%)."""
        f1 = make_pmd_violation(rule="GodClass", message="Possible God Class (WMC=10, ATFD=2, TCC=25%)")
        f2 = make_pmd_violation(rule="GodClass", message="Possible God Class (WMC=10, ATFD=2, TCC=0.25)")
        f3 = make_pmd_violation(rule="GodClass", message="Possible God Class (WMC=10, ATFD=2, TCC=25.5%)")

        assert _metrics(f1)["tcc"] == pytest.approx(0.25)
        assert _metrics(f2)["tcc"] == pytest.approx(0.25)
        assert _metrics(f3)["tcc"] == pytest.approx(0.255)

    def test_invalid_tcc_percentage_handled_safely(self) -> None:
        """TCC values outside [0, 1] (e.g. 150%) must return empty metrics dict rather than crashing."""
        f_bad = make_pmd_violation(rule="GodClass", message="Possible God Class (WMC=10, ATFD=2, TCC=150%)")
        assert _metrics(f_bad) == {}


# ==============================================================================
# Dimension 4: Incomparable Conditions & Error Reporting
# ==============================================================================


class TestIncomparableConditionsAndErrorReporting:
    def test_pmd_ruleset_hash_mismatch(self) -> None:
        """Different PMD ruleset hashes between base and head invalidate comparability."""
        run_base = make_test_run("a" * 40, ruleset_hash="a" * 64)
        run_head = make_test_run("b" * 40, ruleset_hash="f" * 64)

        delta = build_delta(run_base, run_head)
        assert delta.comparable is False
        assert "pmd: analysis condition content differs between revisions" in delta.limitations

    def test_pmd_version_mismatch(self) -> None:
        """PMD version mismatch (7.3.0 vs 7.4.0) causes condition mismatch and unsupported version error."""
        run_base = make_test_run("a" * 40, pmd_version="7.3.0")
        run_head = make_test_run("b" * 40, pmd_version="7.4.0")

        delta = build_delta(run_base, run_head)
        assert delta.comparable is False
        assert any("pmd_version" in lim or "unsupported" in lim or "differs" in lim for lim in delta.limitations)

    def test_cpd_minimum_tokens_mismatch(self) -> None:
        """CPD token threshold mismatch (100 vs 150) causes condition mismatch."""
        run_base = make_test_run("a" * 40, minimum_tokens=100)
        run_head = make_test_run("b" * 40, minimum_tokens=150)

        delta = build_delta(run_base, run_head)
        assert delta.comparable is False
        assert "cpd: analysis condition content differs between revisions" in delta.limitations

    def test_runtime_classpath_hash_mismatch(self) -> None:
        """Runtime classpath mismatch causes tool condition check failure."""
        run_base = make_test_run("a" * 40, runtime_classpath_hashes={"pmd": "1" * 64, "cpd": "2" * 64})
        run_head = make_test_run("b" * 40, runtime_classpath_hashes={"pmd": "3" * 64, "cpd": "2" * 64})

        delta = build_delta(run_base, run_head)
        assert delta.comparable is False
        assert any("runtime classpath does not match" in lim or "differs" in lim for lim in delta.limitations)

    def test_exit_finding_concordance_failure_causes_incomparable(self) -> None:
        """Exit 0 with non-empty findings violates execution concordance and causes run error."""
        v = make_pmd_violation()
        # Adversarially claim exit_code=0 despite having violations
        run_base = make_test_run("a" * 40, pmd_violations=(v,), pmd_exit_code=0)
        run_head = make_test_run("b" * 40)

        delta = build_delta(run_base, run_head)
        assert delta.comparable is False
        assert any("execution exit and report finding count disagree" in lim for lim in delta.limitations)

    def test_zero_java_sources_state_handled_gracefully(self) -> None:
        """no_java_sources state produces clean status without crashing."""
        run_base = make_test_run("a" * 40, source_state="no_java_sources")
        run_head = make_test_run("b" * 40, source_state="no_java_sources")

        delta = build_delta(run_base, run_head)
        assert delta.comparable is True
        assert delta.analysis_status["base"] == "no_sources"
        assert delta.analysis_status["head"] == "no_sources"

    def test_compute_delta_comparable_false_flag(self) -> None:
        """Public compute_delta facade with comparable=False must produce unknown signals and limitations."""
        v = {"rule": "CognitiveComplexity", "path": "Foo.java", "metric": 20, "method": "calc"}
        delta = compute_delta(base_findings=[v], head_findings=[], comparable=False)
        assert delta.comparable is False
        assert len(delta.signals) == 1
        assert delta.signals[0].change == "unknown"
        assert delta.signals[0].matching == "Incomparable analysis conditions"
        assert "Incomparable analysis" in delta.limitations

    def test_compute_delta_delegation_to_build_delta(self) -> None:
        """compute_delta with base_run and head_run must delegate directly to build_delta."""
        run_base = make_test_run("a" * 40)
        run_head = make_test_run("b" * 40)
        delta = compute_delta(base_run=run_base, head_run=run_head)
        assert delta.comparable is True
        assert delta.base_sha == "a" * 40
        assert delta.head_sha == "b" * 40

    def test_compute_delta_findings_path_rename_error_bypass_behavior(self) -> None:
        """Empirically test the facade behavior where compute_delta with finding dicts does not reject cyclic renames."""
        # build_delta rejects cyclic renames:
        cycle_map = {"A.java": "B.java", "B.java": "A.java"}
        assert _rename_errors(cycle_map) == ("rename map contains a cycle",)
        run_base = make_test_run("a" * 40, source_files=("A.java", "B.java"))
        run_head = make_test_run("b" * 40, source_files=("A.java", "B.java"))
        delta_run = compute_delta(base_run=run_base, head_run=run_head, rename_map=cycle_map)
        assert delta_run.comparable is False
        assert "rename map contains a cycle" in delta_run.limitations
