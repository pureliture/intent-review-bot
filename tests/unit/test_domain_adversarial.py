"""Adversarial stress-test suite for domain models (src/review_bot/domain/).

Tests:
1. Frozen immutability across all fields (FrozenInstanceError on mutation & deletion)
2. Dynamic attribute injection defense
3. Mutable default argument leaks and collection isolation
4. Equality invariants (reflexivity, symmetry, value equivalence)
5. Hashability invariants (hashable models in sets/dicts vs unhashable dict-backed models)
6. Shallow immutability and nested mutable container behavior
7. Subclassing and inheritance invariants
8. Round-trip serialization (dataclasses.asdict and json.dumps)
"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, asdict, fields
from typing import Any

import pytest

from review_bot import domain


def build_domain_instances() -> dict[str, Any]:
    """Instantiate valid representative objects for every domain dataclass."""
    insts: dict[str, Any] = {}

    # Changes & Measurements
    insts["Measurement"] = domain.Measurement("metric", 1.0, 2.0)

    # Core v1 Models
    insts["AnalysisError"] = domain.AnalysisError("kind", "msg", "p/F.java", "detail")
    insts["PmdViolation"] = domain.PmdViolation(
        "p/F.java", "src/F.java", 1, 10, 1, 20, "RuleName", "ruleset", 1, "Cls", "m", "pkg", "msg"
    )
    insts["CpdOccurrence"] = domain.CpdOccurrence("p/F.java", "src/F.java", 1, 10)
    insts["CpdDuplication"] = domain.CpdDuplication(
        10, 100, (insts["CpdOccurrence"],), "code_fragment"
    )
    insts["ParsedReport"] = domain.ParsedReport("pmd", "report.xml", True, "valid")
    insts["ExecutionRecord"] = domain.ExecutionRecord(
        schema_version=1,
        tool="pmd",
        exit_code=0,
        tool_version="7.3.0",
        java_version="21",
        java_vendor="OpenJDK",
        classpath_hash="cp_hash",
        profile_hash="prof_hash",
        sha="a" * 40,
        run_id="run_1",
        source_hash="src_hash",
        source_root="src",
        source_roots=("src",),
        source_count=1,
        ruleset_hash="rs_hash",
        minimum_tokens=None,
        language_version="java21",
        encoding="utf-8",
        type_resolution="true",
        started_at="2026-10-06T12:00:00Z",
        completed_at="2026-10-06T12:00:10Z",
        report_exists=True,
        report_hash="rep_hash",
        command=("pmd", "check"),
    )
    insts["ToolAnalysis"] = domain.ToolAnalysis(
        "pmd", "clean", 0, insts["ParsedReport"], insts["ExecutionRecord"]
    )
    insts["SourceExclusion"] = domain.SourceExclusion("path/excluded", "reason")
    insts["AnalysisRun"] = domain.AnalysisRun(
        schema_version=1,
        run_id="run_1",
        sha="a" * 40,
        profile_hash="prof_hash",
        conditions_hash="cond_hash",
        conditions={"key": "val"},
        output_dir="/tmp/out",
        started_at="2026-10-06T12:00:00Z",
        completed_at="2026-10-06T12:00:10Z",
        status="clean",
        source_state="complete",
        source_files=("F.java",),
        source_hash="src_hash",
        exclusions=(),
        pmd=insts["ToolAnalysis"],
        cpd=insts["ToolAnalysis"],
        build_exit_code=0,
    )
    insts["DeltaSignal"] = domain.DeltaSignal("s1", "pmd", "new", (), (), ("F.java",), "exact")
    insts["AnalysisDelta"] = domain.AnalysisDelta(
        1, "a" * 40, "b" * 40, "cond_hash", True, (insts["DeltaSignal"],)
    )
    insts["CodeFile"] = domain.CodeFile("F.java", "class F {}", "hash", "source")
    insts["CodeSnapshot"] = domain.CodeSnapshot("a" * 40, (insts["CodeFile"],))
    insts["PrSnapshot"] = domain.PrSnapshot("org/repo", 1, "a" * 40, "b" * 40, "feature")
    insts["Snapshot"] = domain.Snapshot(
        insts["PrSnapshot"], insts["CodeSnapshot"], insts["CodeSnapshot"]
    )
    insts["Classification"] = domain.Classification("general")
    insts["Evidence"] = domain.Evidence(
        "e1", "code", "base", "a" * 40, "F.java", 1, 10, "class F {}"
    )
    insts["SelectedSignal"] = domain.SelectedSignal(insts["DeltaSignal"], "review", "reason")
    insts["TaskInput"] = domain.TaskInput(
        1,
        insts["Snapshot"],
        insts["AnalysisDelta"],
        insts["Classification"],
        (insts["SelectedSignal"],),
        (insts["Evidence"],),
        {"scope_key": "val"},
        (),
        "hash",
    )
    insts["Citation"] = domain.Citation("e1", "base", "a" * 40, "F.java", 1, 10, "class F {}")
    insts["ReviewFinding"] = domain.ReviewFinding(
        "s1",
        "concern",
        "new",
        "analysis_signal",
        "Title",
        "Reason",
        "Recommendation",
        ("e1",),
        (insts["Citation"],),
    )
    insts["ReviewQuestion"] = domain.ReviewQuestion("s1", "Why?", "Reason", ("e1",))
    insts["ReviewResult"] = domain.ReviewResult(
        1,
        "A",
        "run_key",
        "a" * 40,
        "b" * 40,
        "general",
        insts["PrSnapshot"],
        "completed",
        "ok",
        "Summary",
        (insts["ReviewFinding"],),
        (insts["ReviewQuestion"],),
        (),
        {},
        {},
        insts["AnalysisDelta"],
        {},
        {},
        "2026-10-06T12:00:00Z",
        "review-a/v1",
        "input_hash",
    )
    insts["AnalysisBundle"] = domain.AnalysisBundle(
        1, insts["PrSnapshot"], insts["AnalysisRun"], insts["AnalysisRun"], insts["TaskInput"]
    )
    insts["ModelRequest"] = domain.ModelRequest(
        1, "model_id", "sys", "usr", 30, 100, {"type": "object"}
    )
    insts["ModelResponse"] = domain.ModelResponse("output text")

    # Service Envelope v2 Records
    insts["ChangedFile"] = domain.ChangedFile("F.java")
    insts["UnavailableSource"] = domain.UnavailableSource("source", "reason")
    insts["EnvelopeEvidence"] = domain.EnvelopeEvidence(
        "e1", "pr_text", "v1", {"file": "F.java"}, "content", "digest"
    )
    insts["ReviewSnapshot"] = domain.ReviewSnapshot(
        2,
        "snap_1",
        "org/repo",
        1,
        10,
        20,
        "a" * 40,
        "b" * 40,
        "c" * 40,
        "comparable",
        "2026-10-06T12:00:00Z",
        "2026-10-06T12:00:00Z",
        "d1",
        "d2",
    )
    insts["ModeSelection"] = domain.ModeSelection(
        "sel_1", "snap_1", "all_available", "digest", ("architecture",), ("architecture",)
    )
    insts["PublicationReceipt"] = domain.PublicationReceipt(1, "succeeded", "fp")
    insts["ExecutionReceipt"] = domain.ExecutionReceipt(
        2, "architecture", "sel_1", "snap_1", 1, "fp", "completed"
    )
    insts["DeliveryRecord"] = domain.DeliveryRecord(
        "del_1", "pd", "2026-10-06T12:00:00Z", "opened", 1, 2, 3
    )
    insts["PrRequestRecord"] = domain.PrRequestRecord(1, 2, 3, 1)
    insts["RunRecord"] = domain.RunRecord(
        "run_1", 1, 2, 3, 1, "fp", "queued", "fetch"
    )

    return insts


def test_adversarial_frozen_immutability_all_fields() -> None:
    """Every field in every domain dataclass must reject reassignment and deletion."""
    instances = build_domain_instances()
    tested_fields = 0

    for inst in instances.values():
        for f in fields(inst):
            tested_fields += 1
            # Attempt 1: Reassignment
            with pytest.raises(FrozenInstanceError):
                setattr(inst, f.name, "MALICIOUS_MUTATION")

            # Attempt 2: Deletion
            with pytest.raises(FrozenInstanceError):
                delattr(inst, f.name)

    assert tested_fields >= 180, f"Expected >= 180 fields tested, got {tested_fields}"


def test_adversarial_dynamic_attribute_injection() -> None:
    """Attempting to attach arbitrary undeclared attributes must be blocked."""
    instances = build_domain_instances()

    for inst in instances.values():
        with pytest.raises((FrozenInstanceError, AttributeError)):
            inst.__injected_dynamic_attribute__ = "payload"


def test_adversarial_mutable_default_collection_isolation() -> None:
    """Dataclasses using default factories must NOT share mutable collections across instances."""
    # 1. PrSnapshot
    p1 = domain.PrSnapshot("repo", 1, "a" * 40, "b" * 40, "b1")
    p2 = domain.PrSnapshot("repo", 1, "a" * 40, "b" * 40, "b1")
    assert p1.rename_map is not p2.rename_map
    p1.rename_map["attack.java"] = "pwned.java"
    assert "attack.java" not in p2.rename_map

    # 2. AnalysisDelta
    d1 = domain.AnalysisDelta(1, "a" * 40, "b" * 40, "h", True, ())
    d2 = domain.AnalysisDelta(1, "a" * 40, "b" * 40, "h", True, ())
    assert d1.analysis_status is not d2.analysis_status
    d1.analysis_status["pmd"] = "corrupted"
    assert "pmd" not in d2.analysis_status

    # 3. ExecutionReceipt
    r1 = domain.ExecutionReceipt(2, "arch", "s1", "snap1", 1, "fp", "completed")
    r2 = domain.ExecutionReceipt(2, "arch", "s1", "snap1", 1, "fp", "completed")
    assert r1.coverage is not r2.coverage
    assert r1.model_delivery is not r2.model_delivery
    r1.coverage["tampered"] = True
    assert "tampered" not in r2.coverage

    # 4. ModelResponse
    m1 = domain.ModelResponse("resp1")
    m2 = domain.ModelResponse("resp2")
    assert m1.usage is not m2.usage
    m1.usage["tampered_tokens"] = 9999
    assert "tampered_tokens" not in m2.usage


def test_adversarial_equality_invariants() -> None:
    """Dataclasses must maintain value equality reflexivity and symmetry."""
    insts1 = build_domain_instances()
    insts2 = build_domain_instances()

    for name in insts1:
        i1 = insts1[name]
        i2 = insts2[name]
        assert i1 == i2, f"Value equality failed for {name}"
        assert i2 == i1, f"Symmetric equality failed for {name}"


def test_adversarial_hashability_invariants() -> None:
    """Verify hashable domain types can be members of sets and dict keys,

    and verify dict-backed models raise TypeError as expected by Python dataclass behavior.
    """
    instances = build_domain_instances()

    expected_unhashable = {
        "AnalysisDelta",
        "AnalysisRun",
        "AnalysisBundle",
        "EnvelopeEvidence",
        "ExecutionReceipt",
        "ModelRequest",
        "ModelResponse",
        "PrSnapshot",
        "ReviewResult",
        "Snapshot",
        "TaskInput",
    }

    hashable_count = 0
    unhashable_count = 0

    for name, inst in instances.items():
        if name in expected_unhashable:
            with pytest.raises(TypeError, match="unhashable type: 'dict'"):
                hash(inst)
            unhashable_count += 1
        else:
            h = hash(inst)
            assert isinstance(h, int)
            s = {inst}
            assert inst in s
            d = {inst: "val"}
            assert d[inst] == "val"
            hashable_count += 1

    assert hashable_count >= 25
    assert unhashable_count == len(expected_unhashable)


def test_adversarial_shallow_immutability_documented() -> None:
    """Empirically demonstrate shallow immutability: nested dicts mutate in-place

    and break value equality.
    """
    p1 = domain.PrSnapshot("repo", 1, "a" * 40, "b" * 40, "b1")
    p2 = domain.PrSnapshot("repo", 1, "a" * 40, "b" * 40, "b1")
    assert p1 == p2

    # In-place dictionary mutation succeeds because dict is mutable
    p1.rename_map["Old.java"] = "New.java"
    # Value equality breaks
    assert p1 != p2
    assert p1.rename_map == {"Old.java": "New.java"}


def test_adversarial_subclass_inheritance_frozen_enforcement() -> None:
    """Subclassing a frozen dataclass with a non-frozen dataclass must raise TypeError."""
    from dataclasses import dataclass

    with pytest.raises(TypeError, match="cannot inherit non-frozen dataclass from a frozen one"):

        @dataclass(frozen=False)
        class MutableChild(domain.PrSnapshot):
            extra: str = ""


def test_adversarial_serialization_round_trip() -> None:
    """All domain dataclasses must serialize cleanly via asdict() and json.dumps()."""
    instances = build_domain_instances()

    for name, inst in instances.items():
        d = asdict(inst)
        assert isinstance(d, dict), f"asdict({name}) did not return dict"
        # Must be valid JSON without custom serializers
        serialized = json.dumps(d)
        assert len(serialized) > 2
