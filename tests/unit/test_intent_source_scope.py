from copy import deepcopy

import pytest

from review_bot.application.intent_source_scope import (
    build_source_scope, partition_scoped_original, verify_scoped_partition,
)
from review_bot.application.source_partition import PartitionError, SourceOriginal, make_span
from review_bot.domain.envelope_v2 import ChangedFile


def source(identifier, text, version="head", path="src/value.py"):
    return SourceOriginal(identifier, text, "source", path, version, "a" * 40)


def selected(original, scope):
    return b"".join(original.raw[start:end] for start, end in scope["ranges"][original.evidence_id]).decode()


def test_changed_function_and_reference_chain_keep_exact_original_offsets():
    padding = "\n# unrelated padding\n" * 30
    base = source("base", "def changed():\n    return helper(0)\n" + padding +
        "def helper(value):\n    return value + 1\n" + padding +
        "def unrelated():\n    return 'UNRELATED_BODY'\n", "base")
    head = source("head", base.text.replace("helper(0)", "helper(1)"))
    scope = build_source_scope([base, head], [ChangedFile(head.path)])
    assert "return helper(1)" in selected(head, scope)
    assert "return value + 1" in selected(head, scope)
    assert "UNRELATED_BODY" not in selected(head, scope)
    spans = partition_scoped_original(head, lambda span: True, scope)
    verify_scoped_partition(head, spans, scope)
    for span in spans:
        assert span.source_hash == head.digest
        assert span.text.encode() == head.raw[span.start_byte:span.end_byte]
        assert span.begin_line == head.raw[:span.start_byte].count(b"\n") + 1
    with pytest.raises(PartitionError, match="scope_coverage"):
        verify_scoped_partition(head, spans[:-1], scope)


def test_dependency_depth_is_five_and_version_specific():
    gap = "\n# padding\n" * 30
    body = gap.join(f"def step{i}():\n    return step{i+1}()\n" for i in range(8))
    before = source("base", body, "base")
    after = source("head", body.replace("return step1()", "return step1() + 1"))
    scope = build_source_scope([before, after], [ChangedFile(after.path)])
    assert "def step5" in selected(after, scope)
    assert "def step6" not in selected(after, scope)


@pytest.mark.parametrize("path", ["README.md", "src/broken.py"])
def test_non_ast_fallback_keeps_changed_lines_without_copying_the_full_file(path):
    text = "unrelated text\n" * 80 + "old 값\n" + "other text\n" * 80
    before = source("base", text, "base", path)
    after = source("head", text.replace("old 값", "new 값"), path=path)
    scope = build_source_scope([before, after], [ChangedFile(path)])
    assert "new 값" in selected(after, scope)
    assert len(selected(after, scope).encode()) < len(after.raw) // 4
    assert scope["parser_status"]["head"] in {"unsupported_language", "syntax_error"}


def test_added_deleted_and_renamed_sources_preserve_required_versions():
    old = source("old", "def old():\n    return 0\n", "base", "old.py")
    new = source("new", "def old():\n    return 1\n", path="new.py")
    added = source("added", "class Added:\n    pass\n", path="added.py")
    deleted = source("deleted", "class Deleted:\n    pass\n", "base", "deleted.py")
    scope = build_source_scope([old, new, added, deleted], [
        ChangedFile("new.py", previous_path="old.py", change_type="renamed"),
        ChangedFile("added.py", change_type="added"), ChangedFile("deleted.py", change_type="deleted")])
    assert selected(added, scope) == added.text
    assert selected(deleted, scope) == deleted.text
    assert "return 0" in selected(old, scope) and "return 1" in selected(new, scope)


def test_java_type_reference_keeps_declaration_without_all_unrelated_methods():
    gap = "\n// padding\n" * 30
    text = "package demo;\nclass Value {\n int changed() { return helper(0); }\n" + gap + \
        "int helper(int x) { return x; }\n" + gap + "int unrelated() { return 123456; }\n}\n"
    before = source("base", text, "base", "Value.java")
    after = source("head", text.replace("helper(0)", "helper(1)"), path="Value.java")
    scope = build_source_scope([before, after], [ChangedFile("Value.java")])
    assert "class Value" in selected(after, scope)
    assert "int helper(int x)" in selected(after, scope)
    assert "return 123456" not in selected(after, scope)


def test_python_class_header_ends_at_a_valid_unicode_boundary():
    text = "class Value:\n    한글 = 1\n\n    def changed(self):\n        return self.한글\n"
    before = source("base", text, "base")
    after = source("head", text.replace("한글 = 1", "한글 = 2"))
    scope = build_source_scope([before, after], [ChangedFile(after.path)])
    spans = partition_scoped_original(after, lambda span: True, scope)
    verify_scoped_partition(after, spans, scope)
    assert "한글 = 2" in "".join(span.text for span in spans)


def test_stored_scope_cannot_be_changed_or_removed_to_claim_full_coverage():
    from review_bot.application.partition_review import run_partition_review, validate_partition_result
    from review_bot.domain.partition_review import PartitionReviewLimits
    from tests.fixtures.code_first_intent_review import make_task
    from tests.fixtures.partition_review import PartitionModelClient

    task = make_task(padding=50)
    result = run_partition_review(task, "intent", PartitionModelClient(), PartitionModelClient(group_all=True),
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=100))
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    tampered = deepcopy(result)
    identifier = next(key for key, value in tampered.partition["source_scope"]["ranges"].items() if value)
    tampered.partition["source_scope"]["ranges"][identifier][0][1] -= 1
    with pytest.raises(PartitionError, match="stored_source_scope_changed"):
        validate_partition_result(tampered, task)
    removed = deepcopy(result)
    removed.partition.pop("source_scope")
    with pytest.raises(PartitionError, match="stored_source_scope_missing"):
        validate_partition_result(removed, task)


def test_scope_verifier_rejects_substituted_original_and_overlapping_delivery():
    original = source("head", "a\nb\nc\n")
    scope = {"ranges": {"head": [[0, 6]]}}
    first = make_span(original, 0, 4, "test")
    last = make_span(original, 2, 6, "test")
    with pytest.raises(PartitionError, match="overlap"):
        verify_scoped_partition(original, (first, last), scope)


def test_shared_plan_review_and_fact_checks_never_expand_back_to_unrelated_file_bytes():
    from dataclasses import replace
    import hashlib
    from review_bot.application.intent_context import build_intent_task_input
    from review_bot.application.partition_review import run_partition_review, validate_partition_result
    from review_bot.domain.partition_review import PartitionReviewLimits
    from tests.fixtures.code_first_intent_review import make_task
    from tests.fixtures.partition_review import PartitionModelClient

    template = make_task()
    def extend(code):
        files = []
        for file in code.files:
            if file.path == "src/value.py":
                text = file.content + "\n# padding\n" * 80 + "def unrelated():\n    return 'UNDELIVERED_SENTINEL'\n"
                file = replace(file, content=text, content_hash=hashlib.sha256(text.encode()).hexdigest())
            files.append(file)
        inventory = hashlib.sha256(":".join(f"{f.path}:{f.content_hash}" for f in sorted(files, key=lambda f: f.path)).encode()).hexdigest()
        return replace(code, files=tuple(files), inventory_hash=inventory)
    task = build_intent_task_input(template.snapshot, extend(template.base), extend(template.head),
        related_paths=template.reviewed_paths)
    review, tev = PartitionModelClient(), PartitionModelClient(group_all=True)
    limits = PartitionReviewLimits(max_total_tokens=1000000, max_calls=100)
    plan = run_partition_review(task, "planning", review, tev, limits=limits, available_modes=("intent",))
    assert plan.status == "planned", plan.reason
    result = run_partition_review(task, "intent", review, tev, limits=limits, prepared_plan=plan)
    assert result.status == "completed", result.reason
    validate_partition_result(result, task)
    assert all("UNDELIVERED_SENTINEL" not in request.user_content for request in [*review.requests, *tev.requests])


def test_architecture_refits_full_signal_source_after_a_narrow_common_plan():
    from dataclasses import replace
    import json
    from review_bot.application.context import build_task_input
    from review_bot.application.delta import compute_delta
    from review_bot.application.intent_context import build_intent_task_input
    from review_bot.application.partition_review import run_partition_review, validate_partition_result
    from review_bot.domain.models_v1 import Classification
    from review_bot.domain.partition_review import PartitionReviewLimits
    from tests.fixtures.code_first_intent_review import make_task
    from tests.fixtures.partition_review import PartitionModelClient
    from tests.unit.test_context_and_architecture_review import _make_snapshot

    text = "class A {\n int run() { return 0; }\n" + "// padding\n" * 80 + "int other() { return 7; }\n}\n"
    material = _make_snapshot({"src/A.java": text}, {"src/A.java": text.replace("return 0", "return 1")})
    template = make_task()
    snapshot = replace(template.snapshot, target_base_sha=material.base.sha, comparison_base_sha=material.base.sha,
        head_sha=material.head.sha, changed_files=(ChangedFile("src/A.java"),))
    task = build_intent_task_input(snapshot, replace(material.base, inventory_hash=""), replace(material.head, inventory_hash=""))
    def both(payload, body):
        if payload["stage"] == "select_review_scope":
            body["choice"] = next(o["label"] for o in payload["options"] if o["key"] == "review:architecture,intent")
        return body
    tev, review = PartitionModelClient(group_all=True, transform=both), PartitionModelClient()
    limits = PartitionReviewLimits(max_total_tokens=1000000, max_calls=100)
    plan = run_partition_review(task, "planning", review, tev, limits=limits, available_modes=("architecture", "intent"))
    assert plan.status == "planned", plan.reason
    delta = compute_delta(base_findings=[], head_findings=[{"rule": "CognitiveComplexity", "path": "src/A.java", "metric": 20, "method": "run"}],
        base_sha=material.base.sha, head_sha=material.head.sha)
    architecture = build_task_input(material, delta, Classification("general"))
    result = run_partition_review(architecture, "architecture", review, tev, limits=limits, prepared_plan=plan)
    assert result.status == "completed", result.reason
    validate_partition_result(result, architecture)
    assert any("int other()" in request.user_content
        for request in review.requests if json.loads(request.user_content)["stage"] == "review_partition")


def test_one_hop_dependency_collection_and_unresolved_reference_boundaries():
    import hashlib
    from unittest.mock import Mock
    from review_bot.application.intent_source_scope import discover_one_hop_dependency_paths
    from review_bot.application.snapshot import collect_snapshot_material
    from review_bot.domain.models_v1 import CodeFile
    from review_bot.ports import MergeBaseResult, RepositoryReader

    base_service = (
        "package com.example.order;\n"
        "import com.example.payment.PaymentValidator;\n"
        "public class OrderService {\n"
        "    public boolean process(String id) {\n"
        "        return true;\n"
        "    }\n"
        "}\n"
    )
    head_service = (
        "package com.example.order;\n"
        "import com.example.payment.PaymentValidator;\n"
        "public class OrderService {\n"
        "    public boolean process(String id) {\n"
        "        OrderHelper.audit(id);\n"
        "        ExternalGateway.send(id);\n"
        "        return PaymentValidator.validate(id);\n"
        "    }\n"
        "}\n"
    )
    validator_code = (
        "package com.example.payment;\n"
        "public class PaymentValidator {\n"
        "    public static boolean validate(String id) { return id != null; }\n"
        "}\n"
    )
    helper_code = (
        "package com.example.order;\n"
        "public class OrderHelper {\n"
        "    public static void audit(String id) {}\n"
        "}\n"
    )
    svc_path = "src/main/java/com/example/order/OrderService.java"
    val_path = "src/main/java/com/example/payment/PaymentValidator.java"
    hlp_path = "src/main/java/com/example/order/OrderHelper.java"

    candidates = discover_one_hop_dependency_paths(
        (ChangedFile(svc_path),),
        (CodeFile(svc_path, base_service, hashlib.sha256(base_service.encode()).hexdigest(), "source"),),
        (CodeFile(svc_path, head_service, hashlib.sha256(head_service.encode()).hexdigest(), "source"),),
    )
    assert val_path in candidates
    assert hlp_path in candidates

    repo_blobs = {
        svc_path: head_service.encode(),
        val_path: validator_code.encode(),
        hlp_path: helper_code.encode(),
    }
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult("a" * 40, "a" * 40, "comparable")
    reader.get_changed_files.return_value = (ChangedFile(svc_path),)
    reader.get_rename_map.return_value = {}
    reader.read_tree_blobs.side_effect = lambda sha, *, paths: {
        p: (base_service.encode() if sha == "a" * 40 and p == svc_path else repo_blobs[p])
        for p in paths if p in repo_blobs
    }
    material = collect_snapshot_material(reader, "a" * 40, "b" * 40)
    head_paths = {f.path for f in material.head.files}
    assert {svc_path, val_path, hlp_path} <= head_paths

    originals = [
        SourceOriginal("S-base", base_service, "source", svc_path, "base", "a" * 40),
        SourceOriginal("S-head", head_service, "source", svc_path, "head", "b" * 40),
        SourceOriginal("S-val", validator_code, "context_source", val_path, "head", "b" * 40),
        SourceOriginal("S-hlp", helper_code, "context_source", hlp_path, "head", "b" * 40),
    ]
    scope = build_source_scope(originals, [ChangedFile(svc_path)])
    assert "validate(String id)" in selected(originals[2], scope)
    assert "audit(String id)" in selected(originals[3], scope)
    assert any(
        item["symbol"] == "ExternalGateway" and item["reason"] == "outside_fixed_corpus"
        for item in scope.get("unresolved_references", [])
    )
