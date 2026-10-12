"""Fixed B material, complete prepared scope, and anti-forgery validation."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace
from unittest.mock import Mock

import pytest
import yaml

from review_bot.application.context import hash_json
from review_bot.application.intent_context import (
    IntentInputError, build_intent_evidence, build_intent_task_input, decode_intent_task_input,
    collect_intent_code_snapshot, evidence_digest, intent_scope, strict_json_loads,
)
from review_bot.application.review_a import MockModelClient
from review_bot.application.review_b import run_review_b_pipeline
from review_bot.domain.envelope_v2 import ChangedFile, EnvelopeEvidence
from review_bot.domain.models_v1 import CodeFile, SourceExclusion
from review_bot.config import ConfigValidationError, load_review_config_from_str
from review_bot.ports import RepositoryReader
from tests.fixtures.review_b import BASE_SHA, CAPTURED_AT, HEAD_SHA, digest, make_task, responses, stage_bodies


def wire(task):
    return json.loads(json.dumps(asdict(task)))


def rehash(data):
    data["input_hash"] = hash_json({key: value for key, value in data.items() if key != "input_hash"})
    return data


def test_wire_roundtrip_and_hash_include_text_code_missing_sources_and_scope() -> None:
    task = make_task()
    assert decode_intent_task_input(json.dumps(asdict(task))) == task
    assert make_task(body="A different stated outcome").input_hash != task.input_hash
    assert make_task(unavailable=True).input_hash != task.input_hash
    content = task.head.files[0].content + "# different implementation\n"
    head = replace(task.head, files=(replace(task.head.files[0], content=content, content_hash=digest(content)), task.head.files[1]), inventory_hash="")
    changed = build_intent_task_input(task.snapshot, task.base, head, related_paths=(task.head.files[1].path,))
    assert changed.input_hash != task.input_hash


def test_full_files_are_preserved_but_unrequested_repository_files_are_not_prepared() -> None:
    task = make_task()
    unrelated = CodeFile("elsewhere/unreviewed.py", "outside scope\n", digest("outside scope\n"), "source")
    head = replace(task.head, files=(*task.head.files, unrelated), inventory_hash="")
    scoped = build_intent_task_input(task.snapshot, task.base, head, related_paths=(task.head.files[1].path,))
    assert scoped.head.files == task.head.files
    assert intent_scope(scoped)["repository_wide"] is False
    original = {file.path: file.content for file in task.head.files}
    supplied = {entry.locator["path"]: entry.content for entry in build_intent_evidence(scoped) if entry.locator.get("version") == "head"}
    assert supplied == original


def test_unrelated_repository_completeness_is_not_the_b_scope_gate() -> None:
    task = make_task()
    base = replace(task.base, complete=False, exclusions=(SourceExclusion("elsewhere/binary.bin", "unsupported_encoding"),),
                   errors=("elsewhere/binary.bin: unsupported encoding",))
    scoped = build_intent_task_input(task.snapshot, base, task.head, related_paths=(task.head.files[1].path,))
    assert intent_scope(scoped)["complete"] is True
    assert scoped.base.complete is True


@pytest.mark.parametrize("missing", ["changed_head", "related", "excluded", "unassured"])
def test_required_missing_material_blocks_all_model_calls(missing: str) -> None:
    task = make_task()
    base, head = task.base, task.head
    related = (task.head.files[1].path,)
    if missing == "changed_head":
        head = replace(head, files=(head.files[1],), inventory_hash="")
    elif missing == "related":
        related += ("tests/missing.py",)
    elif missing == "excluded":
        head = replace(head, files=(head.files[0],), exclusions=(SourceExclusion(head.files[1].path, "unsupported_encoding"),), inventory_hash="")
    else:
        base = replace(base, complete=False)
    scoped = build_intent_task_input(task.snapshot, base, head, related_paths=related)
    client = MockModelClient(responses(), input_tokens=100)
    result = run_review_b_pipeline(scoped, client)
    assert result.reason == "incomplete_code_input"
    assert result.scope["missing"]
    assert client.requests == []


def test_rename_and_removed_files_keep_original_base_and_head_locations() -> None:
    task = make_task()
    snapshot = replace(task.snapshot, changed_files=(ChangedFile("src/new.py", "src/request.py", "renamed"),))
    head = replace(task.head, files=(replace(task.head.files[0], path="src/new.py"),), inventory_hash="")
    renamed = build_intent_task_input(snapshot, task.base, head)
    assert renamed.reviewed_paths == ("src/new.py", "src/request.py")
    assert intent_scope(renamed)["complete"] is True
    snapshot = replace(task.snapshot, changed_files=(ChangedFile("src/request.py", change_type="deleted"),))
    deleted = build_intent_task_input(snapshot, task.base, replace(task.head, files=(), inventory_hash=""))
    assert intent_scope(deleted)["complete"] is True
    assert [item.locator["version"] for item in build_intent_evidence(deleted) if item.kind == "source"] == ["base"]


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside", "x/../outside", "~/secret", "C:/secret", "x\\secret", ".git/config", "x//y"])
def test_related_paths_cannot_be_host_reserved_or_traversal_paths(path: str) -> None:
    task = make_task()
    with pytest.raises(IntentInputError):
        build_intent_task_input(task.snapshot, task.base, task.head, related_paths=(path,))


@pytest.mark.parametrize("path", [".env", "config/.env.production", "keys/private.pem", "node_modules/lib/index.js", "build/output.py", "src/.junie/settings"])
def test_operator_cannot_request_trusted_excluded_context(valid_config_yaml, path):
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"]["available_modes"] = ["intent"]
    raw["intent"] = {"context_paths": [path]}
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(yaml.safe_dump(raw))
    task = make_task()
    with pytest.raises(IntentInputError):
        build_intent_task_input(task.snapshot, task.base, task.head, related_paths=(path,))


def test_duplicate_context_paths_are_rejected_and_default_is_empty(valid_config_yaml):
    raw = yaml.safe_load(valid_config_yaml)
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    assert config.intent.context_paths == [] and config.execution.available_modes == ["architecture"]
    raw["intent"] = {"context_paths": ["tests/app.py", "tests/app.py"]}
    with pytest.raises(ConfigValidationError):
        load_review_config_from_str(yaml.safe_dump(raw))


def test_scoped_reader_keeps_full_regular_text_and_records_exclusions():
    reader = Mock(spec=RepositoryReader)
    content = "def execute():\n    return 'whole source'\n"
    reader.read_tree_blobs.return_value = {"app.py": content.encode(), "tests/test_app.py": b"assert True\n", "data.bin": b"\xff\x00"}
    paths = ("tests/test_app.py", "app.py", "data.bin", ".env")
    code = collect_intent_code_snapshot(reader, HEAD_SHA, paths)
    reader.read_tree_blobs.assert_called_once_with(HEAD_SHA, paths=("app.py", "data.bin", "tests/test_app.py"))
    assert code.files[0].content == content and code.files[0].content_hash == digest(content)
    assert [(file.path, file.kind) for file in code.files] == [("app.py", "source"), ("tests/test_app.py", "test")]
    assert {(item.path, item.reason) for item in code.exclusions} == {(".env", "trusted_exclusion"), ("data.bin", "unsupported_text")}
    assert code.complete is False


def test_scoped_reader_rejects_unrequested_blobs_instead_of_quietly_reviewing_them():
    reader = Mock(spec=RepositoryReader)
    reader.read_tree_blobs.return_value = {"outside.py": b"outside\n"}
    with pytest.raises(IntentInputError, match="invalid_scoped_blobs"):
        collect_intent_code_snapshot(reader, HEAD_SHA, ("app.py",))


def test_material_digest_covers_scoped_files_text_versions_and_missing_sources_but_not_read_time():
    from review_bot.application.snapshot import collect_snapshot_material
    from review_bot.domain.envelope_v2 import UnavailableSource
    from review_bot.ports import MergeBaseResult

    task = make_task()
    reader = Mock(spec=RepositoryReader)
    reader.resolve_merge_base.return_value = MergeBaseResult(BASE_SHA, BASE_SHA, "comparable")
    reader.get_changed_files.return_value = task.snapshot.changed_files
    reader.get_rename_map.return_value = {}
    reader.read_tree_blobs.side_effect = lambda sha, *, paths: {
        file.path: file.content.encode() for code in (task.base, task.head) if code.sha == sha for file in code.files if file.path in paths}
    original = task.snapshot.evidence[0]
    first = replace(original, locator={**original.locator, "retrieved_at": CAPTURED_AT})
    second = replace(original, locator={**original.locator, "retrieved_at": "2026-10-07T00:00:00Z"})

    def prepare(evidence=first, missing=(), related=()):
        return collect_snapshot_material(reader, BASE_SHA, HEAD_SHA, evidence=(evidence,),
            unavailable_sources=missing, related_paths=related)

    baseline = prepare()
    assert prepare(second).context_digest == baseline.context_digest
    version = replace(first, locator={**first.locator, "updated_at": "2026-10-06T00:00:00Z"})
    assert prepare(version).context_digest != baseline.context_digest
    assert prepare(missing=(UnavailableSource("issue:12", "forbidden"),)).context_digest != baseline.context_digest
    assert prepare(related=(task.head.files[1].path,)).context_digest != baseline.context_digest
    reader.read_tree_blobs.side_effect = lambda sha, *, paths: {path: b"new whole content\n" for path in paths}
    assert prepare().context_digest != baseline.context_digest


@pytest.mark.parametrize("forgery", ["sha", "path", "line", "boolean_line", "quote", "version", "duplicate_evidence", "extra_field", "cross_repo_issue"])
def test_recomputed_outer_hash_does_not_legitimize_invalid_evidence(forgery: str) -> None:
    task = make_task()
    data = wire(task)
    original = next(item for item in build_intent_evidence(task) if item.kind == "source" and item.locator["version"] == "head")
    supplied = asdict(original)
    data["snapshot"]["evidence"].append(supplied)
    if forgery == "sha":
        supplied["locator"]["sha"] = BASE_SHA
    elif forgery == "path":
        supplied["locator"]["path"] = "elsewhere/hidden.py"
    elif forgery == "line":
        supplied["locator"]["end_line"] = 100
    elif forgery == "boolean_line":
        supplied["locator"]["begin_line"] = True
    elif forgery == "quote":
        supplied["content"] = "invented source"
        supplied["digest"] = digest(supplied["content"])
    elif forgery == "version":
        data["snapshot"]["evidence"][0]["version"] = "invented-version"
    elif forgery == "duplicate_evidence":
        data["snapshot"]["evidence"].append(deepcopy(supplied))
    elif forgery == "extra_field":
        supplied["url"] = "https://untrusted.invalid"
    elif forgery == "cross_repo_issue":
        supplied.update(id="E-ISSUE-1", kind="issue_text", version="e" * 64,
            locator={"repository": "foreign/repo", "issue_number": 3}, content="Foreign issue", digest=digest("Foreign issue"))
    with pytest.raises(IntentInputError):
        decode_intent_task_input(rehash(data))


def test_unavailable_issue_is_fixed_missing_context_not_new_model_evidence() -> None:
    task = make_task(unavailable=True)
    client = MockModelClient(responses(stage_bodies(unavailable=True)), input_tokens=100)
    result = run_review_b_pipeline(task, client)
    assert result.status == "completed"
    assert result.goals.unavailable_context == ("issue:12:forbidden",)
    assert "issue:12:forbidden" not in client.requests[1].user_content
    assert {item.kind for item in build_intent_evidence(task)} == {"pr_text", "source", "test_source"}


@pytest.mark.parametrize("historical", ["missing_version", "later_edit"])
def test_historical_text_is_not_represented_as_original_without_a_valid_time(historical: str) -> None:
    data = wire(make_task())
    data["snapshot"]["as_of"] = "2026-09-30T12:00:00+00:00"
    if historical == "later_edit":
        data["snapshot"]["evidence"][0]["locator"]["updated_at"] = CAPTURED_AT
    with pytest.raises(IntentInputError):
        decode_intent_task_input(rehash(data))


def test_source_is_never_promoted_to_test_run_and_supplied_run_integrity_is_checked() -> None:
    task = make_task()
    assert not any(item.kind == "test_run" for item in build_intent_evidence(task))
    record = {"run_id": "run-1", "sha": HEAD_SHA, "paths": [task.head.files[1].path],
        "command": ["pytest", "tests/test_request.py"], "exit_code": 1, "status": "completed",
        "started_at": "2026-10-01T11:58:00+00:00", "completed_at": "2026-10-01T11:59:00+00:00",
        "output": "assertion failed", "output_hash": digest("assertion failed")}
    item = EnvelopeEvidence("E-RUN-1", "test_run", HEAD_SHA,
        {"repository": task.snapshot.repository, "sha": HEAD_SHA, "run_id": "run-1"}, record, evidence_digest(record, "test_run"))
    with_run = build_intent_task_input(replace(task.snapshot, evidence=(*task.snapshot.evidence, item)), task.base, task.head,
        related_paths=(task.head.files[1].path,))
    assert intent_scope(with_run)["test_run_evidence_ids"] == ["E-RUN-1"]
    client = MockModelClient(responses(), input_tokens=100)
    result = run_review_b_pipeline(with_run, client)
    assert result.status == "completed"
    assert "supplied_test_runs_not_independently_attested" in result.limitations
    assert "E-RUN-1" not in client.requests[0].user_content
    assert "E-RUN-1" in client.requests[1].user_content
    tampered = wire(with_run)
    tampered["snapshot"]["evidence"][-1]["content"]["output"] = "fake success"
    tampered["snapshot"]["evidence"][-1]["digest"] = evidence_digest(tampered["snapshot"]["evidence"][-1]["content"], "test_run")
    with pytest.raises(IntentInputError):
        decode_intent_task_input(rehash(tampered))


@pytest.mark.parametrize("text", ["1e100000", "[" * 25 + "0" + "]" * 25])
def test_json_numbers_and_nesting_are_bounded(text: str) -> None:
    with pytest.raises(IntentInputError):
        strict_json_loads(text)


def test_file_evidence_lines_use_git_lf_lines_not_unicode_string_separators() -> None:
    task = make_task()
    content = "value = 'a\u2028b'\n"
    head = replace(task.head, files=(replace(task.head.files[0], content=content, content_hash=digest(content)),), inventory_hash="")
    fixed = build_intent_task_input(task.snapshot, task.base, head)
    evidence = next(item for item in build_intent_evidence(fixed) if item.locator.get("version") == "head")
    assert evidence.locator["begin_line"] == evidence.locator["end_line"] == 1


def test_unrequested_repository_material_is_not_part_of_the_b_input_byte_limit() -> None:
    from review_bot.application.intent_context import MAX_INPUT_BYTES

    task = make_task()
    content = "x" * (MAX_INPUT_BYTES + 1)
    unrelated = CodeFile("unreviewed/large.txt", content, digest(content), "source")
    head = replace(task.head, files=(*task.head.files, unrelated), inventory_hash="")
    scoped = build_intent_task_input(task.snapshot, task.base, head, related_paths=(task.head.files[1].path,))
    assert scoped == task