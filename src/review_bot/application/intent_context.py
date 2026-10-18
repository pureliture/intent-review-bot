"""Fixed B input, bounded wire decoding, scope checks, and evidence indexing."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, fields, replace
from datetime import datetime
from typing import Any

from review_bot.application.context import hash_json
from review_bot.domain.envelope_v2 import ChangedFile, EnvelopeEvidence, ReviewSnapshot, UnavailableSource
from review_bot.domain.intent_models import IntentTaskInput
from review_bot.domain.models_v1 import CodeFile, CodeSnapshot, SourceExclusion
from review_bot.ports import RepositoryReader

MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_PATHS = 5000
TEXT_KINDS = frozenset({"pr_text", "issue_text"})
CODE_KINDS = frozenset({"source", "test_source", "test_run"})
_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CHANGES = frozenset({"added", "modified", "deleted", "removed", "renamed", "copied", "type_changed"})


class IntentInputError(ValueError):
    """Safe reason codes only; never include source text or exception values."""


def bounded_object(value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise IntentInputError("invalid_fields")
    return value


def bounded_text(value: Any, *, maximum: int = 4000, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value.strip()):
        raise IntentInputError("invalid_string")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise IntentInputError("invalid_string") from None
    return value


def bounded_array(value: Any, *, maximum: int = 20) -> list[Any]:
    if not isinstance(value, list) or len(value) > maximum:
        raise IntentInputError("invalid_array")
    return value


def evidence_id(value: Any) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise IntentInputError("invalid_id")
    return value


def safe_relative_path(value: Any) -> str:
    if (
        not isinstance(value, str) or not value or len(value) > 4096
        or value.startswith(("/", "~")) or any(char in value for char in ("\\", ":"))
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part in ("", ".", "..", ".git", ".junie") for part in value.split("/"))
    ):
        raise IntentInputError("invalid_path")
    return value


def intent_path_exclusion(path: str) -> str | None:
    """Trusted B-only exclusions, never repository-provided ignore instructions."""
    parts = path.lower().split("/")
    if any(part in {".git", ".junie", ".venv", "venv", "node_modules", "__pycache__", "dist", "build", "target"} for part in parts):
        return "trusted_exclusion"
    name = parts[-1]
    if name == ".env" or name.startswith(".env.") or name.endswith((".pem", ".key", ".p12", ".pfx", ".keystore")):
        return "trusted_exclusion"
    return None


def validate_intent_context_paths(paths: tuple[str, ...]) -> tuple[str, ...]:
    if len(paths) > MAX_PATHS:
        raise IntentInputError("context_path_limit_exceeded")
    checked = tuple(safe_relative_path(path) for path in paths)
    _unique(checked)
    if any(intent_path_exclusion(path) for path in checked):
        raise IntentInputError("excluded_context_path")
    return checked


def collect_intent_code_snapshot(
    reader: RepositoryReader, sha: str, paths: tuple[str, ...],
) -> CodeSnapshot:
    """Read only full regular Git blobs in the prepared scope; execute nothing."""
    fixed_sha(sha)
    paths = tuple(sorted({safe_relative_path(path) for path in paths}))
    if len(paths) > MAX_PATHS:
        raise IntentInputError("context_path_limit_exceeded")
    exclusions = [SourceExclusion(path, "trusted_exclusion") for path in paths if intent_path_exclusion(path)]
    allowed = tuple(path for path in paths if not intent_path_exclusion(path))
    blobs = reader.read_tree_blobs(sha, paths=allowed) if allowed else {}
    if not isinstance(blobs, dict) or any(path not in allowed or not isinstance(content, bytes) for path, content in blobs.items()):
        raise IntentInputError("invalid_scoped_blobs")
    files = []
    for path, content in sorted(blobs.items()):
        try:
            text = content.decode("utf-8")
            if b"\x00" in content:
                raise ValueError
        except (UnicodeError, ValueError):
            exclusions.append(SourceExclusion(path, "unsupported_text"))
            continue
        file = CodeFile(path, text, hashlib.sha256(content).hexdigest(), "source")
        files.append(replace(file, kind="test" if _test_file(file) else "source"))
    inventory = hashlib.sha256(":".join(f"{file.path}:{file.content_hash}" for file in files).encode()).hexdigest()
    return CodeSnapshot(sha, tuple(files), tuple(sorted(exclusions, key=lambda item: item.path)),
        inventory, not exclusions, ())


def fixed_sha(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise IntentInputError("invalid_sha")
    return value


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise IntentInputError("invalid_digest")
    return value


def _integer(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise IntentInputError("invalid_integer")
    return value


def _timestamp(value: Any) -> datetime:
    text = bounded_text(value, maximum=64)
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise IntentInputError("invalid_timestamp") from None
    if stamp.tzinfo is None:
        raise IntentInputError("invalid_timestamp")
    return stamp


def _unique(values: list[str] | tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise IntentInputError("duplicate_id_or_path")


def strict_json_loads(text: str, *, max_bytes: int = MAX_INPUT_BYTES) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise IntentInputError("duplicate_json_key")
            result[key] = value
        return result

    def constant(_):
        raise IntentInputError("nonfinite_json")

    try:
        if not isinstance(text, str) or len(text.encode("utf-8")) > max_bytes:
            raise IntentInputError("json_byte_limit")
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
        pending = [(value, 0)]
        count = 0
        while pending:
            item, depth = pending.pop()
            count += 1
            if depth > 24 or count > 100000:
                raise IntentInputError("json_complexity_limit")
            if isinstance(item, float) and not math.isfinite(item):
                raise IntentInputError("nonfinite_json")
            if isinstance(item, dict):
                pending.extend((child, depth + 1) for child in item.values())
            elif isinstance(item, list):
                pending.extend((child, depth + 1) for child in item)
        return value
    except (UnicodeError, RecursionError, json.JSONDecodeError):
        raise IntentInputError("invalid_json") from None


def evidence_digest(content: str | dict[str, object], kind: str) -> str:
    if kind in TEXT_KINDS and isinstance(content, dict):
        bounded_object(content, {"title", "body"})
        title = bounded_text(content["title"], maximum=MAX_INPUT_BYTES, empty=True)
        body = bounded_text(content["body"], maximum=MAX_INPUT_BYTES, empty=True)
        return hashlib.sha256(f"{title}\n{body}".encode("utf-8")).hexdigest()
    if isinstance(content, str):
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
    if isinstance(content, dict):
        return hash_json(content)
    raise IntentInputError("invalid_evidence_content")


def _record(cls, value: Any) -> dict[str, Any]:
    return bounded_object(value, {field.name for field in fields(cls)})


def _decode_code(value: Any, *, allow_empty_inventory: bool = False) -> CodeSnapshot:
    raw = _record(CodeSnapshot, value)
    fixed_sha(raw["sha"])
    files_list = []
    for entry in bounded_array(raw["files"], maximum=MAX_PATHS):
        item = _record(CodeFile, entry)
        safe_relative_path(item["path"])
        bounded_text(item["content"], maximum=MAX_INPUT_BYTES, empty=True)
        if item["kind"] not in ("source", "test", "build"):
            raise IntentInputError("invalid_file_kind")
        if _digest(item["content_hash"]) != evidence_digest(item["content"], "source"):
            raise IntentInputError("code_hash_mismatch")
        files_list.append(CodeFile(**item))
    _unique([file.path for file in files_list])
    exclusions = []
    for entry in bounded_array(raw["exclusions"], maximum=MAX_PATHS):
        item = _record(SourceExclusion, entry)
        safe_relative_path(item["path"])
        bounded_text(item["reason"], maximum=512)
        exclusions.append(SourceExclusion(**item))
    _unique([item.path for item in exclusions])
    if {file.path for file in files_list} & {item.path for item in exclusions}:
        raise IntentInputError("included_excluded_path")
    errors = tuple(bounded_text(item, maximum=4096) for item in bounded_array(raw["errors"], maximum=MAX_PATHS))
    if type(raw["complete"]) is not bool:
        raise IntentInputError("invalid_completeness")
    inventory = hashlib.sha256(":".join(f"{file.path}:{file.content_hash}" for file in sorted(files_list, key=lambda f: f.path)).encode()).hexdigest()
    if not (allow_empty_inventory and raw["inventory_hash"] == "") and _digest(raw["inventory_hash"]) != inventory:
        raise IntentInputError("inventory_hash_mismatch")
    return CodeSnapshot(raw["sha"], tuple(files_list), tuple(exclusions), inventory, raw["complete"], errors)


def _decode_snapshot(value: Any) -> ReviewSnapshot:
    raw = _record(ReviewSnapshot, value)
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 2:
        raise IntentInputError("unsupported_snapshot_version")
    repository = bounded_text(raw["repository"], maximum=201)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or any(part in (".", "..") for part in repository.split("/")):
        raise IntentInputError("invalid_repository")
    for name in ("pr_number", "repository_id", "installation_id"):
        _integer(raw[name], minimum=1)
    for name in ("snapshot_id", "pr_text_digest", "context_digest"):
        _digest(raw[name])
    for name in ("target_base_sha", "head_sha"):
        fixed_sha(raw[name])
    if raw["comparison_base_sha"] is not None:
        fixed_sha(raw["comparison_base_sha"])
    if raw["comparison_status"] not in ("comparable", "incomparable") or (raw["comparison_status"] == "comparable") != (raw["comparison_base_sha"] is not None):
        raise IntentInputError("invalid_comparison")
    _timestamp(raw["captured_at"])
    _timestamp(raw["as_of"])
    changes = []
    for entry in bounded_array(raw["changed_files"], maximum=MAX_PATHS):
        item = _record(ChangedFile, entry)
        safe_relative_path(item["path"])
        if item["previous_path"] is not None:
            safe_relative_path(item["previous_path"])
        if item["change_type"] not in _CHANGES or type(item["has_diff"]) is not bool:
            raise IntentInputError("invalid_change")
        if item["change_type"] in ("renamed", "copied") and not item["previous_path"]:
            raise IntentInputError("missing_previous_path")
        changes.append(ChangedFile(**item))
    _unique([item.path for item in changes])
    evidence = []
    for entry in bounded_array(raw["evidence"], maximum=MAX_PATHS * 2 + 128):
        item = _record(EnvelopeEvidence, entry)
        evidence_id(item["id"])
        if item["kind"] not in TEXT_KINDS | CODE_KINDS:
            raise IntentInputError("unsupported_evidence_kind")
        bounded_text(item["version"], maximum=256)
        if not isinstance(item["locator"], dict):
            raise IntentInputError("invalid_locator")
        if _digest(item["digest"]) != evidence_digest(item["content"], item["kind"]):
            raise IntentInputError("evidence_hash_mismatch")
        evidence.append(EnvelopeEvidence(**item))
    _unique([item.id for item in evidence])
    unavailable = []
    for entry in bounded_array(raw["unavailable_sources"], maximum=128):
        item = _record(UnavailableSource, entry)
        bounded_text(item["source"], maximum=256)
        bounded_text(item["reason"], maximum=512)
        bounded_text(f"{item['source']}:{item['reason']}", maximum=768)
        unavailable.append(UnavailableSource(**item))
    _unique([item.source for item in unavailable])
    return ReviewSnapshot(**{**raw, "changed_files": tuple(changes), "evidence": tuple(evidence), "unavailable_sources": tuple(unavailable)})


def _test_file(file: CodeFile) -> bool:
    path = file.path.lower()
    return file.kind == "test" or bool(set(path.split("/")[:-1]) & {"test", "tests", "__tests__"}) or path.rsplit("/", 1)[-1].startswith("test_") or path.endswith((".test.js", ".spec.js", ".test.ts", ".spec.ts"))


def _code_evidence(task: IntentTaskInput) -> tuple[EnvelopeEvidence, ...]:
    result = []
    for version in ("base", "head"):
        code = getattr(task, version)
        for index, file in enumerate(sorted(code.files, key=lambda f: f.path), 1):
            count = file.content.count("\n") + int(bool(file.content) and not file.content.endswith("\n"))
            result.append(EnvelopeEvidence(
                id=f"E-CODE-{version}-{index:04d}", kind="test_source" if _test_file(file) else "source",
                version=code.sha, locator={"repository": task.snapshot.repository, "version": version,
                    "sha": code.sha, "path": file.path, "begin_line": 1 if count else 0, "end_line": count},
                content=file.content, digest=file.content_hash,
            ))
    return tuple(result)


def _validate_text_evidence(task: IntentTaskInput, item: EnvelopeEvidence) -> None:
    locator = item.locator
    required = {"repository", "pr_number" if item.kind == "pr_text" else "issue_number"}
    if not required <= set(locator) or set(locator) - required - {"updated_at", "retrieved_at"}:
        raise IntentInputError("invalid_text_locator")
    if locator["repository"] != task.snapshot.repository:
        raise IntentInputError("cross_repository_evidence")
    key = "pr_number" if item.kind == "pr_text" else "issue_number"
    _integer(locator[key], minimum=1)
    if key == "pr_number" and locator[key] != task.snapshot.pr_number:
        raise IntentInputError("wrong_pr_evidence")
    if not isinstance(item.content, (str, dict)):
        raise IntentInputError("invalid_text_evidence")
    if isinstance(item.content, str):
        bounded_text(item.content, maximum=MAX_INPUT_BYTES, empty=True)
    if "updated_at" in locator and _timestamp(locator["updated_at"]) > _timestamp(task.snapshot.as_of):
        raise IntentInputError("text_after_as_of")
    if "retrieved_at" in locator:
        _timestamp(locator["retrieved_at"])
    if _timestamp(task.snapshot.as_of) < _timestamp(task.snapshot.captured_at) and "updated_at" not in locator:
        raise IntentInputError("historical_text_version_unverified")
    if item.kind == "pr_text" and item.digest != task.snapshot.pr_text_digest:
        raise IntentInputError("pr_text_hash_mismatch")
    versions = ("head", task.snapshot.head_sha, item.digest) if item.kind == "pr_text" else (item.digest, locator.get("updated_at"))
    if item.version not in versions:
        raise IntentInputError("invalid_text_version")


def _validate_test_run(task: IntentTaskInput, item: EnvelopeEvidence) -> None:
    raw = bounded_object(item.content, {"run_id", "sha", "paths", "command", "exit_code", "status", "started_at", "completed_at", "output", "output_hash"})
    bounded_object(item.locator, {"repository", "sha", "run_id"})
    evidence_id(raw["run_id"])
    if raw["sha"] not in (task.base.sha, task.head.sha) or item.version != raw["sha"] or item.locator != {"repository": task.snapshot.repository, "sha": raw["sha"], "run_id": raw["run_id"]}:
        raise IntentInputError("invalid_test_run_locator")
    paths = [safe_relative_path(path) for path in bounded_array(raw["paths"], maximum=MAX_PATHS)]
    _unique(paths)
    version_paths = {file.path for code in (task.base, task.head) if code.sha == raw["sha"] for file in code.files}
    if not paths or not set(paths) <= set(task.reviewed_paths) & version_paths:
        raise IntentInputError("test_run_outside_scope")
    if raw["status"] != "completed" or type(raw["exit_code"]) is not int or not -128 <= raw["exit_code"] <= 255:
        raise IntentInputError("unverified_test_run")
    command = bounded_array(raw["command"], maximum=128)
    if not command:
        raise IntentInputError("unverified_test_run")
    for part in command:
        bounded_text(part, maximum=4096)
    if not _timestamp(raw["started_at"]) <= _timestamp(raw["completed_at"]) <= _timestamp(task.snapshot.as_of):
        raise IntentInputError("invalid_test_run_time")
    bounded_text(raw["output"], maximum=MAX_INPUT_BYTES, empty=True)
    if _digest(raw["output_hash"]) != evidence_digest(raw["output"], "source"):
        raise IntentInputError("test_run_hash_mismatch")


def build_intent_evidence(task: IntentTaskInput) -> tuple[EnvelopeEvidence, ...]:
    """Only fixed original material becomes Evidence; no model string is accepted."""
    generated = _code_evidence(task)
    by_location = {(item.locator["version"], item.locator["path"]): item for item in generated}
    supplied = {}
    originals = []
    for item in task.snapshot.evidence:
        if item.kind in TEXT_KINDS:
            _validate_text_evidence(task, item)
            originals.append(item)
        elif item.kind == "test_run":
            _validate_test_run(task, item)
            originals.append(item)
        else:
            location = (item.locator.get("version"), item.locator.get("path"))
            expected = by_location.get(location)
            if expected is None or location in supplied or (item.kind, item.version, item.locator, item.content, item.digest) != (expected.kind, expected.version, expected.locator, expected.content, expected.digest):
                raise IntentInputError("invalid_code_evidence_locator")
            if any(type(value) is not type(expected.locator[key]) for key, value in item.locator.items()):
                raise IntentInputError("invalid_code_evidence_locator")
            supplied[location] = item
    codes = [supplied.get((item.locator["version"], item.locator["path"]), item) for item in generated]
    result = (*originals, *codes)
    _unique([item.id for item in result])
    if sum(item.kind == "pr_text" for item in originals) > 1:
        raise IntentInputError("duplicate_pr_text")
    return tuple(result)


def unavailable_context(task: IntentTaskInput, *, text_only: bool = False) -> tuple[str, ...]:
    entries = [f"{item.source}:{item.reason}" for item in task.snapshot.unavailable_sources
               if not text_only or item.source.startswith(("pr_text", "issue", "linked_issue"))]
    if not any(item.kind == "pr_text" for item in task.snapshot.evidence):
        entries.append("pr_text:unavailable")
    return tuple(sorted(set(entries)))


def intent_scope(task: IntentTaskInput) -> dict[str, object]:
    versions = {}
    missing = []
    for version in ("base", "head"):
        code = getattr(task, version)
        versions[version] = {"sha": code.sha, "paths": [file.path for file in code.files],
            "bytes": sum(len(file.content.encode("utf-8")) for file in code.files),
            "absent_paths": sorted(set(task.reviewed_paths) - {file.path for file in code.files}),
            "exclusions": [asdict(item) for item in code.exclusions], "errors": list(code.errors)}
        if not code.complete or code.errors or code.exclusions:
            missing.append(f"{version}:incomplete_prepared_scope")
    present = {version: {file.path for file in getattr(task, version).files} for version in ("base", "head")}
    for change in task.snapshot.changed_files:
        required = []
        if change.change_type not in ("added",):
            required.append(("base", change.previous_path or change.path))
        if change.change_type not in ("removed", "deleted"):
            required.append(("head", change.path))
        missing.extend(f"{version}:{path}:missing" for version, path in required if path not in present[version])
    for path in task.reviewed_paths:
        if path not in present["base"] | present["head"]:
            missing.append(f"context:{path}:missing")
    evidence = build_intent_evidence(task)
    return {"reviewed_paths": list(task.reviewed_paths), "repository_wide": False,
        "complete": not missing, "missing": sorted(set(missing)), "versions": versions,
        "evidence_ids": [item.id for item in evidence], "unavailable_context": list(unavailable_context(task)),
        "tests_executed_by_bot": False, "test_run_evidence_ids": [item.id for item in evidence if item.kind == "test_run"],
        "text_captured_at": task.snapshot.captured_at, "as_of": task.snapshot.as_of}


def _decode_input(value: Any, *, check_hash: bool = True) -> IntentTaskInput:
    raw = _record(IntentTaskInput, value)
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise IntentInputError("unsupported_intent_version")
    snapshot = _decode_snapshot(raw["snapshot"])
    base = _decode_code(raw["base"])
    head = _decode_code(raw["head"])
    paths = tuple(safe_relative_path(path) for path in bounded_array(raw["reviewed_paths"], maximum=MAX_PATHS))
    _unique(paths)
    if paths != tuple(sorted(paths)):
        raise IntentInputError("unordered_reviewed_paths")
    changed = {path for item in snapshot.changed_files for path in (item.path, item.previous_path) if path}
    if not changed <= set(paths) or any(file.path not in paths for code in (base, head) for file in code.files):
        raise IntentInputError("code_outside_scope")
    if base.sha != (snapshot.comparison_base_sha or snapshot.target_base_sha) or head.sha != snapshot.head_sha:
        raise IntentInputError("code_sha_mismatch")
    if base.sha == head.sha and {file.path: file.content_hash for file in base.files} != {file.path: file.content_hash for file in head.files}:
        raise IntentInputError("inconsistent_code_version")
    task = IntentTaskInput(1, snapshot, base, head, paths, raw["input_hash"])
    build_intent_evidence(task)
    payload = asdict(task)
    payload.pop("input_hash")
    expected = hash_json(payload)
    if check_hash and _digest(task.input_hash) != expected:
        raise IntentInputError("input_hash_mismatch")
    return replace(task, input_hash=expected)


def decode_intent_task_input(value: str | dict[str, Any]) -> IntentTaskInput:
    """Decode and verify a fixed B v1 input, without repository/network access."""
    if isinstance(value, dict):
        try:
            value = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError):
            raise IntentInputError("invalid_json") from None
    try:
        return _decode_input(strict_json_loads(value))
    except IntentInputError:
        raise
    except (TypeError, ValueError, KeyError, AttributeError, RecursionError):
        raise IntentInputError("invalid_intent_input") from None


def build_intent_task_input(
    snapshot: ReviewSnapshot, base: CodeSnapshot, head: CodeSnapshot, *, related_paths: tuple[str, ...] = (),
) -> IntentTaskInput:
    """Prepare full files in the declared B scope; never shrink it for a model limit."""
    snapshot = _decode_snapshot(strict_json_loads(json.dumps(asdict(snapshot), ensure_ascii=False, allow_nan=False)))
    paths = tuple(sorted(set(validate_intent_context_paths(related_paths)) |
        {path for item in snapshot.changed_files for path in (item.path, item.previous_path) if path}))
    scoped = []
    for code in (base, head):
        if not isinstance(code, CodeSnapshot) or type(code.complete) is not bool:
            raise IntentInputError("invalid_code_snapshot")
        fixed_sha(code.sha)
        bounded_text(code.inventory_hash, maximum=64, empty=True)
        for file in code.files:
            bounded_text(file.path, maximum=4096)
            _digest(file.content_hash)
        for exclusion in code.exclusions:
            bounded_text(exclusion.path, maximum=4096)
        for error in code.errors:
            bounded_text(error, maximum=4096)
        if code.inventory_hash:
            original_inventory = hashlib.sha256(":".join(f"{file.path}:{file.content_hash}" for file in sorted(code.files, key=lambda file: file.path)).encode()).hexdigest()
            if _digest(code.inventory_hash) != original_inventory:
                raise IntentInputError("inventory_hash_mismatch")
        included = tuple(sorted((file for file in code.files if file.path in paths), key=lambda file: file.path))
        excluded = tuple(item for item in code.exclusions if item.path in paths)
        known_paths = {file.path for file in code.files} | {item.path for item in code.exclusions}
        errors = tuple(error for error in code.errors if any(error.startswith(f"{path}:") for path in paths)
                       or not any(error.startswith(f"{path}:") for path in known_paths))
        complete = not errors and not excluded and (code.complete or bool(code.errors or code.exclusions))
        inventory = hashlib.sha256(":".join(f"{file.path}:{file.content_hash}" for file in included).encode()).hexdigest()
        prepared = CodeSnapshot(code.sha, included, excluded, inventory, complete, errors)
        scoped.append(_decode_code(strict_json_loads(json.dumps(asdict(prepared), ensure_ascii=False, allow_nan=False))))
    task = IntentTaskInput(1, snapshot, scoped[0], scoped[1], paths, "")
    return _decode_input(strict_json_loads(json.dumps(asdict(task), ensure_ascii=False, allow_nan=False)), check_hash=False)