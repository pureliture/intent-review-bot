"""Completed structured review reuse keeps its recorded originals and validation."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import Mock

import pytest

from review_bot.runtime.shared_review_cache import SharedReviewCache

KEY = "a" * 64


def packet(mode="intent"):
    return {"task": {"snapshot_id": "original-snapshot", "input_hash": "original-input"},
        "result": {"schema_version": 13, "status": "planned" if mode == "planning" else "completed",
            "review_type": {"architecture": "A", "intent": "B", "planning": "PLAN"}[mode],
            "snapshot_id": "original-snapshot", "input_hash": "original-input", "summary": "Original review"}}


def decode(value):
    assert value["task"]["snapshot_id"] == value["result"]["snapshot_id"]
    assert value["task"]["input_hash"] == value["result"]["input_hash"]
    return value


def path(root, installation=44, repository=99, mode="intent", key=KEY):
    return root / "shared-review-cache" / str(installation) / str(repository) / mode / f"{key}.json"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@pytest.mark.parametrize("mode", ["architecture", "intent", "planning"])
def test_completed_modes_and_shared_plan_restore_original_json_after_restart(tmp_path, mode):
    original = packet(mode)
    validate = Mock(side_effect=decode)
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, mode, KEY, original, validate_and_decode=validate)
    assert validate.call_count == 1
    assert path(tmp_path, mode=mode).stat().st_mode & 0o777 == 0o600
    restored = SharedReviewCache(tmp_path).load(44, 99, mode, KEY, validate_and_decode=validate)
    assert restored == original and validate.call_count == 2
    assert restored["result"]["snapshot_id"] == "original-snapshot"
    restored["result"]["summary"] = "local mutation"
    assert cache.load(44, 99, mode, KEY, validate_and_decode=decode) == original


def test_namespace_and_generation_key_prevent_cross_review_reuse(tmp_path):
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=decode)
    for installation, repository, mode, key in (
        (45, 99, "intent", KEY), (44, 100, "intent", KEY),
        (44, 99, "architecture", KEY), (44, 99, "intent", "b" * 64),
    ):
        assert cache.load(installation, repository, mode, key, validate_and_decode=decode) is None


@pytest.mark.parametrize("status", ["failed", "partial", "measured", "not_run", "planned"])
def test_partial_or_nonreview_results_do_not_replace_completed_original(tmp_path, status):
    cache = SharedReviewCache(tmp_path)
    original = packet()
    assert cache.save(44, 99, "intent", KEY, original, validate_and_decode=decode)
    invalid = deepcopy(original)
    invalid["result"]["status"] = status
    validate = Mock(side_effect=AssertionError("must not validate incomplete"))
    assert not cache.save(44, 99, "intent", KEY, invalid, validate_and_decode=validate)
    validate.assert_not_called()
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=decode) == original


def test_independent_read_validator_is_required_and_failure_is_a_miss(tmp_path):
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=decode)
    recorded = path(tmp_path).read_bytes()
    validator = Mock(side_effect=ValueError("stored result integrity failed"))
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=validator) is None
    validator.assert_called_once()
    assert path(tmp_path).read_bytes() == recorded
    assert not cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=validator)
    assert path(tmp_path).read_bytes() == recorded


def test_callback_mutation_cannot_change_recorded_originals(tmp_path):
    original = packet()
    def mutate(value):
        value["task"]["snapshot_id"] = "replacement"
        return value
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "intent", KEY, original, validate_and_decode=mutate)
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=decode) == original
    assert original == packet()


@pytest.mark.parametrize("corruption", ["json", "checksum", "identity", "status", "schema", "duplicate"])
def test_corrupt_cache_is_a_miss_and_does_not_get_erased(tmp_path, corruption):
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=decode)
    target = path(tmp_path)
    envelope = json.loads(target.read_bytes())
    if corruption == "json":
        raw = b"malformed private data"
    elif corruption == "duplicate":
        raw = target.read_bytes().replace(b'"schema_version":1', b'"schema_version":1,"schema_version":1')
    else:
        if corruption == "checksum":
            envelope["packet"]["result"]["summary"] = "changed original"
        if corruption == "identity":
            envelope["installation_id"] = 45
        if corruption == "status":
            envelope["packet"]["result"]["status"] = "partial"
        if corruption == "schema":
            envelope["schema_version"] = True
        raw = canonical(envelope)
    target.write_bytes(raw)
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=decode) is None
    assert target.read_bytes() == raw


def test_atomic_write_failure_preserves_original(tmp_path, monkeypatch):
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=decode)
    original = path(tmp_path).read_bytes()
    monkeypatch.setattr("review_bot.runtime.shared_review_cache.os.replace", Mock(side_effect=OSError()))
    changed = packet()
    changed["result"]["summary"] = "replacement"
    assert not cache.save(44, 99, "intent", KEY, changed, validate_and_decode=decode)
    assert path(tmp_path).read_bytes() == original
    assert not list(path(tmp_path).parent.glob(".pending-*"))


def test_symlink_permissions_and_oversize_are_misses(tmp_path):
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=decode)
    target = path(tmp_path)
    original = target.read_bytes()
    target.chmod(0o644)
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=decode) is None
    target.chmod(0o600)
    assert SharedReviewCache(tmp_path, max_bytes=10).load(44, 99, "intent", KEY, validate_and_decode=decode) is None
    target.unlink()
    foreign = tmp_path / "other.json"
    foreign.write_bytes(original)
    target.symlink_to(foreign)
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=decode) is None
    assert not cache.save(44, 99, "intent", KEY, packet(), validate_and_decode=decode)
    assert foreign.read_bytes() == original


def test_invalid_json_nonfinite_and_traversal_never_write(tmp_path):
    cache = SharedReviewCache(tmp_path)
    invalid = packet()
    invalid["task"]["unsafe_number"] = float("nan")
    assert not cache.save(44, 99, "intent", KEY, invalid, validate_and_decode=decode)
    assert not cache.save(44, 99, "../../outside", KEY, packet(), validate_and_decode=decode)
    assert not cache.save(44, 99, "intent", "../outside", packet(), validate_and_decode=decode)
    assert not cache.save(True, 99, "intent", KEY, packet(), validate_and_decode=decode)
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=decode) is None


def test_real_partition_result_roundtrip_revalidates_original_task_and_result(tmp_path):
    from pydantic import TypeAdapter
    from review_bot.application.partition_review import run_partition_review, validate_partition_result
    from review_bot.domain.intent_models import IntentTaskInput
    from review_bot.domain.partition_review import PartitionReviewLimits, PartitionReviewResult
    from tests.fixtures.intent_perspectives import make_task
    from tests.fixtures.partition_review import PartitionModelClient

    task = make_task()
    review, tev = PartitionModelClient(), PartitionModelClient(group_all=True)
    result = run_partition_review(task, "intent", review, tev,
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=128))
    assert result.status == "completed", result.reason
    model_calls = len(review.requests) + len(tev.requests)

    def restore(value):
        original_task = TypeAdapter(IntentTaskInput).validate_python(value["task"])
        original_result = TypeAdapter(PartitionReviewResult).validate_python(value["result"])
        validate_partition_result(original_result, original_task)
        return original_task, original_result

    cache = SharedReviewCache(tmp_path)
    original = {"task": asdict(task), "result": asdict(result)}
    assert cache.save(44, 99, "intent", KEY, original, validate_and_decode=restore)
    restored_task, restored_result = cache.load(44, 99, "intent", KEY, validate_and_decode=restore)
    assert asdict(restored_task) == asdict(task)
    assert asdict(restored_result) == asdict(result)
    assert len(review.requests) + len(tev.requests) == model_calls

    # Even a syntactically valid entry with a recomputed checksum must pass the
    # independent domain validator before it can reach presentation rendering.
    target = path(tmp_path)
    envelope = json.loads(target.read_bytes())
    envelope["packet"]["result"]["input_hash"] = "forged"
    envelope["packet_hash"] = hashlib.sha256(canonical(envelope["packet"])).hexdigest()
    target.write_bytes(canonical(envelope))
    assert cache.load(44, 99, "intent", KEY, validate_and_decode=restore) is None


def test_architecture_cache_codec_restores_real_task_dataclass(tmp_path, worker_review_inputs):
    from review_bot.application.partition_review import run_partition_review
    from review_bot.domain.models_v1 import TaskInput
    from review_bot.domain.partition_review import PartitionReviewLimits
    from review_bot.runtime.stored_partition_review import decode_stored_partition_review
    from tests.fixtures.partition_review import PartitionModelClient

    task, _ = worker_review_inputs
    review, tev = PartitionModelClient(), PartitionModelClient(group_all=True)
    result = run_partition_review(task, "architecture", review, tev,
        limits=PartitionReviewLimits(max_total_tokens=1000000, max_calls=128))
    assert result.status == "completed", result.reason
    cache = SharedReviewCache(tmp_path)
    assert cache.save(44, 99, "architecture", KEY, {"task": asdict(task), "result": asdict(result)},
        validate_and_decode=decode_stored_partition_review)
    restored_task, restored_result = cache.load(44, 99, "architecture", KEY,
        validate_and_decode=decode_stored_partition_review)
    assert isinstance(restored_task, TaskInput)
    assert asdict(restored_task) == asdict(task)
    assert asdict(restored_result) == asdict(result)


@pytest.mark.parametrize("changed", ["ruleset", "tools_lock"])
def test_same_path_recipe_bytes_or_unavailable_recipe_cannot_reuse_previous_a_or_plan(tmp_path, valid_config_yaml, changed):
    from review_bot.application.repository_review_settings import RepositoryReviewSettings
    from review_bot.config import load_review_config_from_str
    from review_bot.runtime.stored_partition_review import shared_review_generation_key
    from tests.fixtures.intent_perspectives import make_task
    from tests.fixtures.partition_review import PartitionModelClient

    config, _ = load_review_config_from_str(valid_config_yaml)
    config.execution.available_modes = ["architecture", "intent"]
    ruleset, lock = tmp_path / "ruleset.xml", tmp_path / "tools.lock.json"
    ruleset.write_bytes(b"<ruleset>original</ruleset>")
    lock.write_bytes(b'{"recipe":"original"}')
    config.architecture.ruleset_resource = str(ruleset)
    config.architecture.tools_lock_path = str(lock)
    task, review, tev = make_task(), PartitionModelClient(), PartitionModelClient()
    bundle = RepositoryReviewSettings()
    cache = SharedReviewCache(tmp_path)

    def key(mode):
        return shared_review_generation_key(task, config, review, tev, bundle, mode)

    original = {mode: key(mode) for mode in ("architecture", "planning", "intent")}
    for mode in ("architecture", "planning"):
        assert cache.save(44, 99, mode, original[mode], packet(mode), validate_and_decode=decode)
    target = ruleset if changed == "ruleset" else lock
    target.write_bytes(target.read_bytes() + b" changed bytes at the same path")
    modified = {mode: key(mode) for mode in original}
    for mode in ("architecture", "planning"):
        assert modified[mode] != original[mode]
        assert cache.load(44, 99, mode, modified[mode], validate_and_decode=decode) is None
    assert modified["intent"] == original["intent"]
    target.unlink()
    for mode in ("architecture", "planning"):
        missing = key(mode)
        assert missing not in (original[mode], modified[mode])
        assert cache.load(44, 99, mode, missing, validate_and_decode=decode) is None
    # Intent-only planning does not require an unused Java analysis recipe.
    config.execution.available_modes = ["intent"]
    absent = key("planning")
    target.write_bytes(b"restored recipe")
    assert key("planning") == absent
    assert review.requests == tev.requests == []
