"""Atomic repository hot settings, trust boundary, and immutable fallback behavior."""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from review_bot.adapters.github.client import GitHubApiClient
from review_bot.adapters.github.repository import (
    GitHubRepositoryCache, GitHubRepositoryError, _DefaultSettingsReader,
)
from review_bot.adapters.repository_review_settings import FileRepositoryReviewSettingsStore
from review_bot.application.repository_review_settings import (
    MAX_SETTINGS_BYTES, SETTINGS_PATH, RepositoryReviewSettings,
    RepositoryReviewSettingsResolver, RepositorySettingsError,
    parse_repository_review_settings,
)
from review_bot.domain.envelope_v2 import RunRecord

SHA = "a" * 40
NEXT_SHA = "b" * 40


def manifest(**overrides) -> bytes:
    return json.dumps({"schema_version": 1, **overrides}, ensure_ascii=False).encode("utf-8")


def source(raw: bytes | None, sha=SHA):
    reader = Mock()
    reader.read_tree_blobs.return_value = {} if raw is None else {SETTINGS_PATH: raw}
    loader = Mock(return_value=(sha, reader))
    return loader, reader


def resolver(tmp_path):
    return RepositoryReviewSettingsResolver(FileRepositoryReviewSettingsStore(tmp_path))


def test_absent_settings_preserve_builtin_and_capture_default_once(tmp_path):
    loader, reader = source(None)
    resolution = resolver(tmp_path).resolve(44, 99, loader)
    assert resolution.bundle.is_builtin
    assert resolution.diagnostic is None
    assert resolution.attempted_sha == SHA
    loader.assert_called_once_with()
    reader.read_tree_blobs.assert_called_once_with(SHA, (SETTINGS_PATH,))


def test_valid_bundle_is_frozen_and_last_valid_survives_restart(tmp_path):
    raw = manifest(review_instructions={"architecture": "변경의 영향과 판단을 설명하세요."},
        presentation={"title_a": "팀 구조 리뷰"},
        comment_templates={"architecture": "{{summary}}\n{{llm_review}}\n{{static_analysis}}"})
    first = resolver(tmp_path).resolve(44, 99, source(raw)[0])
    assert first.bundle.instructions_for("architecture") == "변경의 영향과 판단을 설명하세요."
    assert first.bundle.comment_template("architecture").startswith("{{summary}}")
    assert first.bundle.instructions_for("intent") == ""
    with pytest.raises(FrozenInstanceError):
        first.bundle.source_sha = NEXT_SHA
    later = resolver(tmp_path).resolve(44, 99, source(manifest(review_instructions={
        "architecture": "new", "fact_check": "override forbidden"}), NEXT_SHA)[0])
    assert later.used_fallback
    assert later.diagnostic == "invalid_manifest_slots"
    assert later.bundle == first.bundle
    assert later.attempted_sha == NEXT_SHA
    assert later.bundle.source_sha == SHA
    assert first.bundle.presentation == (("title_a", "팀 구조 리뷰"),)


def test_invalid_or_unavailable_no_history_uses_builtin_without_content_leak(tmp_path):
    secret = "private repository text"
    failed = resolver(tmp_path).resolve(44, 99, Mock(side_effect=RuntimeError(secret)))
    assert failed.used_fallback and failed.bundle.is_builtin
    assert failed.diagnostic == "settings_source_or_state_unavailable"
    invalid = resolver(tmp_path).resolve(44, 99, source(secret.encode())[0])
    assert invalid.bundle.is_builtin and invalid.used_fallback
    assert secret not in invalid.diagnostic


def test_namespace_is_installation_and_repository_not_pr_or_name(tmp_path):
    current = resolver(tmp_path)
    valid = current.resolve(44, 99, source(manifest(review_instructions={"intent": "trusted"}))[0])
    failed = Mock(side_effect=OSError("unavailable"))
    assert current.resolve(44, 99, failed).bundle == valid.bundle
    assert current.resolve(45, 99, failed).bundle.is_builtin
    assert current.resolve(44, 100, failed).bundle.is_builtin


def test_complete_versions_replace_atomically_and_inflight_bundle_never_changes(tmp_path):
    current = resolver(tmp_path)
    first = current.resolve(44, 99, source(manifest(review_instructions={"intent": "first"},
        presentation={"title_b": "first title"}))[0]).bundle
    second = current.resolve(44, 99, source(manifest(review_instructions={"intent": "second"},
        presentation={"title_b": "second title"}), NEXT_SHA)[0]).bundle
    assert first.instructions_for("intent") == "first"
    assert first.presentation == (("title_b", "first title"),)
    assert second.instructions_for("intent") == "second"
    assert second.effective_version != first.effective_version
    # A later intentionally removed file is a valid complete builtin bundle.
    removed = current.resolve(44, 99, source(None, "c" * 40)[0]).bundle
    assert removed.is_builtin
    assert current.resolve(44, 99, Mock(side_effect=OSError())).bundle == removed


def test_generation_and_presentation_hashes_have_independent_effects():
    first = parse_repository_review_settings(manifest(review_instructions={"intent": "criteria"},
        presentation={"title_b": "first"}), SHA)
    label = parse_repository_review_settings(manifest(review_instructions={"intent": "criteria"},
        presentation={"title_b": "second"}), NEXT_SHA)
    criteria = parse_repository_review_settings(manifest(review_instructions={"intent": "new criteria"},
        presentation={"title_b": "first"}), NEXT_SHA)
    assert first.generation_hash == label.generation_hash
    assert first.presentation_hash != label.presentation_hash
    assert first.generation_hash != criteria.generation_hash
    assert first.presentation_hash == criteria.presentation_hash
    same_content_new_commit = parse_repository_review_settings(manifest(
        review_instructions={"intent": "criteria"}, presentation={"title_b": "first"}), NEXT_SHA)
    assert first.effective_version == same_content_new_commit.effective_version


def test_presentation_labels_match_renderer_character_and_utf8_byte_bounds():
    from review_bot.adapters.markdown.developer_presentation import label

    for text in ("x" * 1000, "가" * 341):
        bundle = parse_repository_review_settings(manifest(presentation={"title_b": text}), SHA)
        assert label(dict(bundle.presentation), "title_b", "fallback") == text
    for text in ("x" * 1001, "가" * 342):
        with pytest.raises(RepositorySettingsError, match="text_size_limit"):
            parse_repository_review_settings(manifest(presentation={"title_b": text}), NEXT_SHA)


def test_invalid_presentation_bundle_uses_fallback_without_replacing_last_valid(tmp_path):
    store = FileRepositoryReviewSettingsStore(tmp_path)
    current = RepositoryReviewSettingsResolver(store)
    previous = current.resolve(44, 99, source(manifest(review_instructions={"intent": "trusted"},
        presentation={"title_b": "x" * 1000}))[0])
    assert not previous.used_fallback
    for text in ("x" * 1001, "가" * 342):
        invalid = source(manifest(review_instructions={"intent": "unaccepted"},
            presentation={"title_b": text}), NEXT_SHA)[0]
        resolution = current.resolve(44, 99, invalid)
        assert resolution.used_fallback and resolution.diagnostic == "text_size_limit"
        assert resolution.bundle == previous.bundle
        assert resolution.attempted_sha == NEXT_SHA and resolution.bundle.source_sha == SHA
        assert store.load(44, 99) == previous.bundle
        no_history = current.resolve(44, 100, invalid)
        assert no_history.used_fallback and no_history.bundle.is_builtin
        assert no_history.diagnostic == "text_size_limit"
        assert store.load(44, 100) is None


@pytest.mark.parametrize("raw", [
    b"\xff", b'{"schema_version":1,"schema_version":1}',
    b'{"schema_version":true}', b'{"schema_version":1,"mode":"intent"}',
    manifest(review_instructions={"fact_check": "replace fact checking"}),
    manifest(review_instructions={"intent": "{% include 'private' %}"}),
    manifest(review_instructions={"intent": "\x00"}),
    manifest(review_instructions={"intent": "<script>execute</script>"}),
    manifest(review_instructions={"intent": "x" * 16385}),
    b'{"schema_version":1,"review_instructions":{"intent":"\\ud800"}}',
    manifest(presentation={"permission": "override"}),
    manifest(comment_templates={"architecture": "{{summary}} {{llm_review}}"}),
    manifest(comment_templates={"intent": "{{summary}} {{llm_review}} {{perspectives}} {{unknown}}"}),
    manifest(comment_templates={"intent": "{{summary}} {{summary}} {{llm_review}} {{perspectives}}"}),
    b" " * (MAX_SETTINGS_BYTES + 1),
])
def test_parser_rejects_unsafe_unsupported_and_oversize_content(raw):
    with pytest.raises(RepositorySettingsError):
        parse_repository_review_settings(raw, SHA)


def test_urls_are_inert_instruction_text_and_not_loaded():
    bundle = parse_repository_review_settings(manifest(review_instructions={
        "intent": "Use the policy vocabulary at https://example.com/style as a reference."}), SHA)
    assert "https://example.com" in bundle.instructions_for("intent")


def test_unexpected_reader_paths_are_invalid_and_use_complete_fallback(tmp_path):
    current = resolver(tmp_path)
    valid = current.resolve(44, 99, source(manifest(presentation={"title_a": "trusted"}))[0])
    loader, reader = source(None, NEXT_SHA)
    reader.read_tree_blobs.return_value = {"../settings.json": manifest()}
    failed = current.resolve(44, 99, loader)
    assert failed.bundle == valid.bundle
    assert failed.diagnostic == "invalid_settings_path"


def test_failed_atomic_replace_preserves_previous_record(tmp_path, monkeypatch):
    current = resolver(tmp_path)
    first = current.resolve(44, 99, source(manifest(presentation={"title_a": "original"}))[0])
    monkeypatch.setattr("review_bot.adapters.repository_review_settings.os.replace",
        Mock(side_effect=OSError("disk failed")))
    failed = current.resolve(44, 99, source(manifest(presentation={"title_a": "new"}), NEXT_SHA)[0])
    assert failed.used_fallback and failed.bundle == first.bundle
    assert not list((tmp_path / "repository-review-settings" / "44").glob("*.tmp"))


def test_corrupt_or_foreign_state_is_never_used_as_last_valid(tmp_path):
    store = FileRepositoryReviewSettingsStore(tmp_path)
    valid = resolver(tmp_path).resolve(44, 99, source(manifest(review_instructions={"intent": "x"}))[0])
    record_path = tmp_path / "repository-review-settings" / "44" / "99.json"
    record = json.loads(record_path.read_text())
    record["installation_id"] = 45
    record_path.write_text(json.dumps(record))
    failed = RepositoryReviewSettingsResolver(store).resolve(44, 99, Mock(side_effect=OSError()))
    assert failed.bundle.is_builtin and failed.used_fallback
    assert valid.bundle.instructions_for("intent") == "x"


def test_store_rejects_symlink_and_unsafe_identity(tmp_path):
    root = tmp_path / "repository-review-settings"
    target = tmp_path / "elsewhere"
    target.mkdir()
    root.symlink_to(target)
    assert resolver(tmp_path).resolve(44, 99, source(None)[0]).diagnostic == "settings_state_unavailable"
    with pytest.raises(ValueError):
        resolver(tmp_path).resolve(True, 99, source(None)[0])


def test_default_branch_api_is_repository_bound_and_captures_ref_once():
    calls = []
    def request(req):
        calls.append(req.url.path)
        if req.url.path == "/repos/org/repo":
            return httpx.Response(200, json={"id": 99, "full_name": "org/repo", "default_branch": "release/main"})
        return httpx.Response(200, json={"ref": "refs/heads/release/main", "object": {"type": "commit", "sha": SHA}})
    client = GitHubApiClient(repository="org/repo", repository_id=99, transport=httpx.MockTransport(request))
    gate = Mock(return_value=True)
    assert client.get_default_branch_sha(99, before_request=gate) == SHA
    assert calls == ["/repos/org/repo", "/repos/org/repo/git/ref/heads/release/main"]
    assert gate.call_count == 4
    with pytest.raises(ValueError):
        client.get_default_branch_sha(100)
    assert len(calls) == 2


def test_default_branch_api_rejects_foreign_repository_and_cancellation():
    transport = Mock(return_value=httpx.Response(200, json={"id": 100, "full_name": "fork/repo", "default_branch": "main"}))
    client = GitHubApiClient(repository="org/repo", repository_id=99, transport=httpx.MockTransport(transport))
    with pytest.raises(ValueError, match="invalid_default_branch_repository"):
        client.get_default_branch_sha()
    with pytest.raises(ValueError, match="github_read_gate_closed"):
        client.get_default_branch_sha(before_request=lambda: False)
    assert transport.call_count == 1


def test_default_source_fetch_never_selects_pr_head(tmp_path, synthetic_repo, monkeypatch):
    synthetic_repo.write_file(SETTINGS_PATH, manifest(review_instructions={"intent": "default policy"}))
    default_sha = synthetic_repo.commit("Default branch policy")
    cache = GitHubRepositoryCache(tmp_path / "cache", Mock(return_value="synthetic-token"))
    fetch = Mock()
    monkeypatch.setattr(cache, "_run", fetch)
    # Use real Git objects as immutable source, replacing transport only.
    monkeypatch.setattr("review_bot.adapters.github.repository._DefaultSettingsReader",
        lambda _: _DefaultSettingsReader(synthetic_repo.repo_dir))
    client = SimpleNamespace(repository="org/repo", get_default_branch_sha=Mock(return_value=default_sha))
    run = RunRecord("settings-run", 44, 99, 7, 1, "", "running", "prepare")
    pr = {"base": {"sha": "c" * 40, "repo": {"id": 99, "full_name": "org/repo"}},
        "head": {"sha": "d" * 40, "repo": {"id": 123, "full_name": "fork/repo"}}}
    sha, reader = cache.read_default_settings_source(run, pr, client)
    assert sha == default_sha
    assert reader.read_tree_blobs(sha, (SETTINGS_PATH,))[SETTINGS_PATH].endswith(b"}}")
    client.get_default_branch_sha.assert_called_once_with(99, before_request=None)
    fetch_arguments = fetch.call_args_list[-1].args[0]
    assert fetch_arguments[-1] == default_sha
    assert "c" * 40 not in fetch_arguments and "d" * 40 not in fetch_arguments
    assert not any("pull/" in argument for argument in fetch_arguments)
    pr["base"]["repo"]["id"] = 123
    with pytest.raises(GitHubRepositoryError, match="repository_identity_mismatch"):
        cache.read_default_settings_source(run, pr, client)
    assert client.get_default_branch_sha.call_count == 1


@pytest.mark.parametrize("kind", ["symlink", "parent_symlink", "oversize", "directory"])
def test_default_source_rejects_non_plain_blob_before_read(synthetic_repo, kind):
    if kind == "parent_symlink":
        target = synthetic_repo.repo_dir / "actual"
        target.mkdir()
        (target / "review-settings.json").write_bytes(manifest())
        (synthetic_repo.repo_dir / ".review-bot").symlink_to(target)
    elif kind == "symlink":
        target = synthetic_repo.write_file("actual.json", manifest())
        path = synthetic_repo.repo_dir / SETTINGS_PATH
        path.parent.mkdir(parents=True)
        path.symlink_to(target)
    elif kind == "directory":
        synthetic_repo.write_file(SETTINGS_PATH + "/nested.json", manifest())
    else:
        synthetic_repo.write_file(SETTINGS_PATH, b" " * (MAX_SETTINGS_BYTES + 1))
    sha = synthetic_repo.commit("Malformed settings object")
    reader = _DefaultSettingsReader(synthetic_repo.repo_dir)
    with pytest.raises(RepositorySettingsError):
        reader.read_tree_blobs(sha, (SETTINGS_PATH,))
