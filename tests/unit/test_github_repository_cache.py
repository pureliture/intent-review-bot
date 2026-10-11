"""Private source fetch boundary: fixed identities, no checkout or persisted token."""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from review_bot.adapters.github.repository import GitHubRepositoryCache, GitHubRepositoryError
from review_bot.domain.envelope_v2 import RunRecord
from tests.conftest import SyntheticGitRepo


def request() -> RunRecord:
    return RunRecord("synthetic-run", 44, 99, 7, 1, "", "running", "prepare")


def metadata(base: str, head: str) -> dict:
    return {"head": {"sha": head}, "base": {
        "sha": base, "repo": {"id": 99, "full_name": "org/repo"},
    }}


def test_fetch_uses_private_bare_objects_and_ephemeral_scoped_auth(
    tmp_path: Path, synthetic_repo: SyntheticGitRepo, monkeypatch: pytest.MonkeyPatch,
) -> None:
    topology = synthetic_repo.setup_standard_feature_pr()
    root = tmp_path / "cache"
    root.mkdir(mode=0o700)
    mirror = root / "99.git"
    subprocess.run(["git", "clone", "--bare", str(synthetic_repo.repo_dir), str(mirror)], check=True, capture_output=True)
    captured = []
    execute = subprocess.run

    def fetch(command, **kwargs):
        if "fetch" in command:
            captured.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return execute(command, **kwargs)

    monkeypatch.setattr("review_bot.adapters.github.repository.subprocess.run", fetch)
    # The fetch HTTP transport is replaced, but commit/snapshot reads remain real Git.
    provider = Mock(return_value="synthetic-private-token")
    cache = GitHubRepositoryCache(root, provider)
    reader = cache.prepare(
        request(), metadata(topology["target_base_sha"], topology["head_sha"]),
        SimpleNamespace(repository="org/repo"),
    )

    provider.assert_called_once_with(44, 99)
    assert reader.get_commit(topology["head_sha"]).sha == topology["head_sha"]
    assert reader.read_tree_blobs(topology["head_sha"])["src/main/java/PaymentService.java"]
    command, options = captured[0]
    assert "--no-recurse-submodules" in command and "--no-tags" in command
    assert "--no-auto-maintenance" in command
    assert "https://github.com/org/repo.git" in command
    assert "refs/pull/7/head:refs/review-bot/pr-7" in command
    assert not any("synthetic-private-token" in part for part in command)
    env = options["env"]
    assert "REVIEW_BOT_APP_PRIVATE_KEY_FILE" not in env and "MODEL_API_KEY" not in env
    assert env["GIT_ALLOW_PROTOCOL"] == "https"
    settings = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
    encoded = base64.b64encode(b"x-access-token:synthetic-private-token").decode()
    assert settings["http.https://github.com/.extraheader"] == "Authorization: Basic " + encoded
    assert settings["http.followRedirects"] == "false"
    assert settings["credential.helper"] == ""
    assert "extraheader" not in (mirror / "config").read_text()
    assert "synthetic-private-token" not in (mirror / "config").read_text()
    assert root.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("fault", ["repository", "sha", "name", "installation"])
def test_invalid_binding_is_rejected_before_token_or_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    provider = Mock()
    process = Mock()
    monkeypatch.setattr("review_bot.adapters.github.repository.subprocess.run", process)
    run = request()
    pr = metadata("a" * 40, "b" * 40)
    client = SimpleNamespace(repository="org/repo")
    if fault == "repository":
        pr["base"]["repo"]["id"] = 100
    if fault == "sha":
        pr["head"]["sha"] = "--upload-pack=malicious"
    if fault == "name":
        client.repository = "org/../../private"
    if fault == "installation":
        run = RunRecord("synthetic-run", 0, 99, 7, 1, "", "running", "prepare")
    with pytest.raises(GitHubRepositoryError):
        GitHubRepositoryCache(tmp_path / "cache", provider).prepare(run, pr, client)
    provider.assert_not_called()
    process.assert_not_called()


def test_failed_transport_never_exposes_stderr_or_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, b"", b"synthetic-private-token: private failure")

    monkeypatch.setattr("review_bot.adapters.github.repository.subprocess.run", fail)
    cache = GitHubRepositoryCache(tmp_path / "cache", lambda installation, repository: "synthetic-private-token")
    with pytest.raises(GitHubRepositoryError) as failure:
        cache._run(["fetch", "https://github.com/org/repo.git"], token="synthetic-private-token")
    assert str(failure.value) == "repository_fetch_failed"
    assert "synthetic-private-token" not in str(failure.value)