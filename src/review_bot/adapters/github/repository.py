"""Installation-scoped HTTPS fetches into private bare Git object caches."""
from __future__ import annotations

import base64
import fcntl
import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from review_bot.adapters.git import GitRepositoryReader, _run_git
from review_bot.application.repository_review_settings import (
    MAX_SETTINGS_BYTES, SETTINGS_PATH, RepositorySettingsError,
)
from review_bot.domain.envelope_v2 import RunRecord
from review_bot.ports import GitHubClient


class GitHubRepositoryError(RuntimeError):
    """Safe repository preparation failure, without raw subprocess output."""


class _DefaultSettingsReader(GitRepositoryReader):
    """Bound settings object read; reject links and inspect size before reading bytes."""

    def read_tree_blobs(self, commit_sha: str, paths=None) -> dict[str, bytes]:
        if paths != (SETTINGS_PATH,) or re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None:
            raise RepositorySettingsError("invalid_settings_path")
        parent = _run_git(self.repo_path, ["ls-tree", "-z", "--full-tree", commit_sha, "--", ".review-bot"])
        if parent.stdout and not parent.stdout.startswith(b"040000 tree "):
            raise RepositorySettingsError("invalid_settings_path")
        tree = _run_git(self.repo_path, ["ls-tree", "-z", "--full-tree", commit_sha, "--", SETTINGS_PATH])
        if not tree.stdout:
            return {}
        try:
            metadata, path = tree.stdout.rstrip(b"\0").split(b"\t")
            mode, kind, oid = metadata.split(b" ")
        except ValueError:
            raise RepositorySettingsError("invalid_settings_path") from None
        if (
            path != SETTINGS_PATH.encode("utf-8") or mode not in (b"100644", b"100755")
            or kind != b"blob" or re.fullmatch(rb"[0-9a-f]{40}", oid) is None
        ):
            raise RepositorySettingsError("invalid_settings_path")
        object_id = oid.decode("ascii")
        size = _run_git(self.repo_path, ["cat-file", "-s", object_id])
        try:
            byte_count = int(size.stdout)
        except ValueError:
            raise RepositorySettingsError("invalid_settings_blob") from None
        if not 0 <= byte_count <= MAX_SETTINGS_BYTES:
            raise RepositorySettingsError("manifest_size_limit")
        blob = _run_git(self.repo_path, ["cat-file", "blob", object_id])
        if len(blob.stdout) != byte_count:
            raise RepositorySettingsError("invalid_settings_blob")
        return {SETTINGS_PATH: blob.stdout}


class GitHubRepositoryCache:
    """Fetch exact PR commits without checkout, hooks, submodules or stored credentials."""

    def __init__(
        self, cache_root: Path, token_provider: Callable[[int, int], str], *, timeout_seconds: int = 120,
    ) -> None:
        self.cache_root = cache_root
        self.token_provider = token_provider
        self.timeout_seconds = timeout_seconds

    def _run(self, args: list[str], *, token: str | None = None) -> None:
        env = {
            "PATH": os.environ.get("PATH", ""),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_ALLOW_PROTOCOL": "https",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
        settings = {
            "credential.helper": "",
            "core.hooksPath": os.devnull,
            "core.fsmonitor": "false",
            "http.followRedirects": "false",
            "http.sslVerify": "true",
        }
        if token is not None:
            auth = base64.b64encode(f"x-access-token:{token}".encode()).decode("ascii")
            settings["http.https://github.com/.extraheader"] = f"Authorization: Basic {auth}"
        env["GIT_CONFIG_COUNT"] = str(len(settings))
        for index, (key, value) in enumerate(settings.items()):
            env[f"GIT_CONFIG_KEY_{index}"] = key
            env[f"GIT_CONFIG_VALUE_{index}"] = value
        try:
            result = subprocess.run(
                ["git", "--no-replace-objects", *args], env=env,
                capture_output=True, timeout=self.timeout_seconds, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise GitHubRepositoryError("repository_transport_unavailable") from None
        if result.returncode != 0:
            raise GitHubRepositoryError("repository_fetch_failed")

    def prepare(
        self, run: RunRecord, pr_data: dict[str, Any], github_client: GitHubClient,
    ) -> GitRepositoryReader:
        if run.installation_id <= 0 or run.repository_id <= 0 or run.pr_number <= 0:
            raise GitHubRepositoryError("invalid_repository_target")
        repository = getattr(github_client, "repository", "")
        if not isinstance(repository, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository,
        ) or repository.split("/")[1] in (".", ".."):
            raise GitHubRepositoryError("invalid_repository_name")
        base_repo = pr_data.get("base", {}).get("repo", {})
        if base_repo.get("id") != run.repository_id or base_repo.get("full_name") != repository:
            raise GitHubRepositoryError("repository_identity_mismatch")
        target_base = pr_data.get("base", {}).get("sha", "")
        head = pr_data.get("head", {}).get("sha", "")
        if any(not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (target_base, head)):
            raise GitHubRepositoryError("invalid_commit_sha")
        if self.cache_root.is_symlink():
            raise GitHubRepositoryError("unsafe_repository_cache")
        self.cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.cache_root.stat().st_mode & 0o077:
            raise GitHubRepositoryError("repository_cache_must_be_private")
        repo_path = self.cache_root / f"{run.repository_id}.git"
        if repo_path.is_symlink():
            raise GitHubRepositoryError("unsafe_repository_cache")
        lock_path = self.cache_root / f"{run.repository_id}.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            repo_path.mkdir(mode=0o700, exist_ok=True)
            if not (repo_path / "HEAD").is_file():
                self._run(["init", "--bare", "--template=", str(repo_path)])
            token = self.token_provider(run.installation_id, run.repository_id)
            self._run([
                "-C", str(repo_path), "fetch", "--no-tags", "--no-recurse-submodules",
                "--no-auto-maintenance", "--force", f"https://github.com/{repository}.git",
                target_base, f"refs/pull/{run.pr_number}/head:refs/review-bot/pr-{run.pr_number}",
            ], token=token)
            reader = GitRepositoryReader(repo_path)
            reader.get_commit(target_base)
            reader.get_commit(head)
            return reader

    def read_default_settings_source(
        self, run: RunRecord, pr_data: dict[str, Any], github_client: GitHubClient, *,
        before_request: Callable[[], bool] | None = None,
    ) -> tuple[str, GitRepositoryReader]:
        """Fetch one captured default SHA from the authenticated base repository only."""
        if any(type(value) is not int or value <= 0 for value in (run.installation_id, run.repository_id)):
            raise GitHubRepositoryError("invalid_repository_target")
        repository = getattr(github_client, "repository", "")
        if not isinstance(repository, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", repository,
        ) or repository.split("/")[1] in (".", ".."):
            raise GitHubRepositoryError("invalid_repository_name")
        base_repo = pr_data.get("base", {}).get("repo", {})
        if (
            type(base_repo.get("id")) is not int or base_repo["id"] != run.repository_id
            or base_repo.get("full_name") != repository
        ):
            raise GitHubRepositoryError("repository_identity_mismatch")
        get_default_sha = getattr(github_client, "get_default_branch_sha", None)
        if not callable(get_default_sha):
            raise GitHubRepositoryError("default_settings_source_unavailable")
        default_sha = get_default_sha(run.repository_id, before_request=before_request)
        if not isinstance(default_sha, str) or re.fullmatch(r"[0-9a-f]{40}", default_sha) is None:
            raise GitHubRepositoryError("invalid_default_branch_commit")
        if self.cache_root.is_symlink():
            raise GitHubRepositoryError("unsafe_repository_cache")
        self.cache_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.cache_root.stat().st_mode & 0o077:
            raise GitHubRepositoryError("repository_cache_must_be_private")
        repo_path = self.cache_root / f"{run.repository_id}.git"
        if repo_path.is_symlink():
            raise GitHubRepositoryError("unsafe_repository_cache")
        descriptor = os.open(self.cache_root / f"{run.repository_id}.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if before_request is not None and before_request() is not True:
                raise GitHubRepositoryError("github_read_gate_closed")
            repo_path.mkdir(mode=0o700, exist_ok=True)
            if not (repo_path / "HEAD").is_file():
                self._run(["init", "--bare", "--template=", str(repo_path)])
            token = self.token_provider(run.installation_id, run.repository_id)
            self._run([
                "-C", str(repo_path), "fetch", "--no-tags", "--no-recurse-submodules",
                "--no-auto-maintenance", "--force", f"https://github.com/{repository}.git", default_sha,
            ], token=token)
            if before_request is not None and before_request() is not True:
                raise GitHubRepositoryError("github_read_gate_closed")
            reader = _DefaultSettingsReader(repo_path)
            if reader.get_commit(default_sha).sha != default_sha:
                raise GitHubRepositoryError("invalid_default_branch_commit")
            return default_sha, reader
