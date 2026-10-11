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

from review_bot.adapters.git import GitRepositoryReader
from review_bot.domain.envelope_v2 import RunRecord
from review_bot.ports import GitHubClient


class GitHubRepositoryError(RuntimeError):
    """Safe repository preparation failure, without raw subprocess output."""


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