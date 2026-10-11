"""Per-run composition of installation-scoped GitHub and repository adapters."""
from __future__ import annotations

from pathlib import Path

import httpx

from review_bot.adapters.github.auth import GitHubAppTokenProvider
from review_bot.adapters.github.client import GitHubApiClient
from review_bot.adapters.github.repository import GitHubRepositoryCache, GitHubRepositoryError
from review_bot.adapters.github.webhook import is_installation_allowed, is_repository_allowed
from review_bot.config import ServiceConfig
from review_bot.domain.envelope_v2 import RunRecord


class GitHubAppAdapters:
    """One canonical adapter factory; clients never cross installation/repository bindings."""

    def __init__(
        self, provider: GitHubAppTokenProvider, cache_root: Path,
        service: ServiceConfig, *, transport: httpx.BaseTransport | None = None,
        fetch_timeout_seconds: int = 120,
    ) -> None:
        self.provider = provider
        self.service = service
        self.transport = transport
        self.repository_cache = GitHubRepositoryCache(
            cache_root, provider.get_token, timeout_seconds=fetch_timeout_seconds,
        )

    def github_client(self, run: RunRecord) -> GitHubApiClient:
        if run.installation_id <= 0 or run.repository_id <= 0:
            raise GitHubRepositoryError("installation_required")
        if not is_installation_allowed(run.installation_id, self.service.installation_allowlist):
            raise GitHubRepositoryError("installation_unlisted")
        client = GitHubApiClient(
            token_provider=lambda: self.provider.get_token(run.installation_id, run.repository_id),
            repository_id=run.repository_id, transport=self.transport,
            bot_login=self.provider.get_bot_login(),
        )
        repository = client.resolve_repository(run.repository_id)
        if not is_repository_allowed(repository, self.service.repository_allowlist):
            raise GitHubRepositoryError("repository_unlisted")
        return client