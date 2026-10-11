"""GitHub adapter package."""
from __future__ import annotations

from review_bot.adapters.github.webhook import (
    GitHubCommitRef,
    GitHubInstallation,
    GitHubLabel,
    GitHubRepository,
    GitHubUser,
    PingEvent,
    PullRequestData,
    PullRequestEvent,
    WebhookPayload,
    is_installation_allowed,
    is_repository_allowed,
    verify_github_signature,
)

__all__ = [
    "GitHubCommitRef",
    "GitHubInstallation",
    "GitHubLabel",
    "GitHubRepository",
    "GitHubUser",
    "PingEvent",
    "PullRequestData",
    "PullRequestEvent",
    "WebhookPayload",
    "is_installation_allowed",
    "is_repository_allowed",
    "verify_github_signature",
]
