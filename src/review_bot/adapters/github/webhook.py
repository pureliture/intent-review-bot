"""Webhook payload models, HMAC-SHA256 signature verification, and allowlist validation."""
from __future__ import annotations

import hashlib
import hmac

from pydantic import BaseModel, ConfigDict, Field


class GitHubUser(BaseModel):
    model_config = ConfigDict(extra="ignore")
    login: str
    id: int | None = None


class GitHubCommitRef(BaseModel):
    model_config = ConfigDict(extra="ignore")
    sha: str
    ref: str


class GitHubLabel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    id: int | None = None


class GitHubRepository(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    name: str
    full_name: str
    private: bool = False
    owner: GitHubUser | None = None


class GitHubInstallation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: int
    node_id: str | None = None


class PullRequestData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    number: int
    id: int | None = None
    state: str = "open"
    title: str = ""
    body: str | None = None
    head: GitHubCommitRef
    base: GitHubCommitRef
    labels: list[GitHubLabel] = Field(default_factory=list)
    draft: bool = False


class PullRequestEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str
    number: int
    pull_request: PullRequestData
    repository: GitHubRepository
    installation: GitHubInstallation | None = None
    sender: GitHubUser | None = None


class PingEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    zen: str | None = None
    hook_id: int | None = None
    repository: GitHubRepository | None = None
    sender: GitHubUser | None = None


class WebhookPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")
    action: str | None = None
    number: int | None = None
    repository: GitHubRepository | None = None
    installation: GitHubInstallation | None = None
    pull_request: PullRequestData | None = None
    zen: str | None = None
    hook_id: int | None = None
    sender: GitHubUser | None = None


def verify_github_signature(
    raw_body: bytes,
    signature_header: str | None,
    secret: str,
) -> bool:
    """Verify HMAC SHA-256 signature sent by GitHub in X-Hub-Signature-256."""
    if not signature_header or not secret:
        return False
    if not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def is_repository_allowed(
    repo_full_name: str,
    allowlist: list[str],
) -> bool:
    """Check if repository matches operator allowlist (empty allowlist allows all)."""
    if not allowlist:
        return True
    return repo_full_name in allowlist


def is_installation_allowed(
    installation_id: int | None,
    allowlist: list[int],
) -> bool:
    """Check if installation matches operator allowlist (empty allowlist allows all)."""
    if not allowlist:
        return True
    if installation_id is None:
        return False
    return installation_id in allowlist
