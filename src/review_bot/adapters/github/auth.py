"""Repository-scoped GitHub App installation authentication."""
from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

_REFRESH_MARGIN = 60
_PERMISSIONS = {"contents": "read", "pull_requests": "write"}


class GitHubAppAuthError(RuntimeError):
    """Sanitized authentication failure without credentials or HTTP payloads."""


def _github_api_url(api_url: str) -> str:
    valid = False
    try:
        if isinstance(api_url, str) and all("!" <= char <= "~" for char in api_url):
            url = urlsplit(api_url)
            valid = (
                url.scheme == "https" and url.hostname == "api.github.com"
                and url.port in (None, 443) and url.username is None and url.password is None
                and url.path in ("", "/") and not url.query and not url.fragment
            )
    except Exception:
        pass
    if not valid:
        raise GitHubAppAuthError("Invalid GitHub authentication endpoint")
    return "https://api.github.com"


def _safe_token(token: Any) -> bool:
    return isinstance(token, str) and bool(token) and all("!" <= char <= "~" for char in token)


@dataclass(frozen=True)
class _CachedToken:
    token: str = field(repr=False)
    expires_at: float


class GitHubAppTokenProvider:
    """Issue minimal-permission tokens and cache them only in memory."""

    def __init__(
        self,
        app_id: str,
        private_key_file: Path,
        *,
        api_url: str = "https://api.github.com",
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if (
            not isinstance(app_id, str) or re.fullmatch(r"[0-9]+", app_id) is None
            or not any(char != "0" for char in app_id)
        ):
            raise GitHubAppAuthError("Invalid GitHub App ID")
        self._api_url = _github_api_url(api_url)
        key = None
        try:
            key = serialization.load_pem_private_key(private_key_file.read_bytes(), password=None)
        except Exception:
            pass
        if not isinstance(key, rsa.RSAPrivateKey):
            raise GitHubAppAuthError("Invalid GitHub App private key")
        self._app_id = app_id
        self._key = key
        self._transport = transport
        self._clock = clock
        self._cache: dict[tuple[int, int], _CachedToken] = {}
        self._bot_login: str | None = None
        self._lock = Lock()

    def _now(self) -> float:
        now = float("nan")
        try:
            now = float(self._clock())
        except Exception:
            pass
        if not math.isfinite(now) or now < 60:
            raise GitHubAppAuthError("Invalid GitHub authentication clock")
        return now

    def get_token(self, installation_id: int, repository_id: int) -> str:
        if (
            type(installation_id) is not int or installation_id <= 0
            or type(repository_id) is not int or repository_id <= 0
        ):
            raise GitHubAppAuthError("Invalid GitHub installation or repository ID")
        with self._lock:
            now = self._now()
            scope = (installation_id, repository_id)
            cached = self._cache.get(scope)
            if cached is not None and cached.expires_at > now + _REFRESH_MARGIN:
                return cached.token
            self._cache.pop(scope, None)
            cached = self._issue_token(installation_id, repository_id, now)
            self._cache[scope] = cached
            return cached.token

    def _app_headers(self, now: float) -> dict[str, str]:
        encoded = jwt.encode(
            {"iat": int(now) - 60, "exp": int(now) + 540, "iss": self._app_id},
            self._key, algorithm="RS256",
        )
        if not _safe_token(encoded):
            raise ValueError
        return {
            "Authorization": f"Bearer {encoded}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-review-bot",
        }

    def get_bot_login(self) -> str:
        with self._lock:
            if self._bot_login is not None:
                return self._bot_login
            resp = None
            try:
                headers = self._app_headers(self._now())
                with httpx.Client(
                    timeout=20.0, transport=self._transport, follow_redirects=False,
                ) as client:
                    resp = client.get(f"{_github_api_url(self._api_url)}/app", headers=headers)
            except Exception:
                resp = None
            if resp is None:
                raise GitHubAppAuthError("GitHub App metadata request failed")
            if resp.status_code != 200:
                raise GitHubAppAuthError(f"GitHub App metadata request failed (HTTP {resp.status_code})")
            payload = None
            try:
                payload = resp.json()
            except Exception:
                pass
            if (
                not isinstance(payload, dict) or type(payload.get("id")) is not int
                or str(payload["id"]) != self._app_id.lstrip("0")
                or not isinstance(payload.get("slug"), str)
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*", payload["slug"]) is None
            ):
                raise GitHubAppAuthError("Invalid GitHub App metadata response")
            self._bot_login = f"{payload['slug']}[bot]"
            return self._bot_login

    def _issue_token(self, installation_id: int, repository_id: int, now: float) -> _CachedToken:
        resp = None
        try:
            headers = self._app_headers(now)
            with httpx.Client(
                timeout=20.0, transport=self._transport, follow_redirects=False,
            ) as client:
                resp = client.post(
                    f"{self._api_url}/app/installations/{installation_id}/access_tokens",
                    headers=headers,
                    json={"repository_ids": [repository_id], "permissions": _PERMISSIONS},
                )
        except Exception:
            resp = None
        if resp is None:
            raise GitHubAppAuthError("GitHub App authentication request failed")
        if resp.status_code != 201:
            raise GitHubAppAuthError(f"GitHub App authentication failed (HTTP {resp.status_code})")
        payload = None
        try:
            payload = resp.json()
        except Exception:
            pass
        if not isinstance(payload, dict) or not _safe_token(payload.get("token")):
            raise GitHubAppAuthError("Invalid GitHub App token response")
        expires_at = None
        try:
            expiration = datetime.fromisoformat(payload["expires_at"])
            if expiration.utcoffset() is not None:
                expires_at = expiration.timestamp()
        except Exception:
            pass
        if (
            expires_at is None or not math.isfinite(expires_at)
            or expires_at <= self._now() + _REFRESH_MARGIN
        ):
            raise GitHubAppAuthError("Invalid GitHub App token expiry")
        permissions = payload.get("permissions")
        if (
            not isinstance(permissions, dict)
            or any(permissions.get(name) != access for name, access in _PERMISSIONS.items())
            or any(name not in (*_PERMISSIONS, "metadata") for name in permissions)
            or permissions.get("metadata", "read") != "read"
        ):
            raise GitHubAppAuthError("Invalid GitHub App token permissions")
        if "repository_selection" in payload and payload["repository_selection"] != "selected":
            raise GitHubAppAuthError("Invalid GitHub App token repository scope")
        if "repositories" in payload:
            repositories = payload["repositories"]
            if (
                not isinstance(repositories, list) or len(repositories) != 1
                or not isinstance(repositories[0], dict)
                or type(repositories[0].get("id")) is not int
                or repositories[0]["id"] != repository_id
            ):
                raise GitHubAppAuthError("Invalid GitHub App token repository scope")
        return _CachedToken(payload["token"], expires_at)