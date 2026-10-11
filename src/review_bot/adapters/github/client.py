"""GitHub API client adapter conforming to ports.GitHubClient protocol."""
from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from review_bot.adapters.github.auth import GitHubAppAuthError, _github_api_url, _safe_token

_MAX_COMMENT_PAGES = 100


class GitHubApiClient:
    """Production GitHub REST API client using httpx."""

    def __init__(
        self,
        token: str | None = None,
        repository: str = "",
        api_url: str = "https://api.github.com",
        timeout: float = 20.0,
        *,
        token_provider: Callable[[], str] | None = None,
        transport: httpx.BaseTransport | None = None,
        repository_id: int | None = None,
        bot_login: str | None = None,
    ) -> None:
        if repository_id is not None and (type(repository_id) is not int or repository_id <= 0):
            raise ValueError("Invalid GitHub repository ID")
        if token_provider is not None and not callable(token_provider):
            raise GitHubAppAuthError("Invalid GitHub token provider")
        if bot_login is not None and (
            not isinstance(bot_login, str)
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9-]*\[bot\]", bot_login, re.IGNORECASE | re.ASCII,
            ) is None
        ):
            raise ValueError("Invalid GitHub bot login")
        self.token = token
        self.repository = repository
        self.api_url = _github_api_url(api_url) if token_provider is not None else api_url.rstrip("/")
        self.timeout = timeout
        self._token_provider = token_provider
        self._transport = transport
        self._repository_id = repository_id
        self._bound_repository = repository if repository_id is not None and repository else None
        self._bot_login = bot_login

    @property
    def repository_id(self) -> int | None:
        return self._repository_id

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "github-review-bot",
        }
        token = self.token
        if self._token_provider is not None:
            token = None
            try:
                token = self._token_provider()
            except Exception:
                pass
            if not _safe_token(token):
                raise GitHubAppAuthError("GitHub token provider failed")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    @staticmethod
    def _safe_request(method: str = "GET") -> httpx.Request:
        return httpx.Request(method, "https://api.github.com")

    def _request(self, method: str, url: str | httpx.URL, **kwargs: Any) -> httpx.Response:
        if self._token_provider is not None:
            _github_api_url(self.api_url)
        headers = self._headers()
        options: dict[str, Any] = {"timeout": self.timeout, "follow_redirects": False}
        if self._transport is not None:
            options["transport"] = self._transport
        resp = None
        try:
            with httpx.Client(**options) as client:
                resp = client.request(method, url, headers=headers, **kwargs)
        except Exception:
            resp = None
        if resp is None:
            raise httpx.RequestError("GitHub API request failed", request=self._safe_request(method))
        if not 200 <= resp.status_code < 300:
            request = self._safe_request(method)
            raise httpx.HTTPStatusError(
                f"GitHub API request failed (HTTP {resp.status_code})", request=request,
                response=httpx.Response(resp.status_code, request=request),
            )
        return resp

    def _json(self, resp: httpx.Response, label: str, expected_type: type = dict) -> Any:
        payload = None
        try:
            payload = resp.json()
        except Exception:
            pass
        if not isinstance(payload, expected_type):
            raise httpx.RequestError(f"Invalid GitHub {label} response", request=self._safe_request())
        return payload

    @staticmethod
    def _safe_repository(repo: Any) -> bool:
        return (
            isinstance(repo, str)
            and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+", repo) is not None
            and repo.split("/")[1] not in (".", "..")
        )

    def _check_repository_id(self, repository_id: Any) -> None:
        if type(repository_id) is not int or repository_id <= 0:
            raise ValueError("Invalid GitHub repository ID")
        if self.repository_id is not None and repository_id != self.repository_id:
            raise ValueError("GitHub repository ID does not match bound repository")

    def _repository_for(self, reference: Any = None, **kwargs: Any) -> str:
        if "repository_id" in kwargs:
            self._check_repository_id(kwargs["repository_id"])
        repo = self.repository
        for value in (reference, kwargs.get("repository")):
            if value is None:
                continue
            if isinstance(value, str) and "/" in value:
                if not self._safe_repository(value):
                    raise ValueError("Invalid GitHub repository name")
                if self._bound_repository is not None and value.casefold() != self._bound_repository.casefold():
                    raise ValueError("GitHub repository name does not match bound repository")
                repo = value
            else:
                self._check_repository_id(value)
        if self.repository_id is not None:
            if self._bound_repository is None:
                raise ValueError("Bound GitHub repository has not been resolved")
            if (
                self.repository.casefold() != self._bound_repository.casefold()
                or repo.casefold() != self._bound_repository.casefold()
            ):
                raise ValueError("GitHub repository name does not match bound repository")
            repo = self._bound_repository
        if not self._safe_repository(repo):
            raise ValueError("Invalid GitHub repository name")
        return repo

    def resolve_repository(self, repository_id: int) -> str:
        self._check_repository_id(repository_id)
        resp = self._request("GET", f"{self.api_url}/repositories/{repository_id}")
        payload = self._json(resp, "repository")
        if (
            type(payload.get("id")) is not int or payload["id"] != repository_id
            or not self._safe_repository(payload.get("full_name"))
        ):
            raise httpx.RequestError("Invalid GitHub repository response", request=self._safe_request())
        self.repository = payload["full_name"]
        self._bound_repository = self.repository
        self._repository_id = repository_id
        return self.repository

    def _resolve_repo_and_pr(self, *args: Any, **kwargs: Any) -> tuple[str, int]:
        if len(args) == 1:
            return self._repository_for(**kwargs), int(args[0])
        elif len(args) >= 2:
            repo = self._repository_for(args[0], **kwargs)
            return repo, int(args[1])
        repo = self._repository_for(**kwargs)
        pr_num = int(kwargs.get("pr_number", 0))
        return repo, pr_num

    def get_pull_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        repo, pr_number = self._resolve_repo_and_pr(*args, **kwargs)
        url = f"{self.api_url}/repos/{repo}/pulls/{pr_number}"
        resp = self._request("GET", url)
        return self._json(resp, "pull request")

    def list_comments(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        repo, pr_number = self._resolve_repo_and_pr(*args, **kwargs)
        comments_url = httpx.URL(
            f"{self.api_url}/repos/{repo}/issues/{pr_number}/comments"
        ).copy_set_param("per_page", "100")
        url = comments_url
        comments: list[dict[str, Any]] = []
        visited: set[httpx.URL] = set()
        while True:
            visited.add(url)
            resp = self._request("GET", url)
            page_comments = self._json(resp, "comments", list)
            if any(not isinstance(comment, dict) for comment in page_comments):
                raise httpx.RequestError("Invalid GitHub comments response", request=self._safe_request())
            comments.extend(page_comments)
            next_link = next(
                (link["url"] for link in resp.links.values() if "next" in link.get("rel", "").split()),
                None,
            )
            if next_link is None:
                return comments
            if len(visited) >= _MAX_COMMENT_PAGES:
                raise httpx.RequestError(
                    f"GitHub comments pagination exceeded {_MAX_COMMENT_PAGES} pages",
                    request=self._safe_request(),
                )
            next_url = None
            try:
                next_url = resp.url.join(next_link).copy_set_param("per_page", "100")
            except httpx.InvalidURL:
                pass
            if next_url is None:
                raise httpx.RequestError("Invalid GitHub comments pagination link", request=self._safe_request())
            if (
                (next_url.scheme, next_url.host, next_url.port)
                != (comments_url.scheme, comments_url.host, comments_url.port)
                or next_url.path != comments_url.path
                or next_url.userinfo
                or next_url.fragment
            ):
                raise httpx.RequestError("Unsafe GitHub comments pagination link", request=self._safe_request())
            if next_url in visited:
                raise httpx.RequestError("Cyclic GitHub comments pagination link", request=self._safe_request())
            url = next_url

    def find_marker_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        marker = kwargs.get("marker", "<!-- review-bot:architecture:v1 -->")
        reference = None
        if len(args) == 1:
            pr_number = int(args[0])
        elif len(args) == 2:
            if isinstance(args[1], str) and "<!--" in args[1]:
                pr_number = int(args[0])
                marker = args[1]
            else:
                reference = args[0]
                pr_number = int(args[1])
        elif len(args) >= 3:
            reference = args[0]
            pr_number = int(args[1])
            marker = str(args[2])
        else:
            pr_number = int(kwargs.get("pr_number", 0))

        repo = self._repository_for(reference, **kwargs)
        comments = self.list_comments(repo, pr_number)
        matched = []
        for comment in comments:
            body = comment.get("body", "")
            if marker in body:
                if self._bot_login is not None:
                    user = comment.get("user")
                    if (
                        not isinstance(user, dict) or user.get("type") != "Bot"
                        or not isinstance(user.get("login"), str)
                        or user["login"].casefold() != self._bot_login.casefold()
                    ):
                        raise ValueError("marker_comment_not_owned")
                matched.append(comment)
        if len(matched) > 1:
            raise ValueError(f"Multiple comments with marker '{marker}' found on PR {pr_number}")
        return matched[0] if matched else None

    def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) == 2:
            repo = self._repository_for(**kwargs)
            pr_number = int(args[0])
            body = str(args[1])
        elif len(args) >= 3:
            repo = self._repository_for(args[0], **kwargs)
            pr_number = int(args[1])
            body = str(args[2])
        else:
            repo = self._repository_for(**kwargs)
            pr_number = int(kwargs.get("pr_number", 0))
            body = str(kwargs.get("body", ""))

        url = f"{self.api_url}/repos/{repo}/issues/{pr_number}/comments"
        resp = self._request("POST", url, json={"body": body})
        return self._json(resp, "comment")

    def post_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.create_comment(*args, **kwargs)

    def update_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) == 2:
            repo = self._repository_for(**kwargs)
            comment_id = int(args[0])
            body = str(args[1])
        elif len(args) >= 3:
            repo = self._repository_for(args[0], **kwargs)
            comment_id = int(args[1])
            body = str(args[2])
        else:
            repo = self._repository_for(**kwargs)
            comment_id = int(kwargs.get("comment_id", 0))
            body = str(kwargs.get("body", ""))

        url = f"{self.api_url}/repos/{repo}/issues/comments/{comment_id}"
        resp = self._request("PATCH", url, json={"body": body})
        return self._json(resp, "comment")


class MockGitHubClient:
    """In-memory mock GitHub API client for testing and offline simulation."""

    def __init__(self) -> None:
        self.pull_requests: dict[int, dict[str, Any]] = {}
        self.comments: dict[int, dict[str, Any]] = {}
        self._next_comment_id = 1
        self.posted_comments: list[dict[str, Any]] = []
        self.updated_comments: list[dict[str, Any]] = []

    def set_pull_request(
        self,
        pr_number: int,
        state: str = "open",
        head_sha: str = "b" * 40,
        base_sha: str = "a" * 40,
        branch: str = "feature/test",
        labels: list[str] | None = None,
        repository: str = "owner/repo",
    ) -> None:
        self.pull_requests[pr_number] = {
            "number": pr_number,
            "state": state,
            "head": {"sha": head_sha, "ref": branch},
            "base": {"sha": base_sha, "ref": "main"},
            "labels": [{"name": name} for name in (labels or [])],
            "repository": {"full_name": repository},
        }

    def get_pull_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) == 1:
            pr_number = int(args[0])
        elif len(args) >= 2:
            pr_number = int(args[1])
        else:
            pr_number = int(kwargs.get("pr_number", 0))
        return self.pull_requests.get(
            pr_number,
            {
                "number": pr_number,
                "state": "open",
                "head": {"sha": "b" * 40},
                "base": {"sha": "a" * 40},
            },
        )

    def list_comments(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        if len(args) == 1:
            pr_number = int(args[0])
        elif len(args) >= 2:
            pr_number = int(args[1])
        else:
            pr_number = int(kwargs.get("pr_number", 0))
        return [c for c in self.comments.values() if c.get("pr_number") == pr_number]

    def find_marker_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        marker = kwargs.get("marker", "<!-- review-bot:architecture:v1 -->")
        if len(args) == 1:
            pr_number = int(args[0])
        elif len(args) == 2:
            if isinstance(args[1], str) and "<!--" in args[1]:
                pr_number = int(args[0])
                marker = args[1]
            else:
                pr_number = int(args[1])
        elif len(args) >= 3:
            pr_number = int(args[1])
            marker = str(args[2])
        else:
            pr_number = int(kwargs.get("pr_number", 0))

        matched = []
        for c in self.list_comments(pr_number):
            body = c.get("body", "") if isinstance(c, dict) else getattr(c, "body", "")
            if marker in body:
                matched.append(c)
        if len(matched) > 1:
            raise ValueError(f"Multiple comments with marker '{marker}' found on PR {pr_number}")
        return matched[0] if matched else None

    def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) == 2:
            pr_number = int(args[0])
            body = str(args[1])
        elif len(args) >= 3:
            pr_number = int(args[1])
            body = str(args[2])
        else:
            pr_number = int(kwargs.get("pr_number", 0))
            body = str(kwargs.get("body", ""))

        cid = self._next_comment_id
        self._next_comment_id += 1
        record = {
            "id": cid,
            "pr_number": pr_number,
            "body": body,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self.comments[cid] = record
        self.posted_comments.append(record)
        return record

    def post_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.create_comment(*args, **kwargs)

    def update_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if len(args) == 2:
            comment_id = int(args[0])
            body = str(args[1])
        elif len(args) >= 3:
            comment_id = int(args[1])
            body = str(args[2])
        else:
            comment_id = int(kwargs.get("comment_id", 0))
            body = str(kwargs.get("body", ""))

        if comment_id not in self.comments:
            self.comments[comment_id] = {
                "id": comment_id,
                "body": body,
                "created_at": datetime.now(UTC).isoformat(),
            }
        self.comments[comment_id]["body"] = body
        self.comments[comment_id]["updated_at"] = datetime.now(UTC).isoformat()
        self.updated_comments.append(dict(self.comments[comment_id]))
        return self.comments[comment_id]


# Offline alias
OfflineGitHubClient = MockGitHubClient
