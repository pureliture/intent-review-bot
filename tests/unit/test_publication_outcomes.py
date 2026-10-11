"""Unacknowledged writes are unknown, never falsely reported as absent or fresh."""
from __future__ import annotations

import httpx
import pytest

from review_bot.adapters.github.client import GitHubApiClient
from review_bot.application.publication import publish_review_comment


@pytest.mark.parametrize("existing", [False, True])
def test_write_transport_failure_is_unknown_and_redacted(existing: bool, caplog: pytest.LogCaptureFixture) -> None:
    writes = []

    def network(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/pulls/7"):
            return httpx.Response(200, json={"state": "open", "head": {"sha": "b" * 40}, "base": {"sha": "a" * 40}})
        if request.method == "GET":
            comments = [{"id": 701, "body": "<!-- review-bot:architecture:v1 -->", "user": {"type": "Bot", "login": "review-app[bot]"}}] if existing else []
            return httpx.Response(200, json=comments)
        writes.append(request.method)
        raise httpx.ReadTimeout("synthetic-sensitive-value", request=request)

    client = GitHubApiClient(repository="org/repo", token="synthetic-token", transport=httpx.MockTransport(network), bot_login="review-app[bot]")
    receipt = publish_review_comment(client, 99, 7, "b" * 40, "a" * 40, "Prepared review")
    assert receipt.status == "unknown"
    assert writes == ["PATCH" if existing else "POST"]
    assert "synthetic-sensitive-value" not in receipt.reason
    assert "synthetic-sensitive-value" not in caplog.text