"""Unit tests for Guarded Publication application service."""
from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import httpx
import pytest

from review_bot.adapters.github.client import MockGitHubClient
from review_bot.application.publication import publish_review_comment

MARKER = "<!-- review-bot:architecture:v1 -->"


def test_publication_precheck_pr_closed() -> None:
    client = MockGitHubClient()
    client.set_pull_request(pr_number=1, state="closed", head_sha="b" * 40, base_sha="a" * 40)

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body="Review body",
    )
    assert receipt.status == "failed"
    assert receipt.reason == "pr_not_open"
    assert receipt.comment_id is None
    assert len(client.comments) == 0


def test_publication_precheck_head_sha_mismatch() -> None:
    client = MockGitHubClient()
    client.set_pull_request(pr_number=1, state="open", head_sha="c" * 40, base_sha="a" * 40)

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body="Review body",
    )
    assert receipt.status == "failed"
    assert receipt.reason == "stale_head_sha"
    assert receipt.comment_id is None
    assert len(client.comments) == 0


def test_publication_precheck_base_sha_mismatch() -> None:
    client = MockGitHubClient()
    client.set_pull_request(pr_number=1, state="open", head_sha="b" * 40, base_sha="z" * 40)

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body="Review body",
    )
    assert receipt.status == "failed"
    assert receipt.reason == "stale_base_sha"
    assert receipt.comment_id is None
    assert len(client.comments) == 0


def test_publication_creates_new_comment_when_none_exists() -> None:
    client = MockGitHubClient()
    client.set_pull_request(pr_number=1, state="open", head_sha="b" * 40, base_sha="a" * 40)

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body="### Review Output",
    )
    assert receipt.status == "succeeded"
    assert receipt.comment_id is not None
    assert receipt.reason == "published"
    assert receipt.marker == MARKER
    assert len(client.comments) == 1

    created_comment = client.comments[receipt.comment_id]
    assert created_comment["body"].startswith(MARKER)
    assert "### Review Output" in created_comment["body"]


def test_publication_updates_existing_marker_comment_idempotently() -> None:
    client = MockGitHubClient()
    client.set_pull_request(pr_number=1, state="open", head_sha="b" * 40, base_sha="a" * 40)

    # Initial comment created
    first = client.create_comment(1, f"{MARKER}\nOld review")
    initial_id = first["id"]
    assert len(client.comments) == 1

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body=f"{MARKER}\nUpdated review",
    )
    assert receipt.status == "succeeded"
    assert receipt.comment_id == initial_id
    assert len(client.comments) == 1  # No duplicate created!
    assert client.comments[initial_id]["body"] == f"{MARKER}\nUpdated review"


def test_publication_postcheck_race_head_sha_drift() -> None:
    class RaceMockGitHubClient(MockGitHubClient):
        def __init__(self) -> None:
            super().__init__()
            self._write_count = 0

        def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            res = super().create_comment(*args, **kwargs)
            # Simulate a push occurring during write API call: head moves from b to d
            self.set_pull_request(pr_number=1, state="open", head_sha="d" * 40, base_sha="a" * 40)
            return res

    client = RaceMockGitHubClient()
    client.set_pull_request(pr_number=1, state="open", head_sha="b" * 40, base_sha="a" * 40)

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body="Initial content",
    )
    assert receipt.status == "stale"
    assert receipt.reason == "stale_head_sha"
    assert receipt.comment_id is not None

    # Stale banner must have been applied in-place to comment
    comment_body = client.comments[receipt.comment_id]["body"]
    assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in comment_body


def test_publication_postcheck_race_pr_closed() -> None:
    class ClosedRaceMockGitHubClient(MockGitHubClient):
        def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            res = super().create_comment(*args, **kwargs)
            # Simulate PR being closed during write API call
            self.set_pull_request(pr_number=1, state="closed", head_sha="b" * 40, base_sha="a" * 40)
            return res

    client = ClosedRaceMockGitHubClient()
    client.set_pull_request(pr_number=1, state="open", head_sha="b" * 40, base_sha="a" * 40)

    receipt = publish_review_comment(
        github_client=client,
        repository_id=123,
        pr_number=1,
        expected_head_sha="b" * 40,
        expected_base_sha="a" * 40,
        body="Initial content",
    )
    assert receipt.status == "stale"
    assert receipt.reason == "pr_closed_during_publication"
    assert receipt.comment_id is not None
    assert "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**" in client.comments[receipt.comment_id]["body"]


@pytest.mark.parametrize("operation, reason", [
    ("get_pull_request", "pre_check_failed"),
    ("find_marker_comment", "find_marker_failed"),
    ("create_comment", "create_comment_failed"),
    ("update_comment", "update_comment_failed"),
])
@pytest.mark.parametrize("failure", ["timeout", "http_status"])
def test_publication_http_failure_returns_failed_receipt(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operation: str,
    reason: str,
    failure: str,
) -> None:
    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    comment_id = None
    if operation == "update_comment":
        comment_id = client.create_comment(1, f"{MARKER}\nOld review")["id"]
    request = httpx.Request("GET", "https://api.github.com/repos/org/repo/pulls/1")
    error: httpx.HTTPError
    if failure == "timeout":
        error = httpx.ReadTimeout("GitHub request timed out", request=request)
    else:
        error = httpx.HTTPStatusError(
            "GitHub service unavailable", request=request,
            response=httpx.Response(503, request=request),
        )
    monkeypatch.setattr(client, operation, Mock(side_effect=error))

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    expected_status = "unknown" if failure == "timeout" and operation in ("create_comment", "update_comment") else "failed"
    assert receipt.status == expected_status
    assert receipt.reason.startswith(f"{reason}:")
    assert receipt.comment_id == comment_id
    assert len(receipt.fingerprint) == 64
    assert receipt.marker == MARKER
    assert len(client.comments) == (1 if comment_id is not None else 0)
    if comment_id is not None:
        assert client.comments[comment_id]["body"] == f"{MARKER}\nOld review"
    assert reason in caplog.text


@pytest.mark.parametrize("error", [
    httpx.ReadTimeout("Post-check timed out"),
    RuntimeError("Post-check unavailable"),
    httpx.InvalidURL("Invalid post-check URL"),
    AttributeError("Unexpected post-check error"),
    AssertionError("Unexpected post-check assertion"),
])
def test_publication_postcheck_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    precheck = client.get_pull_request(1)
    monkeypatch.setattr(client, "get_pull_request", Mock(side_effect=[precheck, error]))

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason.startswith("post_check_failed:")
    assert receipt.comment_id in client.comments
    assert len(client.comments) == 1
    assert "post_check_failed" in caplog.text
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("race", ["head_moved", "closed"])
@pytest.mark.parametrize("error", [
    httpx.ReadTimeout("Update timed out"),
    AttributeError("Unexpected stale warning error"),
])
def test_publication_stale_warning_failure_returns_receipt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    race: str, error: Exception,
) -> None:
    class RaceMockGitHubClient(MockGitHubClient):
        def create_comment(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            created = super().create_comment(*args, **kwargs)
            self.set_pull_request(
                1, state="closed" if race == "closed" else "open",
                head_sha="d" * 40, base_sha="a" * 40,
            )
            return created

    client = RaceMockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    monkeypatch.setattr(client, "update_comment", Mock(side_effect=error))

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason.startswith("post_check_failed:")
    assert receipt.comment_id in client.comments
    assert len(client.comments) == 1
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("operation, reason", [
    ("get_pull_request", "pre_check_failed"),
    ("find_marker_comment", "find_marker_failed"),
    ("create_comment", "create_comment_failed"),
    ("update_comment", "update_comment_failed"),
])
@pytest.mark.parametrize("error_type", [AssertionError, AttributeError, httpx.InvalidURL])
def test_publication_unexpected_failure_returns_failed_receipt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    operation: str, reason: str, error_type: type[Exception],
) -> None:
    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    comment_id = None
    if operation == "update_comment":
        comment_id = client.create_comment(1, f"{MARKER}\nOld review")["id"]
    monkeypatch.setattr(client, operation, Mock(side_effect=error_type("Unexpected API error")))

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason.startswith(f"{reason}:")
    assert receipt.comment_id == comment_id
    assert len(client.comments) == (1 if comment_id is not None else 0)
    if comment_id is not None:
        assert client.comments[comment_id]["body"] == f"{MARKER}\nOld review"
    assert reason in caplog.text
    if operation in ("create_comment", "update_comment"):
        assert receipt.reason == f"{reason}:{error_type.__name__}"
        assert error_type.__name__ in caplog.text
        assert "Unexpected API error" not in caplog.text
    else:
        assert any(
            record.exc_info and isinstance(record.exc_info[1], error_type)
            for record in caplog.records
        )


@pytest.mark.parametrize("stage", ["pre_check", "post_check"])
@pytest.mark.parametrize("response", [None, []])
def test_publication_invalid_pr_response_returns_failed_receipt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    stage: str, response: Any,
) -> None:
    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    responses = [response]
    if stage == "post_check":
        responses.insert(0, client.get_pull_request(1))
    monkeypatch.setattr(client, "get_pull_request", Mock(side_effect=responses))

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason.startswith(f"{stage}_failed:")
    if stage == "post_check":
        assert receipt.comment_id in client.comments
        assert len(client.comments) == 1
    else:
        assert receipt.comment_id is None
        assert client.comments == {}
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("field", ["head", "base"])
def test_publication_pr_sha_parsing_failure_returns_failed_receipt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, field: str,
) -> None:
    class InvalidShaInfo(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AttributeError(f"Invalid {field} SHA data")

    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    response = dict(client.get_pull_request(1))
    response[field] = InvalidShaInfo()
    monkeypatch.setattr(client, "get_pull_request", Mock(return_value=response))

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason.startswith("pre_check_failed:")
    assert receipt.comment_id is None
    assert client.comments == {}
    assert any(record.exc_info for record in caplog.records)


@pytest.mark.parametrize("operation, reason", [
    ("find_marker_comment", "update_comment_failed"),
    ("create_comment", "create_comment_failed"),
])
def test_publication_comment_id_parsing_failure_returns_failed_receipt(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    operation: str, reason: str,
) -> None:
    class InvalidComment(dict[str, Any]):
        def get(self, key: str, default: Any = None) -> Any:
            raise AttributeError("Invalid comment ID data")

    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    monkeypatch.setattr(client, operation, Mock(return_value=InvalidComment()))
    update_comment = Mock(wraps=client.update_comment)
    monkeypatch.setattr(client, "update_comment", update_comment)

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason.startswith(f"{reason}:")
    assert receipt.comment_id is None
    assert client.comments == {}
    update_comment.assert_not_called()
    assert receipt.reason == f"{reason}:AttributeError"
    assert "AttributeError" in caplog.text
    assert "Invalid comment ID data" not in caplog.text


@pytest.mark.parametrize("operation", ["create_comment", "find_marker_comment"])
@pytest.mark.parametrize("comment_id", [None, 0, "123", True])
def test_publication_invalid_comment_id_cannot_succeed(
    monkeypatch: pytest.MonkeyPatch, operation: str, comment_id: Any,
) -> None:
    client = MockGitHubClient()
    client.set_pull_request(1, head_sha="b" * 40, base_sha="a" * 40)
    monkeypatch.setattr(client, operation, Mock(return_value={"id": comment_id}))
    update_comment = Mock(wraps=client.update_comment)
    monkeypatch.setattr(client, "update_comment", update_comment)

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=1, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Review body",
    )

    assert receipt.status == "failed"
    assert receipt.reason == "invalid_comment_id"
    assert receipt.comment_id is None
    update_comment.assert_not_called()
