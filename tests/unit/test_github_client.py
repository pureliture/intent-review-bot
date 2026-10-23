"""Unit tests for GitHub comment pagination and guarded publication over HTTP."""
from __future__ import annotations

import json
import gzip
import traceback
import zlib
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from review_bot.adapters.github.client import GitHubApiClient, GitHubResponseLimitError
from review_bot.application.publication import publish_review_comment

COMMENTS_URL = "https://api.github.com/repos/org/repo/issues/42/comments"
MARKER = "<!-- review-bot:architecture:v1 -->"
BOT_LOGIN = "architecture-review[bot]"
HttpHandler = Callable[[httpx.Request], httpx.Response]


@pytest.mark.parametrize("encoding,compress", [("gzip", gzip.compress), ("deflate", zlib.compress)])
def test_bounded_compressed_pr_body_decodes_once_and_keeps_response_metadata(encoding, compress):
    payload = {"number": 305, "body": "압축된 PR 원문을 그대로 보존합니다." * 10}
    decoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    compressed = compress(decoded)
    seen = []

    def handler(request):
        seen.append(request)
        assert request.url.path == "/repos/org/repo/pulls/305"
        return httpx.Response(200, headers={"Content-Encoding": encoding,
            "Content-Length": str(len(compressed)), "ETag": 'W/"original-validator"',
            "Content-Type": "application/json; charset=utf-8"}, stream=httpx.ByteStream(compressed))

    client = GitHubApiClient(token="installation-secret", repository="org/repo", transport=httpx.MockTransport(handler))
    response = client._request("GET", "https://api.github.com/repos/org/repo/pulls/305",
        max_response_bytes=len(decoded), timeout_seconds=1)
    assert response.content == decoded and response.json() == payload
    assert response.headers["ETag"] == 'W/"original-validator"'
    assert response.headers["Content-Type"] == "application/json; charset=utf-8"
    assert "Content-Encoding" not in response.headers
    assert response.headers["Content-Length"] == str(len(decoded))
    assert "Authorization" not in response.request.headers
    assert len(seen) == 1


@pytest.mark.parametrize("encoding,compress", [("gzip", gzip.compress), ("deflate", zlib.compress)])
def test_compressed_pr_limit_applies_to_decoded_body_and_redacts_failure(encoding, compress):
    decoded = json.dumps({"number": 305, "body": "response-secret" * 1000}).encode()
    compressed = compress(decoded)
    assert len(compressed) < 512 < len(decoded)
    client = GitHubApiClient(token="installation-secret", repository="org/repo", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, headers={"Content-Encoding": encoding,
            "Content-Length": str(len(compressed))}, stream=httpx.ByteStream(compressed))))
    with pytest.raises(GitHubResponseLimitError, match="source_response_limit_exceeded") as caught:
        client.get_pull_request(305, max_response_bytes=512, timeout_seconds=1)
    rendered = "".join(traceback.format_exception(caught.value))
    assert "response-secret" not in rendered and "installation-secret" not in rendered


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch) -> Callable[[HttpHandler], None]:
    client_type = httpx.Client

    def install(handler: HttpHandler) -> None:
        monkeypatch.setattr(
            httpx, "Client",
            lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs),
        )

    return install


def test_list_comments_requests_100_per_page(mock_http: Callable[[HttpHandler], None]) -> None:
    requests: list[httpx.Request] = []
    comments = [{"id": 1, "body": "First comment"}]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=comments)

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    assert client.list_comments(42) == comments
    assert len(requests) == 1
    assert requests[0].url.params["per_page"] == "100"


def test_list_comments_follows_next_not_last(mock_http: Callable[[HttpHandler], None]) -> None:
    requests: list[httpx.Request] = []
    comments = [{"id": n, "body": f"Comment {n}"} for n in range(1, 151)]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page") is None:
            return httpx.Response(200, json=comments[:100], headers={
                "Link": f'<{COMMENTS_URL}?page=9>; rel="last", <{COMMENTS_URL}?page=2>; rel="next"',
            })
        assert request.url.params["page"] == "2"
        return httpx.Response(200, json=comments[100:], headers={
            "Link": f'<{COMMENTS_URL}?page=1>; rel="prev"',
        })

    mock_http(handler)
    client = GitHubApiClient(token="test-token", repository="org/repo")

    assert client.list_comments(42) == comments
    assert len(requests) == 2
    assert all(request.url.params["per_page"] == "100" for request in requests)
    assert all(request.headers["Authorization"] == "Bearer test-token" for request in requests)


def test_list_comments_accepts_relative_enterprise_next_link(
    mock_http: Callable[[HttpHandler], None],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.host == "github.example"
        assert request.url.path == "/api/v3/repos/org/repo/issues/42/comments"
        if request.url.params.get("page") is None:
            return httpx.Response(200, json=[{"id": 1}], headers={
                "Link": '<?page=2&per_page=1>; rel="next last"',
            })
        assert request.url.params["page"] == "2"
        return httpx.Response(200, json=[{"id": 2}])

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo", api_url="https://github.example/api/v3")

    assert client.list_comments(42) == [{"id": 1}, {"id": 2}]
    assert len(requests) == 2
    assert all(request.url.params["per_page"] == "100" for request in requests)


@pytest.mark.parametrize("next_link", [
    "https://attacker.example/repos/org/repo/issues/42/comments?page=2",
    "//attacker.example/repos/org/repo/issues/42/comments?page=2",
    "http://api.github.com/repos/org/repo/issues/42/comments?page=2",
    "https://api.github.com/repos/org/other/issues/42/comments?page=2",
    "https://user:password@api.github.com/repos/org/repo/issues/42/comments?page=2",
    "?page=2#fragment",
    "https://api.github.com:bad/repos/org/repo/issues/42/comments?page=2",
])
def test_list_comments_rejects_unsafe_next_link(
    mock_http: Callable[[HttpHandler], None], next_link: str,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[{"id": 1}], headers={
            "Link": f'<{next_link}>; rel="next"',
        })

    mock_http(handler)
    client = GitHubApiClient(token="test-token", repository="org/repo")

    with pytest.raises(httpx.HTTPError, match="pagination") as caught:
        client.list_comments(42)
    assert len(requests) == 1
    assert "Authorization" not in caught.value.request.headers


def test_list_comments_rejects_pagination_cycle(mock_http: Callable[[HttpHandler], None]) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        next_link = (
            f"{COMMENTS_URL}?per_page=100"
            if request.url.params.get("page") == "2"
            else f"{COMMENTS_URL}?page=2&per_page=100"
        )
        return httpx.Response(200, json=[{"id": len(requests)}], headers={
            "Link": f'<{next_link}>; rel="next"',
        })

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    with pytest.raises(httpx.HTTPError, match="pagination"):
        client.list_comments(42)
    assert len(requests) == 2


@pytest.mark.parametrize("has_more", [False, True])
def test_list_comments_bounds_pagination_without_returning_partial_history(
    mock_http: Callable[[HttpHandler], None], has_more: bool,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        page = len(requests)
        assert page <= 100, "Pagination continued beyond 100 pages"
        headers = {"Link": f'<?page={page + 1}>; rel="next"'} if page < 100 or has_more else {}
        return httpx.Response(200, json=[{"id": page}], headers=headers)

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    if has_more:
        with pytest.raises(httpx.RequestError, match="pagination.*100"):
            client.list_comments(42)
    else:
        assert client.list_comments(42) == [{"id": page} for page in range(1, 101)]
    assert len(requests) == 100


@pytest.mark.parametrize("payload", [{"message": "Unexpected object"}, "not a list", None, [None]])
def test_list_comments_rejects_invalid_response_payload(
    mock_http: Callable[[HttpHandler], None], payload: Any,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps(payload))

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    with pytest.raises(httpx.RequestError, match="comments response"):
        client.list_comments(42)


def test_list_comments_does_not_return_partial_history_on_http_failure(
    mock_http: Callable[[HttpHandler], None],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page") == "2":
            return httpx.Response(503, json={"message": "Service unavailable"})
        return httpx.Response(200, json=[{"id": 1}], headers={
            "Link": '<?page=2>; rel="next"',
        })

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    with pytest.raises(httpx.HTTPStatusError):
        client.list_comments(42)
    assert len(requests) == 2


def test_find_marker_comment_checks_duplicates_across_pages(
    mock_http: Callable[[HttpHandler], None],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[{"id": 120, "body": MARKER}])
        return httpx.Response(200, json=[{"id": 5, "body": MARKER}], headers={
            "Link": '<?page=2>; rel="next"',
        })

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    with pytest.raises(ValueError, match="Multiple comments"):
        client.find_marker_comment(42)


def test_publication_updates_marker_comment_on_second_page(
    mock_http: Callable[[HttpHandler], None],
) -> None:
    requests: list[httpx.Request] = []
    comments = [
        {"id": n, "body": MARKER if n == 120 else f"Comment {n}"}
        for n in range(1, 151)
    ]
    updated_bodies: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/repos/org/repo/pulls/42":
            return httpx.Response(200, json={
                "state": "open", "head": {"sha": "b" * 40}, "base": {"sha": "a" * 40},
            })
        if request.method == "GET" and request.url.path == "/repos/org/repo/issues/42/comments":
            if request.url.params.get("page") == "2":
                return httpx.Response(200, json=comments[100:])
            return httpx.Response(200, json=comments[:100], headers={
                "Link": '<?page=2>; rel="next"',
            })
        if request.method == "PATCH" and request.url.path == "/repos/org/repo/issues/comments/120":
            updated_bodies.append(json.loads(request.content)["body"])
            return httpx.Response(200, json={"id": 120})
        raise AssertionError(f"Unexpected GitHub API request: {request.method} {request.url}")

    mock_http(handler)
    client = GitHubApiClient(repository="org/repo")

    receipt = publish_review_comment(
        client, repository_id=123, pr_number=42, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Updated architecture review",
    )

    assert receipt.status == "succeeded"
    assert receipt.comment_id == 120
    assert updated_bodies == [f"{MARKER}\n\nUpdated architecture review"]
    assert not any(request.method == "POST" for request in requests)
    assert len(requests) == 5


@pytest.mark.parametrize("author", [
    {"type": "User", "login": BOT_LOGIN},
    {"type": "Bot", "login": "other-review[bot]"},
    None, {}, "author-response-secret",
    {"type": "Bot"}, {"type": "Bot", "login": None},
    {"type": "bot", "login": BOT_LOGIN},
])
def test_app_bound_marker_comment_ownership_blocks_update_and_creation(author: Any) -> None:
    requests: list[httpx.Request] = []
    comment = {"id": 88, "body": MARKER}
    if author is not None:
        comment["user"] = author

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/repos/org/repo/pulls/42":
            return httpx.Response(200, json={
                "state": "open", "head": {"sha": "b" * 40}, "base": {"sha": "a" * 40},
            })
        if request.method == "GET" and request.url.path == "/repos/org/repo/issues/42/comments":
            return httpx.Response(200, json=[comment])
        return httpx.Response(201, json={"id": 88})

    client = GitHubApiClient(
        repository="org/repo", repository_id=123, bot_login=BOT_LOGIN,
        token_provider=lambda: "installation-token", transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="^marker_comment_not_owned$"):
        client.find_marker_comment(123, 42, MARKER)
    receipt = publish_review_comment(
        client, repository_id=123, pr_number=42, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Architecture review",
    )
    assert receipt.status == "failed"
    assert requests
    assert all(request.method == "GET" for request in requests)


@pytest.mark.parametrize("operation", ["create", "update", "update_casefold"])
def test_app_bound_marker_comment_ownership_allows_only_own_bot_upsert(operation: str) -> None:
    requests: list[httpx.Request] = []
    comments = [{
        "id": 87, "body": "Unrelated human comment", "user": {"type": "User", "login": "human"},
    }]
    if operation != "create":
        comments.append({
            "id": 88, "body": MARKER,
            "user": {
                "type": "Bot", "login": BOT_LOGIN.upper() if operation == "update_casefold" else BOT_LOGIN,
            },
        })

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/repos/org/repo/pulls/42":
            return httpx.Response(200, json={
                "state": "open", "head": {"sha": "b" * 40}, "base": {"sha": "a" * 40},
            })
        if request.method == "GET" and request.url.path == "/repos/org/repo/issues/42/comments":
            return httpx.Response(200, json=comments)
        return httpx.Response(201 if request.method == "POST" else 200, json={"id": 88})

    client = GitHubApiClient(
        repository="org/repo", repository_id=123, bot_login=BOT_LOGIN,
        token_provider=lambda: "installation-token", transport=httpx.MockTransport(handler),
    )
    receipt = publish_review_comment(
        client, repository_id=123, pr_number=42, expected_head_sha="b" * 40,
        expected_base_sha="a" * 40, body="Architecture review",
    )
    assert receipt.status == "succeeded"
    assert receipt.comment_id == 88
    writes = [request for request in requests if request.method != "GET"]
    assert len(writes) == 1
    assert writes[0].method == ("POST" if operation == "create" else "PATCH")
    assert writes[0].url.path == (
        "/repos/org/repo/issues/42/comments" if operation == "create"
        else "/repos/org/repo/issues/comments/88"
    )
    assert json.loads(writes[0].content) == {"body": f"{MARKER}\n\nArchitecture review"}


@pytest.mark.parametrize("other_author", [
    {"type": "User", "login": "human"}, {"type": "Bot", "login": "other-review[bot]"}, None,
])
def test_app_bound_marker_comment_ownership_checks_all_pages(other_author: Any) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=[{"id": 89, "body": MARKER, "user": other_author}])
        return httpx.Response(200, json=[{
            "id": 88, "body": MARKER, "user": {"type": "Bot", "login": BOT_LOGIN},
        }], headers={"Link": '<?page=2>; rel="next"'})

    client = GitHubApiClient(
        repository="org/repo", repository_id=123, bot_login=BOT_LOGIN,
        token_provider=lambda: "installation-token", transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="^marker_comment_not_owned$"):
        client.find_marker_comment(123, 42, MARKER)
    assert len(requests) == 2
    assert all(request.method == "GET" for request in requests)


def test_app_bound_marker_comment_ownership_preserves_duplicate_guard() -> None:
    client = GitHubApiClient(
        repository="org/repo", repository_id=123, bot_login=BOT_LOGIN,
        token_provider=lambda: "installation-token",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[
            {"id": n, "body": MARKER, "user": {"type": "Bot", "login": BOT_LOGIN}}
            for n in (88, 89)
        ])),
    )
    with pytest.raises(ValueError, match="Multiple comments"):
        client.find_marker_comment(123, 42, MARKER)


def test_legacy_marker_comment_with_no_bot_login_retains_existing_behavior() -> None:
    comment = {"id": 88, "body": MARKER, "user": {"type": "User", "login": "human"}}
    client = GitHubApiClient(
        token="static-token", repository="org/repo", bot_login=None,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[comment])),
    )
    assert client.find_marker_comment(42) == comment


@pytest.mark.parametrize("bot_login", [
    "", "human", "wrong[bot]?constructor-secret", "bad\n[bot]", "bad[bot]extra",
    123, False, {}, "봇[bot]", "Kbot[bot]",
])
def test_invalid_bot_login_is_rejected_without_http(bot_login: Any) -> None:
    with pytest.raises(ValueError, match="^Invalid GitHub bot login$") as caught:
        GitHubApiClient(
            repository="org/repo", bot_login=bot_login,
            transport=httpx.MockTransport(lambda request: pytest.fail("Invalid bot login made HTTP request")),
        )
    assert "constructor-secret" not in str(caught.value)


def test_token_provider_refreshes_each_http_request() -> None:
    requests: list[httpx.Request] = []
    tokens = iter(f"request-token-{n}" for n in range(1, 7))

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/repositories/123":
            return httpx.Response(200, json={"id": 123, "full_name": "org/repo"})
        if request.url.path == "/repos/org/repo/pulls/42":
            return httpx.Response(200, json={"number": 42})
        if request.method == "GET":
            if request.url.params.get("page") == "2":
                return httpx.Response(200, json=[{"id": 2}])
            return httpx.Response(200, json=[{"id": 1}], headers={
                "Link": '<?page=2>; rel="next"',
            })
        return httpx.Response(201 if request.method == "POST" else 200, json={"id": 88})

    client = GitHubApiClient(
        token="stale-static-token", token_provider=lambda: next(tokens),
        transport=httpx.MockTransport(handler), repository_id=123,
    )
    assert client.resolve_repository(123) == "org/repo"
    assert client.repository == "org/repo"
    assert client.repository_id == 123
    assert client.get_pull_request(123, 42) == {"number": 42}
    assert client.list_comments(repository_id=123, pr_number=42) == [{"id": 1}, {"id": 2}]
    assert client.post_comment(123, 42, "Review") == {"id": 88}
    assert client.update_comment(123, 88, "Updated review") == {"id": 88}
    assert [request.headers["Authorization"] for request in requests] == [
        f"Bearer request-token-{n}" for n in range(1, 7)
    ]


@pytest.mark.parametrize("operation", [
    lambda client: client.resolve_repository(124),
    lambda client: client.get_pull_request(124, 42),
    lambda client: client.get_pull_request(repository_id=124, pr_number=42),
    lambda client: client.get_pull_request(123, 42, repository_id=124),
    lambda client: client.get_pull_request("other/repo", 42),
    lambda client: client.get_pull_request(123, 42, repository="other/repo"),
    lambda client: client.list_comments(124, 42),
    lambda client: client.list_comments(repository="org/other", pr_number=42),
    lambda client: client.find_marker_comment(124, 42, MARKER),
    lambda client: client.find_marker_comment(repository_id=124, pr_number=42),
    lambda client: client.create_comment(124, 42, "Review"),
    lambda client: client.post_comment(124, 42, "Review"),
    lambda client: client.post_comment(repository_id=124, pr_number=42, body="Review"),
    lambda client: client.post_comment("other/repo", 42, "Review"),
    lambda client: client.update_comment(124, 88, "Review"),
    lambda client: client.update_comment(repository_id=124, comment_id=88, body="Review"),
    lambda client: client.update_comment("org/other", 88, "Review"),
])
def test_bound_repository_mismatch_is_rejected_before_http(operation: Callable[..., Any]) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        pytest.fail("Repository mismatch must not make an HTTP request")

    client = GitHubApiClient(
        token="test-token", repository="org/repo", repository_id=123,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ValueError, match="repository"):
        operation(client)
    assert requests == []


def test_bound_repository_cannot_be_changed_by_mutating_name() -> None:
    client = GitHubApiClient(
        repository="org/repo", repository_id=123,
        transport=httpx.MockTransport(lambda request: pytest.fail("Changed repository made HTTP request")),
    )
    client.repository = "org/other"
    with pytest.raises(ValueError, match="repository"):
        client.post_comment(42, "Review")
    with pytest.raises(AttributeError):
        client.repository_id = 124


@pytest.mark.parametrize("payload", [
    {"id": 124, "full_name": "org/repo"}, {"id": "123", "full_name": "org/repo"},
    {"id": 123}, [], None,
    *({"id": 123, "full_name": name} for name in [
        "", "repo", "org/repo/other", "org/..", "org/.", "../repo",
        "org/repo?token=response-secret", "org/repo#response-secret", "org/%2frepo",
        "org\\evil/repo", "org/repo\nresponse-secret",
    ]),
])
def test_resolve_repository_rejects_invalid_metadata_without_binding(payload: Any) -> None:
    client = GitHubApiClient(
        token="installation-secret",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text=json.dumps(payload))),
    )
    with pytest.raises(httpx.HTTPError, match="repository") as caught:
        client.resolve_repository(123)
    assert client.repository == ""
    assert client.repository_id is None
    assert "response-secret" not in str(caught.value)
    assert "installation-secret" not in repr(caught.value)
    assert "Authorization" not in caught.value.request.headers


def test_resolve_repository_rejects_boolean_response_id() -> None:
    client = GitHubApiClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"id": True, "full_name": "org/repo"}),
    ))
    with pytest.raises(httpx.HTTPError):
        client.resolve_repository(1)


@pytest.mark.parametrize("repository_id", [0, -1, True, "123", None])
def test_resolve_repository_rejects_invalid_ids_before_http(repository_id: Any) -> None:
    client = GitHubApiClient(transport=httpx.MockTransport(
        lambda request: pytest.fail("Invalid repository ID made HTTP request"),
    ))
    with pytest.raises(ValueError, match="repository"):
        client.resolve_repository(repository_id)


@pytest.mark.parametrize("name", ["org/.github", "owner-name/repo_1.2-3"])
def test_resolve_repository_accepts_safe_names(name: str) -> None:
    client = GitHubApiClient(token="test-token", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"id": 123, "full_name": name}),
    ))
    assert client.resolve_repository(123) == name


@pytest.mark.parametrize("operation", [
    lambda client: client.resolve_repository(123),
    lambda client: client.get_pull_request(123, 42),
    lambda client: client.list_comments(123, 42),
    lambda client: client.post_comment(123, 42, "Review"),
    lambda client: client.update_comment(123, 88, "Review"),
])
@pytest.mark.parametrize("status", [401, 403, 500, 302, 307])
def test_client_http_failures_are_sanitized_without_retry_or_redirect(
    operation: Callable[..., Any], status: int, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text="response-secret", headers={
            "Location": "https://attacker.example/steal", "X-Secret": "response-secret",
        })

    client = GitHubApiClient(
        token="installation-secret", repository="org/repo", repository_id=123,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError) as caught:
        operation(client)
    assert len(requests) == 1
    error = caught.value
    assert error.response.status_code == status
    assert "Authorization" not in error.request.headers
    assert error.response.content == b""
    assert "X-Secret" not in error.response.headers
    rendered = "".join(traceback.format_exception(error)) + repr(error)
    assert "installation-secret" not in rendered
    assert "response-secret" not in rendered
    assert error.__context__ is None
    assert error.__cause__ is None
    assert "installation-secret" not in caplog.text
    assert "response-secret" not in caplog.text


@pytest.mark.parametrize("operation", [
    lambda client: client.resolve_repository(123),
    lambda client: client.get_pull_request(123, 42),
    lambda client: client.list_comments(123, 42),
    lambda client: client.post_comment(123, 42, "Review"),
    lambda client: client.update_comment(123, 88, "Review"),
])
def test_client_invalid_json_is_sanitized(operation: Callable[..., Any]) -> None:
    client = GitHubApiClient(
        token="installation-secret", repository="org/repo", repository_id=123,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="response-secret")),
    )
    with pytest.raises(httpx.HTTPError) as caught:
        operation(client)
    error = caught.value
    assert "Authorization" not in error.request.headers
    assert "response-secret" not in "".join(traceback.format_exception(error))
    assert error.__context__ is None


@pytest.mark.parametrize("failure", ["transport", "provider", "invalid-token"])
def test_client_transport_and_callback_failures_are_redacted(failure: str) -> None:
    requests: list[httpx.Request] = []

    def token_provider() -> str:
        if failure == "provider":
            raise ValueError("provider-secret")
        if failure == "invalid-token":
            return "provider-secret\n"
        return "installation-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("transport-secret " + request.headers["Authorization"], request=request)

    client = GitHubApiClient(
        token_provider=token_provider, repository="org/repo", transport=httpx.MockTransport(handler),
    )
    with pytest.raises((httpx.HTTPError, RuntimeError)) as caught:
        client.get_pull_request(42)
    error = caught.value
    rendered = "".join(traceback.format_exception(error)) + repr(error)
    for secret in ("provider-secret", "installation-secret", "transport-secret"):
        assert secret not in rendered
    assert error.__context__ is None
    assert error.__cause__ is None
    assert len(requests) == (1 if failure == "transport" else 0)


@pytest.mark.parametrize("api_url", [
    "http://api.github.com", "https://attacker.example", "https://api.github.com:444",
    "https://api.github.com?secret=provider-secret", "https://user@api.github.com",
])
def test_token_provider_client_rejects_unsafe_origin(api_url: str) -> None:
    with pytest.raises(RuntimeError) as caught:
        GitHubApiClient(token_provider=lambda: "installation-secret", api_url=api_url)
    assert "provider-secret" not in str(caught.value)
    assert "installation-secret" not in repr(caught.value)


def test_bound_marker_lookup_preserves_comment_author_and_marker() -> None:
    comments = [
        {"id": 1, "body": "Human comment", "user": {"login": "alice", "type": "User"}},
        {"id": 2, "body": f"{MARKER}\nReview", "user": {"login": "review-app[bot]", "type": "Bot"}},
    ]
    client = GitHubApiClient(
        repository="org/repo", repository_id=123,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=comments)),
    )
    assert client.find_marker_comment(123, 42, MARKER) == comments[1]
    assert client.find_marker_comment(42, "<!-- other-marker -->") is None


def test_resolved_repository_cannot_be_rebound_to_another_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": 123, "full_name": "org/repo"})

    client = GitHubApiClient(token="test-token", transport=httpx.MockTransport(handler))
    assert client.resolve_repository(123) == "org/repo"
    with pytest.raises(ValueError, match="repository"):
        client.resolve_repository(124)
    with pytest.raises(ValueError, match="repository"):
        client.post_comment(124, 42, "Review")
    assert client.repository_id == 123
    assert client.repository == "org/repo"
    assert len(requests) == 1


def test_bound_repository_requires_a_name_before_publication() -> None:
    client = GitHubApiClient(repository_id=123, transport=httpx.MockTransport(
        lambda request: pytest.fail("Unresolved repository made HTTP request"),
    ))
    with pytest.raises(ValueError, match="resolved"):
        client.post_comment(123, 42, "Review")


def test_mutating_api_url_does_not_send_provider_token_to_another_origin() -> None:
    tokens: list[str] = []

    def token_provider() -> str:
        tokens.append("installation-secret")
        return tokens[-1]

    client = GitHubApiClient(
        repository="org/repo", token_provider=token_provider,
        transport=httpx.MockTransport(lambda request: pytest.fail("Unsafe origin made HTTP request")),
    )
    client.api_url = "https://attacker.example"
    with pytest.raises(RuntimeError):
        client.get_pull_request(42)
    assert tokens == []


@pytest.mark.parametrize("token", ["", None, 123, "installation-secret\n"])
def test_invalid_provider_token_never_falls_back_to_static_token(token: Any) -> None:
    client = GitHubApiClient(
        token="fallback-secret", repository="org/repo", token_provider=lambda: token,
        transport=httpx.MockTransport(lambda request: pytest.fail("Invalid token made HTTP request")),
    )
    with pytest.raises(RuntimeError) as caught:
        client.get_pull_request(42)
    assert "fallback-secret" not in repr(caught.value)
    assert "installation-secret" not in str(caught.value)


def test_static_token_positional_api_and_enterprise_url_remain_compatible() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer legacy-token"
        assert str(request.url) == "https://github.example/api/v3/repos/org/repo/pulls/42"
        return httpx.Response(200, json={"number": 42})

    client = GitHubApiClient(
        "legacy-token", "org/repo", "https://github.example/api/v3", 5.0,
        transport=httpx.MockTransport(handler),
    )
    assert client.get_pull_request(42) == {"number": 42}
    assert client.get_pull_request(123, 42) == {"number": 42}
    assert client.get_pull_request("org/repo", 42) == {"number": 42}
    assert client.get_pull_request(repository="org/repo", pr_number=42) == {"number": 42}
    assert len(requests) == 4
