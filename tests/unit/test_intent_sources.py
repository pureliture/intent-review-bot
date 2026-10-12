"""Current-permission, same-repository text collection and safe missing-source records."""
from dataclasses import asdict

import httpx
import pytest

from review_bot.adapters.github.auth import GitHubAppAuthError
from review_bot.adapters.github.client import GitHubApiClient, GitHubResponseLimitError
from review_bot.adapters.github.intent_sources import IntentCollectionAborted, collect_intent_sources
from review_bot.application.intent_context import evidence_digest


def client_for(issues, *, status=200):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.url.host == "api.github.com"
        assert request.url.path.startswith("/repos/org/repo/issues/")
        if isinstance(status, Exception):
            raise status
        number = int(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(status, json=issues.get(number, {}))

    return GitHubApiClient("synthetic-token", "org/repo", repository_id=99,
        transport=httpx.MockTransport(handler)), calls


def issue(number=12, body="Fixes #99; issue bodies must not cause recursion"):
    return {"number": number, "title": "Keep handling serialized", "body": body,
        "updated_at": "2026-10-01T12:00:00Z"}


def collect(client, body, **kwargs):
    return collect_intent_sources(client, "org/repo", 7,
        {"title": "Serialize handling", "body": body, "updated_at": "2026-10-01T10:00:00Z"}, **kwargs)


def reasons(result):
    return {item.source: item.reason for item in result.unavailable_sources}


def test_same_repo_references_are_deduplicated_and_not_recursive():
    client, calls = client_for({12: issue()})
    result = collect(client, "Fix #12, org/repo#12 and https://github.com/org/repo/issues/12")
    assert [request.url.path for request in calls] == ["/repos/org/repo/issues/12"]
    assert reasons(result) == {}
    assert len(result.evidence) == 2
    original = result.evidence[1]
    assert original.kind == "issue_text" and original.content["body"] == issue()["body"]
    assert original.digest == original.version == evidence_digest(original.content, "issue_text")
    assert original.locator["updated_at"] == issue()["updated_at"]
    assert original.locator["repository"] == "org/repo" and original.locator["issue_number"] == 12
    assert calls[0].extensions["timeout"]["read"] <= 10


@pytest.mark.parametrize("body, expected", [
    ("other/repo#12", "cross_repository_reference"),
    ("https://github.com/other/repo/issues/12", "cross_repository_reference"),
    ("https://evil.invalid/org/repo/issues/12", "unsupported_reference"),
    ("http://github.com/org/repo/issues/12", "unsupported_reference"),
    ("https://github.com@evil.invalid/org/repo/issues/12", "unsupported_reference"),
    ("https://github.com/org/repo/pull/12", "unsupported_reference"),
    ("https://github.com/org/repo/issues/12?token=secret", "unsupported_reference"),
    ("https://github.com:443/org/repo/issues/12", "unsupported_reference"),
])
def test_untrusted_targets_never_drive_http(body, expected):
    client, calls = client_for({})
    result = collect(client, body)
    assert calls == []
    assert list(reasons(result).values()) == [expected]
    assert body not in str([asdict(item) for item in result.unavailable_sources])


@pytest.mark.parametrize("status, expected", [(403, "forbidden"), (404, "not_found"), (500, "request_failed")])
def test_unavailable_issues_do_not_widen_permissions(status, expected):
    client, calls = client_for({}, status=status)
    result = collect(client, "Fixes #12")
    assert reasons(result) == {"issue:12": expected}
    assert len(calls) == 1 and len(result.evidence) == 1


def test_unauthorized_issue_is_a_common_authorization_failure():
    client, calls = client_for({}, status=401)
    with pytest.raises(GitHubAppAuthError):
        collect(client, "Fixes #12 and #13")
    assert len(calls) == 1


def test_reference_and_text_bounds_do_not_truncate_originals():
    client, calls = client_for({12: issue(), 13: issue(13), 14: issue(14)})
    body = "Fixes #12 and #13 and #14"
    result = collect(client, body, max_issues=2)
    assert len(calls) == 2
    assert result.evidence[0].content["body"] == body
    assert reasons(result)["linked_issues"] == "reference_limit_exceeded"
    result = collect(client, "#12", max_text_bytes=30)
    assert reasons(result) == {"issue:12": "text_limit_exceeded"}
    assert [item.kind for item in result.evidence] == ["pr_text"]


def test_stream_response_bytes_are_bounded_without_exposing_content():
    client = GitHubApiClient("synthetic-token", "org/repo", repository_id=99,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"sensitive" * 100)))
    with pytest.raises(GitHubResponseLimitError, match="source_response_limit_exceeded"):
        client.get_issue(12, max_response_bytes=100)


def test_pr_original_response_can_be_bounded_without_changing_legacy_api():
    payload = {"number": 7, "body": "whole" * 200}
    client = GitHubApiClient("synthetic-token", "org/repo", repository_id=99,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    assert client.get_pull_request(7) == payload
    with pytest.raises(GitHubResponseLimitError):
        client.get_pull_request(7, max_response_bytes=100, timeout_seconds=1)


def test_total_original_text_budget_records_omission_without_shortening_text():
    client, calls = client_for({12: issue(body="Complete original details")})
    result = collect(client, "#12", max_total_text_bytes=25)
    assert result.evidence[0].content["body"] == "#12"
    assert reasons(result) == {"issue:12": "text_limit_exceeded"}


def test_authorization_failures_are_observable_without_exposing_credentials():
    client, calls = client_for({}, status=401)
    for count in (1, 2):
        with pytest.raises(httpx.HTTPStatusError):
            client.get_issue(12)
        assert client.authorization_failures == count


def test_historical_text_is_not_reconstructed_from_current_versions():
    old = issue()
    old["updated_at"] = "2026-10-05T12:00:00Z"
    client, calls = client_for({12: old})
    result = collect(client, "Fixes #12", as_of="2026-10-02T00:00:00Z")
    assert reasons(result) == {"issue:12": "historical_text_unavailable"}
    assert len(result.evidence) == 1
    result = collect_intent_sources(client, "org/repo", 7,
        {"title": "Future PR", "body": "Fixes #12", "updated_at": "2026-10-05T00:00:00Z"},
        as_of="2026-10-02T00:00:00Z")
    assert len(calls) == 1, "References in unavailable future PR text are not historical links"
    assert result.evidence == ()
    assert reasons(result)["pr_text"] == "historical_text_unavailable"


def test_absent_references_and_issue_version_are_explicit():
    client, calls = client_for({12: {"number": 12, "title": "No version", "body": "Details"}})
    assert reasons(collect(client, "No linked issue")) == {"linked_issues": "no_explicit_references"}
    assert calls == []
    assert reasons(collect(client, "#12")) == {"issue:12": "text_version_unavailable"}


def test_gate_and_repository_binding_stop_all_reads():
    client, calls = client_for({12: issue()})
    with pytest.raises(IntentCollectionAborted):
        collect(client, "#12", before_request=lambda: False)
    client.repository = "other/repo"
    with pytest.raises(GitHubAppAuthError):
        collect(client, "#12")
    assert calls == []


def test_timeout_reason_does_not_contain_exception_or_secret_values():
    client, calls = client_for({}, status=httpx.ReadTimeout("source-secret"))
    result = collect(client, "#12")
    assert reasons(result) == {"issue:12": "request_timeout"}
    assert "source-secret" not in str(result)


@pytest.mark.parametrize("number", [True, 0, -1, "12", 2**32])
def test_issue_numbers_are_strict_before_any_http(number):
    client, calls = client_for({})
    with pytest.raises(ValueError):
        client.get_issue(number)
    assert calls == []


def test_issue_endpoint_returning_a_pr_is_not_issue_evidence():
    data = issue()
    data["pull_request"] = {"url": "https://github.com/org/repo/pull/12"}
    client, calls = client_for({12: data})
    assert reasons(collect(client, "#12")) == {"issue:12": "unsupported_reference"}