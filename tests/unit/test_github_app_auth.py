"""GitHub App authentication with generated RSA keys and local HTTP transports."""
from __future__ import annotations

import importlib
import json
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from review_bot.adapters.github.auth import GitHubAppAuthError

NOW = 1_760_000_000.0
SECRET = "auth-response-secret"
PERMISSIONS = {"contents": "read", "pull_requests": "write"}


@pytest.fixture(scope="module")
def rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture
def key_file(tmp_path: Path, rsa_key: rsa.RSAPrivateKey) -> Path:
    path = tmp_path / "test-app-key.pem"
    path.write_bytes(rsa_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    return path


def token_response(
    repository_id: int = 123, token: str = SECRET, expires_at: float = NOW + 3600,
) -> dict[str, Any]:
    return {
        "token": token,
        "expires_at": datetime.fromtimestamp(expires_at, UTC).isoformat(),
        "permissions": {**PERMISSIONS, "metadata": "read"},
        "repository_selection": "selected",
        "repositories": [{"id": repository_id, "full_name": "org/repo"}],
    }


def make_provider(
    key_file: Path, handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> Any:
    auth = importlib.import_module("review_bot.adapters.github.auth")
    return auth.GitHubAppTokenProvider(
        "12345", key_file, transport=httpx.MockTransport(handler),
        clock=kwargs.pop("clock", lambda: NOW), **kwargs,
    )


def assert_redacted(error: BaseException, *secrets: str) -> None:
    rendered = "".join(traceback.format_exception(error)) + repr(error)
    for secret in (SECRET, *secrets):
        assert secret not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def test_get_bot_login_authenticates_with_app_jwt_and_caches(
    key_file: Path, rsa_key: rsa.RSAPrivateKey,
) -> None:
    now = [NOW]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": 12345, "slug": "architecture-review"})

    provider = make_provider(key_file, handler, clock=lambda: now[0])
    assert provider.get_bot_login() == "architecture-review[bot]"
    now[0] += 7200
    assert provider.get_bot_login() == "architecture-review[bot]"
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url) == "https://api.github.com/app"
    assert request.content == b""
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    encoded = request.headers["Authorization"].removeprefix("Bearer ")
    assert jwt.get_unverified_header(encoded)["alg"] == "RS256"
    claims = jwt.decode(
        encoded, rsa_key.public_key(), algorithms=["RS256"], issuer="12345",
        options={"verify_exp": False, "verify_iat": False},
    )
    assert claims["iat"] == int(NOW) - 60
    assert int(NOW) < claims["exp"] <= int(NOW) + 600
    assert claims["exp"] - claims["iat"] <= 600
    assert encoded not in repr(provider)
    assert "PRIVATE KEY" not in repr(provider)


def test_get_bot_login_cache_is_provider_local(key_file: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": 12345, "slug": f"review-{len(requests)}"})

    first = make_provider(key_file, handler)
    second = make_provider(key_file, handler)
    assert first.get_bot_login() == "review-1[bot]"
    assert second.get_bot_login() == "review-2[bot]"
    assert first.get_bot_login() == "review-1[bot]"
    assert second.get_bot_login() == "review-2[bot]"
    assert len(requests) == 2


@pytest.mark.parametrize("payload", [
    {"id": 12346, "slug": SECRET}, {"id": "12345", "slug": SECRET},
    {"id": True, "slug": SECRET}, {"id": 12345.0, "slug": SECRET},
    {"slug": SECRET}, {"id": 12345}, {}, None, [], SECRET,
    *({"id": 12345, "slug": slug} for slug in [
        "", "..", "-bot", "org/bot", "bot[bot]", "bot?token=" + SECRET,
        "bot#" + SECRET, "bot\n" + SECRET, "bot\\evil", "%2fbot", "봇", None, 123, [],
    ]),
])
def test_get_bot_login_rejects_invalid_metadata_without_caching(
    key_file: Path, payload: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []
    caplog.set_level("DEBUG")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text=json.dumps(payload))

    provider = make_provider(key_file, handler)
    for _ in range(2):
        with pytest.raises(GitHubAppAuthError, match="metadata") as caught:
            provider.get_bot_login()
        encoded = requests[-1].headers["Authorization"].removeprefix("Bearer ")
        assert_redacted(caught.value, encoded, key_file.read_text())
        assert encoded not in caplog.text
    assert len(requests) == 2
    assert SECRET not in caplog.text


@pytest.mark.parametrize("status_code", [301, 302, 307, 308, 401, 403, 500])
def test_get_bot_login_http_failure_is_redacted_and_not_retried(
    key_file: Path, status_code: int, caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []
    caplog.set_level("DEBUG")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status_code, json={"message": SECRET}, headers={
            "Location": f"https://attacker.example/{SECRET}",
        })

    provider = make_provider(key_file, handler)
    with pytest.raises(GitHubAppAuthError, match=f"HTTP {status_code}") as caught:
        provider.get_bot_login()
    assert len(requests) == 1
    encoded = requests[0].headers["Authorization"].removeprefix("Bearer ")
    assert_redacted(caught.value, encoded, key_file.read_text())
    assert SECRET not in caplog.text
    assert encoded not in caplog.text


def test_get_bot_login_invalid_json_is_redacted(key_file: Path) -> None:
    provider = make_provider(key_file, lambda request: httpx.Response(200, text=SECRET))
    with pytest.raises(GitHubAppAuthError, match="metadata") as caught:
        provider.get_bot_login()
    assert_redacted(caught.value)


def test_get_bot_login_transport_error_is_redacted(key_file: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError(
            SECRET + request.headers["Authorization"], request=request,
        )

    provider = make_provider(key_file, handler)
    with pytest.raises(GitHubAppAuthError) as caught:
        provider.get_bot_login()
    assert len(requests) == 1
    assert_redacted(caught.value, requests[0].headers["Authorization"])


def test_get_bot_login_signing_error_is_redacted(
    key_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_signing(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(SECRET + key_file.read_text())

    monkeypatch.setattr(jwt, "encode", fail_signing)
    provider = make_provider(
        key_file, lambda request: pytest.fail("Invalid JWT must not make an HTTP request"),
    )
    with pytest.raises(GitHubAppAuthError) as caught:
        provider.get_bot_login()
    assert_redacted(caught.value, key_file.read_text())


def test_app_jwt_and_repository_scoped_permissions(
    key_file: Path, rsa_key: rsa.RSAPrivateKey,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_response())

    provider = make_provider(key_file, handler)
    assert provider.get_token(7, 123) == SECRET
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.github.com/app/installations/7/access_tokens"
    assert json.loads(request.content) == {
        "repository_ids": [123], "permissions": PERMISSIONS,
    }
    encoded = request.headers["Authorization"].removeprefix("Bearer ")
    assert jwt.get_unverified_header(encoded)["alg"] == "RS256"
    claims = jwt.decode(
        encoded, rsa_key.public_key(), algorithms=["RS256"], issuer="12345",
        options={"verify_exp": False, "verify_iat": False},
    )
    assert claims["iat"] == int(NOW) - 60
    assert int(NOW) < claims["exp"] <= int(NOW) + 600
    assert claims["exp"] - claims["iat"] <= 600
    assert SECRET not in repr(provider)
    assert SECRET not in repr(provider._cache)
    assert "PRIVATE KEY" not in repr(provider)


def test_tokens_are_cached_and_refreshed_before_expiry(key_file: Path) -> None:
    now = [NOW]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_response(
            token=f"test-token-{len(requests)}", expires_at=now[0] + 3600,
        ))

    provider = make_provider(key_file, handler, clock=lambda: now[0])
    assert provider.get_token(7, 123) == "test-token-1"
    now[0] += 3539
    assert provider.get_token(7, 123) == "test-token-1"
    assert len(requests) == 1
    now[0] += 1
    assert provider.get_token(7, 123) == "test-token-2"
    assert len(requests) == 2
    now[0] += 3601
    assert provider.get_token(7, 123) == "test-token-3"


def test_cache_is_isolated_by_installation_and_repository(key_file: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        repository_id = json.loads(request.content)["repository_ids"][0]
        installation_id = request.url.path.split("/")[3]
        return httpx.Response(201, json=token_response(
            repository_id, token=f"token-{installation_id}-{repository_id}",
        ))

    provider = make_provider(key_file, handler)
    for _ in range(2):
        assert provider.get_token(7, 123) == "token-7-123"
        assert provider.get_token(8, 123) == "token-8-123"
        assert provider.get_token(7, 124) == "token-7-124"
    assert len(requests) == 3


def test_concurrent_cache_access_issues_one_token(key_file: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json=token_response())

    provider = make_provider(key_file, handler)
    with ThreadPoolExecutor(max_workers=4) as executor:
        tokens = list(executor.map(lambda _: provider.get_token(7, 123), range(8)))
    assert tokens == [SECRET] * 8
    assert len(requests) == 1


@pytest.mark.parametrize("field", ["installation_id", "repository_id"])
@pytest.mark.parametrize("value", [0, -1, True, None, "auth-response-secret"])
def test_invalid_scope_ids_fail_before_http(key_file: Path, field: str, value: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Invalid IDs must not make HTTP requests")

    provider = make_provider(key_file, handler)
    auth = importlib.import_module("review_bot.adapters.github.auth")
    ids = {"installation_id": 7, "repository_id": 123, field: value}
    with pytest.raises(auth.GitHubAppAuthError, match="ID") as caught:
        provider.get_token(**ids)
    assert_redacted(caught.value)


@pytest.mark.parametrize("app_id", ["", "0", "-1", SECRET, "1/../2", True])
def test_invalid_app_ids_are_sanitized(key_file: Path, app_id: Any) -> None:
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError, match="ID") as caught:
        auth.GitHubAppTokenProvider(app_id, key_file)
    assert_redacted(caught.value)


@pytest.mark.parametrize("api_url", [
    "http://api.github.com", "https://attacker.example", "//api.github.com",
    "https://api.github.com.attacker.example", "https://api.github.com:444",
    f"https://{SECRET}@api.github.com", f"https://api.github.com?token={SECRET}",
    f"https://api.github.com#{SECRET}", "https://api.github.com/api/v3",
    "https://api.github.com/app/../", f"https://api.github.com:{SECRET}", "not-a-url",
])
def test_provider_rejects_non_github_https_origins(key_file: Path, api_url: str) -> None:
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError) as caught:
        auth.GitHubAppTokenProvider("12345", key_file, api_url=api_url)
    assert_redacted(caught.value)


@pytest.mark.parametrize("api_url", ["https://api.github.com/", "https://api.github.com:443"])
def test_provider_accepts_equivalent_github_origin(key_file: Path, api_url: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.github.com/app/installations/7/access_tokens"
        return httpx.Response(201, json=token_response())

    provider = make_provider(key_file, handler, api_url=api_url)
    assert provider.get_token(7, 123) == SECRET


@pytest.mark.parametrize("kind", ["missing", "invalid", "public", "ec", "encrypted"])
def test_invalid_keys_are_sanitized(
    tmp_path: Path, rsa_key: rsa.RSAPrivateKey, kind: str,
) -> None:
    path = tmp_path / f"{SECRET}.pem"
    if kind == "invalid":
        path.write_text(f"-----BEGIN PRIVATE KEY-----\n{SECRET}\n")
    elif kind == "public":
        path.write_bytes(rsa_key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
        ))
    elif kind == "ec":
        path.write_bytes(ec.generate_private_key(ec.SECP256R1()).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
    elif kind == "encrypted":
        path.write_bytes(rsa_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(SECRET.encode()),
        ))
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError, match="key") as caught:
        auth.GitHubAppTokenProvider("12345", path)
    assert_redacted(caught.value, str(path))


@pytest.mark.parametrize(("field", "value"), [
    ("token", ""), ("token", None), ("token", {"secret": SECRET}),
    ("token", f"{SECRET}\n"), ("token", f"{SECRET}\x00"),
    ("expires_at", SECRET), ("expires_at", None),
    ("expires_at", datetime.fromtimestamp(NOW - 1, UTC).isoformat()),
    ("expires_at", datetime.fromtimestamp(NOW + 59, UTC).isoformat()),
    ("expires_at", datetime.fromtimestamp(NOW + 3600, UTC).replace(tzinfo=None).isoformat()),
    ("repository_selection", "all"), ("repositories", None),
    ("repositories", []), ("repositories", [{"id": 124}]),
    ("repositories", [{"id": "123"}]), ("repositories", [{"id": 123}, {"id": 124}]),
    ("permissions", None), ("permissions", {"contents": "read"}),
    ("permissions", {"contents": "write", "pull_requests": "write"}),
    ("permissions", {**PERMISSIONS, "issues": "write"}),
    ("permissions", {**PERMISSIONS, "metadata": "write"}),
])
def test_bad_token_responses_are_redacted_and_not_cached(
    key_file: Path, field: str, value: Any,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={**token_response(), field: value})

    provider = make_provider(key_file, handler)
    auth = importlib.import_module("review_bot.adapters.github.auth")
    for _ in range(2):
        with pytest.raises(auth.GitHubAppAuthError) as caught:
            provider.get_token(7, 123)
        assert_redacted(caught.value)
    assert len(requests) == 2


@pytest.mark.parametrize("body", [SECRET, "[]", "null", '{"secret": "auth-response-secret"}'])
def test_invalid_json_or_missing_token_response_is_sanitized(key_file: Path, body: str) -> None:
    provider = make_provider(key_file, lambda request: httpx.Response(201, text=body))
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError) as caught:
        provider.get_token(7, 123)
    assert_redacted(caught.value)


@pytest.mark.parametrize("status", [200, 401, 403, 422, 500, 302, 307])
def test_auth_http_failure_is_redacted_without_retry_or_redirect(
    key_file: Path, status: int, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("DEBUG")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, text=SECRET + request.headers["Authorization"], headers={
            "Location": "https://attacker.example/steal",
        })

    provider = make_provider(key_file, handler)
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError) as caught:
        provider.get_token(7, 123)
    assert len(requests) == 1
    assert_redacted(caught.value, requests[0].headers["Authorization"])
    assert SECRET not in caplog.text
    assert requests[0].headers["Authorization"] not in caplog.text


def test_transport_and_signing_failures_do_not_expose_exception_context(
    key_file: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError(SECRET + request.headers["Authorization"], request=request)

    provider = make_provider(key_file, handler)
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError) as caught:
        provider.get_token(7, 123)
    assert len(requests) == 1
    assert_redacted(caught.value, requests[0].headers["Authorization"])

    def broken_encode(*args: Any, **kwargs: Any) -> str:
        raise ValueError(SECRET)

    monkeypatch.setattr(jwt, "encode", broken_encode)
    with pytest.raises(auth.GitHubAppAuthError) as caught:
        provider.get_token(7, 123)
    assert_redacted(caught.value)
    assert len(requests) == 1


@pytest.mark.parametrize("now", [float("nan"), float("inf"), SECRET])
def test_invalid_clock_is_sanitized(key_file: Path, now: Any) -> None:
    provider = make_provider(key_file, lambda request: pytest.fail("Invalid clock made HTTP request"),
                             clock=lambda: now)
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError) as caught:
        provider.get_token(7, 123)
    assert_redacted(caught.value)


def test_failed_refresh_does_not_fall_back_to_expiring_token(key_file: Path) -> None:
    now = [NOW]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(201, json=token_response())
        return httpx.Response(403, text=SECRET)

    provider = make_provider(key_file, handler, clock=lambda: now[0])
    assert provider.get_token(7, 123) == SECRET
    now[0] += 3540
    auth = importlib.import_module("review_bot.adapters.github.auth")
    with pytest.raises(auth.GitHubAppAuthError):
        provider.get_token(7, 123)
    assert len(requests) == 2


def test_provider_drives_repository_resolution_and_comment_http(key_file: Path) -> None:
    from review_bot.adapters.github.client import GitHubApiClient

    now = [NOW]
    requests: list[httpx.Request] = []
    issued: list[str] = []
    body = "<!-- review-bot:architecture:v1 -->\n\nArchitecture review"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/app/installations/7/access_tokens":
            issued.append(f"ghs_test_token_{len(issued) + 1}")
            assert json.loads(request.content) == {
                "repository_ids": [123], "permissions": PERMISSIONS,
            }
            return httpx.Response(201, json=token_response(
                token=issued[-1], expires_at=now[0] + 3600,
            ))
        assert request.headers["Authorization"] == f"Bearer {issued[-1]}"
        if request.url.path == "/repositories/123":
            return httpx.Response(200, json={"id": 123, "full_name": "org/repo"})
        if request.url.path == "/repos/org/repo/pulls/42":
            return httpx.Response(200, json={"number": 42, "state": "open"})
        assert request.method == "POST"
        assert request.url.path == "/repos/org/repo/issues/42/comments"
        assert json.loads(request.content) == {"body": body}
        return httpx.Response(201, json={
            "id": 88, "body": body, "user": {"id": 99, "login": "review-app[bot]"},
        })

    provider = make_provider(key_file, handler, clock=lambda: now[0])
    client = GitHubApiClient(
        token_provider=lambda: provider.get_token(7, 123),
        transport=httpx.MockTransport(handler), repository_id=123,
    )
    assert client.resolve_repository(123) == "org/repo"
    assert client.get_pull_request(123, 42)["state"] == "open"
    now[0] += 3540
    comment = client.post_comment(123, 42, body)
    assert comment["body"] == body
    assert comment["user"]["login"] == "review-app[bot]"
    assert len(issued) == 2
    assert len(requests) == 5


@pytest.mark.parametrize("omitted", [
    ("repositories",), ("repository_selection",), ("repositories", "repository_selection"),
])
def test_optional_scope_metadata_and_long_opaque_token(key_file: Path, omitted: tuple[str, ...]) -> None:
    token = "ghs_12345_" + "opaque-token-segment." * 20
    payload = token_response(token=token)
    payload["permissions"] = PERMISSIONS.copy()
    payload["expires_at"] = datetime.fromtimestamp(NOW + 3600, UTC).isoformat().replace("+00:00", "Z")
    for field in omitted:
        del payload[field]
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert json.loads(request.content)["repository_ids"] == [123]
        return httpx.Response(201, json=payload)

    provider = make_provider(key_file, handler)
    assert provider.get_token(7, 123) == token
    assert provider.get_token(7, 123) == token
    assert len(requests) == 1
    assert token not in repr(provider._cache)