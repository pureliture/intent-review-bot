"""Guarded PR comment publication with idempotency and freshness race verification."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from review_bot.domain.envelope_v2 import PublicationReceipt
from review_bot.ports import GitHubAuthorizationError, GitHubClient

logger = logging.getLogger(__name__)
DEFAULT_A_MARKER = "<!-- review-bot:architecture:v1 -->"


class PublicationAuthorizationError(GitHubAuthorizationError):
    """Preserve a possible comment side effect when source authorization is lost."""

    def __init__(self, receipt: PublicationReceipt) -> None:
        super().__init__("publication_authorization_lost")
        self.receipt = receipt


def _call_client_method(method: Any, *primary_args: Any, fallback_args: Any = None) -> Any:
    """Invoke client method trying primary args first, then fallback args if TypeError occurs."""
    try:
        return method(*primary_args)
    except TypeError:
        if fallback_args is not None:
            return method(*fallback_args)
        raise


def publication_fingerprint(repository_id: int, pr_number: int, head_sha: str, body: str) -> str:
    """Stable identity of the prepared comment within its PR and source version."""
    return hashlib.sha256(f"{repository_id}:{pr_number}:{head_sha}:{body}".encode()).hexdigest()


def publish_review_comment(
    github_client: GitHubClient,
    repository_id: int,
    pr_number: int,
    expected_head_sha: str,
    expected_base_sha: str,
    body: str,
    marker: str = DEFAULT_A_MARKER,
    *,
    publication_guard: Callable[[], bool] | None = None,
    context_guard: Callable[[dict[str, Any]], bool] | None = None,
    source_guard: Callable[[dict[str, Any]], bool] | None = None,
    strict_target: bool = False,
    read_timeout_seconds: float = 10.0,
) -> PublicationReceipt:
    """Guarded idempotent comment publication targeting fixed marker.

    1. Pre-check: PR open state, target head SHA and base SHA freshness.
    2. Marker Discovery: Search existing comments for marker using find_marker_comment.
    3. Idempotent Upsert: Update if found, else create new, checking the caller's fence before writes.
    4. Post-check: Re-query PR state immediately after API return.
       If head or base SHA moved during API call, update comment with stale warning banner
       and return receipt with status='stale'.
    5. Return PublicationReceipt matching domain/envelope_v2.py.
    """
    # Ensure marker is on line 1 of body
    clean_body = body
    if not clean_body.startswith(marker):
        clean_body = f"{marker}\n\n{clean_body.lstrip()}"

    fingerprint = publication_fingerprint(repository_id, pr_number, expected_head_sha, clean_body)

    def read_pr() -> dict[str, Any]:
        if strict_target:
            return github_client.get_pull_request(pr_number, max_response_bytes=524288, timeout_seconds=read_timeout_seconds)
        return _call_client_method(github_client.get_pull_request, pr_number, fallback_args=(repository_id, pr_number))

    def source_is_current(pr: dict[str, Any]) -> bool:
        if source_guard is None:
            return True
        try:
            return source_guard(pr) is True
        except GitHubAuthorizationError:
            raise
        except Exception as exc:
            logger.warning("publication_source_check_failed:%s", type(exc).__name__)
            return False

    def failure_reason(code: str, exc: Exception) -> str:
        if strict_target:
            logger.warning("%s:%s", code, type(exc).__name__)
            return f"{code}:{type(exc).__name__}"
        logger.exception("%s: %s", code, exc)
        return f"{code}: {exc}"

    def context_is_current(pr: dict[str, Any]) -> bool:
        if context_guard is None:
            return True
        try:
            return context_guard(pr) is True
        except Exception as exc:
            logger.warning("publication_context_check_failed:%s", type(exc).__name__)
            return False

    def check_publication_guard(comment_id: int | None = None) -> PublicationReceipt | None:
        if publication_guard is None:
            return None
        try:
            if publication_guard():
                return None
        except Exception:
            logger.exception("publication_guard_failed")
        return PublicationReceipt(
            comment_id=comment_id, status="failed", fingerprint=fingerprint, marker=marker,
            published_at=datetime.now(UTC).isoformat() if comment_id is not None else None,
            reason="publication_fenced",
        )

    # 1. Pre-check PR state
    try:
        if strict_target:
            rejected = check_publication_guard()
            if rejected is not None:
                return rejected
        pr_data = read_pr()
        current_state = pr_data.get("state") if strict_target else pr_data.get("state", "open")
        if current_state != "open":
            return PublicationReceipt(
                comment_id=None,
                status="stale" if strict_target else "failed",
                fingerprint=fingerprint,
                marker=marker,
                reason="pr_not_open",
            )

        if not context_is_current(pr_data):
            return PublicationReceipt(
                comment_id=None, status="stale", fingerprint=fingerprint,
                marker=marker, reason="stale_pr_context",
            )

        head_info = pr_data.get("head")
        current_head = (
            head_info.get("sha")
            if isinstance(head_info, dict)
            else getattr(head_info, "sha", None) or pr_data.get("head_sha", "")
        )
        if expected_head_sha and (strict_target or current_head) and current_head != expected_head_sha:
            return PublicationReceipt(
                comment_id=None,
                status="stale" if strict_target else "failed",
                fingerprint=fingerprint,
                marker=marker,
                reason="stale_head_sha",
            )

        base_info = pr_data.get("base")
        current_base = (
            base_info.get("sha")
            if isinstance(base_info, dict)
            else getattr(base_info, "sha", None) or pr_data.get("base_sha", "")
        )
        if expected_base_sha and (strict_target or current_base) and current_base != expected_base_sha:
            return PublicationReceipt(
                comment_id=None,
                status="stale" if strict_target else "failed",
                fingerprint=fingerprint,
                marker=marker,
                reason="stale_base_sha",
            )
        if not source_is_current(pr_data):
            return PublicationReceipt(None, "stale", fingerprint, marker=marker, reason="stale_intent_sources")
    except GitHubAuthorizationError:
        if strict_target:
            raise PublicationAuthorizationError(PublicationReceipt(None, "failed", fingerprint, marker=marker,
                reason="publication_authorization_lost")) from None
        raise
    except Exception as exc:
        return PublicationReceipt(
            comment_id=None,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            reason=failure_reason("pre_check_failed", exc),
        )

    # 2. Discover existing marker comment
    try:
        if strict_target:
            existing = github_client.find_marker_comment(pr_number, marker, before_request=publication_guard,
                timeout_seconds=read_timeout_seconds, max_response_bytes=8388608)
        else:
            existing = _call_client_method(github_client.find_marker_comment, pr_number, marker,
                fallback_args=(repository_id, pr_number, marker))
    except ValueError as exc:
        return PublicationReceipt(
            comment_id=None,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            reason=f"ambiguous_marker_comment:{type(exc).__name__}" if strict_target else f"ambiguous_marker_comment: {exc}",
        )
    except GitHubAuthorizationError:
        raise
    except Exception as exc:
        return PublicationReceipt(
            comment_id=None,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            reason=failure_reason("find_marker_failed", exc),
        )

    # 3. Post or update comment
    published_id: int | None = None
    if existing is not None:
        comment_id: int | None = None
        try:
            comment_id = (
                existing.get("id")
                if isinstance(existing, dict)
                else getattr(existing, "id", None) or getattr(existing, "comment_id", None)
            )
            if type(comment_id) is not int or comment_id <= 0:
                return PublicationReceipt(
                    comment_id=None, status="failed", fingerprint=fingerprint,
                    marker=marker, reason="invalid_comment_id",
                )
            rejected = check_publication_guard()
            if rejected is not None:
                return rejected
            _call_client_method(
                github_client.update_comment,
                comment_id,
                clean_body,
                fallback_args=(repository_id, comment_id, clean_body),
            )
            published_id = comment_id
        except GitHubAuthorizationError:
            raise
        except Exception as exc:
            logger.warning("update_comment_failed:%s", type(exc).__name__)
            return PublicationReceipt(
                comment_id=comment_id,
                status="unknown" if isinstance(exc, httpx.RequestError) else "failed",
                fingerprint=fingerprint,
                marker=marker,
                reason=f"update_comment_failed:{type(exc).__name__}",
            )
    else:
        try:
            write_method = (
                getattr(github_client, "create_comment", None)
                or github_client.post_comment
            )
            rejected = check_publication_guard()
            if rejected is not None:
                return rejected
            created = _call_client_method(
                write_method,
                pr_number,
                clean_body,
                fallback_args=(repository_id, pr_number, clean_body),
            )
            published_id = (
                created.get("id")
                if isinstance(created, dict)
                else getattr(created, "id", None) or getattr(created, "comment_id", None)
            )
        except GitHubAuthorizationError:
            raise
        except Exception as exc:
            logger.warning("create_comment_failed:%s", type(exc).__name__)
            return PublicationReceipt(
                comment_id=None,
                status="unknown" if isinstance(exc, httpx.RequestError) else "failed",
                fingerprint=fingerprint,
                marker=marker,
                reason=f"create_comment_failed:{type(exc).__name__}",
            )

    if type(published_id) is not int or published_id <= 0:
        return PublicationReceipt(
            comment_id=None, status="failed", fingerprint=fingerprint,
            marker=marker, reason="invalid_comment_id",
        )

    # 4. Post-check PR freshness (concurrency / race detection)
    try:
        if strict_target:
            rejected = check_publication_guard(published_id)
            if rejected is not None:
                return rejected
        post_pr_data = read_pr()
        post_state = post_pr_data.get("state") if strict_target else post_pr_data.get("state", "open")
        post_head_info = post_pr_data.get("head")
        post_head = (
            post_head_info.get("sha")
            if isinstance(post_head_info, dict)
            else getattr(post_head_info, "sha", None) or post_pr_data.get("head_sha", "")
        )

        if post_state != "open":
            stale_banner = (
                "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**\n"
                "> 게시 중 PR 상태가 변경되었습니다: pr_closed\n\n"
            )
            if strict_target:
                stale_banner = "> ⚠️ **Stale review — do not use as a current PR review.**\n> PR state changed during publication.\n\n"
            stale_body = f"{marker}\n\n{stale_banner}" + clean_body[len(marker):].lstrip("\n")
            if published_id is not None:
                rejected = check_publication_guard(published_id)
                if rejected is not None:
                    return rejected
                _call_client_method(
                    github_client.update_comment,
                    published_id,
                    stale_body,
                    fallback_args=(repository_id, published_id, stale_body),
                )
            return PublicationReceipt(
                comment_id=published_id,
                status="stale",
                fingerprint=fingerprint,
                marker=marker,
                published_at=datetime.now(UTC).isoformat(),
                reason="pr_closed_during_publication",
            )

        post_base_info = post_pr_data.get("base")
        post_base = (
            post_base_info.get("sha")
            if isinstance(post_base_info, dict)
            else getattr(post_base_info, "sha", None) or post_pr_data.get("base_sha", "")
        )
        stale_reason = ""
        moved = ""
        if expected_head_sha and (strict_target or post_head) and post_head != expected_head_sha:
            stale_reason = "stale_head_sha"
            moved = "head_sha changed or unavailable" if strict_target else f"head_sha moved to {post_head[:8]}"
        elif expected_base_sha and (strict_target or post_base) and post_base != expected_base_sha:
            stale_reason = "stale_base_sha"
            moved = "base_sha changed or unavailable" if strict_target else f"base_sha moved to {post_base[:8]}"
        elif not context_is_current(post_pr_data):
            stale_reason = "stale_pr_context"
            moved = "PR text or review perspective changed"
        elif not source_is_current(post_pr_data):
            stale_reason = "stale_intent_sources"
            moved = "Intent text or issue versions changed, or freshness could not be verified"
        if stale_reason:
            stale_banner = (
                "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**\n"
                f"> 게시 중 코드·분류·실행이 변경되었습니다: {moved}\n\n"
            )
            if strict_target:
                stale_banner = f"> ⚠️ **Stale review — do not use as a current PR review.**\n> {moved}.\n\n"
            stale_body = f"{marker}\n\n{stale_banner}" + clean_body[len(marker):].lstrip("\n")
            if published_id is not None:
                rejected = check_publication_guard(published_id)
                if rejected is not None:
                    return rejected
                _call_client_method(
                    github_client.update_comment,
                    published_id,
                    stale_body,
                    fallback_args=(repository_id, published_id, stale_body),
                )
            return PublicationReceipt(
                comment_id=published_id,
                status="stale",
                fingerprint=fingerprint,
                marker=marker,
                published_at=datetime.now(UTC).isoformat(),
                reason=stale_reason,
            )
        if strict_target:
            rejected = check_publication_guard(published_id)
            if rejected is not None:
                return rejected
    except GitHubAuthorizationError:
        if strict_target:
            raise PublicationAuthorizationError(PublicationReceipt(published_id, "unknown", fingerprint,
                marker=marker, published_at=datetime.now(UTC).isoformat(), reason="publication_authorization_lost")) from None
        raise
    except Exception as exc:
        return PublicationReceipt(
            comment_id=published_id,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            published_at=datetime.now(UTC).isoformat() if published_id is not None else None,
            reason=failure_reason("post_check_failed", exc),
        )

    return PublicationReceipt(
        comment_id=published_id,
        status="succeeded",
        fingerprint=fingerprint,
        marker=marker,
        published_at=datetime.now(UTC).isoformat(),
        reason="published",
    )
