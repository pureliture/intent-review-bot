"""Guarded PR comment publication with idempotency and freshness race verification."""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import httpx

from review_bot.domain.envelope_v2 import PublicationReceipt
from review_bot.ports import GitHubClient

logger = logging.getLogger(__name__)
DEFAULT_A_MARKER = "<!-- review-bot:architecture:v1 -->"


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
        pr_data = _call_client_method(
            github_client.get_pull_request,
            pr_number,
            fallback_args=(repository_id, pr_number),
        )
        current_state = pr_data.get("state", "open")
        if current_state != "open":
            return PublicationReceipt(
                comment_id=None,
                status="failed",
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
        if expected_head_sha and current_head and current_head != expected_head_sha:
            return PublicationReceipt(
                comment_id=None,
                status="failed",
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
        if expected_base_sha and current_base and current_base != expected_base_sha:
            return PublicationReceipt(
                comment_id=None,
                status="failed",
                fingerprint=fingerprint,
                marker=marker,
                reason="stale_base_sha",
            )
    except Exception as exc:
        logger.exception("pre_check_failed: %s", exc)
        return PublicationReceipt(
            comment_id=None,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            reason=f"pre_check_failed: {exc}",
        )

    # 2. Discover existing marker comment
    try:
        existing = _call_client_method(
            github_client.find_marker_comment,
            pr_number,
            marker,
            fallback_args=(repository_id, pr_number, marker),
        )
    except ValueError as exc:
        return PublicationReceipt(
            comment_id=None,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            reason=f"ambiguous_marker_comment: {exc}",
        )
    except Exception as exc:
        logger.exception("find_marker_failed: %s", exc)
        return PublicationReceipt(
            comment_id=None,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            reason=f"find_marker_failed: {exc}",
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
        post_pr_data = _call_client_method(
            github_client.get_pull_request,
            pr_number,
            fallback_args=(repository_id, pr_number),
        )
        post_state = post_pr_data.get("state", "open")
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
        if expected_head_sha and post_head and post_head != expected_head_sha:
            stale_reason = "stale_head_sha"
            moved = f"head_sha moved to {post_head[:8]}"
        elif expected_base_sha and post_base and post_base != expected_base_sha:
            stale_reason = "stale_base_sha"
            moved = f"base_sha moved to {post_base[:8]}"
        elif not context_is_current(post_pr_data):
            stale_reason = "stale_pr_context"
            moved = "PR text or review perspective changed"
        if stale_reason:
            stale_banner = (
                "> ⚠️ **오래된 결과 — 현재 PR 리뷰로 사용하지 마세요.**\n"
                f"> 게시 중 코드·분류·실행이 변경되었습니다: {moved}\n\n"
            )
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
    except Exception as exc:
        logger.exception("post_check_failed: %s", exc)
        return PublicationReceipt(
            comment_id=published_id,
            status="failed",
            fingerprint=fingerprint,
            marker=marker,
            published_at=datetime.now(UTC).isoformat() if published_id is not None else None,
            reason=f"post_check_failed: {exc}",
        )

    return PublicationReceipt(
        comment_id=published_id,
        status="succeeded",
        fingerprint=fingerprint,
        marker=marker,
        published_at=datetime.now(UTC).isoformat(),
        reason="published",
    )
