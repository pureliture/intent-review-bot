"""Bounded same-repository issue reads; URLs and issue bodies never drive requests."""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

from review_bot.adapters.github.auth import GitHubAppAuthError
from review_bot.adapters.github.client import GitHubResponseLimitError
from review_bot.application.context import hash_json
from review_bot.application.intent_context import evidence_digest
from review_bot.domain.envelope_v2 import EnvelopeEvidence, ReviewSnapshot, UnavailableSource
from review_bot.ports import GitHubClient

MAX_ISSUES = 8
MAX_TEXT_BYTES = 256 * 1024
MAX_TOTAL_TEXT_BYTES = 1024 * 1024
COLLECTION_TIMEOUT_SECONDS = 10.0
_REFERENCE = re.compile(
    r"https?://[^\s<>()\[\]]+|"
    r"(?<![\w/])[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+#[1-9][0-9]{0,9}(?![\w])|"
    r"(?<![\w/#])#[1-9][0-9]{0,9}(?![\w])"
)


class IntentCollectionAborted(RuntimeError):
    """Common deadline, stop or lease gate prevented collection."""


@dataclass(frozen=True)
class IntentSources:
    evidence: tuple[EnvelopeEvidence, ...]
    unavailable_sources: tuple[UnavailableSource, ...]
    captured_at: str
    as_of: str


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else None


def _reference(value: str, repository: str) -> tuple[int | None, str]:
    if value.startswith(("http://", "https://")):
        if "/issues/" not in value and "/pull/" not in value:
            return None, ""
        try:
            url = urlsplit(value.rstrip(".,;!?"))
            match = re.fullmatch(r"/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/issues/([1-9][0-9]{0,9})", url.path)
            if url.scheme != "https" or url.hostname != "github.com" or url.username or url.password or url.port or url.query or url.fragment or match is None:
                return None, "unsupported_reference"
            repo, number = match.groups()
        except ValueError:
            return None, "unsupported_reference"
    elif value.startswith("#"):
        repo, number = repository, value[1:]
    else:
        repo, number = value.rsplit("#", 1)
    if repo.casefold() != repository.casefold():
        return None, "cross_repository_reference"
    parsed = int(number)
    return (parsed, "") if parsed <= 2147483647 else (None, "unsupported_reference")


def collect_intent_sources(
    github_client: GitHubClient | None, repository: str, pr_number: int, pr_data: dict[str, Any], *,
    as_of: str | None = None, max_issues: int = MAX_ISSUES, max_text_bytes: int = MAX_TEXT_BYTES,
    max_total_text_bytes: int = MAX_TOTAL_TEXT_BYTES, timeout_seconds: float = COLLECTION_TIMEOUT_SECONDS,
    before_request: Callable[[], bool] | None = None,
) -> IntentSources:
    """Freeze whole original text or an explicit absence; never widen App permissions."""
    if (
        type(max_issues) is not int or not 1 <= max_issues <= 32
        or type(max_text_bytes) is not int or not 1 <= max_text_bytes <= MAX_TEXT_BYTES
        or type(max_total_text_bytes) is not int or not 1 <= max_total_text_bytes <= MAX_TOTAL_TEXT_BYTES
        or type(timeout_seconds) not in (int, float) or not 0 < timeout_seconds <= COLLECTION_TIMEOUT_SECONDS
        or (as_of is not None and _timestamp(as_of) is None)
    ):
        raise ValueError("invalid_intent_collection_limits")
    bound_repository = getattr(github_client, "repository", repository)
    if isinstance(bound_repository, str) and bound_repository.casefold() != repository.casefold():
        raise GitHubAppAuthError("Intent source repository binding mismatch")
    deadline = time.monotonic() + timeout_seconds
    unavailable: dict[str, UnavailableSource] = {}
    raw_sources: list[tuple[str, int, dict[str, Any], str]] = []
    total_bytes = 0

    def missing(source: str, reason: str) -> None:
        unavailable[source] = UnavailableSource(source, reason)

    def text(kind: str, number: int, data: dict[str, Any]) -> str | None:
        nonlocal total_bytes
        source = "pr_text" if kind == "pr_text" else f"issue:{number}"
        title = "" if data.get("title") is None else data["title"]
        body = "" if data.get("body") is None else data["body"]
        if not isinstance(title, str) or not isinstance(body, str):
            missing(source, "invalid_text")
            return None
        try:
            size = len(f"{title}\n{body}".encode("utf-8"))
        except UnicodeError:
            missing(source, "invalid_text")
            return None
        if size > max_text_bytes or total_bytes + size > max_total_text_bytes:
            missing(source, "text_limit_exceeded")
            return None
        total_bytes += size
        raw_sources.append((kind, number, {"title": title, "body": body, "updated_at": data.get("updated_at")}, datetime.now(UTC).isoformat()))
        return f"{title}\n{body}"

    original = text("pr_text", pr_number, pr_data)
    if as_of is not None:
        pr_version = _timestamp(pr_data.get("updated_at"))
        if pr_version is None or pr_version > _timestamp(as_of):
            original = None
    seen: set[tuple[int | None, str]] = set()
    references = 0
    if original is not None:
        for match in _REFERENCE.finditer(original):
            number, reason = _reference(match.group(), repository)
            if number is None and not reason:
                continue
            identity = (number, match.group() if number is None else "")
            if identity in seen:
                continue
            seen.add(identity)
            references += 1
            if references > max_issues:
                missing("linked_issues", "reference_limit_exceeded")
                break
            source = f"issue:{number}" if number is not None else f"linked_issue:{references}"
            if reason:
                missing(source, reason)
                continue
            if before_request is not None and not before_request():
                raise IntentCollectionAborted("intent_collection_gate_closed")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                missing(source, "collection_timeout")
                continue
            if github_client is None:
                missing(source, "client_unavailable")
                continue
            try:
                data = github_client.get_issue(number, max_response_bytes=524288, timeout_seconds=min(remaining, 10.0))
            except GitHubAppAuthError:
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 401:
                    raise GitHubAppAuthError("Intent source authorization failed") from None
                missing(source, "forbidden" if status == 403 else "not_found" if status == 404 else "request_failed")
                continue
            except GitHubResponseLimitError:
                missing(source, "response_limit_exceeded")
                continue
            except (httpx.TimeoutException, TimeoutError):
                missing(source, "request_timeout")
                continue
            except (httpx.RequestError, ValueError, AttributeError):
                missing(source, "request_failed")
                continue
            if time.monotonic() > deadline:
                missing(source, "collection_timeout")
            elif not isinstance(data, dict) or type(data.get("number")) is not int or data["number"] != number:
                missing(source, "invalid_issue_identity")
            elif "pull_request" in data:
                missing(source, "unsupported_reference")
            else:
                text("issue_text", number, data)
    if references == 0:
        missing("linked_issues", "no_explicit_references" if original is not None else "pr_text_unavailable")
    if before_request is not None and not before_request():
        raise IntentCollectionAborted("intent_collection_gate_closed")
    captured_at = datetime.now(UTC).isoformat()
    as_of = as_of or captured_at
    cutoff = _timestamp(as_of)
    evidence = []
    for kind, number, data, retrieved_at in raw_sources:
        source = "pr_text" if kind == "pr_text" else f"issue:{number}"
        updated_at = _timestamp(data["updated_at"])
        if data["updated_at"] is not None and updated_at is None:
            missing(source, "invalid_text_version")
            continue
        if updated_at is not None and updated_at > cutoff:
            missing(source, "historical_text_unavailable")
            continue
        if updated_at is None and (kind == "issue_text" or cutoff < _timestamp(captured_at)):
            missing(source, "text_version_unavailable")
            continue
        content = {"title": data["title"], "body": data["body"]}
        digest = evidence_digest(content, kind)
        locator = {"repository": repository, "pr_number" if kind == "pr_text" else "issue_number": number, "retrieved_at": retrieved_at}
        if updated_at is not None:
            locator["updated_at"] = data["updated_at"]
        evidence.append(EnvelopeEvidence("E-PR-01" if kind == "pr_text" else f"E-ISSUE-{number}",
            kind, digest, locator, content, digest))
    return IntentSources(tuple(evidence), tuple(unavailable[key] for key in sorted(unavailable)), captured_at, as_of)


def intent_sources_current(
    github_client: GitHubClient, snapshot: ReviewSnapshot, pr_data: dict[str, Any], *,
    timeout_seconds: float = COLLECTION_TIMEOUT_SECONDS, before_request: Callable[[], bool] | None = None,
) -> bool:
    """Recheck frozen originals/versions and missing states within the same repository."""
    original = next((item for item in snapshot.evidence if item.kind == "pr_text"), None)
    if original is None or not isinstance(original.content, dict):
        return False
    if original.content != {"title": pr_data.get("title") or "", "body": pr_data.get("body") or ""}:
        return False
    current = collect_intent_sources(github_client, snapshot.repository, snapshot.pr_number, pr_data,
        timeout_seconds=timeout_seconds, before_request=before_request)

    def identity(evidence, unavailable):
        originals = []
        for item in evidence:
            if item.kind not in ("pr_text", "issue_text"):
                continue
            record = asdict(item)
            record.pop("id")
            record["locator"].pop("retrieved_at", None)
            if item.kind == "pr_text":
                record["locator"].pop("updated_at", None)
            originals.append(record)
        missing = [asdict(item) for item in unavailable if item.source.startswith(("pr_text", "issue", "linked_issue"))]
        return hash_json({"originals": sorted(originals, key=hash_json), "missing": sorted(missing, key=hash_json)})

    return identity(snapshot.evidence, snapshot.unavailable_sources) == identity(current.evidence, current.unavailable_sources)