"""Snapshot assembly integrating Git reader, merge-base resolver, classifier, and selector."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime

from review_bot.application.classification import classify_pr
from review_bot.application.selection import DeadlineModeSelector, LightweightModeSelector, select_modes
from review_bot.domain.envelope_v2 import (
    ChangedFile,
    EnvelopeEvidence,
    ModeSelection,
    ReviewSnapshot,
    UnavailableSource,
)
from review_bot.domain.models_v1 import (
    Classification,
    PrSnapshot,
)
from review_bot.ports import MergeBaseResult, ModeSelectorPort, RepositoryReader


def assemble_pr_snapshot(
    repository: str,
    number: int,
    base_sha: str,
    head_sha: str,
    source_branch: str,
    labels: tuple[str, ...] | list[str] = (),
    changed_files: Sequence[ChangedFile | str] = (),
    rename_map: dict[str, str] | None = None,
) -> PrSnapshot:
    """Build immutable Core v1 PrSnapshot."""
    paths = tuple(f.path if isinstance(f, ChangedFile) else str(f) for f in changed_files)
    renames = dict(rename_map) if rename_map is not None else {}
    return PrSnapshot(
        repository=repository,
        number=number,
        base_sha=base_sha,
        head_sha=head_sha,
        source_branch=source_branch,
        labels=tuple(labels),
        changed_files=paths,
        rename_map=renames,
    )


def assemble_review_snapshot(
    repository: str,
    pr_number: int,
    repository_id: int,
    installation_id: int,
    target_base_sha: str,
    head_sha: str,
    merge_base_result: MergeBaseResult,
    changed_files: Sequence[ChangedFile] = (),
    title: str = "",
    body: str = "",
    as_of: str | None = None,
    evidence: Sequence[EnvelopeEvidence] = (),
    unavailable_sources: Sequence[UnavailableSource] = (),
) -> ReviewSnapshot:
    """Build immutable Service Envelope v2 ReviewSnapshot."""
    now_utc = datetime.now(UTC).isoformat()
    capture_time = now_utc
    as_of_time = as_of or now_utc
    comparison_base_sha = (
        merge_base_result.comparison_base_sha
        if merge_base_result.is_comparable and merge_base_result.comparison_base_sha
        else None
    )
    comparison_status = "comparable" if comparison_base_sha else "incomparable"

    pr_text_raw = f"{title}\n{body}".encode()
    pr_text_digest = hashlib.sha256(pr_text_raw).hexdigest()

    context_raw = f"{repository}:{pr_number}:{target_base_sha}:{head_sha}".encode()
    context_digest = hashlib.sha256(context_raw).hexdigest()

    # Deterministic snapshot ID calculation
    snap_key = (
        f"v2:{repository}:{pr_number}:{repository_id}:{target_base_sha}:"
        f"{head_sha}:{comparison_base_sha}:{pr_text_digest}:{context_digest}"
    )
    snapshot_id = hashlib.sha256(snap_key.encode()).hexdigest()

    return ReviewSnapshot(
        schema_version=2,
        snapshot_id=snapshot_id,
        repository=repository,
        pr_number=pr_number,
        repository_id=repository_id,
        installation_id=installation_id,
        target_base_sha=target_base_sha,
        head_sha=head_sha,
        comparison_base_sha=comparison_base_sha,
        comparison_status=comparison_status,
        captured_at=capture_time,
        as_of=as_of_time,
        pr_text_digest=pr_text_digest,
        context_digest=context_digest,
        changed_files=tuple(changed_files),
        evidence=tuple(evidence),
        unavailable_sources=tuple(unavailable_sources),
    )


def assemble_snapshot_bundle(
    repo_reader: RepositoryReader,
    repository: str,
    pr_number: int,
    repository_id: int,
    installation_id: int,
    target_base_sha: str,
    head_sha: str,
    source_branch: str,
    labels: tuple[str, ...] | list[str] = (),
    title: str = "",
    body: str = "",
    available_modes: Sequence[str] = ("architecture",),
    as_of: str | None = None,
    *,
    selector_policy: str = "all_available",
    selector_uses_model: bool = False,
    mode_selector: ModeSelectorPort | None = None,
) -> tuple[ReviewSnapshot, PrSnapshot, Classification, ModeSelection]:
    """Execute end-to-end M2 snapshot assembly pipeline."""
    title = title or ""
    body = body or ""
    # 1. Merge-base resolution
    merge_base_res = repo_reader.resolve_merge_base(target_base_sha, head_sha)

    # 2. Changed files and rename map determination
    unavailable: list[UnavailableSource] = []
    if merge_base_res.is_comparable and merge_base_res.comparison_base_sha:
        comp_sha = merge_base_res.comparison_base_sha
        changed_files = repo_reader.get_changed_files(comp_sha, head_sha)
        rename_map = repo_reader.get_rename_map(comp_sha, head_sha)
        effective_base_sha = comp_sha
    else:
        changed_files = ()
        rename_map = {}
        effective_base_sha = ""
        unavailable.append(
            UnavailableSource(
                source="merge_base",
                reason=f"Merge base resolution status is {merge_base_res.comparison_status}",
            )
        )

    # 3. Perspective classification
    classification = classify_pr(
        source_branch=source_branch,
        labels=labels,
        title=title,
    )

    # 4. Assemble snapshots
    evidence = ()
    if selector_uses_model:
        evidence = (EnvelopeEvidence(
            id="pr-text", kind="pr_text", version="head",
            locator={"repository": repository, "pr_number": pr_number},
            content={"title": title, "body": body},
            digest=hashlib.sha256(f"{title}\n{body}".encode()).hexdigest(),
        ),)
    review_snapshot = assemble_review_snapshot(
        repository=repository,
        pr_number=pr_number,
        repository_id=repository_id,
        installation_id=installation_id,
        target_base_sha=target_base_sha,
        head_sha=head_sha,
        merge_base_result=merge_base_res,
        changed_files=changed_files,
        title=title,
        body=body,
        as_of=as_of,
        evidence=evidence,
        unavailable_sources=tuple(unavailable),
    )

    pr_snapshot = assemble_pr_snapshot(
        repository=repository,
        number=pr_number,
        base_sha=effective_base_sha,
        head_sha=head_sha,
        source_branch=source_branch,
        labels=labels,
        changed_files=changed_files,
        rename_map=rename_map,
    )

    # 5. Review mode selection
    selection = select_modes(
        snapshot_id=review_snapshot.snapshot_id,
        available_modes=available_modes,
        policy=selector_policy,
    )
    if selector_uses_model and review_snapshot.comparison_status == "comparable":
        selector = mode_selector if mode_selector is not None else LightweightModeSelector()
        if not isinstance(selector, DeadlineModeSelector):
            selector = DeadlineModeSelector(selector)
        selection = selector.select_modes(review_snapshot, pr_snapshot, available_modes)

    return review_snapshot, pr_snapshot, classification, selection
