"""Snapshot assembly integrating Git reader, merge-base resolver, classifier, and selector."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from review_bot.application.classification import classify_pr
from review_bot.application.context import hash_json
from review_bot.application.intent_context import collect_intent_code_snapshot, validate_intent_context_paths
from review_bot.application.selection import (
    DeadlineModeSelector, LightweightModeSelector, constrain_architecture_to_java_gradle, select_modes,
)
from review_bot.domain.envelope_v2 import (
    ChangedFile,
    EnvelopeEvidence,
    ModeSelection,
    ReviewSnapshot,
    UnavailableSource,
)
from review_bot.domain.models_v1 import (
    Classification,
    CodeSnapshot,
    PrSnapshot,
)
from review_bot.ports import MergeBaseResult, ModeSelectorPort, RepositoryReader


class SnapshotCollectionAborted(RuntimeError):
    """The common execution gate closed while preparing fixed material."""


@dataclass(frozen=True)
class SnapshotMaterial:
    merge_base: MergeBaseResult
    changed_files: tuple[ChangedFile, ...]
    rename_map: dict[str, str]
    evidence: tuple[EnvelopeEvidence, ...]
    unavailable_sources: tuple[UnavailableSource, ...]
    context_digest: str
    base: CodeSnapshot | None = None
    head: CodeSnapshot | None = None
    captured_at: str | None = None
    as_of: str | None = None


def collect_snapshot_material(
    repo_reader: RepositoryReader, target_base_sha: str, head_sha: str, *,
    evidence: Sequence[EnvelopeEvidence] = (), unavailable_sources: Sequence[UnavailableSource] = (),
    related_paths: tuple[str, ...] = (), captured_at: str | None = None, as_of: str | None = None,
    before_read: Callable[[], bool] | None = None,
) -> SnapshotMaterial:
    """Freeze B originals and complete scoped Git files before making any selection."""
    def gate() -> None:
        if before_read is not None and not before_read():
            raise SnapshotCollectionAborted("snapshot_collection_gate_closed")

    related_paths = validate_intent_context_paths(related_paths)
    gate()
    merge_base = repo_reader.resolve_merge_base(target_base_sha, head_sha)
    changed = ()
    renames = {}
    base = head = None
    gate()
    if merge_base.is_comparable:
        changed = repo_reader.get_changed_files(merge_base.comparison_base_sha, head_sha)
        gate()
        renames = repo_reader.get_rename_map(merge_base.comparison_base_sha, head_sha)
        paths = tuple(sorted(set(related_paths) | {path for item in changed
            for path in (item.path, item.previous_path) if path}))
        gate()
        base = collect_intent_code_snapshot(repo_reader, merge_base.comparison_base_sha, paths)
        gate()
        head = collect_intent_code_snapshot(repo_reader, head_sha, paths)
        gate()
    stable_evidence = []
    for item in evidence:
        value = asdict(item)
        value["locator"].pop("retrieved_at", None)
        stable_evidence.append(value)
    digest = hash_json({"comparison": asdict(merge_base), "changed_files": [asdict(item) for item in changed],
        "related_paths": sorted(related_paths), "evidence": stable_evidence,
        "unavailable_sources": [asdict(item) for item in unavailable_sources],
        "base": asdict(base) if base is not None else None, "head": asdict(head) if head is not None else None})
    return SnapshotMaterial(merge_base, tuple(changed), renames, tuple(evidence), tuple(unavailable_sources),
        digest, base, head, captured_at, as_of)


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
    *,
    context_digest: str | None = None,
    captured_at: str | None = None,
) -> ReviewSnapshot:
    """Build immutable Service Envelope v2 ReviewSnapshot."""
    now_utc = datetime.now(UTC).isoformat()
    capture_time = captured_at or now_utc
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
    context_digest = context_digest or hashlib.sha256(context_raw).hexdigest()

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
    material: SnapshotMaterial | None = None,
) -> tuple[ReviewSnapshot, PrSnapshot, Classification, ModeSelection]:
    """Execute end-to-end M2 snapshot assembly pipeline."""
    title = title or ""
    body = body or ""
    # 1. Merge-base resolution
    if material is None and "intent" in available_modes:
        material = collect_snapshot_material(repo_reader, target_base_sha, head_sha, evidence=(EnvelopeEvidence(
            id="E-PR-01", kind="pr_text", version="head",
            locator={"repository": repository, "pr_number": pr_number},
            content={"title": title, "body": body}, digest=hashlib.sha256(f"{title}\n{body}".encode()).hexdigest(),
        ),))
    merge_base_res = material.merge_base if material is not None else repo_reader.resolve_merge_base(target_base_sha, head_sha)

    # 2. Changed files and rename map determination
    unavailable: list[UnavailableSource] = list(material.unavailable_sources) if material is not None else []
    if merge_base_res.is_comparable and merge_base_res.comparison_base_sha:
        comp_sha = merge_base_res.comparison_base_sha
        changed_files = material.changed_files if material is not None else repo_reader.get_changed_files(comp_sha, head_sha)
        rename_map = material.rename_map if material is not None else repo_reader.get_rename_map(comp_sha, head_sha)
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
    evidence = material.evidence if material is not None else ()
    if selector_uses_model and material is None:
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
        as_of=material.as_of if material is not None and material.as_of is not None else as_of,
        evidence=evidence,
        unavailable_sources=tuple(unavailable),
        **({"context_digest": hash_json({"repository": repository, "pr_number": pr_number,
            "repository_id": repository_id, "installation_id": installation_id,
            "intent_material": material.context_digest}), "captured_at": material.captured_at}
            if material is not None else {}),
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

    selection = constrain_architecture_to_java_gradle(review_snapshot, pr_snapshot, selection, repo_reader)
    return review_snapshot, pr_snapshot, classification, selection
