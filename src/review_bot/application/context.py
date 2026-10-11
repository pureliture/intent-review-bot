"""TaskInput assembly, Full-or-Not-Run Input Gate, and Evidence indexing."""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import asdict, replace
from difflib import SequenceMatcher

from review_bot.application.classification import (
    classify_pr,  # Re-exported for test compatibility
)
from review_bot.domain.models_v1 import (
    AnalysisDelta,
    Classification,
    CodeFile,
    CodeSnapshot,
    CpdDuplication,
    DeltaSignal,
    Evidence,
    Finding,
    PmdViolation,
    SelectedSignal,
    Snapshot,
    TaskInput,
)


def hash_json(value: object) -> str:
    """Return canonical SHA-256 digest of JSON-serialized object."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fixed_sha(sha: str) -> str:
    if not isinstance(sha, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", sha):
        raise ValueError("A full fixed Git SHA is required, not a branch or revision expression")
    return sha.lower()


def _safe_path(path: str) -> bool:
    return (
        isinstance(path, str)
        and bool(path)
        and not path.startswith("/")
        and "\\" not in path
        and not any(ord(char) < 32 or ord(char) == 127 for char in path)
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )


def _lines(file: CodeFile) -> list[str]:
    return file.content.splitlines(keepends=True)


def _valid_span(file: CodeFile | None, begin: int, end: int) -> bool:
    if file is None or not isinstance(begin, int) or not isinstance(end, int):
        return False
    line_count = len(_lines(file))
    return 1 <= begin <= end <= line_count


def _fragment_span(file: CodeFile, fragment: str | None) -> tuple[int, int] | None:
    if not fragment or file.content.count(fragment) != 1:
        return None
    start = file.content.index(fragment)
    boundaries = []
    offset = 0
    for line in _lines(file):
        offset += len(line)
        boundaries.append(offset)
    return bisect_right(boundaries, start) + 1, bisect_right(boundaries, start + len(fragment) - 1) + 1


def _locations(findings: tuple[Finding, ...]) -> Iterable[tuple[str, int, int, str | None]]:
    for finding in findings:
        if isinstance(finding, PmdViolation):
            yield finding.path, finding.begin_line, finding.end_line, finding.code_fragment
        elif isinstance(finding, CpdDuplication):
            for occurrence in finding.occurrences:
                yield (
                    occurrence.path,
                    occurrence.begin_line,
                    occurrence.end_line,
                    occurrence.code_fragment or finding.code_fragment,
                )


def _changed_spans(snapshot: Snapshot) -> tuple[dict[str, dict[str, list[tuple[int, int]]]], set[tuple[str, str]]]:
    files = {
        version: {file.path: file for file in getattr(snapshot, version).files}
        for version in ("base", "head")
    }
    declared = set(snapshot.pr.changed_files) | set(snapshot.pr.rename_map) | set(snapshot.pr.rename_map.values())
    spans: dict[str, dict[str, list[tuple[int, int]]]] = {"base": {}, "head": {}}
    uncertain: set[tuple[str, str]] = set()
    pairs = [(path, snapshot.pr.rename_map.get(path, path)) for path in files["base"]]
    paired_head = {new for _, new in pairs}
    pairs.extend((path, path) for path in files["head"] if path not in paired_head)
    for old, new in pairs:
        if old not in declared and new not in declared:
            continue
        before = files["base"].get(old)
        after = files["head"].get(new)
        missing = [
            (version, path)
            for version, path, file in (("base", old, before), ("head", new, after))
            if file is None
        ]
        if any(
            not getattr(snapshot, version).complete
            or any(entry.path == path for entry in getattr(snapshot, version).exclusions)
            for version, path in missing
        ):
            uncertain.update((("base", old), ("head", new)))
            continue
        old_lines = _lines(before) if before else []
        new_lines = _lines(after) if after else []
        for operation, i1, i2, j1, j2 in SequenceMatcher(None, old_lines, new_lines, autojunk=False).get_opcodes():
            if operation == "equal":
                continue
            if i1 < i2:
                spans["base"].setdefault(old, []).append((i1 + 1, i2))
            if j1 < j2:
                spans["head"].setdefault(new, []).append((j1 + 1, j2))
    return spans, uncertain


def _overlaps(begin: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(begin <= changed_end and changed_begin <= end for changed_begin, changed_end in spans)


def _provenance(value: object) -> bool:
    return (
        bool(value.strip())
        if isinstance(value, str)
        else isinstance(value, (dict, list, tuple)) and bool(value)
    )


def _relevance(
    snapshot: Snapshot,
    signal: DeltaSignal,
    spans: dict[str, dict[str, list[tuple[int, int]]]],
    uncertain: set[tuple[str, str]],
) -> tuple[str, str]:
    for observation in snapshot.observations:
        if (
            observation.get("related_to_pr") is True
            and _provenance(observation.get("provenance"))
            and signal.signal_id in observation.get("signal_ids", ())
        ):
            return "related", "Caller observation with explicit provenance establishes PR relationship"
    problems = []
    seen = False
    for version, findings in (("base", signal.base_findings), ("head", signal.head_findings)):
        files = {file.path: file for file in getattr(snapshot, version).files}
        for path, begin, end, fragment in _locations(findings):
            seen = True
            file = files.get(path)
            if not _valid_span(file, begin, end) or (version, path) in uncertain:
                problems.append(f"{version}:{path}:{begin}-{end} cannot be verified")
                continue
            changed = spans[version].get(path, [])
            if _overlaps(begin, end, changed):
                return "related", f"{version}:{path}:{begin}-{end} overlaps verified PR changed-line spans"
            fragment_span = _fragment_span(file, fragment)
            if fragment_span:
                fragment_begin, fragment_end = fragment_span
                if (
                    fragment_begin <= end
                    and begin <= fragment_end
                    and _overlaps(fragment_begin, fragment_end, changed)
                ):
                    return (
                        "related",
                        f"Verified {version}:{path}:{fragment_begin}-{fragment_end} source fragment overlaps PR changed-line spans",
                    )
    if problems or not seen:
        return "uncertain", "PR relationship cannot be established: " + ("; ".join(problems) or "no finding locations")
    return "unrelated", "No finding span or verified source fragment overlaps PR changed-line spans"


def _select(
    snapshot: Snapshot,
    delta: AnalysisDelta,
    classification: Classification,
) -> tuple[SelectedSignal, ...]:
    spans, uncertain = _changed_spans(snapshot)
    selected: list[SelectedSignal] = []
    for signal in delta.signals:
        if signal.change == "unknown" or not delta.comparable:
            selected.append(
                SelectedSignal(
                    signal,
                    "limitation",
                    "Unknown or noncomparable analysis must not become an action item",
                )
            )
            continue
        if classification.review_mode == "general" and signal.change in ("new", "worsened"):
            selected.append(
                SelectedSignal(signal, "review", "General review targets new and worsened findings")
            )
            continue
        if classification.review_mode == "refactoring" and signal.change in ("improved", "resolved"):
            selected.append(
                SelectedSignal(
                    signal, "review", "Refactoring review includes improvements and resolutions"
                )
            )
            continue
        relevance, reason = _relevance(snapshot, signal, spans, uncertain)
        if classification.review_mode == "refactoring":
            role = (
                "review"
                if relevance == "related"
                else "limitation"
                if relevance == "uncertain"
                else "context"
            )
            selected.append(SelectedSignal(signal, role, reason))
        elif signal.change == "existing" and relevance != "unrelated":
            selected.append(
                SelectedSignal(
                    signal,
                    "context" if relevance == "related" else "limitation",
                    reason,
                )
            )
    return tuple(selected)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _code_scope(snapshot: CodeSnapshot) -> dict[str, object]:
    coverage = {
        "paths": [file.path for file in snapshot.files],
        "file_count": len(snapshot.files),
        "byte_count": sum(len(file.content.encode("utf-8")) for file in snapshot.files),
    }
    return {
        "sha": snapshot.sha,
        "inventory_hash": snapshot.inventory_hash,
        "inventory_count": len(snapshot.files) + len(snapshot.exclusions),
        "complete": snapshot.complete,
        "errors": list(snapshot.errors),
        "acquired": dict(coverage),
        "provided": dict(coverage),
        "excluded": {
            "entries": [asdict(entry) for entry in snapshot.exclusions],
            "file_count": len(snapshot.exclusions),
            "byte_count": None,
        },
        "content_hashes": {file.path: file.content_hash for file in snapshot.files},
        "kind_counts": {
            kind: sum(file.kind == kind for file in snapshot.files)
            for kind in ("source", "test", "build")
        },
    }


def _validate(snapshot: Snapshot, delta: AnalysisDelta, classification: Classification) -> None:
    for version, code, pr_sha, delta_sha in (
        ("base", snapshot.base, snapshot.pr.base_sha, delta.base_sha),
        ("head", snapshot.head, snapshot.pr.head_sha, delta.head_sha),
    ):
        if len({_fixed_sha(code.sha), _fixed_sha(pr_sha), _fixed_sha(delta_sha)}) != 1:
            raise ValueError(f"Mismatched {version} fixed versions")
        if not re.fullmatch(r"[0-9a-f]{64}", code.inventory_hash):
            raise ValueError(f"Invalid {version} inventory hash")
        paths = set()
        for file in code.files:
            if not _safe_path(file.path) or file.path in paths or file.kind not in ("source", "test", "build"):
                raise ValueError(f"Invalid or duplicate {version} code file: {file.path}")
            paths.add(file.path)
            if hashlib.sha256(file.content.encode("utf-8")).hexdigest() != file.content_hash:
                raise ValueError(f"Invalid {version} source content hash: {file.path}")
    if delta.schema_version != 1:
        raise ValueError("Unsupported delta schema version")
    if not snapshot.pr.repository or snapshot.pr.number <= 0:
        raise ValueError("Repository and PR number must be supplied by the caller")
    if any(not _safe_path(path) for path in snapshot.pr.changed_files):
        raise ValueError("Invalid changed-file path")
    if any(not _safe_path(old) or not _safe_path(new) for old, new in snapshot.pr.rename_map.items()):
        raise ValueError("Invalid rename path")
    if len(set(snapshot.pr.rename_map.values())) != len(snapshot.pr.rename_map):
        raise ValueError("Ambiguous rename map")
    expected_class = classify_pr(snapshot.pr.source_branch, snapshot.pr.labels)
    if classification.review_mode != expected_class.review_mode:
        raise ValueError("Classification does not match the supplied branch and labels")
    ids = set()
    for signal in delta.signals:
        if not signal.signal_id or signal.signal_id in ids:
            raise ValueError("Invalid or duplicate signal ID")
        ids.add(signal.signal_id)
        if signal.change not in ("existing", "new", "worsened", "improved", "resolved", "unknown"):
            raise ValueError("Invalid delta change")
        finding_type = {"pmd": PmdViolation, "cpd": CpdDuplication}.get(signal.tool)
        if finding_type is None or any(
            not isinstance(finding, finding_type)
            for finding in signal.base_findings + signal.head_findings
        ):
            raise ValueError("Signal findings do not match the analysis tool")
    for kind, records in (("criterion", snapshot.explicit_criteria), ("observation", snapshot.observations)):
        for record in records:
            if not isinstance(record, dict):
                raise TypeError(f"{kind} must be an explicit dictionary")
            signal_ids = record.get("signal_ids", ())
            if not isinstance(signal_ids, (list, tuple)) or any(not isinstance(signal_id, str) or not signal_id for signal_id in signal_ids):
                raise ValueError(f"{kind} signal_ids must be a sequence of exact signal IDs")
            if kind == "criterion" and (not _provenance(record.get("provenance")) or not _provenance(record.get("applicability"))):
                raise ValueError("Explicit criteria require provenance and applicability")
            v = record.get("version", "both")
            if v not in ("base", "head", "both"):
                raise ValueError(f"Invalid {kind} version")
            if "sha" in record and (v == "both" or _fixed_sha(record["sha"]) != _fixed_sha(getattr(snapshot, v).sha)):
                raise ValueError(f"Mismatched {kind} SHA")


def check_input_gate(
    snapshot: Snapshot,
    delta: AnalysisDelta | None = None,
    limits: dict[str, object] | None = None,
) -> tuple[bool, str, str]:
    """Validate full code snapshot completeness, errors, and byte/token limits.

    Returns (can_proceed: bool, status: str, reason: str).
    """
    if (
        not snapshot.base.complete
        or not snapshot.head.complete
        or bool(snapshot.base.errors)
        or bool(snapshot.head.errors)
    ):
        return False, "not_run", "incomplete_code_input"

    if limits is not None:
        max_bytes = limits.get("max_input_bytes")
        if max_bytes is not None:
            base_bytes = sum(len(f.content.encode("utf-8")) for f in snapshot.base.files)
            head_bytes = sum(len(f.content.encode("utf-8")) for f in snapshot.head.files)
            if (base_bytes + head_bytes) > int(max_bytes):
                return False, "not_run", "input_limit_exceeded"

        max_tokens = limits.get("max_input_tokens")
        token_count = limits.get("token_count")
        if max_tokens is not None and token_count is not None and int(token_count) > int(max_tokens):
            return False, "not_run", "input_limit_exceeded"

    return True, "can_proceed", ""


def build_task_input(
    snapshot: Snapshot,
    delta: AnalysisDelta,
    classification: Classification,
    limits: dict[str, object] | None = None,
) -> TaskInput:
    """Build immutable TaskInput with full code context and indexed Evidence."""
    _validate(snapshot, delta, classification)
    selected = _select(snapshot, delta, classification)
    limitations: list[str] = list(delta.limitations)
    evidence: dict[str, Evidence] = {}
    code = {
        version: {file.path: file for file in getattr(snapshot, version).files}
        for version in ("base", "head")
    }

    def add_code(
        version: str,
        file: CodeFile,
        begin: int,
        end: int,
        signal_ids: tuple[str, ...] = (),
    ) -> str:
        sha = getattr(snapshot, version).sha
        evidence_id = "code-" + hash_json((version, sha, file.path, file.content_hash, begin, end))
        existing = evidence.get(evidence_id)
        ids = tuple(sorted(set(signal_ids) | set(existing.signal_ids if existing else ())))
        evidence[evidence_id] = Evidence(
            evidence_id,
            "code",
            version,
            sha,
            file.path,
            begin,
            end,
            "".join(_lines(file)[begin - 1:end]) if begin else "",
            ids,
        )
        return evidence_id

    for version in ("base", "head"):
        source = getattr(snapshot, version)
        if not source.complete:
            limitations.append(f"{version} code snapshot is incomplete")
        limitations.extend(f"{version}: {error}" for error in source.errors)
        if source.exclusions:
            limitations.append(
                f"{version} excluded paths are not supplied; excluded byte counts are unavailable in CodeSnapshot"
            )
        for file in sorted(source.files, key=lambda f: f.path):
            line_count = len(_lines(file))
            add_code(version, file, 1 if line_count else 0, line_count)

    for item in selected:
        signal = item.signal
        limitations.extend(signal.limitations)
        if item.role == "limitation":
            limitations.append(f"{signal.signal_id}: {item.reason}")
        for version, findings in (("base", signal.base_findings), ("head", signal.head_findings)):
            if not findings and (signal.base_findings or signal.head_findings):
                continue
            locations = []
            for path, begin, end, _ in _locations(findings):
                file = code[version].get(path)
                location: dict[str, object] = {
                    "path": path,
                    "begin_line": begin,
                    "end_line": end,
                    "content_hash": file.content_hash if file else None,
                    "code_evidence_id": None,
                }
                if _valid_span(file, begin, end):
                    location["code_evidence_id"] = add_code(version, file, begin, end, (signal.signal_id,))
                else:
                    limitations.append(f"{signal.signal_id}: code evidence unavailable at {version}:{path}:{begin}-{end}")
                locations.append(location)
            content = _json({
                "signal": asdict(signal),
                "role": item.role,
                "reason": item.reason,
                "locations": locations,
            })
            evidence_id = "signal-" + hash_json((signal.signal_id, version, content))
            evidence[evidence_id] = Evidence(
                evidence_id,
                "signal",
                version,
                getattr(snapshot, version).sha,
                "",
                0,
                0,
                content,
                (signal.signal_id,),
            )

    for kind, records in (("criterion", snapshot.explicit_criteria), ("observation", snapshot.observations)):
        for index, record in enumerate(records):
            if kind == "observation" and not _provenance(record.get("provenance")):
                limitations.append(f"Observation {index} lacks provenance and cannot establish PR relationship")
            versions = ("base", "head") if record.get("version", "both") == "both" else (record["version"],)
            for version in versions:
                content = _json(record)
                evidence_id = kind + "-" + hash_json((index, version, content))
                ids = tuple(
                    signal_id
                    for signal_id in record.get("signal_ids", ())
                    if any(item.signal.signal_id == signal_id for item in selected)
                )
                evidence[evidence_id] = Evidence(
                    evidence_id,
                    kind,
                    version,
                    getattr(snapshot, version).sha,
                    "",
                    0,
                    0,
                    content,
                    ids,
                )

    evidence_values = tuple(evidence.values())
    scope = {
        "acquisition": "fixed_git_sha_data_only",
        "code": {version: _code_scope(getattr(snapshot, version)) for version in ("base", "head")},
        "changed_files": list(snapshot.pr.changed_files),
        "rename_map": dict(snapshot.pr.rename_map),
        "signals": {
            "available": len(delta.signals),
            "selected": len(selected),
            **{role: sum(item.role == role for item in selected) for role in ("review", "context", "limitation")},
        },
        "evidence": {
            **{kind: sum(item.kind == kind for item in evidence_values) for kind in ("code", "signal", "criterion", "observation")},
            "content_bytes": sum(len(item.content.encode("utf-8")) for item in evidence_values),
        },
    }
    task = TaskInput(
        schema_version=1,
        snapshot=snapshot,
        delta=delta,
        classification=classification,
        signals=selected,
        evidence=evidence_values,
        scope=scope,
        limitations=tuple(dict.fromkeys(limitations)),
        input_hash="",
    )
    return replace(task, input_hash=hash_json(asdict(task)))
