"""Lossless integration: paginate original results, never summarize them to fit."""
from __future__ import annotations

from copy import deepcopy

from review_bot.application.context import hash_json
from review_bot.application.source_partition import PartitionError


def native_integration_pages(rows, sources_for, fits, refine):
    """Retain results verbatim, splitting source delivery only after real overflow."""
    rows = list(rows)
    if not rows:
        raise PartitionError("missing_integration_results")
    def page(inputs, spans):
        return {"inputs": list(inputs), "sources": list(spans)}
    if fits(rows, sources_for(rows)):
        return [page(rows, sources_for(rows))]
    pages, active = [], []
    for row in rows:
        candidate = [*active, row]
        if fits(candidate, sources_for(candidate)):
            active = candidate
            continue
        if active:
            pages.append(page(active, sources_for(active)))
            active = []
        sources = sources_for([row])
        if fits([row], sources):
            active = [row]
            continue
        pending = []
        for span in sources:
            parts = [span] if fits([row], [span]) else refine(span, lambda pieces: fits([row], pieces))
            for part in parts:
                if pending and not fits([row], [*pending, part]):
                    pages.append(page([row], pending))
                    pending = []
                if not fits([row], [part]):
                    raise PartitionError("integration_fixed_result_exceeds_context")
                pending.append(part)
        if pending:
            pages.append(page([row], pending))
    if active:
        pages.append(page(active, sources_for(active)))
    return pages


def combine_final_pages(pages):
    """Preserve every assessment; do not invent a global semantic verdict.

    When one request fits it is the final model judgement. Multiple requests
    produce a composite review with their original per-perspective judgements.
    The program never changes them into a majority/severity vote.
    """
    if not pages:
        raise PartitionError("missing_final_pages")
    if len(pages) == 1:
        return deepcopy(pages[0])
    result = {
        "summary": "\n\n".join(page["summary"] for page in pages),
        "findings": [deepcopy(row) for page in pages for row in page["findings"]],
        "questions": [q for page in pages for q in page["questions"]],
        "perspectives": [],
        "missing_context": sorted({gap for page in pages for gap in page["missing_context"]}),
        "supplied_quotes": [],
        "observation_dispositions": [deepcopy(row) for page in pages for row in page["observation_dispositions"]],
        "context_updates": {key: deepcopy(value) for page in pages for key, value in page["context_updates"].items()},
        "target_reviews": {},
        "source_requests": [deepcopy(row) for page in pages for row in page.get("source_requests", [])],
    }
    quotes = {}
    for page in pages:
        for key, value in page["target_reviews"].items():
            result["target_reviews"].setdefault(key, []).append(deepcopy(value))
        for quote in page["supplied_quotes"]:
            quotes[(quote["evidence_id"], quote["start_byte"], quote["end_byte"])] = quote
    result["supplied_quotes"] = [deepcopy(quotes[key]) for key in sorted(quotes)]
    ids = list(dict.fromkeys(row["perspective_id"] for page in pages for row in page["perspectives"]))
    for identifier in ids:
        assessments = [deepcopy(row) for page in pages for row in page["perspectives"]
            if row["perspective_id"] == identifier]
        result["perspectives"].append({"perspective_id": identifier, "verdict": "partitioned",
            "statement": "\n\n".join(row["statement"] for row in assessments),
            "reasoning": "\n\n".join(row["reasoning"] for row in assessments),
            "failure_condition": "\n\n".join(row["failure_condition"] for row in assessments if row["failure_condition"]),
            "citations": [deepcopy(c) for row in assessments for c in row["citations"]],
            "assessments": assessments})
    return result


def _context_update_applies(previous, update):
    return not (update["state"] == "unresolved" and previous["state"] != "unresolved" and not update["citations"])


def prior_context_updates(pages):
    """Copy accepted state and its complete original evidence, without a summary."""
    states = {}
    for index, page in enumerate(pages):
        for key, update in page["final"]["context_updates"].items():
            previous = states.get(key, {}).get("update", {"state": "unresolved"})
            if _context_update_applies(previous, update):
                states[key] = {"page_index": index, "response_hash": hash_json(page["response"]),
                    "update": deepcopy(update)}
    return states


def reconcile_final_context(rows, pages, final):
    """Replay explicit, source-backed context transitions across all pages.

    A page that has not seen a different page's source cannot erase a verified
    resolution just by reporting uncertainty again without evidence.
    """
    states = {"M" + hash_json((row["observation_id"], gap))[:24]: {
        "description": gap, "state": "unresolved", "reason": "Initial provisional observation.", "citations": []}
        for row in rows for gap in row["missing_context"]}
    history = []
    for index, page in enumerate(pages):
        for key, update in page["context_updates"].items():
            if key not in states:
                raise PartitionError("unknown_context_update")
            previous = deepcopy(states[key])
            applied = _context_update_applies(previous, update)
            if applied:
                states[key] = {"description": previous["description"], **deepcopy(update)}
            history.append({"page": index, "context_id": key, "previous": previous,
                "update": deepcopy(update), "applied": applied})
    known = {row["description"] for row in states.values()}
    other_gaps = {gap for page in pages for gap in page["missing_context"] if gap not in known}
    final["missing_context"] = sorted(other_gaps | {row["description"] for row in states.values() if row["state"] == "unresolved"})
    final["context_updates"] = states
    final["context_update_history"] = history
    return final


def has_insufficient_judgement(rows):
    return any(row["verdict"] == "insufficient_context" or
        has_insufficient_judgement(row.get("assessments", [])) for row in rows)
