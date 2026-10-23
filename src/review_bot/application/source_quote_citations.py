"""Resolve literal quotes only within frozen source and delivered coverage."""
from __future__ import annotations

from collections.abc import Iterable

from review_bot.application.source_partition import PartitionError, SourceOriginal, SourceSpan, make_span


def resolve_quote_citations(
    citations: list[dict[str, str]],
    originals: dict[str, SourceOriginal],
    delivered_spans: Iterable[SourceSpan],
) -> list[SourceSpan]:
    """Require one exact whole-source occurrence, wholly delivered to this request.

    Quote whitespace and Unicode are never normalized. Citation order and every
    duplicate remain observable: duplicates are rejected rather than removed.
    """
    if not isinstance(citations, list):
        raise PartitionError("citation_quote_invalid_array")
    if not isinstance(originals, dict) or any(
        not isinstance(key, str) or not key.strip() or not isinstance(original, SourceOriginal)
        or original.evidence_id != key or not isinstance(original.text, str)
        for key, original in originals.items()
    ):
        raise PartitionError("citation_quote_invalid_originals")
    coverage: dict[str, list[tuple[int, int]]] = {}
    try:
        for span in delivered_spans:
            if not isinstance(span, SourceSpan) or span.evidence_id not in originals:
                raise PartitionError("citation_quote_invalid_delivery")
            original = originals[span.evidence_id]
            canonical = make_span(original, span.start_byte, span.end_byte, span.boundary)
            if canonical != span:
                raise PartitionError("citation_quote_invalid_delivery")
            if span.start_byte < span.end_byte:
                coverage.setdefault(span.evidence_id, []).append((span.start_byte, span.end_byte))
    except (TypeError, UnicodeError, PartitionError):
        raise PartitionError("citation_quote_invalid_delivery") from None
    for key, ranges in coverage.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        coverage[key] = merged
    resolved, seen = [], set()
    for citation in citations:
        if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
            raise PartitionError("citation_quote_invalid_object")
        source_id, quote = citation["source_id"], citation["quote"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise PartitionError("citation_quote_invalid_source_id")
        if not isinstance(quote, str) or not quote.strip():
            raise PartitionError("citation_quote_invalid_quote")
        if source_id not in originals:
            raise PartitionError("citation_quote_unknown_source")
        if source_id not in coverage:
            raise PartitionError("citation_quote_source_not_delivered")
        try:
            needle, raw = quote.encode("utf-8"), originals[source_id].raw
        except UnicodeError:
            raise PartitionError("citation_quote_invalid_quote") from None
        start = raw.find(needle)
        if start < 0:
            raise PartitionError("citation_quote_not_found")
        if raw.find(needle, start + 1) >= 0:
            raise PartitionError("citation_quote_ambiguous")
        end = start + len(needle)
        if not any(begin <= start and end <= stop for begin, stop in coverage[source_id]):
            raise PartitionError("citation_quote_undelivered")
        identity = (source_id, start, end)
        if identity in seen:
            raise PartitionError("citation_quote_duplicate")
        seen.add(identity)
        resolved.append(make_span(originals[source_id], start, end, "citation"))
    return resolved
