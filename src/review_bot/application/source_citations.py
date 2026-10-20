"""Bounded original-range choices; models select IDs and never calculate offsets."""
from __future__ import annotations

from review_bot.application.context import hash_json
from dataclasses import replace
from review_bot.application.source_partition import span_identity, make_span, PartitionError

MAX_CITATION_BYTES = 512


def citation_spans(spans, originals, anchors=()):
    spans = tuple(spans)
    result = {}
    for span in spans:
        raw = span.text.encode("utf-8")
        original = originals[span.evidence_id]
        if span.source_hash != original.digest or original.raw[span.start_byte:span.end_byte] != raw:
            raise PartitionError("citation_source_mismatch")
        cursor, line = 0, span.begin_line
        # Reusing a quote must keep the same ID/range, not progressively shrink
        # it to its first line at each integration boundary.
        line_limit = 8
        while cursor < len(raw):
            end = cursor
            for _ in range(line_limit):
                newline = raw.find(b"\n", end)
                candidate = len(raw) if newline < 0 else newline + 1
                if candidate - cursor > MAX_CITATION_BYTES:
                    end = min(cursor + MAX_CITATION_BYTES, len(raw)) if end == cursor else end
                    break
                end = candidate
                if end == len(raw):
                    break
            while end > cursor:
                try:
                    raw[cursor:end].decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end -= 1
            if end <= cursor:
                raise PartitionError("invalid_citation_unicode_boundary")
            chunk = raw[cursor:end]
            start_byte, end_byte = span.start_byte + cursor, span.start_byte + end
            quote = replace(span, unit_id=span_identity(original, start_byte, end_byte),
                start_byte=start_byte, end_byte=end_byte, begin_line=line,
                end_line=line + chunk.count(b"\n", 0, max(0, len(chunk) - 1)), text=chunk.decode("utf-8"), boundary="citation")
            line += chunk.count(b"\n")
            key = "Q" + hash_json((quote.evidence_id, quote.source_hash, quote.start_byte, quote.end_byte))[:12]
            if key in result and result[key] != quote:
                raise PartitionError("citation_reference_collision")
            result[key] = quote
            cursor = end
    for anchor in anchors:
        identifier, start, end = anchor["evidence_id"], anchor["start_byte"], anchor["end_byte"]
        if not any(span.evidence_id == identifier and span.start_byte <= start < end <= span.end_byte for span in spans):
            continue
        original = originals[identifier]
        quote = make_span(original, start, end, "citation")
        key = "Q" + hash_json((identifier, original.digest, start, end))[:12]
        if key in result and result[key] != quote:
            raise PartitionError("citation_reference_collision")
        result[key] = quote
    return result


def citation_catalog(spans, originals, anchors=()):
    grouped = {}
    for key, span in citation_spans(spans, originals, anchors).items():
        grouped.setdefault(span.evidence_id, []).append({"reference_id": key,
            "begin_line": span.begin_line, "end_line": span.end_line,
            "start_byte": span.start_byte, "end_byte": span.end_byte})
    return [{"evidence_id": key, "path": originals[key].path, "version": originals[key].version,
        "sha": originals[key].sha, "ranges": rows} for key, rows in grouped.items()]


def source_blocks(spans, originals):
    """Put reference IDs beside their exact source, without repeating source bytes.

    Blocks are a presentation of every supplied byte in the SAME request, not
    a new model-call partition or a selection of preferred evidence.
    """
    result = []
    for span in spans:
        blocks = [{"reference_id": key, "begin_line": chunk.begin_line,
            "end_line": chunk.end_line, "text": chunk.text}
            for key, chunk in citation_spans([span], originals).items()]
        if "".join(block["text"] for block in blocks) != span.text:
            raise PartitionError("source_block_coverage_mismatch")
        result.append({"evidence_id": span.evidence_id, "source_hash": span.source_hash,
            "path": span.path, "version": span.version, "sha": span.sha,
            "start_byte": span.start_byte, "end_byte": span.end_byte, "blocks": blocks})
    return result


def signal_context_spans(originals, details):
    """Deterministic CPD partner anchors from already collected fixed originals."""
    sources = {(o.path, o.version): o for o in originals if o.path}
    result = {}
    for target in details:
        if target["tool"] != "cpd":
            continue
        for location in target["locations"]:
            if location["version"] != "head":
                continue
            original = sources.get((location["path"], "head"))
            if original is None:
                raise PartitionError("signal_original_unavailable")
            lines = original.raw.splitlines(keepends=True)
            begin, end = location["begin_line"], location["end_line"]
            if not 1 <= begin <= end <= len(lines):
                raise PartitionError("signal_location_outside_original")
            start = sum(len(line) for line in lines[:begin - 1])
            raw = b"".join(lines[begin - 1:end])
            span = make_span(original, start, start + len(raw), "signal_source")
            result[(span.evidence_id, span.start_byte, span.end_byte)] = span
    return list(result.values())
