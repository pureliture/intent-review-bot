from __future__ import annotations

from review_bot.application.source_citations import citation_spans, citation_catalog, MAX_CITATION_BYTES
from review_bot.application.source_partition import SourceOriginal, make_span


def test_catalog_is_deterministic_and_covers_full_original_without_truncation():
    text = "".join(f"def 함수{i}(): return '한글 원문'\n" for i in range(50)) + "가" * 400
    original = SourceOriginal("S1", text, "source", "app.py", "head", "b" * 40)
    supplied = make_span(original, 0, len(original.raw), "whole_original")
    first = citation_spans([supplied], {"S1": original})
    assert first == citation_spans([supplied], {"S1": original})
    assert b"".join(s.text.encode() for s in first.values()) == original.raw
    assert all(len(s.text.encode()) <= MAX_CITATION_BYTES for s in first.values())
    assert len(first) == len(set(first))
    assert all(row["end_byte"] > row["start_byte"] for c in citation_catalog([supplied], {"S1": original}) for row in c["ranges"])


def test_reusing_a_quote_preserves_its_id_and_entire_range():
    original = SourceOriginal("S1", "".join(f"line {i}\n" for i in range(30)), "source")
    supplied = make_span(original, 0, len(original.raw), "whole_original")
    first = next(iter(citation_spans([supplied], {"S1": original}).values()))
    narrowed = citation_spans([first], {"S1": original})
    assert len(narrowed) == 1
    assert list(narrowed) == [next(iter(citation_spans([supplied], {"S1": original})))]
    assert b"".join(s.text.encode() for s in narrowed.values()) == first.text.encode()
    assert all(first.start_byte <= s.start_byte < s.end_byte <= first.end_byte for s in narrowed.values())


def test_source_blocks_pair_each_reference_with_its_exact_original_and_keep_all_bytes():
    from review_bot.application.source_citations import source_blocks
    text = "import os\n" * 9 + "def test_result():\n    assert actual == '∞'\n" + "한" * 700
    original = SourceOriginal("S1", text, "source", "test_app.py", "head", "b" * 40)
    span = make_span(original, 0, len(original.raw), "whole_original")
    sources = {original.evidence_id: original}
    payload = source_blocks([span], sources)
    assert payload == source_blocks([span], sources)
    blocks = payload[0]["blocks"]
    assert "".join(block["text"] for block in blocks).encode() == original.raw
    ranges = citation_spans([span], sources)
    assert len(blocks) == len(ranges)
    for block in blocks:
        reference = ranges[block["reference_id"]]
        assert block["text"].encode() == original.raw[reference.start_byte:reference.end_byte]
        assert (block["begin_line"], block["end_line"]) == (reference.begin_line, reference.end_line)
    assertion = next(block for block in blocks if "assert actual" in block["text"])
    assert assertion["reference_id"] != blocks[0]["reference_id"]


def test_cpd_partner_anchors_use_only_fixed_originals_and_exact_report_locations():
    from review_bot.application.source_citations import signal_context_spans
    from review_bot.application.source_partition import PartitionError
    import pytest
    originals = [SourceOriginal(name, "header\n" + "가나다" * 200 + "\n", "source", name + ".java", "head", "b" * 40) for name in ("A", "B")]
    target = {"target_id": "cpd-one", "tool": "cpd", "locations": [
        {"path": name + ".java", "version": "head", "begin_line": 2, "end_line": 2} for name in ("A", "B")]}
    spans = signal_context_spans(originals, [target])
    assert {s.path for s in spans} == {"A.java", "B.java"}
    assert all(s.begin_line == 2 and s.text == "가나다" * 200 + "\n" for s in spans)
    assert all(s.text.encode() in originals[i].raw for i, s in enumerate(spans))
    with pytest.raises(PartitionError, match="original_unavailable"):
        signal_context_spans(originals[:1], [target])
