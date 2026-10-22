from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from review_bot.application.source_partition import PartitionError, SourceOriginal, make_span
from review_bot.application.source_quote_citations import resolve_quote_citations


def original(text, source_id="HEAD", version="head"):
    return SourceOriginal(source_id, text, "source", "src/검토.py", version, "b" * 40)


def whole(source):
    return make_span(source, 0, len(source.raw), "delivered")


def quote(source_id, text):
    return {"source_id": source_id, "quote": text}


def test_korean_exact_utf8_bytes_lines_and_metadata_are_canonical():
    source = original("첫 줄\n  한글 인용\n다음 줄\n")
    text = "  한글 인용\n다음 줄"
    resolved = resolve_quote_citations([quote("HEAD", text)], {"HEAD": source}, [whole(source)])
    start = len("첫 줄\n".encode())
    expected = make_span(source, start, start + len(text.encode()), "citation")
    assert resolved == [expected]
    assert (expected.begin_line, expected.end_line) == (2, 3)
    assert expected.text == text and expected.source_hash == hashlib.sha256(source.raw).hexdigest()
    assert (expected.path, expected.version, expected.sha) == (source.path, "head", "b" * 40)


@pytest.mark.parametrize("text,needle", [("ababa", "aba"), ("가가가", "가가")])
def test_overlapping_occurrences_in_whole_original_are_ambiguous_even_if_only_one_is_delivered(text, needle):
    source = original(text)
    delivered = make_span(source, 0, len(needle.encode()), "delivered")
    with pytest.raises(PartitionError, match="^citation_quote_ambiguous$"):
        resolve_quote_citations([quote("HEAD", needle)], {"HEAD": source}, [delivered])


def test_declared_source_and_base_head_are_isolated_and_input_order_is_preserved():
    base, head = original("same literal\nold only", "BASE", "base"), original("same literal\nnew only")
    sources = {"BASE": base, "HEAD": head}
    with pytest.raises(PartitionError, match="^citation_quote_source_not_delivered$"):
        resolve_quote_citations([quote("HEAD", "same literal")], sources, [whole(base)])
    with pytest.raises(PartitionError, match="^citation_quote_not_found$"):
        resolve_quote_citations([quote("HEAD", "old only")], sources, [whole(base), whole(head)])
    resolved = resolve_quote_citations([quote("HEAD", "same literal"), quote("BASE", "same literal")],
        sources, [whole(base), whole(head)])
    assert [span.version for span in resolved] == ["head", "base"]
    assert [span.evidence_id for span in resolved] == ["HEAD", "BASE"]


@pytest.mark.parametrize("text,needle", [("x\r\ny", "x\ny"), ("가", "가"), ("Case", "case"), ("x  y", "x y")])
def test_matching_never_normalizes_newlines_unicode_case_or_whitespace(text, needle):
    source = original(text)
    with pytest.raises(PartitionError, match="^citation_quote_not_found$"):
        resolve_quote_citations([quote("HEAD", needle)], {"HEAD": source}, [whole(source)])


def test_matching_retains_leading_and_trailing_whitespace():
    source = original("prefix\n  selected  \nsuffix")
    result = resolve_quote_citations([quote("HEAD", "  selected  ")], {"HEAD": source}, [whole(source)])
    assert result[0].text == "  selected  "
    assert result[0].start_byte == len("prefix\n".encode())


def test_delivered_union_accepts_unordered_adjacent_and_overlapping_spans_but_rejects_a_gap():
    source = original("prefix SELECTED suffix")
    spans = [make_span(source, 11, 15, "last"), make_span(source, 7, 12, "first")]
    result = resolve_quote_citations([quote("HEAD", "SELECTED")], {"HEAD": source}, iter(spans))
    assert result == [make_span(source, 7, 15, "citation")]
    adjacent = [make_span(source, 10, 15, "second"), make_span(source, 7, 10, "first")]
    assert resolve_quote_citations([quote("HEAD", "SELECTED")], {"HEAD": source}, adjacent) == result
    gap = [make_span(source, 7, 10, "first"), make_span(source, 11, 15, "last")]
    with pytest.raises(PartitionError, match="^citation_quote_undelivered$"):
        resolve_quote_citations([quote("HEAD", "SELECTED")], {"HEAD": source}, gap)


@pytest.mark.parametrize("text", ["", " \t\r\n", None, True, "\ud800"])
def test_quotes_must_be_nonblank_valid_literal_utf8(text):
    source = original("literal")
    with pytest.raises(PartitionError, match="^citation_quote_invalid_quote$"):
        resolve_quote_citations([quote("HEAD", text)], {"HEAD": source}, [whole(source)])


@pytest.mark.parametrize("citation", [{"source_id": "HEAD", "quote": "literal", "start_byte": 0},
    {"source_id": "HEAD", "quote": "literal", "reference_id": "Q1"}, {"source_id": "HEAD"}, "literal"])
def test_model_coordinates_extra_or_missing_fields_are_rejected(citation):
    source = original("literal")
    with pytest.raises(PartitionError, match="^citation_quote_invalid_object$"):
        resolve_quote_citations([citation], {"HEAD": source}, [whole(source)])


@pytest.mark.parametrize("source_id,reason", [("UNKNOWN", "unknown_source"), ("", "invalid_source_id"),
    (" ", "invalid_source_id"), (1, "invalid_source_id")])
def test_source_id_must_be_exact_and_known(source_id, reason):
    source = original("literal")
    with pytest.raises(PartitionError, match="^citation_quote_" + reason + "$"):
        resolve_quote_citations([quote(source_id, "literal")], {"HEAD": source}, [whole(source)])


def test_duplicate_citations_are_rejected_instead_of_silently_dropped():
    source = original("literal")
    citations = [quote("HEAD", "literal"), quote("HEAD", "literal")]
    with pytest.raises(PartitionError, match="^citation_quote_duplicate$"):
        resolve_quote_citations(citations, {"HEAD": source}, [whole(source)])
    assert len(citations) == 2 and source.text == "literal"


def test_quote_can_exceed_512_bytes_and_multiple_lines_without_shortening():
    text = "BEGIN\n" + "한글 원문 보존\n" * 100 + "END"
    source = original("prefix\n" + text + "\nsuffix")
    result = resolve_quote_citations([quote("HEAD", text)], {"HEAD": source}, [whole(source)])
    assert len(text.encode()) > 512
    assert result[0].text == text and result[0].end_byte - result[0].start_byte == len(text.encode())


def test_delivered_metadata_must_match_declared_frozen_source():
    source = original("literal")
    delivered = whole(source)
    for invalid in (replace(delivered, version="base"), replace(delivered, source_hash="forged"),
            replace(delivered, text="changed"), replace(delivered, end_byte=999)):
        with pytest.raises(PartitionError, match="^citation_quote_invalid_delivery$"):
            resolve_quote_citations([quote("HEAD", "literal")], {"HEAD": source}, [invalid])
    with pytest.raises(PartitionError, match="^citation_quote_invalid_originals$"):
        resolve_quote_citations([quote("OTHER", "literal")], {"OTHER": source}, [delivered])


def test_array_contract_empty_citations_and_empty_coverage():
    source = original("literal")
    assert resolve_quote_citations([], {"HEAD": source}, []) == []
    with pytest.raises(PartitionError, match="^citation_quote_invalid_array$"):
        resolve_quote_citations(quote("HEAD", "literal"), {"HEAD": source}, [whole(source)])
    with pytest.raises(PartitionError, match="^citation_quote_source_not_delivered$"):
        resolve_quote_citations([quote("HEAD", "literal")], {"HEAD": source}, [])


@pytest.mark.parametrize("stage", ["fact_check", "fact_rewrite"])
@pytest.mark.parametrize("protocol", ["v3", "v4"])
def test_shared_model_fixture_uses_actual_delivered_text_and_retains_legacy_schema_branch(stage, protocol):
    import json
    from dataclasses import asdict
    from review_bot.application.source_citations import source_blocks, citation_catalog
    from review_bot.domain.models_v1 import ModelRequest
    from tests.fixtures.partition_review import PartitionModelClient

    source = original("def unique_fixture():\n    return '원문 그대로'\n")
    delivered = whole(source)
    sources = {source.evidence_id: source}
    citation_schema = ({"type": "object", "properties": {"source_id": {}, "quote": {}}} if protocol == "v4"
        else {"type": "object", "properties": {"reference_id": {}}})
    schema = {"type": "object", "properties": {"citations": {"type": "array", "items": citation_schema}}}
    payload = {"stage": stage, "mode": "intent", "language": "ko",
        "originals": source_blocks([delivered], sources), "citation_catalog": citation_catalog([delivered], sources),
        "items": [{"item_id": "perspectives:0", "original_item": {"statement": "Synthetic only.",
            "citations": [asdict(delivered)]}, "fields": {"statement": "Synthetic only."}}]}
    request = ModelRequest(1, "partition-fixture", "Synthetic contract only.", json.dumps(payload, ensure_ascii=False), 30, 512, schema)
    client = PartitionModelClient()
    response = json.loads(client.complete(request).text)
    output = (response["items"]["perspectives:0"]["statement"]["citations"] if stage == "fact_check"
        else response["replacement"]["citations"])
    assert len(output) == 1
    if protocol == "v4":
        assert set(output[0]) == {"source_id", "quote"}
        assert resolve_quote_citations(output, sources, [delivered]) == [make_span(source, 0, len(source.raw), "citation")]
        if stage == "fact_rewrite":
            assert response["correction_citations"] == output
    else:
        assert set(output[0]) == {"reference_id"}
        assert output[0]["reference_id"] in {span["reference_id"] for source in payload["citation_catalog"] for span in source["ranges"]}
    assert client.quote_ambiguities == []
