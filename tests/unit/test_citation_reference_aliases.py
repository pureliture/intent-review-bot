from review_bot.application.context import hash_json
from review_bot.application.citation_reference_aliases import (
    alias_model_references,
    restore_model_references,
)


def test_reference_aliases_are_short_round_trip_and_leave_source_text_unchanged():
    original = "Qfac8df8d93a4"
    payload = {
        "citation_catalog": [{"ranges": [{"reference_id": original, "text": f"code {original}"}]}],
        "allowed_reference_ids": [original],
        "prior_response": {"citations": [{"reference_id": original}],
            "validation_error": f"allowed_HEAD_reference_ids=[{original}]"},
    }
    payload["prior_response_hash"] = hash_json(payload["prior_response"])
    schema = {"properties": {"citations": {"items": {"properties": {
        "reference_id": {"type": "string", "enum": [original]}}}}},
        "description": f"HEAD reference: {original}"}

    wire_payload, wire_schema, aliases = alias_model_references(payload, schema)
    alias = wire_payload["citation_catalog"][0]["ranges"][0]["reference_id"]

    assert alias == "R1"
    assert wire_payload["allowed_reference_ids"] == [alias]
    assert wire_payload["prior_response"]["citations"][0]["reference_id"] == alias
    assert wire_payload["prior_response"]["validation_error"] == f"allowed_HEAD_reference_ids=[{alias}]"
    assert wire_payload["prior_response_hash"] == hash_json(wire_payload["prior_response"])
    assert wire_payload["citation_catalog"][0]["ranges"][0]["text"] == f"code {original}"
    assert wire_schema["properties"]["citations"]["items"]["properties"]["reference_id"]["enum"] == [alias]
    assert original not in wire_schema["description"]
    assert restore_model_references({"citations": [{"reference_id": alias}]}, aliases) == {
        "citations": [{"reference_id": original}]}
    restored = restore_model_references(wire_payload, aliases)
    assert restored["prior_response_hash"] == hash_json(restored["prior_response"])


def test_reference_aliases_are_deterministic_and_unknown_output_is_preserved():
    identifiers = ["Qbbb222222222", "Qaaa111111111"]
    invalid_prior = "Qmalformed123"
    payload = {"ranges": [{"reference_id": value} for value in reversed(identifiers)],
        "previous": {"reference_id": invalid_prior}}
    schema = {"properties": {"reference_id": {"enum": identifiers}}}

    wire_payload, wire_schema, aliases = alias_model_references(payload, schema)

    assert [row["reference_id"] for row in wire_payload["ranges"]] == ["R1", "R2"]
    assert wire_payload["previous"]["reference_id"] == "X1"
    assert wire_schema["properties"]["reference_id"]["enum"] == ["R2", "R1"]
    assert aliases == {"R1": "Qaaa111111111", "R2": "Qbbb222222222", "X1": invalid_prior}
    assert restore_model_references({"reference_id": "R99"}, aliases) == {"reference_id": "R99"}
