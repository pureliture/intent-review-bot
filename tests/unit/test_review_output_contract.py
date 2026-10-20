from __future__ import annotations

import json

from review_bot.application.partition_review import _note_schema, _final_schema
from review_bot.application.review_output_contract import render_output_contract


def test_model_readable_contract_tracks_schema_fields_without_copying_reference_enums():
    schema = _note_schema(("Q-private-choice-1", "Q-private-choice-2"), ("problem_success",))
    rendered = render_output_contract(schema)
    contract = json.loads(rendered.splitlines()[-1])
    assert set(contract["fields"]) == set(schema["properties"])
    assert "generic heading" in contract["fields"]["statement"]["meaning"]
    assert "Q-private-choice" not in rendered
    assert "citation_catalog" in rendered


def test_final_contract_exposes_judgement_values_and_exact_map_ownership():
    schema = _final_schema(("Q-secret-reference",), "intent", ("src/A.py",), ("M-private",))
    rendered = render_output_contract(schema)
    contract = json.loads(rendered.splitlines()[-1])
    assert set(contract["fields"]) == set(schema["properties"])
    verdict = contract["fields"]["perspectives"]["items"]["fields"]["verdict"]
    assert verdict["values"] == schema["properties"]["perspectives"]["items"]["properties"]["verdict"]["enum"]
    assert contract["fields"]["target_reviews"]["keys"] == "exact keys from review_targets"
    assert contract["fields"]["context_updates"]["keys"] == "optional known IDs from context_items"
    assert "Q-secret-reference" not in rendered and "M-private" not in rendered
