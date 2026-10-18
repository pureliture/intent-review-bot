"""Signed webhook jobs use the perspective core and retain their private B artifacts."""
import json

import pytest
import yaml

from review_bot.config import load_review_config_from_str
from tests.fixtures.intent_perspectives import PerspectiveModelClient
from tests.integration.test_intent_flow import intent_flow, synthetic_repo, send_event, output


def use_perspectives(flow):
    raw = flow.config.model_dump()
    raw["intent"]["pipeline"] = "perspectives-v1"
    flow.config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    flow.worker.config = flow.config


def aligned_source_response(payload, body):
    refs = [{"evidence_id": x["id"], "start_line": 1, "end_line": 1} for x in payload["evidence"]
        if x["locator"].get("path") == "app.py" and x["locator"].get("version") == "head"]
    assert refs
    for row in body["narrative"].values():
        row.update(statement="고정된 원문에서 요청 직렬화를 위한 잠금 추가를 확인했습니다.", citations=refs)
    for row in body["perspective_decisions"]:
        row["citations"] = refs
    for row in body["judgements"]:
        row.update(verdict="aligned", statement="주어진 원문에서 이 관점의 불일치를 확인하지 못했습니다.",
            failure_condition="", citations=refs)
    return body


def test_signed_webhook_runs_perspectives_without_publishing(intent_flow):
    f = intent_flow
    use_perspectives(f)
    model = PerspectiveModelClient(transform=aligned_source_response)
    selector = PerspectiveModelClient(selector=True)
    f.worker.model_client = model
    f.worker.intent_selection_client = selector
    assert send_event(f).status_code == 202
    assert f.worker.run_once() is True
    directory, status = output(f)
    result = json.loads((directory / "B/result.json").read_text())
    receipt = json.loads((directory / "B/receipt.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert status == "succeeded" and result["status"] == "completed"
    assert result["schema_version"] == manifest["modes"]["intent"]["result_schema_version"] == 3
    assert receipt["b_result"] == result and receipt["input_fingerprint"]
    assert receipt["publication"]["reason"] == "artifact_only"
    assert len(model.requests) == 2 and len(selector.requests) == 8
    for filename in ("catalog.json", "selection.json", "perspectives.json", "judgements.json", "context.json", "passes.json"):
        assert (directory / "B" / filename).is_file()
    f.analyzer.run_analysis.assert_not_called()
    assert not f.comments


@pytest.mark.parametrize("publish", [{"mode": "comment", "b_comments_enabled": False},
    {"mode": "artifact_only", "b_comments_enabled": True}])
def test_perspective_config_rejects_external_publication(intent_flow, publish):
    raw = intent_flow.config.model_dump()
    raw["intent"]["pipeline"] = "perspectives-v1"
    raw["publish"].update(publish)
    with pytest.raises(ValueError, match="artifact_only"):
        load_review_config_from_str(yaml.safe_dump(raw))
