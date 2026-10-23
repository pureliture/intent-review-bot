"""Signed webhook jobs use the perspective core and retain their private B artifacts."""
import json

import pytest
import yaml

from review_bot.config import load_review_config_from_str
from tests.fixtures.partition_review import PartitionModelClient
from tests.integration.test_intent_flow import intent_flow, synthetic_repo, send_event, output


def use_perspectives(flow):
    raw = flow.config.model_dump()
    raw["intent"]["pipeline"] = "perspectives-v1"
    flow.config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    flow.worker.config = flow.config


def test_signed_webhook_runs_perspectives_without_publishing(intent_flow):
    f = intent_flow
    use_perspectives(f)
    model = PartitionModelClient()
    f.worker.model_client = model
    assert send_event(f).status_code == 202
    assert f.worker.run_once() is True
    directory, status = output(f)
    result = json.loads((directory / "B/result.json").read_text())
    receipt = json.loads((directory / "B/receipt.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert status == "succeeded" and result["status"] == "completed"
    assert result["schema_version"] == manifest["modes"]["intent"]["result_schema_version"] == 13
    assert receipt["b_result"] == result and receipt["input_fingerprint"]
    assert receipt["publication"]["reason"] == "artifact_only"
    assert len(model.requests) == result["usage"]["calls"]
    choices = [json.loads(request.user_content) for request in f.planning_model.requests]
    assert len([row for row in choices if row["stage"] == "select_perspective"]) == 8
    assert len(result["perspectives"]) == 8
    for filename in ("partition.json", "perspectives.json", "fact-verification.json", "validation.json"):
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
