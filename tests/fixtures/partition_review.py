"""Synthetic model responses only; actual application/worker logic is exercised."""
from __future__ import annotations

import json

from review_bot.application.architecture_review import ModelProfile
from review_bot.domain.intent_perspectives import PERSPECTIVES
from review_bot.domain.models_v1 import ModelResponse


class PartitionModelClient:
    def __init__(self, *, name="partition-fixture", counter=None, transform=None, group_all=False):
        self.profile = ModelProfile("mock", name, max_input_tokens=50000, max_output_tokens=512, is_mock=True)
        self.counter, self.transform, self.group_all = counter, transform, group_all
        self.requests, self.count_requests = [], []

    def count_input_tokens(self, request):
        self.count_requests.append(request)
        return self.counter(request) if self.counter else 100

    def complete(self, request):
        self.requests.append(request)
        payload = json.loads(request.user_content)
        stage = payload["stage"]
        if set(request.response_schema.get("properties", {})) == {"choice"}:
            options = payload["options"]
            option = options[-1] if stage == "plan_partitions" and not self.group_all else options[0]
            body = {"choice": option["label"]}
        else:
            spans = [*payload.get("originals", payload.get("original_quotes", [])), *payload.get("pr_purpose", [])]
            spans = [{**s, "text": "".join(b["text"] for b in s["blocks"])} if "blocks" in s else s for s in spans]
            spans = sorted((s for s in spans if s["text"]), key=lambda s: s.get("version") != "head")
            source = spans[0]
            def reference_for(citation):
                for catalog in payload["citation_catalog"]:
                    if catalog["evidence_id"] != citation["evidence_id"]:
                        continue
                    for row in catalog["ranges"]:
                        if citation["start_byte"] <= row["start_byte"] < row["end_byte"] <= citation["end_byte"]:
                            return {"reference_id": row["reference_id"]}
                raise AssertionError("No supplied citation range for fixture")
            cite = reference_for(source)
            if stage == "select_review_scope":
                body = {"decision": "apply", "reasoning": "Synthetic applicability fixture.", "citations": [cite],
                    "perspectives": [{"perspective_id": p.id, "decision": "apply"} for p in PERSPECTIVES]
                    if payload["mode"] == "intent" or "intent" in payload.get("available_modes", []) else []}
                if payload["mode"] == "planning":
                    body["selected_modes"] = payload["available_modes"]
            elif stage == "final_review":
                body = {"summary": "Synthetic contract result, not a real review.", "findings": [], "questions": [],
                    "perspectives": [{"perspective_id": p.id, "verdict": "aligned", "statement": "Synthetic judgement.",
                        "reasoning": "Fixture cites an actual delivered original.", "failure_condition": "", "citations": [cite]}
                        for p in PERSPECTIVES] if payload["mode"] == "intent" else [], "missing_context": []}
            else:
                body = {"statement": "Synthetic observation.", "citations": [cite], "missing_context": []}
                if stage == "review_partition":
                    body["draft_assessments"] = {key: {"verdict": "aligned", "statement": "Synthetic provisional judgement.",
                        "reasoning": "Fixture examines fixed source before final review.", "failure_condition": "", "citations": [cite]}
                        for key in payload["draft_assessment_keys"]}
            if stage == "final_review":
                body["source_requests"] = []
                if "correction_reason" in request.response_schema["properties"]:
                    body["correction_reason"] = "Rechecked the fixed originals and corrected the invalid earlier grounding."
                body["context_updates"] = {key: {"state": "resolved", "reason": "Fixed original evidence resolves this provisional gap.",
                    "citations": [cite]} for key in payload.get("context_items", {})}
                body["target_reviews"] = {key: {"state": "reviewed", "reason": "Synthetic explicit review of this target.",
                    "citations": [reference_for(next((span for span in spans if span.get("path") in
                        next((target["paths"] for target in payload.get("review_target_details", []) if target["target_id"] == key), [key])), source))]}
                    for key in payload.get("review_targets", [])}
            if stage in ("reduce_observations", "select_review_scope", "final_review"):
                body["observation_dispositions"] = {row["observation_id"]: {"disposition": "merged", "reason": "Synthetic accounting for this input observation.",
                    "citations": [{"reference_id": request.response_schema["properties"]["observation_dispositions"]["properties"][row["observation_id"]]["properties"]["citations"]["items"]["properties"]["reference_id"]["enum"][0]}]}
                    for row in payload["observations"]}
        if self.transform:
            body = self.transform(payload, body)
        return body if isinstance(body, ModelResponse) else ModelResponse(json.dumps(body, ensure_ascii=False),
            {"input_tokens": self.count_input_tokens(request), "output_tokens": 10})
