"""Literal multi-file truth and request-driven offline model responses for B v2."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace

from review_bot.application.intent_context import build_intent_task_input
from review_bot.application.review_a import ModelProfile
from review_bot.domain.envelope_v2 import ChangedFile
from review_bot.domain.models_v1 import CodeFile, CodeSnapshot, ModelRequest, ModelResponse
from tests.fixtures.review_b import BASE_SHA, HEAD_SHA, digest, make_task as make_v1_task

BASE_SOURCE = 'def render(value):\n    return str(value)\n'
HEAD_SOURCE = 'def render(value):\n    return {"value": value}\n'
CALLER = 'from .value import render\n\ndef label(value):\n    return render(value).strip()\n'
TEST_SOURCE = 'from src.caller import label\n\ndef test_label():\n    assert label(7) == "7"\n'
CONTRACT = "render must provide the string interface consumed by label"
FAILURE = "When label calls render(value).strip(), a dict has no strip attribute."


def make_task(*, body: str | None = None, rename: bool = False, padding: int = 0):
    template = make_v1_task(body=body or "")
    old_path, new_path = "src/value.py", "src/value_v2.py" if rename else "src/value.py"
    suffix = "# fixed padding\n" * padding
    caller_head = 'from .value_v2 import render\n\ndef label(value):\n    return render(value).strip()\n' if rename else CALLER
    base = CodeSnapshot(BASE_SHA, (
        CodeFile(old_path, BASE_SOURCE + suffix, digest(BASE_SOURCE + suffix), "source"),
        CodeFile("src/caller.py", CALLER + suffix, digest(CALLER + suffix), "source"),
        CodeFile("tests/test_label.py", TEST_SOURCE + suffix, digest(TEST_SOURCE + suffix), "test"),
    ))
    head = CodeSnapshot(HEAD_SHA, (
        CodeFile(new_path, HEAD_SOURCE + suffix, digest(HEAD_SOURCE + suffix), "source"),
        CodeFile("src/caller.py", caller_head + suffix, digest(caller_head + suffix), "source"),
        CodeFile("tests/test_label.py", TEST_SOURCE + suffix, digest(TEST_SOURCE + suffix), "test"),
    ))
    changes = (ChangedFile(new_path, previous_path=old_path if rename else None,
        change_type="renamed" if rename else "modified"),)
    if rename:
        changes = (*changes, ChangedFile("src/caller.py"))
    snapshot = replace(template.snapshot, changed_files=changes,
        evidence=template.snapshot.evidence if body is not None else (),
        pr_text_digest=template.snapshot.pr_text_digest if body is not None else digest(""))
    return build_intent_task_input(snapshot, base, head, related_paths=("src/caller.py", "tests/test_label.py"))


def _unique(rows):
    return list({json.dumps(row, sort_keys=True): row for row in rows}.values())


def _cite(entry: dict, line: int) -> dict:
    return {"evidence_id": entry["id"], "start_line": line, "end_line": line}


def _wire_cites(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("evidence_id", "start_line", "end_line")} for row in rows]


def _head_cites(payload: dict) -> list[dict]:
    result = []
    for entry in payload["evidence"]:
        if entry["locator"].get("version") != "head":
            continue
        path = entry["locator"]["path"]
        if path.startswith("src/value"):
            result.append(_cite(entry, 2))
        elif path == "src/caller.py":
            result.append(_cite(entry, 4))
    return _unique(result)


def _relation_update(relation: dict, payload: dict) -> dict:
    paths = {entry["locator"].get("path") for entry in payload["evidence"]}
    both = {relation["from_path"], relation["to_path"]} <= paths
    return {"relation_id": relation["relation_id"], "status": "conflict" if both else "unreviewed",
        "reasoning": "The producer returns a dict while its caller requires strip." if both else "Both original endpoints are needed.",
        "failure_condition": FAILURE if both else "", "citations": _head_cites(payload),
        "missing_context": [] if both else ["Both endpoint files together"]}


def stage_body(request: ModelRequest, *, split: bool = False, defect: bool = True) -> dict:
    """Produce valid wire data from this exact request, without running the target."""
    payload = json.loads(request.user_content)
    stage = payload["stage"]
    paths = payload["reviewed_paths"]
    source_path = next(path for path in paths if path.startswith("src/value") and (not path.endswith("/value.py") or "src/value_v2.py" not in paths))
    if stage == "plan_code_review":
        source_paths = [path for path in paths if path.startswith("src/value")]
        grouped = [source_paths, [path for path in paths if path not in source_paths]] if split else [paths]
        return {"groups": [{"paths": group, "rationale": "Observe related whole files together."} for group in grouped],
            "relations": [{"from_path": source_path, "to_path": "src/caller.py", "contract": CONTRACT, "important": True}], "missing_context": []}
    if stage == "observe_group":
        observations = []
        for entry in payload["evidence"]:
            if entry["locator"].get("version") != "head":
                continue
            path = entry["locator"]["path"]
            line = 2 if path.startswith("src/value") else 4
            statement = "render returns a mapping, rather than its former string." if path.startswith("src/value") else (
                "label calls strip on the returned value." if path == "src/caller.py" else "Test source expects a string; it has not been executed.")
            observations.append({"statement": statement, "citations": [_cite(entry, line)],
                "intent_candidates": ["Possibly change the representation of rendered values."], "missing_context": []})
        relations = payload["relations"]
        updates = [_relation_update(row, payload) for row in relations
            if {row["from_path"], row["to_path"]} <= set(payload["paths"])]
        new = []
        if not relations and source_path in payload["paths"]:
            relation = {"relation_id": "temporary", "from_path": source_path, "to_path": "src/caller.py"}
            state = _relation_update(relation, payload)
            new = [{"from_path": source_path, "to_path": "src/caller.py", "contract": CONTRACT,
                "important": True, **{key: value for key, value in state.items() if key != "relation_id"}}]
        return {"observations": observations, "new_relations": new, "relation_updates": updates,
            "requested_paths": [], "missing_context": []}
    if stage == "recheck_relations":
        return {"relation_updates": [_relation_update(row, payload) for row in payload["relations"]], "missing_context": []}
    observations = [row for group in payload["group_observations"] for row in group["observations"]]
    refs = {"observation_ids": [row["observation_id"] for row in observations],
        "relation_ids": [row["relation_id"] for row in payload["relations"]]}
    if stage == "infer_intent":
        cites = _unique([cite for row in observations for cite in _wire_cites(row["citations"])])
        return {"intents": [{"statement": "The change may be introducing structured rendered values.", **refs,
            "citations": cites, "counterevidence": _wire_cites(observations[-1]["citations"]),
            "alternatives": ["The return-type change could instead be accidental."],
            "uncertainty": "Actual developer intent is unknown; test execution is not supplied.", "missing_context": ["A recorded test run"]}]}
    if stage == "assess":
        refs["intent_ids"] = [row["intent_id"] for row in payload["reconstructed_intents"]]
        conflict = all(row["status"] == "conflict" for row in payload["relations"])
        findings = [{"assessment": "code_defect" if conflict else "insufficient_context", **refs,
            "statement": "The changed return type breaks the caller's string interface." if conflict else "The endpoint contract has not been checked jointly.",
            "reasoning": "The literal producer and consumer show incompatible interfaces.", "failure_condition": FAILURE if conflict else "",
            "citations": _head_cites(payload), "missing_context": [] if conflict else ["Joint endpoint review"]}] if defect else []
        return {"summary": "A return-type/caller contract conflict is visible in the supplied source.",
            "findings": findings, "questions": [{"question": "Is the mapping return type intended?", **refs,
                "citations": _head_cites(payload), "why_needed": "A purpose hypothesis is not a confirmed requirement.",
                "how_answer_changes_assessment": "It changes the suggested repair, not the observed interface conflict."}],
            "relation_updates": [], "review_sufficient": conflict, "no_findings": not findings}
    raise ValueError("unknown_fixture_stage")


def stage_response(request: ModelRequest, *, split: bool = False, defect: bool = True) -> ModelResponse:
    return ModelResponse(json.dumps(stage_body(request, split=split, defect=defect)), {"input_tokens": 100, "output_tokens": 50})


class CodeFirstModelClient:
    """Only the external model seam is mocked; the production core is exercised."""
    def __init__(self, *, split: bool = False, profile: ModelProfile | None = None,
        transform: Callable[[dict, dict], dict | ModelResponse | Exception] | None = None,
        token_counter: Callable[[ModelRequest], int] | None = None,
    ):
        self.profile = profile or ModelProfile("mock", "code-first-fixture", max_output_tokens=512, is_mock=True)
        self.split, self.transform, self.token_counter = split, transform, token_counter
        self.requests: list[ModelRequest] = []
        self.count_requests: list[ModelRequest] = []

    def count_input_tokens(self, request: ModelRequest) -> int:
        self.count_requests.append(request)
        return self.token_counter(request) if self.token_counter else 100

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        payload = json.loads(request.user_content)
        body = stage_body(request, split=self.split)
        result = self.transform(payload, body) if self.transform else body
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, ModelResponse) else ModelResponse(json.dumps(result), {"input_tokens": 100, "output_tokens": 50})