from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy

import pytest

from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
from review_bot.application.agent_review import run_agent_review
from review_bot.application.partition_review import PERSPECTIVE_IDS, prepare_originals, validate_partition_result
from review_bot.application.source_partition import PartitionError
from review_bot.ports import AgentRunError, AgentRunResponse
from tests.fixtures.intent_perspectives import make_task as make_intent_task
from tests.unit.test_partition_review import _pmd_partition_task


class FakeAgentRunner:
    model_id = "stealth/crouton-alpha"
    timeout_seconds = 30

    def __init__(self, tmp_path, output=None, error=None, session=None):
        self.session_store_dir = tmp_path / "agent-sessions"
        self.output = output
        self.error = error
        self.session = session or b'{"type":"assistant","message":{"usage":{"input":10,"output":5,"totalTokens":15}}}\n'
        self.workspace = None
        self.request = None

    def run(self, request):
        self.request = request
        self.workspace = {
            path.relative_to(request.workspace_dir).as_posix()
            for path in request.workspace_dir.rglob("*") if path.is_file()
        }
        if self.error:
            raise self.error
        body = self.output(request) if callable(self.output) else self.output
        if isinstance(body, dict):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, bytes):
            body = body.decode()
        return AgentRunResponse(body, self.session, {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            "model_calls": 1, "cost": {"total": 0.01},
        }, "0.73.1", 0)


def _architecture_body(task):
    originals, meta = prepare_originals(task, "architecture")
    source = next(item for item in originals if item.path == "src/A.java" and item.version == "head")
    return {"summary": "Reviewed architecture.", "findings": [{
        "statement": "A concrete defect.", "reasoning": "The supplied implementation has this behavior.",
        "failure_condition": "When the method is called.", "verdict": "concern",
        "basis": "observed_structure", "target_ids": [meta["targets"][0]],
        "citations": [{"path": source.path, "version": "head", "quote": "void run()"}],
    }]}


def _intent_body(task):
    originals, _meta = prepare_originals(task, "intent")
    source = next(item for item in originals if item.path == "src/value.py" and item.version == "head")
    quote = next(line for line in source.text.splitlines() if line.strip())
    return {"summary": "Compared purpose and implementation.", "questions": [], "perspectives": [
        {"perspective_id": identifier, "assessments": [{
            "verdict": "aligned", "statement": f"Assessment for {identifier}.",
            "reasoning": "Compared the stated intent with supplied source.",
            "failure_condition": "", "citations": [{"path": source.path, "version": "head", "quote": quote}],
        }]} for identifier in PERSPECTIVE_IDS
    ]}


def _run(task, mode, tmp_path, body, *, session=None, runner=None):
    runner = runner or FakeAgentRunner(tmp_path, body, session=session)
    return run_agent_review(task, mode, runner, language="en", instructions="Use the code evidence.")


def test_agent_architecture_result_validates_formats_with_exact_citations_and_trace(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    result = run_agent_review(task, "architecture", runner, language="en", instructions="")

    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert "A concrete defect\\." in markdown
    assert "void run()" in markdown
    assert "Engine: `pi-agent`" in markdown
    assert "0.73.1" in markdown and result.partition["agent"]["skill_sha256"] in markdown
    assert result.partition["engine"] == "pi-agent"
    assert "head/src/A.java" in runner.workspace and "base/src/A.java" in runner.workspace
    assert {"PR.md", "signals.json", "CONTRACT.md"} <= runner.workspace


def test_agent_intent_result_validates_and_formats_all_perspectives(tmp_path):
    task = make_intent_task()
    result = _run(task, "intent", tmp_path, _intent_body(task))

    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert result.status == "completed"
    assert len(result.perspectives) == len(PERSPECTIVE_IDS)
    assert "Assessment for problem\\_success\\." in markdown
    assert "src/value.py" in markdown and "/blob/" in markdown
    assert "Engine: `pi-agent`" in markdown


def test_agent_accepts_one_optional_json_fence(tmp_path):
    task = _pmd_partition_task()
    body = json.dumps(_architecture_body(task))
    result = _run(task, "architecture", tmp_path, f"```json\n{body}\n```")
    assert result.status == "completed"
    validate_partition_result(result, task)


def test_malformed_agent_json_fails_closed(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, "not json")
    assert result.status == "failed" and result.reason == "agent_output_invalid"
    validate_partition_result(result, task)


@pytest.mark.parametrize(("mutation", "reason"), [
    ("absent_quote", "citation_quote_not_found"),
    ("ambiguous_quote", "citation_quote_ambiguous"),
    ("unknown_path", "agent_citation_unknown_path"),
    ("unknown_target", "agent_finding_unknown_target"),
    ("bad_enum", "agent_finding_invalid_verdict"),
    ("no_citations", "agent_item_citations_missing"),
])
def test_invalid_agent_finding_is_rejected_with_reason(tmp_path, mutation, reason):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    finding = body["findings"][0]
    if mutation == "absent_quote":
        finding["citations"][0]["quote"] = "not present"
    elif mutation == "ambiguous_quote":
        finding["citations"][0]["quote"] = "}"
    elif mutation == "unknown_path":
        finding["citations"][0]["path"] = "not-found.java"
    elif mutation == "unknown_target":
        finding["target_ids"] = ["unknown-target"]
    elif mutation == "bad_enum":
        finding["verdict"] = "unexpected"
    elif mutation == "no_citations":
        finding["citations"] = []
    result = _run(task, "architecture", tmp_path, body)
    assert result.status == "partial_review"
    assert result.partition["agent"]["rejected"][0]["reason"] == reason
    validate_partition_result(result, task)


def test_exact_duplicate_agent_items_are_dropped(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    body["findings"].append(deepcopy(body["findings"][0]))
    result = _run(task, "architecture", tmp_path, body)
    assert len(result.findings) == 1
    assert result.partition["agent"]["rejected"] == [{"kind": "finding", "index": 1, "reason": "duplicate"}]
    validate_partition_result(result, task)


def test_distinct_findings_can_share_a_citation(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    second = deepcopy(body["findings"][0])
    second["statement"] = "A separate defect."
    body["findings"].append(second)
    result = _run(task, "architecture", tmp_path, body)
    assert result.status == "completed" and len(result.findings) == 2
    assert len(next(iter(result.partition["final"]["target_reviews"].values()))["citations"]) == 1
    validate_partition_result(result, task)


def test_timeout_agent_error_becomes_failed_result(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, error=AgentRunError("timeout"))
    result = _run(task, "architecture", tmp_path, None, runner=runner)
    assert result.status == "failed" and result.reason == "agent_run_failed:timeout"
    validate_partition_result(result, task)


def test_tampered_stored_agent_citation_is_rejected(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    altered = deepcopy(result)
    altered.partition["final"]["findings"][0]["citations"][0]["quote"] += " tampered"
    with pytest.raises(PartitionError, match="agent_citation_tampered"):
        validate_partition_result(altered, task)


def test_agent_session_is_redacted_hashed_and_gzipped(tmp_path):
    task = _pmd_partition_task()
    session = (b'{"text":"OPENROUTER_API_KEY=secret-value sk-or-v1-supersecret '
        b'Bearer abc.def.ghi"}\n')
    result = _run(task, "architecture", tmp_path, _architecture_body(task), session=session)
    metadata = result.partition["agent"]["session"]
    path = tmp_path / "agent-sessions" / f'{metadata["sha256"]}.jsonl.gz'
    stored = gzip.decompress(path.read_bytes())
    assert hashlib.sha256(stored).hexdigest() == metadata["sha256"]
    assert metadata["bytes"] == len(stored)
    assert b"secret-value" not in stored
    assert b"sk-or-v1-supersecret" not in stored
    assert b"Bearer abc.def.ghi" not in stored
    assert b"[REDACTED]" in stored
