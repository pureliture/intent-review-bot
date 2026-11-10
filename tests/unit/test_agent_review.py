from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
from review_bot.application import agent_review
from review_bot.application.agent_review import (
    AgentOutputError,
    _SEARCH_SCOPE_RULE,
    _contract,
    _initial_agent_prompt,
    _verifier_claims,
    _validate_verifier_output,
    run_agent_review,
)
from review_bot.application.context import hash_json
from review_bot.application.partition_review import PERSPECTIVE_IDS, prepare_originals, validate_partition_result
from review_bot.application.source_partition import PartitionError, SourceOriginal, make_span
from review_bot.ports import AgentRunError, AgentRunFinalization, AgentRunResponse
from tests.fixtures.intent_perspectives import make_task as make_intent_task
from tests.unit.test_partition_review import _pmd_partition_task


class FakeAgentRunner:
    model_id = "stealth/crouton-alpha"
    timeout_seconds = 30

    def __init__(self, tmp_path, output=None, error=None, session=None, repair_output=None, repair_error=None):
        self.session_store_dir = tmp_path / "agent-sessions"
        self.output = output
        self.error = error
        self.repair_output = repair_output
        self.repair_error = repair_error
        self.verify_outputs = []
        self.verify_repair_outputs = []
        self.verify_errors = []
        self.revision_output = None
        self.session = session or b'{"type":"assistant","message":{"usage":{"input":10,"output":5,"totalTokens":15}}}\n'
        self.config = SimpleNamespace(max_proxy_calls=12, max_output_tokens=2048, max_revisions=1)
        self.workspace = None
        self.generation_workspace = None
        self.generation_contract = ""
        self.generation_contract_json = {}
        self.contract = ""
        self.contract_json = {}
        self.generation_workspace_dir = None
        self.verifier_workspace_dir = None
        self.generation_contract_hashes_before_verify = None
        self.generation_contract_hashes_during_verify = None
        self.revision_rows_seen = None
        self.output_path = tmp_path / "fake-out" / "result.json"
        self.omit_output = False
        self.repair_omit_output = None
        self.oversize_output = False
        self.repair_oversize_output = None
        self.request = None
        self.run_modes = []
        self.repair_calls = 0
        self.repair_message = None
        self.repair_messages = []
        self.verify_claim_ids = []
        self.finished = []
        self.proxy_usage = {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13,
            "model_calls": 2, "usage_complete": True, "unknown_usage_attempts": 0,
            "max_prompt_tokens": 5, "compactions": 1, "cost": {"total": 0.025}, "calls": [
                {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "cost": {"total": 0.01}, "status": 200},
                {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": {"total": 0.015}, "status": 200},
            ]}

    def run(self, request):
        self.run_modes.append(request.mode)
        self.request = request
        self.workspace = {
            path.relative_to(request.workspace_dir).as_posix()
            for path in request.workspace_dir.rglob("*") if path.is_file()
        }
        self.contract = (request.workspace_dir / "CONTRACT.md").read_text(encoding="utf-8")
        self.contract_json = json.loads((request.workspace_dir / "contract.json").read_text(encoding="utf-8"))
        if request.mode != "verify":
            self.generation_workspace = set(self.workspace)
            self.generation_contract = self.contract
            self.generation_contract_json = deepcopy(self.contract_json)
            self.generation_workspace_dir = request.workspace_dir
            self.generation_contract_hashes_before_verify = {
                name: hashlib.sha256((request.workspace_dir / name).read_bytes()).hexdigest()
                for name in ("contract.json", "CONTRACT.md")
            }
        else:
            self.verifier_workspace_dir = request.workspace_dir
            self.generation_contract_hashes_during_verify = {
                name: hashlib.sha256((self.generation_workspace_dir / name).read_bytes()).hexdigest()
                for name in ("contract.json", "CONTRACT.md")
            }
        if self.error:
            raise self.error
        if request.mode == "verify":
            if self.verify_errors:
                error = self.verify_errors.pop(0)
                if error:
                    raise error
            claims = json.loads((request.workspace_dir / "claims.json").read_text(encoding="utf-8"))["claims"]
            self.verify_claim_ids.append([claim["id"] for claim in claims])
            body = self.verify_outputs.pop(0) if self.verify_outputs else {"items": {claim["id"]: {
                field: {"state": "supported" if str(text).strip() else "non_factual",
                    "reason": "Checked against the frozen originals.",
                    "citations": claim["cited"] if str(text).strip() else []}
                for field, text in claim["fields"].items()}
                for claim in claims}}
            if callable(body):
                body = body(request, claims)
        else:
            body = self.output(request) if callable(self.output) else self.output
        return self._response(body, self.omit_output, self.oversize_output, "fake-session", {
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
            "model_calls": 1, "cost": {"total": 0.01},
        })

    def _response(self, body, omit_output, oversize_output, session_id, usage):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if omit_output:
            self.output_path.unlink(missing_ok=True)
        elif oversize_output:
            self.output_path.write_bytes(b"x" * 1_048_577)
        else:
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.output_path.write_bytes(body or b"")
        usage = {**usage, "max_prompt_tokens": 5, "compactions": 1}
        return AgentRunResponse("DONE", self.session, usage, "0.73.1", 0,
            session_id, self.output_path)

    def repair(self, request, session_id, message):
        self.repair_calls += 1
        self.repair_message = message
        self.repair_messages.append(message)
        if self.repair_error:
            raise self.repair_error
        if request.mode == "verify":
            claims = json.loads((request.workspace_dir / "claims.json").read_text(encoding="utf-8"))["claims"]
            body = {"items": {claim["id"]: {
                field: {"state": "supported" if str(text).strip() else "non_factual",
                    "reason": "Checked against the frozen originals.",
                    "citations": claim["cited"] if str(text).strip() else []}
                for field, text in claim["fields"].items()}
                for claim in claims}}
            if self.verify_repair_outputs:
                body = self.verify_repair_outputs.pop(0)
            if callable(body):
                body = body(request, claims)
        else:
            body = self.revision_output if "/out/revision.json" in message and self.revision_output is not None else (
                self.repair_output if self.repair_output is not None else self.output)
            body = body(request) if callable(body) else body
            if "/out/revision.json" in message:
                self.revision_rows_seen = json.loads(
                    (request.workspace_dir / "revision_rows.json").read_text(encoding="utf-8"))
                self.output_path = request.workspace_dir / "revision.json"
        return self._response(body,
            self.omit_output if self.repair_omit_output is None else self.repair_omit_output,
            self.oversize_output if self.repair_oversize_output is None else self.repair_oversize_output,
            session_id, {})

    def finish(self, session_id):
        self.finished.append(session_id)
        return AgentRunFinalization(self.session, self.proxy_usage,
            {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "model_calls": 1,
                "cost": {"total": 0.01}},
            str(self.session_store_dir / "proxy-ledgers" / "usage.jsonl"), 1234)


def _architecture_body(task):
    originals, meta = prepare_originals(task, "architecture")
    source = next(item for item in originals if item.path == "src/A.java" and item.version == "head")
    citation = {"path": source.path, "version": "head", "quote": "void run()"}
    return {"summary": "Reviewed architecture.", "summary_citations": [deepcopy(citation)], "findings": [{
        "statement": "A concrete defect.", "reasoning": "The supplied implementation has this behavior.",
        "failure_condition": "When the method is called.", "verdict": "concern",
        "basis": "observed_structure", "target_ids": [meta["targets"][0]],
        "citations": [deepcopy(citation)],
    }]}


def _intent_body(task):
    originals, _meta = prepare_originals(task, "intent")
    source = next(item for item in originals if item.path == "src/value.py" and item.version == "head")
    quote = next(line for line in source.text.splitlines() if line.strip())
    citation = {"path": source.path, "version": "head", "quote": quote}
    return {"summary": "Compared purpose and implementation.", "summary_citations": [deepcopy(citation)],
        "questions": [], "perspectives": [
        {"perspective_id": identifier, "assessments": [{
            "verdict": "aligned", "statement": f"Assessment for {identifier}.",
            "reasoning": "Compared the stated intent with supplied source.",
            "failure_condition": "", "citations": [deepcopy(citation)],
        }]} for identifier in PERSPECTIVE_IDS
    ]}


def _run(task, mode, tmp_path, body, *, session=None, runner=None):
    runner = runner or FakeAgentRunner(tmp_path, body, session=session)
    return run_agent_review(task, mode, runner, language="en", instructions="Use the code evidence.")


@pytest.mark.parametrize(("mode", "skill"), [
    ("architecture", "review-architecture"),
    ("intent", "review-intent"),
    ("verify", "verify-claims"),
])
def test_search_scope_rule_is_in_every_contract(mode, skill):
    assert _SEARCH_SCOPE_RULE in _contract(mode, {"targets": []}, skill, "")


@pytest.mark.parametrize("mode", ["architecture", "intent", "verify"])
def test_search_scope_rule_is_in_initial_prompts(mode):
    assert _SEARCH_SCOPE_RULE in _initial_agent_prompt(mode)


@pytest.mark.parametrize("mode", ["architecture", "intent"])
def test_agent_verifier_claims_exclude_failure_conditions(mode):
    task = _pmd_partition_task() if mode == "architecture" else make_intent_task()
    final = _architecture_body(task) if mode == "architecture" else _intent_body(task)
    final.setdefault("findings", [])
    final.setdefault("perspectives", [])
    final.setdefault("questions", [])

    claims = _verifier_claims(final)
    summary = next(claim for claim in claims if claim["id"] == "summary")
    rows = [claim for claim in claims if claim["id"] != "summary"]

    assert summary["fields"] == {"text": final["summary"]}
    assert rows
    assert all(set(claim["fields"]) == {"statement", "reasoning"} for claim in rows)


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
    assert "timeout 30s" in markdown and "max proxy calls 12" in markdown and "max output tokens 2048" in markdown
    assert "| agent_review | 0 | 2 | 2 | 0 |" in markdown
    assert "Agent cost:" in markdown and "0.025" in markdown
    assert result.partition["engine"] == "pi-agent"
    assert "head/src/A.java" in runner.generation_workspace and "base/src/A.java" in runner.generation_workspace
    assert {"PR.md", "signals.json", "CONTRACT.md"} <= runner.generation_workspace
    assert "contract.json" in runner.generation_workspace
    assert "target_ids may be empty only when basis is observed_structure" in runner.generation_contract
    assert _SEARCH_SCOPE_RULE in runner.generation_contract
    assert "Every finding and assessment needs at least one citation" in runner.generation_contract
    assert "empty intent assessments list means no distinct finding" in runner.generation_contract
    assert runner.generation_contract_json["mode"] == "architecture"
    assert runner.generation_contract_json["allowed_target_ids"] == prepare_originals(task, "architecture")[1]["targets"]
    assert runner.generation_contract_json["required_keys"]["finding"][-1] == "citations"
    assert runner.generation_contract_json["citation_rules"]["empty_target_ids_basis"] == "observed_structure"


def test_agent_result_with_unset_output_cap_records_null_and_hides_it_in_trace(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.config.max_output_tokens = None
    result = run_agent_review(task, "architecture", runner, language="en", instructions="")

    agent = result.partition["agent"]
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")

    assert agent["limits"]["max_output_tokens"] is None
    assert result.model["profile"]["max_output_tokens"] is None
    assert result.partition["identity"]["models"]["agent"]["max_output_tokens"] is None
    assert result.partition["identity"]["models"]["verifier"]["max_output_tokens"] is None
    assert "Agent limits:" in markdown
    assert "max output tokens" not in markdown
    validate_partition_result(result, task)


def test_agent_legacy_output_cap_4096_replays(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.config.max_output_tokens = 4096
    result = run_agent_review(task, "architecture", runner, language="en", instructions="")

    assert result.partition["agent"]["limits"]["max_output_tokens"] == 4096
    assert result.partition["identity"]["models"]["agent"]["max_output_tokens"] == 4096
    assert result.model["profile"]["max_output_tokens"] == 4096
    validate_partition_result(result, task)


def test_agent_intent_result_validates_and_formats_all_perspectives(tmp_path):
    task = make_intent_task()
    runner = FakeAgentRunner(tmp_path, _intent_body(task))
    result = run_agent_review(task, "intent", runner, language="en", instructions="")

    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert result.status == "completed"
    assert len(result.perspectives) == len(PERSPECTIVE_IDS)
    assert "Assessment for problem\\_success\\." in markdown
    assert "src/value.py" in markdown and "/blob/" in markdown
    assert "Engine: `pi-agent`" in markdown
    assert _SEARCH_SCOPE_RULE in runner.generation_contract


def test_intent_empty_assessment_perspective_is_completed_and_omitted_from_comment(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    body["perspectives"][0]["assessments"] = []
    result = _run(task, "intent", tmp_path, body)

    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    private_markdown = format_partition_review_markdown(result, task)
    assert result.status == "completed"
    assert result.perspectives[0]["assessments"] == []
    assert "Assessment for problem_success." not in markdown
    assert "Assessment for " in markdown
    assert result.perspectives[0]["perspective_id"] not in private_markdown


def test_intent_all_empty_assessments_are_partial(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    for perspective in body["perspectives"]:
        perspective["assessments"] = []
    result = _run(task, "intent", tmp_path, body)

    validate_partition_result(result, task)
    assert result.status == "partial_review"
    assert result.reason == "agent_perspectives_incomplete"
    assert len(result.perspectives) == len(PERSPECTIVE_IDS)
    assert all(not item["assessments"] for item in result.perspectives)


def test_agent_fact_verification_supported_items_are_retained(tmp_path):
    task = _pmd_partition_task()
    result = run_agent_review(task, "architecture", FakeAgentRunner(tmp_path, _architecture_body(task)),
        language="en", instructions="")
    validate_partition_result(result, task)
    assert result.status == "completed"
    assert result.partition["fact_verification"]["status"] == "model_supported"
    assert all(row["status"] == "model_supported" for row in result.partition["fact_verification"]["items"])
    assert all(run["duration_ms"] == 0 for run in result.partition["fact_verification"]["runs"]
        if run["role"] == "verify")


@pytest.mark.parametrize(("language", "disclosure"), [
    ("ko", "- 성립 조건은 가정을 설명하는 필드라 사실 검토 대상에서 제외했습니다."),
    ("en", "- Failure conditions describe hypothetical scenarios and are excluded from factual review."),
])
def test_agent_v2_verifies_findings_without_failure_condition_and_discloses_it(
        tmp_path, language, disclosure):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    result = run_agent_review(task, "architecture", FakeAgentRunner(tmp_path, body),
        language=language, instructions="")

    ledger = result.partition["fact_verification"]
    entry = next(row for row in ledger["items"] if row["item_id"] == "findings:0")
    verdicts = entry["attempts"][0]["verdicts"]
    markdown = format_partition_review_markdown(result, task, publication_mode="artifact_only")

    assert result.status == "completed"
    assert ledger["version"] == "agent-verify-v2"
    assert entry["status"] == "model_supported"
    assert set(verdicts) == {"statement", "reasoning"}
    assert all(verdict["state"] == "supported" for verdict in verdicts.values())
    assert "When the method is called" in markdown
    assert disclosure in markdown
    validate_partition_result(result, task)


def test_agent_v2_rejects_failure_condition_verdict(tmp_path):
    task = _pmd_partition_task()
    originals, _meta = prepare_originals(task, "architecture")
    spans = [make_span(original, 0, len(original.raw), "agent_verify") for original in originals]
    final = _architecture_body(task)
    final.update(perspectives=[], questions=[])
    claims = _verifier_claims(final)
    body = {"items": {claim["id"]: {
        field: {"state": "supported", "reason": "The frozen source supports this field.",
            "citations": claim["cited"]}
        for field in claim["fields"]} for claim in claims}}
    body["items"]["findings:0"]["failure_condition"] = {
        "state": "supported", "reason": "It is supported.", "citations": []}

    with pytest.raises(AgentOutputError, match="verifier_output_invalid"):
        _validate_verifier_output(body, claims, originals, spans)


def test_agent_fact_verification_rewrites_contradicted_assessment(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.verify_outputs = [
        lambda req, claims: {"items": {claim["id"]: {
            field: {"state": "contradicted" if claim["id"] == "findings:0" else "supported",
                "reason": "The source contradicts this field." if claim["id"] == "findings:0"
                else "The frozen source supports this field.",
                "citations": claim["cited"]} for field in claim["fields"]} for claim in claims}},
    ]
    runner.revision_output = {"items": {"findings:0": {"action": "revise",
        "item": _architecture_body(task)["findings"][0]}}}
    result = run_agent_review(task, "architecture", runner, language="en", instructions="")
    validate_partition_result(result, task)
    entry = next(row for row in result.partition["fact_verification"]["items"] if row["item_id"] == "findings:0")
    assert len(entry["attempts"]) == 2
    assert result.status == "completed"
    assert runner.verify_claim_ids[-1] == ["findings:0"]
    assert runner.revision_rows_seen == ["findings:0"]
    assert '"findings:0"' in runner.repair_message
    assert '"<row_id>"' in runner.repair_message
    assert '"findings": [' in runner.repair_message
    assert '"summary_citations"' in runner.repair_message
    assert '"statement"' in runner.repair_message
    assert "node /skills/check_output.mjs --revision" in runner.repair_message
    revision_usage = next(run["usage"] for run in result.partition["fact_verification"]["runs"]
        if run["role"] == "revise")
    assert revision_usage["max_prompt_tokens"] == 5 and revision_usage["compactions"] == 1


def test_agent_verifier_uses_isolated_workspace_without_changing_generation_contract(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))

    result = run_agent_review(task, "architecture", runner, language="en", instructions="")

    assert result.status == "completed"
    assert runner.verifier_workspace_dir != runner.generation_workspace_dir
    assert runner.verifier_workspace_dir.parent == runner.generation_workspace_dir.parent
    assert runner.generation_contract_hashes_during_verify == runner.generation_contract_hashes_before_verify
    assert _SEARCH_SCOPE_RULE in runner.contract
    assert runner.contract_json["mode"] == "verify"
    assert not runner.verifier_workspace_dir.exists()
    assert {"head/src/A.java", "base/src/A.java", "PR.md"} <= runner.workspace


def test_agent_fact_verification_supports_multiple_revision_rounds(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    contradicted = lambda _request, claims: {"items": {claim["id"]: {
        field: {"state": "contradicted" if claim["id"] == "findings:0" else "supported",
            "reason": "Recheck this finding." if claim["id"] == "findings:0"
            else "The frozen source supports this field.", "citations": claim["cited"]}
        for field in claim["fields"]} for claim in claims}}
    runner.verify_outputs = [contradicted, contradicted]
    runner.revision_output = {"items": {"findings:0": {"action": "revise",
        "item": _architecture_body(task)["findings"][0]}}}

    result = run_agent_review(task, "architecture", runner, language="en", instructions="", max_revisions=2)
    entry = next(row for row in result.partition["fact_verification"]["items"]
        if row["item_id"] == "findings:0")

    assert result.status == "completed"
    assert len(entry["attempts"]) == 3
    assert len(entry["revision"]) == 2
    assert runner.verify_claim_ids == [
        ["summary", "findings:0"], ["findings:0"], ["findings:0"]]
    validate_partition_result(result, task)


def test_agent_fact_verification_withdraws_finding(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.verify_outputs = [lambda _request, claims: {"items": {claim["id"]: {
        field: {"state": "contradicted" if claim["id"] == "findings:0" else "supported",
            "reason": "The source contradicts this field." if claim["id"] == "findings:0"
            else "The frozen source supports this field.", "citations": claim["cited"]}
        for field in claim["fields"]} for claim in claims}}]
    runner.revision_output = {"items": {"findings:0": {"action": "withdraw"}}}

    result = run_agent_review(task, "architecture", runner, language="en", instructions="")
    entry = next(row for row in result.partition["fact_verification"]["items"]
        if row["item_id"] == "findings:0")

    assert result.status == "completed"
    assert result.findings == ()
    assert entry["status"] == "withdrawn"
    validate_partition_result(result, task)


def test_agent_revision_withdraws_and_revises_architecture_rows_by_original_id(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    second = deepcopy(body["findings"][0])
    second["statement"] = "A second concrete defect."
    body["findings"].append(second)
    runner = FakeAgentRunner(tmp_path, body)

    def initial_verdicts(_request, claims):
        return {"items": {claim["id"]: {
            field: {"state": "contradicted" if claim["id"] in ("findings:0", "findings:1")
                and str(text).strip() else ("supported" if str(text).strip() else "non_factual"),
                "reason": "Recheck this finding.", "citations": claim["cited"] if str(text).strip() else []}
            for field, text in claim["fields"].items()} for claim in claims}}

    runner.verify_outputs = [initial_verdicts]
    revised = deepcopy(second)
    revised["statement"] = "The revised original second finding."
    runner.revision_output = {"items": {
        "findings:0": {"action": "withdraw"},
        "findings:1": {"action": "revise", "item": revised},
    }}

    result = run_agent_review(task, "architecture", runner, language="en", instructions="")
    entries = {entry["item_id"]: entry for entry in result.partition["fact_verification"]["items"]}

    assert result.status == "completed"
    assert [finding["statement"] for finding in result.findings] == ["The revised original second finding."]
    assert entries["findings:0"]["status"] == "withdrawn"
    assert entries["findings:1"]["accepted"]["statement"] == "The revised original second finding."
    assert runner.verify_claim_ids[-1] == ["findings:1"]
    validate_partition_result(result, task)


def test_agent_revision_withdraws_and_revises_intent_assessments_by_original_id(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    second = deepcopy(body["perspectives"][0]["assessments"][0])
    second["statement"] = "A second assessment for this perspective."
    body["perspectives"][0]["assessments"].append(second)
    runner = FakeAgentRunner(tmp_path, body)

    def initial_verdicts(_request, claims):
        return {"items": {claim["id"]: {
            field: {"state": "contradicted" if claim["id"] in ("perspectives:0:0", "perspectives:0:1")
                and str(text).strip() else ("supported" if str(text).strip() else "non_factual"),
                "reason": "Recheck this assessment.", "citations": claim["cited"] if str(text).strip() else []}
            for field, text in claim["fields"].items()} for claim in claims}}

    runner.verify_outputs = [initial_verdicts]
    revised = deepcopy(second)
    revised["statement"] = "The revised original second assessment."
    runner.revision_output = {"items": {
        "perspectives:0:0": {"action": "withdraw"},
        "perspectives:0:1": {"action": "revise", "item": revised},
    }}

    result = run_agent_review(task, "intent", runner, language="en", instructions="")
    entries = {entry["item_id"]: entry for entry in result.partition["fact_verification"]["items"]}

    assert result.status == "completed"
    assert [item["statement"] for item in result.perspectives[0]["assessments"]] == [
        "The revised original second assessment."]
    assert entries["perspectives:0:0"]["status"] == "withdrawn"
    assert entries["perspectives:0:1"]["accepted"]["statement"] == (
        "The revised original second assessment.")
    assert runner.verify_claim_ids[-1] == ["perspectives:0:1"]
    validate_partition_result(result, task)


def test_agent_fact_verification_rejects_summary_withdrawal(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.verify_outputs = [lambda _request, claims: {"items": {claim["id"]: {
        field: {"state": "contradicted" if claim["id"] == "summary" else "supported",
            "reason": "The source contradicts this field." if claim["id"] == "summary"
            else "The frozen source supports this field.", "citations": claim["cited"]}
        for field in claim["fields"]} for claim in claims}}]
    runner.revision_output = {"items": {"summary": {"action": "withdraw"}}}

    result = run_agent_review(task, "architecture", runner, language="en", instructions="")
    entry = next(row for row in result.partition["fact_verification"]["items"]
        if row["item_id"] == "summary")

    assert result.status == "partial_review"
    assert result.reason == "factual_verification_incomplete"
    assert entry["status"] == "unverified"
    assert any(issue["reason"] == "summary_withdrawal_rejected" for issue in entry["host_issues"])
    validate_partition_result(result, task)


def test_agent_verifier_unbound_citation_is_unverifiable(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.verify_outputs = [lambda _request, claims: {"items": {claim["id"]: {
        field: {"state": "supported", "reason": "Checked.", "citations": [
            {"path": "src/missing.java", "version": "head", "quote": "missing"}]}
        for field in claim["fields"]} for claim in claims}}]

    result = run_agent_review(task, "architecture", runner, language="en", instructions="", max_revisions=0)
    summary = result.partition["fact_verification"]["items"][0]

    assert result.status == "partial_review"
    assert summary["attempts"][0]["verdicts"]["text"]["state"] == "unverifiable"
    assert summary["attempts"][0]["verdicts"]["text"]["reason"] == "verifier_citation_unbound"
    validate_partition_result(result, task)


def test_agent_verifier_supported_without_citations_is_rejected(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    invalid = lambda _request, claims: {"items": {claim["id"]: {
        field: {"state": "supported", "reason": "Checked.", "citations": []}
        for field in claim["fields"]} for claim in claims}}
    runner.verify_outputs = [invalid]
    runner.verify_repair_outputs = [invalid]

    result = run_agent_review(task, "architecture", runner, language="en", instructions="", max_revisions=0)

    assert result.status == "partial_review"
    assert result.partition["fact_verification"]["items"][0]["attempts"][0]["verdicts"]["text"]["reason"] == (
        "verifier_output_invalid")
    validate_partition_result(result, task)


def test_agent_verifier_supported_pr_only_citation_is_downgraded(tmp_path):
    task = _pmd_partition_task()
    originals, _ = prepare_originals(task, "architecture")
    pr = SourceOriginal("E-PR-TEST", "A unique PR context statement for testing.", "pr_text")
    originals = [*originals, pr]
    spans = [make_span(original, 0, len(original.raw), "agent_verify") for original in originals]
    claims = [{"id": "summary", "kind": "summary", "fields": {"text": "A factual summary."}, "cited": []}]
    body = {"items": {"summary": {"text": {"state": "supported", "reason": "The PR describes it.",
        "citations": [{"path": "PR.md", "version": "pr", "quote": pr.text}]}}}}

    verdicts, issues = _validate_verifier_output(body, claims, originals, spans)

    assert verdicts["summary"]["text"]["state"] == "unverifiable"
    assert verdicts["summary"]["text"]["reason"] == "verifier_code_citation_missing"
    assert issues == [{"item_id": "summary", "field": "text",
        "reason": "verifier_code_citation_missing", "citations": verdicts["summary"]["text"]["citations"]}]


def test_agent_verification_replay_accepts_code_citation_missing_receipt(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    ledger = result.partition["fact_verification"]
    entry = next(item for item in ledger["items"] if item["item_id"] == "summary")
    verdict = entry["attempts"][0]["verdicts"]["text"]
    verdict.update(state="unverifiable", reason="verifier_code_citation_missing", citations=[])
    issue = {"item_id": "summary", "field": "text",
        "reason": "verifier_code_citation_missing", "citations": []}
    entry["attempts"][0]["host_issues"].append(issue)
    entry["host_issues"].append(issue)
    entry["status"] = "unverified"
    ledger["status"] = "unverified"
    object.__setattr__(result, "status", "partial_review")
    object.__setattr__(result, "reason", "factual_verification_incomplete")
    result.scope["review_complete"] = False

    validate_partition_result(result, task)


@pytest.mark.parametrize("reverify_failure", [False, True])
def test_agent_verifier_run_failure_is_recorded_on_affected_entries(tmp_path, reverify_failure):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    if reverify_failure:
        runner.verify_outputs = [lambda _request, claims: {"items": {claim["id"]: {
            field: {"state": "contradicted" if claim["id"] == "findings:0" and str(text).strip()
                else ("supported" if str(text).strip() else "non_factual"),
                "reason": "Revise this finding.", "citations": claim["cited"] if str(text).strip() else []}
            for field, text in claim["fields"].items()} for claim in claims}}]
        runner.verify_errors = [None, AgentRunError("proxy_budget_exhausted")]
        runner.revision_output = {"items": {"findings:0": {"action": "revise",
            "item": _architecture_body(task)["findings"][0]}}}
    else:
        runner.verify_errors = [AgentRunError("proxy_budget_exhausted")]

    result = run_agent_review(task, "architecture", runner, language="en", instructions="")
    entry = next(row for row in result.partition["fact_verification"]["items"]
        if row["item_id"] == ("findings:0" if reverify_failure else "summary"))

    assert result.status == "partial_review"
    assert {"reason": "verifier_run_failed:proxy_budget_exhausted"} in entry["host_issues"]
    validate_partition_result(result, task)


def test_agent_fact_rewrite_of_nested_assessment_uses_only_code_citations(tmp_path):
    task = make_intent_task()
    runner = FakeAgentRunner(tmp_path, _intent_body(task))
    runner.verify_outputs = [
        lambda req, claims: {"items": {claim["id"]: {
            field: {"state": "contradicted" if claim["id"] == "perspectives:0:0" else "supported",
                "reason": "The source contradicts this field." if claim["id"] == "perspectives:0:0"
                else "The frozen source supports this field.",
                "citations": claim["cited"]} for field in claim["fields"]} for claim in claims}},
    ]
    runner.revision_output = {"items": {"perspectives:0:0": {"action": "revise",
        "item": _intent_body(task)["perspectives"][0]["assessments"][0]}}}
    result = run_agent_review(task, "intent", runner, language="en", instructions="")
    validate_partition_result(result, task)
    entry = next(row for row in result.partition["fact_verification"]["items"]
        if row["item_id"] == "perspectives:0:0")
    assert len(entry["attempts"]) == 2
    assert result.status == "completed"


def test_agent_fact_check_failure_is_partial_and_ledger_tampering_fails_replay(tmp_path):
    task = make_intent_task()
    runner = FakeAgentRunner(tmp_path, _intent_body(task))
    runner.verify_outputs = [{"invalid": "response"}]
    runner.verify_repair_outputs = [{"invalid": "response"}]
    result = run_agent_review(task, "intent", runner, language="en", instructions="", max_revisions=0)

    assert result.status == "partial_review"
    assert result.reason == "factual_verification_incomplete"
    assert runner.repair_calls == 1
    assert all(entry["status"] == "unverified" for entry in result.partition["fact_verification"]["items"])
    assert all(verdict["reason"] == "verifier_output_invalid"
        for entry in result.partition["fact_verification"]["items"]
        for attempt in entry["attempts"] for verdict in attempt["verdicts"].values())
    validate_partition_result(result, task)

    altered = deepcopy(result)
    altered.partition["fact_verification"]["original_final_hash"] = "0" * 64
    with pytest.raises(PartitionError, match="stored_agent_verification_final"):
        validate_partition_result(altered, task)


@pytest.mark.parametrize("mutation", ["verdict", "citation", "final", "status"])
def test_agent_verification_replay_rejects_tampering(tmp_path, mutation):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    altered = deepcopy(result)
    ledger = altered.partition["fact_verification"]
    if mutation == "verdict":
        ledger["items"][0]["attempts"][0]["verdicts"]["text"]["state"] = "contradicted"
    elif mutation == "citation":
        ledger["items"][0]["attempts"][0]["verdicts"]["text"]["citations"][0]["quote"] += " tampered"
    elif mutation == "final":
        altered.partition["final"]["summary"] += " tampered"
        object.__setattr__(altered, "summary", altered.partition["final"]["summary"])
        ledger["accepted_final_hash"] = hash_json(altered.partition["final"])
    else:
        ledger["status"] = "unverified"

    with pytest.raises(PartitionError, match="stored_agent_verification_|agent_"):
        validate_partition_result(altered, task)


def test_agent_verification_replay_accepts_legacy_v1_after_global_version_changes(tmp_path, monkeypatch):
    task = _pmd_partition_task()
    monkeypatch.setattr(agent_review, "_AGENT_VERIFY_VERSION", "agent-verify-v1")

    result = _run(task, "architecture", tmp_path, _architecture_body(task))

    assert result.partition["fact_verification"]["version"] == "agent-verify-v1"
    finding = next(entry for entry in result.partition["fact_verification"]["items"]
        if entry["item_id"] == "findings:0")
    assert set(finding["attempts"][0]["verdicts"]) == {
        "statement", "reasoning", "failure_condition"}
    monkeypatch.undo()
    validate_partition_result(result, task)


def test_agent_verification_replay_rejects_v2_failure_condition_verdict_and_unknown_version(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))

    altered = deepcopy(result)
    finding = next(entry for entry in altered.partition["fact_verification"]["items"]
        if entry["item_id"] == "findings:0")
    finding["attempts"][0]["verdicts"]["failure_condition"] = {
        "state": "unverifiable", "reason": "Not a verified field.", "citations": []}
    with pytest.raises(PartitionError, match="stored_agent_verification_verdict"):
        validate_partition_result(altered, task)

    altered = deepcopy(result)
    altered.partition["fact_verification"]["version"] = "agent-verify-v3"
    with pytest.raises(PartitionError, match="stored_agent_verification_ledger"):
        validate_partition_result(altered, task)


def test_agent_fact_verification_without_review_client_is_not_configured(tmp_path):
    task = make_intent_task()
    result = _run(task, "intent", tmp_path, _intent_body(task))

    assert result.partition["fact_verification"]["version"] == "agent-verify-v2"
    assert result.partition["fact_verification"]["status"] == "model_supported"
    markdown = format_partition_review_markdown(result, task, publication_mode="artifact_only")
    assert "Code claims were checked by a model" in markdown
    validate_partition_result(result, task)


def test_agent_summary_citations_are_bound_to_stored_code_citations(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    result = run_agent_review(task, "architecture", runner, language="en", instructions="")

    validate_partition_result(result, task)
    citations = result.partition["final"]["summary_citations"]
    originals, _meta = prepare_originals(task, "architecture")
    originals_by_id = {item.evidence_id: item for item in originals}
    assert len(citations) == 1
    citation = citations[0]
    source = originals_by_id[citation["evidence_id"]]
    assert source.path and source.version == "head"
    assert citation["quote"] == "void run()"
    assert "summary_citations" in runner.generation_contract
    assert runner.generation_contract_json["required_keys"]["envelope"][1] == "summary_citations"
    assert runner.generation_contract_json["citation_rules"]["summary_citations_required"] is True


def test_bad_summary_citation_is_rejected_and_repaired_once(tmp_path):
    task = _pmd_partition_task()
    first = _architecture_body(task)
    first["summary_citations"][0]["quote"] = "this quote is not in the source"
    corrected = _architecture_body(task)
    runner = FakeAgentRunner(tmp_path, first, repair_output=corrected)
    result = _run(task, "architecture", tmp_path, first, runner=runner)

    assert runner.repair_calls == 1
    assert result.partition["agent"]["repair_attempted"]
    assert result.partition["agent"]["rejected_before"] == [
        {"kind": "envelope", "index": 0, "reason": "agent_summary_citations_invalid"}]
    assert result.status == "completed"
    assert result.partition["final"]["summary_citations"][0]["quote"] == "void run()"


def test_verifier_pr_citation_binds_only_for_verifier(tmp_path):
    task = make_intent_task()
    runner = FakeAgentRunner(tmp_path, _intent_body(task))
    pr_quote = "Intent fixture"

    def pr_verdicts(_request, claims):
        return {"items": {claim["id"]: {
            field: {"state": "supported", "reason": "Checked in PR context.",
                "citations": [{"path": "PR.md", "version": "pr", "quote": pr_quote}]}
            for field in claim["fields"]}
            for claim in claims}}

    runner.verify_outputs = [pr_verdicts]
    result = _run(task, "intent", tmp_path, _intent_body(task), runner=runner)

    assert result.status == "partial_review"
    assert result.reason == "factual_verification_incomplete"
    summary_verdict = result.partition["fact_verification"]["items"][0]["attempts"][0]["verdicts"]["text"]
    assert summary_verdict["state"] == "unverifiable"
    assert summary_verdict["reason"] == "verifier_code_citation_missing"
    assert summary_verdict["citations"][0]["path"] == "PR.md"
    assert summary_verdict["citations"][0]["version"] == "pr"
    assert summary_verdict["citations"][0]["evidence_id"] == "E-PR-01"


def test_generation_pr_citation_is_rejected(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    body["summary_citations"] = [{"path": "PR.md", "version": "pr", "quote": "Intent fixture"}]
    runner = FakeAgentRunner(tmp_path, body, repair_output=body)

    result = _run(task, "intent", tmp_path, body, runner=runner)

    assert result.status == "failed"
    assert result.reason == "agent_summary_citations_invalid"
    validate_partition_result(result, task)


def test_missing_summary_citations_fail_without_a_second_repair(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    del body["summary_citations"]
    runner = FakeAgentRunner(tmp_path, body)
    result = _run(task, "architecture", tmp_path, body, runner=runner)

    assert runner.repair_calls == 1
    assert result.status == "failed"
    assert result.reason == "agent_output_summary_citations_missing"
    validate_partition_result(result, task)


def test_tampered_stored_summary_citation_is_rejected(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    altered = deepcopy(result)
    altered.partition["final"]["summary_citations"][0]["end_byte"] += 1
    with pytest.raises(PartitionError, match="agent_citation_tampered"):
        validate_partition_result(altered, task)


def test_agent_uses_result_file_instead_of_stdout(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    result = _run(task, "architecture", tmp_path, body)
    assert result.status == "completed"
    validate_partition_result(result, task)


def test_malformed_agent_json_fails_closed(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, "not json")
    result = _run(task, "architecture", tmp_path, "not json", runner=runner)
    assert result.status == "failed" and result.reason == "agent_output_json_invalid"
    assert runner.repair_calls == 1 and len(runner.finished) == 1
    assert runner.run_modes == ["architecture"]
    validate_partition_result(result, task)


def test_missing_output_file_is_repaired_with_specific_code(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.omit_output = True
    result = _run(task, "architecture", tmp_path, runner.output, runner=runner)

    assert result.status == "failed" and result.reason == "agent_output_missing"
    assert result.partition["agent"]["rejected_before"][0]["reason"] == "agent_output_missing"
    assert any("agent_output_missing" in message for message in runner.repair_messages)
    assert any("/out/result.json" in message and "node /skills/check_output.mjs" in message
        for message in runner.repair_messages)
    validate_partition_result(result, task)


def test_oversize_output_file_is_rejected_and_repaired(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.oversize_output = True
    result = _run(task, "architecture", tmp_path, runner.output, runner=runner)

    assert result.status == "failed" and result.reason == "agent_output_too_large"
    assert result.partition["agent"]["rejected_before"][0]["reason"] == "agent_output_too_large"
    validate_partition_result(result, task)


def test_json_parse_diagnostic_is_sent_in_repair_message(tmp_path):
    task = _pmd_partition_task()
    corrected = _architecture_body(task)
    runner = FakeAgentRunner(tmp_path, "{", repair_output=corrected)
    result = _run(task, "architecture", tmp_path, "{", runner=runner)

    assert result.status == "completed"
    assert "agent_output_json_invalid" in runner.repair_message
    assert any("offset=" in message and "context=" in message for message in runner.repair_messages)
    assert "node /skills/check_output.mjs" in runner.repair_message
    assert "DONE" in runner.repair_message
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
    assert result.status == "completed"
    assert result.partition["agent"]["repair_attempted"]
    assert result.partition["agent"]["rejected"][0]["reason"] == reason
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert "Rejected items: 1" in markdown and reason in markdown
    validate_partition_result(result, task)


def test_exact_duplicate_agent_items_are_dropped(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    body["findings"].append(deepcopy(body["findings"][0]))
    result = _run(task, "architecture", tmp_path, body)
    assert len(result.findings) == 1
    assert result.partition["agent"]["rejected"] == [{"kind": "finding", "index": 1, "reason": "duplicate"}]
    validate_partition_result(result, task)


def test_untargeted_observed_structure_finding_is_accepted_and_keeps_signal_section(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    body["findings"][0]["target_ids"] = []
    result = _run(task, "architecture", tmp_path, body)

    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert result.status == "completed"
    assert result.findings[0]["target_ids"] == []
    assert "A concrete defect" in markdown and "void run()" in markdown
    assert "PMD/CPD" in markdown


def test_observed_structure_can_have_empty_target_set(tmp_path):
    from dataclasses import replace
    from review_bot.application.context import build_task_input

    base_task = _pmd_partition_task()
    task = build_task_input(base_task.snapshot, replace(base_task.delta, signals=()), base_task.classification)
    body = _architecture_body(base_task)
    body["findings"][0]["target_ids"] = []
    result = _run(task, "architecture", tmp_path, body)

    validate_partition_result(result, task)
    assert result.status == "completed"


def test_analysis_signal_without_target_is_rejected_but_architecture_completes(tmp_path):
    task = _pmd_partition_task()
    body = _architecture_body(task)
    body["findings"][0]["basis"] = "analysis_signal"
    body["findings"][0]["target_ids"] = []
    result = _run(task, "architecture", tmp_path, body)

    assert result.status == "completed"
    assert result.findings == ()
    assert result.partition["agent"]["rejected"][0]["reason"] == "agent_finding_unknown_target"
    validate_partition_result(result, task)


def test_agent_validator_rejects_analysis_signal_without_target(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    altered = deepcopy(result)
    altered.partition["final"]["findings"][0]["basis"] = "analysis_signal"
    altered.partition["final"]["findings"][0]["target_ids"] = []

    with pytest.raises(PartitionError, match="invalid_agent_finding"):
        validate_partition_result(altered, task)


def test_intent_insufficient_context_without_citations_is_valid_and_renders(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    assessment = body["perspectives"][0]["assessments"][0]
    assessment.update(verdict="insufficient_context", citations=[])
    result = _run(task, "intent", tmp_path, body)

    validate_partition_result(result, task)
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert result.status == "partial_review"
    assert result.reason == "factual_verification_incomplete"
    assert result.perspectives[0]["assessments"][0]["citations"] == []
    assert "Review incomplete" in markdown


def test_agent_validator_rejects_concern_without_citations(tmp_path):
    task = make_intent_task()
    result = _run(task, "intent", tmp_path, _intent_body(task))
    altered = deepcopy(result)
    altered.partition["final"]["perspectives"][0]["assessments"][0]["citations"] = []

    with pytest.raises(PartitionError, match="agent_citations_missing"):
        validate_partition_result(altered, task)


def test_intent_concern_without_citation_is_rejected_and_cannot_complete_missing_perspective(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    body["perspectives"][0]["assessments"][0].update(verdict="concern", citations=[])
    result = _run(task, "intent", tmp_path, body)

    assert result.status == "partial_review"
    assert result.partition["agent"]["rejected"][0]["reason"] == "agent_item_citations_missing"
    validate_partition_result(result, task)


def test_intent_rejected_assessment_does_not_downgrade_complete_perspectives(tmp_path):
    task = make_intent_task()
    body = _intent_body(task)
    rejected = deepcopy(body["perspectives"][0]["assessments"][0])
    rejected.update(verdict="concern", citations=[])
    body["perspectives"][0]["assessments"].append(rejected)
    result = _run(task, "intent", tmp_path, body)

    assert result.status == "completed"
    assert result.reason == "agent_items_rejected"
    assert result.partition["agent"]["rejected"][0]["reason"] == "agent_item_citations_missing"
    validate_partition_result(result, task)


def test_repair_success_replaces_first_output_and_never_sends_rejected_prose(tmp_path):
    task = _pmd_partition_task()
    first = _architecture_body(task)
    first["findings"][0]["target_ids"] = ["unknown"]
    corrected = _architecture_body(task)
    runner = FakeAgentRunner(tmp_path, first, repair_output=corrected)
    result = _run(task, "architecture", tmp_path, first, runner=runner)

    assert result.status == "completed" and len(result.findings) == 1
    assert result.partition["agent"]["repair_attempted"]
    assert result.partition["agent"]["before_rejected_count"] == 1
    assert result.partition["agent"]["after_rejected_count"] == 0
    assert "agent_finding_unknown_target" in runner.repair_message
    assert "code=agent_finding_unknown_target" in runner.repair_message
    assert "item_index=0" in runner.repair_message
    assert "A concrete defect." not in runner.repair_message
    assert "/out/result.json" in runner.repair_message
    assert "node /skills/check_output.mjs" in runner.repair_message
    assert runner.repair_calls == 1 and len(runner.finished) == 2
    validate_partition_result(result, task)


def test_repair_failure_is_failed_and_does_not_attempt_a_second_repair(tmp_path):
    task = _pmd_partition_task()
    first = _architecture_body(task)
    first["findings"][0]["target_ids"] = ["unknown"]
    runner = FakeAgentRunner(tmp_path, first, repair_output="still not json")
    result = _run(task, "architecture", tmp_path, first, runner=runner)

    assert result.status == "failed" and result.reason == "agent_output_json_invalid"
    assert runner.repair_calls == 1
    assert result.partition["agent"]["repair_attempted"]
    assert result.partition["agent"]["after_rejected_count"] == 1
    validate_partition_result(result, task)


def test_agent_reported_usage_is_separate_from_proxy_usage_and_rendered(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    agent = result.partition["agent"]

    assert agent["usage"]["input_tokens"] == 9
    assert agent["usage"]["cost"]["total"] == 0.025
    assert agent["usage"]["max_prompt_tokens"] == 5 and agent["usage"]["compactions"] == 1
    assert agent["agent_reported_usage"]["input_tokens"] == 10
    assert agent["proxy_ledger_path"].endswith("usage.jsonl")
    assert result.usage["max_prompt_tokens"] == 5 and result.usage["compactions"] == 1
    verify_usage = next(run["usage"] for run in result.partition["fact_verification"]["runs"]
        if run["role"] == "verify")
    assert verify_usage["max_prompt_tokens"] == 5 and verify_usage["compactions"] == 1
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert "Rejected items: 0" in markdown
    assert "| agent_review | 0 | 2 | 2 | 0 |" in markdown
    assert "Agent usage: input 9 · output 4 · total 13 tokens · max request input 5 · compactions 1" in markdown
    assert "Agent verification usage:" in markdown
    assert "max request input 5 · compactions 1" in markdown


def test_agent_result_validator_accepts_legacy_usage_without_context_metrics(tmp_path):
    task = _pmd_partition_task()
    result = _run(task, "architecture", tmp_path, _architecture_body(task))
    legacy = deepcopy(result)
    for usage in (legacy.partition["agent"]["usage"], legacy.usage):
        usage.pop("max_prompt_tokens")
        usage.pop("compactions")
    for run in legacy.partition["fact_verification"]["runs"]:
        usage = run.get("usage")
        if isinstance(usage, dict):
            usage.pop("max_prompt_tokens", None)
            usage.pop("compactions", None)

    validate_partition_result(legacy, task)


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
    assert runner.run_modes == ["architecture"]
    ledger = result.partition["fact_verification"]
    assert ledger["status"] == "unverified" and ledger["runs"] == []
    assert ledger["items"]
    for entry in ledger["items"]:
        assert entry["status"] == "unverified"
        assert entry["host_issues"] == [{"reason": "generation_failed"}]
        assert all(verdict["state"] == "unverifiable"
            and verdict["reason"] == "generation_failed"
            for verdict in entry["attempts"][0]["verdicts"].values())
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
