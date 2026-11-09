from __future__ import annotations

import gzip
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
from review_bot.application.agent_review import run_agent_review
from review_bot.application.partition_review import PERSPECTIVE_IDS, prepare_originals, validate_partition_result
from review_bot.application.source_partition import PartitionError
from review_bot.domain.partition_review import PartitionReviewLimits
from review_bot.ports import AgentRunError, AgentRunFinalization, AgentRunResponse
from tests.fixtures.intent_perspectives import make_task as make_intent_task
from tests.fixtures.partition_review import PartitionModelClient
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
        self.session = session or b'{"type":"assistant","message":{"usage":{"input":10,"output":5,"totalTokens":15}}}\n'
        self.config = SimpleNamespace(max_proxy_calls=12, max_output_tokens=2048)
        self.workspace = None
        self.contract = ""
        self.contract_json = {}
        self.output_path = tmp_path / "fake-out" / "result.json"
        self.omit_output = False
        self.repair_omit_output = None
        self.oversize_output = False
        self.repair_oversize_output = None
        self.request = None
        self.repair_calls = 0
        self.repair_message = None
        self.finished = []
        self.proxy_usage = {"input_tokens": 9, "output_tokens": 4, "total_tokens": 13,
            "model_calls": 2, "usage_complete": True, "unknown_usage_attempts": 0,
            "cost": {"total": 0.025}, "calls": [
                {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "cost": {"total": 0.01}, "status": 200},
                {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7, "cost": {"total": 0.015}, "status": 200},
            ]}

    def run(self, request):
        self.request = request
        self.workspace = {
            path.relative_to(request.workspace_dir).as_posix()
            for path in request.workspace_dir.rglob("*") if path.is_file()
        }
        self.contract = (request.workspace_dir / "CONTRACT.md").read_text(encoding="utf-8")
        self.contract_json = json.loads((request.workspace_dir / "contract.json").read_text(encoding="utf-8"))
        if self.error:
            raise self.error
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
        return AgentRunResponse("DONE", self.session, usage, "0.73.1", 0,
            session_id, self.output_path)

    def repair(self, request, session_id, message):
        self.repair_calls += 1
        self.repair_message = message
        if self.repair_error:
            raise self.repair_error
        body = self.repair_output if self.repair_output is not None else self.output
        body = body(request) if callable(body) else body
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
    assert "head/src/A.java" in runner.workspace and "base/src/A.java" in runner.workspace
    assert {"PR.md", "signals.json", "CONTRACT.md"} <= runner.workspace
    assert "contract.json" in runner.workspace
    assert "target_ids may be empty only when basis is observed_structure" in runner.contract
    assert "Every finding and assessment needs at least one citation" in runner.contract
    assert "empty intent assessments list means no distinct finding" in runner.contract
    assert runner.contract_json["mode"] == "architecture"
    assert runner.contract_json["allowed_target_ids"] == prepare_originals(task, "architecture")[1]["targets"]
    assert runner.contract_json["required_keys"]["finding"][-1] == "citations"
    assert runner.contract_json["citation_rules"]["empty_target_ids_basis"] == "observed_structure"


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
    task = make_intent_task()
    review_client = PartitionModelClient()
    result = run_agent_review(task, "intent", FakeAgentRunner(tmp_path, _intent_body(task)),
        language="en", instructions="", review_client=review_client,
        fact_limits=PartitionReviewLimits(max_calls=128, max_total_tokens=1_000_000))

    validate_partition_result(result, task)
    assert result.status == "completed"
    assert result.partition["fact_verification"]["status"] == "model_supported"
    assert any(row["item_id"] == "perspectives:0:0"
        for row in result.partition["fact_verification"]["items"])
    assert (result.partition["pre_verification_final"]["perspectives"][0]["assessments"][0]["statement"]
        == result.perspectives[0]["assessments"][0]["statement"])


def test_agent_fact_verification_rewrites_contradicted_assessment(tmp_path):
    task = _pmd_partition_task()
    contradicted = False

    def transform(payload, body):
        nonlocal contradicted
        if payload["stage"] == "fact_check":
            row_id = "findings:0"
            if not contradicted and row_id in body.get("items", {}):
                contradicted = True
                body["items"][row_id]["reasoning"] = {
                    "state": "contradicted", "reason": "The source contradicts this explanation.",
                    "citations": body["items"][row_id]["reasoning"]["citations"]}
        elif payload["stage"] == "fact_rewrite":
            replacements = (body.get("items", {}).values() if "items" in body else [body])
            for replacement in replacements:
                replacement["replacement"]["reasoning"] = "Rewritten against the supplied source."
        return body

    review_client = PartitionModelClient(transform=transform)
    result = run_agent_review(task, "architecture", FakeAgentRunner(tmp_path, _architecture_body(task)),
        language="en", instructions="", review_client=review_client,
        fact_limits=PartitionReviewLimits(max_calls=128, max_total_tokens=1_000_000))

    validate_partition_result(result, task)
    entry = next(row for row in result.partition["fact_verification"]["items"] if row["item_id"] == "findings:0")
    assert len(entry["attempts"]) == 2
    assert result.findings[0]["reasoning"] == "Rewritten against the supplied source."


def test_agent_fact_rewrite_of_nested_assessment_uses_only_code_citations(tmp_path):
    task = make_intent_task()
    contradicted = False

    def transform(payload, body):
        nonlocal contradicted
        if payload["stage"] == "fact_check":
            rewrite_row = body["items"]["perspectives:0:0"]
            if not contradicted:
                contradicted = True
                rewrite_row["reasoning"] = {
                    "state": "contradicted", "reason": "The source contradicts this explanation.",
                    "citations": rewrite_row["reasoning"]["citations"]}
            non_code = next(source for source in payload["citation_catalog"]
                if not source["path"] or source["version"] not in ("head", "base"))
            if "perspectives:1:0" in body["items"]:
                body["items"]["perspectives:1:0"]["statement"]["citations"] = [
                    {"reference_id": non_code["ranges"][0]["reference_id"]}]
        elif payload["stage"] == "fact_rewrite":
            code_source = next(source for source in payload["citation_catalog"]
                if source["path"] and source["version"] in ("head", "base"))
            reference = code_source["ranges"][0]["reference_id"]
            replacements = body.get("items", {}).values() if "items" in body else [body]
            for item in replacements:
                item["replacement"]["reasoning"] = "Rewritten against the supplied source."
                item["replacement"]["citations"] = [{"reference_id": reference}]
                item["correction_citations"] = [{"reference_id": reference}]
        return body

    review_client = PartitionModelClient(transform=transform)
    result = run_agent_review(task, "intent", FakeAgentRunner(tmp_path, _intent_body(task)),
        language="en", instructions="", review_client=review_client,
        fact_limits=PartitionReviewLimits(max_calls=128, max_total_tokens=1_000_000))

    validate_partition_result(result, task)
    entry = next(row for row in result.partition["fact_verification"]["items"]
        if row["item_id"] == "perspectives:0:0")
    assert len(entry["attempts"]) == 2
    assert result.perspectives[0]["assessments"][0]["reasoning"] == "Rewritten against the supplied source."
    originals, _ = prepare_originals(task, "intent")
    originals_by_id = {source.evidence_id: source for source in originals}
    for perspective in result.perspectives:
        for assessment in perspective["assessments"]:
            for citation in assessment["citations"]:
                source = originals_by_id[citation["evidence_id"]]
                assert source.path and source.version in ("head", "base")
                assert (citation["path"], citation["version"]) == (source.path, source.version)
    rewrite_payload = next(json.loads(request.user_content) for request in review_client.requests
        if json.loads(request.user_content)["stage"] == "fact_rewrite")
    assert all(source["path"] and source["version"] in ("head", "base")
        for source in rewrite_payload["citation_catalog"])


def test_agent_fact_check_failure_is_partial_and_ledger_tampering_fails_replay(tmp_path):
    task = make_intent_task()

    def fail_fact_check(payload, body):
        return {"invalid": "response"} if payload["stage"] == "fact_check" else body

    result = run_agent_review(task, "intent", FakeAgentRunner(tmp_path, _intent_body(task)),
        language="en", instructions="", review_client=PartitionModelClient(transform=fail_fact_check),
        fact_limits=PartitionReviewLimits(max_calls=128, max_total_tokens=1_000_000))

    assert result.status == "partial_review"
    assert result.reason == "factual_verification_incomplete"
    validate_partition_result(result, task)

    altered = deepcopy(result)
    altered.partition["fact_verification"]["original_final_hash"] = "0" * 64
    with pytest.raises(PartitionError, match="stored_fact_verification_missing"):
        validate_partition_result(altered, task)


def test_agent_fact_verification_without_review_client_is_not_configured(tmp_path):
    task = make_intent_task()
    result = _run(task, "intent", tmp_path, _intent_body(task))

    assert result.partition["fact_verification"] == "not_configured"
    markdown = format_partition_review_markdown(result, task, publication_mode="artifact_only")
    assert "No factual-review model is configured." in markdown
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
    assert "summary_citations" in runner.contract
    assert runner.contract_json["required_keys"]["envelope"][1] == "summary_citations"
    assert runner.contract_json["citation_rules"]["summary_citations_required"] is True


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
    assert runner.repair_calls == 1 and runner.finished == ["fake-session"]
    validate_partition_result(result, task)


def test_missing_output_file_is_repaired_with_specific_code(tmp_path):
    task = _pmd_partition_task()
    runner = FakeAgentRunner(tmp_path, _architecture_body(task))
    runner.omit_output = True
    result = _run(task, "architecture", tmp_path, runner.output, runner=runner)

    assert result.status == "failed" and result.reason == "agent_output_missing"
    assert result.partition["agent"]["rejected_before"][0]["reason"] == "agent_output_missing"
    assert "agent_output_missing" in runner.repair_message
    assert "/out/result.json" in runner.repair_message
    assert "node /skills/check_output.mjs" in runner.repair_message
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
    assert "offset=" in runner.repair_message and "context=" in runner.repair_message
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
    assert result.status == "completed"
    assert result.perspectives[0]["assessments"][0]["citations"] == []
    assert "Assessment for" in markdown


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
    assert runner.repair_calls == 1 and runner.finished == ["fake-session"]
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
    assert agent["agent_reported_usage"]["input_tokens"] == 10
    assert agent["proxy_ledger_path"].endswith("usage.jsonl")
    markdown = format_partition_review_markdown(result, task, publication_mode="comment")
    assert "Rejected items: 0" in markdown
    assert "| agent_review | 0 | 2 | 2 | 0 |" in markdown


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
