from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from review_bot.application.agent_review import _contract_json


CHECKER = Path(__file__).resolve().parents[2] / "deploy" / "agent-sandbox" / "check_output.mjs"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _run_checker(tmp_path, mode, result, source=b"void run() {}", *, claims=None, pr_text=None):
    workspace = tmp_path / "workspace"
    output = tmp_path / "out"
    (workspace / "head" / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "base").mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=True)
    if source is not None:
        (workspace / "head" / "src" / "A.java").write_bytes(source)
    if pr_text is not None:
        (workspace / "PR.md").write_text(pr_text, encoding="utf-8")
    if mode == "verify":
        (workspace / "claims.json").write_text(json.dumps({"claims": claims or [{
            "id": "summary", "kind": "summary", "fields": {"text": "Review."}, "cited": [],
        }]}), encoding="utf-8")
    contract = _contract_json(mode, {"targets": ["target-1"]})
    (workspace / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    if isinstance(result, bytes):
        (output / "result.json").write_bytes(result)
    else:
        (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
    env = {
        **os.environ,
        "REVIEW_BOT_WORKSPACE_DIR": str(workspace),
        "REVIEW_BOT_OUTPUT_DIR": str(output),
    }
    return subprocess.run(["node", str(CHECKER)], capture_output=True, text=True,
        env=env, timeout=10, check=False)


def _finding(*, verdict="concern", basis="observed_structure", target_ids=None, quote="void run()"):
    return {
        "statement": "One finding.",
        "reasoning": "The supplied source supports this.",
        "failure_condition": "",
        "verdict": verdict,
        "basis": basis,
        "target_ids": [] if target_ids is None else target_ids,
        "citations": [{"path": "src/A.java", "version": "head", "quote": quote}],
    }


def _summary_citations(quote="void run()"):
    return [{"path": "src/A.java", "version": "head", "quote": quote}]


def test_checker_reports_json_parse_message_offset_and_context(tmp_path):
    result = _run_checker(tmp_path, "architecture", b'{"summary":')

    assert result.returncode == 1
    assert "/: agent_output_json_invalid" in result.stdout
    assert "offset=" in result.stdout and "context=" in result.stdout
    assert "summary" in result.stdout


def test_checker_rejects_bad_enum_with_pointer(tmp_path):
    result = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "summary_citations": _summary_citations(),
        "findings": [_finding(verdict="bad_verdict")],
    })

    assert result.returncode == 1
    assert "/findings/0/verdict: agent_output_enum_invalid" in result.stdout


@pytest.mark.parametrize(("quote", "source", "code"), [
    ("absent", b"void run() {}", "agent_output_quote_absent"),
    ("again", b"again again", "agent_output_quote_ambiguous"),
])
def test_checker_rejects_absent_or_ambiguous_quote(tmp_path, quote, source, code):
    result = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "summary_citations": _summary_citations(),
        "findings": [_finding(quote=quote)],
    }, source=source)

    assert result.returncode == 1
    assert f"/findings/0/citations/0/quote: {code}" in result.stdout


def test_checker_requires_target_for_analysis_signal(tmp_path):
    result = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "summary_citations": _summary_citations(),
        "findings": [_finding(basis="analysis_signal")],
    })

    assert result.returncode == 1
    assert "/findings/0/target_ids: agent_output_targets_invalid" in result.stdout


def test_checker_allows_empty_citations_only_for_insufficient_context(tmp_path):
    contract = _contract_json("intent", {"targets": []})
    perspective_id = contract["allowed_perspective_ids"][0]
    result = _run_checker(tmp_path, "intent", {
        "summary": "Review.",
        "summary_citations": _summary_citations(),
        "perspectives": [{
            "perspective_id": perspective_id,
            "assessments": [{
                "verdict": "insufficient_context",
                "statement": "More evidence is required.",
                "reasoning": "The available source does not establish this.",
                "failure_condition": "",
                "citations": [],
            }],
        }],
        "questions": [],
    })

    assert result.returncode == 0
    assert result.stdout == "OK\n"


def test_checker_accepts_empty_assessments_for_required_perspective(tmp_path):
    contract = _contract_json("intent", {"targets": []})
    result = _run_checker(tmp_path, "intent", {
        "summary": "Review.",
        "summary_citations": _summary_citations(),
        "perspectives": [{
            "perspective_id": contract["allowed_perspective_ids"][0],
            "assessments": [],
        }],
        "questions": [],
    })

    assert contract["citation_rules"]["empty_assessments_allowed"] is True
    assert contract["citation_rules"]["empty_assessments_meaning"] == "no_distinct_finding"
    assert result.returncode == 0
    assert result.stdout == "OK\n"


def test_checker_accepts_valid_file_with_exact_unique_quote(tmp_path):
    result = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "summary_citations": _summary_citations(),
        "findings": [_finding(target_ids=["target-1"])],
    })

    assert result.returncode == 0
    assert result.stdout == "OK\n"


def test_checker_requires_summary_citations_when_code_exists(tmp_path):
    missing = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "findings": [],
    })
    empty = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "summary_citations": [],
        "findings": [],
    })

    assert missing.returncode == 1
    assert "/summary_citations: agent_output_summary_citations_missing" in missing.stdout
    assert empty.returncode == 1
    assert "/summary_citations: agent_output_summary_citations_missing" in empty.stdout


def test_checker_allows_empty_summary_citations_without_code(tmp_path):
    result = _run_checker(tmp_path, "architecture", {
        "summary": "No code was available.",
        "summary_citations": [],
        "findings": [],
    }, source=None)

    assert result.returncode == 0
    assert result.stdout == "OK\n"


@pytest.mark.parametrize(("quote", "source", "code"), [
    ("absent", b"void run() {}", "agent_output_quote_absent"),
    ("again", b"again again", "agent_output_quote_ambiguous"),
])
def test_checker_rejects_invalid_summary_quote(tmp_path, quote, source, code):
    result = _run_checker(tmp_path, "architecture", {
        "summary": "Review.",
        "summary_citations": _summary_citations(quote),
        "findings": [],
    }, source=source)

    assert result.returncode == 1
    assert f"/summary_citations/0/quote: {code}" in result.stdout


def test_checker_verify_requires_exact_rows_and_fields(tmp_path):
    claims = [{
        "id": "summary", "kind": "summary", "fields": {"text": "Review."}, "cited": [],
    }, {
        "id": "findings:0", "kind": "finding",
        "fields": {"statement": "One finding.", "reasoning": "The source supports this."},
        "cited": [],
    }]
    result = _run_checker(tmp_path, "verify", {
        "items": {
            "summary": {"text": {"state": "supported", "reason": "Checked.",
                "citations": [{"path": "src/A.java", "version": "head", "quote": "void run()"}]}},
            "findings:0": {
                "statement": {"state": "non_factual", "reason": "A report.", "citations": []},
                "reasoning": {"state": "supported", "reason": "Checked.",
                    "citations": [{"path": "src/A.java", "version": "head", "quote": "void run()"}]},
            },
        },
    }, claims=claims)

    assert result.returncode == 0
    assert result.stdout == "OK\n"

    missing_field = _run_checker(tmp_path / "missing", "verify", {
        "items": {
            "summary": {"text": {"state": "non_factual", "reason": "A report.", "citations": []}},
            "findings:0": {"statement": {"state": "non_factual", "reason": "A report.", "citations": []}},
        },
    }, claims=claims)
    assert missing_field.returncode == 1
    assert "agent_verify_field_coverage_invalid" in missing_field.stdout


def test_checker_verify_rejects_supported_without_citation(tmp_path):
    result = _run_checker(tmp_path, "verify", {
        "items": {"summary": {"text": {"state": "supported", "reason": "Checked.", "citations": []}}},
    })

    assert result.returncode == 1
    assert "/items/summary/text/citations: agent_verify_supported_citations_missing" in result.stdout


def test_checker_verify_allows_pr_text_context_and_rejects_wrong_path(tmp_path):
    valid = _run_checker(tmp_path, "verify", {
        "items": {"summary": {"text": {"state": "unverifiable", "reason": "PR context only.",
            "citations": [{"path": "PR.md", "version": "pr", "quote": "Changed behavior."}]}}},
    }, pr_text="Changed behavior.")
    assert valid.returncode == 0
    assert valid.stdout == "OK\n"

    unsupported = _run_checker(tmp_path / "supported", "verify", {
        "items": {"summary": {"text": {"state": "supported", "reason": "Checked.",
            "citations": [{"path": "PR.md", "version": "pr", "quote": "Changed behavior."}]}}},
    }, pr_text="Changed behavior.")
    assert unsupported.returncode == 1
    assert "/items/summary/text/citations: verifier_code_citation_missing" in unsupported.stdout

    invalid = _run_checker(tmp_path / "invalid", "verify", {
        "items": {"summary": {"text": {"state": "supported", "reason": "Checked.",
            "citations": [{"path": "src/A.java", "version": "pr", "quote": "void run()"}]}}},
    }, pr_text="Changed behavior.")
    assert invalid.returncode == 1
    assert "/items/summary/text/citations/0/path: agent_verify_citation_source_invalid" in invalid.stdout
