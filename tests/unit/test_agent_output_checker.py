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


def _run_checker(tmp_path, mode, result, source=b"void run() {}"):
    workspace = tmp_path / "workspace"
    output = tmp_path / "out"
    (workspace / "head" / "src").mkdir(parents=True, exist_ok=True)
    (workspace / "base").mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=True)
    if source is not None:
        (workspace / "head" / "src" / "A.java").write_bytes(source)
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
