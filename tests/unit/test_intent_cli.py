"""CLI -> real B pipeline -> three mock calls -> private validated files."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from review_bot.application.architecture_review import MockModelClient
from review_bot.runtime import cli
from tests.fixtures.intent_review import make_task, responses


def fixed_input(tmp_path: Path, language: str = "python") -> Path:
    path = tmp_path / "fixed-input.json"
    path.write_text(json.dumps(asdict(make_task(language))), encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.mark.parametrize("language", ["python", "javascript", "java"])
def test_cli_saves_three_stage_results_and_safe_draft_without_analysis_or_posting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, language: str,
) -> None:
    client = MockModelClient(responses(), input_tokens=100)
    monkeypatch.setattr(cli, "build_review_client", lambda profile: client)
    monkeypatch.setattr(cli, "PmdRunner", lambda: pytest.fail("B must not initialize PMD"))
    monkeypatch.setattr(cli, "GitRepositoryReader", lambda *_: pytest.fail("Fixed evaluation must not read a repository"))
    source = fixed_input(tmp_path, language)
    output = tmp_path / "private-output"
    code = cli.main(["intent", "--input", str(source), "--output", str(output), "--model-profile", "ollama-local-128k"])
    assert code == 0
    assert len(client.requests) == 3
    expected = {"input.json", "goals.json", "observations.json", "result.json", "scope.json", "usage.json", "validation.json", "comment.md"}
    assert {path.name for path in (output / "B").iterdir()} == expected
    result = json.loads((output / "B/result.json").read_text())
    assert (result["review_type"], result["status"], result["reason"]) == ("B", "completed", "validated")
    assert json.loads((output / "B/goals.json").read_text())["declared_goals"][0]["goal_id"] == "G1"
    assert json.loads((output / "B/observations.json").read_text())["implementation_observations"][0]["observation_id"] == "O1"
    assert json.loads((output / "B/usage.json").read_text())["total"]["calls"] == 3
    assert json.loads((output / "B/input.json").read_text()) == json.loads(source.read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["publication"]["status"] == "not_attempted"
    assert manifest["publication"]["reason"] == "artifact_only"
    assert not (output / "A").exists()
    assert output.stat().st_mode & 0o077 == 0
    assert (output / "B").stat().st_mode & 0o077 == 0
    assert all(path.stat().st_mode & 0o077 == 0 for path in (output / "B").iterdir())
    draft = (output / "B/comment.md").read_text()
    assert "not executed" in draft
    assert "Mock model" in draft
    assert "outside" in draft
    assert "example/project/blob/" in draft
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Mutual exclusion" not in captured.out


def test_cli_unconfigured_creates_not_run_files_without_model_calls(tmp_path: Path, monkeypatch) -> None:
    def no_profile(profile):
        assert profile is None
        return None
    monkeypatch.setattr(cli, "build_review_client", no_profile)
    output = tmp_path / "not-run"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(output)]) == 0
    result = json.loads((output / "B/result.json").read_text())
    assert result["reason"] == "model_profile_unset"
    assert result["usage"]["calls"] == 0
    assert json.loads((output / "B/goals.json").read_text()) is None
    assert "model_profile_unset" in (output / "B/comment.md").read_text()


def test_cli_tampered_hash_fails_before_model_or_output(tmp_path: Path, monkeypatch, capsys) -> None:
    source = fixed_input(tmp_path)
    data = json.loads(source.read_text())
    data["head"]["files"][0]["content"] += "# tampered"
    source.write_text(json.dumps(data))
    monkeypatch.setattr(cli, "build_review_client", lambda _: pytest.fail("Invalid input must not construct a client"))
    output = tmp_path / "invalid"
    assert cli.main(["intent", "--input", str(source), "--output", str(output)]) == 1
    assert not output.exists()
    assert "invalid_intent_input" in capsys.readouterr().err


@pytest.mark.parametrize("unsafe", ["public_output", "linked_parent", "existing_result", "public_input", "linked_input", "hardlinked_input", "existing_manifest"])
def test_cli_unsafe_paths_fail_before_any_calls_and_preserve_files(tmp_path: Path, monkeypatch, unsafe: str) -> None:
    source = fixed_input(tmp_path)
    output = tmp_path / "output"
    if unsafe == "public_output":
        output.mkdir(mode=0o755)
        output.chmod(0o755)
    elif unsafe == "linked_parent":
        actual = tmp_path / "actual"
        actual.mkdir(mode=0o700)
        link = tmp_path / "link"
        link.symlink_to(actual, target_is_directory=True)
        output = link / "output"
    elif unsafe == "existing_result":
        (output / "B").mkdir(mode=0o700, parents=True)
        output.chmod(0o700)
        (output / "B/result.json").write_text("preserve me")
    elif unsafe == "public_input":
        source.chmod(0o644)
    elif unsafe == "linked_input":
        link = tmp_path / "input-link"
        link.symlink_to(source)
        source = link
    elif unsafe == "hardlinked_input":
        (tmp_path / "input-hardlink").hardlink_to(source)
    elif unsafe == "existing_manifest":
        output.mkdir(mode=0o700)
        (output / "manifest.json").write_text("preserve manifest")
    monkeypatch.setattr(cli, "build_review_client", lambda _: pytest.fail("Unsafe paths must be rejected before model setup"))
    assert cli.main(["intent", "--input", str(source), "--output", str(output)]) == 1
    if unsafe == "existing_result":
        assert (output / "B/result.json").read_text() == "preserve me"
    if unsafe == "existing_manifest":
        assert (output / "manifest.json").read_text() == "preserve manifest"
        assert not (output / "B").exists()


@pytest.mark.parametrize("failure_stage", [None, "observe_changes"])
def test_cli_existing_local_profile_records_native_preflight_usage(tmp_path: Path, monkeypatch, failure_stage) -> None:
    from review_bot.application.declared_intent_review import STAGES
    from tests.fixtures.intent_review import stage_bodies
    from tests.unit.test_ollama_client import chat_response, model_info

    requests = []
    client_type = httpx.Client

    def handler(request):
        requests.append(request)
        assert request.url.host == "127.0.0.1"
        payload = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json=model_info(payload["model"]))
        assert request.url.path == "/api/chat"
        # Zero-measurement contract: no count-only (num_predict==1) call exists.
        # Every chat is a generation; the byte gate proves the budget pre-send.
        assert payload["options"]["num_predict"] != 1
        stage = json.loads(payload["messages"][1]["content"])["stage"]
        if stage == failure_stage:
            raise httpx.ReadTimeout("private transport details", request=request)
        return httpx.Response(200, json=chat_response(
            json.dumps(stage_bodies()[STAGES.index(stage)]),
            prompt_eval_count=100, eval_count=50,
            done_reason="stop",
        ))

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return client_type(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    output = tmp_path / "native-profile"
    if failure_stage:
        assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(output),
            "--model-profile", "ollama-local-128k"]) == 1
        result = json.loads((output / "B/result.json").read_text())
        assert [request.url.path for request in requests] == ["/api/show", *["/api/chat"] * 2]
        assert result["reason"] == "model_timeout"
        assert result["usage"]["calls"] == 2
        assert result["usage"]["token_measurements"] == 0
        assert result["usage"]["metering_input_tokens"] == 0
        assert result["usage"]["metering_output_tokens"] == 0
        assert result["usage"]["input_tokens"] is None
        assert result["usage"]["known_input_tokens"] == 100
        assert result["usage"]["known_output_tokens"] == 50
        assert result["stages"][1]["status"] == "failed"
        assert result["stages"][2]["status"] == "not_attempted"
        assert result["goals"] and not result["observations"]
        assert result["findings"] == []
        assert "private transport details" not in json.dumps(result)
        return
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(output),
        "--model-profile", "ollama-local-128k"]) == 0
    result = json.loads((output / "B/result.json").read_text())
    assert [request.url.path for request in requests] == ["/api/show", *["/api/chat"] * 3]
    assert (result["status"], result["reason"]) == ("completed", "validated")
    assert result["usage"]["calls"] == 3
    assert result["usage"]["token_measurements"] == 0
    assert result["usage"]["metering_input_tokens"] == 0
    assert result["usage"]["metering_output_tokens"] == 0
    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 150
    assert all(stage["token_measurement"] == "not_attempted" for stage in result["stages"])
    assert all(stage["token_measurement_model_calls"] == 0 for stage in result["stages"])
    assert all(stage["measured_input_tokens"] is None for stage in result["stages"])


def test_cli_invalid_final_response_retains_earlier_files_and_usage(tmp_path: Path, monkeypatch) -> None:
    from tests.fixtures.intent_review import stage_bodies

    bodies = stage_bodies()
    bodies[2]["findings"][0]["goal_ids"] = ["invented"]
    client = MockModelClient(responses(bodies), input_tokens=100)
    monkeypatch.setattr(cli, "build_review_client", lambda _: client)
    output = tmp_path / "failed-assessment"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(output)]) == 1
    result = json.loads((output / "B/result.json").read_text())
    assert result["reason"] == "invalid_model_response"
    assert result["usage"]["calls"] == 3
    assert result["usage"]["input_tokens"] == 300
    assert result["findings"] == []
    assert json.loads((output / "B/goals.json").read_text())["declared_goals"]
    assert json.loads((output / "B/observations.json").read_text())["implementation_observations"]
    assert "No validated final assessment" in (output / "B/comment.md").read_text()
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_cli_no_explicit_profile_does_not_inherit_ambient_model_opt_in(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REVIEW_MODEL_PROFILE", "ollama-local-128k")
    monkeypatch.setattr(httpx, "Client", lambda **_: pytest.fail("No profile must not make network calls"))
    output = tmp_path / "ambient-not-run"
    assert cli.main(["intent", "--input", str(fixed_input(tmp_path)), "--output", str(output)]) == 0
    result = json.loads((output / "B/result.json").read_text())
    assert result["reason"] == "model_profile_unset"
    assert result["usage"]["calls"] == result["usage"]["model_calls"] == 0