"""Existing authenticated API -> durable queue -> worker -> private trial result."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import yaml
import pytest
from fastapi.testclient import TestClient

from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import load_review_config_from_str
from review_bot.runtime.intent_api import create_intent_app
from review_bot.runtime.worker import WorkerLoop
from tests.fixtures.intent_perspectives import make_task
from tests.fixtures.partition_review import PartitionModelClient


@pytest.mark.parametrize("repair_gap", [False, True])
def test_fixed_api_worker_runs_partition_trial_and_returns_existing_result_endpoint(tmp_path, valid_config_yaml, monkeypatch, repair_gap):
    from dataclasses import replace
    monkeypatch.setenv("REVIEW_BOT_API_TOKEN", "synthetic-partition-token")
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="offline", available_modes=["intent"], selector_uses_model=False)
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True, "max_total_tokens": 524288,
        "max_calls": 64, "input_target_tokens": 10000}
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    previous = os.umask(0o077)
    store = SQLiteJobStore(root / "jobs.sqlite")
    def review_gap(payload, body):
        if payload["stage"] == "global_review":
            body["missing_context"] = [] if "recheck" in payload else ["Need src/value.py head implementation."]
            if "recheck" in payload:
                cite = payload["requested_originals"][0]["blocks"][0]["reference_id"]
                body["context_updates"] = {key: {"state": "resolved", "reason": "Synthetic explicit source-backed resolution.",
                    "citations": [{"reference_id": cite}]} for key in payload["context_items"] if key.startswith("G")}
        return body
    def select_source(payload, body):
        if payload["stage"] == "plan_fixed_sources":
            found = next((row["evidence_id"] for row in payload["state"]["candidate_inventory"]
                if row["path"] == "src/value.py" and row["version"] == "head"), None)
            key = "source:" + found if found else "no_suitable_source"
            body["choice"] = next(row["label"] for row in payload["options"] if row["key"] == key)
        return body
    review = PartitionModelClient(transform=review_gap if repair_gap else None)
    tev = PartitionModelClient(name="tev", group_all=not repair_gap, transform=select_source if repair_gap else None)
    if repair_gap:
        review.profile = replace(review.profile, max_input_tokens=350)
        review.counter = lambda r: 200 + 100 * len(json.loads(r.user_content)["observations"]) if json.loads(r.user_content)["stage"] == "final_review" else 100

    def forbidden(*args):
        raise AssertionError("fixed trial must not collect GitHub, run PMD, publish, or execute a target")

    headers = {"Authorization": "Bearer synthetic-partition-token"}
    try:
        app = create_intent_app(config, store, root / "artifacts")
        with TestClient(app) as client:
            submitted = client.post("/reviews/intent", json=asdict(make_task()), headers=headers)
            assert submitted.status_code == 202, submitted.text
            urls = submitted.json()
            worker = WorkerLoop(config, store, model_client=review, planning_client=tev,
                github_client_factory=forbidden, repo_reader_factory=forbidden, analyzer_factory=forbidden,
                artifact_root=root / "artifacts")
            assert worker.run_once() is True
            assert client.get(urls["status_url"], headers=headers).json()["status"] == "succeeded"
            response = client.get(urls["result_url"], headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["schema_version"] == 12 and result["status"] == "completed"
            assert len(result["perspectives"]) == 8
            assert result["partition"]["scope_selection"]["decision"] == "apply"
            assert result["origin"]["publication"] == "not_attempted"
            assert review.requests and tev.requests
            assert all(json.loads(r.user_content)["stage"] != "draft_review" for r in review.requests)
            if repair_gap:
                attempts = result["partition"]["global_review"]["attempts"]
                assert len(attempts) == 2 and attempts[0]["source_plan"]["complete"]
                assert attempts[0]["response"]["source_requests"] == [] and attempts[0]["response"]["missing_context"]
                assert attempts[1]["previous"]["prior_response"] == attempts[0]["response"]
                assert any(d["stage"] == "plan_fixed_sources" for d in result["partition"]["decisions"])
                assert all(row["state"] == "resolved" for key, row in result["partition"]["final"]["context_updates"].items() if key.startswith("G"))
        saved = root / "artifacts/fixed" / urls["review_id"] / "attempt-1"
        manifest = json.loads((saved / "manifest.json").read_text())
        assert manifest["status"] == "succeeded"
        assert manifest["publication"]["status"] == "not_attempted"
        assert (saved / "B/partition.json").is_file()
        assert all(p.stat().st_mode & 0o777 == 0o600 for p in saved.rglob("*") if p.is_file())
    finally:
        store.close()
        os.umask(previous)


def test_partition_trial_requires_private_mode_and_explicit_resource_budgets(valid_config_yaml):
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="offline", available_modes=["intent"])
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True}
    import pytest
    with pytest.raises(ValueError, match="max_total_tokens"):
        load_review_config_from_str(yaml.safe_dump(raw))

    raw["partition_evaluation"].update(max_total_tokens=524288, max_calls=64)
    raw["publish"].update(mode="comment", b_comments_enabled=True)
    with pytest.raises(ValueError, match="artifact_only"):
        load_review_config_from_str(yaml.safe_dump(raw))


@pytest.mark.parametrize("failure_stage", ["source_recheck", "global_review"])
def test_failed_source_recheck_is_persisted_and_readable_through_the_authenticated_result_api(tmp_path, valid_config_yaml, monkeypatch, failure_stage):
    from dataclasses import replace
    from review_bot.application.architecture_review import PermanentModelError

    monkeypatch.setenv("REVIEW_BOT_API_TOKEN", "synthetic-failed-partition")
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="offline", available_modes=["intent"], selector_uses_model=False)
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True, "max_total_tokens": 524288, "max_calls": 64}
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    previous = os.umask(0o077)
    store = SQLiteJobStore(root / "jobs.sqlite")

    def fail_recheck(payload, body):
        if failure_stage == "global_review":
            if payload["stage"] == "global_review":
                next(iter(body["observation_dispositions"].values()))["citations"] = [{}]
            return body
        if payload["stage"] == "final_review":
            if "source_recheck" in payload:
                error = PermanentModelError("Invalid or truncated model output")
                error.delivery = "sent"
                error.usage = {"input_tokens": 100, "output_tokens": 512}
                raise error
            original = next(row for row in payload["fixed_original_inventory"] if row["version"] == "head")
            body["source_requests"] = [{"evidence_id": original["evidence_id"], "reason": "Compare this original."}]
        return body

    review = PartitionModelClient(transform=fail_recheck)
    if failure_stage == "global_review":
        review.profile = replace(review.profile, max_input_tokens=350)
        review.counter = lambda r: 200 + 100 * len(json.loads(r.user_content)["observations"]) if json.loads(r.user_content)["stage"] == "final_review" else 100
    headers = {"Authorization": "Bearer synthetic-failed-partition"}
    try:
        with TestClient(create_intent_app(config, store, root / "artifacts")) as api:
            submitted = api.post("/reviews/intent", json=asdict(make_task()), headers=headers)
            assert submitted.status_code == 202
            urls = submitted.json()
            worker = WorkerLoop(config, store, model_client=review,
                planning_client=PartitionModelClient(name="tev", group_all=failure_stage != "global_review"), artifact_root=root / "artifacts")
            assert worker.run_once() is False
            status = api.get(urls["status_url"], headers=headers).json()
            expected_reason = "invalid_global_response" if failure_stage == "global_review" else "model_error"
            assert status["status"] == "failed" and status["error"] == expected_reason
            response = api.get(urls["result_url"], headers=headers)
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "failed" and result["reason"] == expected_reason
            assert not result["summary"] and not result["perspectives"] and not result["findings"]
            assert result["partition"]["review_observations"]
            if failure_stage == "global_review":
                assert result["partition"]["global_review"]["attempts"][0]["response"]
                assert result["partition"]["integration_final"] and "final" not in result["partition"]
            else:
                assert result["partition"]["source_recheck_requests"][0]["response"]
            assert result["usage"]["usage_complete"] and result["usage"]["unknown_usage_attempts"] == 0
    finally:
        store.close()
        os.umask(previous)


def test_cli_partition_option_reaches_same_executor_and_private_artifacts(tmp_path, monkeypatch, capsys):
    from review_bot.runtime import cli
    task = make_task()
    path = tmp_path / "input.json"
    path.write_text(json.dumps(asdict(task), ensure_ascii=False))
    path.chmod(0o600)
    review, tev = PartitionModelClient(), PartitionModelClient(name="tev", group_all=True)
    monkeypatch.setattr(cli, "build_review_client", lambda *args: review)
    monkeypatch.setattr(cli, "build_planning_client", lambda *args: tev)
    output = tmp_path / "output"
    assert cli.main(["intent", "--input", str(path), "--output", str(output),
        "--pipeline", "perspectives-v1", "--partition-evaluation",
        "--max-total-tokens", "524288", "--max-calls", "64"]) == 0
    assert "completed/validated" in capsys.readouterr().out
    result = json.loads((output / "B/result.json").read_text())
    assert result["schema_version"] == 12 and len(result["perspectives"]) == 8
    assert result["origin"]["publication"] == "not_attempted"
    assert tev.requests and review.requests


def test_cli_measure_only_uses_native_counter_without_generating_reviews(tmp_path, monkeypatch, capsys):
    from review_bot.runtime import cli
    path = tmp_path / "input.json"
    path.write_text(json.dumps(asdict(make_task()), ensure_ascii=False))
    path.chmod(0o600)
    review, tev = PartitionModelClient(), PartitionModelClient(name="tev")
    monkeypatch.setattr(cli, "build_review_client", lambda *args: review)
    monkeypatch.setattr(cli, "build_planning_client", lambda *args: tev)
    output = tmp_path / "measured"
    assert cli.main(["intent", "--input", str(path), "--output", str(output), "--pipeline", "perspectives-v1",
        "--partition-evaluation", "--measure-only", "--max-total-tokens", "131072", "--max-calls", "1"]) == 0
    assert "measured/native_input_prepared" in capsys.readouterr().out
    assert not review.requests and not tev.requests
    assert review.count_requests and tev.count_requests
    result = json.loads((output / "B/result.json").read_text())
    assert result["status"] == "measured" and result["usage"]["calls"] == 0
    assert json.loads((output / "manifest.json").read_text())["status"] == "succeeded"


def test_github_worker_connects_architecture_analysis_to_partition_executor(tmp_path, valid_config_yaml, worker_pipeline_ports):
    from review_bot.adapters.github.client import MockGitHubClient
    from review_bot.domain.envelope_v2 import PrRequestRecord
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(available_modes=["architecture"], selector_uses_model=False)
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True, "max_total_tokens": 524288, "max_calls": 64}
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    previous = os.umask(0o077)
    store = SQLiteJobStore(root / "jobs.sqlite")
    github = MockGitHubClient()
    github.set_pull_request(7, repository="org/repo")
    github.pull_requests[7]["base"]["repo"] = {"id": 99, "full_name": "org/repo"}
    github.pull_requests[7].update(title="Synthetic architecture change", body="Synthetic PR", draft=False)
    review, tev = PartitionModelClient(), PartitionModelClient(name="tev", group_all=True)
    try:
        store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        worker = WorkerLoop(config, store, model_client=review, planning_client=tev,
            github_client=github, repo_reader=worker_pipeline_ports["repo_reader"],
            analyzer=worker_pipeline_ports["analyzer"], artifact_root=root / "artifacts")
        assert worker.run_once() is True, store._get_connection().execute("SELECT status,error FROM runs").fetchall()
        results = list((root / "artifacts").rglob("A/result.json"))
        assert len(results) == 1
        result = json.loads(results[0].read_text())
        assert result["schema_version"] == 12 and result["review_type"] == "A"
        assert result["status"] == "completed", result["reason"]
        assert worker_pipeline_ports["analyzer"].run_analysis.call_count == 2
        worker_pipeline_ports["repo_reader"].collect_code_snapshot.assert_not_called()
        assert tev.requests and review.requests
        assert not github.comments
    finally:
        store.close()
        os.umask(previous)


def test_worker_selects_modes_after_shared_plan_and_reuses_it_for_a_and_b(tmp_path, valid_config_yaml, worker_pipeline_ports):
    from review_bot.adapters.github.client import MockGitHubClient
    from review_bot.domain.envelope_v2 import PrRequestRecord
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(available_modes=["architecture", "intent"], selector_uses_model=True)
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True, "max_total_tokens": 1048576, "max_calls": 128}
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    previous = os.umask(0o077)
    store = SQLiteJobStore(root / "jobs.sqlite")
    github = MockGitHubClient()
    github.set_pull_request(7, repository="org/repo")
    github.pull_requests[7]["base"]["repo"] = {"id": 99, "full_name": "org/repo"}
    github.pull_requests[7].update(title="Synthetic shared review", body="Synthetic PR", draft=False)
    review, tev = PartitionModelClient(), PartitionModelClient(name="tev", group_all=True)
    class ForbiddenSelector:
        def select_modes(self, *args):
            raise AssertionError("legacy selector must not run before partition planning")
    try:
        store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        worker = WorkerLoop(config, store, model_client=review, planning_client=tev,
            github_client=github, repo_reader=worker_pipeline_ports["repo_reader"],
            analyzer=worker_pipeline_ports["analyzer"], artifact_root=root / "artifacts", mode_selector=ForbiddenSelector())
        assert worker.run_once() is True
        run_dir = next((root / "artifacts/runs/99/7").iterdir())
        plan = json.loads((run_dir / "execution-plan.json").read_text())
        assert plan["status"] == "planned" and plan["partition"]["selected_modes"] == ["architecture", "intent"]
        plan_id = plan["partition"]["plan_id"]
        results = [json.loads((run_dir / name / "result.json").read_text()) for name in ("A", "B")]
        assert all(r["partition"]["shared_plan_id"] == plan_id for r in results)
        assert all(r["partition"]["plan_reuse"]["reused_group_ids"] for r in results)
        assert results[0]["partition"]["plan_reuse"]["amendments"]  # New PMD facts only.
        assert not results[1]["partition"]["plan_reuse"]["amendments"]
        selections = [r for r in tev.requests if json.loads(r.user_content)["stage"] == "select_review_scope"]
        assert len(selections) == 1
        assert json.loads(selections[0].user_content)["state"]["available_modes"] == ["architecture", "intent"]
        selection = json.loads((run_dir / "selection.json").read_text())
        assert selection["policy_version"].startswith("partitioned-plan/v4-closed:")
        usage = json.loads((run_dir / "usage.json").read_text())
        assert usage["calls"] == len(tev.requests) + len(review.requests)
        assert sum(r["usage"]["calls"] for r in results) + plan["usage"]["calls"] == usage["calls"]
        assert not github.comments
    finally:
        store.close()
        os.umask(previous)


@pytest.mark.parametrize("use_bare", [False, True])
def test_offline_a_trial_uses_rootless_factory_and_private_plan_artifacts(tmp_path, valid_config_yaml, synthetic_repo, monkeypatch, use_bare):
    from review_bot.runtime import cli, service
    from tests.unit.test_delta_adversarial import make_pmd_violation, make_test_run
    from unittest.mock import Mock
    info = synthetic_repo.setup_standard_feature_pr()
    repo = synthetic_repo.repo_dir
    if use_bare:
        import subprocess
        bare = tmp_path / "cache.git"
        subprocess.run(["git", "clone", "--bare", "--no-hardlinks", str(repo), str(bare)], check=True, capture_output=True)
        repo = bare
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(available_modes=["architecture"])
    raw["publish"].update(mode="artifact_only", b_comments_enabled=False)
    raw["partition_evaluation"] = {"enabled": True, "max_total_tokens": 1048576, "max_calls": 128}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    state = tmp_path / "state"
    state.mkdir(mode=0o700)
    monkeypatch.setenv("REVIEW_BOT_STATE_DIR", str(state))
    def analyze(sha, source, *args):
        paths = tuple(sorted(str(p.relative_to(source)) for p in source.rglob("*.java")))
        violations = (make_pmd_violation(path=paths[0], begin_line=1, end_line=1),) if sha == info["head_sha"] else ()
        return make_test_run(sha, source_files=paths, pmd_violations=violations)
    analyzer = Mock()
    analyzer.run_analysis.side_effect = analyze
    factory = Mock(return_value=analyzer)
    monkeypatch.setattr(service, "_architecture_analyzer", factory)
    monkeypatch.setattr(cli, "PmdRunner", Mock(side_effect=AssertionError("No host analyzer fallback")))
    monkeypatch.setattr(cli, "build_review_client", lambda *args: PartitionModelClient())
    monkeypatch.setattr(cli, "build_planning_client", lambda *args: PartitionModelClient(group_all=True))
    output = tmp_path / "output"
    assert cli.main(["offline", "--repo", str(repo), "--base", info["target_base_sha"],
        "--head", info["head_sha"], "--config", str(config_path), "--output", str(output)]) == 0
    factory.assert_called_once()
    assert (output / "execution-plan.json").is_file()
    assert (output / "A/result.json").is_file()
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in output.rglob("*") if p.is_file())
