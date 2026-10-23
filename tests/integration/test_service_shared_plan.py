"""Ordinary GitHub App work is selected by TEV before any eligible PMD analysis."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

import pytest
import yaml

from review_bot.adapters.github.client import MockGitHubClient
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import load_review_config_from_str
from review_bot.domain.envelope_v2 import PrRequestRecord
from review_bot.runtime.worker import WorkerLoop
from tests.fixtures.partition_review import PartitionModelClient


@contextmanager
def _app_worker(tmp_path, valid_config_yaml, ports, selected, *, comments=False, settings_factory=None,
    fail_perspective=None):
    raw = yaml.safe_load(valid_config_yaml)
    raw["execution"].update(mode="github_app", available_modes=["architecture", "intent"],
        selector_uses_model=False)
    raw["intent"] = {"pipeline": "perspectives-v1"}
    raw["publish"].update(mode="comment" if comments else "artifact_only", b_comments_enabled=comments)
    raw.pop("partition_evaluation", None)
    config, _ = load_review_config_from_str(yaml.safe_dump(raw))
    assert not config.partition_evaluation.enabled
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    previous_umask = os.umask(0o077)
    store = SQLiteJobStore(root / "jobs.sqlite")
    github = MockGitHubClient()
    github.bot_login = "synthetic-review[bot]"
    github.set_pull_request(7, repository="org/repo")
    github.pull_requests[7]["base"]["repo"] = {"id": 99, "full_name": "org/repo"}
    github.pull_requests[7].update(title="Synthetic Java change", body="Explain the changed implementation.", draft=False)
    original_create = github.create_comment

    def bot_create(*args, **kwargs):
        record = original_create(*args, **kwargs)
        record["user"] = {"login": github.bot_login, "type": "Bot"}
        return record

    github.create_comment = bot_create
    events = []

    def choose_global_scope(payload, body):
        events.append("tev:" + payload["stage"])
        if payload["stage"] == "select_review_scope":
            assert payload["state"]["field"] == "global_modes"
            assert payload["state"]["available_modes"] == ["architecture", "intent"]
            if selected == "failure":
                body["choice"] = "not-a-declared-choice"
            else:
                key = "review:" + (selected or "none")
                body["choice"] = next(option["label"] for option in payload["options"] if option["key"] == key)
        elif payload["stage"] == "select_perspective" and payload["state"]["field"] == fail_perspective:
            body["choice"] = "not-a-declared-choice"
        return body

    analyze = ports["analyzer"].run_analysis.side_effect

    def recorded_analysis(*args, **kwargs):
        events.append("analyze")
        return analyze(*args, **kwargs)

    ports["analyzer"].run_analysis.side_effect = recorded_analysis
    review = PartitionModelClient()
    tev = PartitionModelClient(name="tev", group_all=True, transform=choose_global_scope)

    class ForbiddenLegacySelector:
        def select_modes(self, *args):
            raise AssertionError("shared global TEV selection must own normal App routing")

    try:
        store.enqueue_request(PrRequestRecord(44, 99, 7, 1))
        worker = WorkerLoop(config, store, model_client=review, planning_client=tev,
            github_client=github, repo_reader=ports["repo_reader"], analyzer=ports["analyzer"],
            artifact_root=root / "artifacts", mode_selector=ForbiddenLegacySelector(),
            repository_settings_factory=settings_factory)
        yield config, worker, github, store, root, events, review, tev
    finally:
        store.close()
        os.umask(previous_umask)


@pytest.mark.parametrize("selected,analysis_calls,run_status", [
    ("", 0, "skipped"),
    ("intent", 0, "succeeded"),
    ("architecture", 2, "succeeded"),
    ("failure", 0, "failed"),
])
def test_normal_app_selects_global_modes_before_eligible_java_analysis(
    tmp_path, valid_config_yaml, worker_pipeline_ports, selected, analysis_calls, run_status,
):
    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, selected) as context:
        _, worker, github, store, root, events, review, tev = context
        outcome = worker.run_once()
        row = store._get_connection().execute("SELECT status,error FROM runs").fetchone()
        assert row[0] == run_status, tuple(row)
        assert outcome is (selected != "failure")
        assert worker_pipeline_ports["analyzer"].run_analysis.call_count == analysis_calls
        worker_pipeline_ports["repo_reader"].collect_code_snapshot.assert_not_called()
        if selected == "failure":
            assert 1 <= events.count("tev:select_review_scope") <= 2
        else:
            assert events.count("tev:select_review_scope") == 1
        assert events.index("tev:observe_partition") < events.index("tev:select_review_scope")
        if analysis_calls:
            assert events.index("tev:select_review_scope") < events.index("analyze")
        run_dir = next((root / "artifacts/runs/99/7").iterdir())
        plan = json.loads((run_dir / "execution-plan.json").read_text())
        if selected == "failure":
            assert plan["status"] == "failed"
            assert not review.requests
        else:
            assert plan["status"] == "planned"
            assert plan["partition"]["selected_modes"] == ([selected] if selected else [])
        assert tev.requests
        assert not github.posted_comments and not github.updated_comments


def test_normal_app_posts_one_intent_pr_comment_using_bot_identity(
    tmp_path, valid_config_yaml, worker_pipeline_ports,
):
    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, "intent", comments=True) as context:
        config, worker, github, store, root, _, review, tev = context
        assert worker.run_once() is True, store._get_connection().execute("SELECT status,error FROM runs").fetchall()
        assert len(github.posted_comments) == 1
        assert not github.updated_comments
        comment = github.posted_comments[0]
        assert comment["pr_number"] == 7
        assert comment["user"] == {"login": "synthetic-review[bot]", "type": "Bot"}
        assert comment["body"].startswith(config.publish.b_marker)
        assert "<details>" in comment["body"]
        assert "[E" not in comment["body"] and "Q_" not in comment["body"]
        worker_pipeline_ports["analyzer"].run_analysis.assert_not_called()
        result_path = next((root / "artifacts").rglob("B/result.json"))
        result = json.loads(result_path.read_text())
        assert result["status"] == "completed" and len(result["perspectives"]) == 8
        assert review.requests and tev.requests


def test_settings_presentation_update_reuses_original_plan_and_review_then_criteria_reruns_review(
    tmp_path, valid_config_yaml, worker_pipeline_ports,
):
    from review_bot.application.repository_review_settings import (
        SettingsResolution, parse_repository_review_settings,
    )

    def settings(sha, *, title, template, instructions=""):
        value = {"schema_version": 1, "presentation": {"title_b": title},
            "comment_templates": {"intent": template}}
        if instructions:
            value["review_instructions"] = {"intent": instructions}
        bundle = parse_repository_review_settings(json.dumps(value).encode("utf-8"), sha)
        return SettingsResolution(bundle, attempted_sha=sha)

    current = [settings("a" * 40, title="처음 팀 검토",
        template="{{summary}}\n{{llm_review}}\n{{perspectives}}")]
    captures = []

    def settings_factory(run, pr_data, github, *, before_request):
        assert before_request()
        assert (run.installation_id, run.repository_id) == (44, 99)
        assert pr_data["base"]["repo"]["id"] == 99
        captures.append(current[0])
        return current[0]

    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, "intent", comments=True,
        settings_factory=settings_factory) as context:
        _, worker, github, store, root, events, review, tev = context
        assert worker.run_once() is True
        first_counts = (len(review.requests), len(tev.requests))
        assert all(first_counts)
        first_comment = github.posted_comments[0]
        comment_id, first_body = first_comment["id"], first_comment["body"]
        assert "처음 팀 검토" in first_body
        runs = root / "artifacts/runs/99/7"
        first_run = next(runs.iterdir())
        original_result = json.loads((first_run / "B/result.json").read_text())
        original_plan = json.loads((first_run / "execution-plan.json").read_text())

        current[0] = settings("b" * 40, title="새로운 팀 검토",
            template="표현 변경\n{{summary}}\n{{perspectives}}\n{{llm_review}}")
        assert current[0].bundle.generation_hash == captures[0].bundle.generation_hash
        assert current[0].bundle.presentation_hash != captures[0].bundle.presentation_hash
        store.enqueue_request(PrRequestRecord(44, 99, 7, 2))
        assert worker.run_once() is True, store._get_connection().execute("SELECT status,error FROM runs").fetchall()
        assert (len(review.requests), len(tev.requests)) == first_counts
        assert len(github.posted_comments) == 1 and len(github.updated_comments) == 1
        updated = github.updated_comments[0]
        assert updated["id"] == comment_id
        assert "새로운 팀 검토" in updated["body"] and "표현 변경" in updated["body"]
        assert updated["body"] != first_body
        second_run = next(directory for directory in runs.iterdir() if directory != first_run)
        assert json.loads((second_run / "B/result.json").read_text()) == original_result
        assert json.loads((second_run / "execution-plan.json").read_text()) == original_plan
        assert json.loads((second_run / "planning-reuse.json").read_text())["new_model_calls"] == 0

        instructions = "변경 목적과 코드의 유지보수 영향을 구체적으로 설명하세요."
        current[0] = settings("c" * 40, title="새로운 팀 검토",
            template="표현 변경\n{{summary}}\n{{perspectives}}\n{{llm_review}}", instructions=instructions)
        assert current[0].bundle.generation_hash != captures[1].bundle.generation_hash
        assert current[0].bundle.presentation_hash == captures[1].bundle.presentation_hash
        store.enqueue_request(PrRequestRecord(44, 99, 7, 3))
        assert worker.run_once() is True, store._get_connection().execute("SELECT status,error FROM runs").fetchall()
        assert len(review.requests) > first_counts[0]
        assert any(instructions in request.system_prompt for request in review.requests[first_counts[0]:])
        assert events.count("tev:select_review_scope") == 1
        assert len(github.posted_comments) == 1 and len(github.updated_comments) == 2
        assert all(comment["id"] == comment_id for comment in github.updated_comments)
        third_run = next(directory for directory in runs.iterdir() if directory not in (first_run, second_run))
        assert json.loads((third_run / "execution-plan.json").read_text()) == original_plan
        assert len(captures) == 3
        worker_pipeline_ports["analyzer"].run_analysis.assert_not_called()


def test_real_default_branch_presentation_commit_reuses_review_with_current_base_fence(
    tmp_path, valid_config_yaml, synthetic_repo,
):
    from unittest.mock import Mock
    from review_bot.adapters.git import GitRepositoryReader
    from review_bot.adapters.repository_review_settings import FileRepositoryReviewSettingsStore
    from review_bot.application.repository_review_settings import (
        RepositoryReviewSettingsResolver, SETTINGS_PATH,
    )

    path = "src/main/java/demo/Service.java"
    synthetic_repo.write_file("build.gradle", "plugins { id 'java' }\n")
    synthetic_repo.write_file(path, "package demo; class Service { int value() { return 0; } }\n")
    synthetic_repo.write_file(SETTINGS_PATH, json.dumps({"schema_version": 1,
        "presentation": {"title_b": "처음 팀 검토"}}, ensure_ascii=False))
    base = synthetic_repo.commit("Initial code and review presentation")
    synthetic_repo.create_branch("feature/cache", base)
    synthetic_repo.checkout("feature/cache")
    synthetic_repo.write_file(path, "package demo; class Service { int value() { return 1; } }\n")
    head = synthetic_repo.commit("Change the reviewed method")
    synthetic_repo.checkout("main")
    reader = GitRepositoryReader(synthetic_repo.repo_dir)
    resolver = RepositoryReviewSettingsResolver(FileRepositoryReviewSettingsStore(tmp_path / "settings-state"))

    def settings_factory(run, pr_data, github, *, before_request):
        assert before_request()
        return resolver.resolve(run.installation_id, run.repository_id,
            lambda: (reader.get_commit("refs/heads/main").sha, reader))

    analyzer = Mock()
    ports = {"repo_reader": reader, "analyzer": analyzer}
    with _app_worker(tmp_path, valid_config_yaml, ports, "intent", comments=True,
        settings_factory=settings_factory) as context:
        _, worker, github, store, root, _, review, tev = context
        github.pull_requests[7]["base"]["sha"] = base
        github.pull_requests[7]["head"].update(sha=head, ref="feature/cache")
        assert worker.run_once() is True
        counts = (len(review.requests), len(tev.requests))
        assert all(counts)
        runs = root / "artifacts/runs/99/7"
        first_run = next(runs.iterdir())
        original_input = json.loads((first_run / "B/input.json").read_text())
        original_result = json.loads((first_run / "B/result.json").read_text())
        original_plan = json.loads((first_run / "execution-plan.json").read_text())
        original_changes = reader.get_changed_files(base, head)
        comment_id = github.posted_comments[0]["id"]

        synthetic_repo.write_file(SETTINGS_PATH, json.dumps({"schema_version": 1,
            "presentation": {"title_b": "변경된 팀 검토"},
            "comment_templates": {"intent": "{{summary}}\n{{perspectives}}\n{{llm_review}}"}}, ensure_ascii=False))
        advanced = synthetic_repo.commit("Update presentation on the default branch only")
        assert advanced != base and reader.get_commit("feature/cache").sha == head
        assert reader.resolve_merge_base(advanced, head).comparison_base_sha == base
        assert reader.get_changed_files(base, head) == original_changes
        github.pull_requests[7]["base"]["sha"] = advanced
        store.enqueue_request(PrRequestRecord(44, 99, 7, 2))
        assert worker.run_once() is True, store._get_connection().execute("SELECT status,error FROM runs").fetchall()
        assert (len(review.requests), len(tev.requests)) == counts
        assert len(github.posted_comments) == 1 and len(github.updated_comments) == 1
        assert github.updated_comments[0]["id"] == comment_id
        assert "변경된 팀 검토" in github.updated_comments[0]["body"]
        second_run = next(directory for directory in runs.iterdir() if directory != first_run)
        assert json.loads((second_run / "snapshot.json").read_text())["target_base_sha"] == advanced
        receipt = json.loads((second_run / "B/receipt.json").read_text())
        assert receipt["publication"]["status"] == "succeeded"
        assert receipt["publication"]["comment_id"] == comment_id
        assert receipt["usage"]["model_calls"] == 0
        assert json.loads((second_run / "B/input.json").read_text()) == original_input
        assert json.loads((second_run / "B/result.json").read_text()) == original_result
        assert json.loads((second_run / "execution-plan.json").read_text()) == original_plan
        assert json.loads((second_run / "repository-settings.json").read_text())["bundle"]["source_sha"] == advanced
        analyzer.run_analysis.assert_not_called()


def test_b_perspective_failure_preserves_selected_a_execution_and_posts_b_status(
    tmp_path, valid_config_yaml, worker_pipeline_ports,
):
    from review_bot.domain.intent_perspectives import PERSPECTIVES

    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, "architecture,intent", comments=True,
        fail_perspective="perspective:" + PERSPECTIVES[1].id) as context:
        config, worker, github, store, root, _, review, tev = context
        assert worker.run_once() is False
        assert worker_pipeline_ports["analyzer"].run_analysis.call_count == 2
        run_dir = next((root / "artifacts/runs/99/7").iterdir())
        plan = json.loads((run_dir / "execution-plan.json").read_text())
        assert plan["status"] == "planned"
        assert plan["partition"]["selected_modes"] == ["architecture", "intent"]
        assert plan["partition"]["deferred_modes"] == {"intent": "invalid_closed_choice"}
        architecture = json.loads((run_dir / "A/result.json").read_text())
        assert architecture["status"] == "completed", architecture["reason"]
        assert architecture["partition"]["shared_plan_id"] == plan["partition"]["plan_id"]
        receipt = json.loads((run_dir / "B/receipt.json").read_text())
        assert receipt["execution_status"] == "failed"
        assert receipt["error_code"] == "partition_mode_planning_deferred:invalid_closed_choice"
        assert not (run_dir / "B/result.json").exists()
        assert all(json.loads(request.user_content)["mode"] == "architecture" for request in review.requests)
        assert len(github.posted_comments) == 2
        intent_comment = next(comment for comment in github.posted_comments if comment["body"].startswith(config.publish.b_marker))
        assert intent_comment["id"] == receipt["publication"]["comment_id"]
        assert receipt["publication"]["status"] == "succeeded"
        assert not list((root / "shared-review-cache/44/99/planning").glob("*.json"))
        assert tev.requests and review.requests


def test_settings_fallback_cannot_hide_authorization_loss_before_model_work(
    tmp_path, valid_config_yaml, worker_pipeline_ports,
):
    from review_bot.application.repository_review_settings import RepositoryReviewSettings, SettingsResolution

    def fallback_after_auth_loss(run, pr_data, github, *, before_request):
        assert before_request()
        github.authorization_failures += 1
        return SettingsResolution(RepositoryReviewSettings(),
            diagnostic="settings_source_or_state_unavailable", used_fallback=True)

    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, "intent", comments=True,
        settings_factory=fallback_after_auth_loss) as context:
        _, worker, github, store, _, _, review, tev = context
        github.authorization_failures = 0
        assert worker.run_once() is False
        status, error = store._get_connection().execute("SELECT status,error FROM runs").fetchone()
        assert status == "failed" and error == "repository_settings_authorization_lost"
        assert not review.requests and not tev.requests
        worker_pipeline_ports["analyzer"].run_analysis.assert_not_called()
        assert not github.posted_comments and not github.updated_comments


def test_analysis_failure_notice_authorization_loss_preserves_receipt_and_stops_b(
    tmp_path, valid_config_yaml, worker_pipeline_ports,
):
    from unittest.mock import Mock
    from review_bot.ports import GitHubAuthorizationError

    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, "architecture,intent", comments=True) as context:
        _, worker, github, store, root, _, review, tev = context
        worker_pipeline_ports["analyzer"].run_analysis.side_effect = RuntimeError("synthetic-analysis-failure")
        github.find_marker_comment = Mock(side_effect=GitHubAuthorizationError("synthetic-auth-loss"))
        assert worker.run_once() is False
        status, error = store._get_connection().execute("SELECT status,error FROM runs").fetchone()
        assert status == "failed" and error == "publication_authorization_lost"
        assert tev.requests and not review.requests
        github.find_marker_comment.assert_called_once()
        run_dir = next((root / "artifacts/runs/99/7").iterdir())
        receipt = json.loads((run_dir / "A/receipt.json").read_text())
        assert receipt["execution_status"] == "failed"
        assert receipt["publication"]["reason"] == "publication_authorization_lost"
        assert receipt["publication"]["input_fingerprint"] is None
        assert not (run_dir / "B/result.json").exists()
        assert not github.posted_comments and not github.updated_comments


def test_common_plan_failure_notice_authorization_loss_stops_pending_notice(
    tmp_path, valid_config_yaml, worker_pipeline_ports,
):
    from unittest.mock import Mock
    from review_bot.ports import GitHubAuthorizationError

    with _app_worker(tmp_path, valid_config_yaml, worker_pipeline_ports, "failure", comments=True) as context:
        _, worker, github, store, root, _, review, tev = context
        github.find_marker_comment = Mock(side_effect=GitHubAuthorizationError("synthetic-auth-loss"))
        assert worker.run_once() is False
        status, error = store._get_connection().execute("SELECT status,error FROM runs").fetchone()
        assert status == "failed" and error == "partition_planning_failed:invalid_closed_choice"
        assert tev.requests and not review.requests
        github.find_marker_comment.assert_called_once()
        worker_pipeline_ports["analyzer"].run_analysis.assert_not_called()
        run_dir = next((root / "artifacts/runs/99/7").iterdir())
        plan = json.loads((run_dir / "execution-plan.json").read_text())
        assert plan["status"] == "failed" and plan["reason"] == "invalid_closed_choice"
        assert not (run_dir / "A/result.json").exists() and not (run_dir / "B/result.json").exists()
        assert not github.posted_comments and not github.updated_comments
