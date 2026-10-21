"""Background Worker execution loop, lease heartbeat management, and generation fencing."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import signal
import socket
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from types import SimpleNamespace
from pathlib import Path
from typing import Any

from review_bot.adapters.github.auth import GitHubAppAuthError
from review_bot.adapters.github.intent_sources import collect_intent_sources, intent_sources_current
from review_bot.adapters.llamacpp import LlamaCppModelClient
from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.markdown.intent_formatter import format_intent_markdown
from review_bot.adapters.beam_cloud import BeamCloudModelClient
from review_bot.adapters.model_http import reset_model_request_context, set_model_request_context
from review_bot.adapters.ollama import OllamaModelClient
from review_bot.application.classification import classify_pr
from review_bot.application.context import build_task_input
from review_bot.application.delta import build_delta
from review_bot.application.intent_context import IntentInputError, build_intent_task_input
from review_bot.application.publication import PublicationAuthorizationError, publication_fingerprint, publish_review_comment
from review_bot.application.architecture_review import run_review_a_pipeline
from review_bot.application.declared_intent_review import run_review_b_pipeline
from review_bot.application.selection import DeadlineModeSelector, LightweightModeSelector, ModelModeSelector
from review_bot.application.snapshot import assemble_snapshot_bundle, collect_snapshot_material
from review_bot.application.repository_review_settings import RepositoryReviewSettings, SettingsResolution
from review_bot.config import ReviewBotConfig, shared_plan_enabled
from review_bot.domain.envelope_v2 import ExecutionReceipt, PublicationReceipt, RunRecord
from review_bot.domain.intent_models import IntentReviewLimits
from review_bot.domain.models_v1 import AnalysisRun, Snapshot
from review_bot.domain.status import RunStage, RunStatus
from review_bot.ports import (
    GitHubAuthorizationError,
    GitHubClient,
    JobStore,
    ModelClient,
    ModeSelectorPort,
    RepositoryReader,
    StaticAnalyzer,
)
from review_bot.runtime.artifacts import RunArtifacts
from review_bot.runtime.mode_fingerprints import architecture_input_fingerprint, intent_input_fingerprint
from review_bot.runtime.shared_review_cache import SharedReviewCache
from review_bot.runtime.stored_partition_review import decode_stored_partition_review, shared_review_generation_key
from review_bot.runtime.model_clients import build_intent_selection_client, build_planning_client, build_review_client, build_selector_client

logger = logging.getLogger("review_bot.worker")


class _RunAborted(Exception):
    """Stop all modes when the common execution gate closes."""


class _ModeFailure(Exception):
    """A sanitized mode-local error which must not suppress other selected modes."""


class _CommonFailure(Exception):
    """Shared repository collection is no longer trustworthy for either mode."""


class WorkerLoop:
    """Worker daemon executing claimed jobs with lease heartbeats and generation fencing.

    Fast selection defaults to a local selector and never reuses model_client.
    Explicit model profiles compose review and, when enabled, a separate
    selector client. run_once prepares models before claiming a lease;
    callers claiming manually must call prepare_models before claiming work.
    For other lightweight models, explicitly inject
    mode_selector=ModelModeSelector(selector_client, model="<provider-model>").
    """

    def __init__(
        self,
        config: ReviewBotConfig | Any,
        job_store: JobStore,
        repo_reader: RepositoryReader | None = None,
        analyzer: StaticAnalyzer | None = None,
        model_client: ModelClient | None = None,
        github_client: GitHubClient | None = None,
        worker_id: str | None = None,
        poll_interval: float = 1.0,
        lease_duration: float = 30.0,
        mode_selector: ModeSelectorPort | None = None,
        github_client_factory: Callable[[RunRecord], GitHubClient] | None = None,
        repo_reader_factory: Callable[[RunRecord, dict[str, Any], GitHubClient], RepositoryReader] | None = None,
        artifact_root: Path | None = None,
        analyzer_factory: Callable[[], StaticAnalyzer] | None = None,
        stop_event: threading.Event | None = None,
        planning_client: ModelClient | None = None,
        intent_selection_client: ModelClient | None = None,
        repository_settings_factory: Callable[..., SettingsResolution] | None = None,
    ) -> None:
        self.config = config
        self.job_store = job_store
        self.repo_reader = repo_reader
        self.analyzer = analyzer
        self.analyzer_factory = analyzer_factory
        self.github_client_factory = github_client_factory
        self.repo_reader_factory = repo_reader_factory
        self.repository_settings_factory = repository_settings_factory
        self._repository_settings = SettingsResolution(RepositoryReviewSettings())
        self.artifact_root = artifact_root
        self._review_cache = SharedReviewCache(artifact_root.parent) if artifact_root is not None else None
        self._generation_keys: dict[str, str] = {}
        model_settings = getattr(config, "model", None)
        profile = getattr(model_settings, "profile", None)
        self.model_client = model_client if model_client is not None else build_review_client(profile, model_settings)
        self.planning_client = planning_client
        self.intent_selection_client = intent_selection_client
        self.github_client = github_client
        self._ollama_selector_client: ModelClient | None = None
        self._models_prepared = False
        if (
            mode_selector is None and profile is not None
            and getattr(getattr(config, "execution", None), "selector_uses_model", False)
            and not shared_plan_enabled(config)
        ):
            self._ollama_selector_client = build_selector_client(profile, model_settings)
            mode_selector = ModelModeSelector(self._ollama_selector_client, model=self._ollama_selector_client.model)
        selector = mode_selector if mode_selector is not None else LightweightModeSelector()
        self.mode_selector = selector if isinstance(selector, DeadlineModeSelector) else DeadlineModeSelector(selector)
        self.worker_id = (
            worker_id
            or f"{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
        )
        self.poll_interval = poll_interval
        self.lease_duration = lease_duration

        self._stop_event = stop_event if stop_event is not None else threading.Event()
        if isinstance(self.intent_selection_client, LlamaCppModelClient):
            self.intent_selection_client.request_cancelled = self._stop_event.is_set
        if isinstance(self.model_client, (OllamaModelClient, LlamaCppModelClient, BeamCloudModelClient)):
            self.model_client.request_cancelled = self._stop_event.is_set
        if self._ollama_selector_client is not None:
            self._ollama_selector_client.request_cancelled = self._stop_event.is_set
        if isinstance(self.planning_client, LlamaCppModelClient):
            self.planning_client.request_cancelled = self._stop_event.is_set
        self._lease_lost = False
        self._current_run_id: str | None = None
        self._partition_budget_key = None
        self._partition_budget = None
        self._partition_plan = self._partition_task = self._partition_planning_result = None
        self._setup_signals()

    def _setup_signals(self) -> None:
        """Register POSIX signal handlers for graceful shutdown if running in main thread."""
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
        except (ValueError, AttributeError):
            # Not in main thread or signal handling unavailable
            pass

    def _handle_signal(self, signum: int, frame: Any) -> None:
        logger.info(f"Signal {signum} received, stopping worker {self.worker_id}...")
        self.stop()

    def stop(self) -> None:
        """Signal worker loop to stop gracefully after current task step."""
        self._stop_event.set()

    def prepare_models(self) -> None:
        """Verify and warm explicitly configured local models before taking a job lease."""
        if self._models_prepared:
            return
        if isinstance(self.model_client, (OllamaModelClient, LlamaCppModelClient, BeamCloudModelClient)):
            self.model_client.warmup()
        if self._ollama_selector_client is not None:
            self._ollama_selector_client.warmup()
        if shared_plan_enabled(self.config):
            self._prepare_partition_planner()
            if isinstance(self.planning_client, LlamaCppModelClient):
                self.planning_client.warmup()
        elif getattr(getattr(self.config, "intent", None), "pipeline", None) == "perspectives-v1":
            self._prepare_intent_selector()
            if isinstance(self.intent_selection_client, LlamaCppModelClient):
                self.intent_selection_client.warmup()
        self._models_prepared = True

    def _prepare_intent_selector(self) -> None:
        if self.intent_selection_client is None:
            self.intent_selection_client = build_intent_selection_client(self.config.model.profile, self.config.model)
            if isinstance(self.intent_selection_client, LlamaCppModelClient):
                self.intent_selection_client.request_cancelled = self._stop_event.is_set

    def _prepare_partition_planner(self) -> None:
        if self.planning_client is None:
            self.planning_client = build_planning_client(self.config.model.profile, self.config.model)
            if isinstance(self.planning_client, LlamaCppModelClient):
                self.planning_client.request_cancelled = self._stop_event.is_set

    def _review_partitioned(self, task_input, mode, run, deadline, language, *, available_modes=()):
        from review_bot.application.partition_review import run_partition_review
        from review_bot.application.model_execution import ModelExecutionBudget
        from review_bot.domain.partition_review import PartitionReviewLimits

        self._prepare_partition_planner()
        trial = self.config.partition_evaluation
        limits = PartitionReviewLimits(max_total_tokens=trial.max_total_tokens or self.config.execution.max_total_tokens,
            max_calls=trial.max_calls or self.config.execution.max_model_calls, input_target_tokens=trial.input_target_tokens,
            observation_output_tokens=trial.observation_output_tokens,
            task_timeout_seconds=max(1, math.ceil(deadline - time.monotonic())))
        key = (run.run_id, run.generation, run.attempts)
        if self._partition_budget_key != key:
            self._partition_budget_key = key
            self._partition_budget = ModelExecutionBudget(limits, deadline, lambda: self._active(run, deadline))
        return run_partition_review(task_input, mode, self.model_client, self.planning_client,
            limits=limits, language=language, before_stage=lambda: self._active(run, deadline),
            shared_budget=self._partition_budget, available_modes=available_modes,
            prepared_plan=getattr(self, "_partition_plan", None) if mode != "planning" else None,
            review_instructions=self._repository_settings.bundle.instructions_for(mode) if mode != "planning" else "")

    def _plan_partitioned(self, task_input, available_modes, run, deadline):
        from review_bot.application.partition_review import validate_partition_result
        plan = self._review_partitioned(task_input, "planning", run, deadline,
            self.config.output.language, available_modes=tuple(available_modes))
        validate_partition_result(plan, task_input)
        self._partition_planning_result = plan
        if plan.status != "planned":
            raise _CommonFailure("partition_planning_failed:" + plan.reason)
        self._partition_plan, self._partition_task = plan, task_input
        return plan

    def _cached_review(self, run, mode):
        key = self._generation_keys.get(mode)
        if self._review_cache is None or key is None:
            return None
        cached = self._review_cache.load(run.installation_id, run.repository_id, mode, key,
            validate_and_decode=decode_stored_partition_review)
        if cached is not None and cached[1].partition.get("deferred_modes"):
            return None
        return cached

    def _save_cached_review(self, run, mode, task, result):
        if getattr(result, "partition", {}).get("deferred_modes"):
            return
        key = self._generation_keys.get(mode)
        if self._review_cache is not None and key is not None:
            self._review_cache.save(run.installation_id, run.repository_id, mode, key,
                {"task": asdict(task), "result": asdict(result)},
                validate_and_decode=decode_stored_partition_review)

    def _presentation_options(self, mode):
        bundle = self._repository_settings.bundle
        return {"presentation": dict(bundle.presentation), "comment_template": bundle.comment_template(mode)}

    def _settings_notice(self, markdown):
        if self._repository_settings.diagnostic is None:
            return markdown
        first, separator, rest = markdown.partition("\n")
        notice = "> 저장소 설정을 적용하지 못해 마지막 정상 설정 또는 내장 기본값으로 검토했습니다."
        return first + "\n" + notice + "\n" + rest if separator else markdown + "\n" + notice

    def _publication_input_key(self, mode, fallback):
        key = self._generation_keys.get(mode)
        if key is None:
            return fallback
        root = Path(__file__).parents[1] / "adapters" / "markdown"
        renderer = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("partition_formatter.py", "developer_presentation.py")}
        return hashlib.sha256(json.dumps({"generation": key, "renderer": renderer,
            "presentation": self._repository_settings.bundle.presentation_hash,
            "settings_notice": self._repository_settings.diagnostic}, sort_keys=True).encode()).hexdigest()

    def _publish_failure_notice(self, run, mode, snapshot, reason, deadline, *, draft=False):
        if (self.config.publish.mode != "comment" or self.github_client is None
                or draft and not self.config.publish.draft_comments
                or mode == "intent" and not self.config.publish.b_comments_enabled):
            return None
        from review_bot.adapters.markdown.developer_presentation import finish, status_sections

        marker = self.config.publish.a_marker if mode == "architecture" else self.config.publish.b_marker
        title = "PMD/CPD·아키텍처 리뷰" if mode == "architecture" else "의도 리뷰"
        body = finish(marker, title, status_sections(SimpleNamespace(status="not_run", reason=reason),
            ko=self.config.output.language == "ko", unreviewed=(title,)),
            scope="검토가 완료되지 않아 코드에 대한 최종 판단은 게시하지 않습니다.",
            max_length=min(60000, self.config.output.max_comment_characters))
        auth_before = getattr(self.github_client, "authorization_failures", None)
        authorization_lost = False
        try:
            publication = publish_review_comment(self.github_client, run.repository_id, run.pr_number,
                snapshot.head_sha, snapshot.target_base_sha, body, marker=marker,
                publication_guard=lambda: self._active(run, deadline),
                context_guard=lambda current: not current.get("draft", False) or self.config.publish.draft_comments,
                strict_target=isinstance(getattr(self.github_client, "bot_login", None), str))
        except PublicationAuthorizationError as error:
            publication, authorization_lost = error.receipt, True
        except GitHubAuthorizationError:
            publication = PublicationReceipt(None, "unknown",
                publication_fingerprint(run.repository_id, run.pr_number, snapshot.head_sha, body), marker=marker)
            authorization_lost = True
        auth_after = getattr(self.github_client, "authorization_failures", None)
        if authorization_lost or (type(auth_before) is int and type(auth_after) is int and auth_after > auth_before):
            has_comment = type(publication.comment_id) is int and publication.comment_id > 0
            publication = replace(publication, status="unknown" if authorization_lost or has_comment else "failed",
                reason="publication_authorization_lost", input_fingerprint=None)
        return publication

    def _review_perspectives(self, task_input, run, deadline, language):
        from review_bot.application.intent_perspective_review import run_intent_perspective_review
        from review_bot.domain.intent_perspectives import IntentPerspectiveLimits

        if shared_plan_enabled(self.config):
            return self._review_partitioned(task_input, "intent", run, deadline, language)
        self._prepare_intent_selector()
        limits = IntentPerspectiveLimits(task_timeout_seconds=max(1, math.ceil(deadline - time.monotonic())))
        return run_intent_perspective_review(task_input, self.model_client,
            selection_client=self.intent_selection_client, language=language,
            limits=limits, before_stage=lambda: self._active(run, deadline))

    def _start_heartbeat(self, run_id: str) -> tuple[threading.Event, threading.Thread]:
        """Launch background thread renewing lease at 1/3 lease_duration interval."""
        heartbeat_stop = threading.Event()
        interval = max(0.1, self.lease_duration / 3.0)

        def _heartbeat_loop() -> None:
            while not heartbeat_stop.wait(timeout=interval):
                try:
                    renewed = self.job_store.renew_lease(
                        run_id, self.worker_id, self.lease_duration
                    )
                except Exception:
                    logger.info("Lease renewal failed for run %s", run_id, exc_info=True)
                    self._lease_lost = True
                    break
                if not renewed:
                    logger.info(
                        f"Failed to renew lease for run {run_id}, ownership lost"
                    )
                    self._lease_lost = True
                    break

        thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"heartbeat-{run_id}",
            daemon=True,
        )
        thread.start()
        return heartbeat_stop, thread

    def _check_fencing_and_lease(self, run_id: str, run_generation: int) -> bool:
        """Check lease ownership and generation fencing. Returns True if valid to proceed."""
        if self._lease_lost:
            logger.info(f"Aborting execution: lease lost for run {run_id}")
            return False

        try:
            owns_lease = self.job_store.renew_lease(run_id, self.worker_id, self.lease_duration)
            is_current = owns_lease and self.job_store.check_generation_fencing(
                run_id, worker_id=self.worker_id, generation=run_generation,
            )
        except Exception:
            self._lease_lost = True
            logger.info("Aborting run %s: could not validate lease fence", run_id, exc_info=True)
            return False

        if not owns_lease:
            self._lease_lost = True
            logger.info("Aborting execution: lease expired or transferred for run %s", run_id)
            return False

        if not is_current:
            logger.info(
                "Aborting run %s generation %s: lease or generation fence is no longer valid",
                run_id, run_generation,
            )
            return False

        return True

    def _update_run(
        self, run: RunRecord, status: RunStatus, stage: RunStage | None = None,
        error: str | None = None,
        *, reason: str | None = None, fingerprint: str | None = None,
    ) -> bool:
        """Check before each transition and atomically reject races at the store boundary."""
        if not self._check_fencing_and_lease(run.run_id, run.generation):
            return False
        applied = self.job_store.update_run(
            run.run_id, status=status, stage=stage, error=error,
            worker_id=self.worker_id, generation=run.generation,
            **({"reason": reason} if reason is not None else {}),
            **({"fingerprint": fingerprint} if fingerprint is not None else {}),
        )
        if not applied:
            # Supersede only an owned stale generation; lease loss must never mutate the run.
            self._check_fencing_and_lease(run.run_id, run.generation)
            logger.info("Run transition rejected by CAS fence for %s", run.run_id)
        return applied

    def process_run(self, run: RunRecord) -> bool:
        """Execute single claimed run through the full analysis and review pipeline."""
        run_id = run.run_id
        self._current_run_id = run_id
        self._lease_lost = False
        self._partition_plan = self._partition_task = self._partition_planning_result = None
        self._generation_keys = {}
        self._repository_settings = SettingsResolution(RepositoryReviewSettings())

        previous_github_client = self.github_client
        previous_repo_reader = self.repo_reader
        artifacts = None
        deadline = time.monotonic() + getattr(getattr(self.config, "execution", None), "task_timeout_seconds", 900)
        receipts: dict[str, ExecutionReceipt] = {}
        heartbeat_stop, heartbeat_thread = self._start_heartbeat(run_id)
        request_scope_lock = threading.Lock()
        request_scope_closed = threading.Event()

        def request_cancelled() -> bool:
            with request_scope_lock:
                return request_scope_closed.is_set() or not self._active(run, deadline)

        request_context = set_model_request_context(request_cancelled, deadline)

        try:
            self.config.publish.validate_b_marker()
            # 1. Stage: PREPARE, with installation/repository-bound dependencies.
            self._stage(run, RunStage.PREPARE, deadline)
            if run.kind == "fixed_intent":
                return self._process_fixed_intent(run, deadline)
            if self.github_client_factory is not None:
                self.github_client = self.github_client_factory(run)
            if self.repo_reader_factory is not None:
                self.repo_reader = None

            head_sha = ""
            target_base_sha = ""
            branch = ""
            draft = False
            labels: tuple[str, ...] = ()
            title = ""
            body = ""
            repo_name = f"repo-{run.repository_id}"
            pr_data = {}

            if self.github_client:
                options = {}
                if "intent" in getattr(getattr(self.config, "execution", None), "available_modes", ("architecture",)):
                    options = {"max_response_bytes": 524288, "timeout_seconds": min(10.0, deadline - time.monotonic())}
                pr_data = self.github_client.get_pull_request(run.pr_number, **options)
                self._require_active(run, deadline)
                draft = pr_data.get("draft") is True
                if pr_data.get("state", "open") != "open":
                    return self._update_run(run, RunStatus.CANCELLED, reason="pr_not_open")
                head_sha = pr_data.get("head", {}).get("sha", "")
                target_base_sha = pr_data.get("base", {}).get("sha", "")
                branch = pr_data.get("head", {}).get("ref", "")
                labels = tuple(
                    lbl.get("name", "") for lbl in pr_data.get("labels", [])
                )
                title = pr_data.get("title") or ""
                body = pr_data.get("body") or ""
                repo_name = pr_data.get("base", {}).get("repo", {}).get(
                    "full_name", repo_name
                )
                if self.repo_reader_factory is not None:
                    self._require_active(run, deadline)
                    self.repo_reader = self.repo_reader_factory(run, pr_data, self.github_client)
                if self.repository_settings_factory is not None:
                    auth_before = getattr(self.github_client, "authorization_failures", None)
                    self._repository_settings = self.repository_settings_factory(
                        run, pr_data, self.github_client,
                        before_request=lambda: self._active(run, deadline))
                    self._require_active(run, deadline)
                    auth_after = getattr(self.github_client, "authorization_failures", None)
                    if type(auth_before) is int and type(auth_after) is int and auth_after > auth_before:
                        raise _CommonFailure("repository_settings_authorization_lost")

            # 2. Stage: SELECT
            self._stage(run, RunStage.SELECT, deadline)

            pr_snap = None
            material = None
            if self.repo_reader:
                execution_cfg = getattr(self.config, "execution", None)
                available_modes = tuple(
                    getattr(execution_cfg, "available_modes", ("architecture",))
                )
                partition_flow = shared_plan_enabled(self.config)
                if "intent" in available_modes or partition_flow:
                    if not pr_data:
                        raise RuntimeError("intent_pr_source_unavailable")
                    sources = collect_intent_sources(self.github_client, repo_name, run.pr_number, pr_data,
                        timeout_seconds=min(10.0, deadline - time.monotonic()),
                        before_request=lambda: self._active(run, deadline))
                    material = collect_snapshot_material(self.repo_reader, target_base_sha, head_sha,
                        evidence=sources.evidence, unavailable_sources=sources.unavailable_sources,
                        related_paths=tuple(self.config.intent.context_paths), captured_at=sources.captured_at,
                        as_of=sources.as_of, before_read=lambda: self._active(run, deadline),
                        include_java_build_manifests=partition_flow)
                self._require_active(run, deadline)
                review_snap, pr_snap, classification, mode_sel = assemble_snapshot_bundle(
                    repo_reader=self.repo_reader,
                    repository=repo_name,
                    pr_number=run.pr_number,
                    repository_id=run.repository_id,
                    installation_id=run.installation_id,
                    target_base_sha=target_base_sha,
                    head_sha=head_sha,
                    source_branch=branch,
                    labels=labels,
                    title=title,
                    body=body,
                    available_modes=available_modes,
                    selector_policy="all_available" if partition_flow else getattr(execution_cfg, "selector_policy", "all_available"),
                    selector_uses_model=False if partition_flow else getattr(execution_cfg, "selector_uses_model", False),
                    mode_selector=None if partition_flow else self.mode_selector,
                    **({"material": material} if material is not None else {}),
                )
                self._require_active(run, deadline)

                if review_snap.comparison_status != "comparable":
                    reason = "; ".join(
                        source.reason for source in review_snap.unavailable_sources
                        if source.source == "merge_base"
                    )
                    error = f"comparison_status=incomparable: {reason}"
                    logger.warning("Aborting run %s: %s", run_id, error)
                    self._update_run(run, RunStatus.FAILED, RunStage.SELECT, error=error)
                    return False

                if partition_flow and mode_sel.selected_modes and review_snap.comparison_status == "comparable":
                    from review_bot.application.selection import record_planned_modes
                    if self.artifact_root is None or material is None:
                        raise _CommonFailure("partition_planning_artifacts_or_material_unavailable")
                    artifacts = RunArtifacts(self.artifact_root, run)
                    artifacts.save_common(snapshot=review_snap, selection=mode_sel, config_hash=self.config.config_hash)
                    artifacts.write_json("repository-settings.json", self._repository_settings)
                    paths = tuple(sorted({f.path for c in (material.base, material.head) for f in c.files}))
                    planning_task = build_intent_task_input(review_snap, material.base, material.head,
                        related_paths=tuple(sorted(set(paths) | set(self.config.intent.context_paths))))
                    artifacts.write_json("planning-input.json", planning_task)
                    self._prepare_partition_planner()
                    self._generation_keys["planning"] = shared_review_generation_key(planning_task, self.config,
                        self.model_client, self.planning_client, self._repository_settings.bundle, "planning")
                    try:
                        cached_plan = self._cached_review(run, "planning")
                        if cached_plan is None:
                            plan = self._plan_partitioned(planning_task, mode_sel.selected_modes, run, deadline)
                            self._require_active(run, deadline)
                            self._save_cached_review(run, "planning", planning_task, plan)
                        else:
                            self._partition_task, self._partition_plan = cached_plan
                            plan = self._partition_planning_result = self._partition_plan
                            artifacts.write_json("observed-planning-input.json", planning_task)
                            artifacts.write_json("planning-input.json", self._partition_task)
                            artifacts.write_json("planning-reuse.json", {"run_key": plan.run_key, "new_model_calls": 0})
                    finally:
                        if self._partition_planning_result is not None:
                            artifacts.save_execution_plan(self._partition_planning_result)
                    for selected_mode in plan.partition["selected_modes"]:
                        base_key = shared_review_generation_key(planning_task, self.config, self.model_client,
                            self.planning_client, self._repository_settings.bundle, selected_mode)
                        self._generation_keys[selected_mode] = hashlib.sha256(json.dumps(
                            {"generation": base_key, "plan_id": plan.partition["plan_id"]}, sort_keys=True).encode()).hexdigest()
                    mode_sel = record_planned_modes(review_snap, available_modes,
                        plan.partition["selected_modes"], plan.partition["plan_id"])
                    artifacts.save_common(snapshot=review_snap, selection=mode_sel, config_hash=self.config.config_hash)
                if not mode_sel.selected_modes:
                    if artifacts is not None:
                        artifacts.finish("skipped")
                    logger.info(
                        "No applicable modes for run %s; skipping downstream execution", run_id,
                    )
                    return self._update_run(
                        run, RunStatus.SKIPPED, RunStage.SELECT, reason="not_applicable",
                    )

                semantic_input = {
                    "repository_id": run.repository_id, "pr_number": run.pr_number,
                    "snapshot_id": review_snap.snapshot_id,
                    "selected_modes": mode_sel.selected_modes,
                    "perspective": classification.review_mode,
                    "config_hash": self.config.config_hash,
                    **({"plan_id": self._partition_plan.partition["plan_id"]} if self._partition_plan is not None else {}),
                }
                fingerprint = hashlib.sha256(json.dumps(semantic_input, sort_keys=True).encode()).hexdigest()
                run = replace(run, fingerprint=fingerprint)
                if not self._update_run(run, RunStatus.RUNNING, fingerprint=fingerprint):
                    return False

            if pr_snap is None:
                self._stage(run, RunStage.PUBLISH, deadline)
                raise _ModeFailure("review_result_unavailable")
            self._require_active(run, deadline)
            if self.artifact_root is not None:
                artifacts = artifacts or RunArtifacts(self.artifact_root, run)
                artifacts.save_common(snapshot=review_snap, selection=mode_sel, config_hash=self.config.config_hash)
            for mode in mode_sel.selected_modes:
                self._require_active(run, deadline)
                try:
                    deferred = self._partition_plan.partition.get("deferred_modes", {}) if self._partition_plan is not None else {}
                    if mode in deferred:
                        error_code = "partition_mode_planning_deferred:" + deferred[mode]
                        receipt = ExecutionReceipt(2, mode, mode_sel.selection_id, review_snap.snapshot_id,
                            run.generation, run.fingerprint, "failed", error_code=error_code,
                            publication=self._publish_failure_notice(run, mode, review_snap, error_code, deadline, draft=draft))
                    elif mode == "architecture":
                        receipt = self._execute_architecture(run, review_snap, pr_snap, classification, mode_sel,
                            artifacts, title, body, draft, deadline)
                    else:
                        receipt = self._execute_intent(run, review_snap, mode_sel, material, artifacts, deadline, draft=draft)
                except (_RunAborted, _CommonFailure, GitHubAuthorizationError):
                    raise
                except Exception as exc:
                    self._require_active(run, deadline)
                    prefix = "worker" if mode_sel.selected_modes == ("architecture",) else mode
                    error_code = str(exc) if isinstance(exc, (_ModeFailure, IntentInputError)) else f"{prefix}_pipeline_error:{type(exc).__name__}"
                    receipt = ExecutionReceipt(2, mode, mode_sel.selection_id, review_snap.snapshot_id,
                        run.generation, run.fingerprint, "failed", error_code=error_code,
                        publication=self._publish_failure_notice(run, mode, review_snap, error_code, deadline, draft=draft))
                self._require_active(run, deadline)
                if receipt.publication is not None:
                    eligible = (receipt.execution_status in ("completed", "not_run") and receipt.publication.status == "succeeded"
                        and type(receipt.publication.comment_id) is int and receipt.publication.comment_id > 0)
                    receipt = replace(receipt, publication=replace(receipt.publication,
                        input_fingerprint=receipt.input_fingerprint if eligible else None))
                if artifacts is not None:
                    try:
                        artifacts.save_receipt(receipt)
                    except (OSError, ValueError, TypeError) as exc:
                        self._require_active(run, deadline)
                        receipt = replace(receipt, execution_status="failed", error_code=f"{mode}_artifact_error:{type(exc).__name__}")
                        if receipt.publication is not None:
                            receipt = replace(receipt, publication=replace(receipt.publication, input_fingerprint=None))
                        artifacts.save_receipt_manifest(receipt)
                publication = receipt.publication
                if not receipt.reused and publication is not None and (
                    publication.status in ("not_attempted", "unknown")
                    or type(publication.comment_id) is int and publication.comment_id > 0
                ):
                    self._record_publication(run, publication, deadline)
                receipts[mode] = receipt
                if publication is not None and publication.reason == "publication_authorization_lost":
                    raise _CommonFailure("publication_authorization_lost")
            failed = [receipt for receipt in receipts.values() if receipt.execution_status == "failed"
                or receipt.publication is not None and receipt.publication.status not in ("succeeded", "not_attempted")]
            all_reused = bool(receipts) and all(receipt.reused for receipt in receipts.values())
            terminal_status = RunStatus.FAILED if failed else RunStatus.SKIPPED if all_reused else RunStatus.SUCCEEDED
            error = "; ".join(receipt.error_code or receipt.publication.reason for receipt in failed) if failed else None
            primary = receipts.get("architecture") or receipts.get("intent")
            publication = primary.publication if primary is not None else None
            reason = "unchanged_input" if all_reused and not failed else publication.reason if not failed and publication is not None and publication.status == "not_attempted" else None
            self._require_active(run, deadline)
            if artifacts is not None:
                if shared_plan_enabled(self.config) and self._partition_budget is not None:
                    calls, tokens, unknown = self._partition_budget.totals()
                    artifacts.write_json("usage.json", {"calls": calls, "known_total_tokens": tokens,
                        "usage_complete": not unknown, "planning": self._partition_plan.usage if self._partition_plan else None,
                        "modes": {mode: receipt.usage for mode, receipt in receipts.items()}})
                artifacts.finish(terminal_status.value, publication)
            stage = RunStage.PUBLISH if not failed or any(receipt.publication for receipt in failed) else None
            applied = self._update_run(run, terminal_status, stage, error=error, reason=reason)
            return applied and not failed

        except _RunAborted:
            return False

        except Exception as exc:
            error_code = str(exc) if isinstance(exc, (_ModeFailure, _CommonFailure, IntentInputError)) else f"worker_pipeline_error:{type(exc).__name__}"
            if (isinstance(exc, _CommonFailure) and str(exc).startswith("partition_planning_failed:")
                    and pr_snap is not None and self._active(run, deadline)):
                try:
                    for pending_mode in mode_sel.selected_modes:
                        notice = self._publish_failure_notice(run, pending_mode, review_snap, error_code, deadline, draft=draft)
                        if notice is not None and (notice.comment_id is not None or notice.status == "unknown"):
                            self._record_publication(run, notice, deadline)
                        if notice is not None and notice.reason == "publication_authorization_lost":
                            break
                except _RunAborted:
                    return False
            if artifacts is not None and self._active(run, deadline):
                try:
                    artifacts.manifest["error_code"] = error_code
                    artifacts.finish("failed")
                except (OSError, ValueError, TypeError):
                    pass
            if self._update_run(run, RunStatus.FAILED, error=error_code):
                logger.error("Error processing run %s: %s", run_id, error_code)
            return False

        finally:
            with request_scope_lock:
                request_scope_closed.set()
            reset_model_request_context(request_context)
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.0)
            self._current_run_id = None
            if self.github_client_factory is not None:
                self.github_client = previous_github_client
            if self.repo_reader_factory is not None:
                self.repo_reader = previous_repo_reader

    def _active(self, run: RunRecord, deadline: float) -> bool:
        if not self._check_fencing_and_lease(run.run_id, run.generation):
            return False
        if self._stop_event.is_set():
            self._update_run(run, RunStatus.CANCELLED, reason="worker_stopped")
            return False
        if time.monotonic() >= deadline:
            self._update_run(run, RunStatus.FAILED, error="task_deadline_exceeded", reason="task_deadline_exceeded")
            return False
        return True

    def _require_active(self, run: RunRecord, deadline: float) -> None:
        if not self._active(run, deadline):
            raise _RunAborted

    def _stage(self, run: RunRecord, stage: RunStage, deadline: float) -> None:
        self._require_active(run, deadline)
        if not self._update_run(run, RunStatus.RUNNING, stage):
            raise _RunAborted

    def _record_publication(self, run: RunRecord, receipt: PublicationReceipt, deadline: float) -> None:
        self._require_active(run, deadline)
        if not self.job_store.record_publication(receipt, run.repository_id, run.pr_number,
            run_id=run.run_id, worker_id=self.worker_id, generation=run.generation):
            self._check_fencing_and_lease(run.run_id, run.generation)
            raise _RunAborted

    def _publication_superseded(
        self, run: RunRecord, receipt: ExecutionReceipt, artifacts: RunArtifacts | None, deadline: float,
    ) -> None:
        self._require_active(run, deadline)
        publication = receipt.publication
        if publication is not None and (type(publication.comment_id) is int or publication.status == "unknown"):
            self._record_publication(run, replace(publication, input_fingerprint=None), deadline)
        if artifacts is not None:
            artifacts.save_receipt(receipt)
            self._require_active(run, deadline)
            artifacts.finish("superseded", publication)
        self._update_run(run, RunStatus.SUPERSEDED, error=receipt.error_code)
        raise _RunAborted

    def _reuse_publication(
        self, run: RunRecord, mode: str, snapshot: Any, selection: Any, input_fingerprint: str | None,
        deadline: float, *, context_guard: Callable[[dict[str, Any]], bool] | None = None,
        source_guard: Callable[[dict[str, Any]], bool] | None = None,
    ) -> ExecutionReceipt | None:
        if input_fingerprint is None or self.github_client is None or self.config.publish.mode != "comment":
            return None
        marker = self.config.publish.a_marker if mode == "architecture" else self.config.publish.b_marker
        if mode == "intent" and (not self.config.publish.b_comments_enabled or not isinstance(getattr(self.github_client, "bot_login", None), str)):
            return None
        self._require_active(run, deadline)
        prior = self.job_store.get_current_publication(run, marker, input_fingerprint=input_fingerprint)
        if not isinstance(prior, PublicationReceipt) or type(prior.comment_id) is not int or prior.comment_id <= 0:
            return None
        auth_before = getattr(self.github_client, "authorization_failures", None)

        def current_sources() -> None:
            self._require_active(run, deadline)
            options = {"max_response_bytes": 524288, "timeout_seconds": min(10.0, deadline - time.monotonic())} if mode == "intent" else {}
            current = self.github_client.get_pull_request(run.pr_number, **options)
            self._require_active(run, deadline)
            reason = None
            if current.get("state") != "open":
                reason = "pr_not_open"
            elif current.get("head", {}).get("sha") != snapshot.head_sha:
                reason = "stale_head_sha"
            elif current.get("base", {}).get("sha") != snapshot.target_base_sha:
                reason = "stale_base_sha"
            elif mode == "intent" and type(current.get("draft")) is not bool:
                reason = "stale_pr_context"
            elif current.get("draft") is True and not self.config.publish.draft_comments:
                reason = "stale_pr_context"
            elif context_guard is not None and context_guard(current) is not True:
                reason = "stale_pr_context"
            elif source_guard is not None and source_guard(current) is not True:
                reason = "stale_intent_sources"
            if reason is not None:
                self._update_run(run, RunStatus.SUPERSEDED, error=f"Freshness check failed: {reason}")
                raise _RunAborted

        current_sources()
        options = {"before_request": lambda: self._active(run, deadline), "timeout_seconds": min(10.0, deadline - time.monotonic()), "max_response_bytes": 8388608} if mode == "intent" else {}
        existing = self.github_client.find_marker_comment(run.repository_id, run.pr_number, marker, **options)
        self._require_active(run, deadline)
        current_sources()
        auth_after = getattr(self.github_client, "authorization_failures", None)
        if type(auth_before) is int and type(auth_after) is int and auth_after > auth_before:
            raise GitHubAppAuthError("Publication authorization lost")
        if not isinstance(existing, dict) or existing.get("id") != prior.comment_id:
            return None
        text = existing.get("body")
        if not isinstance(text, str) or text.count(marker) != 1 or not text.startswith(marker) or publication_fingerprint(
            run.repository_id, run.pr_number, snapshot.head_sha, text,
        ) != prior.fingerprint:
            return None
        return ExecutionReceipt(2, mode, selection.selection_id, snapshot.snapshot_id, run.generation,
            run.fingerprint, "not_run", limitations=("reused_publication_no_new_model_call",),
            model_delivery={"status": "not_attempted"}, publication=replace(prior, reason="unchanged_input"),
            usage={"calls": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0},
            input_fingerprint=input_fingerprint, reused=True)

    def _abort_authorization_loss(
        self, run: RunRecord, artifacts: RunArtifacts | None, receipt: ExecutionReceipt, deadline: float,
    ) -> None:
        self._require_active(run, deadline)
        try:
            if artifacts is not None:
                artifacts.save_receipt(receipt)
            publication = receipt.publication
            if publication is not None and (publication.status == "unknown"
                or type(publication.comment_id) is int and publication.comment_id > 0):
                self._record_publication(run, publication, deadline)
        except _RunAborted:
            raise
        except Exception as exc:
            logger.warning("authorization_receipt_storage_failed:%s", type(exc).__name__)
        raise _CommonFailure("publication_authorization_lost")

    def _execute_architecture(
        self, run: RunRecord, review_snap: Any, pr_snap: Any, classification: Any, mode_sel: Any,
        artifacts: RunArtifacts | None, title: str, body: str, draft: bool, deadline: float,
    ) -> ExecutionReceipt:
        marker = self.config.publish.a_marker

        def context_guard(current: dict[str, Any]) -> bool:
            current_labels = tuple(label.get("name", "") for label in current.get("labels", []))
            current_branch = current.get("head", {}).get("ref", "")
            return ((current.get("title") or "") == title and (current.get("body") or "") == body
                and classify_pr(current_branch, current_labels, title=current.get("title") or "").review_mode == classification.review_mode
                and (current.get("draft") is not True or self.config.publish.draft_comments))

        input_fingerprint = self._publication_input_key("architecture", architecture_input_fingerprint(review_snap, classification, self.config, self.model_client, title=title, body=body))
        if not (draft and not self.config.publish.draft_comments):
            reused = self._reuse_publication(run, "architecture", review_snap, mode_sel, input_fingerprint, deadline, context_guard=context_guard)
            if reused is not None:
                return reused
        cached = self._cached_review(run, "architecture")
        if cached is not None:
            task_input, result = cached
            base_run = head_run = None
            head_sha = pr_snap.head_sha
        else:
            self._stage(run, RunStage.ANALYZE, deadline)
            if self.analyzer is None and self.analyzer_factory is not None:
                self.analyzer = self.analyzer_factory()
            base_run = head_run = None
            effective_base_sha, head_sha = pr_snap.base_sha, pr_snap.head_sha
            if self.analyzer and self.repo_reader:
                arch_cfg = getattr(self.config, "architecture", None)
                ruleset = getattr(arch_cfg, "ruleset_resource", "pmd/ruleset.xml")
                profile = getattr(arch_cfg, "analysis_profile", "java21-jdk-only-v1")
                analysis_runs: list[AnalysisRun] = []
                for sha in (effective_base_sha, head_sha):
                    self._require_active(run, deadline)
                    with tempfile.TemporaryDirectory(prefix="review-bot-tree-") as tmp:
                        source_root = Path(tmp).resolve()
                        try:
                            blobs = self.repo_reader.read_tree_blobs(sha)
                        except Exception as exc:
                            raise _CommonFailure(f"worker_pipeline_error:{type(exc).__name__}") from None
                        self._require_active(run, deadline)
                        for path, content in blobs.items():
                            relative_path = Path(path)
                            dest = source_root / relative_path
                            if relative_path.is_absolute() or ".." in relative_path.parts or not dest.resolve().is_relative_to(source_root):
                                raise ValueError("Unsafe repository path")
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(content)
                        self._require_active(run, deadline)
                        analysis_runs.append(self.analyzer.run_analysis(sha, source_root, ruleset, profile))
                base_run, head_run = analysis_runs
            self._stage(run, RunStage.COMPARE, deadline)
            delta = build_delta(base_run, head_run, rename_map=pr_snap.rename_map) if base_run and head_run else None
            self._stage(run, RunStage.CONTEXT, deadline)
            task_input = None
            if self.repo_reader and delta:
                try:
                    if shared_plan_enabled(self.config):
                        from review_bot.application.partition_review import prepare_partition_architecture_task
                        task_input = prepare_partition_architecture_task(self.repo_reader, pr_snap, delta, classification,
                            before_read=lambda: self._require_active(run, deadline))
                    else:
                        base_code = self.repo_reader.collect_code_snapshot(effective_base_sha)
                        self._require_active(run, deadline)
                        head_code = self.repo_reader.collect_code_snapshot(head_sha)
                except _RunAborted:
                    raise
                except Exception as exc:
                    raise _CommonFailure(f"worker_pipeline_error:{type(exc).__name__}") from None
                self._require_active(run, deadline)
                if task_input is None:
                    task_input = build_task_input(Snapshot(pr=pr_snap, base=base_code, head=head_code), delta, classification)
            self._stage(run, RunStage.REVIEW, deadline)
            if task_input is not None and shared_plan_enabled(self.config):
                result = self._review_partitioned(task_input, "architecture", run, deadline, self.config.output.language)
            else:
                result = run_review_a_pipeline(task_input, self.model_client) if task_input is not None else None
        self._stage(run, RunStage.VALIDATE, deadline)
        self._stage(run, RunStage.PUBLISH, deadline)
        writes_comment = self.config.publish.mode == "comment" and not (draft and not self.config.publish.draft_comments)
        if writes_comment and self.github_client is None:
            raise _ModeFailure("github_client_unavailable")
        if task_input is None or result is None:
            raise _ModeFailure("review_result_unavailable")
        if not writes_comment and artifacts is None:
            raise _ModeFailure("artifact_root_required")
        analysis_failed = any(getattr(value, "status", None) in ("failed", "missing_report") for value in (base_run, head_run))
        status = "failed" if analysis_failed or result.status == "failed" else result.status
        if shared_plan_enabled(self.config) and result.status == "partial_review":
            status = "failed"
        core_receipt = ExecutionReceipt(2, "architecture", mode_sel.selection_id, review_snap.snapshot_id,
            run.generation, result.run_key, status, a_result=result, usage=({"calls": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0} if cached is not None else result.usage),
            coverage=result.scope, limitations=tuple(task_input.limitations),
            model_delivery={"status": "not_attempted" if cached is not None else "sent" if result.usage and result.usage.get("calls", 0) else "not_attempted" if self.model_client is None else "unknown"},
            error_code="analysis_failed" if analysis_failed else result.reason if status == "failed" else None,
            input_fingerprint=input_fingerprint)
        try:
            formatter = format_review_markdown
            if shared_plan_enabled(self.config):
                from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
                formatter = format_partition_review_markdown
            markdown = formatter(result, task_input, marker=marker,
                max_length=self.config.output.max_comment_characters,
                artifact_notice="전체 A 자료는 비공개 서비스 결과 파일에 보존됩니다." if artifacts is not None else None,
                **({"publication_mode": "comment" if writes_comment else "artifact_only"} if shared_plan_enabled(self.config) else {}),
                **self._presentation_options("architecture"))
            markdown = self._settings_notice(markdown)
            self._require_active(run, deadline)
            if artifacts is not None:
                artifacts.save_review(snapshot=review_snap, selection=mode_sel, task_input=task_input,
                    base_run=base_run, head_run=head_run, result=result, markdown=markdown, config_hash=self.config.config_hash)
            self._require_active(run, deadline)
            if status == "completed" and cached is None:
                self._save_cached_review(run, "architecture", task_input, result)
        except (_RunAborted, _CommonFailure, GitHubAuthorizationError):
            raise
        except Exception as exc:
            self._require_active(run, deadline)
            prefix = "worker" if mode_sel.selected_modes == ("architecture",) else "architecture"
            error_code = f"{prefix}_pipeline_error:{type(exc).__name__}"
            return replace(core_receipt, execution_status="failed", error_code=error_code,
                publication=self._publish_failure_notice(run, "architecture", review_snap, error_code, deadline, draft=draft))
        self._require_active(run, deadline)
        if not writes_comment:
            publication = PublicationReceipt(None, "not_attempted",
                publication_fingerprint(run.repository_id, run.pr_number, head_sha, markdown), marker=marker,
                reason="artifact_only" if self.config.publish.mode == "artifact_only" else "draft_comments_disabled")
        else:
            auth_before = getattr(self.github_client, "authorization_failures", None)
            authorization_lost = False
            try:
                publication = publish_review_comment(github_client=self.github_client, repository_id=run.repository_id,
                    pr_number=run.pr_number, expected_head_sha=head_sha, expected_base_sha=review_snap.target_base_sha,
                    body=markdown, marker=marker, publication_guard=lambda: self._active(run, deadline),
                    **({"context_guard": context_guard} if self.github_client_factory is not None else {}))
            except GitHubAuthorizationError:
                authorization_lost = True
                publication = PublicationReceipt(None, "unknown",
                    publication_fingerprint(run.repository_id, run.pr_number, head_sha, markdown), marker=marker,
                    reason="publication_authorization_lost")
            self._require_active(run, deadline)
            auth_after = getattr(self.github_client, "authorization_failures", None)
            if authorization_lost or ("intent" in mode_sel.selected_modes and type(auth_before) is int and type(auth_after) is int and auth_after > auth_before):
                has_comment_id = type(publication.comment_id) is int and publication.comment_id > 0
                publication = replace(publication, status="unknown" if authorization_lost or has_comment_id else "failed",
                    reason="publication_authorization_lost", input_fingerprint=None)
                receipt = replace(core_receipt, execution_status="failed", error_code="publication_authorization_lost", publication=publication)
                self._abort_authorization_loss(run, artifacts, receipt, deadline)
            has_comment_id = type(publication.comment_id) is int and publication.comment_id > 0
            if publication.status != "succeeded" or not has_comment_id:
                stale = publication.status == "stale" or publication.reason in ("stale_head_sha", "stale_base_sha", "pr_not_open")
                reason = "invalid_comment_id" if publication.status == "succeeded" else publication.reason
                error = f"Freshness check failed: {reason}" if stale else f"Publication failed: {reason}"
                receipt = replace(core_receipt, execution_status="failed", error_code=error, publication=publication)
                if stale:
                    self._publication_superseded(run, receipt, artifacts, deadline)
                return receipt
        return replace(core_receipt, publication=publication)

    def _process_fixed_intent(self, run: RunRecord, deadline: float) -> bool:
        from review_bot.adapters.markdown.intent_perspective_formatter import format_intent_perspective_markdown
        from review_bot.application.intent_context import decode_intent_task_input
        from review_bot.runtime.artifacts import IntentArtifacts

        if self.config.intent.pipeline != "perspectives-v1" or self.config.publish.mode != "artifact_only":
            raise _ModeFailure("fixed_intent_requires_artifact_only_perspectives")
        if self.artifact_root is None:
            raise _ModeFailure("artifact_root_required")
        stored = self.job_store.get_fixed_intent_input(run.run_id)
        if stored is None:
            raise _ModeFailure("fixed_intent_input_unavailable")
        task_input = decode_intent_task_input(stored[0])
        if shared_plan_enabled(self.config):
            self._plan_partitioned(task_input, ("intent",), run, deadline)
        self._stage(run, RunStage.REVIEW, deadline)
        result = self._review_perspectives(task_input, run, deadline, stored[1])
        self._stage(run, RunStage.VALIDATE, deadline)
        if shared_plan_enabled(self.config):
            from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
            markdown = format_partition_review_markdown(result, task_input)
        else:
            markdown = format_intent_perspective_markdown(result, task_input)
        output = self.artifact_root / "fixed" / run.run_id / f"attempt-{run.attempts}"
        artifacts = IntentArtifacts(output)
        if shared_plan_enabled(self.config):
            artifacts.save_execution_plan(task_input, self._partition_plan)
        artifacts.save(task_input, result, markdown)
        self._require_active(run, deadline)
        complete = result.status == "completed"
        nonselection = result.reason == "mode_not_selected_in_shared_plan"
        applied = self._update_run(run, RunStatus.SKIPPED if nonselection else RunStatus.SUCCEEDED if complete else RunStatus.FAILED,
            RunStage.VALIDATE, reason=result.reason, error=None if complete or nonselection else result.reason,
            fingerprint=result.run_key)
        return applied and (complete or nonselection)

    def _execute_intent(
        self, run: RunRecord, review_snap: Any, mode_sel: Any, material: Any,
        artifacts: RunArtifacts | None, deadline: float, *, draft: bool = False,
    ) -> ExecutionReceipt:
        self._stage(run, RunStage.CONTEXT, deadline)
        if material is None or material.base is None or material.head is None:
            raise _ModeFailure("intent_material_unavailable")
        if artifacts is None:
            raise _ModeFailure("artifact_root_required")
        task_input = self._partition_task if shared_plan_enabled(self.config) else build_intent_task_input(
            review_snap, material.base, material.head, related_paths=tuple(self.config.intent.context_paths))
        if task_input is None:
            raise _ModeFailure("partition_plan_input_unavailable")
        marker = self.config.publish.b_marker
        code_first = not shared_plan_enabled(self.config) and self.config.intent.pipeline == "code-first-v2"
        perspectives = shared_plan_enabled(self.config) or self.config.intent.pipeline == "perspectives-v1"
        if perspectives:
            if shared_plan_enabled(self.config):
                self._prepare_partition_planner()
            else:
                self._prepare_intent_selector()
            input_fingerprint = intent_input_fingerprint(task_input, self.config, self.model_client,
                selection_client=self.intent_selection_client, planning_client=self.planning_client)
        elif code_first:
            if self.planning_client is None and self.model_client is not None:
                self.planning_client = build_planning_client(self.config.model.profile, self.config.model)
                if isinstance(self.planning_client, LlamaCppModelClient):
                    self.planning_client.request_cancelled = self._stop_event.is_set
            input_fingerprint = intent_input_fingerprint(task_input, self.config, self.model_client,
                planning_client=self.planning_client)
        else:
            input_fingerprint = intent_input_fingerprint(task_input, self.config, self.model_client)

        input_fingerprint = self._publication_input_key("intent", input_fingerprint)

        def source_guard(current: dict[str, Any]) -> bool:
            self._require_active(run, deadline)
            fresh = intent_sources_current(self.github_client, review_snap, current,
                timeout_seconds=min(10.0, deadline - time.monotonic()), before_request=lambda: self._active(run, deadline))
            self._require_active(run, deadline)
            return fresh

        if not (draft and not self.config.publish.draft_comments):
            reused = self._reuse_publication(run, "intent", review_snap, mode_sel, input_fingerprint, deadline, source_guard=source_guard)
            if reused is not None:
                return reused
        self._stage(run, RunStage.REVIEW, deadline)
        cached = self._cached_review(run, "intent")
        if cached is not None:
            task_input, result = cached
            from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
            formatter = format_partition_review_markdown
        else:
            if perspectives:
                from review_bot.adapters.markdown.intent_perspective_formatter import format_intent_perspective_markdown
                result = self._review_perspectives(task_input, run, deadline, self.config.output.language)
                formatter = format_intent_perspective_markdown
                if shared_plan_enabled(self.config):
                    from review_bot.adapters.markdown.partition_formatter import format_partition_review_markdown
                    formatter = format_partition_review_markdown
            elif code_first:
                from review_bot.adapters.markdown.code_intent_formatter import format_code_intent_markdown
                from review_bot.application.code_first_intent_review import run_code_first_review
                from review_bot.domain.code_first_intent_models import CodeFirstReviewLimits

                limits = CodeFirstReviewLimits(task_timeout_seconds=max(1, math.ceil(deadline - time.monotonic())))
                result = run_code_first_review(task_input, self.model_client, planning_client=self.planning_client,
                    before_stage=lambda: self._active(run, deadline), limits=limits)
                formatter = format_code_intent_markdown
            else:
                limits = IntentReviewLimits(task_timeout_seconds=max(1, math.ceil(deadline - time.monotonic())))
                result = run_review_b_pipeline(task_input, self.model_client, before_stage=lambda: self._active(run, deadline), limits=limits)
                formatter = format_intent_markdown
        self._stage(run, RunStage.VALIDATE, deadline)
        execution_status = "failed" if perspectives and result.status != "completed" else result.status
        core_receipt = ExecutionReceipt(2, "intent", mode_sel.selection_id, review_snap.snapshot_id, run.generation,
            result.run_key, execution_status, usage=({"calls": 0, "model_calls": 0, "input_tokens": 0, "output_tokens": 0} if cached is not None else result.usage),
            coverage=result.scope, limitations=result.limitations,
            model_delivery=({"status": "not_attempted"} if cached is not None else result.model_delivery),
            error_code=result.reason if execution_status == "failed" else None, b_result=result, input_fingerprint=input_fingerprint)
        reason = ("artifact_only" if self.config.publish.mode == "artifact_only" else "b_comments_disabled"
            if not self.config.publish.b_comments_enabled else "draft_comments_disabled"
            if draft and not self.config.publish.draft_comments else None)
        try:
            markdown = formatter(result, task_input, marker=marker,
                max_length=min(60000, self.config.output.max_comment_characters),
                publication_mode="comment" if reason is None else "artifact_only",
                **self._presentation_options("intent"))
            markdown = self._settings_notice(markdown)
            self._require_active(run, deadline)
            artifacts.save_intent(task_input, result, markdown)
            self._require_active(run, deadline)
            if result.status == "completed" and cached is None:
                self._save_cached_review(run, "intent", task_input, result)
        except (_RunAborted, _CommonFailure, GitHubAuthorizationError):
            raise
        except Exception as exc:
            self._require_active(run, deadline)
            error_code = f"intent_artifact_error:{type(exc).__name__}"
            return replace(core_receipt, execution_status="failed", error_code=error_code,
                publication=self._publish_failure_notice(run, "intent", review_snap, error_code, deadline, draft=draft))
        self._stage(run, RunStage.PUBLISH, deadline)
        if reason is not None:
            publication = PublicationReceipt(None, "not_attempted",
                publication_fingerprint(run.repository_id, run.pr_number, review_snap.head_sha, markdown), marker=marker, reason=reason)
            return replace(core_receipt, publication=publication)
        if self.github_client is None or not isinstance(getattr(self.github_client, "bot_login", None), str):
            return replace(core_receipt, execution_status="failed", error_code="github_bot_identity_unavailable")
        auth_before = getattr(self.github_client, "authorization_failures", None)
        authorization_lost = False
        try:
            publication = publish_review_comment(self.github_client, run.repository_id, run.pr_number,
                review_snap.head_sha, review_snap.target_base_sha, markdown, marker=marker,
                publication_guard=lambda: self._active(run, deadline),
                context_guard=lambda current: type(current.get("draft")) is bool and (current.get("draft") is not True or self.config.publish.draft_comments),
                source_guard=source_guard, strict_target=True, read_timeout_seconds=min(10.0, deadline - time.monotonic()))
        except PublicationAuthorizationError as exc:
            publication = exc.receipt
            authorization_lost = True
        self._require_active(run, deadline)
        auth_after = getattr(self.github_client, "authorization_failures", None)
        has_comment_id = type(publication.comment_id) is int and publication.comment_id > 0
        if authorization_lost or type(auth_before) is int and type(auth_after) is int and auth_after > auth_before:
            publication = replace(publication, status="unknown" if has_comment_id else "failed",
                reason="publication_authorization_lost", input_fingerprint=None)
            receipt = replace(core_receipt, execution_status="failed", publication=publication, error_code="publication_authorization_lost")
            self._abort_authorization_loss(run, artifacts, receipt, deadline)
        if publication.status != "succeeded" or not has_comment_id:
            stale = publication.status == "stale" or publication.reason in ("stale_head_sha", "stale_base_sha", "pr_not_open")
            reason = "invalid_comment_id" if publication.status == "succeeded" else publication.reason
            receipt = replace(core_receipt, execution_status="failed", publication=publication,
                error_code=f"Freshness check failed: {reason}" if stale else f"Publication failed: {reason}")
            if stale:
                self._publication_superseded(run, receipt, artifacts, deadline)
            return receipt
        return replace(core_receipt, publication=publication)

    def claim_and_execute_once(self) -> bool:
        """Claim and process at most one available run. Returns True if a run was processed."""
        self.prepare_models()
        run = self.job_store.claim_next_run(self.worker_id, self.lease_duration)
        if run is None:
            return False
        return self.process_run(run)

    def run_once(self) -> bool:
        """Alias for claim_and_execute_once for CLI and spec compatibility."""
        return self.claim_and_execute_once()

    def run_forever(self) -> None:
        """Poll and execute runs continuously until stopped."""
        logger.info(f"Worker {self.worker_id} started loop.")
        while not self._stop_event.is_set():
            processed = self.claim_and_execute_once()
            if not processed:
                self._stop_event.wait(timeout=self.poll_interval)
        logger.info(f"Worker {self.worker_id} stopped cleanly.")
