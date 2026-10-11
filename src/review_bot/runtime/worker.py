"""Background Worker execution loop, lease heartbeat management, and generation fencing."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import socket
import tempfile
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.ollama import OLLAMA_PROFILE, SELECTOR_MODEL, OllamaDecisionClient, OllamaModelClient
from review_bot.application.classification import classify_pr
from review_bot.application.context import build_task_input
from review_bot.application.delta import build_delta
from review_bot.application.publication import publication_fingerprint, publish_review_comment
from review_bot.application.review_a import run_review_a_pipeline
from review_bot.application.selection import DeadlineModeSelector, LightweightModeSelector, ModelModeSelector
from review_bot.application.snapshot import assemble_snapshot_bundle
from review_bot.config import ReviewBotConfig
from review_bot.domain.envelope_v2 import PublicationReceipt, RunRecord
from review_bot.domain.models_v1 import AnalysisRun, Snapshot
from review_bot.domain.status import RunStage, RunStatus
from review_bot.ports import (
    GitHubClient,
    JobStore,
    ModelClient,
    ModeSelectorPort,
    RepositoryReader,
    StaticAnalyzer,
)
from review_bot.runtime.artifacts import RunArtifacts
from review_bot.runtime.model_clients import build_review_client

logger = logging.getLogger("review_bot.worker")


class WorkerLoop:
    """Worker daemon executing claimed jobs with lease heartbeats and generation fencing.

    Fast selection defaults to a local selector and never reuses model_client.
    The explicit ollama-local-128k profile composes Granite and, when enabled,
    a separate Tev selector. run_once prepares models before claiming a lease;
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
    ) -> None:
        self.config = config
        self.job_store = job_store
        self.repo_reader = repo_reader
        self.analyzer = analyzer
        self.github_client_factory = github_client_factory
        self.repo_reader_factory = repo_reader_factory
        self.artifact_root = artifact_root
        model_settings = getattr(config, "model", None)
        profile = getattr(model_settings, "profile", None)
        self.model_client = model_client if model_client is not None else build_review_client(profile, model_settings)
        self.github_client = github_client
        self._ollama_selector_client: OllamaDecisionClient | None = None
        self._models_prepared = False
        if (
            mode_selector is None and profile == OLLAMA_PROFILE
            and getattr(getattr(config, "execution", None), "selector_uses_model", False)
        ):
            self._ollama_selector_client = OllamaDecisionClient()
            mode_selector = ModelModeSelector(self._ollama_selector_client, model=SELECTOR_MODEL)
        selector = mode_selector if mode_selector is not None else LightweightModeSelector()
        self.mode_selector = selector if isinstance(selector, DeadlineModeSelector) else DeadlineModeSelector(selector)
        self.worker_id = (
            worker_id
            or f"{socket.gethostname()}-{os.getpid()}-{threading.get_ident()}"
        )
        self.poll_interval = poll_interval
        self.lease_duration = lease_duration

        self._stop_event = threading.Event()
        self._lease_lost = False
        self._current_run_id: str | None = None
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
        if isinstance(self.model_client, OllamaModelClient):
            self.model_client.warmup()
        if self._ollama_selector_client is not None:
            self._ollama_selector_client.warmup()
        self._models_prepared = True

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

        previous_github_client = self.github_client
        previous_repo_reader = self.repo_reader
        artifacts = None
        heartbeat_stop, heartbeat_thread = self._start_heartbeat(run_id)

        try:
            # 1. Stage: PREPARE, with installation/repository-bound dependencies.
            if not self._update_run(run, RunStatus.RUNNING, RunStage.PREPARE):
                return False
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

            if self.github_client:
                pr_data = self.github_client.get_pull_request(run.pr_number)
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
                    self.repo_reader = self.repo_reader_factory(run, pr_data, self.github_client)

            # 2. Stage: SELECT
            if not self._update_run(run, RunStatus.RUNNING, RunStage.SELECT):
                return False

            pr_snap = None
            if self.repo_reader:
                execution_cfg = getattr(self.config, "execution", None)
                available_modes = tuple(
                    getattr(execution_cfg, "available_modes", ("architecture",))
                )
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
                    selector_policy=getattr(execution_cfg, "selector_policy", "all_available"),
                    selector_uses_model=getattr(execution_cfg, "selector_uses_model", False),
                    mode_selector=self.mode_selector,
                )

                if review_snap.comparison_status != "comparable":
                    reason = "; ".join(
                        source.reason for source in review_snap.unavailable_sources
                        if source.source == "merge_base"
                    )
                    error = f"comparison_status=incomparable: {reason}"
                    logger.warning("Aborting run %s: %s", run_id, error)
                    self._update_run(run, RunStatus.FAILED, RunStage.SELECT, error=error)
                    return False

                if not mode_sel.selected_modes:
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
                }
                fingerprint = hashlib.sha256(json.dumps(semantic_input, sort_keys=True).encode()).hexdigest()
                run = replace(run, fingerprint=fingerprint)
                if not self._update_run(run, RunStatus.RUNNING, fingerprint=fingerprint):
                    return False
                marker = self.config.publish.a_marker
                if self.config.publish.mode == "comment" and self.github_client is not None and not (draft and not self.config.publish.draft_comments):
                    prior = self.job_store.get_current_publication(run, marker)
                    if isinstance(prior, PublicationReceipt) and prior.comment_id is not None:
                        existing = self.github_client.find_marker_comment(run.repository_id, run.pr_number, marker)
                        if (
                            isinstance(existing, dict) and existing.get("id") == prior.comment_id
                            and isinstance(existing.get("body"), str)
                            and publication_fingerprint(run.repository_id, run.pr_number, head_sha, existing["body"]) == prior.fingerprint
                        ):
                            return self._update_run(run, RunStatus.SKIPPED, reason="unchanged_input")

            # 3. Stage: ANALYZE
            if not self._update_run(run, RunStatus.RUNNING, RunStage.ANALYZE):
                return False

            base_run = None
            head_run = None
            effective_base_sha = pr_snap.base_sha if pr_snap else ""
            if self.analyzer and self.repo_reader and pr_snap:
                arch_cfg = getattr(self.config, "architecture", None)
                ruleset = getattr(arch_cfg, "ruleset_resource", "pmd/ruleset.xml")
                profile = getattr(
                    arch_cfg, "analysis_profile", "java21-jdk-only-v1"
                )
                analysis_runs: list[AnalysisRun] = []
                for sha in (effective_base_sha, head_sha):
                    if not self._check_fencing_and_lease(run_id, run.generation):
                        return False
                    with tempfile.TemporaryDirectory(prefix="review-bot-tree-") as tmp:
                        source_root = Path(tmp).resolve()
                        blobs = self.repo_reader.read_tree_blobs(sha)
                        if not self._check_fencing_and_lease(run_id, run.generation):
                            return False
                        for path, content in blobs.items():
                            relative_path = Path(path)
                            dest = source_root / relative_path
                            if (
                                relative_path.is_absolute()
                                or ".." in relative_path.parts
                                or not dest.resolve().is_relative_to(source_root)
                            ):
                                raise ValueError(f"Unsafe repository path: {path!r}")
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(content)
                        if not self._check_fencing_and_lease(run_id, run.generation):
                            return False
                        analysis_runs.append(self.analyzer.run_analysis(
                            sha, source_root, ruleset, profile,
                        ))
                base_run, head_run = analysis_runs

            # 4. Stage: COMPARE
            if not self._update_run(run, RunStatus.RUNNING, RunStage.COMPARE):
                return False

            delta = None
            if base_run and head_run and pr_snap:
                delta = build_delta(
                    base_run, head_run, rename_map=pr_snap.rename_map
                )

            # 5. Stage: CONTEXT
            if not self._update_run(run, RunStatus.RUNNING, RunStage.CONTEXT):
                return False

            task_input = None
            if self.repo_reader and delta and pr_snap:
                base_code = self.repo_reader.collect_code_snapshot(
                    effective_base_sha
                )
                if not self._check_fencing_and_lease(run_id, run.generation):
                    return False
                head_code = self.repo_reader.collect_code_snapshot(head_sha)
                core_snapshot = Snapshot(
                    pr=pr_snap, base=base_code, head=head_code
                )
                task_input = build_task_input(
                    core_snapshot, delta, classification
                )

            # 6. Stage: REVIEW
            if not self._update_run(run, RunStatus.RUNNING, RunStage.REVIEW):
                return False

            result = None
            if task_input is not None:
                result = run_review_a_pipeline(task_input, self.model_client)

            # 7. Stage: VALIDATE
            if not self._update_run(run, RunStatus.RUNNING, RunStage.VALIDATE):
                return False

            # 8. Stage: PUBLISH
            if not self._update_run(run, RunStatus.RUNNING, RunStage.PUBLISH):
                return False

            writes_comment = self.config.publish.mode == "comment" and not (draft and not self.config.publish.draft_comments)
            if writes_comment and self.github_client is None:
                self._update_run(run, RunStatus.FAILED, error="github_client_unavailable")
                return False
            if task_input is None or result is None:
                self._update_run(run, RunStatus.FAILED, error="review_result_unavailable")
                return False
            if not writes_comment and self.artifact_root is None:
                self._update_run(run, RunStatus.FAILED, error="artifact_root_required")
                return False

            marker = getattr(
                getattr(self.config, "publish", None),
                "a_marker",
                "<!-- review-bot:architecture:v1 -->",
            )
            markdown = format_review_markdown(
                result, task_input, marker=marker, max_length=self.config.output.max_comment_characters,
                artifact_notice="전체 A 자료는 비공개 서비스 결과 파일에 보존됩니다." if self.artifact_root is not None else None,
            )
            if not self._check_fencing_and_lease(run_id, run.generation):
                return False
            if self.artifact_root is not None:
                artifacts = RunArtifacts(self.artifact_root, run)
                artifacts.save_review(
                    snapshot=review_snap, selection=mode_sel, task_input=task_input,
                    base_run=base_run, head_run=head_run, result=result, markdown=markdown,
                    config_hash=self.config.config_hash,
                )
            if not writes_comment:
                receipt = PublicationReceipt(
                    comment_id=None, status="not_attempted",
                    fingerprint=publication_fingerprint(run.repository_id, run.pr_number, head_sha, markdown),
                    marker=marker, reason="artifact_only" if self.config.publish.mode == "artifact_only" else "draft_comments_disabled",
                )
                if not self.job_store.record_publication(
                    receipt, run.repository_id, run.pr_number, run_id=run_id,
                    worker_id=self.worker_id, generation=run.generation,
                ):
                    return False
                succeeded = self._update_run(run, RunStatus.SUCCEEDED, RunStage.PUBLISH, reason=receipt.reason)
                if succeeded and artifacts is not None:
                    artifacts.finish("succeeded", receipt)
                return succeeded
            def context_guard(current: dict[str, Any]) -> bool:
                current_labels = tuple(label.get("name", "") for label in current.get("labels", []))
                current_branch = current.get("head", {}).get("ref", "")
                return (
                    (current.get("title") or "") == title and (current.get("body") or "") == body
                    and classify_pr(current_branch, current_labels).review_mode == classification.review_mode
                    and (current.get("draft") is not True or self.config.publish.draft_comments)
                )

            receipt = publish_review_comment(
                github_client=self.github_client,
                repository_id=run.repository_id,
                pr_number=run.pr_number,
                expected_head_sha=head_sha,
                expected_base_sha=target_base_sha,
                body=markdown,
                marker=marker,
                publication_guard=lambda: self._check_fencing_and_lease(run_id, run.generation),
                **({"context_guard": context_guard} if self.github_client_factory is not None else {}),
            )
            if not self._check_fencing_and_lease(run_id, run.generation):
                return False
            has_comment_id = type(receipt.comment_id) is int and receipt.comment_id > 0
            if (has_comment_id or receipt.status == "unknown") and not self.job_store.record_publication(
                receipt, run.repository_id, run.pr_number,
                run_id=run_id, worker_id=self.worker_id, generation=run.generation,
            ):
                self._check_fencing_and_lease(run_id, run.generation)
                logger.info("Publication receipt rejected by CAS fence for %s", run_id)
                return False

            if receipt.status != "succeeded" or not has_comment_id:
                stale = receipt.status == "stale" or receipt.reason in (
                    "stale_head_sha", "stale_base_sha", "pr_not_open",
                )
                reason = "invalid_comment_id" if receipt.status == "succeeded" else receipt.reason
                error = f"Freshness check failed: {reason}" if stale else f"Publication failed: {reason}"
                terminal_status = RunStatus.SUPERSEDED if stale else RunStatus.FAILED
                if self._update_run(run, terminal_status, error=error) and artifacts is not None:
                    artifacts.finish(terminal_status.value.lower(), receipt)
                return False

            # Mark run SUCCEEDED
            succeeded = self._update_run(run, RunStatus.SUCCEEDED, RunStage.PUBLISH)
            if succeeded and artifacts is not None:
                artifacts.finish("succeeded", receipt)
            return succeeded

        except Exception as exc:
            error_code = f"worker_pipeline_error:{type(exc).__name__}"
            if self._update_run(run, RunStatus.FAILED, error=error_code):
                logger.error("Error processing run %s: %s", run_id, error_code)
            return False

        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=2.0)
            self._current_run_id = None
            if self.github_client_factory is not None:
                self.github_client = previous_github_client
            if self.repo_reader_factory is not None:
                self.repo_reader = previous_repo_reader

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
