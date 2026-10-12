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
from dataclasses import replace
from pathlib import Path
from typing import Any

from review_bot.adapters.github.auth import GitHubAppAuthError
from review_bot.adapters.github.intent_sources import collect_intent_sources, intent_sources_current
from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.markdown.intent_formatter import format_intent_markdown
from review_bot.adapters.ollama import OLLAMA_PROFILE, SELECTOR_MODEL, OllamaDecisionClient, OllamaModelClient
from review_bot.application.classification import classify_pr
from review_bot.application.context import build_task_input
from review_bot.application.delta import build_delta
from review_bot.application.intent_context import IntentInputError, build_intent_task_input
from review_bot.application.publication import PublicationAuthorizationError, publication_fingerprint, publish_review_comment
from review_bot.application.review_a import run_review_a_pipeline
from review_bot.application.review_b import run_review_b_pipeline
from review_bot.application.selection import DeadlineModeSelector, LightweightModeSelector, ModelModeSelector
from review_bot.application.snapshot import assemble_snapshot_bundle, collect_snapshot_material
from review_bot.config import ReviewBotConfig
from review_bot.domain.envelope_v2 import ExecutionReceipt, PublicationReceipt, RunRecord
from review_bot.domain.models_b import IntentReviewLimits
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
from review_bot.runtime.model_clients import build_review_client

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
        analyzer_factory: Callable[[], StaticAnalyzer] | None = None,
    ) -> None:
        self.config = config
        self.job_store = job_store
        self.repo_reader = repo_reader
        self.analyzer = analyzer
        self.analyzer_factory = analyzer_factory
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
        deadline = time.monotonic() + getattr(getattr(self.config, "execution", None), "task_timeout_seconds", 900)
        receipts: dict[str, ExecutionReceipt] = {}
        heartbeat_stop, heartbeat_thread = self._start_heartbeat(run_id)

        try:
            self.config.publish.validate_b_marker()
            # 1. Stage: PREPARE, with installation/repository-bound dependencies.
            self._stage(run, RunStage.PREPARE, deadline)
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

            # 2. Stage: SELECT
            self._stage(run, RunStage.SELECT, deadline)

            pr_snap = None
            material = None
            if self.repo_reader:
                execution_cfg = getattr(self.config, "execution", None)
                available_modes = tuple(
                    getattr(execution_cfg, "available_modes", ("architecture",))
                )
                if "intent" in available_modes:
                    if not pr_data:
                        raise RuntimeError("intent_pr_source_unavailable")
                    sources = collect_intent_sources(self.github_client, repo_name, run.pr_number, pr_data,
                        timeout_seconds=min(10.0, deadline - time.monotonic()),
                        before_request=lambda: self._active(run, deadline))
                    material = collect_snapshot_material(self.repo_reader, target_base_sha, head_sha,
                        evidence=sources.evidence, unavailable_sources=sources.unavailable_sources,
                        related_paths=tuple(self.config.intent.context_paths), captured_at=sources.captured_at,
                        as_of=sources.as_of, before_read=lambda: self._active(run, deadline))
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
                    selector_policy=getattr(execution_cfg, "selector_policy", "all_available"),
                    selector_uses_model=getattr(execution_cfg, "selector_uses_model", False),
                    mode_selector=self.mode_selector,
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

            if pr_snap is None:
                self._stage(run, RunStage.PUBLISH, deadline)
                raise _ModeFailure("review_result_unavailable")
            self._require_active(run, deadline)
            if self.artifact_root is not None:
                artifacts = RunArtifacts(self.artifact_root, run)
                artifacts.save_common(snapshot=review_snap, selection=mode_sel, config_hash=self.config.config_hash)
            for mode in mode_sel.selected_modes:
                self._require_active(run, deadline)
                try:
                    if mode == "architecture":
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
                        run.generation, run.fingerprint, "failed", error_code=error_code)
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
                artifacts.finish(terminal_status.value, publication)
            stage = RunStage.PUBLISH if not failed or any(receipt.publication for receipt in failed) else None
            applied = self._update_run(run, terminal_status, stage, error=error, reason=reason)
            return applied and not failed

        except _RunAborted:
            return False

        except Exception as exc:
            error_code = str(exc) if isinstance(exc, (_ModeFailure, _CommonFailure, IntentInputError)) else f"worker_pipeline_error:{type(exc).__name__}"
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

        input_fingerprint = architecture_input_fingerprint(review_snap, classification, self.config, self.model_client, title=title, body=body)
        if not (draft and not self.config.publish.draft_comments):
            reused = self._reuse_publication(run, "architecture", review_snap, mode_sel, input_fingerprint, deadline, context_guard=context_guard)
            if reused is not None:
                return reused
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
                base_code = self.repo_reader.collect_code_snapshot(effective_base_sha)
                self._require_active(run, deadline)
                head_code = self.repo_reader.collect_code_snapshot(head_sha)
            except _RunAborted:
                raise
            except Exception as exc:
                raise _CommonFailure(f"worker_pipeline_error:{type(exc).__name__}") from None
            self._require_active(run, deadline)
            task_input = build_task_input(Snapshot(pr=pr_snap, base=base_code, head=head_code), delta, classification)
        self._stage(run, RunStage.REVIEW, deadline)
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
        markdown = format_review_markdown(result, task_input, marker=marker,
            max_length=self.config.output.max_comment_characters,
            artifact_notice="전체 A 자료는 비공개 서비스 결과 파일에 보존됩니다." if artifacts is not None else None)
        self._require_active(run, deadline)
        if artifacts is not None:
            artifacts.save_review(snapshot=review_snap, selection=mode_sel, task_input=task_input,
                base_run=base_run, head_run=head_run, result=result, markdown=markdown, config_hash=self.config.config_hash)
        self._require_active(run, deadline)
        if not writes_comment:
            publication = PublicationReceipt(None, "not_attempted",
                publication_fingerprint(run.repository_id, run.pr_number, head_sha, markdown), marker=marker,
                reason="artifact_only" if self.config.publish.mode == "artifact_only" else "draft_comments_disabled")
        else:
            auth_before = getattr(self.github_client, "authorization_failures", None)
            publication = publish_review_comment(github_client=self.github_client, repository_id=run.repository_id,
                pr_number=run.pr_number, expected_head_sha=head_sha, expected_base_sha=review_snap.target_base_sha,
                body=markdown, marker=marker, publication_guard=lambda: self._active(run, deadline),
                **({"context_guard": context_guard} if self.github_client_factory is not None else {}))
            self._require_active(run, deadline)
            auth_after = getattr(self.github_client, "authorization_failures", None)
            if "intent" in mode_sel.selected_modes and type(auth_before) is int and type(auth_after) is int and auth_after > auth_before:
                raise GitHubAppAuthError("Publication authorization lost")
            has_comment_id = type(publication.comment_id) is int and publication.comment_id > 0
            if publication.status != "succeeded" or not has_comment_id:
                stale = publication.status == "stale" or publication.reason in ("stale_head_sha", "stale_base_sha", "pr_not_open")
                reason = "invalid_comment_id" if publication.status == "succeeded" else publication.reason
                error = f"Freshness check failed: {reason}" if stale else f"Publication failed: {reason}"
                receipt = ExecutionReceipt(2, "architecture", mode_sel.selection_id, review_snap.snapshot_id,
                    run.generation, result.run_key, "failed", a_result=result, usage=result.usage,
                    error_code=error, publication=publication, input_fingerprint=input_fingerprint)
                if stale:
                    self._publication_superseded(run, receipt, artifacts, deadline)
                return receipt
        analysis_failed = any(getattr(value, "status", None) in ("failed", "missing_report") for value in (base_run, head_run))
        status = "failed" if analysis_failed or result.status == "failed" else result.status
        return ExecutionReceipt(2, "architecture", mode_sel.selection_id, review_snap.snapshot_id,
            run.generation, result.run_key, status, a_result=result, usage=result.usage,
            coverage=result.scope, limitations=tuple(task_input.limitations),
            model_delivery={"status": "sent" if result.usage and result.usage.get("calls", 0) else "not_attempted" if self.model_client is None else "unknown"},
            error_code="analysis_failed" if analysis_failed else result.reason if status == "failed" else None,
            publication=publication, input_fingerprint=input_fingerprint)

    def _execute_intent(
        self, run: RunRecord, review_snap: Any, mode_sel: Any, material: Any,
        artifacts: RunArtifacts | None, deadline: float, *, draft: bool = False,
    ) -> ExecutionReceipt:
        self._stage(run, RunStage.CONTEXT, deadline)
        if material is None or material.base is None or material.head is None:
            raise _ModeFailure("intent_material_unavailable")
        if artifacts is None:
            raise _ModeFailure("artifact_root_required")
        task_input = build_intent_task_input(review_snap, material.base, material.head,
            related_paths=tuple(self.config.intent.context_paths))
        marker = self.config.publish.b_marker
        input_fingerprint = intent_input_fingerprint(task_input, self.config, self.model_client)

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
        limits = IntentReviewLimits(task_timeout_seconds=max(1, math.ceil(deadline - time.monotonic())))
        result = run_review_b_pipeline(task_input, self.model_client, before_stage=lambda: self._active(run, deadline), limits=limits)
        self._stage(run, RunStage.VALIDATE, deadline)
        core_receipt = ExecutionReceipt(2, "intent", mode_sel.selection_id, review_snap.snapshot_id, run.generation,
            result.run_key, result.status, usage=result.usage,
            coverage=result.scope, limitations=result.limitations, model_delivery=result.model_delivery,
            error_code=result.reason if result.status == "failed" else None, b_result=result, input_fingerprint=input_fingerprint)
        reason = ("artifact_only" if self.config.publish.mode == "artifact_only" else "b_comments_disabled"
            if not self.config.publish.b_comments_enabled else "draft_comments_disabled"
            if draft and not self.config.publish.draft_comments else result.reason if result.status != "completed" else None)
        try:
            markdown = format_intent_markdown(result, task_input, marker=marker,
                max_length=min(60000, self.config.output.max_comment_characters),
                publication_mode="comment" if reason is None else "artifact_only")
            self._require_active(run, deadline)
            artifacts.save_intent(task_input, result, markdown)
        except (OSError, ValueError, TypeError) as exc:
            self._require_active(run, deadline)
            return replace(core_receipt, execution_status="failed", error_code=f"intent_artifact_error:{type(exc).__name__}")
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
            artifacts.save_receipt(receipt)
            if has_comment_id:
                self._record_publication(run, publication, deadline)
            raise _CommonFailure("publication_authorization_lost")
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
