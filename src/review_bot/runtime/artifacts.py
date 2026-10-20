"""Private, atomic service-v2 output files; never persist authentication objects."""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from review_bot.domain.envelope_v2 import RunRecord


def _intent_artifact_values(task_input: Any, result: Any) -> dict[str, Any]:
    values = {"input.json": task_input, "result.json": result, "scope.json": result.scope,
        "usage.json": {"total": result.usage, "stages": [asdict(stage) for stage in result.stages]},
        "validation.json": {"status": result.status, "reason": result.reason, "errors": result.validation_errors,
            "stages": [{"stage": stage.stage, "status": stage.status, "reason": stage.reason} for stage in result.stages]}}
    if result.schema_version == 7:
        values.update({"partition.json": result.partition, "perspectives.json": result.perspectives})
    elif result.schema_version == 3:
        from review_bot.domain.intent_perspectives import PERSPECTIVES

        values.update({"catalog.json": {"version": result.catalog_version, "hash": result.catalog_hash,
            "perspectives": [asdict(item) for item in PERSPECTIVES]},
            "selection.json": result.selection, "perspectives.json": {
            "effective": result.perspectives, "history": result.perspective_history},
            "judgements.json": {"current": result.judgements, "history": result.judgement_history},
            "context.json": {"gaps": result.context_gaps, "history": result.context_history, "requests": result.requests},
            "passes.json": result.passes})
    elif result.schema_version == 2:
        values.update({"plan.json": result.plan, "observations.json": result.group_observations,
            "intents.json": result.reconstructed_intents, "relations.json": result.relations})
    else:
        values.update({"goals.json": result.goals, "observations.json": result.observations})
    return values


class RunArtifacts:
    """Shared private originals with independent A/ and B/ mode outputs."""

    def __init__(self, root: Path, run: RunRecord) -> None:
        if run.repository_id <= 0 or run.pr_number <= 0:
            raise ValueError("invalid_artifact_target")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run.run_id):
            raise ValueError("invalid_artifact_run_id")
        root = _intent_path(root)
        self.root = root / "runs" / str(run.repository_id) / str(run.pr_number) / run.run_id
        current = root
        for part in ("", "runs", str(run.repository_id), str(run.pr_number), run.run_id):
            if part:
                current = current / part
            if current.is_symlink():
                raise ValueError("unsafe_artifact_directory")
            current.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = current.lstat()
            if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
                raise ValueError("artifact_directory_must_be_private")
        self.manifest: dict[str, Any] = {
            "schema_version": 2,
            "run_id": run.run_id,
            "installation_id": run.installation_id,
            "repository_id": run.repository_id,
            "pr_number": run.pr_number,
            "generation": run.generation,
            "status": "running",
            "publication": None,
            "modes": {},
        }

    def _write(self, relative_path: str, content: str) -> None:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.parent not in (Path("."), Path("A"), Path("B")):
            raise ValueError("invalid_artifact_path")
        _intent_directory(self.root)
        _intent_directory(self.root / relative.parent)
        target = self.root / relative_path
        if target.exists() or target.is_symlink():
            metadata = target.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ValueError("unsafe_artifact_file")
        fd, temp_path = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, target)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def write_json(self, relative_path: str, value: Any) -> None:
        data = asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
        self._write(relative_path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def save_common(self, *, snapshot: Any, selection: Any, config_hash: str) -> None:
        self.manifest.update({"snapshot_id": snapshot.snapshot_id, "selection_id": selection.selection_id,
            "selected_modes": selection.selected_modes, "config_hash": config_hash})
        self.write_json("snapshot.json", snapshot)
        self.write_json("selection.json", selection)
        self.write_json("manifest.json", self.manifest)

    def save_execution_plan(self, plan: Any) -> None:
        self.write_json("execution-plan.json", plan)
        self.manifest["planning"] = {"plan_id": plan.partition.get("plan_id"), "status": plan.status,
            "reason": plan.reason, "usage": plan.usage, "model_delivery": plan.model_delivery}
        self.write_json("manifest.json", self.manifest)

    def save_review(
        self, *, snapshot: Any, selection: Any, task_input: Any,
        base_run: Any, head_run: Any, result: Any, markdown: str, config_hash: str,
    ) -> None:
        self.save_common(snapshot=snapshot, selection=selection, config_hash=config_hash)
        self.manifest["fingerprint"] = result.input_hash
        self.manifest["modes"]["architecture"] = {"input_hash": result.input_hash,
            "input_fingerprint": result.run_key, "status": result.status, "reason": result.reason,
            "publication": None}
        self.write_json("A/input.json", task_input)
        self.write_json("A/analysis.json", {"base": asdict(base_run), "head": asdict(head_run)})
        self.write_json("A/delta.json", task_input.delta)
        self.write_json("A/scope.json", result.scope)
        self.write_json("A/result.json", result)
        self.write_json("A/usage.json", result.usage)
        if result.schema_version == 7:
            self.write_json("A/partition.json", result.partition)
        self.write_json("A/validation.json", {"status": result.status, "reason": result.reason})
        self._write("A/comment.md", markdown)
        self.write_json("manifest.json", self.manifest)

    def save_intent(self, task_input: Any, result: Any, markdown: str) -> None:
        _intent_directory(self.root / "B")
        values = _intent_artifact_values(task_input, result)
        for name, value in values.items():
            self.write_json("B/" + name, value)
        self._write("B/comment.md", markdown)
        self.manifest["modes"]["intent"] = {"input_hash": task_input.input_hash,
            "input_fingerprint": result.run_key, "prompt_version": result.prompt_version,
            "prompt_hash": result.prompt_hash, "status": result.status, "reason": result.reason,
            "publication": None}
        if result.schema_version in (2, 3, 7):
            self.manifest["modes"]["intent"]["result_schema_version"] = result.schema_version
        self.write_json("manifest.json", self.manifest)

    def save_receipt(self, receipt: Any) -> None:
        directory = {"architecture": "A", "intent": "B"}.get(receipt.mode)
        if directory is None:
            raise ValueError("invalid_artifact_mode")
        _intent_directory(self.root / directory)
        self.write_json(directory + "/receipt.json", receipt)
        self.save_receipt_manifest(receipt)

    def save_receipt_manifest(self, receipt: Any) -> None:
        record = self.manifest["modes"].setdefault(receipt.mode, {})
        record.update({"input_fingerprint": receipt.input_fingerprint, "core_fingerprint": receipt.fingerprint,
            "reused": receipt.reused, "execution_status": receipt.execution_status,
            "error_code": receipt.error_code, "usage": receipt.usage,
            "model_delivery": receipt.model_delivery,
            "publication": asdict(receipt.publication) if receipt.publication else None})
        if receipt.reused:
            record.update({"status": "not_run", "reason": "unchanged_input"})
        self.write_json("manifest.json", self.manifest)

    def finish(self, status: str, publication: Any = None) -> None:
        self.manifest["status"] = status
        if publication is not None:
            self.manifest["publication"] = asdict(publication)
        self.write_json("manifest.json", self.manifest)


def _intent_path(path: Path) -> Path:
    if ".." in path.parts or any(ord(char) < 32 or ord(char) == 127 for char in os.fspath(path)):
        raise ValueError("unsafe_intent_path")
    path = path.absolute()
    if path == Path("/") or Path.home().is_relative_to(path):
        raise ValueError("unsafe_intent_path")
    for parent in reversed(path.parents):
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            continue
        sticky_root = metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid not in (0, os.geteuid()) or (metadata.st_mode & 0o022 and not sticky_root):
            raise ValueError("unsafe_intent_parent")
    return path


def _intent_directory(path: Path) -> Path:
    path = _intent_path(path)
    for directory in (*reversed(path.parents), path):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        metadata = directory.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("unsafe_intent_directory")
    metadata = path.lstat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise ValueError("intent_directory_must_be_private")
    return path


def read_private_intent_input(path: Path, *, max_bytes: int) -> str:
    """Bounded read of an owned, private regular file, never a link or device."""
    path = _intent_path(path)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) not in (0o400, 0o600) or metadata.st_size > max_bytes:
        raise ValueError("unsafe_intent_input")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_mode, opened.st_nlink) != (metadata.st_dev, metadata.st_ino, metadata.st_uid, metadata.st_mode, metadata.st_nlink):
            raise ValueError("unsafe_intent_input")
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("intent_input_limit_exceeded")
    return data.decode("utf-8")


class IntentArtifacts:
    """Fresh private evaluation outputs, isolated from existing A/service files."""

    def __init__(self, root: Path, *, mode="intent") -> None:
        if mode not in ("intent", "architecture"):
            raise ValueError("invalid_private_artifact_mode")
        self.mode = mode
        self.root = _intent_directory(root)
        self.directory = self.root / ("B" if mode == "intent" else "A")
        manifest = self.root / "manifest.json"
        if manifest.exists() or manifest.is_symlink() or self.directory.exists() or self.directory.is_symlink():
            raise ValueError("intent_output_already_exists")
        self.directory.mkdir(mode=0o700)
        _intent_directory(self.directory)

    def _write(self, directory: Path, name: str, content: str) -> None:
        if directory not in (self.root, self.directory) or Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("invalid_intent_artifact_path")
        _intent_directory(directory)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        pending = f".pending-{uuid4().hex}"
        created = False
        try:
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
            created = True
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(pending, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
            os.fsync(directory_fd)
        finally:
            if created:
                os.unlink(pending, dir_fd=directory_fd)
            os.close(directory_fd)

    def save_execution_plan(self, task_input: Any, plan: Any) -> None:
        self._write(self.root, "planning-input.json", json.dumps(asdict(task_input), ensure_ascii=False, indent=2) + "\n")
        self._write(self.root, "execution-plan.json", json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n")

    def save(self, task_input: Any, result: Any, markdown: str) -> None:
        publication = {"status": "not_attempted", "reason": "artifact_only"}
        values = _intent_artifact_values(task_input, result)
        for name, value in values.items():
            data = asdict(value) if is_dataclass(value) else value
            self._write(self.directory, name, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        self._write(self.directory, "comment.md", markdown)
        manifest = {"schema_version": 2, "snapshot_id": result.snapshot_id, "selected_modes": [self.mode],
            "status": "failed" if result.status == "failed" else "succeeded", "publication": publication,
            "modes": {self.mode: {"input_hash": task_input.input_hash, "input_fingerprint": result.run_key,
                "prompt_version": result.prompt_version, "prompt_hash": result.prompt_hash,
                "status": result.status, "reason": result.reason, "publication": publication}}}
        if result.schema_version in (3, 7):
            manifest["status"] = "succeeded" if result.status == "completed" or (
                result.schema_version == 7 and result.status == "measured") else "failed"
            if result.reason == "mode_not_selected_in_shared_plan":
                manifest["status"] = "skipped"
        if result.schema_version in (2, 3, 7):
            manifest["modes"][self.mode]["result_schema_version"] = result.schema_version
        self._write(self.root, "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
