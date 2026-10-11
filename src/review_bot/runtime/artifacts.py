"""Private, atomic service-v2 output files; never persist authentication objects."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from review_bot.domain.envelope_v2 import RunRecord


class RunArtifacts:
    """One run's private outputs under runs/<repository>/<PR>/<run>/A."""

    def __init__(self, root: Path, run: RunRecord) -> None:
        if run.repository_id <= 0 or run.pr_number <= 0:
            raise ValueError("invalid_artifact_target")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run.run_id):
            raise ValueError("invalid_artifact_run_id")
        self.root = root / "runs" / str(run.repository_id) / str(run.pr_number) / run.run_id
        current = root
        for part in ("", "runs", str(run.repository_id), str(run.pr_number), run.run_id, "A"):
            if part:
                current = current / part
            if current.is_symlink():
                raise ValueError("unsafe_artifact_directory")
            current.mkdir(mode=0o700, parents=True, exist_ok=True)
            if current.stat().st_mode & 0o077:
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
        }

    def _write(self, relative_path: str, content: str) -> None:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts or relative.parent not in (Path("."), Path("A")):
            raise ValueError("invalid_artifact_path")
        target = self.root / relative_path
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
        self._write(relative_path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    def save_review(
        self, *, snapshot: Any, selection: Any, task_input: Any,
        base_run: Any, head_run: Any, result: Any, markdown: str, config_hash: str,
    ) -> None:
        self.manifest.update({
            "snapshot_id": snapshot.snapshot_id,
            "selection_id": selection.selection_id,
            "selected_modes": selection.selected_modes,
            "config_hash": config_hash,
            "fingerprint": result.input_hash,
        })
        self.write_json("snapshot.json", snapshot)
        self.write_json("selection.json", selection)
        self.write_json("A/input.json", task_input)
        self.write_json("A/analysis.json", {"base": asdict(base_run), "head": asdict(head_run)})
        self.write_json("A/delta.json", task_input.delta)
        self.write_json("A/scope.json", result.scope)
        self.write_json("A/result.json", result)
        self.write_json("A/usage.json", result.usage)
        self.write_json("A/validation.json", {"status": result.status, "reason": result.reason})
        self._write("A/comment.md", markdown)
        self.write_json("manifest.json", self.manifest)

    def finish(self, status: str, publication: Any = None) -> None:
        self.manifest["status"] = status
        if publication is not None:
            self.manifest["publication"] = asdict(publication)
        self.write_json("manifest.json", self.manifest)