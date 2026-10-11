"""Service-v2 artifacts stay atomic and private in one run directory."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_bot.domain.envelope_v2 import RunRecord
from review_bot.runtime.artifacts import RunArtifacts


def test_private_atomic_json_and_terminal_manifest(tmp_path: Path) -> None:
    run = RunRecord("run-example", 44, 99, 7, 1, "", "running", "prepare")
    artifacts = RunArtifacts(tmp_path / "state", run)
    artifacts.write_json("A/result.json", {"status": "not_run", "reason": "model_profile_unset"})
    artifacts.finish("succeeded")
    result = artifacts.root / "A/result.json"
    assert json.loads(result.read_text())["reason"] == "model_profile_unset"
    assert json.loads((artifacts.root / "manifest.json").read_text())["publication"] is None
    assert result.stat().st_mode & 0o077 == 0
    assert artifacts.root.stat().st_mode & 0o077 == 0
    assert (artifacts.root / "A").stat().st_mode & 0o077 == 0
    assert list(result.parent.iterdir()) == [result]


@pytest.mark.parametrize("target", ["../escape.json", "/private/escape.json", "A/../../escape.json", "nested/result.json"])
def test_paths_cannot_escape_the_run(tmp_path: Path, target: str) -> None:
    run = RunRecord("run-example", 44, 99, 7, 1, "", "running", "prepare")
    artifacts = RunArtifacts(tmp_path / "state", run)
    with pytest.raises(ValueError, match="invalid_artifact_path"):
        artifacts.write_json(target, {"private": "source"})
    assert not (tmp_path / "state/runs/99/7/escape.json").exists()


@pytest.mark.parametrize("run_id", ["../escaped", "/escaped", "has/slash"])
def test_unsafe_run_id_rejected(tmp_path: Path, run_id: str) -> None:
    run = RunRecord(run_id, 44, 99, 7, 1, "", "running", "prepare")
    with pytest.raises(ValueError, match="invalid_artifact_run_id"):
        RunArtifacts(tmp_path / "state", run)


def test_nonprivate_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "shared"
    root.mkdir(mode=0o755)
    run = RunRecord("run-example", 44, 99, 7, 1, "", "running", "prepare")
    with pytest.raises(ValueError, match="artifact_directory_must_be_private"):
        RunArtifacts(root, run)