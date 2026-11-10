from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "tooling") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "tooling"))

import fetch_tools as cli_fetch_tools


class _Response:
    def __init__(self, content: bytes, url: str = "https://repo.maven.apache.org/artifact.jar"):
        self._stream = io.BytesIO(content)
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self) -> str:
        return self._url

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


def _lock(tmp_path: Path, content: bytes = b"correct") -> tuple[Path, Path, str]:
    lockfile = tmp_path / "tools.lock.json"
    lib_dir = tmp_path / "tooling" / "lib"
    lib_dir.mkdir(parents=True)
    name = "pmd-test.jar"
    lockfile.write_text(json.dumps({
        "artifacts": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "category": "pmd_runtime",
                "relative_path": f"tooling/lib/{name}",
                "url": "https://repo.maven.apache.org/maven2/test/pmd-test.jar",
            },
        },
    }), encoding="utf-8")
    return lockfile, lib_dir, name


def test_fetch_tools_rejects_mismatch_without_writing(tmp_path, monkeypatch, capsys):
    lockfile, lib_dir, name = _lock(tmp_path)
    monkeypatch.setattr(
        cli_fetch_tools,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"wrong!"),
    )

    assert cli_fetch_tools.fetch_tools(lockfile, lib_dir) == 1
    assert not (lib_dir / name).exists()
    assert list(lib_dir.iterdir()) == []
    assert "mismatch" in capsys.readouterr().err


def test_fetch_tools_downloads_and_verifies_missing_file(tmp_path, monkeypatch, capsys):
    lockfile, lib_dir, name = _lock(tmp_path)
    monkeypatch.setattr(
        cli_fetch_tools,
        "urlopen",
        lambda *_args, **_kwargs: _Response(b"correct"),
    )

    assert cli_fetch_tools.fetch_tools(lockfile, lib_dir) == 0
    assert (lib_dir / name).read_bytes() == b"correct"
    assert f"Downloaded and verified: {name}" in capsys.readouterr().out


def test_fetch_tools_rejects_existing_corrupt_file(tmp_path, monkeypatch):
    lockfile, lib_dir, name = _lock(tmp_path)
    bad_content = b"corrupt"
    (lib_dir / name).write_bytes(bad_content)

    def unexpected_download(*_args, **_kwargs):
        pytest.fail("existing files must be checked instead of redownloaded")

    monkeypatch.setattr(cli_fetch_tools, "urlopen", unexpected_download)

    assert cli_fetch_tools.fetch_tools(lockfile, lib_dir) == 1
    assert (lib_dir / name).read_bytes() == bad_content
