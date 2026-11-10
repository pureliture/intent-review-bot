#!/usr/bin/env python3
"""Download missing locked PMD tools from HTTPS Maven Central URLs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOCKFILE = ROOT / "tooling" / "tools.lock.json"
DEFAULT_LIB_DIR = ROOT / "tooling" / "lib"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ToolFetchError(Exception):
    pass


def _verify_file(path: Path, name: str, spec: dict[str, object]) -> None:
    if path.is_symlink():
        raise ToolFetchError(f"Symlink rejected: {path}")
    if not path.is_file():
        raise ToolFetchError(f"Missing artifact: {name}")

    expected_size = spec.get("size_bytes")
    expected_sha = spec.get("sha256")
    if type(expected_size) is not int or expected_size < 0:
        raise ToolFetchError(f"Invalid locked size for {name}")
    if not isinstance(expected_sha, str) or not _SHA256_PATTERN.fullmatch(expected_sha):
        raise ToolFetchError(f"Invalid locked SHA-256 for {name}")

    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ToolFetchError(
            f"Size mismatch on {name}: expected {expected_size} bytes, got {actual_size} bytes"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha:
        raise ToolFetchError(f"SHA-256 mismatch on {name}: expected {expected_sha}, got {digest}")


def _download(url: str, destination: Path, name: str, spec: dict[str, object]) -> None:
    if urlsplit(url).scheme != "https" or not urlsplit(url).netloc:
        raise ToolFetchError(f"HTTPS URL required for {name}")
    expected_size = spec.get("size_bytes")
    expected_sha = spec.get("sha256")
    if type(expected_size) is not int or expected_size < 0:
        raise ToolFetchError(f"Invalid locked size for {name}")
    if not isinstance(expected_sha, str) or not _SHA256_PATTERN.fullmatch(expected_sha):
        raise ToolFetchError(f"Invalid locked SHA-256 for {name}")

    with urlopen(url, timeout=60) as response:
        final_url = response.geturl() if hasattr(response, "geturl") else url
        if urlsplit(final_url).scheme != "https":
            raise ToolFetchError(f"HTTPS redirect required for {name}")
        digest = hashlib.sha256()
        actual_size = 0
        with destination.open("xb") as output:
            while chunk := response.read(65536):
                actual_size += len(chunk)
                digest.update(chunk)
                output.write(chunk)

    if actual_size != expected_size:
        raise ToolFetchError(
            f"Size mismatch on {name}: expected {expected_size} bytes, got {actual_size} bytes"
        )
    actual_sha = digest.hexdigest()
    if actual_sha != expected_sha:
        raise ToolFetchError(f"SHA-256 mismatch on {name}: expected {expected_sha}, got {actual_sha}")


def fetch_tools(lockfile_path: Path, lib_dir: Path) -> int:
    try:
        lock = json.loads(lockfile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: could not read tools lock: {exc}", file=sys.stderr)
        return 1
    artifacts = lock.get("artifacts") if isinstance(lock, dict) else None
    if not isinstance(artifacts, dict) or not artifacts:
        print("Error: tools lock has no valid artifacts mapping", file=sys.stderr)
        return 1

    try:
        lib_dir.mkdir(parents=True, exist_ok=True)
        staged: list[tuple[Path, Path]] = []
        reports: list[str] = []
        with tempfile.TemporaryDirectory(prefix=".fetch-tools-", dir=lib_dir) as temporary_dir:
            staging_dir = Path(temporary_dir)
            for name in sorted(artifacts):
                spec = artifacts[name]
                if not isinstance(name, str) or not isinstance(spec, dict):
                    raise ToolFetchError("Invalid artifact entry in tools lock")
                if spec.get("category") == "ruleset":
                    continue
                relative_path = spec.get("relative_path")
                if (not isinstance(relative_path, str)
                    or Path(relative_path).parts[:2] != ("tooling", "lib")
                    or Path(relative_path).name != name
                    or Path(name).name != name):
                    raise ToolFetchError(f"Invalid tooling/lib path for {name}")
                target = lib_dir / name
                if target.exists() or target.is_symlink():
                    _verify_file(target, name, spec)
                    reports.append(f"[OK] Existing artifact verified: {name}")
                    continue

                url = spec.get("url")
                if not isinstance(url, str):
                    raise ToolFetchError(f"Missing HTTPS URL for {name}")
                staged_file = staging_dir / name
                _download(url, staged_file, name, spec)
                staged.append((staged_file, target))
                reports.append(f"[OK] Downloaded and verified: {name}")

            for staged_file, target in staged:
                os.replace(staged_file, target)
        for report in reports:
            print(report)
        return 0
    except (OSError, ToolFetchError, URLError, HTTPException, ValueError) as exc:
        print(f"Error: tool fetch failed: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    parser.add_argument("--lib-dir", type=Path, default=DEFAULT_LIB_DIR)
    args = parser.parse_args(argv)
    return fetch_tools(args.lockfile, args.lib_dir)


if __name__ == "__main__":
    raise SystemExit(main())
