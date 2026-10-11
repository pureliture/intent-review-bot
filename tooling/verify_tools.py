#!/usr/bin/env python3
"""Standalone CLI verification tool for PMD/CPD tools lockfile integrity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

KNOWN_JDK21_SEARCH_PATHS = (
    Path.home() / "Library/Java/JavaVirtualMachines/corretto-21.0.6/Contents/Home",
    Path("/Library/Java/JavaVirtualMachines/corretto-21.jdk/Contents/Home"),
    Path("/usr/lib/jvm/java-21-amazon-corretto"),
    Path("/usr/lib/jvm/java-21-openjdk-amd64"),
    Path("/usr/lib/jvm/java-21-openjdk"),
    Path("/opt/java/corretto-21"),
)


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 64 KiB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_java_candidate(explicit: str | None = None) -> Path | None:
    """Find a candidate java executable."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        if (p / "bin" / "java").is_file():
            return p / "bin" / "java"
        return None

    if "JAVA_HOME" in os.environ:
        home = Path(os.environ["JAVA_HOME"])
        if (home / "bin" / "java").is_file():
            return home / "bin" / "java"

    for search_home in KNOWN_JDK21_SEARCH_PATHS:
        candidate = search_home / "bin" / "java"
        if candidate.is_file():
            return candidate

    which_java = shutil.which("java")
    if which_java:
        return Path(which_java)

    return None


def verify_java_version(java_bin: Path, verbose: bool = False) -> tuple[bool, str]:
    """Execute java and verify major version is 21 in sanitized environment."""
    clean_env = {
        k: v for k, v in os.environ.items()
        if k not in (
            "JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS",
            "JAVA_OPTS", "GRADLE_OPTS", "CLASSPATH",
        )
    }
    clean_env["LC_ALL"] = "C.UTF-8"
    clean_env["LANG"] = "C.UTF-8"

    try:
        proc = subprocess.run(
            [str(java_bin), "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=clean_env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"Failed to execute java binary: {exc}"

    output = proc.stderr + "\n" + proc.stdout
    match = re.search(r"^\s*java\.version\s*=\s*(\S+)", output, re.MULTILINE)
    if not match:
        # Fallback to parsing -version directly
        match = re.search(r'version "(\d+)(?:\.|\+|-)', output)
        if not match:
            return False, f"Could not determine java.version from output: {output[:200]}"
        version_str = match.group(1)
        major = version_str
    else:
        version_str = match.group(1)
        major = version_str.split(".")[0]

    if major != "21":
        return False, f"Java major version is {major} ({version_str}), expected 21"

    if verbose:
        print(f"[OK] JDK 21 verified: {java_bin} (version {version_str})")
    return True, version_str


def verify_tools(
    lockfile_path: Path,
    lib_dir: Path,
    ruleset_path: Path,
    java_path: str | None = None,
    skip_java: bool = False,
    strict: bool = False,
    verbose: bool = False,
) -> int:
    """Main verification routine. Returns 0 on success, 1 on error."""
    if not lockfile_path.is_file():
        print(f"Error: Lockfile not found: {lockfile_path}", file=sys.stderr)
        return 1

    try:
        data = json.loads(lockfile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: Malformed lockfile JSON: {exc}", file=sys.stderr)
        return 1

    expected_keys = {"pmd_version", "java_version", "artifacts"}
    for k in expected_keys:
        if k not in data:
            print(f"Error: Missing required lockfile key '{k}'", file=sys.stderr)
            return 1

    if data["pmd_version"] != "7.3.0":
        print(f"Error: Unsupported pmd_version: {data['pmd_version']}", file=sys.stderr)
        return 1

    if data["java_version"] != "21":
        print(f"Error: Unsupported java_version: {data['java_version']}", file=sys.stderr)
        return 1

    artifacts = data.get("artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        print("Error: Lockfile has empty or invalid 'artifacts' mapping", file=sys.stderr)
        return 1

    failures: list[str] = []
    checked_files: set[str] = set()

    for name, spec in artifacts.items():
        expected_sha = spec.get("sha256")
        expected_size = spec.get("size_bytes")
        category = spec.get("category", "")
        relative_path = spec.get("relative_path")

        if category == "ruleset":
            target = ruleset_path
            if not target.is_file() and relative_path:
                project_root = lockfile_path.parent.parent
                cand = project_root / relative_path
                if cand.is_file():
                    target = cand
        else:
            target = lib_dir / name
            checked_files.add(name)

        if not target.exists():
            failures.append(f"Missing artifact: {name} (expected at {target})")
            continue

        if target.is_symlink():
            failures.append(f"Symlink rejected: {target} must not be a symbolic link")
            continue

        metadata = target.stat()
        if not stat.S_ISREG(metadata.st_mode):
            failures.append(f"Not a regular file: {target}")
            continue

        if metadata.st_size != expected_size:
            failures.append(
                f"Size mismatch on {name}: expected {expected_size} bytes, got {metadata.st_size} bytes"
            )
            continue

        actual_sha = compute_sha256(target)
        if actual_sha != expected_sha:
            failures.append(
                f"SHA-256 mismatch on {name}: expected {expected_sha}, got {actual_sha}"
            )
            continue

        if verbose:
            print(f"[OK] Artifact verified: {name} ({metadata.st_size} bytes, SHA-256 matches)")

    if strict and lib_dir.is_dir():
        for item in lib_dir.iterdir():
            if item.name not in checked_files and not item.name.startswith("."):
                failures.append(f"Strict mode violation: unexpected file in lib-dir: {item.name}")

    if not skip_java:
        java_bin = find_java_candidate(java_path)
        if not java_bin:
            failures.append("No JDK 21 binary found. Specify via --java or set JAVA_HOME.")
        else:
            ok, msg = verify_java_version(java_bin, verbose=verbose)
            if not ok:
                failures.append(f"JDK 21 verification failed: {msg}")

    if failures:
        print(f"FAILED: {len(failures)} verification check(s) failed:", file=sys.stderr)
        for fail in failures:
            print(f"  - {fail}", file=sys.stderr)
        return 1

    print(f"SUCCESS: Verified {len(artifacts)} locked artifacts and JDK 21 environment.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify PMD 7.3.0 and JDK 21 tools lockfile integrity.")
    project_root = Path(__file__).resolve().parent.parent
    parser.add_argument(
        "--lockfile",
        type=Path,
        default=project_root / "tooling" / "tools.lock.json",
        help="Path to tools.lock.json",
    )
    parser.add_argument(
        "--lib-dir",
        type=Path,
        default=project_root / "tooling" / "lib",
        help="Directory containing tool JARs",
    )
    parser.add_argument(
        "--ruleset",
        type=Path,
        default=project_root / "src" / "review_bot" / "resources" / "pmd" / "ruleset.xml",
        help="Path to ruleset.xml",
    )
    parser.add_argument(
        "--java",
        type=str,
        default=None,
        help="Path to java binary or JAVA_HOME",
    )
    parser.add_argument(
        "--skip-java",
        action="store_true",
        help="Skip JDK 21 verification",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if unlisted files exist in lib-dir",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose verification logs",
    )

    args = parser.parse_args()
    code = verify_tools(
        lockfile_path=args.lockfile,
        lib_dir=args.lib_dir,
        ruleset_path=args.ruleset,
        java_path=args.java,
        skip_java=args.skip_java,
        strict=args.strict,
        verbose=args.verbose,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
