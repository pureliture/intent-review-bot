"""Bot-owned PMD 7.3.0 and CPD tool provisioning, checksum locking, and JDK 21 discovery."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KNOWN_JDK21_SEARCH_PATHS = (
    Path.home() / "Library/Java/JavaVirtualMachines/corretto-21.0.6/Contents/Home",
    Path("/Library/Java/JavaVirtualMachines/corretto-21.jdk/Contents/Home"),
    Path("/usr/lib/jvm/java-21-amazon-corretto"),
    Path("/usr/lib/jvm/java-21-openjdk-amd64"),
    Path("/usr/lib/jvm/java-21-openjdk"),
    Path("/opt/java/corretto-21"),
)


class ToolVerificationError(RuntimeError):
    """Raised when static analysis tool verification fails."""

    def __init__(self, reason: str, details: str, path: Path | None = None) -> None:
        self.reason = reason
        self.details = details
        self.path = path
        super().__init__(f"Tool verification failed [{reason}]: {details}")


@dataclass(frozen=True)
class ArtifactLock:
    """Locked specification of an analysis artifact."""

    name: str
    sha256: str
    size_bytes: int
    category: str
    relative_path: str | None = None


@dataclass(frozen=True)
class ToolsLock:
    """Locked specification of all tools and runtime requirements."""

    pmd_version: str
    java_version: str
    artifacts: dict[str, ArtifactLock]
    jdk_metadata: dict[str, Any]


@dataclass(frozen=True)
class JdkIdentity:
    """Verified JDK 21 identity and paths."""

    executable: Path
    java_home: Path
    version: str
    major_version: int
    vendor: str
    release_hash: str | None
    executable_hash: str


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file in 64 KiB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_tools_lock(lock_path: Path | str | None = None) -> ToolsLock:
    """Load and validate tooling/tools.lock.json against schema contract."""
    if lock_path is None:
        # Default relative to project root
        current = Path(__file__).resolve()
        # file is in src/review_bot/adapters/pmd/ -> project root is parents[3]
        project_root = current.parents[4]
        lock_file = project_root / "tooling" / "tools.lock.json"
    else:
        lock_file = Path(lock_path)

    if not lock_file.is_file():
        raise ToolVerificationError(
            "missing_lockfile",
            f"Lockfile does not exist: {lock_file}",
            path=lock_file,
        )

    try:
        raw = json.loads(lock_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ToolVerificationError(
            "invalid_lockfile_json",
            f"Could not parse lockfile JSON: {exc}",
            path=lock_file,
        ) from exc

    for key in ("pmd_version", "java_version", "artifacts"):
        if key not in raw:
            raise ToolVerificationError(
                "schema_contract_violation",
                f"Missing required key '{key}' in tools.lock.json",
                path=lock_file,
            )

    if raw["pmd_version"] != "7.3.0":
        raise ToolVerificationError(
            "version_mismatch",
            f"Expected PMD version 7.3.0, got {raw['pmd_version']}",
            path=lock_file,
        )

    if raw["java_version"] != "21":
        raise ToolVerificationError(
            "version_mismatch",
            f"Expected Java version 21, got {raw['java_version']}",
            path=lock_file,
        )

    artifacts_dict: dict[str, ArtifactLock] = {}
    for name, item in raw.get("artifacts", {}).items():
        if not isinstance(item, dict):
            continue
        artifacts_dict[name] = ArtifactLock(
            name=name,
            sha256=item.get("sha256", ""),
            size_bytes=int(item.get("size_bytes", 0)),
            category=item.get("category", ""),
            relative_path=item.get("relative_path"),
        )

    jdk_meta = raw.get("jdk_identity") or raw.get("jdk") or {}
    return ToolsLock(
        pmd_version=raw["pmd_version"],
        java_version=raw["java_version"],
        artifacts=artifacts_dict,
        jdk_metadata=jdk_meta,
    )


def get_clean_env(java_home: Path | None = None) -> dict[str, str]:
    """Return a pristine environment dictionary stripping JVM injection variables and secrets."""
    hazardous = {
        "JAVA_TOOL_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "_JAVA_OPTIONS",
        "JAVA_OPTS",
        "GRADLE_OPTS",
        "CLASSPATH",
        "GITHUB_TOKEN",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_WEBHOOK_SECRET",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "MODEL_API_KEY",
    }
    clean = {k: v for k, v in os.environ.items() if k not in hazardous}
    if java_home is not None:
        clean["JAVA_HOME"] = str(java_home)
    clean["LC_ALL"] = "C.UTF-8"
    clean["LANG"] = "C.UTF-8"
    return clean


def verify_jdk(
    java_executable: Path | str | None = None,
    java_home: Path | str | None = None,
) -> JdkIdentity:
    """Probe and verify that the target Java runtime is JDK 21 (Corretto / Adoptium)."""
    candidate_bin: Path | None = None
    candidate_home: Path | None = None

    if java_executable is not None:
        p = Path(java_executable)
        if not p.is_file():
            raise ToolVerificationError("jdk_not_found", f"Specified Java executable not found: {p}", path=p)
        candidate_bin = p
        candidate_home = p.parent.parent
    elif java_home is not None:
        h = Path(java_home)
        b = h / "bin" / "java"
        if not b.is_file():
            raise ToolVerificationError("jdk_not_found", f"Java binary not found in specified JAVA_HOME: {b}", path=b)
        candidate_bin = b
        candidate_home = h
    elif "JAVA_HOME" in os.environ:
        h = Path(os.environ["JAVA_HOME"])
        b = h / "bin" / "java"
        if b.is_file():
            candidate_bin = b
            candidate_home = h

    if candidate_bin is None:
        for search_home in KNOWN_JDK21_SEARCH_PATHS:
            b = search_home / "bin" / "java"
            if b.is_file():
                candidate_bin = b
                candidate_home = search_home
                break

    if candidate_bin is None:
        which = shutil.which("java")
        if which:
            candidate_bin = Path(which)
            candidate_home = candidate_bin.parent.parent

    if candidate_bin is None or not candidate_bin.is_file():
        raise ToolVerificationError(
            "jdk_not_found",
            "No Java executable found to verify.",
        )

    if candidate_bin.is_symlink():
        # Check if the real path exists and is regular file
        real = candidate_bin.resolve()
        if not real.is_file():
            raise ToolVerificationError(
                "symlink_rejected",
                f"Java binary symlink resolves to non-file: {candidate_bin}",
                path=candidate_bin,
            )

    env = get_clean_env(candidate_home)
    try:
        proc = subprocess.run(
            [str(candidate_bin), "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            check=False,
        )
    except Exception as exc:
        raise ToolVerificationError(
            "jdk_execution_failed",
            f"Failed to execute java binary: {exc}",
            path=candidate_bin,
        ) from exc

    output = proc.stderr + "\n" + proc.stdout
    match_version = re.search(r"^\s*java\.version\s*=\s*(\S+)", output, re.MULTILINE)
    match_vendor = re.search(r"^\s*java\.vendor\s*=\s*(.*?)\s*$", output, re.MULTILINE)
    match_home = re.search(r"^\s*java\.home\s*=\s*(.*?)\s*$", output, re.MULTILINE)

    if not match_version:
        fallback_match = re.search(r'version "(\d+)(?:\.|\+|-)', output)
        if not fallback_match:
            raise ToolVerificationError(
                "jdk_probe_failed",
                f"Could not parse java.version from JVM output: {output[:300]}",
                path=candidate_bin,
            )
        version_str = fallback_match.group(1)
        major = int(version_str)
    else:
        version_str = match_version.group(1)
        try:
            major = int(version_str.split(".")[0])
        except ValueError as exc:
            raise ToolVerificationError(
                "jdk_version_mismatch",
                f"Invalid java.version format: {version_str}",
                path=candidate_bin,
            ) from exc

    if major != 21:
        raise ToolVerificationError(
            "jdk_version_mismatch",
            f"Java runtime major version is {major} ({version_str}), required JDK 21",
            path=candidate_bin,
        )

    vendor = match_vendor.group(1).strip() if match_vendor else "Unknown"
    home_path = Path(match_home.group(1).strip()) if match_home else candidate_home or candidate_bin.parent.parent

    # Binary and release hashes
    exec_hash = compute_sha256(candidate_bin)
    release_file = home_path / "release"
    release_hash: str | None = None
    if release_file.is_file() and not release_file.is_symlink():
        release_hash = compute_sha256(release_file)

    return JdkIdentity(
        executable=candidate_bin,
        java_home=home_path,
        version=version_str,
        major_version=major,
        vendor=vendor,
        release_hash=release_hash,
        executable_hash=exec_hash,
    )


def find_pmd_jars(
    search_dirs: Sequence[Path] | None = None,
    project_root: Path | None = None,
    expected_names: set[str] | Sequence[str] | None = None,
) -> dict[str, Path]:
    """Discover PMD runtime JARs across candidate directories without invoking Gradle."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[4]

    expected_set: set[str] | None = set(expected_names) if expected_names is not None else None
    if expected_set is None:
        lock_file = project_root / "tooling" / "tools.lock.json"
        if lock_file.is_file():
            try:
                raw_lock = json.loads(lock_file.read_text(encoding="utf-8"))
                expected_set = set(raw_lock.get("artifacts", {}).keys())
            except (OSError, json.JSONDecodeError):
                expected_set = None

    candidates: list[Path] = []
    if search_dirs:
        candidates.extend(search_dirs)

    # Common local search locations
    candidates.append(project_root / "tooling" / "lib")
    if "PMD_LIB_DIR" in os.environ:
        candidates.append(Path(os.environ["PMD_LIB_DIR"]))
    if "PMD_HOME" in os.environ:
        candidates.append(Path(os.environ["PMD_HOME"]) / "lib")
    candidates.append(Path("/opt/pmd/lib"))
    candidates.append(Path.home() / ".gradle" / "caches" / "modules-2" / "files-2.1")

    discovered: dict[str, Path] = {}
    for candidate_dir in candidates:
        if not candidate_dir.is_dir():
            continue
        # Scan files in directory (recursively if gradle cache or lib)
        is_gradle_cache = "gradle" in str(candidate_dir).lower()
        pattern = "**/*.jar" if is_gradle_cache else "*.jar"
        for jar_path in candidate_dir.glob(pattern):
            if (expected_set is None or jar_path.name in expected_set) and jar_path.is_symlink():
                raise ToolVerificationError(
                    "symlink_rejected",
                    f"JAR artifact must not be a symlink: {jar_path}",
                    path=jar_path,
                )
            if jar_path.name not in discovered and jar_path.is_file():
                discovered[jar_path.name] = jar_path

    return discovered


def compute_classpath_hash(classpath: Sequence[Path]) -> str:
    """Compute deterministic composite SHA-256 hash of classpath inventory."""
    hasher = hashlib.sha256()
    for jar in sorted(classpath, key=lambda p: p.name):
        hasher.update(jar.name.encode("utf-8"))
        hasher.update(b":")
        hasher.update(compute_sha256(jar).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def verify_tool_installation(
    tools_lock_path: Path | str | None = None,
    lib_dir: Path | str | None = None,
    ruleset_path: Path | str | None = None,
    java_home: Path | str | None = None,
    search_dirs: Sequence[Path] | None = None,
) -> tuple[tuple[Path, ...], JdkIdentity, Path]:
    """Verify all 24 required JARs, ruleset.xml, and JDK 21 against tools.lock.json.

    Returns:
        (resolved_classpath, jdk_identity, verified_ruleset_path)
    """
    lock = load_tools_lock(tools_lock_path)
    project_root = Path(__file__).resolve().parents[4]

    # Verify JDK 21
    jdk = verify_jdk(java_home=java_home)

    # Verify Ruleset
    if ruleset_path is not None:
        ruleset_file = Path(ruleset_path)
    else:
        ruleset_lock = lock.artifacts.get("ruleset.xml")
        rel = ruleset_lock.relative_path if ruleset_lock else "src/review_bot/resources/pmd/ruleset.xml"
        ruleset_file = project_root / (rel or "src/review_bot/resources/pmd/ruleset.xml")

    if not ruleset_file.exists():
        raise ToolVerificationError("missing_artifact", f"Ruleset not found: {ruleset_file}", ruleset_file)
    if ruleset_file.is_symlink():
        raise ToolVerificationError("symlink_rejected", f"Ruleset must not be a symlink: {ruleset_file}", ruleset_file)
    if not stat.S_ISREG(ruleset_file.stat().st_mode):
        raise ToolVerificationError("not_regular_file", f"Ruleset must be a regular file: {ruleset_file}", ruleset_file)

    ruleset_spec = lock.artifacts.get("ruleset.xml")
    if ruleset_spec:
        if ruleset_file.stat().st_size != ruleset_spec.size_bytes:
            raise ToolVerificationError(
                "size_mismatch",
                f"Ruleset size mismatch: expected {ruleset_spec.size_bytes}, got {ruleset_file.stat().st_size}",
                ruleset_file,
            )
        actual_ruleset_hash = compute_sha256(ruleset_file)
        if actual_ruleset_hash != ruleset_spec.sha256:
            raise ToolVerificationError(
                "checksum_mismatch",
                f"Ruleset checksum mismatch: expected {ruleset_spec.sha256}, got {actual_ruleset_hash}",
                ruleset_file,
            )

    # Locate and verify JARs
    all_search_dirs = list(search_dirs or [])
    if lib_dir is not None:
        all_search_dirs.insert(0, Path(lib_dir))

    jar_pool = find_pmd_jars(
        all_search_dirs,
        project_root=project_root,
        expected_names=set(lock.artifacts.keys()),
    )

    resolved_jars: dict[str, Path] = {}
    for name, spec in lock.artifacts.items():
        if spec.category == "ruleset":
            continue

        if name not in jar_pool:
            raise ToolVerificationError(
                "missing_artifact",
                f"Required runtime JAR '{name}' not found in search paths",
            )

        jar_path = jar_pool[name]
        if jar_path.is_symlink():
            raise ToolVerificationError(
                "symlink_rejected",
                f"JAR artifact must not be a symlink: {jar_path}",
                jar_path,
            )
        if not stat.S_ISREG(jar_path.stat().st_mode):
            raise ToolVerificationError(
                "not_regular_file",
                f"JAR artifact must be a regular file: {jar_path}",
                jar_path,
            )

        actual_size = jar_path.stat().st_size
        if actual_size != spec.size_bytes:
            raise ToolVerificationError(
                "size_mismatch",
                f"JAR size mismatch on {name}: expected {spec.size_bytes}, got {actual_size}",
                jar_path,
            )

        actual_sha = compute_sha256(jar_path)
        if actual_sha != spec.sha256:
            raise ToolVerificationError(
                "checksum_mismatch",
                f"JAR checksum mismatch on {name}: expected {spec.sha256}, got {actual_sha}",
                jar_path,
            )

        resolved_jars[name] = jar_path

    # Deterministic classpath ordering:
    # 1. pmd-cli-7.3.0.jar
    # 2. pmd-java-7.3.0.jar
    # 3. pmd-core-7.3.0.jar
    # 4. all other runtime jars sorted alphabetically
    priority_order = ("pmd-cli-7.3.0.jar", "pmd-java-7.3.0.jar", "pmd-core-7.3.0.jar")
    ordered_classpath: list[Path] = []
    for prio in priority_order:
        if prio in resolved_jars:
            ordered_classpath.append(resolved_jars[prio])

    for name in sorted(resolved_jars.keys()):
        if name not in priority_order:
            ordered_classpath.append(resolved_jars[name])

    return tuple(ordered_classpath), jdk, ruleset_file
