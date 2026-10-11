"""Git repository adapter implementing object reads, merge-base resolution, and diffing."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from review_bot.domain.envelope_v2 import ChangedFile
from review_bot.domain.models_v1 import (
    CodeFile,
    CodeSnapshot,
    SourceExclusion,
)
from review_bot.ports import Commit, MergeBaseResult


class GitError(Exception):
    """Base exception for Git adapter operations."""


class GitCommandError(GitError):
    """Raised when a git command exits with non-zero status."""

    def __init__(self, command: list[str], returncode: int, stderr: str) -> None:
        super().__init__(f"Git command failed ({returncode}): {' '.join(command)}\n{stderr}")
        self.command = command
        self.returncode = returncode
        self.stderr = stderr


class CommitNotFoundError(GitError):
    """Raised when a requested commit SHA cannot be resolved."""


# --- Validation Helpers ---

_SHA_REGEX = re.compile(r"^[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?\Z")
_GIT_TIMEOUT_SECONDS = 30


def _fixed_sha(sha: str) -> str:
    """Validate and normalize a fixed commit or object SHA."""
    if not isinstance(sha, str) or not _SHA_REGEX.match(sha):
        raise ValueError(f"Invalid full Git SHA: {sha!r}")
    return sha.lower()


def _safe_path(path: str) -> bool:
    """Ensure path is relative, normalized, and free of traversal or control characters."""
    if not isinstance(path, str) or not path:
        return False
    if path.startswith(("/", "\\")):
        return False
    if any(ord(c) < 32 or ord(c) == 127 for c in path):
        return False
    parts = path.split("/")
    return all(p not in ("", ".", "..") for p in parts) and "\\" not in path


def _run_git(
    repo_path: Path | str,
    args: list[str],
    input_bytes: bytes | None = None,
    check: bool = True,
    timeout: int = _GIT_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Execute git subprocess in isolated environment."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
    })
    cmd = [
        "git",
        "--no-replace-objects",
        "--no-optional-locks",
        "-c", "core.fsmonitor=false",
        "-c", "protocol.allow=never",
        "-C", str(repo_path),
        *args,
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            capture_output=True,
            env=env,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(f"Git command timed out after {timeout}s: {' '.join(cmd)}") from exc
    except OSError as exc:
        raise GitError(f"Failed to execute git: {exc}") from exc

    if check and proc.returncode != 0:
        stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
        raise GitCommandError(cmd, proc.returncode, stderr_text)
    return proc


# --- Standalone Functions ---

def resolve_merge_base(
    repo_path: Path | str, target_base_sha: str, head_sha: str
) -> MergeBaseResult:
    """Compute three-dot merge base separating target_base_sha from comparison_base_sha."""
    t_sha = _fixed_sha(target_base_sha)
    h_sha = _fixed_sha(head_sha)

    # Identical commit check
    if t_sha == h_sha:
        exists_proc = _run_git(
            repo_path, ["cat-file", "-e", f"{t_sha}^{{commit}}"], check=False
        )
        if exists_proc.returncode != 0:
            return MergeBaseResult(
                target_base_sha=t_sha,
                comparison_base_sha=None,
                comparison_status="unavailable",
                candidates=(),
            )
        return MergeBaseResult(
            target_base_sha=t_sha,
            comparison_base_sha=t_sha,
            comparison_status="comparable",
            candidates=(t_sha,),
        )

    proc = _run_git(repo_path, ["merge-base", "--all", t_sha, h_sha], check=False)
    if proc.returncode != 0:
        return MergeBaseResult(
            target_base_sha=t_sha,
            comparison_base_sha=None,
            comparison_status="unavailable",
            candidates=(),
        )

    bases = [line.strip().lower() for line in proc.stdout.decode("ascii").splitlines() if line.strip()]
    if not bases:
        return MergeBaseResult(
            target_base_sha=t_sha,
            comparison_base_sha=None,
            comparison_status="unavailable",
            candidates=(),
        )
    elif len(bases) == 1:
        return MergeBaseResult(
            target_base_sha=t_sha,
            comparison_base_sha=bases[0],
            comparison_status="comparable",
            candidates=(bases[0],),
        )
    else:
        return MergeBaseResult(
            target_base_sha=t_sha,
            comparison_base_sha=None,
            comparison_status="ambiguous",
            candidates=tuple(bases),
        )


def read_tree_blobs(
    repo_path: Path | str,
    commit_sha: str,
    paths: tuple[str, ...] | list[str] | None = None,
) -> dict[str, bytes]:
    """Read regular blob contents at commit_sha using git cat-file."""
    c_sha = _fixed_sha(commit_sha)
    allowed_paths = set(paths) if paths is not None else None

    # List tree entries
    proc = _run_git(repo_path, ["ls-tree", "-r", "-z", "--full-tree", c_sha])
    entries = proc.stdout.split(b"\0")

    path_to_oid: dict[str, str] = {}
    for entry in entries:
        if not entry:
            continue
        try:
            metadata, raw_path = entry.split(b"\t", 1)
            mode, obj_type, oid = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue

        if not _safe_path(path):
            continue
        if allowed_paths is not None and path not in allowed_paths:
            continue
        if mode not in ("100644", "100755") or obj_type != "blob":
            continue

        path_to_oid[path] = oid

    if not path_to_oid:
        return {}

    # Bulk fetch unique OIDs via git cat-file --batch
    unique_oids = list(dict.fromkeys(path_to_oid.values()))
    batch_input = "".join(f"{oid}\n" for oid in unique_oids).encode("ascii")
    batch_proc = _run_git(repo_path, ["cat-file", "--batch"], input_bytes=batch_input)
    raw_output = batch_proc.stdout

    oid_to_content: dict[str, bytes] = {}
    offset = 0
    for oid in unique_oids:
        nl = raw_output.find(b"\n", offset)
        if nl < 0:
            break
        header = raw_output[offset:nl].split()
        if len(header) != 3 or header[1] != b"blob":
            break
        size = int(header[2])
        content_start = nl + 1
        content_end = content_start + size
        content = raw_output[content_start:content_end]
        oid_to_content[oid] = content
        offset = content_end + 1  # trailing newline

    return {path: oid_to_content[oid] for path, oid in path_to_oid.items() if oid in oid_to_content}


def get_changed_files(
    repo_path: Path | str, base_sha: str, head_sha: str
) -> tuple[ChangedFile, ...]:
    """Detect file changes and renames between base_sha and head_sha."""
    b_sha = _fixed_sha(base_sha)
    h_sha = _fixed_sha(head_sha)

    proc = _run_git(
        repo_path,
        ["diff", "--no-ext-diff", "--no-textconv", "--find-renames", "-z", "--name-status", b_sha, h_sha],
    )
    tokens = proc.stdout.split(b"\0")
    files: list[ChangedFile] = []
    idx = 0

    while idx < len(tokens) and tokens[idx]:
        status_code = tokens[idx].decode("ascii", errors="replace").strip()
        idx += 1
        if idx >= len(tokens):
            break

        path1 = tokens[idx].decode("utf-8", errors="replace")
        idx += 1

        if status_code.startswith("R"):
            if idx >= len(tokens):
                break
            dest_path = tokens[idx].decode("utf-8", errors="replace")
            idx += 1
            if _safe_path(path1) and _safe_path(dest_path):
                files.append(ChangedFile(path=dest_path, previous_path=path1, change_type="renamed", has_diff=True))
        elif status_code.startswith("C"):
            if idx >= len(tokens):
                break
            dest_path = tokens[idx].decode("utf-8", errors="replace")
            idx += 1
            if _safe_path(path1) and _safe_path(dest_path):
                files.append(ChangedFile(path=dest_path, previous_path=path1, change_type="copied", has_diff=True))
        elif status_code.startswith("A"):
            if _safe_path(path1):
                files.append(ChangedFile(path=path1, previous_path=None, change_type="added", has_diff=True))
        elif status_code.startswith("D"):
            if _safe_path(path1):
                files.append(ChangedFile(path=path1, previous_path=None, change_type="deleted", has_diff=True))
        else:
            if _safe_path(path1):
                files.append(ChangedFile(path=path1, previous_path=None, change_type="modified", has_diff=True))

    return tuple(files)


def get_rename_map(
    repo_path: Path | str, base_sha: str, head_sha: str
) -> dict[str, str]:
    """Compute safe rename map {previous_path: current_path}."""
    changed_files = get_changed_files(repo_path, base_sha, head_sha)
    renames: dict[str, str] = {}
    for f in changed_files:
        if f.change_type == "renamed" and f.previous_path:
            renames[f.previous_path] = f.path

    # Injective check
    if len(set(renames.values())) != len(renames):
        raise ValueError("Ambiguous rename map: multiple files renamed to same destination")
    return renames


def collect_code_snapshot(repo_path: Path | str, sha: str) -> CodeSnapshot:
    """Full code gathering for a commit SHA."""
    c_sha = _fixed_sha(sha)
    blobs = read_tree_blobs(repo_path, c_sha)

    files: list[CodeFile] = []
    exclusions: list[SourceExclusion] = []
    errors: list[str] = []

    for path, content_bytes in blobs.items():
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        try:
            content_str = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            exclusions.append(SourceExclusion(path=path, reason="unsupported_encoding"))
            errors.append(f"{path}: unsupported encoding")
            continue

        # Classify file kind
        lower_path = path.lower()
        if lower_path.endswith(".java"):
            kind = "test" if ("/test/" in lower_path or lower_path.startswith("test/")) else "source"
        elif any(lower_path.endswith(ext) for ext in (".gradle", ".gradle.kts", ".xml", ".properties")):
            kind = "build"
        else:
            kind = "source"

        files.append(CodeFile(path=path, content=content_str, content_hash=content_hash, kind=kind))

    files.sort(key=lambda f: f.path)
    inv_str = ":".join(f"{f.path}:{f.content_hash}" for f in files)
    inventory_hash = hashlib.sha256(inv_str.encode("utf-8")).hexdigest()

    return CodeSnapshot(
        sha=c_sha,
        files=tuple(files),
        exclusions=tuple(exclusions),
        inventory_hash=inventory_hash,
        complete=len(errors) == 0,
        errors=tuple(errors),
    )


# --- Class GitRepositoryReader ---

class GitRepositoryReader:
    """RepositoryReader protocol implementation backed by a local Git directory."""

    def __init__(self, repo_path: Path | str) -> None:
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists() and not (self.repo_path / "HEAD").exists():
            raise ValueError(f"Not a valid git repository: {self.repo_path}")

    def get_commit(self, sha: str) -> Commit:
        if not isinstance(sha, str) or not sha or sha.startswith("-"):
            raise CommitNotFoundError(f"Commit not found: {sha}")
        c_sha = _fixed_sha(sha) if _SHA_REGEX.match(sha) else sha
        proc = _run_git(
            self.repo_path,
            ["log", "-1", "--format=%H%n%T%n%P%n%an <%ae>%n%aI%n%B", "--end-of-options", c_sha],
            check=False,
        )
        if proc.returncode != 0:
            raise CommitNotFoundError(f"Commit not found: {c_sha}")
        lines = proc.stdout.decode("utf-8", errors="replace").splitlines()
        if len(lines) < 5:
            raise CommitNotFoundError(f"Commit not found: {c_sha}")
        return Commit(
            sha=lines[0].strip(),
            tree_sha=lines[1].strip(),
            parents=tuple(lines[2].strip().split()) if lines[2].strip() else (),
            author=lines[3].strip(),
            committed_at=lines[4].strip(),
            message="\n".join(lines[5:]).strip(),
        )

    def read_tree_blobs(
        self, commit_sha: str, paths: tuple[str, ...] | list[str] | None = None
    ) -> dict[str, bytes]:
        return read_tree_blobs(self.repo_path, commit_sha, paths)

    def compute_merge_base(
        self, base_sha: str, head_sha: str, all_bases: bool = False
    ) -> tuple[str, ...]:
        args = ["merge-base"]
        if all_bases:
            args.append("--all")
        args.extend([_fixed_sha(base_sha), _fixed_sha(head_sha)])
        proc = _run_git(self.repo_path, args, check=False)
        if proc.returncode != 0:
            return ()
        return tuple(line.strip().lower() for line in proc.stdout.decode("ascii").splitlines() if line.strip())

    def resolve_merge_base(
        self, target_base_sha: str, head_sha: str
    ) -> MergeBaseResult:
        return resolve_merge_base(self.repo_path, target_base_sha, head_sha)

    def get_changed_files(
        self, base_sha: str, head_sha: str
    ) -> tuple[ChangedFile, ...]:
        return get_changed_files(self.repo_path, base_sha, head_sha)

    def get_rename_map(
        self, base_sha: str, head_sha: str
    ) -> dict[str, str]:
        return get_rename_map(self.repo_path, base_sha, head_sha)

    def collect_code_snapshot(
        self, sha: str
    ) -> CodeSnapshot:
        return collect_code_snapshot(self.repo_path, sha)
