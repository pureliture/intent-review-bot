"""Private completed structured results, reusable when only presentation changes.

Callers own canonical source material and must decode the original task/result
and run validate_partition_result in validate_and_decode. No persisted snapshot,
task or finding identifiers are rewritten when a packet is restored.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MODES = frozenset({"architecture", "intent", "planning"})


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")


def _identity(installation_id: int, repository_id: int, mode: str) -> None:
    if any(type(value) is not int or value <= 0 for value in (installation_id, repository_id)):
        raise ValueError("invalid_review_cache_identity")
    if mode not in _MODES:
        raise ValueError("invalid_review_cache_mode")


def _unique(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_review_cache_key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("invalid_review_cache_constant")


def _complete_packet(packet: Any, mode: str) -> bool:
    if not isinstance(packet, dict) or set(packet) != {"task", "result"}:
        return False
    result = packet["result"]
    return (
        isinstance(packet["task"], dict) and isinstance(result, dict)
        and type(result.get("schema_version")) is int and result["schema_version"] == 13
        and result.get("status") == ("planned" if mode == "planning" else "completed")
        and result.get("review_type") == {"architecture": "A", "intent": "B", "planning": "PLAN"}[mode]
    )


class SharedReviewCache:
    def __init__(self, state_root: Path, *, max_bytes: int = 64 * 1024 * 1024) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid_review_cache_byte_bound")
        self.root = Path(state_root) / "shared-review-cache"
        self.max_bytes = max_bytes

    def _path(self, installation_id: int, repository_id: int, mode: str, generation_key: str) -> Path:
        _identity(installation_id, repository_id, mode)
        if not isinstance(generation_key, str) or _DIGEST.fullmatch(generation_key) is None:
            raise ValueError("invalid_review_cache_generation_key")
        directory = self.root
        for component in ("", str(installation_id), str(repository_id), mode):
            if component:
                directory = directory / component
            if directory.is_symlink():
                raise ValueError("unsafe_review_cache_directory")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            metadata = directory.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
            ):
                raise ValueError("review_cache_directory_must_be_private")
        return directory / f"{generation_key}.json"

    def load(
        self, installation_id: int, repository_id: int, mode: str, generation_key: str, *,
        validate_and_decode: Callable[[dict], _T],
    ) -> _T | None:
        """A corrupt, partial, inaccessible or independently invalid packet is a miss."""
        try:
            path = self._path(installation_id, repository_id, mode, generation_key)
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size > self.max_bytes
                ):
                    return None
                raw = stream.read(self.max_bytes + 1)
            if len(raw) > self.max_bytes:
                return None
            envelope = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                parse_constant=_reject_constant)
            if not isinstance(envelope, dict) or set(envelope) != {
                "schema_version", "installation_id", "repository_id", "mode", "generation_key",
                "packet", "packet_hash",
            } or type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
                return None
            if (
                type(envelope["installation_id"]) is not int or envelope["installation_id"] != installation_id
                or type(envelope["repository_id"]) is not int or envelope["repository_id"] != repository_id
                or envelope["mode"] != mode or envelope["generation_key"] != generation_key
                or not _complete_packet(envelope["packet"], mode)
                or hashlib.sha256(_canonical(envelope["packet"])).hexdigest() != envelope["packet_hash"]
            ):
                return None
            return validate_and_decode(envelope["packet"])
        except Exception:
            return None

    def save(
        self, installation_id: int, repository_id: int, mode: str, generation_key: str,
        packet: dict, *, validate_and_decode: Callable[[dict], _T],
    ) -> bool:
        """Validate the exact JSON packet before one atomic complete-entry write."""
        temporary = None
        try:
            path = self._path(installation_id, repository_id, mode, generation_key)
            # Preserve the exact object traversal used by strict replay;
            # canonical ordering remains only the checksum contract.
            encoded_packet = json.dumps(packet, ensure_ascii=False, sort_keys=False,
                separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(encoded_packet) > self.max_bytes:
                return False
            packet_hash = hashlib.sha256(_canonical(packet)).hexdigest()
            frozen_packet = json.loads(encoded_packet, object_pairs_hook=_unique,
                parse_constant=_reject_constant)
            if not _complete_packet(frozen_packet, mode) or validate_and_decode(frozen_packet) is None:
                return False
            # The callback gets an isolated decoded object; preserve the recorded bytes.
            envelope = {"schema_version": 1, "installation_id": installation_id,
                "repository_id": repository_id, "mode": mode, "generation_key": generation_key,
                "packet": json.loads(encoded_packet), "packet_hash": packet_hash}
            raw = json.dumps(envelope, ensure_ascii=False, sort_keys=False,
                separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(raw) > self.max_bytes:
                return False
            if path.exists() or path.is_symlink():
                metadata = path.lstat()
                if (
                    not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1 or stat.S_IMODE(metadata.st_mode) != 0o600
                ):
                    return False
            descriptor, temporary = tempfile.mkstemp(prefix=".pending-", suffix=".json", dir=path.parent)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return True
        except Exception:
            return False
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
