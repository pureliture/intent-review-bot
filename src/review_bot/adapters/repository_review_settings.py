"""Private, atomic last-known-valid settings records in the existing state root."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from review_bot.application.repository_review_settings import (
    MAX_SETTINGS_BYTES,
    RepositoryReviewSettings,
    RepositorySettingsError,
    _unique_object,
    _validate_identity,
    parse_repository_review_settings,
)


class FileRepositoryReviewSettingsStore:
    """One entire version per installation/repository; no prompt fragments are cached."""

    def __init__(self, state_root: Path) -> None:
        self.root = Path(state_root) / "repository-review-settings"

    def _directory(self, installation_id: int, repository_id: int) -> Path:
        _validate_identity(installation_id, repository_id)
        for directory in (self.root, self.root / str(installation_id)):
            if directory.is_symlink():
                raise RepositorySettingsError("unsafe_settings_state_path")
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            if directory.stat().st_mode & 0o077:
                raise RepositorySettingsError("settings_state_must_be_private")
        return self.root / str(installation_id)

    @contextmanager
    def transaction(self, installation_id: int, repository_id: int):
        directory = self._directory(installation_id, repository_id)
        descriptor = os.open(directory / f"{repository_id}.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            yield

    def load(self, installation_id: int, repository_id: int) -> RepositoryReviewSettings | None:
        directory = self._directory(installation_id, repository_id)
        try:
            descriptor = os.open(directory / f"{repository_id}.json", os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(descriptor, "rb") as stream:
            if os.fstat(stream.fileno()).st_mode & 0o077:
                raise RepositorySettingsError("settings_state_must_be_private")
            raw = stream.read(MAX_SETTINGS_BYTES * 2 + 1)
        if len(raw) > MAX_SETTINGS_BYTES * 2:
            raise RepositorySettingsError("settings_state_size_limit")
        try:
            record = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
            if not isinstance(record, dict) or set(record) != {
                "installation_id", "repository_id", "source_sha", "manifest", "effective_version",
            } or record["installation_id"] != installation_id or record["repository_id"] != repository_id:
                raise RepositorySettingsError("settings_state_identity_mismatch")
            manifest = json.dumps(record["manifest"], ensure_ascii=False).encode("utf-8")
            bundle = parse_repository_review_settings(manifest, record["source_sha"])
            if bundle.effective_version != record["effective_version"]:
                raise RepositorySettingsError("settings_state_integrity_mismatch")
            return bundle
        except (KeyError, UnicodeError, ValueError, TypeError, RecursionError):
            raise RepositorySettingsError("invalid_settings_state") from None

    def save(self, installation_id: int, repository_id: int, bundle: RepositoryReviewSettings) -> None:
        directory = self._directory(installation_id, repository_id)
        # Revalidate the complete object before it becomes the durable fallback.
        manifest = bundle.manifest()
        validated = parse_repository_review_settings(
            json.dumps(manifest, ensure_ascii=False).encode("utf-8"), bundle.source_sha,
        )
        if validated != bundle:
            raise RepositorySettingsError("invalid_settings_state")
        payload = json.dumps({"installation_id": installation_id, "repository_id": repository_id,
            "source_sha": bundle.source_sha, "manifest": manifest,
            "effective_version": bundle.effective_version}, ensure_ascii=False, sort_keys=True).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{repository_id}-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, directory / f"{repository_id}.json")
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
