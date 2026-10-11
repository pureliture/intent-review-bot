"""Fail-closed CLI composition of the signed receiver and installation-bound worker."""
from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import uvicorn

from review_bot.adapters.github.auth import GitHubAppTokenProvider
from review_bot.adapters.pmd.container import RootlessPmdAnalyzer, _IMAGE_PIN
from review_bot.adapters.sqlite.store import SQLiteJobStore
from review_bot.config import ReviewBotConfig, load_review_config, validate_active_environment
from review_bot.domain.envelope_v2 import RunRecord
from review_bot.ports import GitHubClient, RepositoryReader
from review_bot.runtime.github_app import GitHubAppAdapters
from review_bot.runtime.webhook import create_webhook_app
from review_bot.runtime.worker import WorkerLoop


class ServiceConfigurationError(ValueError):
    """Operator-facing errors containing field names, never raw values or paths."""


def load_service_config(path: str | None) -> ReviewBotConfig:
    if not path:
        raise ServiceConfigurationError("Service commands require --config")
    try:
        config, _ = load_review_config(path)
    except Exception:
        raise ServiceConfigurationError("Cannot load operator --config") from None
    if config.execution.mode != "github_app":
        raise ServiceConfigurationError("execution.mode must be github_app")
    if config.execution.backend != "github_app":
        raise ServiceConfigurationError("execution.backend must be github_app")
    if config.execution.worker_concurrency != 1:
        raise ServiceConfigurationError("execution.worker_concurrency must be 1")
    if config.service.worker != "synchronous":
        raise ServiceConfigurationError("service.worker must be synchronous")
    if not config.service.sqlite_wal:
        raise ServiceConfigurationError("service.sqlite_wal must be true")
    if not config.service.lease_and_generation_fencing:
        raise ServiceConfigurationError("service.lease_and_generation_fencing must be true")
    isolation = config.architecture.isolation
    if isolation.runtime != "rootless_docker":
        raise ServiceConfigurationError("architecture.isolation.runtime must be rootless_docker")
    if not isolation.read_only_root:
        raise ServiceConfigurationError("architecture.isolation.read_only_root must be true")
    if not isolation.no_new_privileges:
        raise ServiceConfigurationError("architecture.isolation.no_new_privileges must be true")
    image = config.architecture.tool_image_digest
    if image is None or _IMAGE_PIN.fullmatch(image) is None:
        raise ServiceConfigurationError("architecture.tool_image_digest requires a precise sha256 image pin")
    try:
        validate_active_environment(config)
    except Exception:
        raise ServiceConfigurationError("Required GitHub App environment references must be set") from None
    if not os.environ.get(config.service.state_root_env, "").strip():
        raise ServiceConfigurationError("service.state_root_env must reference a private absolute directory")
    return config


def _checked_path(path: Path) -> Path:
    text = os.fspath(path)
    if (
        path.anchor != "/" or path == Path("/") or ".." in path.parts
        or Path.home().is_relative_to(path)
        or any(ord(char) < 32 or ord(char) == 127 or char in ',"' for char in text)
    ):
        raise ServiceConfigurationError("Unsafe private service path")
    for parent in reversed(path.parents):
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            continue
        sticky_root = metadata.st_uid == 0 and bool(metadata.st_mode & stat.S_ISVTX)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid not in (0, os.geteuid())
            or (metadata.st_mode & 0o022 and not sticky_root)
        ):
            raise ServiceConfigurationError("Private service paths require trusted non-symlink parents")
    return path


def _private_directory(path: Path) -> Path:
    path = _checked_path(path)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise ServiceConfigurationError("Service state directories must be owned and mode 0700")
    return path


def _private_file(path: Path, *, missing_ok: bool = False, key: bool = False) -> bool:
    path = _checked_path(path)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise ServiceConfigurationError("Private service file is unavailable") from None
    modes = (0o400, 0o600) if key else (0o600,)
    if (
        not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) not in modes or metadata.st_nlink != 1
    ):
        raise ServiceConfigurationError("Service files must be private, owned and not linked")
    return True


def _check_database_files(database: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        _private_file(Path(str(database) + suffix), missing_ok=True)


@contextmanager
def open_service_state(config: ReviewBotConfig) -> Iterator[tuple[Path, SQLiteJobStore]]:
    previous_umask = os.umask(0o077)
    job_store = None
    try:
        root = _private_directory(Path(os.environ[config.service.state_root_env]))
        for name in ("repositories", "analysis", "artifacts"):
            _private_directory(root / name)
        database = root / "jobs.sqlite"
        _check_database_files(database)
        if not _private_file(database, missing_ok=True):
            try:
                descriptor = os.open(database, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            except FileExistsError:
                _private_file(database)
            else:
                os.close(descriptor)
        job_store = SQLiteJobStore(database)
        _check_database_files(database)
        yield root, job_store
    except ServiceConfigurationError:
        raise
    except Exception:
        raise ServiceConfigurationError("Private service state or runtime is unavailable") from None
    finally:
        try:
            if job_store is not None:
                job_store.close()
        finally:
            os.umask(previous_umask)


class _ClaimedAppAdapters:
    """Load the key only when WorkerLoop invokes an adapter for claimed work."""

    def __init__(self, config: ReviewBotConfig, cache_root: Path) -> None:
        self.config = config
        self.cache_root = cache_root
        self._adapters: GitHubAppAdapters | None = None

    def _get(self) -> GitHubAppAdapters:
        if self._adapters is None:
            try:
                key = Path(os.environ[self.config.service.app_private_key_file_env])
                _private_file(key, key=True)
                provider = GitHubAppTokenProvider(os.environ[self.config.service.app_id_env], key)
                self._adapters = GitHubAppAdapters(provider, self.cache_root, self.config.service)
            except Exception:
                raise ServiceConfigurationError("GitHub App credentials or adapters are unavailable") from None
        return self._adapters

    def github_client(self, run: RunRecord) -> GitHubClient:
        return self._get().github_client(run)

    def repository_reader(self, run: RunRecord, pr_data: dict[str, Any], client: GitHubClient) -> RepositoryReader:
        return self._get().repository_cache.prepare(run, pr_data, client)


def run_webhook(config_path: str | None, *, host: str = "127.0.0.1", port: int = 8000) -> int:
    config = load_service_config(config_path)
    if not host or not 1 <= port <= 65535:
        raise ServiceConfigurationError("A host and port in 1..65535 are required")
    with open_service_state(config) as (_, job_store):
        app = create_webhook_app(config, job_store)
        uvicorn.run(app, host=host, port=port)
    return 0


def run_worker(config_path: str | None, *, once: bool = False) -> int:
    config = load_service_config(config_path)
    with open_service_state(config) as (root, job_store):
        isolation = config.architecture.isolation
        analyzer = RootlessPmdAnalyzer(
            config.architecture.tool_image_digest, root / "analysis",
            memory_mib=isolation.memory_mib, pids=isolation.pids, cpus=isolation.cpus,
            timeout_seconds=config.architecture.tool_timeout_seconds,
        )
        adapters = _ClaimedAppAdapters(config, root / "repositories")
        loop = WorkerLoop(
            config, job_store, analyzer=analyzer, github_client_factory=adapters.github_client,
            repo_reader_factory=adapters.repository_reader, artifact_root=root / "artifacts",
        )
        if once:
            loop.prepare_models()
            run = job_store.claim_next_run(loop.worker_id, loop.lease_duration)
            if run is None:
                return 0
            return 0 if loop.process_run(run) else 1
        loop.run_forever()
    return 0