"""LXD VM launchers for host and one-shot guest execution."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from typing import Any

from review_bot.adapters.vm_user_data import one_shot_user_data
from review_bot.application.ondemand import ManagedInstance

_JOB_ID = re.compile(r"^[0-9a-f]{64}$")
_VM_ID = re.compile(r"^rb-[0-9a-f]{20}$")
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_INSTANCE_CONFIG_KEY = re.compile(r"^user\.[a-z0-9][a-z0-9.-]*$")
_RESERVED_CONFIG = {
    "user.review-bot.managed",
    "user.review-bot.job-id",
    "user.review-bot.vm-id",
    "user.review-bot.max-lifetime-seconds",
}


class LxdVmLauncher:
    def __init__(
        self,
        image: str,
        *,
        profiles: tuple[str, ...] = ("default",),
        project: str | None = None,
        instance_config: dict[str, str] | None = None,
        lxc: str = "lxc",
        run: Callable[..., Any] = subprocess.run,
    ) -> None:
        if not isinstance(image, str) or not image or any(char.isspace() for char in image):
            raise ValueError("invalid_lxd_image")
        if not isinstance(profiles, tuple) or any(
            not isinstance(profile, str) or _PROFILE_NAME.fullmatch(profile) is None
            for profile in profiles
        ):
            raise ValueError("invalid_lxd_profile")
        if instance_config is not None and not isinstance(instance_config, dict):
            raise ValueError("invalid_lxd_instance_config")
        config = dict(instance_config or {})
        if any(
            not isinstance(key, str)
            or key in _RESERVED_CONFIG
            or _INSTANCE_CONFIG_KEY.fullmatch(key) is None
            or not isinstance(value, str)
            or any(char in value for char in ("\x00", "\n", "\r"))
            for key, value in config.items()
        ):
            raise ValueError("invalid_lxd_instance_config")
        self.image = image
        self.profiles = profiles
        self.project = project
        self.instance_config = config
        self.lxc = lxc
        self.run = run

    def _invoke(self, args: list[str], *, timeout: int, error: str):
        try:
            return self.run(
                [self.lxc, *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(error) from None
        except UnicodeDecodeError:
            raise RuntimeError(error) from None

    def _project_args(self) -> list[str]:
        return ["--project", self.project] if self.project else []

    def launch(self, job_id: str, *, max_lifetime_seconds: int, launch_token: str) -> str:
        if (
            not isinstance(job_id, str)
            or _JOB_ID.fullmatch(job_id) is None
            or type(max_lifetime_seconds) is not int
            or not 60 <= max_lifetime_seconds <= 6 * 3600
            or not isinstance(launch_token, str)
            or not launch_token
        ):
            raise ValueError("invalid_launch_request")

        name = "rb-" + hashlib.sha256(launch_token.encode("utf-8")).hexdigest()[:20]
        args = [
            "launch",
            self.image,
            name,
            "--vm",
            "--ephemeral",
            *self._project_args(),
        ]
        for profile in self.profiles:
            args.extend(("--profile", profile))
        for key, value in (
            ("user.review-bot.managed", "true"),
            ("user.review-bot.job-id", job_id),
            ("user.review-bot.vm-id", name),
            ("user.review-bot.max-lifetime-seconds", str(max_lifetime_seconds)),
        ):
            args.extend(("-c", f"{key}={value}"))
        for key, value in sorted(self.instance_config.items()):
            args.extend(("-c", f"{key}={value}"))
        args.extend((
            "-c",
            f"cloud-init.user-data={one_shot_user_data(job_id, max_lifetime_seconds)}",
        ))

        result = self._invoke(args, timeout=300, error="lxd_launch_failed")
        if result.returncode == 0:
            return name
        existing = self._invoke(
            [
                "config",
                "get",
                *self._project_args(),
                name,
                "user.review-bot.job-id",
            ],
            timeout=60,
            error="lxd_launch_failed",
        )
        if (
            existing.returncode == 0
            and isinstance(existing.stdout, str)
            and existing.stdout.strip() == job_id
        ):
            return name
        raise RuntimeError("lxd_launch_failed")

    def terminate(self, vm_id: str) -> None:
        if not isinstance(vm_id, str) or _VM_ID.fullmatch(vm_id) is None:
            raise ValueError("invalid_vm_id")
        result = self._invoke(
            ["delete", "--force", *self._project_args(), vm_id],
            timeout=60,
            error="lxd_terminate_failed",
        )
        if result.returncode == 0:
            return
        info = self._invoke(
            ["info", *self._project_args(), vm_id],
            timeout=60,
            error="lxd_terminate_failed",
        )
        if info.returncode == 0:
            raise RuntimeError("lxd_terminate_failed")

    def list_managed_instances(self) -> list[ManagedInstance]:
        result = self._invoke(
            ["list", *self._project_args(), "--format", "json"],
            timeout=60,
            error="lxd_list_failed",
        )
        if result.returncode != 0:
            raise RuntimeError("lxd_list_failed")
        try:
            instances = json.loads(result.stdout)
            if not isinstance(instances, list):
                raise TypeError
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError("lxd_list_failed") from None

        managed = []
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            config = instance.get("config")
            if not isinstance(config, dict):
                continue
            name = instance.get("name")
            job_id = config.get("user.review-bot.job-id")
            if (
                config.get("user.review-bot.managed") != "true"
                or not isinstance(job_id, str)
                or _JOB_ID.fullmatch(job_id) is None
                or instance.get("status") == "Stopped"
                or not isinstance(name, str)
                or _VM_ID.fullmatch(name) is None
                or any(config.get(key) != value for key, value in self.instance_config.items())
            ):
                continue
            managed.append(ManagedInstance(name, job_id))
        return managed


class LxdGuestLauncher:
    def __init__(self, vm_id: str) -> None:
        if not isinstance(vm_id, str) or _VM_ID.fullmatch(vm_id) is None:
            raise ValueError("invalid_vm_id")
        self.vm_id = vm_id

    def launch(self, job_id: str, *, max_lifetime_seconds: int, launch_token: str) -> str:
        raise RuntimeError("unsupported_in_guest")

    def terminate(self, vm_id: str) -> None:
        if vm_id == self.vm_id:
            return
        raise RuntimeError("lxd_guest_cannot_terminate_other_vm")

    def list_managed_instances(self) -> list[ManagedInstance]:
        raise RuntimeError("unsupported_in_guest")
