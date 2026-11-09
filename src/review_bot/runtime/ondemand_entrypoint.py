"""VM-side CLI composition for a one-shot on-demand review."""
from __future__ import annotations

import re
import sys
import urllib.request
from collections.abc import Callable

from review_bot.application.ondemand import OnDemandCoordinator, ReviewJob
from review_bot.runtime.ondemand_vm import run_job

_INSTANCE_ID = re.compile(r"^i-[0-9a-f]+$")
_IMDS = "http://169.254.169.254"


def read_instance_id(*, urlopen=urllib.request.urlopen) -> str:
    token_request = urllib.request.Request(
        f"{_IMDS}/latest/api/token",
        data=b"",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        method="PUT",
    )
    with urlopen(token_request, timeout=2.0) as response:
        token = response.read(4096).decode("ascii").strip()
    if not token:
        raise RuntimeError("instance_metadata_unavailable")
    identity_request = urllib.request.Request(
        f"{_IMDS}/latest/meta-data/instance-id",
        headers={"X-aws-ec2-metadata-token": token},
        method="GET",
    )
    with urlopen(identity_request, timeout=2.0) as response:
        instance_id = response.read(128).decode("ascii").strip()
    if not _INSTANCE_ID.fullmatch(instance_id):
        raise RuntimeError("instance_metadata_unavailable")
    return instance_id


def run_ondemand(
    job_id: str,
    *,
    vm_id: str | None = None,
    config_path: str | None = None,
    coordinator: OnDemandCoordinator | None = None,
    execute: Callable[[ReviewJob, Callable[[], bool]], tuple[dict[str, str], dict]] | None = None,
) -> int:
    try:
        self_vm_id = vm_id or read_instance_id()
        if coordinator is None:
            from review_bot.runtime.aws_lambda import _coordinator_from_environment

            coordinator = _coordinator_from_environment()
        if execute is None:
            from review_bot.runtime.service import execute_ondemand_review

            def execute_review(job, is_current):
                return execute_ondemand_review(config_path, job, is_current)

            execute = execute_review
        result = run_job(coordinator, job_id, self_vm_id, execute, self_vm_id=self_vm_id)
    except Exception:
        sys.stderr.write("Error: on-demand review could not start\n")
        return 1
    return 1 if result is not None and result.state in ("failed", "dead_letter") else 0
