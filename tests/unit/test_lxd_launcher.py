from __future__ import annotations

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest

from review_bot.adapters.aws.ec2_launcher import Ec2VmLauncher
from review_bot.adapters.lxd.launcher import LxdGuestLauncher, LxdVmLauncher
from review_bot.adapters.vm_user_data import one_shot_user_data
from review_bot.application.ondemand import ManagedInstance
from review_bot.runtime.aws_lambda import _launcher_from_environment

JOB_ID = "a" * 64
VM_ID = "rb-" + hashlib.sha256(b"launch-token").hexdigest()[:20]


class FakeRunner:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if not self.results:
            raise AssertionError(f"unexpected command: {argv}")
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def completed(returncode=0, stdout="", stderr="private lxc error"):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_launch_uses_expected_lxd_vm_argv_and_shared_user_data():
    runner = FakeRunner(completed())
    launcher = LxdVmLauncher(
        "ubuntu:24.04",
        profiles=("default", "review-bot-worker"),
        project="reviews",
        instance_config={
            "user.review-bot.region": "us-east-1",
            "user.review-bot.job-table": "jobs",
        },
        run=runner,
    )

    assert launcher.launch(
        JOB_ID,
        max_lifetime_seconds=601,
        launch_token="launch-token",
    ) == VM_ID
    argv, options = runner.calls[0]
    assert argv == [
        "lxc", "launch", "ubuntu:24.04", VM_ID, "--vm", "--ephemeral",
        "--project", "reviews", "--profile", "default",
        "--profile", "review-bot-worker",
        "-c", "user.review-bot.managed=true",
        "-c", f"user.review-bot.job-id={JOB_ID}",
        "-c", f"user.review-bot.vm-id={VM_ID}",
        "-c", "user.review-bot.max-lifetime-seconds=601",
        "-c", "user.review-bot.job-table=jobs",
        "-c", "user.review-bot.region=us-east-1",
        "-c", f"cloud-init.user-data={one_shot_user_data(JOB_ID, 601)}",
    ]
    assert options == {
        "capture_output": True,
        "text": True,
        "check": False,
        "timeout": 300,
    }


@pytest.mark.parametrize(
    ("job_id", "lifetime", "token"),
    [
        ("bad", 600, "token"),
        (JOB_ID, 59, "token"),
        (JOB_ID, 21601, "token"),
        (JOB_ID, True, "token"),
        (JOB_ID, 600, ""),
        (JOB_ID, 600, None),
    ],
)
def test_launch_rejects_invalid_requests(job_id, lifetime, token):
    launcher = LxdVmLauncher("ubuntu:24.04", run=FakeRunner())
    with pytest.raises(ValueError, match="invalid_launch_request"):
        launcher.launch(job_id, max_lifetime_seconds=lifetime, launch_token=token)


@pytest.mark.parametrize(
    "instance_config",
    [
        {"user.review-bot.managed": "true"},
        {"user.review-bot.job-id": JOB_ID},
        {"user.review-bot.vm-id": VM_ID},
        {"user.review-bot.max-lifetime-seconds": "600"},
        {"not-user.key": "value"},
        {"user.review-bot.value": "line\nbreak"},
        {"user.review-bot.value": "carriage\rreturn"},
        {"user.review-bot.value": "nul\x00value"},
        {"user.review-bot.value": 1},
        [("user.review-bot.value", "value")],
    ],
)
def test_invalid_instance_config_is_rejected(instance_config):
    with pytest.raises(ValueError, match="invalid_lxd_instance_config"):
        LxdVmLauncher("ubuntu:24.04", instance_config=instance_config)


@pytest.mark.parametrize(
    ("image", "profiles", "error"),
    [
        ("", ("default",), "invalid_lxd_image"),
        ("ubuntu 24.04", ("default",), "invalid_lxd_image"),
        ("ubuntu:24.04", ("bad profile",), "invalid_lxd_profile"),
        ("ubuntu:24.04", ("",), "invalid_lxd_profile"),
    ],
)
def test_invalid_image_or_profile_is_rejected(image, profiles, error):
    with pytest.raises(ValueError, match=error):
        LxdVmLauncher(image, profiles=profiles)


def test_launch_failure_is_idempotent_only_for_same_job():
    runner = FakeRunner(
        completed(returncode=1),
        completed(stdout=JOB_ID),
    )
    launcher = LxdVmLauncher("ubuntu:24.04", project="reviews", run=runner)
    assert launcher.launch(
        JOB_ID,
        max_lifetime_seconds=600,
        launch_token="launch-token",
    ) == VM_ID
    assert runner.calls[1][0] == [
        "lxc", "config", "get", "--project", "reviews", VM_ID,
        "user.review-bot.job-id",
    ]

    runner = FakeRunner(completed(returncode=1), completed(stdout="b" * 64))
    launcher = LxdVmLauncher("ubuntu:24.04", run=runner)
    with pytest.raises(RuntimeError, match="lxd_launch_failed"):
        launcher.launch(JOB_ID, max_lifetime_seconds=600, launch_token="launch-token")

    runner = FakeRunner(completed(returncode=1), completed(returncode=1))
    launcher = LxdVmLauncher("ubuntu:24.04", run=runner)
    with pytest.raises(RuntimeError, match="lxd_launch_failed"):
        launcher.launch(JOB_ID, max_lifetime_seconds=600, launch_token="launch-token")


def test_launch_timeout_uses_generic_error():
    runner = FakeRunner(subprocess.TimeoutExpired(["lxc"], 300))
    launcher = LxdVmLauncher("ubuntu:24.04", run=runner)
    with pytest.raises(RuntimeError, match="lxd_launch_failed"):
        launcher.launch(JOB_ID, max_lifetime_seconds=600, launch_token="launch-token")


def test_terminate_validates_name_and_handles_absent_instances():
    launcher = LxdVmLauncher("ubuntu:24.04", run=FakeRunner(completed()))
    with pytest.raises(ValueError, match="invalid_vm_id"):
        launcher.terminate("t1")
    launcher.terminate(VM_ID)

    runner = FakeRunner(completed(returncode=1), completed(returncode=1))
    LxdVmLauncher("ubuntu:24.04", run=runner).terminate(VM_ID)
    assert [call[0][1] for call in runner.calls] == ["delete", "info"]

    runner = FakeRunner(completed(returncode=1), completed())
    with pytest.raises(RuntimeError, match="lxd_terminate_failed"):
        LxdVmLauncher("ubuntu:24.04", run=runner).terminate(VM_ID)


def test_list_managed_instances_filters_invalid_and_other_deployments():
    def row(name, *, job_id=JOB_ID, managed="true", status="Running", team="blue"):
        return {
            "name": name,
            "status": status,
            "config": {
                "user.review-bot.managed": managed,
                "user.review-bot.job-id": job_id,
                "user.review-bot.team": team,
            },
        }

    rows = [
        row(VM_ID),
        row("not-managed", managed="false"),
        row("bad-job", job_id="bad"),
        row("stopped", status="Stopped"),
        row("not-a-vm-id"),
        row("other-team", team="red"),
    ]
    launcher = LxdVmLauncher(
        "ubuntu:24.04",
        project="reviews",
        instance_config={"user.review-bot.team": "blue"},
        run=FakeRunner(completed(stdout=json.dumps(rows))),
    )
    assert launcher.list_managed_instances() == [ManagedInstance(VM_ID, JOB_ID)]
    assert launcher.run.calls[0][0] == [
        "lxc", "list", "--project", "reviews", "--format", "json",
    ]


@pytest.mark.parametrize(
    "result",
    [completed(returncode=1), completed(stdout="{bad json"), completed(stdout="{}")],
)
def test_list_failure_raises_generic_error(result):
    launcher = LxdVmLauncher("ubuntu:24.04", run=FakeRunner(result))
    with pytest.raises(RuntimeError, match="lxd_list_failed"):
        launcher.list_managed_instances()


def test_guest_launcher_only_terminates_itself():
    launcher = LxdGuestLauncher(VM_ID)
    launcher.terminate(VM_ID)
    with pytest.raises(ValueError, match="invalid_vm_id"):
        LxdGuestLauncher("t1")
    with pytest.raises(RuntimeError, match="lxd_guest_cannot_terminate_other_vm"):
        launcher.terminate("rb-" + "b" * 20)
    with pytest.raises(RuntimeError, match="unsupported_in_guest"):
        launcher.launch(JOB_ID, max_lifetime_seconds=600, launch_token="token")
    with pytest.raises(RuntimeError, match="unsupported_in_guest"):
        launcher.list_managed_instances()


def test_launcher_from_environment_defaults_to_ec2_with_injected_client():
    client = object()
    calls = []
    launcher = _launcher_from_environment(
        {
            "AWS_REGION": "us-east-1",
            "REVIEW_BOT_LAUNCH_TEMPLATE_ID": "lt-123",
        },
        ec2_client_factory=lambda **kwargs: calls.append(kwargs) or client,
    )
    assert isinstance(launcher, Ec2VmLauncher)
    assert launcher.client is client
    assert launcher.launch_template_id == "lt-123"
    assert calls == [{"region_name": "us-east-1"}]


def test_launcher_from_environment_parses_lxd_settings():
    launcher = _launcher_from_environment(
        {
            "REVIEW_BOT_VM_PROVIDER": "lxd",
            "REVIEW_BOT_LXD_IMAGE": "review-bot-worker",
            "REVIEW_BOT_LXD_PROFILES": "default, review-bot-worker",
            "REVIEW_BOT_LXD_PROJECT": "reviews",
            "REVIEW_BOT_INSTANCE_TAGS": json.dumps({
                "user.review-bot.region": "us-east-1",
            }),
        },
        ec2_client_factory=lambda *_args, **_kwargs: pytest.fail("EC2 client was created"),
    )
    assert isinstance(launcher, LxdVmLauncher)
    assert launcher.profiles == ("default", "review-bot-worker")
    assert launcher.project == "reviews"
    assert launcher.instance_config == {"user.review-bot.region": "us-east-1"}


def test_launcher_from_environment_requires_lxd_image_and_supports_guest():
    factory = lambda *_args, **_kwargs: pytest.fail("EC2 client was created")
    with pytest.raises(ValueError, match="REVIEW_BOT_LXD_IMAGE is required"):
        _launcher_from_environment({"REVIEW_BOT_VM_PROVIDER": "lxd"}, ec2_client_factory=factory)
    guest = _launcher_from_environment(
        {"REVIEW_BOT_VM_PROVIDER": "lxd-guest", "REVIEW_BOT_VM_ID": VM_ID},
        ec2_client_factory=factory,
    )
    assert isinstance(guest, LxdGuestLauncher)
    assert guest.vm_id == VM_ID
    with pytest.raises(ValueError, match="unsupported_vm_provider"):
        _launcher_from_environment(
            {"REVIEW_BOT_VM_PROVIDER": "gce"},
            ec2_client_factory=factory,
        )
