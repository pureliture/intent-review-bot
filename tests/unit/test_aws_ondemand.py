"""AWS reference path verified against moto's local (no-cost, no-network) DynamoDB/EC2 emulation."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from review_bot.adapters.aws.dynamodb_job_store import DynamoDbJobStore  # noqa: E402
from review_bot.adapters.aws.ec2_launcher import Ec2VmLauncher  # noqa: E402
from review_bot.application.ondemand import ManagedInstance, OnDemandCoordinator  # noqa: E402
from review_bot.runtime.aws_lambda import build_intake_handler, build_recovery_handler  # noqa: E402
from review_bot.runtime.ondemand_vm import run_job  # noqa: E402

SECRET = "test-webhook-secret"
REGION = "us-east-1"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def aws(monkeypatch):
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        monkeypatch.setenv(key, "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    with moto.mock_aws():
        ddb = boto3.client("dynamodb", region_name=REGION)
        ddb.create_table(TableName="jobs", BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": n, "AttributeType": "S"} for n in ("pk", "pr_key", "active")],
            KeySchema=[{"AttributeName": "pk", "KeyType": "HASH"}],
            GlobalSecondaryIndexes=[{"IndexName": f"{n}_index", "KeySchema": [{"AttributeName": a, "KeyType": "HASH"}],
                "Projection": {"ProjectionType": "KEYS_ONLY"}} for n, a in (("pr", "pr_key"), ("active", "active"))])
        ec2 = boto3.client("ec2", region_name=REGION)
        image = ec2.describe_images()["Images"][0]["ImageId"]
        template = ec2.create_launch_template(LaunchTemplateName="review-vm",
            LaunchTemplateData={"ImageId": image, "InstanceType": "t3.micro"})["LaunchTemplate"]["LaunchTemplateId"]
        clock = Clock()
        coordinator = OnDemandCoordinator(DynamoDbJobStore(ddb, "jobs"), Ec2VmLauncher(ec2, template), clock=clock,
            max_attempts=2, lease_seconds=60)
        yield coordinator, ec2, clock


def webhook(action="synchronize", head="b" * 40, delivery="d-1", secret=SECRET, pr=5):
    body = json.dumps({"action": action, "installation": {"id": 11},
        "repository": {"id": 22, "full_name": "o/r"},
        "pull_request": {"number": pr, "base": {"sha": "a" * 40}, "head": {"sha": head}}}).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {"headers": {"X-Hub-Signature-256": signature, "X-GitHub-Delivery": delivery, "X-GitHub-Event": "pull_request"},
        "body": base64.b64encode(body).decode(), "isBase64Encoded": True}


def states(ec2):
    return sorted(i["State"]["Name"] for r in ec2.describe_instances()["Reservations"] for i in r["Instances"])


def test_signed_webhook_launches_exactly_one_vm_even_on_redelivery(aws):
    coordinator, ec2, _ = aws
    handler = build_intake_handler(coordinator, lambda: SECRET, allowed_installations=(11,))
    assert json.loads(handler(webhook())["body"])["outcome"] == "accepted"
    assert json.loads(handler(webhook())["body"])["outcome"] == "duplicate_delivery"
    assert json.loads(handler(webhook(delivery="d-2"))["body"])["outcome"] == "duplicate_job"
    assert states(ec2) == ["running"]
    instance = ec2.describe_instances()["Reservations"][0]["Instances"][0]
    tags = {t["Key"]: t["Value"] for t in instance["Tags"]}
    assert tags["review-bot:managed"] == "true" and len(tags["review-bot:job-id"]) == 64


def test_bad_signature_disallowed_installation_and_replay_are_rejected_without_vm(aws):
    coordinator, ec2, _ = aws
    handler = build_intake_handler(coordinator, lambda: SECRET, allowed_installations=(99,))
    assert handler(webhook(secret="wrong"))["statusCode"] == 401
    assert handler(webhook())["statusCode"] == 403
    open_handler = build_intake_handler(coordinator, lambda: SECRET)
    assert open_handler(webhook())["statusCode"] == 202
    tampered = webhook(head="c" * 40)
    assert open_handler(tampered)["statusCode"] == 409
    assert states(ec2) == ["running"]


def test_vm_run_persists_trace_in_dynamodb_and_terminates_itself(aws):
    coordinator, ec2, _ = aws
    handler = build_intake_handler(coordinator, lambda: SECRET)
    handler(webhook())
    job = next(iter(coordinator.store.list_active()))
    calls = []

    def execute(claimed, is_current):
        calls.append(is_current())
        return {"architecture": "completed", "intent": "partial_review"}, {"stages": [{"stage": "final_review", "calls": 3}]}

    finished = run_job(coordinator, job.job_id, "vm-worker", execute, heartbeat_seconds=0.01)
    assert calls == [True]
    assert finished.state == "partial"
    stored = coordinator.store.get(job.job_id)
    assert stored.trace == {"stages": [{"stage": "final_review", "calls": 3}]}
    assert stored.mode_results == {"architecture": "completed", "intent": "partial_review"}
    assert states(ec2) == ["terminated"], "trace remains readable after the VM is gone"


def test_superseded_vm_cannot_write_and_lost_vm_is_relaunched_by_recovery(aws):
    coordinator, ec2, clock = aws
    handler = build_intake_handler(coordinator, lambda: SECRET)
    handler(webhook())
    old = next(iter(coordinator.store.list_active())).job_id
    assert coordinator.claim(old, "vm-old")
    handler(webhook(head="c" * 40, delivery="d-2"))
    assert coordinator.store.get(old).state == "superseded"
    assert not coordinator.is_current(old, "vm-old")
    new = next(j for j in coordinator.store.list_active())
    assert coordinator.claim(new.job_id, "vm-new")
    clock.now += 61
    result = build_recovery_handler(coordinator)()
    assert result == {"recovered": 1, "relaunched": 1}
    assert coordinator.store.get(new.job_id).attempts == 1 and coordinator.store.get(new.job_id).state == "launching"
    assert states(ec2).count("running") == 1 and states(ec2).count("terminated") == 2


def test_failed_claim_terminates_self_without_releasing_another_jobs_vm(aws):
    coordinator, ec2, _ = aws
    handler = build_intake_handler(coordinator, lambda: SECRET)
    handler(webhook(pr=5, delivery="d-first"))
    handler(webhook(pr=6, head="c" * 40, delivery="d-second"))
    jobs = sorted(coordinator.store.list_active(), key=lambda item: item.pr_number)
    job_with_other_vm, self_job = jobs
    other_vm, second_job_vm = job_with_other_vm.vm_id, self_job.vm_id
    self_vm = ec2.run_instances(ImageId=ec2.describe_images()["Images"][0]["ImageId"],
        InstanceType="t3.micro", MinCount=1, MaxCount=1)["Instances"][0]["InstanceId"]
    assert coordinator.claim(job_with_other_vm.job_id, "already-running")
    finished = run_job(coordinator, job_with_other_vm.job_id, "duplicate-worker",
        lambda *_: pytest.fail("failed claim must not execute"), self_vm_id=self_vm)
    assert finished is None
    current = coordinator.store.get(job_with_other_vm.job_id)
    assert current.vm_id == other_vm
    instances = {item["InstanceId"]: item["State"]["Name"] for reservation in ec2.describe_instances()["Reservations"]
        for item in reservation["Instances"]}
    assert instances[other_vm] == "running"
    assert instances[second_job_vm] == "running"
    assert instances[self_vm] == "terminated"


def test_failed_claim_with_own_vm_releases_it(aws):
    coordinator, ec2, _ = aws
    handler = build_intake_handler(coordinator, lambda: SECRET)
    handler(webhook())
    job = next(iter(coordinator.store.list_active()))
    assert coordinator.claim(job.job_id, "owner")
    assert run_job(coordinator, job.job_id, "duplicate",
        lambda *_: pytest.fail("failed claim must not execute"), self_vm_id=job.vm_id) is None
    assert states(ec2) == ["terminated"]


def test_cancelled_job_retains_state_after_terminate_failure_and_retries(aws, monkeypatch):
    coordinator, ec2, _ = aws
    handler = build_intake_handler(coordinator, lambda: SECRET)
    handler(webhook())
    job = next(iter(coordinator.store.list_active()))
    original_terminate = coordinator.launcher.terminate
    monkeypatch.setattr(coordinator.launcher, "terminate",
        lambda _vm: (_ for _ in ()).throw(RuntimeError("temporary")))

    handler(webhook(action="closed", delivery="close-job"))
    cancelled = coordinator.store.get(job.job_id)
    assert cancelled.state == "cancelled"
    assert cancelled.termination_error == "RuntimeError"
    assert states(ec2) == ["running"]

    monkeypatch.setattr(coordinator.launcher, "terminate", original_terminate)
    coordinator.recover()
    assert coordinator.store.get(job.job_id).state == "cancelled"
    assert coordinator.store.get(job.job_id).termination_error is None
    assert states(ec2) == ["terminated"]


def test_ec2_managed_instance_listing_paginates():
    class PagedClient:
        def __init__(self):
            self.calls = []

        def describe_instances(self, **kwargs):
            self.calls.append(kwargs)
            if "NextToken" not in kwargs:
                return {"Reservations": [{"Instances": [{"InstanceId": "i-one", "Tags": [
                    {"Key": "review-bot:job-id", "Value": "a" * 64}]}]}], "NextToken": "page-2"}
            return {"Reservations": [{"Instances": [{"InstanceId": "i-two", "Tags": [
                {"Key": "review-bot:job-id", "Value": "b" * 64}]}]}]}

    client = PagedClient()
    launcher = Ec2VmLauncher(client, "lt-id")
    assert launcher.list_managed_instances() == [
        ManagedInstance("i-one", "a" * 64), ManagedInstance("i-two", "b" * 64),
    ]
    assert client.calls[1]["NextToken"] == "page-2"
    assert {item["Name"] for item in client.calls[0]["Filters"]} == {
        "tag:review-bot:managed", "tag:review-bot:job-id", "instance-state-name",
    }


def test_ec2_managed_instance_listing_filters_deployment_tags():
    class Client:
        def __init__(self):
            self.filters = None

        def describe_instances(self, **kwargs):
            self.filters = kwargs["Filters"]
            return {"Reservations": []}

    client = Client()
    launcher = Ec2VmLauncher(client, "lt-id", instance_tags={
        "team": "review-bot", "review-bot:job-table": "review-bot-prod",
    })

    assert launcher.list_managed_instances() == []
    assert client.filters[-2:] == [
        {"Name": "tag:review-bot:job-table", "Values": ["review-bot-prod"]},
        {"Name": "tag:team", "Values": ["review-bot"]},
    ]


def test_ec2_launcher_adds_bootstrap_tags_without_overriding_job_tags(aws):
    coordinator, ec2, _ = aws
    launcher = Ec2VmLauncher(
        ec2,
        coordinator.launcher.launch_template_id,
        instance_tags={
            "review-bot:job-table": "jobs",
            "review-bot:managed": "false",
            "review-bot:job-id": "0" * 64,
        },
    )
    job_id = "a" * 64
    vm_id = launcher.launch(job_id, max_lifetime_seconds=600, launch_token="launch-token")
    instance = next(
        instance
        for reservation in ec2.describe_instances()["Reservations"]
        for instance in reservation["Instances"]
        if instance["InstanceId"] == vm_id
    )
    tags = {tag["Key"]: tag["Value"] for tag in instance["Tags"]}
    assert tags["review-bot:managed"] == "true"
    assert tags["review-bot:job-id"] == job_id
    assert tags["review-bot:job-table"] == "jobs"
    user_data = base64.b64decode(
        ec2.describe_instance_attribute(InstanceId=vm_id, Attribute="userData")["UserData"]["Value"]
    ).decode()
    assert f"# review-bot-ondemand --job-id {job_id}" in user_data
    assert "# review-bot-max-lifetime-seconds 600" in user_data
    assert "exec /opt/review-bot/bin/review-bot-ondemand" not in user_data


def test_concurrent_claims_on_dynamodb_yield_single_owner(aws):
    coordinator, _, _ = aws
    build_intake_handler(coordinator, lambda: SECRET)(webhook())
    job_id = next(iter(coordinator.store.list_active())).job_id
    owners, barrier = [], threading.Barrier(4)

    def claim(name):
        barrier.wait()
        if coordinator.claim(job_id, name):
            owners.append(name)

    threads = [threading.Thread(target=claim, args=(f"vm-{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(owners) == 1
