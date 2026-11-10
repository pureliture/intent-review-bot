"""AWS Lambda reference handlers: signed webhook intake and scheduled VM-loss recovery.

Only metadata leaves the handler. Raw payloads, signatures and secrets are never logged or stored.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections.abc import Callable

from review_bot.adapters.github.webhook import verify_github_signature
from review_bot.application.ondemand import IntakeEvent, OnDemandCoordinator

MAX_PAYLOAD_BYTES = 1024 * 1024


def _response(status: int, outcome: str) -> dict:
    return {"statusCode": status, "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"outcome": outcome})}


def build_intake_handler(coordinator: OnDemandCoordinator, secret: Callable[[], str], *,
    allowed_installations: tuple[int, ...] = (), allowed_repositories: tuple[str, ...] = ()) -> Callable:
    def handler(event: dict, _context=None) -> dict:
        headers = {str(k).lower(): v for k, v in (event.get("headers") or {}).items()}
        body = event.get("body") or ""
        raw = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("utf-8")
        if len(raw) > MAX_PAYLOAD_BYTES:
            return _response(413, "payload_too_large")
        if not verify_github_signature(raw, headers.get("x-hub-signature-256"), secret()):
            return _response(401, "invalid_signature")
        delivery_id = headers.get("x-github-delivery")
        kind = headers.get("x-github-event")
        if not delivery_id:
            return _response(400, "missing_delivery")
        if kind == "ping":
            return _response(200, "pong")
        if kind != "pull_request":
            return _response(202, "ignored_event")
        try:
            payload = json.loads(raw)
            pr = payload["pull_request"]
            intake = IntakeEvent(delivery_id, hashlib.sha256(raw).hexdigest(), str(payload["action"]),
                int(payload["installation"]["id"]), int(payload["repository"]["id"]), int(pr["number"]),
                str(pr["base"]["sha"]), str(pr["head"]["sha"]))
            repository = str(payload["repository"]["full_name"])
        except (ValueError, KeyError, TypeError):
            return _response(400, "invalid_payload")
        if allowed_installations and intake.installation_id not in allowed_installations:
            return _response(403, "installation_not_allowed")
        if allowed_repositories and repository not in allowed_repositories:
            return _response(403, "repository_not_allowed")
        decision = coordinator.accept(intake)
        if decision.outcome == "replay_mismatch":
            return _response(409, decision.outcome)
        if decision.launch and decision.job_id:
            coordinator.launch(decision.job_id)
        return _response(202, decision.outcome)

    return handler


def build_recovery_handler(coordinator: OnDemandCoordinator) -> Callable:
    def handler(_event=None, _context=None) -> dict:
        actions = coordinator.recover()
        relaunched = [job_id for job_id, action in actions if action == "requeued" and coordinator.launch(job_id)]
        return {"recovered": len(actions), "relaunched": len(relaunched)}

    return handler


def _coordinator_from_environment() -> OnDemandCoordinator:  # pragma: no cover - requires AWS
    import boto3

    from review_bot.adapters.aws.dynamodb_job_store import DynamoDbJobStore
    from review_bot.adapters.aws.ec2_launcher import Ec2VmLauncher

    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    store = DynamoDbJobStore(boto3.client("dynamodb", region_name=region), os.environ["REVIEW_BOT_JOB_TABLE"])
    instance_tags = json.loads(os.environ.get("REVIEW_BOT_INSTANCE_TAGS", "{}"))
    if not isinstance(instance_tags, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in instance_tags.items()
    ):
        raise ValueError("REVIEW_BOT_INSTANCE_TAGS must be a string mapping")
    launcher = Ec2VmLauncher(boto3.client("ec2", region_name=region),
        os.environ.get("REVIEW_BOT_LAUNCH_TEMPLATE_ID", ""), instance_tags=instance_tags)
    return OnDemandCoordinator(store, launcher, clock=time.time,
        max_vm_lifetime_seconds=int(os.environ.get("REVIEW_BOT_MAX_VM_SECONDS", "3600")))


def _secret_from_environment() -> str:  # pragma: no cover - requires AWS
    import boto3

    return boto3.client("secretsmanager").get_secret_value(SecretId=os.environ["REVIEW_BOT_WEBHOOK_SECRET_ARN"])["SecretString"]


def intake(event, context):  # pragma: no cover - Lambda entry point
    ids = tuple(int(v) for v in os.environ.get("REVIEW_BOT_ALLOWED_INSTALLATIONS", "").split(",") if v)
    repos = tuple(v for v in os.environ.get("REVIEW_BOT_ALLOWED_REPOSITORIES", "").split(",") if v)
    return build_intake_handler(_coordinator_from_environment(), _secret_from_environment,
        allowed_installations=ids, allowed_repositories=repos)(event, context)


def recovery(event, context):  # pragma: no cover - Lambda entry point
    return build_recovery_handler(_coordinator_from_environment())(event, context)
