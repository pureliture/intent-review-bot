"""EC2 implementation of ``VmLauncherPort`` for one-shot Linux review workers."""
from __future__ import annotations

import re

from review_bot.application.ondemand import ManagedInstance

_JOB_ID = re.compile(r"^[0-9a-f]{64}$")


class Ec2VmLauncher:
    """Launches from an operator-owned launch template (AMI, subnet, SG, instance profile).

    The VM receives only the job id. It shuts itself down after ``max_lifetime_seconds`` and
    ``InstanceInitiatedShutdownBehavior=terminate`` turns every shutdown into termination.
    """

    def __init__(self, client, launch_template_id: str, *, tag_prefix: str = "review-bot",
        instance_tags: dict[str, str] | None = None) -> None:
        self.client, self.launch_template_id, self.tag_prefix = client, launch_template_id, tag_prefix
        self.instance_tags = dict(instance_tags or {})

    def launch(self, job_id: str, *, max_lifetime_seconds: int, launch_token: str) -> str:
        if not _JOB_ID.match(job_id) or type(max_lifetime_seconds) is not int or not 60 <= max_lifetime_seconds <= 6 * 3600:
            raise ValueError("invalid_launch_request")
        minutes = max(1, max_lifetime_seconds // 60)
        user_data = ("#!/bin/bash\nset -eu\n"
            f"shutdown -h +{minutes}\n"
            f"# review-bot-max-lifetime-seconds {max_lifetime_seconds}\n"
            f"# review-bot-ondemand --job-id {job_id}\n")
        tags = {
            **self.instance_tags,
            f"{self.tag_prefix}:job-id": job_id,
            f"{self.tag_prefix}:managed": "true",
        }
        response = self.client.run_instances(
            LaunchTemplate={"LaunchTemplateId": self.launch_template_id, "Version": "$Default"},
            MinCount=1, MaxCount=1, ClientToken=launch_token[:64],
            InstanceInitiatedShutdownBehavior="terminate", UserData=user_data,
            MetadataOptions={
                "HttpTokens": "required",
                "HttpPutResponseHopLimit": 1,
                "InstanceMetadataTags": "enabled",
            },
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": key, "Value": value} for key, value in tags.items()]}],
        )
        return response["Instances"][0]["InstanceId"]

    def terminate(self, vm_id: str) -> None:
        self.client.terminate_instances(InstanceIds=[vm_id])

    def list_managed_instances(self) -> list[ManagedInstance]:
        instances = []
        filters = [
            {"Name": f"tag:{self.tag_prefix}:managed", "Values": ["true"]},
            {"Name": f"tag:{self.tag_prefix}:job-id", "Values": ["*"]},
            {"Name": "instance-state-name", "Values": ["pending", "running"]},
        ]
        filters.extend(
            {"Name": f"tag:{key}", "Values": [self.instance_tags[key]]}
            for key in sorted(self.instance_tags)
        )
        token = None
        while True:
            request = {"Filters": filters}
            if token is not None:
                request["NextToken"] = token
            page = self.client.describe_instances(**request)
            for reservation in page.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    tags = {tag.get("Key"): tag.get("Value") for tag in instance.get("Tags", [])}
                    vm_id = instance.get("InstanceId")
                    job_id = tags.get(f"{self.tag_prefix}:job-id")
                    if isinstance(vm_id, str) and isinstance(job_id, str) and job_id:
                        instances.append(ManagedInstance(vm_id, job_id))
            token = page.get("NextToken")
            if not token:
                return instances
