"""EC2 implementation of ``VmLauncherPort`` for one-shot Linux review workers."""
from __future__ import annotations

import re

_JOB_ID = re.compile(r"^[0-9a-f]{64}$")


class Ec2VmLauncher:
    """Launches from an operator-owned launch template (AMI, subnet, SG, instance profile).

    The VM receives only the job id. It shuts itself down after ``max_lifetime_seconds`` and
    ``InstanceInitiatedShutdownBehavior=terminate`` turns every shutdown into termination.
    """

    def __init__(self, client, launch_template_id: str, *, tag_prefix: str = "review-bot") -> None:
        self.client, self.launch_template_id, self.tag_prefix = client, launch_template_id, tag_prefix

    def launch(self, job_id: str, *, max_lifetime_seconds: int, launch_token: str) -> str:
        if not _JOB_ID.match(job_id) or type(max_lifetime_seconds) is not int or not 60 <= max_lifetime_seconds <= 6 * 3600:
            raise ValueError("invalid_launch_request")
        minutes = max(1, max_lifetime_seconds // 60)
        user_data = ("#!/bin/bash\nset -eu\n"
            f"shutdown -h +{minutes}\n"
            f"exec /opt/review-bot/bin/review-bot-ondemand --job-id {job_id}\n")
        response = self.client.run_instances(
            LaunchTemplate={"LaunchTemplateId": self.launch_template_id, "Version": "$Default"},
            MinCount=1, MaxCount=1, ClientToken=launch_token[:64],
            InstanceInitiatedShutdownBehavior="terminate", UserData=user_data,
            MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": f"{self.tag_prefix}:job-id", "Value": job_id},
                {"Key": f"{self.tag_prefix}:managed", "Value": "true"}]}],
        )
        return response["Instances"][0]["InstanceId"]

    def terminate(self, vm_id: str) -> None:
        self.client.terminate_instances(InstanceIds=[vm_id])
