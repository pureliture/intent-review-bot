"""Shared one-shot bootstrap payload for VM launchers."""
from __future__ import annotations


def one_shot_user_data(job_id: str, max_lifetime_seconds: int) -> str:
    minutes = max(1, max_lifetime_seconds // 60)
    return (
        "#!/bin/bash\nset -eu\n"
        f"shutdown -h +{minutes}\n"
        f"# review-bot-max-lifetime-seconds {max_lifetime_seconds}\n"
        f"# review-bot-ondemand --job-id {job_id}\n"
    )
