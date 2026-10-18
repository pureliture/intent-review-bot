"""Guard the Compose mount options which are validated only at container creation."""
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize("service", ["web", "worker"])
def test_tmpfs_options_are_one_absolute_mount(service):
    config = yaml.safe_load((Path(__file__).parents[2] / "compose.yaml").read_text())
    target = config["services"][service]
    assert target["tmpfs"] == ["/tmp:mode=1777,size=64m"]
    assert target["read_only"] is True
    assert target["cap_drop"] == ["ALL"]
    assert target["security_opt"] == ["no-new-privileges:true"]
    assert target["environment"]["REVIEW_BOT_STATE_DIR"] == "/var/lib/review-bot"
    assert target["environment"]["REVIEW_BOT_MODEL_STATE_DIR"] == "/var/lib/review-bot-models"
