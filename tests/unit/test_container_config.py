"""Guard the Compose mount options which are validated only at container creation."""
from pathlib import Path
import runpy

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


@pytest.mark.parametrize("service", ["web", "worker"])
def test_nonroot_services_receive_readable_network_configuration(service):
    config = yaml.safe_load((Path(__file__).parents[2] / "compose.yaml").read_text())
    mounts = {item["target"]: item for item in config["services"][service]["volumes"]}
    for target in ("/etc/hosts", "/etc/resolv.conf"):
        item = mounts[target]
        assert item["type"] == "bind" and item["read_only"] is True
        assert item["bind"]["create_host_path"] is False
        assert "REVIEW_BOT_NETWORK_CONFIG_DIR" in item["source"]
    assert mounts["/etc/hosts"]["source"].endswith("/hosts")
    assert mounts["/etc/resolv.conf"]["source"].endswith("/resolv.conf")
    assert "configs" not in config and "configs" not in config["services"][service]
    assert "extra_hosts" not in config["services"][service]


def network_preparer():
    return runpy.run_path(str(Path(__file__).parents[2] / "deploy/prepare-network.py"))["prepare_network"]


def test_network_preparation_keeps_host_resolver_and_private_directory(tmp_path):
    resolver = tmp_path / "source-resolver"
    resolver.write_text("nameserver 127.0.0.53\nsearch synthetic.invalid\n")
    directory = tmp_path / "network"
    prepare = network_preparer()
    prepare(directory, "100.64.0.1", resolver=resolver)
    assert directory.stat().st_mode & 0o777 == 0o700
    for name in ("hosts", "resolv.conf"):
        assert (directory / name).stat().st_mode & 0o777 == 0o644
    assert "100.64.0.1 ubuntu-ms-7e07" in (directory / "hosts").read_text()
    assert (directory / "resolv.conf").read_text().endswith(resolver.read_text())
    prepare(directory, "100.64.0.2", resolver=resolver)
    assert "100.64.0.2 ubuntu-ms-7e07" in (directory / "hosts").read_text()
    assert resolver.read_text() == "nameserver 127.0.0.53\nsearch synthetic.invalid\n"


@pytest.mark.parametrize("linked", [False, True])
def test_network_preparation_preserves_unrelated_files(tmp_path, linked):
    resolver = tmp_path / "resolver"
    resolver.write_text("nameserver 127.0.0.53\n")
    directory = tmp_path / "network"
    directory.mkdir(mode=0o700)
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("preserve me")
    if linked:
        (directory / "hosts").symlink_to(unrelated)
    else:
        (directory / "hosts").write_text("preserve me")
    with pytest.raises(ValueError, match="unrelated"):
        network_preparer()(directory, "100.64.0.1", resolver=resolver)
    assert unrelated.read_text() == "preserve me"
    assert (directory / "hosts").read_text() == "preserve me"
