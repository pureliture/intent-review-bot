"""Smoke tests for review-bot CLI entrypoint."""
from __future__ import annotations

import pytest

from review_bot import __version__
from review_bot.adapters.pi_agent_docker import proxy_image_tag
from review_bot.runtime.cli import create_parser, main


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    parser = create_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out


def test_cli_parser_offline() -> None:
    parser = create_parser()
    args = parser.parse_args(
        ["offline", "--repo", "/path/to/repo", "--base", "main", "--head", "feature/abc"]
    )
    assert args.subcommand == "offline"
    assert args.repo == "/path/to/repo"
    assert args.base == "main"
    assert args.head == "feature/abc"


def test_cli_parser_webhook() -> None:
    parser = create_parser()
    args = parser.parse_args(["webhook", "--host", "0.0.0.0", "--port", "9000"])
    assert args.subcommand == "webhook"
    assert args.host == "0.0.0.0"
    assert args.port == 9000


def test_cli_parser_worker() -> None:
    parser = create_parser()
    args = parser.parse_args(["worker", "--once", "--config", "conf.yaml"])
    assert args.subcommand == "worker"
    assert args.once is True
    assert args.config == "conf.yaml"


def test_cli_main_invocation(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main(["offline"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "review-bot subcommand 'offline' initialized." in captured.out


def test_cli_agent_proxy_image(capsys: pytest.CaptureFixture[str]) -> None:
    expected = proxy_image_tag()
    assert main(["agent-proxy-image"]) == 0
    assert capsys.readouterr().out.strip() == expected


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage: review-bot" in captured.out


def test_ondemand_run_rejects_lifetime_below_minimum() -> None:
    with pytest.raises(SystemExit) as exc:
        main([
            "ondemand-run",
            "--job-id", "a" * 64,
            "--vm-lifetime-seconds", "59",
        ])
    assert exc.value.code == 2


def test_ondemand_recover_prints_json(monkeypatch, capsys: pytest.CaptureFixture[str]) -> None:
    from review_bot.runtime import aws_lambda

    coordinator = object()
    monkeypatch.setattr(aws_lambda, "_coordinator_from_environment", lambda: coordinator)
    monkeypatch.setattr(
        aws_lambda,
        "build_recovery_handler",
        lambda actual: (lambda: {"recovered": int(actual is coordinator), "relaunched": 1}),
    )
    assert main(["ondemand-recover"]) == 0
    assert capsys.readouterr().out.strip() == '{"recovered": 1, "relaunched": 1}'
