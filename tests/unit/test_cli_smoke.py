"""Smoke tests for review-bot CLI entrypoint."""
from __future__ import annotations

import pytest

from review_bot import __version__
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


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage: review-bot" in captured.out
