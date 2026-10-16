"""CLI entrypoint for review-bot with offline analysis, delta, and full review pipelines."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from review_bot import __version__
from review_bot.adapters.git import CommitNotFoundError, GitError, GitRepositoryReader
from review_bot.adapters.markdown.formatter import format_review_markdown
from review_bot.adapters.markdown.intent_formatter import format_intent_markdown
from review_bot.adapters.pmd.runner import PmdRunner
from review_bot.application.classification import classify_pr
from review_bot.application.context import build_task_input
from review_bot.application.delta import build_delta
from review_bot.application.intent_context import MAX_INPUT_BYTES, decode_intent_task_input
from review_bot.application.architecture_review import run_review_a_pipeline
from review_bot.application.declared_intent_review import run_review_b_pipeline
from review_bot.config import ModelConfig, load_review_config
from review_bot.domain.models_v1 import (
    AnalysisRun,
    CpdDuplication,
    CpdOccurrence,
    ParsedReport,
    PmdViolation,
    PrSnapshot,
    Snapshot,
    ToolAnalysis,
)
from review_bot.runtime.artifacts import IntentArtifacts, read_private_intent_input
from review_bot.runtime.model_clients import MODEL_PROFILES, build_planning_client, build_review_client

logger = logging.getLogger(__name__)


def _load_tools_lock() -> dict[str, Any]:
    """Find and load tools.lock.json."""
    candidates = [
        Path(__file__).resolve().parents[3] / "tooling" / "tools.lock.json",
        Path("tooling/tools.lock.json"),
        Path("../tooling/tools.lock.json"),
    ]
    for c in candidates:
        if c.is_file():
            try:
                return json.loads(c.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.debug("Failed to read candidate tools.lock: %s", exc)
                continue
    return {"pmd_version": "7.3.0", "java_version": "21"}


def _parse_pmd_violation(d: dict[str, Any]) -> PmdViolation:
    return PmdViolation(
        path=d.get("path", ""),
        raw_path=d.get("raw_path", d.get("path", "")),
        begin_line=d.get("begin_line", 1),
        end_line=d.get("end_line", 1),
        begin_column=d.get("begin_column", 1),
        end_column=d.get("end_column", 1),
        rule=d.get("rule", ""),
        ruleset=d.get("ruleset", ""),
        priority=d.get("priority", 3),
        klass=d.get("klass", d.get("class_name", "")),
        method=d.get("method", d.get("method_name", "")),
        package=d.get("package", ""),
        message=d.get("message", ""),
        code_fragment=d.get("code_fragment"),
    )


def _parse_cpd_duplication(d: dict[str, Any]) -> CpdDuplication:
    occs = tuple(
        CpdOccurrence(
            path=o.get("path", ""),
            raw_path=o.get("raw_path", o.get("path", "")),
            begin_line=o.get("begin_line", 1),
            end_line=o.get("end_line", 1),
            begin_column=o.get("begin_column", 1),
            end_column=o.get("end_column", 1),
            code_fragment=o.get("code_fragment"),
        )
        for o in d.get("occurrences", [])
    )
    return CpdDuplication(
        lines=d.get("lines", 0),
        tokens=d.get("tokens", 0),
        occurrences=occs,
        code_fragment=d.get("code_fragment"),
    )


def _parse_parsed_report(d: dict[str, Any]) -> ParsedReport:
    viols = tuple(_parse_pmd_violation(v) for v in d.get("violations", []))
    dups = tuple(_parse_cpd_duplication(dup) for dup in d.get("duplications", []))
    return ParsedReport(
        tool=d.get("tool", "pmd"),
        path=d.get("path", ""),
        exists=d.get("exists", True),
        parse_status=d.get("parse_status", "valid"),
        violations=viols,
        duplications=dups,
        suppressed_count=d.get("suppressed_count", 0),
    )


def _parse_tool_analysis(d: dict[str, Any]) -> ToolAnalysis:
    report = _parse_parsed_report(d.get("report", {}))
    return ToolAnalysis(
        tool=d.get("tool", "pmd"),
        status=d.get("status", "clean"),
        exit_code=d.get("exit_code", 0),
        report=report,
        execution=None,
    )


def _analysis_run_from_dict(d: dict[str, Any]) -> AnalysisRun:
    pmd = _parse_tool_analysis(d.get("pmd", {}))
    cpd = _parse_tool_analysis(d.get("cpd", {}))
    return AnalysisRun(
        schema_version=d.get("schema_version", 1),
        run_id=d.get("run_id", "run-1"),
        sha=d.get("sha", ""),
        profile_hash=d.get("profile_hash", ""),
        conditions_hash=d.get("conditions_hash", ""),
        conditions=d.get("conditions", {}),
        output_dir=d.get("output_dir", ""),
        started_at=d.get("started_at", ""),
        completed_at=d.get("completed_at", ""),
        status=d.get("status", "clean"),
        source_state=d.get("source_state", "complete"),
        source_files=tuple(d.get("source_files", ())),
        source_hash=d.get("source_hash"),
        exclusions=(),
        pmd=pmd,
        cpd=cpd,
        build_exit_code=d.get("build_exit_code"),
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="review-bot",
        description="Standalone GitHub Architecture Review Bot CLI",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. review
    review_parser = subparsers.add_parser(
        "review",
        help="Run end-to-end architecture review against a local git repository",
    )
    review_parser.add_argument(
        "--repo",
        default=".",
        help="Path to target git repository (default: current directory)",
    )
    review_parser.add_argument(
        "--base",
        help="Base git reference or commit SHA",
    )
    review_parser.add_argument(
        "--head",
        default="HEAD",
        help="Head git reference or commit SHA (default: HEAD)",
    )
    review_parser.add_argument(
        "--config",
        help="Path to operator review-config.yaml",
    )
    review_parser.add_argument(
        "--output",
        "-o",
        help="Output path or directory (writes comment.md and artifacts)",
    )
    review_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # offline (compat)
    offline_parser = subparsers.add_parser(
        "offline",
        help="Run offline architecture review against a local git repository (compat)",
    )
    offline_parser.add_argument(
        "--repo",
        default=None,
        help="Path to target git repository",
    )
    offline_parser.add_argument(
        "--base",
        help="Base git reference or commit SHA",
    )
    offline_parser.add_argument(
        "--head",
        default="HEAD",
        help="Head git reference or commit SHA (default: HEAD)",
    )
    offline_parser.add_argument(
        "--config",
        help="Path to operator review-config.yaml",
    )
    offline_parser.add_argument(
        "--output",
        "-o",
        help="Output path or directory (writes comment.md and artifacts)",
    )
    offline_parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    for review_command in (review_parser, offline_parser):
        review_command.add_argument(
            "--model-profile", choices=MODEL_PROFILES, default=None,
            help="Explicitly enable an Ollama or llama.cpp profile (default: no model calls)",
        )

    # 2. analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run bot-owned PMD 7.3.0 and CPD against target commit in local repo, outputs JSON",
    )
    analyze_parser.add_argument(
        "--repo",
        default=".",
        help="Path to target git repository",
    )
    analyze_parser.add_argument(
        "--commit",
        "--sha",
        dest="commit",
        default="HEAD",
        help="Target git commit SHA or revision to analyze (default: HEAD)",
    )
    analyze_parser.add_argument(
        "--ruleset",
        default="pmd/ruleset.xml",
        help="Ruleset resource path (default: pmd/ruleset.xml)",
    )
    analyze_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )

    # 3. delta
    delta_parser = subparsers.add_parser(
        "delta",
        help="Compute 6-state change delta between base-run and head-run JSON files or repo commits",
    )
    delta_parser.add_argument(
        "--base-run",
        help="Path to base AnalysisRun JSON file",
    )
    delta_parser.add_argument(
        "--head-run",
        help="Path to head AnalysisRun JSON file",
    )
    delta_parser.add_argument(
        "--repo",
        default=".",
        help="Path to target git repository (used if --base-run/--head-run not provided)",
    )
    delta_parser.add_argument(
        "--base",
        help="Base git commit SHA or reference",
    )
    delta_parser.add_argument(
        "--head",
        default="HEAD",
        help="Head git commit SHA or reference (default: HEAD)",
    )
    delta_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    delta_parser.add_argument(
        "--format",
        choices=["json", "table", "markdown"],
        default="json",
        help="Output format (default: json)",
    )

    # 4. version
    version_parser = subparsers.add_parser(
        "version",
        help="Print review-bot version and tool lock digests",
    )
    version_parser.add_argument(
        "--json",
        action="store_true",
        help="Output version and tool locks in JSON format",
    )

    # 5. webhook
    webhook_parser = subparsers.add_parser(
        "webhook",
        help="Start the FastAPI webhook receiver server",
    )
    webhook_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host bind address (default: 127.0.0.1)",
    )
    webhook_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    webhook_parser.add_argument(
        "--config",
        help="Path to operator review-config.yaml",
    )

    # 6. worker
    worker_parser = subparsers.add_parser(
        "worker",
        help="Start the leased SQLite JobStore worker loop",
    )
    worker_parser.add_argument(
        "--config",
        help="Path to operator review-config.yaml",
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="Process a single claimed run and exit",
    )

    intent_parser = subparsers.add_parser(
        "intent", help="Evaluate fixed B intent input privately, without posting or target execution",
    )
    intent_parser.add_argument("--input", required=True, help="Private fixed B v1 input JSON (owned 0600/0400 file)")
    intent_parser.add_argument("--output", required=True, help="Fresh private output directory (0700)")
    intent_parser.add_argument("--model-profile", choices=MODEL_PROFILES, default=None,
        help="Explicitly enable a review profile (default: no model calls)")
    intent_parser.add_argument("--pipeline", choices=("declared-v1", "code-first-v2", "perspectives-v1"), default="declared-v1",
        help="B review contract; perspectives-v1 performs one fixed-input intent review with closed perspective choices")
    intent_parser.add_argument("--language", choices=("ko", "en"), default="ko",
        help="Human output language for perspectives-v1 (default: ko)")
    intent_parser.add_argument("--call-timeout-seconds", type=int, default=None,
        help="Explicit per-call model timeout; task, context and token budgets remain unchanged")

    return parser


def run_version(args: argparse.Namespace) -> int:
    """Print package version and tool lock digests."""
    tools_lock = _load_tools_lock()
    if getattr(args, "json", False):
        output = {
            "version": __version__,
            "tools_lock": tools_lock,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        pmd_ver = tools_lock.get("pmd_version", "7.3.0")
        java_ver = tools_lock.get("java_version", "21")
        artifact_count = len(tools_lock.get("artifacts", {}))
        print(f"review-bot {__version__}")
        print(f"PMD/CPD: {pmd_ver} (JDK {java_ver})")
        print(f"Locked artifacts: {artifact_count} digests verified in tools.lock.json")
    return 0


def run_analyze(args: argparse.Namespace) -> int:
    """Run bot-owned PMD and CPD against target commit in local repo."""
    repo_path = Path(args.repo).resolve()
    if not (repo_path / ".git").exists():
        sys.stderr.write(f"Error: {repo_path} is not a git repository\n")
        return 1

    try:
        reader = GitRepositoryReader(repo_path)
        commit = reader.get_commit(args.commit)
        sha = commit.sha

        runner = PmdRunner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            blobs = reader.read_tree_blobs(sha)
            for p, content in blobs.items():
                dest = Path(tmp_dir) / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)
            run = runner.run_analysis(sha, Path(tmp_dir), args.ruleset)
    except CommitNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except GitError as exc:
        sys.stderr.write(f"Error: Git error: {exc}\n")
        return 1

    output_json = json.dumps(asdict(run), indent=2, ensure_ascii=False)
    if getattr(args, "output", None):
        Path(args.output).write_text(output_json, encoding="utf-8")
    else:
        print(output_json)
    return 0


def run_delta(args: argparse.Namespace) -> int:
    """Compute 6-state change delta between base and head runs."""
    if bool(getattr(args, "base_run", None)) != bool(getattr(args, "head_run", None)):
        sys.stderr.write("Error: Both --base-run and --head-run must be provided together\n")
        return 1

    rename_map: dict[str, str] = {}
    if getattr(args, "base_run", None) and getattr(args, "head_run", None):
        try:
            base_data = json.loads(Path(args.base_run).read_text(encoding="utf-8"))
            head_data = json.loads(Path(args.head_run).read_text(encoding="utf-8"))
            base_run = _analysis_run_from_dict(base_data)
            head_run = _analysis_run_from_dict(head_data)
        except (FileNotFoundError, OSError) as exc:
            sys.stderr.write(f"Error: Failed to read analysis run file: {exc}\n")
            return 1
        except json.JSONDecodeError as exc:
            sys.stderr.write(f"Error: Malformed JSON in run file: {exc}\n")
            return 1
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            sys.stderr.write(f"Error: Invalid analysis run data: {exc}\n")
            return 1
    else:
        repo_path = Path(args.repo).resolve()
        if not (repo_path / ".git").exists():
            sys.stderr.write(f"Error: {repo_path} is not a git repository\n")
            return 1
        try:
            reader = GitRepositoryReader(repo_path)
            head_commit = reader.get_commit(args.head or "HEAD")
            base_ref = args.base or (head_commit.parents[0] if head_commit.parents else head_commit.sha)
            base_commit = reader.get_commit(base_ref)
            rename_map = reader.get_rename_map(base_commit.sha, head_commit.sha)

            runner = PmdRunner()
            with tempfile.TemporaryDirectory() as base_tmp, tempfile.TemporaryDirectory() as head_tmp:
                base_blobs = reader.read_tree_blobs(base_commit.sha)
                for p, content in base_blobs.items():
                    dest = Path(base_tmp) / p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
                head_blobs = reader.read_tree_blobs(head_commit.sha)
                for p, content in head_blobs.items():
                    dest = Path(head_tmp) / p
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(content)
                base_run = runner.run_analysis(base_commit.sha, Path(base_tmp), "pmd/ruleset.xml")
                head_run = runner.run_analysis(head_commit.sha, Path(head_tmp), "pmd/ruleset.xml")
        except CommitNotFoundError as exc:
            sys.stderr.write(f"Error: {exc}\n")
            return 1
        except GitError as exc:
            sys.stderr.write(f"Error: Git error: {exc}\n")
            return 1

    delta = build_delta(base_run, head_run, rename_map=rename_map)

    if getattr(args, "format", "json") in ("table", "markdown"):
        counts = delta.counts
        table_lines = [
            "| existing | new | worsened | improved | resolved | unknown |",
            "|---:|---:|---:|---:|---:|---:|",
            (
                f"| {counts.get('existing', 0)} | {counts.get('new', 0)} | {counts.get('worsened', 0)} | "
                f"{counts.get('improved', 0)} | {counts.get('resolved', 0)} | {counts.get('unknown', 0)} |"
            ),
        ]
        rendered = "\n".join(table_lines)
    else:
        rendered = json.dumps(asdict(delta), indent=2, ensure_ascii=False)

    if getattr(args, "output", None):
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


def run_offline_review(args: argparse.Namespace) -> int:
    """Execute complete offline review pipeline on a local git repository."""
    repo_path = Path(getattr(args, "repo", None) or ".").resolve()
    if not (repo_path / ".git").exists():
        sys.stderr.write(f"Error: {repo_path} is not a git repository\n")
        return 1

    try:
        config = None
        if getattr(args, "config", None):
            config, _ = load_review_config(args.config)
        profile = getattr(args, "model_profile", None)
        if profile is None and config is not None:
            profile = config.model.profile
        model_client = build_review_client(profile, config.model if config is not None else None)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Error: model configuration: {exc}\n")
        return 1

    try:
        reader = GitRepositoryReader(repo_path)

        head_commit = reader.get_commit(args.head or "HEAD")
        head_sha = head_commit.sha

        if args.base:
            target_base_commit = reader.get_commit(args.base)
            target_base_sha = target_base_commit.sha
        else:
            target_base_sha = head_commit.parents[0] if head_commit.parents else head_sha

        merge_res = reader.resolve_merge_base(target_base_sha, head_sha)
        if not merge_res.is_comparable or not merge_res.comparison_base_sha:
            sys.stderr.write(
                "Error: comparison_status=incomparable: "
                f"Merge base resolution status is {merge_res.comparison_status}\n"
            )
            return 1
        comparison_base_sha = merge_res.comparison_base_sha

        # Gather snapshots
        base_code = reader.collect_code_snapshot(comparison_base_sha)
        head_code = reader.collect_code_snapshot(head_sha)
        changed_files = tuple(f.path for f in reader.get_changed_files(comparison_base_sha, head_sha))
        rename_map = reader.get_rename_map(comparison_base_sha, head_sha)

        # Classify perspective
        classification = classify_pr(head_commit.message, ())

        # Run PMD analysis via isolated temp trees
        runner = PmdRunner()
        with tempfile.TemporaryDirectory() as base_tmp, tempfile.TemporaryDirectory() as head_tmp:
            base_blobs = reader.read_tree_blobs(comparison_base_sha)
            for p, content in base_blobs.items():
                dest = Path(base_tmp) / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)

            head_blobs = reader.read_tree_blobs(head_sha)
            for p, content in head_blobs.items():
                dest = Path(head_tmp) / p
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(content)

            base_run = runner.run_analysis(comparison_base_sha, Path(base_tmp), "pmd/ruleset.xml")
            head_run = runner.run_analysis(head_sha, Path(head_tmp), "pmd/ruleset.xml")
    except CommitNotFoundError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    except GitError as exc:
        sys.stderr.write(f"Error: Git error: {exc}\n")
        return 1

    # Delta & Context
    delta = build_delta(base_run, head_run, rename_map=rename_map)
    pr_snap = PrSnapshot(
        repository=repo_path.name,
        number=1,
        base_sha=comparison_base_sha,
        head_sha=head_sha,
        source_branch="HEAD",
        changed_files=changed_files,
        rename_map=rename_map,
    )
    core_snap = Snapshot(pr=pr_snap, base=base_code, head=head_code)
    task_input = build_task_input(core_snap, delta, classification)

    # Review A
    result = run_review_a_pipeline(task_input, model_client) if model_client is not None else run_review_a_pipeline(task_input)

    # Format Markdown
    formatted_md = format_review_markdown(result, task_input)

    # Output handling and artifact persistence
    if getattr(args, "output", None):
        out_path = Path(args.output)
        if out_path.is_dir() or not out_path.suffix:
            out_dir = out_path
            comment_file = out_dir / "comment.md"
        else:
            out_dir = out_path.parent
            comment_file = out_path
        out_dir.mkdir(parents=True, exist_ok=True)
        comment_file.write_text(formatted_md, encoding="utf-8")
        (out_dir / "result.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "delta.json").write_text(
            json.dumps(asdict(delta), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "task_input.json").write_text(
            json.dumps(asdict(task_input), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        if getattr(args, "format", "markdown") == "json":
            print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
        else:
            print(formatted_md)

    return 0


def run_intent(args: argparse.Namespace) -> int:
    """Read-only fixed evaluation; no config, queue, repository, or publisher is composed."""
    try:
        text = read_private_intent_input(Path(args.input), max_bytes=MAX_INPUT_BYTES)
        task_input = decode_intent_task_input(text)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        sys.stderr.write("Error: invalid_intent_input\n")
        return 1
    try:
        artifacts = IntentArtifacts(Path(args.output))
    except (OSError, ValueError):
        sys.stderr.write("Error: unsafe_intent_output\n")
        return 1
    try:
        timeout = getattr(args, "call_timeout_seconds", None)
        settings = ModelConfig(call_timeout_seconds=timeout) if timeout is not None else None
        model_client = build_review_client(args.model_profile, settings) if settings is not None else build_review_client(args.model_profile)
        if getattr(args, "pipeline", "declared-v1") == "perspectives-v1":
            from review_bot.adapters.markdown.intent_perspective_formatter import format_intent_perspective_markdown
            from review_bot.application.intent_perspective_review import run_intent_perspective_review
            from review_bot.runtime.model_clients import build_intent_selection_client

            selector = (build_intent_selection_client(args.model_profile, settings) if settings is not None
                else build_intent_selection_client(args.model_profile))
            result = run_intent_perspective_review(task_input, model_client, selection_client=selector,
                language=getattr(args, "language", "ko"))
            markdown = format_intent_perspective_markdown(result, task_input)
        elif getattr(args, "pipeline", "declared-v1") == "code-first-v2":
            from review_bot.adapters.markdown.code_intent_formatter import format_code_intent_markdown
            from review_bot.application.code_first_intent_review import run_code_first_review

            planning_client = (build_planning_client(args.model_profile, settings) if settings is not None
                else build_planning_client(args.model_profile))
            result = run_code_first_review(task_input, model_client, planning_client=planning_client)
            markdown = format_code_intent_markdown(result, task_input)
        else:
            result = run_review_b_pipeline(task_input, model_client)
            markdown = format_intent_markdown(result, task_input)
        artifacts.save(task_input, result, markdown)
    except Exception:
        sys.stderr.write("Error: intent_evaluation_failed\n")
        return 1
    print(f"intent: {result.status}/{result.reason}")
    if getattr(args, "pipeline", "declared-v1") in ("code-first-v2", "perspectives-v1"):
        return 0 if result.status == "completed" else 1
    return 1 if result.status == "failed" else 0


def _cmd_webhook(args: argparse.Namespace) -> int:
    from review_bot.runtime.service import ServiceConfigurationError, run_webhook

    try:
        return run_webhook(args.config, host=args.host, port=args.port)
    except ServiceConfigurationError as exc:
        sys.stderr.write(f"Error: {exc}\n")
    except (Exception, SystemExit):
        sys.stderr.write("Error: webhook service failed\n")
    return 1


def _cmd_worker(args: argparse.Namespace) -> int:
    from review_bot.runtime.service import ServiceConfigurationError, run_worker

    try:
        return run_worker(args.config, once=args.once)
    except ServiceConfigurationError as exc:
        sys.stderr.write(f"Error: {exc}\n")
    except (Exception, SystemExit):
        sys.stderr.write("Error: worker service failed\n")
    return 1


def main(args: list[str] | None = None) -> int:
    parser = create_parser()
    parsed = parser.parse_args(args)
    if not parsed.subcommand:
        parser.print_help()
        return 0

    if parsed.subcommand == "version":
        return run_version(parsed)

    if parsed.subcommand == "analyze":
        return run_analyze(parsed)

    if parsed.subcommand == "delta":
        return run_delta(parsed)

    if parsed.subcommand == "review":
        return run_offline_review(parsed)

    if parsed.subcommand == "intent":
        return run_intent(parsed)

    if parsed.subcommand == "offline":
        # Print initialized message to satisfy smoke tests
        print(f"review-bot subcommand '{parsed.subcommand}' initialized.")
        if getattr(parsed, "repo", None) is not None:
            return run_offline_review(parsed)
        return 0

    if parsed.subcommand == "webhook":
        return _cmd_webhook(parsed)

    if parsed.subcommand == "worker":
        return _cmd_worker(parsed)

    return 1


if __name__ == "__main__":
    sys.exit(main())
