"""Command line entry point. ``raglab run`` executes the configured grid."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

from raglab import __version__
from raglab.config import ConfigError, load_config
from raglab.llm import LLMNotConfigured


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raglab",
        description="Compare RAG chunking and retrieval strategies on a labeled dataset.",
    )
    parser.add_argument("--version", action="version", version=f"raglab {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the chunker by retriever grid and write results")
    run.add_argument("--config", default="configs/scifact.yaml", help="YAML config path")
    run.add_argument("--max-queries", type=int, help="Evaluate a seeded subset of labeled queries")
    run.add_argument("--output", help="Directory for CSV, JSON, markdown, and the chart")
    run.add_argument("--check", help="Fail if any metric in this JSON file is below its minimum")
    run.add_argument("--retrievers", help="Comma-separated retrievers, replacing the config list")
    run.add_argument("--chunkers", help="Comma-separated chunkers, replacing the config list")
    run.add_argument("--generation", choices=["auto", "on", "off"])
    run.set_defaults(func=_cmd_run)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, LLMNotConfigured) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        logging.getLogger("raglab").exception("raglab failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _cmd_run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    config = load_config(args.config)
    if args.max_queries is not None:
        if args.max_queries < 1:
            raise ConfigError("--max-queries must be >= 1")
        config = replace(config, max_queries=args.max_queries)
    if args.output:
        config = replace(config, output_dir=_cli_path(args.output))
    if args.check:
        check_path = _cli_path(args.check)
        if not check_path.exists():
            raise ConfigError(f"Threshold file not found: {check_path}")
        config = replace(config, check_path=check_path)
    if args.retrievers:
        names = [name.strip() for name in args.retrievers.split(",") if name.strip()]
        if not names:
            raise ConfigError("--retrievers did not include any names")
        config = replace(config, retrievers=names, force_retrievers=True)
    if args.chunkers:
        allowed = {spec.name: spec for spec in config.chunkers}
        chosen = []
        for name in [item.strip() for item in args.chunkers.split(",") if item.strip()]:
            if name not in allowed:
                known = ", ".join(spec.name for spec in config.chunkers)
                raise ConfigError(f"Unknown chunker '{name}' for this config. Known: {known}.")
            chosen.append(allowed[name])
        if not chosen:
            raise ConfigError("--chunkers did not include any names")
        config = replace(config, chunkers=chosen)
    if args.generation:
        generation = replace(config.generation, mode=args.generation)
        config = replace(config, generation=generation)
    from raglab.report import render_markdown
    from raglab.runner import run_benchmark

    result = run_benchmark(config)
    print(render_markdown(result))
    if result["threshold_failures"]:
        print("Regression gate failed:")
        for failure in result["threshold_failures"]:
            print(f"- {failure}")
        return 1
    return 0


def _cli_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path
