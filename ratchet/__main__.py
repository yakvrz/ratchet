from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from ratchet.adapters import load_adapter
from ratchet.config import RatchetRunConfig, ensure_search_path, resolve_run_config
from ratchet.io import file_sha256, load_eval_cases
from ratchet.optimizer import RatchetOptimizer
from ratchet.preflight import run_preflight_check, validate_search_space
from ratchet.scaffold import SUPPORTED_TEMPLATES, init_scaffold


def load_runtime(config: RatchetRunConfig) -> tuple[Any, tuple[Any, ...], Any]:
    os.environ["RATCHET_ENV_FILE"] = config.env_file
    ensure_search_path(config)
    adapter = load_adapter(config.adapter)
    cases = load_eval_cases(config.evals)
    search_space = adapter.search_space()
    validate_search_space(search_space)
    return adapter, cases, search_space


def run_optimizer(
    *,
    adapter_spec: str | None = None,
    evals_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    env_file: str | None = ".env",
    dev_budget: int | None = 20,
    holdout_top_k: int | None = 5,
    harnesser_model: str | None = "gpt-5.4",
    harnesser_reasoning: str | None = "medium",
    harnesser_enabled: bool | None = True,
    max_case_retries: int | None = 2,
    case_timeout_s: int | None = 180,
    fail_fast: bool | None = False,
    config: RatchetRunConfig | None = None,
) -> Path:
    config = config or resolve_run_config(
        config_path=None,
        adapter=adapter_spec,
        evals_path=evals_path,
        out_dir=out_dir,
        env_file=env_file,
        dev_budget=dev_budget,
        holdout_top_k=holdout_top_k,
        harnesser_model=harnesser_model,
        harnesser_reasoning=harnesser_reasoning,
        harnesser_enabled=harnesser_enabled,
        max_case_retries=max_case_retries,
        case_timeout_s=case_timeout_s,
        fail_fast=fail_fast,
    )
    adapter, cases, search_space = load_runtime(config)
    optimizer = RatchetOptimizer(
        adapter=adapter,
        search_space=search_space,
        out_dir=config.out,
        env_path=config.env_file,
        dev_budget=config.dev_budget,
        holdout_top_k=config.holdout_top_k,
        harnesser_model=config.harnesser_model,
        harnesser_reasoning=config.harnesser_reasoning,
        harnesser_enabled=config.harnesser_enabled,
        max_case_retries=config.max_case_retries,
        case_timeout_s=config.case_timeout_s,
        fail_fast=config.fail_fast,
        run_metadata={
            **config.to_manifest_dict(),
            "evals_sha256": file_sha256(config.evals),
        },
    )
    result = optimizer.run(cases)
    print(
        f"Selected candidate: {result.selected_candidate_hash} "
        f"({'promoted' if result.promoted else 'baseline kept'})"
    )
    print(f"Report: {config.out / 'report.md'}")
    return config.out


def run_check(
    *,
    config: RatchetRunConfig,
    sample_limit: int = 2,
) -> dict[str, Any]:
    adapter, cases, search_space = load_runtime(config)
    summary = run_preflight_check(
        adapter_spec=config.adapter,
        adapter=adapter,
        search_space=search_space,
        cases=cases,
        sample_limit=sample_limit,
    )
    print("Ratchet check passed.")
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return summary.to_dict()


def paired_demo_defaults() -> tuple[str, Path, Path]:
    root = Path(__file__).resolve().parents[1]
    adapter_spec = "examples.northstar.adapter:adapter"
    evals_path = root / "examples" / "northstar" / "evals.jsonl"
    out_dir = root / "results" / "paired-demo"
    return adapter_spec, evals_path, out_dir


def _apply_run_overrides(args: argparse.Namespace, *, default_adapter: str | None = None, default_evals: Path | None = None, default_out: Path | None = None) -> RatchetRunConfig:
    harnesser_enabled = None
    if getattr(args, "disable_harnesser", False):
        harnesser_enabled = False
    return resolve_run_config(
        config_path=getattr(args, "config", None),
        adapter=getattr(args, "adapter", None) or default_adapter,
        evals_path=getattr(args, "evals", None) or default_evals,
        out_dir=getattr(args, "out", None) or default_out,
        env_file=getattr(args, "env_file", None),
        dev_budget=getattr(args, "dev_budget", None),
        holdout_top_k=getattr(args, "holdout_top_k", None),
        harnesser_model=getattr(args, "harnesser_model", None),
        harnesser_reasoning=getattr(args, "harnesser_reasoning", None),
        harnesser_enabled=harnesser_enabled,
        max_case_retries=getattr(args, "max_case_retries", None),
        case_timeout_s=getattr(args, "case_timeout_s", None),
        fail_fast=True if getattr(args, "fail_fast", False) else None,
    )


def _resolve_check_config(args: argparse.Namespace) -> RatchetRunConfig:
    out_dir = getattr(args, "out", None) or (Path.cwd() / "results" / "check")
    return _apply_run_overrides(args, default_out=out_dir)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to ratchet.toml")
    parser.add_argument("--adapter", help="Adapter import path, e.g. package.module:adapter")
    parser.add_argument("--evals", help="Path to evals JSONL")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--env-file", help="Path to .env with OPENAI_API_KEY")
    parser.add_argument("--dev-budget", type=int, help="Max dev candidate evaluations after baseline")
    parser.add_argument("--holdout-top-k", type=int, help="Holdout validation budget")
    parser.add_argument("--harnesser-model", help="Model used by the diagnoser/proposer loop")
    parser.add_argument("--harnesser-reasoning", help="Reasoning effort for the diagnoser/proposer loop")
    parser.add_argument(
        "--disable-harnesser",
        action="store_true",
        help="Disable the LLM diagnoser/proposer and use heuristic proposals only",
    )
    parser.add_argument("--max-case-retries", type=int, help="Per-case retry budget after the first attempt")
    parser.add_argument("--case-timeout-s", type=int, help="Per-case timeout in seconds")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the run immediately after the first case that returns an error trace",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ratchet: attachable agent optimizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Generate a scaffolded Ratchet adapter for a Python agent.")
    init_parser.add_argument("--template", default="python_function", choices=sorted(SUPPORTED_TEMPLATES))
    init_parser.add_argument("--out", required=True, help="Directory to create the scaffold in")

    run_parser = subparsers.add_parser("run", help="Optimize an adapter-backed harness against evals.")
    add_run_arguments(run_parser)

    check_parser = subparsers.add_parser("check", help="Validate adapter/eval wiring before a full optimization run.")
    add_run_arguments(check_parser)
    check_parser.add_argument("--sample-limit", type=int, default=2, help="How many cases to probe during preflight")

    paired_parser = subparsers.add_parser("paired-demo", help="Run the built-in Northstar example.")
    add_run_arguments(paired_parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "init":
        root = init_scaffold(out_dir=args.out, template=args.template)
        print(f"Scaffold created at {root}")
        return

    if args.command == "run":
        config = _apply_run_overrides(args)
        run_optimizer(config=config)
        return

    if args.command == "check":
        config = _resolve_check_config(args)
        run_check(config=config, sample_limit=args.sample_limit)
        return

    if args.command == "paired-demo":
        adapter_spec, evals_path, out_dir = paired_demo_defaults()
        config = _apply_run_overrides(
            args,
            default_adapter=adapter_spec,
            default_evals=evals_path,
            default_out=out_dir,
        )
        run_optimizer(config=config)
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
