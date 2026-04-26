from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from ratchet.adapters import adapter_fingerprint, load_adapter
from ratchet.config import RatchetConfigError, RatchetRunConfig, ensure_search_path, resolve_run_config
from ratchet.errors import OptimizerModelError
from ratchet.io import file_sha256, load_eval_cases
from ratchet.optimizer import RatchetOptimizer
from ratchet.preflight import run_preflight_check
from ratchet.scaffold import SUPPORTED_TEMPLATES, init_scaffold


def load_runtime(config: RatchetRunConfig) -> tuple[Any, tuple[Any, ...]]:
    os.environ["RATCHET_ENV_FILE"] = config.env_file
    ensure_search_path(config)
    adapter = load_adapter(config.adapter)
    cases = load_eval_cases(config.evals)
    return adapter, cases


def run_optimizer(
    *,
    adapter_spec: str | None = None,
    evals_path: Path | str | None = None,
    out_dir: Path | str | None = None,
    env_file: str | None = ".env",
    dev_budget: int | None = 20,
    holdout_budget: int | None = 5,
    objective_mode: str | None = "correctness",
    allowed_models: list[str] | None = None,
    allowed_edits: list[str] | None = None,
    optimizer_model: str | None = "gpt-5.4",
    optimizer_reasoning: str | None = "medium",
    samples_per_case: int | None = 1,
    max_case_retries: int | None = 2,
    case_timeout_s: int | None = 180,
    fail_fast: bool | None = False,
    sanitize_examples: bool | None = None,
    config: RatchetRunConfig | None = None,
) -> Path:
    config = config or resolve_run_config(
        config_path=None,
        adapter=adapter_spec,
        evals_path=evals_path,
        out_dir=out_dir,
        env_file=env_file,
        dev_budget=dev_budget,
        holdout_budget=holdout_budget,
        objective_mode=objective_mode,
        allowed_models=allowed_models,
        allowed_edits=allowed_edits,
        optimizer_model=optimizer_model,
        optimizer_reasoning=optimizer_reasoning,
        samples_per_case=samples_per_case,
        max_case_retries=max_case_retries,
        case_timeout_s=case_timeout_s,
        fail_fast=fail_fast,
        sanitize_examples=sanitize_examples,
    )
    adapter, cases = load_runtime(config)
    optimizer = RatchetOptimizer(
        adapter=adapter,
        out_dir=config.out,
        env_path=config.env_file,
        dev_budget=config.dev_budget,
        holdout_budget=config.holdout_budget,
        objective=config.objective,
        optimizer_model=config.optimizer_model,
        optimizer_reasoning=config.optimizer_reasoning,
        samples_per_case=config.samples_per_case,
        max_case_retries=config.max_case_retries,
        case_timeout_s=config.case_timeout_s,
        fail_fast=config.fail_fast,
        run_metadata={
            **config.to_manifest_dict(),
            "adapter_fingerprint": adapter_fingerprint(config.adapter),
            "evals_sha256": file_sha256(config.evals),
        },
        progress_callback=_print_progress_event,
    )
    result = optimizer.run(cases)
    print(
        f"Selected patch: {result.selected_patch_hash} "
        f"({'promoted' if result.promoted else 'baseline kept'})"
    )
    print(f"Report: {config.out / 'report.md'}")
    return config.out


def _print_progress_event(row: dict[str, Any]) -> None:
    event = row.get("event")
    message: str | None = None
    if event == "run_started":
        message = (
            f"run started: dev={row.get('dev_cases')} holdout={row.get('holdout_cases')} "
            f"dev_budget={row.get('dev_budget')}"
        )
    elif event == "baseline_dev_started":
        message = f"baseline dev started: cases={row.get('case_count')}"
    elif event == "baseline_dev_completed":
        message = _format_score_message("baseline dev", row)
    elif event == "baseline_holdout_started":
        message = f"baseline holdout started: cases={row.get('case_count')}"
    elif event == "baseline_holdout_completed":
        message = _format_score_message("baseline holdout", row)
    elif event == "iteration_started":
        message = (
            f"iteration {row.get('iteration')}: frontier={row.get('frontier_width')} "
            f"dev_evals={row.get('dev_evaluations')}/{row.get('dev_budget')}"
        )
    elif event == "parent_started":
        message = (
            f"parent {row.get('parent_rank')}: patch={_short_hash(row.get('parent_patch_hash'))} "
            f"score={row.get('mean_score')} pass={row.get('pass_count')}/{row.get('case_count')}"
        )
    elif event == "diagnosis_started":
        message = (
            f"diagnosis started: parent={row.get('parent_rank')} "
            f"failures={row.get('failure_count')}"
        )
    elif event == "search_hypothesis_ready":
        families = row.get("active_families") or []
        message = (
            f"hypothesis ready: families={','.join(families[:5]) or 'none'} "
            f"contexts={row.get('active_context_count')}"
        )
    elif event == "proposal_started":
        families = row.get("active_families") or []
        message = (
            f"proposal started: budget={row.get('proposal_budget')} "
            f"families={','.join(families[:5]) or 'none'}"
        )
    elif event == "proposal_completed":
        message = (
            f"proposals: returned={row.get('returned_count')} valid={row.get('valid_count')} "
            f"invalid={row.get('invalid_count')} duplicates={row.get('duplicate_count')}"
        )
    elif event == "candidate_evaluation_started":
        message = (
            f"candidate started: family={row.get('transform_family')} "
            f"patch={_short_hash(row.get('patch_hash'))}"
        )
    elif event == "candidate_evaluated":
        status = "accepted" if row.get("accepted") else "rejected"
        reason = row.get("rejection_reason")
        suffix = f" ({reason})" if reason else ""
        message = f"candidate {status}: family={row.get('transform_family')}{suffix}"
    elif event == "retry_started":
        message = f"retrying parent {row.get('parent_rank')}: {row.get('reason')}"
    elif event == "frontier_updated":
        message = f"frontier updated: accepted={row.get('accepted_count')}"
    elif event == "holdout_candidate_started":
        message = f"holdout started: patch={_short_hash(row.get('patch_hash'))}"
    elif event == "holdout_candidate_completed":
        status = "passed" if row.get("passed_final_gate") else "rejected"
        message = f"holdout {status}: patch={_short_hash(row.get('patch_hash'))}"
    elif event == "holdout_validation_skipped":
        message = f"holdout skipped: {row.get('reason')}"
    elif event == "run_completed":
        status = "promoted" if row.get("promoted") else "baseline kept"
        message = f"run completed: {status}, selected={_short_hash(row.get('selected_patch_hash'))}"

    if message:
        print(f"[ratchet] {message}", file=sys.stderr, flush=True)


def _format_score_message(label: str, row: dict[str, Any]) -> str:
    return (
        f"{label} complete: score={row.get('mean_score')} "
        f"pass={row.get('pass_count')}/{row.get('case_count')} "
        f"cost=${float(row.get('mean_cost_usd') or 0):.4f}/case "
        f"latency={float(row.get('median_latency_s') or 0):.2f}s"
    )


def _short_hash(value: object) -> str:
    text = str(value or "")
    return text[:8] if text else "unknown"


def run_check(
    *,
    config: RatchetRunConfig,
    sample_limit: int = 2,
) -> dict[str, Any]:
    adapter, cases = load_runtime(config)
    summary = run_preflight_check(
        adapter_spec=config.adapter,
        adapter=adapter,
        cases=cases,
        objective=config.objective,
        sample_limit=sample_limit,
        optimizer_model=config.optimizer_model if os.environ.get("RATCHET_CHECK_OPTIMIZER_MODEL") == "1" else None,
        optimizer_env_path=config.env_file,
    )
    print("Ratchet check passed.")
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return summary.to_dict()


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _apply_run_overrides(
    args: argparse.Namespace,
    *,
    default_adapter: str | None = None,
    default_evals: Path | None = None,
    default_out: Path | None = None,
) -> RatchetRunConfig:
    return resolve_run_config(
        config_path=getattr(args, "config", None),
        adapter=getattr(args, "adapter", None) or default_adapter,
        evals_path=getattr(args, "evals", None) or default_evals,
        out_dir=getattr(args, "out", None) or default_out,
        env_file=getattr(args, "env_file", None),
        dev_budget=getattr(args, "dev_budget", None),
        holdout_budget=getattr(args, "holdout_budget", None),
        objective_mode=getattr(args, "mode", None),
        allowed_models=_split_csv(getattr(args, "allowed_models", None)),
        allowed_edits=_split_csv(getattr(args, "allowed_edits", None)),
        optimizer_model=getattr(args, "optimizer_model", None),
        optimizer_reasoning=getattr(args, "optimizer_reasoning", None),
        samples_per_case=getattr(args, "samples_per_case", None),
        max_case_retries=getattr(args, "max_case_retries", None),
        case_timeout_s=getattr(args, "case_timeout_s", None),
        fail_fast=True if getattr(args, "fail_fast", False) else None,
        sanitize_examples=True if getattr(args, "sanitize_examples", False) else None,
    )


def _resolve_check_config(args: argparse.Namespace) -> RatchetRunConfig:
    out_dir = getattr(args, "out", None) or (Path.cwd() / "results" / "check")
    return _apply_run_overrides(args, default_out=out_dir)


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to ratchet.toml")
    parser.add_argument("--adapter", help="Adapter import path, e.g. package.module:adapter")
    parser.add_argument("--evals", help="Path to evals JSONL")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--env-file", help="Path to .env with model provider API keys")
    parser.add_argument("--dev-budget", type=int, help="Max dev patch evaluations after baseline")
    parser.add_argument("--holdout-budget", type=int, help="Holdout finalist validation budget")
    parser.add_argument("--mode", choices=["correctness", "cost", "latency"], help="Primary optimization objective")
    parser.add_argument("--allowed-models", help="Comma-separated model allowlist for change_model patches")
    parser.add_argument("--allowed-edits", help="Comma-separated edit kinds to allow")
    parser.add_argument("--optimizer-model", help="Model used by Ratchet's diagnosis/proposal loop")
    parser.add_argument("--optimizer-reasoning", help="Reasoning effort for Ratchet's diagnosis/proposal loop")
    parser.add_argument("--samples-per-case", type=int, help="Number of repeated samples to evaluate per patch/case")
    parser.add_argument("--max-case-retries", type=int, help="Per-case retry budget after the first attempt")
    parser.add_argument("--case-timeout-s", type=int, help="Per-case timeout in seconds")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the run immediately after the first case that returns an error trace",
    )
    parser.add_argument(
        "--sanitize-examples",
        action="store_true",
        help="Redact raw dev example text before sending diagnostic examples to the optimizer model",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ratchet: eval-backed agent optimizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Generate a scaffolded Ratchet adapter for a Python agent.")
    init_parser.add_argument("--template", default="python_function", choices=sorted(SUPPORTED_TEMPLATES))
    init_parser.add_argument("--out", required=True, help="Directory to create the scaffold in")

    optimize_parser = subparsers.add_parser("optimize", help="Optimize an adapter-backed agent against evals.")
    add_run_arguments(optimize_parser)

    run_parser = subparsers.add_parser("run", help="Alias for optimize.")
    add_run_arguments(run_parser)

    check_parser = subparsers.add_parser("check", help="Validate adapter/eval/spec wiring before optimization.")
    add_run_arguments(check_parser)
    check_parser.add_argument("--sample-limit", type=int, default=2, help="How many cases to probe during preflight")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            root = init_scaffold(out_dir=args.out, template=args.template)
            print(f"Scaffold created at {root}")
            return 0

        if args.command in {"optimize", "run"}:
            config = _apply_run_overrides(args)
            run_optimizer(config=config)
            return 0

        if args.command == "check":
            config = _resolve_check_config(args)
            run_check(config=config, sample_limit=args.sample_limit)
            return 0

        raise ValueError(f"Unsupported command: {args.command}")
    except RatchetConfigError as exc:
        print(f"Ratchet config error: {exc}", file=sys.stderr)
        return 2
    except OptimizerModelError as exc:
        print(f"Ratchet optimizer model error: {exc}", file=sys.stderr)
        return 4
    except (TypeError, ValueError) as exc:
        print(f"Ratchet preflight error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
