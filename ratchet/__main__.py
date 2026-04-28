from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from ratchet.adapters import adapter_fingerprint, load_adapter
from ratchet.config import RatchetConfigError, RatchetRunConfig, ensure_search_path, resolve_run_config
from ratchet.eval_health import EvalHealthReport, render_eval_health_markdown, run_eval_health_check
from ratchet.errors import OptimizerModelError
from ratchet.io import file_sha256, load_eval_cases
from ratchet.optimizer import RatchetOptimizer
from ratchet.partial import write_partial_run_outputs
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
    case_concurrency: int | None = 1,
    stage_case_concurrency: int | None = None,
    max_case_retries: int | None = 2,
    case_timeout_s: int | None = 180,
    fail_fast: bool | None = False,
    sanitize_examples: bool | None = None,
    expensive_candidate_cost_ratio: float | None = None,
    max_expensive_full_dev_candidates: int | None = None,
    max_expensive_holdout_candidates: int | None = None,
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
        case_concurrency=case_concurrency,
        stage_case_concurrency=stage_case_concurrency,
        max_case_retries=max_case_retries,
        case_timeout_s=case_timeout_s,
        fail_fast=fail_fast,
        sanitize_examples=sanitize_examples,
        expensive_candidate_cost_ratio=expensive_candidate_cost_ratio,
        max_expensive_full_dev_candidates=max_expensive_full_dev_candidates,
        max_expensive_holdout_candidates=max_expensive_holdout_candidates,
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
        case_concurrency=config.case_concurrency,
        stage_case_concurrency=config.stage_case_concurrency,
        max_case_retries=config.max_case_retries,
        case_timeout_s=config.case_timeout_s,
        fail_fast=config.fail_fast,
        expensive_candidate_cost_ratio=config.expensive_candidate_cost_ratio,
        max_expensive_full_dev_candidates=config.max_expensive_full_dev_candidates,
        max_expensive_holdout_candidates=config.max_expensive_holdout_candidates,
        run_metadata={
            **config.to_manifest_dict(),
            "adapter_fingerprint": adapter_fingerprint(config.adapter),
            "evals_sha256": file_sha256(config.evals),
        },
        progress_callback=CliProgressPrinter(),
    )
    try:
        result = optimizer.run(cases)
    except KeyboardInterrupt:
        write_partial_run_outputs(
            config.out,
            status="interrupted",
            reason="KeyboardInterrupt received before run completion.",
        )
        print(f"Partial report: {config.out / 'partial_report.md'}", file=sys.stderr)
        raise
    except Exception as exc:
        write_partial_run_outputs(
            config.out,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        print(f"Partial report: {config.out / 'partial_report.md'}", file=sys.stderr)
        raise
    print(
        f"Selected patch: {result.selected_patch_hash} "
        f"({'promoted' if result.promoted else 'baseline kept'})"
    )
    print(f"Report: {config.out / 'report.md'}")
    return config.out


class CliProgressPrinter:
    def __call__(self, row: dict[str, Any]) -> None:
        rendered = self.format(row)
        if rendered:
            print(rendered, file=sys.stderr, flush=True)

    def format(self, row: dict[str, Any]) -> str | None:
        event = str(row.get("event") or "")
        phase, message = _progress_message(event, row)
        if message is None:
            return None
        return f"[ratchet {_format_elapsed(row.get('elapsed_s'))}] {phase:<10} {message}"


def _print_progress_event(row: dict[str, Any]) -> None:
    CliProgressPrinter()(row)


def _progress_message(event: str, row: dict[str, Any]) -> tuple[str, str | None]:
    if event == "run_started":
        return (
            "RUN",
            "started "
            f"objective={row.get('objective')} train={row.get('train_cases')} dev={row.get('dev_cases')} "
            f"holdout={row.get('holdout_cases')} dev_budget={row.get('dev_budget')} "
            f"holdout_budget={row.get('holdout_budget')} concurrency={row.get('case_concurrency')}"
            f"/{row.get('stage_case_concurrency', row.get('case_concurrency'))} "
            f"examples={row.get('proposal_example_count')}",
        )
    if event == "baseline_dev_started":
        return "BASELINE", f"dev evaluation started cases={row.get('case_count')}"
    if event == "baseline_dev_completed":
        return "BASELINE", "dev complete " + _score_summary(row)
    if event == "baseline_holdout_started":
        return "BASELINE", f"holdout evaluation started cases={row.get('case_count')}"
    if event == "baseline_holdout_completed":
        return "BASELINE", "holdout complete " + _score_summary(row)
    if event == "iteration_started":
        return (
            "SEARCH",
            f"iteration={row.get('iteration')} frontier={row.get('frontier_width')} "
            f"dev_evals={row.get('dev_evaluations')}/{row.get('dev_budget')}",
        )
    if event == "parent_started":
        return (
            "PARENT",
            f"rank={row.get('parent_rank')} patch={_short_hash(row.get('parent_patch_hash'))} "
            + _score_summary(row),
        )
    if event == "diagnosis_started":
        return "DIAGNOSE", f"started parent={row.get('parent_rank')} failures={row.get('failure_count')}"
    if event == "diagnosis_completed":
        diagnostics = row.get("call_diagnostics") or {}
        cached = " cached" if row.get("cached") else ""
        return (
            "DIAGNOSE",
            " ".join(
                part
                for part in (
                    f"complete diagnoses={row.get('diagnosis_count')}{cached}",
                    _call_summary(diagnostics),
                    _short_reason(row.get("analysis")),
                )
                if part
            ),
        )
    if event == "task_theory_ready":
        modes = _join_limited(row.get("residual_failure_modes") or [], limit=3)
        cached = " cached" if row.get("cached") else ""
        return (
            "THEORY",
            f"bottleneck={row.get('bottleneck_class')} confidence={row.get('confidence')}{cached} "
            f"residual={modes or 'none'}",
        )
    if event == "search_hypothesis_ready":
        families = _join_limited(row.get("active_families") or [], limit=5)
        return "HYPOTHESIS", f"active={families or 'none'} contexts={row.get('active_context_count')}"
    if event == "research_controller_started":
        return (
            "RESEARCH",
            f"started stage={row.get('stage')} candidates={row.get('candidate_count')} "
            f"max_select={row.get('max_select')}",
        )
    if event == "research_controller_completed":
        diagnostics = row.get("call_diagnostics") or {}
        selected = _join_limited([_short_hash(item) for item in row.get("selected_candidate_ids") or []], limit=4)
        return (
            "RESEARCH",
            " ".join(
                part
                for part in (
                    f"complete stage={row.get('stage')} selected={selected or 'none'}",
                    _call_summary(diagnostics),
                    _short_reason(row.get("rationale")),
                )
                if part
            ),
        )
    if event == "proposal_started":
        families = _join_limited(row.get("active_families") or [], limit=5)
        retry = " retry" if row.get("proposal_retry") else ""
        return (
            "PROPOSE",
            f"started{retry} parent={row.get('parent_rank')} budget={row.get('proposal_budget')} "
            f"families={families or 'none'}",
        )
    if event == "proposal_completed":
        diagnostics = row.get("call_diagnostics") or {}
        return (
            "PROPOSE",
            " ".join(
                part
                for part in (
                    f"complete returned={row.get('returned_count')} valid={row.get('valid_count')} "
                    f"invalid={row.get('invalid_count')} duplicates={row.get('duplicate_count')}",
                    _call_summary(diagnostics),
                )
                if part
            ),
        )
    if event == "candidate_evaluation_started":
        return (
            "CANDIDATE",
            f"queued family={row.get('transform_family')} patch={_short_hash(row.get('patch_hash'))}",
        )
    if event == "candidate_stage_started":
        return (
            "STAGE",
            f"{_stage_name(row.get('stage'))} started candidates={row.get('candidate_count')} "
            f"cases={row.get('case_count')}",
        )
    if event == "candidate_stage_completed":
        return (
            "STAGE",
            f"{_stage_name(row.get('stage'))} complete advanced={row.get('advanced_count')} "
            f"accepted={row.get('accepted_count')} rejected={row.get('rejected_count')} "
            f"screened={row.get('screened_count')}",
        )
    if event == "candidate_evaluated":
        status = row.get("frontier_status") or ("accepted" if row.get("accepted") else "rejected")
        reason = row.get("rejection_reason") or row.get("constraint_warning")
        return (
            "CANDIDATE",
            f"{status} family={row.get('transform_family')} patch={_short_hash(row.get('patch_hash'))} "
            f"score_delta={_format_signed(row.get('score_delta'), digits=3)} "
            f"cost_delta={_format_money_delta(row.get('cost_delta'))} "
            f"latency_delta={_format_seconds_delta(row.get('latency_delta'))} "
            f"stages={row.get('stage_count')} full_dev={_format_bool(row.get('full_dev_evaluated'))}"
            + (f" reason={_short_reason(reason)}" if reason else ""),
        )
    if event == "retry_started":
        return "RETRY", f"parent={row.get('parent_rank')} reason={_short_reason(row.get('reason'))}"
    if event == "frontier_updated":
        patches = _join_limited([_short_hash(item) for item in row.get("frontier_patch_hashes") or []], limit=4)
        return "FRONTIER", f"accepted={row.get('accepted_count')} selectable={row.get('selectable_parent_count')} patches={patches}"
    if event == "search_stopped":
        return "SEARCH", f"stopped reason={_short_reason(row.get('reason'))}"
    if event == "simplification_started":
        return (
            "SIMPLIFY",
            f"started parent={_short_hash(row.get('parent_patch_hash'))} variant={_short_hash(row.get('patch_hash'))}",
        )
    if event == "simplification_completed":
        status = "accepted" if row.get("accepted") else "rejected"
        reason = row.get("rejection_reason")
        return (
            "SIMPLIFY",
            f"{status} variant={_short_hash(row.get('variant_patch_hash'))} {_score_summary(row)}"
            + (f" reason={_short_reason(reason)}" if reason else ""),
        )
    if event == "confirmation_started":
        return (
            "CONFIRM",
            f"started patch={_short_hash(row.get('patch_hash'))} cases={row.get('case_count')} "
            f"samples={row.get('sample_count')}",
        )
    if event == "confirmation_completed":
        status = "passed" if row.get("passed") else "failed"
        return "CONFIRM", f"{status} patch={_short_hash(row.get('patch_hash'))} reason={_short_reason(row.get('reason'))}"
    if event == "confirmation_skipped":
        return "CONFIRM", f"skipped patch={_short_hash(row.get('patch_hash'))} reason={_short_reason(row.get('reason'))}"
    if event == "holdout_candidate_started":
        return "HOLDOUT", f"started patch={_short_hash(row.get('patch_hash'))} cases={row.get('case_count')}"
    if event == "holdout_candidate_completed":
        status = row.get("finalist_status") or ("validated" if row.get("passed_final_gate") else "rejected")
        reason = row.get("rejection_reason")
        return (
            "HOLDOUT",
            f"{status} patch={_short_hash(row.get('patch_hash'))} {_score_summary(row)}"
            + (f" reason={_short_reason(reason)}" if reason else ""),
        )
    if event == "holdout_validation_skipped":
        patch = row.get("patch_hash")
        patch_text = f" patch={_short_hash(patch)}" if patch else ""
        return "HOLDOUT", f"skipped{patch_text} reason={_short_reason(row.get('reason'))}"
    if event == "case_batch_started":
        return (
            "BATCH",
            f"started split={row.get('split')} patch={_short_hash(row.get('patch_hash'))} "
            f"fresh={row.get('fresh_count')} cases={row.get('case_count')} samples={row.get('sample_count')} "
            f"concurrency={row.get('concurrency')}",
        )
    if event == "case_batch_completed":
        return (
            "BATCH",
            f"complete split={row.get('split')} patch={_short_hash(row.get('patch_hash'))} "
            f"fresh={row.get('fresh_count')} concurrency={row.get('concurrency')}",
        )
    if event == "run_completed":
        status = "promoted" if row.get("promoted") else "baseline kept"
        return (
            "DONE",
            f"{status} selected={_short_hash(row.get('selected_patch_hash'))} "
            f"accepted_dev={row.get('accepted_dev_patches')} holdout_validations={row.get('holdout_validations')} "
            f"reason={_short_reason(row.get('selection_reason'))}",
        )
    return event.upper()[:10] or "EVENT", None


def _format_elapsed(value: object) -> str:
    seconds = int(float(value or 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _score_summary(row: dict[str, Any]) -> str:
    return (
        f"score={_format_number(row.get('mean_score'), digits=3)} "
        f"pass={_format_count(row.get('pass_count'))}/{_format_count(row.get('case_count'))} "
        f"cost={_format_money(row.get('mean_cost_usd'))}/case "
        f"latency={_format_seconds(row.get('median_latency_s'))}"
    )


def _call_summary(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return ""
    parts: list[str] = []
    model = diagnostics.get("model")
    if model:
        parts.append(f"model={model}")
    input_tokens = diagnostics.get("input_tokens")
    output_tokens = diagnostics.get("output_tokens")
    total_tokens = diagnostics.get("total_tokens")
    if input_tokens is not None or output_tokens is not None:
        parts.append(f"tokens={_format_count(input_tokens)}/{_format_count(output_tokens)}")
    elif total_tokens is not None:
        parts.append(f"tokens={_format_count(total_tokens)}")
    prompt_tokens = diagnostics.get("prompt_approx_tokens")
    if prompt_tokens is not None:
        parts.append(f"prompt~={_format_count(prompt_tokens)}tok")
    prompt_chars = diagnostics.get("prompt_chars")
    if prompt_chars is not None:
        parts.append(f"prompt={_format_count(prompt_chars)}ch")
    elapsed = diagnostics.get("elapsed_s")
    if elapsed is not None:
        parts.append(f"call={_format_seconds(elapsed)}")
    finish_reason = diagnostics.get("finish_reason")
    if finish_reason:
        parts.append(f"finish={finish_reason}")
    return " ".join(parts)


def _format_number(value: object, *, digits: int) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_signed(value: object, *, digits: int) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_count(value: object) -> str:
    if value is None:
        return "?"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _format_bool(value: object) -> str:
    return "yes" if bool(value) else "no"


def _format_money(value: object) -> str:
    if value is None:
        return "$?"
    try:
        return f"${float(value):.4f}"
    except (TypeError, ValueError):
        return f"${value}"


def _format_money_delta(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "+" if amount >= 0 else "-"
    return f"{sign}${abs(amount):.4f}"


def _format_seconds(value: object) -> str:
    if value is None:
        return "?s"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return f"{value}s"


def _format_seconds_delta(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.2f}s"
    except (TypeError, ValueError):
        return str(value)


def _stage_name(value: object) -> str:
    text = str(value or "stage")
    return text.replace("_", "-")


def _join_limited(values: list[Any] | tuple[Any, ...], *, limit: int) -> str:
    items = [str(item) for item in values if item is not None and str(item)]
    if not items:
        return ""
    shown = items[:limit]
    if len(items) > limit:
        shown.append(f"+{len(items) - limit}")
    return ",".join(shown)


def _short_reason(value: object, *, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


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


def run_eval_health(
    *,
    config: RatchetRunConfig,
    sample_limit: int | None = None,
    repeats: int | None = None,
    strict: bool = False,
) -> EvalHealthReport:
    adapter, cases = load_runtime(config)
    report = run_eval_health_check(
        adapter_spec=config.adapter,
        adapter=adapter,
        cases=cases,
        config=config.eval_health,
        sample_limit=sample_limit,
        repeats=repeats,
        case_timeout_s=config.case_timeout_s,
        evaluation_samples_per_case=config.samples_per_case,
        case_concurrency=config.case_concurrency,
    )
    out_dir = config.out / "eval_health"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_health.json").write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    (out_dir / "eval_health.md").write_text(render_eval_health_markdown(report))
    _print_eval_health_summary(report, out_dir=out_dir, strict=strict)
    return report


def _print_eval_health_summary(report: EvalHealthReport, *, out_dir: Path, strict: bool) -> None:
    issue_counts: dict[str, int] = {"fatal": 0, "warning": 0, "info": 0}
    for issue in report.issues:
        issue_counts[issue.severity] = issue_counts.get(issue.severity, 0) + 1
    print(
        "Ratchet eval health: "
        f"{report.status} "
        f"(fatal={issue_counts['fatal']} warning={issue_counts['warning']} info={issue_counts['info']})"
    )
    split_counts = report.split_summary.get("by_split", {})
    print(
        "Splits: "
        f"train={split_counts.get('train', 0)} "
        f"dev={split_counts.get('dev', 0)} "
        f"holdout={split_counts.get('holdout', 0)}"
    )
    probe = report.baseline_probe
    if probe.get("checked"):
        runtime = ((probe.get("runtime_feasibility") or {}).get("estimated_eval_sweep") or {})
        print(
            "Baseline probe: "
            f"cases={len(probe.get('sampled_case_ids') or [])} "
            f"repeats={probe.get('repeats')} "
            f"pass_rate={float(probe.get('pass_rate') or 0.0):.3f} "
            f"errors={int(probe.get('error_attempt_count') or 0)}/{int(probe.get('attempt_count') or 0)} "
            f"unstable={int(probe.get('unstable_case_count') or 0)}"
        )
        if runtime:
            print(
                "Estimated eval sweep: "
                f"attempts={runtime.get('case_attempts')} "
                f"wall={float(runtime.get('wall_time_s') or 0.0):.1f}s "
                f"cost=${float(runtime.get('cost_usd') or 0.0):.6f} "
                f"tokens={int(runtime.get('total_tokens') or 0)}"
            )
    else:
        print(f"Baseline probe: skipped ({probe.get('reason')})")
    for issue in report.issues[:10]:
        print(f"- {issue.severity.upper()} {issue.code}: {issue.message}")
    if len(report.issues) > 10:
        print(f"- ... {len(report.issues) - 10} more issue(s) in eval_health.json")
    if strict and report.warning and not report.fatal:
        print("Strict mode: warnings make this check fail.")
    print(f"Report: {out_dir / 'eval_health.md'}")
    print(f"JSON: {out_dir / 'eval_health.json'}")


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
        case_concurrency=getattr(args, "case_concurrency", None),
        stage_case_concurrency=getattr(args, "stage_case_concurrency", None),
        max_case_retries=getattr(args, "max_case_retries", None),
        case_timeout_s=getattr(args, "case_timeout_s", None),
        fail_fast=True if getattr(args, "fail_fast", False) else None,
        sanitize_examples=True if getattr(args, "sanitize_examples", False) else None,
        expensive_candidate_cost_ratio=getattr(args, "expensive_candidate_cost_ratio", None),
        max_expensive_full_dev_candidates=getattr(args, "max_expensive_full_dev_candidates", None),
        max_expensive_holdout_candidates=getattr(args, "max_expensive_holdout_candidates", None),
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
    parser.add_argument("--case-concurrency", type=int, help="Maximum concurrent case evaluations per patch")
    parser.add_argument(
        "--stage-case-concurrency",
        type=int,
        help="Maximum concurrent case evaluations across a multi-patch stage; defaults to --case-concurrency.",
    )
    parser.add_argument("--max-case-retries", type=int, help="Per-case retry budget after the first attempt")
    parser.add_argument("--case-timeout-s", type=int, help="Per-case timeout in seconds")
    parser.add_argument(
        "--expensive-candidate-cost-ratio",
        type=float,
        help="Treat candidates above this cost ratio as expensive for evaluation-budget caps.",
    )
    parser.add_argument(
        "--max-expensive-full-dev-candidates",
        type=int,
        help="Maximum expensive candidates to evaluate on full dev; omit for no cap.",
    )
    parser.add_argument(
        "--max-expensive-holdout-candidates",
        type=int,
        help="Maximum expensive finalists to validate on holdout; omit for no cap.",
    )
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


def add_eval_health_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to ratchet.toml")
    parser.add_argument("--adapter", help="Adapter import path, e.g. package.module:adapter")
    parser.add_argument("--evals", help="Path to evals JSONL")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--env-file", help="Path to .env with model provider API keys")
    parser.add_argument("--sample-limit", type=int, help="Maximum dev/holdout cases to probe with the baseline")
    parser.add_argument("--repeats", type=int, help="Repeated baseline probes per sampled case; use 0 to skip probes")
    parser.add_argument("--case-timeout-s", type=int, help="Per-case timeout in seconds for baseline probes")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when eval health reports warnings as well as fatal issues",
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

    eval_health_parser = subparsers.add_parser("eval-health", help="Check eval-set and grader health before optimization.")
    add_eval_health_arguments(eval_health_parser)

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

        if args.command == "eval-health":
            config = _apply_run_overrides(args)
            report = run_eval_health(
                config=config,
                sample_limit=getattr(args, "sample_limit", None),
                repeats=getattr(args, "repeats", None),
                strict=bool(getattr(args, "strict", False)),
            )
            if report.fatal or (bool(getattr(args, "strict", False)) and report.warning):
                return 5
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
