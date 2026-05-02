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
from ratchet.ideation_benchmark import write_ideation_assessment
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
    optimizer_model: str | None = "gpt-5.4",
    optimizer_reasoning: str | None = "medium",
    search_planner_model: str | None = None,
    search_planner_reasoning: str | None = None,
    candidate_implementer_model: str | None = None,
    candidate_implementer_reasoning: str | None = None,
    samples_per_case: int | None = 1,
    case_concurrency: int | None = 1,
    stage_case_concurrency: int | None = None,
    max_case_retries: int | None = 2,
    case_timeout_s: int | None = 180,
    fail_fast: bool | None = False,
    sanitize_examples: bool | None = None,
    expensive_candidate_cost_ratio: float | None = None,
    max_dev_measurement_cost_usd: float | None = None,
    max_holdout_measurement_cost_usd: float | None = None,
    max_dev_measurement_tool_calls: int | None = None,
    max_holdout_measurement_tool_calls: int | None = None,
    max_dev_measurement_turns: int | None = None,
    max_holdout_measurement_turns: int | None = None,
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
        optimizer_model=optimizer_model,
        optimizer_reasoning=optimizer_reasoning,
        search_planner_model=search_planner_model,
        search_planner_reasoning=search_planner_reasoning,
        candidate_implementer_model=candidate_implementer_model,
        candidate_implementer_reasoning=candidate_implementer_reasoning,
        samples_per_case=samples_per_case,
        case_concurrency=case_concurrency,
        stage_case_concurrency=stage_case_concurrency,
        max_case_retries=max_case_retries,
        case_timeout_s=case_timeout_s,
        fail_fast=fail_fast,
        sanitize_examples=sanitize_examples,
        expensive_candidate_cost_ratio=expensive_candidate_cost_ratio,
        max_dev_measurement_cost_usd=max_dev_measurement_cost_usd,
        max_holdout_measurement_cost_usd=max_holdout_measurement_cost_usd,
        max_dev_measurement_tool_calls=max_dev_measurement_tool_calls,
        max_holdout_measurement_tool_calls=max_holdout_measurement_tool_calls,
        max_dev_measurement_turns=max_dev_measurement_turns,
        max_holdout_measurement_turns=max_holdout_measurement_turns,
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
        search_planner_model=config.search_planner_model,
        search_planner_reasoning=config.search_planner_reasoning,
        candidate_implementer_model=config.candidate_implementer_model,
        candidate_implementer_reasoning=config.candidate_implementer_reasoning,
        samples_per_case=config.samples_per_case,
        case_concurrency=config.case_concurrency,
        stage_case_concurrency=config.stage_case_concurrency,
        max_case_retries=config.max_case_retries,
        case_timeout_s=config.case_timeout_s,
        fail_fast=config.fail_fast,
        expensive_candidate_cost_ratio=config.expensive_candidate_cost_ratio,
        max_dev_measurement_cost_usd=config.max_dev_measurement_cost_usd,
        max_holdout_measurement_cost_usd=config.max_holdout_measurement_cost_usd,
        max_dev_measurement_tool_calls=config.max_dev_measurement_tool_calls,
        max_holdout_measurement_tool_calls=config.max_holdout_measurement_tool_calls,
        max_dev_measurement_turns=config.max_dev_measurement_turns,
        max_holdout_measurement_turns=config.max_holdout_measurement_turns,
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
    outcome = "promoted optimized candidate" if result.promoted else "kept baseline"
    print(f"Ratchet finished: {outcome}; selected candidate {result.selected_candidate_id}")
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
        return f"[{_format_elapsed(row.get('elapsed_s'))}] {phase:<9} {message}"


def _print_progress_event(row: dict[str, Any]) -> None:
    CliProgressPrinter()(row)


def _progress_message(event: str, row: dict[str, Any]) -> tuple[str, str | None]:
    if event == "run_started":
        total_cases = row.get("total_cases")
        if total_cases is None:
            total_cases = sum(int(row.get(key) or 0) for key in ("train_cases", "dev_cases", "holdout_cases"))
        return (
            "Run",
            f"{total_cases} cases for {row.get('objective')} "
            f"(train {row.get('train_cases')}, dev {row.get('dev_cases')}, holdout {row.get('holdout_cases')}); "
            f"candidate budget dev {row.get('dev_budget')}, holdout {row.get('holdout_budget')}; "
            f"examples {row.get('proposal_example_count')}",
        )
    if event == "baseline_dev_started":
        return "Observe", f"measuring baseline on {row.get('case_count')} dev cases"
    if event == "baseline_dev_completed":
        return "Observe", f"baseline {_score_brief(row)}; {_failure_brief(row)}; {_slice_brief(row)}"
    if event == "baseline_holdout_started":
        return "Holdout", f"measuring protected baseline reference on {row.get('case_count')} cases"
    if event == "baseline_holdout_completed":
        return "Holdout", "baseline reference " + _score_brief(row)
    if event == "evidence_packet_ready":
        weak = _join_limited(row.get("weak_slices") or [], limit=5)
        modes = _join_limited(row.get("residual_failure_modes") or [], limit=5)
        extras = []
        if int(row.get("tool_error_case_count") or 0):
            extras.append(f"{row.get('tool_error_case_count')} tool-error cases")
        if int(row.get("invalid_output_count") or 0):
            extras.append(f"{row.get('invalid_output_count')} invalid outputs")
        missing_examples = _join_limited(row.get("weak_labels_without_examples") or [], limit=3)
        if missing_examples:
            extras.append(f"missing train examples for {missing_examples}")
        detail = "; ".join(part for part in (f"weak slices {weak}" if weak else "", f"modes {modes}" if modes else "", ", ".join(extras)) if part)
        return "Diagnose", detail or "no residual failures found"
    if event == "iteration_started":
        return (
            "Search",
            f"round {row.get('iteration')} with {row.get('frontier_width')} parent(s); "
            f"candidate budget used {row.get('dev_evaluations')}/{row.get('dev_budget')}",
        )
    if event == "parent_started":
        return (
            "Frontier",
            f"parent #{row.get('parent_rank')} is {_short_hash(row.get('parent_candidate_id'))}: "
            + _score_brief(row),
        )
    if event == "search_planner_started":
        return (
            "Plan",
            f"asking planner to choose from {row.get('surface_opportunity_count')} legal surface opportunities",
        )
    if event == "search_planner_completed":
        diagnostics = row.get("call_diagnostics") or {}
        return (
            "Plan",
            " ".join(
                part
                for part in (
                    f"planned {row.get('brief_count')} brief(s)",
                    _call_summary(diagnostics),
                )
                if part
            ),
        )
    if event == "search_plan_ready":
        mechanisms = _join_limited(row.get("target_mechanisms") or [], limit=4)
        diagnosis = _short_reason(row.get("diagnosis"), limit=160)
        brief_text = _briefs_brief(row.get("briefs") or [])
        return (
            "Plan",
            f"thinks {diagnosis}; will try {brief_text or (mechanisms or 'no mechanisms')}",
        )
    if event == "proposal_started":
        retry = " retry" if row.get("proposal_retry") else ""
        return (
            "Build",
            f"implementing{retry} planned briefs as transform programs; budget {row.get('proposal_budget')}",
        )
    if event == "proposal_completed":
        diagnostics = row.get("call_diagnostics") or {}
        return (
            "Build",
            " ".join(
                part
                for part in (
                    f"{row.get('returned_count')} candidates, {row.get('valid_count')} compiled, "
                    f"{row.get('invalid_count')} contract failures, {row.get('duplicate_count')} duplicates",
                    _call_summary(diagnostics),
                )
                if part
            ),
        )
    if event == "candidate_evaluation_started":
        return "Candidate", None
    if event == "candidate_stage_started":
        return (
            "Test",
            f"{_stage_name(row.get('stage'))}: {row.get('candidate_count')} candidates on {row.get('case_count')} cases",
        )
    if event == "candidate_stage_completed":
        return (
            "Learn",
            f"{_stage_name(row.get('stage'))}: {row.get('advanced_count')} continue, "
            f"{row.get('accepted_count')} promotable, {row.get('rejected_count')} rejected, "
            f"{row.get('screened_count')} screened",
        )
    if event == "candidate_evaluated":
        status = row.get("frontier_status") or ("accepted" if row.get("accepted") else "rejected")
        reason = row.get("rejection_reason") or row.get("constraint_warning")
        return (
            "Learn",
            f"{_short_hash(row.get('candidate_id'))} {_humanize_key(status)} via "
            f"{_humanize_key(row.get('surface_mechanism'))}: "
            f"score {_format_signed(row.get('score_delta'), digits=3)}, "
            f"fixed {row.get('fixed_count', '?')}, regressed {row.get('regressed_count', '?')}, "
            f"cost {_format_money_delta(row.get('cost_delta'))}, "
            f"full-dev {_format_bool(row.get('full_dev_evaluated'))}"
            + (f" reason={_short_reason(reason)}" if reason else ""),
        )
    if event == "retry_started":
        return "Retry", f"parent #{row.get('parent_rank')} because {_short_reason(row.get('reason'))}"
    if event == "frontier_updated":
        candidates = _join_limited([_short_hash(item) for item in row.get("frontier_candidate_ids") or []], limit=4)
        return "Frontier", f"{row.get('accepted_count')} accepted; next parents {candidates or 'none'}"
    if event == "search_stopped":
        return "Search", f"stopping search: {_short_reason(row.get('reason'))}"
    if event == "simplification_started":
        return (
            "Simplify",
            f"testing simpler variant={_short_hash(row.get('candidate_id'))} of parent={_short_hash(row.get('parent_candidate_id'))}",
        )
    if event == "simplification_completed":
        status = "accepted" if row.get("accepted") else "rejected"
        reason = row.get("rejection_reason")
        return (
            "Simplify",
            f"{status} variant={_short_hash(row.get('variant_candidate_id'))} {_score_brief(row)}"
            + (f" reason={_short_reason(reason)}" if reason else ""),
        )
    if event == "confirmation_started":
        return (
            "Guard",
            f"rechecking {_short_hash(row.get('candidate_id'))} on {row.get('case_count')} sensitive cases before holdout",
        )
    if event == "confirmation_completed":
        status = "passed" if row.get("passed") else "failed"
        return "Guard", f"{_short_hash(row.get('candidate_id'))} {status}: {_short_reason(row.get('reason'))}"
    if event == "confirmation_skipped":
        return "Guard", f"skipped {_short_hash(row.get('candidate_id'))}: {_short_reason(row.get('reason'))}"
    if event == "holdout_candidate_started":
        return "Holdout", f"validating finalist candidate={_short_hash(row.get('candidate_id'))} on {row.get('case_count')} protected case(s)"
    if event == "holdout_candidate_completed":
        status = row.get("finalist_status") or ("validated" if row.get("passed_final_gate") else "rejected")
        reason = row.get("rejection_reason")
        return (
            "Holdout",
            f"{_humanize_key(status)} candidate={_short_hash(row.get('candidate_id'))} {_score_brief(row)}"
            + (f" reason={_short_reason(reason)}" if reason else ""),
        )
    if event == "holdout_validation_skipped":
        candidate = row.get("candidate_id")
        candidate_text = f" candidate={_short_hash(candidate)}" if candidate else ""
        return "Holdout", f"skipped{candidate_text} reason={_short_reason(row.get('reason'))}"
    if event == "case_batch_started":
        return "Evaluate", None
    if event == "case_batch_completed":
        return "Evaluate", None
    if event == "run_completed":
        status = "promoted" if row.get("promoted") else "baseline kept"
        return (
            "Decide",
            f"{status}; selected {_short_hash(row.get('selected_candidate_id'))}; "
            f"dev finalists {row.get('accepted_dev_candidates')}, holdout validations {row.get('holdout_validations')}; "
            f"{_short_reason(row.get('selection_reason'))}",
        )
    return event.upper()[:10] or "EVENT", None


def _format_elapsed(value: object) -> str:
    seconds = int(float(value or 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _score_brief(row: dict[str, Any]) -> str:
    return (
        f"score {_format_number(row.get('mean_score'), digits=3)} "
        f"({_format_count(row.get('pass_count'))}/{_format_count(row.get('case_count'))} pass), "
        f"cost {_format_money(row.get('mean_cost_usd'))}/case, "
        f"latency {_format_seconds(row.get('median_latency_s'))}"
    )


def _failure_brief(row: dict[str, Any]) -> str:
    labels = row.get("failure_labels")
    if not isinstance(labels, dict) or not labels:
        return "no failure labels"
    parts = [f"{key}={value}" for key, value in list(labels.items())[:3]]
    remaining = len(labels) - len(parts)
    if remaining > 0:
        parts.append(f"+{remaining} more")
    return "failures " + ", ".join(parts)


def _slice_brief(row: dict[str, Any]) -> str:
    metrics = row.get("category_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return "no slice breakdown"
    weak: list[tuple[str, float, int, int]] = []
    for label, values in metrics.items():
        if not isinstance(values, dict):
            continue
        count = int(values.get("count") or 0)
        passed = int(values.get("pass_count") or 0)
        if count and passed < count:
            weak.append((str(label), passed / count, passed, count))
    weak.sort(key=lambda item: (item[1], item[0]))
    if not weak:
        return "all reported slices passed"
    parts = [f"{label} {passed}/{count}" for label, _, passed, count in weak[:4]]
    if len(weak) > len(parts):
        parts.append(f"+{len(weak) - len(parts)} more")
    return "weak slices " + ", ".join(parts)


def _briefs_brief(briefs: object) -> str:
    if not isinstance(briefs, list):
        return ""
    normalized = [brief for brief in briefs if isinstance(brief, dict)]
    normalized.sort(key=lambda item: int(item.get("priority") or 1))
    parts: list[str] = []
    for brief in normalized[:4]:
        mechanism = _humanize_key(brief.get("mechanism_class"))
        slices = _join_limited(brief.get("target_slices") or [], limit=3)
        brief_id = str(brief.get("brief_id") or "brief")
        target = f" on {slices}" if slices else ""
        parts.append(f"{brief_id} ({mechanism}{target})")
    if len(normalized) > len(parts):
        parts.append(f"+{len(normalized) - len(parts)} more")
    return "; ".join(parts)


def _call_summary(diagnostics: dict[str, Any]) -> str:
    if not diagnostics:
        return ""
    parts: list[str] = []
    model = diagnostics.get("model")
    if model:
        parts.append(str(model))
    total_tokens = diagnostics.get("total_tokens")
    if total_tokens is None:
        input_tokens = diagnostics.get("input_tokens")
        output_tokens = diagnostics.get("output_tokens")
        if input_tokens is not None or output_tokens is not None:
            try:
                total_tokens = int(input_tokens or 0) + int(output_tokens or 0)
            except (TypeError, ValueError):
                total_tokens = None
    if total_tokens is not None:
        parts.append(f"{_format_count(total_tokens)} tokens")
    elapsed = diagnostics.get("elapsed_s")
    if elapsed is not None:
        parts.append(_format_seconds(elapsed))
    finish_reason = diagnostics.get("finish_reason")
    if finish_reason and finish_reason != "stop":
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


def _humanize_key(value: object) -> str:
    text = str(value or "")
    return text.replace("_", " ") if text else "n/a"


def _join_limited(values: list[Any] | tuple[Any, ...], *, limit: int) -> str:
    items = [str(item) for item in values if item is not None and str(item)]
    if not items:
        return ""
    shown = items[:limit]
    if len(items) > limit:
        shown.append(f"+{len(items) - limit}")
    return ", ".join(shown)


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


def run_release_check(
    *,
    config: RatchetRunConfig,
    sample_limit: int | None = None,
    repeats: int | None = None,
) -> EvalHealthReport:
    adapter, cases = load_runtime(config)
    preflight_limit = config.eval_health.sample_limit if sample_limit is None else sample_limit
    print("Release check: preflight")
    preflight = run_preflight_check(
        adapter_spec=config.adapter,
        adapter=adapter,
        cases=cases,
        objective=config.objective,
        sample_limit=preflight_limit,
        optimizer_model=config.optimizer_model if os.environ.get("RATCHET_CHECK_OPTIMIZER_MODEL") == "1" else None,
        optimizer_env_path=config.env_file,
    )
    materialized = preflight.materialization.get("verified_surfaces") or []
    print(
        "Preflight passed: "
        f"{len(cases)} eval cases, "
        f"{len(preflight.generated_surface)} generated surface(s), "
        f"verified surfaces={_join_limited(materialized, limit=6) or 'none'}"
    )

    print("Release check: eval health")
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
    out_dir = config.out / "release_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval_health.json").write_text(json.dumps(report.to_dict(), indent=2, default=str) + "\n")
    (out_dir / "eval_health.md").write_text(render_eval_health_markdown(report))
    _print_eval_health_summary(report, out_dir=out_dir, strict=True)
    if report.fatal or report.warning:
        print("Release check failed: eval health must be healthy with no warnings.")
    else:
        print("Release check passed.")
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
        optimizer_model=getattr(args, "optimizer_model", None),
        optimizer_reasoning=getattr(args, "optimizer_reasoning", None),
        search_planner_model=getattr(args, "search_planner_model", None),
        search_planner_reasoning=getattr(args, "search_planner_reasoning", None),
        candidate_implementer_model=getattr(args, "candidate_implementer_model", None),
        candidate_implementer_reasoning=getattr(args, "candidate_implementer_reasoning", None),
        samples_per_case=getattr(args, "samples_per_case", None),
        case_concurrency=getattr(args, "case_concurrency", None),
        stage_case_concurrency=getattr(args, "stage_case_concurrency", None),
        max_case_retries=getattr(args, "max_case_retries", None),
        case_timeout_s=getattr(args, "case_timeout_s", None),
        fail_fast=True if getattr(args, "fail_fast", False) else None,
        sanitize_examples=True if getattr(args, "sanitize_examples", False) else None,
        expensive_candidate_cost_ratio=getattr(args, "expensive_candidate_cost_ratio", None),
        max_dev_measurement_cost_usd=getattr(args, "max_dev_measurement_cost_usd", None),
        max_holdout_measurement_cost_usd=getattr(args, "max_holdout_measurement_cost_usd", None),
        max_dev_measurement_tool_calls=getattr(args, "max_dev_measurement_tool_calls", None),
        max_holdout_measurement_tool_calls=getattr(args, "max_holdout_measurement_tool_calls", None),
        max_dev_measurement_turns=getattr(args, "max_dev_measurement_turns", None),
        max_holdout_measurement_turns=getattr(args, "max_holdout_measurement_turns", None),
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
    parser.add_argument("--dev-budget", type=int, help="Max dev candidate evaluations after baseline")
    parser.add_argument("--holdout-budget", type=int, help="Holdout finalist validation budget")
    parser.add_argument("--mode", choices=["correctness", "cost", "latency"], help="Primary optimization objective")
    parser.add_argument("--allowed-models", help="Comma-separated model allowlist for model-config transforms")
    parser.add_argument("--optimizer-model", help="Model used by Ratchet's research loop")
    parser.add_argument("--optimizer-reasoning", help="Reasoning effort for Ratchet's research loop")
    parser.add_argument("--search-planner-model", help="Override model for Ratchet's search planner")
    parser.add_argument("--search-planner-reasoning", help="Override reasoning effort for Ratchet's search planner")
    parser.add_argument("--candidate-implementer-model", help="Override model for Ratchet's candidate implementer")
    parser.add_argument("--candidate-implementer-reasoning", help="Override reasoning effort for Ratchet's candidate implementer")
    parser.add_argument("--samples-per-case", type=int, help="Number of repeated samples to evaluate per candidate/case")
    parser.add_argument("--case-concurrency", type=int, help="Maximum concurrent case evaluations per candidate")
    parser.add_argument(
        "--stage-case-concurrency",
        type=int,
        help="Maximum concurrent case evaluations across a multi-candidate stage; defaults to --case-concurrency.",
    )
    parser.add_argument("--max-case-retries", type=int, help="Per-case retry budget after the first attempt")
    parser.add_argument("--case-timeout-s", type=int, help="Per-case timeout in seconds")
    parser.add_argument(
        "--expensive-candidate-cost-ratio",
        type=float,
        help="Report candidates above this deployed cost ratio as expensive tradeoffs.",
    )
    parser.add_argument(
        "--max-dev-measurement-cost-usd",
        type=float,
        help="Maximum candidate measurement spend across dev stages; omit for no dollar ceiling.",
    )
    parser.add_argument(
        "--max-holdout-measurement-cost-usd",
        type=float,
        help="Maximum candidate measurement spend for holdout validation; omit for no dollar ceiling.",
    )
    parser.add_argument(
        "--max-dev-measurement-tool-calls",
        type=int,
        help="Maximum candidate tool calls across dev stages; omit for no tool-call ceiling.",
    )
    parser.add_argument(
        "--max-holdout-measurement-tool-calls",
        type=int,
        help="Maximum candidate tool calls for holdout validation; omit for no tool-call ceiling.",
    )
    parser.add_argument(
        "--max-dev-measurement-turns",
        type=int,
        help="Maximum candidate interaction turns across dev stages; omit for no turn ceiling.",
    )
    parser.add_argument(
        "--max-holdout-measurement-turns",
        type=int,
        help="Maximum candidate interaction turns for holdout validation; omit for no turn ceiling.",
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


def add_release_check_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to ratchet.toml")
    parser.add_argument("--adapter", help="Adapter import path, e.g. package.module:adapter")
    parser.add_argument("--evals", help="Path to evals JSONL")
    parser.add_argument("--out", help="Output directory")
    parser.add_argument("--env-file", help="Path to .env with model provider API keys")
    parser.add_argument("--sample-limit", type=int, help="Maximum dev/holdout cases to probe")
    parser.add_argument("--repeats", type=int, help="Repeated baseline probes per sampled case")
    parser.add_argument("--case-timeout-s", type=int, help="Per-case timeout in seconds for baseline probes")


def add_ideation_assessment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Completed Ratchet run output directory")
    parser.add_argument("--spec", help="Optional ideation assessment spec JSON")
    parser.add_argument("--out", help="Output JSON path; defaults to RUN_DIR/ideation_assessment.json")


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
    check_parser.add_argument("--sample-limit", type=int, default=1, help="How many cases to probe during preflight")

    eval_health_parser = subparsers.add_parser("eval-health", help="Check eval-set and grader health before optimization.")
    add_eval_health_arguments(eval_health_parser)

    release_parser = subparsers.add_parser(
        "release-check",
        help="Run preflight plus strict eval-health gates for a release-candidate config.",
    )
    add_release_check_arguments(release_parser)

    assess_parser = subparsers.add_parser(
        "assess-ideation",
        help="Assess optimizer ideation quality from an existing run directory.",
    )
    add_ideation_assessment_arguments(assess_parser)

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

        if args.command == "release-check":
            config = _apply_run_overrides(args)
            report = run_release_check(
                config=config,
                sample_limit=getattr(args, "sample_limit", None),
                repeats=getattr(args, "repeats", None),
            )
            if report.fatal or report.warning:
                return 5
            return 0

        if args.command == "assess-ideation":
            assessment = write_ideation_assessment(
                args.run_dir,
                spec_path=getattr(args, "spec", None),
                out_path=getattr(args, "out", None),
            )
            summary = assessment.get("summary") or {}
            print(
                "Ideation assessment: "
                f"{summary.get('passed_checks')}/{summary.get('total_checks')} checks passed; "
                f"valid_impl_rate={float(summary.get('valid_implementation_rate') or 0.0):.3f}; "
                f"selected_holdout_delta={float(summary.get('selected_holdout_score_delta') or 0.0):+.3f}"
            )
            print(f"JSON: {getattr(args, 'out', None) or Path(args.run_dir) / 'ideation_assessment.json'}")
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
