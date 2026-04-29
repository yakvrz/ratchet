from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from ratchet.evidence_ledger import confirmation_stability_result
from ratchet.objectives import behavior_flip_summary, compare_summaries
from ratchet.results import PatchSummary, RatchetResult
from ratchet.transform_program import CompiledCandidate
from ratchet.types import EvalCase, OptimizationObjective


LOW_OUTPUT_TOKEN_RATIO = 0.25


def runtime_reliability_diagnostics(
    reference: PatchSummary,
    candidate: PatchSummary,
) -> dict[str, Any]:
    flip_summary = behavior_flip_summary(reference, candidate)
    fixed_ids = set(flip_summary["fixed_case_ids"])
    runtime_only = _is_runtime_only_patch(candidate.patch)
    runtime_involved = _touches_runtime(candidate.patch)
    reference_by_id = _representative_evaluations(reference)
    candidate_by_id = _representative_evaluations(candidate)
    fixed_invalid: list[str] = []
    low_token_fixed: list[str] = []
    finish_reason_changed: list[str] = []
    finish_reasons: dict[str, list[str]] = defaultdict(list)
    for case_id in sorted(set(reference_by_id) & set(candidate_by_id)):
        reference_eval = reference_by_id[case_id]
        candidate_eval = candidate_by_id[case_id]
        reference_finish_reason = str(reference_eval.record.diagnostics.metadata.get("finish_reason") or "")
        candidate_finish_reason = str(candidate_eval.record.diagnostics.metadata.get("finish_reason") or "")
        if reference_finish_reason != candidate_finish_reason:
            finish_reason_changed.append(case_id)
    for case_id in sorted(fixed_ids):
        reference_eval = reference_by_id.get(case_id)
        candidate_eval = candidate_by_id.get(case_id)
        if reference_eval is None or candidate_eval is None:
            continue
        if _invalid_output(reference_eval):
            fixed_invalid.append(case_id)
        cap = _requested_output_cap(reference_eval) or _requested_output_cap(candidate_eval)
        output_tokens = reference_eval.record.metrics.output_tokens
        if cap and output_tokens <= max(1, int(cap * LOW_OUTPUT_TOKEN_RATIO)):
            low_token_fixed.append(case_id)
        for evaluation in (reference_eval, candidate_eval):
            finish_reason = str(evaluation.record.diagnostics.metadata.get("finish_reason") or "")
            if finish_reason:
                finish_reasons[case_id].append(finish_reason)
    baseline_runtime_defect_fixed = bool(runtime_involved and ((fixed_invalid and low_token_fixed) or finish_reason_changed))
    return {
        "patch_hash": candidate.patch_hash,
        "runtime_only": runtime_only,
        "runtime_involved": runtime_involved,
        "runtime_finding": baseline_runtime_defect_fixed,
        "diagnostic_class": (
            "baseline_runtime_defect_fixed"
            if baseline_runtime_defect_fixed
            else "no_runtime_reliability_finding"
        ),
        "baseline_runtime_defect_fixed": baseline_runtime_defect_fixed,
        "reason": (
            "Baseline runtime defect fixed: runtime patch corrected invalid outputs where baseline runs ended far below the requested output cap."
            if baseline_runtime_defect_fixed
            else "No runtime reliability suspicion detected."
        ),
        "fixed_invalid_output_case_ids": fixed_invalid,
        "low_token_fixed_case_ids": low_token_fixed,
        "finish_reason_changed_case_ids": finish_reason_changed,
        "regressed_case_ids": list(flip_summary["regressed_case_ids"]),
        "finish_reasons_by_case": dict(finish_reasons),
    }


def confirmation_case_subset(
    reference: PatchSummary,
    candidate: PatchSummary,
    dev_cases: tuple[EvalCase, ...],
    *,
    stable_limit: int = 3,
) -> tuple[EvalCase, ...]:
    flip_summary = behavior_flip_summary(reference, candidate)
    selected_ids = [
        *flip_summary["fixed_case_ids"],
        *flip_summary["regressed_case_ids"],
    ]
    reference_passed = {
        case_id: case_passed
        for case_id, _, _, _, case_passed in reference._case_rows()
    }
    candidate_passed = {
        case_id: case_passed
        for case_id, _, _, _, case_passed in candidate._case_rows()
    }
    for case in dev_cases:
        if len(selected_ids) >= len(flip_summary["fixed_case_ids"]) + len(flip_summary["regressed_case_ids"]) + stable_limit:
            break
        if case.id in selected_ids:
            continue
        if reference_passed.get(case.id) == candidate_passed.get(case.id):
            selected_ids.append(case.id)
    case_by_id = {case.id: case for case in dev_cases}
    return tuple(case_by_id[case_id] for case_id in selected_ids if case_id in case_by_id)


def confirmation_result(
    *,
    reference: PatchSummary,
    candidate: PatchSummary,
    confirmation_reference: PatchSummary,
    confirmation_candidate: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> dict[str, Any]:
    objective = objective or OptimizationObjective()
    stability = confirmation_stability_result(
        reference=reference,
        candidate=candidate,
        repeated_reference=confirmation_reference,
        repeated_candidate=confirmation_candidate,
        objective=objective,
    )
    return {
        "patch_hash": candidate.patch_hash,
        "diagnostic_class": stability["status"],
        **stability,
    }


def build_run_profile(result: RatchetResult, out_dir: Path) -> dict[str, Any]:
    progress_rows = _read_jsonl(out_dir / "progress.jsonl")
    optimizer_calls = _optimizer_call_profile(result.optimizer_call_diagnostics)
    run_cost = _run_cost_profile(result, optimizer_calls)
    return {
        "elapsed_s": max((float(row.get("elapsed_s", 0.0)) for row in progress_rows), default=0.0),
        "phase_durations_s": _phase_durations(progress_rows),
        "phase_attempt_durations_s": _phase_attempt_durations(progress_rows),
        "slowest_cases": _case_metric_extremes(result, metric="latency_s", limit=8),
        "highest_token_cases": _case_metric_extremes(result, metric="total_tokens", limit=8),
        "patch_profiles": _patch_profiles(result),
        "patch_deltas_vs_baseline": _patch_deltas_vs_baseline(result),
        "optimizer_calls": optimizer_calls,
        "run_cost": run_cost,
        "cache_events": {
            "case_cache_hits": sum(1 for row in progress_rows if row.get("event") == "case_cache_hit"),
            "case_completed": sum(1 for row in progress_rows if row.get("event") == "case_completed"),
            "diagnosis_cache_hits": sum(
                1
                for row in progress_rows
                if row.get("event") == "diagnosis_completed" and row.get("cached")
            ),
            "task_theory_cache_hits": sum(
                1
                for row in progress_rows
                if row.get("event") == "task_theory_ready" and row.get("cached")
            ),
        },
        "cache_hit_rate": _cache_hit_rate(progress_rows),
    }


def _run_cost_profile(result: RatchetResult, optimizer_calls: dict[str, Any]) -> dict[str, Any]:
    seen_evaluations: set[tuple[str, str, int]] = set()
    eval_cost = 0.0
    eval_input_tokens = 0
    eval_output_tokens = 0
    eval_total_tokens = 0
    eval_count = 0
    for summary in [
        result.baseline_dev,
        result.baseline_holdout,
        *result.accepted_dev_patches,
        *result.holdout_patches,
    ]:
        for evaluation in summary.evaluations:
            key = (summary.patch_hash, evaluation.case.id, evaluation.sample_index)
            if key in seen_evaluations:
                continue
            seen_evaluations.add(key)
            metrics = evaluation.record.metrics
            eval_count += 1
            eval_cost += float(metrics.cost_usd or 0.0)
            eval_input_tokens += int(metrics.input_tokens or 0)
            eval_output_tokens += int(metrics.output_tokens or 0)
            eval_total_tokens += int(metrics.total_tokens or 0)
    optimizer_totals = optimizer_calls.get("totals") or {}
    optimizer_cost = float(optimizer_totals.get("cost_usd") or 0.0)
    optimizer_input_tokens = int(optimizer_totals.get("input_tokens") or 0)
    optimizer_output_tokens = int(optimizer_totals.get("output_tokens") or 0)
    optimizer_total_tokens = int(optimizer_totals.get("total_tokens") or 0)
    return {
        "eval_cost_usd": eval_cost,
        "optimizer_cost_usd": optimizer_cost,
        "total_cost_usd": eval_cost + optimizer_cost,
        "eval_case_evaluations": eval_count,
        "eval_input_tokens": eval_input_tokens,
        "eval_output_tokens": eval_output_tokens,
        "eval_tokens": eval_total_tokens,
        "optimizer_input_tokens": optimizer_input_tokens,
        "optimizer_output_tokens": optimizer_output_tokens,
        "optimizer_tokens": optimizer_total_tokens,
        "total_input_tokens": eval_input_tokens + optimizer_input_tokens,
        "total_output_tokens": eval_output_tokens + optimizer_output_tokens,
        "total_tokens": eval_total_tokens + optimizer_total_tokens,
    }


def quality_cost_tradeoffs(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in proposals:
        reason = str(row.get("rejection_reason") or row.get("constraint_warning") or "")
        if row.get("transform_family") != "model_substitution" or "cost constraint rejected" not in reason:
            continue
        rows.append(
            {
                "patch_hash": row.get("patch_hash"),
                "transform_instance": row.get("transform_instance"),
                "rejection_reason": reason,
                "comparison_to_parent": row.get("comparison_to_parent"),
                "metrics": _compact_metrics(row.get("metrics") or {}),
                "patch": row.get("patch"),
            }
        )
    return rows


def _representative_evaluations(summary: PatchSummary) -> dict[str, Any]:
    rows = {}
    for case_id, evaluations, _, _, _ in summary._case_rows():
        rows[case_id] = next((item for item in evaluations if not item.grade.passed), evaluations[0])
    return rows


def _invalid_output(evaluation: Any) -> bool:
    output = evaluation.record.output
    return (
        any("invalid_output" in label for label in evaluation.grade.labels)
        or (isinstance(output, dict) and "invalid_output" in output)
        or bool(evaluation.record.diagnostics.metadata.get("invalid_output"))
    )


def _requested_output_cap(evaluation: Any) -> int | None:
    value = evaluation.record.diagnostics.metadata.get("requested_output_cap")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_runtime_only_patch(patch: CompiledCandidate | None) -> bool:
    return patch is not None and bool(patch.program.patches) and all(
        transform.op.op in {"set_model_config", "set_retry_policy", "set_turn_limit", "set_tool_call_limit"}
        for transform in patch.program.patches
    )


def _touches_runtime(patch: CompiledCandidate | None) -> bool:
    return patch is not None and any(
        transform.op.op in {"set_model_config", "set_retry_policy", "set_turn_limit", "set_tool_call_limit"}
        for transform in patch.program.patches
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line))
        except json.JSONDecodeError:
            continue
    return rows


def _phase_durations(rows: list[dict[str, Any]]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for name, intervals in _phase_intervals(rows).items():
        duration = _union_interval_duration(intervals)
        if duration:
            durations[name] = round(duration, 3)
    return durations


def _phase_attempt_durations(rows: list[dict[str, Any]]) -> dict[str, float]:
    durations: dict[str, float] = {}
    for name, intervals in _phase_intervals(rows).items():
        duration = sum(max(end - start, 0.0) for start, end in intervals)
        if duration:
            durations[name] = round(duration, 3)
    return durations


def _phase_intervals(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    pairings = [
        ("baseline_dev", "baseline_dev_started", "baseline_dev_completed"),
        ("baseline_holdout", "baseline_holdout_started", "baseline_holdout_completed"),
        ("diagnosis", "diagnosis_started", "diagnosis_completed"),
        ("proposal", "proposal_started", "proposal_completed"),
        ("candidate_evaluation", "candidate_evaluation_started", "candidate_evaluated"),
        ("confirmation", "confirmation_started", "confirmation_completed"),
        ("holdout_validation", "holdout_candidate_started", "holdout_candidate_completed"),
    ]
    intervals: dict[str, list[tuple[float, float]]] = {}
    for name, start_event, end_event in pairings:
        starts = _event_rows(rows, start_event)
        ends = _event_rows(rows, end_event)
        phase_intervals: list[tuple[float, float]] = []
        for start, end in zip(starts, ends):
            phase_intervals.append((float(start.get("elapsed_s", 0.0)), float(end.get("elapsed_s", 0.0))))
        if phase_intervals:
            intervals[name] = phase_intervals
    return intervals


def _union_interval_duration(intervals: list[tuple[float, float]]) -> float:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((start, end) for start, end in intervals if end > start):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return sum(end - start for start, end in merged)


def _event_rows(rows: list[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("event") == event]


def _case_metric_extremes(result: RatchetResult, *, metric: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, summaries in {
        "baseline_dev": [result.baseline_dev],
        "baseline_holdout": [result.baseline_holdout],
        "accepted_dev": result.accepted_dev_patches,
        "holdout": result.holdout_patches,
    }.items():
        for summary in summaries:
            for evaluation in summary.evaluations:
                metrics = evaluation.record.metrics
                rows.append(
                    {
                        "split_group": split_name,
                        "patch_hash": summary.patch_hash,
                        "case_id": evaluation.case.id,
                        "latency_s": metrics.latency_s,
                        "total_tokens": metrics.total_tokens,
                        "input_tokens": metrics.input_tokens,
                        "output_tokens": metrics.output_tokens,
                        "cost_usd": metrics.cost_usd,
                    }
                )
    rows.sort(key=lambda item: float(item.get(metric, 0.0)), reverse=True)
    return rows[:limit]


def _patch_profiles(result: RatchetResult) -> list[dict[str, Any]]:
    summaries = [result.baseline_dev, result.baseline_holdout, *result.accepted_dev_patches, *result.holdout_patches]
    seen: set[tuple[str, str]] = set()
    rows = []
    for summary in summaries:
        key = (summary.split, summary.patch_hash)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "patch_hash": summary.patch_hash,
                "split": summary.split,
                "case_count": summary.case_count,
                "pass_count": summary.pass_count,
                "mean_score": summary.mean_score,
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
                "operation_count": summary.operation_count,
            }
        )
    return rows


def _patch_deltas_vs_baseline(result: RatchetResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, baseline, summaries in [
        ("dev", result.baseline_dev, result.accepted_dev_patches),
        ("holdout", result.baseline_holdout, result.holdout_patches),
    ]:
        for summary in summaries:
            if set(summary.grouped_evaluations) != set(baseline.grouped_evaluations):
                continue
            comparison = compare_summaries(baseline, summary)
            rows.append(
                {
                    "patch_hash": summary.patch_hash,
                    "split": split_name,
                    "score_delta": comparison.score_delta,
                    "score_ci": comparison.score_ci,
                    "cost_delta": comparison.cost_delta,
                    "cost_ci": comparison.cost_ci,
                    "token_delta": comparison.token_delta,
                    "token_ci": comparison.token_ci,
                    "latency_delta": comparison.latency_delta,
                    "latency_ci": comparison.latency_ci,
                }
            )
    return rows


def _cache_hit_rate(rows: list[dict[str, Any]]) -> float:
    hits = sum(1 for row in rows if row.get("event") == "case_cache_hit")
    fresh = sum(1 for row in rows if row.get("event") == "case_completed")
    total = hits + fresh
    return round(hits / total, 4) if total else 0.0


def _optimizer_call_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "call_count": len(rows),
        "elapsed_s": round(sum(float(row.get("elapsed_s") or 0.0) for row in rows), 3),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
        "cost_usd": sum(float(row.get("cost_usd") or 0.0) for row in rows if row.get("cost_usd") is not None),
    }
    by_component: dict[str, dict[str, Any]] = {}
    for row in rows:
        component = str(row.get("component") or "unknown")
        item = by_component.setdefault(
            component,
            {"call_count": 0, "elapsed_s": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
        )
        item["call_count"] += 1
        item["elapsed_s"] = round(float(item["elapsed_s"]) + float(row.get("elapsed_s") or 0.0), 3)
        item["input_tokens"] += int(row.get("input_tokens") or 0)
        item["output_tokens"] += int(row.get("output_tokens") or 0)
        item["total_tokens"] += int(row.get("total_tokens") or 0)
        if row.get("cost_usd") is not None:
            item["cost_usd"] += float(row.get("cost_usd") or 0.0)
    return {
        "totals": totals,
        "by_component": by_component,
        "calls": rows,
    }


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_hash": metrics.get("patch_hash"),
        "case_count": metrics.get("case_count"),
        "pass_count": metrics.get("pass_count"),
        "mean_score": metrics.get("mean_score"),
        "mean_cost_usd": metrics.get("mean_cost_usd"),
        "median_latency_s": metrics.get("median_latency_s"),
    }
