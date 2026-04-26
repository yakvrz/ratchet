from __future__ import annotations

from dataclasses import dataclass
import random
import statistics
from typing import Any

from ratchet.results import PatchSummary, Comparison
from ratchet.types import OptimizationObjective


NON_INFERIORITY_MARGIN = 0.01
DEFAULT_COST_GUARD = 3.0
DEFAULT_LATENCY_GUARD = 3.0
DEFAULT_COST_MODE_LATENCY_GUARD = 1.15
FINALIST_STATUSES = {"validated", "directional", "failed"}


@dataclass(frozen=True)
class FinalGateResult:
    status: str
    reason: str | None
    comparison: Comparison

    @property
    def validated(self) -> bool:
        return self.status == "validated"

    @property
    def directional(self) -> bool:
        return self.status == "directional"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "validated": self.validated,
            "directional": self.directional,
            "comparison": self.comparison.to_dict(),
        }


def bootstrap_mean_ci(values: list[float], iterations: int = 2000, seed: int = 7) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        boot = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(statistics.fmean(boot))
    samples.sort()
    lower_index = int(0.025 * iterations)
    upper_index = int(0.975 * iterations)
    return samples[lower_index], samples[upper_index]


def compare_summaries(reference: PatchSummary, patch_summary: PatchSummary) -> Comparison:
    reference_by_id = _case_metric_rows(reference)
    patch_by_id = _case_metric_rows(patch_summary)
    if set(reference_by_id) != set(patch_by_id):
        raise ValueError("Patch summaries must cover the same cases for paired comparison.")
    case_ids = list(reference_by_id)
    score_deltas = [
        patch_by_id[case_id]["score"] - reference_by_id[case_id]["score"]
        for case_id in case_ids
    ]
    cost_deltas = [
        patch_by_id[case_id]["cost"] - reference_by_id[case_id]["cost"]
        for case_id in case_ids
    ]
    token_deltas = [
        patch_by_id[case_id]["tokens"] - reference_by_id[case_id]["tokens"]
        for case_id in case_ids
    ]
    latency_deltas = [
        patch_by_id[case_id]["latency"] - reference_by_id[case_id]["latency"]
        for case_id in case_ids
    ]
    return Comparison(
        score_delta=statistics.fmean(score_deltas),
        score_ci=bootstrap_mean_ci(score_deltas),
        cost_delta=statistics.fmean(cost_deltas),
        cost_ci=bootstrap_mean_ci(cost_deltas),
        token_delta=statistics.fmean(token_deltas),
        token_ci=bootstrap_mean_ci(token_deltas),
        latency_delta=statistics.fmean(latency_deltas),
        latency_ci=bootstrap_mean_ci(latency_deltas),
    )


def _case_metric_rows(summary: PatchSummary) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for case_id, evaluations, mean_score, _, _ in summary._case_rows():
        rows[case_id] = {
            "score": mean_score,
            "cost": statistics.fmean(evaluation.record.metrics.cost_usd for evaluation in evaluations),
            "tokens": statistics.fmean(float(evaluation.record.metrics.total_tokens) for evaluation in evaluations),
            "latency": statistics.median(evaluation.record.metrics.latency_s for evaluation in evaluations),
        }
    return rows


def behavior_flip_summary(reference: PatchSummary, patch_summary: PatchSummary) -> dict[str, Any]:
    reference_rows = {case_id: case_passed for case_id, _, _, _, case_passed in reference._case_rows()}
    patch_rows = {case_id: case_passed for case_id, _, _, _, case_passed in patch_summary._case_rows()}
    if set(reference_rows) != set(patch_rows):
        raise ValueError("Patch summaries must cover the same cases for flip comparison.")
    fixed: list[str] = []
    regressed: list[str] = []
    for case_id, reference_passed in reference_rows.items():
        patch_passed = patch_rows[case_id]
        if reference_passed == patch_passed:
            continue
        if not reference_passed and patch_passed:
            fixed.append(case_id)
        elif reference_passed and not patch_passed:
            regressed.append(case_id)
    return {
        "fixed_case_ids": fixed,
        "regressed_case_ids": regressed,
        "fixed_count": len(fixed),
        "regressed_count": len(regressed),
    }


def patch_satisfies_constraints(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> bool:
    return constraint_rejection_reason(baseline, patch_summary, objective) is None


def constraint_rejection_reason(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> str | None:
    constraints = objective.constraints
    score_delta = patch_summary.mean_score - baseline.mean_score
    required_score_delta = constraints.min_correctness_delta
    if required_score_delta is None:
        required_score_delta = 0.0 if objective.mode == "correctness" else -NON_INFERIORITY_MARGIN
    if score_delta + 1e-9 < required_score_delta:
        return (
            "correctness constraint rejected patch "
            f"(score delta {score_delta:.4f} < required {required_score_delta:.4f})"
        )
    max_cost_ratio = constraints.max_cost_ratio
    if max_cost_ratio is None:
        max_cost_ratio = DEFAULT_COST_GUARD if objective.mode == "correctness" else None
    if max_cost_ratio is not None and baseline.mean_cost_usd > 0:
        if patch_summary.mean_cost_usd > baseline.mean_cost_usd * max_cost_ratio:
            return (
                "cost constraint rejected patch "
                f"(${patch_summary.mean_cost_usd:.6f} > {max_cost_ratio:.2f}x baseline)"
            )
    max_latency_ratio = constraints.max_latency_ratio
    if max_latency_ratio is None:
        if objective.mode == "correctness":
            max_latency_ratio = DEFAULT_LATENCY_GUARD
        elif objective.mode == "cost":
            max_latency_ratio = DEFAULT_COST_MODE_LATENCY_GUARD
    if max_latency_ratio is not None and baseline.median_latency_s > 0:
        if patch_summary.median_latency_s > baseline.median_latency_s * max_latency_ratio:
            return (
                "latency constraint rejected patch "
                f"({patch_summary.median_latency_s:.3f}s > {max_latency_ratio:.2f}x baseline)"
            )
    return None


def objective_improved(
    reference: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> bool:
    return objective_rejection_reason(reference, patch_summary, objective) is None


def objective_rejection_reason(
    reference: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> str | None:
    if objective.mode == "correctness":
        if patch_summary.pass_count > reference.pass_count:
            return None
        if patch_summary.pass_count < reference.pass_count:
            return "correctness objective rejected pass count regression"
        if patch_summary.mean_score > reference.mean_score + NON_INFERIORITY_MARGIN:
            return None
        return "correctness objective did not improve pass count or mean score"
    if objective.mode == "cost":
        if patch_summary.mean_score < reference.mean_score - NON_INFERIORITY_MARGIN:
            return "cost objective rejected correctness tradeoff"
        if patch_summary.mean_cost_usd >= reference.mean_cost_usd:
            return "cost objective did not reduce mean cost"
        return None
    if objective.mode == "latency":
        if patch_summary.mean_score < reference.mean_score - NON_INFERIORITY_MARGIN:
            return "latency objective rejected correctness tradeoff"
        if patch_summary.median_latency_s >= reference.median_latency_s:
            return "latency objective did not reduce median latency"
        return None
    raise ValueError(f"Unsupported optimization mode: {objective.mode}")


def patch_rejection_reason(
    *,
    baseline: PatchSummary,
    reference: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> str | None:
    constraint_reason = constraint_rejection_reason(baseline, patch_summary, objective)
    if constraint_reason is not None:
        return constraint_reason
    return objective_rejection_reason(reference, patch_summary, objective)


def objective_sort_key(summary: PatchSummary, objective: OptimizationObjective) -> tuple[Any, ...]:
    if objective.mode == "correctness":
        return (
            -summary.pass_count,
            -summary.mean_score,
            summary.mean_cost_usd,
            summary.median_latency_s,
            summary.operation_count,
            summary.patch_hash,
        )
    if objective.mode == "cost":
        return (
            summary.mean_cost_usd,
            -summary.pass_count,
            -summary.mean_score,
            summary.median_latency_s,
            summary.operation_count,
            summary.patch_hash,
        )
    return (
        summary.median_latency_s,
        -summary.pass_count,
        -summary.mean_score,
        summary.mean_cost_usd,
        summary.operation_count,
        summary.patch_hash,
    )


def final_gate(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> tuple[bool, Comparison]:
    gate = final_gate_status(baseline, patch_summary, objective)
    return gate.validated, gate.comparison


def final_gate_rejection_reason(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> tuple[str | None, Comparison]:
    gate = final_gate_status(baseline, patch_summary, objective)
    return (None if gate.validated else gate.reason), gate.comparison


def final_gate_status(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> FinalGateResult:
    objective = objective or OptimizationObjective()
    comparison = compare_summaries(baseline, patch_summary)
    constraint_reason = constraint_rejection_reason(baseline, patch_summary, objective)
    if constraint_reason is not None:
        return FinalGateResult(status="failed", reason=constraint_reason, comparison=comparison)
    objective_reason = objective_rejection_reason(baseline, patch_summary, objective)
    if objective_reason is not None:
        return FinalGateResult(status="failed", reason=objective_reason, comparison=comparison)
    uncertainty_reason = uncertainty_rejection_reason(comparison, objective)
    if uncertainty_reason is not None:
        return FinalGateResult(status="directional", reason=uncertainty_reason, comparison=comparison)
    return FinalGateResult(status="validated", reason=None, comparison=comparison)


def uncertainty_rejection_reason(
    comparison: Comparison,
    objective: OptimizationObjective,
) -> str | None:
    if objective.mode == "correctness":
        if comparison.score_ci[0] <= 0.0:
            return (
                "correctness uncertainty rejected patch "
                f"(score CI lower {comparison.score_ci[0]:.4f} <= 0.0000)"
            )
        return None
    if objective.mode == "cost":
        if comparison.score_ci[0] < -NON_INFERIORITY_MARGIN:
            return (
                "cost uncertainty rejected correctness tradeoff "
                f"(score CI lower {comparison.score_ci[0]:.4f} < {-NON_INFERIORITY_MARGIN:.4f})"
            )
        if comparison.cost_ci[1] >= 0.0:
            return (
                "cost uncertainty rejected patch "
                f"(cost CI upper {comparison.cost_ci[1]:.6f} >= 0.000000)"
            )
        return None
    if objective.mode == "latency":
        if comparison.score_ci[0] < -NON_INFERIORITY_MARGIN:
            return (
                "latency uncertainty rejected correctness tradeoff "
                f"(score CI lower {comparison.score_ci[0]:.4f} < {-NON_INFERIORITY_MARGIN:.4f})"
            )
        if comparison.latency_ci[1] >= 0.0:
            return (
                "latency uncertainty rejected patch "
                f"(latency CI upper {comparison.latency_ci[1]:.4f} >= 0.0000)"
            )
        return None
    raise ValueError(f"Unsupported optimization mode: {objective.mode}")


def pareto_frontier(summaries: list[PatchSummary]) -> list[dict[str, Any]]:
    frontier: list[PatchSummary] = []
    for summary in summaries:
        dominated = False
        for other in summaries:
            if other is summary:
                continue
            no_worse = (
                other.mean_score >= summary.mean_score
                and other.mean_cost_usd <= summary.mean_cost_usd
                and other.median_latency_s <= summary.median_latency_s
            )
            strictly_better = (
                other.mean_score > summary.mean_score
                or other.mean_cost_usd < summary.mean_cost_usd
                or other.median_latency_s < summary.median_latency_s
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(summary)
    frontier.sort(key=lambda summary: (-summary.mean_score, summary.mean_cost_usd, summary.median_latency_s))
    return [summary.to_dict() for summary in frontier]
