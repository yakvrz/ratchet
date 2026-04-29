from __future__ import annotations

from dataclasses import dataclass
from math import comb
import random
import statistics
from typing import Any

from ratchet.results import PassSignificance, PatchSummary, Comparison
from ratchet.types import OptimizationObjective


NON_INFERIORITY_MARGIN = 0.01
DEFAULT_LATENCY_GUARD = 3.0
DEFAULT_COST_MODE_LATENCY_GUARD = 1.15
SCORE_EQUIVALENCE_FLOOR = 0.05
COST_EQUIVALENCE_FRACTION = 0.0
LATENCY_EQUIVALENCE_FRACTION = 0.0
SIGNIFICANCE_ALPHA = 0.10
FINALIST_STATUSES = {"validated", "directional", "failed", "unstable"}


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


def mcnemar_pvalue(fixed: int, regressed: int) -> float:
    """One-sided exact binomial test on discordant pass/fail pairs.

    H0: P(fix) == P(regress); H1: candidate fixes more cases than it regresses.
    Returns the probability of seeing at least ``fixed`` successes out of
    ``fixed + regressed`` trials under a fair coin. Returns 1.0 when there are
    no discordant pairs (i.e., no evidence either way).
    """
    n = fixed + regressed
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(fixed, n + 1))
    return tail / (2 ** n)


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
    flips = behavior_flip_summary(reference, patch_summary)
    pass_significance = PassSignificance(
        fixed_count=flips["fixed_count"],
        regressed_count=flips["regressed_count"],
        n_discordant=flips["fixed_count"] + flips["regressed_count"],
        n_cases=len(case_ids),
        p_value=mcnemar_pvalue(flips["fixed_count"], flips["regressed_count"]),
    )
    return Comparison(
        score_delta=statistics.fmean(score_deltas),
        score_ci=bootstrap_mean_ci(score_deltas),
        cost_delta=statistics.fmean(cost_deltas),
        cost_ci=bootstrap_mean_ci(cost_deltas),
        token_delta=statistics.fmean(token_deltas),
        token_ci=bootstrap_mean_ci(token_deltas),
        latency_delta=statistics.fmean(latency_deltas),
        latency_ci=bootstrap_mean_ci(latency_deltas),
        pass_significance=pass_significance,
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


@dataclass(frozen=True)
class GatePredicate:
    """Single source of truth for objective semantics.

    Acceptance, holdout validation, sorting, finalist confirmation, and
    recommendation tie-breaking all derive from this one object so that
    behavior cannot drift between call sites.
    """

    objective: OptimizationObjective

    @property
    def mode(self) -> str:
        return self.objective.mode

    def constraint_reason(
        self,
        baseline: PatchSummary,
        candidate: PatchSummary,
    ) -> str | None:
        constraints = self.objective.constraints
        score_delta = candidate.mean_score - baseline.mean_score
        required_score_delta = constraints.min_correctness_delta
        if required_score_delta is None:
            required_score_delta = 0.0 if self.mode == "correctness" else -NON_INFERIORITY_MARGIN
        if score_delta + 1e-9 < required_score_delta:
            return (
                "correctness constraint rejected patch "
                f"(score delta {score_delta:.4f} < required {required_score_delta:.4f})"
            )
        max_cost_ratio = constraints.max_cost_ratio
        if max_cost_ratio is not None and baseline.mean_cost_usd > 0:
            if candidate.mean_cost_usd > baseline.mean_cost_usd * max_cost_ratio:
                return (
                    "cost constraint rejected patch "
                    f"(${candidate.mean_cost_usd:.6f} > {max_cost_ratio:.2f}x baseline)"
                )
        max_latency_ratio = constraints.max_latency_ratio
        if max_latency_ratio is None:
            if self.mode == "correctness":
                max_latency_ratio = DEFAULT_LATENCY_GUARD
            elif self.mode == "cost":
                max_latency_ratio = DEFAULT_COST_MODE_LATENCY_GUARD
        if max_latency_ratio is not None and baseline.median_latency_s > 0:
            if candidate.median_latency_s > baseline.median_latency_s * max_latency_ratio:
                return (
                    "latency constraint rejected patch "
                    f"({candidate.median_latency_s:.3f}s > {max_latency_ratio:.2f}x baseline)"
                )
        return None

    def improvement_reason(
        self,
        reference: PatchSummary,
        candidate: PatchSummary,
    ) -> str | None:
        if self.mode == "correctness":
            if candidate.pass_count > reference.pass_count:
                return None
            if candidate.pass_count < reference.pass_count:
                return "correctness objective rejected pass count regression"
            if candidate.mean_score > reference.mean_score + NON_INFERIORITY_MARGIN:
                return None
            return "correctness objective did not improve pass count or mean score"
        if self.mode == "cost":
            if candidate.mean_score < reference.mean_score - NON_INFERIORITY_MARGIN:
                return "cost objective rejected correctness tradeoff"
            if candidate.mean_cost_usd >= reference.mean_cost_usd:
                return "cost objective did not reduce mean cost"
            return None
        if self.mode == "latency":
            if candidate.mean_score < reference.mean_score - NON_INFERIORITY_MARGIN:
                return "latency objective rejected correctness tradeoff"
            if candidate.median_latency_s >= reference.median_latency_s:
                return "latency objective did not reduce median latency"
            return None
        raise ValueError(f"Unsupported optimization mode: {self.mode}")

    def confidence_reason(self, comparison: Comparison) -> str | None:
        if self.mode == "correctness":
            sig = comparison.pass_significance
            if sig is None:
                return "correctness uncertainty rejected patch (missing paired pass-flip significance)"
            if sig.p_value > SIGNIFICANCE_ALPHA:
                return (
                    "correctness uncertainty rejected patch "
                    f"(fixed {sig.fixed_count}, regressed {sig.regressed_count}, "
                    f"paired pass-flip p-value {sig.p_value:.4f} > alpha {SIGNIFICANCE_ALPHA:.2f})"
                )
            return None
        if self.mode == "cost":
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
        if self.mode == "latency":
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
        raise ValueError(f"Unsupported optimization mode: {self.mode}")

    def dev_gate_reason(
        self,
        *,
        baseline: PatchSummary,
        reference: PatchSummary,
        candidate: PatchSummary,
    ) -> str | None:
        reason = self.constraint_reason(baseline, candidate)
        if reason is not None:
            return reason
        return self.improvement_reason(reference, candidate)

    def confirmation_reason(
        self,
        *,
        baseline: PatchSummary,
        candidate: PatchSummary,
        regressed_case_ids: list[str],
        comparison: Comparison | None = None,
    ) -> str | None:
        reason = self.dev_gate_reason(baseline=baseline, reference=baseline, candidate=candidate)
        if reason is not None:
            return reason
        if regressed_case_ids:
            return (
                f"confirmation observed regressions on {len(regressed_case_ids)} case(s)"
            )
        return None

    def final_gate(
        self,
        baseline: PatchSummary,
        candidate: PatchSummary,
    ) -> FinalGateResult:
        comparison = compare_summaries(baseline, candidate)
        constraint_reason = self.constraint_reason(baseline, candidate)
        if constraint_reason is not None:
            return FinalGateResult(status="failed", reason=constraint_reason, comparison=comparison)
        improvement_reason = self.improvement_reason(baseline, candidate)
        if improvement_reason is not None:
            return FinalGateResult(status="failed", reason=improvement_reason, comparison=comparison)
        confidence_reason = self.confidence_reason(comparison)
        if confidence_reason is not None:
            return FinalGateResult(status="directional", reason=confidence_reason, comparison=comparison)
        return FinalGateResult(status="validated", reason=None, comparison=comparison)

    def sort_key(self, summary: PatchSummary) -> tuple[Any, ...]:
        if self.mode == "correctness":
            return (
                -summary.pass_count,
                -summary.mean_score,
                summary.mean_cost_usd,
                summary.median_latency_s,
                summary.operation_count,
                summary.patch_hash,
            )
        if self.mode == "cost":
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

    def primary_metric(self, summary: PatchSummary) -> float:
        """Higher-is-better view of the primary axis. Used for equivalence bands."""
        if self.mode == "correctness":
            return summary.mean_score
        if self.mode == "cost":
            return -summary.mean_cost_usd
        return -summary.median_latency_s

    def equivalence_margin(self, reference: PatchSummary) -> float:
        """How wide the noise band is around the primary axis, expressed in
        the same units as ``primary_metric``."""
        if self.mode == "correctness":
            one_case_delta = 1.0 / max(reference.case_count, 1)
            return max(SCORE_EQUIVALENCE_FLOOR, one_case_delta)
        if self.mode == "cost":
            return reference.mean_cost_usd * COST_EQUIVALENCE_FRACTION
        return reference.median_latency_s * LATENCY_EQUIVALENCE_FRACTION

    def secondary_sort_key(self, summary: PatchSummary) -> tuple[Any, ...]:
        """Tiebreak ordering applied within an equivalence band on the primary axis."""
        if self.mode == "correctness":
            return (
                summary.mean_cost_usd,
                summary.median_latency_s,
                summary.operation_count,
                -summary.mean_score,
                summary.patch_hash,
            )
        if self.mode == "cost":
            return (
                -summary.mean_score,
                summary.median_latency_s,
                summary.operation_count,
                summary.mean_cost_usd,
                summary.patch_hash,
            )
        return (
            -summary.mean_score,
            summary.mean_cost_usd,
            summary.operation_count,
            summary.median_latency_s,
            summary.patch_hash,
        )

    def select_recommended(
        self,
        candidates: list[PatchSummary],
    ) -> tuple[PatchSummary, dict[str, Any]]:
        """Pick a recommendation from already-validated candidates.

        Best on the primary axis wins; within an equivalence band any
        candidate may be promoted on the secondary tiebreaker. Returns the
        chosen summary plus a structured rationale dict.
        """
        if not candidates:
            raise ValueError("select_recommended requires at least one candidate.")
        ranked = sorted(candidates, key=self.sort_key)
        highest_quality = ranked[0]
        margin = self.equivalence_margin(highest_quality)
        best_primary = self.primary_metric(highest_quality)
        equivalent = [
            summary
            for summary in candidates
            if best_primary - self.primary_metric(summary) <= margin + 1e-9
        ]
        policy = self._recommendation_policy()
        selected = self._select_by_policy(
            highest_quality=highest_quality,
            equivalent=equivalent,
            candidates=candidates,
            policy=policy,
        )
        if selected.patch_hash == highest_quality.patch_hash:
            reason = f"Promoted highest-quality validated patch for {self.mode} objective."
        else:
            reason = (
                f"Promoted `{policy}` validated patch within primary-axis equivalence margin "
                f"for {self.mode} objective (margin {margin:.4f})."
            )
        variants = self._frontier_variants(
            highest_quality=highest_quality,
            equivalent=equivalent,
            candidates=candidates,
        )
        return selected, {
            "recommended_patch_hash": selected.patch_hash,
            "highest_quality_patch_hash": highest_quality.patch_hash,
            "validated_candidate_count": len(candidates),
            "equivalence_margin": margin,
            "recommendation_policy": policy,
            "reason": reason,
            "frontier_variants": variants,
            "recommended_metrics": selected.to_dict(),
            "highest_quality_metrics": highest_quality.to_dict(),
        }

    def _recommendation_policy(self) -> str:
        normalized = [_normalize_policy_token(item) for item in self.objective.tie_breakers]
        if self.mode == "correctness":
            for token in normalized:
                if token in {"highest_correctness", "highest_quality", "maximize_correctness"}:
                    return "highest_correctness"
                if token in {"lowest_cost_within_quality_margin", "lower_cost"}:
                    return "lowest_cost_within_quality_margin"
                if token in {"lowest_latency_within_quality_margin", "lower_latency"}:
                    return "lowest_latency_within_quality_margin"
                if token in {"simplest_within_quality_margin", "smaller_patch", "simpler"}:
                    return "simplest_within_quality_margin"
                if token == "balanced":
                    return "balanced"
            return "lowest_cost_within_quality_margin"
        if self.mode == "cost":
            return "lowest_cost"
        return "lowest_latency"

    def _select_by_policy(
        self,
        *,
        highest_quality: PatchSummary,
        equivalent: list[PatchSummary],
        candidates: list[PatchSummary],
        policy: str,
    ) -> PatchSummary:
        if policy == "highest_correctness":
            return highest_quality
        if policy in {"lowest_cost", "lowest_cost_within_quality_margin"}:
            return min(
                equivalent,
                key=lambda summary: (
                    summary.mean_cost_usd,
                    -summary.mean_score,
                    summary.median_latency_s,
                    summary.operation_count,
                    summary.patch_hash,
                ),
            )
        if policy in {"lowest_latency", "lowest_latency_within_quality_margin"}:
            return min(
                equivalent,
                key=lambda summary: (
                    summary.median_latency_s,
                    -summary.mean_score,
                    summary.mean_cost_usd,
                    summary.operation_count,
                    summary.patch_hash,
                ),
            )
        if policy == "simplest_within_quality_margin":
            return min(
                equivalent,
                key=lambda summary: (
                    summary.operation_count,
                    summary.mean_cost_usd,
                    summary.median_latency_s,
                    -summary.mean_score,
                    summary.patch_hash,
                ),
            )
        if policy == "balanced":
            return sorted(equivalent, key=self.secondary_sort_key)[0]
        return sorted(candidates, key=self.sort_key)[0]

    def _frontier_variants(
        self,
        *,
        highest_quality: PatchSummary,
        equivalent: list[PatchSummary],
        candidates: list[PatchSummary],
    ) -> list[dict[str, Any]]:
        variants: list[tuple[str, PatchSummary]] = [("highest_quality", highest_quality)]
        if self.mode == "correctness":
            variants.extend(
                [
                    (
                        "lowest_cost_within_margin",
                        self._select_by_policy(
                            highest_quality=highest_quality,
                            equivalent=equivalent,
                            candidates=candidates,
                            policy="lowest_cost_within_quality_margin",
                        ),
                    ),
                    (
                        "lowest_latency_within_margin",
                        self._select_by_policy(
                            highest_quality=highest_quality,
                            equivalent=equivalent,
                            candidates=candidates,
                            policy="lowest_latency_within_quality_margin",
                        ),
                    ),
                    (
                        "simplest_within_margin",
                        self._select_by_policy(
                            highest_quality=highest_quality,
                            equivalent=equivalent,
                            candidates=candidates,
                            policy="simplest_within_quality_margin",
                        ),
                    ),
                ]
            )
        seen: set[tuple[str, str]] = set()
        rows: list[dict[str, Any]] = []
        for role, summary in variants:
            key = (role, summary.patch_hash)
            if key in seen:
                continue
            seen.add(key)
            rows.append(_frontier_variant_row(role, summary))
        return rows


def _predicate(objective: OptimizationObjective | None) -> GatePredicate:
    return GatePredicate(objective or OptimizationObjective())


def patch_satisfies_constraints(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> bool:
    return _predicate(objective).constraint_reason(baseline, patch_summary) is None


def constraint_rejection_reason(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> str | None:
    return _predicate(objective).constraint_reason(baseline, patch_summary)


def objective_improved(
    reference: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> bool:
    return _predicate(objective).improvement_reason(reference, patch_summary) is None


def objective_rejection_reason(
    reference: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> str | None:
    return _predicate(objective).improvement_reason(reference, patch_summary)


def patch_rejection_reason(
    *,
    baseline: PatchSummary,
    reference: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective,
) -> str | None:
    predicate = _predicate(objective)
    return predicate.dev_gate_reason(
        baseline=baseline,
        reference=reference,
        candidate=patch_summary,
    )


def objective_sort_key(summary: PatchSummary, objective: OptimizationObjective) -> tuple[Any, ...]:
    return _predicate(objective).sort_key(summary)


def final_gate(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> tuple[bool, Comparison]:
    gate = _predicate(objective).final_gate(baseline, patch_summary)
    return gate.validated, gate.comparison


def final_gate_rejection_reason(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> tuple[str | None, Comparison]:
    gate = _predicate(objective).final_gate(baseline, patch_summary)
    return (None if gate.validated else gate.reason), gate.comparison


def final_gate_status(
    baseline: PatchSummary,
    patch_summary: PatchSummary,
    objective: OptimizationObjective | None = None,
) -> FinalGateResult:
    return _predicate(objective).final_gate(baseline, patch_summary)


def uncertainty_rejection_reason(
    comparison: Comparison,
    objective: OptimizationObjective,
) -> str | None:
    return _predicate(objective).confidence_reason(comparison)


def select_recommended_patch(
    candidates: list[PatchSummary],
    objective: OptimizationObjective,
) -> tuple[PatchSummary, dict[str, Any]]:
    return _predicate(objective).select_recommended(candidates)


def _normalize_policy_token(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _frontier_variant_row(role: str, summary: PatchSummary) -> dict[str, Any]:
    return {
        "role": role,
        "patch_hash": summary.patch_hash,
        "pass_count": summary.pass_count,
        "case_count": summary.case_count,
        "mean_score": summary.mean_score,
        "mean_cost_usd": summary.mean_cost_usd,
        "mean_total_tokens": summary.mean_total_tokens,
        "median_latency_s": summary.median_latency_s,
        "operation_count": summary.operation_count,
        "operations": [
            {
                "op": operation.op,
                "target": operation.target,
                "value_summary": _operation_value_summary(operation.value),
            }
            for operation in summary.patch.operations
        ],
    }


def _operation_value_summary(value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(item, dict) for item in value):
        source_ids = [item.get("source_case_id") for item in value if item.get("source_case_id")]
        if source_ids:
            return source_ids
    if isinstance(value, str):
        return value if len(value) <= 160 else value[:157] + "..."
    return value


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
