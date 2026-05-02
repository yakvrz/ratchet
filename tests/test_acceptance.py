from __future__ import annotations

import unittest

from ratchet.objectives import (
    GatePredicate,
    compare_summaries,
    final_gate,
    final_gate_status,
    objective_rejection_reason,
    select_recommended_candidate,
)
from ratchet.reporting import build_outcome_analysis
from ratchet.results import CandidateSummary, CaseEvaluation
from ratchet.types import (
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    OptimizationConstraints,
    OptimizationObjective,
    RunRecord,
)


def make_summary(
    candidate_id_value: str,
    scores: list[float],
    costs: list[float],
    tokens: list[int],
    latencies: list[float],
) -> CandidateSummary:
    evaluations = []
    for index, score in enumerate(scores, start=1):
        case = EvalCase(id=f"case-{index}", split="holdout", input=f"case {index}")
        evaluations.append(
            CaseEvaluation(
                case=case,
                record=RunRecord(
                    output="ok",
                    metrics=OperationalMetrics(
                        latency_s=latencies[index - 1],
                        input_tokens=tokens[index - 1] // 2,
                        output_tokens=tokens[index - 1] // 2,
                        total_tokens=tokens[index - 1],
                        cost_usd=costs[index - 1],
                    ),
                    diagnostics=DiagnosticTrace(),
                ),
                grade=GradeResult(score=score, passed=score == 1.0, labels=[]),
            )
        )
    return CandidateSummary(
        candidate_id=candidate_id_value,
        candidate=None,
        split="holdout",
        evaluations=evaluations,
    )


def make_repeated_summary(
    candidate_id_value: str,
    case_sample_scores: list[list[float]],
) -> CandidateSummary:
    evaluations = []
    for case_index, sample_scores in enumerate(case_sample_scores, start=1):
        case = EvalCase(id=f"case-{case_index}", split="holdout", input=f"case {case_index}")
        for sample_index, score in enumerate(sample_scores):
            evaluations.append(
                CaseEvaluation(
                    case=case,
                    record=RunRecord(
                        output="ok",
                        metrics=OperationalMetrics(
                            latency_s=1.0,
                            input_tokens=50,
                            output_tokens=50,
                            total_tokens=100,
                            cost_usd=0.002,
                        ),
                        diagnostics=DiagnosticTrace(),
                    ),
                    grade=GradeResult(score=score, passed=score == 1.0, labels=[]),
                    sample_index=sample_index,
                )
            )
    return CandidateSummary(
        candidate_id=candidate_id_value,
        candidate=None,
        split="holdout",
        evaluations=evaluations,
    )


class AcceptanceGateTests(unittest.TestCase):
    def test_equal_quality_lower_cost_and_tokens_passes_final_gate(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        candidate_summary = make_summary("candidate_summary", [1.0, 1.0, 1.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.05, 1.05, 1.05])
        passed, _ = final_gate(baseline, candidate_summary, OptimizationObjective(mode="cost"))
        self.assertTrue(passed)

    def test_cost_mode_default_allows_confident_noninferior_correctness(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        candidate_summary = make_summary("candidate_summary", [0.995, 0.995, 0.995], [0.002, 0.002, 0.002], [100, 100, 100], [1.0, 1.0, 1.0])
        passed, comparison = final_gate(baseline, candidate_summary, OptimizationObjective(mode="cost"))
        self.assertGreaterEqual(comparison.score_ci[0], -0.01)
        self.assertLess(comparison.cost_ci[1], 0.0)
        self.assertTrue(passed)

    def test_explicit_min_correctness_delta_keeps_strict_cost_guard(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        candidate_summary = make_summary("candidate_summary", [0.995, 0.995, 0.995], [0.002, 0.002, 0.002], [100, 100, 100], [1.0, 1.0, 1.0])
        objective = OptimizationObjective(
            mode="cost",
            constraints=OptimizationConstraints(min_correctness_delta=0.0),
        )
        passed, _ = final_gate(baseline, candidate_summary, objective)
        self.assertFalse(passed)

    def test_equal_quality_slower_than_guard_fails(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        candidate_summary = make_summary("candidate_summary", [1.0, 1.0, 1.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.3, 1.3, 1.3])
        passed, _ = final_gate(baseline, candidate_summary, OptimizationObjective(mode="cost"))
        self.assertFalse(passed)

    def test_higher_quality_patch_can_pass_final_gate_without_efficiency_gain(self) -> None:
        baseline = make_summary("baseline", [0.0, 0.0, 0.0, 0.0], [0.002] * 4, [100] * 4, [1.0] * 4)
        candidate_summary = make_summary("candidate_summary", [1.0, 1.0, 1.0, 1.0], [0.004] * 4, [300] * 4, [1.0] * 4)
        comparison = compare_summaries(baseline, candidate_summary)
        self.assertGreater(comparison.score_ci[0], 0.0)
        passed, _ = final_gate(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertTrue(passed)

    def test_correctness_mode_does_not_apply_implicit_cost_guard(self) -> None:
        baseline = make_summary("baseline", [0.0] * 12, [0.001] * 12, [100] * 12, [1.0] * 12)
        candidate_summary = make_summary("candidate_summary", [1.0] * 12, [0.050] * 12, [300] * 12, [1.0] * 12)
        gate = final_gate_status(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(gate.status, "validated")

    def test_explicit_correctness_cost_ratio_remains_a_hard_constraint(self) -> None:
        baseline = make_summary("baseline", [0.0] * 12, [0.001] * 12, [100] * 12, [1.0] * 12)
        candidate_summary = make_summary("candidate_summary", [1.0] * 12, [0.050] * 12, [300] * 12, [1.0] * 12)
        objective = OptimizationObjective(
            mode="correctness",
            constraints=OptimizationConstraints(max_cost_ratio=3.0),
        )
        gate = final_gate_status(baseline, candidate_summary, objective)
        self.assertEqual(gate.status, "failed")
        self.assertIn("cost constraint", gate.reason or "")

    def test_correctness_dev_acceptance_allows_material_score_gain_despite_pass_count_regression(self) -> None:
        baseline = make_summary("baseline", [1.0, 0.0, 0.0], [0.002] * 3, [100] * 3, [1.0] * 3)
        candidate_summary = make_summary("candidate_summary", [0.9, 0.9, 0.9], [0.002] * 3, [100] * 3, [1.0] * 3)
        reason = objective_rejection_reason(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertIsNone(reason)

    def test_correctness_dev_acceptance_rejects_pass_count_regression_without_score_gain(self) -> None:
        baseline = make_summary("baseline", [1.0, 0.0, 0.0], [0.002] * 3, [100] * 3, [1.0] * 3)
        candidate_summary = make_summary("candidate_summary", [0.9, 0.05, 0.05], [0.002] * 3, [100] * 3, [1.0] * 3)
        reason = objective_rejection_reason(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(reason, "correctness objective rejected pass count regression without material score gain")

    def test_correctness_dev_acceptance_allows_equal_pass_count_score_gain(self) -> None:
        baseline = make_summary("baseline", [1.0, 0.2, 0.2], [0.002] * 3, [100] * 3, [1.0] * 3)
        candidate_summary = make_summary("candidate_summary", [1.0, 0.5, 0.5], [0.002] * 3, [100] * 3, [1.0] * 3)
        reason = objective_rejection_reason(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertIsNone(reason)

    def test_correctness_dev_acceptance_allows_higher_pass_count(self) -> None:
        baseline = make_summary("baseline", [0.0, 0.0], [0.002] * 2, [100] * 2, [1.0] * 2)
        candidate_summary = make_summary("candidate_summary", [1.0, 0.0], [0.002] * 2, [100] * 2, [1.0] * 2)
        reason = objective_rejection_reason(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertIsNone(reason)

    def test_small_holdout_gain_passes_deterministic_final_gate(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 0.0, 0.0], [0.002] * 4, [100] * 4, [1.0] * 4)
        candidate_summary = make_summary("candidate_summary", [1.0, 1.0, 1.0, 0.0], [0.004] * 4, [300] * 4, [1.0] * 4)
        comparison = compare_summaries(baseline, candidate_summary)
        self.assertEqual(comparison.score_ci[0], 0.0)
        passed, _ = final_gate(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertTrue(passed)

    def test_underpowered_holdout_gain_is_validated_by_objective_gate(self) -> None:
        baseline = make_summary("baseline", [1.0] * 18 + [0.0] * 6, [0.002] * 24, [100] * 24, [1.0] * 24)
        candidate_summary = make_summary("candidate_summary", [1.0] * 20 + [0.0] * 4, [0.002] * 24, [100] * 24, [1.0] * 24)
        gate = final_gate_status(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(gate.status, "validated")
        self.assertTrue(gate.validated)
        self.assertIsNone(gate.reason)

    def test_score_only_holdout_gain_is_validated_by_objective_gate(self) -> None:
        baseline = make_summary("baseline", [0.4] * 8, [0.002] * 8, [100] * 8, [1.0] * 8)
        candidate_summary = make_summary("candidate_summary", [0.8] * 8, [0.002] * 8, [100] * 8, [1.0] * 8)
        comparison = compare_summaries(baseline, candidate_summary)
        self.assertGreater(comparison.score_ci[0], 0.0)
        gate = final_gate_status(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(gate.status, "validated")
        self.assertTrue(gate.validated)
        self.assertEqual(gate.comparison.pass_significance.fixed_count, 0)
        self.assertEqual(gate.comparison.pass_significance.regressed_count, 0)

    def test_score_gain_with_pass_regression_is_validated_by_objective_gate(self) -> None:
        baseline = make_summary("baseline", [1.0, 0.0, 0.0, 0.0], [0.002] * 4, [100] * 4, [1.0] * 4)
        candidate_summary = make_summary("candidate_summary", [0.9, 0.9, 0.9, 0.9], [0.002] * 4, [100] * 4, [1.0] * 4)
        gate = final_gate_status(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(gate.status, "validated")
        self.assertIsNone(gate.reason)

    def test_repeated_sample_pass_flip_significance_uses_case_majorities(self) -> None:
        baseline = make_repeated_summary("baseline", [[0.0, 0.0, 1.0]] * 4)
        candidate_summary = make_repeated_summary("candidate_summary", [[1.0, 1.0, 0.0]] * 4)
        gate = final_gate_status(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(gate.status, "validated")
        self.assertEqual(gate.comparison.pass_significance.fixed_count, 4)
        self.assertEqual(gate.comparison.pass_significance.regressed_count, 0)

    def test_ci_positive_holdout_gain_is_validated(self) -> None:
        baseline = make_summary("baseline", [0.0] * 24, [0.002] * 24, [100] * 24, [1.0] * 24)
        candidate_summary = make_summary("candidate_summary", [1.0] * 24, [0.002] * 24, [100] * 24, [1.0] * 24)
        gate = final_gate_status(baseline, candidate_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(gate.status, "validated")
        self.assertTrue(gate.validated)

    def test_recommendation_prefers_cheaper_equivalent_validated_frontier(self) -> None:
        highest = make_summary("highest", [1.0] * 24, [0.010] * 24, [900] * 24, [1.0] * 24)
        cheaper = make_summary("cheaper", [1.0] * 23 + [0.0], [0.002] * 24, [120] * 24, [1.0] * 24)
        selected, recommendation = select_recommended_candidate(
            [highest, cheaper],
            OptimizationObjective(mode="correctness"),
        )
        self.assertEqual(selected.candidate_id, "cheaper")
        self.assertEqual(recommendation["highest_quality_candidate_id"], "highest")
        self.assertEqual(recommendation["validated_candidate_count"], 2)
        self.assertGreater(recommendation["equivalence_margin"], 0.0)
        self.assertEqual(recommendation["recommendation_policy"], "lowest_cost_within_quality_margin")
        self.assertTrue(recommendation["frontier_variants"])

    def test_recommendation_highest_correctness_policy_picks_best_score(self) -> None:
        highest = make_summary("highest", [1.0] * 24, [0.010] * 24, [900] * 24, [1.0] * 24)
        cheaper = make_summary("cheaper", [1.0] * 23 + [0.0], [0.002] * 24, [120] * 24, [1.0] * 24)
        selected, recommendation = select_recommended_candidate(
            [highest, cheaper],
            OptimizationObjective(mode="correctness", tie_breakers=["highest_correctness"]),
        )
        self.assertEqual(selected.candidate_id, "highest")
        self.assertEqual(recommendation["recommendation_policy"], "highest_correctness")
        variants = {item["role"]: item for item in recommendation["frontier_variants"]}
        self.assertEqual(variants["highest_quality"]["candidate_id"], "highest")
        self.assertEqual(variants["lowest_cost_within_margin"]["candidate_id"], "cheaper")

    def test_recommendation_keeps_highest_quality_when_no_equivalent_alternative(self) -> None:
        highest = make_summary("highest", [1.0] * 24, [0.010] * 24, [900] * 24, [1.0] * 24)
        far_cheaper_far_worse = make_summary(
            "weak", [0.5] * 24, [0.001] * 24, [100] * 24, [1.0] * 24
        )
        selected, recommendation = select_recommended_candidate(
            [highest, far_cheaper_far_worse],
            OptimizationObjective(mode="correctness"),
        )
        self.assertEqual(selected.candidate_id, "highest")
        self.assertEqual(recommendation["highest_quality_candidate_id"], "highest")

    def test_recommendation_cost_mode_picks_cheapest_validated_patch(self) -> None:
        cheap = make_summary("cheap", [1.0] * 12, [0.001] * 12, [100] * 12, [1.0] * 12)
        cheaper = make_summary("cheaper", [1.0] * 12, [0.0008] * 12, [100] * 12, [1.0] * 12)
        selected, _ = select_recommended_candidate(
            [cheap, cheaper],
            OptimizationObjective(mode="cost"),
        )
        self.assertEqual(selected.candidate_id, "cheaper")

    def test_recommendation_latency_mode_picks_fastest_validated_patch(self) -> None:
        fast = make_summary("fast", [1.0] * 12, [0.002] * 12, [100] * 12, [0.8] * 12)
        faster = make_summary("faster", [1.0] * 12, [0.002] * 12, [100] * 12, [0.6] * 12)
        selected, _ = select_recommended_candidate(
            [fast, faster],
            OptimizationObjective(mode="latency"),
        )
        self.assertEqual(selected.candidate_id, "faster")

    def test_predicate_dev_gate_short_circuits_on_constraint_violation(self) -> None:
        baseline = make_summary("baseline", [1.0] * 6, [0.002] * 6, [100] * 6, [1.0] * 6)
        bad = make_summary("bad", [1.0] * 6, [0.020] * 6, [100] * 6, [1.0] * 6)
        objective = OptimizationObjective(
            mode="correctness",
            constraints=OptimizationConstraints(max_cost_ratio=2.0),
        )
        predicate = GatePredicate(objective)
        reason = predicate.dev_gate_reason(baseline=baseline, reference=baseline, candidate=bad)
        self.assertIsNotNone(reason)
        self.assertIn("cost constraint", reason or "")

    def test_predicate_confirmation_flags_regressions(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0, 1.0], [0.002] * 4, [100] * 4, [1.0] * 4)
        candidate = make_summary("candidate", [1.0, 1.0, 1.0, 0.0], [0.002] * 4, [100] * 4, [1.0] * 4)
        predicate = GatePredicate(OptimizationObjective(mode="correctness"))
        reason = predicate.confirmation_reason(
            baseline=baseline,
            candidate=candidate,
            regressed_case_ids=["case-4"],
        )
        self.assertIsNotNone(reason)

    def test_predicate_confirmation_does_not_require_tiny_subset_significance(self) -> None:
        baseline = make_summary("baseline", [0.0, 0.0], [0.002] * 2, [100] * 2, [1.0] * 2)
        candidate = make_summary("candidate", [1.0, 1.0], [0.002] * 2, [100] * 2, [1.0] * 2)
        predicate = GatePredicate(OptimizationObjective(mode="correctness"))
        reason = predicate.confirmation_reason(
            baseline=baseline,
            candidate=candidate,
            regressed_case_ids=[],
        )
        self.assertIsNone(reason)

    def test_latency_mode_requires_confident_latency_gain(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.002] * 3, [100] * 3, [1.0, 1.0, 1.0])
        candidate_summary = make_summary("candidate_summary", [1.0, 1.0, 1.0], [0.002] * 3, [100] * 3, [0.7, 0.7, 0.7])
        passed, comparison = final_gate(baseline, candidate_summary, OptimizationObjective(mode="latency"))
        self.assertLess(comparison.latency_ci[1], 0.0)
        self.assertTrue(passed)

    def test_repeated_sample_no_failures_are_reported_by_case_count(self) -> None:
        evaluations = []
        for case_index in range(2):
            case = EvalCase(id=f"case-{case_index}", split="dev", input="ok")
            for sample_index in range(3):
                evaluations.append(
                    CaseEvaluation(
                        case=case,
                        record=RunRecord(
                            output="ok",
                            metrics=OperationalMetrics(
                                latency_s=1.0,
                                input_tokens=10,
                                output_tokens=5,
                                total_tokens=15,
                                cost_usd=0.001,
                            ),
                            diagnostics=DiagnosticTrace(),
                        ),
                        grade=GradeResult(score=1.0, passed=True),
                        sample_index=sample_index,
                    )
                )
        baseline = CandidateSummary(
            candidate_id="baseline",
            candidate=None,
            split="dev",
            evaluations=evaluations,
        )
        outcome = build_outcome_analysis(
            objective=OptimizationObjective(mode="correctness"),
            promoted=False,
            baseline_dev=baseline,
            accepted_dev_candidates=[],
            holdout_candidates=[],
            events=[],
        )
        self.assertEqual(outcome["status"], "no_failures")

    def test_failure_labels_ignore_sample_failures_when_case_majority_passes(self) -> None:
        evaluations = []
        case = EvalCase(id="case-split", split="dev", input="ok")
        for sample_index, passed in enumerate([True, True, False]):
            evaluations.append(
                CaseEvaluation(
                    case=case,
                    record=RunRecord(
                        output="ok" if passed else "wrong",
                        metrics=OperationalMetrics(
                            latency_s=1.0,
                            input_tokens=10,
                            output_tokens=5,
                            total_tokens=15,
                            cost_usd=0.001,
                        ),
                        diagnostics=DiagnosticTrace(),
                    ),
                    grade=GradeResult(
                        score=1.0 if passed else 0.0,
                        passed=passed,
                        labels=[] if passed else ["sample_failure"],
                    ),
                    sample_index=sample_index,
                )
            )
        summary = CandidateSummary(
            candidate_id="split",
            candidate=None,
            split="dev",
            evaluations=evaluations,
        )

        self.assertEqual(summary.pass_count, 1)
        self.assertEqual(summary.failure_labels, {})
        self.assertEqual(summary.failed_examples(), [])


if __name__ == "__main__":
    unittest.main()
