from __future__ import annotations

import unittest

from ratchet.objectives import compare_summaries, final_gate, objective_rejection_reason
from ratchet.reporting import build_outcome_analysis
from ratchet.results import PatchSummary, CaseEvaluation
from ratchet.types import (
    AgentPatch,
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    OptimizationConstraints,
    OptimizationObjective,
    RunRecord,
)


def make_summary(
    patch_hash_value: str,
    scores: list[float],
    costs: list[float],
    tokens: list[int],
    latencies: list[float],
) -> PatchSummary:
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
    return PatchSummary(
        patch_hash=patch_hash_value,
        patch=AgentPatch(metadata={"name": patch_hash_value}),
        split="holdout",
        evaluations=evaluations,
    )


class AcceptanceGateTests(unittest.TestCase):
    def test_equal_quality_lower_cost_and_tokens_passes_final_gate(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        patch_summary = make_summary("patch_summary", [1.0, 1.0, 1.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.05, 1.05, 1.05])
        passed, _ = final_gate(baseline, patch_summary, OptimizationObjective(mode="cost"))
        self.assertTrue(passed)

    def test_cost_mode_default_allows_confident_noninferior_correctness(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        patch_summary = make_summary("patch_summary", [0.995, 0.995, 0.995], [0.002, 0.002, 0.002], [100, 100, 100], [1.0, 1.0, 1.0])
        passed, comparison = final_gate(baseline, patch_summary, OptimizationObjective(mode="cost"))
        self.assertGreaterEqual(comparison.score_ci[0], -0.01)
        self.assertLess(comparison.cost_ci[1], 0.0)
        self.assertTrue(passed)

    def test_explicit_min_correctness_delta_keeps_strict_cost_guard(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        patch_summary = make_summary("patch_summary", [0.995, 0.995, 0.995], [0.002, 0.002, 0.002], [100, 100, 100], [1.0, 1.0, 1.0])
        objective = OptimizationObjective(
            mode="cost",
            constraints=OptimizationConstraints(min_correctness_delta=0.0),
        )
        passed, _ = final_gate(baseline, patch_summary, objective)
        self.assertFalse(passed)

    def test_equal_quality_slower_than_guard_fails(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        patch_summary = make_summary("patch_summary", [1.0, 1.0, 1.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.3, 1.3, 1.3])
        passed, _ = final_gate(baseline, patch_summary, OptimizationObjective(mode="cost"))
        self.assertFalse(passed)

    def test_higher_quality_patch_can_pass_final_gate_without_efficiency_gain(self) -> None:
        baseline = make_summary("baseline", [0.0, 0.0, 0.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.0, 1.0, 1.0])
        patch_summary = make_summary("patch_summary", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [300, 300, 300], [1.0, 1.0, 1.0])
        comparison = compare_summaries(baseline, patch_summary)
        self.assertGreater(comparison.score_ci[0], 0.0)
        passed, _ = final_gate(baseline, patch_summary, OptimizationObjective(mode="correctness"))
        self.assertTrue(passed)

    def test_correctness_dev_acceptance_rejects_pass_count_regression_despite_score_gain(self) -> None:
        baseline = make_summary("baseline", [1.0, 0.0, 0.0], [0.002] * 3, [100] * 3, [1.0] * 3)
        patch_summary = make_summary("patch_summary", [0.9, 0.9, 0.9], [0.002] * 3, [100] * 3, [1.0] * 3)
        reason = objective_rejection_reason(baseline, patch_summary, OptimizationObjective(mode="correctness"))
        self.assertEqual(reason, "correctness objective rejected pass count regression")

    def test_correctness_dev_acceptance_allows_equal_pass_count_score_gain(self) -> None:
        baseline = make_summary("baseline", [1.0, 0.2, 0.2], [0.002] * 3, [100] * 3, [1.0] * 3)
        patch_summary = make_summary("patch_summary", [1.0, 0.5, 0.5], [0.002] * 3, [100] * 3, [1.0] * 3)
        reason = objective_rejection_reason(baseline, patch_summary, OptimizationObjective(mode="correctness"))
        self.assertIsNone(reason)

    def test_correctness_dev_acceptance_allows_higher_pass_count(self) -> None:
        baseline = make_summary("baseline", [0.0, 0.0], [0.002] * 2, [100] * 2, [1.0] * 2)
        patch_summary = make_summary("patch_summary", [1.0, 0.0], [0.002] * 2, [100] * 2, [1.0] * 2)
        reason = objective_rejection_reason(baseline, patch_summary, OptimizationObjective(mode="correctness"))
        self.assertIsNone(reason)

    def test_small_uncertain_holdout_gain_fails_final_gate(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 0.0, 0.0], [0.002] * 4, [100] * 4, [1.0] * 4)
        patch_summary = make_summary("patch_summary", [1.0, 1.0, 1.0, 0.0], [0.004] * 4, [300] * 4, [1.0] * 4)
        comparison = compare_summaries(baseline, patch_summary)
        self.assertEqual(comparison.score_ci[0], 0.0)
        passed, _ = final_gate(baseline, patch_summary, OptimizationObjective(mode="correctness"))
        self.assertFalse(passed)

    def test_latency_mode_requires_confident_latency_gain(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.002] * 3, [100] * 3, [1.0, 1.0, 1.0])
        patch_summary = make_summary("patch_summary", [1.0, 1.0, 1.0], [0.002] * 3, [100] * 3, [0.7, 0.7, 0.7])
        passed, comparison = final_gate(baseline, patch_summary, OptimizationObjective(mode="latency"))
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
        baseline = PatchSummary(
            patch_hash="baseline",
            patch=AgentPatch.empty(),
            split="dev",
            evaluations=evaluations,
        )
        outcome = build_outcome_analysis(
            objective=OptimizationObjective(mode="correctness"),
            promoted=False,
            baseline_dev=baseline,
            accepted_dev_patches=[],
            holdout_patches=[],
            decision_log=[],
        )
        self.assertEqual(outcome["status"], "no_failures")


if __name__ == "__main__":
    unittest.main()
