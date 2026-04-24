from __future__ import annotations

import unittest

from ratchet.optimizer import CandidateSummary, CaseEvaluation, compare_summaries, final_gate
from ratchet.types import DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, RunRecord


def make_summary(
    candidate_hash: str,
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
        candidate_hash=candidate_hash,
        candidate={"candidate": candidate_hash},
        split="holdout",
        evaluations=evaluations,
    )


class AcceptanceGateTests(unittest.TestCase):
    def test_equal_quality_lower_cost_and_tokens_passes_final_gate(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        candidate = make_summary("candidate", [1.0, 1.0, 1.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.05, 1.05, 1.05])
        passed, _ = final_gate(baseline, candidate)
        self.assertTrue(passed)

    def test_equal_quality_slower_than_guard_fails(self) -> None:
        baseline = make_summary("baseline", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [200, 200, 200], [1.0, 1.0, 1.0])
        candidate = make_summary("candidate", [1.0, 1.0, 1.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.3, 1.3, 1.3])
        passed, _ = final_gate(baseline, candidate)
        self.assertFalse(passed)

    def test_higher_quality_but_less_efficient_candidate_improves_dev_quality_not_final_gate(self) -> None:
        baseline = make_summary("baseline", [0.0, 0.0, 0.0], [0.002, 0.002, 0.002], [100, 100, 100], [1.0, 1.0, 1.0])
        candidate = make_summary("candidate", [1.0, 1.0, 1.0], [0.004, 0.004, 0.004], [300, 300, 300], [1.0, 1.0, 1.0])
        comparison = compare_summaries(baseline, candidate)
        self.assertGreater(comparison.score_ci[0], 0.0)
        passed, _ = final_gate(baseline, candidate)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
