from __future__ import annotations

import unittest

from ratchet.evidence_ledger import build_evidence_summary, confirmation_stability_result
from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.types import DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


def summary(
    candidate_id: str,
    scores: list[float],
    *,
    invalid_indices: set[int] | None = None,
    finish_reason: str = "stop",
) -> CandidateSummary:
    invalid_indices = invalid_indices or set()
    evaluations: list[CaseEvaluation] = []
    for index, score in enumerate(scores, start=1):
        invalid = index in invalid_indices
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(id=f"case-{index}", split="dev", input=f"input {index}", expected=f"expected {index}"),
                record=RunRecord(
                    output={"invalid_output": "bad"} if invalid else {"answer": "ok"},
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=20,
                        output_tokens=5,
                        total_tokens=25,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(
                        metadata={
                            "finish_reason": finish_reason,
                            "invalid_output": invalid,
                        }
                    ),
                ),
                grade=GradeResult(
                    score=score,
                    passed=score >= 1.0,
                    labels=["invalid_output"] if invalid else ([] if score >= 1.0 else ["failed"]),
                ),
            )
        )
    return CandidateSummary(
        candidate_id=candidate_id,
        candidate=None,
        split="dev",
        evaluations=evaluations,
    )


class EvidenceLedgerTests(unittest.TestCase):
    def test_low_sample_one_case_gain_is_low_confidence(self) -> None:
        reference = summary("baseline", [1.0, 0.0, 1.0, 1.0])
        candidate = summary("candidate", [1.0, 1.0, 1.0, 1.0])

        evidence = build_evidence_summary(
            candidate_id="candidate",
            stage="small_dev",
            reference=reference,
            baseline=reference,
            candidate=candidate,
            mechanism_class="semantic_boundary_rewrite",
            affordance_ids=["prompt.semantic"],
            comparison_group="group",
            candidate_role="atomic",
            rejection_reason=None,
            constraint_warning=None,
        )

        self.assertEqual(evidence.confidence_tier, "low")
        self.assertEqual(evidence.pass_gain, 1)

    def test_runtime_invalid_output_fix_requires_repeat_evidence(self) -> None:
        reference = summary("baseline", [0.0, 1.0], invalid_indices={1})
        candidate = summary("candidate", [1.0, 1.0])

        evidence = build_evidence_summary(
            candidate_id="candidate",
            stage="small_dev",
            reference=reference,
            baseline=reference,
            candidate=candidate,
            mechanism_class="runtime_defect_fix",
            affordance_ids=["runtime.output_cap"],
            comparison_group="runtime",
            candidate_role="atomic",
            rejection_reason=None,
            constraint_warning=None,
        )

        self.assertIn("runtime_repeat_required", evidence.baseline_instability_flags)
        self.assertEqual(evidence.confidence_tier, "unstable")

    def test_moving_baseline_confirmation_becomes_unstable(self) -> None:
        original_reference = summary("baseline", [0.0, 1.0], invalid_indices={1})
        original_candidate = summary("candidate", [1.0, 1.0])
        repeated_reference = summary("baseline-repeat", [1.0, 1.0])
        repeated_candidate = summary("candidate-repeat", [1.0, 1.0])

        result = confirmation_stability_result(
            reference=original_reference,
            candidate=original_candidate,
            repeated_reference=repeated_reference,
            repeated_candidate=repeated_candidate,
            objective=OptimizationObjective(mode="correctness"),
        )

        self.assertEqual(result["status"], "runtime_instability")
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
