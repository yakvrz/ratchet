from __future__ import annotations

import unittest

from ratchet.evidence import build_behavior_diagnostics, build_proposal_example_bank
from ratchet.experiments import build_task_theory
from ratchet.results import CaseEvaluation, PatchSummary
from ratchet.types import AgentPatch, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


def _evaluation(
    *,
    case_id: str,
    expected: str,
    actual: str,
    passed: bool,
) -> CaseEvaluation:
    return CaseEvaluation(
        case=EvalCase(
            id=case_id,
            split="dev",
            input=f"message for {expected}",
            expected={"label": expected},
            metadata={"category": expected},
        ),
        record=RunRecord(
            output={"label": actual},
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.001,
            ),
        ),
        grade=GradeResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            labels=[] if passed else ["wrong_label", f"expected:{expected}", f"actual:{actual}"],
        ),
    )


class EvidenceTests(unittest.TestCase):
    def test_proposal_example_bank_balances_train_labels(self) -> None:
        cases = tuple(
            EvalCase(
                id=f"train-{label}-{index}",
                split="train",
                input=f"{label} example {index}",
                expected={"label": label},
                metadata={"category": label},
            )
            for label in ("alpha", "beta")
            for index in range(3)
        )

        bank = build_proposal_example_bank(cases, limit=4)

        self.assertEqual(bank.label_field, "label")
        self.assertEqual(bank.label_counts, {"alpha": 2, "beta": 2})
        self.assertEqual(len(bank.examples), 4)

    def test_behavior_diagnostics_reports_confusions_and_weak_labels(self) -> None:
        summary = PatchSummary(
            patch_hash="baseline",
            patch=AgentPatch.empty(),
            split="dev",
            evaluations=[
                _evaluation(case_id="dev-1", expected="alpha", actual="alpha", passed=True),
                _evaluation(case_id="dev-2", expected="beta", actual="alpha", passed=False),
                _evaluation(case_id="dev-3", expected="beta", actual="alpha", passed=False),
            ],
        )

        diagnostics = build_behavior_diagnostics(summary)

        self.assertIn("beta", diagnostics["weak_labels"])
        self.assertEqual(
            diagnostics["confusions"][0],
            {"expected": "beta", "actual": "alpha", "count": 2, "case_ids": ["dev-2", "dev-3"]},
        )

    def test_task_theory_exposes_experiment_opportunities_and_example_sources(self) -> None:
        summary = PatchSummary(
            patch_hash="baseline",
            patch=AgentPatch.empty(),
            split="dev",
            evaluations=[
                _evaluation(case_id="dev-1", expected="alpha", actual="alpha", passed=True),
                _evaluation(case_id="dev-2", expected="beta", actual="alpha", passed=False),
                _evaluation(case_id="dev-3", expected="beta", actual="alpha", passed=False),
            ],
        )
        train_cases = (
            EvalCase(id="train-alpha-1", split="train", input="alpha sample", expected={"label": "alpha"}),
            EvalCase(id="train-beta-1", split="train", input="beta sample", expected={"label": "beta"}),
            EvalCase(id="train-beta-2", split="train", input="beta sample 2", expected={"label": "beta"}),
        )

        theory = build_task_theory(
            summary=summary,
            diagnoses=[],
            objective=OptimizationObjective(),
            proposal_example_bank=build_proposal_example_bank(train_cases),
        )

        self.assertEqual(theory.bottleneck_class, "semantic_boundary_confusion")
        self.assertEqual(
            theory.example_coverage["target_label_source_case_ids"],
            {"alpha": ["train-alpha-1"], "beta": ["train-beta-1", "train-beta-2"]},
        )
        opportunity = theory.experiment_opportunities[0]
        self.assertEqual(opportunity["mechanism_class"], "semantic_boundary_rewrite")
        self.assertIn("confusion:beta->alpha", opportunity["target_slices"])
        self.assertEqual(
            opportunity["source_case_ids_by_label"],
            {"beta": ["train-beta-1", "train-beta-2"], "alpha": ["train-alpha-1"]},
        )


if __name__ == "__main__":
    unittest.main()
