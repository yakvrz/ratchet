from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ratchet.evidence_ledger import EvidenceLedger
from ratchet.optimizer import (
    CandidateEvaluationState,
    RatchetOptimizer,
    _candidate_batch_concurrency_limit,
    _eligible_for_full_dev_from_small_signal,
    _finalize_candidate_state,
)
from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.candidates import CandidateProposal, CandidateSurfaceApplication
from ratchet.surface_search import TransformContextKey
from ratchet.types import AgentSpec, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


class FakeAdapter:
    def agent_spec(self) -> AgentSpec:
        return AgentSpec(name="sample", model="base", instructions={"system_prompt": "Answer."})

    def run_case(self, case: EvalCase, candidate=None) -> RunRecord:
        return RunRecord(
            output="ok",
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.01,
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return GradeResult(score=1.0, passed=True)

    def export(self, candidate, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)


class FailingAdapter(FakeAdapter):
    def run_case(self, case: EvalCase, candidate=None) -> RunRecord:
        raise RuntimeError("provider quota exhausted")


def _summary(
    *,
    candidate_id: str = "baseline",
    score: float = 1.0,
    passed: bool = True,
    cost_usd: float = 0.01,
    latency_s: float = 1.0,
) -> CandidateSummary:
    case = EvalCase(id="case-1", split="dev", input="x")
    return CandidateSummary(
        candidate_id=candidate_id,
        candidate=None,
        split="dev",
        evaluations=[
            CaseEvaluation(
                case=case,
                record=RunRecord(
                    output="ok",
                    metrics=OperationalMetrics(
                        latency_s=latency_s,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=cost_usd,
                    ),
                ),
                grade=GradeResult(score=score, passed=passed),
            )
        ],
    )


def _evaluation_state() -> CandidateEvaluationState:
    surface = surface_from_agent_spec(FakeAdapter().agent_spec())
    program = TransformProgram.from_dict(
        {
            "candidate_id": "prompt",
            "patches": [
                {
                    "hook": "before_model_call",
                    "op": "add_context_section",
                    "section": "extra",
                    "content": "Be complete.",
                }
            ],
        }
    )
    proposal = CandidateProposal(
        program=program,
        applications=[CandidateSurfaceApplication("surface.surface_context.system_prompt")],
        experiment_id="intent-1",
        hypothesis="Add context.",
    )
    compiled = TransformCompiler().compile_or_raise(program, surface)
    return CandidateEvaluationState(
        proposal=proposal,
        compiled_candidate=compiled,
        candidate_id="candidate-1",
        proposal_candidate_id="proposal-1",
        transform_context=TransformContextKey.from_candidate(proposal),
    )


class OptimizerConcurrencyTests(unittest.TestCase):
    def test_optimizer_rejects_hard_timeout_with_threaded_concurrency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "case_timeout_s requires serial case execution"):
                RatchetOptimizer(
                    FakeAdapter(),
                    Path(tmp),
                    case_timeout_s=180,
                    case_concurrency=2,
                )

    def test_zero_measurement_budget_screens_before_smoke_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            optimizer = RatchetOptimizer(
                FakeAdapter(),
                Path(tmp),
                case_timeout_s=0,
                max_dev_measurement_cost_usd=0.0,
            )
            state = _evaluation_state()

            kept = optimizer._filter_candidate_stage_by_measurement_budget(
                [state],
                reference=_summary(),
                stage_name="smoke",
                stage_cases=(EvalCase(id="case-1", split="dev", input="x"),),
                evidence_ledger=EvidenceLedger(),
            )

        self.assertEqual(kept, [])
        self.assertIsNone(state.summary)
        self.assertEqual(state.frontier_status, "screened_out")
        self.assertIn("measurement_budget_exhausted", str(state.rejection_reason))

    def test_gemini_pro_model_substitution_throttles_candidate_batch(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="gemini-3-flash-preview",
                model_options=["gemini-3-flash-preview", "gemini-3-pro-preview"],
            )
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "pro-model",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "set_model_config",
                            "field": "model_name",
                            "value": "gemini-3-pro-preview",
                        }
                    ],
                }
            ),
            surface,
        )

        self.assertEqual(_candidate_batch_concurrency_limit([None, candidate]), 1)

    def test_non_model_candidate_does_not_throttle_candidate_batch(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="gemini-3-flash-preview", instructions={"system_prompt": "Answer."})
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "prompt",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "add_context_section",
                            "section": "extra",
                            "content": "Be complete.",
                        }
                    ],
                }
            ),
            surface,
        )

        self.assertEqual(_candidate_batch_concurrency_limit([None, candidate]), 10_000)

    def test_flat_correctness_with_lower_cost_is_efficiency_frontier_only(self) -> None:
        state = _evaluation_state()
        state.summary = _summary(candidate_id="candidate-1", score=1.0, cost_usd=0.005)
        state.rejection_reason = "no positive correctness gain"

        _finalize_candidate_state(state, _summary(score=1.0, cost_usd=0.01), OptimizationObjective())

        self.assertEqual(state.frontier_status, "efficiency_frontier")
        self.assertFalse(state.accepted)

    def test_positive_correctness_gain_is_promotable_dev(self) -> None:
        state = _evaluation_state()
        state.summary = _summary(candidate_id="candidate-1", score=1.0)

        _finalize_candidate_state(state, _summary(score=0.0), OptimizationObjective())

        self.assertEqual(state.frontier_status, "promotable_dev")
        self.assertTrue(state.accepted)

    def test_small_dev_flat_candidate_does_not_reach_full_dev(self) -> None:
        state = _evaluation_state()
        state.stage_rows.append(
            {
                "stage": "small_dev",
                "comparison_to_parent": {"score_delta": 0.0},
                "behavior_flip_summary": {"fixed_count": 0, "regressed_count": 0},
                "metrics": {"behavioral": {"failure_labels": {"invalid_output": 1}}},
            }
        )

        eligible, reason = _eligible_for_full_dev_from_small_signal(state, OptimizationObjective())
        self.assertFalse(eligible)
        self.assertIn("no positive correctness signal", reason)

    def test_small_dev_targeted_failure_reduction_reaches_full_dev(self) -> None:
        state = _evaluation_state()
        state.stage_rows.append(
            {
                "stage": "small_dev",
                "comparison_to_parent": {"score_delta": 0.0},
                "behavior_flip_summary": {"fixed_count": 2, "regressed_count": 0},
                "metrics": {"behavioral": {"failure_labels": {"invalid_output": 1}}},
            }
        )

        eligible, reason = _eligible_for_full_dev_from_small_signal(state, OptimizationObjective())
        self.assertTrue(eligible)
        self.assertEqual(reason, "")

    def test_small_dev_stage_does_not_expand_to_full_dev_on_many_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            optimizer = RatchetOptimizer(FakeAdapter(), Path(tmp), case_timeout_s=0)
            reference = CandidateSummary(
                candidate_id="baseline",
                candidate=None,
                split="dev",
                evaluations=[
                    CaseEvaluation(
                        case=EvalCase(id=f"case-{index}", split="dev", input="x"),
                        record=RunRecord(
                            output="ok",
                            metrics=OperationalMetrics(
                                latency_s=1.0,
                                input_tokens=10,
                                output_tokens=5,
                                total_tokens=15,
                                cost_usd=0.01,
                            ),
                        ),
                        grade=GradeResult(score=1.0 if index % 4 == 0 else 0.0, passed=index % 4 == 0),
                    )
                    for index in range(16)
                ],
            )
            dev_cases = tuple(evaluation.case for evaluation in reference.evaluations)

            stages = optimizer._progressive_eval_stages(reference, dev_cases)

        self.assertEqual([name for name, _ in stages], ["smoke", "small_dev", "full_dev"])
        small_cases = next(cases for name, cases in stages if name == "small_dev")
        self.assertLess(len(small_cases), len(dev_cases))

    def test_systemic_runtime_errors_abort_evaluation(self) -> None:
        cases = tuple(EvalCase(id=f"case-{index}", split="holdout", input="x") for index in range(4))
        with tempfile.TemporaryDirectory() as tmp:
            optimizer = RatchetOptimizer(
                FailingAdapter(),
                Path(tmp),
                case_timeout_s=0,
                max_case_retries=0,
                case_concurrency=2,
            )

            with self.assertRaisesRegex(RuntimeError, "runtime errors made the measurement invalid"):
                optimizer.evaluate_candidate(None, cases)


if __name__ == "__main__":
    unittest.main()
