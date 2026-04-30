from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ratchet.evidence_ledger import EvidenceLedger
from ratchet.optimizer import CandidateEvaluationState, RatchetOptimizer, _candidate_batch_concurrency_limit
from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.transforms import CandidateAffordanceApplication, CandidateProposal, TransformContextKey
from ratchet.types import AgentSpec, EvalCase, GradeResult, OperationalMetrics, RunRecord


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


def _summary() -> CandidateSummary:
    case = EvalCase(id="case-1", split="dev", input="x")
    return CandidateSummary(
        candidate_id="baseline",
        candidate=None,
        split="dev",
        evaluations=[
            CaseEvaluation(
                case=case,
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
                grade=GradeResult(score=1.0, passed=True),
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
        applications=[CandidateAffordanceApplication("surface.surface_context.system_prompt")],
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


if __name__ == "__main__":
    unittest.main()
