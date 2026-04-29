from __future__ import annotations

import unittest

from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.transforms import build_search_hypothesis
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


def _summary(labels: list[list[str]]) -> CandidateSummary:
    evaluations: list[CaseEvaluation] = []
    for index, case_labels in enumerate(labels, start=1):
        passed = not case_labels
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(id=f"case-{index}", split="dev", input=f"input {index}"),
                record=RunRecord(
                    output="ok" if passed else "wrong",
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(metadata={"finish_reason": "stop"}),
                ),
                grade=GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=case_labels),
            )
        )
    return CandidateSummary(candidate_id="baseline", candidate=None, split="dev", evaluations=evaluations)


class TransformLibraryTests(unittest.TestCase):
    def test_search_hypothesis_uses_surface_spec(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="large", instructions={"system_prompt": "Answer."})
        )

        hypothesis = build_search_hypothesis(
            summary=_summary([["wrong_label"], []]),
            surface=surface,
            objective=OptimizationObjective(),
            history=[],
        )

        self.assertIn("prompt_rewrite", hypothesis.active_families)
        self.assertTrue(hypothesis.context_states)

    def test_compiler_rejects_unsupported_hook(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad",
                "patches": [
                    {"hook": "before_tool_call", "op": "normalize_tool_args", "target": "tool_call"}
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "unsupported_hook")


if __name__ == "__main__":
    unittest.main()
