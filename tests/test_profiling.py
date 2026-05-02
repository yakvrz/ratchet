from __future__ import annotations

import unittest

from ratchet.profiling import runtime_reliability_diagnostics
from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, InteractionTurn, OperationalMetrics, RunRecord, ToolCallTrace


def _candidate() -> object:
    program = TransformProgram.from_dict(
        {
            "candidate_id": "runtime",
            "patches": [{"hook": "before_model_call", "op": "set_model_config", "field": "max_tokens", "value": 1024}],
        }
    )
    return TransformCompiler().compile_or_raise(program, surface_from_agent_spec(AgentSpec(name="sample", model="base")))


def _summary(candidate: object | None, *, passed: bool, invalid: bool, finish_reason: str = "stop") -> CandidateSummary:
    evaluation = CaseEvaluation(
        case=EvalCase(id="case-1", split="dev", input="x"),
        record=RunRecord(
            output={"invalid_output": "bad"} if invalid else {"answer": "ok"},
            metrics=OperationalMetrics(latency_s=1.0, input_tokens=100, output_tokens=8, total_tokens=108, cost_usd=0.001),
            diagnostics=DiagnosticTrace(metadata={"invalid_output": invalid, "finish_reason": finish_reason, "requested_output_cap": 1024}),
        ),
        grade=GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=[] if passed else ["invalid_output"]),
    )
    return CandidateSummary(candidate_id="candidate" if candidate else "baseline", candidate=candidate, split="dev", evaluations=[evaluation])


def _tool_problem_summary(candidate: object | None, *, passed: bool, tool_error: bool) -> CandidateSummary:
    evaluation = CaseEvaluation(
        case=EvalCase(id="case-1", split="dev", input="x"),
        record=RunRecord(
            output={"answer": "ok"},
            metrics=OperationalMetrics(latency_s=1.0, input_tokens=50, output_tokens=50, total_tokens=100, cost_usd=0.001),
            diagnostics=DiagnosticTrace(
                turns=[
                    InteractionTurn(
                        index=1,
                        actor="assistant",
                        tool_calls=[
                            ToolCallTrace(
                                name="mutate",
                                status="error" if tool_error else "ok",
                                error="order_not_inspected" if tool_error else None,
                            )
                        ],
                    )
                ]
            ),
        ),
        grade=GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=[] if passed else ["tool_error:order_not_inspected"]),
    )
    return CandidateSummary(candidate_id="candidate" if candidate else "baseline", candidate=candidate, split="dev", evaluations=[evaluation])


class ProfilingTests(unittest.TestCase):
    def test_runtime_transform_is_reported_as_runtime_involved(self) -> None:
        candidate = _candidate()
        diagnostics = runtime_reliability_diagnostics(
            _summary(None, passed=False, invalid=True, finish_reason="length"),
            _summary(candidate, passed=True, invalid=False, finish_reason="stop"),
        )

        self.assertTrue(diagnostics["runtime_only"])
        self.assertTrue(diagnostics["runtime_involved"])

    def test_fixed_tool_errors_are_reported_as_confirmation_worthy(self) -> None:
        candidate = _candidate()
        diagnostics = runtime_reliability_diagnostics(
            _tool_problem_summary(None, passed=False, tool_error=True),
            _tool_problem_summary(candidate, passed=True, tool_error=False),
        )

        self.assertTrue(diagnostics["tool_trajectory_defect_fixed"])
        self.assertEqual(diagnostics["fixed_tool_problem_case_ids"], ["case-1"])


if __name__ == "__main__":
    unittest.main()
