from __future__ import annotations

import unittest

from ratchet.profiling import runtime_reliability_diagnostics
from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, RunRecord


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


class ProfilingTests(unittest.TestCase):
    def test_runtime_transform_is_reported_as_runtime_involved(self) -> None:
        candidate = _candidate()
        diagnostics = runtime_reliability_diagnostics(
            _summary(None, passed=False, invalid=True, finish_reason="length"),
            _summary(candidate, passed=True, invalid=False, finish_reason="stop"),
        )

        self.assertTrue(diagnostics["runtime_only"])
        self.assertTrue(diagnostics["runtime_involved"])


if __name__ == "__main__":
    unittest.main()
