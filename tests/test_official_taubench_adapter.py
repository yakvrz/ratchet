from __future__ import annotations

import unittest

from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord
from samples.taubench_official_agent.ratchet_adapter import OfficialTauBenchAdapter


class FakeTauBenchRunner:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def run_task(self, **kwargs: object) -> RunRecord:
        self.calls.append({"runner": self.kwargs, "task": kwargs})
        return RunRecord(
            output={"benchmark": "tau-bench", "task_id": kwargs["task_id"], "reward": 1.0, "info": {}},
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=100,
                output_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
                model_calls=2,
                tool_calls=3,
                turns=5,
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=["get_order_details"],
                terminal_state={"reward": 1.0},
                terminal_reason="success",
            ),
        )


class OfficialTauBenchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeTauBenchRunner.calls = []

    def test_adapter_runs_official_runner_and_grades_reward(self) -> None:
        adapter = OfficialTauBenchAdapter(env_path="/missing/.env", runner_factory=FakeTauBenchRunner)
        case = EvalCase(
            id="dev-retail-0",
            split="dev",
            input="official task",
            expected={"reward": 1.0},
            metadata={"env": "retail", "task_split": "test", "task_id": 0},
        )

        record = adapter.run_case(case)
        grade = adapter.grade(case, record.output)

        self.assertTrue(grade.passed)
        self.assertEqual(FakeTauBenchRunner.calls[0]["task"]["model"], "gpt-4o")
        self.assertEqual(FakeTauBenchRunner.calls[0]["task"]["model_provider"], "openai")
        self.assertEqual(record.diagnostics.metadata["benchmark_fidelity"], "official_tau_bench_simulator")
        self.assertEqual(record.metrics.tool_calls, 3)

    def test_model_substitution_maps_provider(self) -> None:
        adapter = OfficialTauBenchAdapter(env_path="/missing/.env", runner_factory=FakeTauBenchRunner)
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "sonnet",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "set_model_config",
                            "field": "model_name",
                            "value": "claude-3-5-sonnet-20241022",
                        }
                    ],
                }
            ),
            adapter.surface_spec(),
        )
        case = EvalCase(
            id="dev-retail-1",
            split="dev",
            input="official task",
            expected={"reward": 1.0},
            metadata={"env": "retail", "task_split": "test", "task_id": 1},
        )

        adapter.run_case(case, candidate)

        self.assertEqual(FakeTauBenchRunner.calls[0]["task"]["model"], "claude-3-5-sonnet-20241022")
        self.assertEqual(FakeTauBenchRunner.calls[0]["task"]["model_provider"], "anthropic")


if __name__ == "__main__":
    unittest.main()
