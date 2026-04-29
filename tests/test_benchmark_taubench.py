from __future__ import annotations

from types import SimpleNamespace
import unittest

from ratchet.benchmarks import taubench_result_to_run_record


class TauBenchBridgeTests(unittest.TestCase):
    def test_result_converter_preserves_original_tau_bench_tool_trajectory(self) -> None:
        result = SimpleNamespace(
            task_id=7,
            trial=0,
            reward=1.0,
            info={"r_actions": 1.0},
            traj=[
                {"role": "user", "content": "Where is my order?", "usage": {"input_tokens": 4, "output_tokens": 5}, "cost": 0.001},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_order",
                                "arguments": {"order_id": "O-1"},
                            }
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 10},
                    "cost": 0.002,
                },
                {"role": "tool", "tool_name": "get_order", "observation": '{"status":"delivered"}'},
            ],
        )

        record = taubench_result_to_run_record(result)

        self.assertEqual(record.output["benchmark"], "tau-bench")
        self.assertEqual(record.output["reward"], 1.0)
        self.assertEqual(record.metrics.model_calls, 2)
        self.assertEqual(record.metrics.tool_calls, 1)
        self.assertEqual(record.metrics.turns, 3)
        self.assertEqual(record.metrics.cost_usd, 0.003)
        self.assertEqual(record.diagnostics.terminal_reason, "success")
        self.assertEqual(record.diagnostics.tool_calls, ["get_order"])


if __name__ == "__main__":
    unittest.main()
