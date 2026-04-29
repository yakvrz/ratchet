from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import EvalCase
from samples.taubench_action_agent.ratchet_adapter import TauBenchActionAdapter


class FakeUsage:
    input_tokens = 31
    output_tokens = 7


class FakeOutputItem:
    type = "message"


class FakeResponse:
    usage = FakeUsage()
    output = [FakeOutputItem()]
    finish_reason = "stop"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create_response(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(json.dumps({"actions": [{"name": "get_order"}], "message": "Done"}))


class TransformArchitectureTauBenchTests(unittest.TestCase):
    def test_compiled_transform_runs_through_tau_bench_action_surface(self) -> None:
        client = FakeClient()
        from samples.taubench_action_agent.agent import TauBenchActionRunner

        adapter = TauBenchActionAdapter(runner=TauBenchActionRunner(client=client))
        surface = adapter.surface_spec()
        program = TransformProgram.from_dict(
            {
                "candidate_id": "C_context_runtime_guard",
                "hypothesis_id": "H_context_surface",
                "patches": [
                    {
                        "op": "define_state",
                        "field": "task_input",
                        "type": "string",
                        "initial": {"$ref": "case.input"},
                    },
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": "current_task_state",
                        "position": "after:policy_rule",
                        "content": {
                            "task_input": {"$ref": "state.task_input"},
                            "discipline": "Prefer necessary tau-bench workflow actions and avoid speculative extras.",
                        },
                    },
                    {
                        "hook": "before_model_call",
                        "op": "set_model_config",
                        "field": "max_tokens",
                        "value": 777,
                    },
                    {
                        "hook": "before_user_response",
                        "op": "validate",
                        "target": "draft_response",
                        "checks": ["json_object", "actions_array"],
                        "on_fail": {
                            "op": "rewrite_response",
                            "replacement": {"actions": [], "message": "Unable to produce a valid action plan."},
                        },
                    },
                ],
            }
        )
        candidate = TransformCompiler().compile_or_raise(program, surface)
        case = EvalCase(
            id="taubench-proof-1",
            split="dev",
            input="Customer asks where order O-1 is. Available tools: get_order.",
            expected={"actions": [{"name": "get_order"}]},
        )

        record = adapter.run_case(case, candidate)
        grade = adapter.grade(case, record.output)

        self.assertTrue(grade.passed)
        self.assertEqual(client.calls[0]["max_output_tokens"], 777)
        self.assertIn("[current_task_state]", str(client.calls[0]["instructions"]))
        self.assertEqual(record.diagnostics.metadata["transform_candidate_id"], "C_context_runtime_guard")
        trace = record.diagnostics.metadata["transform_trace"]
        self.assertGreaterEqual(len(trace), 4)
        self.assertIn("current_task_state", record.diagnostics.metadata["rendered_context_sections"])
        self.assertEqual(record.diagnostics.metadata["transform_compile_report"]["status"], "compiled")

    def test_compiler_rejects_unsupported_tau_bench_tool_hook(self) -> None:
        adapter = TauBenchActionAdapter(runner=None)
        program = TransformProgram.from_dict(
            {
                "candidate_id": "C_bad_tool_hook",
                "patches": [
                    {
                        "hook": "before_tool_call",
                        "op": "normalize_tool_args",
                        "target": "tool_call",
                    }
                ],
            }
        )

        candidate = TransformCompiler().compile(program, adapter.surface_spec())

        self.assertEqual(candidate.report.status, "rejected")
        self.assertEqual(candidate.report.rejection.code, "unsupported_hook")

    def test_compiled_candidate_export_is_inspectable(self) -> None:
        adapter = TauBenchActionAdapter(runner=None)
        program = TransformProgram.from_dict(
            {
                "candidate_id": "C_exportable",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": "completion_discipline",
                        "content": "Only report actions represented in the JSON action list.",
                    }
                ],
            }
        )
        candidate = TransformCompiler().compile_or_raise(program, adapter.surface_spec())
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            adapter.export(candidate, out_dir)
            payload = json.loads((out_dir / "compiled_candidate.json").read_text())
            surface = json.loads((out_dir / "surface_spec.json").read_text())

        self.assertEqual(payload["report"]["status"], "compiled")
        self.assertEqual(surface["agent_id"], "taubench-action-agent")


if __name__ == "__main__":
    unittest.main()
