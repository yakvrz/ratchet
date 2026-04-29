from __future__ import annotations

import json
import unittest

from ratchet.adapter_generation import GeneratedSingleCallAdapter
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import AgentPatch, EvalCase, PatchOperation
from samples.banking77_intent_agent.ratchet_adapter import Banking77IntentAdapter
from samples.bfcl_function_calling_agent.ratchet_adapter import BfclFunctionCallingAdapter
from samples.clinc150_intent_agent.ratchet_adapter import Clinc150IntentAdapter


class FakeUsage:
    input_tokens = 100
    output_tokens = 10


class FakeOutputItem:
    type = "message"


class FakeResponse:
    usage = FakeUsage()
    output = [FakeOutputItem()]
    finish_reason = "stop"

    def __init__(self, payload: object) -> None:
        self.output_text = json.dumps(payload)


class FakeClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def create_response(self, **kwargs: object) -> FakeResponse:
        self.calls.append(kwargs)
        return FakeResponse(self.payload)


class GeneratedSampleAdapterTests(unittest.TestCase):
    def test_bfcl_uses_generated_single_call_adapter(self) -> None:
        adapter = BfclFunctionCallingAdapter(
            client=FakeClient({"calls": [{"name": "calculate_density", "arguments": {"mass": 45, "volume": 15}}]})
        )
        case = EvalCase(
            id="bfcl-1",
            split="dev",
            input=json.dumps(
                {
                    "question": "What is the density of a substance with a mass of 45 kg and a volume of 15 m3?",
                    "functions": [
                        {
                            "name": "calculate_density",
                            "description": "Calculate density.",
                            "parameters": {"type": "dict", "properties": {}},
                        }
                    ],
                }
            ),
            expected={"ground_truth": [{"calculate_density": {"mass": [45], "volume": [15]}}]},
        )

        record = adapter.run_case(case)
        grade = adapter.grade(case, record.output)

        self.assertIsInstance(adapter, GeneratedSingleCallAdapter)
        self.assertTrue(grade.passed)
        self.assertIn("task_rule", record.diagnostics.metadata["rendered_context_sections"])

    def test_banking77_uses_generated_single_call_adapter(self) -> None:
        client = FakeClient({"label": "cash_withdrawal_charge"})
        adapter = Banking77IntentAdapter(client=client)
        case = EvalCase(
            id="banking-1",
            split="dev",
            input="I was charged for withdrawing cash.",
            expected={"label": "cash_withdrawal_charge"},
        )

        record = adapter.run_case(case)
        grade = adapter.grade(case, record.output)

        self.assertIsInstance(adapter, GeneratedSingleCallAdapter)
        self.assertTrue(grade.passed)
        self.assertIn("label_rule", record.diagnostics.metadata["rendered_context_sections"])

        program = TransformProgram.from_dict(
            {
                "id": "C_add_label_rule",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "replace_context_section",
                        "section": "label_rule",
                        "content": "Use concise labels.",
                    }
                ],
            }
        )
        candidate = TransformCompiler().compile_or_raise(program, adapter.surface_spec())
        patched = adapter.run_case(case, candidate)
        self.assertIn("Use concise labels.", str(client.calls[-1]["instructions"]))
        self.assertEqual(
            patched.diagnostics.metadata["transform_compile_report"]["status"],
            "compiled",
        )
        self.assertTrue(patched.diagnostics.metadata["transform_trace"])

        with self.assertRaisesRegex(TypeError, "CompiledCandidate"):
            adapter.run_case(
                case,
                AgentPatch(
                    operations=[
                        PatchOperation(
                            op="add_instruction",
                            target="instructions.label_rule",
                            value="legacy patches are not adapter candidates",
                        )
                    ]
                ),  # type: ignore[arg-type]
            )

    def test_clinc150_uses_generated_single_call_adapter(self) -> None:
        adapter = Clinc150IntentAdapter(client=FakeClient({"label": "weather"}))
        case = EvalCase(
            id="clinc-1",
            split="dev",
            input="What is the weather tomorrow?",
            expected={"label": "weather"},
        )

        record = adapter.run_case(case)
        grade = adapter.grade(case, record.output)

        self.assertIsInstance(adapter, GeneratedSingleCallAdapter)
        self.assertTrue(grade.passed)
        self.assertIn("label_rule", record.diagnostics.metadata["rendered_context_sections"])


if __name__ == "__main__":
    unittest.main()
