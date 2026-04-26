from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ratchet.adapters import adapter_fingerprint
from ratchet.types import AgentPatch, AgentSpec, EvalCase, PatchOperation


class FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeOutputItem:
    def __init__(
        self,
        item_type: str,
        *,
        name: str = "",
        arguments: str = "{}",
        call_id: str = "",
    ) -> None:
        self.type = item_type
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class FakeResponse:
    def __init__(
        self,
        response_id: str,
        output: list[FakeOutputItem],
        *,
        output_text: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.id = response_id
        self.output = output
        self.output_text = output_text
        self.usage = FakeUsage(input_tokens, output_tokens)


class FakeResponsesClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls = 0

    def create_response(self, **_: object) -> FakeResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


class AdapterFingerprintTests(unittest.TestCase):
    def test_fingerprint_includes_sidecar_behavior_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "sidecar_adapter.py"
            prompt_path = root / "judge_prompt.md"
            module_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult

                    class Adapter:
                        def agent_spec(self):
                            return AgentSpec(name="sidecar", model="primary")

                        def run_case(self, case: EvalCase, patch: AgentPatch | None = None):
                            raise NotImplementedError

                        def grade(self, case, output):
                            return GradeResult(score=1.0, passed=True)

                        def export(self, patch, out_dir):
                            Path(out_dir).mkdir(parents=True, exist_ok=True)

                    adapter = Adapter()
                    """
                ).strip()
            )
            prompt_path.write_text("judge v1")
            sys.path.insert(0, str(root))
            try:
                first = adapter_fingerprint("sidecar_adapter:adapter")
                prompt_path.write_text("judge v2")
                second = adapter_fingerprint("sidecar_adapter:adapter")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("sidecar_adapter", None)
                importlib.invalidate_caches()

        self.assertNotEqual(first["source_tree_sha256"], second["source_tree_sha256"])

    def test_fingerprint_includes_nested_behavior_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "prompts"
            nested.mkdir()
            module_path = root / "nested_adapter.py"
            prompt_path = nested / "judge_prompt.md"
            module_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult

                    class Adapter:
                        def agent_spec(self):
                            return AgentSpec(name="nested", model="primary")

                        def run_case(self, case: EvalCase, patch: AgentPatch | None = None):
                            raise NotImplementedError

                        def grade(self, case, output):
                            return GradeResult(score=1.0, passed=True)

                        def export(self, patch, out_dir):
                            Path(out_dir).mkdir(parents=True, exist_ok=True)

                    adapter = Adapter()
                    """
                ).strip()
            )
            prompt_path.write_text("nested judge v1")
            sys.path.insert(0, str(root))
            try:
                first = adapter_fingerprint("nested_adapter:adapter")
                prompt_path.write_text("nested judge v2")
                second = adapter_fingerprint("nested_adapter:adapter")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("nested_adapter", None)
                importlib.invalidate_caches()

        self.assertNotEqual(first["source_tree_sha256"], second["source_tree_sha256"])

    def test_fingerprint_includes_optional_adapter_cache_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "custom_adapter.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from ratchet.types import AgentPatch, AgentSpec, GradeResult

                    VERSION = "v1"

                    class Adapter:
                        def agent_spec(self):
                            return AgentSpec(name="custom", model="primary")

                        def run_case(self, case, patch: AgentPatch | None = None):
                            raise NotImplementedError

                        def grade(self, case, output):
                            return GradeResult(score=1.0, passed=True)

                        def export(self, patch, out_dir):
                            Path(out_dir).mkdir(parents=True, exist_ok=True)

                        def cache_fingerprint(self):
                            return {"version": VERSION}

                    adapter = Adapter()
                    """
                ).strip()
            )
            sys.path.insert(0, str(root))
            try:
                module = importlib.import_module("custom_adapter")
                first = adapter_fingerprint("custom_adapter:adapter")
                module.VERSION = "v2"
                second = adapter_fingerprint("custom_adapter:adapter")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("custom_adapter", None)
                importlib.invalidate_caches()

        self.assertNotEqual(first["custom_fingerprint_sha256"], second["custom_fingerprint_sha256"])

    def test_kashi_few_shot_patch_reaches_agent_prompt(self) -> None:
        agent_path = Path(__file__).resolve().parents[1] / "samples" / "kashi_agent" / "agent.py"
        module_spec = importlib.util.spec_from_file_location("kashi_agent_prompt_test", agent_path)
        if module_spec is None or module_spec.loader is None:
            raise AssertionError("Could not load Kashi sample agent.")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        try:
            module_spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_spec.name, None)

        spec = AgentSpec(
            name="kashi-test",
            model="gpt-4o-2024-08-06",
            instructions={"base": "Base instruction."},
            few_shot=[{"messages": [{"role": "user", "content": "Synthetic example."}]}],
        )
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_few_shot",
                    target="few_shot",
                    value={"messages": [{"role": "user", "content": "Patched example."}]},
                )
            ]
        )

        config = module.KashiAgentConfig.from_spec(spec.apply_patch(patch))
        rendered = "\n".join(config.instructions)

        self.assertIn("Synthetic example.", rendered)
        self.assertIn("Patched example.", rendered)

    def test_tool_loop_accounts_final_continuation_usage(self) -> None:
        from samples.public_docs_agent.agent import PublicDocsAgentRunner

        client = FakeResponsesClient(
            [
                FakeResponse(
                    "resp-1",
                    [
                        FakeOutputItem(
                            "function_call",
                            name="docs_search",
                            arguments='{"query": "python list append"}',
                            call_id="call-1",
                        )
                    ],
                    output_text="",
                    input_tokens=10,
                    output_tokens=2,
                ),
                FakeResponse(
                    "resp-2",
                    [FakeOutputItem("message")],
                    output_text='{"answer": "list.append()"}',
                    input_tokens=20,
                    output_tokens=4,
                ),
            ]
        )
        runner = PublicDocsAgentRunner(client=client)
        record = runner.run_case(
            {
                "model": "gpt-5.4-mini",
                "reasoning_effort": "low",
                "prompt_output_rule": "Return JSON.",
                "prompt_grounding_rule": "Use docs.",
                "prompt_tool_rule": "Use docs_search.",
                "prompt_fallback_rule": "Return unknown.",
                "prompt_few_shot": "",
                "docs_search_enabled": "on",
                "docs_search_description": "Search docs.",
                "knowledge_mode": "raw",
                "retrieval_top_k": "1",
                "output_cap": "64",
                "max_tool_rounds": "1",
            },
            EvalCase(id="dev-1", split="dev", input="Which method appends to a list?"),
        )

        self.assertEqual(record.metrics.input_tokens, 30)
        self.assertEqual(record.metrics.output_tokens, 6)
        self.assertEqual(record.diagnostics.metadata["response_ids"], ["resp-1", "resp-2"])
        self.assertEqual(record.diagnostics.metadata["output_item_types"], [["function_call"], ["message"]])

    def test_python_api_grounding_rejects_parser_fallback_output(self) -> None:
        from samples.python_api_grounding_agent.ratchet_adapter import PythonApiGroundingAdapter

        grade = PythonApiGroundingAdapter().grade(
            EvalCase(
                id="api-dev-01",
                split="dev",
                input="Which option?",
                expected={"answer": "unknown"},
            ),
            {"answer": "unknown", "invalid_output": ""},
        )

        self.assertFalse(grade.passed)
        self.assertIn("invalid_output", grade.labels)

    def test_tool_loop_raises_when_final_response_still_requests_tools(self) -> None:
        from samples.public_docs_agent.agent import PublicDocsAgentRunner

        client = FakeResponsesClient(
            [
                FakeResponse(
                    "resp-1",
                    [
                        FakeOutputItem(
                            "function_call",
                            name="docs_search",
                            arguments='{"query": "python list append"}',
                            call_id="call-1",
                        )
                    ],
                    output_text="",
                    input_tokens=10,
                    output_tokens=2,
                ),
                FakeResponse(
                    "resp-2",
                    [
                        FakeOutputItem(
                            "function_call",
                            name="docs_search",
                            arguments='{"query": "python list extend"}',
                            call_id="call-2",
                        )
                    ],
                    output_text="",
                    input_tokens=20,
                    output_tokens=4,
                ),
            ]
        )
        runner = PublicDocsAgentRunner(client=client)

        with self.assertRaisesRegex(RuntimeError, "tool round budget exhausted"):
            runner.run_case(
                {
                    "model": "gpt-5.4-mini",
                    "reasoning_effort": "low",
                    "prompt_output_rule": "Return JSON.",
                    "prompt_grounding_rule": "Use docs.",
                    "prompt_tool_rule": "Use docs_search.",
                    "prompt_fallback_rule": "Return unknown.",
                    "prompt_few_shot": "",
                    "docs_search_enabled": "on",
                    "docs_search_description": "Search docs.",
                    "knowledge_mode": "raw",
                    "retrieval_top_k": "1",
                    "output_cap": "64",
                    "max_tool_rounds": "1",
                },
                EvalCase(id="dev-1", split="dev", input="Which method appends to a list?"),
            )


if __name__ == "__main__":
    unittest.main()
