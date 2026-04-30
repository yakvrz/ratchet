from __future__ import annotations

import unittest

from ratchet.tool_loop import GeneratedToolLoopAdapter, ToolLoopModelResponse
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import AgentSpec, AgentTool, EvalCase


class _Response:
    def __init__(self, observation: str, *, reward: float = 0.0, done: bool = False) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = {"source": "fake"}


class _FakeEnvironment:
    wiki = "Use tools to inspect state before responding."
    tools_info = [
        {
            "type": "function",
            "function": {
                "name": "lookup_order",
                "description": "Look up an order.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]

    def __init__(self) -> None:
        self.actions: list[tuple[str, dict[str, object]]] = []

    def reset(self, task_index: int | None = None) -> _Response:
        return _Response("Need status for order A1.")

    def step(self, action: dict[str, object]) -> _Response:
        self.actions.append((str(action["name"]), dict(action["kwargs"])))
        if action["name"] == "lookup_order":
            return _Response("Order A1 is delivered.")
        return _Response("###STOP###", reward=1.0, done=True)


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs: object) -> ToolLoopModelResponse:
        self.calls += 1
        if self.calls == 1:
            return ToolLoopModelResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_order",
                                "arguments": '{"order_id": " A1 "}',
                            },
                        }
                    ],
                },
                input_tokens=10,
                output_tokens=5,
            )
        return ToolLoopModelResponse(
            message={"role": "assistant", "content": "The order is delivered."},
            input_tokens=12,
            output_tokens=6,
        )


class _RepeatingToolClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **kwargs: object) -> ToolLoopModelResponse:
        self.calls += 1
        if self.calls <= 2:
            return ToolLoopModelResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{self.calls}",
                            "type": "function",
                            "function": {
                                "name": "lookup_order",
                                "arguments": '{"order_id": "A1"}',
                            },
                        }
                    ],
                },
                input_tokens=10,
                output_tokens=5,
            )
        return ToolLoopModelResponse(
            message={"role": "assistant", "content": "The order is delivered."},
            input_tokens=12,
            output_tokens=6,
        )


class _ToolDescriptionClient:
    def __init__(self) -> None:
        self.tool_descriptions: list[str] = []

    def complete(self, **kwargs: object) -> ToolLoopModelResponse:
        tools = kwargs["tools"]
        assert isinstance(tools, list)
        function = tools[0]["function"]
        self.tool_descriptions.append(str(function["description"]))
        return ToolLoopModelResponse(
            message={"role": "assistant", "content": "The order is delivered."},
            input_tokens=12,
            output_tokens=6,
        )


class ToolLoopAdapterTests(unittest.TestCase):
    def test_tool_loop_executes_transform_hooks_around_environment_tools(self) -> None:
        env = _FakeEnvironment()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(
                name="fake-tool-loop",
                model="gpt-4o",
                model_options=["gpt-4o"],
                tools={"lookup_order": AgentTool(name="lookup_order", metadata={"side_effect": "read"})},
            ),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_FakeClient(),
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "tool_loop_hooks",
                    "patches": [
                        {"op": "define_state", "field": "observed", "type": "list[object]", "initial": []},
                        {"hook": "before_tool_call", "op": "normalize_tool_args", "target": "tool_call"},
                        {
                            "hook": "after_tool_result",
                            "op": "append_state",
                            "field": "observed",
                            "value": {
                                "tool": {"$ref": "tool_call.name"},
                                "status": {"$ref": "tool_result.status"},
                            },
                        },
                    ],
                }
            ),
            adapter.surface_spec(),
        )
        case = EvalCase(id="case-1", split="dev", input="", expected={"reward": 1.0})

        record = adapter.run_case(case, candidate)
        grade = adapter.grade(case, record.output)

        self.assertTrue(grade.passed)
        self.assertEqual(env.actions[0], ("lookup_order", {"order_id": "A1"}))
        self.assertEqual(record.metrics.model_calls, 2)
        self.assertEqual(record.metrics.tool_calls, 1)
        trace_ops = [item["op"] for item in record.diagnostics.metadata["transform_trace"]]
        self.assertIn("normalize_tool_args", trace_ops)
        self.assertIn("append_state", trace_ops)

    def test_tool_loop_validation_can_replan_duplicate_tool_calls(self) -> None:
        env = _FakeEnvironment()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_RepeatingToolClient(),
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "duplicate_guard",
                    "patches": [
                        {
                            "hook": "before_tool_call",
                            "op": "validate",
                            "target": "tool_call",
                            "checks": ["not_duplicate_tool_call"],
                            "on_fail": {
                                "hook": "before_tool_call",
                                "op": "replan",
                                "message": "The proposed tool call repeats an already observed result. Continue with the next step.",
                            },
                        }
                    ],
                }
            ),
            adapter.surface_spec(),
        )
        case = EvalCase(id="case-1", split="dev", input="", expected={"reward": 1.0})

        record = adapter.run_case(case, candidate)

        self.assertEqual(env.actions, [("lookup_order", {"order_id": "A1"}), ("respond", {"content": "The order is delivered."})])
        trace = record.diagnostics.metadata["transform_trace"]
        self.assertTrue(any(item["op"] == "validate" and item["result"] == "failed" for item in trace))
        self.assertTrue(any(item["op"] == "replan" for item in trace))

    def test_tool_description_rewrite_changes_model_presentation_only(self) -> None:
        env = _FakeEnvironment()
        client = _ToolDescriptionClient()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=client,
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "tool_description",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "rewrite_tool_description",
                            "tool": "lookup_order",
                            "append": "Use after the user supplies an order id.",
                        }
                    ],
                }
            ),
            adapter.surface_spec(),
        )
        case = EvalCase(id="case-1", split="dev", input="", expected={"reward": 1.0})

        adapter.run_case(case, candidate)

        self.assertIn("Use after the user supplies an order id.", client.tool_descriptions[0])
        self.assertEqual(env.actions, [("respond", {"content": "The order is delivered."})])

    def test_taubench_sample_uses_generic_tool_loop_adapter(self) -> None:
        from samples.taubench_agent.ratchet_adapter import adapter

        self.assertIsInstance(adapter, GeneratedToolLoopAdapter)
        surface = adapter.surface_spec()
        self.assertTrue(surface.hooks["before_tool_call"].supported)
        self.assertTrue(surface.hooks["after_tool_result"].supported)


if __name__ == "__main__":
    unittest.main()
