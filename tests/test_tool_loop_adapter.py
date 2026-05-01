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

    def __init__(self, tools_info: list[dict[str, object]] | None = None) -> None:
        self.actions: list[tuple[str, dict[str, object]]] = []
        if tools_info is not None:
            self.tools_info = tools_info

    def reset(self, task_index: int | None = None) -> _Response:
        return _Response("Need status for order A1.")

    def step(self, action: dict[str, object]) -> _Response:
        self.actions.append((str(action["name"]), dict(action["kwargs"])))
        if action["name"] == "lookup_order":
            return _Response('{"status": "success", "order": {"order_id": "A1", "status": "delivered"}}')
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


class _CaptureToolThenResponseClient:
    def __init__(self) -> None:
        self.calls = 0
        self.system_prompts: list[str] = []

    def complete(self, **kwargs: object) -> ToolLoopModelResponse:
        self.calls += 1
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.system_prompts.append(str(messages[0]["content"]))
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


class _TwoToolClient:
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
                            "function": {"name": "lookup_order", "arguments": '{"order_id": "A1"}'},
                        }
                    ],
                },
                input_tokens=10,
                output_tokens=5,
            )
        if self.calls == 2:
            return ToolLoopModelResponse(
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "other_tool", "arguments": '{"order_id": "Z999"}'},
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


class _CaptureSystemClient:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def complete(self, **kwargs: object) -> ToolLoopModelResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.system_prompts.append(str(messages[0]["content"]))
        return ToolLoopModelResponse(
            message={"role": "assistant", "content": "The order is delivered."},
            input_tokens=12,
            output_tokens=6,
        )


class ToolLoopAdapterTests(unittest.TestCase):
    def _case(self) -> EvalCase:
        return EvalCase(id="case-1", split="dev", input="", expected={"reward": 1.0})

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
            adapter.surface_spec((self._case(),)),
        )
        case = self._case()

        record = adapter.run_case(case, candidate)
        grade = adapter.grade(case, record.output)

        self.assertTrue(grade.passed)
        self.assertEqual(env.actions[0], ("lookup_order", {"order_id": "A1"}))
        self.assertEqual(record.metrics.model_calls, 2)
        self.assertEqual(record.metrics.tool_calls, 1)
        trace_ops = [item["op"] for item in record.diagnostics.metadata["transform_trace"]]
        self.assertIn("normalize_tool_args", trace_ops)
        self.assertIn("append_state", trace_ops)

    def test_tool_loop_state_can_use_parsed_tool_result_and_context_templates(self) -> None:
        env = _FakeEnvironment()
        client = _CaptureToolThenResponseClient()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(
                name="fake-tool-loop",
                model="gpt-4o",
                model_options=["gpt-4o"],
                tools={"lookup_order": AgentTool(name="lookup_order", metadata={"side_effect": "read"})},
            ),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=client,
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "parsed_state",
                    "patches": [
                        {"op": "define_state", "field": "last_status", "type": "string", "initial": "unknown"},
                        {
                            "hook": "after_tool_result",
                            "op": "set_state",
                            "field": "last_status",
                            "value": {"$ref": "tool_result.parsed.status"},
                        },
                        {
                            "hook": "before_model_call",
                            "op": "add_context_section",
                            "section": "status_state",
                            "content": "Last observed status: {{state.last_status}}",
                        },
                    ],
                }
            ),
            adapter.surface_spec((self._case(),)),
        )

        adapter.run_case(self._case(), candidate)

        self.assertEqual(len(client.system_prompts), 2)
        self.assertIn("Last observed status: unknown", client.system_prompts[0])
        self.assertIn("Last observed status: success", client.system_prompts[1])

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
            adapter.surface_spec((self._case(),)),
        )
        case = self._case()

        record = adapter.run_case(case, candidate)

        self.assertEqual(env.actions, [("lookup_order", {"order_id": "A1"}), ("respond", {"content": "The order is delivered."})])
        trace = record.diagnostics.metadata["transform_trace"]
        self.assertTrue(any(item["op"] == "validate" and item["result"] == "failed" for item in trace))
        self.assertTrue(any(item["op"] == "replan" for item in trace))

    def test_before_tool_call_patch_can_target_specific_tool(self) -> None:
        env = _FakeEnvironment(
            tools_info=[
                *_FakeEnvironment.tools_info,
                {
                    "type": "function",
                    "function": {
                        "name": "other_tool",
                        "description": "Another tool.",
                        "parameters": {
                            "type": "object",
                            "properties": {"order_id": {"type": "string"}},
                            "required": ["order_id"],
                        },
                    },
                },
            ]
        )
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(
                name="fake-tool-loop",
                model="gpt-4o",
                model_options=["gpt-4o"],
                tools={
                    "lookup_order": AgentTool(name="lookup_order", metadata={"side_effect": "read"}),
                    "other_tool": AgentTool(name="other_tool", metadata={"side_effect": "read"}),
                },
            ),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_TwoToolClient(),
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "tool_specific_guard",
                    "patches": [
                        {
                            "hook": "before_tool_call",
                            "op": "validate",
                            "tool": "other_tool",
                            "target": "tool_call",
                            "checks": [{"type": "referenced_args_observed"}],
                            "on_fail": {
                                "hook": "before_tool_call",
                                "op": "replan",
                                "content": "Use observed identifiers only.",
                            },
                        }
                    ],
                }
            ),
            adapter.surface_spec((self._case(),)),
        )

        record = adapter.run_case(self._case(), candidate)

        trace = record.diagnostics.metadata["transform_trace"]
        self.assertTrue(any(item["op"] == "validate" and item["result"] == "skipped_tool" for item in trace))
        self.assertTrue(any(item["op"] == "validate" and item["result"] == "failed" for item in trace))
        self.assertTrue(any(item["op"] == "replan" for item in trace))
        self.assertTrue(any(turn.message == "Use observed identifiers only." for turn in record.diagnostics.turns))

    def test_tool_loop_validation_accepts_structured_schema_check(self) -> None:
        env = _FakeEnvironment()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_FakeClient(),
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "schema_guard",
                    "patches": [
                        {
                            "hook": "before_tool_call",
                            "op": "validate",
                            "target": "tool_call",
                            "checks": [{"type": "args_schema_valid"}],
                            "on_fail": {
                                "hook": "before_tool_call",
                                "op": "replan",
                                "message": "Repair the tool arguments before continuing.",
                            },
                        }
                    ],
                }
            ),
            adapter.surface_spec((self._case(),)),
        )

        record = adapter.run_case(self._case(), candidate)

        self.assertEqual(env.actions[0], ("lookup_order", {"order_id": " A1 "}))
        trace = record.diagnostics.metadata["transform_trace"]
        self.assertTrue(any(item["op"] == "validate" and item["result"] == "passed" for item in trace))

    def test_tool_loop_schema_validation_rejects_wrong_argument_type(self) -> None:
        class BadArgClient:
            def complete(self, **kwargs: object) -> ToolLoopModelResponse:
                return ToolLoopModelResponse(
                    message={
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_bad",
                                "type": "function",
                                "function": {"name": "lookup_order", "arguments": '{"order_id": 123}'},
                            }
                        ],
                    },
                    input_tokens=10,
                    output_tokens=5,
                )

        env = _FakeEnvironment()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: env,
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=BadArgClient(),
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "schema_guard",
                    "patches": [
                        {
                            "hook": "before_tool_call",
                            "op": "validate",
                            "target": "tool_call",
                            "checks": [{"type": "args_schema_valid"}],
                            "on_fail": {
                                "hook": "before_tool_call",
                                "op": "terminate",
                                "message": "Invalid tool arguments.",
                            },
                        }
                    ],
                }
            ),
            adapter.surface_spec((self._case(),)),
        )

        record = adapter.run_case(self._case(), candidate)

        self.assertEqual(env.actions, [])
        trace = record.diagnostics.metadata["transform_trace"]
        self.assertTrue(any(item["op"] == "validate" and item["result"] == "failed" for item in trace))

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
            adapter.surface_spec((self._case(),)),
        )
        case = self._case()

        adapter.run_case(case, candidate)

        self.assertIn("Use after the user supplies an order id.", client.tool_descriptions[0])
        self.assertEqual(env.actions, [("respond", {"content": "The order is delivered."})])

    def test_taubench_sample_uses_generic_tool_loop_adapter(self) -> None:
        from samples.taubench_agent.ratchet_adapter import adapter

        self.assertIsInstance(adapter, GeneratedToolLoopAdapter)

    def test_tool_loop_surface_probe_infers_context_and_tools(self) -> None:
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: _FakeEnvironment(),
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_FakeClient(),
        )
        surface = adapter.surface_spec((self._case(),))

        section_names = surface.context.graph.section_names()
        self.assertIn("domain_policy", section_names)
        self.assertIn("tool_instructions", section_names)
        self.assertIn("recent_messages", section_names)
        self.assertTrue(surface.tools.tools_available)
        before_tool_call = surface.hooks["before_tool_call"].to_dict()
        check_names = {item["type"] for item in before_tool_call["validation_checks"]}
        self.assertIn("args_schema_valid", check_names)
        self.assertIn("not_duplicate_tool_call", check_names)
        self.assertEqual(surface.tools.tools[0].name, "lookup_order")
        self.assertEqual(surface.tools.tools[0].metadata["side_effect"], "read")

    def test_tool_loop_fingerprint_includes_environment_source(self) -> None:
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: _FakeEnvironment(),
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_FakeClient(),
        )

        fingerprint = adapter.fingerprint()

        self.assertEqual(fingerprint["adapter_kind"], "generated_tool_loop")
        self.assertIn("source_tree_sha256", fingerprint["environment_factory"])
        self.assertTrue(fingerprint["environment_factory"]["source_tree_sha256"])

    def test_tool_loop_runtime_uses_inferred_surface_context_graph(self) -> None:
        client = _CaptureSystemClient()
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: _FakeEnvironment(),
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=client,
        )
        surface = adapter.surface_spec((self._case(),))
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "replace_domain_policy",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "replace_context_section",
                            "section": "domain_policy",
                            "content": "Replacement policy text.",
                        }
                    ],
                }
            ),
            surface,
        )

        record = adapter.run_case(self._case(), candidate)

        self.assertIsNone(record.metrics.error)
        self.assertTrue(client.system_prompts)
        self.assertIn("[domain_policy]\nReplacement policy text.", client.system_prompts[0])
        self.assertNotIn("Use tools to inspect state before responding.", client.system_prompts[0])

    def test_tool_metadata_inference_uses_tool_action_not_return_text(self) -> None:
        adapter = GeneratedToolLoopAdapter(
            agent_spec=AgentSpec(name="fake-tool-loop", model="gpt-4o", model_options=["gpt-4o"]),
            environment_factory=lambda case, config: _FakeEnvironment(
                tools_info=[
                    {
                        "type": "function",
                        "function": {
                            "name": "find_user_id_by_email",
                            "description": "Find user id by email. If the user is not found, the function will return an error message.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "return_delivered_order_items",
                            "description": "Return some items of a delivered order.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    },
                ]
            ),
            action_factory=lambda name, args: {"name": name, "kwargs": args},
            client=_FakeClient(),
        )

        tools = {tool.name: tool for tool in adapter.surface_spec((self._case(),)).tools.tools}

        self.assertEqual(tools["find_user_id_by_email"].metadata["side_effect"], "read")
        self.assertEqual(tools["return_delivered_order_items"].metadata["side_effect"], "mutating")


if __name__ == "__main__":
    unittest.main()
