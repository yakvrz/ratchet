from __future__ import annotations

import unittest

from ratchet.surface_opportunities import generate_surface_opportunities
from ratchet.surfaces import surface_from_agent_spec, tool_loop_surface_from_agent_spec
from ratchet.types import AgentSpec, AgentTool


class SurfaceOpportunityTests(unittest.TestCase):
    def test_generation_uses_surface_spec_targets(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                model_options=["base", "larger"],
                instructions={"system_prompt": "Classify."},
                output_contract="Return JSON.",
                tools={"search": AgentTool(name="search", description="Search docs.")},
            )
        )

        opportunities = generate_surface_opportunities(surface)

        ids = {item.surface_opportunity_id for item in opportunities}
        self.assertTrue(any(item.target_name == "system_prompt" for item in opportunities))
        self.assertTrue(any(item.target_name == "output_contract" for item in opportunities))
        self.assertIn("surface.surface_model.model_config", ids)
        model_opportunity = next(item for item in opportunities if item.mechanism == "surface_model")
        self.assertEqual(model_opportunity.value_schema["current_model"], "base")
        self.assertEqual(model_opportunity.value_schema["model_name"]["allowed_values"], ["base", "larger"])

    def test_tool_surface_is_exposed_as_tool_loop_surface(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                tools={"refund_order": AgentTool(name="refund_order", description="Refund an order.")},
            )
        )

        opportunities = generate_surface_opportunities(
            surface,
            active_mechanisms=["surface_tool_loop"],
            evidence={"tool_trajectory_defect": True},
        )

        tool = next(item for item in opportunities if item.target_name == "refund_order")
        self.assertEqual(tool.family, "surface_program")
        self.assertEqual(tool.mechanism, "surface_tool_loop")

    def test_tool_loop_surface_exposes_generic_tool_surface_opportunity_without_static_tools(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Use tools before responding.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "Look up records.",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )

        opportunities = generate_surface_opportunities(
            surface,
            active_mechanisms=["surface_tool_loop"],
            evidence={"tool_trajectory_defect": True},
        )

        tool_loop = [item for item in opportunities if item.target_name == "tool_loop"]
        self.assertTrue(tool_loop)
        self.assertIn("validate", tool_loop[0].ops)
        self.assertIn("before_tool_call", tool_loop[0].value_schema["hooks"])

    def test_tool_loop_surface_derives_inspect_before_mutate_affordance_from_identifier_flow(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Inspect records before mutation.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_orders",
                            "description": "List orders.",
                            "parameters": {
                                "type": "object",
                                "properties": {"user_id": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "description": "Inspect one order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "cancel_order",
                            "description": "Cancel an order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        },
                    },
                ],
                "tool_result_schemas": {
                    "list_orders": {
                        "type": "object",
                        "properties": {
                            "orders": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"order_id": {"type": "string"}},
                                },
                            }
                        },
                    },
                    "get_order": {
                        "type": "object",
                        "properties": {
                            "order": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            }
                        },
                    },
                },
            },
        )

        opportunities = generate_surface_opportunities(
            surface,
            active_mechanisms=["surface_tool_loop"],
            evidence={"tool_trajectory_defect": True},
        )

        affordance = surface.affordances[0]
        self.assertEqual(affordance["identifier"], "order_id")
        self.assertEqual(affordance["inspected_state_field"], "inspected_order_ids")
        self.assertEqual(affordance["listed_state_field"], "listed_order_ids")
        producer_refs = {producer["ref"] for producer in affordance["produced_by"]}
        self.assertIn("tool_result.parsed.orders[].order_id", producer_refs)
        self.assertIn("tool_result.parsed.order.order_id", producer_refs)
        ids = {item.surface_opportunity_id for item in opportunities}
        self.assertIn("surface.surface_tool_loop.inspect_before_mutate_order_id", ids)


if __name__ == "__main__":
    unittest.main()
