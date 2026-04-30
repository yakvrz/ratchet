from __future__ import annotations

import unittest

from ratchet.affordances import generate_optimization_affordances
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

        opportunities = generate_optimization_affordances(surface)

        ids = {item.affordance_id for item in opportunities}
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

        opportunities = generate_optimization_affordances(
            surface,
            active_families=["surface_tool_loop"],
            evidence={"tool_trajectory_defect": True},
        )

        tool = next(item for item in opportunities if item.target_name == "refund_order")
        self.assertEqual(tool.family, "surface_program")
        self.assertEqual(tool.mechanism, "surface_tool_loop")

    def test_tool_loop_surface_exposes_generic_tool_affordance_without_static_tools(self) -> None:
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

        opportunities = generate_optimization_affordances(
            surface,
            active_families=["surface_tool_loop"],
            evidence={"tool_trajectory_defect": True},
        )

        tool_loop = [item for item in opportunities if item.target_name == "tool_loop"]
        self.assertTrue(tool_loop)
        self.assertIn("validate", tool_loop[0].ops)
        self.assertIn("before_tool_call", tool_loop[0].value_schema["hooks"])


if __name__ == "__main__":
    unittest.main()
