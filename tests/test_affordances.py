from __future__ import annotations

import unittest

from ratchet.affordances import generate_optimization_affordances
from ratchet.surfaces import surface_from_agent_spec, tool_loop_surface_from_agent_spec
from ratchet.types import AgentSpec, AgentTool, OptimizationObjective


class OptimizationAffordanceTests(unittest.TestCase):
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

        affordances = generate_optimization_affordances(surface)

        ids = {item.affordance_id for item in affordances}
        self.assertTrue(any(item.target_name == "system_prompt" for item in affordances))
        self.assertTrue(any(item.target_name == "output_contract" for item in affordances))
        self.assertIn("model_substitution.model_capability_probe.generic_policy.model_config", ids)
        model_affordance = next(item for item in affordances if item.family == "model_substitution")
        self.assertEqual(model_affordance.value_schema["current_model"], "base")
        self.assertEqual(model_affordance.value_schema["model_name"]["allowed_values"], ["base", "larger"])

    def test_tool_trajectory_evidence_activates_tool_policy_mechanism(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                tools={"refund_order": AgentTool(name="refund_order", description="Refund an order.")},
            )
        )

        affordances = generate_optimization_affordances(
            surface,
            active_families=["tool_policy_revision"],
            evidence={"tool_trajectory_defect": True},
        )

        self.assertEqual(affordances[0].family, "tool_policy_revision")
        self.assertEqual(affordances[0].mechanism, "tool_selection_policy")

    def test_tool_loop_surface_exposes_generic_tool_affordance_without_static_tools(self) -> None:
        surface = tool_loop_surface_from_agent_spec(AgentSpec(name="interactive", model="base"))

        affordances = generate_optimization_affordances(
            surface,
            active_families=["tool_policy_revision"],
            evidence={"tool_trajectory_defect": True},
        )

        tool_loop = [item for item in affordances if item.target_name == "tool_loop"]
        self.assertTrue(tool_loop)
        self.assertIn("validate", tool_loop[0].ops)
        self.assertIn("before_tool_call", tool_loop[0].value_schema["hooks"])


if __name__ == "__main__":
    unittest.main()
