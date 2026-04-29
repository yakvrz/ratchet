from __future__ import annotations

import unittest

from ratchet.affordances import generate_optimization_affordances
from ratchet.surfaces import surface_from_agent_spec
from ratchet.types import AgentSpec, AgentTool, OptimizationObjective


class OptimizationAffordanceTests(unittest.TestCase):
    def test_generation_uses_surface_spec_targets(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
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


if __name__ == "__main__":
    unittest.main()
