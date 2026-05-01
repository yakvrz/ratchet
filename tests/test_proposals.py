from __future__ import annotations

import unittest

from ratchet.experiments import ExperimentIntent
from ratchet.proposals import _surface_affordance_proposals
from ratchet.surface_opportunities import generate_surface_opportunities
from ratchet.surfaces import tool_loop_surface_from_agent_spec
from ratchet.types import AgentSpec


class SurfaceAffordanceProposalTests(unittest.TestCase):
    def test_identifier_flow_affordance_emits_composed_state_guard_candidate(self) -> None:
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
                            "parameters": {"type": "object"},
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
                    }
                },
            },
        )
        opportunities = generate_surface_opportunities(surface, active_mechanisms=["surface_tool_loop"])
        affordance_id = "surface.surface_tool_loop.inspect_before_mutate_order_id"
        intent = ExperimentIntent(
            intent_id="intent_tool_loop",
            mechanism_class="surface_tool_loop",
            hypothesis="Ground mutating calls in observed identifiers.",
            surface_opportunity_ids=[affordance_id],
            candidate_roles=["composed"],
        )

        proposals = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=opportunities,
            experiment_intents=[intent],
        )

        self.assertEqual(len(proposals), 1)
        candidate = proposals[0]
        self.assertEqual(candidate.experiment_id, "intent_tool_loop")
        self.assertEqual(candidate.candidate_role, "composed")
        patches = [patch.to_dict() for patch in candidate.program.patches]
        self.assertIn(
            {
                "hook": "after_tool_result",
                "op": "append_state",
                "field": "observed_order_ids",
                "value": {"$ref": "tool_result.parsed.orders[].order_id"},
                "extend": True,
                "when": {"tool_call.name": "list_orders"},
            },
            patches,
        )
        self.assertTrue(
            any(
                patch.get("op") == "validate"
                and patch.get("tool") == "cancel_order"
                and patch.get("checks") == [
                    {"type": "tool_arg_in_state", "state_field": "observed_order_ids", "arg": "order_id"}
                ]
                for patch in patches
            )
        )


if __name__ == "__main__":
    unittest.main()
