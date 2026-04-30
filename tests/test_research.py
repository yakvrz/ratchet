from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.experiments import ResearchState
from ratchet.research import ResearchPlanner, ResearchTheorist


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _Client:
    def create_response(self, **_: object) -> _Response:
        return _Response(
            '{"experiment_intents":[{"intent_id":"intent_1",'
            '"mechanism_class":"surface_context",'
            '"hypothesis":"test","surface_opportunity_ids":["missing"]}]}'
        )


class _RepairClient:
    def __init__(self) -> None:
        self.calls = 0

    def create_response(self, **_: object) -> _Response:
        self.calls += 1
        if self.calls == 1:
            return _Response('{"experiment_intents":{"intent_id":"bad_shape"}}')
        return _Response(
            '{"experiment_intents":[{"intent_id":"intent_1",'
            '"mechanism_class":"surface_context",'
            '"hypothesis":"test","surface_opportunity_ids":["known"]}]}'
        )


class _TheoryClient:
    def create_response(self, **_: object) -> _Response:
        return _Response(
            """{
              "summary":"The baseline is failing before using the available tool-loop surface.",
              "hypotheses":[{
                "hypothesis":"The agent is not validating tool calls against observations.",
                "mechanism":"surface_tool_loop",
                "target_slices":["tool tasks"]
              }],
              "opportunities":[{
                "description":"Measure whether a before-tool-call validator improves reliability.",
                "surface_opportunity_ids":["tool_loop"]
              }]
            }"""
        )


class ResearchRoleTests(unittest.TestCase):
    def test_research_planner_rejects_unknown_surface_opportunity_id(self) -> None:
        planner = ResearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        planner._client = _Client()
        state = ResearchState(
            objective={},
            budget={},
            parent={},
            research_theory={},
            behavior_profile={},
            surface_opportunities=[{"surface_opportunity_id": "known"}],
            prior_experiment_outcomes=[],
            frontier={},
        )

        with self.assertRaises(OptimizerModelError):
            planner.plan(state=state)

    def test_research_planner_repairs_schema_invalid_payload(self) -> None:
        planner = ResearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        client = _RepairClient()
        planner._client = client
        state = ResearchState(
            objective={},
            budget={},
            parent={},
            research_theory={},
            behavior_profile={},
            surface_opportunities=[{"surface_opportunity_id": "known"}],
            prior_experiment_outcomes=[],
            frontier={},
        )

        intents = planner.plan(state=state)

        self.assertEqual(client.calls, 2)
        self.assertEqual([intent.intent_id for intent in intents], ["intent_1"])
        self.assertTrue(planner.last_call_diagnostics and planner.last_call_diagnostics.get("repair_attempted"))

    def test_research_theorist_normalizes_internal_ids_and_aliases(self) -> None:
        theorist = ResearchTheorist(env_path=".env", model="fake", reasoning_effort="low")
        theorist._client = _TheoryClient()

        theory = theorist.build_theory(state={}, surface_opportunity_ids={"tool_loop"})

        self.assertEqual(theory.theory_id, "T_001")
        self.assertEqual(theory.primary_hypothesis_id, "H_001")
        self.assertEqual(theory.hypotheses[0].statement, "The agent is not validating tool calls against observations.")
        self.assertEqual(theory.hypotheses[0].mechanism_class, "surface_tool_loop")
        self.assertEqual(theory.experiment_opportunities[0].opportunity_id, "O_001")
        self.assertEqual(theory.experiment_opportunities[0].hypothesis_ids, ["H_001"])


if __name__ == "__main__":
    unittest.main()
