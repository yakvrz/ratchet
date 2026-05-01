from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.research import SearchPlanner


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _Client:
    def create_response(self, **_: object) -> _Response:
        return _Response(
            '{"plan_id":"P1","diagnosis":"test","hypotheses":["h"],'
            '"target_mechanisms":["surface_context"],'
            '"briefs":[{"brief_id":"B1","mechanism_class":"surface_context",'
            '"hypothesis":"test","surface_opportunity_ids":["missing"]}]}'
        )


class _RepairClient:
    def __init__(self) -> None:
        self.calls = 0

    def create_response(self, **_: object) -> _Response:
        self.calls += 1
        if self.calls == 1:
            return _Response('{"briefs":')
        return _Response(
            '{"plan_id":"P1","diagnosis":"test","hypotheses":["h"],'
            '"target_mechanisms":["surface_context"],'
            '"briefs":[{"brief_id":"B1","mechanism_class":"surface_context",'
            '"hypothesis":"test","surface_opportunity_ids":["known"]}]}'
        )


class _AliasClient:
    def create_response(self, **_: object) -> _Response:
        return _Response(
            """{
              "summary":"The baseline is failing before using the tool-loop surface.",
              "hypotheses":["The agent is not validating tool calls against observations."],
              "candidate_briefs":[{
                "id":"tool_loop_brief",
                "mechanism":"surface_tool_loop",
                "description":"Measure whether before-tool-call validation improves reliability.",
                "surface_opportunity_ids":["tool_loop"]
              }]
            }"""
        )


class SearchPlannerTests(unittest.TestCase):
    def test_rejects_unknown_surface_opportunity_id(self) -> None:
        planner = SearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        planner._client = _Client()

        with self.assertRaises(OptimizerModelError):
            planner.plan(state={}, surface_opportunity_ids={"known"})

    def test_repairs_schema_invalid_payload(self) -> None:
        planner = SearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        client = _RepairClient()
        planner._client = client

        plan = planner.plan(state={}, surface_opportunity_ids={"known"})

        self.assertEqual(client.calls, 2)
        self.assertEqual([brief.brief_id for brief in plan.briefs], ["B1"])
        self.assertTrue(planner.last_call_diagnostics and planner.last_call_diagnostics.get("repair_attempted"))

    def test_normalizes_internal_ids_and_aliases(self) -> None:
        planner = SearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        planner._client = _AliasClient()

        plan = planner.plan(state={}, surface_opportunity_ids={"tool_loop"})

        self.assertEqual(plan.plan_id, "P_001")
        self.assertEqual(plan.diagnosis, "The baseline is failing before using the tool-loop surface.")
        self.assertEqual(plan.briefs[0].brief_id, "tool_loop_brief")
        self.assertEqual(plan.briefs[0].mechanism_class, "surface_tool_loop")


if __name__ == "__main__":
    unittest.main()
