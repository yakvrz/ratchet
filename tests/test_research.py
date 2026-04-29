from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.experiments import ResearchState
from ratchet.research import ResearchPlanner


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _Client:
    def create_response(self, **_: object) -> _Response:
        return _Response(
            '{"experiment_intents":[{"intent_id":"intent_1",'
            '"mechanism_class":"semantic_boundary_rewrite",'
            '"hypothesis":"test","affordance_ids":["missing"]}]}'
        )


class ResearchRoleTests(unittest.TestCase):
    def test_research_planner_rejects_unknown_affordance_id(self) -> None:
        planner = ResearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        planner._client = _Client()
        state = ResearchState(
            objective={},
            budget={},
            parent={},
            task_theory={},
            behavior_profile={},
            affordances=[{"affordance_id": "known"}],
            prior_experiment_outcomes=[],
            frontier={},
        )

        with self.assertRaises(OptimizerModelError):
            planner.plan(state=state)


if __name__ == "__main__":
    unittest.main()
