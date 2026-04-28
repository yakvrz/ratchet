from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.research import ResearchAction, ResearchDecision, validate_research_decision


class ResearchControllerTests(unittest.TestCase):
    def test_validate_research_decision_accepts_known_candidate_ids(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            stage="full_dev",
            candidate_ids=["a", "b"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["a"],
            skipped_candidate_reasons={"b": "lower information value"},
        )

        self.assertEqual(validate_research_decision(decision, [action]), action)

    def test_validate_research_decision_rejects_unknown_action(self) -> None:
        action = ResearchAction(action_id="evaluate_full_dev", action_type="evaluate_candidates")
        decision = ResearchDecision(action_id="stop", action_type="stop")

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_validate_research_decision_rejects_unknown_candidate(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["b"],
            skipped_candidate_reasons={"a": "not selected"},
        )

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_validate_research_decision_rejects_over_selection(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a", "b"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["a", "b"],
        )

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_validate_research_decision_requires_skip_reasons(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a", "b"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["a"],
        )

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])


if __name__ == "__main__":
    unittest.main()
