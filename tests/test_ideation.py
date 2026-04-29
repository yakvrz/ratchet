from __future__ import annotations

import unittest

from ratchet.ideation import build_ideation_metrics


class IdeationMetricsTests(unittest.TestCase):
    def test_counts_valid_current_proposal_rows_without_type_marker(self) -> None:
        metrics = build_ideation_metrics(
            decision_log=[
                {
                    "type": "research_plan",
                    "experiment_intents": [
                        {
                            "intent_id": "intent_1",
                            "mechanism_class": "semantic_boundary_rewrite",
                            "affordance_ids": ["aff_prompt"],
                        }
                    ],
                }
            ],
            proposals=[
                {
                    "candidate": {"experiment_id": "intent_1"},
                    "transform_family": "prompt_rewrite",
                    "mechanism_class": "semantic_boundary_rewrite",
                    "accepted": True,
                    "frontier_status": "promotable",
                },
                {
                    "type": "candidate_proposal",
                    "valid": False,
                    "invalid_reason": "unknown affordance_id",
                },
            ],
            finalist_statuses=[],
        )

        self.assertEqual(metrics["implementer"]["raw_candidate_count"], 2)
        self.assertEqual(metrics["implementer"]["valid_candidate_count"], 1)
        self.assertEqual(metrics["implementer"]["invalid_candidate_count"], 1)
        self.assertEqual(metrics["implementer"]["implemented_intent_count"], 1)
        self.assertEqual(metrics["discovery"]["stage_counts"]["promotable_dev"], 1)


if __name__ == "__main__":
    unittest.main()
