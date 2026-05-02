from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ratchet.ideation_benchmark import IdeationAssessmentSpec, assess_ideation_run, write_ideation_assessment


class IdeationBenchmarkTests(unittest.TestCase):
    def test_assessment_reads_run_artifacts_without_affecting_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "selected_candidate_id": "patch-1",
                        "promoted": True,
                        "finalist_statuses": [{"candidate_id": "patch-1", "status": "validated"}],
                        "run_cost": {"total_cost_usd": 0.12, "optimizer_tokens": 1000},
                    }
                )
            )
            (root / "candidate_metrics.json").write_text(
                json.dumps(
                    {
                        "baseline_holdout": {"behavioral": {"mean_score": 0.5}},
                        "selected_holdout": {"behavioral": {"mean_score": 0.75}},
                    }
                )
            )
            (root / "ideation_metrics.json").write_text(
                json.dumps(
                    {
                        "implementer": {
                            "valid_implementation_rate": 1.0,
                            "raw_candidate_count": 1,
                            "valid_candidate_count": 1,
                        },
                        "planner": {"brief_mechanisms": {"surface_context": 1}},
                    }
                )
            )
            (root / "proposals.jsonl").write_text(
                json.dumps(
                    {
                        "candidate": {"experiment_id": "intent-1"},
                        "candidate_id": "patch-1",
                        "surface_mechanism": "surface_context",
                        "mechanism_class": "surface_context",
                        "accepted": True,
                        "full_dev_evaluated": True,
                        "comparison_to_parent": {"score_delta": 0.25},
                    }
                )
                + "\n"
            )
            (root / "search_plans.jsonl").write_text(
                json.dumps(
                    {
                        "search_plan": {
                            "briefs": [
                                {"brief_id": "brief-1", "mechanism_class": "surface_context"}
                            ],
                        }
                    }
                )
                + "\n"
            )

            assessment = assess_ideation_run(
                root,
                spec=IdeationAssessmentSpec(
                    task_id="fake",
                    mechanisms_of_interest=["surface_context"],
                    pivotal_mechanisms=["surface_context"],
                    min_valid_implementation_rate=0.9,
                    min_holdout_score_delta=0.1,
                ),
            )

            self.assertTrue(all(assessment["checks"].values()))
            self.assertEqual(assessment["summary"]["validated_candidate_count"], 1)
            self.assertEqual(assessment["summary"]["selected_holdout_score_delta"], 0.25)

            written = write_ideation_assessment(root)
            self.assertTrue((root / "ideation_assessment.json").exists())
            self.assertEqual(written["run_dir"], str(root))


if __name__ == "__main__":
    unittest.main()
