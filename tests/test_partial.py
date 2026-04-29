from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ratchet.partial import write_partial_run_outputs


class PartialRunOutputTests(unittest.TestCase):
    def test_partial_outputs_identify_incomplete_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            progress_path = out_dir / "progress.jsonl"
            progress_path.write_text(
                "\n".join(
                    [
                        json.dumps({"event": "run_started", "elapsed_s": 0.0}),
                        json.dumps(
                            {
                                "event": "case_started",
                                "elapsed_s": 1.0,
                                "candidate_id": "patch-1",
                                "case_id": "case-1",
                                "sample_index": 0,
                                "split": "dev",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "case_started",
                                "elapsed_s": 2.0,
                                "candidate_id": "patch-1",
                                "case_id": "case-2",
                                "sample_index": 0,
                                "split": "dev",
                            }
                        ),
                        json.dumps(
                            {
                                "event": "case_completed",
                                "elapsed_s": 3.0,
                                "candidate_id": "patch-1",
                                "case_id": "case-1",
                                "sample_index": 0,
                                "split": "dev",
                            }
                        ),
                    ]
                )
                + "\n"
            )
            (out_dir / "case_results.jsonl").write_text(json.dumps({"candidate_id": "patch-1"}) + "\n")

            manifest = write_partial_run_outputs(out_dir, status="failed", reason="test failure")

            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["case_result_count"], 1)
            self.assertEqual(len(manifest["incomplete_cases"]), 1)
            self.assertEqual(manifest["incomplete_cases"][0]["case_id"], "case-2")
            self.assertTrue((out_dir / "partial_run_manifest.json").exists())
            self.assertIn("case=`case-2`", (out_dir / "partial_report.md").read_text())


if __name__ == "__main__":
    unittest.main()
