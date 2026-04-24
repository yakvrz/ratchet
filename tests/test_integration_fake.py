from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ratchet.__main__ import run_optimizer
from ratchet.adapters import load_adapter
from ratchet.io import load_eval_cases
from ratchet.optimizer import RatchetOptimizer
from tests.fixtures.fake_adapter import adapter as fake_adapter


class FakeClient:
    def create_response(self, **_: object) -> object:
        class Response:
            output_text = "not-json"

        return Response()


class FakeAdapterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_adapter.reset()

    def write_evals(self, directory: Path) -> Path:
        evals_path = directory / "fake_evals.jsonl"
        rows = [
            {"id": "dev-1", "split": "dev", "input": "policy", "expected": "policy", "metadata": {"needs_tool": False}},
            {"id": "dev-2", "split": "dev", "input": "math", "expected": "math", "metadata": {"needs_tool": True}},
            {"id": "hold-1", "split": "holdout", "input": "policy", "expected": "policy", "metadata": {"needs_tool": False}},
            {"id": "hold-2", "split": "holdout", "input": "math", "expected": "math", "metadata": {"needs_tool": True}},
        ]
        evals_path.write_text("\n".join(json.dumps(row) for row in rows))
        return evals_path

    def test_run_optimizer_loads_adapter_evals_and_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            out_dir = root / "run"
            run_optimizer(
                adapter_spec="tests.fixtures.fake_adapter:adapter",
                evals_path=evals_path,
                out_dir=out_dir,
                env_file=".env",
                dev_budget=6,
                holdout_top_k=3,
                harnesser_model="gpt-5.4-mini",
                harnesser_reasoning="low",
                harnesser_enabled=False,
            )
            self.assertTrue((out_dir / "case_results.jsonl").exists())
            self.assertTrue((out_dir / "candidate_metrics.json").exists())
            self.assertTrue((out_dir / "decision_log.json").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "optimized_candidate.json").exists())
            self.assertTrue((out_dir / "exported_candidate" / "candidate.json").exists())
            report = (out_dir / "report.md").read_text()
            self.assertIn("## Run Health", report)
            self.assertIn("## Selected Candidate", report)

    def test_runtime_error_is_persisted_and_resume_retries_failed_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            out_dir = root / "run"
            fake_adapter.always_fail_case_id = "dev-1"
            run_optimizer(
                adapter_spec="tests.fixtures.fake_adapter:adapter",
                evals_path=evals_path,
                out_dir=out_dir,
                env_file=".env",
                dev_budget=4,
                holdout_top_k=2,
                harnesser_model="gpt-5.4-mini",
                harnesser_reasoning="low",
                harnesser_enabled=False,
            )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertGreater(manifest["stats"]["runtime_errors"], 0)
            case_lines_after_failure = (out_dir / "case_results.jsonl").read_text().splitlines()

            fake_adapter.reset()
            run_optimizer(
                adapter_spec="tests.fixtures.fake_adapter:adapter",
                evals_path=evals_path,
                out_dir=out_dir,
                env_file=".env",
                dev_budget=4,
                holdout_top_k=2,
                harnesser_model="gpt-5.4-mini",
                harnesser_reasoning="low",
                harnesser_enabled=False,
            )
            case_lines_after_resume = (out_dir / "case_results.jsonl").read_text().splitlines()
            self.assertGreater(len(case_lines_after_resume), len(case_lines_after_failure))

    def test_fail_fast_stops_after_error_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            out_dir = root / "run"
            fake_adapter.always_fail_case_id = "dev-1"
            with self.assertRaises(RuntimeError):
                run_optimizer(
                    adapter_spec="tests.fixtures.fake_adapter:adapter",
                    evals_path=evals_path,
                    out_dir=out_dir,
                    env_file=".env",
                    dev_budget=4,
                    holdout_top_k=2,
                    harnesser_model="gpt-5.4-mini",
                    harnesser_reasoning="low",
                    harnesser_enabled=False,
                    fail_fast=True,
                )

    def test_timeout_and_grader_error_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            fake_adapter.sleep_case_id = "dev-1"
            fake_adapter.sleep_seconds = 1.5
            fake_adapter.bad_grade_case_id = "hold-1"
            optimizer = RatchetOptimizer(
                adapter=adapter,
                search_space=adapter.search_space(),
                out_dir=root / "run",
                env_path=".env",
                dev_budget=4,
                holdout_top_k=2,
                harnesser_enabled=False,
                max_case_retries=1,
                case_timeout_s=1,
            )
            optimizer.run(cases)
            manifest = json.loads((root / "run" / "run_manifest.json").read_text())
            self.assertGreaterEqual(manifest["stats"]["timeouts"], 1)
            self.assertGreaterEqual(manifest["stats"]["grader_errors"], 1)
            self.assertGreaterEqual(manifest["stats"]["retries"], 2)

    def test_invalid_llm_output_falls_back_to_heuristic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            optimizer = RatchetOptimizer(
                adapter=adapter,
                search_space=adapter.search_space(),
                out_dir=root / "run",
                env_path=".env",
                dev_budget=4,
                holdout_top_k=2,
                harnesser_enabled=True,
            )
            optimizer.diagnoser._client = FakeClient()
            optimizer.proposer._client = FakeClient()
            result = optimizer.run(cases)
            self.assertTrue(
                any(
                    "heuristic structural proposals" in event["proposal_analysis"]
                    for event in result.decision_log
                    if event["type"] == "proposal_iteration"
                )
            )


if __name__ == "__main__":
    unittest.main()
