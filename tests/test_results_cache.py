from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ratchet.results import CaseEvaluation, ResultStore
from ratchet.types import EvalCase, GradeResult, OperationalMetrics, RunRecord


def _evaluation(case: EvalCase) -> CaseEvaluation:
    return CaseEvaluation(
        case=case,
        record=RunRecord(
            output="ok",
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.01,
            ),
        ),
        grade=GradeResult(score=1.0, passed=True),
    )


class ResultStoreCacheTests(unittest.TestCase):
    def test_shared_cache_reuses_rows_across_run_directories(self) -> None:
        case = EvalCase(id="case-1", split="dev", input="x", expected="ok")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / ".ratchet" / "cache" / "case_results.jsonl"
            first = ResultStore(root / "run-a", cache_namespace="same", shared_cache_path=shared)
            first.put("baseline", None, _evaluation(case))

            second = ResultStore(root / "run-b", cache_namespace="same", shared_cache_path=shared)
            cached = second.get("baseline", case, candidate=None)

            self.assertIsNotNone(cached)
            assert cached is not None
            self.assertTrue(cached.cached)
            self.assertEqual(cached.cache_source, "shared")
            local_rows = (root / "run-b" / "case_results.jsonl").read_text()
            self.assertIn('"cache_source": "shared"', local_rows)

    def test_shared_cache_namespace_mismatch_does_not_reuse_rows(self) -> None:
        case = EvalCase(id="case-1", split="dev", input="x", expected="ok")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / ".ratchet" / "cache" / "case_results.jsonl"
            first = ResultStore(root / "run-a", cache_namespace="first", shared_cache_path=shared)
            first.put("baseline", None, _evaluation(case))

            second = ResultStore(root / "run-b", cache_namespace="second", shared_cache_path=shared)

            self.assertIsNone(second.get("baseline", case, candidate=None))

    def test_run_local_namespace_mismatch_fails_fast(self) -> None:
        case = EvalCase(id="case-1", split="dev", input="x", expected="ok")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ResultStore(root / "run", cache_namespace="first")
            store.put("baseline", None, _evaluation(case))

            with self.assertRaisesRegex(ValueError, "Run-local cache namespace mismatch"):
                ResultStore(root / "run", cache_namespace="second")


if __name__ == "__main__":
    unittest.main()
