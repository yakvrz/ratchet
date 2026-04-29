from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import textwrap
import unittest

from ratchet.config import EvalHealthConfig, load_run_config
from ratchet.eval_health import run_eval_health_check
from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, RunRecord


class StableAdapter:
    def agent_spec(self) -> AgentSpec:
        return AgentSpec(name="stable", model="local")

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        return RunRecord(
            output=case.expected,
            metrics=OperationalMetrics(
                latency_s=0.01,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.0,
            ),
            diagnostics=DiagnosticTrace(metadata={"model": "local-stable"}),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return GradeResult(score=1.0 if output == case.expected else 0.0, passed=output == case.expected)

    def export(self, candidate: CompiledCandidate | None, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)


class FlakyAdapter(StableAdapter):
    def __init__(self) -> None:
        self.calls = 0

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        self.calls += 1
        expected = dict(case.expected)
        output = expected if self.calls % 2 else {"label": "wrong"}
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=0.01,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.0,
            ),
        )


class HeavyAdapter(StableAdapter):
    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        return RunRecord(
            output=case.expected,
            metrics=OperationalMetrics(
                latency_s=12.0,
                input_tokens=1000,
                output_tokens=250,
                total_tokens=1250,
                cost_usd=0.75,
            ),
            diagnostics=DiagnosticTrace(metadata={"model": "expensive-model"}),
        )


def cases() -> tuple[EvalCase, ...]:
    return (
        EvalCase(id="train-a", split="train", input="same text", expected={"label": "a"}, metadata={"category": "a"}),
        EvalCase(id="dev-a", split="dev", input="same text", expected={"label": "a"}, metadata={"category": "a"}),
        EvalCase(id="dev-b", split="dev", input="dev b", expected={"label": "b"}, metadata={"category": "b"}),
        EvalCase(id="hold-a", split="holdout", input="hold a", expected={"label": "a"}, metadata={"category": "a"}),
        EvalCase(id="hold-b", split="holdout", input="hold b", expected={"label": "b"}, metadata={"category": "b"}),
    )


class EvalHealthTests(unittest.TestCase):
    def test_static_health_reports_leakage_and_small_holdout(self) -> None:
        report = run_eval_health_check(
            adapter_spec="tests.test_eval_health:adapter",
            adapter=StableAdapter(),
            cases=cases(),
            config=EvalHealthConfig(sample_limit=0, repeats=0, min_holdout_cases=5),
        )

        self.assertEqual(report.status, "warning")
        codes = {issue.code for issue in report.issues}
        self.assertIn("train_eval_leakage", codes)
        self.assertIn("holdout_split_too_small", codes)
        self.assertEqual(report.baseline_probe["checked"], False)

    def test_dynamic_health_reports_instability(self) -> None:
        report = run_eval_health_check(
            adapter_spec="tests.test_eval_health:flaky",
            adapter=FlakyAdapter(),
            cases=cases(),
            config=EvalHealthConfig(
                sample_limit=2,
                repeats=2,
                min_holdout_cases=1,
                max_unstable_case_rate=0.0,
            ),
        )

        self.assertEqual(report.status, "warning")
        self.assertIn("baseline_probe_unstable", {issue.code for issue in report.issues})
        self.assertGreater(report.baseline_probe["unstable_case_count"], 0)

    def test_runtime_health_reports_unfeasible_eval_sweep(self) -> None:
        report = run_eval_health_check(
            adapter_spec="tests.test_eval_health:heavy",
            adapter=HeavyAdapter(),
            cases=cases(),
            config=EvalHealthConfig(
                sample_limit=2,
                repeats=1,
                min_holdout_cases=1,
                max_mean_latency_s=1.0,
                max_p95_latency_s=1.0,
                max_mean_cost_usd=0.1,
                max_estimated_eval_cost_usd=1.0,
                max_estimated_eval_wall_time_s=10.0,
                max_estimated_eval_tokens=1000,
            ),
            evaluation_samples_per_case=2,
            case_concurrency=1,
        )

        codes = {issue.code for issue in report.issues}
        self.assertIn("runtime_mean_latency_high", codes)
        self.assertIn("runtime_p95_latency_high", codes)
        self.assertIn("runtime_mean_cost_high", codes)
        self.assertIn("runtime_estimated_eval_cost_high", codes)
        self.assertIn("runtime_estimated_eval_wall_time_high", codes)
        self.assertIn("runtime_estimated_eval_tokens_high", codes)
        self.assertEqual(report.baseline_probe["runtime_feasibility"]["estimated_eval_sweep"]["case_attempts"], 8)
        self.assertIn("expensive-model", report.baseline_probe["runtime_feasibility"]["models"])
        self.assertEqual(report.baseline_probe["runtime_feasibility"]["pricing_basis"], "adapter_reported_cost_usd")

    def test_eval_health_config_table_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "evals.jsonl").write_text("")
            config_path = root / "ratchet.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"

                    [ratchet.eval_health]
                    sample_limit = 3
                    repeats = 1
                    min_holdout_cases = 2
                    max_unstable_case_rate = 0.5
                    max_mean_latency_s = 9.0
                    max_estimated_eval_tokens = 12345
                    """
                ).strip()
            )

            loaded = load_run_config(config_path)

            self.assertEqual(loaded.eval_health.sample_limit, 3)
            self.assertEqual(loaded.eval_health.repeats, 1)
            self.assertEqual(loaded.eval_health.min_holdout_cases, 2)
            self.assertEqual(loaded.eval_health.max_unstable_case_rate, 0.5)
            self.assertEqual(loaded.eval_health.max_mean_latency_s, 9.0)
            self.assertEqual(loaded.eval_health.max_estimated_eval_tokens, 12345)

    def test_eval_health_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"id": "dev-1", "split": "dev", "input": "x", "expected": "x"},
                {"id": "hold-1", "split": "holdout", "input": "y", "expected": "y"},
            ]
            (root / "evals.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            config_path = root / "ratchet.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "tests.test_eval_health:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"

                    [ratchet.eval_health]
                    sample_limit = 0
                    repeats = 0
                    min_holdout_cases = 1
                    """
                ).strip()
            )

            completed = subprocess.run(
                [sys.executable, "-m", "ratchet", "eval-health", "--config", str(config_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                text=True,
                capture_output=True,
                check=True,
            )

            self.assertIn("Ratchet eval health:", completed.stdout)
            self.assertTrue((root / "results" / "run" / "eval_health" / "eval_health.json").exists())
            self.assertTrue((root / "results" / "run" / "eval_health" / "eval_health.md").exists())


adapter = StableAdapter()
flaky = FlakyAdapter()
heavy = HeavyAdapter()


if __name__ == "__main__":
    unittest.main()
