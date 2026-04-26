from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from ratchet.scaffold import init_scaffold
from ratchet.preflight import run_preflight_check
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


FUNCTION_AGENT_BODY = """from __future__ import annotations

from typing import Any


def run_agent(spec: dict[str, Any], case_payload: dict[str, Any]) -> dict[str, Any]:
    expected = str(case_payload["expected"])
    cheaper = spec["model"] == "cheaper"
    grounded = "grounded" in " ".join(spec.get("instructions", {}).values()).lower()
    total_tokens = 180
    cost_usd = 0.009
    latency_s = 1.3
    if cheaper:
        total_tokens -= 60
        cost_usd -= 0.005
        latency_s -= 0.2
    if grounded:
        total_tokens -= 35
        cost_usd -= 0.001
        latency_s -= 0.1
    output = expected if grounded else "wrong"
    return {
        "output": output,
        "raw_output_text": output,
        "tool_calls": [],
        "latency_s": latency_s,
        "input_tokens": total_tokens // 2,
        "output_tokens": total_tokens // 2,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }
"""


CLI_AGENT_BODY = """from __future__ import annotations

import json
import sys
from typing import Any


def run_agent(spec: dict[str, Any], case_payload: dict[str, Any]) -> dict[str, Any]:
    expected = str(case_payload["expected"])
    cheaper = spec["model"] == "cheaper"
    grounded = "grounded" in " ".join(spec.get("instructions", {}).values()).lower()
    total_tokens = 200
    cost_usd = 0.011
    latency_s = 1.4
    if cheaper:
        total_tokens -= 70
        cost_usd -= 0.006
        latency_s -= 0.25
    if grounded:
        total_tokens -= 45
        cost_usd -= 0.001
        latency_s -= 0.1
    output = expected if grounded else "wrong"
    return {
        "output": output,
        "raw_output_text": output,
        "tool_calls": [],
        "latency_s": latency_s,
        "input_tokens": total_tokens // 2,
        "output_tokens": total_tokens // 2,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def main() -> None:
    request = json.loads(sys.stdin.read())
    response = run_agent(
        spec=dict(request["spec"]),
        case_payload=dict(request["case"]),
    )
    sys.stdout.write(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
"""


BROKEN_ADAPTER_BODY = """from __future__ import annotations

from pathlib import Path

from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult


class BrokenAdapter:
    def agent_spec(self) -> AgentSpec:
        return AgentSpec(name="broken", model="primary")

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> dict[str, str]:
        return {"output": "wrong"}

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return GradeResult(score=1.0, passed=True, labels=[])

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)


adapter = BrokenAdapter()
"""


class IgnoringExportAdapter:
    def agent_spec(self) -> AgentSpec:
        return AgentSpec(
            name="ignores-export",
            model="primary",
            instructions={"system_prompt": "Answer."},
            output_contract="Return text.",
        )

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        return RunRecord(
            output=str(case.expected),
            metrics=OperationalMetrics(
                latency_s=0.0,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return GradeResult(score=1.0, passed=True)

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), sort_keys=True))


class CliConfigIntegrationTests(unittest.TestCase):
    def run_cli(self, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "ratchet", *args],
            cwd=str(cwd or Path(__file__).resolve().parents[1]),
            text=True,
            capture_output=True,
            check=True,
        )

    def write_evals(self, path: Path) -> None:
        rows = [
            {"id": "dev-1", "split": "dev", "input": "first", "expected": "alpha", "metadata": {"category": "sample"}},
            {"id": "dev-2", "split": "dev", "input": "second", "expected": "beta", "metadata": {"category": "sample"}},
            {"id": "hold-1", "split": "holdout", "input": "third", "expected": "gamma", "metadata": {"category": "sample"}},
            {"id": "hold-2", "split": "holdout", "input": "fourth", "expected": "delta", "metadata": {"category": "sample"}},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows))

    def test_check_succeeds_on_scaffolded_python_function_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "function-agent", template="python_function")
            (root / "agent.py").write_text(FUNCTION_AGENT_BODY)
            self.write_evals(root / "evals.sample.jsonl")
            completed = self.run_cli("check", "--config", str(root / "ratchet.toml"), "--sample-limit", "2")
            self.assertIn("Ratchet check passed.", completed.stdout)

    def test_check_fails_clearly_on_invalid_adapter_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ratchet_adapter.py").write_text(BROKEN_ADAPTER_BODY)
            rows = [
                {"id": "dev-1", "split": "dev", "input": "x", "expected": "x"},
                {"id": "hold-1", "split": "holdout", "input": "y", "expected": "y"},
            ]
            (root / "evals.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            (root / "ratchet.toml").write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "ratchet_adapter:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"
                    env_file = ".env"
                    """
                ).strip()
            )
            with self.assertRaises(subprocess.CalledProcessError) as context:
                self.run_cli("check", "--config", str(root / "ratchet.toml"))
            self.assertIn("run_case returned dict", context.exception.stderr)

    def test_check_fails_when_export_does_not_materialize_generated_targets(self) -> None:
        cases = (
            EvalCase(id="dev-1", split="dev", input="x", expected="x"),
            EvalCase(id="hold-1", split="holdout", input="y", expected="y"),
        )
        with self.assertRaisesRegex(ValueError, "Materialization audit failed"):
            run_preflight_check(
                adapter_spec="tests.test_cli_config:adapter",
                adapter=IgnoringExportAdapter(),
                cases=cases,
                objective=OptimizationObjective(),
                sample_limit=2,
            )

    def test_optimize_with_zero_dev_budget_runs_scaffolded_python_function_agent_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "function-agent", template="python_function")
            (root / "agent.py").write_text(FUNCTION_AGENT_BODY)
            self.write_evals(root / "evals.sample.jsonl")
            config_path = root / "ratchet.toml"
            config_path.write_text(config_path.read_text().replace("dev_budget = 8", "dev_budget = 0"))
            self.run_cli("optimize", "--config", str(config_path))
            out_dir = root / "results" / "run"
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            selected = json.loads((out_dir / "selected_patch.json").read_text())
            summary = (out_dir / "summary.html").read_text()
            report = (out_dir / "report.md").read_text()
            self.assertIn("selected_patch_hash", manifest)
            self.assertFalse(selected["promoted"])
            applied = json.loads((out_dir / "exported_patch" / "agent_spec.json").read_text())
            self.assertNotIn("grounded", " ".join(applied["instructions"].values()).lower())
            self.assertIn("<h2>What Changed</h2>", summary)
            self.assertIn('src="plots/scorecard.svg"', summary)
            self.assertIn("## Selected Patch", report)

    def test_optimize_with_zero_dev_budget_runs_scaffolded_python_cli_agent_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "cli-agent", template="python_cli")
            (root / "agent_cli.py").write_text(CLI_AGENT_BODY)
            self.write_evals(root / "evals.sample.jsonl")
            config_path = root / "ratchet.toml"
            config_path.write_text(config_path.read_text().replace("dev_budget = 8", "dev_budget = 0"))
            self.run_cli("optimize", "--config", str(config_path))
            out_dir = root / "results" / "run"
            selected = json.loads((out_dir / "selected_patch.json").read_text())
            self.assertFalse(selected["promoted"])


if __name__ == "__main__":
    unittest.main()
