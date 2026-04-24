from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest
import sys

from ratchet.scaffold import init_scaffold


FUNCTION_AGENT_BODY = """from __future__ import annotations

from typing import Any


def run_agent(
    candidate: dict[str, str],
    case_payload: dict[str, Any],
    hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = str(case_payload["expected"])
    cheaper = candidate["model"] == "cheaper"
    grounded = "grounded" in candidate["system_prompt"].lower()
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
    return {
        "output": expected,
        "raw_output_text": expected,
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

from ratchet.code_artifacts import CodeArtifactLoader
from ratchet.types import CodeArtifactSpec


def run_agent(
    candidate: dict[str, str],
    case_payload: dict[str, Any],
    hooks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    expected = str(case_payload["expected"])
    cheaper = candidate["model"] == "cheaper"
    grounded = "grounded" in candidate["system_prompt"].lower()
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
    return {
        "output": expected,
        "raw_output_text": expected,
        "tool_calls": [],
        "latency_s": latency_s,
        "input_tokens": total_tokens // 2,
        "output_tokens": total_tokens // 2,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def main() -> None:
    request = json.loads(sys.stdin.read())
    loader = CodeArtifactLoader()
    hook_specs = [CodeArtifactSpec.from_dict(item) for item in request.get("hook_specs", [])]
    hook_sources = dict(request.get("hook_sources", {}))
    hooks = loader.build_hooks(hook_sources, hook_specs) if hook_specs else {}
    response = run_agent(
        candidate=dict(request["candidate"]),
        case_payload=dict(request["case"]),
        hooks=hooks,
    )
    sys.stdout.write(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
"""


BROKEN_ADAPTER_BODY = """from __future__ import annotations

import json
from pathlib import Path

from ratchet.types import EvalCase, GradeResult, RunRecord, SearchSpace, TextArtifactSpec


class BrokenAdapter:
    def baseline(self) -> dict[str, str]:
        return {"mode": "default"}

    def search_space(self) -> SearchSpace:
        return SearchSpace(
            text_artifacts=[
                TextArtifactSpec(
                    name="mode",
                    kind="prompt",
                    default="default",
                    max_chars=32,
                    description="Broken check fixture.",
                )
            ]
        )

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> dict[str, str]:
        return {"output": "wrong"}

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return GradeResult(score=1.0, passed=True, labels=[])

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))


adapter = BrokenAdapter()
"""


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

    def test_run_with_config_optimizes_scaffolded_python_function_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "function-agent", template="python_function")
            (root / "agent.py").write_text(FUNCTION_AGENT_BODY)
            self.write_evals(root / "evals.sample.jsonl")
            self.run_cli("run", "--config", str(root / "ratchet.toml"), "--disable-harnesser")
            out_dir = root / "results" / "run"
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            selected = json.loads((out_dir / "optimized_candidate.json").read_text())
            report = (out_dir / "report.md").read_text()
            self.assertIn("selected_candidate_hash", manifest)
            self.assertTrue(selected["promoted"])
            self.assertEqual(selected["candidate"]["model"], "cheaper")
            self.assertIn("grounded", selected["candidate"]["system_prompt"].lower())
            self.assertIn("## Selected Candidate", report)

    def test_run_with_config_optimizes_scaffolded_python_cli_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "cli-agent", template="python_cli")
            (root / "agent_cli.py").write_text(CLI_AGENT_BODY)
            self.write_evals(root / "evals.sample.jsonl")
            self.run_cli("run", "--config", str(root / "ratchet.toml"), "--disable-harnesser")
            out_dir = root / "results" / "run"
            selected = json.loads((out_dir / "optimized_candidate.json").read_text())
            self.assertTrue(selected["promoted"])
            self.assertEqual(selected["candidate"]["model"], "cheaper")


if __name__ == "__main__":
    unittest.main()
