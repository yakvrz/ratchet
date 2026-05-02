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
from ratchet.config import RatchetConfigError, load_run_config, resolve_run_config
from ratchet.__main__ import CliProgressPrinter
from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


FUNCTION_AGENT_BODY = """from __future__ import annotations

from typing import Any


class Usage:
    input_tokens = 90
    output_tokens = 14


class Response:
    usage = Usage()
    output = []
    finish_reason = "stop"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class Client:
    def create_response(self, **kwargs: Any) -> Response:
        answers = {"first": "alpha", "second": "beta", "third": "gamma", "fourth": "delta"}
        return Response(answers[str(kwargs["input"])])


def build_model_input(case_payload: dict[str, Any]) -> Any:
    return case_payload["input"]


def parse_model_output(raw_output_text: str) -> object:
    return raw_output_text.strip()


def create_model_client() -> object:
    return Client()
"""


CLI_AGENT_BODY = """from __future__ import annotations

import json
import sys
from typing import Any


def run_agent(spec: dict[str, Any], case_payload: dict[str, Any]) -> dict[str, Any]:
    return {"input": case_payload["input"]}


class Usage:
    input_tokens = 100
    output_tokens = 16


class Response:
    usage = Usage()
    output = []
    finish_reason = "stop"

    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class Client:
    def create_response(self, **kwargs: Any) -> Response:
        answers = {"first": "alpha", "second": "beta", "third": "gamma", "fourth": "delta"}
        return Response(answers[str(kwargs["input"])])


def create_model_client() -> object:
    return Client()


def main() -> None:
    request = json.loads(sys.stdin.read())
    response = run_agent({}, dict(request["case"]))
    sys.stdout.write(json.dumps(response, sort_keys=True))


if __name__ == "__main__":
    main()
"""


BROKEN_ADAPTER_BODY = """from __future__ import annotations

from pathlib import Path

from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, EvalCase, GradeResult


class BrokenAdapter:
    def agent_spec(self) -> AgentSpec:
        return AgentSpec(name="broken", model="primary")

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> dict[str, str]:
        return {"output": "wrong"}

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return GradeResult(score=1.0, passed=True, labels=[])

    def export(self, candidate: CompiledCandidate | None, out_dir: Path) -> None:
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

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
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

    def export(self, candidate: CompiledCandidate | None, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate.to_dict() if candidate else None, sort_keys=True))


class CandidateAwareExportAdapter(IgnoringExportAdapter):
    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        if candidate is not None:
            return RunRecord(
                output={"message": "RATCHET_TRANSFORM_SENTINEL_RESPONSE"},
                metrics=OperationalMetrics(
                    latency_s=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                ),
                diagnostics=DiagnosticTrace(
                    metadata={
                        "transform_trace": [
                            {
                                "hook": "before_user_response",
                                "op": "rewrite_response",
                                "fields": {"message": "RATCHET_TRANSFORM_SENTINEL_RESPONSE"},
                            }
                        ]
                    }
                ),
            )
        return super().run_case(case, candidate)


class NoneAgentSpecAdapter(IgnoringExportAdapter):
    def agent_spec(self) -> None:
        return None


class WrongAgentSpecAdapter(IgnoringExportAdapter):
    def agent_spec(self) -> str:
        return "not an AgentSpec"


class RaisingAgentSpecAdapter(IgnoringExportAdapter):
    def agent_spec(self) -> AgentSpec:
        raise RuntimeError("spec unavailable")


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

    def test_progress_formatter_renders_key_events_consistently(self) -> None:
        printer = CliProgressPrinter()

        run_line = printer.format(
            {
                "event": "run_started",
                "elapsed_s": 3.2,
                "objective": "correctness",
                "train_cases": 20,
                "dev_cases": 30,
                "holdout_cases": 10,
                "dev_budget": 8,
                "holdout_budget": 2,
                "case_concurrency": 4,
                "stage_case_concurrency": 12,
                "proposal_example_count": 12,
            }
        )
        self.assertIsNotNone(run_line)
        self.assertIn("[00:03] Run", run_line)
        self.assertIn("60 cases for correctness", run_line)
        self.assertIn("train 20, dev 30, holdout 10", run_line)
        self.assertIn("candidate budget dev 8, holdout 2", run_line)

        proposal_line = printer.format(
            {
                "event": "proposal_completed",
                "elapsed_s": 72,
                "returned_count": 5,
                "valid_count": 4,
                "invalid_count": 1,
                "duplicate_count": 0,
                "call_diagnostics": {
                    "model": "gemini-3-flash-preview",
                    "input_tokens": 1200,
                    "output_tokens": 300,
                    "prompt_approx_tokens": 1000,
                    "elapsed_s": 2.4,
                    "finish_reason": "stop",
                },
            }
        )
        self.assertIsNotNone(proposal_line)
        self.assertIn("Build", proposal_line)
        self.assertIn("5 candidates, 4 compiled, 1 contract failures", proposal_line)
        self.assertIn("gemini-3-flash-preview", proposal_line)
        self.assertIn("1500 tokens", proposal_line)

        candidate_line = printer.format(
            {
                "event": "candidate_evaluated",
                "elapsed_s": 125,
                "frontier_status": "screened_out",
                "surface_mechanism": "surface_examples",
                "candidate_id": "abcdef123456",
                "score_delta": 0.125,
                "cost_delta": -0.002,
                "latency_delta": 0.31,
                "fixed_count": 3,
                "regressed_count": 1,
                "stage_count": 2,
                "full_dev_evaluated": False,
                "rejection_reason": "small-dev regression",
            }
        )
        self.assertIsNotNone(candidate_line)
        self.assertIn("Learn", candidate_line)
        self.assertIn("abcdef12", candidate_line)
        self.assertIn("score +0.125", candidate_line)
        self.assertIn("cost -$0.0020", candidate_line)
        self.assertIn("fixed 3, regressed 1", candidate_line)
        self.assertIn("full-dev no", candidate_line)

        evidence_line = printer.format(
            {
                "event": "evidence_packet_ready",
                "elapsed_s": 40,
                "weak_slices": ["ambiguity", "cancel"],
                "residual_failure_modes": ["tool_trajectory", "weak_slices"],
                "tool_error_case_count": 8,
                "invalid_output_count": 0,
            }
        )
        self.assertIsNotNone(evidence_line)
        self.assertIn("Diagnose", evidence_line)
        self.assertIn("weak slices ambiguity, cancel", evidence_line)
        self.assertIn("8 tool-error cases", evidence_line)

        plan_line = printer.format(
            {
                "event": "search_plan_ready",
                "elapsed_s": 50,
                "diagnosis": "The agent mutates orders before inspecting them and guesses on ambiguous requests.",
                "briefs": [
                    {
                        "brief_id": "inspect-before-mutate",
                        "mechanism_class": "surface_tool_loop",
                        "target_slices": ["cancel", "address"],
                        "priority": 1,
                    }
                ],
            }
        )
        self.assertIsNotNone(plan_line)
        self.assertIn("Plan", plan_line)
        self.assertIn("thinks The agent mutates", plan_line)
        self.assertIn("inspect-before-mutate", plan_line)

        batch_line = printer.format({"event": "case_batch_started", "elapsed_s": 80, "fresh_count": 12})
        self.assertIsNone(batch_line)

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
            self.assertEqual(context.exception.returncode, 3)
            self.assertIn("run_case returned dict", context.exception.stderr)

    def test_check_accepts_compiled_candidate_export_surface(self) -> None:
        cases = (
            EvalCase(id="dev-1", split="dev", input="x", expected="x"),
            EvalCase(id="hold-1", split="holdout", input="y", expected="y"),
        )
        run_preflight_check(
            adapter_spec="tests.test_cli_config:adapter",
            adapter=CandidateAwareExportAdapter(),
            cases=cases,
            objective=OptimizationObjective(),
            sample_limit=2,
        )

    def test_check_rejects_adapter_that_exports_but_ignores_candidate_execution(self) -> None:
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

    def test_check_rejects_none_agent_spec(self) -> None:
        cases = (
            EvalCase(id="dev-1", split="dev", input="x", expected="x"),
            EvalCase(id="hold-1", split="holdout", input="y", expected="y"),
        )
        with self.assertRaisesRegex(TypeError, "agent_spec\\(\\) returned None"):
            run_preflight_check(
                adapter_spec="tests.test_cli_config:none_spec",
                adapter=NoneAgentSpecAdapter(),
                cases=cases,
                objective=OptimizationObjective(),
                sample_limit=2,
            )

    def test_check_rejects_wrong_type_agent_spec(self) -> None:
        cases = (
            EvalCase(id="dev-1", split="dev", input="x", expected="x"),
            EvalCase(id="hold-1", split="holdout", input="y", expected="y"),
        )
        with self.assertRaisesRegex(TypeError, "returned str, expected AgentSpec"):
            run_preflight_check(
                adapter_spec="tests.test_cli_config:wrong_spec",
                adapter=WrongAgentSpecAdapter(),
                cases=cases,
                objective=OptimizationObjective(),
                sample_limit=2,
            )

    def test_check_wraps_raising_agent_spec(self) -> None:
        cases = (
            EvalCase(id="dev-1", split="dev", input="x", expected="x"),
            EvalCase(id="hold-1", split="holdout", input="y", expected="y"),
        )
        with self.assertRaisesRegex(TypeError, "agent_spec\\(\\) failed: spec unavailable"):
            run_preflight_check(
                adapter_spec="tests.test_cli_config:raising_spec",
                adapter=RaisingAgentSpecAdapter(),
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
            selected = json.loads((out_dir / "selected_candidate.json").read_text())
            summary = (out_dir / "summary.html").read_text()
            report = (out_dir / "report.md").read_text()
            self.assertIn("selected_candidate_id", manifest)
            self.assertEqual(manifest["simplification_results"], [])
            self.assertFalse(selected["promoted"])
            exported_surface = json.loads((out_dir / "exported_candidate" / "surface_spec.json").read_text())
            self.assertEqual(exported_surface["agent_id"], "scaffolded-python-function-agent")
            self.assertIn("<h2>What Changed</h2>", summary)
            self.assertIn('src="plots/scorecard.svg"', summary)
            self.assertIn("## Selected Candidate", report)

    def test_sanitize_examples_can_be_configured_and_overridden(self) -> None:
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
                    case_timeout_s = 0
                    stage_case_concurrency = 12
                    sanitize_examples = true
                    """
                ).strip()
            )
            loaded = load_run_config(config_path)
            self.assertTrue(loaded.objective.constraints.sanitize_examples)
            self.assertEqual(loaded.stage_case_concurrency, 12)
            overridden = resolve_run_config(
                config_path=config_path,
                adapter=None,
                evals_path=None,
                out_dir=None,
                env_file=None,
                dev_budget=None,
                holdout_budget=None,
                objective_mode=None,
                allowed_models=None,
                optimizer_model=None,
                optimizer_reasoning=None,
                samples_per_case=None,
                case_concurrency=None,
                stage_case_concurrency=None,
                max_case_retries=None,
                case_timeout_s=None,
                fail_fast=None,
                sanitize_examples=False,
            )
            self.assertFalse(overridden.objective.constraints.sanitize_examples)

    def test_config_rejects_hard_timeout_with_threaded_case_concurrency(self) -> None:
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
                    case_timeout_s = 180
                    case_concurrency = 2
                    """
                ).strip()
            )
            with self.assertRaisesRegex(RatchetConfigError, "case_timeout_s requires serial case execution"):
                load_run_config(config_path)

    def test_measurement_budgets_replace_expensive_candidate_caps(self) -> None:
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
                    max_dev_measurement_cost_usd = 0.25
                    max_holdout_measurement_cost_usd = 0.10
                    max_dev_measurement_tool_calls = 40
                    max_holdout_measurement_tool_calls = 12
                    max_dev_measurement_turns = 80
                    max_holdout_measurement_turns = 24
                    """
                ).strip()
            )

            loaded = load_run_config(config_path)
            self.assertEqual(loaded.max_dev_measurement_cost_usd, 0.25)
            self.assertEqual(loaded.max_holdout_measurement_cost_usd, 0.10)
            self.assertEqual(loaded.max_dev_measurement_tool_calls, 40)
            self.assertEqual(loaded.max_holdout_measurement_tool_calls, 12)
            self.assertEqual(loaded.max_dev_measurement_turns, 80)
            self.assertEqual(loaded.max_holdout_measurement_turns, 24)

            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"
                    max_expensive_full_dev_candidates = 1
                    """
                ).strip()
            )
            with self.assertRaisesRegex(RatchetConfigError, "max_expensive_full_dev_candidates"):
                load_run_config(config_path)

            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"
                    max_dev_measurement_cost_usd = -0.01
                    """
                ).strip()
            )
            with self.assertRaisesRegex(RatchetConfigError, "max_dev_measurement_cost_usd"):
                load_run_config(config_path)

            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"
                    max_dev_measurement_turns = -1
                    """
                ).strip()
            )
            with self.assertRaisesRegex(RatchetConfigError, "max_dev_measurement_turns"):
                load_run_config(config_path)

    def test_optimizer_role_models_fall_back_to_default_and_can_override(self) -> None:
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
                    optimizer_model = "default-model"
                    optimizer_reasoning = "medium"
                    search_planner_model = "planner-model"
                    search_planner_reasoning = "high"
                    candidate_implementer_model = "implementer-model"
                    """
                ).strip()
            )

            loaded = load_run_config(config_path)
            self.assertEqual(
                loaded.optimizer_role_models(),
                {
                    "search_planner": "planner-model",
                    "candidate_implementer": "implementer-model",
                },
            )
            self.assertEqual(
                loaded.optimizer_role_reasoning(),
                {
                    "search_planner": "high",
                    "candidate_implementer": "medium",
                },
            )

            overridden = resolve_run_config(
                config_path=config_path,
                adapter=None,
                evals_path=None,
                out_dir=None,
                env_file=None,
                dev_budget=None,
                holdout_budget=None,
                objective_mode=None,
                allowed_models=None,
                optimizer_model=None,
                optimizer_reasoning=None,
                samples_per_case=None,
                case_concurrency=None,
                stage_case_concurrency=None,
                max_case_retries=None,
                case_timeout_s=None,
                fail_fast=None,
                candidate_implementer_model="override-implementer",
            )
            self.assertEqual(overridden.optimizer_role_models()["candidate_implementer"], "override-implementer")

    def test_config_rejects_removed_optimizer_role_keys(self) -> None:
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
                    diagnoser_model = "removed"
                    """
                ).strip()
            )

            with self.assertRaisesRegex(RatchetConfigError, "diagnoser_model"):
                load_run_config(config_path)

    def test_config_rejects_unknown_top_level_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "ratchet.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"
                    dev_bugget = 50
                    """
                ).strip()
            )

            with self.assertRaisesRegex(RatchetConfigError, "dev_bugget"):
                load_run_config(config_path)

    def test_cli_config_error_uses_distinct_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "ratchet.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"
                    dev_bugget = 50
                    """
                ).strip()
            )

            completed = subprocess.run(
                [sys.executable, "-m", "ratchet", "check", "--config", str(config_path)],
                cwd=str(Path(__file__).resolve().parents[1]),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("Ratchet config error", completed.stderr)

    def test_config_rejects_missing_required_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "ratchet.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    evals = "evals.jsonl"
                    out = "results/run"
                    """
                ).strip()
            )

            with self.assertRaisesRegex(RatchetConfigError, "adapter"):
                load_run_config(config_path)

    def test_config_wraps_malformed_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "ratchet.toml"
            config_path.write_text("[ratchet\nadapter =")

            with self.assertRaisesRegex(RatchetConfigError, "Invalid TOML"):
                load_run_config(config_path)

    def test_config_rejects_unknown_nested_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "ratchet.toml"
            config_path.write_text(
                textwrap.dedent(
                    """
                    [ratchet]
                    adapter = "pkg.module:adapter"
                    evals = "evals.jsonl"
                    out = "results/run"

                    [ratchet.objective]
                    mode = "correctness"

                    [ratchet.objective.constraints]
                    typo_ratio = 2.0
                    """
                ).strip()
            )

            with self.assertRaisesRegex(RatchetConfigError, "typo_ratio"):
                load_run_config(config_path)

    def test_optimize_with_zero_dev_budget_runs_scaffolded_python_cli_agent_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = init_scaffold(Path(tmp) / "cli-agent", template="python_cli")
            (root / "agent_cli.py").write_text(CLI_AGENT_BODY)
            self.write_evals(root / "evals.sample.jsonl")
            config_path = root / "ratchet.toml"
            config_path.write_text(config_path.read_text().replace("dev_budget = 8", "dev_budget = 0"))
            self.run_cli("optimize", "--config", str(config_path))
            out_dir = root / "results" / "run"
            selected = json.loads((out_dir / "selected_candidate.json").read_text())
            self.assertFalse(selected["promoted"])


if __name__ == "__main__":
    unittest.main()
