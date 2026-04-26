from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ratchet.__main__ import run_optimizer
from ratchet.adapters import load_adapter
from ratchet.errors import OptimizerModelError
from ratchet.io import load_eval_cases
from ratchet.optimizer import RatchetOptimizer
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    OptimizationObjective,
    RunRecord,
)
from tests.fixtures.fake_adapter import adapter as fake_adapter


class InvalidJsonClient:
    def create_response(self, **_: object) -> object:
        class Response:
            output_text = "not-json"

        return Response()


class FakeDiagnosisClient:
    def create_response(self, **kwargs: object) -> object:
        payload = json.loads(str(kwargs["input"]).split("\n\n", 1)[1])
        case_ids = [item["case_id"] for item in payload["failed_examples"]]
        target_names = [
            item["name"]
            for item in payload["editable_targets"]
            if item["name"] in {"instructions.system_prompt", "tools.search.enabled", "model"}
        ]
        diagnoses = [
            {
                "case_ids": case_ids,
                "category": "behavior_gap",
                "root_cause": "The current agent behavior does not satisfy the failed examples.",
                "target_names": target_names,
                "evidence": [{"case_ids": case_ids}],
            }
        ]
        return type("Response", (), {"output_text": json.dumps({"diagnoses": diagnoses})})()


class FakePatchClient:
    def create_response(self, **kwargs: object) -> object:
        prompt = str(kwargs.get("input", ""))
        if '"mode": "cost"' in prompt:
            patches = [
                {
                    "operations": [{"op": "change_model", "target": "model", "value": "small"}],
                    "rationale": "Try a cheaper allowed model.",
                    "expected_effect": "Lower cost while preserving current correctness.",
                }
            ]
        else:
            patches = [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Answer with exact grounded facts.",
                        }
                    ],
                    "rationale": "Ground answers more explicitly.",
                    "expected_effect": "Fix cases where the baseline answers wrong.",
                },
                {
                    "operations": [
                        {"op": "set_runtime_param", "target": "tools.search.enabled", "value": True}
                    ],
                    "rationale": "Enable search for tool-dependent cases.",
                    "expected_effect": "Allow grounded answers when a tool is needed.",
                },
            ]

        return type("Response", (), {"output_text": json.dumps({"patches": patches})})()


class ShapeInvalidPatchClient:
    def create_response(self, **_: object) -> object:
        patches = [
            {
                "operations": [
                    {"op": "set_runtime_param", "target": "tools.search.enabled", "value": "true"}
                ],
                "rationale": "Malformed boolean value.",
                "expected_effect": "Should be rejected by target schema.",
            }
        ]
        return type("Response", (), {"output_text": json.dumps({"patches": patches})})()


BRANCH_SPEC = AgentSpec(
    name="branching-agent",
    model="large",
    instructions={"system_prompt": "Answer from learned routing rules."},
    output_contract="Return the exact expected token.",
)


class BranchingAdapter:
    def agent_spec(self) -> AgentSpec:
        return BRANCH_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        spec = BRANCH_SPEC.apply_patch(patch)
        instruction_text = " ".join(spec.instructions.values()).lower()
        expected = str(case.expected)
        solved = expected in instruction_text
        return RunRecord(
            output=expected if solved else "wrong",
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=50,
                output_tokens=5,
                total_tokens=55,
                cost_usd=0.001,
            ),
            diagnostics=DiagnosticTrace(raw_output_text=expected if solved else "wrong"),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        passed = str(output) == str(case.expected)
        return GradeResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            labels=[] if passed else ["failed"],
        )

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))


class BranchingDiagnosisClient:
    def create_response(self, **kwargs: object) -> object:
        payload = json.loads(str(kwargs["input"]).split("\n\n", 1)[1])
        failed_case_ids = [item["case_id"] for item in payload["failed_examples"]]
        target = next(item["name"] for item in payload["editable_targets"] if item["kind"] == "instruction")
        diagnoses = [
            {
                "case_ids": [case_id],
                "category": f"cluster-{index}",
                "root_cause": f"Missing generalized rule for cluster {index}.",
                "target_names": [target],
                "evidence": [{"case_id": case_id}],
            }
            for index, case_id in enumerate(failed_case_ids, start=1)
        ]
        return type("Response", (), {"output_text": json.dumps({"diagnoses": diagnoses})})()


class BranchingPatchClient:
    def __init__(self) -> None:
        self.diagnosis_counts: list[int] = []

    def create_response(self, **kwargs: object) -> object:
        payload = json.loads(str(kwargs["input"]).split("\n\n", 1)[1])
        self.diagnosis_counts.append(len(payload.get("diagnoses", [])))
        values = [
            str(operation.get("value", "")).lower()
            for operation in payload["current_patch"].get("operations", [])
        ]
        has_alpha = any("alpha" in value for value in values)
        has_beta = any("beta" in value for value in values)
        if not has_alpha and not has_beta:
            patches = [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Route alpha cases to alpha.",
                        }
                    ],
                    "rationale": "Try the alpha cluster.",
                    "expected_effect": "Fix alpha cases.",
                },
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Route beta cases to beta.",
                        }
                    ],
                    "rationale": "Try the beta cluster.",
                    "expected_effect": "Fix beta cases.",
                },
            ]
        elif has_beta and not has_alpha:
            patches = [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Route alpha cases to alpha.",
                        }
                    ],
                    "rationale": "Compose the beta branch with the alpha rule.",
                    "expected_effect": "Fix both clusters.",
                }
            ]
        else:
            patches = []
        return type("Response", (), {"output_text": json.dumps({"patches": patches})})()


class RetryPatchClient:
    def __init__(self) -> None:
        self.history_lengths: list[int] = []

    def create_response(self, **kwargs: object) -> object:
        payload = json.loads(str(kwargs["input"]).split("\n\n", 1)[1])
        self.history_lengths.append(len(payload.get("recent_history", [])))
        if len(self.history_lengths) == 1:
            patches = [
                {
                    "operations": [
                        {"op": "set_runtime_param", "target": "runtime.output_cap", "value": 80}
                    ],
                    "rationale": "Try a harmless runtime cap first.",
                    "expected_effect": "This should be rejected because behavior is unchanged.",
                }
            ]
        else:
            patches = [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Answer with exact grounded facts.",
                        },
                        {"op": "set_runtime_param", "target": "tools.search.enabled", "value": True},
                    ],
                    "rationale": "Use rejection evidence to make a behavioral patch.",
                    "expected_effect": "Fix all fake eval cases.",
                }
            ]
        return type("Response", (), {"output_text": json.dumps({"patches": patches})})()


class FakeAdapterIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        fake_adapter.reset()

    def write_evals(self, directory: Path, *, expected_prefix: str = "") -> Path:
        evals_path = directory / "fake_evals.jsonl"
        prefix = f"{expected_prefix}-" if expected_prefix else ""
        rows = [
            {"id": "dev-1", "split": "dev", "input": "policy", "expected": f"{prefix}policy", "metadata": {"needs_tool": False}},
            {"id": "dev-2", "split": "dev", "input": "math", "expected": f"{prefix}math", "metadata": {"needs_tool": True}},
            {"id": "hold-1", "split": "holdout", "input": "policy", "expected": f"{prefix}policy", "metadata": {"needs_tool": False}},
            {"id": "hold-2", "split": "holdout", "input": "math", "expected": f"{prefix}math", "metadata": {"needs_tool": True}},
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
                dev_budget=0,
                holdout_budget=3,
                optimizer_model="gpt-5.4-mini",
                optimizer_reasoning="low",
            )
            self.assertTrue((out_dir / "case_results.jsonl").exists())
            self.assertTrue((out_dir / "patch_metrics.json").exists())
            self.assertTrue((out_dir / "decision_log.json").exists())
            self.assertTrue((out_dir / "run_manifest.json").exists())
            self.assertTrue((out_dir / "summary.html").exists())
            self.assertTrue((out_dir / "plots" / "progress.svg").exists())
            self.assertTrue((out_dir / "plots" / "efficiency_progress.svg").exists())
            self.assertTrue((out_dir / "report.md").exists())
            self.assertTrue((out_dir / "selected_patch.json").exists())
            self.assertTrue((out_dir / "exported_patch" / "patch.json").exists())
            selected = json.loads((out_dir / "selected_patch.json").read_text())
            self.assertFalse(selected["promoted"])
            self.assertEqual(selected["patch"]["operations"], [])

    def test_llm_proposer_drives_optimization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            optimizer = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "run",
                env_path=".env",
                dev_budget=6,
                holdout_budget=3,
            )
            optimizer.diagnoser._client = FakeDiagnosisClient()
            optimizer.proposer._client = FakePatchClient()
            result = optimizer.run(cases)

            self.assertTrue(result.promoted)
            ops = result.selected_patch.to_dict()["operations"]
            self.assertTrue(any(op["op"] == "add_instruction" for op in ops))
            self.assertTrue(any(op["target"] == "tools.search.enabled" for op in ops))

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
                dev_budget=0,
                holdout_budget=2,
                optimizer_model="gpt-5.4-mini",
                optimizer_reasoning="low",
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
                dev_budget=0,
                holdout_budget=2,
                optimizer_model="gpt-5.4-mini",
                optimizer_reasoning="low",
            )
            case_lines_after_resume = (out_dir / "case_results.jsonl").read_text().splitlines()
            self.assertGreater(len(case_lines_after_resume), len(case_lines_after_failure))

    def test_resume_does_not_reuse_cache_after_eval_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root, expected_prefix="first")
            out_dir = root / "run"
            run_optimizer(
                adapter_spec="tests.fixtures.fake_adapter:adapter",
                evals_path=evals_path,
                out_dir=out_dir,
                env_file=".env",
                dev_budget=0,
                holdout_budget=0,
                optimizer_model="gpt-5.4-mini",
                optimizer_reasoning="low",
            )

            self.write_evals(root, expected_prefix="second")
            run_optimizer(
                adapter_spec="tests.fixtures.fake_adapter:adapter",
                evals_path=evals_path,
                out_dir=out_dir,
                env_file=".env",
                dev_budget=0,
                holdout_budget=0,
                optimizer_model="gpt-5.4-mini",
                optimizer_reasoning="low",
            )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            self.assertEqual(manifest["stats"]["cache_hits"], 0)
            self.assertEqual(manifest["stats"]["fresh_case_evaluations"], 4)

    def test_samples_per_case_records_and_aggregates_repeated_evaluations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            out_dir = root / "run"
            run_optimizer(
                adapter_spec="tests.fixtures.fake_adapter:adapter",
                evals_path=evals_path,
                out_dir=out_dir,
                env_file=".env",
                dev_budget=0,
                holdout_budget=0,
                optimizer_model="gpt-5.4-mini",
                optimizer_reasoning="low",
                samples_per_case=3,
            )
            manifest = json.loads((out_dir / "run_manifest.json").read_text())
            metrics = json.loads((out_dir / "patch_metrics.json").read_text())
            case_records = [json.loads(line) for line in (out_dir / "case_results.jsonl").read_text().splitlines()]

            self.assertEqual(manifest["samples_per_case"], 3)
            self.assertEqual(manifest["stats"]["fresh_case_evaluations"], 12)
            self.assertEqual(metrics["baseline_dev"]["case_count"], 2)
            self.assertEqual(metrics["baseline_dev"]["sample_count"], 6)
            self.assertEqual(metrics["baseline_dev"]["samples_per_case"], 3)
            self.assertEqual({row["sample_index"] for row in case_records}, {0, 1, 2})

    def test_holdout_budget_caps_finalist_validations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            out_dir = root / "run"
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            optimizer = RatchetOptimizer(
                adapter=adapter,
                out_dir=out_dir,
                env_path=".env",
                dev_budget=6,
                holdout_budget=0,
            )
            optimizer.diagnoser._client = FakeDiagnosisClient()
            optimizer.proposer._client = FakePatchClient()
            optimizer.run(cases)
            metrics = json.loads((out_dir / "patch_metrics.json").read_text())
            decision_log = json.loads((out_dir / "decision_log.json").read_text())
            self.assertEqual(metrics["holdout_patches"], [])
            self.assertTrue(
                any(event["type"] == "holdout_validation_skipped" for event in decision_log)
            )

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
                    dev_budget=0,
                    holdout_budget=2,
                    optimizer_model="gpt-5.4-mini",
                    optimizer_reasoning="low",
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
                out_dir=root / "run",
                env_path=".env",
                dev_budget=0,
                holdout_budget=2,
                max_case_retries=1,
                case_timeout_s=1,
            )
            optimizer.run(cases)
            manifest = json.loads((root / "run" / "run_manifest.json").read_text())
            self.assertGreaterEqual(manifest["stats"]["timeouts"], 1)
            self.assertGreaterEqual(manifest["stats"]["grader_errors"], 1)
            self.assertGreaterEqual(manifest["stats"]["retries"], 2)

    def test_invalid_llm_output_does_not_fall_back_to_static_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            optimizer = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "run",
                env_path=".env",
                dev_budget=4,
                holdout_budget=2,
            )
            optimizer.diagnoser._client = FakeDiagnosisClient()
            optimizer.proposer._client = InvalidJsonClient()
            with self.assertRaises(OptimizerModelError):
                optimizer.run(cases)

    def test_shape_invalid_model_patch_is_reported_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            optimizer = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "run",
                env_path=".env",
                dev_budget=4,
                holdout_budget=2,
            )
            optimizer.diagnoser._client = FakeDiagnosisClient()
            optimizer.proposer._client = ShapeInvalidPatchClient()
            result = optimizer.run(cases)

            self.assertFalse(result.promoted)
            self.assertEqual(result.outcome_analysis["status"], "proposals_invalid")
            self.assertEqual(result.outcome_analysis["latest_proposal_stats"]["raw_count"], 1)
            self.assertEqual(result.outcome_analysis["latest_proposal_stats"]["valid_count"], 0)

    def test_objective_modes_select_different_winners(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            correctness = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "correctness",
                dev_budget=6,
                holdout_budget=3,
                objective=OptimizationObjective(mode="correctness"),
            )
            correctness.diagnoser._client = FakeDiagnosisClient()
            correctness.proposer._client = FakePatchClient()
            correctness_result = correctness.run(cases)
            cost = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "cost",
                dev_budget=6,
                holdout_budget=3,
                objective=OptimizationObjective(mode="cost"),
            )
            cost.diagnoser._client = FakeDiagnosisClient()
            cost.proposer._client = FakePatchClient()
            cost_result = cost.run(cases)
            self.assertNotEqual(correctness_result.selected_patch_hash, cost_result.selected_patch_hash)

    def test_frontier_expands_alternate_accepted_branch_after_best_branch_stalls(self) -> None:
        cases = (
            EvalCase(id="dev-alpha", split="dev", input="alpha", expected="alpha"),
            EvalCase(id="dev-beta", split="dev", input="beta", expected="beta"),
            EvalCase(id="hold-alpha", split="holdout", input="alpha", expected="alpha"),
            EvalCase(id="hold-beta", split="holdout", input="beta", expected="beta"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch_client = BranchingPatchClient()
            optimizer = RatchetOptimizer(
                adapter=BranchingAdapter(),
                out_dir=root / "run",
                env_path=".env",
                dev_budget=4,
                holdout_budget=2,
            )
            optimizer.diagnoser._client = BranchingDiagnosisClient()
            optimizer.proposer._client = patch_client
            result = optimizer.run(cases)
            decision_log = json.loads((root / "run" / "decision_log.json").read_text())

            values = [
                str(operation["value"]).lower()
                for operation in result.selected_patch.to_dict()["operations"]
            ]
            self.assertTrue(result.promoted)
            self.assertTrue(any("alpha" in value for value in values))
            self.assertTrue(any("beta" in value for value in values))
            self.assertTrue(any(count >= 2 for count in patch_client.diagnosis_counts))
            self.assertTrue(
                any(
                    event.get("type") == "proposal_iteration"
                    and event.get("iteration") == 2
                    and event.get("parent_rank") == 2
                    for event in decision_log
                )
            )

    def test_rejected_batch_gets_one_model_driven_retry_with_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            patch_client = RetryPatchClient()
            optimizer = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "run",
                env_path=".env",
                dev_budget=3,
                holdout_budget=0,
            )
            optimizer.diagnoser._client = FakeDiagnosisClient()
            optimizer.proposer._client = patch_client
            result = optimizer.run(cases)

            retry_evaluations = [
                event
                for event in result.decision_log
                if event.get("type") == "proposal_evaluation" and event.get("proposal_retry")
            ]
            self.assertEqual(len(patch_client.history_lengths), 2)
            self.assertEqual(patch_client.history_lengths[0], 0)
            self.assertGreater(patch_client.history_lengths[1], 0)
            self.assertEqual(len(retry_evaluations), 1)
            self.assertTrue(retry_evaluations[0]["accepted"])
            self.assertEqual(retry_evaluations[0]["retry_reason"], "no_accepted_candidates_from_parent")
            self.assertTrue(result.accepted_dev_patches)

    def test_rejected_batch_retry_does_not_run_when_dev_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evals_path = self.write_evals(root)
            cases = load_eval_cases(evals_path)
            adapter = load_adapter("tests.fixtures.fake_adapter:adapter")
            patch_client = RetryPatchClient()
            optimizer = RatchetOptimizer(
                adapter=adapter,
                out_dir=root / "run",
                env_path=".env",
                dev_budget=1,
                holdout_budget=0,
            )
            optimizer.diagnoser._client = FakeDiagnosisClient()
            optimizer.proposer._client = patch_client
            result = optimizer.run(cases)

            self.assertEqual(len(patch_client.history_lengths), 1)
            self.assertFalse(any(event.get("proposal_retry") for event in result.decision_log))


if __name__ == "__main__":
    unittest.main()
