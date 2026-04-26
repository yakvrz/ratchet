from __future__ import annotations

import json
import unittest

from ratchet.diagnosis import FailureDiagnoser
from ratchet.errors import OptimizerModelError
from ratchet.proposals import ProposalEngine
from ratchet.results import PatchSummary, CaseEvaluation
from ratchet.surface import SurfaceGenerator
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    OptimizationConstraints,
    OptimizationObjective,
    RunRecord,
)


class FakePatchClient:
    def __init__(self, patches: list[dict[str, object]]) -> None:
        self.patches = patches

    def create_response(self, **_: object) -> object:
        candidates = [
            {
                "transform_family": _family_for_patch(patch),
                "transform_instance": str(patch.get("rationale", "")) or _family_for_patch(patch),
                "target_slice": "global",
                "hypothesis": str(patch.get("rationale", "")),
                "expected_effects": {"summary": patch.get("expected_effect", "")},
                "evaluation_plan": "full_dev",
                "patch": patch,
            }
            for patch in self.patches
        ]
        return type("Response", (), {"output_text": json.dumps({"candidates": candidates})})()


def _family_for_patch(patch: dict[str, object]) -> str:
    operations = patch.get("operations", [])
    if not isinstance(operations, list) or not operations:
        return "prompt_rewrite"
    operation = operations[0]
    if not isinstance(operation, dict):
        return "prompt_rewrite"
    op = str(operation.get("op", ""))
    target = str(operation.get("target", ""))
    if op == "change_model":
        return "model_substitution"
    if op == "set_retrieval_param":
        return "retrieval_tuning"
    if target.startswith("output") or op == "add_output_constraint":
        return "output_contract_tightening"
    return "prompt_rewrite"


class FakeDiagnosisClient:
    def create_response(self, **kwargs: object) -> object:
        payload = json.loads(str(kwargs["input"]).split("\n\n", 1)[1])
        case_id = payload["failed_examples"][0]["case_id"]
        target = next(
            item["name"]
            for item in payload["editable_targets"]
            if item["kind"] == "instruction"
        )
        diagnoses = [
            {
                "case_ids": [case_id],
                "category": "prompt_ambiguity",
                "root_cause": "Instructions do not distinguish the expected behavior.",
                "target_names": [target],
                "evidence": [{"case_id": case_id}],
            }
        ]
        return type("Response", (), {"output_text": json.dumps({"diagnoses": diagnoses})})()


class InvalidJsonClient:
    def create_response(self, **_: object) -> object:
        return type("Response", (), {"output_text": "not-json"})()


class BarePatchClient:
    def __init__(self, patches: list[dict[str, object]]) -> None:
        self.patches = patches

    def create_response(self, **_: object) -> object:
        return type("Response", (), {"output_text": json.dumps({"patches": self.patches})})()


def make_summary(patch_hash: str, scores: list[float]) -> PatchSummary:
    evaluations = []
    for index, score in enumerate(scores, start=1):
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(id=f"case-{index}", split="dev", input=f"case {index}"),
                record=RunRecord(
                    output="ok" if score == 1.0 else "wrong",
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(),
                ),
                grade=GradeResult(score=score, passed=score == 1.0),
            )
        )
    return PatchSummary(
        patch_hash=patch_hash,
        patch=AgentPatch(),
        split="dev",
        evaluations=evaluations,
    )


class GeneratedSurfaceTests(unittest.TestCase):
    def test_surface_exists_even_without_agent_spec(self) -> None:
        targets = SurfaceGenerator().generate(None, OptimizationObjective())
        self.assertEqual(targets[0].name, "wrapper_instruction")
        self.assertIn("add_instruction", targets[0].allowed_ops)

    def test_diagnoser_targets_generated_instruction_surface(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        summary = make_summary("baseline", [0.0, 1.0])
        diagnoser = FailureDiagnoser(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        diagnoser._client = FakeDiagnosisClient()
        diagnoses, _ = diagnoser.diagnose(summary, surface)

        self.assertEqual(diagnoses[0].category, "prompt_ambiguity")
        self.assertIn("instructions.system_prompt", diagnoses[0].target_names)

    def test_diagnoser_json_failure_is_fatal(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        summary = make_summary("baseline", [0.0, 1.0])
        diagnoser = FailureDiagnoser(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        diagnoser._client = InvalidJsonClient()

        with self.assertRaises(OptimizerModelError):
            diagnoser.diagnose(summary, surface)

    def test_llm_proposer_validates_returned_patch(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["instruction", "output"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        summary = make_summary("baseline", [0.0, 1.0])
        diagnoser = FailureDiagnoser(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        diagnoser._client = FakeDiagnosisClient()
        diagnoses, _ = diagnoser.diagnose(summary, surface)
        engine = ProposalEngine(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = FakePatchClient(
            [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Answer with grounded evidence.",
                        }
                    ],
                    "rationale": "Ground the behavior.",
                    "expected_effect": "Improve failed cases.",
                }
            ]
        )

        proposals, analysis = engine.propose(
            summary,
            surface,
            objective=objective,
            diagnosis=diagnoses[0],
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        self.assertTrue(proposals)
        self.assertEqual(proposals[0].patch.operations[0].op, "add_instruction")
        self.assertEqual(proposals[0].transform_family, "prompt_rewrite")
        self.assertIn("Validated LLM transform candidate proposals", analysis)

    def test_llm_proposer_preserves_model_rank_and_logs_deferred_candidates(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="old",
            model_options=["old", "new"],
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = ProposalEngine(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = FakePatchClient(
            [
                {
                    "operations": [
                        {"op": "change_model", "target": "model", "value": "new"}
                    ],
                    "rationale": "Try a stronger allowed model.",
                    "expected_effect": "Improve capability on failed cases.",
                },
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Answer with grounded evidence.",
                        }
                    ],
                    "rationale": "Ground the behavior.",
                    "expected_effect": "Improve failed cases.",
                },
                {
                    "operations": [
                        {
                            "op": "add_output_constraint",
                            "target": "output_contract",
                            "value": "Keep the response concise.",
                        }
                    ],
                    "rationale": "Constrain output style.",
                    "expected_effect": "Reduce output drift.",
                },
            ]
        )

        proposals, _ = engine.propose(
            make_summary("baseline", [0.0, 1.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            proposal_budget=1,
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0].patch.operations[0].op, "change_model")
        self.assertEqual(proposals[0].transform_family, "model_substitution")
        self.assertEqual(engine.last_stats.valid_count, 3)
        self.assertEqual(engine.last_stats.returned_count, 1)
        self.assertEqual(len(engine.last_candidate_rows), 3)
        self.assertEqual(
            [row["scheduled"] for row in engine.last_candidate_rows],
            [True, False, False],
        )

    def test_llm_proposer_accepts_categorical_retrieval_patch(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            model_options=["small", "large"],
            instructions={"system_prompt": "Answer helpfully."},
            retrieval={"knowledge_mode": "raw", "top_k": 6},
            runtime={"output_cap": "128"},
        )
        objective = OptimizationObjective(mode="cost")
        surface = SurfaceGenerator().generate(spec, objective)
        engine = ProposalEngine(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = FakePatchClient(
            [
                {
                    "operations": [
                        {
                            "op": "set_retrieval_param",
                            "target": "retrieval.knowledge_mode",
                            "value": "semantic",
                        }
                    ],
                    "rationale": "Change retrieval mode.",
                    "expected_effect": "Explore a categorical retrieval setting.",
                }
            ]
        )

        proposals, _ = engine.propose(
            make_summary("baseline", [1.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        targets = [operation.target for candidate in proposals for operation in candidate.patch.operations]
        self.assertEqual(targets, ["retrieval.knowledge_mode"])

    def test_llm_proposer_rejects_legacy_bare_patches(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        engine = ProposalEngine(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = BarePatchClient(
            [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Answer exactly.",
                        }
                    ],
                    "rationale": "Legacy bare patch.",
                    "expected_effect": "Should not be accepted.",
                }
            ]
        )

        proposals, _ = engine.propose(
            make_summary("baseline", [0.0]),
            surface,
            objective=OptimizationObjective(),
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        self.assertEqual(proposals, [])
        self.assertEqual(engine.last_stats.raw_count, 0)

    def test_llm_proposer_rejects_inactive_family_candidate(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = ProposalEngine(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = FakePatchClient(
            [
                {
                    "operations": [
                        {
                            "op": "add_output_constraint",
                            "target": "output_contract",
                            "value": "Return concise text.",
                        }
                    ],
                    "rationale": "Output family has no current signal.",
                    "expected_effect": "Should be inactive.",
                }
            ]
        )

        proposals, _ = engine.propose(
            make_summary("baseline", [1.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        self.assertEqual(proposals, [])
        self.assertIn("inactive transform family", next(iter(engine.last_stats.invalid_reasons or {})))


if __name__ == "__main__":
    unittest.main()
