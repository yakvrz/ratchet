from __future__ import annotations

import json
import unittest

from ratchet.diagnosis import FailureDiagnoser
from ratchet.evidence import build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentIntent, build_task_theory
from ratchet.affordances import generate_optimization_affordances
from ratchet.optimizer import _task_theory_with_affordance_opportunities
from ratchet.proposals import CandidateImplementer, prompt_size_profile
from ratchet.results import PatchSummary, CaseEvaluation
from ratchet.surface import SurfaceGenerator
from ratchet.transforms import BehaviorProfile, SearchHypothesis, TransformFamilyState
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

    def create_response(self, **kwargs: object) -> object:
        prompt = str(kwargs.get("input", ""))
        candidates = [
            {
                "experiment_id": "exp_1",
                "candidate_role": "atomic",
                "comparison_group": "exp_1",
                "target_slice": "global",
                "hypothesis": str(patch.get("rationale", "")),
                "expected_effects": {"summary": patch.get("expected_effect", "")},
                "evaluation_plan": "full_dev",
                "_test_patch": patch,
                "_test_family": _family_for_patch(patch),
                "_test_mechanism": _mechanism_for_family(_family_for_patch(patch)),
            }
            for patch in self.patches
        ]
        _attach_affordance_ids(candidates, prompt)
        return experiment_response(candidates)


def experiment_response(candidates: list[dict[str, object]], *, mechanism: str = "semantic_boundary_rewrite") -> object:
    return type(
        "Response",
        (),
        {
            "output_text": json.dumps(
                {
                    "experiments": [
                        {
                            "experiment_id": "exp_1",
                            "mechanism_class": mechanism,
                            "mechanism": mechanism,
                            "hypothesis": "Test a controlled optimization mechanism.",
                            "target_slices": ["global"],
                            "measurements": ["score_delta", "cost_delta", "latency_delta"],
                            "candidate_roles": ["atomic"],
                            "candidates": candidates,
                        }
                    ]
                }
            )
        },
    )()


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
    if op == "set_runtime_param":
        return "runtime_tuning"
    if op == "add_few_shot":
        return "targeted_few_shot"
    if target.startswith("output") or op == "add_output_constraint":
        return "output_contract_tightening"
    return "prompt_rewrite"


def _mechanism_for_family(family: str) -> str:
    if family == "model_substitution":
        return "model_capability_probe"
    if family == "output_contract_tightening":
        return "output_contract_fix"
    if family == "targeted_few_shot":
        return "representative_examples"
    if family == "runtime_tuning":
        return "runtime_defect_fix"
    return "semantic_boundary_rewrite"


def _attach_affordance_ids(candidates: list[object], prompt: str) -> None:
    try:
        payload = json.loads(prompt.split("\n\n", 1)[1])
    except Exception:
        return
    affordances = payload.get("optimization_affordances") or []
    if not isinstance(affordances, list):
        return
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("affordance_ids"):
            continue
        matches = _matching_affordance_ids(candidate, affordances)
        if matches:
            candidate["applications"] = _candidate_applications(candidate, matches)
            for key in (
                "transform_family",
                "mechanism_class",
                "affordance_ids",
                "intervention",
                "transform_instance",
                "_test_patch",
                "_test_family",
                "_test_mechanism",
            ):
                candidate.pop(key, None)


def _matching_affordance_ids(candidate: dict[str, object], affordances: list[object]) -> list[str]:
    family = str(candidate.get("transform_family") or candidate.get("_test_family") or "")
    mechanism = str(candidate.get("mechanism_class") or candidate.get("_test_mechanism") or "")
    operations = _candidate_operations(candidate)
    matches: list[str] = []
    for affordance in affordances:
        if not isinstance(affordance, dict):
            continue
        if (affordance.get("family") or affordance.get("transform_family")) != family:
            continue
        if (affordance.get("mechanism") or affordance.get("mechanism_class")) != mechanism:
            continue
        if not operations:
            if affordance.get("target_kind") == "few_shot":
                matches.append(str(affordance.get("affordance_id") or ""))
            continue
        for operation in operations:
            if (
                operation.get("op") in set(affordance.get("ops") or affordance.get("allowed_ops") or [])
                and operation.get("target") in {affordance.get("target_name"), affordance.get("target_path")}
            ):
                matches.append(str(affordance.get("affordance_id") or ""))
    return [item for item in matches if item]


def _candidate_applications(candidate: dict[str, object], affordance_ids: list[str]) -> list[dict[str, object]]:
    intervention = candidate.get("intervention")
    if isinstance(intervention, dict) and intervention.get("kind") == "example_selection":
        payload = intervention.get("payload")
        return [
            {
                "affordance_id": affordance_ids[0],
                "selection": dict(payload) if isinstance(payload, dict) else {},
                "rationale": str(candidate.get("hypothesis") or ""),
            }
        ]
    operations = _candidate_operations(candidate)
    return [
        {
            "affordance_id": affordance_ids[min(index, len(affordance_ids) - 1)],
            "operation": operation,
            "rationale": str(candidate.get("hypothesis") or ""),
        }
        for index, operation in enumerate(operations)
    ]


def _candidate_operations(candidate: dict[str, object]) -> list[dict[str, object]]:
    intervention = candidate.get("intervention")
    if "_test_patch" in candidate:
        patch = candidate.get("_test_patch")
        operations = patch.get("operations") if isinstance(patch, dict) else None
        return [operation for operation in operations or [] if isinstance(operation, dict)]
    if not isinstance(intervention, dict) or intervention.get("kind") == "example_selection":
        return []
    payload = intervention.get("payload")
    if not isinstance(payload, dict):
        return []
    patch = payload.get("patch")
    if not isinstance(patch, dict):
        return []
    operations = patch.get("operations")
    if not isinstance(operations, list):
        return []
    return [operation for operation in operations if isinstance(operation, dict)]


def search_hypothesis_with_budget(budget_allocation: dict[str, float]) -> SearchHypothesis:
    family_states = {
        family: TransformFamilyState(
            family=family,
            state="active",
            suitability=share,
            budget_share=share,
            reason="test allocation",
        )
        for family, share in budget_allocation.items()
    }
    return SearchHypothesis(
        family_states=family_states,
        context_states={},
        target_slices=["global"],
        profile=BehaviorProfile(
            mean_score=0.0,
            pass_count=0,
            case_count=1,
            pass_rate=0.0,
            failure_labels={"failed": 1},
            category_metrics={},
            invalid_output_rate=0.0,
            mean_cost_usd=0.001,
            mean_total_tokens=100.0,
            median_latency_s=1.0,
            high_cost_case_ids=[],
            high_latency_case_ids=[],
            target_slices=["global"],
            weak_slice_count=1,
            runtime_error_rate=0.0,
            length_finish_rate=0.0,
            parser_fallback_rate=0.0,
        ),
        budget_allocation=budget_allocation,
        rationale="test allocation",
    )


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


class RawCandidateClient:
    def __init__(self, candidates: list[object]) -> None:
        self.candidates = candidates

    def create_response(self, **kwargs: object) -> object:
        prompt = str(kwargs.get("input", ""))
        candidates = list(self.candidates)
        _attach_affordance_ids(candidates, prompt)
        return type(
            "Response",
            (),
            {
                "output_text": json.dumps(
                    {
                        "experiments": [
                            {
                                "experiment_id": "exp_1",
                                "mechanism_class": "semantic_boundary_rewrite",
                                "mechanism": "semantic_boundary_rewrite",
                                "hypothesis": "Malformed candidate test.",
                                "candidates": candidates,
                            }
                        ]
                    }
                )
            },
        )()


class CapturingPatchClient:
    def __init__(self) -> None:
        self.input_text = ""

    def create_response(self, **kwargs: object) -> object:
        self.input_text = str(kwargs.get("input", ""))
        return type("Response", (), {"output_text": json.dumps({"experiments": []})})()


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


def make_labeled_summary(patch_hash: str, labels: list[list[str]]) -> PatchSummary:
    evaluations = []
    for index, case_labels in enumerate(labels, start=1):
        passed = not case_labels
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(id=f"case-{index}", split="dev", input=f"case {index}"),
                record=RunRecord(
                    output="ok" if passed else "wrong",
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(),
                ),
                grade=GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=case_labels),
            )
        )
    return PatchSummary(
        patch_hash=patch_hash,
        patch=AgentPatch(),
        split="dev",
        evaluations=evaluations,
    )


def make_sensitive_failed_summary() -> PatchSummary:
    case = EvalCase(
        id="case-secret",
        split="dev",
        input="customer ssn 123-45-6789",
        expected="private expected answer",
    )
    return PatchSummary(
        patch_hash="baseline",
        patch=AgentPatch(),
        split="dev",
        evaluations=[
            CaseEvaluation(
                case=case,
                record=RunRecord(
                    output="private wrong output",
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(raw_output_text="private raw transcript"),
                ),
                grade=GradeResult(
                    score=0.0,
                    passed=False,
                    labels=["failed"],
                    notes="private grading note",
                ),
            )
        ],
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

    def test_task_theory_exposes_model_capability_when_model_affordance_can_test_residual_failures(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="small",
            model_options=["small", "large"],
            instructions={"system_prompt": "Classify."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["instruction", "model"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        summary = make_labeled_summary("baseline", [["wrong_label"], [], ["wrong_label"], []])
        task_theory = build_task_theory(
            summary=summary,
            diagnoses=[],
            objective=objective,
        )
        affordances = generate_optimization_affordances(
            surface,
            objective=objective,
            active_families=["prompt_rewrite", "model_substitution"],
            evidence={"bottleneck_class": task_theory.bottleneck_class},
        )

        enriched = _task_theory_with_affordance_opportunities(
            task_theory=task_theory,
            affordances=affordances,
            current_dev=summary,
            proposals_log=[],
            objective=objective,
        )

        mechanisms = [
            row.get("mechanism_class")
            for row in enriched["experiment_opportunities"]
        ]
        self.assertIn("model_capability_probe", mechanisms)

    def test_task_theory_exposes_model_efficiency_without_residual_failures(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            model_options=["small", "large"],
            instructions={"system_prompt": "Classify."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective(
            mode="cost",
            constraints=OptimizationConstraints(allowed_edits=["model"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        summary = make_labeled_summary("baseline", [[], [], []])
        task_theory = build_task_theory(
            summary=summary,
            diagnoses=[],
            objective=objective,
        )
        affordances = generate_optimization_affordances(
            surface,
            objective=objective,
            active_families=["model_substitution"],
            evidence={"bottleneck_class": task_theory.bottleneck_class},
        )

        enriched = _task_theory_with_affordance_opportunities(
            task_theory=task_theory,
            affordances=affordances,
            current_dev=summary,
            proposals_log=[],
            objective=objective,
        )

        efficiency_opportunities = [
            row
            for row in enriched["experiment_opportunities"]
            if row.get("mechanism_class") == "efficiency_probe"
        ]
        self.assertTrue(efficiency_opportunities)
        self.assertIn(
            "model_substitution.efficiency_probe.model_choice.model",
            efficiency_opportunities[0]["affordance_ids"],
        )

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

    def test_candidate_implementer_validates_returned_patch(self) -> None:
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
        engine = CandidateImplementer(
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
        self.assertIn("Validated transform candidate implementations", analysis)

    def test_candidate_implementer_preserves_model_rank_and_logs_deferred_candidates(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="old",
            model_options=["old", "new"],
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = CandidateImplementer(
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

        self.assertEqual(len(proposals), 3)
        self.assertEqual(proposals[0].patch.operations[0].op, "change_model")
        self.assertEqual(proposals[0].transform_family, "model_substitution")
        self.assertEqual(engine.last_stats.valid_count, 3)
        self.assertEqual(engine.last_stats.returned_count, 3)
        self.assertEqual(len(engine.last_candidate_rows), 3)

    def test_candidate_implementer_preserves_same_group_arms(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = CandidateImplementer(
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
                    "rationale": "First prompt candidate.",
                    "expected_effect": "Improve failed cases.",
                },
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Use exact wording from the task.",
                        }
                    ],
                    "rationale": "Second prompt candidate.",
                    "expected_effect": "Improve failed cases differently.",
                },
                {
                    "operations": [
                        {
                            "op": "add_output_constraint",
                            "target": "output_contract",
                            "value": "Return only the final answer text.",
                        }
                    ],
                    "rationale": "Output contract candidate.",
                    "expected_effect": "Reduce format drift.",
                },
            ]
        )

        proposals, _ = engine.propose(
            make_summary("baseline", [0.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            search_hypothesis=search_hypothesis_with_budget(
                {"prompt_rewrite": 0.75, "output_contract_tightening": 0.25}
            ),
            proposal_budget=2,
        )

        self.assertEqual(
            [proposal.transform_family for proposal in proposals],
            ["prompt_rewrite", "prompt_rewrite", "output_contract_tightening"],
        )
        self.assertEqual(engine.last_stats.raw_count, 3)
        self.assertEqual(engine.last_stats.valid_count, 3)
        self.assertEqual(engine.last_stats.returned_count, 3)
        self.assertEqual(len(engine.last_candidate_rows), 3)
        self.assertFalse(engine.last_stats.invalid_reasons)

    def test_candidate_implementer_preserves_distinct_family_budget_groups(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        prompt_candidate = {
            "transform_family": "prompt_rewrite",
            "mechanism_class": "semantic_boundary_rewrite",
            "experiment_id": "exp_1",
            "candidate_role": "atomic",
            "target_slice": "global",
            "hypothesis": "Try a prompt edit.",
            "expected_effects": {"summary": "Improve failed cases."},
            "evaluation_plan": "full_dev",
            "intervention": {
                "kind": "patch",
                "payload": {
                    "patch": {
                        "operations": [
                            {
                                "op": "add_instruction",
                                "target": "instructions.system_prompt",
                                "value": "Answer with grounded evidence.",
                            }
                        ],
                        "rationale": "Prompt candidate.",
                        "expected_effect": "Improve failed cases.",
                    }
                },
            },
        }
        second_prompt = json.loads(json.dumps(prompt_candidate))
        prompt_candidate["comparison_group"] = "prompt_group_1"
        second_prompt["comparison_group"] = "prompt_group_2"
        second_prompt["intervention"]["payload"]["patch"]["operations"][0]["value"] = "Use exact wording from the task."
        engine._client = RawCandidateClient([prompt_candidate, second_prompt])

        proposals, _ = engine.propose(
            make_summary("baseline", [0.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            search_hypothesis=search_hypothesis_with_budget({"prompt_rewrite": 1.0}),
            proposal_budget=1,
        )

        self.assertEqual(len(proposals), 2)
        self.assertEqual(engine.last_stats.valid_count, 2)
        self.assertEqual(engine.last_stats.returned_count, 2)
        self.assertFalse(engine.last_invalid_candidate_rows)

    def test_candidate_implementer_accepts_categorical_runtime_patch(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            model_options=["small", "large"],
            instructions={"system_prompt": "Answer helpfully."},
            runtime={"reasoning_effort": "low", "output_cap": 128},
        )
        objective = OptimizationObjective(mode="cost")
        surface = SurfaceGenerator().generate(spec, objective)
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = FakePatchClient(
            [
                {
                    "operations": [
                        {
                            "op": "set_runtime_param",
                            "target": "runtime.reasoning_effort",
                            "value": "medium",
                        }
                    ],
                    "rationale": "Change runtime reasoning effort.",
                    "expected_effect": "Explore a categorical runtime setting.",
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
        self.assertEqual(targets, ["runtime.reasoning_effort"])

    def test_candidate_implementer_passes_task_theory_and_affordances(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        client = CapturingPatchClient()
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = client

        proposals, _ = engine.propose(
            make_labeled_summary("baseline", [["invalid_output"], []]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        self.assertEqual(proposals, [])
        self.assertNotIn('"planner_guidance"', client.input_text)
        self.assertIn('"optimization_affordances"', client.input_text)
        self.assertIn('"experiment_opportunity_mechanisms"', client.input_text)
        self.assertIn('"output_contract_fix"', client.input_text)
        self.assertIn("selection.source_case_ids", client.input_text)
        self.assertIn("no experiments returned", engine.last_stats.plan_audit["warnings"])

    def test_candidate_implementer_rejects_experiments_outside_requested_intents(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        client = FakePatchClient(
            [
                {
                    "operations": [
                        {
                            "op": "add_instruction",
                            "target": "instructions.system_prompt",
                            "value": "Clarify failed cases.",
                        }
                    ],
                    "rationale": "Clarify behavior.",
                    "expected_effect": "Improve correctness.",
                }
            ]
        )
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = client

        proposals, _ = engine.propose(
            make_summary("baseline", [0.0, 1.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            experiment_intents=[
                ExperimentIntent(
                    intent_id="requested_intent",
                    mechanism_class="semantic_boundary_rewrite",
                    hypothesis="Implement only this requested intent.",
                    affordance_ids=[
                        "prompt_rewrite.semantic_boundary_rewrite.task_instructions.instructions_system_prompt"
                    ],
                )
            ],
        )

        self.assertEqual(proposals, [])
        self.assertIn("requested_intent", engine.last_stats.plan_audit["missing_intent_ids"])
        self.assertTrue(
            any("does not match any requested experiment_intent" in reason for reason in engine.last_stats.invalid_reasons or {})
        )

    def test_candidate_implementer_prompt_includes_experiment_opportunities(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Classify intent."},
            output_contract="Return JSON with label.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        client = CapturingPatchClient()
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = client
        summary = PatchSummary(
            patch_hash="baseline",
            patch=AgentPatch.empty(),
            split="dev",
            evaluations=[
                CaseEvaluation(
                    case=EvalCase(
                        id="dev-beta-1",
                        split="dev",
                        input="beta message",
                        expected={"label": "beta"},
                    ),
                    record=RunRecord(
                        output={"label": "alpha"},
                        metrics=OperationalMetrics(
                            latency_s=1.0,
                            input_tokens=10,
                            output_tokens=5,
                            total_tokens=15,
                            cost_usd=0.001,
                        ),
                        diagnostics=DiagnosticTrace(),
                    ),
                    grade=GradeResult(
                        score=0.0,
                        passed=False,
                        labels=["wrong_label", "expected:beta", "actual:alpha"],
                    ),
                )
            ],
        )
        train_cases = (
            EvalCase(id="train-alpha-1", split="train", input="alpha sample", expected={"label": "alpha"}),
            EvalCase(id="train-beta-1", split="train", input="beta sample", expected={"label": "beta"}),
        )

        proposals, _ = engine.propose(
            summary,
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            search_hypothesis=search_hypothesis_with_budget(
                {"prompt_rewrite": 0.5, "targeted_few_shot": 0.5}
            ),
            proposal_example_bank=build_proposal_example_bank(train_cases),
            proposal_example_cases=train_cases,
        )

        self.assertEqual(proposals, [])
        self.assertIn('"experiment_opportunity_mechanisms"', client.input_text)
        self.assertIn('"expected":"beta"', client.input_text)
        self.assertIn('"actual":"alpha"', client.input_text)
        self.assertIn('"train-beta-1"', client.input_text)

    def test_targeted_few_shot_source_ids_materialize_from_train_bank(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Classify."},
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        train_cases = (
            EvalCase(id="train-card-1", split="train", input="card charge I do not know", expected={"label": "card_payment_not_recognised"}),
            EvalCase(id="train-card-2", split="train", input="unknown card transaction", expected={"label": "card_payment_not_recognised"}),
        )
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = RawCandidateClient(
            [
                {
                    "transform_family": "targeted_few_shot",
                    "mechanism_class": "representative_examples",
                    "candidate_role": "atomic",
                    "intervention": {
                        "kind": "example_selection",
                        "payload": {
                            "source_case_ids": ["train-card-1", "train-card-2"],
                            "selection_strategy": "representative",
                        },
                    },
                    "hypothesis": "Anchor the weak card payment label.",
                }
            ]
        )

        proposals, _ = engine.propose(
            make_labeled_summary("baseline", [["wrong_label"], []]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            search_hypothesis=search_hypothesis_with_budget({"targeted_few_shot": 1.0}),
            proposal_example_bank=build_proposal_example_bank(train_cases),
            proposal_example_cases=train_cases,
            proposal_budget=1,
        )

        self.assertEqual(len(proposals), 1)
        first_value = proposals[0].patch.operations[0].value
        self.assertEqual(first_value[0]["source_case_id"], "train-card-1")
        self.assertEqual(first_value[0]["input"], "card charge I do not know")
        self.assertEqual(first_value[0]["output"], {"label": "card_payment_not_recognised"})
        self.assertEqual(len(first_value), 2)
        self.assertEqual(proposals[0].candidate_role, "atomic")
        self.assertEqual(engine.last_stats.valid_count, 1)
        self.assertEqual(engine.last_stats.returned_count, 1)
        self.assertFalse(engine.last_stats.invalid_reasons)
        self.assertTrue(engine.last_candidate_rows[0]["materialization"]["materialized"])

    def test_targeted_few_shot_rejects_inline_examples_and_unknown_ids(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Classify."},
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        train_cases = (
            EvalCase(id="train-card-1", split="train", input="card charge I do not know", expected={"label": "card_payment_not_recognised"}),
        )
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = RawCandidateClient(
            [
                {
                    "transform_family": "targeted_few_shot",
                    "mechanism_class": "representative_examples",
                    "candidate_role": "atomic",
                    "hypothesis": "Inline examples should be rejected.",
                    "intervention": {
                        "kind": "patch",
                        "payload": {
                            "patch": {
                                "operations": [
                                    {
                                        "op": "add_few_shot",
                                        "target": "few_shot",
                                        "value": [{}],
                                    }
                                ],
                                "rationale": "Malformed inline example.",
                                "expected_effect": "Should be rejected.",
                            }
                        },
                    },
                },
                {
                    "transform_family": "targeted_few_shot",
                    "mechanism_class": "representative_examples",
                    "candidate_role": "atomic",
                    "intervention": {
                        "kind": "example_selection",
                        "payload": {"source_case_ids": ["missing-train-case"]},
                    },
                    "hypothesis": "Unknown IDs should be rejected.",
                },
            ]
        )

        proposals, _ = engine.propose(
            make_labeled_summary("baseline", [["wrong_label"], []]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            search_hypothesis=search_hypothesis_with_budget({"targeted_few_shot": 1.0}),
            proposal_example_bank=build_proposal_example_bank(train_cases),
            proposal_example_cases=train_cases,
            proposal_budget=2,
        )

        self.assertEqual(proposals, [])
        reasons = engine.last_stats.invalid_reasons or {}
        self.assertTrue(any("must use selection" in reason for reason in reasons))
        self.assertTrue(any("unknown few-shot source_case_ids" in reason for reason in reasons))

    def test_targeted_few_shot_rejects_source_ids_outside_applications(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Classify."},
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        train_cases = (
            EvalCase(id="train-card-1", split="train", input="card charge I do not know", expected={"label": "card_payment_not_recognised"}),
        )
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = RawCandidateClient(
            [
                {
                    "transform_family": "targeted_few_shot",
                    "mechanism_class": "representative_examples",
                    "candidate_role": "atomic",
                    "transform_parameters": {"source_case_ids": ["train-card-1"]},
                    "intervention": {"kind": "example_selection", "payload": {}},
                    "hypothesis": "Candidate-level transform_parameters should not satisfy the contract.",
                }
            ]
        )

        proposals, _ = engine.propose(
            make_labeled_summary("baseline", [["wrong_label"], []]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
            search_hypothesis=search_hypothesis_with_budget({"targeted_few_shot": 1.0}),
            proposal_example_bank=build_proposal_example_bank(train_cases),
            proposal_example_cases=train_cases,
            proposal_budget=1,
        )

        self.assertEqual(proposals, [])
        self.assertTrue(
            any("candidate transform_parameters are derived" in reason for reason in (engine.last_stats.invalid_reasons or {}))
        )

    def test_candidate_implementer_rejects_bare_patches(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        engine = CandidateImplementer(
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
                    "rationale": "Bare patch.",
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

    def test_candidate_implementer_logs_malformed_raw_candidates(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = RawCandidateClient(
            [
                "not an object",
                {
                    "transform_family": "prompt_rewrite",
                    "hypothesis": "missing patch should be logged",
                    "patch": 1,
                },
            ]
        )

        proposals, _ = engine.propose(
            make_summary("baseline", [0.0]),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        self.assertEqual(proposals, [])
        self.assertEqual(engine.last_stats.raw_count, 2)
        self.assertEqual(engine.last_stats.valid_count, 0)
        self.assertEqual(engine.last_stats.invalid_count, 2)
        self.assertEqual(len(engine.last_invalid_candidate_rows), 2)
        reasons = engine.last_stats.invalid_reasons or {}
        self.assertIn("candidate entry is not an object", reasons)
        self.assertTrue(any(reason.startswith("malformed candidate:") for reason in reasons))

    def test_candidate_implementer_rejects_inactive_family_candidate(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective()
        surface = SurfaceGenerator().generate(spec, objective)
        engine = CandidateImplementer(
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

    def test_candidate_implementer_redacts_diagnostic_examples_when_configured(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer helpfully."},
            output_contract="Return text.",
        )
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(sanitize_examples=True),
        )
        surface = SurfaceGenerator().generate(spec, objective)
        client = CapturingPatchClient()
        engine = CandidateImplementer(
            env_path=".env",
            model="gpt-5.4-mini",
            reasoning_effort="low",
        )
        engine._client = client

        proposals, _ = engine.propose(
            make_sensitive_failed_summary(),
            surface,
            objective=objective,
            diagnosis=None,
            seen_hashes=set(),
            current_spec=spec,
            history=[],
        )

        self.assertEqual(proposals, [])
        self.assertIn('"sanitized":true', client.input_text)
        self.assertIn("[redacted by sanitize_examples]", client.input_text)
        self.assertNotIn("123-45-6789", client.input_text)
        self.assertNotIn("private expected answer", client.input_text)
        self.assertNotIn("private wrong output", client.input_text)
        self.assertNotIn("private raw transcript", client.input_text)
        self.assertNotIn("private grading note", client.input_text)
        self.assertEqual(engine.last_call_diagnostics["prompt_chars"], len(client.input_text))
        self.assertEqual(
            engine.last_call_diagnostics["prompt_approx_tokens"],
            prompt_size_profile(client.input_text)["approx_tokens"],
        )
        self.assertLess(prompt_size_profile(client.input_text)["chars"], 30000)


if __name__ == "__main__":
    unittest.main()
