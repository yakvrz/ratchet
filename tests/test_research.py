from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentIntent, ResearchState
from ratchet.optimizer import CandidateEvaluationState, _measurement_action
from ratchet.research import MeasurementSelector, ResearchPlanner
from ratchet.transforms import CandidateAffordanceApplication, CandidateProposal, TransformContextKey
from ratchet.types import AgentPatch, PatchOperation


class _Response:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class _RepairClient:
    def __init__(self, outputs: list[str] | None = None) -> None:
        self.calls = 0
        self.outputs = outputs

    def create_response(self, **_: object) -> _Response:
        self.calls += 1
        if self.outputs is not None:
            return _Response(self.outputs[self.calls - 1])
        if self.calls == 1:
            return _Response('{"action_id":"evaluate_full_dev","action_type":"evaluate_candidates","selected_candidate_ids":[')
        return _Response(
            '{"action_id":"evaluate_full_dev","action_type":"evaluate_candidates",'
            '"selected_candidate_ids":["a"],"rationale":"fixed","expected_information":"info",'
            '"risks":"none","skipped_candidate_reasons":{"b":"skip"}}'
        )


def _state(index: int) -> CandidateEvaluationState:
    candidate = CandidateProposal(
        comparison_group="same-group",
        patch=AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system",
                    value=f"rule {index}",
                )
            ]
        ),
        applications=[
            CandidateAffordanceApplication(
                affordance_id="prompt_rewrite.semantic_boundary_rewrite.instruction.instructions_system",
                operation=PatchOperation(
                    op="add_instruction",
                    target="instructions.system",
                    value=f"rule {index}",
                ),
            )
        ],
    )
    return CandidateEvaluationState(
        candidate=candidate,
        patch=candidate.patch,
        patch_hash=f"patch-{index}",
        proposal_patch_hash=f"proposal-{index}",
        transform_context=TransformContextKey.from_candidate(candidate),
    )


class ResearchRoleTests(unittest.TestCase):
    def test_research_planner_rejects_unknown_affordance_id(self) -> None:
        planner = ResearchPlanner(env_path=".env", model="fake", reasoning_effort="low")
        planner._client = _RepairClient(
            [
                (
                    '{"experiment_intents":[{"intent_id":"intent_1",'
                    '"mechanism_class":"semantic_boundary_rewrite",'
                    '"hypothesis":"test","affordance_ids":["missing"]}]}'
                )
            ]
        )
        state = ResearchState(
            objective={},
            budget={},
            parent={},
            task_theory={},
            behavior_profile={},
            affordances=[
                {
                    "affordance_id": "aff_prompt",
                    "transform_family": "prompt_rewrite",
                    "mechanism_class": "semantic_boundary_rewrite",
                }
            ],
            prior_experiment_outcomes=[],
            frontier={},
        )

        with self.assertRaises(OptimizerModelError):
            planner.plan(state)

    def test_measurement_selector_rejects_created_candidate_ids(self) -> None:
        selector = MeasurementSelector(env_path=".env", model="fake", reasoning_effort="low")
        selector._client = _RepairClient(
            [
                (
                    '{"selected_candidate_ids":["new_candidate"],"rationale":"bad",'
                    '"expected_information":"info","risks":"none",'
                    '"skipped_candidate_reasons":{"a":"skip"}}'
                )
            ]
        )

        with self.assertRaises(OptimizerModelError):
            selector.select(
                stage="full_dev",
                state={"evidence_ledger": {"candidate_evidence": [{"candidate_id": "a"}]}},
                candidate_ids=["a"],
                max_select=1,
            )

    def test_measurement_selector_repairs_invalid_json_syntax(self) -> None:
        selector = MeasurementSelector(env_path=".env", model="fake", reasoning_effort="low")
        selector._client = _RepairClient()

        decision = selector.select(
            stage="full_dev",
            state={
                "evidence_ledger": {
                    "candidate_evidence": [
                        {"candidate_id": "a"},
                        {"candidate_id": "b"},
                    ]
                }
            },
            candidate_ids=["a", "b"],
            max_select=1,
        )

        self.assertEqual(decision.selected_candidate_ids, ["a"])
        self.assertEqual(selector._client.calls, 2)
        self.assertTrue(selector.last_call_diagnostics["repair_attempted"])

    def test_measurement_selector_fills_missing_skip_reasons(self) -> None:
        selector = MeasurementSelector(env_path=".env", model="fake", reasoning_effort="low")
        selector._client = _RepairClient(
            [
                (
                    '{"selected_candidate_ids":["a"],"rationale":"pick a",'
                    '"expected_information":"info","risks":"none",'
                    '"skipped_candidate_reasons":{}}'
                )
            ]
        )

        decision = selector.select(
            stage="full_dev",
            state={
                "evidence_ledger": {
                    "candidate_evidence": [
                        {"candidate_id": "a"},
                        {"candidate_id": "b"},
                    ]
                }
            },
            candidate_ids=["a", "b"],
            max_select=1,
        )

        self.assertEqual(
            decision.skipped_candidate_reasons["b"],
            "not selected by measurement selector",
        )

    def test_measurement_selector_requires_evidence_ledger(self) -> None:
        selector = MeasurementSelector(env_path=".env", model="fake", reasoning_effort="low")
        selector._client = _RepairClient(
            [
                (
                    '{"selected_candidate_ids":[],"rationale":"bad",'
                    '"expected_information":"info","risks":"none",'
                    '"skipped_candidate_reasons":{"a":"skip"}}'
                )
            ]
        )

        with self.assertRaises(OptimizerModelError):
            selector.select(
                stage="full_dev",
                state={"candidates": [{"candidate_id": "a"}]},
                candidate_ids=["a"],
                max_select=1,
            )

    def test_experiment_intent_normalizes_all_family_marker_and_rejects_unknown_roles(self) -> None:
        intent = ExperimentIntent.from_dict(
            {
                "intent_id": "intent_1",
                "mechanism_class": "runtime_defect_fix",
                "hypothesis": "Test runtime.",
                "allowed_families": ["all"],
            },
        )

        self.assertEqual(intent.allowed_families, [])
        with self.assertRaises(ValueError):
            ExperimentIntent.from_dict(
                {
                    "intent_id": "intent_2",
                    "mechanism_class": "runtime_defect_fix",
                    "hypothesis": "Test runtime.",
                    "candidate_roles": ["generator"],
                },
            )

    def test_late_full_dev_action_exposes_one_hard_selection_slot(self) -> None:
        action = _measurement_action(
            stage_name="full_dev",
            states=[_state(1), _state(2), _state(3)],
            dev_evaluations_used=4,
            dev_budget=8,
        )

        self.assertEqual(action.max_select, 1)
        self.assertTrue(action.metadata["late_budget"])
        self.assertGreater(action.metadata["raw_max_select"], action.max_select)


if __name__ == "__main__":
    unittest.main()
