from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentIntent
from ratchet.optimizer import CandidateEvaluationState, _research_evaluate_action
from ratchet.transforms import CandidateProposal, TransformContextKey
from ratchet.types import AgentPatch, PatchOperation
from ratchet.research import ResearchAction, ResearchDecision, validate_research_decision
from ratchet.research import ResearchController


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
        transform_family="prompt_rewrite",
        mechanism_class="semantic_boundary_rewrite",
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
    )
    return CandidateEvaluationState(
        candidate=candidate,
        patch=candidate.patch,
        patch_hash=f"patch-{index}",
        proposal_patch_hash=f"proposal-{index}",
        transform_context=TransformContextKey.from_candidate(candidate),
    )


class ResearchControllerTests(unittest.TestCase):
    def test_validate_research_decision_accepts_known_candidate_ids(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            stage="full_dev",
            candidate_ids=["a", "b"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["a"],
            skipped_candidate_reasons={"b": "lower information value"},
        )

        self.assertEqual(validate_research_decision(decision, [action]), action)

    def test_validate_research_decision_rejects_unknown_action(self) -> None:
        action = ResearchAction(action_id="evaluate_full_dev", action_type="evaluate_candidates")
        decision = ResearchDecision(action_id="stop", action_type="stop")

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_validate_research_decision_rejects_unknown_candidate(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["b"],
            skipped_candidate_reasons={"a": "not selected"},
        )

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_validate_research_decision_rejects_over_selection(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a", "b"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["a", "b"],
        )

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_validate_research_decision_requires_skip_reasons(self) -> None:
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a", "b"],
            max_select=1,
        )
        decision = ResearchDecision(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            selected_candidate_ids=["a"],
        )

        with self.assertRaises(OptimizerModelError):
            validate_research_decision(decision, [action])

    def test_plan_experiments_decision_requires_experiment_intents(self) -> None:
        action = ResearchAction(action_id="plan_experiments", action_type="plan_experiments")
        with self.assertRaises(OptimizerModelError):
            validate_research_decision(
                ResearchDecision(action_id="plan_experiments", action_type="plan_experiments"),
                [action],
            )

        decision = ResearchDecision(
            action_id="plan_experiments",
            action_type="plan_experiments",
            experiment_intents=[
                ExperimentIntent(
                    intent_id="intent_1",
                    mechanism_class="semantic_boundary_rewrite",
                    hypothesis="Test whether a boundary rewrite improves failed cases.",
                )
            ],
        )

        self.assertEqual(validate_research_decision(decision, [action]), action)

    def test_late_full_dev_action_exposes_one_hard_selection_slot(self) -> None:
        action = _research_evaluate_action(
            stage_name="full_dev",
            states=[_state(1), _state(2), _state(3)],
            dev_evaluations_used=4,
            dev_budget=8,
        )

        self.assertEqual(action.max_select, 1)
        self.assertTrue(action.metadata["late_budget"])
        self.assertGreater(action.metadata["raw_max_select"], action.max_select)

    def test_research_controller_repairs_invalid_json_response(self) -> None:
        controller = ResearchController(env_path=".env", model="fake", reasoning_effort="low")
        client = _RepairClient()
        controller._client = client
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a", "b"],
            max_select=1,
        )

        decision = controller.decide(state={}, allowed_actions=[action])

        self.assertEqual(decision.selected_candidate_ids, ["a"])
        self.assertEqual(client.calls, 2)
        self.assertTrue((controller.last_call_diagnostics or {}).get("repair_attempted"))

    def test_research_controller_repairs_invalid_decision_contract(self) -> None:
        controller = ResearchController(env_path=".env", model="fake", reasoning_effort="low")
        client = _RepairClient(
            [
                '{"action_id":"evaluate_full_dev","action_type":"evaluate_candidates",'
                '"selected_candidate_ids":["a"],"rationale":"bad","expected_information":"info",'
                '"risks":"none","skipped_candidate_reasons":{}}',
                '{"action_id":"evaluate_full_dev","action_type":"evaluate_candidates",'
                '"selected_candidate_ids":["a"],"rationale":"fixed","expected_information":"info",'
                '"risks":"none","skipped_candidate_reasons":{"b":"skip"}}',
            ]
        )
        controller._client = client
        action = ResearchAction(
            action_id="evaluate_full_dev",
            action_type="evaluate_candidates",
            candidate_ids=["a", "b"],
            max_select=1,
        )

        decision = controller.decide(state={}, allowed_actions=[action])

        self.assertEqual(decision.selected_candidate_ids, ["a"])
        self.assertEqual(decision.skipped_candidate_reasons, {"b": "skip"})
        self.assertEqual(client.calls, 2)
        self.assertTrue((controller.last_call_diagnostics or {}).get("repair_attempted"))


if __name__ == "__main__":
    unittest.main()
