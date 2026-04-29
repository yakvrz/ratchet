from __future__ import annotations

import unittest

from ratchet.errors import OptimizerModelError
from ratchet.evidence_ledger import EvidenceLedger
from ratchet.experiments import ExperimentIntent, ResearchState
from ratchet.optimizer import (
    CandidateEvaluationState,
    _compact_recent_history_for_theory,
    _measurement_action,
    _measurement_budget_exhausted,
    _research_state_packet,
)
from ratchet.results import CaseEvaluation, PatchSummary
from ratchet.research import MeasurementSelector, ResearchPlanner
from ratchet.transforms import CandidateAffordanceApplication, CandidateProposal, TransformContextKey
from ratchet.types import AgentPatch, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, PatchOperation, RunRecord


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
                affordance_id="prompt_rewrite.semantic_boundary_rewrite.task_instructions.instructions_system",
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


def _summary(patch_hash: str, scores: list[float], *, cost_usd: float = 0.001) -> PatchSummary:
    evaluations: list[CaseEvaluation] = []
    for index, score in enumerate(scores, start=1):
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(id=f"case-{index}", split="dev", input=f"input {index}", expected="ok"),
                record=RunRecord(
                    output={"answer": "ok"},
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=cost_usd,
                    ),
                    diagnostics=DiagnosticTrace(metadata={"finish_reason": "stop"}),
                ),
                grade=GradeResult(score=score, passed=score >= 1.0),
            )
        )
    return PatchSummary(
        patch_hash=patch_hash,
        patch=AgentPatch.empty(),
        split="dev",
        evaluations=evaluations,
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

    def test_selector_state_includes_measurement_budget_and_deployed_ratios(self) -> None:
        baseline = _summary("baseline", [1.0, 0.0], cost_usd=0.001)
        candidate_summary = _summary("patch-1", [1.0, 1.0], cost_usd=0.010)
        state = _state(1)
        state.summary = candidate_summary
        ledger = EvidenceLedger()
        ledger.add(
            candidate_id=state.patch_hash,
            stage="small_dev",
            reference=baseline,
            baseline=baseline,
            candidate=candidate_summary,
            mechanism_class=state.candidate.mechanism_class,
            affordance_ids=state.candidate.affordance_ids,
            comparison_group=state.candidate.comparison_group,
            candidate_role=state.candidate.candidate_role,
            rejection_reason=None,
            constraint_warning=None,
        )

        packet = _research_state_packet(
            objective=OptimizationObjective(),
            stage_name="full_dev",
            reference=baseline,
            baseline=baseline,
            states=[state],
            proposals_log=[],
            dev_evaluations_used=1,
            dev_budget=4,
            evidence_ledger=ledger,
            stage_cases=tuple(
                EvalCase(id=f"case-{index}", split="dev", input="", expected="")
                for index in range(1, 5)
            ),
            samples_per_case=1,
            measurement_cost_used_usd=0.02,
            max_measurement_cost_usd=0.05,
            measurement_tool_calls_used=1,
            max_measurement_tool_calls=10,
            measurement_turns_used=2,
            max_measurement_turns=20,
        )

        row = packet["evidence_ledger"]["candidate_evidence"][0]
        self.assertAlmostEqual(packet["budget"]["remaining_measurement_budget_usd"], 0.03)
        self.assertAlmostEqual(row["marginal_measurement_cost_usd"], 0.02)
        self.assertAlmostEqual(row["remaining_measurement_budget_usd"], 0.03)
        self.assertAlmostEqual(row["deployed_cost_ratio"], 10.0)
        self.assertEqual(packet["budget"]["remaining_measurement_tool_calls"], 9)
        self.assertEqual(packet["budget"]["remaining_measurement_turns"], 18)

    def test_measurement_budget_guard_blocks_only_hard_budget_overrun(self) -> None:
        self.assertFalse(_measurement_budget_exhausted(used_usd=0.05, marginal_usd=0.04, max_usd=0.10))
        self.assertTrue(_measurement_budget_exhausted(used_usd=0.08, marginal_usd=0.04, max_usd=0.10))
        self.assertFalse(_measurement_budget_exhausted(used_usd=10.0, marginal_usd=10.0, max_usd=None))

    def test_experiment_intent_rejects_unknown_roles(self) -> None:
        intent = ExperimentIntent.from_dict(
            {
                "intent_id": "intent_1",
                "mechanism_class": "runtime_defect_fix",
                "hypothesis": "Test runtime.",
                "affordance_ids": ["runtime.output_cap"],
            },
        )

        self.assertEqual(intent.affordance_ids, ["runtime.output_cap"])
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

    def test_research_theorist_history_compaction_drops_raw_stage_payloads(self) -> None:
        large_payload = {"case_rows": ["x" * 1000 for _ in range(20)]}
        rows = _compact_recent_history_for_theory(
            [
                {
                    "candidate": True,
                    "patch_hash": "patch-1",
                    "parent_patch_hash": "parent",
                    "hypothesis": "h" * 1000,
                    "expected_effects": "e" * 1000,
                    "mechanism_class": "semantic_boundary_rewrite",
                    "transform_family": "prompt_rewrite",
                    "candidate_role": "atomic",
                    "target_slice": "slice",
                    "accepted": False,
                    "frontier_status": "failed",
                    "rejection_reason": "regressed",
                    "behavior_flip_summary": {
                        "fixed_count": 1,
                        "regressed_count": 2,
                        "invalid_output_delta": -1,
                        "raw": large_payload,
                    },
                    "evaluation_stages": [
                        {
                            "stage": "full_dev",
                            "case_count": 96,
                            "passed": False,
                            "comparison_to_parent": {
                                "score_delta": -0.1,
                                "pass_rate_delta": -0.1,
                                "cost_delta": 0.01,
                                "latency_delta": 0.2,
                                "token_delta": 20,
                            },
                            "behavior_flip_summary": large_payload,
                        }
                    ],
                }
            ],
            limit=4,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertNotIn("evaluation_stages", row)
        self.assertNotIn("raw", row["behavior_flips"])
        self.assertLessEqual(len(row["hypothesis"]), 320)
        self.assertEqual(row["latest_stage"]["stage"], "full_dev")


if __name__ == "__main__":
    unittest.main()
