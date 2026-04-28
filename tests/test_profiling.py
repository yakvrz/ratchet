from __future__ import annotations

import unittest

from ratchet.evidence import ProposalExample, ProposalExampleBank
from ratchet.profiling import (
    _phase_attempt_durations,
    _phase_durations,
    quality_cost_tradeoffs,
    runtime_reliability_diagnostics,
)
from ratchet.proposals import _few_shot_count_variants, _materialize_candidate_references
from ratchet.optimizer import (
    CandidateEvaluationState,
    _prefer_simple_few_shot_strategy,
    _select_full_dev_candidates,
    _simplification_variants,
)
from ratchet.results import CaseEvaluation, Comparison, PatchSummary
from ratchet.transforms import CandidateProposal, Intervention, TransformContextKey
from ratchet.types import (
    AgentPatch,
    DiagnosticTrace,
    EditableTarget,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    OptimizationObjective,
    PatchOperation,
    RunRecord,
)
from ratchet.validation import PatchValidator


def summary(
    patch: AgentPatch,
    *,
    passed: bool,
    output: object,
    labels: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> PatchSummary:
    case = EvalCase(id="case-1", split="dev", input="x", expected={"label": "ok"})
    evaluation = CaseEvaluation(
        case=case,
        record=RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=100,
                output_tokens=8,
                total_tokens=108,
                cost_usd=0.001,
            ),
            diagnostics=DiagnosticTrace(
                raw_output_text=str(output),
                metadata=metadata or {},
            ),
        ),
        grade=GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=labels or []),
    )
    return PatchSummary(
        patch_hash="candidate" if patch.operations else "baseline",
        patch=patch,
        split="dev",
        evaluations=[evaluation],
    )


def selection_summary(
    patch: AgentPatch,
    *,
    pass_count: int,
    case_count: int = 10,
    cost_usd: float = 0.001,
) -> PatchSummary:
    evaluations = []
    for index in range(case_count):
        passed = index < pass_count
        case = EvalCase(id=f"case-{index}", split="dev", input=f"x {index}", expected={"label": "ok"})
        evaluations.append(
            CaseEvaluation(
                case=case,
                record=RunRecord(
                    output={"label": "ok" if passed else "wrong"},
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=100,
                        output_tokens=8,
                        total_tokens=108,
                        cost_usd=cost_usd,
                    ),
                    diagnostics=DiagnosticTrace(raw_output_text="{}"),
                ),
                grade=GradeResult(score=1.0 if passed else 0.0, passed=passed),
            )
        )
    return PatchSummary(
        patch_hash=f"summary-{pass_count}-{cost_usd}",
        patch=patch,
        split="dev",
        evaluations=evaluations,
    )


def candidate_state(
    candidate: CandidateProposal,
    *,
    pass_count: int,
    cost_usd: float = 0.001,
    score_delta: float | None = None,
) -> CandidateEvaluationState:
    return CandidateEvaluationState(
        candidate=candidate,
        patch=candidate.patch,
        patch_hash=f"patch-{candidate.transform_family}-{pass_count}-{cost_usd}",
        proposal_patch_hash=f"proposal-{candidate.transform_family}-{pass_count}-{cost_usd}",
        transform_context=TransformContextKey.from_candidate(candidate),
        summary=selection_summary(candidate.patch, pass_count=pass_count, cost_usd=cost_usd),
        comparison=(
            Comparison(
                score_delta=score_delta,
                score_ci=(score_delta, score_delta),
                cost_delta=0.0,
                cost_ci=(0.0, 0.0),
                token_delta=0.0,
                token_ci=(0.0, 0.0),
                latency_delta=0.0,
                latency_ci=(0.0, 0.0),
            )
            if score_delta is not None
            else None
        ),
    )


def few_shot_candidate(example_count: int, *, comparison_group: str = "few-shot-exp") -> CandidateProposal:
    examples = [
        {"source_case_id": f"train-{index}", "input": f"example {index}", "output": {"label": "ok"}}
        for index in range(example_count)
    ]
    return CandidateProposal(
        transform_family="targeted_few_shot",
        intervention=Intervention(kind="example_selection", payload={}),
        transform_parameters={
            "few_shot_example_count": example_count,
            "original_few_shot_example_count": 3,
            "selection_strategy": "representative",
        },
        comparison_group=comparison_group,
        candidate_role="compression" if example_count < 3 else "atomic",
        patch=AgentPatch(
            operations=[PatchOperation(op="add_few_shot", target="few_shot", value=examples)],
            metadata={
                "few_shot_variant": {
                    "example_count": example_count,
                    "original_example_count": 3,
                    "selection_strategy": "representative",
                    "source_case_ids": [item["source_case_id"] for item in examples],
                }
            },
        ),
    )


def prompt_candidate(name: str, *, comparison_group: str = "prompt-exp") -> CandidateProposal:
    return CandidateProposal(
        transform_family="prompt_rewrite",
        intervention=Intervention(kind="patch", payload={}),
        transform_instance=name,
        comparison_group=comparison_group,
        patch=AgentPatch(
            operations=[
                PatchOperation(
                    op="revise_instruction",
                    target="instructions.system",
                    value=f"Classify carefully: {name}",
                )
            ]
        ),
    )


def output_contract_candidate(name: str, *, comparison_group: str = "contract-exp") -> CandidateProposal:
    return CandidateProposal(
        transform_family="output_contract_tightening",
        mechanism_class="output_contract_fix",
        intervention=Intervention(kind="patch", payload={}),
        transform_instance=name,
        comparison_group=comparison_group,
        patch=AgentPatch(
            operations=[
                PatchOperation(
                    op="add_output_constraint",
                    target="output_contract",
                    value=f"Return compact valid JSON only: {name}",
                )
            ]
        ),
    )


def experiment_role_candidate(
    name: str,
    *,
    role: str,
    comparison_group: str = "experiment-exp",
) -> CandidateProposal:
    candidate = prompt_candidate(name, comparison_group=comparison_group)
    return CandidateProposal(
        transform_family=candidate.transform_family,
        intervention=candidate.intervention,
        transform_instance=candidate.transform_instance,
        transform_parameters=dict(candidate.transform_parameters),
        mechanism_class=candidate.mechanism_class,
        experiment_id=candidate.experiment_id,
        candidate_role=role,
        comparison_group=candidate.comparison_group,
        target_slice=candidate.target_slice,
        hypothesis=candidate.hypothesis,
        expected_effects=dict(candidate.expected_effects),
        evaluation_plan=candidate.evaluation_plan,
        patch=candidate.patch,
    )


class ProfilingTests(unittest.TestCase):
    def test_phase_durations_report_wall_time_for_overlapping_attempts(self) -> None:
        rows = [
            {"event": "candidate_evaluation_started", "elapsed_s": 10.0},
            {"event": "candidate_evaluation_started", "elapsed_s": 11.0},
            {"event": "candidate_evaluated", "elapsed_s": 20.0},
            {"event": "candidate_evaluated", "elapsed_s": 21.0},
        ]

        self.assertEqual(_phase_durations(rows)["candidate_evaluation"], 11.0)
        self.assertEqual(_phase_attempt_durations(rows)["candidate_evaluation"], 20.0)

    def test_runtime_only_invalid_output_fix_below_cap_is_suspicious(self) -> None:
        baseline = summary(
            AgentPatch.empty(),
            passed=False,
            output={"label": "invalid", "invalid_output": "{\"label\""},
            labels=["invalid_output"],
            metadata={"requested_output_cap": 512, "finish_reason": "stop", "invalid_output": True},
        )
        candidate_patch = AgentPatch(
            operations=[
                PatchOperation(op="set_runtime_param", target="runtime.output_cap", value=1024)
            ]
        )
        candidate = summary(
            candidate_patch,
            passed=True,
            output={"label": "ok"},
            metadata={"requested_output_cap": 1024, "finish_reason": "stop"},
        )

        diagnostics = runtime_reliability_diagnostics(baseline, candidate)

        self.assertTrue(diagnostics["runtime_finding"])
        self.assertTrue(diagnostics["baseline_runtime_defect_fixed"])
        self.assertEqual(diagnostics["diagnostic_class"], "baseline_runtime_defect_fixed")
        self.assertNotIn("suspicious", diagnostics)
        self.assertEqual(diagnostics["fixed_invalid_output_case_ids"], ["case-1"])
        self.assertEqual(diagnostics["low_token_fixed_case_ids"], ["case-1"])

    def test_few_shot_source_references_are_materialized(self) -> None:
        bank = ProposalExampleBank(
            examples=[
                ProposalExample(
                    case_id="train-1",
                    input="How do I verify identity?",
                    expected={"label": "verify_my_identity"},
                    metadata={"category": "identity"},
                    label="verify_my_identity",
                )
            ],
            label_counts={"verify_my_identity": 1},
            metadata_categories={"identity": 1},
            label_field="label",
        )
        candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            intervention=Intervention(kind="example_selection", payload={"source_case_ids": ["train-1"]}),
            transform_parameters={"source_case_ids": ["train-1"]},
            hypothesis="Add a representative identity example.",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_few_shot",
                        target="few_shot",
                        value=[{"source_case_id": "train-1", "purpose": "identity confusion"}],
                    )
                ]
            ),
        )

        materialized, materialization = _materialize_candidate_references(candidate, bank)

        value = materialized.patch.operations[0].value
        self.assertEqual(value[0]["input"], "How do I verify identity?")
        self.assertEqual(value[0]["output"], {"label": "verify_my_identity"})
        self.assertEqual(materialization["source_case_ids"], ["train-1"])

    def test_reference_only_few_shot_candidate_is_materialized(self) -> None:
        bank = ProposalExampleBank(
            examples=[
                ProposalExample(
                    case_id="train-1",
                    input="I do not recognize this card payment.",
                    expected={"label": "card_payment_not_recognised"},
                    metadata={"category": "card_payment_not_recognised"},
                    label="card_payment_not_recognised",
                ),
                ProposalExample(
                    case_id="train-2",
                    input="Why was I charged a fee for using my card?",
                    expected={"label": "card_payment_fee_charged"},
                    metadata={"category": "card_payment_fee_charged"},
                    label="card_payment_fee_charged",
                ),
            ],
            label_counts={"card_payment_not_recognised": 1, "card_payment_fee_charged": 1},
            metadata_categories={"card_payment_not_recognised": 1, "card_payment_fee_charged": 1},
            label_field="label",
        )
        candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            transform_instance="contrastive_card_examples",
            intervention=Intervention(
                kind="example_selection",
                payload={"source_case_ids": ["train-1", "train-2"], "selection_strategy": "contrastive"},
            ),
            transform_parameters={
                "source_case_ids": ["train-1", "train-2"],
                "selection_strategy": "contrastive",
            },
            hypothesis="Contrast unknown card payments against recognized card fees.",
            patch=AgentPatch.empty(),
        )

        materialized, materialization = _materialize_candidate_references(candidate, bank)

        self.assertEqual(len(materialized.patch.operations), 1)
        operation = materialized.patch.operations[0]
        self.assertEqual(operation.op, "add_few_shot")
        self.assertEqual(operation.target, "few_shot")
        self.assertEqual([item["source_case_id"] for item in operation.value], ["train-1", "train-2"])
        self.assertEqual(operation.value[0]["output"], {"label": "card_payment_not_recognised"})
        self.assertTrue(materialized.patch.metadata["materialized_few_shot"])
        self.assertEqual(materialization["source_case_ids"], ["train-1", "train-2"])

    def test_few_shot_candidate_expands_to_one_two_three_shot_variants(self) -> None:
        candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            transform_instance="contrastive_card_examples",
            intervention=Intervention(
                kind="example_selection",
                payload={
                    "source_case_ids": ["train-1", "train-2", "train-3", "train-4"],
                    "selection_strategy": "contrastive",
                },
            ),
            transform_parameters={
                "source_case_ids": ["train-1", "train-2", "train-3", "train-4"],
                "selection_strategy": "contrastive",
            },
            hypothesis="Compare several card payment intents.",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_few_shot",
                        target="few_shot",
                        value=[
                            {"source_case_id": "train-1"},
                            {"source_case_id": "train-2"},
                            {"source_case_id": "train-3"},
                            {"source_case_id": "train-4"},
                        ],
                    )
                ]
            ),
        )

        variants = _few_shot_count_variants(candidate)

        self.assertEqual(
            [variant.transform_parameters["few_shot_example_count"] for variant in variants],
            [1, 2, 3],
        )
        self.assertEqual(
            [len(variant.patch.operations[0].value) for variant in variants],
            [1, 2, 3],
        )
        self.assertEqual(variants[2].patch.metadata["few_shot_variant"]["selection_strategy"], "contrastive")
        self.assertEqual(variants[2].transform_parameters["source_case_ids"], ["train-1", "train-2", "train-3"])
        self.assertEqual(variants[0].patch.metadata["few_shot_source_case_ids"], ["train-1"])
        self.assertEqual(variants[2].patch.metadata["few_shot_source_case_ids"], ["train-1", "train-2", "train-3"])

    def test_contrastive_few_shot_is_not_frontier_preferred_when_representative_ties(self) -> None:
        representative_candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            transform_parameters={
                "selection_strategy": "representative",
                "few_shot_example_count": 1,
            },
            hypothesis="Representative example.",
            patch=AgentPatch(
                operations=[
                    PatchOperation(op="add_few_shot", target="few_shot", value=[{"source_case_id": "train-1"}])
                ]
            ),
        )
        contrastive_candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            transform_parameters={
                "selection_strategy": "contrastive",
                "few_shot_example_count": 2,
            },
            hypothesis="Contrastive examples.",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_few_shot",
                        target="few_shot",
                        value=[{"source_case_id": "train-1"}, {"source_case_id": "train-2"}],
                    )
                ]
            ),
        )
        representative_summary = summary(representative_candidate.patch, passed=True, output={"label": "ok"})
        contrastive_summary = summary(contrastive_candidate.patch, passed=True, output={"label": "ok"})

        filtered = _prefer_simple_few_shot_strategy(
            [
                (contrastive_candidate, contrastive_summary, None),
                (representative_candidate, representative_summary, None),
            ]
        )

        self.assertEqual([row[0].transform_parameters["selection_strategy"] for row in filtered], ["representative"])

    def test_full_dev_selection_keeps_two_few_shot_compression_variants(self) -> None:
        one_shot = candidate_state(few_shot_candidate(1), pass_count=8, cost_usd=0.001)
        two_shot = candidate_state(few_shot_candidate(2), pass_count=9, cost_usd=0.002)
        three_shot = candidate_state(few_shot_candidate(3), pass_count=9, cost_usd=0.003)

        selected = _select_full_dev_candidates(
            [one_shot, two_shot, three_shot],
            OptimizationObjective(mode="correctness"),
        )

        self.assertEqual([state.candidate.transform_parameters["few_shot_example_count"] for state in selected], [2, 3])
        self.assertEqual(one_shot.frontier_status, "screened_out")
        self.assertIn("few-shot compression ranking", one_shot.rejection_reason or "")

    def test_full_dev_selection_prefers_smaller_few_shot_variant_when_tied(self) -> None:
        one_shot = candidate_state(few_shot_candidate(1), pass_count=9, cost_usd=0.001)
        two_shot = candidate_state(few_shot_candidate(2), pass_count=9, cost_usd=0.001)
        three_shot = candidate_state(few_shot_candidate(3), pass_count=9, cost_usd=0.001)

        selected = _select_full_dev_candidates(
            [three_shot, two_shot, one_shot],
            OptimizationObjective(mode="correctness"),
        )

        self.assertEqual([state.candidate.transform_parameters["few_shot_example_count"] for state in selected], [1, 2])
        self.assertEqual(three_shot.frontier_status, "screened_out")

    def test_full_dev_selection_keeps_one_ordinary_candidate_per_group(self) -> None:
        weak = candidate_state(prompt_candidate("weak"), pass_count=7)
        good = candidate_state(prompt_candidate("good"), pass_count=9)
        best = candidate_state(prompt_candidate("best"), pass_count=10)

        selected = _select_full_dev_candidates(
            [weak, good, best],
            OptimizationObjective(mode="correctness"),
        )

        self.assertEqual([state.candidate.transform_instance for state in selected], ["best"])
        self.assertEqual(weak.frontier_status, "screened_out")
        self.assertEqual(good.frontier_status, "screened_out")

    def test_full_dev_selection_preserves_strong_signal_output_contract_candidate(self) -> None:
        best_absolute = candidate_state(prompt_candidate("best", comparison_group="exp"), pass_count=10, score_delta=0.02)
        strong_contract = candidate_state(
            output_contract_candidate("close-json", comparison_group="exp"),
            pass_count=7,
            score_delta=0.22,
        )
        weak = candidate_state(prompt_candidate("weak", comparison_group="exp"), pass_count=6, score_delta=0.01)

        selected = _select_full_dev_candidates(
            [weak, strong_contract, best_absolute],
            OptimizationObjective(mode="correctness"),
        )

        selected_instances = {state.candidate.transform_instance for state in selected}
        self.assertEqual(selected_instances, {"best", "close-json"})
        self.assertEqual(weak.frontier_status, "screened_out")
        self.assertIn("experiment-aware batch ranking", weak.rejection_reason or "")

    def test_full_dev_selection_preserves_control_or_ablation_role(self) -> None:
        atomic = candidate_state(experiment_role_candidate("atomic", role="atomic"), pass_count=10)
        control = candidate_state(experiment_role_candidate("control", role="control"), pass_count=8)
        weak = candidate_state(experiment_role_candidate("weak", role="atomic"), pass_count=7)

        selected = _select_full_dev_candidates(
            [weak, control, atomic],
            OptimizationObjective(mode="correctness"),
        )

        selected_roles = {state.candidate.candidate_role for state in selected}
        selected_instances = {state.candidate.transform_instance for state in selected}
        self.assertIn("control", selected_roles)
        self.assertEqual(selected_instances, {"atomic", "control"})
        self.assertEqual(weak.frontier_status, "screened_out")

    def test_full_dev_selection_preserves_competitive_composed_candidate(self) -> None:
        atomic = candidate_state(experiment_role_candidate("atomic", role="atomic"), pass_count=10, score_delta=0.08)
        composed = candidate_state(
            experiment_role_candidate("composed", role="composed"),
            pass_count=9,
            score_delta=0.06,
        )
        weak = candidate_state(experiment_role_candidate("weak", role="composed"), pass_count=8, score_delta=0.01)

        selected = _select_full_dev_candidates(
            [weak, composed, atomic],
            OptimizationObjective(mode="correctness"),
        )

        selected_instances = {state.candidate.transform_instance for state in selected}
        self.assertEqual(selected_instances, {"atomic", "composed"})
        self.assertEqual(weak.frontier_status, "screened_out")
        self.assertIn("experiment role ranking", weak.rejection_reason or "")

    def test_reference_only_few_shot_candidate_can_be_parsed_without_patch(self) -> None:
        candidate = CandidateProposal.from_dict(
            {
                "transform_family": "targeted_few_shot",
                "transform_instance": "identity_examples",
                "intervention": {
                    "kind": "example_selection",
                    "payload": {"source_case_ids": ["train-1"]},
                },
                "hypothesis": "Use a representative train example.",
            }
        )

        self.assertTrue(candidate.patch.is_empty)
        self.assertEqual(candidate.transform_parameters["source_case_ids"], ["train-1"])

    def test_candidate_parser_rejects_missing_intervention(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit intervention"):
            CandidateProposal.from_dict(
                {
                    "transform_family": "targeted_few_shot",
                    "transform_instance": "identity_examples",
                    "hypothesis": "Use a representative train example.",
                }
            )

    def test_candidate_parser_rejects_model_authored_transform_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "transform_parameters are derived"):
            CandidateProposal.from_dict(
                {
                    "transform_family": "targeted_few_shot",
                    "transform_instance": "identity_examples",
                    "transform_parameters": {"source_case_ids": ["train-1"]},
                    "intervention": {
                        "kind": "example_selection",
                        "payload": {"source_case_ids": ["train-1"]},
                    },
                    "hypothesis": "Use a representative train example.",
                }
            )

    def test_unknown_few_shot_reference_is_rejected_after_materialization(self) -> None:
        bank = ProposalExampleBank(
            examples=[],
            label_counts={},
            metadata_categories={},
            label_field="label",
        )
        candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            intervention=Intervention(kind="example_selection", payload={"source_case_ids": ["missing"]}),
            transform_parameters={"source_case_ids": ["missing"]},
            hypothesis="Try a missing train example.",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_few_shot",
                        target="few_shot",
                        value=[{"source_case_id": "missing"}],
                    )
                ]
            ),
        )
        materialized, _ = _materialize_candidate_references(candidate, bank)
        target = EditableTarget(
            name="few_shot",
            kind="few_shot",
            path="few_shot",
            current_value=[],
            allowed_ops=["add_few_shot"],
            value_schema={"type": "array"},
        )

        is_valid, reason = PatchValidator().validate_with_reason(
            materialized.patch,
            current_spec=None,
            surface=[target],
            objective=OptimizationObjective(),
            proposal_example_case_ids=bank.case_ids,
        )

        self.assertFalse(is_valid)
        self.assertIn("requires proposal-safe train examples", reason or "")

    def test_string_few_shot_source_ids_do_not_expand_character_by_character(self) -> None:
        bank = ProposalExampleBank(
            examples=[
                ProposalExample(
                    case_id="train-1",
                    input="input",
                    expected={"label": "label"},
                    metadata={},
                    label="label",
                )
            ],
            label_counts={"label": 1},
            metadata_categories={},
            label_field="label",
        )
        candidate = CandidateProposal(
            transform_family="targeted_few_shot",
            intervention=Intervention(kind="example_selection", payload={"source_case_ids": "train-1"}),
            transform_parameters={"source_case_ids": "train-1"},
            hypothesis="Malformed source_case_ids should be rejected, not repaired.",
            patch=AgentPatch(
                operations=[
                    PatchOperation(op="add_few_shot", target="few_shot", value=[{}])
                ]
            ),
        )

        materialized, materialization = _materialize_candidate_references(candidate, bank)

        self.assertEqual(materialization, {})
        self.assertEqual(materialized.patch.operations[0].value, [{}])

    def test_cost_rejected_model_substitution_is_reported_as_tradeoff(self) -> None:
        rows = quality_cost_tradeoffs(
            [
                {
                    "transform_family": "model_substitution",
                    "patch_hash": "patch-1",
                    "rejection_reason": "cost constraint rejected patch ($0.02 > 3.00x baseline)",
                    "metrics": {"pass_count": 10, "case_count": 12, "mean_cost_usd": 0.02},
                }
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["patch_hash"], "patch-1")

    def test_cost_rejected_model_substitution_uses_constraint_warning(self) -> None:
        rows = quality_cost_tradeoffs(
            [
                {
                    "transform_family": "model_substitution",
                    "patch_hash": "patch-1",
                    "constraint_warning": "cost constraint rejected patch ($0.02 > 3.00x baseline)",
                    "metrics": {"pass_count": 10, "case_count": 12, "mean_cost_usd": 0.02},
                }
            ]
        )
        self.assertEqual(len(rows), 1)

    def test_simplification_variants_remove_ops_and_reduce_few_shot(self) -> None:
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_few_shot",
                    target="few_shot",
                    value=[
                        {"source_case_id": "a"},
                        {"source_case_id": "b"},
                        {"source_case_id": "c"},
                    ],
                ),
                PatchOperation(op="set_runtime_param", target="runtime.output_cap", value=2048),
            ]
        )

        variants = _simplification_variants(patch)
        simplification_types = {variant.metadata["simplification"]["type"] for variant in variants}

        self.assertIn("remove_operation", simplification_types)
        self.assertIn("reduce_few_shot", simplification_types)
        self.assertTrue(
            any(
                operation.op == "add_few_shot" and len(operation.value) == 1
                for variant in variants
                for operation in variant.operations
                if isinstance(operation.value, list)
            )
        )


if __name__ == "__main__":
    unittest.main()
