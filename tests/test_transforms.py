from __future__ import annotations

import unittest

from ratchet.results import CaseEvaluation, PatchSummary
from ratchet.surface import SurfaceGenerator
from ratchet.transforms import (
    CandidateAffordanceApplication,
    CandidateProposal,
    Intervention,
    TransformContextKey,
    build_search_hypothesis,
    select_branch_history,
    summarize_transform_context_results,
    summarize_transform_results,
    transform_registry,
    validate_candidate_transform,
)
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    OptimizationObjective,
    PatchOperation,
    RunRecord,
)


def make_summary(labels: list[list[str]], *, mode: str = "dev") -> PatchSummary:
    evaluations = []
    for index, case_labels in enumerate(labels, start=1):
        passed = not case_labels
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(
                    id=f"case-{index}",
                    split=mode,
                    input=f"case {index}",
                    metadata={"category": "format" if index == 1 else "semantic"},
                ),
                record=RunRecord(
                    output={"answer": "ok"} if passed else "bad",
                    metrics=OperationalMetrics(
                        latency_s=float(index),
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001 * index,
                    ),
                ),
                grade=GradeResult(
                    score=1.0 if passed else 0.0,
                    passed=passed,
                    labels=case_labels,
                ),
            )
        )
    return PatchSummary(
        patch_hash="baseline",
        patch=AgentPatch.empty(),
        split=mode,
        evaluations=evaluations,
    )


def proposal(
    *,
    transform_family: str,
    patch: AgentPatch,
    mechanism_class: str = "semantic_boundary_rewrite",
    experiment_id: str = "exp_1",
    candidate_role: str = "atomic",
    **kwargs: object,
) -> CandidateProposal:
    transform_instance = str(kwargs.pop("transform_instance", ""))
    intervention = kwargs.pop("intervention", None)
    transform_parameters = kwargs.pop("transform_parameters", {})
    operations = list(patch.operations)
    selection = dict(kwargs.pop("selection", {}))
    explicit_example_selection = intervention is not None and getattr(intervention, "kind", "") == "example_selection"
    if explicit_example_selection:
        selection = dict(getattr(intervention, "payload", {}) or {})
    if (
        explicit_example_selection
        and not selection
        and isinstance(transform_parameters, dict)
        and "source_case_ids" in transform_parameters
    ):
        selection = {"source_case_ids": transform_parameters["source_case_ids"]}
    target_segment = "few_shot" if selection and not operations else (
        operations[0].target.replace(".", "_") if operations else "instructions_system_prompt"
    )
    applications = [
        CandidateAffordanceApplication(
            affordance_id=f"{transform_family}.{mechanism_class}.candidate.{target_segment}",
            operation=operations[0] if operations else None,
            selection=selection,
            rationale=transform_instance or str(patch.rationale or transform_family),
        )
    ]
    return CandidateProposal(
        experiment_id=experiment_id,
        candidate_role=candidate_role,
        patch=patch,
        applications=applications,
        **kwargs,
    )


class TransformLibraryTests(unittest.TestCase):
    def test_registry_exposes_valid_transform_families(self) -> None:
        registry = transform_registry()
        self.assertIn("prompt_rewrite", registry)
        self.assertIn("output_contract_tightening", registry)
        self.assertIn("model_substitution", registry)
        for name, family in registry.items():
            self.assertEqual(name, family.name)
            self.assertTrue(family.supported_edit_kinds)
            self.assertTrue(family.supported_ops)
            self.assertGreaterEqual(family.complexity_cost, 0.0)

    def test_invalid_output_activates_output_transforms(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer."},
            output_contract="Return JSON.",
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        hypothesis = build_search_hypothesis(
            summary=make_summary([["invalid_output"], []]),
            surface=surface,
            objective=OptimizationObjective(),
            history=[],
        )

        self.assertIn("output_contract_tightening", hypothesis.active_families)
        self.assertGreater(hypothesis.family_states["output_contract_tightening"].suitability, 0.0)
        self.assertIn("failure_label:invalid_output", hypothesis.target_slices)

    def test_cost_objective_activates_efficiency_families(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            model_options=["small", "large"],
            instructions={"system_prompt": "Answer."},
            retrieval={"top_k": 5},
            runtime={"output_cap": 128},
        )
        objective = OptimizationObjective(mode="cost")
        surface = SurfaceGenerator().generate(spec, objective)
        hypothesis = build_search_hypothesis(
            summary=make_summary([[]]),
            surface=surface,
            objective=objective,
            history=[],
        )

        self.assertIn("model_substitution", hypothesis.active_families)
        self.assertIn("runtime_tuning", hypothesis.active_families)

    def test_targeted_few_shot_requires_proposal_examples(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Classify."},
        )
        objective = OptimizationObjective(
            constraints=OptimizationObjective().constraints
        )
        surface = SurfaceGenerator().generate(spec, objective)
        without_examples = build_search_hypothesis(
            summary=make_summary([["wrong_label"], []]),
            surface=surface,
            objective=objective,
            history=[],
            proposal_example_count=0,
        )
        with_examples = build_search_hypothesis(
            summary=make_summary([["wrong_label"], []]),
            surface=surface,
            objective=objective,
            history=[],
            proposal_example_count=3,
        )

        self.assertNotIn("targeted_few_shot", without_examples.active_families)
        self.assertIn("targeted_few_shot", with_examples.active_families)

    def test_targeted_few_shot_parameters_are_required(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Classify."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_few_shot",
                    target="few_shot",
                    value=[
                        {
                            "source_case_id": "train-1",
                            "input": "hello",
                            "output": {"label": "greeting"},
                            "purpose": "representative",
                        }
                    ],
                )
            ]
        )
        missing = proposal(
            transform_family="targeted_few_shot",
            mechanism_class="representative_examples",
            patch=patch,
        )
        missing_reference_only = proposal(
            transform_family="targeted_few_shot",
            mechanism_class="representative_examples",
            intervention=Intervention(kind="example_selection", payload={}),
            patch=AgentPatch.empty(),
        )
        valid = proposal(
            transform_family="targeted_few_shot",
            mechanism_class="representative_examples",
            intervention=Intervention(kind="example_selection", payload={"source_case_ids": ["train-1"]}),
            transform_parameters={"source_case_ids": ["train-1"]},
            patch=AgentPatch(
                operations=list(patch.operations),
                metadata={"materialized_few_shot": True},
            ),
        )
        reference_only = proposal(
            transform_family="targeted_few_shot",
            mechanism_class="representative_examples",
            intervention=Intervention(kind="example_selection", payload={"source_case_ids": ["train-1"]}),
            transform_parameters={"source_case_ids": ["train-1"]},
            patch=AgentPatch.empty(),
        )
        malformed = proposal(
            transform_family="targeted_few_shot",
            mechanism_class="representative_examples",
            intervention=Intervention(kind="example_selection", payload={"source_case_ids": "train-1"}),
            transform_parameters={"source_case_ids": "train-1"},
            patch=AgentPatch.empty(),
        )

        self.assertIn("must use selection", validate_candidate_transform(missing, surface=surface) or "")
        self.assertIn(
            "requires selection.source_case_ids",
            validate_candidate_transform(missing_reference_only, surface=surface) or "",
        )
        self.assertIsNone(validate_candidate_transform(valid, surface=surface))
        self.assertIsNone(validate_candidate_transform(reference_only, surface=surface))
        self.assertIn("requires selection.source_case_ids", validate_candidate_transform(malformed, surface=surface) or "")
        inline = proposal(
            transform_family="targeted_few_shot",
            mechanism_class="representative_examples",
            transform_parameters={"source_case_ids": ["train-1"]},
            patch=patch,
        )
        self.assertIn("must use selection", validate_candidate_transform(inline, surface=surface) or "")

    def test_candidate_validation_rejects_unknown_and_incompatible_family(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        unknown = proposal(
            transform_family="not_real",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_instruction",
                        target="instructions.system_prompt",
                        value="Answer exactly.",
                    )
                ]
            ),
        )
        incompatible = proposal(
            transform_family="model_substitution",
            mechanism_class="model_capability_probe",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_instruction",
                        target="instructions.system_prompt",
                        value="Answer exactly.",
                    )
                ]
            ),
        )

        self.assertIn("unknown transform family", validate_candidate_transform(unknown, surface=surface) or "")
        self.assertIn("incompatible", validate_candidate_transform(incompatible, surface=surface) or "")

    def test_context_identity_ignores_free_form_instance_text(self) -> None:
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Return valid JSON with all required fields.",
                )
            ]
        )
        first = TransformContextKey.from_candidate(
            proposal(
                transform_family="prompt_rewrite",
                transform_instance="tighten output v1",
                patch=patch,
            )
        )
        second = TransformContextKey.from_candidate(
            proposal(
                transform_family="prompt_rewrite",
                transform_instance="renamed but same mechanism",
                patch=patch,
            )
        )

        self.assertEqual(first.id, second.id)
        self.assertNotEqual(first.transform_instance, second.transform_instance)

    def test_candidate_validation_enforces_context_lifecycle(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        rejected_patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Return valid JSON with all required fields.",
                )
            ]
        )
        rejected_candidate = proposal(
            transform_family="prompt_rewrite",
            transform_instance="failed format tightening",
            patch=rejected_patch,
        )
        hypothesis = build_search_hypothesis(
            summary=make_summary([["failed"], []]),
            surface=surface,
            objective=OptimizationObjective(),
            history=[
                {
                    "transform_family": "prompt_rewrite",
                    "transform_context": TransformContextKey.from_candidate(rejected_candidate).to_dict(),
                    "accepted": False,
                    "parent_patch_hash": "baseline",
                    "patch_hash": "failed-format",
                    "proposal": rejected_patch.to_dict(),
                    "comparison_to_parent": {"score_delta": 0.0},
                }
            ],
        )
        renamed_same_context = proposal(
            transform_family="prompt_rewrite",
            transform_instance="same idea with a new name",
            patch=rejected_patch,
        )
        distinct_mechanism = proposal(
            transform_family="prompt_rewrite",
            transform_instance="grounded answer mechanism",
            patch=AgentPatch(
                operations=[
                    PatchOperation(
                        op="add_instruction",
                        target="instructions.system_prompt",
                        value="Cite source evidence before giving the answer.",
                    )
                ]
            ),
        )

        self.assertIn(
            "inactive transform context",
            validate_candidate_transform(
                renamed_same_context,
                surface=surface,
                search_hypothesis=hypothesis,
            )
            or "",
        )
        self.assertIsNone(
            validate_candidate_transform(
                distinct_mechanism,
                surface=surface,
                search_hypothesis=hypothesis,
            )
        )

    def test_prompt_hypothesis_view_is_compact(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"system_prompt": "Answer.", "output_rule": "Return JSON."},
            output_contract="Return JSON.",
        )
        hypothesis = build_search_hypothesis(
            summary=make_summary([["invalid_output"], []]),
            surface=SurfaceGenerator().generate(spec, OptimizationObjective()),
            objective=OptimizationObjective(),
            history=[],
        )

        prompt_view = hypothesis.to_prompt_dict()

        self.assertIn("active_contexts", prompt_view)
        self.assertNotIn("context_states", prompt_view)
        self.assertLessEqual(
            len(prompt_view["active_contexts"]),
            len(hypothesis.context_states),
        )

    def test_transform_result_summary_promotes_and_constrains_families(self) -> None:
        summaries = summarize_transform_results(
            [
                {
                    "transform_family": "prompt_rewrite",
                    "accepted": True,
                    "comparison_to_parent": {"score_delta": 0.5, "cost_delta": 0.0, "latency_delta": 0.0},
                },
                {
                    "transform_family": "model_substitution",
                    "accepted": False,
                    "comparison_to_parent": {"score_delta": -0.5, "cost_delta": -0.1, "latency_delta": 0.0},
                },
                {
                    "transform_family": "model_substitution",
                    "accepted": False,
                    "comparison_to_parent": {"score_delta": -0.5, "cost_delta": -0.1, "latency_delta": 0.0},
                },
            ]
        )

        self.assertEqual(summaries["prompt_rewrite"]["state"], "promoted")
        self.assertEqual(summaries["model_substitution"]["state"], "constrained")

    def test_lifecycle_history_controls_active_families(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            model_options=["small", "large"],
            instructions={"system_prompt": "Answer."},
            output_contract="Return JSON.",
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        summary = make_summary([["invalid_output"], []])
        model_context = TransformContextKey(
            family="model_substitution",
            target_names=("model",),
            ops=("change_model",),
            transform_instance="try small",
        )
        constrained = build_search_hypothesis(
            summary=summary,
            surface=surface,
            objective=OptimizationObjective(),
            history=[
                {
                    "transform_family": "model_substitution",
                    "transform_context": model_context.to_dict(),
                    "accepted": False,
                    "parent_patch_hash": "baseline",
                    "patch_hash": "small",
                    "comparison_to_parent": {"score_delta": -0.5},
                }
            ],
        )
        prompt_context = TransformContextKey(
            family="prompt_rewrite",
            target_names=("instructions.system_prompt",),
            ops=("add_instruction",),
            transform_instance="add grounding",
        )
        promoted = build_search_hypothesis(
            summary=summary,
            surface=surface,
            objective=OptimizationObjective(),
            history=[
                {
                    "transform_family": "prompt_rewrite",
                    "transform_context": prompt_context.to_dict(),
                    "accepted": True,
                    "parent_patch_hash": "baseline",
                    "patch_hash": "prompt",
                    "comparison_to_parent": {"score_delta": 0.5},
                }
            ],
        )
        output_context = TransformContextKey(
            family="output_contract_tightening",
            target_names=("output_contract",),
            ops=("add_output_constraint",),
            transform_instance="tighten json",
        )
        paused = build_search_hypothesis(
            summary=summary,
            surface=surface,
            objective=OptimizationObjective(),
            history=[
                {
                    "transform_family": "output_contract_tightening",
                    "transform_context": output_context.to_dict(),
                    "accepted": False,
                    "parent_patch_hash": "baseline",
                    "patch_hash": "output",
                    "comparison_to_parent": {"score_delta": 0.0},
                }
            ],
        )

        self.assertEqual(constrained.context_states[model_context.id].state, "constrained")
        self.assertIn("model_substitution", constrained.active_families)
        self.assertTrue(constrained.context_states[model_context.id].constraints)
        self.assertEqual(promoted.family_states["prompt_rewrite"].state, "promoted")
        self.assertIn("prompt_rewrite", promoted.active_families)
        self.assertEqual(paused.context_states[output_context.id].state, "paused")
        self.assertIn("output_contract_tightening", paused.active_families)

    def test_branch_history_excludes_siblings_and_includes_ancestor(self) -> None:
        rows = [
            {"parent_patch_hash": "baseline", "patch_hash": "alpha", "accepted": True, "transform_family": "prompt_rewrite"},
            {"parent_patch_hash": "baseline", "patch_hash": "beta", "accepted": True, "transform_family": "model_substitution"},
            {"parent_patch_hash": "alpha", "patch_hash": "alpha2", "accepted": False, "transform_family": "prompt_rewrite"},
        ]

        selected = select_branch_history(rows, "alpha")

        self.assertEqual([row["patch_hash"] for row in selected], ["alpha", "alpha2"])

    def test_context_lifecycle_keeps_other_prompt_targets_active(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="large",
            instructions={"output_rule": "Return JSON.", "tool_rule": "Use tools."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        summary = make_summary([["failed"], []])
        output_context = TransformContextKey(
            family="prompt_rewrite",
            target_names=("instructions.output_rule",),
            ops=("revise_instruction",),
            transform_instance="tighten output",
        )

        hypothesis = build_search_hypothesis(
            summary=summary,
            surface=surface,
            objective=OptimizationObjective(),
            history=[
                {
                    "transform_family": "prompt_rewrite",
                    "transform_context": output_context.to_dict(),
                    "accepted": False,
                    "parent_patch_hash": "baseline",
                    "patch_hash": "failed-output",
                    "comparison_to_parent": {"score_delta": -0.5},
                }
            ],
        )

        self.assertEqual(hypothesis.context_states[output_context.id].state, "constrained")
        self.assertEqual(hypothesis.family_states["prompt_rewrite"].state, "active")
        self.assertIn("prompt_rewrite", hypothesis.active_families)

    def test_context_summary_can_demote_late_regression(self) -> None:
        context = TransformContextKey(
            family="prompt_rewrite",
            target_names=("instructions.system_prompt",),
            ops=("add_instruction",),
            transform_instance="grounding",
        )
        summaries = summarize_transform_context_results(
            [
                {
                    "transform_family": "prompt_rewrite",
                    "transform_context": context.to_dict(),
                    "accepted": True,
                    "comparison_to_parent": {"score_delta": 0.5},
                },
                {
                    "transform_family": "prompt_rewrite",
                    "transform_context": context.to_dict(),
                    "accepted": False,
                    "comparison_to_parent": {"score_delta": -0.5},
                },
            ]
        )

        self.assertEqual(summaries[context.id]["state"], "constrained")


if __name__ == "__main__":
    unittest.main()
