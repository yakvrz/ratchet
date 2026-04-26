from __future__ import annotations

import unittest

from ratchet.evidence import ProposalExample, ProposalExampleBank
from ratchet.profiling import quality_cost_tradeoffs, runtime_reliability_diagnostics
from ratchet.proposals import _materialize_candidate_references
from ratchet.optimizer import _simplification_variants
from ratchet.results import CaseEvaluation, PatchSummary
from ratchet.transforms import CandidateProposal
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


class ProfilingTests(unittest.TestCase):
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

    def test_unknown_few_shot_reference_is_rejected_after_materialization(self) -> None:
        bank = ProposalExampleBank(
            examples=[],
            label_counts={},
            metadata_categories={},
            label_field="label",
        )
        candidate = CandidateProposal(
            transform_family="targeted_few_shot",
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
