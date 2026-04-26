from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

import ratchet
from ratchet.io import agent_spec_hash, load_eval_cases, patch_hash
from ratchet.optimizer import case_timeout
from ratchet.surface import SurfaceGenerator
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    AgentTool,
    EvalCase,
    OptimizationConstraints,
    OptimizationObjective,
    PatchOperation,
)
from ratchet.validation import PatchValidator


class V2PatchSurfaceTests(unittest.TestCase):
    def make_spec(self) -> AgentSpec:
        return AgentSpec(
            name="unit-agent",
            model="large",
            model_options=["small", "large"],
            instructions={"system_prompt": "Answer politely."},
            tools={"search": AgentTool(name="search", description="Search.", policy="Use search.", enabled=False)},
            retrieval={"top_k": 4},
            output_contract="Return text.",
            runtime={"output_cap": 128},
        )

    def test_agent_spec_patch_application_changes_spec(self) -> None:
        spec = self.make_spec()
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Answer with exact grounded facts.",
                ),
                PatchOperation(op="set_runtime_param", target="tools.search.enabled", value=True),
            ]
        )

        updated = spec.apply_patch(patch)

        self.assertTrue(updated.tools["search"].enabled)
        self.assertIn("exact grounded", updated.instructions["system_prompt"].lower())
        self.assertNotEqual(agent_spec_hash(spec), agent_spec_hash(updated))

    def test_agent_spec_apply_patch_preserves_original_spec(self) -> None:
        spec = self.make_spec()
        original = spec.to_dict()
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Answer with exact grounded facts.",
                ),
                PatchOperation(op="set_runtime_param", target="runtime.output_cap", value=256),
            ]
        )

        updated = spec.apply_patch(patch)

        self.assertEqual(spec.to_dict(), original)
        self.assertNotEqual(updated.to_dict(), original)
        self.assertEqual(spec.runtime["output_cap"], 128)
        self.assertEqual(updated.runtime["output_cap"], 256)

    def test_public_api_exports_user_facing_errors_not_transform_internals(self) -> None:
        self.assertIn("OptimizerModelError", ratchet.__all__)
        self.assertIn("RatchetConfigError", ratchet.__all__)
        self.assertNotIn("TransformContextState", ratchet.__all__)
        self.assertNotIn("TransformFamilyState", ratchet.__all__)

    def test_case_timeout_fails_fast_in_worker_thread(self) -> None:
        errors: list[str] = []

        def worker() -> None:
            try:
                with case_timeout(1):
                    pass
            except RuntimeError as exc:
                errors.append(str(exc))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        self.assertEqual(len(errors), 1)
        self.assertIn("main thread", errors[0])

    def test_surface_generator_derives_targets_from_agent_spec(self) -> None:
        objective = OptimizationObjective(
            mode="correctness",
            constraints=OptimizationConstraints(
                allowed_edits=["instruction", "tool", "model", "retrieval", "runtime", "output"],
                allowed_models=["small", "large"],
            ),
        )

        targets = SurfaceGenerator().generate(self.make_spec(), objective)
        names = {target.name for target in targets}

        self.assertIn("instructions.system_prompt", names)
        self.assertIn("tools.search.enabled", names)
        self.assertIn("tools.search.description", names)
        self.assertIn("retrieval.top_k", names)
        self.assertIn("runtime.output_cap", names)
        self.assertIn("model", names)
        schemas = {target.name: target.value_schema for target in targets}
        self.assertEqual(schemas["runtime.output_cap"]["type"], "integer")
        self.assertEqual(schemas["tools.search.enabled"]["type"], "boolean")
        self.assertEqual(schemas["model"]["shape"], "categorical")

    def test_surface_generator_memoizes_by_spec_and_objective(self) -> None:
        class CountingSurfaceGenerator(SurfaceGenerator):
            def __init__(self) -> None:
                super().__init__()
                self.uncached_calls = 0

            def _generate_uncached(
                self,
                spec: AgentSpec | None,
                objective: OptimizationObjective,
            ):
                self.uncached_calls += 1
                return super()._generate_uncached(spec, objective)

        spec = self.make_spec()
        generator = CountingSurfaceGenerator()
        correctness = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["instruction"])
        )
        cost = OptimizationObjective(
            mode="cost",
            constraints=OptimizationConstraints(allowed_edits=["instruction"]),
        )

        first = generator.generate(spec, correctness)
        second = generator.generate(spec, correctness)
        third = generator.generate(spec.apply_patch(AgentPatch.empty()), correctness)
        fourth = generator.generate(spec, cost)

        self.assertEqual(generator.uncached_calls, 2)
        self.assertIsNot(first, second)
        self.assertEqual([target.to_dict() for target in first], [target.to_dict() for target in second])
        self.assertEqual([target.to_dict() for target in first], [target.to_dict() for target in third])
        self.assertEqual([target.to_dict() for target in first], [target.to_dict() for target in fourth])

    def test_few_shot_patch_accepts_single_or_multiple_examples(self) -> None:
        spec = self.make_spec()
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["few_shot"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_few_shot",
                    target="few_shot",
                    value=[
                        {
                            "source_case_id": "train-1",
                            "input": "How do I reset access?",
                            "output": {"label": "account_help"},
                            "purpose": "representative account help example",
                        },
                        {
                            "source_case_id": "train-2",
                            "input": "Why was I charged?",
                            "output": {"label": "fee_question"},
                            "purpose": "representative fee question example",
                        },
                    ],
                )
            ]
        )

        self.assertTrue(
            validator.validate(
                patch,
                current_spec=spec,
                surface=surface,
                objective=objective,
                proposal_example_case_ids={"train-1", "train-2"},
            )
        )
        updated = spec.apply_patch(patch)
        self.assertEqual(len(updated.few_shot), 2)
        self.assertEqual(updated.few_shot[0]["output"]["label"], "account_help")

    def test_few_shot_patch_rejects_ignored_extra_fields(self) -> None:
        spec = self.make_spec()
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["few_shot"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_few_shot",
                    target="few_shot",
                    value=[
                        {
                            "source_case_id": "train-1",
                            "input": "How do I reset access?",
                            "output": {"label": "account_help"},
                            "purpose": "representative account help example",
                            "ignored": "this renderer would ignore me",
                        }
                    ],
                )
            ]
        )

        is_valid, reason = validator.validate_with_reason(
            patch,
            current_spec=spec,
            surface=surface,
            objective=objective,
            proposal_example_case_ids={"train-1"},
        )
        self.assertFalse(is_valid)
        self.assertIn("unsupported key 'ignored'", reason or "")

    def test_patch_validator_rejects_invalid_model_and_unsupported_target(self) -> None:
        spec = self.make_spec()
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["model"], allowed_models=["small"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()

        invalid_model = AgentPatch(
            operations=[PatchOperation(op="change_model", target="model", value="not-allowed")]
        )
        invalid_target = AgentPatch(
            operations=[PatchOperation(op="change_model", target="judge.model", value="small")]
        )

        self.assertFalse(
            validator.validate(invalid_model, current_spec=spec, surface=surface, objective=objective)
        )
        self.assertFalse(
            validator.validate(invalid_target, current_spec=spec, surface=surface, objective=objective)
        )

    def test_patch_validator_rejects_value_that_violates_target_schema(self) -> None:
        spec = self.make_spec()
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["runtime", "tool"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()

        string_runtime = AgentPatch(
            operations=[PatchOperation(op="set_runtime_param", target="runtime.output_cap", value="128")]
        )
        string_bool = AgentPatch(
            operations=[PatchOperation(op="set_runtime_param", target="tools.search.enabled", value="true")]
        )

        self.assertFalse(
            validator.validate(string_runtime, current_spec=spec, surface=surface, objective=objective)
        )
        self.assertFalse(
            validator.validate(string_bool, current_spec=spec, surface=surface, objective=objective)
        )

    def test_patch_validator_rejects_eval_memorization(self) -> None:
        spec = self.make_spec()
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["instruction"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()
        case = EvalCase(
            id="case-123",
            split="dev",
            input="Customer identifier 928374651 asks about the renewal waiver for Project Monarch.",
            expected="Use the private expected answer that should never be copied into a patch.",
            metadata={"example_uuid": "123e4567-e89b-12d3-a456-426614174000"},
        )

        copied_id = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Special-case case-123.",
                )
            ]
        )
        copied_number = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Remember customer identifier 928374651.",
                )
            ]
        )
        copied_expected = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Use the private expected answer that should never be copied into a patch.",
                )
            ]
        )
        copied_uuid = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Route 123e4567-e89b-12d3-a456-426614174000 specially.",
                )
            ]
        )

        for patch in (copied_id, copied_number, copied_expected, copied_uuid):
            self.assertFalse(
                validator.validate(
                    patch,
                    current_spec=spec,
                    surface=surface,
                    objective=objective,
                    evidence_cases=[case],
                )
            )

    def test_patch_validator_restricts_few_shot_to_proposal_examples(self) -> None:
        spec = self.make_spec()
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["few_shot", "instruction"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()
        train_case = EvalCase(
            id="train-1",
            split="train",
            input="Private train wording for access reset with a distinctive protected phrase",
            expected={"label": "account_help"},
        )
        valid_few_shot = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_few_shot",
                    target="few_shot",
                    value=[
                        {
                            "source_case_id": "train-1",
                            "input": "Private train wording for access reset with a distinctive protected phrase",
                            "output": {"label": "account_help"},
                            "purpose": "representative train example",
                        }
                    ],
                )
            ]
        )
        prompt_copy = AgentPatch(
            operations=[
                PatchOperation(
                    op="add_instruction",
                    target="instructions.system_prompt",
                    value="Always remember Private train wording for access reset with a distinctive protected phrase.",
                )
            ]
        )

        self.assertTrue(
            validator.validate(
                valid_few_shot,
                current_spec=spec,
                surface=surface,
                objective=objective,
                proposal_example_case_ids={"train-1"},
                proposal_example_cases=[train_case],
            )
        )
        self.assertFalse(
            validator.validate(
                prompt_copy,
                current_spec=spec,
                surface=surface,
                objective=objective,
                proposal_example_case_ids={"train-1"},
                proposal_example_cases=[train_case],
            )
        )

    def test_patch_validator_allows_eval_literals_already_in_agent_spec(self) -> None:
        spec = AgentSpec(
            name="intent-agent",
            model="large",
            instructions={
                "label_rule": "Allowed labels: card_payment_fee_charged, extra_charge_on_statement.",
                "decision_rule": "Choose a label.",
            },
        )
        objective = OptimizationObjective(
            constraints=OptimizationConstraints(allowed_edits=["instruction"])
        )
        surface = SurfaceGenerator().generate(spec, objective)
        validator = PatchValidator()
        case = EvalCase(
            id="case-1",
            split="dev",
            input="Why was I charged for a card payment?",
            expected={"label": "card_payment_fee_charged"},
            metadata={"category": "card_payment_fee_charged"},
        )
        patch = AgentPatch(
            operations=[
                PatchOperation(
                    op="revise_instruction",
                    target="instructions.decision_rule",
                    value="Prefer card_payment_fee_charged for card payment fee complaints.",
                )
            ]
        )

        self.assertTrue(
            validator.validate(
                patch,
                current_spec=spec,
                surface=surface,
                objective=objective,
                evidence_cases=[case],
            )
        )

    def test_patch_hash_is_deterministic(self) -> None:
        first = patch_hash(
            AgentPatch(operations=[PatchOperation(op="change_model", target="model", value="small")])
        )
        second = patch_hash(
            AgentPatch(operations=[PatchOperation(op="change_model", target="model", value="small")])
        )
        self.assertEqual(first, second)

    def test_load_eval_cases_rejects_duplicate_ids(self) -> None:
        rows = [
            {"id": "case-1", "split": "dev", "input": "a"},
            {"id": "case-1", "split": "holdout", "input": "b"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evals.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "duplicate case id"):
                load_eval_cases(path)


if __name__ == "__main__":
    unittest.main()
