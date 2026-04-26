from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ratchet.io import agent_spec_hash, load_eval_cases, patch_hash
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
