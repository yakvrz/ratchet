from __future__ import annotations

import unittest

from ratchet.affordances import generate_optimization_affordances, validate_candidate_affordances
from ratchet.surface import SurfaceGenerator
from ratchet.types import AgentSpec, AgentTool, OptimizationObjective


class OptimizationAffordanceTests(unittest.TestCase):
    def test_generation_covers_current_editable_families(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="base",
            model_options=["base", "strong"],
            instructions={"system_prompt": "Classify."},
            output_contract="Return JSON.",
            retrieval={"top_k": 3},
            runtime={"output_cap": 120},
            tools={
                "search": AgentTool(
                    name="search",
                    description="Search docs.",
                    policy="Use for unknown facts.",
                    enabled=True,
                )
            },
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())

        affordances = generate_optimization_affordances(surface)
        keys = {(item.transform_family, item.target_kind) for item in affordances}

        self.assertIn(("prompt_rewrite", "instruction"), keys)
        self.assertIn(("output_contract_tightening", "output"), keys)
        self.assertIn(("targeted_few_shot", "few_shot"), keys)
        self.assertIn(("model_substitution", "model"), keys)
        self.assertIn(("runtime_tuning", "runtime"), keys)
        self.assertIn(("retrieval_tuning", "retrieval"), keys)
        self.assertIn(("verifier_retry", "verifier"), keys)

    def test_validation_requires_affordance_to_cover_operation(self) -> None:
        spec = AgentSpec(
            name="sample",
            model="base",
            instructions={"system_prompt": "Classify."},
        )
        surface = SurfaceGenerator().generate(spec, OptimizationObjective())
        affordances = generate_optimization_affordances(surface, active_families=["prompt_rewrite"])
        affordance = next(
            item
            for item in affordances
            if item.transform_family == "prompt_rewrite" and item.mechanism_class == "semantic_boundary_rewrite"
        )

        self.assertIsNone(
            validate_candidate_affordances(
                affordance_ids=[affordance.affordance_id],
                transform_family="prompt_rewrite",
                mechanism_class="semantic_boundary_rewrite",
                operations=[{"op": "add_instruction", "target": "instructions.system_prompt"}],
                affordances=affordances,
            )
        )
        self.assertIn(
            "not covered",
            validate_candidate_affordances(
                affordance_ids=[affordance.affordance_id],
                transform_family="prompt_rewrite",
                mechanism_class="semantic_boundary_rewrite",
                operations=[{"op": "add_output_constraint", "target": "output_contract"}],
                affordances=affordances,
            )
            or "",
        )


if __name__ == "__main__":
    unittest.main()
