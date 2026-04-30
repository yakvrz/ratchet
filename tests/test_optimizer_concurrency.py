from __future__ import annotations

import unittest

from ratchet.optimizer import _candidate_batch_concurrency_limit
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import AgentSpec


class OptimizerConcurrencyTests(unittest.TestCase):
    def test_gemini_pro_model_substitution_throttles_candidate_batch(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="gemini-3-flash-preview",
                model_options=["gemini-3-flash-preview", "gemini-3-pro-preview"],
            )
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "pro-model",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "set_model_config",
                            "field": "model_name",
                            "value": "gemini-3-pro-preview",
                        }
                    ],
                }
            ),
            surface,
        )

        self.assertEqual(_candidate_batch_concurrency_limit([None, candidate]), 1)

    def test_non_model_candidate_does_not_throttle_candidate_batch(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="gemini-3-flash-preview", instructions={"system_prompt": "Answer."})
        )
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "prompt",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "add_context_section",
                            "section": "extra",
                            "content": "Be complete.",
                        }
                    ],
                }
            ),
            surface,
        )

        self.assertEqual(_candidate_batch_concurrency_limit([None, candidate]), 10_000)


if __name__ == "__main__":
    unittest.main()
