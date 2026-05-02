from __future__ import annotations

import sys
import unittest
from pathlib import Path

from ratchet.adapters import load_adapter
from ratchet.config import ensure_search_path, load_run_config
from ratchet.io import load_eval_cases
from ratchet.tool_loop import GeneratedToolLoopAdapter
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram


ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"


class DemoAdapterTests(unittest.TestCase):
    def tearDown(self) -> None:
        for module_name in ("ratchet_adapter", "ratchet_adapter_expanded", "order_desk_env", "expanded_tasks"):
            sys.modules.pop(module_name, None)
        try:
            sys.path.remove(str(DEMO))
        except ValueError:
            pass

    def test_demo_config_loads_from_project_root(self) -> None:
        config = load_run_config(DEMO / "ratchet.diagnostic_expanded.toml")

        self.assertEqual(config.adapter, "ratchet_adapter_expanded:adapter")
        self.assertEqual(config.evals, DEMO / "evals.diagnostic_expanded.jsonl")
        self.assertEqual(config.out, DEMO / "results" / "diagnostic-expanded")
        self.assertEqual(config.env_file, str(ROOT / ".env"))
        self.assertEqual(config.search_planner_model, "gemini-3-pro-preview")
        self.assertEqual(config.candidate_implementer_model, "gemini-3-pro-preview")

    def test_demo_adapter_and_eval_surface_load_cleanly(self) -> None:
        config = load_run_config(DEMO / "ratchet.diagnostic_expanded.toml")
        ensure_search_path(config)
        adapter = load_adapter(config.adapter)
        cases = load_eval_cases(config.evals)
        representative_cases = tuple(cases[:4])

        self.assertIsInstance(adapter, GeneratedToolLoopAdapter)
        self.assertEqual({case.split for case in cases}, {"train", "dev", "holdout"})
        self.assertEqual(len([case for case in cases if case.split == "dev"]), 48)
        self.assertEqual(len([case for case in cases if case.split == "holdout"]), 48)

        surface = adapter.surface_spec(representative_cases)
        section_names = surface.context.graph.section_names()
        tool_names = {tool.name for tool in surface.tools.tools}

        self.assertIn("domain_policy", section_names)
        self.assertIn("tool_instructions", section_names)
        self.assertIn("recent_messages", section_names)
        self.assertIn("get_order", tool_names)
        self.assertIn("cancel_order", tool_names)
        self.assertIn("modify_address", tool_names)
        self.assertIn("return_item", tool_names)

    def test_demo_surface_accepts_generic_transform_program(self) -> None:
        config = load_run_config(DEMO / "ratchet.diagnostic_expanded.toml")
        ensure_search_path(config)
        adapter = load_adapter(config.adapter)
        cases = load_eval_cases(config.evals)
        surface = adapter.surface_spec(tuple(cases[:4]))

        program = TransformProgram.from_dict(
            {
                "candidate_id": "demo_context_patch",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "replace_context_section",
                        "section": "domain_policy",
                        "content": "Inspect the relevant order details before any mutating action.",
                    }
                ],
            }
        )
        compiled = TransformCompiler().compile_or_raise(program, surface)

        self.assertEqual(compiled.program.candidate_id, "demo_context_patch")
        self.assertEqual(compiled.report.status, "compiled")


if __name__ == "__main__":
    unittest.main()
