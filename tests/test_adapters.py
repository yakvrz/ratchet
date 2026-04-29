from __future__ import annotations

import importlib
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from ratchet.adapters import adapter_fingerprint
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentSpec


class AdapterFingerprintTests(unittest.TestCase):
    def test_fingerprint_includes_sidecar_behavior_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "sidecar_adapter.py"
            prompt_path = root / "judge_prompt.md"
            module_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from ratchet.transform_program import CompiledCandidate
                    from ratchet.types import AgentSpec, EvalCase, GradeResult

                    class Adapter:
                        def agent_spec(self):
                            return AgentSpec(name="sidecar", model="primary")

                        def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None):
                            raise NotImplementedError

                        def grade(self, case, output):
                            return GradeResult(score=1.0, passed=True)

                        def export(self, patch, out_dir):
                            Path(out_dir).mkdir(parents=True, exist_ok=True)

                    adapter = Adapter()
                    """
                ).strip()
            )
            prompt_path.write_text("judge v1")
            sys.path.insert(0, str(root))
            try:
                first = adapter_fingerprint("sidecar_adapter:adapter")
                prompt_path.write_text("judge v2")
                second = adapter_fingerprint("sidecar_adapter:adapter")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("sidecar_adapter", None)
                importlib.invalidate_caches()

        self.assertNotEqual(first["source_tree_sha256"], second["source_tree_sha256"])

    def test_fingerprint_includes_nested_behavior_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "prompts"
            nested.mkdir()
            module_path = root / "nested_adapter.py"
            prompt_path = nested / "judge_prompt.md"
            module_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from ratchet.transform_program import CompiledCandidate
                    from ratchet.types import AgentSpec, EvalCase, GradeResult

                    class Adapter:
                        def agent_spec(self):
                            return AgentSpec(name="nested", model="primary")

                        def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None):
                            raise NotImplementedError

                        def grade(self, case, output):
                            return GradeResult(score=1.0, passed=True)

                        def export(self, patch, out_dir):
                            Path(out_dir).mkdir(parents=True, exist_ok=True)

                    adapter = Adapter()
                    """
                ).strip()
            )
            prompt_path.write_text("nested judge v1")
            sys.path.insert(0, str(root))
            try:
                first = adapter_fingerprint("nested_adapter:adapter")
                prompt_path.write_text("nested judge v2")
                second = adapter_fingerprint("nested_adapter:adapter")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("nested_adapter", None)
                importlib.invalidate_caches()

        self.assertNotEqual(first["source_tree_sha256"], second["source_tree_sha256"])

    def test_fingerprint_includes_optional_adapter_cache_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module_path = root / "custom_adapter.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    from pathlib import Path
                    from ratchet.transform_program import CompiledCandidate
                    from ratchet.types import AgentSpec, GradeResult

                    VERSION = "v1"

                    class Adapter:
                        def agent_spec(self):
                            return AgentSpec(name="custom", model="primary")

                        def run_case(self, case, candidate: CompiledCandidate | None = None):
                            raise NotImplementedError

                        def grade(self, case, output):
                            return GradeResult(score=1.0, passed=True)

                        def export(self, patch, out_dir):
                            Path(out_dir).mkdir(parents=True, exist_ok=True)

                        def cache_fingerprint(self):
                            return {"version": VERSION}

                    adapter = Adapter()
                    """
                ).strip()
            )
            sys.path.insert(0, str(root))
            try:
                module = importlib.import_module("custom_adapter")
                first = adapter_fingerprint("custom_adapter:adapter")
                module.VERSION = "v2"
                second = adapter_fingerprint("custom_adapter:adapter")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop("custom_adapter", None)
                importlib.invalidate_caches()

        self.assertNotEqual(first["custom_fingerprint_sha256"], second["custom_fingerprint_sha256"])

    def test_few_shot_renderer_includes_examples(self) -> None:
        rendered = render_few_shot_prompt(
            [
                {"messages": [{"role": "user", "content": "Synthetic example."}]},
                {"messages": [{"role": "user", "content": "Generated context example."}]},
            ]
        )

        self.assertIn("Synthetic example.", rendered)
        self.assertIn("Generated context example.", rendered)

if __name__ == "__main__":
    unittest.main()
