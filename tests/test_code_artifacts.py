from __future__ import annotations

import unittest

from ratchet.code_artifacts import CodeArtifactLoader, compile_code_artifact, default_hook_source
from ratchet.types import CodeArtifactSpec


class CodeArtifactTests(unittest.TestCase):
    def test_valid_code_artifact_compiles_and_runs(self) -> None:
        spec = CodeArtifactSpec(
            name="post_answer_validator_hook",
            language="python",
            callable_name="post_answer_validator_hook",
            signature="(output, context)",
            default="def post_answer_validator_hook(output, context):\n    return output\n",
            max_chars=400,
            max_lines=8,
        )
        hook = compile_code_artifact(
            spec,
            "def post_answer_validator_hook(output, context):\n    return {'answer': 'unknown'}\n",
        )
        self.assertEqual(hook({"answer": "x"}, {}), {"answer": "unknown"})

    def test_invalid_code_artifact_import_is_rejected(self) -> None:
        spec = CodeArtifactSpec(
            name="hook",
            language="python",
            callable_name="hook",
            signature="(output, context)",
            default="def hook(output, context):\n    return output\n",
            max_chars=400,
            max_lines=8,
        )
        with self.assertRaises(ValueError):
            compile_code_artifact(spec, "import os\n\ndef hook(output, context):\n    return output\n")

    def test_hook_signature_mismatch_fails(self) -> None:
        spec = CodeArtifactSpec(
            name="hook",
            language="python",
            callable_name="hook",
            signature="(output, context)",
            default="def hook(output, context):\n    return output\n",
            max_chars=400,
            max_lines=8,
        )
        with self.assertRaises(ValueError):
            compile_code_artifact(spec, "def hook(output):\n    return output\n")

    def test_default_hook_source_loader_builds_identity_hooks(self) -> None:
        spec = CodeArtifactSpec(
            name="post_tool_context_hook",
            language="python",
            callable_name="post_tool_context_hook",
            signature="(cards, context)",
            default="",
            max_chars=400,
            max_lines=8,
        )
        loader = CodeArtifactLoader()
        hooks = loader.build_hooks(
            {"post_tool_context_hook": default_hook_source(spec)},
            [CodeArtifactSpec.from_dict({**spec.to_dict(), "default": default_hook_source(spec)})],
        )
        self.assertEqual(
            hooks["post_tool_context_hook"]([{"text": "one"}], {}),
            [{"text": "one"}],
        )


if __name__ == "__main__":
    unittest.main()
