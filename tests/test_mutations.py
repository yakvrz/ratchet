from __future__ import annotations

import unittest

from ratchet.io import candidate_hash, normalize_candidate
from ratchet.preflight import validate_search_space
from ratchet.types import CodeArtifactSpec, ComponentSpec, EnumKnobSpec, SearchSpace, TextArtifactSpec


class SearchSpaceTests(unittest.TestCase):
    def test_mixed_search_space_normalizes_enum_and_text_defaults(self) -> None:
        search_space = SearchSpace(
            enum_knobs=[EnumKnobSpec(name="model", kind="model", values=["a", "b"], default="a")],
            text_artifacts=[TextArtifactSpec(name="prompt", kind="prompt", default="plain", max_chars=40)],
        )
        candidate = normalize_candidate({"model": "b"}, search_space)
        self.assertEqual(candidate, {"model": "b", "prompt": "plain"})

    def test_dependency_filtering_resets_inactive_text_artifact(self) -> None:
        search_space = SearchSpace(
            enum_knobs=[EnumKnobSpec(name="tool", kind="tool", values=["off", "on"], default="off")],
            text_artifacts=[
                TextArtifactSpec(
                    name="tool_description",
                    kind="tool",
                    default="Use tool.",
                    max_chars=80,
                    depends_on={"tool": ["on"]},
                )
            ],
        )
        candidate = normalize_candidate(
            {"tool": "off", "tool_description": "Custom tool instructions"},
            search_space,
        )
        self.assertEqual(candidate["tool_description"], "Use tool.")

    def test_component_defaults_and_dependencies_are_normalized(self) -> None:
        search_space = SearchSpace(
            enum_knobs=[EnumKnobSpec(name="tool", kind="tool", values=["off", "on"], default="on")],
            text_artifacts=[
                TextArtifactSpec(
                    name="validator_rule",
                    kind="component",
                    default="Force unknown when ungrounded.",
                    max_chars=80,
                    depends_on={"validator": ["on"]},
                )
            ],
            components=[
                ComponentSpec(
                    name="validator",
                    kind="validator",
                    values=["off", "on"],
                    default="off",
                    depends_on={"tool": ["on"]},
                )
            ],
        )
        candidate = normalize_candidate(
            {
                "tool": "off",
                "validator": "on",
                "validator_rule": "Custom stricter validator.",
            },
            search_space,
        )
        self.assertEqual(candidate["validator"], "off")
        self.assertEqual(candidate["validator_rule"], "Force unknown when ungrounded.")

    def test_rewrite_bounds_are_enforced(self) -> None:
        search_space = SearchSpace(
            text_artifacts=[TextArtifactSpec(name="prompt", kind="prompt", default="plain", max_chars=5)]
        )
        with self.assertRaises(ValueError):
            normalize_candidate({"prompt": "too-long"}, search_space)

    def test_code_artifact_defaults_and_dependencies_are_normalized(self) -> None:
        search_space = SearchSpace(
            components=[
                ComponentSpec(
                    name="validator",
                    kind="validator",
                    values=["off", "on"],
                    default="off",
                )
            ],
            code_artifacts=[
                CodeArtifactSpec(
                    name="post_answer_validator_hook",
                    language="python",
                    callable_name="post_answer_validator_hook",
                    signature="(output, context)",
                    default="def post_answer_validator_hook(output, context):\n    return output\n",
                    max_chars=200,
                    max_lines=4,
                    depends_on={"validator": ["on"]},
                )
            ],
        )
        candidate = normalize_candidate(
            {
                "validator": "off",
                "post_answer_validator_hook": (
                    "def post_answer_validator_hook(output, context):\n"
                    "    return {'answer': 'unknown'}\n"
                ),
            },
            search_space,
        )
        self.assertEqual(candidate["validator"], "off")
        self.assertEqual(
            candidate["post_answer_validator_hook"],
            "def post_answer_validator_hook(output, context):\n    return output\n",
        )

    def test_code_artifact_rewrite_bounds_are_enforced(self) -> None:
        search_space = SearchSpace(
            code_artifacts=[
                CodeArtifactSpec(
                    name="hook",
                    language="python",
                    callable_name="hook",
                    signature="(output, context)",
                    default="def hook(output, context):\n    return output\n",
                    max_chars=60,
                    max_lines=2,
                )
            ]
        )
        with self.assertRaises(ValueError):
            normalize_candidate(
                {"hook": "def hook(output, context):\n    value = output\n    return value\n"},
                search_space,
            )

    def test_validate_search_space_rejects_duplicate_names(self) -> None:
        search_space = SearchSpace(
            enum_knobs=[EnumKnobSpec(name="shared", kind="model", values=["a"], default="a")],
            text_artifacts=[TextArtifactSpec(name="shared", kind="prompt", default="x", max_chars=10)],
        )
        with self.assertRaises(ValueError):
            validate_search_space(search_space)

    def test_candidate_hash_is_deterministic(self) -> None:
        first = candidate_hash({"b": "2", "a": "1"})
        second = candidate_hash({"a": "1", "b": "2"})
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
