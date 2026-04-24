from __future__ import annotations

import json
import os
from pathlib import Path
import re

from ratchet.code_artifacts import CodeArtifactLoader, default_hook_source
from ratchet.types import (
    CodeArtifactSpec,
    ComponentSpec,
    EnumKnobSpec,
    EvalCase,
    GradeResult,
    RunRecord,
    SearchSpace,
    TextArtifactSpec,
)

from agent import PythonApiGroundingRunner


ANSWER_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?"
)


def canonicalize_answer(text: str) -> str:
    stripped = text.strip().replace("`", "")
    lowered = stripped.lower()
    if lowered == "unknown":
        return "unknown"
    match = ANSWER_TOKEN_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return stripped.rstrip(".")


def build_search_space() -> SearchSpace:
    validator_spec = CodeArtifactSpec(
        name="post_answer_validator_hook",
        language="python",
        callable_name="post_answer_validator_hook",
        signature="(output, context)",
        default="",
        max_chars=900,
        max_lines=20,
        depends_on={"answer_validator_enabled": ["on"]},
        description="Bounded source-level hook that can validate or rewrite the final answer payload.",
    )
    query_spec = CodeArtifactSpec(
        name="pre_tool_query_hook",
        language="python",
        callable_name="pre_tool_query_hook",
        signature="(query, context)",
        default="",
        max_chars=700,
        max_lines=16,
        depends_on={"docs_search_enabled": ["on"]},
        description="Bounded source-level hook that can rewrite a search query before docs_search.",
    )
    context_spec = CodeArtifactSpec(
        name="post_tool_context_hook",
        language="python",
        callable_name="post_tool_context_hook",
        signature="(cards, context)",
        default="",
        max_chars=700,
        max_lines=16,
        depends_on={"docs_search_enabled": ["on"]},
        description="Bounded source-level hook that can filter or reorder retrieved docs cards.",
    )
    def with_default(spec: CodeArtifactSpec) -> CodeArtifactSpec:
        return CodeArtifactSpec.from_dict({**spec.to_dict(), "default": default_hook_source(spec)})
    return SearchSpace(
        enum_knobs=[
            EnumKnobSpec(
                name="model",
                kind="model",
                values=["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
                default="gpt-5.4",
                description="Model choice for the grounded API harness.",
            ),
            EnumKnobSpec(
                name="reasoning_effort",
                kind="reasoning",
                values=["low", "none"],
                default="low",
                description="Reasoning effort passed to the Responses API.",
            ),
            EnumKnobSpec(
                name="docs_search_enabled",
                kind="tool",
                values=["off", "on"],
                default="on",
                description="Whether the docs_search tool is available to the agent.",
            ),
            EnumKnobSpec(
                name="knowledge_mode",
                kind="kb",
                values=["raw", "distilled"],
                default="raw",
                depends_on={"docs_search_enabled": ["on"]},
                description="Search either fuller snippets or shorter distilled cards.",
            ),
            EnumKnobSpec(
                name="retrieval_top_k",
                kind="param",
                values=["6", "4", "2", "1"],
                default="6",
                depends_on={"docs_search_enabled": ["on"]},
                description="Number of docs cards returned by docs_search.",
            ),
            EnumKnobSpec(
                name="output_cap",
                kind="param",
                values=["140", "96", "64"],
                default="140",
                description="max_output_tokens for each model turn.",
            ),
            EnumKnobSpec(
                name="max_tool_rounds",
                kind="param",
                values=["4", "2", "1"],
                default="4",
                description="Maximum number of tool continuation rounds.",
            ),
        ],
        text_artifacts=[
            TextArtifactSpec(
                name="prompt_output_rule",
                kind="prompt",
                default="Return JSON with a single string field named answer.",
                max_chars=180,
                description="Prompt clause controlling output format discipline.",
            ),
            TextArtifactSpec(
                name="prompt_grounding_rule",
                kind="prompt",
                default="Prefer exact literals from docs_search results, but if the search is inconclusive you may provide the most likely Python symbol or argument.",
                max_chars=220,
                description="Prompt clause controlling whether the answer must be grounded or may rely on prior knowledge.",
            ),
            TextArtifactSpec(
                name="prompt_tool_rule",
                kind="prompt",
                default="Always call docs_search before answering.",
                max_chars=220,
                depends_on={"docs_search_enabled": ["on"]},
                description="Prompt clause controlling search-tool usage.",
            ),
            TextArtifactSpec(
                name="prompt_fallback_rule",
                kind="prompt",
                default="If the tool is inconclusive, provide your best answer.",
                max_chars=220,
                description="Prompt clause controlling unknown/refusal behavior.",
            ),
            TextArtifactSpec(
                name="docs_search_description",
                kind="tool",
                default="Search a frozen snapshot of public Python API notes and return relevant grounded snippets.",
                max_chars=220,
                depends_on={"docs_search_enabled": ["on"]},
                description="Tool artifact describing docs_search behavior to the model.",
            ),
            TextArtifactSpec(
                name="answer_validator_rule",
                kind="component",
                default="If the chosen answer is not directly supported by retrieved evidence, replace it with unknown.",
                max_chars=240,
                depends_on={"answer_validator_enabled": ["on"]},
                description="Internal validator policy for post-answer grounding checks.",
            ),
        ],
        components=[
            ComponentSpec(
                name="answer_validator_enabled",
                kind="validator",
                values=["off", "on"],
                default="off",
                depends_on={"docs_search_enabled": ["on"]},
                description="Enable a bounded post-answer validator component.",
            ),
        ],
        code_artifacts=[
            with_default(query_spec),
            with_default(context_spec),
            with_default(validator_spec),
        ],
    )


class PythonApiGroundingAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: PythonApiGroundingRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner
        self._search_space = build_search_space()
        self._hook_loader = CodeArtifactLoader()

    def baseline(self) -> dict[str, str]:
        return {
            "model": "gpt-5.4",
            "reasoning_effort": "low",
            "prompt_output_rule": "Return JSON with a single string field named answer.",
            "prompt_grounding_rule": "Prefer exact literals from docs_search results, but if the search is inconclusive you may provide the most likely Python symbol or argument.",
            "prompt_tool_rule": "Always call docs_search before answering.",
            "prompt_fallback_rule": "If the tool is inconclusive, provide your best answer.",
            "answer_validator_enabled": "off",
            "answer_validator_rule": "If the chosen answer is not directly supported by retrieved evidence, replace it with unknown.",
            "docs_search_enabled": "on",
            "docs_search_description": "Search a frozen snapshot of public Python API notes and return relevant grounded snippets.",
            "knowledge_mode": "raw",
            "retrieval_top_k": "6",
            "output_cap": "140",
            "max_tool_rounds": "4",
        }

    def search_space(self) -> SearchSpace:
        return self._search_space

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        if self._runner is None:
            self._runner = PythonApiGroundingRunner(env_path=self.env_path)
        hooks = self._hook_loader.build_hooks(candidate, self._search_space.code_artifacts)
        return self._runner.run_case(candidate, case, hooks=hooks)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if not isinstance(output, dict) or "answer" not in output:
            return GradeResult(
                score=0.0,
                passed=False,
                labels=["invalid_output"],
                notes="Could not parse answer payload.",
            )
        expected_payload = case.expected
        if not isinstance(expected_payload, dict):
            raise ValueError("Python API grounding grader requires dict expected payloads.")
        actual_answer = canonicalize_answer(str(output["answer"]))
        candidates = [canonicalize_answer(str(expected_payload["answer"]))]
        candidates.extend(canonicalize_answer(str(alias)) for alias in case.metadata.get("aliases", []))
        for expected_answer in candidates:
            if (
                actual_answer == expected_answer
                or actual_answer.endswith(expected_answer)
                or expected_answer.endswith(actual_answer)
            ):
                return GradeResult(score=1.0, passed=True, labels=[])
        labels = ["wrong_field:answer"]
        if str(expected_payload["answer"]).lower() == "unknown":
            labels.append("unexpected_answer")
        return GradeResult(
            score=0.0,
            passed=False,
            labels=labels,
            notes=f"actual={actual_answer!r} output={output}",
        )

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))
        (out_dir / "README.md").write_text(
            "# Exported Python API Grounding Candidate\n\n"
            "This bundle contains the selected knobs for the standalone grounded Python API sample harness.\n"
        )


adapter = PythonApiGroundingAdapter()
