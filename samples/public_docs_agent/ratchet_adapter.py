from __future__ import annotations

import json
import os
from pathlib import Path
import re

from ratchet.types import EnumKnobSpec, EvalCase, GradeResult, RunRecord, SearchSpace, TextArtifactSpec
from agent import PublicDocsAgentRunner


ANSWER_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)"
)


def canonicalize_answer(text: str) -> str:
    stripped = text.strip().replace("`", "")
    match = ANSWER_TOKEN_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return stripped.rstrip(".")


class PublicDocsAdapter:
    def __init__(self, env_path: str | None = None, runner: PublicDocsAgentRunner | None = None) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def baseline(self) -> dict[str, str]:
        return {
            "model": "gpt-5.4",
            "reasoning_effort": "low",
            "prompt_output_rule": "Return strict JSON with exactly one string field named answer.",
            "prompt_grounding_rule": "Copy the exact symbol, method, function, or argument from grounded docs results only.",
            "prompt_tool_rule": "Always call docs_search before answering.",
            "prompt_fallback_rule": "If the answer is not grounded in the docs results, return {\"answer\": \"unknown\"}.",
            "docs_search_enabled": "on",
            "docs_search_description": "Search a frozen snapshot of public Python documentation and return the most relevant grounded snippets.",
            "knowledge_mode": "raw",
            "retrieval_top_k": "4",
            "output_cap": "120",
            "max_tool_rounds": "4",
        }

    def search_space(self) -> SearchSpace:
        return SearchSpace(
            enum_knobs=[
                EnumKnobSpec(
                    name="model",
                    kind="model",
                    values=["gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"],
                    default="gpt-5.4",
                    description="Model choice for the docs QA harness.",
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
                    values=["4", "2", "1"],
                    default="4",
                    depends_on={"docs_search_enabled": ["on"]},
                    description="Number of docs cards returned by docs_search.",
                ),
                EnumKnobSpec(
                    name="output_cap",
                    kind="param",
                    values=["120", "80", "48"],
                    default="120",
                    description="max_output_tokens for each model turn.",
                ),
                EnumKnobSpec(
                    name="max_tool_rounds",
                    kind="param",
                    values=["4"],
                    default="4",
                    description="Maximum number of tool continuation rounds.",
                ),
            ],
            text_artifacts=[
                TextArtifactSpec(
                    name="prompt_output_rule",
                    kind="prompt",
                    default="Return strict JSON with exactly one string field named answer.",
                    max_chars=180,
                    description="Prompt clause controlling output format discipline.",
                ),
                TextArtifactSpec(
                    name="prompt_grounding_rule",
                    kind="prompt",
                    default="Copy the exact symbol, method, function, or argument from grounded docs results only.",
                    max_chars=220,
                    description="Prompt clause controlling grounding and exactness.",
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
                    default="If the answer is not grounded in the docs results, return {\"answer\": \"unknown\"}.",
                    max_chars=220,
                    description="Prompt clause controlling fallback behavior.",
                ),
                TextArtifactSpec(
                    name="docs_search_description",
                    kind="tool",
                    default="Search a frozen snapshot of public Python documentation and return the most relevant grounded snippets.",
                    max_chars=220,
                    depends_on={"docs_search_enabled": ["on"]},
                    description="Tool artifact describing docs_search behavior to the model.",
                ),
            ],
        )

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        if self._runner is None:
            self._runner = PublicDocsAgentRunner(env_path=self.env_path)
        return self._runner.run_case(candidate, case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if not isinstance(output, dict) or "answer" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes="Could not parse answer payload.")
        expected_payload = case.expected
        if not isinstance(expected_payload, dict):
            raise ValueError("Public docs grader requires dict expected payloads.")
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
        return GradeResult(
            score=0.0,
            passed=False,
            labels=["wrong_field:answer"],
            notes=f"actual={actual_answer!r} output={output}",
        )

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))
        (out_dir / "README.md").write_text(
            "# Exported Public Docs Candidate\n\n"
            "This bundle contains the selected knobs for the standalone public-docs QA sample harness.\n"
        )


adapter = PublicDocsAdapter()
