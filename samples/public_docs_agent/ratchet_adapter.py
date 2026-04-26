from __future__ import annotations

import json
import os
from pathlib import Path
import re

from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, AgentTool, EvalCase, GradeResult, RunRecord

try:
    from agent import PublicDocsAgentRunner
except ModuleNotFoundError:
    from .agent import PublicDocsAgentRunner


ANSWER_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)"
)


BASE_SPEC = AgentSpec(
    name="public-docs-agent",
    model="gpt-5.4",
    model_options=["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4"],
    instructions={
        "output_rule": "Return strict JSON with exactly one string field named answer.",
        "grounding_rule": "Copy the exact symbol, method, function, or argument from grounded docs results only.",
        "tool_rule": "Always call docs_search before answering.",
        "fallback_rule": "If the answer is not grounded in the docs results, return {\"answer\": \"unknown\"}.",
    },
    tools={
        "docs_search": AgentTool(
            name="docs_search",
            description="Search a frozen snapshot of public Python documentation and return the most relevant grounded snippets.",
            policy="Use docs_search before answering Python documentation questions.",
            enabled=True,
        )
    },
    retrieval={"knowledge_mode": "raw", "top_k": 4},
    output_contract="Return strict JSON with exactly one string field named answer.",
    runtime={"reasoning_effort": "low", "output_cap": 120, "max_tool_rounds": 4},
)


def agent_config_from_spec(spec: AgentSpec) -> dict[str, str]:
    return {
        "model": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "low")),
        "prompt_output_rule": spec.instructions.get("output_rule", ""),
        "prompt_grounding_rule": spec.instructions.get("grounding_rule", ""),
        "prompt_tool_rule": spec.instructions.get("tool_rule", ""),
        "prompt_fallback_rule": spec.instructions.get("fallback_rule", ""),
        "prompt_few_shot": render_few_shot_prompt(spec.few_shot),
        "docs_search_enabled": "on" if spec.tools["docs_search"].enabled else "off",
        "docs_search_description": spec.tools["docs_search"].description,
        "knowledge_mode": str(spec.retrieval.get("knowledge_mode", "raw")),
        "retrieval_top_k": str(spec.retrieval.get("top_k", 4)),
        "output_cap": str(spec.runtime.get("output_cap", 120)),
        "max_tool_rounds": str(spec.runtime.get("max_tool_rounds", 4)),
    }


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

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            self._runner = PublicDocsAgentRunner(env_path=self.env_path)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if not isinstance(output, dict) or "answer" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes="Could not parse answer payload.")
        expected_payload = case.expected
        if not isinstance(expected_payload, dict):
            raise ValueError("Public docs grader requires dict expected payloads.")
        actual_answer = canonicalize_answer(str(output["answer"]))
        expected_answers = [canonicalize_answer(str(expected_payload["answer"]))]
        expected_answers.extend(canonicalize_answer(str(alias)) for alias in case.metadata.get("aliases", []))
        for expected_answer in expected_answers:
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

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


adapter = PublicDocsAdapter()
