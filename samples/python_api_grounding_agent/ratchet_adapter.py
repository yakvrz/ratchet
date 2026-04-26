from __future__ import annotations

import json
import os
from pathlib import Path
import re

from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, AgentTool, EvalCase, GradeResult, RunRecord

try:
    from agent import PythonApiGroundingRunner
except ModuleNotFoundError:
    from .agent import PythonApiGroundingRunner


ANSWER_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?"
)


BASE_SPEC = AgentSpec(
    name="python-api-grounding-agent",
    model="gemini-2.5-flash",
    model_options=[
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    instructions={
        "output_rule": "Return JSON with a single string field named answer.",
        "grounding_rule": "Prefer exact literals from docs_search results, but if the search is inconclusive you may provide the most likely Python symbol or argument.",
        "tool_rule": "Always call docs_search before answering.",
        "fallback_rule": "If the tool is inconclusive, provide your best answer.",
        "validator_rule": "If the chosen answer is not directly supported by retrieved evidence, replace it with unknown.",
    },
    tools={
        "docs_search": AgentTool(
            name="docs_search",
            description="Search a frozen snapshot of public Python API notes and return relevant grounded snippets.",
            policy="Use docs_search before answering Python API questions.",
            enabled=True,
        )
    },
    retrieval={"knowledge_mode": "raw", "top_k": 6},
    output_contract="Return JSON with a single string field named answer.",
    runtime={"reasoning_effort": "low", "output_cap": 140, "max_tool_rounds": 4, "answer_validator_enabled": False},
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
        "answer_validator_enabled": "on" if spec.runtime.get("answer_validator_enabled") or spec.runtime.get("verifier_retry") else "off",
        "answer_validator_rule": spec.instructions.get("validator_rule", ""),
        "docs_search_enabled": "on" if spec.tools["docs_search"].enabled else "off",
        "docs_search_description": spec.tools["docs_search"].description,
        "knowledge_mode": str(spec.retrieval.get("knowledge_mode", "raw")),
        "retrieval_top_k": str(spec.retrieval.get("top_k", 6)),
        "output_cap": str(spec.runtime.get("output_cap", 140)),
        "max_tool_rounds": str(spec.runtime.get("max_tool_rounds", 4)),
    }


def canonicalize_answer(text: str) -> str:
    stripped = text.strip().replace("`", "")
    lowered = stripped.lower()
    if lowered == "unknown":
        return "unknown"
    match = ANSWER_TOKEN_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return stripped.rstrip(".")


class PythonApiGroundingAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: PythonApiGroundingRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            self._runner = PythonApiGroundingRunner(env_path=self.env_path)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case, hooks={})

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if not isinstance(output, dict) or "answer" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes="Could not parse answer payload.")
        if "invalid_output" in output:
            return GradeResult(
                score=0.0,
                passed=False,
                labels=["invalid_output"],
                notes=f"Model did not return the required JSON answer payload: {output.get('invalid_output')!r}",
            )
        expected_payload = case.expected
        if not isinstance(expected_payload, dict):
            raise ValueError("Python API grounding grader requires dict expected payloads.")
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
        labels = ["wrong_field:answer"]
        if str(expected_payload["answer"]).lower() == "unknown":
            labels.append("unexpected_answer")
        return GradeResult(
            score=0.0,
            passed=False,
            labels=labels,
            notes=f"actual={actual_answer!r} output={output}",
        )

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


adapter = PythonApiGroundingAdapter()
