from __future__ import annotations

import json
import os
from pathlib import Path

from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, AgentTool, EvalCase, GradeResult, RunRecord

try:
    from agent import RunbookActionRunner
except ModuleNotFoundError:
    from .agent import RunbookActionRunner


def normalize_answer(text: str) -> str:
    return str(text).strip().replace("`", "").rstrip(".").lower()


BASE_SPEC = AgentSpec(
    name="runbook-action-agent",
    model="gpt-5.4",
    model_options=["gpt-5.4-mini", "gpt-5.4"],
    instructions={
        "output_rule": "Return JSON with a single string field named answer.",
        "grounding_rule": "Choose the next step only from runbook_search evidence. If the evidence is weak, you may still provide the most likely action.",
        "tool_rule": "Always call runbook_search before answering.",
        "fallback_rule": "If the runbook is inconclusive, provide your best action.",
        "validator_rule": "If the chosen action is not directly supported by retrieved runbook evidence, replace it with unknown.",
    },
    tools={
        "runbook_search": AgentTool(
            name="runbook_search",
            description="Search a frozen incident runbook snapshot and return grounded remediation snippets.",
            policy="Use runbook_search before selecting a remediation action.",
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
        "runbook_search_enabled": "on" if spec.tools["runbook_search"].enabled else "off",
        "runbook_search_description": spec.tools["runbook_search"].description,
        "knowledge_mode": str(spec.retrieval.get("knowledge_mode", "raw")),
        "retrieval_top_k": str(spec.retrieval.get("top_k", 6)),
        "output_cap": str(spec.runtime.get("output_cap", 140)),
        "max_tool_rounds": str(spec.runtime.get("max_tool_rounds", 4)),
    }


class RunbookActionAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: RunbookActionRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            self._runner = RunbookActionRunner(env_path=self.env_path)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case, hooks={})

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if not isinstance(output, dict) or "answer" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes="Expected JSON payload.")
        expected = case.expected
        if not isinstance(expected, dict):
            raise ValueError("Runbook action grader requires dict expected payloads.")
        actual = normalize_answer(output.get("answer", ""))
        expected_answer = normalize_answer(expected["answer"])
        if actual == expected_answer:
            return GradeResult(score=1.0, passed=True, labels=[])
        return GradeResult(
            score=0.0,
            passed=False,
            labels=["wrong_field:answer"],
            notes=f"actual={actual!r} output={output}",
        )

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


adapter = RunbookActionAdapter()
