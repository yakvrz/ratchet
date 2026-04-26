from __future__ import annotations

import json
import os
from pathlib import Path

from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, AgentTool, EvalCase, GradeResult, RunRecord

try:
    from agent import PolicyTriageRunner
except ModuleNotFoundError:
    from .agent import PolicyTriageRunner


BASE_SPEC = AgentSpec(
    name="policy-triage-agent",
    model="gpt-5.4-mini",
    model_options=["gpt-5.4-mini", "gpt-5.4"],
    instructions={
        "output_rule": "Return JSON with decision and amount only.",
        "grounding_rule": "Base the decision and amount only on grounded policy evidence from policy_search.",
        "tool_rule": "Always call policy_search before deciding.",
        "fallback_rule": "If the policy is unclear or requires review, choose escalate with amount 0.",
        "validator_rule": "If the decision is not directly supported by retrieved policy evidence, replace it with escalate and amount 0.",
    },
    tools={
        "policy_search": AgentTool(
            name="policy_search",
            description="Search a frozen expense-policy snapshot and return grounded snippets.",
            policy="Use policy_search before deciding reimbursement eligibility and amount.",
            enabled=True,
        )
    },
    retrieval={"knowledge_mode": "distilled", "top_k": 1},
    output_contract="Return JSON with decision and amount only.",
    runtime={"reasoning_effort": "none", "output_cap": 64, "max_tool_rounds": 1, "decision_validator_enabled": True},
)


def agent_config_from_spec(spec: AgentSpec) -> dict[str, str]:
    return {
        "model": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "none")),
        "prompt_output_rule": spec.instructions.get("output_rule", ""),
        "prompt_grounding_rule": spec.instructions.get("grounding_rule", ""),
        "prompt_tool_rule": spec.instructions.get("tool_rule", ""),
        "prompt_fallback_rule": spec.instructions.get("fallback_rule", ""),
        "prompt_few_shot": render_few_shot_prompt(spec.few_shot),
        "decision_validator_enabled": "on" if spec.runtime.get("decision_validator_enabled") or spec.runtime.get("verifier_retry") else "off",
        "decision_validator_rule": spec.instructions.get("validator_rule", ""),
        "policy_search_enabled": "on" if spec.tools["policy_search"].enabled else "off",
        "policy_search_description": spec.tools["policy_search"].description,
        "knowledge_mode": str(spec.retrieval.get("knowledge_mode", "distilled")),
        "retrieval_top_k": str(spec.retrieval.get("top_k", 1)),
        "output_cap": str(spec.runtime.get("output_cap", 64)),
        "max_tool_rounds": str(spec.runtime.get("max_tool_rounds", 1)),
    }


class PolicyTriageAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: PolicyTriageRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            self._runner = PolicyTriageRunner(env_path=self.env_path)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case, hooks={})

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if not isinstance(output, dict):
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes="Expected JSON object.")
        expected = case.expected
        if not isinstance(expected, dict):
            raise ValueError("Policy triage grader requires dict expected payloads.")
        decision = str(output.get("decision", "")).strip().lower()
        amount = int(output.get("amount", -1))
        expected_decision = str(expected["decision"]).strip().lower()
        expected_amount = int(expected["amount"])
        labels: list[str] = []
        score = 0.0
        if decision == expected_decision:
            score += 0.5
        else:
            labels.append("wrong_field:decision")
        if amount == expected_amount:
            score += 0.5
        else:
            labels.append("wrong_field:amount")
        return GradeResult(
            score=score,
            passed=score == 1.0,
            labels=labels,
            notes=f"actual={output}",
        )

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


adapter = PolicyTriageAdapter()
