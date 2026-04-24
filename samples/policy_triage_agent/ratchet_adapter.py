from __future__ import annotations

import json
import os
from pathlib import Path

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

from agent import PolicyTriageRunner


def build_search_space() -> SearchSpace:
    query_spec = CodeArtifactSpec(
        name="pre_tool_query_hook",
        language="python",
        callable_name="pre_tool_query_hook",
        signature="(query, context)",
        default="",
        max_chars=600,
        max_lines=12,
        depends_on={"policy_search_enabled": ["on"]},
        description="Bounded source-level hook that can rewrite a policy search query.",
    )
    context_spec = CodeArtifactSpec(
        name="post_tool_context_hook",
        language="python",
        callable_name="post_tool_context_hook",
        signature="(cards, context)",
        default="",
        max_chars=600,
        max_lines=12,
        depends_on={"policy_search_enabled": ["on"]},
        description="Bounded source-level hook that can filter or reorder retrieved policy cards.",
    )
    validator_spec = CodeArtifactSpec(
        name="post_answer_validator_hook",
        language="python",
        callable_name="post_answer_validator_hook",
        signature="(output, context)",
        default="",
        max_chars=600,
        max_lines=12,
        depends_on={"decision_validator_enabled": ["on"]},
        description="Bounded source-level hook that can validate or rewrite the final decision payload.",
    )

    def with_default(spec: CodeArtifactSpec) -> CodeArtifactSpec:
        return CodeArtifactSpec.from_dict({**spec.to_dict(), "default": default_hook_source(spec)})

    return SearchSpace(
        enum_knobs=[
            EnumKnobSpec(
                name="model",
                kind="model",
                values=["gpt-5.4", "gpt-5.4-mini"],
                default="gpt-5.4-mini",
                description="Model choice for reimbursement triage.",
            ),
            EnumKnobSpec(
                name="reasoning_effort",
                kind="reasoning",
                values=["low", "none"],
                default="none",
                description="Reasoning effort for the Responses API.",
            ),
            EnumKnobSpec(
                name="policy_search_enabled",
                kind="tool",
                values=["off", "on"],
                default="on",
                description="Whether the policy_search tool is available.",
            ),
            EnumKnobSpec(
                name="knowledge_mode",
                kind="kb",
                values=["raw", "distilled"],
                default="distilled",
                depends_on={"policy_search_enabled": ["on"]},
                description="Use fuller snippets or shorter distilled policy cards.",
            ),
            EnumKnobSpec(
                name="retrieval_top_k",
                kind="param",
                values=["4", "2", "1"],
                default="1",
                depends_on={"policy_search_enabled": ["on"]},
                description="Number of policy cards returned by policy_search.",
            ),
            EnumKnobSpec(
                name="output_cap",
                kind="param",
                values=["128", "96", "64"],
                default="64",
                description="max_output_tokens per turn.",
            ),
            EnumKnobSpec(
                name="max_tool_rounds",
                kind="param",
                values=["3", "2", "1"],
                default="1",
                description="Maximum number of tool continuation rounds.",
            ),
        ],
        text_artifacts=[
            TextArtifactSpec(
                name="prompt_output_rule",
                kind="prompt",
                default="Return JSON with decision and amount only.",
                max_chars=180,
                description="Prompt clause controlling output-format discipline.",
            ),
            TextArtifactSpec(
                name="prompt_grounding_rule",
                kind="prompt",
                default="Base the decision and amount only on grounded policy evidence from policy_search.",
                max_chars=220,
                description="Prompt clause controlling grounded policy reasoning.",
            ),
            TextArtifactSpec(
                name="prompt_tool_rule",
                kind="prompt",
                default="Always call policy_search before deciding.",
                max_chars=200,
                depends_on={"policy_search_enabled": ["on"]},
                description="Prompt clause controlling tool usage.",
            ),
            TextArtifactSpec(
                name="prompt_fallback_rule",
                kind="prompt",
                default="If the policy is unclear or requires review, choose escalate with amount 0.",
                max_chars=220,
                description="Prompt clause controlling escalation behavior.",
            ),
            TextArtifactSpec(
                name="policy_search_description",
                kind="tool",
                default="Search a frozen expense-policy snapshot and return grounded snippets.",
                max_chars=220,
                depends_on={"policy_search_enabled": ["on"]},
                description="Tool description shown to the model.",
            ),
            TextArtifactSpec(
                name="decision_validator_rule",
                kind="component",
                default="If the decision is not directly supported by retrieved policy evidence, replace it with escalate and amount 0.",
                max_chars=240,
                depends_on={"decision_validator_enabled": ["on"]},
                description="Internal validator policy for post-answer checks.",
            ),
        ],
        components=[
            ComponentSpec(
                name="decision_validator_enabled",
                kind="validator",
                values=["off", "on"],
                default="on",
                depends_on={"policy_search_enabled": ["on"]},
                description="Enable a bounded post-answer validation component.",
            )
        ],
        code_artifacts=[with_default(query_spec), with_default(context_spec), with_default(validator_spec)],
    )


class PolicyTriageAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: PolicyTriageRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner
        self._search_space = build_search_space()
        self._hook_loader = CodeArtifactLoader()

    def baseline(self) -> dict[str, str]:
        return {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
            "prompt_output_rule": "Return JSON with decision and amount only.",
            "prompt_grounding_rule": "Base the decision and amount only on grounded policy evidence from policy_search.",
            "prompt_tool_rule": "Always call policy_search before deciding.",
            "prompt_fallback_rule": "If the policy is unclear or requires review, choose escalate with amount 0.",
            "decision_validator_enabled": "on",
            "decision_validator_rule": "If the decision is not directly supported by retrieved policy evidence, replace it with escalate and amount 0.",
            "policy_search_enabled": "on",
            "policy_search_description": "Search a frozen expense-policy snapshot and return grounded snippets.",
            "knowledge_mode": "distilled",
            "retrieval_top_k": "1",
            "output_cap": "64",
            "max_tool_rounds": "1",
        }

    def search_space(self) -> SearchSpace:
        return self._search_space

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        if self._runner is None:
            self._runner = PolicyTriageRunner(env_path=self.env_path)
        hooks = self._hook_loader.build_hooks(candidate, self._search_space.code_artifacts)
        return self._runner.run_case(candidate, case, hooks=hooks)

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

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))
        (out_dir / "README.md").write_text(
            "# Exported Policy Triage Candidate\n\n"
            "This bundle contains the selected knobs for the standalone policy triage sample harness.\n"
        )


adapter = PolicyTriageAdapter()
