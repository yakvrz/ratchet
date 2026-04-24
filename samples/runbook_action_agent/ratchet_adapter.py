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

from agent import RunbookActionRunner


def normalize_answer(text: str) -> str:
    return str(text).strip().replace("`", "").rstrip(".").lower()


def build_search_space() -> SearchSpace:
    query_spec = CodeArtifactSpec(
        name="pre_tool_query_hook",
        language="python",
        callable_name="pre_tool_query_hook",
        signature="(query, context)",
        default="",
        max_chars=700,
        max_lines=16,
        depends_on={"runbook_search_enabled": ["on"]},
        description="Bounded source-level hook that can rewrite a runbook search query.",
    )
    context_spec = CodeArtifactSpec(
        name="post_tool_context_hook",
        language="python",
        callable_name="post_tool_context_hook",
        signature="(cards, context)",
        default="",
        max_chars=700,
        max_lines=16,
        depends_on={"runbook_search_enabled": ["on"]},
        description="Bounded source-level hook that can filter or reorder retrieved runbook cards.",
    )
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

    def with_default(spec: CodeArtifactSpec) -> CodeArtifactSpec:
        default_source = default_hook_source(spec)
        if spec.name == "post_answer_validator_hook":
            default_source = """def post_answer_validator_hook(output, context):
    if not isinstance(output, dict):
        return output
    answer = str(output.get("answer", "unknown")).strip().lower()
    if answer == "unknown":
        return {"answer": "unknown"}
    question = str(context.get("case_input", "")).split("Options:", 1)[0].lower()
    words = [word.strip(".,:;!?()[]<>\\\"'") for word in question.split()]
    focus = [word for word in words if len(word) >= 5]
    haystack = "\\n".join(
        f"{card.get('doc_id', '')} {card.get('title', '')} {card.get('text', '')}".lower()
        for card in context.get("retrieved_cards", [])
        if isinstance(card, dict)
    )
    topical = sum(1 for word in focus if word and word in haystack)
    if answer in haystack and topical >= 1:
        return {"answer": output.get("answer", "unknown")}
    return {"answer": "unknown"}
"""
        return CodeArtifactSpec.from_dict({**spec.to_dict(), "default": default_source})

    return SearchSpace(
        enum_knobs=[
            EnumKnobSpec(
                name="model",
                kind="model",
                values=["gpt-5.4", "gpt-5.4-mini"],
                default="gpt-5.4",
                description="Model choice for the runbook-action harness.",
            ),
            EnumKnobSpec(
                name="reasoning_effort",
                kind="reasoning",
                values=["low", "none"],
                default="low",
                description="Reasoning effort passed to the Responses API.",
            ),
            EnumKnobSpec(
                name="runbook_search_enabled",
                kind="tool",
                values=["off", "on"],
                default="on",
                description="Whether the runbook_search tool is available.",
            ),
            EnumKnobSpec(
                name="knowledge_mode",
                kind="kb",
                values=["raw", "distilled"],
                default="raw",
                depends_on={"runbook_search_enabled": ["on"]},
                description="Use fuller snippets or shorter distilled runbook cards.",
            ),
            EnumKnobSpec(
                name="retrieval_top_k",
                kind="param",
                values=["6", "3", "1"],
                default="6",
                depends_on={"runbook_search_enabled": ["on"]},
                description="Number of runbook cards returned by runbook_search.",
            ),
            EnumKnobSpec(
                name="output_cap",
                kind="param",
                values=["140", "96", "64"],
                default="140",
                description="max_output_tokens per turn.",
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
                description="Prompt clause controlling output discipline.",
            ),
            TextArtifactSpec(
                name="prompt_grounding_rule",
                kind="prompt",
                default="Choose the next step only from runbook_search evidence. If the evidence is weak, you may still provide the most likely action.",
                max_chars=220,
                description="Prompt clause controlling grounded action selection.",
            ),
            TextArtifactSpec(
                name="prompt_tool_rule",
                kind="prompt",
                default="Always call runbook_search before answering.",
                max_chars=220,
                depends_on={"runbook_search_enabled": ["on"]},
                description="Prompt clause controlling tool usage.",
            ),
            TextArtifactSpec(
                name="prompt_fallback_rule",
                kind="prompt",
                default="If the runbook is inconclusive, provide your best action.",
                max_chars=220,
                description="Prompt clause controlling unknown behavior.",
            ),
            TextArtifactSpec(
                name="runbook_search_description",
                kind="tool",
                default="Search a frozen incident runbook snapshot and return grounded remediation snippets.",
                max_chars=220,
                depends_on={"runbook_search_enabled": ["on"]},
                description="Tool description shown to the model.",
            ),
            TextArtifactSpec(
                name="answer_validator_rule",
                kind="component",
                default="If the chosen action is not directly supported by retrieved runbook evidence, replace it with unknown.",
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
                depends_on={"runbook_search_enabled": ["on"]},
                description="Enable a bounded post-answer validation component.",
            ),
        ],
        code_artifacts=[with_default(query_spec), with_default(context_spec), with_default(validator_spec)],
    )


class RunbookActionAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: RunbookActionRunner | None = None,
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
            "prompt_grounding_rule": "Choose the next step only from runbook_search evidence. If the evidence is weak, you may still provide the most likely action.",
            "prompt_tool_rule": "Always call runbook_search before answering.",
            "prompt_fallback_rule": "If the runbook is inconclusive, provide your best action.",
            "answer_validator_enabled": "off",
            "answer_validator_rule": "If the chosen action is not directly supported by retrieved runbook evidence, replace it with unknown.",
            "runbook_search_enabled": "on",
            "runbook_search_description": "Search a frozen incident runbook snapshot and return grounded remediation snippets.",
            "knowledge_mode": "raw",
            "retrieval_top_k": "6",
            "output_cap": "140",
            "max_tool_rounds": "4",
        }

    def search_space(self) -> SearchSpace:
        return self._search_space

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        if self._runner is None:
            self._runner = RunbookActionRunner(env_path=self.env_path)
        hooks = self._hook_loader.build_hooks(candidate, self._search_space.code_artifacts)
        return self._runner.run_case(candidate, case, hooks=hooks)

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

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))
        (out_dir / "README.md").write_text(
            "# Exported Runbook Action Candidate\n\n"
            "This bundle contains the selected knobs for the standalone runbook-action sample harness.\n"
        )


adapter = RunbookActionAdapter()
