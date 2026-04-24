from __future__ import annotations

import json
import os
from pathlib import Path

from ratchet.benchmark import extract_first_number, normalize_text
from ratchet.harness import NorthstarHarnessRunner
from ratchet.types import EnumKnobSpec, EvalCase, GradeResult, RunRecord, SearchSpace, TextArtifactSpec


class NorthstarAdapter:
    def __init__(self, env_path: str | None = None, runner: NorthstarHarnessRunner | None = None) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def baseline(self) -> dict[str, str]:
        return {
            "model": "gpt-5.4-mini",
            "reasoning_effort": "none",
            "kb_tool_enabled": "off",
            "calculator_tool_enabled": "off",
            "kb_tool_description": "Search the Northstar Fulfillment handbook and fee deck for grounded policy details.",
            "calculator_tool_description": "Evaluate arithmetic expressions with decimals.",
            "knowledge_mode": "raw",
            "prompt_identity_rule": "Answer the user's benchmark question directly.",
            "prompt_answer_rule": "Return a concise final answer without explanation.",
            "prompt_kb_rule": "Use kb_lookup for policies, fees, queues, SLAs, labels, codes, or schedules.",
            "prompt_calc_rule": "Use calculator for arithmetic instead of mental math.",
            "prompt_fallback_rule": "If you truly cannot ground the answer, reply with exactly: unknown.",
            "output_cap": "80",
            "max_tool_rounds": "4",
        }

    def search_space(self) -> SearchSpace:
        return SearchSpace(
            enum_knobs=[
                EnumKnobSpec(
                    name="model",
                    kind="model",
                    values=["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4"],
                    default="gpt-5.4-mini",
                    description="Choose the base model for the harnessed agent.",
                ),
                EnumKnobSpec(
                    name="reasoning_effort",
                    kind="reasoning",
                    values=["none", "low"],
                    default="none",
                    description="Reasoning effort passed to the Responses API.",
                ),
                EnumKnobSpec(
                    name="kb_tool_enabled",
                    kind="tool",
                    values=["off", "on"],
                    default="off",
                    description="Whether the handbook retrieval tool is available.",
                ),
                EnumKnobSpec(
                    name="calculator_tool_enabled",
                    kind="tool",
                    values=["off", "on"],
                    default="off",
                    description="Whether the arithmetic calculator tool is available.",
                ),
                EnumKnobSpec(
                    name="knowledge_mode",
                    kind="kb",
                    values=["raw", "distilled"],
                    default="raw",
                    depends_on={"kb_tool_enabled": ["on"]},
                    description="Knowledge-card variant used by kb_lookup.",
                ),
                EnumKnobSpec(
                    name="output_cap",
                    kind="param",
                    values=["80", "60", "40"],
                    default="80",
                    description="max_output_tokens for each model call.",
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
                    name="kb_tool_description",
                    kind="tool",
                    default="Search the Northstar Fulfillment handbook and fee deck for grounded policy details.",
                    max_chars=220,
                    depends_on={"kb_tool_enabled": ["on"]},
                    description="Editable tool description for kb_lookup.",
                ),
                TextArtifactSpec(
                    name="calculator_tool_description",
                    kind="tool",
                    default="Evaluate arithmetic expressions with decimals.",
                    max_chars=180,
                    depends_on={"calculator_tool_enabled": ["on"]},
                    description="Editable tool description for calculator.",
                ),
                TextArtifactSpec(
                    name="prompt_identity_rule",
                    kind="prompt",
                    default="Answer the user's benchmark question directly.",
                    max_chars=180,
                    description="Prompt clause defining the task identity.",
                ),
                TextArtifactSpec(
                    name="prompt_answer_rule",
                    kind="prompt",
                    default="Return a concise final answer without explanation.",
                    max_chars=180,
                    description="Prompt clause defining answer style.",
                ),
                TextArtifactSpec(
                    name="prompt_kb_rule",
                    kind="prompt",
                    default="Use kb_lookup for policies, fees, queues, SLAs, labels, codes, or schedules.",
                    max_chars=220,
                    depends_on={"kb_tool_enabled": ["on"]},
                    description="Prompt clause guiding kb tool usage.",
                ),
                TextArtifactSpec(
                    name="prompt_calc_rule",
                    kind="prompt",
                    default="Use calculator for arithmetic instead of mental math.",
                    max_chars=180,
                    depends_on={"calculator_tool_enabled": ["on"]},
                    description="Prompt clause guiding calculator usage.",
                ),
                TextArtifactSpec(
                    name="prompt_fallback_rule",
                    kind="prompt",
                    default="If you truly cannot ground the answer, reply with exactly: unknown.",
                    max_chars=180,
                    description="Prompt clause controlling fallback behavior.",
                ),
            ],
        )

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        if self._runner is None:
            self._runner = NorthstarHarnessRunner(env_path=self.env_path)
        return self._runner.run_case(candidate, case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        prediction = str(output)
        if self._is_correct(case, prediction):
            return GradeResult(score=1.0, passed=True, labels=[])
        labels = self._failure_labels(case, prediction)
        return GradeResult(score=0.0, passed=False, labels=labels, notes="Northstar exact-match grader")

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))
        (out_dir / "README.md").write_text(
            "# Exported Northstar Candidate\n\n"
            "This bundle contains the selected candidate knobs for the built-in Northstar adapter.\n"
            "Run Ratchet with `--adapter examples.northstar.adapter:adapter` and this candidate JSON as input "
            "to reproduce the selected harness settings.\n"
        )

    def _is_correct(self, case: EvalCase, prediction: str) -> bool:
        answer_type = str(case.metadata.get("answer_type", "text"))
        if answer_type == "number":
            value = extract_first_number(prediction)
            expected = case.expected
            if value is None or expected is None:
                return False
            tolerance = float(case.metadata.get("tolerance", 0.01))
            return abs(value - float(expected)) <= tolerance
        normalized_prediction = normalize_text(prediction)
        candidates = [str(case.expected)] if case.expected is not None else []
        candidates.extend(str(alias) for alias in case.metadata.get("aliases", []))
        for candidate in candidates:
            normalized_candidate = normalize_text(candidate)
            if normalized_prediction == normalized_candidate or normalized_candidate in normalized_prediction:
                return True
        return False

    def _failure_labels(self, case: EvalCase, prediction: str) -> list[str]:
        if prediction.strip().lower() == "unknown":
            return ["unknown"]
        category = str(case.metadata.get("category", "text"))
        if category == "math":
            return ["wrong_math_answer"]
        return ["failed"]


adapter = NorthstarAdapter()
