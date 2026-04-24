from __future__ import annotations

import json
from pathlib import Path
import time

from ratchet.types import (
    DiagnosticTrace,
    EnumKnobSpec,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    RunRecord,
    SearchSpace,
    TextArtifactSpec,
)


class FakeAdapter:
    def __init__(self) -> None:
        self.fail_once_case_id: str | None = None
        self.fail_once_candidate: dict[str, str] | None = None
        self.always_fail_case_id: str | None = None
        self.sleep_case_id: str | None = None
        self.sleep_seconds: float = 0.0
        self.bad_grade_case_id: str | None = None
        self._failed = False

    def reset(self) -> None:
        self.fail_once_case_id = None
        self.fail_once_candidate = None
        self.always_fail_case_id = None
        self.sleep_case_id = None
        self.sleep_seconds = 0.0
        self.bad_grade_case_id = None
        self._failed = False

    def baseline(self) -> dict[str, str]:
        return {
            "system_prompt": "Answer politely.",
            "search_tool_enabled": "off",
            "search_tool_description": "Search the knowledge base.",
            "model": "large",
        }

    def search_space(self) -> SearchSpace:
        return SearchSpace(
            enum_knobs=[
                EnumKnobSpec(
                    name="search_tool_enabled",
                    kind="tool",
                    values=["off", "on"],
                    default="off",
                    description="Toggle the task tool.",
                ),
                EnumKnobSpec(
                    name="model",
                    kind="model",
                    values=["large", "small"],
                    default="large",
                    description="Model size knob.",
                ),
            ],
            text_artifacts=[
                TextArtifactSpec(
                    name="system_prompt",
                    kind="prompt",
                    default="Answer politely.",
                    max_chars=160,
                    description="Prompt artifact controlling exactness.",
                ),
                TextArtifactSpec(
                    name="search_tool_description",
                    kind="tool",
                    default="Search the knowledge base.",
                    max_chars=180,
                    depends_on={"search_tool_enabled": ["on"]},
                    description="Tool artifact describing search behavior.",
                ),
            ],
        )

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        if self.always_fail_case_id == case.id:
            raise RuntimeError("Injected persistent fake adapter failure.")
        if (
            self.fail_once_case_id == case.id
            and self.fail_once_candidate == candidate
            and not self._failed
        ):
            self._failed = True
            raise RuntimeError("Injected fake adapter failure.")
        if self.sleep_case_id == case.id and self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        needs_tool = bool(case.metadata.get("needs_tool", False))
        small_ok = bool(case.metadata.get("small_ok", True))
        exact_prompt = "exact grounded" in candidate["system_prompt"].lower()
        tool_on = candidate["search_tool_enabled"] == "on"
        tool_description = candidate["search_tool_description"].lower()
        tool_sharp = "exact grounded facts" in tool_description
        model = candidate["model"]
        solved = exact_prompt and (tool_on or not needs_tool) and (small_ok or model != "small")
        answer = str(case.expected) if solved else "wrong"

        input_tokens = 160 if model == "large" else 60
        output_tokens = 20 if solved else 10
        total_tokens = input_tokens + output_tokens + (30 if tool_on and needs_tool else 0)
        if tool_sharp and tool_on:
            total_tokens -= 10
        cost_usd = 0.004 if model == "large" else 0.001
        if tool_on and needs_tool:
            cost_usd += 0.0001
        if tool_sharp and tool_on:
            cost_usd -= 0.00005
        latency_s = 1.0 if model == "large" else 1.05
        if tool_on and needs_tool:
            latency_s += 0.02

        return RunRecord(
            output=answer,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=cost_usd,
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=["fake_tool"] if tool_on and needs_tool else [],
                raw_output_text=answer,
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        if self.bad_grade_case_id == case.id:
            raise ValueError("Injected fake grader failure.")
        correct = str(output) == str(case.expected)
        if correct:
            return GradeResult(score=1.0, passed=True, labels=[])
        return GradeResult(score=0.0, passed=False, labels=["failed"], notes="Fake exact-match grader")

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True))


adapter = FakeAdapter()
