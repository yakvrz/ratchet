from __future__ import annotations

import json
from pathlib import Path
import time

from ratchet.types import (
    AgentPatch,
    AgentSpec,
    AgentTool,
    DiagnosticTrace,
    EvalCase,
    GradeResult,
    OperationalMetrics,
    RunRecord,
)


BASE_SPEC = AgentSpec(
    name="fake-agent",
    model="large",
    model_options=["small", "large"],
    instructions={"system_prompt": "Answer politely."},
    tools={
        "search": AgentTool(
            name="search",
            description="Search the knowledge base.",
            policy="Use search when needed.",
            enabled=False,
        )
    },
    output_contract="Return the final answer text.",
    runtime={"output_cap": 80},
)


class FakeAdapter:
    def __init__(self) -> None:
        self.fail_once_case_id: str | None = None
        self.fail_once_patch_hash: str | None = None
        self.always_fail_case_id: str | None = None
        self.sleep_case_id: str | None = None
        self.sleep_seconds: float = 0.0
        self.bad_grade_case_id: str | None = None
        self._failed = False

    def reset(self) -> None:
        self.fail_once_case_id = None
        self.fail_once_patch_hash = None
        self.always_fail_case_id = None
        self.sleep_case_id = None
        self.sleep_seconds = 0.0
        self.bad_grade_case_id = None
        self._failed = False

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self.always_fail_case_id == case.id:
            raise RuntimeError("Injected persistent fake adapter failure.")
        patch_payload = json.dumps((patch or AgentPatch.empty()).to_dict(), sort_keys=True)
        if (
            self.fail_once_case_id == case.id
            and self.fail_once_patch_hash == patch_payload
            and not self._failed
        ):
            self._failed = True
            raise RuntimeError("Injected fake adapter failure.")
        if self.sleep_case_id == case.id and self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

        spec = BASE_SPEC.apply_patch(patch)
        needs_tool = bool(case.metadata.get("needs_tool", False))
        small_ok = bool(case.metadata.get("small_ok", True))
        prompt_text = " ".join(spec.instructions.values()).lower()
        exact_prompt = "exact grounded" in prompt_text
        prompt_covers_tool_cases = "tool-dependent cases as solvable" in prompt_text
        tool = spec.tools["search"]
        tool_on = tool.enabled
        tool_sharp = "exact grounded facts" in tool.description.lower()
        solved = exact_prompt and (tool_on or prompt_covers_tool_cases or not needs_tool) and (small_ok or spec.model != "small")
        answer = str(case.expected) if solved else "wrong"

        input_tokens = 160 if spec.model == "large" else 60
        output_tokens = 20 if solved else 10
        total_tokens = input_tokens + output_tokens + (30 if tool_on and needs_tool else 0)
        if tool_sharp and tool_on:
            total_tokens -= 10
        cost_usd = 0.004 if spec.model == "large" else 0.001
        if tool_on and needs_tool:
            cost_usd += 0.0001
        if tool_sharp and tool_on:
            cost_usd -= 0.00005
        latency_s = 1.0 if spec.model == "large" else 1.05
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

    def export(self, patch: AgentPatch | None, out_dir: Path) -> None:
        patch = patch or AgentPatch.empty()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(BASE_SPEC.apply_patch(patch).to_dict(), indent=2, sort_keys=True))


adapter = FakeAdapter()
