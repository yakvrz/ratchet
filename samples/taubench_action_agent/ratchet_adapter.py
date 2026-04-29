from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

from ratchet.model_client import ResponsesModelClient
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, RunRecord

try:
    from agent import TauBenchActionRunner
except ModuleNotFoundError:
    from .agent import TauBenchActionRunner


BASE_SPEC = AgentSpec(
    name="taubench-action-agent",
    model="gemini-2.5-flash-lite",
    model_options=[
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    instructions={
        "task_rule": "Infer the customer-service workflow actions required by the tau-bench task.",
        "policy_rule": "Use the domain policy excerpt and available tool names. Do not choose tools that are not listed.",
        "action_rule": (
            "Focus on required tool/action names. Include an action only when the task context implies it is necessary "
            "to satisfy the user while following policy."
        ),
        "sequencing_rule": "Return actions in the likely execution order. Prefer a short necessary sequence over extra speculative steps.",
        "output_rule": "Return JSON with actions and message. Each action has name and arguments. Use {} when arguments are uncertain.",
    },
    output_contract="Return JSON: {\"actions\":[{\"name\":\"tool_name\",\"arguments\":{}}],\"message\":\"...\"}.",
    runtime={"reasoning_effort": "low", "output_cap": 512},
)


def agent_config_from_spec(spec: AgentSpec) -> dict[str, str]:
    return {
        "model": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "low")),
        "output_cap": str(spec.runtime.get("output_cap", 512)),
        "task_rule": spec.instructions.get("task_rule", ""),
        "policy_rule": spec.instructions.get("policy_rule", ""),
        "action_rule": spec.instructions.get("action_rule", ""),
        "sequencing_rule": spec.instructions.get("sequencing_rule", ""),
        "output_rule": spec.instructions.get("output_rule", ""),
        "few_shot": render_few_shot_prompt(spec.few_shot),
    }


class TauBenchActionAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: TauBenchActionRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            client = ResponsesModelClient(env_path=self.env_path)
            self._runner = TauBenchActionRunner(client=client)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        expected = case.expected
        if not isinstance(expected, dict):
            raise ValueError("tau-bench action grader requires dict expected payloads.")
        if not isinstance(output, dict) or "actions" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes=f"output={output!r}")
        if "invalid_output" in output:
            return GradeResult(
                score=0.0,
                passed=False,
                labels=["invalid_output"],
                notes=f"raw={output.get('invalid_output')!r}",
            )
        expected_names = _action_names(expected.get("actions", []))
        actual_names = _action_names(output.get("actions", []))
        score, labels, notes = _score_action_names(expected_names, actual_names)
        return GradeResult(score=score, passed=score >= 1.0, labels=labels, notes=notes)

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


def _action_names(actions: Any) -> list[str]:
    if not isinstance(actions, list):
        return []
    names = []
    for action in actions:
        if isinstance(action, dict) and action.get("name"):
            names.append(str(action["name"]))
    return names


def _score_action_names(expected: list[str], actual: list[str]) -> tuple[float, list[str], str | None]:
    if not expected and not actual:
        return 1.0, [], None
    if not expected:
        return 0.0, ["extra_action"], f"expected no actions; actual={actual!r}"
    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    correct = sum(min(expected_counts[name], actual_counts[name]) for name in expected_counts)
    precision = correct / max(len(actual), 1)
    recall = correct / len(expected)
    score = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    labels = []
    missing = list((expected_counts - actual_counts).elements())
    extra = list((actual_counts - expected_counts).elements())
    if missing:
        labels.append("missing_action")
    if extra:
        labels.append("extra_action")
    if actual != expected:
        labels.append("wrong_sequence")
    notes = f"expected={expected!r} actual={actual!r}"
    return (1.0 if actual == expected else round(score, 4)), labels, notes


adapter = TauBenchActionAdapter()
