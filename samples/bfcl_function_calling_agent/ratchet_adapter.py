from __future__ import annotations

from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any

from ratchet.model_client import ResponsesModelClient
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, RunRecord, TargetSemantics

try:
    from agent import BfclFunctionCallingRunner
except ModuleNotFoundError:
    from .agent import BfclFunctionCallingRunner


BASE_SPEC = AgentSpec(
    name="bfcl-function-calling-agent",
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
        "task_rule": "Choose the function call or calls needed to satisfy the user request.",
        "schema_rule": (
            "Use only functions listed in available_functions. Function names and argument names must match "
            "the provided schema exactly."
        ),
        "argument_rule": (
            "Infer arguments directly from the user request. Prefer literal values from the request. "
            "Do not invent optional arguments unless the request or schema default makes them necessary."
        ),
        "no_call_rule": "If no listed function is relevant, return an empty calls array.",
        "decision_rule": (
            "Prefer the first function whose name or description overlaps the request. For multiple possible calls, "
            "return each call in the same order the request mentions it."
        ),
        "output_rule": "Return JSON with a calls array. Each call has name and arguments.",
    },
    output_contract="Return JSON: {\"calls\":[{\"name\":\"function_name\",\"arguments\":{...}}]}.",
    runtime={"reasoning_effort": "low", "output_cap": 512},
    target_semantics={
        "task_rule": TargetSemantics(
            role="task_instructions",
            axes=["task_framing", "function_calling"],
            scope="global",
            risks=["broad_behavior_shift"],
            measurement_hints=["score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "schema_rule": TargetSemantics(
            role="schema_adherence_policy",
            axes=["schema_grounding", "argument_name_validity"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["wrong_call_delta", "invalid_output_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "argument_rule": TargetSemantics(
            role="argument_extraction_policy",
            axes=["argument_selection", "literal_value_grounding"],
            scope="slice",
            risks=["neighbor_case_regression"],
            measurement_hints=["wrong_argument_delta", "target_slice_score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "no_call_rule": TargetSemantics(
            role="no_call_policy",
            axes=["tool_relevance_boundary", "abstention"],
            scope="slice",
            risks=["false_negative_calls", "false_positive_calls"],
            measurement_hints=["wrong_call_count_delta", "target_slice_score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "decision_rule": TargetSemantics(
            role="decision_policy",
            axes=["tool_selection", "call_ordering", "tie_breaking"],
            scope="global",
            risks=["wrong_tool_regression", "broad_behavior_shift"],
            measurement_hints=["wrong_call_delta", "wrong_call_count_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "output_rule": TargetSemantics(
            role="output_format_rule",
            axes=["format_validity", "parser_compatibility"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "output_contract": TargetSemantics(
            role="external_output_contract",
            axes=["format_validity", "parser_compatibility", "contract_preservation"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "few_shot": TargetSemantics(
            role="example_bank",
            axes=["example_anchoring", "target_slice_recall"],
            scope="slice",
            risks=["neighbor_case_regression", "example_overfit"],
            measurement_hints=["target_slice_score_delta", "non_target_regression", "example_token_delta"],
            confidence=1.0,
            source="adapter",
        ),
        "model": TargetSemantics(
            role="model_choice",
            axes=["model_capability", "cost_latency_tradeoff"],
            scope="global",
            risks=["cost_latency_regression", "quality_regression"],
            measurement_hints=["score_delta", "cost_delta", "latency_delta"],
            confidence=1.0,
            source="adapter",
        ),
        "output_cap": TargetSemantics(
            role="output_budget_control",
            axes=["completion_integrity", "cost_latency_tradeoff"],
            scope="global",
            risks=["truncation_regression", "cost_latency_regression"],
            measurement_hints=["finish_reason_delta", "invalid_output_delta", "score_delta", "latency_delta"],
            confidence=1.0,
            source="adapter",
        ),
        "reasoning_effort": TargetSemantics(
            role="reasoning_effort_control",
            axes=["reasoning_depth", "cost_latency_tradeoff"],
            scope="global",
            risks=["cost_latency_regression", "quality_regression"],
            measurement_hints=["score_delta", "cost_delta", "latency_delta"],
            confidence=1.0,
            source="adapter",
        ),
    },
)


def agent_config_from_spec(spec: AgentSpec) -> dict[str, str]:
    return {
        "model": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "low")),
        "output_cap": str(spec.runtime.get("output_cap", 512)),
        "task_rule": spec.instructions.get("task_rule", ""),
        "schema_rule": spec.instructions.get("schema_rule", ""),
        "argument_rule": spec.instructions.get("argument_rule", ""),
        "no_call_rule": spec.instructions.get("no_call_rule", ""),
        "decision_rule": spec.instructions.get("decision_rule", ""),
        "output_rule": spec.instructions.get("output_rule", ""),
        "few_shot": render_few_shot_prompt(spec.few_shot),
    }


class BfclFunctionCallingAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: BfclFunctionCallingRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            client = ResponsesModelClient(env_path=self.env_path)
            self._runner = BfclFunctionCallingRunner(client=client)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        expected = case.expected
        if not isinstance(expected, dict):
            raise ValueError("BFCL grader requires dict expected payloads.")
        if not isinstance(output, dict) or "calls" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes=f"output={output!r}")
        if "invalid_output" in output:
            return GradeResult(
                score=0.0,
                passed=False,
                labels=["invalid_output"],
                notes=f"raw={output.get('invalid_output')!r}",
            )
        expected_calls = _expected_calls(expected.get("ground_truth", []))
        actual_calls = _actual_calls(output.get("calls", []))
        passed, labels, notes = _compare_calls(expected_calls, actual_calls)
        return GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=labels, notes=notes)

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


def _expected_calls(ground_truth: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(ground_truth, list):
        return calls
    for item in ground_truth:
        if not isinstance(item, dict):
            continue
        for name, arguments in item.items():
            calls.append({"name": str(name), "arguments": arguments if isinstance(arguments, dict) else {}})
    return calls


def _actual_calls(raw_calls: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if not isinstance(raw_calls, list):
        return calls
    for item in raw_calls:
        if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("arguments"), dict):
            calls.append({"name": item["name"], "arguments": item["arguments"]})
    return calls


def _compare_calls(expected_calls: list[dict[str, Any]], actual_calls: list[dict[str, Any]]) -> tuple[bool, list[str], str | None]:
    if len(expected_calls) != len(actual_calls):
        return (
            False,
            ["wrong_call_count", f"expected_count:{len(expected_calls)}", f"actual_count:{len(actual_calls)}"],
            f"expected {len(expected_calls)} call(s), got {len(actual_calls)}",
        )
    remaining = list(actual_calls)
    for expected in expected_calls:
        match_index = next((index for index, actual in enumerate(remaining) if _call_matches(expected, actual)), None)
        if match_index is None:
            expected_name = expected.get("name")
            actual_names = ",".join(str(call.get("name")) for call in actual_calls)
            return (
                False,
                ["wrong_call", f"expected:{expected_name}", f"actual:{actual_names}"],
                f"expected call {expected!r}; actual calls {actual_calls!r}",
            )
        remaining.pop(match_index)
    return True, [], None


def _call_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if str(expected.get("name")) != str(actual.get("name")):
        return False
    expected_args = expected.get("arguments") if isinstance(expected.get("arguments"), dict) else {}
    actual_args = actual.get("arguments") if isinstance(actual.get("arguments"), dict) else {}
    for key, allowed_values in expected_args.items():
        if key not in actual_args:
            if _argument_missing_allowed(allowed_values):
                continue
            return False
        if not _value_allowed(actual_args[key], allowed_values):
            return False
    expected_keys = set(expected_args)
    for key, value in actual_args.items():
        if key not in expected_keys and not _empty_value(value):
            return False
    return True


def _argument_missing_allowed(allowed_values: Any) -> bool:
    if not isinstance(allowed_values, list):
        return False
    return any(_empty_value(value) for value in allowed_values)


def _value_allowed(actual: Any, allowed_values: Any) -> bool:
    values = allowed_values if isinstance(allowed_values, list) else [allowed_values]
    return any(_values_equal(actual, expected) for expected in values)


def _values_equal(actual: Any, expected: Any) -> bool:
    if _empty_value(expected) and _empty_value(actual):
        return True
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(_values_equal(left, right) for left, right in zip(actual, expected))
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(_values_equal(actual[key], expected[key]) for key in actual)
    return str(actual).strip().lower() == str(expected).strip().lower()


def _empty_value(value: Any) -> bool:
    return value is None or value == "" or value == []


adapter = BfclFunctionCallingAdapter()
