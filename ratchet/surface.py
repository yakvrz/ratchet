from __future__ import annotations

from typing import Any

from ratchet.io import stable_digest
from ratchet.types import AgentSpec, EditableTarget, OptimizationObjective, TargetSemantics


def _text_value_schema(max_chars: int) -> dict[str, Any]:
    return {"type": "string", "shape": "text", "maxLength": max_chars}


def _infer_value_schema(
    value: Any,
    *,
    choices: list[str] | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    if choices:
        return {"type": "string", "shape": "categorical", "enum": list(choices)}
    if isinstance(value, bool):
        return {"type": "boolean", "shape": "boolean"}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "integer", "shape": "numeric"}
    if isinstance(value, float):
        return {"type": "number", "shape": "numeric"}
    if isinstance(value, dict):
        return {"type": "object", "shape": "structured"}
    if isinstance(value, list):
        return {"type": "array", "shape": "list"}
    schema: dict[str, Any] = {"type": "string", "shape": "text"}
    if max_chars is not None:
        schema["maxLength"] = max_chars
    return schema


class SurfaceGenerator:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[EditableTarget, ...]] = {}

    def generate(
        self,
        spec: AgentSpec | None,
        objective: OptimizationObjective,
    ) -> list[EditableTarget]:
        key = stable_digest(
            {
                "agent_spec": spec.to_dict() if spec is not None else None,
                "objective": objective.to_dict(),
            }
        )
        cached = self._cache.get(key)
        if cached is None:
            cached = tuple(self._generate_uncached(spec, objective))
            self._cache[key] = cached
        return list(cached)

    def _generate_uncached(
        self,
        spec: AgentSpec | None,
        objective: OptimizationObjective,
    ) -> list[EditableTarget]:
        if spec is None:
            return [
                EditableTarget(
                    name="wrapper_instruction",
                    kind="instruction",
                    path="instructions.wrapper",
                    current_value="",
                    allowed_ops=["add_instruction", "revise_instruction"],
                    description="Generic wrapper instruction applied around the original agent when no AgentSpec is available.",
                    max_chars=1200,
                    value_schema=_text_value_schema(1200),
                    semantics=TargetSemantics(
                        role="wrapper_instruction",
                        axes=["task_framing", "instruction_following"],
                        scope="global",
                        risks=["broad_behavior_shift"],
                        measurement_hints=["score_delta", "non_target_regression"],
                        confidence=0.7,
                        source="inferred",
                    ),
                )
            ]
        constraints = objective.constraints
        allowed = set(constraints.allowed_edits)
        targets: list[EditableTarget] = []
        if "model" in allowed:
            choices = constraints.allowed_models or spec.model_options
            choices = [item for item in choices if item != spec.model]
            if choices:
                targets.append(
                    EditableTarget(
                        name="model",
                        kind="model",
                        path="model",
                        current_value=spec.model,
                        allowed_ops=["change_model"],
                        description="Model used by the optimized agent.",
                        choices=choices,
                        value_schema=_infer_value_schema(spec.model, choices=choices),
                        semantics=_target_semantics(spec, name="model", kind="model", path="model"),
                    )
                )
        if "instruction" in allowed:
            for name, text in sorted(spec.instructions.items()):
                max_chars = max(len(text) + 700, 1200)
                targets.append(
                    EditableTarget(
                        name=f"instructions.{name}",
                        kind="instruction",
                        path=f"instructions.{name}",
                        current_value=text,
                        allowed_ops=["add_instruction", "revise_instruction"],
                        description=f"Instruction section {name}.",
                        max_chars=max_chars,
                        value_schema=_text_value_schema(max_chars),
                        semantics=_target_semantics(
                            spec,
                            name=f"instructions.{name}",
                            kind="instruction",
                            path=f"instructions.{name}",
                        ),
                    )
                )
        if "output" in allowed:
            max_chars = max(len(spec.output_contract) + 500, 900)
            targets.append(
                EditableTarget(
                    name="output_contract",
                    kind="output",
                    path="output_contract",
                    current_value=spec.output_contract,
                    allowed_ops=["add_output_constraint"],
                    description="Externally visible answer format and output constraints.",
                    max_chars=max_chars,
                    value_schema=_text_value_schema(max_chars),
                    semantics=_target_semantics(
                        spec,
                        name="output_contract",
                        kind="output",
                        path="output_contract",
                    ),
                )
            )
        if "tool" in allowed:
            for name, tool in sorted(spec.tools.items()):
                description_max_chars = max(len(tool.description) + 400, 700)
                targets.append(
                    EditableTarget(
                        name=f"tools.{name}.description",
                        kind="tool",
                        path=f"tools.{name}.description",
                        current_value=tool.description,
                        allowed_ops=["revise_tool_description"],
                        description=f"Description for tool {name}.",
                        max_chars=description_max_chars,
                        value_schema=_text_value_schema(description_max_chars),
                        semantics=_target_semantics(
                            spec,
                            name=f"tools.{name}.description",
                            kind="tool",
                            path=f"tools.{name}.description",
                        ),
                    )
                )
                policy_max_chars = max(len(tool.policy) + 400, 700)
                targets.append(
                    EditableTarget(
                        name=f"tools.{name}.policy",
                        kind="tool",
                        path=f"tools.{name}.policy",
                        current_value=tool.policy,
                        allowed_ops=["revise_tool_policy"],
                        description=f"Use policy for tool {name}.",
                        max_chars=policy_max_chars,
                        value_schema=_text_value_schema(policy_max_chars),
                        semantics=_target_semantics(
                            spec,
                            name=f"tools.{name}.policy",
                            kind="tool",
                            path=f"tools.{name}.policy",
                        ),
                    )
                )
                if not tool.enabled:
                    targets.append(
                        EditableTarget(
                            name=f"tools.{name}.enabled",
                            kind="tool",
                            path=f"tools.{name}.enabled",
                            current_value=False,
                            allowed_ops=["set_runtime_param"],
                            description=f"Enable tool {name}.",
                            value_schema=_infer_value_schema(False),
                            semantics=_target_semantics(
                                spec,
                                name=f"tools.{name}.enabled",
                                kind="tool",
                                path=f"tools.{name}.enabled",
                            ),
                        )
                    )
        if "runtime" in allowed:
            for key, value in sorted(spec.runtime.items()):
                targets.append(
                    EditableTarget(
                        name=f"runtime.{key}",
                        kind="runtime",
                        path=f"runtime.{key}",
                        current_value=value,
                        allowed_ops=["set_runtime_param"],
                        description=f"Runtime parameter {key}.",
                        value_schema=_infer_value_schema(value),
                        semantics=_target_semantics(
                            spec,
                            name=f"runtime.{key}",
                            kind="runtime",
                            path=f"runtime.{key}",
                        ),
                    )
                )
        if "few_shot" in allowed:
            targets.append(
                EditableTarget(
                    name="few_shot",
                    kind="few_shot",
                    path="few_shot",
                    current_value=spec.few_shot,
                    allowed_ops=["add_few_shot"],
                    description=(
                        "Few-shot examples appended to the agent spec. Values must be arrays of proposal-safe "
                        "examples that cite source_case_id from the provided train example bank."
                    ),
                    semantics=_target_semantics(spec, name="few_shot", kind="few_shot", path="few_shot"),
                    value_schema={
                        "type": "array",
                        "shape": "few_shot_examples",
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "required": ["source_case_id", "input", "output", "purpose"],
                            "properties": {
                                "source_case_id": {"type": "string", "maxLength": 160},
                                "input": {"type": "string", "maxLength": 1200},
                                "output": {"type": "object"},
                                "purpose": {"type": "string", "maxLength": 240},
                            },
                            "additionalProperties": False,
                        },
                    },
                )
            )
        if "verifier" in allowed:
            targets.append(
                EditableTarget(
                    name="verifier_retry",
                    kind="verifier",
                    path="runtime.verifier_retry",
                    current_value=spec.runtime.get("verifier_retry", False),
                    allowed_ops=["add_verifier_retry"],
                    description="Generic verifier/retry wrapper setting.",
                    value_schema=_infer_value_schema(False),
                    semantics=_target_semantics(
                        spec,
                        name="verifier_retry",
                        kind="verifier",
                        path="runtime.verifier_retry",
                    ),
                )
            )
        return targets


def _target_semantics(spec: AgentSpec, *, name: str, kind: str, path: str) -> TargetSemantics:
    for key in _semantic_keys(name=name, path=path):
        explicit = spec.target_semantics.get(key)
        if explicit is not None:
            return explicit
    return _infer_target_semantics(name=name, kind=kind, path=path)


def _semantic_keys(*, name: str, path: str) -> list[str]:
    keys = [name, path]
    for value in (name, path):
        if value.startswith("instructions."):
            keys.append(value.removeprefix("instructions."))
        if value.startswith("runtime."):
            keys.append(value.removeprefix("runtime."))
    return list(dict.fromkeys(keys))


def _infer_target_semantics(*, name: str, kind: str, path: str) -> TargetSemantics:
    simple_name = name.rsplit(".", 1)[-1]
    if kind == "instruction":
        return _infer_instruction_semantics(simple_name)
    if kind == "output":
        return TargetSemantics(
            role="external_output_contract",
            axes=["format_validity", "parser_compatibility", "contract_preservation"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "non_target_regression"],
            confidence=0.9,
            source="inferred",
        )
    if kind == "few_shot":
        return TargetSemantics(
            role="example_bank",
            axes=["example_anchoring", "target_slice_recall"],
            scope="slice",
            risks=["neighbor_label_regression", "example_overfit"],
            measurement_hints=["target_slice_score_delta", "non_target_regression", "example_token_delta"],
            confidence=0.9,
            source="inferred",
        )
    if kind == "model":
        return TargetSemantics(
            role="model_choice",
            axes=["model_capability", "cost_latency_tradeoff"],
            scope="global",
            risks=["cost_latency_regression", "quality_regression"],
            measurement_hints=["score_delta", "cost_delta", "latency_delta"],
            confidence=0.95,
            source="inferred",
        )
    if kind == "runtime":
        return _infer_runtime_semantics(simple_name)
    if kind == "tool":
        return _infer_tool_semantics(name=name)
    if kind == "verifier":
        return TargetSemantics(
            role="verifier_retry_policy",
            axes=["runtime_reliability", "contract_preservation"],
            scope="global",
            risks=["latency_regression", "cost_latency_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "latency_delta"],
            confidence=0.8,
            source="inferred",
        )
    return TargetSemantics(
        role=f"{kind}_policy",
        axes=[f"{kind}_behavior"],
        scope="global",
        risks=["quality_regression"],
        measurement_hints=["score_delta", "non_target_regression"],
        confidence=0.4,
        source="inferred",
    )


def _infer_instruction_semantics(name: str) -> TargetSemantics:
    semantics_by_name = {
        "task_rule": TargetSemantics(
            role="task_instructions",
            axes=["task_framing", "instruction_following"],
            scope="global",
            risks=["broad_behavior_shift"],
            measurement_hints=["score_delta", "non_target_regression"],
            confidence=0.9,
            source="inferred",
        ),
        "label_rule": TargetSemantics(
            role="label_space",
            axes=["classification_boundary", "label_validity"],
            scope="global",
            risks=["label_space_regression"],
            measurement_hints=["target_label_score_delta", "wrong_label_delta", "non_target_regression"],
            confidence=0.95,
            source="inferred",
        ),
        "label_descriptions": TargetSemantics(
            role="label_description",
            axes=["classification_boundary", "semantic_grounding"],
            scope="slice",
            risks=["neighbor_label_regression"],
            measurement_hints=["target_label_score_delta", "confusion_delta", "non_target_regression"],
            confidence=0.95,
            source="inferred",
        ),
        "label_aliases": TargetSemantics(
            role="label_alias_mapping",
            axes=["classification_boundary", "confusion_resolution"],
            scope="slice",
            risks=["neighbor_label_regression"],
            measurement_hints=["target_label_score_delta", "confusion_delta", "non_target_regression"],
            confidence=0.95,
            source="inferred",
        ),
        "confusable_label_rules": TargetSemantics(
            role="confusable_label_policy",
            axes=["classification_boundary", "confusion_resolution", "tie_breaking"],
            scope="slice",
            risks=["neighbor_label_regression"],
            measurement_hints=["target_label_score_delta", "confusion_delta", "non_target_regression"],
            confidence=0.95,
            source="inferred",
        ),
        "decision_rule": TargetSemantics(
            role="decision_policy",
            axes=["selection_policy", "tie_breaking"],
            scope="global",
            risks=["broad_behavior_shift"],
            measurement_hints=["score_delta", "confusion_delta", "non_target_regression"],
            confidence=0.9,
            source="inferred",
        ),
        "schema_rule": TargetSemantics(
            role="schema_adherence_policy",
            axes=["schema_grounding", "argument_name_validity"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["wrong_call_delta", "invalid_output_delta", "non_target_regression"],
            confidence=0.9,
            source="inferred",
        ),
        "argument_rule": TargetSemantics(
            role="argument_extraction_policy",
            axes=["argument_selection", "literal_value_grounding"],
            scope="slice",
            risks=["neighbor_case_regression"],
            measurement_hints=["wrong_argument_delta", "target_slice_score_delta", "non_target_regression"],
            confidence=0.9,
            source="inferred",
        ),
        "no_call_rule": TargetSemantics(
            role="no_call_policy",
            axes=["tool_relevance_boundary", "abstention"],
            scope="slice",
            risks=["false_negative_calls", "false_positive_calls"],
            measurement_hints=["wrong_call_count_delta", "target_slice_score_delta", "non_target_regression"],
            confidence=0.9,
            source="inferred",
        ),
        "output_rule": TargetSemantics(
            role="output_format_rule",
            axes=["format_validity", "parser_compatibility"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "non_target_regression"],
            confidence=0.95,
            source="inferred",
        ),
    }
    return semantics_by_name.get(
        name,
        TargetSemantics(
            role="instruction_policy",
            axes=["instruction_following"],
            scope="global",
            risks=["quality_regression"],
            measurement_hints=["score_delta", "non_target_regression"],
            confidence=0.55,
            source="inferred",
        ),
    )


def _infer_runtime_semantics(name: str) -> TargetSemantics:
    if name in {"output_cap", "max_output_tokens", "max_tokens"}:
        return TargetSemantics(
            role="output_budget_control",
            axes=["completion_integrity", "cost_latency_tradeoff"],
            scope="global",
            risks=["truncation_regression", "cost_latency_regression"],
            measurement_hints=["finish_reason_delta", "invalid_output_delta", "score_delta", "latency_delta"],
            confidence=0.9,
            source="inferred",
        )
    if name == "reasoning_effort":
        return TargetSemantics(
            role="reasoning_effort_control",
            axes=["reasoning_depth", "cost_latency_tradeoff"],
            scope="global",
            risks=["cost_latency_regression", "quality_regression"],
            measurement_hints=["score_delta", "cost_delta", "latency_delta"],
            confidence=0.9,
            source="inferred",
        )
    return TargetSemantics(
        role="runtime_control",
        axes=["runtime_reliability", "cost_latency_tradeoff"],
        scope="global",
        risks=["cost_latency_regression", "quality_regression"],
        measurement_hints=["score_delta", "cost_delta", "latency_delta"],
        confidence=0.7,
        source="inferred",
    )


def _infer_tool_semantics(*, name: str) -> TargetSemantics:
    if name.endswith(".description"):
        return TargetSemantics(
            role="tool_description",
            axes=["tool_selection", "schema_grounding", "argument_grounding"],
            scope="slice",
            risks=["wrong_tool_regression", "argument_regression"],
            measurement_hints=[
                "wrong_call_delta",
                "invalid_tool_call_delta",
                "tool_error_delta",
                "target_slice_score_delta",
                "non_target_regression",
            ],
            confidence=0.85,
            source="inferred",
        )
    if name.endswith(".policy"):
        return TargetSemantics(
            role="tool_policy",
            axes=["tool_selection", "action_policy", "precondition_checking", "stop_continue_boundary"],
            scope="slice",
            risks=["wrong_tool_regression", "tool_overuse", "tool_underuse"],
            measurement_hints=[
                "wrong_call_delta",
                "tool_call_delta",
                "tool_error_delta",
                "turn_delta",
                "target_slice_score_delta",
                "non_target_regression",
            ],
            confidence=0.85,
            source="inferred",
        )
    return TargetSemantics(
        role="tool_enablement",
        axes=["tool_availability", "runtime_reliability"],
        scope="global",
        risks=["tool_surface_regression"],
        measurement_hints=["score_delta", "wrong_call_delta", "non_target_regression"],
        confidence=0.75,
        source="inferred",
    )
