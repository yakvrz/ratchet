from __future__ import annotations

from typing import Any

from ratchet.types import AgentSpec, EditableTarget, OptimizationObjective


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
    def generate(
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
                        )
                    )
        if "retrieval" in allowed:
            for key, value in sorted(spec.retrieval.items()):
                targets.append(
                    EditableTarget(
                        name=f"retrieval.{key}",
                        kind="retrieval",
                        path=f"retrieval.{key}",
                        current_value=value,
                        allowed_ops=["set_retrieval_param"],
                        description=f"Retrieval parameter {key}.",
                        value_schema=_infer_value_schema(value),
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
                    description="Few-shot examples appended to the agent spec.",
                    value_schema={"type": "object", "shape": "structured"},
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
                )
            )
        return targets
