from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ratchet.capabilities import validation_check_schema
from ratchet.surface_opportunities import SurfaceOpportunity
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_program import TransformProgram


@dataclass(frozen=True)
class TransformContract:
    hook_ops: dict[str, list[str]]
    op_hooks: dict[str, list[str]]
    required_params: dict[str, list[str]]
    available_refs: dict[str, list[str]]
    validation_checks: dict[str, list[dict[str, Any]]]
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    opportunity_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def allowed_hooks(self) -> list[str]:
        return sorted(self.hook_ops)

    @property
    def allowed_ops(self) -> list[str]:
        return sorted(self.op_hooks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_ops": {key: list(value) for key, value in sorted(self.hook_ops.items())},
            "op_hooks": {key: list(value) for key, value in sorted(self.op_hooks.items())},
            "required_params": {key: list(value) for key, value in sorted(self.required_params.items())},
            "available_refs": {key: list(value) for key, value in sorted(self.available_refs.items())},
            "validation_checks": {
                key: [dict(item) for item in value]
                for key, value in sorted(self.validation_checks.items())
            },
            "examples": {
                key: [dict(item) for item in value]
                for key, value in sorted(self.examples.items())
            },
            "op_schema": transform_patch_schema_for_contract(self),
            "op_policy": [
                "Use only hook/op pairs listed in hook_ops; omitted hook means on_task_start.",
                "Use required_params for each op; do not invent prose-only operations.",
                "References are only legal when their root appears in available_refs for that hook.",
                "Compiler validation remains authoritative; these examples are shapes, not recipes.",
            ],
            "opportunity_contracts": {
                key: dict(value)
                for key, value in sorted(self.opportunity_contracts.items())
            },
        }


def build_transform_contract(
    surface: SurfaceSpec,
    surface_opportunities: list[SurfaceOpportunity],
) -> TransformContract:
    hook_ops: dict[str, list[str]] = {}
    op_hooks: dict[str, list[str]] = {}
    available_refs: dict[str, list[str]] = {}
    validation_checks: dict[str, list[dict[str, Any]]] = {}
    for hook_name, hook in sorted(surface.hooks.items()):
        if not hook.supported or not hook.allowed_ops:
            continue
        hook_ops[hook_name] = sorted(hook.allowed_ops)
        available_refs[hook_name] = list(hook.available_inputs)
        if hook.validation_checks:
            validation_checks[hook_name] = [dict(item) for item in hook.validation_checks]
        for op in hook.allowed_ops:
            op_hooks.setdefault(op, []).append(hook_name)
    required_params = _required_params(surface)
    examples = _examples(surface, hook_ops)
    opportunity_contracts = {
        opportunity.surface_opportunity_id: {
            "mechanism": opportunity.mechanism,
            "target_name": opportunity.target_name,
            "target_kind": opportunity.target_kind,
            "allowed_ops": list(opportunity.ops),
            "target_path": opportunity.target_path,
            "safe_patterns": list((opportunity.value_schema or {}).get("safe_patterns") or [])[:5],
        }
        for opportunity in surface_opportunities
    }
    return TransformContract(
        hook_ops={key: list(value) for key, value in sorted(hook_ops.items())},
        op_hooks={key: sorted(value) for key, value in sorted(op_hooks.items())},
        required_params=required_params,
        available_refs=available_refs,
        validation_checks=validation_checks,
        examples=examples,
        opportunity_contracts=opportunity_contracts,
    )


def transform_patch_schema_for_contract(contract: TransformContract) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hook": {"type": "string", "enum": contract.allowed_hooks},
            "op": {"type": "string", "enum": contract.allowed_ops},
            "section": {"type": "string", "maxLength": 160},
            "field": {"type": "string", "maxLength": 160},
            "target": {"type": "string", "maxLength": 160},
            "tool": {"type": "string", "maxLength": 160},
            "position": {"type": "string", "maxLength": 160},
            "content": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 2400},
                    {"type": "object", "minProperties": 1, "additionalProperties": True},
                    {"type": "array", "minItems": 1, "maxItems": 8, "items": {}},
                    {"type": "number"},
                    {"type": "boolean"},
                ]
            },
            "value": {},
            "initial": {},
            "type": {"type": "string", "maxLength": 160},
            "checks": {"type": "array", "items": validation_check_schema(), "maxItems": 12},
            "on_fail": {"type": "object", "additionalProperties": True},
            "when": {"type": "object", "additionalProperties": True},
            "unless": {"type": "object", "additionalProperties": True},
        },
        "required": ["op"],
        "additionalProperties": True,
    }


def contract_example_programs(contract: TransformContract) -> list[TransformProgram]:
    programs: list[TransformProgram] = []
    for family, examples in contract.examples.items():
        for index, patch in enumerate(examples):
            programs.append(
                TransformProgram.from_dict(
                    {
                        "candidate_id": f"contract_example_{family}_{index}",
                        "patches": [patch],
                    }
                )
            )
    return programs


def _required_params(surface: SurfaceSpec) -> dict[str, list[str]]:
    required: dict[str, list[str]] = {
        "add_context_section": ["hook", "op", "section", "content"],
        "replace_context_section": ["hook", "op", "section", "content"],
        "remove_context_section": ["hook", "op", "section"],
        "move_context_section": ["hook", "op", "section", "position"],
        "reorder_context_sections": ["hook", "op", "order"],
        "render_state_section": ["hook", "op", "section", "fields"],
        "define_state": ["op", "field", "type", "initial"],
        "set_state": ["hook", "op", "field", "value"],
        "append_state": ["hook", "op", "field", "value"],
        "merge_state": ["hook", "op", "field", "value"],
        "clear_state": ["hook", "op", "field"],
        "validate": ["hook", "op", "target", "checks"],
        "validate_claims": ["hook", "op", "target", "checks"],
        "rewrite_tool_description": ["hook", "op", "tool", "content_or_append"],
        "normalize_tool_args": ["hook", "op"],
        "repair_tool_args": ["hook", "op"],
        "set_model_config": ["hook", "op", "field", "value"],
        "rewrite_response": ["hook", "op", "content"],
        "block_response": ["hook", "op", "message"],
        "log_event": ["hook", "op"],
        "trace_annotation": ["hook", "op", "fields"],
    }
    if surface.model.model_options:
        required["set_model_config"].append("model_name_value_in_model_options")
    return required


def _examples(surface: SurfaceSpec, hook_ops: dict[str, list[str]]) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {}
    if "before_model_call" in hook_ops:
        editable_section = next(iter(surface.context.editable_sections), None)
        if editable_section and "replace_context_section" in hook_ops["before_model_call"]:
            examples.setdefault("surface_context", []).append(
                {
                    "hook": "before_model_call",
                    "op": "replace_context_section",
                    "section": editable_section,
                    "content": "Complete replacement instruction text.",
                }
            )
        if "add_context_section" in hook_ops["before_model_call"]:
            examples.setdefault("surface_context", []).append(
                {
                    "hook": "before_model_call",
                    "op": "add_context_section",
                    "section": "generated_guidance",
                    "content": "Focused additional instruction text.",
                }
            )
        if surface.model.model_options and "set_model_config" in hook_ops["before_model_call"]:
            target_model = next(
                (model for model in surface.model.model_options if model != surface.model.current_model),
                surface.model.model_options[0],
            )
            examples.setdefault("surface_model", []).append(
                {
                    "hook": "before_model_call",
                    "op": "set_model_config",
                    "field": "model_name",
                    "value": target_model,
                }
            )
    if "on_task_start" in hook_ops and "define_state" in hook_ops["on_task_start"]:
        examples.setdefault("surface_state", []).append(
            {"op": "define_state", "field": "candidate_state", "type": "list[string]", "initial": []}
        )
    if "before_tool_call" in hook_ops and "validate" in hook_ops["before_tool_call"]:
        examples.setdefault("surface_tool_loop", []).append(
            {
                "hook": "before_tool_call",
                "op": "validate",
                "target": "tool_call",
                "checks": [{"type": "args_schema_valid"}],
                "on_fail": {"op": "replan", "message": "Use a valid tool call."},
            }
        )
    if "before_user_response" in hook_ops and "validate" in hook_ops["before_user_response"]:
        examples.setdefault("surface_response", []).append(
            {
                "hook": "before_user_response",
                "op": "validate",
                "target": "draft_response",
                "checks": [{"type": "json_object"}],
            }
        )
    return examples
