from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.context_graph import ContextGraph, ContextSection
from ratchet.types import AgentSpec, TargetSemantics


SUPPORTED_HOOKS = {
    "on_task_start",
    "after_user_message",
    "before_model_call",
    "after_model_call",
    "before_tool_call",
    "after_tool_result",
    "on_tool_error",
    "before_user_response",
    "on_task_end",
}


@dataclass(frozen=True)
class HookSurface:
    name: str
    supported: bool
    available_inputs: tuple[str, ...] = ()
    allowed_outputs: tuple[str, ...] = ()
    allowed_ops: tuple[str, ...] = ()
    method: str = "native"

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_HOOKS:
            raise ValueError(f"Unsupported hook surface: {self.name}")
        if self.method not in {"native", "emulated", "unsupported"}:
            raise ValueError(f"Unsupported hook method: {self.method}")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["available_inputs"] = list(self.available_inputs)
        row["allowed_outputs"] = list(self.allowed_outputs)
        row["allowed_ops"] = list(self.allowed_ops)
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HookSurface":
        return cls(
            name=str(payload["name"]),
            supported=bool(payload.get("supported", False)),
            available_inputs=tuple(str(item) for item in payload.get("available_inputs", [])),
            allowed_outputs=tuple(str(item) for item in payload.get("allowed_outputs", [])),
            allowed_ops=tuple(str(item) for item in payload.get("allowed_ops", [])),
            method=str(payload.get("method", "native")),
        )


@dataclass(frozen=True)
class ContextSurface:
    graph: ContextGraph
    editable_sections: tuple[str, ...]
    generated_sections_allowed: bool = True
    removable_sections_allowed: bool = True
    reorderable_sections_allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "editable_sections": list(self.editable_sections),
            "generated_sections_allowed": self.generated_sections_allowed,
            "removable_sections_allowed": self.removable_sections_allowed,
            "reorderable_sections_allowed": self.reorderable_sections_allowed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextSurface":
        return cls(
            graph=ContextGraph.from_dict(dict(payload["graph"])),
            editable_sections=tuple(str(item) for item in payload.get("editable_sections", [])),
            generated_sections_allowed=bool(payload.get("generated_sections_allowed", True)),
            removable_sections_allowed=bool(payload.get("removable_sections_allowed", True)),
            reorderable_sections_allowed=bool(payload.get("reorderable_sections_allowed", True)),
        )


@dataclass(frozen=True)
class StateSurface:
    supports_persistent_state: bool = True
    existing_fields: tuple[str, ...] = ()
    add_fields_allowed: bool = True
    internal_only_fields: bool = True
    expose_to_context: bool = True
    typed_state_supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["existing_fields"] = list(self.existing_fields)
        return row


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolSpec":
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            schema=dict(payload.get("schema", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ToolSurface:
    tools: tuple[ToolSpec, ...] = ()
    tool_schema_rewrite_allowed: bool = False
    tool_description_rewrite_allowed: bool = False
    tool_call_interception_allowed: bool = False
    tool_execution_modification_allowed: bool = False
    tool_result_rewrite_allowed: bool = False
    tool_metadata_allowed: bool = True

    @property
    def tools_available(self) -> bool:
        return bool(self.tools)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [tool.to_dict() for tool in self.tools],
            "tools_available": self.tools_available,
            "tool_schema_rewrite_allowed": self.tool_schema_rewrite_allowed,
            "tool_description_rewrite_allowed": self.tool_description_rewrite_allowed,
            "tool_call_interception_allowed": self.tool_call_interception_allowed,
            "tool_execution_modification_allowed": self.tool_execution_modification_allowed,
            "tool_result_rewrite_allowed": self.tool_result_rewrite_allowed,
            "tool_metadata_allowed": self.tool_metadata_allowed,
        }


@dataclass(frozen=True)
class ModelSurface:
    provider: str = "configurable"
    current_model: str = ""
    model_options: tuple[str, ...] = ()
    model_name_configurable: bool = True
    temperature_configurable: bool = False
    max_tokens_configurable: bool = True
    reasoning_effort_configurable: bool = True
    tool_choice_mode_configurable: bool = False
    auxiliary_model_calls_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["model_options"] = list(self.model_options)
        return row


@dataclass(frozen=True)
class ResponseSurface:
    draft_response_interception_allowed: bool = True
    response_rewrite_allowed: bool = True
    response_blocking_allowed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceSpec:
    agent_id: str
    context: ContextSurface
    hooks: dict[str, HookSurface]
    state: StateSurface
    tools: ToolSurface
    model: ModelSurface
    response: ResponseSurface
    immutable_boundaries: tuple[str, ...]
    safety_constraints: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.agent_id:
            raise ValueError("SurfaceSpec agent_id must be non-empty.")
        missing = sorted(SUPPORTED_HOOKS - set(self.hooks))
        if missing:
            raise ValueError(f"SurfaceSpec missing hook surfaces: {', '.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "context": self.context.to_dict(),
            "hooks": {name: hook.to_dict() for name, hook in sorted(self.hooks.items())},
            "state": self.state.to_dict(),
            "tools": self.tools.to_dict(),
            "model": self.model.to_dict(),
            "response": self.response.to_dict(),
            "immutable_boundaries": list(self.immutable_boundaries),
            "safety_constraints": list(self.safety_constraints),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SurfaceTarget:
    name: str
    kind: str
    path: str
    current_value: Any
    allowed_ops: tuple[str, ...]
    description: str = ""
    choices: tuple[str, ...] = ()
    max_chars: int | None = None
    value_schema: dict[str, Any] = field(default_factory=dict)
    semantics: TargetSemantics = field(default_factory=TargetSemantics)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["allowed_ops"] = list(self.allowed_ops)
        row["choices"] = list(self.choices)
        row["semantics"] = self.semantics.to_dict()
        return row


def surface_targets(surface: SurfaceSpec) -> list[SurfaceTarget]:
    targets: list[SurfaceTarget] = []
    for section in surface.context.graph.sections:
        if section.name not in surface.context.editable_sections:
            continue
        targets.append(
            SurfaceTarget(
                name=section.name,
                kind="context",
                path=f"context.{section.name}",
                current_value=section.content,
                allowed_ops=tuple(_context_allowed_ops(surface)),
                description=f"Context section {section.name}.",
                max_chars=8000,
            )
        )
    if surface.context.generated_sections_allowed:
        targets.append(
            SurfaceTarget(
                name="generated_context",
                kind="context",
                path="context.generated",
                current_value=None,
                allowed_ops=("add_context_section", "render_state_section"),
                description="Generated context sections that candidates may add to the model context graph.",
            )
        )
    if surface.response.draft_response_interception_allowed:
        response_ops = ["validate", "validate_claims"]
        if surface.response.response_rewrite_allowed:
            response_ops.append("rewrite_response")
        if surface.response.response_blocking_allowed:
            response_ops.append("block_response")
        targets.append(
            SurfaceTarget(
                name="draft_response",
                kind="response",
                path="response.draft",
                current_value=None,
                allowed_ops=tuple(response_ops),
                description="Draft response before it is returned to the user or evaluator.",
            )
        )
    if surface.state.supports_persistent_state:
        state_ops = ["define_state"]
        if surface.state.add_fields_allowed:
            state_ops.extend(["set_state", "append_state", "merge_state", "clear_state", "expose_state"])
        targets.append(
            SurfaceTarget(
                name="state",
                kind="state",
                path="state",
                current_value={"existing_fields": list(surface.state.existing_fields)},
                allowed_ops=tuple(state_ops),
                description="Typed per-task transform state.",
            )
        )
    model_ops = []
    if (
        surface.model.model_name_configurable
        or surface.model.temperature_configurable
        or surface.model.max_tokens_configurable
        or surface.model.reasoning_effort_configurable
        or surface.model.tool_choice_mode_configurable
    ):
        model_ops.append("set_model_config")
    if surface.model.auxiliary_model_calls_allowed:
        model_ops.append("call_model")
    if model_ops:
        targets.append(
            SurfaceTarget(
                name="model_config",
                kind="model",
                path="model",
                current_value=surface.model.to_dict(),
                allowed_ops=tuple(model_ops),
                description="Model invocation configuration exposed by the adapter.",
                choices=tuple(surface.model.model_options),
                value_schema=_model_config_value_schema(surface.model),
            )
        )
    tool_ops = []
    if surface.tools.tool_metadata_allowed:
        tool_ops.append("annotate_tool")
    if surface.tools.tool_description_rewrite_allowed:
        tool_ops.append("rewrite_tool_description")
    if surface.tools.tool_call_interception_allowed:
        tool_ops.extend(["normalize_tool_args", "repair_tool_args", "validate", "replan"])
        targets.append(
            SurfaceTarget(
                name="tool_loop",
                kind="tool",
                path="tools.*",
                current_value={
                    "tools_available": surface.tools.tools_available,
                    "tool_call_interception_allowed": True,
                    "tool_result_rewrite_allowed": surface.tools.tool_result_rewrite_allowed,
                },
                allowed_ops=tuple(tool_ops),
                description=(
                    "Generic tool-call middleware for validating, normalizing, blocking, or replanning "
                    "proposed tool calls before environment execution. This surface must not modify true "
                    "tool implementations or fabricate tool results."
                ),
                value_schema={
                    "hooks": ["before_model_call", "before_tool_call", "after_tool_result"],
                    "available_refs": ["tools", "tool_call", "tool_schema", "tool_metadata", "message_history", "state"],
                    "safe_patterns": [
                        "rewrite tool descriptions at before_model_call without changing execution semantics",
                        "validate tool_call with args_schema_valid then replan on failure",
                        "validate tool_call with not_duplicate_tool_call then replan on failure",
                        "normalize tool_call args before execution",
                        "define state on_task_start and append real tool_result observations after_tool_result",
                    ],
                    "forbidden": [
                        "rewrite_tool_result",
                        "modify tool implementation",
                        "read hidden evaluator or expected answers",
                    ],
                },
            )
        )
    for tool in surface.tools.tools:
        if tool_ops:
            targets.append(
                SurfaceTarget(
                    name=tool.name,
                    kind="tool",
                    path=f"tools.{tool.name}",
                    current_value=tool.to_dict(),
                    allowed_ops=tuple(tool_ops),
                    description=tool.description or f"Tool {tool.name}.",
                )
            )
    return targets


def _context_allowed_ops(surface: SurfaceSpec) -> list[str]:
    ops = ["add_context_section", "replace_context_section"]
    if surface.context.removable_sections_allowed:
        ops.append("remove_context_section")
    if surface.context.reorderable_sections_allowed:
        ops.extend(["move_context_section", "reorder_context_sections"])
    if surface.state.expose_to_context:
        ops.append("render_state_section")
    return ops


def _model_config_value_schema(model: ModelSurface) -> dict[str, Any]:
    fields = []
    if model.model_name_configurable:
        fields.append("model_name")
    if model.temperature_configurable:
        fields.append("temperature")
    if model.max_tokens_configurable:
        fields.append("max_tokens")
    if model.reasoning_effort_configurable:
        fields.append("reasoning_effort")
    if model.tool_choice_mode_configurable:
        fields.append("tool_choice_mode")
    return {
        "operation": "set_model_config",
        "fields": fields,
        "current_model": model.current_model,
        "model_name": {
            "allowed_values": list(model.model_options),
        },
    }


def unsupported_hooks() -> dict[str, HookSurface]:
    return {
        name: HookSurface(name=name, supported=False, method="unsupported")
        for name in SUPPORTED_HOOKS
    }


def surface_from_agent_spec(spec: AgentSpec) -> SurfaceSpec:
    sections = [
        ContextSection(name=name, role="system", content=text, required=True)
        for name, text in spec.instructions.items()
    ]
    if spec.output_contract:
        sections.append(
            ContextSection(
                name="output_contract",
                role="system",
                content=spec.output_contract,
                required=True,
            )
        )
    hooks = unsupported_hooks()
    for name, inputs, ops in [
        (
            "on_task_start",
            ("case", "state", "context"),
            ("define_state", "set_state", "log_event", "trace_annotation"),
        ),
        (
            "before_model_call",
            ("case", "state", "context", "model_config"),
            (
                "add_context_section",
                "remove_context_section",
                "replace_context_section",
                "move_context_section",
                "reorder_context_sections",
                "render_state_section",
                "set_model_config",
                "log_event",
                "trace_annotation",
            ),
        ),
        (
            "after_model_call",
            ("state", "raw_response", "message_history"),
            ("set_state", "append_state", "log_event", "trace_annotation"),
        ),
        (
            "before_user_response",
            ("state", "draft_response", "message_history"),
            (
                "validate",
                "validate_claims",
                "rewrite_response",
                "block_response",
                "log_event",
                "trace_annotation",
            ),
        ),
        (
            "on_task_end",
            ("state", "output", "trace"),
            ("log_event", "trace_annotation"),
        ),
    ]:
        hooks[name] = HookSurface(
            name=name,
            supported=True,
            available_inputs=inputs,
            allowed_outputs=("continue", "modified_context", "modified_response", "modified_model_config"),
            allowed_ops=ops,
        )
    return SurfaceSpec(
        agent_id=spec.name,
        context=ContextSurface(
            graph=ContextGraph(tuple(sections)),
            editable_sections=tuple(section.name for section in sections),
            generated_sections_allowed=True,
            removable_sections_allowed=True,
            reorderable_sections_allowed=True,
        ),
        hooks=hooks,
        state=StateSurface(existing_fields=("messages",)),
        tools=ToolSurface(
            tools=tuple(
                ToolSpec(name=name, description=tool.description, metadata=dict(tool.metadata))
                for name, tool in sorted(spec.tools.items())
            ),
            tool_description_rewrite_allowed=True,
            tool_metadata_allowed=True,
        ),
        model=ModelSurface(current_model=spec.model, model_options=tuple(spec.model_options)),
        response=ResponseSurface(),
        immutable_boundaries=(
            "evaluator",
            "hidden_task_labels",
            "gold_answers",
            "environment_state_except_via_tools",
            "true_tool_semantics",
        ),
        safety_constraints=(
            "no hidden label leakage",
            "no evaluator modification",
            "no tool-result fabrication",
            "no benchmark-specific task-id overfitting",
        ),
        metadata={"source": "agent_spec"},
    )


def tool_loop_surface_from_agent_spec(spec: AgentSpec) -> SurfaceSpec:
    surface = surface_from_agent_spec(spec)
    hooks = unsupported_hooks()
    hook_specs = [
        (
            "on_task_start",
            ("case", "state", "context"),
            ("define_state", "set_state", "log_event", "trace_annotation"),
            ("continue",),
        ),
        (
            "after_user_message",
            ("case", "state", "message_history"),
            ("set_state", "append_state", "log_event", "trace_annotation"),
            ("continue",),
        ),
        (
            "before_model_call",
            ("case", "state", "context", "model_config", "message_history", "tools"),
            (
                "add_context_section",
                "remove_context_section",
                "replace_context_section",
                "move_context_section",
                "reorder_context_sections",
                "render_state_section",
                "rewrite_tool_description",
                "set_model_config",
                "log_event",
                "trace_annotation",
            ),
            ("continue", "modified_context", "modified_model_config", "modified_tool_presentation"),
        ),
        (
            "after_model_call",
            ("state", "raw_response", "message_history"),
            ("set_state", "append_state", "log_event", "trace_annotation"),
            ("continue",),
        ),
        (
            "before_tool_call",
            ("state", "tool_call", "tool_schema", "tool_metadata", "message_history"),
            (
                "validate",
                "normalize_tool_args",
                "block",
                "allow",
                "replan",
                "log_event",
                "trace_annotation",
            ),
            ("allow", "block", "modified_tool_call", "replan_instruction"),
        ),
        (
            "after_tool_result",
            ("state", "tool_call", "tool_result", "message_history"),
            ("set_state", "append_state", "merge_state", "log_event", "trace_annotation"),
            ("continue",),
        ),
        (
            "on_tool_error",
            ("state", "tool_call", "tool_error", "message_history"),
            ("set_state", "append_state", "retry", "replan", "log_event", "trace_annotation"),
            ("continue", "retry", "replan_instruction"),
        ),
        (
            "before_user_response",
            ("state", "draft_response", "message_history"),
            (
                "validate",
                "validate_claims",
                "rewrite_response",
                "block_response",
                "log_event",
                "trace_annotation",
            ),
            ("continue", "modified_response", "blocked_response"),
        ),
        (
            "on_task_end",
            ("state", "output", "trace"),
            ("log_event", "trace_annotation"),
            ("continue",),
        ),
    ]
    for name, inputs, ops, outputs in hook_specs:
        hooks[name] = HookSurface(
            name=name,
            supported=True,
            available_inputs=inputs,
            allowed_outputs=outputs,
            allowed_ops=ops,
        )
    return SurfaceSpec(
        agent_id=surface.agent_id,
        context=surface.context,
        hooks=hooks,
        state=surface.state,
        tools=ToolSurface(
            tools=surface.tools.tools,
            tool_schema_rewrite_allowed=False,
            tool_description_rewrite_allowed=True,
            tool_call_interception_allowed=True,
            tool_execution_modification_allowed=False,
            tool_result_rewrite_allowed=False,
            tool_metadata_allowed=True,
        ),
        model=ModelSurface(
            provider=surface.model.provider,
            current_model=surface.model.current_model,
            model_options=surface.model.model_options,
            model_name_configurable=surface.model.model_name_configurable,
            temperature_configurable=True,
            max_tokens_configurable=surface.model.max_tokens_configurable,
            reasoning_effort_configurable=surface.model.reasoning_effort_configurable,
            tool_choice_mode_configurable=True,
            auxiliary_model_calls_allowed=surface.model.auxiliary_model_calls_allowed,
        ),
        response=surface.response,
        immutable_boundaries=surface.immutable_boundaries,
        safety_constraints=surface.safety_constraints,
        metadata={"source": "tool_loop_agent_spec"},
    )
