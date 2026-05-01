from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.capabilities import validation_checks_for_hook
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
    validation_checks: tuple[dict[str, Any], ...] = ()
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
        row["validation_checks"] = [dict(item) for item in self.validation_checks]
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HookSurface":
        return cls(
            name=str(payload["name"]),
            supported=bool(payload.get("supported", False)),
            available_inputs=tuple(str(item) for item in payload.get("available_inputs", [])),
            allowed_outputs=tuple(str(item) for item in payload.get("allowed_outputs", [])),
            allowed_ops=tuple(str(item) for item in payload.get("allowed_ops", [])),
            validation_checks=tuple(dict(item) for item in payload.get("validation_checks", []) if isinstance(item, dict)),
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
    result_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolSpec":
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            schema=dict(payload.get("schema", {})),
            result_schema=dict(payload.get("result_schema", {})),
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
    affordances: tuple[dict[str, Any], ...] = ()
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
            "affordances": [dict(item) for item in self.affordances],
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
                value_schema={
                    "validation_checks": [
                        spec.to_dict() for spec in validation_checks_for_hook("before_user_response")
                    ],
                    "safe_patterns": [
                        "validate draft response with completion_claims_supported then rewrite_response on failure",
                        "validate structured output with json_object, actions_array, or required_output_keys",
                    ],
                },
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
        for hook_name, hook_description, hook_ops in [
            (
                "before_tool_call",
                "Validate, normalize, block, or replan proposed tool calls before environment execution.",
                ("validate", "normalize_tool_args", "block", "allow", "replan"),
            ),
            (
                "after_tool_result",
                "Update state from real tool observations after environment execution.",
                ("set_state", "append_state", "merge_state"),
            ),
            (
                "on_tool_error",
                "Route real tool errors into bounded retry or replan behavior.",
                ("set_state", "append_state", "retry", "replan"),
            ),
        ]:
            if surface.hooks[hook_name].supported:
                targets.append(
                    SurfaceTarget(
                        name=hook_name,
                        kind="tool",
                        path=f"hooks.{hook_name}",
                        current_value=surface.hooks[hook_name].to_dict(),
                        allowed_ops=hook_ops,
                        description=hook_description,
                        value_schema={
                            "validation_checks": [
                                spec.to_dict() for spec in validation_checks_for_hook(hook_name)
                            ],
                        },
                    )
                )
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
                        "validate tool_call with {\"type\":\"args_schema_valid\"} then replan on failure",
                        "validate tool_call with {\"type\":\"not_duplicate_tool_call\"} then replan on failure",
                        "validate mutating tool_call with {\"type\":\"mutating_tool_requires_confirmation\"} then replan on failure",
                        "validate referenced ids with {\"type\":\"referenced_args_observed\"} then replan on failure",
                        "normalize tool_call args before execution",
                        "define state on_task_start and append real tool_result observations after_tool_result",
                    ],
                    "validation_checks": [
                        spec.to_dict() for spec in validation_checks_for_hook("before_tool_call")
                    ],
                    "forbidden": [
                        "rewrite_tool_result",
                        "modify tool implementation",
                        "read hidden evaluator or expected answers",
                    ],
                },
            )
        )
        for affordance in surface.affordances:
            if affordance.get("kind") != "inspect_before_mutate":
                continue
            identifier = str(affordance.get("identifier") or "")
            if not identifier:
                continue
            targets.append(
                SurfaceTarget(
                    name=f"inspect_before_mutate.{identifier}",
                    kind="tool",
                    path=f"affordances.inspect_before_mutate.{identifier}",
                    current_value=dict(affordance),
                    allowed_ops=("define_state", "append_state", "render_state_section", "validate", "replan"),
                    description=(
                        f"Generic inspect-before-mutate scaffold for {identifier}: collect identifiers from "
                        "read tool results, expose them in state/context, and validate mutating tool calls "
                        "before environment execution."
                    ),
                    value_schema={
                        "affordance": dict(affordance),
                        "safe_patterns": [
                            "define a state list for observed identifiers",
                            "append identifiers from real after_tool_result observations",
                            "render the identifier state before model calls",
                            "validate mutating tool arguments with tool_arg_in_state and replan on failure",
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
            validation_checks=tuple(spec.to_dict() for spec in validation_checks_for_hook(name)),
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


def tool_loop_surface_from_agent_spec(spec: AgentSpec, *, probe: dict[str, Any]) -> SurfaceSpec:
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
            validation_checks=tuple(spec.to_dict() for spec in validation_checks_for_hook(name)),
        )
    context_sections = tuple(_tool_loop_context_sections(spec, probe))
    tool_specs = tuple(_tool_loop_tool_specs(spec, probe))
    return SurfaceSpec(
        agent_id=surface.agent_id,
        context=ContextSurface(
            graph=ContextGraph(context_sections),
            editable_sections=tuple(section.name for section in context_sections),
            generated_sections_allowed=True,
            removable_sections_allowed=True,
            reorderable_sections_allowed=True,
        ),
        hooks=hooks,
        state=surface.state,
        tools=ToolSurface(
            tools=tool_specs,
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
        affordances=tuple(_tool_loop_affordances(tool_specs)),
        metadata={"source": "tool_loop_agent_spec"},
    )


def _tool_loop_context_sections(spec: AgentSpec, probe: dict[str, Any]) -> list[ContextSection]:
    sections: list[ContextSection] = []
    domain_policy = str(probe.get("domain_policy") or "").strip()
    if domain_policy:
        sections.append(
            ContextSection(
                name="domain_policy",
                role="system",
                content=domain_policy,
                required=True,
                metadata={"source": "environment"},
            )
        )
    for name, text in spec.instructions.items():
        if text:
            sections.append(
                ContextSection(
                    name=name,
                    role="system",
                    content=text,
                    required=True,
                    metadata={"source": "agent_spec.instructions"},
                )
            )
    tool_summary = _tool_instruction_text(probe.get("tools") or [])
    if tool_summary:
        sections.append(
            ContextSection(
                name="tool_instructions",
                role="system",
                content=tool_summary,
                required=True,
                metadata={"source": "environment.tools_info"},
            )
        )
    if spec.output_contract:
        sections.append(
            ContextSection(
                name="output_contract",
                role="system",
                content=spec.output_contract,
                required=True,
                metadata={"source": "agent_spec.output_contract"},
            )
        )
    sections.append(
        ContextSection(
            name="recent_messages",
            role="system",
            content="Runtime conversation history supplied as messages after the system prompt.",
            required=True,
            metadata={"dynamic": True, "source": "runtime.message_history"},
        )
    )
    return sections


def _tool_loop_tool_specs(spec: AgentSpec, probe: dict[str, Any]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    static_tools = dict(spec.tools)
    result_schemas = probe.get("tool_result_schemas") if isinstance(probe.get("tool_result_schemas"), dict) else {}
    for raw_tool in probe.get("tools") or []:
        if not isinstance(raw_tool, dict):
            continue
        function = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        static_tool = static_tools.get(name)
        description = str(function.get("description") or (static_tool.description if static_tool is not None else ""))
        schema = dict(function.get("parameters") or function.get("schema") or {})
        result_schema = _tool_result_schema(name, function, result_schemas)
        metadata = _inferred_tool_metadata(name, description, schema)
        if static_tool is not None:
            metadata.update(static_tool.metadata)
        result_paths = _schema_leaf_paths(result_schema)
        if result_paths:
            metadata["result_paths"] = result_paths
        specs.append(
            ToolSpec(
                name=name,
                description=description,
                schema=schema,
                result_schema=result_schema,
                metadata=metadata,
            )
        )
    return sorted(specs, key=lambda item: item.name)


def _tool_result_schema(name: str, function: dict[str, Any], result_schemas: Any) -> dict[str, Any]:
    for key in ("result_schema", "returns", "output_schema"):
        raw_schema = function.get(key)
        if isinstance(raw_schema, dict):
            return dict(raw_schema)
    if isinstance(result_schemas, dict):
        raw_schema = result_schemas.get(name)
        if isinstance(raw_schema, dict):
            return dict(raw_schema)
    return {}


def _tool_loop_affordances(tools: tuple[ToolSpec, ...]) -> list[dict[str, Any]]:
    flows = _identifier_flows(tools)
    affordances: list[dict[str, Any]] = []
    for identifier, flow in sorted(flows.items()):
        producers = [
            producer
            for producer in flow["produced_by"]
            if producer.get("side_effect") in {"read", "internal"}
        ]
        mutating_consumers = [
            consumer
            for consumer in flow["consumed_by"]
            if consumer.get("side_effect") in {"mutating", "destructive"}
        ]
        if not producers or not mutating_consumers:
            continue
        affordances.append(
            {
                "kind": "inspect_before_mutate",
                "identifier": identifier,
                "state_field": f"observed_{identifier.removesuffix('_id')}_ids",
                "produced_by": producers,
                "consumed_by": mutating_consumers,
                "required_surfaces": [
                    "after_tool_result",
                    "before_tool_call",
                    "surface_state",
                    "surface_context",
                ],
            }
        )
    return affordances


def _identifier_flows(tools: tuple[ToolSpec, ...]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    flows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for tool in tools:
        side_effect = str(tool.metadata.get("side_effect") or "unknown")
        for path in _schema_identifier_paths(tool.result_schema):
            identifier = path.rsplit(".", 1)[-1].replace("[]", "")
            flow = flows.setdefault(identifier, {"produced_by": [], "consumed_by": []})
            flow["produced_by"].append(
                {
                    "tool": tool.name,
                    "path": path,
                    "ref": f"tool_result.parsed.{path}",
                    "side_effect": side_effect,
                }
            )
        for arg in _schema_identifier_args(tool.schema):
            flow = flows.setdefault(arg, {"produced_by": [], "consumed_by": []})
            flow["consumed_by"].append(
                {
                    "tool": tool.name,
                    "arg": arg,
                    "side_effect": side_effect,
                }
            )
    return flows


def _schema_identifier_args(schema: dict[str, Any]) -> list[str]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    return sorted(
        str(name)
        for name, subschema in properties.items()
        if _is_identifier_field(str(name), subschema)
    )


def _schema_identifier_paths(schema: dict[str, Any]) -> list[str]:
    return [
        path
        for path, subschema in _schema_leaf_path_items(schema)
        if _is_identifier_field(path.rsplit(".", 1)[-1].replace("[]", ""), subschema)
    ]


def _schema_leaf_paths(schema: dict[str, Any]) -> list[str]:
    return [path for path, _subschema in _schema_leaf_path_items(schema)]


def _schema_leaf_path_items(schema: dict[str, Any], prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(schema, dict) or not schema:
        return []
    schema_type = schema.get("type")
    if schema_type == "object" or isinstance(schema.get("properties"), dict):
        rows: list[tuple[str, dict[str, Any]]] = []
        for name, subschema in sorted(schema.get("properties", {}).items()):
            if not isinstance(subschema, dict):
                continue
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            child_rows = _schema_leaf_path_items(subschema, child_prefix)
            rows.extend(child_rows or [(child_prefix, subschema)])
        return rows
    if schema_type == "array" or isinstance(schema.get("items"), dict):
        item_schema = schema.get("items") if isinstance(schema.get("items"), dict) else {}
        array_prefix = f"{prefix}[]"
        child_rows = _schema_leaf_path_items(item_schema, array_prefix)
        return child_rows or [(array_prefix, schema)]
    return [(prefix, schema)] if prefix else []


def _is_identifier_field(name: str, schema: Any) -> bool:
    if not name.endswith("_id"):
        return False
    if not isinstance(schema, dict):
        return True
    schema_type = schema.get("type")
    return schema_type in {None, "string", "integer", "number"}


def _tool_instruction_text(raw_tools: list[Any]) -> str:
    rows: list[str] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue
        function = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        description = str(function.get("description") or "").strip()
        rows.append(f"- {name}: {description}" if description else f"- {name}")
    return "\n".join(rows)


def _inferred_tool_metadata(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    name_tokens = _identifier_tokens(name)
    description_head = description.split(".", 1)[0]
    description_tokens = _identifier_tokens(description_head)
    action_token = name_tokens[0] if name_tokens else ""
    mutating_tokens = {
        "create",
        "update",
        "delete",
        "cancel",
        "change",
        "modify",
        "submit",
        "send",
        "book",
        "refund",
        "return",
        "transfer",
        "purchase",
        "pay",
        "exchange",
    }
    destructive_tokens = {"delete", "cancel", "remove", "close"}
    read_tokens = {"get", "list", "search", "find", "lookup", "check", "retrieve", "view", "calculate"}
    internal_tokens = {"think", "reason", "log"}
    if action_token in destructive_tokens:
        side_effect = "destructive"
        risk = "high"
    elif action_token in read_tokens:
        side_effect = "read"
        risk = "low"
    elif action_token in internal_tokens:
        side_effect = "internal"
        risk = "low"
    elif action_token in mutating_tokens:
        side_effect = "mutating"
        risk = "medium"
    elif set(description_tokens) & destructive_tokens:
        side_effect = "destructive"
        risk = "high"
    elif set(description_tokens) & read_tokens:
        side_effect = "read"
        risk = "low"
    elif set(description_tokens) & mutating_tokens:
        side_effect = "mutating"
        risk = "medium"
    else:
        side_effect = "unknown"
        risk = "unknown"
    return {
        "side_effect": side_effect,
        "risk": risk,
        "schema_keys": sorted(str(key) for key in schema.keys()),
        "source": "surface_probe",
    }


def _identifier_tokens(value: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in value:
        if char.isalnum():
            current.append(char.lower())
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens
