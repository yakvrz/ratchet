from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


PATCH_OPS = {
    "add_context_section",
    "add_instruction",
    "append_state",
    "block_response",
    "clear_state",
    "define_state",
    "expose_state",
    "hide_state",
    "extract_claims",
    "log_event",
    "merge_state",
    "move_context_section",
    "normalize_tool_args",
    "annotate_tool",
    "remove_context_section",
    "render_state_section",
    "reorder_context_sections",
    "replace_context_section",
    "repair_tool_args",
    "rewrite_response",
    "rewrite_tool_description",
    "revise_instruction",
    "add_output_constraint",
    "revise_tool_description",
    "revise_tool_policy",
    "set_model_config",
    "set_runtime_param",
    "set_state",
    "trace_annotation",
    "validate",
    "validate_claims",
    "change_model",
    "add_few_shot",
    "add_verifier_retry",
}

EDIT_KINDS = {
    "context",
    "instruction",
    "output",
    "response",
    "state",
    "tool",
    "runtime",
    "model",
    "few_shot",
    "verifier",
}


@dataclass(frozen=True)
class TargetSemantics:
    role: str = "generic_policy"
    axes: list[str] = field(default_factory=list)
    scope: str = "global"
    risks: list[str] = field(default_factory=list)
    measurement_hints: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "default"

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("TargetSemantics role must be non-empty.")
        if self.scope not in {"local", "slice", "global"}:
            raise ValueError(f"Unsupported TargetSemantics scope: {self.scope}")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("TargetSemantics confidence must be between 0.0 and 1.0.")
        object.__setattr__(self, "axes", [str(item) for item in self.axes])
        object.__setattr__(self, "risks", [str(item) for item in self.risks])
        object.__setattr__(self, "measurement_hints", [str(item) for item in self.measurement_hints])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | "TargetSemantics" | None) -> "TargetSemantics":
        if payload is None:
            return cls()
        if isinstance(payload, TargetSemantics):
            return payload
        return cls(
            role=str(payload.get("role", "generic_policy")),
            axes=[str(item) for item in payload.get("axes", [])],
            scope=str(payload.get("scope", "global")),
            risks=[str(item) for item in payload.get("risks", [])],
            measurement_hints=[str(item) for item in payload.get("measurement_hints", [])],
            confidence=float(payload.get("confidence", 0.0)),
            source=str(payload.get("source", "default")),
        )


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str = ""
    policy: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentTool name must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTool":
        return cls(
            name=str(payload["name"]),
            description=str(payload.get("description", "")),
            policy=str(payload.get("policy", "")),
            enabled=bool(payload.get("enabled", True)),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class AgentSpec:
    name: str
    model: str
    instructions: dict[str, str] = field(default_factory=dict)
    tools: dict[str, AgentTool] = field(default_factory=dict)
    output_contract: str = ""
    runtime: dict[str, Any] = field(default_factory=dict)
    model_options: list[str] = field(default_factory=list)
    few_shot: list[dict[str, Any]] = field(default_factory=list)
    target_semantics: dict[str, TargetSemantics] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("AgentSpec name must be non-empty.")
        if not self.model:
            raise ValueError("AgentSpec model must be non-empty.")
        if self.model_options and self.model not in self.model_options:
            raise ValueError(f"AgentSpec model {self.model!r} is not in model_options.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "instructions": dict(self.instructions),
            "tools": {name: tool.to_dict() for name, tool in sorted(self.tools.items())},
            "output_contract": self.output_contract,
            "runtime": dict(self.runtime),
            "model_options": list(self.model_options),
            "few_shot": [dict(item) for item in self.few_shot],
            "target_semantics": {
                key: semantics.to_dict() for key, semantics in sorted(self.target_semantics.items())
            },
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentSpec":
        return cls(
            name=str(payload["name"]),
            model=str(payload["model"]),
            instructions={str(key): str(value) for key, value in payload.get("instructions", {}).items()},
            tools={
                str(name): AgentTool.from_dict({**dict(tool), "name": str(name)})
                if isinstance(tool, dict) and "name" not in tool
                else AgentTool.from_dict(dict(tool))
                for name, tool in payload.get("tools", {}).items()
            },
            output_contract=str(payload.get("output_contract", "")),
            runtime=dict(payload.get("runtime", {})),
            model_options=[str(item) for item in payload.get("model_options", [])],
            few_shot=[dict(item) for item in payload.get("few_shot", [])],
            target_semantics={
                str(key): TargetSemantics.from_dict(value)
                for key, value in payload.get("target_semantics", {}).items()
            },
            metadata=dict(payload.get("metadata", {})),
        )

    def apply_patch(self, patch: "AgentPatch | None") -> "AgentSpec":
        if patch is None or not patch.operations:
            return self
        spec = self.to_dict()
        for operation in patch.operations:
            _apply_operation(spec, operation)
        return AgentSpec.from_dict(spec)


@dataclass(frozen=True)
class EditableTarget:
    name: str
    kind: str
    path: str
    current_value: Any
    allowed_ops: list[str]
    description: str = ""
    choices: list[str] = field(default_factory=list)
    max_chars: int | None = None
    value_schema: dict[str, Any] = field(default_factory=dict)
    semantics: TargetSemantics = field(default_factory=TargetSemantics)

    def __post_init__(self) -> None:
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"Unsupported editable target kind: {self.kind}")
        if not self.name or not self.path:
            raise ValueError("EditableTarget name and path must be non-empty.")
        unsupported = sorted(set(self.allowed_ops) - PATCH_OPS)
        if unsupported:
            raise ValueError(f"Unsupported editable target ops: {', '.join(unsupported)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EditableTarget":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            path=str(payload["path"]),
            current_value=payload.get("current_value"),
            allowed_ops=[str(item) for item in payload.get("allowed_ops", [])],
            description=str(payload.get("description", "")),
            choices=[str(item) for item in payload.get("choices", [])],
            max_chars=int(payload["max_chars"]) if payload.get("max_chars") is not None else None,
            value_schema=dict(payload.get("value_schema", {})),
            semantics=TargetSemantics.from_dict(payload.get("semantics")),
        )


@dataclass(frozen=True)
class PatchOperation:
    op: str
    target: str
    value: Any
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.op not in PATCH_OPS:
            raise ValueError(f"Unsupported patch operation: {self.op}")
        if not self.target:
            raise ValueError("PatchOperation target must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchOperation":
        return cls(
            op=str(payload["op"]),
            target=str(payload["target"]),
            value=payload.get("value"),
            rationale=str(payload.get("rationale", "")),
        )


@dataclass(frozen=True)
class AgentPatch:
    operations: list[PatchOperation] = field(default_factory=list)
    rationale: str = ""
    expected_effect: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operations": [operation.to_dict() for operation in self.operations],
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentPatch":
        return cls(
            operations=[PatchOperation.from_dict(item) for item in payload.get("operations", [])],
            rationale=str(payload.get("rationale", "")),
            expected_effect=str(payload.get("expected_effect", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def empty(cls) -> "AgentPatch":
        return cls()

    @property
    def is_empty(self) -> bool:
        return not self.operations


@dataclass(frozen=True)
class OptimizationConstraints:
    allowed_edits: list[str] = field(default_factory=lambda: sorted(EDIT_KINDS))
    allowed_models: list[str] = field(default_factory=list)
    max_cost_ratio: float | None = None
    max_latency_ratio: float | None = None
    min_correctness_delta: float | None = None
    max_patch_operations: int = 2
    sanitize_examples: bool = False

    def __post_init__(self) -> None:
        unsupported = sorted(set(self.allowed_edits) - EDIT_KINDS)
        if unsupported:
            raise ValueError(f"Unsupported allowed edit kinds: {', '.join(unsupported)}")
        if self.max_patch_operations <= 0:
            raise ValueError("max_patch_operations must be positive.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "OptimizationConstraints":
        payload = dict(payload or {})
        return cls(
            allowed_edits=[str(item) for item in payload.get("allowed_edits", sorted(EDIT_KINDS))],
            allowed_models=[str(item) for item in payload.get("allowed_models", [])],
            max_cost_ratio=(
                float(payload["max_cost_ratio"]) if payload.get("max_cost_ratio") is not None else None
            ),
            max_latency_ratio=(
                float(payload["max_latency_ratio"]) if payload.get("max_latency_ratio") is not None else None
            ),
            min_correctness_delta=(
                float(payload["min_correctness_delta"])
                if payload.get("min_correctness_delta") is not None
                else None
            ),
            max_patch_operations=int(payload.get("max_patch_operations", 2)),
            sanitize_examples=bool(payload.get("sanitize_examples", False)),
        )


@dataclass(frozen=True)
class OptimizationObjective:
    mode: str = "correctness"
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
    tie_breakers: list[str] = field(default_factory=lambda: ["lower_cost", "lower_latency", "smaller_patch"])

    def __post_init__(self) -> None:
        if self.mode not in {"correctness", "cost", "latency"}:
            raise ValueError(f"Unsupported optimization mode: {self.mode}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "constraints": self.constraints.to_dict(),
            "tie_breakers": list(self.tie_breakers),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "OptimizationObjective":
        payload = dict(payload or {})
        return cls(
            mode=str(payload.get("mode", "correctness")),
            constraints=OptimizationConstraints.from_dict(payload.get("constraints")),
            tie_breakers=[str(item) for item in payload.get("tie_breakers", ["lower_cost", "lower_latency", "smaller_patch"])],
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    split: str
    input: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "holdout"}:
            raise ValueError(f"Unsupported split: {self.split}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(payload["id"]),
            split=str(payload["split"]),
            input=str(payload["input"]),
            expected=payload.get("expected"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class OperationalMetrics:
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model_calls: int = 1
    tool_calls: int = 0
    turns: int = 1
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperationalMetrics":
        return cls(
            latency_s=float(payload.get("latency_s", 0.0)),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            model_calls=int(payload.get("model_calls", 1)),
            tool_calls=int(payload.get("tool_calls", 0)),
            turns=int(payload.get("turns", 1)),
            error=payload.get("error"),
        )


@dataclass(frozen=True)
class ToolCallTrace:
    name: str
    arguments: Any = None
    result: Any = None
    status: str = "ok"
    latency_s: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ToolCallTrace name must be non-empty.")
        if self.status not in {"ok", "error", "invalid", "skipped"}:
            raise ValueError(f"Unsupported ToolCallTrace status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | "ToolCallTrace") -> "ToolCallTrace":
        if isinstance(payload, ToolCallTrace):
            return payload
        return cls(
            name=str(payload["name"]),
            arguments=payload.get("arguments"),
            result=payload.get("result"),
            status=str(payload.get("status", "ok")),
            latency_s=float(payload["latency_s"]) if payload.get("latency_s") is not None else None,
            error=payload.get("error"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class InteractionTurn:
    index: int
    actor: str
    message: Any = None
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    outcome: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("InteractionTurn index must be non-negative.")
        if not self.actor:
            raise ValueError("InteractionTurn actor must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "actor": self.actor,
            "message": self.message,
            "tool_calls": [tool_call.to_dict() for tool_call in self.tool_calls],
            "outcome": self.outcome,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | "InteractionTurn") -> "InteractionTurn":
        if isinstance(payload, InteractionTurn):
            return payload
        return cls(
            index=int(payload.get("index", 0)),
            actor=str(payload.get("actor", "agent")),
            message=payload.get("message"),
            tool_calls=[
                ToolCallTrace.from_dict(dict(item))
                for item in payload.get("tool_calls", [])
                if isinstance(item, dict)
            ],
            outcome=str(payload.get("outcome", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class DiagnosticTrace:
    tool_calls: list[str] = field(default_factory=list)
    raw_output_text: str = ""
    turns: list[InteractionTurn] = field(default_factory=list)
    terminal_state: dict[str, Any] = field(default_factory=dict)
    terminal_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        tool_names = list(self.tool_calls)
        for turn in self.turns:
            tool_names.extend(tool_call.name for tool_call in turn.tool_calls)
        object.__setattr__(self, "tool_calls", list(dict.fromkeys(str(name) for name in tool_names if name)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_calls": list(self.tool_calls),
            "raw_output_text": self.raw_output_text,
            "turns": [turn.to_dict() for turn in self.turns],
            "terminal_state": dict(self.terminal_state),
            "terminal_reason": self.terminal_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosticTrace":
        return cls(
            tool_calls=[str(item) for item in payload.get("tool_calls", [])],
            raw_output_text=str(payload.get("raw_output_text", "")),
            turns=[
                InteractionTurn.from_dict(dict(item))
                for item in payload.get("turns", [])
                if isinstance(item, dict)
            ],
            terminal_state=dict(payload.get("terminal_state", {})),
            terminal_reason=str(payload.get("terminal_reason", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class RunRecord:
    output: Any
    metrics: OperationalMetrics
    diagnostics: DiagnosticTrace = field(default_factory=DiagnosticTrace)

    def __post_init__(self) -> None:
        derived_tool_calls = sum(len(turn.tool_calls) for turn in self.diagnostics.turns)
        if not derived_tool_calls:
            derived_tool_calls = len(self.diagnostics.tool_calls)
        derived_turns = max(1, len(self.diagnostics.turns))
        if (
            self.metrics.tool_calls == 0
            and derived_tool_calls > 0
        ) or (
            self.metrics.turns == 1
            and derived_turns > 1
        ):
            object.__setattr__(
                self,
                "metrics",
                replace(
                    self.metrics,
                    tool_calls=derived_tool_calls if self.metrics.tool_calls == 0 else self.metrics.tool_calls,
                    turns=derived_turns if self.metrics.turns == 1 else self.metrics.turns,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "metrics": self.metrics.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        return cls(
            output=payload.get("output"),
            metrics=OperationalMetrics.from_dict(payload.get("metrics", {})),
            diagnostics=DiagnosticTrace.from_dict(payload.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class GradeResult:
    score: float
    passed: bool
    labels: list[str] = field(default_factory=list)
    notes: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Grade score must be between 0 and 1, got {self.score}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GradeResult":
        return cls(
            score=float(payload["score"]),
            passed=bool(payload["passed"]),
            labels=[str(label) for label in payload.get("labels", [])],
            notes=payload.get("notes"),
        )


@dataclass(frozen=True)
class FailureDiagnosis:
    case_ids: list[str]
    category: str
    root_cause: str
    target_names: list[str]
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureDiagnosis":
        return cls(
            case_ids=[str(item) for item in payload.get("case_ids", [])],
            category=str(payload["category"]),
            root_cause=str(payload.get("root_cause", "")),
            target_names=[str(item) for item in payload.get("target_names", payload.get("target_keys", []))],
            evidence=[dict(item) for item in payload.get("evidence", [])],
        )


def _apply_operation(spec: dict[str, Any], operation: PatchOperation) -> None:
    target = operation.target
    value = operation.value
    if operation.op == "change_model":
        model = str(value)
        options = [str(item) for item in spec.get("model_options", [])]
        if options and model not in options:
            raise ValueError(f"Model {model!r} is not in AgentSpec model_options.")
        spec["model"] = model
        return

    if operation.op in {"add_instruction", "revise_instruction"}:
        section = _strip_prefix(target, "instructions.")
        instructions = dict(spec.get("instructions", {}))
        current = str(instructions.get(section, ""))
        if operation.op == "add_instruction" and current:
            text = str(value).strip()
            instructions[section] = f"{current.rstrip()}\n\n{text}" if text not in current else current
        else:
            instructions[section] = str(value)
        spec["instructions"] = instructions
        return

    if operation.op == "add_output_constraint":
        current = str(spec.get("output_contract", ""))
        text = str(value).strip()
        spec["output_contract"] = f"{current.rstrip()}\n\n{text}" if current and text not in current else text or current
        return

    if operation.op in {"revise_tool_description", "revise_tool_policy"}:
        parts = target.split(".")
        if len(parts) < 3 or parts[0] != "tools":
            raise ValueError(f"Invalid tool target: {target}")
        tool_name = parts[1]
        field_name = "description" if operation.op == "revise_tool_description" else "policy"
        tools = {name: dict(tool) for name, tool in spec.get("tools", {}).items()}
        if tool_name not in tools:
            raise ValueError(f"Unknown tool target: {tool_name}")
        tools[tool_name][field_name] = str(value)
        spec["tools"] = tools
        return

    if operation.op == "set_runtime_param":
        if target.startswith("tools.") and target.endswith(".enabled"):
            parts = target.split(".")
            if len(parts) != 3:
                raise ValueError(f"Invalid tool enabled target: {target}")
            tool_name = parts[1]
            tools = {name: dict(tool) for name, tool in spec.get("tools", {}).items()}
            if tool_name not in tools:
                raise ValueError(f"Unknown tool target: {tool_name}")
            tools[tool_name]["enabled"] = bool(value)
            spec["tools"] = tools
            return
        key = _strip_prefix(target, "runtime.")
        runtime = dict(spec.get("runtime", {}))
        runtime[key] = value
        spec["runtime"] = runtime
        return

    if operation.op == "add_few_shot":
        few_shot = [dict(item) for item in spec.get("few_shot", [])]
        if isinstance(value, list):
            few_shot.extend(dict(item) if isinstance(item, dict) else {"text": str(item)} for item in value)
        else:
            few_shot.append(dict(value) if isinstance(value, dict) else {"text": str(value)})
        spec["few_shot"] = few_shot
        return

    if operation.op == "add_verifier_retry":
        runtime = dict(spec.get("runtime", {}))
        runtime["verifier_retry"] = value if value is not None else True
        spec["runtime"] = runtime
        return

    raise ValueError(f"Unsupported patch operation: {operation.op}")


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value
