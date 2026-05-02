from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


CONTEXT_OPS = {
    "add_context_section",
    "remove_context_section",
    "replace_context_section",
    "move_context_section",
    "reorder_context_sections",
    "render_state_section",
}
STATE_OPS = {"define_state", "set_state", "append_state", "merge_state", "clear_state", "expose_state", "hide_state"}
TOOL_OPS = {
    "annotate_tool",
    "rewrite_tool_description",
    "group_tools",
    "hide_tool",
    "normalize_tool_args",
    "repair_tool_args",
    "select_tool_mode",
}
VALIDATION_OPS = {
    "validate",
    "schema_check",
    "consistency_check",
    "support_check",
    "precondition_check",
    "ambiguity_check",
    "claim_check",
    "validate_claims",
}
CONTROL_OPS = {"allow", "block", "route", "replan", "retry", "fallback", "ask_user", "terminate", "continue"}
MODEL_OPS = {"call_model", "set_model_config"}
RESPONSE_OPS = {"extract_claims", "rewrite_response", "block_response", "add_response_disclosure"}
LIMIT_OPS = {"set_retry_policy", "set_turn_limit", "set_tool_call_limit", "define_completion_criteria", "validate_completion"}
INSTRUMENTATION_OPS = {"log_event", "trace_annotation", "metric_counter", "capture_snapshot"}
TRANSFORM_OPS = (
    CONTEXT_OPS
    | STATE_OPS
    | TOOL_OPS
    | VALIDATION_OPS
    | CONTROL_OPS
    | MODEL_OPS
    | RESPONSE_OPS
    | LIMIT_OPS
    | INSTRUMENTATION_OPS
)


@dataclass(frozen=True)
class TransformOp:
    op: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.op not in TRANSFORM_OPS:
            raise ValueError(f"Unsupported transform op: {self.op}")

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, **dict(self.params)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformOp":
        if not isinstance(payload, dict):
            raise ValueError("transform operation must be an object")
        op = str(payload.get("op") or "")
        if not op:
            raise ValueError("transform operation requires op")
        params = {key: value for key, value in payload.items() if key not in {"op", "hook", "when", "unless"}}
        return cls(op=op, params=params)


@dataclass(frozen=True)
class TransformPatch:
    op: TransformOp
    hook: str | None = None
    when: dict[str, Any] = field(default_factory=dict)
    unless: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = self.op.to_dict()
        if self.hook:
            row["hook"] = self.hook
        if self.when:
            row["when"] = dict(self.when)
        if self.unless:
            row["unless"] = dict(self.unless)
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformPatch":
        if not isinstance(payload, dict):
            raise ValueError("transform candidate must be an object")
        return cls(
            op=TransformOp.from_dict(payload),
            hook=str(payload["hook"]) if payload.get("hook") is not None else None,
            when=dict(payload.get("when", {})),
            unless=dict(payload.get("unless", {})),
        )


@dataclass(frozen=True)
class TransformProgram:
    candidate_id: str
    patches: tuple[TransformPatch, ...]
    hypothesis_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("TransformProgram candidate_id must be non-empty.")
        if not self.patches:
            raise ValueError("TransformProgram requires at least one patch.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "hypothesis_id": self.hypothesis_id,
            "patches": [patch.to_dict() for patch in self.patches],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformProgram":
        if not isinstance(payload, dict):
            raise ValueError("transform program must be an object")
        raw_patches = payload.get("patches")
        if not isinstance(raw_patches, list) or not raw_patches:
            raise ValueError("transform program requires non-empty patches[]")
        return cls(
            candidate_id=str(payload.get("candidate_id") or payload.get("id") or ""),
            hypothesis_id=str(payload.get("hypothesis_id") or ""),
            patches=tuple(TransformPatch.from_dict(dict(item)) for item in raw_patches if isinstance(item, dict)),
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "TransformProgram":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class CompileIssue:
    code: str
    message: str
    patch_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompileReport:
    candidate_id: str
    status: str
    modified_surfaces: tuple[str, ...] = ()
    added_state_fields: tuple[str, ...] = ()
    added_context_sections: tuple[str, ...] = ()
    added_validators: tuple[str, ...] = ()
    estimated_overhead: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    rejection: CompileIssue | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "modified_surfaces": list(self.modified_surfaces),
            "added_state_fields": list(self.added_state_fields),
            "added_context_sections": list(self.added_context_sections),
            "added_validators": list(self.added_validators),
            "estimated_overhead": dict(self.estimated_overhead),
            "warnings": list(self.warnings),
            "rejection": self.rejection.to_dict() if self.rejection else None,
        }


@dataclass(frozen=True)
class CandidateDiff:
    added_state_fields: tuple[str, ...] = ()
    added_context_sections: tuple[str, ...] = ()
    context_changes: tuple[str, ...] = ()
    hook_changes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tool_changes: tuple[str, ...] = ()
    immutable_boundaries: str = "unchanged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_state_fields": list(self.added_state_fields),
            "added_context_sections": list(self.added_context_sections),
            "context_changes": list(self.context_changes),
            "hook_changes": {key: list(value) for key, value in sorted(self.hook_changes.items())},
            "tool_changes": list(self.tool_changes),
            "immutable_boundaries": self.immutable_boundaries,
        }


@dataclass(frozen=True)
class CompiledCandidate:
    program: TransformProgram
    operations_by_hook: dict[str, tuple[TransformPatch, ...]]
    report: CompileReport
    diff: CandidateDiff

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "operations_by_hook": {
                hook: [patch.to_dict() for patch in patches]
                for hook, patches in sorted(self.operations_by_hook.items())
            },
            "report": self.report.to_dict(),
            "diff": self.diff.to_dict(),
        }


def references_in_value(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, dict):
        raw_ref = value.get("$ref") or value.get("ref")
        if isinstance(raw_ref, str):
            refs.add(raw_ref)
        for item in value.values():
            refs.update(references_in_value(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(references_in_value(item))
    return refs
