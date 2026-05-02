from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any


@dataclass(frozen=True)
class ValidationCheckSpec:
    name: str
    hooks: tuple[str, ...]
    required_inputs: tuple[str, ...]
    description: str
    parameters_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.name,
            "hooks": list(self.hooks),
            "required_inputs": list(self.required_inputs),
            "description": self.description,
            "parameters_schema": dict(self.parameters_schema),
        }


VALIDATION_CHECKS: dict[str, ValidationCheckSpec] = {
    "json_object": ValidationCheckSpec(
        name="json_object",
        hooks=("before_user_response",),
        required_inputs=("draft_response",),
        description="Draft response must be a JSON object.",
        parameters_schema={},
    ),
    "actions_array": ValidationCheckSpec(
        name="actions_array",
        hooks=("before_user_response",),
        required_inputs=("draft_response",),
        description="Draft JSON response must contain an actions array.",
        parameters_schema={},
    ),
    "required_output_keys": ValidationCheckSpec(
        name="required_output_keys",
        hooks=("before_user_response",),
        required_inputs=("draft_response",),
        description="Draft JSON response must contain the listed top-level keys.",
        parameters_schema={"required": ["keys"], "properties": {"keys": {"type": "array", "items": {"type": "string"}}}},
    ),
    "args_schema_valid": ValidationCheckSpec(
        name="args_schema_valid",
        hooks=("before_tool_call",),
        required_inputs=("tool_call", "tool_schema"),
        description="Tool-call arguments must satisfy the tool JSON schema exposed by the environment.",
        parameters_schema={},
    ),
    "not_duplicate_tool_call": ValidationCheckSpec(
        name="not_duplicate_tool_call",
        hooks=("before_tool_call",),
        required_inputs=("tool_call", "message_history"),
        description="The proposed tool call must not repeat an identical prior tool call.",
        parameters_schema={},
    ),
    "mutating_tool_requires_confirmation": ValidationCheckSpec(
        name="mutating_tool_requires_confirmation",
        hooks=("before_tool_call",),
        required_inputs=("tool_call", "tool_metadata", "message_history"),
        description="Mutating or destructive tool calls require an explicit recent user confirmation.",
        parameters_schema={},
    ),
    "referenced_args_observed": ValidationCheckSpec(
        name="referenced_args_observed",
        hooks=("before_tool_call",),
        required_inputs=("tool_call", "message_history"),
        description="Identifier-like string arguments must appear in prior user/tool observations.",
        parameters_schema={},
    ),
    "tool_arg_in_state": ValidationCheckSpec(
        name="tool_arg_in_state",
        hooks=("before_tool_call",),
        required_inputs=("tool_call", "state"),
        description="A named tool-call argument must be present in a configured transform state list.",
        parameters_schema={
            "required": ["state_field", "arg"],
            "properties": {
                "state_field": {"type": "string"},
                "arg": {"type": "string"},
                "state_key": {"type": "string"},
            },
        },
    ),
    "completion_claims_supported": ValidationCheckSpec(
        name="completion_claims_supported",
        hooks=("before_user_response",),
        required_inputs=("draft_response", "state"),
        description="Completion-like final responses require a successful completed action in transform state.",
        parameters_schema={"properties": {"state_field": {"type": "string"}}},
    ),
    "clarification_response": ValidationCheckSpec(
        name="clarification_response",
        hooks=("before_user_response",),
        required_inputs=("draft_response",),
        description=(
            "Draft response must be an explicit clarification request rather than an implicit choice prompt "
            "or unsupported completion."
        ),
        parameters_schema={
            "properties": {
                "markers": {"type": "array", "items": {"type": "string"}},
                "allow_question_only": {"type": "boolean"},
            }
        },
    ),
}


def validation_checks_for_hook(hook: str) -> list[ValidationCheckSpec]:
    return [spec for spec in VALIDATION_CHECKS.values() if hook in spec.hooks]


def validation_check_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": sorted(VALIDATION_CHECKS)},
            "keys": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 20},
            "state_field": {"type": "string", "maxLength": 120},
            "arg": {"type": "string", "maxLength": 120},
            "state_key": {"type": "string", "maxLength": 120},
            "markers": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 20},
            "allow_question_only": {"type": "boolean"},
        },
        "required": ["type"],
        "additionalProperties": True,
    }


def normalize_validation_check(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        if raw in VALIDATION_CHECKS:
            return {"type": raw}
        return None
    if not isinstance(raw, dict):
        return None
    if set(raw) == {"required_output_keys"} and isinstance(raw.get("required_output_keys"), list):
        return {"type": "required_output_keys", "keys": list(raw["required_output_keys"])}
    check_type = raw.get("type")
    if isinstance(check_type, str) and check_type in VALIDATION_CHECKS:
        return dict(raw)
    return None


def validate_check_payload(raw: Any, *, hook: str) -> str | None:
    check = normalize_validation_check(raw)
    if check is None:
        return f"Validation check {raw!r} is not implemented by the runtime."
    check_type = str(check["type"])
    spec = VALIDATION_CHECKS[check_type]
    if hook not in spec.hooks:
        return f"Validation check {check_type!r} is not available at hook {hook!r}."
    if check_type == "required_output_keys":
        keys = check.get("keys")
        if not isinstance(keys, list) or not keys or not all(isinstance(item, str) and item for item in keys):
            return "required_output_keys requires non-empty keys[]."
    if check_type == "completion_claims_supported":
        state_field = check.get("state_field", "completed_actions")
        if not isinstance(state_field, str) or not state_field:
            return "completion_claims_supported state_field must be a non-empty string."
    if check_type == "tool_arg_in_state":
        state_field = check.get("state_field")
        arg = check.get("arg")
        state_key = check.get("state_key")
        if not isinstance(state_field, str) or not state_field:
            return "tool_arg_in_state requires non-empty state_field."
        if not isinstance(arg, str) or not arg:
            return "tool_arg_in_state requires non-empty arg."
        if state_key is not None and (not isinstance(state_key, str) or not state_key):
            return "tool_arg_in_state state_key must be a non-empty string when provided."
    if check_type == "clarification_response":
        markers = check.get("markers")
        if markers is not None and (
            not isinstance(markers, list) or not markers or not all(isinstance(item, str) and item.strip() for item in markers)
        ):
            return "clarification_response markers must be a non-empty string array when provided."
        allow_question_only = check.get("allow_question_only")
        if allow_question_only is not None and not isinstance(allow_question_only, bool):
            return "clarification_response allow_question_only must be boolean when provided."
    return None


def run_validation_check(raw: Any, ctx: Any) -> bool:
    check = normalize_validation_check(raw)
    if check is None:
        raise ValueError(f"Unsupported validation check: {raw!r}")
    check_type = str(check["type"])
    if check_type == "json_object":
        return isinstance(ctx.draft_response, dict)
    if check_type == "actions_array":
        return isinstance(ctx.draft_response, dict) and isinstance(ctx.draft_response.get("actions"), list)
    if check_type == "required_output_keys":
        keys = check.get("keys")
        return isinstance(keys, list) and isinstance(ctx.draft_response, dict) and all(key in ctx.draft_response for key in keys)
    if check_type == "args_schema_valid":
        return _tool_args_satisfy_schema(ctx)
    if check_type == "not_duplicate_tool_call":
        return not _has_prior_matching_tool_call(ctx)
    if check_type == "mutating_tool_requires_confirmation":
        if _tool_side_effect(ctx) not in {"mutating", "destructive"}:
            return True
        return _recent_user_confirmation(ctx)
    if check_type == "referenced_args_observed":
        return _referenced_args_observed(ctx)
    if check_type == "tool_arg_in_state":
        return _tool_arg_in_state(
            ctx,
            state_field=str(check["state_field"]),
            arg=str(check["arg"]),
            state_key=str(check.get("state_key") or check["arg"]),
        )
    if check_type == "completion_claims_supported":
        return _completion_claims_supported(ctx, state_field=str(check.get("state_field") or "completed_actions"))
    if check_type == "clarification_response":
        markers = check.get("markers")
        return _clarification_response(
            ctx,
            markers=[str(item) for item in markers] if isinstance(markers, list) else None,
            allow_question_only=bool(check.get("allow_question_only", False)),
        )
    raise ValueError(f"Unsupported validation check: {check_type!r}")


def _tool_args_satisfy_schema(ctx: Any) -> bool:
    if not isinstance(ctx.tool_call, dict) or not isinstance(ctx.tool_call.get("args"), dict):
        return False
    schema = ctx.tool_schema if isinstance(ctx.tool_schema, dict) else {}
    return _validate_json_schema_value(ctx.tool_call["args"], schema)


def _validate_json_schema_value(value: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict) or not schema:
        return True
    if "enum" in schema and value not in schema.get("enum", []):
        return False
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, dict):
            return False
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        if any(key not in value for key in required):
            return False
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict) and not _validate_json_schema_value(value[key], child_schema):
                return False
        return True
    if expected_type == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            return all(_validate_json_schema_value(item, item_schema) for item in value)
        return True
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int | float) and not isinstance(value, bool))
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _has_prior_matching_tool_call(ctx: Any) -> bool:
    if not isinstance(ctx.tool_call, dict):
        return False
    current_name = str(ctx.tool_call.get("name") or "")
    current_args = _stable_json(ctx.tool_call.get("args") or {})
    for message in ctx.message_history:
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or str(function.get("name") or "") != current_name:
                continue
            if _stable_json(_tool_call_args(function.get("arguments"))) == current_args:
                return True
    return False


def _tool_call_args(raw_args: Any) -> Any:
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return raw_args
    return raw_args or {}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _tool_side_effect(ctx: Any) -> str:
    metadata = ctx.tool_metadata if isinstance(ctx.tool_metadata, dict) else {}
    side_effect = str(metadata.get("side_effect") or "").lower()
    if side_effect:
        return side_effect
    name = str((ctx.tool_call or {}).get("name") or "").lower() if isinstance(ctx.tool_call, dict) else ""
    if any(word in name for word in ("delete", "cancel", "modify", "create", "update", "return", "exchange", "transfer")):
        return "mutating"
    return "read"


def _recent_user_confirmation(ctx: Any) -> bool:
    for message in reversed(ctx.message_history[-4:]):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = str(message.get("content") or "").strip().lower()
        if re.fullmatch(r"(yes|y|confirmed|confirm|i confirm|go ahead|please proceed|proceed|correct)", content):
            return True
    return False


def _referenced_args_observed(ctx: Any) -> bool:
    if not isinstance(ctx.tool_call, dict) or not isinstance(ctx.tool_call.get("args"), dict):
        return False
    observed_text = "\n".join(
        str(message.get("content") or "")
        for message in ctx.message_history
        if isinstance(message, dict) and message.get("role") in {"user", "tool"}
    )
    if not observed_text:
        return True
    for value in _flatten_values(ctx.tool_call["args"]):
        if not isinstance(value, str):
            continue
        token = value.strip()
        if _looks_like_identifier(token) and not _observed_text_contains_identifier(observed_text, token):
            return False
    return True


def _tool_arg_in_state(ctx: Any, *, state_field: str, arg: str, state_key: str) -> bool:
    if not isinstance(ctx.tool_call, dict) or not isinstance(ctx.tool_call.get("args"), dict):
        return False
    value = ctx.tool_call["args"].get(arg)
    if value is None:
        return False
    state = ctx.state if isinstance(ctx.state, dict) else {}
    observed = state.get(state_field)
    if not isinstance(observed, list):
        return False
    expected = _canonical_identifier(value)
    for item in observed:
        if _canonical_identifier(item) == expected:
            return True
        if isinstance(item, dict) and _canonical_identifier(item.get(state_key)) == expected:
            return True
    return False


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        rows: list[Any] = []
        for child in value.values():
            rows.extend(_flatten_values(child))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(_flatten_values(child))
        return rows
    return [value]


def _looks_like_identifier(value: str) -> bool:
    if len(value) < 4:
        return False
    return bool(re.search(r"\d", value) or re.fullmatch(r"[A-Za-z_]+_[A-Za-z0-9_]+", value))


def _observed_text_contains_identifier(observed_text: str, value: str) -> bool:
    if value in observed_text:
        return True
    canonical = _canonical_identifier(value)
    if not canonical:
        return False
    return canonical in {_canonical_identifier(match) for match in re.findall(r"[#A-Za-z0-9_:-]{4,}", observed_text)}


def _canonical_identifier(value: Any) -> str:
    if not isinstance(value, str):
        return str(value)
    normalized = value.strip()
    if normalized.startswith("#") and len(normalized) > 1:
        normalized = normalized[1:]
    return normalized.casefold()


def _completion_claims_supported(ctx: Any, *, state_field: str) -> bool:
    text = str(ctx.draft_response or "").lower()
    if not any(marker in text for marker in ("done", "completed", "submitted", "cancelled", "canceled", "modified", "created", "processed")):
        return True
    state = ctx.state if isinstance(ctx.state, dict) else {}
    completed = state.get(state_field)
    if isinstance(completed, list):
        return bool(completed)
    return bool(completed)


DEFAULT_CLARIFICATION_MARKERS = (
    "which",
    "what",
    "clarify",
    "confirm",
    "specify",
    "provide",
    "choose",
    "select",
)


def _clarification_response(ctx: Any, *, markers: list[str] | None, allow_question_only: bool) -> bool:
    text = str(ctx.draft_response or "").strip().lower()
    if not text:
        return False
    if _completion_claim_text(text):
        return False
    marker_set = tuple(marker.strip().lower() for marker in (markers or list(DEFAULT_CLARIFICATION_MARKERS)) if marker.strip())
    has_marker = any(re.search(rf"\b{re.escape(marker)}\b", text) for marker in marker_set)
    has_question_shape = "?" in text or any(
        re.search(rf"\b{re.escape(marker)}\b", text) for marker in ("please", "need", "missing", "unclear")
    )
    if allow_question_only:
        return has_question_shape
    return has_marker and has_question_shape


def _completion_claim_text(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "done",
            "completed",
            "submitted",
            "cancelled",
            "canceled",
            "modified",
            "created",
            "processed",
            "updated",
            "changed",
            "return has been",
            "refund has been",
        )
    )
