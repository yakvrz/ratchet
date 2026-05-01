from __future__ import annotations

from dataclasses import dataclass, field
import copy
import json
import re
from typing import Any

from ratchet.capabilities import run_validation_check
from ratchet.context_graph import ContextGraph, ContextSection
from ratchet.transform_program import CompiledCandidate, TransformPatch
from ratchet.types import EvalCase


@dataclass
class RuntimeContext:
    case: EvalCase
    context: ContextGraph
    model_config: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)
    message_history: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_call: dict[str, Any] | None = None
    tool_schema: dict[str, Any] | None = None
    tool_metadata: dict[str, Any] | None = None
    tool_result: Any = None
    tool_error: Any = None
    raw_response: Any = None
    draft_response: Any = None
    output: Any = None
    trace_annotations: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def annotate(self, *, hook: str, op: str, result: str = "applied", fields: dict[str, Any] | None = None) -> None:
        self.trace_annotations.append(
            {
                "hook": hook,
                "op": op,
                "result": result,
                "fields": dict(fields or {}),
            }
        )


class TransformRuntime:
    def __init__(self, candidate: CompiledCandidate | None) -> None:
        self.candidate = candidate

    def run_hook(self, hook: str, ctx: RuntimeContext) -> RuntimeContext:
        if self.candidate is None:
            return ctx
        for patch in self.candidate.operations_by_hook.get(hook, ()):
            if not _condition_matches(patch.when, ctx):
                ctx.annotate(hook=hook, op=patch.op.op, result="skipped_when")
                continue
            if patch.unless and _condition_matches(patch.unless, ctx):
                ctx.annotate(hook=hook, op=patch.op.op, result="skipped_unless")
                continue
            self._apply_patch(hook, patch, ctx)
        return ctx

    def _apply_patch(self, hook: str, patch: TransformPatch, ctx: RuntimeContext) -> None:
        op = patch.op.op
        params = patch.op.params
        if hook == "before_tool_call" and "tool" in params:
            selected_tool = str(params["tool"])
            actual_tool = str((ctx.tool_call or {}).get("name") or "")
            if selected_tool != actual_tool:
                ctx.annotate(hook=hook, op=op, result="skipped_tool", fields={"tool": selected_tool, "actual": actual_tool})
                return
        if op == "define_state":
            field = str(params["field"])
            ctx.state[field] = copy.deepcopy(resolve_value(params.get("initial"), ctx))
            ctx.annotate(hook=hook, op=op, fields={"field": field})
            return
        if op == "set_state":
            field = str(params["field"])
            ctx.state[field] = resolve_value(params.get("value"), ctx)
            ctx.annotate(hook=hook, op=op, fields={"field": field})
            return
        if op == "append_state":
            field = str(params["field"])
            value = resolve_value(params.get("value"), ctx)
            existing = ctx.state.setdefault(field, [])
            if not isinstance(existing, list):
                raise TypeError(f"State field {field!r} is not appendable.")
            if params.get("extend") is True and isinstance(value, list):
                existing.extend(item for item in value if item not in existing)
            elif value not in existing:
                existing.append(value)
            ctx.annotate(hook=hook, op=op, fields={"field": field})
            return
        if op == "merge_state":
            field = str(params["field"])
            value = resolve_value(params.get("value"), ctx)
            existing = ctx.state.setdefault(field, {})
            if not isinstance(existing, dict) or not isinstance(value, dict):
                raise TypeError(f"State field {field!r} is not mergeable.")
            existing.update(value)
            ctx.annotate(hook=hook, op=op, fields={"field": field})
            return
        if op == "clear_state":
            field = str(params["field"])
            ctx.state[field] = []
            ctx.annotate(hook=hook, op=op, fields={"field": field})
            return
        if op == "add_context_section":
            section = ContextSection(
                name=str(params["section"]),
                role=str(params.get("role", "system")),
                content=resolve_value(params.get("content", ""), ctx),
                required=bool(params.get("required", False)),
                visibility=str(params.get("visibility", "model_visible")),
                metadata=dict(params.get("metadata", {})),
            )
            ctx.context = ctx.context.add_section(section, position=params.get("position"))
            ctx.annotate(hook=hook, op=op, fields={"section": section.name})
            return
        if op == "remove_context_section":
            section = str(params["section"])
            ctx.context = ctx.context.remove_section(section)
            ctx.annotate(hook=hook, op=op, fields={"section": section})
            return
        if op == "replace_context_section":
            section = str(params["section"])
            ctx.context = ctx.context.replace_section(section, resolve_value(params.get("content", ""), ctx))
            ctx.annotate(hook=hook, op=op, fields={"section": section})
            return
        if op == "move_context_section":
            section = str(params["section"])
            ctx.context = ctx.context.move_section(section, position=str(params["position"]))
            ctx.annotate(hook=hook, op=op, fields={"section": section})
            return
        if op == "reorder_context_sections":
            order = [str(item) for item in params["order"]]
            ctx.context = ctx.context.reorder(order)
            ctx.annotate(hook=hook, op=op, fields={"order": order})
            return
        if op == "render_state_section":
            section_name = str(params["section"])
            fields = params.get("fields")
            if fields is None:
                content = dict(ctx.state)
            elif isinstance(fields, list):
                content = {str(field): ctx.state.get(str(field)) for field in fields}
            else:
                raise TypeError("render_state_section fields must be an array when provided.")
            section = ContextSection(
                name=section_name,
                role=str(params.get("role", "system")),
                content=content,
            )
            ctx.context = ctx.context.add_section(section, position=params.get("position"))
            ctx.annotate(hook=hook, op=op, fields={"section": section_name})
            return
        if op == "set_model_config":
            field = str(params["field"])
            ctx.model_config[field] = resolve_value(params.get("value"), ctx)
            ctx.annotate(hook=hook, op=op, fields={"field": field})
            return
        if op == "normalize_tool_args":
            if ctx.tool_call is None:
                raise TypeError("normalize_tool_args requires a tool_call in the runtime context.")
            ctx.tool_call["args"] = _normalize_args(ctx.tool_call.get("args"), params)
            ctx.annotate(hook=hook, op=op, fields={"tool": ctx.tool_call.get("name")})
            return
        if op == "rewrite_tool_description":
            tool_name = str(params["tool"])
            content = str(params.get("content") or "")
            append = str(params.get("append") or "")
            if not content and not append:
                raise ValueError("rewrite_tool_description requires content or append.")
            rewritten = False
            for tool in ctx.tools:
                function = tool.get("function") if isinstance(tool, dict) else None
                if not isinstance(function, dict) or str(function.get("name") or "") != tool_name:
                    continue
                current = str(function.get("description") or "")
                function["description"] = content if content else f"{current}\n\n{append}".strip()
                rewritten = True
            if not rewritten:
                raise KeyError(f"Tool {tool_name!r} is not available for description rewrite.")
            ctx.annotate(hook=hook, op=op, fields={"tool": tool_name})
            return
        if op in {"log_event", "trace_annotation"}:
            ctx.annotate(hook=hook, op=op, fields=resolve_value(params.get("fields", {}), ctx))
            return
        if op in {"validate", "validate_claims"}:
            result = _validate(params, ctx)
            ctx.annotate(hook=hook, op=op, result="passed" if result else "failed", fields={"target": params.get("target")})
            if not result:
                on_fail = params.get("on_fail")
                if isinstance(on_fail, dict):
                    self._apply_patch(hook, TransformPatch.from_dict(on_fail), ctx)
            return
        if op == "rewrite_response":
            ctx.draft_response = _rewrite_response(params, ctx)
            ctx.output = ctx.draft_response
            ctx.annotate(hook=hook, op=op, fields={"response_preview": _preview_value(ctx.draft_response)})
            return
        if op == "block_response":
            ctx.output = {"blocked": True, "message": str(params.get("message", "Response blocked by transform."))}
            ctx.annotate(hook=hook, op=op)
            return
        if op in {"allow", "continue"}:
            ctx.annotate(hook=hook, op=op)
            return
        if op in {"block", "replan", "ask_user", "terminate", "retry"}:
            ctx.state["_control"] = {"op": op, "message": params.get("message", params.get("content"))}
            ctx.annotate(hook=hook, op=op)
            return
        raise NotImplementedError(f"Runtime op {op!r} is not implemented.")


def resolve_value(value: Any, ctx: RuntimeContext) -> Any:
    if isinstance(value, dict):
        raw_ref = value.get("$ref") or value.get("ref")
        if isinstance(raw_ref, str):
            return _resolve_ref(raw_ref, ctx)
        return {key: resolve_value(item, ctx) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_value(item, ctx) for item in value]
    if isinstance(value, str):
        return _resolve_template(value, ctx)
    return value


TEMPLATE_REF_PATTERN = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")


def _resolve_template(value: str, ctx: RuntimeContext) -> str:
    def replace(match: re.Match[str]) -> str:
        resolved = _resolve_ref(match.group(1), ctx)
        if resolved is None:
            return ""
        if isinstance(resolved, str):
            return resolved
        if isinstance(resolved, (dict, list)):
            return json.dumps(resolved, sort_keys=True, default=str)
        return str(resolved)

    return TEMPLATE_REF_PATTERN.sub(replace, value)


def _resolve_ref(ref: str, ctx: RuntimeContext) -> Any:
    parts = ref.split(".")
    root = parts[0]
    if root == "state":
        current: Any = ctx.state
    elif root == "case":
        current = ctx.case
    elif root == "context":
        current = ctx.context
    elif root == "model_config":
        current = ctx.model_config
    elif root == "message_history":
        current = ctx.message_history
    elif root == "tool_call":
        current = ctx.tool_call
    elif root == "tool_schema":
        current = ctx.tool_schema
    elif root == "tool_metadata":
        current = ctx.tool_metadata
    elif root == "tool_result":
        current = ctx.tool_result
    elif root == "tool_error":
        current = ctx.tool_error
    elif root == "draft_response":
        current = ctx.draft_response
    elif root == "output":
        current = ctx.output
    elif root == "trace":
        current = ctx.trace_annotations
    else:
        raise KeyError(f"Unknown runtime reference root {root!r}.")
    for part in parts[1:]:
        current = _resolve_ref_part(current, part)
    return current


def _resolve_ref_part(current: Any, part: str) -> Any:
    if part.endswith("[]"):
        key = part[:-2]
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key)
        if current is None:
            return []
        if not isinstance(current, list):
            raise TypeError(f"Runtime reference expected list at {part!r}.")
        return current
    if isinstance(current, list):
        values: list[Any] = []
        for item in current:
            value = _resolve_ref_part(item, part)
            if isinstance(value, list):
                values.extend(value)
            elif value is not None:
                values.append(value)
        return values
    if isinstance(current, dict):
        return current.get(part)
    return getattr(current, part)


def _condition_matches(condition: dict[str, Any], ctx: RuntimeContext) -> bool:
    if not condition:
        return True
    for ref, expected in condition.items():
        actual = _resolve_ref(str(ref), ctx)
        if isinstance(expected, dict):
            if "exists" in expected and bool(actual is not None) != bool(expected["exists"]):
                return False
            if "not_empty" in expected and bool(actual) != bool(expected["not_empty"]):
                return False
            if "equals" in expected and actual != expected["equals"]:
                return False
        elif actual != expected:
            return False
    return True


def _normalize_args(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str):
        normalized = value.strip() if params.get("trim_strings", True) else value
        return normalized
    if isinstance(value, list):
        return [_normalize_args(item, params) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_args(item, params) for key, item in value.items()}
    return value


def _validate(params: dict[str, Any], ctx: RuntimeContext) -> bool:
    checks = params.get("checks")
    if not isinstance(checks, list):
        return True
    for check in checks:
        if not run_validation_check(check, ctx):
            return False
    return True


def _preview_value(value: Any, limit: int = 500) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else f"{text[:limit]}..."


def _rewrite_response(params: dict[str, Any], ctx: RuntimeContext) -> Any:
    replacement = params.get("replacement")
    if replacement is not None:
        return resolve_value(replacement, ctx)
    if isinstance(ctx.draft_response, dict):
        rewritten = dict(ctx.draft_response)
        for key in params.get("remove_keys", []):
            rewritten.pop(str(key), None)
        if "message" in params:
            rewritten["message"] = str(params["message"])
        return rewritten
    if "message" in params:
        return str(params["message"])
    return ctx.draft_response
