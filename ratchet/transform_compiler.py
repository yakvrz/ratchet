from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from ratchet.capabilities import normalize_validation_check, validate_check_payload
from ratchet.surfaces import SurfaceSpec, SUPPORTED_HOOKS
from ratchet.transform_program import (
    CandidateDiff,
    CompileIssue,
    CompileReport,
    CompiledCandidate,
    TransformPatch,
    TransformProgram,
    references_in_value,
)


FORBIDDEN_REFERENCE_ROOTS = {
    "evaluator",
    "gold",
    "gold_answers",
    "hidden",
    "hidden_labels",
    "hidden_task_goal",
    "expected",
}
CASE_ID_CONDITION_PATTERN = re.compile(r"\b(case|task)[_-]?id\b", re.IGNORECASE)
REFERENCE_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+(?:\[\])?)*$")
CONDITION_OPERATOR_KEYS = {"exists", "not_empty", "equals"}
FORBIDDEN_TRACE_LITERALS = (
    "###stop###",
    "hidden task",
    "gold answer",
    "evaluator label",
)


class TransformCompileError(ValueError):
    def __init__(self, issue: CompileIssue) -> None:
        super().__init__(issue.message)
        self.issue = issue


class TransformCompiler:
    def compile(self, program: TransformProgram, surface: SurfaceSpec) -> CompiledCandidate:
        try:
            return self._compile(program, surface)
        except TransformCompileError as exc:
            report = CompileReport(
                candidate_id=program.candidate_id,
                status="rejected",
                rejection=exc.issue,
            )
            return CompiledCandidate(
                program=program,
                operations_by_hook={},
                report=report,
                diff=CandidateDiff(),
            )

    def compile_or_raise(self, program: TransformProgram, surface: SurfaceSpec) -> CompiledCandidate:
        compiled = self.compile(program, surface)
        if compiled.report.status != "compiled":
            issue = compiled.report.rejection or CompileIssue("compile_error", "candidate rejected")
            raise TransformCompileError(issue)
        return compiled

    def _compile(self, program: TransformProgram, surface: SurfaceSpec) -> CompiledCandidate:
        state_fields = set(surface.state.existing_fields)
        defined_state_fields: list[str] = []
        added_context_sections: list[str] = []
        validators: list[str] = []
        modified_surfaces: set[str] = set()
        hook_changes: dict[str, list[str]] = defaultdict(list)
        operations_by_hook: dict[str, list[TransformPatch]] = defaultdict(list)

        for index, patch in enumerate(program.patches):
            self._validate_patch(index, patch, surface, state_fields)
            op = patch.op.op
            hook = patch.hook or "on_task_start"
            operations_by_hook[hook].append(patch)
            modified_surfaces.add(hook)
            hook_changes[hook].append(op)
            if op == "define_state":
                field = self._required_string(index, patch, "field")
                if field in state_fields:
                    self._reject(index, "duplicate_state_field", f"State field {field!r} already exists.")
                state_fields.add(field)
                defined_state_fields.append(field)
            if op in {"add_context_section", "render_state_section"}:
                section = self._required_string(index, patch, "section")
                added_context_sections.append(section)
            if op in {"validate", "validate_claims", "schema_check", "support_check", "precondition_check"}:
                validators.append(str(patch.op.params.get("target") or op))

        diff = CandidateDiff(
            added_state_fields=tuple(defined_state_fields),
            added_context_sections=tuple(added_context_sections),
            context_changes=tuple(self._context_changes(program.patches)),
            hook_changes={key: tuple(value) for key, value in sorted(hook_changes.items())},
        )
        report = CompileReport(
            candidate_id=program.candidate_id,
            status="compiled",
            modified_surfaces=tuple(sorted(modified_surfaces)),
            added_state_fields=tuple(defined_state_fields),
            added_context_sections=tuple(added_context_sections),
            added_validators=tuple(validators),
            estimated_overhead={
                "tokens_per_turn": "medium" if added_context_sections else "none",
                "extra_model_calls": sum(1 for patch in program.patches if patch.op.op == "call_model"),
                "extra_tool_calls": 0,
            },
            warnings=tuple(self._warnings(program.patches, surface)),
        )
        return CompiledCandidate(
            program=program,
            operations_by_hook={key: tuple(value) for key, value in sorted(operations_by_hook.items())},
            report=report,
            diff=diff,
        )

    def _validate_patch(
        self,
        index: int,
        patch: TransformPatch,
        surface: SurfaceSpec,
        state_fields: set[str],
    ) -> None:
        hook_name = patch.hook or "on_task_start"
        if hook_name not in SUPPORTED_HOOKS:
            self._reject(index, "unknown_hook", f"Unknown hook {hook_name!r}.")
        hook = surface.hooks[hook_name]
        if not hook.supported:
            self._reject(index, "unsupported_hook", f"Candidate uses {hook_name}, but this surface does not support it.")
        op = patch.op.op
        if op not in hook.allowed_ops:
            self._reject(index, "unsupported_operation", f"Operation {op!r} is not allowed at hook {hook_name!r}.")
        self._validate_boundaries(index, patch)
        self._validate_context_op(index, patch, surface)
        self._validate_state_op(index, patch, surface, state_fields)
        self._validate_model_op(index, patch, surface)
        self._validate_response_op(index, patch, surface)
        self._validate_tool_op(index, patch, surface)
        self._validate_validation_checks(index, patch, hook_name, state_fields)
        self._validate_conditions(index, patch, hook.available_inputs, state_fields)
        self._validate_on_fail(index, patch, hook.allowed_ops, state_fields)
        self._validate_references(index, patch, hook.available_inputs, state_fields)

    def _validate_context_op(self, index: int, patch: TransformPatch, surface: SurfaceSpec) -> None:
        op = patch.op.op
        params = patch.op.params
        if op == "add_context_section" and not surface.context.generated_sections_allowed:
            self._reject(index, "context_generation_unsupported", "Surface does not allow generated context sections.")
        if op == "remove_context_section" and not surface.context.removable_sections_allowed:
            self._reject(index, "context_removal_unsupported", "Surface does not allow context section removal.")
        if op in {"move_context_section", "reorder_context_sections"} and not surface.context.reorderable_sections_allowed:
            self._reject(index, "context_reorder_unsupported", "Surface does not allow context section reordering.")
        if op == "remove_context_section":
            section = self._required_string(index, patch, "section")
            existing = [item for item in surface.context.graph.sections if item.name == section]
            if not existing:
                self._reject(index, "unknown_context_section", f"Unknown context section {section!r}.")
            if existing[0].required:
                self._reject(index, "required_context_section", f"Cannot remove required section {section!r}.")
        if op == "replace_context_section":
            section = self._required_string(index, patch, "section")
            if section not in surface.context.editable_sections:
                self._reject(index, "readonly_context_section", f"Context section {section!r} is not editable.")
        if op in {"add_context_section", "replace_context_section"}:
            self._validate_context_content(index, patch)
        if op == "reorder_context_sections":
            order = params.get("order")
            if not isinstance(order, list) or not all(isinstance(item, str) and item for item in order):
                self._reject(index, "invalid_context_order", "reorder_context_sections requires order[] of section names.")

    def _validate_context_content(self, index: int, patch: TransformPatch) -> None:
        params = patch.op.params
        if "content" not in params:
            hint = " Use 'content', not 'value'." if "value" in params else ""
            self._reject(
                index,
                "context_content_required",
                f"Operation {patch.op.op!r} requires non-empty 'content'.{hint}",
            )
        content = params["content"]
        if content is None:
            self._reject(index, "context_content_required", f"Operation {patch.op.op!r} requires non-empty 'content'.")
        if isinstance(content, str) and not content.strip():
            self._reject(index, "context_content_required", f"Operation {patch.op.op!r} requires non-empty 'content'.")
        if isinstance(content, (list, tuple, dict)) and not content:
            self._reject(index, "context_content_required", f"Operation {patch.op.op!r} requires non-empty 'content'.")

    def _validate_state_op(
        self,
        index: int,
        patch: TransformPatch,
        surface: SurfaceSpec,
        state_fields: set[str],
    ) -> None:
        op = patch.op.op
        if op in {"define_state", "set_state", "append_state", "merge_state", "clear_state", "expose_state", "hide_state"}:
            if not surface.state.supports_persistent_state:
                self._reject(index, "state_unsupported", "Surface does not support persistent state.")
        if op == "define_state":
            if not surface.state.add_fields_allowed:
                self._reject(index, "state_add_unsupported", "Surface does not allow new state fields.")
            field = self._required_string(index, patch, "field")
            type_name = self._required_string(index, patch, "type")
            if not type_name:
                self._reject(index, "state_type_required", f"State field {field!r} requires a type.")
        if op in {"set_state", "append_state", "merge_state", "clear_state", "expose_state", "hide_state"}:
            field = self._required_string(index, patch, "field")
            if field not in state_fields:
                self._reject(index, "unknown_state_field", f"State field {field!r} is referenced before definition.")

    def _validate_model_op(self, index: int, patch: TransformPatch, surface: SurfaceSpec) -> None:
        if patch.op.op == "call_model" and not surface.model.auxiliary_model_calls_allowed:
            self._reject(index, "auxiliary_model_call_unsupported", "Surface does not allow auxiliary model calls.")
        if patch.op.op == "set_model_config":
            field = self._required_string(index, patch, "field")
            allowed = {
                "model_name": surface.model.model_name_configurable,
                "temperature": surface.model.temperature_configurable,
                "max_tokens": surface.model.max_tokens_configurable,
                "reasoning_effort": surface.model.reasoning_effort_configurable,
                "tool_choice_mode": surface.model.tool_choice_mode_configurable,
            }
            if field not in allowed or not allowed[field]:
                self._reject(index, "model_config_unsupported", f"Model config field {field!r} is not configurable.")
            if field == "model_name":
                value = patch.op.params.get("value")
                if not isinstance(value, str) or not value:
                    self._reject(index, "model_name_required", "set_model_config field 'model_name' requires a non-empty string value.")
                if surface.model.model_options and value not in surface.model.model_options:
                    self._reject(
                        index,
                        "model_name_not_allowed",
                        f"Model name {value!r} is not in this surface's model_options.",
                    )

    def _validate_response_op(self, index: int, patch: TransformPatch, surface: SurfaceSpec) -> None:
        if patch.op.op in {"extract_claims", "validate_claims"} and not surface.response.draft_response_interception_allowed:
            self._reject(index, "response_interception_unsupported", "Surface does not allow draft response interception.")
        if patch.op.op == "rewrite_response" and not surface.response.response_rewrite_allowed:
            self._reject(index, "response_rewrite_unsupported", "Surface does not allow response rewriting.")
        if patch.op.op == "rewrite_response":
            params = patch.op.params
            if "message" not in params and "replacement" not in params:
                self._reject(index, "response_rewrite_content_required", "rewrite_response requires message or replacement.")
        if patch.op.op == "block_response" and not surface.response.response_blocking_allowed:
            self._reject(index, "response_block_unsupported", "Surface does not allow response blocking.")
        if patch.op.op == "block_response" and "message" not in patch.op.params:
            self._reject(index, "response_block_message_required", "block_response requires message.")

    def _validate_tool_op(self, index: int, patch: TransformPatch, surface: SurfaceSpec) -> None:
        op = patch.op.op
        if patch.hook == "before_tool_call" and "tool" in patch.op.params:
            tool_name = self._required_string(index, patch, "tool")
            known_tools = {tool.name for tool in surface.tools.tools}
            if known_tools and tool_name not in known_tools:
                self._reject(index, "unknown_tool", f"Tool {tool_name!r} is not part of this surface.")
        if op == "rewrite_tool_description" and not surface.tools.tool_description_rewrite_allowed:
            self._reject(index, "tool_description_rewrite_unsupported", "Surface does not allow tool description rewrites.")
        if op == "rewrite_tool_description":
            self._required_string(index, patch, "tool")
            content = str(patch.op.params.get("content") or "").strip()
            append = str(patch.op.params.get("append") or "").strip()
            if not content and not append:
                self._reject(
                    index,
                    "tool_description_content_required",
                    "rewrite_tool_description requires non-empty content or append.",
                )
        if op in {"normalize_tool_args", "repair_tool_args"} and not surface.tools.tool_call_interception_allowed:
            self._reject(index, "tool_call_interception_unsupported", "Surface does not allow tool-call interception.")
        if op == "annotate_tool" and not surface.tools.tool_metadata_allowed:
            self._reject(index, "tool_metadata_unsupported", "Surface does not allow tool metadata.")
        if op == "rewrite_tool_result":
            self._reject(index, "tool_result_tampering", "Transform programs may not rewrite tool results.")

    def _validate_validation_checks(
        self,
        index: int,
        patch: TransformPatch,
        hook_name: str,
        state_fields: set[str],
    ) -> None:
        if patch.op.op not in {"validate", "validate_claims"}:
            return
        checks = patch.op.params.get("checks")
        if checks is None:
            self._reject(index, "validation_checks_required", "validate requires implemented checks[].")
        if not isinstance(checks, list):
            self._reject(index, "invalid_validation_checks", "validate requires checks[] when checks are provided.")
        if not checks:
            self._reject(index, "validation_checks_required", "validate requires at least one implemented check.")
        for check in checks:
            error = validate_check_payload(check, hook=hook_name)
            if error is not None:
                self._reject(index, "unsupported_validation_check", error)
            normalized = normalize_validation_check(check)
            if isinstance(normalized, dict):
                state_field = normalized.get("state_field")
                if isinstance(state_field, str) and state_field and state_field not in state_fields:
                    self._reject(
                        index,
                        "unknown_state_field",
                        f"Validation check {normalized.get('type')!r} references undefined state field {state_field!r}.",
                    )
        patch.op.params["checks"] = [normalize_validation_check(check) for check in checks]

    def _validate_references(
        self,
        index: int,
        patch: TransformPatch,
        available_inputs: tuple[str, ...],
        state_fields: set[str],
    ) -> None:
        payload = patch.to_dict()
        for ref in references_in_value(payload):
            root = ref.split(".", 1)[0]
            if root in FORBIDDEN_REFERENCE_ROOTS:
                self._reject(index, "immutable_boundary_violation", f"Candidate attempts to access forbidden reference {ref!r}.")
            if root == "state":
                parts = ref.split(".")
                if len(parts) < 2 or parts[1] not in state_fields:
                    self._reject(index, "unknown_state_field", f"State reference {ref!r} is not defined.")
            elif root not in available_inputs and root not in {"candidate", "surface"}:
                self._reject(index, "unavailable_reference", f"Reference {ref!r} is unavailable at this hook.")

    def _validate_conditions(
        self,
        index: int,
        patch: TransformPatch,
        available_inputs: tuple[str, ...],
        state_fields: set[str],
    ) -> None:
        for condition_name, condition in (("when", patch.when), ("unless", patch.unless)):
            if condition is None:
                continue
            if not isinstance(condition, dict):
                self._reject(index, "invalid_condition", f"{condition_name} must be an object of runtime refs to expected values.")
            for ref, expected in condition.items():
                if not isinstance(ref, str) or not REFERENCE_KEY_PATTERN.fullmatch(ref):
                    self._reject(
                        index,
                        "invalid_condition",
                        f"{condition_name} key {ref!r} is not a runtime reference; supported shape is {{'tool_call.name': 'tool'}} or {{'state.field': {{'exists': true}}}}.",
                    )
                root = ref.split(".", 1)[0]
                if root in FORBIDDEN_REFERENCE_ROOTS:
                    self._reject(index, "immutable_boundary_violation", f"Candidate condition attempts forbidden reference {ref!r}.")
                if root == "state":
                    parts = ref.split(".")
                    if len(parts) < 2 or parts[1] not in state_fields:
                        self._reject(index, "unknown_state_field", f"Condition state reference {ref!r} is not defined.")
                elif root not in available_inputs:
                    self._reject(index, "unavailable_reference", f"Condition reference {ref!r} is unavailable at this hook.")
                if isinstance(expected, dict):
                    unknown_keys = set(expected) - CONDITION_OPERATOR_KEYS
                    if unknown_keys:
                        self._reject(
                            index,
                            "invalid_condition",
                            f"{condition_name} for {ref!r} uses unsupported operator(s): {sorted(unknown_keys)}.",
                        )
                    if "exists" in expected and not isinstance(expected["exists"], bool):
                        self._reject(index, "invalid_condition", f"{condition_name} operator 'exists' requires a boolean.")
                    if "not_empty" in expected and not isinstance(expected["not_empty"], bool):
                        self._reject(index, "invalid_condition", f"{condition_name} operator 'not_empty' requires a boolean.")

    def _validate_on_fail(
        self,
        index: int,
        patch: TransformPatch,
        hook_allowed_ops: tuple[str, ...],
        state_fields: set[str],
    ) -> None:
        if patch.op.op not in {"validate", "validate_claims"}:
            return
        on_fail = patch.op.params.get("on_fail")
        if on_fail is None:
            return
        if not isinstance(on_fail, dict):
            self._reject(index, "invalid_on_fail", "validate on_fail must be a transform operation object.")
        try:
            failure_patch = TransformPatch.from_dict(on_fail)
        except ValueError as exc:
            self._reject(index, "invalid_on_fail", str(exc))
        if failure_patch.hook is not None and failure_patch.hook != patch.hook:
            self._reject(index, "invalid_on_fail", "validate on_fail must run at the same hook; omit hook inside on_fail.")
        if failure_patch.op.op not in hook_allowed_ops:
            self._reject(
                index,
                "invalid_on_fail",
                f"on_fail operation {failure_patch.op.op!r} is not allowed at hook {(patch.hook or 'on_task_start')!r}.",
            )
        if failure_patch.op.op == "rewrite_response":
            if "message" not in failure_patch.op.params and "replacement" not in failure_patch.op.params:
                self._reject(index, "invalid_on_fail", "rewrite_response on_fail requires message or replacement.")
        if failure_patch.op.op == "block_response" and "message" not in failure_patch.op.params:
            self._reject(index, "invalid_on_fail", "block_response on_fail requires message.")
        if failure_patch.op.op in {"set_state", "append_state", "merge_state", "clear_state", "expose_state", "hide_state"}:
            field = failure_patch.op.params.get("field")
            if not isinstance(field, str) or field not in state_fields:
                self._reject(index, "unknown_state_field", f"on_fail references undefined state field {field!r}.")

    def _validate_boundaries(self, index: int, patch: TransformPatch) -> None:
        payload_text = json.dumps(patch.to_dict(), sort_keys=True, default=str)
        for forbidden in ("hidden_task_goal", "hidden_label", "gold_answer", "evaluator"):
            if forbidden in payload_text:
                self._reject(index, "immutable_boundary_violation", f"Candidate references immutable boundary {forbidden!r}.")
        lowered_payload = payload_text.lower()
        for forbidden in FORBIDDEN_TRACE_LITERALS:
            if forbidden in lowered_payload:
                self._reject(
                    index,
                    "immutable_boundary_violation",
                    f"Candidate embeds evaluator or simulator artifact {forbidden!r}.",
                )
        condition_text = json.dumps({"when": patch.when, "unless": patch.unless}, sort_keys=True, default=str)
        if CASE_ID_CONDITION_PATTERN.search(condition_text):
            self._reject(index, "task_id_overfitting", "Candidate condition appears to branch on case/task id.")

    def _required_string(self, index: int, patch: TransformPatch, key: str) -> str:
        value = patch.op.params.get(key)
        if not isinstance(value, str) or not value:
            self._reject(index, "missing_required_field", f"Operation {patch.op.op!r} requires non-empty {key!r}.")
        return value

    def _context_changes(self, patches: tuple[TransformPatch, ...]) -> list[str]:
        rows = []
        for patch in patches:
            if patch.op.op in {
                "add_context_section",
                "remove_context_section",
                "replace_context_section",
                "move_context_section",
                "reorder_context_sections",
                "render_state_section",
            }:
                rows.append(patch.op.op)
        return rows

    def _warnings(self, patches: tuple[TransformPatch, ...], surface: SurfaceSpec) -> list[str]:
        warnings = []
        if any(patch.op.op == "rewrite_response" for patch in patches):
            warnings.append("response_rewrite may change user-facing style")
        if any(patch.op.op == "call_model" for patch in patches) and not surface.model.auxiliary_model_calls_allowed:
            warnings.append("auxiliary model calls are unavailable on this surface")
        return warnings

    def _reject(self, index: int, code: str, message: str) -> None:
        raise TransformCompileError(CompileIssue(code=code, message=message, patch_index=index))
