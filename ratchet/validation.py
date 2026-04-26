from __future__ import annotations

import json
import re
from typing import Any

from ratchet.types import AgentPatch, AgentSpec, EditableTarget, EvalCase, OptimizationObjective


UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
LONG_NUMBER_PATTERN = re.compile(r"\b\d{6,}\b")
MIN_METADATA_COPY_CHARS = 12
MIN_LONG_COPY_CHARS = 40
LONG_COPY_WORDS = 8


def _schema_type_matches(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "null":
        return value is None
    return True


def _value_schema_error(value: Any, schema: dict[str, Any]) -> str | None:
    if not schema:
        return None
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if not any(_schema_type_matches(value, str(item)) for item in schema_type):
            return f"value does not match allowed schema types {schema_type}"
    elif schema_type is not None and not _schema_type_matches(value, str(schema_type)):
        return f"value does not match schema type {schema_type!r}"
    if "enum" in schema and value not in schema["enum"]:
        return "value is not one of the target enum choices"
    if isinstance(value, str) and schema.get("maxLength") is not None:
        if len(value) > int(schema["maxLength"]):
            return "string value exceeds target maxLength"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if schema.get("minimum") is not None and value < float(schema["minimum"]):
            return "numeric value is below target minimum"
        if schema.get("maximum") is not None and value > float(schema["maximum"]):
            return "numeric value is above target maximum"
    return None


class PatchValidator:
    def validate_with_reason(
        self,
        patch: AgentPatch,
        *,
        current_spec: AgentSpec | None,
        surface: list[EditableTarget],
        objective: OptimizationObjective,
        evidence_cases: list[EvalCase] | None = None,
    ) -> tuple[bool, str | None]:
        if patch.is_empty:
            return False, "empty patch"
        if len(patch.operations) > objective.constraints.max_patch_operations:
            return False, "patch exceeds max_patch_operations"
        target_by_name = {target.name: target for target in surface}
        target_by_path = {target.path: target for target in surface}
        seen_targets: set[str] = set()
        for operation in patch.operations:
            target = target_by_name.get(operation.target) or target_by_path.get(operation.target)
            if target is None:
                return False, f"unknown target {operation.target!r}"
            if target.name in seen_targets:
                return False, f"duplicate target {target.name!r}"
            seen_targets.add(target.name)
            if operation.op not in target.allowed_ops:
                return False, f"operation {operation.op!r} is not allowed for target {target.name!r}"
            if target.max_chars is not None and isinstance(operation.value, str):
                if len(operation.value) > target.max_chars:
                    return False, f"value for target {target.name!r} exceeds max_chars"
            if target.choices and str(operation.value) not in target.choices:
                return False, f"value for target {target.name!r} is not an allowed choice"
            schema_error = _value_schema_error(operation.value, target.value_schema)
            if schema_error is not None:
                return False, f"{target.name}: {schema_error}"
            evidence_error = _eval_evidence_copy_error(
                operation.value,
                target.current_value,
                evidence_cases or [],
            )
            if evidence_error is not None:
                return False, evidence_error
            if operation.op == "change_model" and target.choices:
                if str(operation.value) not in target.choices:
                    return False, "model value is not in the allowed model set"
        if current_spec is not None:
            try:
                current_spec.apply_patch(patch)
            except Exception as exc:
                return False, f"patch does not apply cleanly: {exc}"
        return True, None

    def validate(
        self,
        patch: AgentPatch,
        *,
        current_spec: AgentSpec | None,
        surface: list[EditableTarget],
        objective: OptimizationObjective,
        evidence_cases: list[EvalCase] | None = None,
    ) -> bool:
        is_valid, _ = self.validate_with_reason(
            patch,
            current_spec=current_spec,
            surface=surface,
            objective=objective,
            evidence_cases=evidence_cases,
        )
        return is_valid


def _eval_evidence_copy_error(value: Any, current_value: Any, cases: list[EvalCase]) -> str | None:
    if not cases:
        return None
    patch_text = _stringify_value(value)
    current_text = _stringify_value(current_value)
    patch_norm = _normalize_copy_text(patch_text)
    current_norm = _normalize_copy_text(current_text)
    for case in _unique_cases(cases):
        case_id = str(case.id)
        if case_id and _contains_new_literal(patch_text, current_text, case_id):
            return f"patch value copies eval case id {case_id!r}"
        for label, text in _case_evidence_texts(case):
            for token in UUID_PATTERN.findall(text):
                if _contains_new_literal(patch_text, current_text, token):
                    return f"patch value copies eval {label} UUID"
            for token in LONG_NUMBER_PATTERN.findall(text):
                if _contains_new_literal(patch_text, current_text, token):
                    return f"patch value copies eval {label} numeric identifier"
            if label == "metadata" and len(text.strip()) >= MIN_METADATA_COPY_CHARS:
                if _contains_new_literal(patch_text, current_text, text.strip()):
                    return "patch value copies eval metadata value"
            copied = _copied_long_evidence_fragment(patch_norm, current_norm, text)
            if copied is not None:
                return f"patch value copies eval {label} fragment"
    return None


def _stringify_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def _unique_cases(cases: list[EvalCase]) -> list[EvalCase]:
    seen: set[str] = set()
    unique: list[EvalCase] = []
    for case in cases:
        if case.id in seen:
            continue
        seen.add(case.id)
        unique.append(case)
    return unique


def _case_evidence_texts(case: EvalCase) -> list[tuple[str, str]]:
    rows = [("input", str(case.input))]
    if case.expected is not None:
        rows.extend(("expected", item) for item in _flatten_text(case.expected))
    rows.extend(("metadata", item) for item in _flatten_text(case.metadata))
    return [(label, text) for label, text in rows if text]


def _flatten_text(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, dict):
        rows: list[str] = []
        for key, item in value.items():
            rows.append(str(key))
            rows.extend(_flatten_text(item))
        return rows
    if isinstance(value, list):
        rows = []
        for item in value:
            rows.extend(_flatten_text(item))
        return rows
    return [str(value)]


def _contains_new_literal(patch_text: str, current_text: str, literal: str) -> bool:
    if not literal:
        return False
    return literal in patch_text and literal not in current_text


def _normalize_copy_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _copied_long_evidence_fragment(
    patch_norm: str,
    current_norm: str,
    evidence_text: str,
) -> str | None:
    evidence_norm = _normalize_copy_text(evidence_text)
    if len(evidence_norm) < MIN_LONG_COPY_CHARS:
        return None
    if evidence_norm in patch_norm and evidence_norm not in current_norm:
        return evidence_norm
    words = evidence_norm.split()
    if len(words) < LONG_COPY_WORDS:
        return None
    for index in range(0, len(words) - LONG_COPY_WORDS + 1):
        fragment = " ".join(words[index : index + LONG_COPY_WORDS])
        if len(fragment) < MIN_LONG_COPY_CHARS:
            continue
        if fragment in patch_norm and fragment not in current_norm:
            return fragment
    return None
