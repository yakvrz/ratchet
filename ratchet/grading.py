from __future__ import annotations

import json
import re
from typing import Any, Iterable

from ratchet.types import EvalCase, GradeResult


def normalize_text(text: str) -> str:
    lowered = text.strip().lower()
    lowered = lowered.replace("**", " ")
    lowered = re.sub(r"answer\s*:\s*", "", lowered)
    lowered = re.sub(r"[^a-z0-9:._-]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def extract_first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        return None
    return float(match.group(0))


def extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def output_text(output: Any, *, field: str | None = None) -> str:
    if field is not None:
        if not isinstance(output, dict):
            raise ValueError(f"Expected dict output to extract field {field!r}.")
        return str(output[field])
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if len(output) == 1:
            return str(next(iter(output.values())))
        raise ValueError("Dict output requires a field name for text grading.")
    return str(output)


def exact_text_grade(
    case: EvalCase,
    output: Any,
    *,
    field: str | None = None,
    expected: str | None = None,
    aliases: Iterable[str] = (),
) -> GradeResult:
    expected_values = []
    if expected is not None:
        expected_values.append(expected)
    elif case.expected is not None:
        expected_values.append(str(case.expected))
    expected_values.extend(str(alias) for alias in aliases)
    normalized_prediction = normalize_text(output_text(output, field=field))
    for candidate in expected_values:
        normalized_candidate = normalize_text(candidate)
        if normalized_candidate == normalized_prediction or normalized_candidate in normalized_prediction:
            return GradeResult(score=1.0, passed=True, labels=[])
    return GradeResult(
        score=0.0,
        passed=False,
        labels=["exact_text_mismatch"],
        notes=f"prediction={output!r}",
    )


def numeric_tolerance_grade(
    case: EvalCase,
    output: Any,
    *,
    field: str | None = None,
    expected: float | None = None,
    tolerance: float = 0.01,
) -> GradeResult:
    prediction = extract_first_number(output_text(output, field=field))
    if expected is None:
        if case.expected is None:
            raise ValueError("numeric_tolerance_grade requires an expected value.")
        expected = float(case.expected)
    if prediction is not None and abs(prediction - expected) <= tolerance:
        return GradeResult(score=1.0, passed=True, labels=[])
    return GradeResult(
        score=0.0,
        passed=False,
        labels=["numeric_mismatch"],
        notes=f"prediction={prediction!r} expected={expected:.4f} tolerance={tolerance:.4f}",
    )


def json_field_grade(
    case: EvalCase,
    output: Any,
    *,
    required_fields: Iterable[str],
    numeric_tolerances: dict[str, float] | None = None,
) -> GradeResult:
    if not isinstance(output, dict):
        return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes="Expected dict output.")
    expected_payload = case.expected
    if not isinstance(expected_payload, dict):
        raise ValueError("json_field_grade requires case.expected to be a dict.")
    numeric_tolerances = dict(numeric_tolerances or {})
    failures: list[str] = []
    checked_fields = list(required_fields)
    for field_name in checked_fields:
        if field_name not in output:
            failures.append(f"missing_field:{field_name}")
            continue
        expected_value = expected_payload.get(field_name)
        actual_value = output[field_name]
        if field_name in numeric_tolerances:
            try:
                actual_number = float(actual_value)
                expected_number = float(expected_value)
            except (TypeError, ValueError):
                failures.append(f"non_numeric_field:{field_name}")
                continue
            if abs(actual_number - expected_number) > numeric_tolerances[field_name]:
                failures.append(f"wrong_field:{field_name}")
            continue
        if actual_value != expected_value:
            failures.append(f"wrong_field:{field_name}")
    if not failures:
        return GradeResult(score=1.0, passed=True, labels=[])
    score = max(0.0, 1.0 - (len(failures) / max(len(checked_fields), 1)))
    return GradeResult(
        score=round(score, 4),
        passed=False,
        labels=failures,
        notes=f"output={output}",
    )
