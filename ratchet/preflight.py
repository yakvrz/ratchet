from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import tempfile
from typing import Any

from ratchet.adapters import AdapterProtocol, checked_agent_spec
from ratchet.model_client import ResponsesModelClient, validate_optimizer_model_access
from ratchet.surface import SurfaceGenerator
from ratchet.types import AgentPatch, EditableTarget, EvalCase, GradeResult, OptimizationObjective, PatchOperation, RunRecord
from ratchet.validation import PatchValidator


@dataclass(frozen=True)
class CheckSummary:
    adapter: str
    agent_spec: dict[str, Any] | None
    generated_surface: list[dict[str, Any]]
    sample_cases: list[dict[str, Any]]
    stability: dict[str, Any]
    materialization: dict[str, Any]
    optimizer_model_access: dict[str, Any]
    exported_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_check_cases(cases: tuple[EvalCase, ...], sample_limit: int = 2) -> tuple[EvalCase, ...]:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be at least 1.")
    dev_cases = [case for case in cases if case.split == "dev"]
    holdout_cases = [case for case in cases if case.split == "holdout"]
    if not dev_cases or not holdout_cases:
        raise ValueError("Eval file must include both dev and holdout cases for preflight check.")
    selected: list[EvalCase] = [dev_cases[0], holdout_cases[0]]
    remaining = [case for case in cases if case.id not in {selected[0].id, selected[1].id}]
    for case in remaining:
        if len(selected) >= sample_limit:
            break
        selected.append(case)
    return tuple(selected[:sample_limit])


def _run_checked_case(
    adapter: AdapterProtocol,
    case: EvalCase,
) -> tuple[RunRecord, GradeResult]:
    record = adapter.run_case(case, None)
    if not isinstance(record, RunRecord):
        raise TypeError(f"run_case returned {type(record).__name__}, expected RunRecord.")
    try:
        json.dumps(record.output, sort_keys=True)
    except TypeError as error:
        raise TypeError("run_case returned a non-JSON-serializable output.") from error
    grade = adapter.grade(case, record.output)
    if not isinstance(grade, GradeResult):
        raise TypeError(f"grade returned {type(grade).__name__}, expected GradeResult.")
    return record, grade


def _stability_summary(
    adapter: AdapterProtocol,
    sample_cases: tuple[EvalCase, ...],
) -> dict[str, Any]:
    first_pass: list[tuple[RunRecord, GradeResult]] = []
    second_pass: list[tuple[RunRecord, GradeResult]] = []
    for case in sample_cases:
        first_pass.append(_run_checked_case(adapter, case))
    for case in sample_cases:
        second_pass.append(_run_checked_case(adapter, case))

    pass_flips: list[str] = []
    output_drift: list[str] = []
    score_deltas: list[float] = []
    for case, (first_record, first_grade), (second_record, second_grade) in zip(sample_cases, first_pass, second_pass):
        if first_grade.passed != second_grade.passed:
            pass_flips.append(case.id)
        if first_record.output != second_record.output:
            output_drift.append(case.id)
        score_deltas.append(abs(first_grade.score - second_grade.score))
    mean_score_drift = sum(score_deltas) / max(len(score_deltas), 1)
    stable = not pass_flips and mean_score_drift <= 0.01
    if not stable:
        raise ValueError(
            "Preflight stability check failed: repeated baseline runs changed externally graded behavior."
        )
    return {
        "stable": stable,
        "pass_flip_case_ids": pass_flips,
        "output_drift_case_ids": output_drift,
        "mean_score_drift": round(mean_score_drift, 4),
    }


def run_preflight_check(
    *,
    adapter_spec: str,
    adapter: AdapterProtocol,
    cases: tuple[EvalCase, ...],
    objective: OptimizationObjective,
    sample_limit: int = 2,
    optimizer_model: str | None = None,
    optimizer_env_path: str = ".env",
    optimizer_client: ResponsesModelClient | None = None,
) -> CheckSummary:
    spec = checked_agent_spec(adapter, adapter_spec=adapter_spec)
    surface = SurfaceGenerator().generate(spec, objective)
    sample_cases = select_check_cases(cases, sample_limit=sample_limit)
    sample_rows: list[dict[str, Any]] = []
    for case in sample_cases:
        record, grade = _run_checked_case(adapter, case)
        sample_rows.append(
            {
                "case_id": case.id,
                "split": case.split,
                "score": grade.score,
                "passed": grade.passed,
                "error": record.metrics.error,
                "labels": list(grade.labels),
            }
        )

    stability = _stability_summary(adapter, sample_cases)
    materialization = _materialization_audit(adapter, spec, surface, objective)
    optimizer_model_access = (
        validate_optimizer_model_access(
            env_path=optimizer_env_path,
            model=optimizer_model,
            client=optimizer_client,
        )
        if optimizer_model is not None
        else {"checked": False, "reason": "optimizer_model was not provided"}
    )

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = Path(tmp) / "export"
        adapter.export(AgentPatch.empty(), export_dir)

    return CheckSummary(
        adapter=adapter_spec,
        agent_spec=spec.to_dict() if spec else None,
        generated_surface=[target.to_dict() for target in surface],
        sample_cases=sample_rows,
        stability=stability,
        materialization=materialization,
        optimizer_model_access=optimizer_model_access,
        exported_path="validated in temporary directory",
    )


def _materialization_audit(
    adapter: AdapterProtocol,
    spec: Any,
    surface: list[EditableTarget],
    objective: OptimizationObjective,
) -> dict[str, Any]:
    if not surface:
        return {"checked": False, "verified_kinds": [], "skipped_kinds": [], "checks": []}
    targets_by_kind: dict[str, EditableTarget] = {}
    for target in surface:
        targets_by_kind.setdefault(target.kind, target)
        if _preferred_materialization_target(target):
            targets_by_kind[target.kind] = target
    checks: list[dict[str, Any]] = []
    validator = PatchValidator()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index, (kind, target) in enumerate(sorted(targets_by_kind.items())):
            patch, expected = _sentinel_patch_for_target(target, index)
            is_valid, invalid_reason = validator.validate_with_reason(
                patch,
                current_spec=spec,
                surface=surface,
                objective=objective,
            )
            if not is_valid:
                checks.append(
                    {
                        "kind": kind,
                        "target": target.name,
                        "verified": False,
                        "reason": invalid_reason or "sentinel patch was invalid",
                    }
                )
                continue
            export_dir = root / kind
            adapter.export(patch, export_dir)
            exported_text = _exported_review_text(export_dir)
            verified = expected in exported_text
            checks.append(
                {
                    "kind": kind,
                    "target": target.name,
                    "verified": verified,
                    "expected": expected,
                    "reason": None if verified else "patched sentinel was not found in exported review artifacts",
                }
            )
    failed = [check for check in checks if not check["verified"]]
    if failed:
        failed_rows = ", ".join(f"{check['kind']}:{check['target']}" for check in failed)
        raise ValueError(f"Materialization audit failed for generated targets: {failed_rows}")
    return {
        "checked": True,
        "verified_kinds": [str(check["kind"]) for check in checks],
        "skipped_kinds": [],
        "checks": checks,
    }


def _preferred_materialization_target(target: EditableTarget) -> bool:
    schema_types = _schema_types(target)
    return bool(schema_types & {"string", "object", "array"}) or target.kind in {"model", "verifier"}


def _sentinel_patch_for_target(target: EditableTarget, index: int) -> tuple[AgentPatch, str]:
    op = _sentinel_op(target)
    sentinel = f"RATCHET_MATERIALIZATION_SENTINEL_{target.kind}_{index}"
    schema_types = _schema_types(target)
    if op == "change_model":
        value = target.choices[0]
        expected = f'"model": "{value}"'
    elif op == "add_few_shot":
        value = [
            {
                "source_case_id": "preflight-sentinel",
                "input": sentinel,
                "output": {"label": sentinel},
                "purpose": "Preflight materialization audit sentinel.",
            }
        ]
        expected = sentinel
    elif schema_types == {"boolean"}:
        value = not bool(target.current_value)
        leaf = target.path.rsplit(".", 1)[-1]
        expected = f'"{leaf}": {str(value).lower()}'
    elif schema_types == {"integer"}:
        value = 912345 + index
        expected = str(value)
    elif schema_types == {"number"}:
        value = 912345.25 + index
        expected = str(value)
    elif "object" in schema_types:
        value = {"ratchet_materialization_sentinel": sentinel}
        expected = sentinel
    elif "array" in schema_types:
        value = [{"ratchet_materialization_sentinel": sentinel}]
        expected = sentinel
    else:
        value = sentinel
        expected = sentinel
    return (
        AgentPatch(
            operations=[
                PatchOperation(
                    op=op,
                    target=target.name,
                    value=value,
                    rationale="Preflight materialization audit sentinel.",
                )
            ],
            rationale="Preflight materialization audit sentinel.",
            expected_effect="Verify adapter export materializes generated targets.",
        ),
        expected,
    )


def _schema_types(target: EditableTarget) -> set[str]:
    schema_type = target.value_schema.get("type")
    if isinstance(schema_type, list):
        return {str(item) for item in schema_type}
    if schema_type is None:
        return set()
    return {str(schema_type)}


def _sentinel_op(target: EditableTarget) -> str:
    for op in (
        "change_model",
        "add_few_shot",
        "add_instruction",
        "revise_instruction",
        "add_output_constraint",
        "revise_tool_description",
        "revise_tool_policy",
        "set_runtime_param",
        "add_verifier_retry",
    ):
        if op in target.allowed_ops:
            return op
    return target.allowed_ops[0]


def _exported_review_text(root: Path) -> str:
    rows: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "patch.json"):
        try:
            rows.append(path.read_text())
        except UnicodeDecodeError:
            continue
    return "\n".join(rows)
