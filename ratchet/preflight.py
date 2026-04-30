from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import tempfile
from typing import Any

from ratchet.adapters import AdapterProtocol, checked_agent_spec, checked_surface_spec
from ratchet.model_client import ResponsesModelClient, validate_optimizer_model_access
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.types import EvalCase, GradeResult, OptimizationObjective, RunRecord


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
    first_pass: list[tuple[RunRecord, GradeResult]] | None = None,
) -> dict[str, Any]:
    first_pass_rows: list[tuple[RunRecord, GradeResult]] = list(first_pass or [])
    second_pass: list[tuple[RunRecord, GradeResult]] = []
    for case in sample_cases[len(first_pass_rows):]:
        first_pass_rows.append(_run_checked_case(adapter, case))
    for case in sample_cases:
        second_pass.append(_run_checked_case(adapter, case))

    pass_flips: list[str] = []
    output_drift: list[str] = []
    score_deltas: list[float] = []
    for case, (first_record, first_grade), (second_record, second_grade) in zip(sample_cases, first_pass_rows, second_pass):
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
    surface = checked_surface_spec(adapter, adapter_spec=adapter_spec)
    sample_cases = select_check_cases(cases, sample_limit=sample_limit)
    sample_rows: list[dict[str, Any]] = []
    first_pass: list[tuple[RunRecord, GradeResult]] = []
    for case in sample_cases:
        record, grade = _run_checked_case(adapter, case)
        first_pass.append((record, grade))
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

    stability = _stability_summary(adapter, sample_cases, first_pass=first_pass)
    materialization = _materialization_audit(adapter, surface, sample_case=sample_cases[0])
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
        adapter.export(None, export_dir)

    return CheckSummary(
        adapter=adapter_spec,
        agent_spec=spec.to_dict() if spec else None,
        generated_surface=[surface.to_dict()],
        sample_cases=sample_rows,
        stability=stability,
        materialization=materialization,
        optimizer_model_access=optimizer_model_access,
        exported_path="validated in temporary directory",
    )


def _materialization_audit(
    adapter: AdapterProtocol,
    surface: SurfaceSpec,
    *,
    sample_case: EvalCase,
) -> dict[str, Any]:
    compiler = TransformCompiler()
    raw_programs = _sentinel_transform_programs(surface)
    execution_programs = _sentinel_execution_programs(surface)
    if not raw_programs and not execution_programs:
        return {
            "checked": False,
            "verified_surfaces": [],
            "skipped_surfaces": [],
            "checks": [],
            "execution_checks": [],
        }
    checks: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index, (surface_name, raw_program, expected) in enumerate(raw_programs):
            candidate = compiler.compile_or_raise(TransformProgram.from_dict(raw_program), surface)
            export_dir = root / surface_name
            adapter.export(candidate, export_dir)
            exported_text = _exported_review_text(export_dir)
            verified = expected in exported_text
            checks.append(
                {
                    "surface": surface_name,
                    "verified": verified,
                    "expected": expected,
                    "reason": None if verified else "compiled transform sentinel was not found in exported artifacts",
                }
            )
    execution_checks: list[dict[str, Any]] = []
    for surface_name, raw_program, expected in execution_programs:
        candidate = compiler.compile_or_raise(TransformProgram.from_dict(raw_program), surface)
        record = adapter.run_case(sample_case, candidate)
        if not isinstance(record, RunRecord):
            raise TypeError(f"run_case returned {type(record).__name__}, expected RunRecord.")
        output_text = json.dumps(record.output, sort_keys=True, default=str)
        trace = record.diagnostics.metadata.get("transform_trace", [])
        trace_text = json.dumps(trace, sort_keys=True, default=str)
        verified = expected in output_text or expected in trace_text
        execution_checks.append(
            {
                "surface": surface_name,
                "verified": verified,
                "expected": expected,
                "reason": None if verified else "compiled transform sentinel was not observed during run_case execution",
            }
        )
    failed = [check for check in checks if not check["verified"]]
    failed.extend(check for check in execution_checks if not check["verified"])
    if failed:
        failed_rows = ", ".join(str(check["surface"]) for check in failed)
        raise ValueError(f"Materialization audit failed for transform surfaces: {failed_rows}")
    return {
        "checked": True,
        "verified_surfaces": [str(check["surface"]) for check in checks],
        "skipped_surfaces": [],
        "checks": checks,
        "execution_checks": execution_checks,
    }


def _sentinel_transform_programs(surface: SurfaceSpec) -> list[tuple[str, dict[str, Any], str]]:
    programs: list[tuple[str, dict[str, Any], str]] = []
    sentinel = "RATCHET_TRANSFORM_SENTINEL_CONTEXT"
    editable = set(surface.context.editable_sections)
    section = next((item.name for item in surface.context.graph.sections if item.name in editable), None)
    if section is not None:
        programs.append(
            (
                "context",
                {
                    "id": "preflight_context_sentinel",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "replace_context_section",
                            "section": section,
                            "content": sentinel,
                        }
                    ],
                },
                sentinel,
            )
        )
    if surface.context.generated_sections_allowed:
        generated_sentinel = "RATCHET_TRANSFORM_SENTINEL_GENERATED_SECTION"
        programs.append(
            (
                "generated_context",
                {
                    "id": "preflight_generated_context_sentinel",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "add_context_section",
                            "section": "preflight_generated_context_sentinel",
                            "content": generated_sentinel,
                            "position": "end",
                        }
                    ],
                },
                generated_sentinel,
            )
        )
    if surface.model.max_tokens_configurable:
        programs.append(
            (
                "model_config",
                {
                    "id": "preflight_model_config_sentinel",
                    "patches": [
                        {
                            "hook": "before_model_call",
                            "op": "set_model_config",
                            "field": "max_tokens",
                            "value": 321,
                        }
                    ],
                },
                '"field": "max_tokens"',
            )
        )
    return programs


def _sentinel_execution_programs(surface: SurfaceSpec) -> list[tuple[str, dict[str, Any], str]]:
    hook = surface.hooks["before_user_response"]
    if not hook.supported or "rewrite_response" not in hook.allowed_ops:
        return []
    sentinel = "RATCHET_TRANSFORM_SENTINEL_RESPONSE"
    return [
        (
            "response_execution",
            {
                "id": "preflight_response_execution_sentinel",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "rewrite_response",
                        "message": sentinel,
                    }
                ],
            },
            sentinel,
        )
    ]


def _exported_review_text(root: Path) -> str:
    rows: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "patch.json"):
        try:
            rows.append(path.read_text())
        except UnicodeDecodeError:
            continue
    return "\n".join(rows)
