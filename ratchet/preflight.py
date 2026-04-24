from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import tempfile
from typing import Any

from ratchet.adapters import AdapterProtocol
from ratchet.code_artifacts import compile_code_artifact
from ratchet.io import normalize_candidate
from ratchet.types import EvalCase, GradeResult, RunRecord, SearchSpace


@dataclass(frozen=True)
class CheckSummary:
    adapter: str
    baseline_candidate: dict[str, str]
    sample_cases: list[dict[str, Any]]
    stability: dict[str, Any]
    exported_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_search_space(search_space: SearchSpace) -> None:
    all_specs = search_space.all_specs()
    if not all_specs:
        raise ValueError("Adapter search_space() returned no knobs.")
    knob_names = [spec.name for spec in all_specs]
    duplicates = sorted({name for name in knob_names if knob_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Adapter search_space() contains duplicate knob names: {', '.join(duplicates)}")
    for spec in search_space.code_artifacts:
        compile_code_artifact(spec, spec.default)


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
    return tuple(selected)


def _run_checked_case(
    adapter: AdapterProtocol,
    baseline_candidate: dict[str, str],
    case: EvalCase,
) -> tuple[RunRecord, GradeResult]:
    record = adapter.run_case(baseline_candidate, case)
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
    baseline_candidate: dict[str, str],
    sample_cases: tuple[EvalCase, ...],
) -> dict[str, Any]:
    first_pass: list[tuple[RunRecord, GradeResult]] = []
    second_pass: list[tuple[RunRecord, GradeResult]] = []
    for case in sample_cases:
        first_pass.append(_run_checked_case(adapter, baseline_candidate, case))
    for case in sample_cases:
        second_pass.append(_run_checked_case(adapter, baseline_candidate, case))

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
    search_space: SearchSpace,
    cases: tuple[EvalCase, ...],
    sample_limit: int = 2,
) -> CheckSummary:
    validate_search_space(search_space)
    baseline_candidate = normalize_candidate(adapter.baseline(), search_space)
    sample_cases = select_check_cases(cases, sample_limit=sample_limit)
    sample_rows: list[dict[str, Any]] = []
    for case in sample_cases:
        record, grade = _run_checked_case(adapter, baseline_candidate, case)
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

    stability = _stability_summary(adapter, baseline_candidate, sample_cases)

    with tempfile.TemporaryDirectory() as tmp:
        export_dir = Path(tmp) / "export"
        adapter.export(baseline_candidate, export_dir)

    return CheckSummary(
        adapter=adapter_spec,
        baseline_candidate=baseline_candidate,
        sample_cases=sample_rows,
        stability=stability,
        exported_path="validated in temporary directory",
    )
