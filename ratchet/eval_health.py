from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import contextlib
import json
import re
import signal
import statistics
from typing import Any, Iterable, Literal

from ratchet.adapters import AdapterProtocol
from ratchet.config import EvalHealthConfig
from ratchet.types import EvalCase, GradeResult, RunRecord


Severity = Literal["fatal", "warning", "info"]
HealthStatus = Literal["healthy", "warning", "fatal"]


@dataclass(frozen=True)
class EvalHealthIssue:
    severity: Severity
    code: str
    message: str
    case_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvalHealthReport:
    status: HealthStatus
    issues: list[EvalHealthIssue]
    split_summary: dict[str, Any]
    label_summary: dict[str, Any]
    category_summary: dict[str, Any]
    leakage_summary: dict[str, Any]
    baseline_probe: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision": _health_decision(self.status),
            "summary": _report_summary(self),
            "issues": [issue.to_dict() for issue in self.issues],
            "split_summary": self.split_summary,
            "runtime_feasibility": (self.baseline_probe.get("runtime_feasibility") or {}),
            "baseline_probe": self.baseline_probe,
            "label_summary": self.label_summary,
            "category_summary": self.category_summary,
            "leakage_summary": self.leakage_summary,
        }

    @property
    def fatal(self) -> bool:
        return any(issue.severity == "fatal" for issue in self.issues)

    @property
    def warning(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)


def run_eval_health_check(
    *,
    adapter_spec: str,
    adapter: AdapterProtocol,
    cases: tuple[EvalCase, ...],
    config: EvalHealthConfig | None = None,
    sample_limit: int | None = None,
    repeats: int | None = None,
    case_timeout_s: int = 180,
    evaluation_samples_per_case: int = 1,
    case_concurrency: int = 1,
) -> EvalHealthReport:
    health_config = config or EvalHealthConfig()
    effective_sample_limit = health_config.sample_limit if sample_limit is None else sample_limit
    effective_repeats = health_config.repeats if repeats is None else repeats
    effective_config = EvalHealthConfig(
        sample_limit=effective_sample_limit,
        repeats=effective_repeats,
        min_dev_cases=health_config.min_dev_cases,
        min_holdout_cases=health_config.min_holdout_cases,
        min_cases_per_category=health_config.min_cases_per_category,
        max_runtime_error_rate=health_config.max_runtime_error_rate,
        max_unstable_case_rate=health_config.max_unstable_case_rate,
        max_mean_latency_s=health_config.max_mean_latency_s,
        max_p95_latency_s=health_config.max_p95_latency_s,
        max_mean_cost_usd=health_config.max_mean_cost_usd,
        max_estimated_eval_cost_usd=health_config.max_estimated_eval_cost_usd,
        max_estimated_eval_wall_time_s=health_config.max_estimated_eval_wall_time_s,
        max_estimated_eval_tokens=health_config.max_estimated_eval_tokens,
    )
    issues: list[EvalHealthIssue] = []
    split_summary = _split_summary(cases)
    label_summary = _label_summary(cases)
    category_summary = _category_summary(cases)
    leakage_summary = _leakage_summary(cases)

    issues.extend(_static_issues(cases, effective_config, split_summary, label_summary, category_summary, leakage_summary))
    baseline_probe = _baseline_probe(
        adapter=adapter,
        cases=cases,
        config=effective_config,
        case_timeout_s=case_timeout_s,
        evaluation_samples_per_case=evaluation_samples_per_case,
        case_concurrency=case_concurrency,
    )
    issues.extend(_probe_issues(baseline_probe, effective_config))
    status = _status(issues)
    return EvalHealthReport(
        status=status,
        issues=issues,
        split_summary={
            **split_summary,
            "adapter": adapter_spec,
            "config": effective_config.to_dict(),
        },
        label_summary=label_summary,
        category_summary=category_summary,
        leakage_summary=leakage_summary,
        baseline_probe=baseline_probe,
    )


def render_eval_health_markdown(report: EvalHealthReport) -> str:
    payload = report.to_dict()
    issue_counts = payload["summary"]["issue_counts"]
    split_counts = report.split_summary.get("by_split", {})
    probe = report.baseline_probe
    runtime = dict(probe.get("runtime_feasibility") or {})
    estimated = dict(runtime.get("estimated_eval_sweep") or {})
    observed = dict(runtime.get("observed_per_case") or {})
    model_rows = runtime.get("models") or {}
    lines = [
        "# Eval Health Report",
        "",
        f"Status: `{report.status}`",
        f"Decision: {payload['decision']}",
        "",
        "## Summary",
        "",
        f"- Total cases: {report.split_summary.get('total_cases', 0)}",
        (
            "- Splits: "
            f"train={split_counts.get('train', 0)} "
            f"dev={split_counts.get('dev', 0)} "
            f"holdout={split_counts.get('holdout', 0)}"
        ),
        f"- Issues: fatal={issue_counts['fatal']} warning={issue_counts['warning']} info={issue_counts['info']}",
    ]
    if probe.get("checked"):
        lines.extend(
            [
                (
                    "- Baseline probe: "
                    f"{len(probe.get('sampled_case_ids') or [])} cases x {probe.get('repeats')} repeats, "
                    f"pass_rate={float(probe.get('pass_rate') or 0.0):.3f}, "
                    f"errors={int(probe.get('error_attempt_count') or 0)}/{int(probe.get('attempt_count') or 0)}, "
                    f"unstable={int(probe.get('unstable_case_count') or 0)}"
                ),
                (
                    "- Runtime estimate: "
                    f"{estimated.get('case_attempts', 0)} attempts, "
                    f"{float(estimated.get('wall_time_s') or 0.0):.1f}s wall, "
                    f"${float(estimated.get('cost_usd') or 0.0):.6f}, "
                    f"{int(estimated.get('total_tokens') or 0)} tokens"
                ),
            ]
        )
    else:
        lines.append(f"- Baseline probe: skipped ({probe.get('reason')})")

    lines.extend(["", "## Issues", ""])
    if not report.issues:
        lines.append("- none")
    for issue in report.issues:
        case_text = f" cases={','.join(issue.case_ids[:8])}" if issue.case_ids else ""
        lines.append(f"- `{issue.severity}` `{issue.code}`: {issue.message}{case_text}")

    lines.extend(
        [
            "",
            "## Runtime Feasibility",
            "",
            f"- Pricing basis: `{runtime.get('pricing_basis', 'unknown')}`",
            f"- Case concurrency: {runtime.get('case_concurrency', 'n/a')}",
            f"- Runnable cases: {runtime.get('runnable_case_count', 'n/a')}",
            f"- Samples per case: {runtime.get('samples_per_case', 'n/a')}",
        ]
    )
    if estimated:
        lines.extend(
            [
                f"- Estimated attempts: {estimated.get('case_attempts')}",
                f"- Estimated wall time: {float(estimated.get('wall_time_s') or 0.0):.3f}s",
                f"- Estimated serial time: {float(estimated.get('serial_time_s') or 0.0):.3f}s",
                f"- Estimated cost: ${float(estimated.get('cost_usd') or 0.0):.6f}",
                f"- Estimated tokens: {int(estimated.get('total_tokens') or 0)}",
            ]
        )
    latency = observed.get("latency_s") or {}
    cost = observed.get("cost_usd") or {}
    tokens = observed.get("total_tokens") or {}
    lines.extend(
        [
            f"- Observed latency: mean={_format_optional_number(latency.get('mean'))}s p95={_format_optional_number(latency.get('p95'))}s max={_format_optional_number(latency.get('max'))}s",
            f"- Observed cost: mean=${_format_optional_number(cost.get('mean'), digits=6)} p95=${_format_optional_number(cost.get('p95'), digits=6)} max=${_format_optional_number(cost.get('max'), digits=6)}",
            f"- Observed tokens: mean={_format_optional_number(tokens.get('mean'))} p95={_format_optional_number(tokens.get('p95'))} max={_format_optional_number(tokens.get('max'))}",
            "",
            "### Probe Models",
            "",
        ]
    )
    if not model_rows:
        lines.append("- none recorded")
    else:
        lines.extend(
            [
                "| Model | Attempts | Mean latency | P95 latency | Mean cost | Mean tokens | Pass rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for model, row in sorted(model_rows.items()):
            lines.append(
                "| "
                f"`{model}` | "
                f"{row.get('attempt_count', 0)} | "
                f"{_format_optional_number(row.get('mean_latency_s'))}s | "
                f"{_format_optional_number(row.get('p95_latency_s'))}s | "
                f"${_format_optional_number(row.get('mean_cost_usd'), digits=6)} | "
                f"{_format_optional_number(row.get('mean_total_tokens'))} | "
                f"{_format_optional_number(row.get('pass_rate'), digits=3)} |"
            )

    lines.extend(
        [
            "",
            "## Coverage And Balance",
            "",
            "| Split | Cases | Labels | Categories | Min/category | Max/category |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for split in ("train", "dev", "holdout"):
        categories = (report.category_summary.get("by_split") or {}).get(split, {})
        labels = (report.label_summary.get("by_split") or {}).get(split, {})
        category_counts = [int(value) for value in categories.values()]
        lines.append(
            "| "
            f"{split} | "
            f"{split_counts.get(split, 0)} | "
            f"{len(labels)} | "
            f"{len(categories)} | "
            f"{min(category_counts) if category_counts else 0} | "
            f"{max(category_counts) if category_counts else 0} |"
        )

    leakage = report.leakage_summary
    lines.extend(
        [
            "",
            "## Leakage And Duplicates",
            "",
            f"- Normalized duplicate inputs: {len(leakage.get('duplicate_normalized_inputs') or [])}",
            f"- Normalized duplicate input+expected pairs: {len(leakage.get('duplicate_normalized_input_expected') or [])}",
            f"- Train to eval leaks: {len(leakage.get('train_to_eval_leaks') or [])}",
            f"- Dev/holdout overlaps: {len(leakage.get('dev_holdout_overlaps') or [])}",
            "",
            "## Sampled Cases",
            "",
        ]
    )
    if not probe.get("attempts"):
        lines.append("- none")
    else:
        lines.extend(
            [
                "| Case | Split | Repeat | Passed | Score | Latency | Cost | Tokens | Model |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for attempt in probe.get("attempts", []):
            lines.append(
                "| "
                f"`{attempt.get('case_id')}` | "
                f"{attempt.get('split')} | "
                f"{attempt.get('repeat_index')} | "
                f"{attempt.get('passed', 'error')} | "
                f"{_format_optional_number(attempt.get('score'), digits=3)} | "
                f"{_format_optional_number(attempt.get('latency_s'))}s | "
                f"${_format_optional_number(attempt.get('cost_usd'), digits=6)} | "
                f"{attempt.get('total_tokens', 0)} | "
                f"`{_attempt_model(attempt)}` |"
            )
    return "\n".join(lines) + "\n"


def _split_summary(cases: tuple[EvalCase, ...]) -> dict[str, Any]:
    counts = Counter(case.split for case in cases)
    return {
        "total_cases": len(cases),
        "by_split": dict(sorted(counts.items())),
    }


def _label_summary(cases: tuple[EvalCase, ...]) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        by_split[case.split][_expected_label(case.expected)] += 1
    labels_by_split = {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(by_split.items())
    }
    all_labels = sorted({label for counts in by_split.values() for label in counts})
    missing_by_split = {
        split: sorted(set(all_labels) - set(counts))
        for split, counts in sorted(by_split.items())
    }
    return {
        "by_split": labels_by_split,
        "unique_labels": all_labels,
        "missing_labels_by_split": missing_by_split,
    }


def _category_summary(cases: tuple[EvalCase, ...]) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        by_split[case.split][_category(case)] += 1
    categories_by_split = {
        split: dict(sorted(counts.items()))
        for split, counts in sorted(by_split.items())
    }
    all_categories = sorted({category for counts in by_split.values() for category in counts})
    missing_by_split = {
        split: sorted(set(all_categories) - set(counts))
        for split, counts in sorted(by_split.items())
    }
    return {
        "by_split": categories_by_split,
        "unique_categories": all_categories,
        "missing_categories_by_split": missing_by_split,
    }


def _leakage_summary(cases: tuple[EvalCase, ...]) -> dict[str, Any]:
    by_input: dict[str, list[EvalCase]] = defaultdict(list)
    by_input_expected: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        normalized_input = _normalize_text(case.input)
        by_input[normalized_input].append(case)
        by_input_expected[f"{normalized_input}\0{_expected_label(case.expected)}"].append(case)
    duplicate_inputs = _duplicate_rows(by_input.values())
    duplicate_input_expected = _duplicate_rows(by_input_expected.values())
    train_leaks = [
        row
        for row in duplicate_input_expected
        if "train" in row["splits"] and bool(set(row["splits"]) & {"dev", "holdout"})
    ]
    dev_holdout_overlaps = [
        row
        for row in duplicate_input_expected
        if "dev" in row["splits"] and "holdout" in row["splits"]
    ]
    return {
        "duplicate_normalized_inputs": duplicate_inputs,
        "duplicate_normalized_input_expected": duplicate_input_expected,
        "train_to_eval_leaks": train_leaks,
        "dev_holdout_overlaps": dev_holdout_overlaps,
    }


def _static_issues(
    cases: tuple[EvalCase, ...],
    config: EvalHealthConfig,
    split_summary: dict[str, Any],
    label_summary: dict[str, Any],
    category_summary: dict[str, Any],
    leakage_summary: dict[str, Any],
) -> list[EvalHealthIssue]:
    issues: list[EvalHealthIssue] = []
    counts = dict(split_summary.get("by_split") or {})
    if counts.get("dev", 0) < config.min_dev_cases:
        issues.append(
            EvalHealthIssue(
                severity="fatal",
                code="dev_split_too_small",
                message=f"Dev split has {counts.get('dev', 0)} cases; minimum is {config.min_dev_cases}.",
                metadata={"dev_cases": counts.get("dev", 0), "min_dev_cases": config.min_dev_cases},
            )
        )
    if counts.get("holdout", 0) < config.min_holdout_cases:
        severity: Severity = "fatal" if counts.get("holdout", 0) == 0 else "warning"
        issues.append(
            EvalHealthIssue(
                severity=severity,
                code="holdout_split_too_small",
                message=f"Holdout split has {counts.get('holdout', 0)} cases; recommended minimum is {config.min_holdout_cases}.",
                metadata={"holdout_cases": counts.get("holdout", 0), "min_holdout_cases": config.min_holdout_cases},
            )
        )
    for split in ("dev", "holdout"):
        missing_labels = (label_summary.get("missing_labels_by_split") or {}).get(split, [])
        if missing_labels and counts.get(split, 0):
            issues.append(
                EvalHealthIssue(
                    severity="warning",
                    code=f"{split}_missing_labels",
                    message=f"{split} split is missing {len(missing_labels)} label(s) present elsewhere.",
                    metadata={"labels": missing_labels[:20], "omitted": max(0, len(missing_labels) - 20)},
                )
            )
        missing_categories = (category_summary.get("missing_categories_by_split") or {}).get(split, [])
        if missing_categories and counts.get(split, 0):
            issues.append(
                EvalHealthIssue(
                    severity="warning",
                    code=f"{split}_missing_categories",
                    message=f"{split} split is missing {len(missing_categories)} categor(y/ies) present elsewhere.",
                    metadata={"categories": missing_categories[:20], "omitted": max(0, len(missing_categories) - 20)},
                )
            )
        low_support = _low_support_categories(category_summary, split, config.min_cases_per_category)
        if low_support:
            issues.append(
                EvalHealthIssue(
                    severity="warning",
                    code=f"{split}_low_category_support",
                    message=f"{split} split has {len(low_support)} categor(y/ies) below {config.min_cases_per_category} cases.",
                    metadata={"categories": low_support[:20], "omitted": max(0, len(low_support) - 20)},
                )
            )
    train_leaks = leakage_summary.get("train_to_eval_leaks") or []
    if train_leaks:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="train_eval_leakage",
                message=f"Found {len(train_leaks)} normalized train input/expected duplicate(s) in dev or holdout.",
                case_ids=_case_ids_from_duplicate_rows(train_leaks)[:20],
                metadata={"duplicates": train_leaks[:10], "omitted": max(0, len(train_leaks) - 10)},
            )
        )
    dev_holdout_overlaps = leakage_summary.get("dev_holdout_overlaps") or []
    if dev_holdout_overlaps:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="dev_holdout_overlap",
                message=f"Found {len(dev_holdout_overlaps)} normalized dev/holdout input/expected duplicate(s).",
                case_ids=_case_ids_from_duplicate_rows(dev_holdout_overlaps)[:20],
                metadata={"duplicates": dev_holdout_overlaps[:10], "omitted": max(0, len(dev_holdout_overlaps) - 10)},
            )
        )
    if not cases:
        issues.append(
            EvalHealthIssue(
                severity="fatal",
                code="empty_eval_file",
                message="Eval file contains no cases.",
            )
        )
    return issues


def _baseline_probe(
    *,
    adapter: AdapterProtocol,
    cases: tuple[EvalCase, ...],
    config: EvalHealthConfig,
    case_timeout_s: int,
    evaluation_samples_per_case: int,
    case_concurrency: int,
) -> dict[str, Any]:
    selected_cases = _select_probe_cases(cases, limit=config.sample_limit)
    if config.sample_limit == 0 or config.repeats == 0:
        return {
            "checked": False,
            "reason": "baseline probes disabled by eval_health sample_limit=0 or repeats=0",
            "sampled_case_ids": [],
            "repeats": config.repeats,
            "runtime_feasibility": _runtime_feasibility(
                cases=cases,
                attempts=[],
                latency_values=[],
                cost_values=[],
                token_values=[],
                evaluation_samples_per_case=evaluation_samples_per_case,
                case_concurrency=case_concurrency,
            ),
        }
    attempts: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in selected_cases:
        for repeat_index in range(config.repeats):
            row = _probe_case(adapter, case, repeat_index=repeat_index, case_timeout_s=case_timeout_s)
            attempts.append(row)
            by_case[case.id].append(row)
    error_attempts = [row for row in attempts if row.get("error")]
    graded_attempts = [row for row in attempts if not row.get("error")]
    unstable_cases = []
    output_drift_case_ids = []
    for case_id, rows in sorted(by_case.items()):
        successful_rows = [row for row in rows if not row.get("error")]
        if len(successful_rows) < 2:
            continue
        pass_values = {bool(row.get("passed")) for row in successful_rows}
        scores = [float(row.get("score", 0.0)) for row in successful_rows]
        outputs = {json.dumps(row.get("output"), sort_keys=True, default=str) for row in successful_rows}
        score_delta = max(scores) - min(scores)
        if len(pass_values) > 1 or score_delta > 0.01:
            unstable_cases.append(
                {
                    "case_id": case_id,
                    "pass_values": sorted(pass_values),
                    "score_delta": round(score_delta, 4),
                }
            )
        if len(outputs) > 1:
            output_drift_case_ids.append(case_id)
    latency_values = [float(row.get("latency_s", 0.0)) for row in graded_attempts]
    cost_values = [float(row.get("cost_usd", 0.0)) for row in graded_attempts]
    token_values = [int(row.get("total_tokens", 0)) for row in graded_attempts]
    runtime_feasibility = _runtime_feasibility(
        cases=cases,
        attempts=graded_attempts,
        latency_values=latency_values,
        cost_values=cost_values,
        token_values=token_values,
        evaluation_samples_per_case=evaluation_samples_per_case,
        case_concurrency=case_concurrency,
    )
    return {
        "checked": True,
        "sampled_case_ids": [case.id for case in selected_cases],
        "repeats": config.repeats,
        "attempt_count": len(attempts),
        "successful_attempt_count": len(graded_attempts),
        "error_attempt_count": len(error_attempts),
        "runtime_error_rate": round(len(error_attempts) / len(attempts), 4) if attempts else 0.0,
        "unstable_case_count": len(unstable_cases),
        "unstable_case_rate": round(len(unstable_cases) / len(selected_cases), 4) if selected_cases else 0.0,
        "unstable_cases": unstable_cases,
        "output_drift_case_ids": output_drift_case_ids,
        "mean_score": round(statistics.fmean(float(row.get("score", 0.0)) for row in graded_attempts), 4)
        if graded_attempts
        else 0.0,
        "pass_rate": round(statistics.fmean(float(bool(row.get("passed"))) for row in graded_attempts), 4)
        if graded_attempts
        else 0.0,
        "latency_s": _metric_summary(latency_values),
        "cost_usd": _metric_summary(cost_values),
        "total_tokens": _metric_summary(token_values),
        "runtime_feasibility": runtime_feasibility,
        "attempts": attempts,
    }


def _probe_case(
    adapter: AdapterProtocol,
    case: EvalCase,
    *,
    repeat_index: int,
    case_timeout_s: int,
) -> dict[str, Any]:
    try:
        with _case_timeout(case_timeout_s):
            record = adapter.run_case(case, None)
        if not isinstance(record, RunRecord):
            raise TypeError(f"run_case returned {type(record).__name__}, expected RunRecord.")
        try:
            json.dumps(record.output, sort_keys=True)
        except TypeError as error:
            raise TypeError("run_case returned a non-JSON-serializable output.") from error
        with _case_timeout(case_timeout_s):
            grade = adapter.grade(case, record.output)
        if not isinstance(grade, GradeResult):
            raise TypeError(f"grade returned {type(grade).__name__}, expected GradeResult.")
        return {
            "case_id": case.id,
            "split": case.split,
            "repeat_index": repeat_index,
            "passed": grade.passed,
            "score": grade.score,
            "labels": list(grade.labels),
            "notes": grade.notes,
            "output": record.output,
            "latency_s": record.metrics.latency_s,
            "input_tokens": record.metrics.input_tokens,
            "output_tokens": record.metrics.output_tokens,
            "total_tokens": record.metrics.total_tokens,
            "cost_usd": record.metrics.cost_usd,
            "metric_error": record.metrics.error,
            "diagnostics": record.diagnostics.to_dict(),
        }
    except Exception as error:
        return {
            "case_id": case.id,
            "split": case.split,
            "repeat_index": repeat_index,
            "error": f"{type(error).__name__}: {error}",
        }


def _probe_issues(probe: dict[str, Any], config: EvalHealthConfig) -> list[EvalHealthIssue]:
    if not probe.get("checked"):
        return [
            EvalHealthIssue(
                severity="info",
                code="baseline_probe_skipped",
                message=str(probe.get("reason", "Baseline probes were skipped.")),
            )
        ]
    issues: list[EvalHealthIssue] = []
    attempt_count = int(probe.get("attempt_count") or 0)
    error_count = int(probe.get("error_attempt_count") or 0)
    runtime_error_rate = float(probe.get("runtime_error_rate") or 0.0)
    if attempt_count and error_count == attempt_count:
        issues.append(
            EvalHealthIssue(
                severity="fatal",
                code="baseline_probe_all_failed",
                message="Every sampled baseline probe failed before grading.",
                case_ids=list(probe.get("sampled_case_ids") or []),
                metadata={"runtime_error_rate": runtime_error_rate},
            )
        )
    elif runtime_error_rate > config.max_runtime_error_rate:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="baseline_probe_error_rate_high",
                message=(
                    f"Sampled baseline runtime/grader error rate is {runtime_error_rate:.2%}; "
                    f"configured maximum is {config.max_runtime_error_rate:.2%}."
                ),
                case_ids=_error_case_ids(probe),
                metadata={"runtime_error_rate": runtime_error_rate, "max_runtime_error_rate": config.max_runtime_error_rate},
            )
        )
    runtime_feasibility = dict(probe.get("runtime_feasibility") or {})
    issues.extend(_runtime_feasibility_issues(runtime_feasibility, config))
    unstable_rate = float(probe.get("unstable_case_rate") or 0.0)
    if unstable_rate > config.max_unstable_case_rate:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="baseline_probe_unstable",
                message=(
                    f"Sampled baseline instability rate is {unstable_rate:.2%}; "
                    f"configured maximum is {config.max_unstable_case_rate:.2%}."
                ),
                case_ids=[str(row.get("case_id")) for row in probe.get("unstable_cases", [])],
                metadata={"unstable_cases": probe.get("unstable_cases", [])},
            )
        )
    output_drift_case_ids = list(probe.get("output_drift_case_ids") or [])
    if output_drift_case_ids and not probe.get("unstable_cases"):
        issues.append(
            EvalHealthIssue(
                severity="info",
                code="baseline_probe_output_drift",
                message="Some repeated baseline probes produced different outputs without changing pass/score.",
                case_ids=output_drift_case_ids,
            )
        )
    return issues


def _runtime_feasibility_issues(
    runtime_feasibility: dict[str, Any],
    config: EvalHealthConfig,
) -> list[EvalHealthIssue]:
    if not runtime_feasibility.get("estimated"):
        return []
    issues: list[EvalHealthIssue] = []
    observed = dict(runtime_feasibility.get("observed_per_case") or {})
    estimated = dict(runtime_feasibility.get("estimated_eval_sweep") or {})
    mean_latency = _optional_float((observed.get("latency_s") or {}).get("mean"))
    p95_latency = _optional_float((observed.get("latency_s") or {}).get("p95"))
    mean_cost = _optional_float((observed.get("cost_usd") or {}).get("mean"))
    estimated_cost = _optional_float(estimated.get("cost_usd"))
    estimated_wall_time = _optional_float(estimated.get("wall_time_s"))
    estimated_tokens = _optional_float(estimated.get("total_tokens"))

    if mean_latency is not None and mean_latency > config.max_mean_latency_s:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="runtime_mean_latency_high",
                message=(
                    f"Sampled baseline mean latency is {mean_latency:.2f}s/case; "
                    f"configured maximum is {config.max_mean_latency_s:.2f}s."
                ),
                metadata={"observed_mean_latency_s": mean_latency, "max_mean_latency_s": config.max_mean_latency_s},
            )
        )
    if p95_latency is not None and p95_latency > config.max_p95_latency_s:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="runtime_p95_latency_high",
                message=(
                    f"Sampled baseline p95 latency is {p95_latency:.2f}s/case; "
                    f"configured maximum is {config.max_p95_latency_s:.2f}s."
                ),
                metadata={"observed_p95_latency_s": p95_latency, "max_p95_latency_s": config.max_p95_latency_s},
            )
        )
    if mean_cost is not None and mean_cost > config.max_mean_cost_usd:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="runtime_mean_cost_high",
                message=(
                    f"Sampled baseline mean cost is ${mean_cost:.6f}/case; "
                    f"configured maximum is ${config.max_mean_cost_usd:.6f}."
                ),
                metadata={"observed_mean_cost_usd": mean_cost, "max_mean_cost_usd": config.max_mean_cost_usd},
            )
        )
    if estimated_cost is not None and estimated_cost > config.max_estimated_eval_cost_usd:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="runtime_estimated_eval_cost_high",
                message=(
                    f"Estimated full dev+holdout evaluation sweep cost is ${estimated_cost:.4f}; "
                    f"configured maximum is ${config.max_estimated_eval_cost_usd:.4f}."
                ),
                metadata={"estimated_eval_sweep": estimated, "max_estimated_eval_cost_usd": config.max_estimated_eval_cost_usd},
            )
        )
    if estimated_wall_time is not None and estimated_wall_time > config.max_estimated_eval_wall_time_s:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="runtime_estimated_eval_wall_time_high",
                message=(
                    f"Estimated full dev+holdout evaluation sweep wall time is {estimated_wall_time:.1f}s; "
                    f"configured maximum is {config.max_estimated_eval_wall_time_s:.1f}s."
                ),
                metadata={
                    "estimated_eval_sweep": estimated,
                    "max_estimated_eval_wall_time_s": config.max_estimated_eval_wall_time_s,
                },
            )
        )
    if estimated_tokens is not None and estimated_tokens > config.max_estimated_eval_tokens:
        issues.append(
            EvalHealthIssue(
                severity="warning",
                code="runtime_estimated_eval_tokens_high",
                message=(
                    f"Estimated full dev+holdout evaluation sweep uses {int(estimated_tokens)} tokens; "
                    f"configured maximum is {config.max_estimated_eval_tokens}."
                ),
                metadata={
                    "estimated_eval_sweep": estimated,
                    "max_estimated_eval_tokens": config.max_estimated_eval_tokens,
                },
            )
        )
    return issues


def _select_probe_cases(cases: tuple[EvalCase, ...], *, limit: int) -> tuple[EvalCase, ...]:
    if limit <= 0:
        return ()
    selected: list[EvalCase] = []
    seen_ids: set[str] = set()
    groups: dict[tuple[str, str], list[EvalCase]] = defaultdict(list)
    for case in cases:
        if case.split in {"dev", "holdout"}:
            groups[(case.split, _category(case))].append(case)
    for split in ("dev", "holdout"):
        for key in sorted(key for key in groups if key[0] == split):
            if len(selected) >= limit:
                break
            case = groups[key][0]
            selected.append(case)
            seen_ids.add(case.id)
    for case in cases:
        if len(selected) >= limit:
            break
        if case.split not in {"dev", "holdout"} or case.id in seen_ids:
            continue
        selected.append(case)
        seen_ids.add(case.id)
    return tuple(selected)


def _low_support_categories(category_summary: dict[str, Any], split: str, minimum: int) -> list[dict[str, Any]]:
    counts = ((category_summary.get("by_split") or {}).get(split) or {})
    return [
        {"category": category, "count": count}
        for category, count in sorted(counts.items())
        if int(count) < minimum
    ]


def _duplicate_rows(groups: Iterable[list[EvalCase]]) -> list[dict[str, Any]]:
    rows = []
    for items in groups:
        if len(items) < 2:
            continue
        rows.append(
            {
                "case_ids": [case.id for case in items],
                "splits": sorted({case.split for case in items}),
                "labels": sorted({_expected_label(case.expected) for case in items}),
            }
        )
    return rows


def _case_ids_from_duplicate_rows(rows: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        ids.extend(str(case_id) for case_id in row.get("case_ids", []))
    return sorted(set(ids))


def _error_case_ids(probe: dict[str, Any]) -> list[str]:
    return sorted(
        {
            str(row.get("case_id"))
            for row in probe.get("attempts", [])
            if row.get("error")
        }
    )


def _metric_summary(values: list[float] | list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "p95": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "min": round(min(numeric), 6),
        "mean": round(statistics.fmean(numeric), 6),
        "median": round(statistics.median(numeric), 6),
        "p95": round(_percentile(numeric, 0.95), 6),
        "max": round(max(numeric), 6),
    }


def _runtime_feasibility(
    *,
    cases: tuple[EvalCase, ...],
    attempts: list[dict[str, Any]],
    latency_values: list[float],
    cost_values: list[float],
    token_values: list[int],
    evaluation_samples_per_case: int,
    case_concurrency: int,
) -> dict[str, Any]:
    runnable_case_count = sum(1 for case in cases if case.split in {"dev", "holdout"})
    samples_per_case = max(1, int(evaluation_samples_per_case))
    concurrency = max(1, int(case_concurrency))
    eval_attempts = runnable_case_count * samples_per_case
    observed = {
        "latency_s": _metric_summary(latency_values),
        "cost_usd": _metric_summary(cost_values),
        "total_tokens": _metric_summary(token_values),
    }
    models = _model_metric_summary(attempts)
    pricing_basis = _pricing_basis(attempts)
    if not latency_values and not cost_values and not token_values:
        return {
            "estimated": False,
            "pricing_basis": pricing_basis,
            "models": models,
            "runnable_case_count": runnable_case_count,
            "samples_per_case": samples_per_case,
            "case_concurrency": concurrency,
            "observed_per_case": observed,
            "estimated_eval_sweep": {},
        }
    mean_latency = float(observed["latency_s"]["mean"] or 0.0)
    mean_cost = float(observed["cost_usd"]["mean"] or 0.0)
    mean_tokens = float(observed["total_tokens"]["mean"] or 0.0)
    return {
        "estimated": True,
        "pricing_basis": pricing_basis,
        "models": models,
        "runnable_case_count": runnable_case_count,
        "samples_per_case": samples_per_case,
        "case_concurrency": concurrency,
        "observed_per_case": observed,
        "estimated_eval_sweep": {
            "case_attempts": eval_attempts,
            "wall_time_s": round((mean_latency * eval_attempts) / concurrency, 3),
            "serial_time_s": round(mean_latency * eval_attempts, 3),
            "cost_usd": round(mean_cost * eval_attempts, 6),
            "total_tokens": int(round(mean_tokens * eval_attempts)),
        },
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_metric_summary(attempts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        grouped[_attempt_model(attempt)].append(attempt)
    rows: dict[str, dict[str, Any]] = {}
    for model, items in sorted(grouped.items()):
        latencies = [float(item.get("latency_s", 0.0)) for item in items]
        costs = [float(item.get("cost_usd", 0.0)) for item in items]
        tokens = [int(item.get("total_tokens", 0)) for item in items]
        rows[model] = {
            "attempt_count": len(items),
            "pass_rate": round(statistics.fmean(float(bool(item.get("passed"))) for item in items), 4),
            "mean_latency_s": round(statistics.fmean(latencies), 6) if latencies else None,
            "p95_latency_s": round(_percentile(latencies, 0.95), 6) if latencies else None,
            "mean_cost_usd": round(statistics.fmean(costs), 6) if costs else None,
            "mean_total_tokens": round(statistics.fmean(tokens), 6) if tokens else None,
        }
    return rows


def _attempt_model(attempt: dict[str, Any]) -> str:
    diagnostics = attempt.get("diagnostics")
    if isinstance(diagnostics, dict):
        metadata = diagnostics.get("metadata")
        if isinstance(metadata, dict) and metadata.get("model"):
            return str(metadata["model"])
    return "unknown"


def _pricing_basis(attempts: list[dict[str, Any]]) -> str:
    cost_values = [attempt.get("cost_usd") for attempt in attempts if attempt.get("cost_usd") is not None]
    if any(float(value or 0.0) > 0.0 for value in cost_values):
        return "adapter_reported_cost_usd"
    if cost_values:
        return "unavailable_or_zero_reported_cost"
    return "not_measured"


def _health_decision(status: HealthStatus) -> str:
    if status == "fatal":
        return "Fix eval health issues before running optimize."
    if status == "warning":
        return "Review warnings before running optimize."
    return "Safe to run optimize."


def _report_summary(report: EvalHealthReport) -> dict[str, Any]:
    issue_counts = {"fatal": 0, "warning": 0, "info": 0}
    for issue in report.issues:
        issue_counts[issue.severity] = issue_counts.get(issue.severity, 0) + 1
    probe = report.baseline_probe
    runtime = dict(probe.get("runtime_feasibility") or {})
    estimated = dict(runtime.get("estimated_eval_sweep") or {})
    split_counts = dict(report.split_summary.get("by_split") or {})
    return {
        "status": report.status,
        "decision": _health_decision(report.status),
        "issue_counts": issue_counts,
        "splits": {
            "train": int(split_counts.get("train", 0)),
            "dev": int(split_counts.get("dev", 0)),
            "holdout": int(split_counts.get("holdout", 0)),
        },
        "baseline_probe": {
            "checked": bool(probe.get("checked")),
            "sampled_case_count": len(probe.get("sampled_case_ids") or []),
            "repeats": int(probe.get("repeats") or 0),
            "pass_rate": probe.get("pass_rate"),
            "runtime_error_rate": probe.get("runtime_error_rate"),
            "unstable_case_count": int(probe.get("unstable_case_count") or 0),
        },
        "runtime_estimate": {
            "models": sorted((runtime.get("models") or {}).keys()),
            "pricing_basis": runtime.get("pricing_basis"),
            "case_attempts": estimated.get("case_attempts"),
            "wall_time_s": estimated.get("wall_time_s"),
            "cost_usd": estimated.get("cost_usd"),
            "total_tokens": estimated.get("total_tokens"),
        },
    }


def _format_optional_number(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


@contextlib.contextmanager
def _case_timeout(timeout_s: int) -> Iterable[None]:
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Case exceeded {timeout_s} second timeout.")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _category(case: EvalCase) -> str:
    value = case.metadata.get("category")
    if value is None:
        return "uncategorized"
    return str(value)


def _expected_label(expected: Any) -> str:
    if isinstance(expected, dict) and "label" in expected:
        return str(expected["label"])
    try:
        return json.dumps(expected, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        return str(expected)


def _normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9_ -]+", "", text)
    return text.strip()


def _status(issues: list[EvalHealthIssue]) -> HealthStatus:
    if any(issue.severity == "fatal" for issue in issues):
        return "fatal"
    if any(issue.severity == "warning" for issue in issues):
        return "warning"
    return "healthy"
