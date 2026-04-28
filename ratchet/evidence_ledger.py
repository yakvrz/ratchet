from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ratchet.objectives import behavior_flip_summary, compare_summaries
from ratchet.results import Comparison, PatchSummary
from ratchet.types import OptimizationObjective


LOW_CONFIDENCE_CASE_COUNT = 12
MEDIUM_CONFIDENCE_CASE_COUNT = 24


@dataclass(frozen=True)
class EvidenceSummary:
    candidate_id: str
    stage: str
    case_ids: list[str]
    case_count: int
    reference_patch_hash: str
    baseline_patch_hash: str
    candidate_patch_hash: str
    comparison_to_reference: dict[str, Any]
    behavior_flip_summary: dict[str, Any]
    effect_size: float
    pass_gain: int
    fixed_count: int
    regressed_count: int
    invalid_output_delta: int
    finish_reason_delta: dict[str, int]
    token_delta: float
    cost_delta: float
    latency_delta: float
    sign_consistency: str
    confidence_tier: str
    baseline_instability_flags: list[str]
    measurement_cost: dict[str, Any]
    mechanism_class: str
    affordance_ids: list[str]
    comparison_group: str
    candidate_role: str
    rejection_reason: str | None = None
    constraint_warning: str | None = None
    passed_stage: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceLedger:
    records: list[EvidenceSummary] = field(default_factory=list)

    def add(
        self,
        *,
        candidate_id: str,
        stage: str,
        reference: PatchSummary,
        baseline: PatchSummary,
        candidate: PatchSummary,
        mechanism_class: str,
        affordance_ids: list[str],
        comparison_group: str,
        candidate_role: str,
        rejection_reason: str | None,
        constraint_warning: str | None,
    ) -> EvidenceSummary:
        summary = build_evidence_summary(
            candidate_id=candidate_id,
            stage=stage,
            reference=reference,
            baseline=baseline,
            candidate=candidate,
            mechanism_class=mechanism_class,
            affordance_ids=affordance_ids,
            comparison_group=comparison_group,
            candidate_role=candidate_role,
            rejection_reason=rejection_reason,
            constraint_warning=constraint_warning,
        )
        self.records.append(summary)
        return summary

    def latest(self, candidate_id: str) -> EvidenceSummary | None:
        for record in reversed(self.records):
            if record.candidate_id == candidate_id:
                return record
        return None

    def by_candidate(self, candidate_id: str) -> list[EvidenceSummary]:
        return [record for record in self.records if record.candidate_id == candidate_id]

    def selector_rows(self, candidate_ids: Iterable[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate_id in candidate_ids:
            latest = self.latest(candidate_id)
            if latest is not None:
                row = latest.to_dict()
                row["stage_history"] = [record.to_dict() for record in self.by_candidate(candidate_id)]
                rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self.records],
            "summary": ledger_summary(self.records),
        }


def build_evidence_summary(
    *,
    candidate_id: str,
    stage: str,
    reference: PatchSummary,
    baseline: PatchSummary,
    candidate: PatchSummary,
    mechanism_class: str,
    affordance_ids: list[str],
    comparison_group: str,
    candidate_role: str,
    rejection_reason: str | None,
    constraint_warning: str | None,
) -> EvidenceSummary:
    comparison = compare_summaries(reference, candidate)
    flip_summary = behavior_flip_summary(reference, candidate)
    fixed_count = len(flip_summary["fixed_case_ids"])
    regressed_count = len(flip_summary["regressed_case_ids"])
    pass_gain = fixed_count - regressed_count
    invalid_output_delta = len(_invalid_case_ids(candidate)) - len(_invalid_case_ids(reference))
    finish_reason_delta = _counter_delta(_finish_reason_counts(reference), _finish_reason_counts(candidate))
    flags = _baseline_instability_flags(reference=reference, baseline=baseline, candidate=candidate)
    confidence = _confidence_tier(
        case_count=candidate.case_count,
        pass_gain=pass_gain,
        fixed_count=fixed_count,
        regressed_count=regressed_count,
        score_delta=comparison.score_delta,
        flags=flags,
    )
    return EvidenceSummary(
        candidate_id=candidate_id,
        stage=stage,
        case_ids=list(candidate.grouped_evaluations),
        case_count=candidate.case_count,
        reference_patch_hash=reference.patch_hash,
        baseline_patch_hash=baseline.patch_hash,
        candidate_patch_hash=candidate.patch_hash,
        comparison_to_reference=comparison.to_dict(),
        behavior_flip_summary=flip_summary,
        effect_size=round(comparison.score_delta, 6),
        pass_gain=pass_gain,
        fixed_count=fixed_count,
        regressed_count=regressed_count,
        invalid_output_delta=invalid_output_delta,
        finish_reason_delta=finish_reason_delta,
        token_delta=comparison.token_delta,
        cost_delta=comparison.cost_delta,
        latency_delta=comparison.latency_delta,
        sign_consistency=_sign_consistency(pass_gain=pass_gain, score_delta=comparison.score_delta),
        confidence_tier=confidence,
        baseline_instability_flags=flags,
        measurement_cost={
            "candidate_samples": candidate.sample_count,
            "candidate_mean_cost_usd": candidate.mean_cost_usd,
            "candidate_mean_total_tokens": candidate.mean_total_tokens,
            "estimated_total_cost_usd": candidate.mean_cost_usd * candidate.sample_count,
            "estimated_total_tokens": candidate.mean_total_tokens * candidate.sample_count,
        },
        mechanism_class=mechanism_class,
        affordance_ids=list(affordance_ids),
        comparison_group=comparison_group,
        candidate_role=candidate_role,
        rejection_reason=rejection_reason,
        constraint_warning=constraint_warning,
        passed_stage=rejection_reason is None,
    )


def confirmation_stability_result(
    *,
    reference: PatchSummary,
    candidate: PatchSummary,
    repeated_reference: PatchSummary,
    repeated_candidate: PatchSummary,
    objective: OptimizationObjective,
) -> dict[str, Any]:
    comparison = compare_summaries(repeated_reference, repeated_candidate)
    flip_summary = behavior_flip_summary(repeated_reference, repeated_candidate)
    original_flip_summary = behavior_flip_summary(reference, candidate)
    regressed_count = len(flip_summary["regressed_case_ids"])
    fixed_count = len(flip_summary["fixed_case_ids"])
    original_fixed_invalid = set(original_flip_summary["fixed_case_ids"]) & set(_invalid_case_ids(reference))
    repeated_reference_invalid = set(_invalid_case_ids(repeated_reference))
    repeated_candidate_invalid = set(_invalid_case_ids(repeated_candidate))
    invalid_fixed_again = sorted((original_fixed_invalid | repeated_reference_invalid) - repeated_candidate_invalid)
    if regressed_count:
        status = "failed"
        reason = f"stability check observed regressions on {regressed_count} case(s)"
        passed = False
    elif original_fixed_invalid and not repeated_reference_invalid and fixed_count <= 0:
        status = "runtime_instability"
        reason = "repeated baseline no longer reproduced the invalid-output/runtime failures that made the dev gain appear large"
        passed = False
    elif invalid_fixed_again:
        status = "runtime_defect_confirmed"
        reason = "candidate repeated the runtime/output fix against a fresh paired baseline"
        passed = True
    elif fixed_count > 0 or comparison.score_delta > 0:
        status = "semantic_gain_after_runtime_fix"
        reason = "candidate repeated a positive non-regressing signal after paired runtime check"
        passed = True
    else:
        status = "runtime_instability"
        reason = "paired repeat did not reproduce a positive candidate signal"
        passed = False
    return {
        "status": status,
        "passed": passed,
        "reason": reason,
        "objective": objective.to_dict(),
        "case_ids": list(repeated_reference.grouped_evaluations),
        "reference_metrics": repeated_reference.to_dict(),
        "candidate_metrics": repeated_candidate.to_dict(),
        "comparison_to_reference": comparison.to_dict(),
        "behavior_flip_summary": flip_summary,
        "original_behavior_flip_summary": original_flip_summary,
        "invalid_output": {
            "original_fixed_invalid_case_ids": sorted(original_fixed_invalid),
            "repeated_reference_invalid_case_ids": sorted(repeated_reference_invalid),
            "repeated_candidate_invalid_case_ids": sorted(repeated_candidate_invalid),
            "invalid_fixed_again_case_ids": invalid_fixed_again,
        },
    }


def ledger_summary(records: list[EvidenceSummary]) -> dict[str, Any]:
    confidence_counts = Counter(record.confidence_tier for record in records)
    stage_counts = Counter(record.stage for record in records)
    instability_counts = Counter(flag for record in records for flag in record.baseline_instability_flags)
    return {
        "record_count": len(records),
        "stage_counts": dict(sorted(stage_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "baseline_instability_counts": dict(sorted(instability_counts.items())),
        "measurement_cost": {
            "estimated_total_cost_usd": round(
                sum(record.measurement_cost.get("estimated_total_cost_usd", 0.0) for record in records),
                8,
            ),
            "estimated_total_tokens": int(
                sum(record.measurement_cost.get("estimated_total_tokens", 0.0) for record in records)
            ),
        },
    }


def _confidence_tier(
    *,
    case_count: int,
    pass_gain: int,
    fixed_count: int,
    regressed_count: int,
    score_delta: float,
    flags: list[str],
) -> str:
    if flags:
        return "unstable"
    if case_count < LOW_CONFIDENCE_CASE_COUNT:
        return "low"
    if abs(pass_gain) < 2 and abs(score_delta) < 0.10:
        return "low"
    if case_count >= MEDIUM_CONFIDENCE_CASE_COUNT and pass_gain >= 2 and regressed_count == 0:
        return "high"
    if fixed_count or regressed_count or abs(score_delta) >= 0.05:
        return "medium"
    return "low"


def _sign_consistency(*, pass_gain: int, score_delta: float) -> str:
    if pass_gain > 0 and score_delta >= 0:
        return "positive"
    if pass_gain < 0 and score_delta <= 0:
        return "negative"
    if pass_gain == 0 and abs(score_delta) < 1e-9:
        return "flat"
    return "mixed"


def _baseline_instability_flags(
    *,
    reference: PatchSummary,
    baseline: PatchSummary,
    candidate: PatchSummary,
) -> list[str]:
    flags: list[str] = []
    if reference.patch_hash != baseline.patch_hash:
        return flags
    reference_invalid = len(_invalid_case_ids(reference))
    candidate_invalid = len(_invalid_case_ids(candidate))
    if candidate_invalid < reference_invalid:
        flags.append("runtime_repeat_required")
    reference_finish = _finish_reason_counts(reference)
    candidate_finish = _finish_reason_counts(candidate)
    if reference_finish.get("length", 0) > candidate_finish.get("length", 0):
        flags.append("runtime_repeat_required")
    return sorted(set(flags))


def _invalid_case_ids(summary: PatchSummary) -> list[str]:
    rows: list[str] = []
    for case_id, evaluations in summary.grouped_evaluations.items():
        if any(_invalid_output(evaluation) for evaluation in evaluations):
            rows.append(case_id)
    return sorted(rows)


def _invalid_output(evaluation: Any) -> bool:
    output = evaluation.record.output
    return (
        any("invalid_output" in label for label in evaluation.grade.labels)
        or (isinstance(output, dict) and "invalid_output" in output)
        or bool(evaluation.record.diagnostics.metadata.get("invalid_output"))
        or bool(evaluation.record.diagnostics.metadata.get("parser_fallback"))
    )


def _finish_reason_counts(summary: PatchSummary) -> Counter[str]:
    counts: Counter[str] = Counter()
    for evaluations in summary.grouped_evaluations.values():
        for evaluation in evaluations:
            finish_reason = str(evaluation.record.diagnostics.metadata.get("finish_reason") or "")
            if finish_reason:
                counts[finish_reason] += 1
    return counts


def _counter_delta(reference: Counter[str], candidate: Counter[str]) -> dict[str, int]:
    keys = sorted(set(reference) | set(candidate))
    return {key: candidate.get(key, 0) - reference.get(key, 0) for key in keys}
