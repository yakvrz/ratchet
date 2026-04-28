from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.results import PatchSummary
from ratchet.types import FailureDiagnosis, OptimizationObjective


MECHANISM_CLASSES = {
    "runtime_defect_fix",
    "output_contract_fix",
    "semantic_boundary_rewrite",
    "representative_examples",
    "contrastive_examples",
    "model_capability_probe",
    "efficiency_probe",
    "ablation",
}


MECHANISMS_BY_FAMILY: dict[str, set[str]] = {
    "prompt_rewrite": {"runtime_defect_fix", "semantic_boundary_rewrite", "output_contract_fix", "ablation"},
    "output_contract_tightening": {"output_contract_fix", "runtime_defect_fix", "ablation"},
    "targeted_few_shot": {"representative_examples", "contrastive_examples", "semantic_boundary_rewrite", "ablation"},
    "model_substitution": {"runtime_defect_fix", "model_capability_probe", "efficiency_probe", "ablation"},
    "tool_policy_revision": {"efficiency_probe", "semantic_boundary_rewrite", "ablation"},
    "retrieval_tuning": {"efficiency_probe", "semantic_boundary_rewrite", "ablation"},
    "runtime_tuning": {"runtime_defect_fix", "efficiency_probe", "output_contract_fix", "ablation"},
    "verifier_retry": {"output_contract_fix", "semantic_boundary_rewrite", "ablation"},
}


CANDIDATE_ROLES = {"atomic", "composed", "control", "ablation", "compression"}


@dataclass(frozen=True)
class TaskTheory:
    bottleneck_class: str
    residual_failure_modes: list[str]
    label_confusions: list[dict[str, Any]]
    weak_slices: list[str]
    runtime_defects: dict[str, Any]
    output_defects: dict[str, Any]
    example_coverage: dict[str, Any]
    cost_latency_profile: dict[str, Any]
    confidence: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    mechanism: str
    hypothesis: str
    mechanism_label: str = ""
    target_slices: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    candidate_roles: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mechanism not in MECHANISM_CLASSES:
            raise ValueError(f"unknown experiment mechanism {self.mechanism!r}")
        if not self.experiment_id:
            raise ValueError("experiment_id must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, fallback_id: str) -> "ExperimentSpec":
        raw_mechanism = str(payload.get("mechanism") or "")
        mechanism_class = str(payload.get("mechanism_class") or raw_mechanism)
        if mechanism_class not in MECHANISM_CLASSES:
            candidate_mechanisms = [
                str(candidate.get("mechanism_class") or "")
                for candidate in payload.get("candidates", [])
                if isinstance(candidate, dict)
            ]
            valid_candidate_mechanisms = [
                mechanism for mechanism in candidate_mechanisms if mechanism in MECHANISM_CLASSES
            ]
            if valid_candidate_mechanisms:
                mechanism_class = valid_candidate_mechanisms[0]
        return cls(
            experiment_id=str(payload.get("experiment_id") or payload.get("id") or fallback_id),
            mechanism=mechanism_class,
            hypothesis=str(payload.get("hypothesis") or ""),
            mechanism_label=raw_mechanism if raw_mechanism != mechanism_class else "",
            target_slices=[str(item) for item in payload.get("target_slices", []) if item],
            measurements=[str(item) for item in payload.get("measurements", payload.get("expected_measurements", [])) if item],
            candidate_roles=[str(item) for item in payload.get("candidate_roles", []) if item],
        )


def build_task_theory(
    *,
    summary: PatchSummary,
    diagnoses: list[FailureDiagnosis],
    objective: OptimizationObjective,
    proposal_example_bank: ProposalExampleBank | None = None,
) -> TaskTheory:
    diagnostics = build_behavior_diagnostics(summary)
    runtime = dict(diagnostics.get("runtime_reliability") or {})
    invalid_case_ids = list(diagnostics.get("invalid_output_case_ids") or [])
    confusions = list(diagnostics.get("confusions") or [])
    weak_labels = [str(item) for item in diagnostics.get("weak_labels", [])]
    diagnosis_categories = sorted({diagnosis.category for diagnosis in diagnoses if diagnosis.category})
    evidence: list[str] = []
    if runtime.get("length_finish_case_ids") or runtime.get("parser_fallback_case_ids"):
        bottleneck = "runtime_or_output_defect"
        evidence.append("runtime/output trace defects observed")
    elif invalid_case_ids:
        bottleneck = "output_contract"
        evidence.append("invalid output failures observed")
    elif confusions or weak_labels:
        bottleneck = "semantic_boundary_confusion"
        evidence.append("label or slice confusions observed")
    elif objective.mode in {"cost", "latency"}:
        bottleneck = "efficiency_tradeoff"
        evidence.append(f"{objective.mode} objective active")
    elif summary.pass_count < summary.case_count:
        bottleneck = "general_correctness_gap"
        evidence.append("failing cases observed")
    else:
        bottleneck = "no_observed_failures"
        evidence.append("current branch has no observed failures")
    label_counts = proposal_example_bank.label_counts if proposal_example_bank is not None else {}
    missing_weak_examples = [label for label in weak_labels if label not in label_counts]
    return TaskTheory(
        bottleneck_class=bottleneck,
        residual_failure_modes=_residual_failure_modes(
            invalid_case_ids=invalid_case_ids,
            confusions=confusions,
            weak_labels=weak_labels,
            diagnosis_categories=diagnosis_categories,
        ),
        label_confusions=confusions[:12],
        weak_slices=weak_labels[:20],
        runtime_defects={
            "finish_reason_counts": runtime.get("finish_reason_counts", {}),
            "length_finish_case_ids": runtime.get("length_finish_case_ids", []),
            "parser_fallback_case_ids": runtime.get("parser_fallback_case_ids", []),
            "low_output_token_length_case_ids": runtime.get("low_output_token_length_case_ids", []),
        },
        output_defects={
            "invalid_output_case_ids": invalid_case_ids,
            "invalid_output_count": len(invalid_case_ids),
        },
        example_coverage={
            "example_count": len(proposal_example_bank.examples) if proposal_example_bank else 0,
            "label_counts": dict(label_counts),
            "weak_labels_without_examples": missing_weak_examples,
        },
        cost_latency_profile={
            "mean_cost_usd": summary.mean_cost_usd,
            "mean_total_tokens": summary.mean_total_tokens,
            "median_latency_s": summary.median_latency_s,
        },
        confidence="medium" if summary.case_count >= 20 else "low",
        evidence=evidence,
    )


def mechanism_error_for_family(family: str, mechanism: str) -> str | None:
    if mechanism not in MECHANISM_CLASSES:
        return f"unknown mechanism class {mechanism!r}"
    allowed = MECHANISMS_BY_FAMILY.get(family)
    if allowed is None:
        return None
    if mechanism not in allowed:
        return f"mechanism class {mechanism!r} is incompatible with transform family {family!r}"
    return None


def _residual_failure_modes(
    *,
    invalid_case_ids: list[str],
    confusions: list[dict[str, Any]],
    weak_labels: list[str],
    diagnosis_categories: list[str],
) -> list[str]:
    modes: list[str] = []
    if invalid_case_ids:
        modes.append("invalid_output")
    if confusions:
        modes.append("label_confusion")
    if weak_labels:
        modes.append("weak_slices")
    modes.extend(category for category in diagnosis_categories if category not in modes)
    return modes[:12]
