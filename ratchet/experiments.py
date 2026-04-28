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
    experiment_opportunities: list[dict[str, Any]] = field(default_factory=list)

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
    example_source_ids = _example_source_ids_by_label(proposal_example_bank)
    target_example_labels = _target_example_labels(confusions=confusions, weak_labels=weak_labels)
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
            "target_label_source_case_ids": {
                label: example_source_ids.get(label, [])[:4]
                for label in target_example_labels
                if example_source_ids.get(label)
            },
        },
        cost_latency_profile={
            "mean_cost_usd": summary.mean_cost_usd,
            "mean_total_tokens": summary.mean_total_tokens,
            "median_latency_s": summary.median_latency_s,
        },
        confidence="medium" if summary.case_count >= 20 else "low",
        evidence=evidence,
        experiment_opportunities=_experiment_opportunities(
            bottleneck=bottleneck,
            runtime=runtime,
            invalid_case_ids=invalid_case_ids,
            confusions=confusions,
            weak_labels=weak_labels,
            example_source_ids=example_source_ids,
            objective=objective,
        ),
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


def _experiment_opportunities(
    *,
    bottleneck: str,
    runtime: dict[str, Any],
    invalid_case_ids: list[str],
    confusions: list[dict[str, Any]],
    weak_labels: list[str],
    example_source_ids: dict[str, list[str]],
    objective: OptimizationObjective,
) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    if runtime.get("length_finish_case_ids") or runtime.get("parser_fallback_case_ids"):
        opportunities.append(
            {
                "mechanism_class": "runtime_defect_fix",
                "target_slices": _slice_ids("runtime", runtime.get("length_finish_case_ids", []))
                + _slice_ids("parser_fallback", runtime.get("parser_fallback_case_ids", [])),
                "candidate_roles": ["atomic", "control"],
                "measurements": ["finish_reason_delta", "invalid_output_delta", "score_delta", "latency_delta"],
                "rationale": "Trace evidence suggests the current branch may be failing before semantic behavior is measurable.",
                "disconfirming_result": "Output reliability metrics do not improve on affected cases.",
            }
        )
    if invalid_case_ids:
        opportunities.append(
            {
                "mechanism_class": "output_contract_fix",
                "target_slices": _slice_ids("invalid_output", invalid_case_ids),
                "candidate_roles": ["atomic", "control"],
                "measurements": ["invalid_output_delta", "score_delta", "non_target_regression"],
                "rationale": "Invalid outputs should be tested as contract/format failures before adding semantic complexity.",
                "disconfirming_result": "Invalid-output cases remain invalid or regress elsewhere.",
            }
        )
    for row in confusions[:4]:
        if not isinstance(row, dict):
            continue
        expected = str(row.get("expected") or "")
        actual = str(row.get("actual") or "")
        if not expected or not actual:
            continue
        labels = [label for label in (expected, actual) if label]
        opportunities.append(
            {
                "mechanism_class": "semantic_boundary_rewrite",
                "target_slices": [f"confusion:{expected}->{actual}", f"label:{expected}"],
                "candidate_roles": ["control", "atomic", "composed"],
                "compatible_mechanisms": ["representative_examples", "contrastive_examples"],
                "source_labels": labels,
                "source_case_ids_by_label": {
                    label: example_source_ids.get(label, [])[:3]
                    for label in labels
                    if example_source_ids.get(label)
                },
                "measurements": ["target_slice_score_delta", "non_target_regression", "cost_delta"],
                "rationale": "Observed expected-vs-actual label confusion needs a boundary test, not only a generic prompt improvement.",
                "disconfirming_result": "The confusion persists on target cases or causes non-target regressions.",
            }
        )
    for label in weak_labels[:4]:
        if any(label == str(row.get("expected") or "") for row in confusions[:4] if isinstance(row, dict)):
            continue
        opportunities.append(
            {
                "mechanism_class": "representative_examples",
                "target_slices": [f"label:{label}"],
                "candidate_roles": ["atomic", "compression"],
                "compatible_mechanisms": ["semantic_boundary_rewrite", "contrastive_examples"],
                "source_labels": [label],
                "source_case_ids_by_label": {
                    label: example_source_ids.get(label, [])[:4]
                }
                if example_source_ids.get(label)
                else {},
                "measurements": ["target_slice_score_delta", "example_token_delta", "non_target_regression"],
                "rationale": "Weak label evidence with train coverage can test whether example anchoring is the missing signal.",
                "disconfirming_result": "Few-shot variants do not improve the weak label after compression.",
            }
        )
    if bottleneck in {"efficiency_tradeoff", "no_observed_failures"} or objective.mode in {"cost", "latency"}:
        opportunities.append(
            {
                "mechanism_class": "efficiency_probe",
                "target_slices": ["global"],
                "candidate_roles": ["atomic", "ablation"],
                "measurements": ["score_delta", "cost_delta", "latency_delta"],
                "rationale": "The current objective can improve through cost or latency reductions if quality is preserved.",
                "disconfirming_result": "Cost or latency improves only by violating the quality constraint.",
            }
        )
    if not opportunities and bottleneck == "general_correctness_gap":
        opportunities.append(
            {
                "mechanism_class": "semantic_boundary_rewrite",
                "target_slices": ["failed_cases"],
                "candidate_roles": ["atomic", "control"],
                "measurements": ["score_delta", "non_target_regression", "cost_delta"],
                "rationale": "Failures exist but are not yet explained by a sharper slice; test a minimal semantic hypothesis.",
                "disconfirming_result": "No failed-case improvement on the current branch.",
            }
        )
    return opportunities[:8]


def _target_example_labels(*, confusions: list[dict[str, Any]], weak_labels: list[str]) -> list[str]:
    labels: list[str] = []
    for row in confusions:
        if not isinstance(row, dict):
            continue
        for key in ("expected", "actual"):
            label = str(row.get(key) or "")
            if label and label not in labels:
                labels.append(label)
    for label in weak_labels:
        if label and label not in labels:
            labels.append(label)
    return labels[:12]


def _example_source_ids_by_label(bank: ProposalExampleBank | None) -> dict[str, list[str]]:
    if bank is None:
        return {}
    rows: dict[str, list[str]] = {}
    for example in bank.examples:
        if not example.label:
            continue
        rows.setdefault(example.label, []).append(example.case_id)
    return {label: case_ids[:6] for label, case_ids in sorted(rows.items())}


def _slice_ids(prefix: str, case_ids: Any) -> list[str]:
    if not isinstance(case_ids, list):
        return []
    return [f"{prefix}:{case_id}" for case_id in case_ids[:6] if case_id]
