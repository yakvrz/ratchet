from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.results import CandidateSummary
from ratchet.types import FailureDiagnosis, OptimizationObjective


MECHANISM_CLASSES = {
    "runtime_defect_fix",
    "output_contract_fix",
    "semantic_boundary_rewrite",
    "representative_examples",
    "contrastive_examples",
    "model_capability_probe",
    "efficiency_probe",
    "tool_selection_policy",
    "tool_argument_grounding",
    "tool_precondition_policy",
    "interaction_completion_policy",
    "ablation",
}


MECHANISMS_BY_FAMILY: dict[str, set[str]] = {
    "prompt_rewrite": {"runtime_defect_fix", "semantic_boundary_rewrite", "output_contract_fix", "ablation"},
    "output_contract_tightening": {"output_contract_fix", "runtime_defect_fix", "ablation"},
    "targeted_few_shot": {"representative_examples", "contrastive_examples", "semantic_boundary_rewrite", "ablation"},
    "model_substitution": {"runtime_defect_fix", "model_capability_probe", "efficiency_probe", "ablation"},
    "runtime_tuning": {"runtime_defect_fix", "efficiency_probe", "output_contract_fix", "ablation"},
    "tool_policy_revision": {
        "efficiency_probe",
        "semantic_boundary_rewrite",
        "tool_selection_policy",
        "tool_argument_grounding",
        "tool_precondition_policy",
        "interaction_completion_policy",
        "ablation",
    },
    "verifier_retry": {"runtime_defect_fix", "output_contract_fix", "semantic_boundary_rewrite", "ablation"},
}


CANDIDATE_ROLES = {"atomic", "composed", "control", "ablation", "compression"}


@dataclass(frozen=True)
class EvidencePacket:
    residual_failure_modes: list[str]
    label_confusions: list[dict[str, Any]]
    weak_slices: list[str]
    runtime_defects: dict[str, Any]
    output_defects: dict[str, Any]
    tool_defects: dict[str, Any]
    example_coverage: dict[str, Any]
    cost_latency_profile: dict[str, Any]
    behavior_diagnostics: dict[str, Any]
    diagnosis_categories: list[str]
    confidence: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalHypothesis:
    hypothesis_id: str
    statement: str
    mechanism_class: str
    target_slices: list[str] = field(default_factory=list)
    supporting_evidence: list[str] = field(default_factory=list)
    competing_evidence: list[str] = field(default_factory=list)
    disconfirming_result: str = ""
    confidence: str = "low"

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            raise ValueError("hypothesis_id must be non-empty")
        if not self.statement:
            raise ValueError("hypothesis statement must be non-empty")
        if self.mechanism_class not in MECHANISM_CLASSES:
            raise ValueError(f"unknown hypothesis mechanism_class {self.mechanism_class!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CausalHypothesis":
        return cls(
            hypothesis_id=str(payload.get("hypothesis_id") or payload.get("id") or ""),
            statement=str(payload.get("statement") or ""),
            mechanism_class=str(payload.get("mechanism_class") or ""),
            target_slices=[str(item) for item in payload.get("target_slices", []) if item],
            supporting_evidence=[str(item) for item in payload.get("supporting_evidence", []) if item],
            competing_evidence=[str(item) for item in payload.get("competing_evidence", []) if item],
            disconfirming_result=str(payload.get("disconfirming_result") or ""),
            confidence=str(payload.get("confidence") or "low"),
        )


@dataclass(frozen=True)
class ResearchOpportunity:
    opportunity_id: str
    hypothesis_ids: list[str]
    mechanism_class: str
    target_slices: list[str]
    rationale: str
    measurements: list[str] = field(default_factory=list)
    disconfirming_result: str = ""
    candidate_roles: list[str] = field(default_factory=list)
    compatible_mechanisms: list[str] = field(default_factory=list)
    affordance_ids: list[str] = field(default_factory=list)
    priority: int = 1

    def __post_init__(self) -> None:
        if not self.opportunity_id:
            raise ValueError("opportunity_id must be non-empty")
        if self.mechanism_class not in MECHANISM_CLASSES:
            raise ValueError(f"unknown opportunity mechanism_class {self.mechanism_class!r}")
        if not self.hypothesis_ids:
            raise ValueError("opportunity hypothesis_ids must be non-empty")
        if not self.rationale:
            raise ValueError("opportunity rationale must be non-empty")
        unknown_roles = sorted(set(self.candidate_roles) - CANDIDATE_ROLES)
        if unknown_roles:
            raise ValueError(f"unknown opportunity candidate_roles: {unknown_roles}")
        if self.priority < 1:
            raise ValueError("opportunity priority must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchOpportunity":
        return cls(
            opportunity_id=str(payload.get("opportunity_id") or payload.get("id") or ""),
            hypothesis_ids=[str(item) for item in payload.get("hypothesis_ids", []) if item],
            mechanism_class=str(payload.get("mechanism_class") or ""),
            target_slices=[str(item) for item in payload.get("target_slices", []) if item],
            rationale=str(payload.get("rationale") or ""),
            measurements=[str(item) for item in payload.get("measurements", []) if item],
            disconfirming_result=str(payload.get("disconfirming_result") or ""),
            candidate_roles=[str(item) for item in payload.get("candidate_roles", []) if item],
            compatible_mechanisms=[str(item) for item in payload.get("compatible_mechanisms", []) if item],
            affordance_ids=[str(item) for item in payload.get("affordance_ids", []) if item],
            priority=int(payload.get("priority") or 1),
        )


@dataclass(frozen=True)
class TheoryUpdate:
    update_id: str
    hypothesis_id: str
    status: str
    evidence: list[str] = field(default_factory=list)
    implication: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TheoryUpdate":
        return cls(
            update_id=str(payload.get("update_id") or payload.get("id") or ""),
            hypothesis_id=str(payload.get("hypothesis_id") or ""),
            status=str(payload.get("status") or ""),
            evidence=[str(item) for item in payload.get("evidence", []) if item],
            implication=str(payload.get("implication") or ""),
        )


@dataclass(frozen=True)
class ResearchTheory:
    theory_id: str
    summary: str
    primary_hypothesis_id: str
    hypotheses: list[CausalHypothesis]
    experiment_opportunities: list[ResearchOpportunity]
    disconfirmed_explanations: list[str] = field(default_factory=list)
    surprising_observations: list[str] = field(default_factory=list)
    prior_lessons: list[str] = field(default_factory=list)
    uncertainty: str = ""
    confidence: str = "low"

    def __post_init__(self) -> None:
        if not self.theory_id:
            raise ValueError("theory_id must be non-empty")
        if not self.summary:
            raise ValueError("research theory summary must be non-empty")
        if not self.hypotheses:
            raise ValueError("research theory hypotheses must be non-empty")
        hypothesis_ids = [hypothesis.hypothesis_id for hypothesis in self.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise ValueError("research theory hypothesis_ids must be unique")
        if self.primary_hypothesis_id not in set(hypothesis_ids):
            raise ValueError("primary_hypothesis_id must reference a hypothesis")
        opportunity_ids = [opportunity.opportunity_id for opportunity in self.experiment_opportunities]
        if len(opportunity_ids) != len(set(opportunity_ids)):
            raise ValueError("research theory opportunity_ids must be unique")
        known_hypotheses = set(hypothesis_ids)
        for opportunity in self.experiment_opportunities:
            unknown = sorted(set(opportunity.hypothesis_ids) - known_hypotheses)
            if unknown:
                raise ValueError(f"opportunity {opportunity.opportunity_id!r} cites unknown hypothesis_ids: {unknown}")

    @property
    def bottleneck_class(self) -> str:
        primary = next(
            (hypothesis for hypothesis in self.hypotheses if hypothesis.hypothesis_id == self.primary_hypothesis_id),
            self.hypotheses[0],
        )
        return primary.mechanism_class

    @property
    def residual_failure_modes(self) -> list[str]:
        return sorted({hypothesis.mechanism_class for hypothesis in self.hypotheses})

    def to_dict(self) -> dict[str, Any]:
        return {
            "theory_id": self.theory_id,
            "summary": self.summary,
            "primary_hypothesis_id": self.primary_hypothesis_id,
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "experiment_opportunities": [opportunity.to_dict() for opportunity in self.experiment_opportunities],
            "disconfirmed_explanations": list(self.disconfirmed_explanations),
            "surprising_observations": list(self.surprising_observations),
            "prior_lessons": list(self.prior_lessons),
            "uncertainty": self.uncertainty,
            "confidence": self.confidence,
            "bottleneck_class": self.bottleneck_class,
            "residual_failure_modes": self.residual_failure_modes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchTheory":
        hypotheses = [
            CausalHypothesis.from_dict(item)
            for item in payload.get("hypotheses", [])
            if isinstance(item, dict)
        ]
        opportunities = [
            ResearchOpportunity.from_dict(item)
            for item in payload.get("experiment_opportunities", [])
            if isinstance(item, dict)
        ]
        return cls(
            theory_id=str(payload.get("theory_id") or payload.get("id") or ""),
            summary=str(payload.get("summary") or ""),
            primary_hypothesis_id=str(payload.get("primary_hypothesis_id") or ""),
            hypotheses=hypotheses,
            experiment_opportunities=opportunities,
            disconfirmed_explanations=[str(item) for item in payload.get("disconfirmed_explanations", []) if item],
            surprising_observations=[str(item) for item in payload.get("surprising_observations", []) if item],
            prior_lessons=[str(item) for item in payload.get("prior_lessons", []) if item],
            uncertainty=str(payload.get("uncertainty") or ""),
            confidence=str(payload.get("confidence") or "low"),
        )


@dataclass(frozen=True)
class TaskTheory:
    bottleneck_class: str
    residual_failure_modes: list[str]
    label_confusions: list[dict[str, Any]]
    weak_slices: list[str]
    runtime_defects: dict[str, Any]
    output_defects: dict[str, Any]
    tool_defects: dict[str, Any]
    example_coverage: dict[str, Any]
    cost_latency_profile: dict[str, Any]
    confidence: str
    evidence: list[str] = field(default_factory=list)
    experiment_opportunities: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentIntent:
    intent_id: str
    mechanism_class: str
    hypothesis: str
    target_slices: list[str] = field(default_factory=list)
    candidate_roles: list[str] = field(default_factory=list)
    measurements: list[str] = field(default_factory=list)
    affordance_ids: list[str] = field(default_factory=list)
    success_criteria: str = ""
    disconfirming_result: str = ""
    priority: int = 1

    def __post_init__(self) -> None:
        if not self.intent_id:
            raise ValueError("intent_id must be non-empty")
        if self.mechanism_class not in MECHANISM_CLASSES:
            raise ValueError(f"unknown intent mechanism_class {self.mechanism_class!r}")
        unknown_roles = sorted(set(self.candidate_roles) - CANDIDATE_ROLES)
        if unknown_roles:
            raise ValueError(f"unknown intent candidate_roles: {unknown_roles}")
        if not self.affordance_ids:
            raise ValueError("intent affordance_ids must be non-empty")
        if self.priority < 1:
            raise ValueError("intent priority must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentIntent":
        return cls(
            intent_id=str(payload.get("intent_id") or ""),
            mechanism_class=str(payload.get("mechanism_class") or ""),
            hypothesis=str(payload.get("hypothesis") or ""),
            target_slices=[str(item) for item in payload.get("target_slices", []) if item],
            candidate_roles=[str(item) for item in payload.get("candidate_roles", []) if item],
            measurements=[str(item) for item in payload.get("measurements", payload.get("expected_measurements", [])) if item],
            affordance_ids=[str(item) for item in payload.get("affordance_ids", []) if item],
            success_criteria=str(payload.get("success_criteria") or ""),
            disconfirming_result=str(payload.get("disconfirming_result") or ""),
            priority=int(payload.get("priority") or 1),
        )


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
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentSpec":
        raw_mechanism = str(payload.get("mechanism") or "")
        mechanism_class = str(payload.get("mechanism_class") or "")
        return cls(
            experiment_id=str(payload.get("experiment_id") or ""),
            mechanism=mechanism_class,
            hypothesis=str(payload.get("hypothesis") or ""),
            mechanism_label=raw_mechanism if raw_mechanism != mechanism_class else "",
            target_slices=[str(item) for item in payload.get("target_slices", []) if item],
            measurements=[str(item) for item in payload.get("measurements", payload.get("expected_measurements", [])) if item],
            candidate_roles=[str(item) for item in payload.get("candidate_roles", []) if item],
        )


@dataclass(frozen=True)
class ResearchState:
    objective: dict[str, Any]
    budget: dict[str, Any]
    parent: dict[str, Any]
    task_theory: dict[str, Any]
    behavior_profile: dict[str, Any]
    affordances: list[dict[str, Any]]
    prior_experiment_outcomes: list[dict[str, Any]] = field(default_factory=list)
    frontier: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateImplementation:
    intent_id: str
    candidate_id: str
    affordance_ids: list[str]
    transform_family: str
    mechanism_class: str
    candidate_role: str
    hypothesis: str
    intervention: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasurementDecision:
    stage: str
    selected_candidate_ids: list[str]
    rationale: str
    expected_information: str = ""
    risks: str = ""
    skipped_candidate_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_task_theory(
    *,
    summary: CandidateSummary,
    diagnoses: list[FailureDiagnosis],
    objective: OptimizationObjective,
    proposal_example_bank: ProposalExampleBank | None = None,
) -> TaskTheory:
    packet = build_evidence_packet(
        summary=summary,
        diagnoses=diagnoses,
        objective=objective,
        proposal_example_bank=proposal_example_bank,
    )
    diagnostics = packet.behavior_diagnostics
    runtime = dict(packet.runtime_defects)
    tool_interaction = dict(packet.tool_defects)
    invalid_case_ids = list(packet.output_defects.get("invalid_output_case_ids", []))
    confusions = list(packet.label_confusions)
    weak_labels = list(packet.weak_slices)
    evidence = list(packet.evidence)
    if runtime.get("length_finish_case_ids") or runtime.get("parser_fallback_case_ids"):
        bottleneck = "runtime_or_output_defect"
    elif _has_tool_trajectory_defect(tool_interaction):
        bottleneck = "tool_trajectory"
    elif invalid_case_ids:
        bottleneck = "output_contract"
    elif confusions or weak_labels:
        bottleneck = "semantic_boundary_confusion"
    elif objective.mode in {"cost", "latency"}:
        bottleneck = "efficiency_tradeoff"
    elif summary.pass_count < summary.case_count:
        bottleneck = "general_correctness_gap"
    else:
        bottleneck = "no_observed_failures"
    return TaskTheory(
        bottleneck_class=bottleneck,
        residual_failure_modes=list(packet.residual_failure_modes),
        label_confusions=list(packet.label_confusions),
        weak_slices=list(packet.weak_slices),
        runtime_defects=dict(packet.runtime_defects),
        output_defects=dict(packet.output_defects),
        tool_defects=dict(packet.tool_defects),
        example_coverage=dict(packet.example_coverage),
        cost_latency_profile=dict(packet.cost_latency_profile),
        confidence=packet.confidence,
        evidence=evidence,
        experiment_opportunities=_experiment_opportunities(
            bottleneck=bottleneck,
            runtime=runtime,
            invalid_case_ids=invalid_case_ids,
            tool_interaction=tool_interaction,
            confusions=confusions,
            weak_labels=weak_labels,
            example_source_ids=_example_source_ids_by_label(proposal_example_bank),
            objective=objective,
        ),
    )


def build_evidence_packet(
    *,
    summary: CandidateSummary,
    diagnoses: list[FailureDiagnosis],
    objective: OptimizationObjective,
    proposal_example_bank: ProposalExampleBank | None = None,
) -> EvidencePacket:
    diagnostics = build_behavior_diagnostics(summary)
    runtime = dict(diagnostics.get("runtime_reliability") or {})
    tool_interaction = dict(diagnostics.get("tool_interaction") or {})
    invalid_case_ids = list(diagnostics.get("invalid_output_case_ids") or [])
    confusions = list(diagnostics.get("confusions") or [])
    weak_labels = [str(item) for item in diagnostics.get("weak_labels", [])]
    diagnosis_categories = sorted({diagnosis.category for diagnosis in diagnoses if diagnosis.category})
    evidence: list[str] = []
    if runtime.get("length_finish_case_ids") or runtime.get("parser_fallback_case_ids"):
        evidence.append("runtime/output trace defects observed")
    if _has_tool_trajectory_defect(tool_interaction):
        evidence.append("tool/environment trajectory defects observed")
    if invalid_case_ids:
        evidence.append("invalid output failures observed")
    if confusions or weak_labels:
        evidence.append("label or slice confusions observed")
    if objective.mode in {"cost", "latency"}:
        evidence.append(f"{objective.mode} objective active")
    if summary.pass_count < summary.case_count:
        evidence.append("failing cases observed")
    if not evidence:
        evidence.append("current branch has no observed failures")
    label_counts = proposal_example_bank.label_counts if proposal_example_bank is not None else {}
    missing_weak_examples = [label for label in weak_labels if label not in label_counts]
    example_source_ids = _example_source_ids_by_label(proposal_example_bank)
    target_example_labels = _target_example_labels(confusions=confusions, weak_labels=weak_labels)
    return EvidencePacket(
        residual_failure_modes=_residual_failure_modes(
            invalid_case_ids=invalid_case_ids,
            tool_interaction=tool_interaction,
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
        tool_defects={
            "tool_call_counts": tool_interaction.get("tool_call_counts", {}),
            "tool_status_counts": tool_interaction.get("tool_status_counts", {}),
            "turn_outcome_counts": tool_interaction.get("turn_outcome_counts", {}),
            "terminal_reason_counts": tool_interaction.get("terminal_reason_counts", {}),
            "tool_error_case_ids": tool_interaction.get("tool_error_case_ids", []),
            "invalid_tool_call_case_ids": tool_interaction.get("invalid_tool_call_case_ids", []),
            "premature_stop_case_ids": tool_interaction.get("premature_stop_case_ids", []),
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
            "mean_model_calls": summary.mean_model_calls,
            "mean_tool_calls": summary.mean_tool_calls,
            "mean_turns": summary.mean_turns,
            "median_latency_s": summary.median_latency_s,
        },
        confidence="medium" if summary.case_count >= 20 else "low",
        evidence=evidence,
        behavior_diagnostics=diagnostics,
        diagnosis_categories=diagnosis_categories,
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


def compatible_families_for_mechanism(
    mechanism: str,
    *,
    active_families: list[str] | None = None,
) -> list[str]:
    active = set(active_families or MECHANISMS_BY_FAMILY)
    return sorted(
        family
        for family, mechanisms in MECHANISMS_BY_FAMILY.items()
        if family in active and mechanism in mechanisms
    )


def _residual_failure_modes(
    *,
    invalid_case_ids: list[str],
    tool_interaction: dict[str, Any],
    confusions: list[dict[str, Any]],
    weak_labels: list[str],
    diagnosis_categories: list[str],
) -> list[str]:
    modes: list[str] = []
    if invalid_case_ids:
        modes.append("invalid_output")
    if _has_tool_trajectory_defect(tool_interaction):
        modes.append("tool_trajectory")
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
    tool_interaction: dict[str, Any],
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
    if _has_tool_trajectory_defect(tool_interaction):
        target_slices = (
            _slice_ids("invalid_tool", tool_interaction.get("invalid_tool_call_case_ids", []))
            + _slice_ids("tool_error", tool_interaction.get("tool_error_case_ids", []))
            + _slice_ids("premature_stop", tool_interaction.get("premature_stop_case_ids", []))
        )
        opportunities.append(
            {
                "mechanism_class": "tool_selection_policy",
                "target_slices": target_slices or ["tool_trajectory"],
                "candidate_roles": ["atomic", "control"],
                "compatible_mechanisms": [
                    "tool_argument_grounding",
                    "tool_precondition_policy",
                    "interaction_completion_policy",
                ],
                "measurements": [
                    "score_delta",
                    "tool_call_delta",
                    "tool_error_delta",
                    "turn_delta",
                    "non_target_regression",
                ],
                "rationale": "Tool/environment traces indicate failure in action selection, arguments, ordering, or stopping.",
                "disconfirming_result": "Tool trajectory errors persist or success improves only through excessive extra turns.",
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


def _has_tool_trajectory_defect(tool_interaction: dict[str, Any]) -> bool:
    return bool(
        tool_interaction.get("tool_error_case_ids")
        or tool_interaction.get("invalid_tool_call_case_ids")
        or tool_interaction.get("premature_stop_case_ids")
        or tool_interaction.get("turn_outcome_counts")
    )


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
