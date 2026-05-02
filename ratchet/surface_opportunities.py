from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.surfaces import SurfaceSpec, SurfaceTarget, surface_targets
from ratchet.types import OptimizationObjective


@dataclass(frozen=True)
class OpportunityComposition:
    can_pair_with: list[str] = field(default_factory=list)
    should_not_pair_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceOpportunity:
    surface_opportunity_id: str
    label: str
    family: str
    mechanism: str
    target_name: str
    target_kind: str
    target_path: str
    ops: list[str]
    value_schema: dict[str, Any]
    semantic_role: str
    behavioral_axes: list[str]
    expected_scope: str
    risk: str
    measurements: list[str]
    composition: OpportunityComposition = field(default_factory=OpportunityComposition)
    suitability: float = 0.0
    evidence: list[str] = field(default_factory=list)
    budget_hint: float = 0.0
    expected_cost_impact: str = "unknown"
    expected_latency_impact: str = "unknown"
    description: str = ""

    @property
    def surface_mechanism(self) -> str:
        return self.family

    @property
    def mechanism_class(self) -> str:
        return self.mechanism

    @property
    def allowed_ops(self) -> list[str]:
        return list(self.ops)

    @property
    def required_measurements(self) -> list[str]:
        return list(self.measurements)

    @property
    def risk_level(self) -> str:
        if self.risk in {"high", "neighbor_label_regression", "contract_regression"}:
            return "medium"
        if self.risk in {"medium", "cost_latency_regression"}:
            return "low_medium"
        return "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_opportunity_id": self.surface_opportunity_id,
            "label": self.label,
            "family": self.family,
            "mechanism": self.mechanism,
            "target": self.target_name,
            "target_kind": self.target_kind,
            "target_path": self.target_path,
            "ops": list(self.ops),
            "value_schema": dict(self.value_schema),
            "semantic_role": self.semantic_role,
            "behavioral_axes": list(self.behavioral_axes),
            "expected_scope": self.expected_scope,
            "risk": self.risk,
            "measurements": list(self.measurements),
            "composition": self.composition.to_dict(),
            "suitability": self.suitability,
            "evidence": list(self.evidence),
            "budget_hint": self.budget_hint,
            "expected_cost_impact": self.expected_cost_impact,
            "expected_latency_impact": self.expected_latency_impact,
            "description": self.description,
        }


def generate_surface_opportunities(
    surface: SurfaceSpec,
    *,
    objective: OptimizationObjective | None = None,
    active_mechanisms: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> list[SurfaceOpportunity]:
    objective = objective or OptimizationObjective()
    if not isinstance(surface, SurfaceSpec):
        raise TypeError(f"generate_surface_opportunities requires SurfaceSpec, got {type(surface).__name__}.")
    return _generate_surface_opportunities(
        surface,
        objective=objective,
        active_mechanisms=active_mechanisms,
        evidence=evidence or {},
    )


def _generate_surface_opportunities(
    surface: SurfaceSpec,
    *,
    objective: OptimizationObjective,
    active_mechanisms: list[str] | None,
    evidence: dict[str, Any],
) -> list[SurfaceOpportunity]:
    surface_opportunities: list[SurfaceOpportunity] = []
    for target in surface_targets(surface):
        ops = sorted(target.allowed_ops)
        if not ops:
            continue
        mechanism = _surface_mechanism(target)
        if active_mechanisms is not None and mechanism not in set(active_mechanisms):
            continue
        surface_opportunities.append(
            SurfaceOpportunity(
                surface_opportunity_id=_surface_opportunity_id(mechanism, target),
                label=f"{target.kind} surface: {target.name}",
                family="surface_program",
                mechanism=mechanism,
                target_name=target.name,
                target_kind=target.kind,
                target_path=target.path,
                ops=ops,
                value_schema=target.value_schema,
                semantic_role=target.semantics.role,
                behavioral_axes=_axes_for_target(mechanism, target),
                expected_scope=target.semantics.scope,
                risk=_risk_for_target(target, "medium"),
                measurements=_surface_measurements(target),
                suitability=_suitability(
                    mechanism=mechanism,
                    target=target,
                    objective=objective,
                    evidence=evidence,
                ),
                evidence=_evidence_for(mechanism, target, evidence),
                budget_hint=_surface_budget_hint(target),
                expected_cost_impact=_surface_cost_impact(target),
                expected_latency_impact=_surface_latency_impact(target),
                description=target.description,
            )
        )
    return sorted(
        _dedupe_surface_opportunities(surface_opportunities),
        key=lambda item: (-item.suitability, item.surface_opportunity_id),
    )


def _surface_mechanism(target: SurfaceTarget) -> str:
    if target.kind == "instruction":
        return "surface_context"
    if target.kind == "output":
        return "surface_output"
    if target.kind == "state":
        return "surface_state"
    if target.kind == "tool":
        return "surface_tool_loop"
    if target.kind == "model":
        return "surface_model"
    if target.kind == "response":
        return "surface_response"
    if target.kind == "few_shot":
        return "surface_examples"
    if target.kind == "runtime":
        return "surface_runtime"
    return f"surface_{target.kind}"


def _surface_opportunity_id(mechanism: str, target: SurfaceTarget) -> str:
    safe_name = _canonical_segment(target.name)
    return f"surface.{mechanism}.{safe_name}"


def _surface_measurements(target: SurfaceTarget) -> list[str]:
    base = ["score_delta", "regression_cases", "cost_delta", "latency_delta"]
    if target.kind == "tool":
        return [*base, "tool_call_delta", "tool_error_delta", "turn_delta"]
    if target.kind in {"state", "response"}:
        return [*base, "runtime_error_delta"]
    return base


def _surface_budget_hint(target: SurfaceTarget) -> float:
    if target.kind in {"model", "tool"}:
        return 0.8
    if target.kind in {"state", "response", "runtime"}:
        return 0.7
    return 0.5


def _surface_cost_impact(target: SurfaceTarget) -> str:
    if target.kind == "model":
        return "model-dependent"
    if target.kind in {"instruction", "output", "few_shot"}:
        return "token_overhead"
    return "low"


def _surface_latency_impact(target: SurfaceTarget) -> str:
    if target.kind == "model":
        return "model-dependent"
    if target.kind in {"tool", "state", "response"}:
        return "low"
    return "token-dependent"


def validate_candidate_surface_applications(
    *,
    applications: list[Any],
    surface_opportunities: list[SurfaceOpportunity],
) -> str | None:
    if not applications:
        return "candidate must include at least one surface opportunity application"
    by_id = {surface_opportunity.surface_opportunity_id: surface_opportunity for surface_opportunity in surface_opportunities}
    selection_count = 0
    for application in applications:
        surface_opportunity_id = str(getattr(application, "surface_opportunity_id", ""))
        surface_opportunity = by_id.get(surface_opportunity_id)
        if surface_opportunity is None:
            return f"unknown surface_opportunity_id {surface_opportunity_id!r}"
        selection = getattr(application, "selection", None)
        source_ids = selection.get("source_case_ids") if isinstance(selection, dict) else None
        if source_ids:
            selection_count += 1
            if surface_opportunity.target_kind != "few_shot":
                return f"surface opportunity {surface_opportunity_id!r} does not support example selection"
            if not isinstance(source_ids, list) or not all(isinstance(item, str) and item for item in source_ids):
                return f"surface opportunity {surface_opportunity_id!r} example selection requires non-empty source_case_ids"
        elif source_ids is not None and not isinstance(source_ids, list):
            return f"surface opportunity {surface_opportunity_id!r} example selection requires source_case_ids[]"
        else:
            continue
    return None


def surface_opportunity_family(surface_opportunity_id: str) -> str:
    return surface_opportunity_id.split(".", 1)[0] if "." in surface_opportunity_id else ""


def surface_opportunity_mechanism(surface_opportunity_id: str) -> str:
    parts = surface_opportunity_id.split(".")
    return parts[1] if len(parts) > 1 else ""


def _canonical_segment(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace(" ", "_").lower()


def _axes_for_target(mechanism: str, target: SurfaceTarget) -> list[str]:
    return _merge_unique(_axes_for_mechanism(mechanism, target.semantics.role), target.semantics.axes)


def _risk_for_target(target: SurfaceTarget, default: str) -> str:
    return target.semantics.risks[0] if target.semantics.risks else default


def _axes_for_mechanism(mechanism: str, role: str) -> list[str]:
    axes_by_mechanism = {
        "surface_context": ["context_graph", "instruction_ordering", "policy_visibility"],
        "surface_output": ["format_validity", "external_contract"],
        "surface_state": ["typed_state", "fact_memory", "state_visibility"],
        "surface_tool_loop": ["tool_choice", "argument_grounding", "precondition_checking", "completion_integrity"],
        "surface_model": ["model_config", "cost_latency_tradeoff", "capability_limit"],
        "surface_response": ["claim_support", "final_response_guarding"],
        "surface_examples": ["example_anchoring", "target_slice_recall"],
        "surface_runtime": ["runtime_control", "retry_limits", "loop_termination"],
    }
    return axes_by_mechanism.get(mechanism, [role])


def _suitability(*, mechanism: str, target: SurfaceTarget, objective: OptimizationObjective, evidence: dict[str, Any]) -> float:
    score = 0.25
    role = target.semantics.role
    bottleneck = str(evidence.get("bottleneck_class") or "")
    residual_modes = {str(item) for item in evidence.get("residual_failure_modes", []) if item}
    evidence_text = str(evidence).lower()
    if mechanism == "surface_context" and (
        bottleneck in {"semantic_boundary_confusion", "general_correctness_gap"}
        or {"label_confusion", "weak_slices", "general_correctness_gap"} & residual_modes
    ):
        score += 0.35
    if mechanism in {"surface_output", "surface_response"} and (
        bottleneck == "output_contract" or evidence.get("invalid_output") or "invalid_output" in residual_modes
    ):
        score += 0.45
    if mechanism == "surface_response" and (
        "clarification" in evidence_text or "ambiguous" in evidence_text or "ambiguity" in evidence_text
    ):
        score += 0.35
    if mechanism == "surface_runtime" and evidence.get("runtime_defect"):
        score += 0.45
    if mechanism == "surface_tool_loop" and (
        evidence.get("tool_trajectory_defect") or "tool_trajectory" in residual_modes
    ):
        score += 0.45
    if mechanism == "surface_examples" and evidence.get("example_coverage"):
        score += 0.30
    if mechanism == "surface_model" and objective.mode == "correctness":
        score += 0.15
    if mechanism == "surface_model" and objective.mode in {"cost", "latency"}:
        score += 0.45
    if mechanism == "surface_context" and role in {
        "argument_extraction_policy",
        "confusable_label_policy",
        "decision_policy",
        "label_alias_mapping",
        "label_description",
        "label_space",
        "schema_adherence_policy",
        "tool_description",
        "tool_policy",
    }:
        score += 0.10
    if mechanism in {"surface_output", "surface_response"} and role in {
        "external_output_contract",
        "output_budget_control",
        "output_format_rule",
        "schema_adherence_policy",
        "response_guarding",
    }:
        score += 0.10
    if mechanism == "surface_runtime" and role in {
        "output_budget_control",
        "runtime_control",
        "verifier_retry_policy",
    }:
        score += 0.10
    if mechanism == "surface_model" and role in {
        "model_choice",
        "reasoning_effort_control",
        "output_budget_control",
        "runtime_control",
    }:
        score += 0.10
    if mechanism == "surface_tool_loop" and role in {
        "tool_description",
        "tool_policy",
        "tool_relevance_boundary",
    }:
        score += 0.15
    score += min(target.semantics.confidence, 1.0) * 0.05
    if target.name in set(evidence.get("diagnosis_target_names") or []):
        score += 0.20
    return round(min(score, 1.0), 3)


def _evidence_for(mechanism: str, target: SurfaceTarget, evidence: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    if evidence.get("bottleneck_class"):
        rows.append(f"research theory bottleneck: {evidence['bottleneck_class']}")
    if target.name in set(evidence.get("diagnosis_target_names") or []):
        rows.append(f"diagnosis targeted {target.name}")
    if target.semantics.source != "default":
        rows.append(
            f"target semantics: {target.semantics.role} ({target.semantics.source}, "
            f"confidence={target.semantics.confidence:.2f})"
        )
    if mechanism == "surface_examples" and evidence.get("example_coverage"):
        rows.append("proposal-safe train examples cover relevant labels")
    if mechanism == "surface_runtime" and evidence.get("runtime_defect"):
        rows.append("runtime finish/parser defects observed")
    if mechanism in {"surface_output", "surface_response"} and evidence.get("invalid_output"):
        rows.append("invalid-output failures observed")
    if mechanism == "surface_tool_loop" and evidence.get("tool_trajectory_defect"):
        rows.append("tool/environment trajectory defects observed")
    return rows


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*left, *right]:
        if item and item not in merged:
            merged.append(item)
    return merged


def _dedupe_surface_opportunities(surface_opportunities: list[SurfaceOpportunity]) -> list[SurfaceOpportunity]:
    by_id: dict[str, SurfaceOpportunity] = {}
    for surface_opportunity in surface_opportunities:
        current = by_id.get(surface_opportunity.surface_opportunity_id)
        if current is None or surface_opportunity.suitability > current.suitability:
            by_id[surface_opportunity.surface_opportunity_id] = surface_opportunity
    return list(by_id.values())
