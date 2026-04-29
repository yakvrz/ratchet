from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from ratchet.experiments import MECHANISMS_BY_FAMILY, mechanism_error_for_family
from ratchet.transforms import TransformFamily, transform_registry
from ratchet.types import EditableTarget, OptimizationObjective


@dataclass(frozen=True)
class AffordanceComposition:
    can_pair_with: list[str] = field(default_factory=list)
    should_not_pair_with: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizationAffordance:
    affordance_id: str
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
    composition: AffordanceComposition = field(default_factory=AffordanceComposition)
    suitability: float = 0.0
    evidence: list[str] = field(default_factory=list)
    budget_hint: float = 0.0
    expected_cost_impact: str = "unknown"
    expected_latency_impact: str = "unknown"
    description: str = ""

    @property
    def transform_family(self) -> str:
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
            "affordance_id": self.affordance_id,
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


class AffordanceProvider(Protocol):
    def generate(
        self,
        target: EditableTarget,
        *,
        family: TransformFamily,
        objective: OptimizationObjective,
        active_families: set[str],
        evidence: dict[str, Any],
    ) -> list[OptimizationAffordance]:
        ...


def generate_optimization_affordances(
    surface: list[EditableTarget],
    *,
    objective: OptimizationObjective | None = None,
    active_families: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
) -> list[OptimizationAffordance]:
    objective = objective or OptimizationObjective()
    registry = transform_registry()
    active = set(active_families or registry)
    providers = _providers()
    affordances: list[OptimizationAffordance] = []
    for target in surface:
        for family_name in sorted(active):
            family = registry.get(family_name)
            if family is None or target.kind not in family.supported_edit_kinds:
                continue
            for provider in providers:
                affordances.extend(
                    provider.generate(
                        target,
                        family=family,
                        objective=objective,
                        active_families=active,
                        evidence=evidence or {},
                    )
                )
    return sorted(
        _dedupe_affordances(affordances),
        key=lambda item: (-item.suitability, item.affordance_id),
    )


def validate_candidate_applications(
    *,
    applications: list[Any],
    affordances: list[OptimizationAffordance],
) -> str | None:
    if not applications:
        return "candidate must include at least one affordance application"
    by_id = {affordance.affordance_id: affordance for affordance in affordances}
    for application in applications:
        affordance_id = str(getattr(application, "affordance_id", ""))
        affordance = by_id.get(affordance_id)
        if affordance is None:
            return f"unknown affordance_id {affordance_id!r}"
        operation = getattr(application, "operation", None)
        selection = getattr(application, "selection", None)
        if operation is not None:
            if operation.op not in affordance.ops:
                return f"operation {operation.op!r} is not allowed by affordance {affordance_id!r}"
            if operation.target not in {affordance.target_name, affordance.target_path}:
                return f"operation target {operation.target!r} is not covered by affordance {affordance_id!r}"
        elif selection:
            if affordance.family != "targeted_few_shot":
                return f"affordance {affordance_id!r} does not support example selection"
            source_ids = selection.get("source_case_ids")
            if not isinstance(source_ids, list) or not all(isinstance(item, str) and item for item in source_ids):
                return f"affordance {affordance_id!r} example selection requires non-empty source_case_ids"
        else:
            return f"affordance application {affordance_id!r} must include operation or selection"
    return None


def affordance_family(affordance_id: str) -> str:
    return affordance_id.split(".", 1)[0] if "." in affordance_id else ""


def affordance_mechanism(affordance_id: str) -> str:
    parts = affordance_id.split(".")
    return parts[1] if len(parts) > 1 else ""


class PromptAffordanceProvider:
    def generate(
        self,
        target: EditableTarget,
        *,
        family: TransformFamily,
        objective: OptimizationObjective,
        active_families: set[str],
        evidence: dict[str, Any],
    ) -> list[OptimizationAffordance]:
        if family.name != "prompt_rewrite" or target.kind != "instruction":
            return []
        role = _semantic_role(target)
        mechanisms = ["semantic_boundary_rewrite", "output_contract_fix"]
        if evidence.get("runtime_defect"):
            mechanisms.append("runtime_defect_fix")
        return [
            _make_affordance(
                target=target,
                family=family,
                mechanism=mechanism,
                label=_label("Rewrite", role),
                semantic_role=role,
                behavioral_axes=_axes_for_mechanism(mechanism, role),
                expected_scope="slice" if mechanism == "semantic_boundary_rewrite" else "global",
                risk="neighbor_label_regression" if mechanism == "semantic_boundary_rewrite" else "contract_regression",
                composition=AffordanceComposition(
                    can_pair_with=["targeted_few_shot.contrastive_examples.*", "targeted_few_shot.representative_examples.*"],
                    should_not_pair_with=[],
                ),
                suitability=_suitability(mechanism=mechanism, target=target, objective=objective, evidence=evidence),
                evidence=_evidence_for(mechanism, target, evidence),
            )
            for mechanism in mechanisms
            if _supports(family, target, mechanism)
        ]


class OutputContractAffordanceProvider:
    def generate(self, target: EditableTarget, *, family: TransformFamily, objective: OptimizationObjective, active_families: set[str], evidence: dict[str, Any]) -> list[OptimizationAffordance]:
        if family.name != "output_contract_tightening" or target.kind != "output":
            return []
        return [
            _make_affordance(
                target=target,
                family=family,
                mechanism="output_contract_fix",
                label="Tighten output contract",
                semantic_role="output_contract",
                behavioral_axes=["format_validity", "parser_compatibility"],
                expected_scope="global",
                risk="contract_regression",
                composition=AffordanceComposition(can_pair_with=["runtime_tuning.output_contract_fix.*"]),
                suitability=_suitability(mechanism="output_contract_fix", target=target, objective=objective, evidence=evidence),
                evidence=_evidence_for("output_contract_fix", target, evidence),
            )
        ]


class FewShotAffordanceProvider:
    def generate(self, target: EditableTarget, *, family: TransformFamily, objective: OptimizationObjective, active_families: set[str], evidence: dict[str, Any]) -> list[OptimizationAffordance]:
        if family.name != "targeted_few_shot" or target.kind != "few_shot":
            return []
        rows = []
        for mechanism in ("representative_examples", "contrastive_examples"):
            rows.append(
                _make_affordance(
                    target=target,
                    family=family,
                    mechanism=mechanism,
                    label=("Select contrastive examples" if mechanism == "contrastive_examples" else "Select representative examples"),
                    semantic_role="example_bank",
                    behavioral_axes=["example_anchoring", "classification_boundary"],
                    expected_scope="slice",
                    risk="neighbor_label_regression",
                    composition=AffordanceComposition(
                        can_pair_with=["prompt_rewrite.semantic_boundary_rewrite.*"],
                        should_not_pair_with=[],
                    ),
                    suitability=_suitability(mechanism=mechanism, target=target, objective=objective, evidence=evidence),
                    evidence=_evidence_for(mechanism, target, evidence),
                )
            )
        return rows


class ModelAffordanceProvider:
    def generate(self, target: EditableTarget, *, family: TransformFamily, objective: OptimizationObjective, active_families: set[str], evidence: dict[str, Any]) -> list[OptimizationAffordance]:
        if family.name != "model_substitution" or target.kind != "model":
            return []
        mechanisms = ["model_capability_probe", "efficiency_probe"]
        return [
            _make_affordance(
                target=target,
                family=family,
                mechanism=mechanism,
                label="Probe model capability" if mechanism == "model_capability_probe" else "Probe model efficiency",
                semantic_role="model_choice",
                behavioral_axes=["model_capability"] if mechanism == "model_capability_probe" else ["cost_latency_tradeoff"],
                expected_scope="global",
                risk="cost_latency_regression" if mechanism == "model_capability_probe" else "quality_regression",
                composition=AffordanceComposition(should_not_pair_with=["prompt_rewrite.semantic_boundary_rewrite.*"]),
                suitability=_suitability(mechanism=mechanism, target=target, objective=objective, evidence=evidence),
                evidence=_evidence_for(mechanism, target, evidence),
            )
            for mechanism in mechanisms
        ]


class RuntimeAffordanceProvider:
    def generate(self, target: EditableTarget, *, family: TransformFamily, objective: OptimizationObjective, active_families: set[str], evidence: dict[str, Any]) -> list[OptimizationAffordance]:
        if family.name != "runtime_tuning" or target.kind != "runtime":
            return []
        mechanisms = ["runtime_defect_fix", "efficiency_probe"]
        if "output" in target.name:
            mechanisms.append("output_contract_fix")
        return [
            _make_affordance(
                target=target,
                family=family,
                mechanism=mechanism,
                label=_label("Tune", _semantic_role(target)),
                semantic_role=_semantic_role(target),
                behavioral_axes=_axes_for_mechanism(mechanism, _semantic_role(target)),
                expected_scope="global",
                risk="cost_latency_regression",
                composition=AffordanceComposition(can_pair_with=["output_contract_tightening.output_contract_fix.*"]),
                suitability=_suitability(mechanism=mechanism, target=target, objective=objective, evidence=evidence),
                evidence=_evidence_for(mechanism, target, evidence),
            )
            for mechanism in mechanisms
            if _supports(family, target, mechanism)
        ]


class PassthroughAffordanceProvider:
    def generate(self, target: EditableTarget, *, family: TransformFamily, objective: OptimizationObjective, active_families: set[str], evidence: dict[str, Any]) -> list[OptimizationAffordance]:
        if family.name not in {"retrieval_tuning", "tool_policy_revision", "verifier_retry"}:
            return []
        mechanisms = sorted(MECHANISMS_BY_FAMILY.get(family.name, set()) - {"ablation"})
        return [
            _make_affordance(
                target=target,
                family=family,
                mechanism=mechanism,
                label=_label("Revise", _semantic_role(target)),
                semantic_role=_semantic_role(target),
                behavioral_axes=_axes_for_mechanism(mechanism, _semantic_role(target)),
                expected_scope="slice" if mechanism == "semantic_boundary_rewrite" else "global",
                risk="quality_regression",
                suitability=_suitability(mechanism=mechanism, target=target, objective=objective, evidence=evidence),
                evidence=_evidence_for(mechanism, target, evidence),
            )
            for mechanism in mechanisms
            if _supports(family, target, mechanism)
        ]


def _providers() -> list[AffordanceProvider]:
    return [
        PromptAffordanceProvider(),
        OutputContractAffordanceProvider(),
        FewShotAffordanceProvider(),
        ModelAffordanceProvider(),
        RuntimeAffordanceProvider(),
        PassthroughAffordanceProvider(),
    ]


def _make_affordance(
    *,
    target: EditableTarget,
    family: TransformFamily,
    mechanism: str,
    label: str,
    semantic_role: str,
    behavioral_axes: list[str],
    expected_scope: str,
    risk: str,
    suitability: float,
    evidence: list[str],
    composition: AffordanceComposition | None = None,
) -> OptimizationAffordance:
    ops = [op for op in target.allowed_ops if op in family.supported_ops]
    return OptimizationAffordance(
        affordance_id=_affordance_id(family.name, mechanism, target),
        label=label,
        family=family.name,
        mechanism=mechanism,
        target_name=target.name,
        target_kind=target.kind,
        target_path=target.path,
        ops=ops,
        value_schema=dict(target.value_schema),
        semantic_role=semantic_role,
        behavioral_axes=behavioral_axes,
        expected_scope=expected_scope,
        risk=risk,
        measurements=_measurements_for(mechanism, family),
        composition=composition or AffordanceComposition(),
        suitability=suitability,
        evidence=evidence,
        budget_hint=max(0.05, round(suitability / 4.0, 3)),
        expected_cost_impact=_impact(family.expected_effects, "cost"),
        expected_latency_impact=_impact(family.expected_effects, "latency"),
        description=target.description,
    )


def _supports(family: TransformFamily, target: EditableTarget, mechanism: str) -> bool:
    return (
        target.kind in family.supported_edit_kinds
        and bool(set(target.allowed_ops) & set(family.supported_ops))
        and mechanism_error_for_family(family.name, mechanism) is None
    )


def _affordance_id(family: str, mechanism: str, target: EditableTarget) -> str:
    return ".".join(
        [
            family,
            mechanism,
            target.kind,
            _canonical_segment(target.name),
        ]
    )


def _canonical_segment(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace(" ", "_").lower()


def _semantic_role(target: EditableTarget) -> str:
    name = target.name.lower()
    if "label" in name or "intent" in name or "alias" in name:
        return "label_alias_mapping"
    if "system" in name or "instruction" in name:
        return "task_instructions"
    if "output" in name or target.kind == "output":
        return "output_contract"
    if "runtime" in name or target.kind == "runtime":
        return "runtime_control"
    if target.kind == "few_shot":
        return "example_bank"
    return f"{target.kind}_policy"


def _label(verb: str, role: str) -> str:
    return f"{verb} {role.replace('_', ' ')}"


def _axes_for_mechanism(mechanism: str, role: str) -> list[str]:
    axes_by_mechanism = {
        "semantic_boundary_rewrite": ["classification_boundary", "confusion_resolution"],
        "output_contract_fix": ["format_validity", "contract_preservation"],
        "runtime_defect_fix": ["runtime_reliability", "completion_integrity"],
        "representative_examples": ["example_anchoring", "target_slice_recall"],
        "contrastive_examples": ["classification_boundary", "neighbor_label_regression"],
        "model_capability_probe": ["model_capability", "global_correctness"],
        "efficiency_probe": ["cost_latency_tradeoff", "correctness_preservation"],
    }
    return axes_by_mechanism.get(mechanism, [role])


def _measurements_for(mechanism: str, family: TransformFamily) -> list[str]:
    measurements_by_mechanism = {
        "semantic_boundary_rewrite": ["target_slice_score_delta", "confusion_delta", "non_target_regression"],
        "output_contract_fix": ["invalid_output_delta", "score_delta", "non_target_regression"],
        "runtime_defect_fix": ["finish_reason_delta", "invalid_output_delta", "score_delta", "latency_delta"],
        "representative_examples": ["target_slice_score_delta", "example_token_delta", "non_target_regression"],
        "contrastive_examples": ["target_label_score_delta", "neighbor_label_regression", "example_token_delta"],
        "model_capability_probe": ["score_delta", "cost_delta", "latency_delta"],
        "efficiency_probe": ["cost_delta", "latency_delta", "correctness_guard"],
        "ablation": ["score_delta", "complexity_delta", "cost_delta"],
    }
    return measurements_by_mechanism.get(mechanism, list(family.required_measurements))


def _suitability(*, mechanism: str, target: EditableTarget, objective: OptimizationObjective, evidence: dict[str, Any]) -> float:
    score = 0.25
    bottleneck = str(evidence.get("bottleneck_class") or "")
    if mechanism == "semantic_boundary_rewrite" and bottleneck == "semantic_boundary_confusion":
        score += 0.45
    if mechanism == "output_contract_fix" and (bottleneck == "output_contract" or evidence.get("invalid_output")):
        score += 0.45
    if mechanism == "runtime_defect_fix" and evidence.get("runtime_defect"):
        score += 0.45
    if mechanism in {"representative_examples", "contrastive_examples"} and evidence.get("example_coverage"):
        score += 0.30
    if mechanism == "model_capability_probe" and objective.mode == "correctness":
        score += 0.15
    if mechanism == "efficiency_probe" and objective.mode in {"cost", "latency"}:
        score += 0.45
    if target.name in set(evidence.get("diagnosis_target_names") or []):
        score += 0.20
    return round(min(score, 1.0), 3)


def _evidence_for(mechanism: str, target: EditableTarget, evidence: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    if evidence.get("bottleneck_class"):
        rows.append(f"task theory bottleneck: {evidence['bottleneck_class']}")
    if target.name in set(evidence.get("diagnosis_target_names") or []):
        rows.append(f"diagnosis targeted {target.name}")
    if mechanism in {"representative_examples", "contrastive_examples"} and evidence.get("example_coverage"):
        rows.append("proposal-safe train examples cover relevant labels")
    if mechanism == "runtime_defect_fix" and evidence.get("runtime_defect"):
        rows.append("runtime finish/parser defects observed")
    if mechanism == "output_contract_fix" and evidence.get("invalid_output"):
        rows.append("invalid-output failures observed")
    return rows


def _impact(effects: dict[str, str], key: str) -> str:
    return effects.get(key, "unknown")


def _dedupe_affordances(affordances: list[OptimizationAffordance]) -> list[OptimizationAffordance]:
    by_id: dict[str, OptimizationAffordance] = {}
    for affordance in affordances:
        current = by_id.get(affordance.affordance_id)
        if current is None or affordance.suitability > current.suitability:
            by_id[affordance.affordance_id] = affordance
    return list(by_id.values())
