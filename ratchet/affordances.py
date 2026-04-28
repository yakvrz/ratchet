from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ratchet.experiments import MECHANISMS_BY_FAMILY, mechanism_error_for_family
from ratchet.io import stable_digest
from ratchet.transforms import TransformFamily, transform_registry
from ratchet.types import EditableTarget


@dataclass(frozen=True)
class OptimizationAffordance:
    affordance_id: str
    target_name: str
    target_kind: str
    target_path: str
    transform_family: str
    mechanism_class: str
    allowed_ops: list[str]
    value_schema: dict[str, Any]
    expected_cost_impact: str
    expected_latency_impact: str
    risk_level: str
    required_measurements: list[str]
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_optimization_affordances(
    surface: list[EditableTarget],
    *,
    active_families: list[str] | None = None,
) -> list[OptimizationAffordance]:
    registry = transform_registry()
    family_names = list(active_families or registry)
    affordances: list[OptimizationAffordance] = []
    for target in surface:
        for family_name in family_names:
            family = registry.get(family_name)
            if family is None:
                continue
            if target.kind not in family.supported_edit_kinds:
                continue
            ops = [op for op in target.allowed_ops if op in family.supported_ops]
            if not ops:
                continue
            for mechanism in sorted(MECHANISMS_BY_FAMILY.get(family.name, set())):
                if mechanism_error_for_family(family.name, mechanism) is not None:
                    continue
                affordances.append(_affordance(target=target, family=family, mechanism=mechanism, ops=ops))
    return sorted(affordances, key=lambda item: item.affordance_id)


def validate_candidate_affordances(
    *,
    affordance_ids: list[str],
    transform_family: str,
    mechanism_class: str,
    operations: list[dict[str, str]],
    affordances: list[OptimizationAffordance],
) -> str | None:
    if not affordance_ids:
        return "candidate must cite at least one affordance_id"
    by_id = {affordance.affordance_id: affordance for affordance in affordances}
    selected: list[OptimizationAffordance] = []
    unknown = [affordance_id for affordance_id in affordance_ids if affordance_id not in by_id]
    if unknown:
        return "unknown affordance_id(s): " + ", ".join(unknown[:6])
    for affordance_id in affordance_ids:
        affordance = by_id[affordance_id]
        if affordance.transform_family != transform_family:
            return (
                f"affordance {affordance_id!r} belongs to transform family "
                f"{affordance.transform_family!r}, not {transform_family!r}"
            )
        if affordance.mechanism_class != mechanism_class:
            return (
                f"affordance {affordance_id!r} belongs to mechanism "
                f"{affordance.mechanism_class!r}, not {mechanism_class!r}"
            )
        selected.append(affordance)
    if not operations:
        if any(affordance.target_kind == "few_shot" for affordance in selected):
            return None
        return "non-few-shot affordance candidates must include patch operations"
    for operation in operations:
        op_name = operation.get("op", "")
        target_name = operation.get("target", "")
        if not any(
            (target_name in {affordance.target_name, affordance.target_path})
            and op_name in affordance.allowed_ops
            for affordance in selected
        ):
            return f"operation {op_name!r} on target {target_name!r} is not covered by cited affordance_ids"
    return None


def _affordance(
    *,
    target: EditableTarget,
    family: TransformFamily,
    mechanism: str,
    ops: list[str],
) -> OptimizationAffordance:
    digest = stable_digest(
        {
            "target": target.name,
            "family": family.name,
            "mechanism": mechanism,
            "ops": ops,
        }
    )[:10]
    effects = family.expected_effects
    return OptimizationAffordance(
        affordance_id=f"aff_{digest}",
        target_name=target.name,
        target_kind=target.kind,
        target_path=target.path,
        transform_family=family.name,
        mechanism_class=mechanism,
        allowed_ops=list(ops),
        value_schema=dict(target.value_schema),
        expected_cost_impact=_impact(effects, "cost"),
        expected_latency_impact=_impact(effects, "latency"),
        risk_level=_risk_level(family),
        required_measurements=list(family.required_measurements),
        description=target.description,
    )


def _impact(effects: dict[str, str], key: str) -> str:
    value = effects.get(key)
    if value is None:
        return "unknown"
    return value


def _risk_level(family: TransformFamily) -> str:
    if family.complexity_cost >= 1.5:
        return "medium"
    if family.risks:
        return "low_medium"
    return "low"
