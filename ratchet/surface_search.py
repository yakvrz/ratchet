from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ratchet.candidates import CandidateProposal
from ratchet.transform_program import TransformPatch


TRANSFORM_LIFECYCLE_STATES = {
    "available",
    "active",
    "promotable_dev",
    "paused",
    "constrained",
}


@dataclass(frozen=True)
class TransformContextKey:
    family: str
    target_names: tuple[str, ...] = ()
    ops: tuple[str, ...] = ()
    target_slice: str = "global"
    mechanism: tuple[str, ...] = ()
    transform_instance: str = "candidate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _normalize_token(self.family, default="unknown"))
        object.__setattr__(self, "target_names", tuple(sorted(_normalize_token(item) for item in self.target_names if item)))
        object.__setattr__(self, "ops", tuple(sorted(_normalize_token(item) for item in self.ops if item)))
        object.__setattr__(self, "target_slice", _normalize_token(self.target_slice, default="global"))
        object.__setattr__(self, "mechanism", tuple(sorted(_normalize_token(item) for item in self.mechanism if item)))
        object.__setattr__(self, "transform_instance", _normalize_token(self.transform_instance, default="candidate"))

    @property
    def id(self) -> str:
        return "|".join(
            [
                self.family,
                ",".join(self.target_names) or "-",
                ",".join(self.ops) or "-",
                self.target_slice,
                ",".join(self.mechanism) or "generic",
            ]
        )

    @property
    def scope_id(self) -> str:
        return "|".join(
            [
                self.family,
                ",".join(self.target_names) or "-",
                ",".join(self.ops) or "-",
                self.target_slice,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope_id": self.scope_id,
            "family": self.family,
            "target_names": list(self.target_names),
            "ops": list(self.ops),
            "target_slice": self.target_slice,
            "mechanism": list(self.mechanism),
            "transform_instance": self.transform_instance,
        }

    @classmethod
    def from_candidate(cls, candidate: CandidateProposal) -> TransformContextKey:
        patches = tuple(candidate.program.patches)
        return cls(
            family=candidate.surface_mechanism,
            target_names=tuple(_transform_patch_target(patch) for patch in patches),
            ops=tuple(patch.op.op for patch in patches),
            target_slice=candidate.target_slice,
            mechanism=(
                *tuple(_transform_patch_mechanism_signature(patch) for patch in patches),
                *_parameter_mechanism_signature(candidate.transform_parameters),
            ),
            transform_instance=candidate.transform_instance or candidate.hypothesis or "candidate",
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TransformContextKey:
        existing = row.get("transform_context")
        if isinstance(existing, dict):
            return cls(
                family=str(existing.get("family") or row.get("surface_mechanism") or "unknown"),
                target_names=tuple(str(item) for item in existing.get("target_names", [])),
                ops=tuple(str(item) for item in existing.get("ops", [])),
                target_slice=str(existing.get("target_slice") or row.get("target_slice") or "global"),
                mechanism=tuple(str(item) for item in existing.get("mechanism", [])),
                transform_instance=str(existing.get("transform_instance") or row.get("transform_instance") or "candidate"),
            )
        candidate_payload = row.get("proposal_candidate") if isinstance(row.get("proposal_candidate"), dict) else {}
        if not candidate_payload:
            candidate_payload = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        program_payload = row.get("proposal") or candidate_payload.get("program") or {}
        raw_patches = program_payload.get("patches", []) if isinstance(program_payload, dict) else []
        patches = [TransformPatch.from_dict(item) for item in raw_patches if isinstance(item, dict)]
        return cls(
            family=str(row.get("surface_mechanism") or "unknown"),
            target_names=tuple(_transform_patch_target(patch) for patch in patches),
            ops=tuple(patch.op.op for patch in patches),
            target_slice=str(row.get("target_slice") or "global"),
            mechanism=tuple(_transform_patch_mechanism_signature(patch) for patch in patches),
            transform_instance=str(row.get("transform_instance") or row.get("hypothesis") or "candidate"),
        )


@dataclass(frozen=True)
class TransformContextState:
    key: TransformContextKey
    state: str
    suitability: float
    reason: str
    evidence: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    accepted_count: int = 0
    rejected_count: int = 0
    recent_result_count: int = 0
    last_score_delta: float | None = None

    def __post_init__(self) -> None:
        if self.state not in TRANSFORM_LIFECYCLE_STATES:
            raise ValueError(f"Unsupported transform context lifecycle state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "state": self.state,
            "suitability": self.suitability,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "constraints": list(self.constraints),
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "recent_result_count": self.recent_result_count,
            "last_score_delta": self.last_score_delta,
        }


def _context_lifecycle_state(
    *,
    key: TransformContextKey,
    rows: list[dict[str, Any]],
    suitability: float,
    evidence: list[str],
) -> TransformContextState:
    recent = rows[-5:]
    accepted = [row for row in recent if row.get("accepted")]
    rejected = [row for row in recent if not row.get("accepted")]
    last_delta = _row_score_delta(recent[-1]) if recent else None
    accepted_weight = sum(1.0 / (len(recent) - index) for index, row in enumerate(recent) if row.get("accepted"))
    rejected_weight = sum(1.0 / (len(recent) - index) for index, row in enumerate(recent) if not row.get("accepted"))
    if recent and last_delta is not None and last_delta < 0:
        state = "constrained"
        suitability = min(round(max(suitability * 0.35, 0.05), 4), suitability)
        reason = "Latest same-context candidate regressed score; require a materially distinct context before retrying."
    elif accepted and accepted_weight >= rejected_weight:
        state = "promotable_dev"
        suitability = round(max(suitability * 1.35, suitability + 0.15), 4)
        reason = "Recent same-context evidence earned finalist eligibility on dev."
    elif len(rejected) >= 2:
        state = "constrained"
        suitability = min(round(max(suitability * 0.35, 0.05), 4), suitability)
        reason = "Repeated same-context candidates failed the objective gate."
    elif rejected:
        if suitability >= 0.75 and evidence:
            state = "active"
            reason = "One same-context candidate failed, but current evidence remains strong."
        else:
            state = "paused"
            suitability = 0.0
            reason = "One same-context candidate failed; waiting for stronger evidence before retrying."
    elif suitability > 0:
        state = "active"
        reason = _suitability_reason(key.family, evidence, suitability)
    else:
        state = "available"
        reason = _suitability_reason(key.family, evidence, suitability)
    return TransformContextState(
        key=key,
        state=state,
        suitability=suitability,
        reason=reason,
        evidence=evidence,
        constraints=_constraints_for_lifecycle_state(state),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        recent_result_count=len(recent),
        last_score_delta=last_delta,
    )


def _context_summary_reason(state: str) -> str:
    if state == "promotable_dev":
        return "Recent same-context evidence earned finalist eligibility on dev."
    if state == "constrained":
        return "Same-context evidence regressed or repeatedly failed."
    if state == "paused":
        return "Same-context evidence failed once without enough evidence to retry immediately."
    if state == "active":
        return "Context remains active under current evidence."
    return "No evaluated evidence for this context."


def _constraints_for_lifecycle_state(state: str) -> list[str]:
    if state == "constrained":
        return [
            "Do not propose near-duplicates of failed instances from this mechanism.",
            "Only retry this mechanism with a materially different target, slice, parameterization, or expected mechanism.",
        ]
    if state == "paused":
        return ["Do not retry this mechanism unless later evidence makes it active again."]
    return []


def _row_score_delta(row: dict[str, Any]) -> float | None:
    comparison = row.get("comparison_to_parent") or {}
    if "score_delta" not in comparison:
        return None
    return float(comparison["score_delta"])


def _suitability_reason(family: str, evidence: list[str], suitability: float) -> str:
    if suitability <= 0:
        return f"{family} has no current evidence signal."
    return f"{family} is plausible because " + "; ".join(evidence) + "."


def _transform_patch_target(patch: TransformPatch) -> str:
    params = patch.op.params
    for key in ("section", "field", "target", "tool"):
        value = params.get(key)
        if isinstance(value, str) and value:
            return value
    return patch.hook or "global"


def _transform_patch_mechanism_signature(patch: TransformPatch) -> str:
    op = patch.op.op
    params = patch.op.params
    if op == "set_model_config":
        return f"{str(params.get('field', 'model_config'))}:{_value_class(params.get('value'))}"
    if op in {
        "add_context_section",
        "replace_context_section",
        "render_state_section",
        "rewrite_tool_description",
        "rewrite_response",
    }:
        return f"{op}:text:{_text_mechanism_class(str(params.get('content') or params.get('message') or params.get('append') or ''))}"
    return f"{op}:{_mapping_shape(params)}"


def _parameter_mechanism_signature(parameters: dict[str, Any]) -> tuple[str, ...]:
    if not parameters:
        return ()
    rows: list[str] = []
    for key in sorted(parameters):
        value = parameters[key]
        if key == "source_case_ids" and isinstance(value, list):
            rows.append(f"{key}:count={len(value)}")
            continue
        if key in {"target_labels", "affected_slices"} and isinstance(value, list):
            labels = ",".join(sorted(_normalize_token(str(item)) for item in value)[:6])
            rows.append(f"{key}:{labels}")
            continue
        rows.append(f"{_normalize_token(str(key))}:{_value_class(value)}")
    return tuple(rows)


def _value_class(value: Any) -> str:
    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return f"string:{_text_mechanism_class(value)}"
    if isinstance(value, list):
        return f"list:{len(value)}"
    if isinstance(value, dict):
        return _mapping_shape(value)
    if value is None:
        return "null"
    return type(value).__name__


def _mapping_shape(value: Any) -> str:
    if not isinstance(value, dict):
        return _value_class(value)
    keys = ",".join(sorted(str(key) for key in value.keys())[:8])
    return f"object:{keys or '-'}"


def _text_mechanism_class(text: str) -> str:
    normalized = _normalize_token(text)
    classes = []
    keyword_groups = {
        "format_contract": ("json", "schema", "format", "field", "valid", "parse", "contract"),
        "grounding": ("source", "evidence", "cite", "citation", "ground", "fact", "document"),
        "fallback": ("unknown", "cannot", "insufficient", "not available", "fallback"),
        "tool_use": ("tool", "search", "web", "lookup"),
        "classification": ("label", "category", "class", "priority", "intent"),
        "brevity": ("concise", "short", "brief", "limit"),
    }
    for label, keywords in keyword_groups.items():
        if any(keyword in normalized for keyword in keywords):
            classes.append(label)
    if not classes:
        classes.append("semantic_instruction")
    word_count = len(normalized.split())
    if word_count <= 12:
        length = "short"
    elif word_count <= 60:
        length = "medium"
    else:
        length = "long"
    return "+".join([*classes, length])


def _normalize_token(value: str, *, default: str = "") -> str:
    normalized = " ".join(str(value).strip().lower().split())
    return normalized or default
