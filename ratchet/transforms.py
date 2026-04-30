from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.experiments import CANDIDATE_ROLES
from ratchet.results import Comparison, CandidateSummary
from ratchet.surfaces import SurfaceSpec, SurfaceTarget, surface_targets
from ratchet.transform_program import TransformPatch, TransformProgram
from ratchet.types import FailureDiagnosis, OptimizationObjective


TRANSFORM_LIFECYCLE_STATES = {
    "available",
    "active",
    "promoted",
    "paused",
    "constrained",
}


@dataclass(frozen=True)
class TransformFamily:
    name: str
    category: str
    purpose: str
    supported_edit_kinds: list[str]
    supported_ops: list[str]
    activation_signals: list[str]
    expected_effects: dict[str, str]
    risks: list[str]
    required_measurements: list[str]
    complexity_cost: float
    parameter_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorProfile:
    mean_score: float
    pass_count: int
    case_count: int
    pass_rate: float
    failure_labels: dict[str, int]
    category_metrics: dict[str, dict[str, float | int]]
    invalid_output_rate: float
    mean_cost_usd: float
    mean_total_tokens: float
    median_latency_s: float
    high_cost_case_ids: list[str]
    high_latency_case_ids: list[str]
    target_slices: list[str]
    weak_slice_count: int
    runtime_error_rate: float
    length_finish_rate: float
    parser_fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformFamilyState:
    family: str
    state: str
    suitability: float
    budget_share: float
    reason: str
    evidence: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in TRANSFORM_LIFECYCLE_STATES:
            raise ValueError(f"Unsupported transform lifecycle state: {self.state}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    def from_candidate(cls, candidate: "CandidateProposal") -> "TransformContextKey":
        patches = tuple(candidate.program.patches)
        return cls(
            family=candidate.transform_family,
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
    def from_row(cls, row: dict[str, Any]) -> "TransformContextKey":
        existing = row.get("transform_context")
        if isinstance(existing, dict):
            return cls(
                family=str(existing.get("family") or row.get("transform_family") or "unknown"),
                target_names=tuple(str(item) for item in existing.get("target_names", [])),
                ops=tuple(str(item) for item in existing.get("ops", [])),
                target_slice=str(existing.get("target_slice") or row.get("target_slice") or "global"),
                mechanism=tuple(str(item) for item in existing.get("mechanism", [])),
                transform_instance=str(existing.get("transform_instance") or row.get("transform_instance") or "candidate"),
            )
        candidate_payload = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        program_payload = row.get("proposal") or candidate_payload.get("program") or {}
        raw_patches = program_payload.get("patches", []) if isinstance(program_payload, dict) else []
        patches = [TransformPatch.from_dict(item) for item in raw_patches if isinstance(item, dict)]
        return cls(
            family=str(row.get("transform_family") or "unknown"),
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


@dataclass(frozen=True)
class SearchHypothesis:
    family_states: dict[str, TransformFamilyState]
    context_states: dict[str, TransformContextState]
    target_slices: list[str]
    profile: BehaviorProfile
    budget_allocation: dict[str, float]
    rationale: str

    @property
    def active_families(self) -> list[str]:
        return [
            name
            for name, state in sorted(
                self.family_states.items(),
                key=lambda item: (-item[1].suitability, item[0]),
            )
            if state.state in {"active", "promoted"}
            or (state.state == "constrained" and state.suitability > 0)
        ]

    @property
    def active_contexts(self) -> list[str]:
        return [
            context_id
            for context_id, state in sorted(
                self.context_states.items(),
                key=lambda item: (-item[1].suitability, item[0]),
            )
            if state.state in {"active", "promoted"}
            or (state.state == "constrained" and state.suitability > 0)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_states": {
                name: state.to_dict() for name, state in sorted(self.family_states.items())
            },
            "context_states": {
                context_id: state.to_dict() for context_id, state in sorted(self.context_states.items())
            },
            "active_families": self.active_families,
            "active_contexts": self.active_contexts,
            "target_slices": list(self.target_slices),
            "profile": self.profile.to_dict(),
            "budget_allocation": dict(sorted(self.budget_allocation.items())),
            "rationale": self.rationale,
        }

    def to_prompt_dict(self, *, max_contexts_per_family: int = 3, max_constrained_contexts: int = 8) -> dict[str, Any]:
        ranked_contexts = sorted(
            self.context_states.values(),
            key=lambda state: (-state.suitability, state.key.family, state.key.scope_id, state.key.id),
        )
        active_contexts: list[dict[str, Any]] = []
        counts_by_family: Counter[str] = Counter()
        for state in ranked_contexts:
            if state.state not in {"active", "promoted"}:
                continue
            if counts_by_family[state.key.family] >= max_contexts_per_family:
                continue
            counts_by_family[state.key.family] += 1
            active_contexts.append(_context_prompt_row(state))
        constrained_contexts = [
            _context_prompt_row(state)
            for state in ranked_contexts
            if state.state in {"constrained", "paused"} and state.recent_result_count > 0
        ][:max_constrained_contexts]
        return {
            "family_states": {
                name: {
                    "state": state.state,
                    "suitability": state.suitability,
                    "budget_share": state.budget_share,
                    "reason": state.reason,
                    "constraints": list(state.constraints),
                }
                for name, state in sorted(self.family_states.items())
            },
            "active_families": self.active_families,
            "active_contexts": active_contexts,
            "constrained_or_paused_contexts": constrained_contexts,
            "target_slices": list(self.target_slices[:8]),
            "profile": {
                "mean_score": self.profile.mean_score,
                "pass_rate": self.profile.pass_rate,
                "failure_labels": dict(self.profile.failure_labels),
                "invalid_output_rate": self.profile.invalid_output_rate,
                "mean_cost_usd": self.profile.mean_cost_usd,
                "median_latency_s": self.profile.median_latency_s,
                "weak_slice_count": self.profile.weak_slice_count,
                "runtime_error_rate": self.profile.runtime_error_rate,
                "length_finish_rate": self.profile.length_finish_rate,
                "parser_fallback_rate": self.profile.parser_fallback_rate,
            },
            "budget_allocation": dict(sorted(self.budget_allocation.items())),
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class Intervention:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Intervention":
        return cls(
            kind=str(payload.get("kind", "")),
            payload=dict(payload.get("payload", {})),
        )


@dataclass(frozen=True)
class CandidateAffordanceApplication:
    affordance_id: str
    selection: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    @property
    def family(self) -> str:
        if self.affordance_id.startswith("surface."):
            return self.mechanism
        parts = self.affordance_id.split(".")
        return parts[0] if parts else ""

    @property
    def mechanism(self) -> str:
        if self.affordance_id.startswith("surface."):
            parts = self.affordance_id.split(".")
            return parts[1] if len(parts) > 1 else "surface"
        parts = self.affordance_id.split(".")
        return parts[1] if len(parts) > 1 else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "affordance_id": self.affordance_id,
            "surface_opportunity_id": self.affordance_id,
            "surface": self.mechanism,
            "selection": dict(self.selection),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateAffordanceApplication":
        affordance_id = str(payload.get("surface_opportunity_id") or payload.get("affordance_id") or "")
        if not affordance_id:
            raise ValueError("application requires surface_opportunity_id")
        raw_operation = payload.get("operation")
        raw_selection = payload.get("selection")
        if isinstance(raw_operation, dict):
            raise ValueError("applications cite surface affordances; transform edits belong in program.patches")
        selection = dict(raw_selection) if isinstance(raw_selection, dict) else {}
        return cls(
            affordance_id=affordance_id,
            selection=selection,
            rationale=str(payload.get("rationale") or ""),
        )


@dataclass(frozen=True)
class CandidateProposal:
    program: TransformProgram
    applications: list[CandidateAffordanceApplication]
    experiment_id: str = ""
    candidate_role: str = "atomic"
    comparison_group: str = ""
    target_slice: str = "global"
    hypothesis: str = ""
    expected_effects: dict[str, Any] = field(default_factory=dict)
    evaluation_plan: str = "full_dev"

    @property
    def affordance_ids(self) -> list[str]:
        return [application.affordance_id for application in self.applications]

    @property
    def transform_family(self) -> str:
        return self.applications[0].family if self.applications else ""

    @property
    def mechanism_class(self) -> str:
        return self.applications[0].mechanism if self.applications else ""

    @property
    def transform_instance(self) -> str:
        return "; ".join(application.rationale for application in self.applications if application.rationale)[:240]

    @property
    def transform_parameters(self) -> dict[str, Any]:
        source_ids: list[str] = []
        strategies: list[str] = []
        for application in self.applications:
            raw_ids = application.selection.get("source_case_ids")
            if isinstance(raw_ids, list):
                source_ids.extend(str(item) for item in raw_ids if isinstance(item, str) and item)
            if application.selection.get("selection_strategy"):
                strategies.append(str(application.selection["selection_strategy"]))
        row: dict[str, Any] = {}
        if source_ids:
            row["source_case_ids"] = source_ids
        if strategies:
            row["selection_strategies"] = sorted(set(strategies))
        if "few_shot_example_count" in self.program.metadata:
            row["few_shot_example_count"] = self.program.metadata["few_shot_example_count"]
        return row

    @property
    def intervention(self) -> Intervention:
        if self.program.patches:
            return Intervention(kind="transform_program", payload={"program": self.program.to_dict()})
        if self.applications and self.applications[0].selection:
            return Intervention(kind="example_selection", payload=dict(self.applications[0].selection))
        return Intervention(kind="transform_program", payload={"program": self.program.to_dict()})

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_family": self.transform_family,
            "intervention": self.intervention.to_dict(),
            "transform_instance": self.transform_instance,
            "transform_parameters": dict(self.transform_parameters),
            "mechanism_class": self.mechanism_class,
            "experiment_id": self.experiment_id,
            "candidate_role": self.candidate_role,
            "comparison_group": self.comparison_group,
            "affordance_ids": list(self.affordance_ids),
            "surface_opportunity_ids": list(self.affordance_ids),
            "target_slice": self.target_slice,
            "transform_context": TransformContextKey.from_candidate(self).to_dict(),
            "hypothesis": self.hypothesis,
            "expected_effects": dict(self.expected_effects),
            "evaluation_plan": self.evaluation_plan,
            "applications": [application.to_dict() for application in self.applications],
            "program": self.program.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateProposal":
        if "patch" in payload:
            raise ValueError("candidate must emit a typed transform program, not a legacy patch")
        if payload.get("transform_parameters"):
            raise ValueError("candidate transform_parameters are derived; put candidate-specific data in applications[]")
        if payload.get("transform_family") or payload.get("mechanism_class") or payload.get("affordance_ids"):
            raise ValueError("candidate must cite applications[]; family, mechanism, and affordance_ids are derived")
        raw_program = payload.get("program") or payload.get("transform_program")
        if not isinstance(raw_program, dict):
            raw_patches = payload.get("patches")
            if isinstance(raw_patches, list):
                raw_program = {
                    "id": str(payload.get("candidate_id") or payload.get("experiment_id") or ""),
                    "hypothesis_id": str(payload.get("hypothesis_id") or ""),
                    "patches": raw_patches,
                    "metadata": dict(payload.get("metadata") or {}),
                }
        if not isinstance(raw_program, dict):
            raise ValueError("candidate requires program or patches[]")
        program_payload = dict(raw_program)
        if not program_payload.get("id") and not program_payload.get("candidate_id"):
            program_payload["id"] = str(payload.get("experiment_id") or payload.get("candidate_id") or "")
        metadata = {
            **dict(program_payload.get("metadata") or {}),
            "hypothesis": str(payload.get("hypothesis", "")),
            "expected_effects": dict(payload.get("expected_effects", {})),
        }
        program_payload["metadata"] = metadata
        program = TransformProgram.from_dict(program_payload)
        raw_applications = payload.get("applications")
        if not isinstance(raw_applications, list) or not raw_applications:
            raise ValueError("candidate requires non-empty applications[]")
        applications = [
            CandidateAffordanceApplication.from_dict(application)
            for application in raw_applications
            if isinstance(application, dict)
        ]
        if len(applications) != len(raw_applications):
            raise ValueError("candidate applications must be objects")
        return cls(
            program=program,
            applications=applications,
            experiment_id=str(payload.get("experiment_id", "")),
            candidate_role=str(payload.get("candidate_role", "atomic") or "atomic"),
            comparison_group=str(payload.get("comparison_group", "")),
            target_slice=str(payload.get("target_slice", "global") or "global"),
            hypothesis=str(payload.get("hypothesis", "")),
            expected_effects=dict(payload.get("expected_effects", {})),
            evaluation_plan=str(payload.get("evaluation_plan", "full_dev") or "full_dev"),
        )


@dataclass(frozen=True)
class TransformResultSummary:
    family: str
    proposed_count: int = 0
    evaluated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    best_score_delta: float | None = None
    best_cost_delta: float | None = None
    best_latency_delta: float | None = None
    state: str = "available"
    reason: str = "No candidates evaluated for this surface mechanism."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRANSFORM_FAMILIES: dict[str, TransformFamily] = {
    "surface_context": TransformFamily(
        name="surface_context",
        category="context",
        purpose="Modify model-visible context graph sections through typed DSL patches.",
        supported_edit_kinds=["instruction"],
        supported_ops=["add_context_section", "replace_context_section", "remove_context_section", "move_context_section", "reorder_context_sections"],
        activation_signals=["branch_failures", "correctness_gap", "weak_slices"],
        expected_effects={"correctness": "variable", "cost": "token-dependent", "latency": "token-dependent"},
        risks=["overconstrains behavior", "moves failures between slices"],
        required_measurements=["score_delta", "regression_cases", "cost_delta", "latency_delta"],
        complexity_cost=1.0,
    ),
    "surface_tool_loop": TransformFamily(
        name="surface_tool_loop",
        category="tool_loop",
        purpose="Modify tool presentation and tool-loop middleware without changing true tool semantics.",
        supported_edit_kinds=["tool"],
        supported_ops=["annotate_tool", "rewrite_tool_description", "validate", "normalize_tool_args", "repair_tool_args", "replan"],
        activation_signals=["tool_dependent_slice", "correctness_gap"],
        expected_effects={"correctness": "variable", "cost": "variable", "latency": "variable"},
        risks=["tool overuse", "tool underuse", "tool-call regression"],
        required_measurements=["score_delta", "tool_call_delta", "tool_error_delta", "turn_delta"],
        complexity_cost=1.25,
    ),
    "surface_state": TransformFamily(
        name="surface_state",
        category="state",
        purpose="Add and maintain typed runtime state for context, validation, and response guards.",
        supported_edit_kinds=["state"],
        supported_ops=["define_state", "store_state", "append_state", "merge_state", "clear_state", "expose_state"],
        activation_signals=["branch_failures", "tool_dependent_slice"],
        expected_effects={"correctness": "variable", "cost": "low", "latency": "low"},
        risks=["state drift", "extra prompt tokens when exposed"],
        required_measurements=["score_delta", "runtime_error_delta", "cost_delta"],
        complexity_cost=1.25,
    ),
    "surface_response": TransformFamily(
        name="surface_response",
        category="response",
        purpose="Validate, block, or rewrite draft responses at declared response hooks.",
        supported_edit_kinds=["response"],
        supported_ops=["extract_claims", "validate", "validate_claims", "rewrite_response", "block_response"],
        activation_signals=["invalid_output", "runtime_truncation"],
        expected_effects={"correctness": "variable", "cost": "low", "latency": "low"},
        risks=["over-suppresses valid responses"],
        required_measurements=["score_delta", "invalid_output_delta", "cost_delta"],
        complexity_cost=1.0,
    ),
    "surface_output": TransformFamily(
        name="surface_output",
        category="output",
        purpose="Modify explicit output-contract sections and output schemas.",
        supported_edit_kinds=["output"],
        supported_ops=["add_context_section", "replace_context_section", "validate"],
        activation_signals=["invalid_output", "runtime_truncation"],
        expected_effects={"correctness": "variable", "cost": "token-dependent", "latency": "token-dependent"},
        risks=["format improves while semantics regress"],
        required_measurements=["score_delta", "invalid_output_delta", "regression_cases"],
        complexity_cost=1.0,
    ),
    "surface_runtime": TransformFamily(
        name="surface_runtime",
        category="runtime",
        purpose="Tune declared retry, turn, and runtime-control surfaces.",
        supported_edit_kinds=["runtime"],
        supported_ops=["set_retry_policy", "set_turn_limit", "set_tool_call_limit", "validate_completion"],
        activation_signals=["runtime_errors", "runtime_truncation", "cost_objective", "latency_objective"],
        expected_effects={"correctness": "variable", "cost": "variable", "latency": "variable"},
        risks=["hidden correctness tradeoff"],
        required_measurements=["score_delta", "turn_delta", "latency_delta", "cost_delta"],
        complexity_cost=1.0,
    ),
    "surface_model": TransformFamily(
        name="surface_model",
        category="model",
        purpose="Change declared model invocation settings.",
        supported_edit_kinds=["model"],
        supported_ops=["set_model_config"],
        activation_signals=["correctness_gap", "cost_objective", "latency_objective"],
        expected_effects={"correctness": "variable", "cost": "variable", "latency": "variable"},
        risks=["global behavior change", "cost or latency regression"],
        required_measurements=["score_delta", "cost_delta", "latency_delta"],
        complexity_cost=1.0,
    ),
    "surface_examples": TransformFamily(
        name="surface_examples",
        category="examples",
        purpose="Expose proposal-safe examples as typed context sections.",
        supported_edit_kinds=["few_shot"],
        supported_ops=["add_context_section"],
        activation_signals=["weak_slices"],
        expected_effects={"correctness": "variable", "cost": "increase", "latency": "token-dependent"},
        risks=["example overfitting", "prompt growth"],
        required_measurements=["score_delta", "example_token_delta", "regression_cases"],
        complexity_cost=1.5,
    ),
}

def transform_registry() -> dict[str, TransformFamily]:
    return dict(TRANSFORM_FAMILIES)


def build_behavior_profile(summary: CandidateSummary) -> BehaviorProfile:
    case_rows = summary._case_rows()
    costs = []
    latencies = []
    cost_by_case: dict[str, float] = {}
    latency_by_case: dict[str, float] = {}
    for case_id, evaluations, _, _, _ in case_rows:
        case_cost = sum(evaluation.record.metrics.cost_usd for evaluation in evaluations) / max(len(evaluations), 1)
        case_latency = sorted(evaluation.record.metrics.latency_s for evaluation in evaluations)[len(evaluations) // 2]
        costs.append(case_cost)
        latencies.append(case_latency)
        cost_by_case[case_id] = case_cost
        latency_by_case[case_id] = case_latency
    high_cost_threshold = _high_metric_threshold(costs)
    high_latency_threshold = _high_metric_threshold(latencies)
    invalid_count = 0
    runtime_error_count = 0
    length_finish_count = 0
    parser_fallback_count = 0
    for _, evaluations, _, _, case_passed in case_rows:
        if any(evaluation.record.metrics.error for evaluation in evaluations):
            runtime_error_count += 1
        representative = next((item for item in evaluations if not item.grade.passed), evaluations[0])
        metadata = representative.record.diagnostics.metadata
        if str(metadata.get("finish_reason") or "") == "length":
            length_finish_count += 1
        if metadata.get("parser_fallback"):
            parser_fallback_count += 1
        if case_passed:
            continue
        labels = [label for evaluation in evaluations if not evaluation.grade.passed for label in evaluation.grade.labels]
        if any("invalid_output" in label or "output" in label for label in labels):
            invalid_count += 1
    target_slices = _target_slices(summary)
    return BehaviorProfile(
        mean_score=summary.mean_score,
        pass_count=summary.pass_count,
        case_count=summary.case_count,
        pass_rate=summary.pass_rate,
        failure_labels=summary.failure_labels,
        category_metrics=summary.category_metrics,
        invalid_output_rate=(invalid_count / summary.case_count if summary.case_count else 0.0),
        mean_cost_usd=summary.mean_cost_usd,
        mean_total_tokens=summary.mean_total_tokens,
        median_latency_s=summary.median_latency_s,
        high_cost_case_ids=[
            case_id for case_id, value in sorted(cost_by_case.items()) if value >= high_cost_threshold and value > 0
        ],
        high_latency_case_ids=[
            case_id for case_id, value in sorted(latency_by_case.items()) if value >= high_latency_threshold and value > 0
        ],
        target_slices=target_slices,
        weak_slice_count=sum(
            1
            for metrics in summary.category_metrics.values()
            if int(metrics.get("count", 0)) > int(metrics.get("pass_count", 0))
        ),
        runtime_error_rate=(runtime_error_count / summary.case_count if summary.case_count else 0.0),
        length_finish_rate=(length_finish_count / summary.case_count if summary.case_count else 0.0),
        parser_fallback_rate=(parser_fallback_count / summary.case_count if summary.case_count else 0.0),
    )


def build_search_hypothesis(
    *,
    summary: CandidateSummary,
    surface: SurfaceSpec,
    objective: OptimizationObjective,
    history: list[dict[str, Any]],
    parent_candidate_id: str | None = None,
    diagnoses: list[FailureDiagnosis] | None = None,
    proposal_example_count: int = 0,
) -> SearchHypothesis:
    if not isinstance(surface, SurfaceSpec):
        raise TypeError(f"build_search_hypothesis requires SurfaceSpec, got {type(surface).__name__}.")
    profile = build_behavior_profile(summary)
    targets = surface_targets(surface)
    branch_history = select_branch_history(history, parent_candidate_id or summary.candidate_id)
    diagnosis_signals = _diagnosis_signals(diagnoses or [])
    context_states: dict[str, TransformContextState] = {}
    for target in targets:
        ops = tuple(sorted(str(op) for op in target.allowed_ops if op))
        if not ops:
            continue
        family = _surface_mechanism_for_target(target)
        context_key = TransformContextKey(
            family=family,
            target_names=(target.name,),
            ops=ops,
            target_slice="global",
            mechanism=(target.semantics.role or target.kind,),
        )
        suitability, evidence = _surface_context_suitability(
            target=target,
            family=family,
            profile=profile,
            objective=objective,
            diagnosis_signals=diagnosis_signals,
            proposal_example_count=proposal_example_count,
        )
        context_states[context_key.id] = _context_lifecycle_state(
            key=context_key,
            rows=_rows_for_context(branch_history, context_key),
            suitability=suitability,
            evidence=evidence,
        )
    for row in branch_history:
        context_key = TransformContextKey.from_row(row)
        if context_key.id in context_states:
            continue
        suitability, evidence = _row_context_suitability(context_key=context_key, profile=profile, objective=objective)
        context_states[context_key.id] = _context_lifecycle_state(
            key=context_key,
            rows=_rows_for_context(branch_history, context_key),
            suitability=suitability,
            evidence=evidence,
        )
    family_states = _aggregate_family_states(context_states)
    allocation = _budget_allocation(family_states)
    family_states = {
        name: TransformFamilyState(
            family=state.family,
            state=state.state,
            suitability=state.suitability,
            budget_share=allocation.get(name, 0.0),
            reason=state.reason,
            evidence=list(state.evidence),
            constraints=list(state.constraints),
        )
        for name, state in family_states.items()
    }
    return SearchHypothesis(
        family_states=family_states,
        context_states=context_states,
        target_slices=profile.target_slices,
        profile=profile,
        budget_allocation=allocation,
        rationale="Search hypothesis derived from the inferred optimization surface, current behavior profile, diagnoses, objective, and branch-local surface-program history.",
    )


def validate_candidate_transform(
    candidate: CandidateProposal,
    *,
    surface: SurfaceSpec,
    search_hypothesis: SearchHypothesis | None = None,
) -> str | None:
    if not candidate.experiment_id:
        return "candidate must belong to an experiment"
    if candidate.candidate_role not in CANDIDATE_ROLES:
        return f"unknown candidate role {candidate.candidate_role!r}"
    if not candidate.applications:
        return "candidate must include at least one surface opportunity application"
    for application in candidate.applications:
        if application.affordance_id.startswith("surface."):
            continue
        return f"candidate application must cite a surface_opportunity_id, got {application.affordance_id!r}"
    if search_hypothesis is not None:
        if any(not application.affordance_id.startswith("surface.") for application in candidate.applications):
            eligibility_error = validate_candidate_context(candidate, search_hypothesis=search_hypothesis, surface=surface)
            if eligibility_error is not None:
                return eligibility_error
    return None


def validate_candidate_context(
    candidate: CandidateProposal,
    *,
    search_hypothesis: SearchHypothesis,
    surface: SurfaceSpec | None = None,
) -> str | None:
    for family_name in sorted({application.family for application in candidate.applications}):
        family_state = search_hypothesis.family_states.get(family_name)
        if family_state is None or family_name not in search_hypothesis.active_families:
            return f"inactive surface mechanism {family_name!r}"
    combined_key = TransformContextKey.from_candidate(candidate)
    exact_state = search_hypothesis.context_states.get(combined_key.id)
    if exact_state is not None and exact_state.state in {"paused", "available"}:
        return f"inactive transform context {combined_key.id!r}"
    if exact_state is not None and exact_state.state == "constrained":
        return f"constrained transform context {combined_key.id!r} requires a materially distinct mechanism"
    for application in candidate.applications:
        if application.selection:
            operation_key = TransformContextKey(
                family=application.family,
                target_names=("proposal_selected_examples",),
                ops=("add_context_section",),
                target_slice=candidate.target_slice,
                mechanism=(application.mechanism,),
                transform_instance=application.rationale or candidate.hypothesis or "candidate",
            )
            operation_error = _operation_context_error(operation_key, search_hypothesis)
            if operation_error is not None:
                return operation_error
            continue
    return None


def select_branch_history(history: list[dict[str, Any]], parent_candidate_id: str | None) -> list[dict[str, Any]]:
    if not parent_candidate_id:
        return list(history)
    producing_parent_by_child: dict[str, str] = {}
    for row in history:
        child = row.get("candidate_id")
        parent = row.get("parent_candidate_id")
        if row.get("accepted") and isinstance(child, str) and isinstance(parent, str):
            producing_parent_by_child[child] = parent
    lineage = {parent_candidate_id}
    cursor = parent_candidate_id
    while cursor in producing_parent_by_child:
        cursor = producing_parent_by_child[cursor]
        if cursor in lineage:
            break
        lineage.add(cursor)
    return [
        row
        for row in history
        if row.get("candidate_id") in lineage or row.get("parent_candidate_id") == parent_candidate_id
    ]


def summarize_transform_results(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    context_summaries = summarize_transform_context_results(proposals)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposed_counts: Counter[str] = Counter()
    for row in proposals:
        family = str(row.get("transform_family") or "unknown")
        if row.get("type") == "candidate_proposal":
            proposed_counts[family] += 1
        else:
            grouped[family].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for family in sorted(set(grouped) | set(proposed_counts)):
        rows = grouped.get(family, [])
        evaluated_count = len(rows)
        proposed_count = max(proposed_counts.get(family, 0), evaluated_count)
        accepted_rows = [row for row in rows if row.get("accepted")]
        comparisons = [row.get("comparison_to_parent") or {} for row in rows]
        score_deltas = [float(item["score_delta"]) for item in comparisons if "score_delta" in item]
        cost_deltas = [float(item["cost_delta"]) for item in comparisons if "cost_delta" in item]
        latency_deltas = [float(item["latency_delta"]) for item in comparisons if "latency_delta" in item]
        score_regressed = any(delta < 0 for delta in score_deltas)
        if accepted_rows:
            state = "promoted"
            reason = "At least one candidate from this surface mechanism improved the configured objective on dev."
        elif evaluated_count >= 2 or score_regressed:
            state = "constrained"
            reason = (
                "At least one candidate from this surface mechanism regressed score; future attempts should use a distinct target, slice, or instance."
                if score_regressed
                else "Multiple evaluated candidates failed the configured objective gate; future attempts should avoid near-duplicate instances."
            )
        elif evaluated_count == 1:
            state = "paused"
            reason = "The evaluated candidate failed the configured objective gate."
        else:
            state = "available"
            reason = "No candidates evaluated for this surface mechanism."
        summaries[family] = TransformResultSummary(
            family=family,
            proposed_count=proposed_count,
            evaluated_count=evaluated_count,
            accepted_count=len(accepted_rows),
            rejected_count=max(evaluated_count - len(accepted_rows), 0),
            best_score_delta=max(score_deltas) if score_deltas else None,
            best_cost_delta=min(cost_deltas) if cost_deltas else None,
            best_latency_delta=min(latency_deltas) if latency_deltas else None,
            state=state,
            reason=reason,
        ).to_dict()
        family_contexts = [
            summary
            for summary in context_summaries.values()
            if ((summary.get("key") or {}).get("family") == family)
        ]
        if family_contexts:
            if any(summary.get("state") == "promoted" for summary in family_contexts):
                summaries[family]["state"] = "promoted"
            elif any(summary.get("state") == "active" for summary in family_contexts):
                summaries[family]["state"] = "active"
            elif any(summary.get("state") == "constrained" for summary in family_contexts):
                summaries[family]["state"] = "constrained"
            elif any(summary.get("state") == "paused" for summary in family_contexts):
                summaries[family]["state"] = "paused"
    return summaries


def summarize_transform_context_results(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        if "accepted" not in row:
            continue
        key = TransformContextKey.from_row(row)
        grouped[key.id].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for context_id, rows in sorted(grouped.items()):
        key = TransformContextKey.from_row(rows[-1])
        comparisons = [row.get("comparison_to_parent") or {} for row in rows]
        score_deltas = [float(item["score_delta"]) for item in comparisons if "score_delta" in item]
        cost_deltas = [float(item["cost_delta"]) for item in comparisons if "cost_delta" in item]
        latency_deltas = [float(item["latency_delta"]) for item in comparisons if "latency_delta" in item]
        accepted_rows = [row for row in rows if row.get("accepted")]
        state = _context_lifecycle_state(
            key=key,
            rows=rows,
            suitability=0.0,
            evidence=[],
        ).state
        reason = _context_summary_reason(state)
        summaries[context_id] = {
            "key": key.to_dict(),
            "state": state,
            "proposed_count": len(rows),
            "evaluated_count": len(rows),
            "accepted_count": len(accepted_rows),
            "rejected_count": max(len(rows) - len(accepted_rows), 0),
            "best_score_delta": max(score_deltas) if score_deltas else None,
            "best_cost_delta": min(cost_deltas) if cost_deltas else None,
            "best_latency_delta": min(latency_deltas) if latency_deltas else None,
            "reason": reason,
        }
    return summaries


def summarize_affordance_results(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        applications = row.get("applications") or candidate.get("applications") if isinstance(candidate, dict) else []
        if not isinstance(applications, list):
            continue
        for application in applications:
            if not isinstance(application, dict):
                continue
            affordance_id = str(application.get("affordance_id") or "")
            if affordance_id:
                grouped[affordance_id].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for affordance_id, rows in sorted(grouped.items()):
        evaluated_rows = [row for row in rows if "accepted" in row]
        valid_rows = [row for row in rows if row.get("valid") is not False]
        accepted_rows = [row for row in evaluated_rows if row.get("accepted")]
        comparisons = [row.get("comparison_to_parent") or {} for row in evaluated_rows]
        score_deltas = [float(item["score_delta"]) for item in comparisons if "score_delta" in item]
        cost_deltas = [float(item["cost_delta"]) for item in comparisons if "cost_delta" in item]
        latency_deltas = [float(item["latency_delta"]) for item in comparisons if "latency_delta" in item]
        invalid_reasons = Counter(
            str(row.get("invalid_reason"))
            for row in rows
            if row.get("valid") is False and row.get("invalid_reason")
        )
        if accepted_rows:
            state = "promoted"
            reason = "At least one application of this surface opportunity improved the configured objective on dev."
        elif score_deltas and any(delta < 0 for delta in score_deltas):
            state = "constrained"
            reason = "At least one evaluated application regressed score."
        elif len(evaluated_rows) >= 2:
            state = "constrained"
            reason = "Multiple evaluated applications failed the configured objective gate."
        elif evaluated_rows:
            state = "paused"
            reason = "The evaluated application did not improve the configured objective."
        elif invalid_reasons:
            state = "invalid"
            reason = "No application reached evaluation because implementation validation failed."
        else:
            state = "proposed"
            reason = "Affordance was proposed but not evaluated."
        summaries[affordance_id] = {
            "affordance_id": affordance_id,
            "family": _affordance_id_part(affordance_id, 0),
            "mechanism": _affordance_id_part(affordance_id, 1),
            "state": state,
            "proposed_count": len(rows),
            "valid_count": len(valid_rows),
            "evaluated_count": len(evaluated_rows),
            "accepted_count": len(accepted_rows),
            "rejected_count": max(len(evaluated_rows) - len(accepted_rows), 0),
            "invalid_count": max(len(rows) - len(valid_rows), 0),
            "best_score_delta": max(score_deltas) if score_deltas else None,
            "best_cost_delta": min(cost_deltas) if cost_deltas else None,
            "best_latency_delta": min(latency_deltas) if latency_deltas else None,
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
            "candidate_ids": [
                str(row.get("candidate_id"))
                for row in evaluated_rows
                if row.get("candidate_id")
            ][:8],
            "reason": reason,
        }
    return summaries


def _affordance_id_part(affordance_id: str, index: int) -> str:
    parts = affordance_id.split(".")
    return parts[index] if len(parts) > index else ""


def observe_transform_result(
    *,
    family: str,
    context_key: TransformContextKey | None = None,
    accepted: bool,
    comparison: Comparison,
    rejection_reason: str | None,
) -> dict[str, Any]:
    if accepted:
        state = "promoted"
        reason = "Candidate improved the configured objective on dev."
    elif comparison.score_delta < 0:
        state = "constrained"
        reason = rejection_reason or "Candidate regressed score; future attempts should be materially distinct."
    else:
        state = "paused"
        reason = rejection_reason or "Candidate did not improve the configured objective."
    return {
        "type": "transform_observation",
        "transform_family": family,
        "transform_context": context_key.to_dict() if context_key else None,
        "state": state,
        "reason": reason,
        "comparison_to_parent": comparison.to_dict(),
    }


def _diagnosis_signals(diagnoses: list[FailureDiagnosis]) -> dict[str, set[str]]:
    target_names: set[str] = set()
    categories: set[str] = set()
    case_ids: set[str] = set()
    for diagnosis in diagnoses:
        target_names.update(diagnosis.target_names)
        if diagnosis.category:
            categories.add(_normalize_token(diagnosis.category))
        case_ids.update(diagnosis.case_ids)
    return {
        "target_names": target_names,
        "categories": categories,
        "case_ids": case_ids,
        "target_slices": {f"diagnosis:{category}" for category in categories},
    }


def _surface_mechanism_for_target(target: SurfaceTarget) -> str:
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
    return f"surface_{_normalize_token(target.kind)}"


def _surface_context_suitability(
    *,
    target: SurfaceTarget,
    family: str,
    profile: BehaviorProfile,
    objective: OptimizationObjective,
    diagnosis_signals: dict[str, set[str]],
    proposal_example_count: int,
) -> tuple[float, list[str]]:
    suitability = 0.2 + min(max(target.semantics.confidence, 0.0), 1.0) * 0.15
    evidence: list[str] = [f"inferred editable {target.kind} surface"]
    if profile.pass_count < profile.case_count:
        suitability += 0.1
        evidence.append("branch has residual correctness failures")
    if profile.invalid_output_rate > 0 and family in {"surface_output", "surface_response", "surface_runtime"}:
        suitability += 0.3
        evidence.append("invalid or incomplete outputs observed")
    if profile.runtime_error_rate > 0 and family == "surface_runtime":
        suitability += 0.25
        evidence.append("runtime errors observed")
    if profile.length_finish_rate > 0 and family in {"surface_runtime", "surface_output"}:
        suitability += 0.2
        evidence.append("finish_reason=length observed")
    if profile.parser_fallback_rate > 0 and family in {"surface_output", "surface_response"}:
        suitability += 0.2
        evidence.append("parser fallback observed")
    if profile.weak_slice_count > 0 and family in {"surface_context", "surface_examples"}:
        suitability += 0.15
        evidence.append("weak slices are available")
    if objective.mode in {"cost", "latency"} and family in {"surface_model", "surface_runtime", "surface_context"}:
        suitability += 0.25
        evidence.append(f"{objective.mode} objective is active")
    if family == "surface_examples" and proposal_example_count <= 0:
        return 0.0, ["No proposal-safe train examples are available for example-surface patches."]
    if target.name in diagnosis_signals["target_names"]:
        suitability += 0.2
        evidence.append("diagnosis points at this editable target")
    if diagnosis_signals["categories"] and family in {"surface_context", "surface_tool_loop", "surface_response"}:
        suitability += 0.05
        evidence.append("diagnosis categories provide targetable failure context")
    return round(min(max(suitability, 0.0), 1.0), 4), _unique(evidence)


def _row_context_suitability(
    *,
    context_key: TransformContextKey,
    profile: BehaviorProfile,
    objective: OptimizationObjective,
) -> tuple[float, list[str]]:
    suitability = 0.35
    evidence = ["surface-program context seen in branch history"]
    if profile.pass_count < profile.case_count:
        suitability += 0.1
    if context_key.family == "surface_model" and objective.mode in {"cost", "latency"}:
        suitability += 0.2
    return round(min(suitability, 1.0), 4), evidence


def _rows_for_context(rows: list[dict[str, Any]], context_key: TransformContextKey) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if "accepted" in row and TransformContextKey.from_row(row).id == context_key.id
    ]


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
        adjusted = min(round(max(suitability * 0.35, 0.05), 4), suitability)
        reason = "Latest same-context candidate regressed score; require a materially distinct context before retrying."
        suitability = adjusted
    elif accepted and accepted_weight >= rejected_weight:
        state = "promoted"
        suitability = round(max(suitability * 1.35, suitability + 0.15), 4)
        reason = "Recent same-context evidence improved the dev objective."
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


def _aggregate_family_states(context_states: dict[str, TransformContextState]) -> dict[str, TransformFamilyState]:
    grouped: dict[str, list[TransformContextState]] = defaultdict(list)
    for context_state in context_states.values():
        grouped[context_state.key.family].append(context_state)
    family_states: dict[str, TransformFamilyState] = {}
    for family_name in sorted(grouped):
        states = grouped.get(family_name, [])
        if any(state.state == "promoted" for state in states):
            state_name = "promoted"
        elif any(state.state == "active" for state in states):
            state_name = "active"
        elif any(state.state == "constrained" and state.suitability > 0 for state in states):
            state_name = "constrained"
        elif any(state.state == "paused" for state in states):
            state_name = "paused"
        else:
            state_name = "available"
        suitability = max((state.suitability for state in states), default=0.0)
        evidence = _unique(item for state in states for item in state.evidence)
        constraints = _unique(item for state in states for item in state.constraints)
        family_states[family_name] = TransformFamilyState(
            family=family_name,
            state=state_name,
            suitability=suitability,
            budget_share=0.0,
            reason=_family_state_reason(family_name, state_name, states),
            evidence=evidence,
            constraints=constraints,
        )
    return family_states


def _budget_allocation(states: dict[str, TransformFamilyState]) -> dict[str, float]:
    active = {
        name: state.suitability
        for name, state in states.items()
        if state.state in {"active", "promoted", "constrained"} and state.suitability > 0
    }
    total = sum(active.values())
    if total <= 0:
        return {}
    return {name: round(value / total, 4) for name, value in active.items()}


def _constraints_for_lifecycle_state(state: str) -> list[str]:
    if state == "constrained":
        return [
            "Do not propose near-duplicates of failed instances from this family.",
            "Only retry this family with a materially different target, slice, parameterization, or expected mechanism.",
        ]
    if state == "paused":
        return ["Do not retry this family unless later evidence makes it active again."]
    return []


def _family_state_reason(
    family_name: str,
    state: str,
    states: list[TransformContextState],
) -> str:
    counts = Counter(state.state for state in states)
    if state == "promoted":
        return f"{family_name} has at least one promoted branch-local transform context."
    if state == "active":
        return f"{family_name} has viable branch-local transform contexts."
    if state == "constrained":
        return f"{family_name} is only viable through constrained contexts that require materially distinct retries."
    if state == "paused":
        return f"{family_name} is paused across current branch contexts pending stronger evidence."
    return f"{family_name} has no active branch-local evidence signal. Context states: {dict(counts)}."


def _context_summary_reason(state: str) -> str:
    if state == "promoted":
        return "Recent same-context evidence improved the dev objective."
    if state == "constrained":
        return "Same-context evidence regressed or repeatedly failed."
    if state == "paused":
        return "Same-context evidence failed once without enough evidence to retry immediately."
    if state == "active":
        return "Context remains active under current evidence."
    return "No evaluated evidence for this context."


def _context_prompt_row(state: TransformContextState) -> dict[str, Any]:
    key = state.key.to_dict()
    return {
        "id": key["id"],
        "scope_id": key["scope_id"],
        "family": key["family"],
        "target_names": key["target_names"],
        "ops": key["ops"],
        "target_slice": key["target_slice"],
        "mechanism": key["mechanism"],
        "state": state.state,
        "suitability": state.suitability,
        "reason": state.reason,
        "constraints": list(state.constraints),
        "accepted_count": state.accepted_count,
        "rejected_count": state.rejected_count,
    }


def _operation_context_error(
    operation_key: TransformContextKey,
    search_hypothesis: SearchHypothesis,
) -> str | None:
    exact_state = search_hypothesis.context_states.get(operation_key.id)
    if exact_state is not None:
        if exact_state.state in {"active", "promoted"}:
            return None
        if exact_state.state == "constrained":
            return f"constrained transform context {operation_key.id!r} requires a materially distinct mechanism"
        return f"inactive transform context {operation_key.id!r}"
    same_scope_states = [
        state
        for state in search_hypothesis.context_states.values()
        if _context_scope_covers(state.key, operation_key)
    ]
    if any(state.state in {"active", "promoted"} for state in same_scope_states):
        return None
    if any(
        state.state in {"constrained", "paused"} and state.key.mechanism != operation_key.mechanism
        for state in same_scope_states
    ):
        return None
    if same_scope_states:
        return f"inactive transform context scope {operation_key.scope_id!r}"
    if operation_key.family in search_hypothesis.active_families:
        return None
    return f"inactive surface mechanism {operation_key.family!r}"


def _context_scope_covers(known: TransformContextKey, candidate: TransformContextKey) -> bool:
    if known.family != candidate.family or known.target_slice != candidate.target_slice:
        return False
    if not set(candidate.target_names).issubset(set(known.target_names)):
        return False
    if not set(candidate.ops).issubset(set(known.ops)):
        return False
    return True


def _row_score_delta(row: dict[str, Any]) -> float | None:
    comparison = row.get("comparison_to_parent") or {}
    if "score_delta" not in comparison:
        return None
    return float(comparison["score_delta"])


def _suitability_reason(family: str, evidence: list[str], suitability: float) -> str:
    if suitability <= 0:
        return f"{family} has no current evidence signal."
    return f"{family} is plausible because " + "; ".join(evidence) + "."


def _target_slices(summary: CandidateSummary) -> list[str]:
    slices: list[str] = []
    for category, metrics in summary.category_metrics.items():
        count = int(metrics.get("count", 0))
        pass_count = int(metrics.get("pass_count", 0))
        if count > pass_count:
            slices.append(f"category:{category}")
    for label, count in summary.failure_labels.items():
        if count:
            slices.append(f"failure_label:{label}")
    return sorted(set(slices))


def _high_metric_threshold(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, int(0.75 * (len(ordered) - 1)))]


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


def _few_shot_shape(value: Any) -> str:
    if not isinstance(value, list):
        return _value_class(value)
    key_sets = sorted(
        ",".join(sorted(str(key) for key in item.keys()))
        for item in value
        if isinstance(item, dict)
    )
    return f"count={len(value)};keys={';'.join(key_sets[:3]) or '-'}"


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


def _unique(values: Any) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows
