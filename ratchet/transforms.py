from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from ratchet.results import Comparison, PatchSummary
from ratchet.types import AgentPatch, EditableTarget, FailureDiagnosis, OptimizationObjective, PatchOperation


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
        operations = tuple(candidate.patch.operations)
        if candidate.transform_family == "targeted_few_shot" and not operations:
            source_ids = candidate.transform_parameters.get("source_case_ids")
            mechanism = _parameter_mechanism_signature(candidate.transform_parameters)
            if isinstance(source_ids, list):
                mechanism = (f"few_shot:count={len(source_ids)}", *mechanism)
            return cls(
                family=candidate.transform_family,
                target_names=("few_shot",),
                ops=("add_few_shot",),
                target_slice=candidate.target_slice,
                mechanism=mechanism,
                transform_instance=candidate.transform_instance or candidate.hypothesis or "candidate",
            )
        return cls(
            family=candidate.transform_family,
            target_names=tuple(operation.target for operation in operations),
            ops=tuple(operation.op for operation in operations),
            target_slice=candidate.target_slice,
            mechanism=(
                *tuple(_operation_mechanism_signature(operation) for operation in operations),
                *_parameter_mechanism_signature(candidate.transform_parameters),
            ),
            transform_instance=candidate.transform_instance or candidate.hypothesis or "candidate",
        )

    @classmethod
    def from_operation(
        cls,
        *,
        family: str,
        operation: PatchOperation,
        target_slice: str = "global",
        transform_instance: str = "candidate",
    ) -> "TransformContextKey":
        return cls(
            family=family,
            target_names=(operation.target,),
            ops=(operation.op,),
            target_slice=target_slice,
            mechanism=(_operation_mechanism_signature(operation),),
            transform_instance=transform_instance,
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
        patch_payload = row.get("proposal") or (row.get("candidate") or {}).get("patch") or row.get("patch") or {}
        operations = patch_payload.get("operations", []) if isinstance(patch_payload, dict) else []
        operation_objects = [
            PatchOperation.from_dict(operation)
            for operation in operations
            if isinstance(operation, dict)
        ]
        return cls(
            family=str(row.get("transform_family") or "unknown"),
            target_names=tuple(operation.target for operation in operation_objects),
            ops=tuple(operation.op for operation in operation_objects),
            target_slice=str(row.get("target_slice") or "global"),
            mechanism=tuple(_operation_mechanism_signature(operation) for operation in operation_objects),
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
class CandidateProposal:
    patch: AgentPatch
    transform_family: str
    transform_instance: str = ""
    transform_parameters: dict[str, Any] = field(default_factory=dict)
    target_slice: str = "global"
    hypothesis: str = ""
    expected_effects: dict[str, Any] = field(default_factory=dict)
    evaluation_plan: str = "full_dev"

    def to_dict(self) -> dict[str, Any]:
        return {
            "transform_family": self.transform_family,
            "transform_instance": self.transform_instance,
            "transform_parameters": dict(self.transform_parameters),
            "target_slice": self.target_slice,
            "transform_context": TransformContextKey.from_candidate(self).to_dict(),
            "hypothesis": self.hypothesis,
            "expected_effects": dict(self.expected_effects),
            "evaluation_plan": self.evaluation_plan,
            "patch": self.patch.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateProposal":
        if "patch" in payload:
            patch_payload = payload["patch"]
        elif str(payload.get("transform_family", "")) == "targeted_few_shot":
            patch_payload = {"operations": []}
        else:
            patch_payload = payload
        return cls(
            patch=AgentPatch.from_dict(dict(patch_payload)),
            transform_family=str(payload.get("transform_family", "")),
            transform_instance=str(payload.get("transform_instance", "")),
            transform_parameters=dict(payload.get("transform_parameters", {})),
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
    reason: str = "No candidates evaluated for this transform family."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


TRANSFORM_FAMILIES: dict[str, TransformFamily] = {
    "prompt_rewrite": TransformFamily(
        name="prompt_rewrite",
        category="instructions",
        purpose="Revise or add instruction text to change agent behavior.",
        supported_edit_kinds=["instruction"],
        supported_ops=["add_instruction", "revise_instruction"],
        activation_signals=["semantic_failures", "invalid_output", "general_correctness_gap"],
        expected_effects={"correctness": "possible_increase", "cost": "neutral", "latency": "neutral"},
        risks=["overconstrains behavior", "moves failures between slices"],
        required_measurements=["score_delta", "regressions", "cost_delta", "latency_delta"],
        complexity_cost=1.0,
        parameter_contract={
            "recommended": {
                "mechanism_class": ["rubric_clarification", "label_disambiguation", "format_instruction", "grounding", "fallback_policy"],
                "target_names": "editable instruction targets touched by the patch",
                "affected_slices": "labels, categories, or failure slices expected to move",
            }
        },
    ),
    "output_contract_tightening": TransformFamily(
        name="output_contract_tightening",
        category="output",
        purpose="Tighten externally visible output format or schema instructions.",
        supported_edit_kinds=["output", "instruction"],
        supported_ops=["add_output_constraint", "add_instruction", "revise_instruction"],
        activation_signals=["invalid_output", "output_contract_failure"],
        expected_effects={"invalid_output_rate": "decrease", "correctness": "possible_increase"},
        risks=["may reduce semantic quality while fixing format"],
        required_measurements=["invalid_output_rate_delta", "score_delta", "regressions"],
        complexity_cost=1.0,
        parameter_contract={
            "recommended": {
                "mechanism_class": ["json_schema_clarification", "required_field_rule", "parser_compatibility"],
                "affected_failure_modes": "invalid-output or output-contract labels addressed",
            }
        },
    ),
    "targeted_few_shot": TransformFamily(
        name="targeted_few_shot",
        category="examples",
        purpose="Add representative examples for weak labels or slices.",
        supported_edit_kinds=["few_shot"],
        supported_ops=["add_few_shot"],
        activation_signals=["weak_slice", "label_confusion", "invalid_output"],
        expected_effects={"target_slice_score": "possible_increase", "cost": "increase"},
        risks=["overfits examples", "increases prompt length"],
        required_measurements=["target_slice_score_delta", "non_target_regressions", "token_delta"],
        complexity_cost=1.5,
        parameter_contract={
            "required": {
                "source_case_ids": "train example IDs from proposal_example_bank",
            },
            "recommended": {
                "target_labels": "labels or slices addressed",
                "selection_strategy": ["representative", "contrastive", "hard_negative"],
                "affected_confusions": "expected->actual confusion pairs the examples target",
            },
        },
    ),
    "model_substitution": TransformFamily(
        name="model_substitution",
        category="model",
        purpose="Change to another allowed model.",
        supported_edit_kinds=["model"],
        supported_ops=["change_model"],
        activation_signals=["capability_gap", "cost_objective", "latency_objective"],
        expected_effects={"correctness": "variable", "cost": "variable", "latency": "variable"},
        risks=["changes behavior globally", "cost or latency regression"],
        required_measurements=["score_delta", "cost_delta", "latency_delta"],
        complexity_cost=1.0,
        parameter_contract={
            "recommended": {
                "from_model": "current model",
                "to_model": "allowed replacement model",
                "expected_tradeoff": "quality, cost, or latency tradeoff being tested",
            }
        },
    ),
    "tool_policy_revision": TransformFamily(
        name="tool_policy_revision",
        category="tools",
        purpose="Revise tool description, tool use policy, or enabled state.",
        supported_edit_kinds=["tool"],
        supported_ops=["revise_tool_description", "revise_tool_policy", "set_runtime_param"],
        activation_signals=["tool_errors", "retrieval_or_tool_dependent_failures"],
        expected_effects={"correctness": "possible_increase", "tool_calls": "variable"},
        risks=["overuses or underuses tools"],
        required_measurements=["score_delta", "tool_calls", "cost_delta", "latency_delta"],
        complexity_cost=1.2,
        parameter_contract={
            "recommended": {
                "tool_name": "tool whose description, policy, or enablement changes",
                "gating_signal": "when the tool should be used differently",
            }
        },
    ),
    "retrieval_tuning": TransformFamily(
        name="retrieval_tuning",
        category="retrieval",
        purpose="Tune retrieval parameters exposed by the agent policy.",
        supported_edit_kinds=["retrieval"],
        supported_ops=["set_retrieval_param"],
        activation_signals=["retrieval_dependent_failures", "cost_objective", "latency_objective"],
        expected_effects={"correctness": "variable", "cost": "variable", "latency": "variable"},
        risks=["misses evidence", "adds context noise"],
        required_measurements=["score_delta", "token_delta", "cost_delta", "latency_delta"],
        complexity_cost=1.0,
        parameter_contract={
            "recommended": {
                "retrieval_param": "retrieval setting being changed",
                "expected_tradeoff": "recall, noise, cost, or latency tradeoff",
            }
        },
    ),
    "runtime_tuning": TransformFamily(
        name="runtime_tuning",
        category="runtime",
        purpose="Tune runtime controls such as caps, reasoning settings, or validator flags.",
        supported_edit_kinds=["runtime"],
        supported_ops=["set_runtime_param"],
        activation_signals=["cost_objective", "latency_objective", "invalid_output"],
        expected_effects={"cost": "variable", "latency": "variable", "correctness": "variable"},
        risks=["hidden correctness tradeoff"],
        required_measurements=["score_delta", "cost_delta", "latency_delta"],
        complexity_cost=1.0,
        parameter_contract={
            "recommended": {
                "runtime_param": "runtime setting being changed",
                "expected_tradeoff": "quality, cost, or latency tradeoff",
            }
        },
    ),
    "verifier_retry": TransformFamily(
        name="verifier_retry",
        category="verification",
        purpose="Enable verifier or repair retry behavior exposed by the policy.",
        supported_edit_kinds=["verifier"],
        supported_ops=["add_verifier_retry"],
        activation_signals=["invalid_output", "persistent_high_risk_failures"],
        expected_effects={"correctness": "possible_increase", "cost": "increase", "latency": "increase"},
        risks=["adds model calls", "masks underlying prompt issues"],
        required_measurements=["score_delta", "cost_delta", "latency_delta", "retry_rate"],
        complexity_cost=2.0,
        parameter_contract={
            "recommended": {
                "trigger": "failure condition or high-risk slice that should invoke verification",
                "retry_limit": "number of repair attempts",
            }
        },
    ),
}


SIGNAL_WEIGHTS_BY_FAMILY: dict[str, dict[str, float]] = {
    "branch_failures": {
        "prompt_rewrite": 1.0,
        "output_contract_tightening": 1.0,
        "targeted_few_shot": 1.0,
        "model_substitution": 1.0,
        "tool_policy_revision": 1.0,
        "retrieval_tuning": 1.0,
        "runtime_tuning": 1.0,
        "verifier_retry": 1.0,
    },
    "invalid_output": {
        "output_contract_tightening": 1.0,
        "prompt_rewrite": 1.0,
        "targeted_few_shot": 1.0,
        "verifier_retry": 1.0,
        "runtime_tuning": 1.0,
        "retrieval_tuning": -0.6,
        "tool_policy_revision": -0.6,
        "model_substitution": -0.6,
    },
    "correctness_gap": {
        "prompt_rewrite": 1.0,
        "model_substitution": 1.0,
        "targeted_few_shot": 1.0,
        "tool_policy_revision": 1.0,
        "retrieval_tuning": 1.0,
    },
    "tool_dependent_slice": {
        "retrieval_tuning": 1.0,
        "tool_policy_revision": 1.0,
    },
    "cost_objective": {
        "model_substitution": 1.0,
        "retrieval_tuning": 1.0,
        "runtime_tuning": 1.0,
        "tool_policy_revision": 1.0,
    },
    "high_cost_cases": {
        "retrieval_tuning": 1.0,
        "runtime_tuning": 1.0,
        "model_substitution": 1.0,
    },
    "latency_objective": {
        "model_substitution": 1.0,
        "retrieval_tuning": 1.0,
        "runtime_tuning": 1.0,
        "tool_policy_revision": 1.0,
    },
    "high_latency_cases": {
        "retrieval_tuning": 1.0,
        "runtime_tuning": 1.0,
        "model_substitution": 1.0,
    },
    "weak_slices": {
        "targeted_few_shot": 1.0,
        "prompt_rewrite": 1.0,
        "verifier_retry": 1.0,
    },
    "runtime_errors": {
        "runtime_tuning": 1.0,
        "verifier_retry": 1.0,
        "model_substitution": 1.0,
    },
    "runtime_truncation": {
        "runtime_tuning": 1.0,
        "output_contract_tightening": 1.0,
        "prompt_rewrite": 0.4,
        "model_substitution": 0.2,
        "targeted_few_shot": -0.4,
    },
}


SIGNAL_REASONS = {
    "branch_failures": "current branch has failing cases",
    "invalid_output": "invalid output failures observed",
    "correctness_gap": "correctness gap observed",
    "tool_dependent_slice": "tool-dependent slice signal observed",
    "cost_objective": "cost objective active",
    "high_cost_cases": "high-cost cases observed",
    "latency_objective": "latency objective active",
    "high_latency_cases": "high-latency cases observed",
    "weak_slices": "weak or failing slices available",
    "runtime_errors": "runtime errors observed",
    "runtime_truncation": "finish_reason=length or parser fallback observed",
}


def transform_registry() -> dict[str, TransformFamily]:
    return dict(TRANSFORM_FAMILIES)


def build_behavior_profile(summary: PatchSummary) -> BehaviorProfile:
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
    summary: PatchSummary,
    surface: list[EditableTarget],
    objective: OptimizationObjective,
    history: list[dict[str, Any]],
    parent_patch_hash: str | None = None,
    diagnoses: list[FailureDiagnosis] | None = None,
    proposal_example_count: int = 0,
) -> SearchHypothesis:
    profile = build_behavior_profile(summary)
    surface_kinds = {target.kind for target in surface}
    surface_ops = {op for target in surface for op in target.allowed_ops}
    branch_history = select_branch_history(history, parent_patch_hash or summary.patch_hash)
    diagnosis_signals = _diagnosis_signals(diagnoses or [])
    context_states: dict[str, TransformContextState] = {}
    for family in TRANSFORM_FAMILIES.values():
        if family.name == "targeted_few_shot" and proposal_example_count <= 0:
            context_key = TransformContextKey(family=family.name)
            context_states[context_key.id] = TransformContextState(
                key=context_key,
                state="available",
                suitability=0.0,
                reason="No proposal-safe train examples are available for targeted few-shot selection.",
                constraints=["Add train/search examples before proposing targeted few-shot patches."],
            )
            continue
        if not (set(family.supported_edit_kinds) & surface_kinds) or not (set(family.supported_ops) & surface_ops):
            context_key = TransformContextKey(family=family.name)
            context_states[context_key.id] = TransformContextState(
                key=context_key,
                state="available",
                suitability=0.0,
                reason="No compatible editable target is available.",
                constraints=["No compatible editable target or operation is available."],
            )
            continue
        base_suitability, base_evidence = _suitability(family, profile, objective)
        for context_key in _candidate_context_keys(
            family=family,
            surface=surface,
            profile=profile,
            diagnosis_signals=diagnosis_signals,
        ):
            suitability, evidence = _context_suitability(
                family=family,
                context_key=context_key,
                base_suitability=base_suitability,
                base_evidence=base_evidence,
                diagnosis_signals=diagnosis_signals,
            )
            context_states[context_key.id] = _context_lifecycle_state(
                key=context_key,
                rows=_rows_for_context(branch_history, context_key),
                suitability=suitability,
                evidence=evidence,
            )
    for row in branch_history:
        context_key = TransformContextKey.from_row(row)
        if context_key.family not in TRANSFORM_FAMILIES:
            continue
        if context_key.id in context_states:
            continue
        family = TRANSFORM_FAMILIES[context_key.family]
        base_suitability, base_evidence = _suitability(family, profile, objective)
        suitability, evidence = _context_suitability(
            family=family,
            context_key=context_key,
            base_suitability=base_suitability,
            base_evidence=base_evidence,
            diagnosis_signals=diagnosis_signals,
        )
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
        rationale="Search hypothesis derived from current behavior profile, diagnoses, editable surface, objective, and branch-local transform context history.",
    )


def validate_candidate_transform(
    candidate: CandidateProposal,
    *,
    surface: list[EditableTarget],
    search_hypothesis: SearchHypothesis | None = None,
) -> str | None:
    registry = TRANSFORM_FAMILIES
    family = registry.get(candidate.transform_family)
    if family is None:
        return f"unknown transform family {candidate.transform_family!r}"
    parameter_error = _transform_parameter_contract_error(candidate, family)
    if parameter_error is not None:
        return parameter_error
    if not candidate.patch.operations and candidate.transform_family != "targeted_few_shot":
        return "candidate patch must include at least one operation"
    target_by_name = {target.name: target for target in surface}
    target_by_path = {target.path: target for target in surface}
    for operation in candidate.patch.operations:
        target = target_by_name.get(operation.target) or target_by_path.get(operation.target)
        if target is None:
            return f"unknown target {operation.target!r}"
        if target.kind not in family.supported_edit_kinds:
            return f"target kind {target.kind!r} is incompatible with transform family {family.name!r}"
        if operation.op not in family.supported_ops:
            return f"operation {operation.op!r} is incompatible with transform family {family.name!r}"
    if search_hypothesis is not None:
        eligibility_error = validate_candidate_context(candidate, search_hypothesis=search_hypothesis)
        if eligibility_error is not None:
            return eligibility_error
    return None


def _transform_parameter_contract_error(candidate: CandidateProposal, family: TransformFamily) -> str | None:
    required = family.parameter_contract.get("required")
    if not isinstance(required, dict):
        return None
    for key in required:
        if key not in candidate.transform_parameters:
            return f"transform family {family.name!r} requires transform_parameters.{key}"
        value = candidate.transform_parameters[key]
        if key.endswith("_ids") or key == "source_case_ids":
            if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
                return f"transform_parameters.{key} must be a non-empty string array"
    return None


def validate_candidate_context(
    candidate: CandidateProposal,
    *,
    search_hypothesis: SearchHypothesis,
) -> str | None:
    family_state = search_hypothesis.family_states.get(candidate.transform_family)
    if family_state is None or candidate.transform_family not in search_hypothesis.active_families:
        return f"inactive transform family {candidate.transform_family!r}"
    combined_key = TransformContextKey.from_candidate(candidate)
    exact_state = search_hypothesis.context_states.get(combined_key.id)
    if exact_state is not None and exact_state.state in {"paused", "available"}:
        return f"inactive transform context {combined_key.id!r}"
    if exact_state is not None and exact_state.state == "constrained":
        return f"constrained transform context {combined_key.id!r} requires a materially distinct mechanism"
    for operation in candidate.patch.operations:
        operation_key = TransformContextKey.from_operation(
            family=candidate.transform_family,
            operation=operation,
            target_slice=candidate.target_slice,
            transform_instance=candidate.transform_instance or candidate.hypothesis or "candidate",
        )
        operation_error = _operation_context_error(operation_key, search_hypothesis)
        if operation_error is not None:
            return operation_error
    return None


def select_branch_history(history: list[dict[str, Any]], parent_patch_hash: str | None) -> list[dict[str, Any]]:
    if not parent_patch_hash:
        return list(history)
    producing_parent_by_child: dict[str, str] = {}
    for row in history:
        child = row.get("patch_hash")
        parent = row.get("parent_patch_hash")
        if row.get("accepted") and isinstance(child, str) and isinstance(parent, str):
            producing_parent_by_child[child] = parent
    lineage = {parent_patch_hash}
    cursor = parent_patch_hash
    while cursor in producing_parent_by_child:
        cursor = producing_parent_by_child[cursor]
        if cursor in lineage:
            break
        lineage.add(cursor)
    return [
        row
        for row in history
        if row.get("patch_hash") in lineage or row.get("parent_patch_hash") == parent_patch_hash
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
    for family in sorted(set(grouped) | set(proposed_counts) | set(TRANSFORM_FAMILIES)):
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
            reason = "At least one candidate from this transform family improved the configured objective on dev."
        elif evaluated_count >= 2 or score_regressed:
            state = "constrained"
            reason = (
                "At least one candidate from this transform family regressed score; future attempts should use a distinct target, slice, or instance."
                if score_regressed
                else "Multiple evaluated candidates failed the configured objective gate; future attempts should avoid near-duplicate instances."
            )
        elif evaluated_count == 1:
            state = "paused"
            reason = "The evaluated candidate failed the configured objective gate."
        else:
            state = "available"
            reason = "No candidates evaluated for this transform family."
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


def _suitability(
    family: TransformFamily,
    profile: BehaviorProfile,
    objective: OptimizationObjective,
) -> tuple[float, list[str]]:
    signals = _evidence_signals(profile, objective)
    score = -family.complexity_cost * 0.03
    evidence: list[str] = []
    for signal, strength in signals.items():
        family_weight = SIGNAL_WEIGHTS_BY_FAMILY.get(signal, {}).get(family.name, 0.0)
        if family_weight == 0:
            continue
        score += strength * family_weight
        evidence.append(SIGNAL_REASONS.get(signal, signal))
    return max(round(score, 4), 0.0), _unique(evidence)


def _evidence_signals(
    profile: BehaviorProfile,
    objective: OptimizationObjective,
) -> dict[str, float]:
    signals: dict[str, float] = {}
    has_failures = profile.pass_count < profile.case_count
    failure_rate = 1.0 - profile.pass_rate
    if has_failures:
        signals["branch_failures"] = min(0.15 + failure_rate * 0.25, 0.4)
    if profile.invalid_output_rate > 0:
        signals["invalid_output"] = 0.3 + min(profile.invalid_output_rate * 0.35, 0.35)
    if objective.mode == "correctness" and has_failures:
        signals["correctness_gap"] = 0.25
        if _metadata_flag_present(profile, "needs_tool"):
            signals["tool_dependent_slice"] = 0.2
    if objective.mode == "cost":
        signals["cost_objective"] = 0.55
        if profile.high_cost_case_ids:
            signals["high_cost_cases"] = 0.15
    if objective.mode == "latency":
        signals["latency_objective"] = 0.55
        if profile.high_latency_case_ids:
            signals["high_latency_cases"] = 0.15
    if profile.weak_slice_count:
        signals["weak_slices"] = min(0.1 + profile.weak_slice_count * 0.05, 0.25)
    if profile.runtime_error_rate > 0:
        signals["runtime_errors"] = 0.2
    if profile.length_finish_rate > 0 or profile.parser_fallback_rate > 0:
        signals["runtime_truncation"] = min(
            0.2 + (profile.length_finish_rate + profile.parser_fallback_rate) * 0.3,
            0.5,
        )
    return signals


def _candidate_context_keys(
    *,
    family: TransformFamily,
    surface: list[EditableTarget],
    profile: BehaviorProfile,
    diagnosis_signals: dict[str, set[str]],
) -> list[TransformContextKey]:
    slices = sorted(diagnosis_signals["target_slices"] or set(profile.target_slices[:3]) or {"global"})
    keys: list[TransformContextKey] = []
    for target in surface:
        ops = tuple(sorted(set(target.allowed_ops) & set(family.supported_ops)))
        if not ops or target.kind not in family.supported_edit_kinds:
            continue
        target_slices = slices
        if diagnosis_signals["target_names"] and target.name not in diagnosis_signals["target_names"]:
            target_slices = ["global"]
        for target_slice in target_slices[:3]:
            keys.append(
                TransformContextKey(
                    family=family.name,
                    target_names=(target.name,),
                    ops=ops,
                    target_slice=target_slice,
                    transform_instance="candidate",
                )
            )
    return keys or [TransformContextKey(family=family.name)]


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


def _context_suitability(
    *,
    family: TransformFamily,
    context_key: TransformContextKey,
    base_suitability: float,
    base_evidence: list[str],
    diagnosis_signals: dict[str, set[str]],
) -> tuple[float, list[str]]:
    suitability = base_suitability
    evidence = list(base_evidence)
    if set(context_key.target_names) & diagnosis_signals["target_names"]:
        suitability += 0.15
        evidence.append("diagnosis points at this editable target")
    if diagnosis_signals["categories"] and context_key.family in {
        "prompt_rewrite",
        "output_contract_tightening",
        "targeted_few_shot",
        "tool_policy_revision",
        "retrieval_tuning",
    }:
        suitability += 0.05
        evidence.append("diagnosis categories provide targetable failure context")
    if context_key.target_slice in diagnosis_signals["target_slices"]:
        suitability += 0.05
        evidence.append("context targets a diagnosed failure slice")
    return round(max(suitability, 0.0), 4), _unique(evidence)


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
    for family_name in sorted(TRANSFORM_FAMILIES):
        states = grouped.get(family_name, [])
        if not states:
            family_states[family_name] = TransformFamilyState(
                family=family_name,
                state="available",
                suitability=0.0,
                budget_share=0.0,
                reason="No transform contexts are available for this family.",
            )
            continue
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
    return f"inactive transform family {operation_key.family!r}"


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


def _target_slices(summary: PatchSummary) -> list[str]:
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


def _metadata_flag_present(profile: BehaviorProfile, key: str) -> bool:
    # BehaviorProfile intentionally stores aggregate slice facts, not raw cases.
    # For now, infer common metadata signals from target slice names when present.
    return any(key in slice_name for slice_name in profile.target_slices)


def _high_metric_threshold(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, int(0.75 * (len(ordered) - 1)))]


def _operation_mechanism_signature(operation: PatchOperation) -> str:
    value = operation.value
    if operation.op == "change_model":
        return f"model:{_normalize_token(str(value), default='unknown')}"
    if operation.op in {"set_runtime_param", "set_retrieval_param"}:
        return f"{operation.op}:{_value_class(value)}"
    if operation.op == "add_few_shot":
        return f"few_shot:{_few_shot_shape(value)}"
    if operation.op == "add_verifier_retry":
        return f"verifier:{_mapping_shape(value)}"
    if operation.op in {
        "add_instruction",
        "revise_instruction",
        "add_output_constraint",
        "revise_tool_description",
        "revise_tool_policy",
    }:
        return f"text:{_text_mechanism_class(str(value))}"
    return f"{operation.op}:{_value_class(value)}"


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
        "tool_use": ("tool", "search", "retrieve", "retrieval", "web", "lookup"),
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
