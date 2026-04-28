from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentSpec, MECHANISMS_BY_FAMILY, TaskTheory, build_task_theory
from ratchet.io import extract_json_object, patch_hash
from ratchet.model_client import (
    ResponsesModelClient,
    combine_response_diagnostics,
    error_response_diagnostics,
    response_diagnostics,
)
from ratchet.patches import compose_patches
from ratchet.results import PatchSummary
from ratchet.transforms import (
    CandidateProposal,
    Intervention,
    SearchHypothesis,
    build_search_hypothesis,
    transform_registry,
    validate_candidate_transform,
)
from ratchet.types import AgentPatch, AgentSpec, EditableTarget, FailureDiagnosis, OptimizationObjective, PatchOperation
from ratchet.types import EvalCase
from ratchet.validation import PatchValidator


MAX_PROPOSALS_PER_ITERATION = 8
PROPOSER_INSTRUCTIONS = (
    "You are Ratchet's task-agnostic patch proposer. Return JSON with experiments[] and optional "
    "target_considerations[]. Keep text concise. Use planner_guidance and search_hypothesis to choose "
    "mechanism_class values from the allowed mechanism list. Each candidate must name an active "
    "transform_family, a candidate_role in atomic/composed/control/ablation/compression, a hypothesis, "
    "and one intervention. Normal candidates use intervention.kind='patch' with payload.patch.operations; "
    "few-shot candidates use only intervention.kind='example_selection' with payload.source_case_ids from "
    "proposal_example_bank. Do not inline few-shot examples. Patch operations must use editable_targets, "
    "allowed_ops, value_schema, and the declared transform family's supported ops. Declare family by the "
    "actual operation: instruction ops -> prompt_rewrite/output_contract_tightening, set_runtime_param -> "
    "runtime_tuning, change_model -> model_substitution, source_case_ids/add_few_shot -> targeted_few_shot. "
    "Do not copy diagnostic_only_examples into patch values; only proposal-safe train examples may be copied, "
    "and only through source_case_id. Prefer minimal, independently evaluable patches. For cost/latency modes, "
    "preserve correctness and explore model/runtime/retrieval/tool efficiency even when failures are absent. "
    "Return empty experiments only when no safe evaluable candidate exists."
)


@dataclass
class ProposalStats:
    raw_count: int = 0
    valid_count: int = 0
    returned_count: int = 0
    invalid_count: int = 0
    duplicate_count: int = 0
    error: str | None = None
    invalid_reasons: dict[str, int] | None = None
    target_considerations: list[dict[str, Any]] | None = None
    plan_audit: dict[str, Any] | None = None
    raw_output_text: str = ""
    call_diagnostics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_count": self.raw_count,
            "valid_count": self.valid_count,
            "returned_count": self.returned_count,
            "invalid_count": self.invalid_count,
            "duplicate_count": self.duplicate_count,
            "error": self.error,
            "invalid_reasons": dict(self.invalid_reasons or {}),
            "target_considerations": list(self.target_considerations or []),
            "plan_audit": dict(self.plan_audit or {}),
            "raw_output_text": self.raw_output_text,
            "call_diagnostics": dict(self.call_diagnostics or {}),
        }


class ProposalEngine:
    def __init__(
        self,
        *,
        env_path: str,
        model: str,
        reasoning_effort: str,
    ) -> None:
        self.env_path = env_path
        self.model = model
        self.reasoning_effort = reasoning_effort
        self._client: ResponsesModelClient | None = None
        self.last_stats = ProposalStats()
        self.last_candidate_rows: list[dict[str, Any]] = []
        self.last_invalid_candidate_rows: list[dict[str, Any]] = []
        self.last_call_diagnostics: dict[str, Any] | None = None
        self._last_raw_candidate_count = 0
        self._last_parse_invalid_reasons: Counter[str] = Counter()
        self._last_parse_invalid_candidate_rows: list[dict[str, Any]] = []
        self._last_plan_audit: dict[str, Any] = {}

    def propose(
        self,
        summary: PatchSummary,
        surface: list[EditableTarget],
        *,
        objective: OptimizationObjective,
        seen_hashes: set[str],
        current_spec: AgentSpec | None,
        history: list[dict[str, Any]],
        search_hypothesis: SearchHypothesis | None = None,
        diagnosis: FailureDiagnosis | None = None,
        diagnoses: list[FailureDiagnosis] | None = None,
        task_theory: TaskTheory | None = None,
        proposal_example_bank: ProposalExampleBank | None = None,
        proposal_example_cases: tuple[EvalCase, ...] = (),
        proposal_budget: int = MAX_PROPOSALS_PER_ITERATION,
    ) -> tuple[list[CandidateProposal], str]:
        proposals: list[CandidateProposal] = []
        analysis_parts: list[str] = []
        invalid_reasons: Counter[str] = Counter()
        proposal_budget = max(0, proposal_budget)
        if search_hypothesis is None:
            search_hypothesis = build_search_hypothesis(
                summary=summary,
                surface=surface,
                objective=objective,
                history=history,
                proposal_example_count=len(proposal_example_bank.examples) if proposal_example_bank else 0,
            )
        diagnosis_context = list(diagnoses or ([] if diagnosis is None else [diagnosis]))
        if task_theory is None:
            task_theory = build_task_theory(
                summary=summary,
                diagnoses=diagnosis_context,
                objective=objective,
                proposal_example_bank=proposal_example_bank,
            )
        llm_proposals, target_considerations = self._llm_proposals(
            summary,
            surface,
            objective=objective,
            diagnoses=diagnosis_context,
            history=history,
            search_hypothesis=search_hypothesis,
            task_theory=task_theory,
            proposal_example_bank=proposal_example_bank,
            proposal_budget=proposal_budget,
        )
        proposals.extend(llm_proposals)
        analysis_parts.append("LLM proposer returned transform candidate proposals.")
        invalid_reasons.update(self._last_parse_invalid_reasons)
        validator = PatchValidator()
        valid: list[CandidateProposal] = []
        budget_valid: list[CandidateProposal] = []
        local_seen: set[str] = set()
        family_quotas = _family_budget_quotas(
            search_hypothesis.budget_allocation,
            proposal_budget=proposal_budget,
        )
        family_counts: Counter[str] = Counter()
        family_budget_groups: set[tuple[str, str]] = set()
        scheduled_group_indices: dict[str, int] = {}
        scheduled_group_flags: dict[str, bool] = {}
        group_count = 0
        candidate_rows: list[dict[str, Any]] = []
        invalid_candidate_rows: list[dict[str, Any]] = []
        invalid_candidate_rows.extend(self._last_parse_invalid_candidate_rows)
        for raw_candidate in proposals:
            reference_error = _targeted_few_shot_reference_error(raw_candidate)
            if reference_error is not None:
                invalid_reasons[reference_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(raw_candidate, reference_error))
                continue
            materialized_candidate, materialization = _materialize_candidate_references(raw_candidate, proposal_example_bank)
            materialization_error = materialization.get("error")
            if materialization_error:
                reason = str(materialization_error)
                invalid_reasons[reason] += 1
                invalid_candidate_rows.append(
                    _invalid_candidate_row(raw_candidate, reason, materialization=materialization)
                )
                continue
            group_valid: list[tuple[CandidateProposal, dict[str, Any], str]] = []
            for candidate in _few_shot_count_variants(materialized_candidate):
                variant_materialization = _few_shot_variant_materialization(candidate, materialization)
                family_error = validate_candidate_transform(
                    candidate,
                    surface=surface,
                    search_hypothesis=search_hypothesis,
                )
                if family_error is not None:
                    invalid_reasons[family_error] += 1
                    invalid_candidate_rows.append(
                        _invalid_candidate_row(candidate, family_error, materialization=variant_materialization)
                    )
                    continue
                is_valid, invalid_reason = validator.validate_with_reason(
                    candidate.patch,
                    current_spec=current_spec,
                    surface=surface,
                    objective=objective,
                    evidence_cases=[evaluation.case for evaluation in summary.evaluations],
                    proposal_example_case_ids=proposal_example_bank.case_ids if proposal_example_bank is not None else None,
                    proposal_example_cases=list(proposal_example_cases),
                )
                if not is_valid:
                    reason = invalid_reason or "invalid patch"
                    invalid_reasons[reason] += 1
                    invalid_candidate_rows.append(
                        _invalid_candidate_row(candidate, reason, materialization=variant_materialization)
                    )
                    continue
                digest = patch_hash(compose_patches(summary.patch, candidate.patch))
                if digest in seen_hashes or digest in local_seen:
                    invalid_reasons["duplicate patch"] += 1
                    invalid_candidate_rows.append(
                        _invalid_candidate_row(candidate, "duplicate patch", materialization=variant_materialization)
                    )
                    continue
                local_seen.add(digest)
                group_valid.append((candidate, variant_materialization, digest))
            if not group_valid:
                continue
            group_family = group_valid[0][0].transform_family
            budget_group = _candidate_budget_group(group_valid[0][0])
            family_budget_group = (group_family, budget_group)
            raw_quota = family_quotas.get(group_family)
            if raw_quota is None:
                quota = proposal_budget if not family_quotas else 0
            else:
                quota = max(1, raw_quota)
            if family_budget_group not in family_budget_groups and family_counts[group_family] >= quota:
                reason = f"transform family budget exceeded for {group_family!r} (quota {quota})"
                invalid_reasons[reason] += len(group_valid)
                for candidate, variant_materialization, _digest in group_valid:
                    invalid_candidate_rows.append(
                        _invalid_candidate_row(candidate, reason, materialization=variant_materialization)
                    )
                continue
            if family_budget_group not in family_budget_groups:
                family_counts[group_family] += 1
                family_budget_groups.add(family_budget_group)
            if budget_group not in scheduled_group_indices:
                group_count += 1
                scheduled_group_indices[budget_group] = group_count
                scheduled_group_flags[budget_group] = group_count <= proposal_budget
            proposal_group = scheduled_group_indices[budget_group]
            group_scheduled = scheduled_group_flags[budget_group]
            for variant_rank, (candidate, variant_materialization, digest) in enumerate(group_valid, start=1):
                valid.append(candidate)
                budget_valid.append(candidate)
                candidate_rows.append(
                    {
                        "rank": len(candidate_rows) + 1,
                        "proposal_group": proposal_group,
                        "variant_rank": variant_rank,
                        "proposal_patch_hash": patch_hash(candidate.patch),
                        "patch_hash": digest,
                        "proposal": candidate.patch.to_dict(),
                        "candidate": candidate.to_dict(),
                        "transform_family": candidate.transform_family,
                        "mechanism_class": candidate.mechanism_class,
                        "experiment_id": candidate.experiment_id,
                        "candidate_role": candidate.candidate_role,
                        "comparison_group": candidate.comparison_group,
                        "transform_instance": candidate.transform_instance,
                        "target_slice": candidate.target_slice,
                        "hypothesis": candidate.hypothesis,
                        "evaluation_plan": candidate.evaluation_plan,
                        "materialization": variant_materialization,
                        "scheduled": group_scheduled,
                        "family_quota": quota,
                        "family_rank": family_counts[candidate.transform_family],
                    }
                )
        returned_hashes = {
            row["patch_hash"]
            for row in candidate_rows
            if row.get("scheduled") and isinstance(row.get("patch_hash"), str)
        }
        returned = [
            candidate
            for candidate in budget_valid
            if patch_hash(compose_patches(summary.patch, candidate.patch)) in returned_hashes
        ]
        self.last_candidate_rows = candidate_rows
        self.last_invalid_candidate_rows = invalid_candidate_rows
        self.last_stats = ProposalStats(
            raw_count=self._last_raw_candidate_count,
            valid_count=len(budget_valid),
            returned_count=len(returned),
            invalid_count=sum(count for reason, count in invalid_reasons.items() if reason != "duplicate patch"),
            duplicate_count=invalid_reasons.get("duplicate patch", 0),
            error=None,
            invalid_reasons=dict(sorted(invalid_reasons.items())),
            target_considerations=target_considerations,
            plan_audit=self._last_plan_audit,
            raw_output_text=self._last_raw_output_text,
            call_diagnostics=self.last_call_diagnostics,
        )
        if valid:
            analysis_parts.append("Validated LLM transform candidate proposals.")
        else:
            analysis_parts.append("No valid LLM transform candidate proposals.")
        analysis_parts.append(
            "Proposal counts: "
            f"raw={self.last_stats.raw_count}, valid={self.last_stats.valid_count}, "
            f"returned={self.last_stats.returned_count}, invalid={self.last_stats.invalid_count}, "
            f"duplicate={self.last_stats.duplicate_count}."
        )
        return returned, " ".join(analysis_parts)

    def _llm_proposals(
        self,
        summary: PatchSummary,
        surface: list[EditableTarget],
        *,
        objective: OptimizationObjective,
        diagnoses: list[FailureDiagnosis],
        history: list[dict[str, Any]],
        search_hypothesis: SearchHypothesis,
        task_theory: TaskTheory,
        proposal_example_bank: ProposalExampleBank | None,
        proposal_budget: int,
    ) -> tuple[list[CandidateProposal], list[dict[str, Any]]]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        self._last_plan_audit = {}
        target_kinds = sorted({target.kind for target in surface})
        registry = transform_registry()
        active_family_rows = [
            _compact_transform_family(registry[name].to_dict())
            for name in search_hypothesis.active_families
            if name in registry
        ]
        behavior_diagnostics = build_behavior_diagnostics(summary)
        compact_diagnostics = _compact_behavior_diagnostics(behavior_diagnostics)
        planner_guidance = _planner_guidance(
            task_theory=task_theory,
            search_hypothesis=search_hypothesis,
            proposal_budget=proposal_budget,
        )
        prompt = {
            "objective": objective.to_dict(),
            "proposal_budget": proposal_budget,
            "target_kinds": target_kinds,
            "transform_library": active_family_rows,
            "search_hypothesis": search_hypothesis.to_prompt_dict(
                max_contexts_per_family=2,
                max_constrained_contexts=5,
            ),
            "task_theory": _compact_task_theory(task_theory),
            "planner_guidance": planner_guidance,
            "proposal_policy": {
                "empty_patches_allowed": (
                    "Only when no listed editable target can plausibly improve the objective without violating constraints."
                ),
                "cost_or_latency_without_failures": (
                    "If correctness is currently saturated and the objective is cost or latency, still propose minimal "
                    "efficiency patches from the generated surface so the eval loop can validate the tradeoff."
                ),
                "candidate_portfolio": (
                    "Generate an ordered portfolio of distinct, independently evaluable patches up to proposal_budget. "
                    "Do not let prompt edits crowd out other plausible target kinds; rank by expected objective impact "
                    "and constraint risk."
                ),
            },
            "current_patch": _compact_patch(summary.patch.to_dict()),
            "behavior": {
                "mean_score": summary.mean_score,
                "pass_count": summary.pass_count,
                "pass_rate": summary.pass_rate,
                "failure_labels": _top_mapping(summary.failure_labels, limit=12),
            },
            "behavior_diagnostics": compact_diagnostics,
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "diagnoses": [_compact_diagnosis(diagnosis) for diagnosis in diagnoses[:4]],
            "primary_diagnosis": _compact_diagnosis(diagnoses[0]) if diagnoses else None,
            "editable_targets": [_compact_editable_target(target) for target in surface],
            "diagnostic_only_examples": {
                "usage": (
                    "dev examples for diagnosis only. Do not copy their case IDs, inputs, or expected outputs into patches."
                    if not objective.constraints.sanitize_examples
                    else "dev failure metadata for diagnosis only. Raw input, expected, output, notes, and raw_output_text fields are redacted."
                ),
                "sanitized": objective.constraints.sanitize_examples,
                "examples": summary.failed_examples(
                    limit=3,
                    max_text_chars=260,
                    sanitize_text=objective.constraints.sanitize_examples,
                ),
            },
            "proposal_example_bank": (
                _compact_proposal_example_bank(
                    proposal_example_bank,
                    target_labels=_target_labels_for_examples(compact_diagnostics),
                    max_examples=8,
                    max_per_label=2,
                )
                if proposal_example_bank is not None
                else {
                    "usage": "no proposal-safe train examples available",
                    "example_count": 0,
                    "examples": [],
                }
            ),
            "recent_history": _compact_recent_history(history, limit=4),
        }
        try:
            started_at = time.perf_counter()
            prompt_input = _proposal_prompt_input(prompt, proposal_budget=proposal_budget)
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "ratchet_patch_proposals",
                        "strict": False,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "target_considerations": {
                                    "type": "array",
                                    "maxItems": max(len(target_kinds), 1),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "target_kind": {"type": "string", "maxLength": 80},
                                            "decision": {"type": "string", "maxLength": 40},
                                            "rationale": {"type": "string", "maxLength": 280},
                                        },
                                        "required": ["target_kind", "decision", "rationale"],
                                    },
                                },
                                "experiments": {
                                    "type": "array",
                                    "maxItems": max(proposal_budget, 0),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "experiment_id": {"type": "string", "maxLength": 80},
                                            "mechanism_class": {"type": "string", "maxLength": 80},
                                            "mechanism": {"type": "string", "maxLength": 160},
                                            "hypothesis": {"type": "string", "maxLength": 360},
                                            "target_slices": {"type": "array", "items": {"type": "string"}},
                                            "measurements": {"type": "array", "items": {"type": "string"}},
                                            "candidate_roles": {"type": "array", "items": {"type": "string"}},
                                            "candidates": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "transform_family": {"type": "string", "maxLength": 80},
                                                        "mechanism_class": {"type": "string", "maxLength": 80},
                                                        "candidate_role": {"type": "string", "maxLength": 40},
                                                        "comparison_group": {"type": "string", "maxLength": 80},
                                                        "transform_instance": {"type": "string", "maxLength": 160},
                                                        "target_slice": {"type": "string", "maxLength": 160},
                                                        "hypothesis": {"type": "string", "maxLength": 360},
                                                        "expected_effects": {"type": "object"},
                                                        "evaluation_plan": {"type": "string", "maxLength": 240},
                                                        "intervention": _intervention_schema(),
                                                    },
                                                    "required": ["transform_family", "candidate_role", "hypothesis", "intervention"],
                                                },
                                            },
                                        },
                                        "required": ["experiment_id", "mechanism_class", "hypothesis", "candidates"],
                                    },
                                },
                            },
                            "required": ["experiments"],
                        },
                    }
                },
                input=prompt_input,
                max_output_tokens=3500,
            )
            self.last_call_diagnostics = {
                "component": "proposer",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": approximate_prompt_tokens(prompt_input),
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
        except Exception as exc:
            self.last_call_diagnostics = {
                "component": "proposer",
                "prompt_chars": len(prompt_input) if "prompt_input" in locals() else None,
                "prompt_approx_tokens": approximate_prompt_tokens(prompt_input) if "prompt_input" in locals() else None,
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Optimizer proposer failed: {exc}") from exc
        self._last_raw_output_text = response.output_text
        self._last_raw_candidate_count = 0
        self._last_parse_invalid_reasons = Counter()
        self._last_parse_invalid_candidate_rows = []
        try:
            payload = extract_json_object(response.output_text)
        except Exception as exc:
            primary_diagnostics = self.last_call_diagnostics or {}
            repair_started_at = time.perf_counter()
            try:
                repair_response = self._client.create_response(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "ratchet_patch_proposals_repair",
                            "strict": False,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "target_considerations": {"type": "array"},
                                    "experiments": {"type": "array"},
                                },
                                "required": ["experiments"],
                            },
                        }
                    },
                    input=(
                        "The previous proposer response was invalid JSON. "
                        "Return only a valid JSON object with target_considerations and experiments. "
                        "Preserve the intended experiment groups and candidate patches where possible; do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:9000]}"
                    ),
                    max_output_tokens=3500,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="proposer",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                payload = extract_json_object(repair_response.output_text)
                self._last_raw_output_text = repair_response.output_text
            except Exception as repair_exc:
                self.last_call_diagnostics = {
                    **primary_diagnostics,
                    "component": "proposer",
                    "repair_attempted": True,
                    "repair_error": str(repair_exc),
                }
                raise OptimizerModelError(
                    f"Optimizer proposer returned invalid JSON: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        candidates: list[CandidateProposal] = []
        raw_experiments = payload.get("experiments")
        if not isinstance(raw_experiments, list):
            self._last_parse_invalid_reasons["experiments field is not an array"] += 1
            raw_experiments = []
        self._last_raw_candidate_count = sum(
            len(raw.get("candidates", []))
            for raw in raw_experiments
            if isinstance(raw, dict) and isinstance(raw.get("candidates"), list)
        )
        for experiment_index, raw_experiment in enumerate(raw_experiments, start=1):
            if not isinstance(raw_experiment, dict):
                reason = "experiment entry is not an object"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason)
                )
                continue
            try:
                experiment = ExperimentSpec.from_dict(raw_experiment, fallback_id=f"exp_{experiment_index}")
            except Exception as exc:
                reason = f"malformed experiment: {exc}"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason)
                )
                continue
            raw_candidates = raw_experiment.get("candidates")
            if not isinstance(raw_candidates, list):
                reason = "experiment candidates field is not an array"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason)
                )
                continue
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    reason = "candidate entry is not an object"
                    self._last_parse_invalid_reasons[reason] += 1
                    self._last_parse_invalid_candidate_rows.append(
                        _invalid_raw_candidate_row(raw_candidate, reason)
                    )
                    continue
                candidate_payload = {
                    **raw_candidate,
                    "experiment_id": raw_candidate.get("experiment_id") or experiment.experiment_id,
                    "mechanism_class": raw_candidate.get("mechanism_class") or experiment.mechanism,
                    "comparison_group": raw_candidate.get("comparison_group") or experiment.experiment_id,
                    "target_slice": raw_candidate.get("target_slice")
                    or (experiment.target_slices[0] if experiment.target_slices else "global"),
                }
                try:
                    candidates.append(CandidateProposal.from_dict(candidate_payload))
                except Exception as exc:
                    reason = f"malformed candidate: {exc}"
                    self._last_parse_invalid_reasons[reason] += 1
                    self._last_parse_invalid_candidate_rows.append(
                        _invalid_raw_candidate_row(raw_candidate, reason)
                    )
                    continue
        self._last_plan_audit = _audit_experiment_plan(
            raw_experiments=raw_experiments,
            parsed_candidates=candidates,
            planner_guidance=planner_guidance,
            task_theory=task_theory,
            proposal_budget=proposal_budget,
        )
        considerations = [
            {
                "target_kind": str(item.get("target_kind", "")),
                "decision": str(item.get("decision", "")),
                "rationale": str(item.get("rationale", "")),
            }
            for item in payload.get("target_considerations", [])
            if isinstance(item, dict)
        ]
        return candidates, considerations


def _patch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {"type": "string"},
                        "target": {"type": "string"},
                        "value": {
                            "anyOf": [
                                {"type": "string", "maxLength": 1600},
                                {"type": "number"},
                                {"type": "integer"},
                                {"type": "boolean"},
                                {"type": "object", "additionalProperties": True, "maxProperties": 12},
                                {"type": "array", "items": {}, "maxItems": 8},
                                {"type": "null"},
                            ]
                        },
                        "rationale": {"type": "string", "maxLength": 240},
                    },
                    "required": ["op", "target", "value"],
                },
            },
            "rationale": {"type": "string", "maxLength": 360},
            "expected_effect": {"type": "string", "maxLength": 240},
            "metadata": {"type": "object"},
        },
        "required": ["operations", "rationale", "expected_effect"],
    }


def _intervention_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["patch", "example_selection"],
            },
            "payload": {
                "type": "object",
                "properties": {
                    "patch": _patch_schema(),
                    "source_case_ids": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 160},
                        "maxItems": 8,
                    },
                    "selection_strategy": {"type": "string", "maxLength": 80},
                    "target_labels": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 120},
                        "maxItems": 12,
                    },
                    "affected_confusions": {
                        "type": "array",
                        "items": {"type": "string", "maxLength": 160},
                        "maxItems": 12,
                    },
                },
                "additionalProperties": True,
                "maxProperties": 8,
            },
        },
        "required": ["kind", "payload"],
    }


def _compact_transform_family(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name"),
        "category": row.get("category"),
        "purpose": row.get("purpose"),
        "supported_edit_kinds": row.get("supported_edit_kinds", []),
        "supported_ops": row.get("supported_ops", []),
        "activation_signals": row.get("activation_signals", [])[:6]
        if isinstance(row.get("activation_signals"), list)
        else row.get("activation_signals", []),
        "required_measurements": row.get("required_measurements", [])[:6]
        if isinstance(row.get("required_measurements"), list)
        else row.get("required_measurements", []),
        "complexity_cost": row.get("complexity_cost"),
        "parameter_contract": row.get("parameter_contract", {}),
    }


def _compact_task_theory(task_theory: TaskTheory) -> dict[str, Any]:
    row = task_theory.to_dict()
    return {
        "bottleneck_class": row.get("bottleneck_class"),
        "residual_failure_modes": list(row.get("residual_failure_modes") or [])[:8],
        "label_confusions": list(row.get("label_confusions") or [])[:8],
        "weak_slices": list(row.get("weak_slices") or [])[:12],
        "runtime_defects": row.get("runtime_defects", {}),
        "output_defects": row.get("output_defects", {}),
        "example_coverage": {
            "example_count": (row.get("example_coverage") or {}).get("example_count"),
            "weak_labels_without_examples": list(
                (row.get("example_coverage") or {}).get("weak_labels_without_examples") or []
            )[:12],
            "label_counts": _top_mapping((row.get("example_coverage") or {}).get("label_counts") or {}, limit=20),
        },
        "cost_latency_profile": row.get("cost_latency_profile", {}),
        "confidence": row.get("confidence"),
        "evidence": list(row.get("evidence") or [])[:6],
    }


def _planner_guidance(
    *,
    task_theory: TaskTheory,
    search_hypothesis: SearchHypothesis,
    proposal_budget: int,
) -> dict[str, Any]:
    active_mechanisms = _active_mechanisms(search_hypothesis)
    primary, secondary, rationale = _mechanisms_for_bottleneck(task_theory.bottleneck_class)
    primary = [mechanism for mechanism in primary if mechanism in active_mechanisms]
    secondary = [mechanism for mechanism in secondary if mechanism in active_mechanisms and mechanism not in primary]
    if not primary:
        primary = list(secondary[:2])
        secondary = secondary[2:]
    experiment_count = max(0, min(proposal_budget, 3))
    return {
        "bottleneck_class": task_theory.bottleneck_class,
        "primary_mechanisms": primary[:3],
        "secondary_mechanisms": secondary[:4],
        "active_mechanisms": active_mechanisms,
        "recommended_experiment_count": experiment_count,
        "rationale": rationale,
        "planning_rules": [
            "Each experiment should test one mechanism, not a grab bag of unrelated edits.",
            "Prefer one strong experiment with controls over many shallow single-candidate variants.",
            "Use composed candidates only when the mechanism needs multiple compatible transform families.",
            "If few-shot is proposed, use proposal-safe source_case_ids and make the target labels/slices explicit.",
        ],
    }


def _active_mechanisms(search_hypothesis: SearchHypothesis) -> list[str]:
    mechanisms: set[str] = set()
    for family in search_hypothesis.active_families:
        mechanisms.update(MECHANISMS_BY_FAMILY.get(family, set()))
    return sorted(mechanisms)


def _mechanisms_for_bottleneck(bottleneck_class: str) -> tuple[list[str], list[str], str]:
    if bottleneck_class == "runtime_or_output_defect":
        return (
            ["runtime_defect_fix", "output_contract_fix"],
            ["semantic_boundary_rewrite", "model_capability_probe"],
            "Current evidence points first at runtime/output reliability before semantic specialization.",
        )
    if bottleneck_class == "output_contract":
        return (
            ["output_contract_fix"],
            ["runtime_defect_fix", "semantic_boundary_rewrite"],
            "Invalid or contract-shaped failures should be tested as output contract fixes first.",
        )
    if bottleneck_class == "semantic_boundary_confusion":
        return (
            ["semantic_boundary_rewrite", "representative_examples", "contrastive_examples"],
            ["model_capability_probe", "ablation"],
            "Label, slice, or semantic confusion calls for boundary rewrites and example anchoring experiments.",
        )
    if bottleneck_class == "efficiency_tradeoff":
        return (
            ["efficiency_probe", "model_capability_probe"],
            ["ablation", "runtime_defect_fix"],
            "Efficiency objectives should test cost/latency mechanisms while preserving quality.",
        )
    if bottleneck_class == "no_observed_failures":
        return (
            ["efficiency_probe", "ablation"],
            ["model_capability_probe"],
            "With no observed failures, useful experiments should reduce cost, latency, or complexity.",
        )
    return (
        ["semantic_boundary_rewrite", "model_capability_probe", "representative_examples"],
        ["output_contract_fix", "ablation"],
        "General correctness gaps need semantic, capability, or example-anchoring experiments.",
    )


def _audit_experiment_plan(
    *,
    raw_experiments: list[Any],
    parsed_candidates: list[CandidateProposal],
    planner_guidance: dict[str, Any],
    task_theory: TaskTheory,
    proposal_budget: int,
) -> dict[str, Any]:
    experiment_count = sum(1 for item in raw_experiments if isinstance(item, dict))
    mechanism_counts = Counter(candidate.mechanism_class for candidate in parsed_candidates)
    role_counts = Counter(candidate.candidate_role for candidate in parsed_candidates)
    primary_mechanisms = [str(item) for item in planner_guidance.get("primary_mechanisms", [])]
    candidate_mechanisms = set(mechanism_counts)
    missing_primary = [
        mechanism for mechanism in primary_mechanisms if mechanism not in candidate_mechanisms
    ]
    warnings: list[str] = []
    if proposal_budget > 0 and experiment_count == 0:
        warnings.append("no experiments returned")
    if experiment_count > 0 and not parsed_candidates:
        warnings.append("experiments contained no parseable candidates")
    if parsed_candidates and missing_primary and len(missing_primary) == len(primary_mechanisms):
        warnings.append("no candidate used a primary mechanism from planner guidance")
    if task_theory.bottleneck_class == "semantic_boundary_confusion":
        has_examples = bool(candidate_mechanisms.intersection({"representative_examples", "contrastive_examples"}))
        has_rewrite = "semantic_boundary_rewrite" in candidate_mechanisms
        if has_examples and not has_rewrite:
            warnings.append("example experiment lacks a semantic-boundary rewrite control")
    if role_counts.get("composed", 0) and not (role_counts.get("control", 0) or role_counts.get("ablation", 0)):
        warnings.append("composed candidate lacks a control or ablation in the returned plan")
    return {
        "experiment_count": experiment_count,
        "raw_candidate_count": sum(
            len(item.get("candidates", []))
            for item in raw_experiments
            if isinstance(item, dict) and isinstance(item.get("candidates"), list)
        ),
        "parsed_candidate_count": len(parsed_candidates),
        "candidate_mechanisms": dict(sorted(mechanism_counts.items())),
        "candidate_roles": dict(sorted(role_counts.items())),
        "primary_mechanisms": primary_mechanisms,
        "missing_primary_mechanisms": missing_primary,
        "warnings": warnings,
    }


def _compact_behavior_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_field": diagnostics.get("label_field"),
        "per_label": list(diagnostics.get("per_label") or [])[:16],
        "weak_labels": list(diagnostics.get("weak_labels") or [])[:12],
        "confusions": list(diagnostics.get("confusions") or [])[:8],
        "overpredicted_labels": list(diagnostics.get("overpredicted_labels") or [])[:8],
        "invalid_output_case_ids": list(diagnostics.get("invalid_output_case_ids") or [])[:8],
        "runtime_reliability": diagnostics.get("runtime_reliability", {}),
        "category_metrics": _compact_category_metrics(diagnostics.get("category_metrics") or {}),
    }


def _compact_category_metrics(metrics: dict[str, Any], *, limit: int = 16) -> dict[str, Any]:
    rows = sorted(
        (
            (name, value)
            for name, value in metrics.items()
            if isinstance(value, dict)
        ),
        key=lambda item: (
            float(item[1].get("pass_rate", item[1].get("mean_score", 1.0)) or 1.0),
            str(item[0]),
        ),
    )
    return {str(name): value for name, value in rows[:limit]}


def _compact_diagnosis(diagnosis: FailureDiagnosis) -> dict[str, Any]:
    return {
        "case_ids": diagnosis.case_ids[:8],
        "category": diagnosis.category,
        "root_cause": diagnosis.root_cause[:500],
        "target_names": diagnosis.target_names[:8],
        "evidence": diagnosis.evidence[:4],
    }


def _compact_editable_target(target: EditableTarget) -> dict[str, Any]:
    current_value = target.current_value
    if isinstance(current_value, str):
        compact_value: Any = current_value[:700]
    elif isinstance(current_value, list):
        compact_value = {"type": "list", "count": len(current_value), "sample": current_value[:2]}
    elif isinstance(current_value, dict):
        compact_value = {"type": "object", "keys": sorted(str(key) for key in current_value.keys())[:16]}
    else:
        compact_value = current_value
    return {
        "name": target.name,
        "kind": target.kind,
        "path": target.path,
        "current_value": compact_value,
        "allowed_ops": list(target.allowed_ops),
        "description": target.description[:240],
        "choices": list(target.choices),
        "max_chars": target.max_chars,
        "value_schema": dict(target.value_schema),
    }


def _compact_proposal_example_bank(
    bank: ProposalExampleBank,
    *,
    target_labels: set[str],
    max_examples: int,
    max_per_label: int,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    selected = []
    for example in sorted(
        bank.examples,
        key=lambda item: (
            0 if item.label in target_labels else 1,
            item.label or "",
            item.case_id,
        ),
    ):
        label = example.label or "unlabeled"
        if label_counts[label] >= max_per_label:
            continue
        label_counts[label] += 1
        selected.append(example)
        if len(selected) >= max_examples:
            break
    return {
        "usage": (
            "proposal-safe train examples. For targeted_few_shot, select source_case_ids only; "
            "Ratchet materializes inputs and expected outputs."
        ),
        "label_field": bank.label_field,
        "example_count": len(bank.examples),
        "included_example_count": len(selected),
        "label_counts": dict(bank.label_counts),
        "metadata_categories": _top_mapping(bank.metadata_categories, limit=20),
        "examples": [example.to_dict() for example in selected],
    }


def _top_mapping(mapping: dict[str, Any], *, limit: int) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in sorted(
            mapping.items(),
            key=lambda item: (-_numeric_value(item[1]), str(item[0])),
        )[:limit]
    }


def _numeric_value(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compact_recent_history(history: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in history[-limit:]:
        comparison = row.get("comparison_to_parent") or {}
        metrics = row.get("metrics") or {}
        patch = row.get("proposal") or {}
        rows.append(
            {
                "iteration": row.get("iteration"),
                "attempt": row.get("attempt"),
                "parent_patch_hash": row.get("parent_patch_hash"),
                "patch_hash": row.get("patch_hash"),
            "transform_family": row.get("transform_family"),
            "mechanism_class": row.get("mechanism_class"),
            "experiment_id": row.get("experiment_id"),
            "candidate_role": row.get("candidate_role"),
            "comparison_group": row.get("comparison_group"),
            "transform_instance": row.get("transform_instance"),
                "transform_parameters": _value_summary(
                    (row.get("candidate") or {}).get("transform_parameters") or row.get("transform_parameters")
                ),
                "transform_context": row.get("transform_context"),
                "target_slice": row.get("target_slice"),
                "hypothesis": row.get("hypothesis"),
                "accepted": row.get("accepted"),
                "rejection_reason": row.get("rejection_reason"),
                "score_delta": comparison.get("score_delta"),
                "cost_delta": comparison.get("cost_delta"),
                "latency_delta": comparison.get("latency_delta"),
                "pass_count": metrics.get("pass_count"),
                "case_count": metrics.get("case_count"),
                "mean_score": metrics.get("mean_score"),
                "operations": [
                    {
                        "op": operation.get("op"),
                        "target": operation.get("target"),
                        "value_summary": _value_summary(operation.get("value")),
                    }
                    for operation in patch.get("operations", [])
                    if isinstance(operation, dict)
                ],
            }
        )
    return rows


def _proposal_prompt_input(prompt: dict[str, Any], *, proposal_budget: int) -> str:
    return (
        f"{PROPOSER_INSTRUCTIONS} proposal_budget={proposal_budget}. "
        "Allowed mechanisms: runtime_defect_fix, output_contract_fix, semantic_boundary_rewrite, "
        "representative_examples, contrastive_examples, model_capability_probe, efficiency_probe, ablation.\n\n"
        f"{json.dumps(prompt, separators=(',', ':'), default=str)}"
    )


def approximate_prompt_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def prompt_size_profile(text: str) -> dict[str, int]:
    return {
        "chars": len(text),
        "approx_tokens": approximate_prompt_tokens(text),
    }


def _compact_patch(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "operations": [
            {
                "op": operation.get("op"),
                "target": operation.get("target"),
                "value_summary": _value_summary(operation.get("value")),
            }
            for operation in patch.get("operations", [])
            if isinstance(operation, dict)
        ],
        "rationale": str(patch.get("rationale") or "")[:240],
        "expected_effect": str(patch.get("expected_effect") or "")[:240],
    }


def _target_labels_for_examples(behavior_diagnostics: dict[str, Any]) -> set[str]:
    labels = {str(label) for label in behavior_diagnostics.get("weak_labels", []) if label}
    for row in behavior_diagnostics.get("confusions", []):
        if not isinstance(row, dict):
            continue
        if row.get("expected"):
            labels.add(str(row["expected"]))
        if row.get("actual"):
            labels.add(str(row["actual"]))
    return labels


def _invalid_candidate_row(
    candidate: CandidateProposal,
    reason: str,
    *,
    materialization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "proposal_patch_hash": patch_hash(candidate.patch),
        "proposal": candidate.patch.to_dict(),
        "candidate": candidate.to_dict(),
        "transform_family": candidate.transform_family,
        "transform_instance": candidate.transform_instance,
        "transform_parameters": candidate.transform_parameters,
        "target_slice": candidate.target_slice,
        "hypothesis": candidate.hypothesis,
        "evaluation_plan": candidate.evaluation_plan,
        "materialization": materialization or {},
        "scheduled": False,
        "valid": False,
        "invalid_reason": reason,
    }


def _invalid_raw_candidate_row(raw_candidate: Any, reason: str) -> dict[str, Any]:
    return {
        "proposal_patch_hash": None,
        "proposal": {},
        "candidate": {},
        "raw_candidate": _value_summary(raw_candidate),
        "transform_family": None,
        "transform_instance": None,
        "transform_parameters": {},
        "target_slice": None,
        "hypothesis": "",
        "evaluation_plan": "",
        "materialization": {},
        "scheduled": False,
        "valid": False,
        "invalid_reason": reason,
    }


def _family_budget_quotas(
    budget_allocation: dict[str, float],
    *,
    proposal_budget: int,
) -> dict[str, int]:
    if proposal_budget <= 0:
        return {family: 0 for family in budget_allocation}
    positive = {
        family: max(0.0, float(share))
        for family, share in budget_allocation.items()
        if float(share) > 0.0
    }
    if not positive:
        return {}
    total = sum(positive.values())
    scaled = {
        family: (share / total) * proposal_budget
        for family, share in positive.items()
    }
    quotas = {family: int(value) for family, value in scaled.items()}
    remaining = proposal_budget - sum(quotas.values())
    ranked_remainders = sorted(
        scaled.items(),
        key=lambda item: (-(item[1] - int(item[1])), item[0]),
    )
    for family, _ in ranked_remainders[:remaining]:
        quotas[family] += 1
    return quotas


def _candidate_budget_group(candidate: CandidateProposal) -> str:
    comparison_group = candidate.comparison_group or candidate.experiment_id or "default"
    target_slice = candidate.target_slice or "global"
    return "|".join(
        [
            candidate.transform_family,
            str(comparison_group),
            str(target_slice),
        ]
    )


def _targeted_few_shot_reference_error(candidate: CandidateProposal) -> str | None:
    if candidate.transform_family != "targeted_few_shot":
        return None
    if candidate.intervention.kind != "example_selection":
        return "targeted_few_shot must use example_selection intervention"
    if candidate.patch.operations:
        return "targeted_few_shot must use example_selection intervention, not inline add_few_shot values"
    return None


def _materialize_candidate_references(
    candidate: CandidateProposal,
    proposal_example_bank: ProposalExampleBank | None,
) -> tuple[CandidateProposal, dict[str, Any]]:
    if proposal_example_bank is None:
        if candidate.transform_family == "targeted_few_shot":
            return (
                candidate,
                {
                    "type": "few_shot_reference_expansion",
                    "materialized": False,
                    "error": "targeted_few_shot requires a proposal example bank",
                },
            )
        return candidate, {}
    example_by_id = {example.case_id: example for example in proposal_example_bank.examples}
    operations: list[PatchOperation] = []
    changed = False
    materialized_rows: list[dict[str, Any]] = []
    transform_parameters = dict(candidate.transform_parameters)
    raw_parameter_source_ids = _example_selection_source_ids(candidate)
    parameter_source_ids = (
        [str(item) for item in raw_parameter_source_ids if isinstance(item, str) and item]
        if isinstance(raw_parameter_source_ids, list)
        else []
    )
    if candidate.transform_family == "targeted_few_shot" and parameter_source_ids:
        unknown_source_ids = [source_id for source_id in parameter_source_ids if source_id not in example_by_id]
        if unknown_source_ids:
            return (
                candidate,
                {
                    "type": "few_shot_reference_expansion",
                    "materialized": False,
                    "source_case_ids": parameter_source_ids,
                    "unknown_source_case_ids": unknown_source_ids,
                    "error": "unknown few-shot source_case_ids: " + ", ".join(unknown_source_ids[:6]),
                },
            )
    candidate_operations = list(candidate.patch.operations)
    if candidate.transform_family == "targeted_few_shot" and not candidate_operations and parameter_source_ids:
        candidate_operations = [
            PatchOperation(
                op="add_few_shot",
                target="few_shot",
                value=[{"source_case_id": source_id} for source_id in parameter_source_ids],
                rationale="Materialize proposer-selected train examples.",
            )
        ]
    for operation in candidate_operations:
        if operation.op != "add_few_shot":
            operations.append(operation)
            continue
        raw_items = operation.value if isinstance(operation.value, list) else [operation.value]
        source_ids = _few_shot_source_ids(raw_items) or parameter_source_ids
        if not source_ids:
            operations.append(operation)
            continue
        materialized_items: list[dict[str, Any]] = []
        for index, source_id in enumerate(source_ids):
            raw_item = raw_items[index] if index < len(raw_items) and isinstance(raw_items[index], dict) else {}
            example = example_by_id.get(source_id)
            if example is None:
                materialized_items.append(dict(raw_item, source_case_id=source_id))
                continue
            item = {
                "source_case_id": source_id,
                "input": example.input,
                "output": example.expected,
                "purpose": str(raw_item.get("purpose") or candidate.hypothesis or "proposal-selected train example")[:240],
            }
            materialized_items.append(item)
            materialized_rows.append({"source_case_id": source_id, "label": example.label})
        operations.append(
            PatchOperation(
                op=operation.op,
                target=operation.target,
                value=materialized_items,
                rationale=operation.rationale,
            )
        )
        changed = True
        transform_parameters["source_case_ids"] = source_ids
    if not changed:
        return candidate, {}
    patch = AgentPatch(
        operations=operations,
        rationale=candidate.patch.rationale,
        expected_effect=candidate.patch.expected_effect,
        metadata={
            **candidate.patch.metadata,
            "materialized_few_shot": True,
            "few_shot_source_case_ids": [row["source_case_id"] for row in materialized_rows],
        },
    )
    return (
        CandidateProposal(
            patch=patch,
            transform_family=candidate.transform_family,
            intervention=candidate.intervention,
            transform_instance=candidate.transform_instance,
            transform_parameters=transform_parameters,
            mechanism_class=candidate.mechanism_class,
            experiment_id=candidate.experiment_id,
            candidate_role=candidate.candidate_role,
            comparison_group=candidate.comparison_group,
            target_slice=candidate.target_slice,
            hypothesis=candidate.hypothesis,
            expected_effects=dict(candidate.expected_effects),
            evaluation_plan=candidate.evaluation_plan,
        ),
        {
            "type": "few_shot_reference_expansion",
            "materialized": True,
            "source_case_ids": [row["source_case_id"] for row in materialized_rows],
            "source_labels": [row["label"] for row in materialized_rows],
            "raw_patch": candidate.patch.to_dict(),
        },
    )


def _few_shot_count_variants(candidate: CandidateProposal) -> list[CandidateProposal]:
    if candidate.transform_family != "targeted_few_shot":
        return [candidate]
    operations = list(candidate.patch.operations)
    few_shot_indexes = [
        index
        for index, operation in enumerate(operations)
        if operation.op == "add_few_shot" and isinstance(operation.value, list)
    ]
    if not few_shot_indexes:
        return [candidate]
    op_index = few_shot_indexes[0]
    operation = operations[op_index]
    examples = list(operation.value)
    if len(examples) <= 1:
        return [_annotated_few_shot_variant(candidate, keep_count=len(examples), original_count=len(examples))]
    variants: list[CandidateProposal] = []
    for keep_count in range(1, min(3, len(examples)) + 1):
        reduced_operations = list(operations)
        reduced_operations[op_index] = PatchOperation(
            op=operation.op,
            target=operation.target,
            value=examples[:keep_count],
            rationale=operation.rationale,
        )
        variants.append(
            _annotated_few_shot_variant(
                CandidateProposal(
                    patch=AgentPatch(
                        operations=reduced_operations,
                        rationale=candidate.patch.rationale,
                        expected_effect=candidate.patch.expected_effect,
                        metadata=dict(candidate.patch.metadata),
                    ),
                    transform_family=candidate.transform_family,
                    intervention=candidate.intervention,
                    transform_instance=candidate.transform_instance,
                    transform_parameters=dict(candidate.transform_parameters),
                    mechanism_class=candidate.mechanism_class,
                    experiment_id=candidate.experiment_id,
                    candidate_role=candidate.candidate_role,
                    comparison_group=candidate.comparison_group,
                    target_slice=candidate.target_slice,
                    hypothesis=candidate.hypothesis,
                    expected_effects=dict(candidate.expected_effects),
                    evaluation_plan=candidate.evaluation_plan,
                ),
                keep_count=keep_count,
                original_count=len(examples),
            )
        )
    return variants


def _annotated_few_shot_variant(
    candidate: CandidateProposal,
    *,
    keep_count: int,
    original_count: int,
) -> CandidateProposal:
    source_ids = _candidate_few_shot_source_ids(candidate)
    kept_source_ids = source_ids[:keep_count] if source_ids else []
    strategy = str(candidate.transform_parameters.get("selection_strategy") or "unspecified")
    intervention_payload = dict(candidate.intervention.payload)
    if kept_source_ids:
        intervention_payload["source_case_ids"] = kept_source_ids
    transform_parameters = {
        **candidate.transform_parameters,
        "source_case_ids": kept_source_ids or candidate.transform_parameters.get("source_case_ids", []),
        "few_shot_example_count": keep_count,
        "original_few_shot_example_count": original_count,
        "selection_strategy": strategy,
    }
    patch = AgentPatch(
        operations=list(candidate.patch.operations),
        rationale=candidate.patch.rationale,
        expected_effect=candidate.patch.expected_effect,
        metadata={
            **candidate.patch.metadata,
            "few_shot_source_case_ids": kept_source_ids,
            "few_shot_variant": {
                "example_count": keep_count,
                "original_example_count": original_count,
                "selection_strategy": strategy,
                "source_case_ids": kept_source_ids,
            },
        },
    )
    return CandidateProposal(
        patch=patch,
        transform_family=candidate.transform_family,
        intervention=Intervention(kind=candidate.intervention.kind, payload=intervention_payload),
        transform_instance=f"{candidate.transform_instance or 'few_shot'}_{keep_count}_shot",
        transform_parameters=transform_parameters,
        mechanism_class=candidate.mechanism_class,
        experiment_id=candidate.experiment_id,
        candidate_role="compression" if keep_count < original_count else candidate.candidate_role,
        comparison_group=candidate.comparison_group,
        target_slice=candidate.target_slice,
        hypothesis=candidate.hypothesis,
        expected_effects={
            **candidate.expected_effects,
            "few_shot_example_count": keep_count,
            "selection_strategy": strategy,
        },
        evaluation_plan=candidate.evaluation_plan,
    )


def _candidate_few_shot_source_ids(candidate: CandidateProposal) -> list[str]:
    for operation in candidate.patch.operations:
        if operation.op == "add_few_shot" and isinstance(operation.value, list):
            return _few_shot_source_ids(operation.value)
    raw = _example_selection_source_ids(candidate)
    if isinstance(raw, list):
        return [str(item) for item in raw if isinstance(item, str) and item]
    return []


def _example_selection_source_ids(candidate: CandidateProposal) -> Any:
    if candidate.intervention.kind == "example_selection":
        return candidate.intervention.payload.get("source_case_ids", [])
    return []


def _few_shot_variant_materialization(
    candidate: CandidateProposal,
    materialization: dict[str, Any],
) -> dict[str, Any]:
    if candidate.transform_family != "targeted_few_shot":
        return materialization
    variant = candidate.patch.metadata.get("few_shot_variant")
    if not isinstance(variant, dict):
        return materialization
    rows = dict(materialization)
    source_ids = variant.get("source_case_ids")
    if isinstance(source_ids, list):
        rows["source_case_ids"] = [str(item) for item in source_ids]
    rows["few_shot_variant"] = dict(variant)
    return rows


def _few_shot_source_ids(raw_items: list[Any]) -> list[str]:
    rows: list[str] = []
    for item in raw_items:
        if isinstance(item, dict) and isinstance(item.get("source_case_id"), str) and item["source_case_id"]:
            rows.append(item["source_case_id"])
    return rows


def _value_summary(value: Any) -> Any:
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, list):
        return {"type": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value.keys())[:12]}
    return value
