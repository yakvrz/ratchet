from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.affordances import OptimizationAffordance, generate_optimization_affordances, validate_candidate_affordances
from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.errors import OptimizerModelError
from ratchet.experiments import CANDIDATE_ROLES, ExperimentIntent, ExperimentSpec, TaskTheory, build_task_theory
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
PROPOSER_MAX_OUTPUT_TOKENS = 8000
PROPOSER_INSTRUCTIONS = (
    "You are Ratchet's task-agnostic candidate implementer. Return JSON with experiments[] and optional "
    "target_considerations[]. Keep text concise. Implement experiment_intents exactly: they define the "
    "research questions, mechanisms, target slices, controls, and measurements. Treat task_theory.experiment_opportunities "
    "as supporting evidence only; they are not patch recipes. Each candidate must name an active "
    "transform_family, a candidate_role in atomic/composed/control/ablation/compression, a hypothesis, "
    "and one intervention. Normal candidates use intervention.kind='patch' with payload.patch.operations; "
    "few-shot candidates use only intervention.kind='example_selection' with payload.source_case_ids from "
    "proposal_example_bank. Do not inline few-shot examples. Each candidate must cite affordance_ids from "
    "optimization_affordances. Patch operations must use editable_targets, "
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


class CandidateImplementer:
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
        experiment_intents: list[ExperimentIntent] | None = None,
        affordances: list[OptimizationAffordance] | None = None,
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
        active_affordances = list(affordances or generate_optimization_affordances(surface, active_families=search_hypothesis.active_families))
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
            experiment_intents=experiment_intents or [],
            affordances=active_affordances,
        )
        proposals.extend(llm_proposals)
        analysis_parts.append("Candidate implementer returned transform candidate proposals.")
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
            candidate = materialized_candidate
            family_error = validate_candidate_transform(
                candidate,
                surface=surface,
                search_hypothesis=search_hypothesis,
            )
            if family_error is not None:
                invalid_reasons[family_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, family_error, materialization=materialization))
                continue
            affordance_error = validate_candidate_affordances(
                affordance_ids=candidate.affordance_ids,
                transform_family=candidate.transform_family,
                mechanism_class=candidate.mechanism_class,
                operations=[
                    {"op": operation.op, "target": operation.target}
                    for operation in candidate.patch.operations
                ],
                affordances=active_affordances,
            )
            if affordance_error is not None:
                invalid_reasons[affordance_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, affordance_error, materialization=materialization))
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
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, reason, materialization=materialization))
                continue
            digest = patch_hash(compose_patches(summary.patch, candidate.patch))
            if digest in seen_hashes or digest in local_seen:
                invalid_reasons["duplicate patch"] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, "duplicate patch", materialization=materialization))
                continue
            local_seen.add(digest)
            group_valid.append((candidate, materialization, digest))
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
                        "affordance_ids": list(candidate.affordance_ids),
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
            analysis_parts.append("Validated transform candidate implementations.")
        else:
            analysis_parts.append("No valid transform candidate implementations.")
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
        experiment_intents: list[ExperimentIntent],
        affordances: list[OptimizationAffordance],
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
        prompt = {
            "objective": objective.to_dict(),
            "proposal_budget": proposal_budget,
            "target_kinds": target_kinds,
            "transform_library": active_family_rows,
            "search_hypothesis": _compact_search_hypothesis(search_hypothesis),
            "task_theory": _compact_task_theory(task_theory),
            "experiment_intents": [_compact_experiment_intent(intent) for intent in experiment_intents],
            "optimization_affordances": [_compact_affordance(affordance) for affordance in affordances],
            "proposal_policy": {
                "experiment_intents": (
                    "If experiment_intents is non-empty, every returned experiment_id must exactly match one "
                    "intent_id. Each candidate must cite affordance_ids from that intent and from optimization_affordances."
                ),
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
            "diagnoses": [_compact_diagnosis(diagnosis) for diagnosis in diagnoses[:3]],
            "editable_targets": [_compact_editable_target(target) for target in surface],
            "diagnostic_only_examples": {
                "usage": (
                    "dev examples for diagnosis only. Do not copy their case IDs, inputs, or expected outputs into patches."
                    if not objective.constraints.sanitize_examples
                    else "dev failure metadata for diagnosis only. Raw input, expected, output, notes, and raw_output_text fields are redacted."
                ),
                "sanitized": objective.constraints.sanitize_examples,
                "examples": summary.failed_examples(
                    limit=2,
                    max_text_chars=180,
                    sanitize_text=objective.constraints.sanitize_examples,
                ),
            },
            "proposal_example_bank": (
                _compact_proposal_example_bank(
                    proposal_example_bank,
                    target_labels=_target_labels_for_examples(compact_diagnostics),
                    max_examples=4,
                    max_per_label=1,
                )
                if proposal_example_bank is not None
                else {
                    "usage": "no proposal-safe train examples available",
                    "example_count": 0,
                    "examples": [],
                }
            ),
            "recent_history": _compact_recent_history(history, limit=3),
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
                                            "candidate_roles": {
                                                "type": "array",
                                                "items": {"type": "string", "enum": sorted(CANDIDATE_ROLES)},
                                            },
                                            "candidates": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "transform_family": {"type": "string", "maxLength": 80},
                                                        "mechanism_class": {"type": "string", "maxLength": 80},
                                                        "candidate_role": {"type": "string", "enum": sorted(CANDIDATE_ROLES)},
                                                        "comparison_group": {"type": "string", "maxLength": 80},
                                                        "transform_instance": {"type": "string", "maxLength": 160},
                                                        "target_slice": {"type": "string", "maxLength": 160},
                                                        "hypothesis": {"type": "string", "maxLength": 360},
                                                        "expected_effects": {"type": "object"},
                                                        "evaluation_plan": {"type": "string", "maxLength": 240},
                                                        "affordance_ids": {
                                                            "type": "array",
                                                            "items": {"type": "string", "maxLength": 80},
                                                        },
                                                        "intervention": _intervention_schema(),
                                                    },
                                                    "required": ["transform_family", "candidate_role", "hypothesis", "affordance_ids", "intervention"],
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
                max_output_tokens=PROPOSER_MAX_OUTPUT_TOKENS,
            )
            self.last_call_diagnostics = {
                "component": "candidate_implementer",
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
                "component": "candidate_implementer",
                "prompt_chars": len(prompt_input) if "prompt_input" in locals() else None,
                "prompt_approx_tokens": approximate_prompt_tokens(prompt_input) if "prompt_input" in locals() else None,
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Candidate implementer failed: {exc}") from exc
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
                        "The previous candidate-implementer response was invalid JSON. "
                        "Return only a valid JSON object with target_considerations and experiments. "
                        "Preserve the intended experiment groups and candidate patches where possible; do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:9000]}"
                    ),
                    max_output_tokens=PROPOSER_MAX_OUTPUT_TOKENS,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="candidate_implementer",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                payload = extract_json_object(repair_response.output_text)
                self._last_raw_output_text = repair_response.output_text
            except Exception as repair_exc:
                self.last_call_diagnostics = {
                    **primary_diagnostics,
                    "component": "candidate_implementer",
                    "repair_attempted": True,
                    "repair_error": str(repair_exc),
                }
                raise OptimizerModelError(
                    f"Candidate implementer returned invalid JSON: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        candidates: list[CandidateProposal] = []
        intent_by_id = {intent.intent_id: intent for intent in experiment_intents}
        intent_ids = set(intent_by_id)
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
                experiment = ExperimentSpec.from_dict(raw_experiment)
            except Exception as exc:
                reason = f"malformed experiment: {exc}"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason)
                )
                continue
            if intent_ids and experiment.experiment_id not in intent_ids:
                reason = f"experiment_id {experiment.experiment_id!r} does not match any requested experiment_intent"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason)
                )
                continue
            intent = intent_by_id.get(experiment.experiment_id)
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
                    candidate = CandidateProposal.from_dict(candidate_payload)
                except Exception as exc:
                    reason = f"malformed candidate: {exc}"
                    self._last_parse_invalid_reasons[reason] += 1
                    self._last_parse_invalid_candidate_rows.append(
                        _invalid_raw_candidate_row(raw_candidate, reason)
                    )
                    continue
                if intent is not None and intent.allowed_families and candidate.transform_family not in set(intent.allowed_families):
                    reason = (
                        f"candidate family {candidate.transform_family!r} is not allowed by experiment intent "
                        f"{intent.intent_id!r}"
                    )
                    self._last_parse_invalid_reasons[reason] += 1
                    self._last_parse_invalid_candidate_rows.append(
                        _invalid_raw_candidate_row(raw_candidate, reason)
                    )
                    continue
                if intent is not None and intent.affordance_ids:
                    unknown_for_intent = sorted(set(candidate.affordance_ids) - set(intent.affordance_ids))
                    if unknown_for_intent:
                        reason = (
                            f"candidate affordance_ids {unknown_for_intent} are not allowed by experiment intent "
                            f"{intent.intent_id!r}"
                        )
                        self._last_parse_invalid_reasons[reason] += 1
                        self._last_parse_invalid_candidate_rows.append(
                            _invalid_raw_candidate_row(raw_candidate, reason)
                        )
                        continue
                candidates.append(candidate)
        self._last_plan_audit = _audit_experiment_plan(
            raw_experiments=raw_experiments,
            parsed_candidates=candidates,
            experiment_intents=experiment_intents,
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
        "purpose": str(row.get("purpose") or "")[:180],
        "supported_edit_kinds": row.get("supported_edit_kinds", []),
        "supported_ops": row.get("supported_ops", []),
        "complexity_cost": row.get("complexity_cost"),
    }


def _compact_search_hypothesis(search_hypothesis: SearchHypothesis) -> dict[str, Any]:
    row = search_hypothesis.to_prompt_dict(
        max_contexts_per_family=1,
        max_constrained_contexts=2,
    )
    return {
        "family_states": {
            name: {
                "state": value.get("state"),
                "suitability": value.get("suitability"),
                "budget_share": value.get("budget_share"),
                "constraints": list(value.get("constraints") or [])[:3],
            }
            for name, value in (row.get("family_states") or {}).items()
            if isinstance(value, dict)
        },
        "active_families": list(row.get("active_families") or []),
        "active_contexts": [
            _compact_context_prompt_row(context)
            for context in list(row.get("active_contexts") or [])[:5]
            if isinstance(context, dict)
        ],
        "constrained_or_paused_contexts": [
            _compact_context_prompt_row(context)
            for context in list(row.get("constrained_or_paused_contexts") or [])[:3]
            if isinstance(context, dict)
        ],
        "target_slices": list(row.get("target_slices") or [])[:6],
        "profile": row.get("profile", {}),
        "budget_allocation": row.get("budget_allocation", {}),
        "rationale": str(row.get("rationale") or "")[:240],
    }


def _compact_context_prompt_row(row: dict[str, Any]) -> dict[str, Any]:
    key = row.get("key") if isinstance(row.get("key"), dict) else {}
    return {
        "family": key.get("family") or row.get("family"),
        "targets": list(key.get("target_names") or row.get("target_names") or [])[:3],
        "ops": list(key.get("ops") or row.get("ops") or [])[:3],
        "target_slice": key.get("target_slice") or row.get("target_slice"),
        "state": row.get("state"),
        "suitability": row.get("suitability"),
        "constraints": list(row.get("constraints") or [])[:3],
    }


def _compact_task_theory(task_theory: TaskTheory) -> dict[str, Any]:
    row = task_theory.to_dict()
    return {
        "bottleneck_class": row.get("bottleneck_class"),
        "residual_failure_modes": list(row.get("residual_failure_modes") or [])[:6],
        "label_confusions": list(row.get("label_confusions") or [])[:4],
        "weak_slices": list(row.get("weak_slices") or [])[:6],
        "runtime_defects": row.get("runtime_defects", {}),
        "output_defects": row.get("output_defects", {}),
        "example_coverage": {
            "example_count": (row.get("example_coverage") or {}).get("example_count"),
            "weak_labels_without_examples": list(
                (row.get("example_coverage") or {}).get("weak_labels_without_examples") or []
            )[:8],
            "target_label_source_case_ids": {
                str(label): list(case_ids)[:4]
                for label, case_ids in ((row.get("example_coverage") or {}).get("target_label_source_case_ids") or {}).items()
                if isinstance(case_ids, list)
            },
            "label_counts": _top_mapping((row.get("example_coverage") or {}).get("label_counts") or {}, limit=8),
        },
        "cost_latency_profile": row.get("cost_latency_profile", {}),
        "confidence": row.get("confidence"),
        "evidence": list(row.get("evidence") or [])[:4],
        "experiment_opportunity_mechanisms": [
            str(item.get("mechanism_class"))
            for item in list(row.get("experiment_opportunities") or [])[:5]
            if isinstance(item, dict) and item.get("mechanism_class")
        ],
    }


def _compact_experiment_intent(intent: ExperimentIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "mechanism_class": intent.mechanism_class,
        "hypothesis": intent.hypothesis[:360],
        "target_slices": list(intent.target_slices)[:5],
        "candidate_roles": list(intent.candidate_roles)[:5],
        "measurements": list(intent.measurements)[:5],
        "allowed_families": list(intent.allowed_families)[:5],
        "affordance_ids": list(intent.affordance_ids)[:8],
        "success_criteria": intent.success_criteria[:240],
        "disconfirming_result": intent.disconfirming_result[:240],
        "priority": intent.priority,
    }


def _compact_affordance(affordance: OptimizationAffordance) -> dict[str, Any]:
    return {
        "affordance_id": affordance.affordance_id,
        "target_name": affordance.target_name,
        "target_kind": affordance.target_kind,
        "transform_family": affordance.transform_family,
        "mechanism_class": affordance.mechanism_class,
        "allowed_ops": list(affordance.allowed_ops),
        "value_schema": affordance.value_schema,
        "expected_cost_impact": affordance.expected_cost_impact,
        "expected_latency_impact": affordance.expected_latency_impact,
        "risk_level": affordance.risk_level,
        "required_measurements": list(affordance.required_measurements)[:5],
    }


def _audit_experiment_plan(
    *,
    raw_experiments: list[Any],
    parsed_candidates: list[CandidateProposal],
    experiment_intents: list[ExperimentIntent],
    task_theory: TaskTheory,
    proposal_budget: int,
) -> dict[str, Any]:
    experiment_count = sum(1 for item in raw_experiments if isinstance(item, dict))
    mechanism_counts = Counter(candidate.mechanism_class for candidate in parsed_candidates)
    role_counts = Counter(candidate.candidate_role for candidate in parsed_candidates)
    primary_mechanisms = [intent.mechanism_class for intent in experiment_intents]
    opportunity_mechanisms = [
        str(item.get("mechanism_class"))
        for item in task_theory.experiment_opportunities
        if isinstance(item, dict) and item.get("mechanism_class")
    ]
    candidate_mechanisms = set(mechanism_counts)
    requested_intent_ids = {intent.intent_id for intent in experiment_intents}
    returned_intent_ids = {
        str(item.get("experiment_id") or "")
        for item in raw_experiments
        if isinstance(item, dict)
    }
    intent_by_id = {intent.intent_id: intent for intent in experiment_intents}
    mechanism_mismatch_ids = sorted(
        str(item.get("experiment_id") or "")
        for item in raw_experiments
        if isinstance(item, dict)
        and str(item.get("experiment_id") or "") in intent_by_id
        and str(item.get("mechanism_class") or "")
        and str(item.get("mechanism_class") or "")
        != intent_by_id[str(item.get("experiment_id") or "")].mechanism_class
    )
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
    missing_opportunities = [
        mechanism for mechanism in opportunity_mechanisms[:2] if mechanism not in candidate_mechanisms
    ]
    if parsed_candidates and missing_opportunities and len(missing_opportunities) == min(2, len(opportunity_mechanisms)):
        warnings.append("no candidate directly tested a top experiment opportunity")
    missing_intents = sorted(requested_intent_ids - returned_intent_ids)
    if requested_intent_ids and not (requested_intent_ids & returned_intent_ids):
        warnings.append("candidate implementer did not return any requested experiment intent IDs")
    if mechanism_mismatch_ids:
        warnings.append("returned experiment mechanism differed from requested intent mechanism")
    if task_theory.bottleneck_class == "semantic_boundary_confusion":
        has_examples = bool(candidate_mechanisms.intersection({"representative_examples", "contrastive_examples"}))
        has_rewrite = "semantic_boundary_rewrite" in candidate_mechanisms
        if has_examples and not has_rewrite:
            warnings.append("example experiment lacks a semantic-boundary rewrite control")
        if _semantic_opportunity_has_examples(task_theory) and has_rewrite and not has_examples:
            warnings.append("semantic-boundary plan did not test available example anchoring")
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
        "requested_intent_ids": sorted(requested_intent_ids),
        "returned_intent_ids": sorted(item for item in returned_intent_ids if item),
        "missing_intent_ids": missing_intents,
        "mechanism_mismatch_intent_ids": mechanism_mismatch_ids,
        "primary_mechanisms": primary_mechanisms,
        "opportunity_mechanisms": opportunity_mechanisms,
        "missing_primary_mechanisms": missing_primary,
        "warnings": warnings,
    }


def _semantic_opportunity_has_examples(task_theory: TaskTheory) -> bool:
    for row in task_theory.experiment_opportunities:
        if row.get("mechanism_class") != "semantic_boundary_rewrite":
            continue
        source_ids = row.get("source_case_ids_by_label")
        if isinstance(source_ids, dict) and any(source_ids.values()):
            return True
    return False


def _compact_behavior_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_field": diagnostics.get("label_field"),
        "per_label": _compact_per_label(list(diagnostics.get("per_label") or [])[:8]),
        "weak_labels": list(diagnostics.get("weak_labels") or [])[:8],
        "confusions": list(diagnostics.get("confusions") or [])[:6],
        "overpredicted_labels": list(diagnostics.get("overpredicted_labels") or [])[:5],
        "invalid_output_case_ids": list(diagnostics.get("invalid_output_case_ids") or [])[:8],
        "runtime_reliability": _compact_runtime_reliability(diagnostics.get("runtime_reliability") or {}),
        "category_metrics": _compact_category_metrics(diagnostics.get("category_metrics") or {}, limit=4),
    }


def _compact_per_label(rows: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        compact.append(
            {
                "label": row.get("label"),
                "support": row.get("support"),
                "pass_count": row.get("pass_count"),
                "pass_rate": row.get("pass_rate"),
                "case_ids": list(row.get("case_ids") or [])[:4],
            }
        )
    return compact


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
    return {
        str(name): {
            "pass_rate": value.get("pass_rate"),
            "pass_count": value.get("pass_count"),
            "case_count": value.get("case_count"),
            "mean_score": value.get("mean_score"),
        }
        for name, value in rows[:limit]
    }


def _compact_runtime_reliability(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "finish_reason_counts": row.get("finish_reason_counts", {}),
        "length_finish_case_ids": list(row.get("length_finish_case_ids") or [])[:4],
        "parser_fallback_case_ids": list(row.get("parser_fallback_case_ids") or [])[:4],
        "low_output_token_length_case_ids": list(row.get("low_output_token_length_case_ids") or [])[:4],
    }


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
        compact_value: Any = current_value[:160]
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
        "description": target.description[:90],
        "choices": list(target.choices)[:8],
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
        "label_counts": _top_mapping(bank.label_counts, limit=12),
        "metadata_categories": _top_mapping(bank.metadata_categories, limit=12),
        "examples": [_compact_proposal_example(example) for example in selected],
    }


def _compact_proposal_example(example: Any) -> dict[str, Any]:
    row = example.to_dict()
    return {
        "case_id": row.get("case_id"),
        "input": _value_summary(row.get("input")),
        "expected": row.get("expected"),
        "label": row.get("label"),
        "metadata": row.get("metadata"),
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
                rationale="Materialize implementer-selected train examples.",
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
            "few_shot_example_count": len(materialized_rows),
        },
    )
    transform_parameters["few_shot_example_count"] = len(materialized_rows)
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
            affordance_ids=list(candidate.affordance_ids),
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


def _example_selection_source_ids(candidate: CandidateProposal) -> Any:
    if candidate.intervention.kind == "example_selection":
        return candidate.intervention.payload.get("source_case_ids", [])
    return []


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
