from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.affordances import OptimizationAffordance, generate_optimization_affordances, validate_candidate_applications
from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.errors import OptimizerModelError
from ratchet.experiments import (
    CANDIDATE_ROLES,
    EvidencePacket,
    ExperimentIntent,
    ExperimentSpec,
    ResearchTheory,
    TaskTheory,
    build_evidence_packet,
    build_task_theory,
)
from ratchet.io import extract_json_object, transform_program_hash
from ratchet.model_client import (
    ResponsesModelClient,
    combine_response_diagnostics,
    error_response_diagnostics,
    response_diagnostics,
)
from ratchet.results import CandidateSummary
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformPatch, TransformProgram
from ratchet.transforms import (
    CandidateProposal,
    Intervention,
    SearchHypothesis,
    build_search_hypothesis,
    transform_registry,
    validate_candidate_transform,
)
from ratchet.types import AgentSpec, FailureDiagnosis, OptimizationObjective
from ratchet.types import EvalCase


MAX_PROPOSALS_PER_ITERATION = 8
PROPOSER_MAX_OUTPUT_TOKENS = 8000
PROPOSER_INSTRUCTIONS = (
    "You are Ratchet's task-agnostic candidate implementer. Return JSON with experiments[] and optional "
    "affordance_considerations[]. Keep text concise. Implement experiment_intents exactly: they define the "
    "research questions, mechanisms, target slices, controls, and measurements. Treat research_theory "
    "as the causal context for implementation; opportunities are not candidate recipes. Each candidate must include "
    "a typed transform program under candidate.program and applications[] citing relevant optimization_affordances. "
    "A candidate without program.patches[] is invalid. Programs must use the transform DSL hook/op schema, not legacy candidate operations. "
    "For add_context_section or replace_context_section, emit one focused patch with non-empty content containing the actual rendered instruction or context data; prefer a concise string. Do not emit empty content objects, repeated placeholder patches, or context text in value. "
    "Every set_model_config candidate needs field and value; every define_state candidate needs field, type, and initial. Use selection.source_case_ids "
    "only for few-shot examples from proposal_example_bank. Do not inline few-shot examples. Family, mechanism, "
    "measurements, and risks are derived from cited affordances; do not emit candidate-level transform_family, "
    "mechanism_class, affordance_ids, patch, or intervention fields. "
    "Do not copy diagnostic_only_examples into candidate values; only proposal-safe train examples may be copied, "
    "and only through source_case_id. Prefer minimal, independently evaluable candidates. For cost/latency modes, "
    "preserve correctness and explore model/runtime/tool efficiency even when failures are absent. "
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
    affordance_considerations: list[dict[str, Any]] | None = None
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
            "affordance_considerations": list(self.affordance_considerations or []),
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
        summary: CandidateSummary,
        surface: SurfaceSpec,
        *,
        objective: OptimizationObjective,
        seen_hashes: set[str],
        current_spec: AgentSpec | None,
        history: list[dict[str, Any]],
        search_hypothesis: SearchHypothesis | None = None,
        diagnosis: FailureDiagnosis | None = None,
        diagnoses: list[FailureDiagnosis] | None = None,
        task_theory: TaskTheory | None = None,
        research_theory: ResearchTheory | None = None,
        evidence_packet: EvidencePacket | None = None,
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
        if evidence_packet is None:
            evidence_packet = build_evidence_packet(
                summary=summary,
                diagnoses=diagnosis_context,
                objective=objective,
                proposal_example_bank=proposal_example_bank,
            )
        if task_theory is None:
            task_theory = build_task_theory(
                summary=summary,
                diagnoses=diagnosis_context,
                objective=objective,
                proposal_example_bank=proposal_example_bank,
            )
        if research_theory is None:
            research_theory = _legacy_research_theory_from_task_theory(task_theory)
        active_affordances = list(affordances or generate_optimization_affordances(
            surface,
            objective=objective,
            active_families=search_hypothesis.active_families,
            evidence=_affordance_evidence(evidence_packet, diagnosis_context),
        ))
        llm_proposals, affordance_considerations = self._llm_proposals(
            summary,
            surface,
            objective=objective,
            diagnoses=diagnosis_context,
            history=history,
            search_hypothesis=search_hypothesis,
            research_theory=research_theory,
            evidence_packet=evidence_packet,
            proposal_example_bank=proposal_example_bank,
            proposal_budget=proposal_budget,
            experiment_intents=experiment_intents or [],
            affordances=active_affordances,
        )
        proposals.extend(llm_proposals)
        analysis_parts.append("Candidate implementer returned transform candidate proposals.")
        invalid_reasons.update(self._last_parse_invalid_reasons)
        raw_count = self._last_raw_candidate_count
        invalid_candidate_rows = list(self._last_parse_invalid_candidate_rows)
        budget_valid, candidate_rows, validation_invalid_rows, validation_invalid_reasons = self._validate_candidate_proposals(
            proposals,
            surface=surface,
            search_hypothesis=search_hypothesis,
            active_affordances=active_affordances,
            seen_hashes=seen_hashes,
            proposal_example_bank=proposal_example_bank,
        )
        invalid_candidate_rows.extend(validation_invalid_rows)
        invalid_reasons.update(validation_invalid_reasons)
        if not budget_valid and invalid_candidate_rows and proposal_budget > 0:
            repair_feedback = _repair_feedback_rows(invalid_candidate_rows)
            repaired_proposals, repaired_considerations = self._llm_proposals(
                summary,
                surface,
                objective=objective,
                diagnoses=diagnosis_context,
                history=history,
                search_hypothesis=search_hypothesis,
                research_theory=research_theory,
                evidence_packet=evidence_packet,
                proposal_example_bank=proposal_example_bank,
                proposal_budget=proposal_budget,
                experiment_intents=experiment_intents or [],
                affordances=active_affordances,
                repair_feedback=repair_feedback,
            )
            raw_count += self._last_raw_candidate_count
            affordance_considerations.extend(repaired_considerations)
            invalid_reasons.update(self._last_parse_invalid_reasons)
            invalid_candidate_rows.extend(self._last_parse_invalid_candidate_rows)
            repaired_valid, repaired_rows, repaired_invalid_rows, repaired_invalid_reasons = self._validate_candidate_proposals(
                repaired_proposals,
                surface=surface,
                search_hypothesis=search_hypothesis,
                active_affordances=active_affordances,
                seen_hashes=seen_hashes,
                proposal_example_bank=proposal_example_bank,
            )
            if repaired_valid:
                analysis_parts.append("Repaired invalid transform candidates using compiler feedback.")
            budget_valid.extend(repaired_valid)
            candidate_rows.extend(repaired_rows)
            invalid_candidate_rows.extend(repaired_invalid_rows)
            invalid_reasons.update(repaired_invalid_reasons)
        self.last_candidate_rows = candidate_rows
        self.last_invalid_candidate_rows = invalid_candidate_rows
        self.last_stats = ProposalStats(
            raw_count=raw_count,
            valid_count=len(budget_valid),
            returned_count=len(budget_valid),
            invalid_count=sum(count for reason, count in invalid_reasons.items() if reason != "duplicate candidate"),
            duplicate_count=invalid_reasons.get("duplicate candidate", 0),
            error=None,
            invalid_reasons=dict(sorted(invalid_reasons.items())),
            affordance_considerations=affordance_considerations,
            plan_audit=self._last_plan_audit,
            raw_output_text=self._last_raw_output_text,
            call_diagnostics=self.last_call_diagnostics,
        )
        if budget_valid:
            analysis_parts.append("Validated transform candidate implementations.")
        else:
            analysis_parts.append("No valid transform candidate implementations.")
        analysis_parts.append(
            "Proposal counts: "
            f"raw={self.last_stats.raw_count}, valid={self.last_stats.valid_count}, "
            f"returned={self.last_stats.returned_count}, invalid={self.last_stats.invalid_count}, "
            f"duplicate={self.last_stats.duplicate_count}."
        )
        return budget_valid, " ".join(analysis_parts)

    def _validate_candidate_proposals(
        self,
        proposals: list[CandidateProposal],
        *,
        surface: SurfaceSpec,
        search_hypothesis: SearchHypothesis,
        active_affordances: list[OptimizationAffordance],
        seen_hashes: set[str],
        proposal_example_bank: ProposalExampleBank | None,
    ) -> tuple[list[CandidateProposal], list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
        valid: list[CandidateProposal] = []
        candidate_rows: list[dict[str, Any]] = []
        invalid_candidate_rows: list[dict[str, Any]] = []
        invalid_reasons: Counter[str] = Counter()
        local_seen: set[str] = set()
        group_count = 0
        group_indices: dict[str, int] = {}
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
            affordance_error = validate_candidate_applications(
                applications=candidate.applications,
                affordances=active_affordances,
            )
            if affordance_error is not None:
                invalid_reasons[affordance_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, affordance_error, materialization=materialization))
                continue
            compile_report = TransformCompiler().compile(candidate.program, surface).report
            if compile_report.status != "compiled":
                issue = compile_report.rejection
                reason = (
                    f"transform compile rejected candidate: {issue.code}: {issue.message}"
                    if issue is not None
                    else "transform compile rejected candidate"
                )
                invalid_reasons[reason] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, reason, materialization=materialization))
                continue
            digest = transform_program_hash(candidate.program)
            if digest in seen_hashes or digest in local_seen:
                invalid_reasons["duplicate transform program"] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, "duplicate transform program", materialization=materialization))
                continue
            local_seen.add(digest)
            budget_group = _candidate_budget_group(candidate)
            if budget_group not in group_indices:
                group_count += 1
                group_indices[budget_group] = group_count
            valid.append(candidate)
            candidate_rows.append(
                {
                    "rank": len(candidate_rows) + 1,
                    "proposal_group": group_indices[budget_group],
                    "variant_rank": 1,
                    "proposal_program_hash": digest,
                    "candidate_id": digest,
                    "proposal": candidate.program.to_dict(),
                    "candidate": candidate.to_dict(),
                    "applications": [application.to_dict() for application in candidate.applications],
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
                    "materialization": materialization,
                }
            )
        return valid, candidate_rows, invalid_candidate_rows, invalid_reasons

    def _llm_proposals(
        self,
        summary: CandidateSummary,
        surface: SurfaceSpec,
        *,
        objective: OptimizationObjective,
        diagnoses: list[FailureDiagnosis],
        history: list[dict[str, Any]],
        search_hypothesis: SearchHypothesis,
        research_theory: ResearchTheory,
        evidence_packet: EvidencePacket,
        proposal_example_bank: ProposalExampleBank | None,
        proposal_budget: int,
        experiment_intents: list[ExperimentIntent],
        affordances: list[OptimizationAffordance],
        repair_feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[list[CandidateProposal], list[dict[str, Any]]]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        self._last_plan_audit = {}
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
            "transform_library": active_family_rows,
            "surface_spec": _compact_surface_spec(surface),
            "search_hypothesis": _compact_search_hypothesis(search_hypothesis),
            "research_theory": _research_theory_prompt_view(research_theory),
            "task_theory": _research_theory_prompt_view(research_theory),
            "evidence_packet": _compact_evidence_packet(evidence_packet),
            "experiment_intents": [_compact_experiment_intent(intent) for intent in experiment_intents],
            "optimization_affordances": [_compact_affordance(affordance) for affordance in affordances],
            "proposal_policy": {
                "experiment_intents": (
                    "If experiment_intents is non-empty, every returned experiment_id must exactly match one "
                    "intent_id. Each candidate must cite affordance_ids from that intent and from optimization_affordances."
                ),
                "empty_patches_allowed": (
                    "Only when no listed optimization affordance can plausibly improve the objective without violating constraints."
                ),
                "cost_or_latency_without_failures": (
                    "If correctness is currently saturated and the objective is cost or latency, still propose minimal "
                    "efficiency candidates from the affordance surface so the eval loop can validate the tradeoff."
                ),
                "candidate_portfolio": (
                    "Generate an ordered portfolio of distinct, independently evaluable candidates up to proposal_budget. "
                    "Do not let prompt edits crowd out other plausible target kinds; rank by expected objective impact "
                    "and constraint risk."
                ),
                "context_patch_content": (
                    "For context rewrites, content must be the complete replacement or inserted text. "
                    "A placeholder object like {} is invalid; repeated identical context patches are invalid."
                ),
                "repair_feedback": (
                    "When repair_feedback is non-empty, return repaired versions of those candidates only. "
                    "Fix the compiler or parser errors structurally; do not repeat invalid empty content fields."
                ),
            },
            "repair_feedback": {
                "usage": "previous candidate programs failed validation; repair them instead of creating unrelated candidates",
                "invalid_candidates": list(repair_feedback or []),
            },
            "current_candidate": summary.candidate.to_dict() if summary.candidate is not None else None,
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
                        "name": "ratchet_candidate_proposals",
                        "strict": False,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "affordance_considerations": {
                                    "type": "array",
                                    "maxItems": max(len(affordances), 1),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "affordance_id": {"type": "string", "maxLength": 180},
                                            "decision": {"type": "string", "maxLength": 40},
                                            "rationale": {"type": "string", "maxLength": 280},
                                        },
                                        "required": ["affordance_id", "decision", "rationale"],
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
                                                        "candidate_role": {"type": "string", "enum": sorted(CANDIDATE_ROLES)},
                                                        "comparison_group": {"type": "string", "maxLength": 80},
                                                        "target_slice": {"type": "string", "maxLength": 160},
                                                        "hypothesis": {"type": "string", "maxLength": 360},
                                                        "expected_effects": {"type": "object"},
                                                        "evaluation_plan": {"type": "string", "maxLength": 240},
                                                        "program": _program_schema(),
                                                        "patches": {
                                                            "type": "array",
                                                            "items": _transform_patch_schema(),
                                                            "minItems": 1,
                                                            "maxItems": 12,
                                                        },
                                                        "applications": {
                                                            "type": "array",
                                                            "minItems": 1,
                                                            "maxItems": 3,
                                                            "items": _application_schema(),
                                                        },
                                                    },
                                                    "required": ["candidate_role", "hypothesis", "applications", "program"],
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
                            "name": "ratchet_candidate_proposals_repair",
                            "strict": False,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "affordance_considerations": {"type": "array"},
                                    "experiments": {"type": "array"},
                                },
                                "required": ["experiments"],
                            },
                        }
                    },
                    input=(
                        "The previous candidate-implementer response was invalid JSON. "
                        "Return only a valid JSON object with affordance_considerations and experiments. "
                        "Preserve the intended experiment groups and candidate programs where possible; do not add prose.\n\n"
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
            research_theory=research_theory,
            proposal_budget=proposal_budget,
        )
        considerations = [
            {
                "affordance_id": str(item.get("affordance_id", "")),
                "decision": str(item.get("decision", "")),
                "rationale": str(item.get("rationale", "")),
            }
            for item in payload.get("affordance_considerations", [])
            if isinstance(item, dict)
        ]
        return candidates, considerations


def _application_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "affordance_id": {"type": "string", "maxLength": 180},
            "selection": {
                "type": "object",
                "properties": {
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
                },
            },
            "rationale": {"type": "string", "maxLength": 240},
        },
        "required": ["affordance_id"],
    }


def _program_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "maxLength": 120},
            "hypothesis_id": {"type": "string", "maxLength": 120},
            "metadata": {"type": "object", "additionalProperties": True, "maxProperties": 12},
            "patches": {
                "type": "array",
                "items": _transform_patch_schema(),
                "minItems": 1,
                "maxItems": 12,
            },
        },
        "required": ["patches"],
    }


def _transform_patch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "hook": {
                "type": "string",
                "enum": [
                    "on_task_start",
                    "after_user_message",
                    "before_model_call",
                    "after_model_call",
                    "before_tool_call",
                    "after_tool_result",
                    "on_tool_error",
                    "before_user_response",
                    "on_task_end",
                ],
            },
            "op": {"type": "string", "maxLength": 80},
            "section": {"type": "string", "maxLength": 160},
            "field": {"type": "string", "maxLength": 160},
            "target": {"type": "string", "maxLength": 160},
            "tool": {"type": "string", "maxLength": 160},
            "position": {"type": "string", "maxLength": 160},
            "content": {
                "anyOf": [
                    {"type": "string", "minLength": 1, "maxLength": 2400},
                    {"type": "object", "minProperties": 1, "additionalProperties": True},
                    {"type": "array", "minItems": 1, "maxItems": 8, "items": {}},
                    {"type": "number"},
                    {"type": "boolean"},
                ]
            },
            "value": {},
            "initial": {},
            "type": {"type": "string", "maxLength": 160},
            "checks": {"type": "array", "items": {}, "maxItems": 12},
            "on_fail": {"type": "object", "additionalProperties": True},
            "when": {"type": "object", "additionalProperties": True},
            "unless": {"type": "object", "additionalProperties": True},
        },
        "required": ["op"],
        "additionalProperties": True,
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


def _research_theory_prompt_view(research_theory: ResearchTheory) -> dict[str, Any]:
    row = research_theory.to_dict()
    return {
        "theory_id": row.get("theory_id"),
        "summary": str(row.get("summary") or "")[:900],
        "primary_hypothesis_id": row.get("primary_hypothesis_id"),
        "hypotheses": [
            {
                "hypothesis_id": item.get("hypothesis_id"),
                "statement": str(item.get("statement") or "")[:700],
                "mechanism_class": item.get("mechanism_class"),
                "target_slices": list(item.get("target_slices") or [])[:6],
                "supporting_evidence": list(item.get("supporting_evidence") or [])[:5],
                "competing_evidence": list(item.get("competing_evidence") or [])[:4],
                "disconfirming_result": str(item.get("disconfirming_result") or "")[:320],
                "confidence": item.get("confidence"),
            }
            for item in list(row.get("hypotheses") or [])[:5]
            if isinstance(item, dict)
        ],
        "experiment_opportunities": [
            {
                "opportunity_id": item.get("opportunity_id"),
                "hypothesis_ids": list(item.get("hypothesis_ids") or [])[:4],
                "mechanism_class": item.get("mechanism_class"),
                "target_slices": list(item.get("target_slices") or [])[:6],
                "rationale": str(item.get("rationale") or "")[:600],
                "measurements": list(item.get("measurements") or [])[:6],
                "disconfirming_result": str(item.get("disconfirming_result") or "")[:320],
                "candidate_roles": list(item.get("candidate_roles") or [])[:5],
                "compatible_mechanisms": list(item.get("compatible_mechanisms") or [])[:5],
                "affordance_ids": list(item.get("affordance_ids") or [])[:10],
                "priority": item.get("priority"),
            }
            for item in list(row.get("experiment_opportunities") or [])[:6]
            if isinstance(item, dict)
        ],
        "experiment_opportunity_mechanisms": [
            str(item.get("mechanism_class"))
            for item in list(row.get("experiment_opportunities") or [])[:6]
            if isinstance(item, dict) and item.get("mechanism_class")
        ],
        "disconfirmed_explanations": list(row.get("disconfirmed_explanations") or [])[:5],
        "surprising_observations": list(row.get("surprising_observations") or [])[:5],
        "prior_lessons": list(row.get("prior_lessons") or [])[:5],
        "uncertainty": str(row.get("uncertainty") or "")[:500],
        "confidence": row.get("confidence"),
    }


def _compact_evidence_packet(evidence_packet: EvidencePacket) -> dict[str, Any]:
    row = evidence_packet.to_dict()
    return {
        "residual_failure_modes": list(row.get("residual_failure_modes") or [])[:8],
        "label_confusions": list(row.get("label_confusions") or [])[:6],
        "weak_slices": list(row.get("weak_slices") or [])[:8],
        "runtime_defects": row.get("runtime_defects", {}),
        "output_defects": row.get("output_defects", {}),
        "tool_defects": row.get("tool_defects", {}),
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
        "diagnosis_categories": list(row.get("diagnosis_categories") or [])[:8],
        "confidence": row.get("confidence"),
        "evidence": list(row.get("evidence") or [])[:8],
    }


def _compact_experiment_intent(intent: ExperimentIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "mechanism_class": intent.mechanism_class,
        "hypothesis": intent.hypothesis[:360],
        "target_slices": list(intent.target_slices)[:5],
        "candidate_roles": list(intent.candidate_roles)[:5],
        "measurements": list(intent.measurements)[:5],
        "affordance_ids": list(intent.affordance_ids)[:8],
        "success_criteria": intent.success_criteria[:240],
        "disconfirming_result": intent.disconfirming_result[:240],
        "priority": intent.priority,
    }


def _compact_affordance(affordance: OptimizationAffordance) -> dict[str, Any]:
    return {
        "affordance_id": affordance.affordance_id,
        "label": affordance.label,
        "target_name": affordance.target_name,
        "target_kind": affordance.target_kind,
        "target_path": affordance.target_path,
        "family": affordance.family,
        "mechanism": affordance.mechanism,
        "ops": list(affordance.ops),
        "value_schema": affordance.value_schema,
        "semantic_role": affordance.semantic_role,
        "behavioral_axes": list(affordance.behavioral_axes)[:5],
        "expected_scope": affordance.expected_scope,
        "risk": affordance.risk,
        "measurements": list(affordance.measurements)[:5],
        "composition": affordance.composition.to_dict(),
        "suitability": affordance.suitability,
        "evidence": list(affordance.evidence)[:4],
        "budget_hint": affordance.budget_hint,
    }


def _affordance_evidence(evidence_packet: EvidencePacket, diagnoses: list[FailureDiagnosis]) -> dict[str, Any]:
    packet = evidence_packet.to_dict()
    runtime = packet.get("runtime_defects") or {}
    output = packet.get("output_defects") or {}
    tool = packet.get("tool_defects") or {}
    return {
        "bottleneck_class": ",".join(packet.get("residual_failure_modes") or []),
        "runtime_defect": bool(runtime.get("length_finish_case_ids") or runtime.get("parser_fallback_case_ids")),
        "invalid_output": bool(output.get("invalid_output_count")),
        "tool_trajectory_defect": bool(
            tool.get("tool_error_case_ids")
            or tool.get("invalid_tool_call_case_ids")
            or tool.get("premature_stop_case_ids")
            or tool.get("turn_outcome_counts")
        ),
        "example_coverage": bool((packet.get("example_coverage") or {}).get("example_count")),
        "diagnosis_target_names": sorted({target for diagnosis in diagnoses for target in diagnosis.target_names}),
    }


def _audit_experiment_plan(
    *,
    raw_experiments: list[Any],
    parsed_candidates: list[CandidateProposal],
    experiment_intents: list[ExperimentIntent],
    research_theory: ResearchTheory,
    proposal_budget: int,
) -> dict[str, Any]:
    experiment_count = sum(1 for item in raw_experiments if isinstance(item, dict))
    mechanism_counts = Counter(candidate.mechanism_class for candidate in parsed_candidates)
    role_counts = Counter(candidate.candidate_role for candidate in parsed_candidates)
    primary_mechanisms = [intent.mechanism_class for intent in experiment_intents]
    opportunity_mechanisms = [
        str(item.get("mechanism_class"))
        for item in research_theory.to_dict().get("experiment_opportunities", [])
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
    if research_theory.bottleneck_class == "semantic_boundary_rewrite":
        has_examples = bool(candidate_mechanisms.intersection({"representative_examples", "contrastive_examples"}))
        has_rewrite = "semantic_boundary_rewrite" in candidate_mechanisms
        if has_examples and not has_rewrite:
            warnings.append("example experiment lacks a semantic-boundary rewrite control")
        if _semantic_opportunity_has_examples(research_theory) and has_rewrite and not has_examples:
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


def _semantic_opportunity_has_examples(research_theory: ResearchTheory) -> bool:
    for row in research_theory.to_dict().get("experiment_opportunities", []):
        if row.get("mechanism_class") != "semantic_boundary_rewrite":
            continue
        source_ids = row.get("source_case_ids_by_label")
        if isinstance(source_ids, dict) and any(source_ids.values()):
            return True
    return False


def _legacy_research_theory_from_task_theory(task_theory: TaskTheory) -> ResearchTheory:
    opportunities = []
    hypothesis_id = "legacy_h1"
    mechanism = _legacy_mechanism_for_bottleneck(task_theory.bottleneck_class)
    for index, row in enumerate(task_theory.experiment_opportunities or [], start=1):
        if not isinstance(row, dict):
            continue
        opportunities.append(
            {
                "opportunity_id": f"legacy_opp_{index}",
                "hypothesis_ids": [hypothesis_id],
                "mechanism_class": str(row.get("mechanism_class") or mechanism),
                "target_slices": list(row.get("target_slices") or [])[:6],
                "rationale": str(row.get("rationale") or "Legacy task-theory opportunity."),
                "measurements": list(row.get("measurements") or [])[:6],
                "disconfirming_result": str(row.get("disconfirming_result") or ""),
                "candidate_roles": list(row.get("candidate_roles") or ["atomic"])[:5],
                "compatible_mechanisms": list(row.get("compatible_mechanisms") or [])[:5],
                "affordance_ids": list(row.get("affordance_ids") or [])[:10],
                "priority": index,
            }
        )
    if not opportunities:
        opportunities.append(
            {
                "opportunity_id": "legacy_opp_1",
                "hypothesis_ids": [hypothesis_id],
                "mechanism_class": mechanism,
                "target_slices": ["failed_cases"],
                "rationale": "Legacy compatibility theory from deterministic task evidence.",
                "measurements": ["score_delta", "non_target_regression"],
                "disconfirming_result": "No improvement on staged eval.",
                "candidate_roles": ["atomic"],
                "priority": 1,
            }
        )
    return ResearchTheory.from_dict(
        {
            "theory_id": "legacy_task_theory",
            "summary": "Compatibility research theory derived from legacy task-theory evidence.",
            "primary_hypothesis_id": hypothesis_id,
            "hypotheses": [
                {
                    "hypothesis_id": hypothesis_id,
                    "statement": "Legacy deterministic task theory identified the active optimization mechanism.",
                    "mechanism_class": mechanism,
                    "target_slices": list(task_theory.weak_slices[:6]) or ["failed_cases"],
                    "supporting_evidence": list(task_theory.evidence[:6]),
                    "competing_evidence": [],
                    "disconfirming_result": "No improvement on staged eval.",
                    "confidence": task_theory.confidence,
                }
            ],
            "experiment_opportunities": opportunities,
            "confidence": task_theory.confidence,
        }
    )


def _legacy_mechanism_for_bottleneck(bottleneck: str) -> str:
    if bottleneck == "runtime_or_output_defect":
        return "runtime_defect_fix"
    if bottleneck == "tool_trajectory":
        return "tool_selection_policy"
    if bottleneck == "output_contract":
        return "output_contract_fix"
    if bottleneck == "efficiency_tradeoff":
        return "efficiency_probe"
    return "semantic_boundary_rewrite"


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


def _compact_surface_spec(surface: SurfaceSpec) -> dict[str, Any]:
    return {
        "agent_id": surface.agent_id,
        "context": {
            "sections": [
                {
                    "name": section.name,
                    "role": section.role,
                    "required": section.required,
                    "editable": section.name in surface.context.editable_sections,
                    "content_shape": _value_shape(section.content),
                }
                for section in surface.context.graph.sections
            ],
            "generated_sections_allowed": surface.context.generated_sections_allowed,
            "removable_sections_allowed": surface.context.removable_sections_allowed,
            "reorderable_sections_allowed": surface.context.reorderable_sections_allowed,
        },
        "hooks": {
            name: {
                "supported": hook.supported,
                "method": hook.method,
                "available_inputs": list(hook.available_inputs),
                "allowed_ops": list(hook.allowed_ops),
            }
            for name, hook in sorted(surface.hooks.items())
            if hook.supported
        },
        "state": surface.state.to_dict(),
        "tools": {
            "tools": [tool.to_dict() for tool in surface.tools.tools],
            "tool_description_rewrite_allowed": surface.tools.tool_description_rewrite_allowed,
            "tool_call_interception_allowed": surface.tools.tool_call_interception_allowed,
            "tool_metadata_allowed": surface.tools.tool_metadata_allowed,
        },
        "model": surface.model.to_dict(),
        "response": surface.response.to_dict(),
        "immutable_boundaries": list(surface.immutable_boundaries),
        "safety_constraints": list(surface.safety_constraints),
    }


def _value_shape(value: Any) -> Any:
    if isinstance(value, str):
        return {"type": "string", "chars": len(value), "prefix": value[:180]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value), "sample": value[:2]}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value.keys())[:16]}
    return {"type": type(value).__name__, "value": value}


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
        candidate = row.get("proposal") or {}
        program = candidate.get("program") if isinstance(candidate.get("program"), dict) else {}
        rows.append(
            {
                "iteration": row.get("iteration"),
                "attempt": row.get("attempt"),
                "parent_candidate_id": row.get("parent_candidate_id"),
                "candidate_id": row.get("candidate_id"),
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
                        "hook": operation.get("hook"),
                        "section": operation.get("section"),
                        "field": operation.get("field"),
                        "target": operation.get("target"),
                        "content_summary": _value_summary(operation.get("content")),
                    }
                    for operation in program.get("patches", [])
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
    digest = transform_program_hash(candidate.program)
    return {
        "proposal_program_hash": digest,
        "proposal": candidate.program.to_dict(),
        "candidate": candidate.to_dict(),
        "applications": [application.to_dict() for application in candidate.applications],
        "transform_family": candidate.transform_family,
        "mechanism_class": candidate.mechanism_class,
        "affordance_ids": list(candidate.affordance_ids),
        "transform_instance": candidate.transform_instance,
        "transform_parameters": candidate.transform_parameters,
        "target_slice": candidate.target_slice,
        "hypothesis": candidate.hypothesis,
        "evaluation_plan": candidate.evaluation_plan,
        "materialization": materialization or {},
        "valid": False,
        "invalid_reason": reason,
    }


def _invalid_raw_candidate_row(raw_candidate: Any, reason: str) -> dict[str, Any]:
    return {
        "proposal_program_hash": None,
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
        "valid": False,
        "invalid_reason": reason,
    }


def _repair_feedback_rows(rows: list[dict[str, Any]], *, limit: int = 4, max_chars: int = 5000) -> list[dict[str, Any]]:
    feedback: list[dict[str, Any]] = []
    for row in rows[:limit]:
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        proposal = row.get("proposal") if isinstance(row.get("proposal"), dict) else {}
        applications = row.get("applications") if isinstance(row.get("applications"), list) else []
        payload = {
            "invalid_reason": row.get("invalid_reason"),
            "experiment_id": row.get("experiment_id") or candidate.get("experiment_id"),
            "candidate_role": row.get("candidate_role") or candidate.get("candidate_role"),
            "comparison_group": row.get("comparison_group") or candidate.get("comparison_group"),
            "target_slice": row.get("target_slice") or candidate.get("target_slice"),
            "hypothesis": row.get("hypothesis") or candidate.get("hypothesis"),
            "evaluation_plan": row.get("evaluation_plan") or candidate.get("evaluation_plan"),
            "applications": applications or candidate.get("applications", []),
            "program": proposal or candidate.get("program", {}),
        }
        text = json.dumps(payload, separators=(",", ":"), default=str)
        if len(text) > max_chars:
            payload["program"] = _value_summary(payload["program"])
            payload["applications"] = _value_summary(payload["applications"])
        feedback.append(payload)
    return feedback


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
    for application in candidate.applications:
        if application.family != "targeted_few_shot":
            continue
        if not application.selection:
            return "targeted_few_shot affordance applications must use selection, not inline add_few_shot values"
    return None


def _materialize_candidate_references(
    candidate: CandidateProposal,
    proposal_example_bank: ProposalExampleBank | None,
) -> tuple[CandidateProposal, dict[str, Any]]:
    if proposal_example_bank is None:
        if any(application.family == "targeted_few_shot" for application in candidate.applications):
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
    materialized_rows: list[dict[str, Any]] = []
    raw_parameter_source_ids = _example_selection_source_ids(candidate)
    parameter_source_ids = (
        [str(item) for item in raw_parameter_source_ids if isinstance(item, str) and item]
        if isinstance(raw_parameter_source_ids, list)
        else []
    )
    if parameter_source_ids:
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
    if not parameter_source_ids:
        return candidate, {}
    examples = []
    for source_id in parameter_source_ids:
        example = example_by_id[source_id]
        examples.append(
            {
                "source_case_id": source_id,
                "input": example.input,
                "output": example.expected,
                "purpose": candidate.hypothesis or "proposal-selected train example",
            }
        )
        materialized_rows.append({"source_case_id": source_id, "label": example.label})
    materialized_program = TransformProgram(
        candidate_id=candidate.program.candidate_id,
        hypothesis_id=candidate.program.hypothesis_id,
        patches=(
            *candidate.program.patches,
            TransformPatch.from_dict(
                {
                    "hook": "before_model_call",
                    "op": "add_context_section",
                    "section": "proposal_selected_examples",
                    "position": "end",
                    "content": examples,
                }
            ),
        ),
        metadata={
            **candidate.program.metadata,
            "materialized_few_shot": True,
            "few_shot_source_case_ids": [row["source_case_id"] for row in materialized_rows],
            "few_shot_example_count": len(materialized_rows),
        },
    )
    return (
        CandidateProposal(
            program=materialized_program,
            applications=list(candidate.applications),
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
            "raw_program": candidate.program.to_dict(),
        },
    )


def _example_selection_source_ids(candidate: CandidateProposal) -> Any:
    rows: list[str] = []
    for application in candidate.applications:
        raw_ids = application.selection.get("source_case_ids")
        if isinstance(raw_ids, list):
            rows.extend(str(item) for item in raw_ids if isinstance(item, str) and item)
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
