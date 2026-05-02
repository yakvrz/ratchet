from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.surface_opportunities import SurfaceOpportunity, generate_surface_opportunities, validate_candidate_surface_applications
from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.errors import OptimizerModelError
from ratchet.experiments import (
    CANDIDATE_ROLES,
    EvidencePacket,
    ExperimentSpec,
    SearchBrief,
    SearchPlan,
    build_evidence_packet,
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
from ratchet.transform_contract import TransformContract, build_transform_contract, transform_patch_schema_for_contract
from ratchet.transform_program import CompiledCandidate, TransformPatch, TransformProgram
from ratchet.candidates import CandidateProposal, CandidateSurfaceApplication, Intervention
from ratchet.transform_validation import (
    validate_candidate_transform,
)
from ratchet.types import AgentSpec, OptimizationObjective
from ratchet.types import EvalCase
from ratchet.capabilities import validation_check_schema


MAX_PROPOSALS_PER_ITERATION = 8
PROPOSER_MAX_OUTPUT_TOKENS = 8000
PROPOSER_INSTRUCTIONS = (
    "You are Ratchet's task-agnostic candidate implementer. Return JSON with experiments[]. "
    "Keep text concise. Implement search_plan briefs exactly: they define the research questions, target slices, "
    "measurements, and legal surface opportunities. "
    "Each candidate must include a typed transform program under candidate.program and applications[] citing "
    "relevant surface_opportunities. "
    "A candidate without program.patches[] is invalid. Programs must use the transform DSL hook/op schema, not untyped candidate operations. "
    "For add_context_section or replace_context_section, emit one focused patch with non-empty content containing the actual rendered instruction or context data; prefer a concise string. Do not emit empty content objects, repeated placeholder patches, or context text in value. "
    "Every set_model_config candidate needs field and value; for field=model_name, value must be one of the model values exposed by the cited model surface and should differ from the current model. Every define_state candidate needs field, type, and initial. Use selection.source_case_ids "
    "only for few-shot examples from proposal_example_bank; omit source_case_ids for all context, tool-loop, state, response, model, and runtime candidates. Do not inline few-shot examples. Family and mechanism "
    "labels are derived from cited surface_opportunities; do not emit candidate-level surface_mechanism, "
    "mechanism_class, surface_opportunity_ids, patch, or intervention fields. "
    "For interactive tool-loop surfaces, prefer general middleware programs: define state on_task_start, "
    "normalize or validate tool_call at before_tool_call, replan on validation failure, append real tool_result "
    "observations at after_tool_result, rewrite model-facing tool descriptions at before_model_call, and guard final responses at before_user_response. "
    "State updates must be executable: use {\"$ref\":\"tool_result.parsed...\"}, {\"$ref\":\"tool_call.args...\"}, or literal values, never prose like 'update from tool result'. "
    "Context strings may interpolate refs with {{state.field}} after the state field is defined. For inspect-before-mutate mechanisms, compose the scaffold: define a state list, append trusted identifiers after successful read/inspection tool results, expose that list in context when useful, and validate mutating tool args with {\"type\":\"tool_arg_in_state\",\"state_field\":\"...\",\"arg\":\"order_id\"} plus replan on failure. Every validate patch "
    "must use the structured executable checks[] advertised by the cited surface, e.g. {\"type\":\"args_schema_valid\"} or {\"type\":\"not_duplicate_tool_call\"}, plus an executable on_fail "
    "operation such as replan; prose-only validation content is invalid. Never rewrite tool_result, "
    "modify tool implementations, branch on task/case IDs, or use benchmark-specific domain rules. "
    "Never mention simulator control tokens, evaluator sentinels, hidden labels, gold answers, or trace-only stop markers in candidate content, even as examples of what not to do. "
    "Do not copy diagnostic_only_examples into candidate values; only proposal-safe train examples may be copied, "
    "and only through source_case_id. Do not return no-op, log-only, or control/baseline candidates; baseline comparisons are automatic measurement infrastructure. Prefer minimal, independently evaluable candidates. For cost/latency modes, "
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
    surface_opportunity_considerations: list[dict[str, Any]] | None = None
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
            "surface_opportunity_considerations": list(self.surface_opportunity_considerations or []),
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
        search_plan: SearchPlan | None = None,
        evidence_packet: EvidencePacket | None = None,
        proposal_example_bank: ProposalExampleBank | None = None,
        proposal_example_cases: tuple[EvalCase, ...] = (),
        proposal_budget: int = MAX_PROPOSALS_PER_ITERATION,
        surface_opportunities: list[SurfaceOpportunity] | None = None,
    ) -> tuple[list[CandidateProposal], str]:
        proposals: list[CandidateProposal] = []
        analysis_parts: list[str] = []
        invalid_reasons: Counter[str] = Counter()
        proposal_budget = max(0, proposal_budget)
        if evidence_packet is None:
            evidence_packet = build_evidence_packet(
                summary=summary,
                diagnoses=[],
                objective=objective,
                proposal_example_bank=proposal_example_bank,
            )
        if search_plan is None:
            raise ValueError("CandidateImplementer requires a SearchPlan from the search planner.")
        active_surface_opportunities = list(surface_opportunities or generate_surface_opportunities(
            surface,
            objective=objective,
            active_mechanisms=search_plan.active_mechanisms,
            evidence=_surface_opportunity_evidence(evidence_packet),
        ))
        transform_contract = build_transform_contract(surface, active_surface_opportunities)
        structural_proposals = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=active_surface_opportunities,
            search_plan=search_plan,
            proposal_budget=proposal_budget,
            parent_candidate=summary.candidate,
        )
        proposals.extend(structural_proposals)
        if structural_proposals:
            analysis_parts.append(
                f"Added {len(structural_proposals)} surface-derived composed scaffold candidate(s)."
            )
        remaining_proposal_budget = max(0, proposal_budget - len(structural_proposals))
        if remaining_proposal_budget:
            llm_proposals, surface_opportunity_considerations = self._llm_proposals(
                summary,
                surface,
                objective=objective,
                history=history,
                search_plan=search_plan,
                evidence_packet=evidence_packet,
                proposal_example_bank=proposal_example_bank,
                proposal_budget=remaining_proposal_budget,
                surface_opportunities=active_surface_opportunities,
                transform_contract=transform_contract,
            )
            raw_count = len(structural_proposals) + self._last_raw_candidate_count
        else:
            llm_proposals = []
            surface_opportunity_considerations = []
            self._last_raw_candidate_count = 0
            self._last_parse_invalid_reasons = Counter()
            self._last_parse_invalid_candidate_rows = []
            self._last_raw_output_text = ""
            self.last_call_diagnostics = {}
            raw_count = len(structural_proposals)
        proposals.extend(llm_proposals)
        analysis_parts.append("Candidate implementer returned transform candidate proposals.")
        invalid_reasons.update(self._last_parse_invalid_reasons)
        invalid_candidate_rows = list(self._last_parse_invalid_candidate_rows)
        budget_valid, candidate_rows, validation_invalid_rows, validation_invalid_reasons = self._validate_candidate_proposals(
            proposals,
            surface=surface,
            active_surface_opportunities=active_surface_opportunities,
            seen_hashes=seen_hashes,
            proposal_example_bank=proposal_example_bank,
        )
        invalid_candidate_rows.extend(validation_invalid_rows)
        invalid_reasons.update(validation_invalid_reasons)
        base_plan_audit = dict(self._last_plan_audit or {})
        repair_targets = _repair_targets_for_uncovered_briefs(
            invalid_candidate_rows,
            valid_candidates=budget_valid,
            search_plan=search_plan,
        )
        if repair_targets and proposal_budget > 0:
            repair_feedback = _repair_feedback_rows(repair_targets)
            repaired_proposals, repaired_considerations = self._llm_proposals(
                summary,
                surface,
                objective=objective,
                history=history,
                search_plan=search_plan,
                evidence_packet=evidence_packet,
                proposal_example_bank=proposal_example_bank,
                proposal_budget=proposal_budget,
                surface_opportunities=active_surface_opportunities,
                transform_contract=transform_contract,
                repair_feedback=repair_feedback,
            )
            raw_count += self._last_raw_candidate_count
            surface_opportunity_considerations.extend(repaired_considerations)
            invalid_reasons.update(self._last_parse_invalid_reasons)
            invalid_candidate_rows.extend(self._last_parse_invalid_candidate_rows)
            repaired_proposals, escaped_repair_rows, escaped_repair_reasons = _filter_repaired_proposals(
                repaired_proposals,
                repair_targets=repair_targets,
            )
            invalid_candidate_rows.extend(escaped_repair_rows)
            invalid_reasons.update(escaped_repair_reasons)
            repaired_valid, repaired_rows, repaired_invalid_rows, repaired_invalid_reasons = self._validate_candidate_proposals(
                repaired_proposals,
                surface=surface,
                active_surface_opportunities=active_surface_opportunities,
                seen_hashes=seen_hashes,
                proposal_example_bank=proposal_example_bank,
            )
            if repaired_valid:
                analysis_parts.append("Repaired invalid transform candidates using compiler feedback.")
            budget_valid.extend(repaired_valid)
            candidate_rows.extend(repaired_rows)
            invalid_candidate_rows.extend(repaired_invalid_rows)
            invalid_reasons.update(repaired_invalid_reasons)
        plan_audit = _coverage_audit(
            search_plan=search_plan,
            valid_rows=candidate_rows,
            invalid_rows=invalid_candidate_rows,
            base_audit=base_plan_audit,
        )
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
            surface_opportunity_considerations=surface_opportunity_considerations,
            plan_audit=plan_audit,
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
        active_surface_opportunities: list[SurfaceOpportunity],
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
            )
            if family_error is not None:
                invalid_reasons[family_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, family_error, materialization=materialization))
                continue
            surface_opportunity_error = validate_candidate_surface_applications(
                applications=candidate.applications,
                surface_opportunities=active_surface_opportunities,
            )
            if surface_opportunity_error is not None:
                invalid_reasons[surface_opportunity_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, surface_opportunity_error, materialization=materialization))
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
                    "surface_mechanism": candidate.surface_mechanism,
                    "mechanism_class": candidate.mechanism_class,
                    "experiment_id": candidate.experiment_id,
                    "candidate_role": candidate.candidate_role,
                    "comparison_group": candidate.comparison_group,
                    "surface_opportunity_ids": list(candidate.surface_opportunity_ids),
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
        history: list[dict[str, Any]],
        search_plan: SearchPlan,
        evidence_packet: EvidencePacket,
        proposal_example_bank: ProposalExampleBank | None,
        proposal_budget: int,
        surface_opportunities: list[SurfaceOpportunity],
        transform_contract: TransformContract,
        repair_feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[list[CandidateProposal], list[dict[str, Any]]]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        self._last_plan_audit = {}
        behavior_diagnostics = build_behavior_diagnostics(summary)
        compact_diagnostics = _compact_behavior_diagnostics(behavior_diagnostics)
        prompt = {
            "objective": objective.to_dict(),
            "proposal_budget": proposal_budget,
            "transform_library": {
                "language": "typed hook DSL",
                "ops_are_validated_by": "transform_compiler",
                "contract_is_derived_from": "SurfaceSpec hooks, operation rules, refs, and validation checks",
            },
            "transform_contract": transform_contract.to_dict(),
            "surface_spec": _compact_surface_spec(surface),
            "search_plan": _compact_search_plan(search_plan),
            "evidence_packet": _compact_evidence_packet(evidence_packet),
            "surface_opportunities": [_compact_surface_opportunity(surface_opportunity) for surface_opportunity in surface_opportunities],
            "proposal_policy": {
                "search_plan": (
                    "Every returned experiment_id must exactly match a search_plan brief_id. Each candidate must cite "
                    "surface_opportunity_ids from that brief and from surface_opportunities."
                ),
                "empty_patches_allowed": (
                    "Only when no listed surface opportunity can plausibly improve the objective without violating constraints."
                ),
                "cost_or_latency_without_failures": (
                    "If correctness is currently saturated and the objective is cost or latency, still propose minimal "
                    "efficiency candidates from the inferred surface so the eval loop can validate the tradeoff."
                ),
                "candidate_portfolio": (
                    "Generate an ordered portfolio of distinct, independently evaluable candidates up to proposal_budget. "
                    "Cover distinct requested mechanism_class values before returning multiple candidates for the same mechanism. "
                    "When proposal_budget is smaller than the number of requested intents, prioritize mechanism diversity first, "
                    "then expected objective impact and constraint risk. Do not let prompt edits crowd out tool-loop, state, "
                    "or response mechanisms selected by the planner."
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
                                "surface_opportunity_considerations": {
                                    "type": "array",
                                    "maxItems": max(len(surface_opportunities), 1),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "surface_opportunity_id": {"type": "string", "maxLength": 180},
                                            "decision": {"type": "string", "maxLength": 40},
                                            "rationale": {"type": "string", "maxLength": 280},
                                        },
                                        "required": ["surface_opportunity_id", "decision", "rationale"],
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
                                                        "candidate_role": {
                                                            "type": "string",
                                                            "enum": sorted(CANDIDATE_ROLES - {"control"}),
                                                        },
                                                        "comparison_group": {"type": "string", "maxLength": 80},
                                                        "target_slice": {"type": "string", "maxLength": 160},
                                                        "hypothesis": {"type": "string", "maxLength": 360},
                                                        "expected_effects": {"type": "object"},
                                                        "evaluation_plan": {"type": "string", "maxLength": 240},
                                                        "program": _program_schema(transform_contract),
                                                        "patches": {
                                                            "type": "array",
                                                            "items": _transform_patch_schema(transform_contract),
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
                                    "surface_opportunity_considerations": {"type": "array"},
                                    "experiments": {"type": "array"},
                                },
                                "required": ["experiments"],
                            },
                        }
                    },
                    input=(
                        "The previous candidate-implementer response was invalid JSON. "
                        "Return only a valid JSON object with surface_opportunity_considerations and experiments. "
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
        brief_by_id = {brief.brief_id: brief for brief in search_plan.briefs}
        brief_ids = set(brief_by_id)
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
                    _invalid_raw_candidate_row(raw_experiment, reason, raw_experiment=raw_experiment)
                )
                continue
            if brief_ids and experiment.experiment_id not in brief_ids:
                reason = f"experiment_id {experiment.experiment_id!r} does not match any search plan brief_id"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason, raw_experiment=raw_experiment)
                )
                continue
            brief = brief_by_id.get(experiment.experiment_id)
            raw_candidates = raw_experiment.get("candidates")
            if not isinstance(raw_candidates, list):
                reason = "experiment candidates field is not an array"
                self._last_parse_invalid_reasons[reason] += 1
                self._last_parse_invalid_candidate_rows.append(
                    _invalid_raw_candidate_row(raw_experiment, reason, raw_experiment=raw_experiment)
                )
                continue
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, dict):
                    reason = "candidate entry is not an object"
                    self._last_parse_invalid_reasons[reason] += 1
                    self._last_parse_invalid_candidate_rows.append(
                        _invalid_raw_candidate_row(raw_candidate, reason, experiment=experiment)
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
                        _invalid_raw_candidate_row(raw_candidate, reason, experiment=experiment, candidate_payload=candidate_payload)
                    )
                    continue
                if brief is not None and brief.surface_opportunity_ids:
                    unknown_for_brief = sorted(set(candidate.surface_opportunity_ids) - set(brief.surface_opportunity_ids))
                    if unknown_for_brief:
                        reason = (
                            f"candidate surface_opportunity_ids {unknown_for_brief} are not allowed by search brief "
                            f"{brief.brief_id!r}"
                        )
                        self._last_parse_invalid_reasons[reason] += 1
                        self._last_parse_invalid_candidate_rows.append(
                            _invalid_raw_candidate_row(raw_candidate, reason, experiment=experiment, candidate_payload=candidate_payload)
                        )
                        continue
                candidates.append(candidate)
        self._last_plan_audit = _audit_experiment_plan(
            raw_experiments=raw_experiments,
            parsed_candidates=candidates,
            search_plan=search_plan,
            proposal_budget=proposal_budget,
        )
        considerations = [
            {
                "surface_opportunity_id": str(item.get("surface_opportunity_id", "")),
                "surface_opportunity_id": str(item.get("surface_opportunity_id") or item.get("surface_opportunity_id") or ""),
                "decision": str(item.get("decision", "")),
                "rationale": str(item.get("rationale", "")),
            }
            for item in payload.get("surface_opportunity_considerations", payload.get("surface_opportunity_considerations", []))
            if isinstance(item, dict)
        ]
        return candidates, considerations


def _application_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "surface_opportunity_id": {"type": "string", "maxLength": 180},
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
        "required": ["surface_opportunity_id"],
    }


def _program_schema(transform_contract: TransformContract | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "maxLength": 120},
            "hypothesis_id": {"type": "string", "maxLength": 120},
            "metadata": {"type": "object", "additionalProperties": True, "maxProperties": 12},
            "patches": {
                "type": "array",
                "items": _transform_patch_schema(transform_contract),
                "minItems": 1,
                "maxItems": 12,
            },
        },
        "required": ["patches"],
    }


def _transform_patch_schema(transform_contract: TransformContract | None = None) -> dict[str, Any]:
    if transform_contract is not None:
        return transform_patch_schema_for_contract(transform_contract)
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
            "checks": {"type": "array", "items": validation_check_schema(), "maxItems": 12},
            "on_fail": {"type": "object", "additionalProperties": True},
            "when": {"type": "object", "additionalProperties": True},
            "unless": {"type": "object", "additionalProperties": True},
        },
        "required": ["op"],
        "additionalProperties": True,
    }


def _compact_surface_mechanism(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": row.get("name"),
        "category": row.get("category"),
        "purpose": str(row.get("purpose") or "")[:180],
        "supported_edit_kinds": row.get("supported_edit_kinds", []),
        "supported_ops": row.get("supported_ops", []),
        "complexity_cost": row.get("complexity_cost"),
    }


def _compact_search_plan(search_plan: SearchPlan) -> dict[str, Any]:
    row = search_plan.to_dict()
    return {
        "plan_id": row.get("plan_id"),
        "diagnosis": str(row.get("diagnosis") or "")[:1200],
        "hypotheses": list(row.get("hypotheses") or [])[:6],
        "target_mechanisms": list(row.get("target_mechanisms") or [])[:6],
        "active_mechanisms": list(row.get("active_mechanisms") or [])[:6],
        "briefs": [_compact_search_brief(brief) for brief in search_plan.briefs[:6]],
        "confidence": row.get("confidence"),
    }


def _compact_search_brief(brief: SearchBrief) -> dict[str, Any]:
    return {
        "brief_id": brief.brief_id,
        "mechanism_class": brief.mechanism_class,
        "hypothesis": brief.hypothesis[:600],
        "target_slices": list(brief.target_slices)[:5],
        "candidate_roles": list(brief.candidate_roles)[:5],
        "measurements": list(brief.measurements)[:5],
        "surface_opportunity_ids": list(brief.surface_opportunity_ids)[:8],
        "success_criteria": brief.success_criteria[:240],
        "disconfirming_result": brief.disconfirming_result[:240],
        "priority": brief.priority,
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


def _compact_surface_opportunity(surface_opportunity: SurfaceOpportunity) -> dict[str, Any]:
    return {
        "surface_opportunity_id": surface_opportunity.surface_opportunity_id,
        "label": surface_opportunity.label,
        "target_name": surface_opportunity.target_name,
        "target_kind": surface_opportunity.target_kind,
        "target_path": surface_opportunity.target_path,
        "ops": list(surface_opportunity.ops),
        "value_schema": surface_opportunity.value_schema,
        "semantic_role": surface_opportunity.semantic_role,
        "behavioral_axes": list(surface_opportunity.behavioral_axes)[:5],
        "expected_scope": surface_opportunity.expected_scope,
        "risk": surface_opportunity.risk,
        "measurements": list(surface_opportunity.measurements)[:5],
        "composition": surface_opportunity.composition.to_dict(),
        "suitability": surface_opportunity.suitability,
        "evidence": list(surface_opportunity.evidence)[:4],
        "budget_hint": surface_opportunity.budget_hint,
    }


def _surface_affordance_proposals(
    *,
    surface: SurfaceSpec,
    surface_opportunities: list[SurfaceOpportunity],
    search_plan: SearchPlan,
    proposal_budget: int,
    parent_candidate: CompiledCandidate | None = None,
) -> list[CandidateProposal]:
    opportunity_by_target = {item.target_name: item for item in surface_opportunities}
    primary: list[CandidateProposal] = []
    ablations: list[CandidateProposal] = []
    for affordance in surface.affordances:
        if affordance.get("kind") != "inspect_before_mutate":
            continue
        identifier = str(affordance.get("identifier") or "")
        inspected_state_field = _inspected_state_field(identifier)
        listed_state_field = _listed_state_field(identifier)
        if not identifier:
            continue
        if _parent_has_affordance(parent_candidate, kind="inspect_before_mutate", identifier=identifier):
            continue
        opportunity = opportunity_by_target.get(f"inspect_before_mutate.{identifier}")
        if opportunity is None:
            continue
        brief = _brief_for_opportunity(opportunity.surface_opportunity_id, search_plan.briefs)
        patches: list[dict[str, Any]] = [
            {
                "op": "define_state",
                "field": inspected_state_field,
                "type": "list[string]",
                "initial": [],
            }
        ]
        listed_patches: list[dict[str, Any]] = []
        for producer in list(affordance.get("produced_by") or [])[:3]:
            if not isinstance(producer, dict):
                continue
            tool = str(producer.get("tool") or "")
            ref = str(producer.get("ref") or "")
            if not tool or not ref:
                continue
            if _producer_is_inspection(producer):
                field = inspected_state_field
                target_patches = patches
            else:
                field = listed_state_field
                target_patches = listed_patches
            target_patches.append(
                {
                    "hook": "after_tool_result",
                    "op": "append_state",
                    "field": field,
                    "value": {"$ref": ref},
                    "extend": "[]" in ref,
                    "when": {"tool_call.name": tool},
                }
            )
        if listed_patches:
            patches.insert(
                1,
                {
                    "op": "define_state",
                    "field": listed_state_field,
                    "type": "list[string]",
                    "initial": [],
                },
            )
            patches.extend(listed_patches)
        if not any(
            patch.get("op") == "append_state" and patch.get("field") == inspected_state_field
            for patch in patches
        ):
            continue
        rendered_fields = [inspected_state_field]
        if listed_patches:
            rendered_fields.append(listed_state_field)
        patches.append(
            {
                "hook": "before_model_call",
                "op": "render_state_section",
                "section": "inspected_identifiers",
                "fields": rendered_fields,
                "position": "before:recent_messages",
            }
        )
        for consumer in list(affordance.get("consumed_by") or [])[:5]:
            if not isinstance(consumer, dict):
                continue
            tool = str(consumer.get("tool") or "")
            arg = str(consumer.get("arg") or "")
            if not tool or arg != identifier:
                continue
            patches.append(
                {
                    "hook": "before_tool_call",
                    "op": "validate",
                    "tool": tool,
                    "target": "tool_call",
                    "checks": [
                        {
                            "type": "tool_arg_in_state",
                            "state_field": inspected_state_field,
                            "arg": identifier,
                        }
                    ],
                    "on_fail": {
                        "op": "replan",
                        "message": (
                            f"Inspect the relevant record and observe its {identifier} through a tool result "
                            "before using this mutating tool."
                        ),
                    },
                }
            )
        if len(patches) <= 3:
            continue
        program = TransformProgram.from_dict(
            {
                "candidate_id": f"structural_inspect_before_mutate_{identifier}",
                "patches": patches[:12],
                "metadata": {
                    "source": "surface_affordance",
                    "affordance_kind": "inspect_before_mutate",
                    "identifier": identifier,
                },
            }
        )
        primary_candidate = _affordance_candidate(
            program=program,
            opportunity_id=opportunity.surface_opportunity_id,
            experiment_id=brief.brief_id if brief is not None else f"surface_affordance_{identifier}",
            candidate_role="composed",
            comparison_group=f"inspect_before_mutate.{identifier}",
            identifier=identifier,
            rationale=f"Compose inspected-{identifier} state tracking with mutating-tool validation.",
            hypothesis=(
                f"Mutating tool calls that consume {identifier} should be grounded in identifiers "
                "previously observed from inspection tool results."
            ),
        )
        primary.append(primary_candidate)
        context_ablated = [patch for patch in patches if patch.get("op") != "render_state_section"]
        if len(context_ablated) != len(patches):
            ablation_program = TransformProgram.from_dict(
                {
                    "candidate_id": f"structural_inspect_before_mutate_{identifier}_no_context",
                    "patches": context_ablated[:12],
                    "metadata": {
                        "source": "surface_affordance",
                        "affordance_kind": "inspect_before_mutate",
                        "identifier": identifier,
                        "ablation": "without_state_context_rendering",
                    },
                }
            )
            ablations.append(
                _affordance_candidate(
                    program=ablation_program,
                    opportunity_id=opportunity.surface_opportunity_id,
                    experiment_id=brief.brief_id if brief is not None else f"surface_affordance_{identifier}",
                    candidate_role="ablation",
                    comparison_group=f"inspect_before_mutate.{identifier}",
                    identifier=identifier,
                    rationale=(
                        f"Ablate model-visible {identifier} state while preserving stateful mutating-tool validation."
                    ),
                    hypothesis=(
                        f"Stateful validation for {identifier} may account for most of the benefit without "
                        "rendering inspected identifiers into model context."
                    ),
                )
            )
    capacity = max(0, proposal_budget)
    if capacity == 0:
        return []
    selected = primary[:capacity]
    remaining = capacity - len(selected)
    if remaining > 0:
        selected.extend(ablations[:remaining])
    return selected


def _affordance_candidate(
    *,
    program: TransformProgram,
    opportunity_id: str,
    experiment_id: str,
    candidate_role: str,
    comparison_group: str,
    identifier: str,
    rationale: str,
    hypothesis: str,
) -> CandidateProposal:
    return CandidateProposal(
        program=program,
        applications=[
            CandidateSurfaceApplication(
                surface_opportunity_id=opportunity_id,
                rationale=rationale,
            )
        ],
        experiment_id=experiment_id,
        candidate_role=candidate_role,
        comparison_group=comparison_group,
        target_slice="global",
        hypothesis=hypothesis,
        expected_effects={
            "score": "increase if failures involve ungrounded mutating tool calls",
            "cost": "low token overhead from state rendering" if candidate_role != "ablation" else "lower token overhead than composed scaffold",
            "latency": "low",
            "identifier": identifier,
        },
        evaluation_plan="staged_dev_then_holdout",
    )


def _brief_for_opportunity(
    surface_opportunity_id: str,
    search_briefs: list[SearchBrief],
) -> SearchBrief | None:
    for brief in search_briefs:
        if surface_opportunity_id in brief.surface_opportunity_ids:
            return brief
    for brief in search_briefs:
        if brief.mechanism_class == "surface_tool_loop":
            return brief
    return None


def _parent_has_affordance(
    parent_candidate: CompiledCandidate | None,
    *,
    kind: str,
    identifier: str,
) -> bool:
    if parent_candidate is None:
        return False
    metadata = parent_candidate.program.metadata
    if metadata.get("affordance_kind") == kind and metadata.get("identifier") == identifier:
        return True
    inspected_state_field = _inspected_state_field(identifier)
    for patch in parent_candidate.program.patches:
        if patch.op.op != "validate":
            continue
        checks = patch.op.params.get("checks")
        if not isinstance(checks, list):
            continue
        for check in checks:
            if not isinstance(check, dict):
                continue
            if check.get("type") == "tool_arg_in_state" and check.get("state_field") == inspected_state_field:
                return True
    return False


def _inspected_state_field(identifier: str) -> str:
    return f"inspected_{identifier.removesuffix('_id')}_ids"


def _listed_state_field(identifier: str) -> str:
    return f"listed_{identifier.removesuffix('_id')}_ids"


def _producer_is_inspection(producer: dict[str, Any]) -> bool:
    tool = str(producer.get("tool") or "").lower()
    path = str(producer.get("path") or "").lower()
    if tool.startswith(("get_", "inspect_", "retrieve_")):
        return True
    if tool.startswith(("list_", "search_", "find_")):
        return False
    return "[]" not in path


def _surface_opportunity_evidence(evidence_packet: EvidencePacket) -> dict[str, Any]:
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
    }


def _audit_experiment_plan(
    *,
    raw_experiments: list[Any],
    parsed_candidates: list[CandidateProposal],
    search_plan: SearchPlan,
    proposal_budget: int,
) -> dict[str, Any]:
    experiment_count = sum(1 for item in raw_experiments if isinstance(item, dict))
    mechanism_counts = Counter(candidate.mechanism_class for candidate in parsed_candidates)
    role_counts = Counter(candidate.candidate_role for candidate in parsed_candidates)
    primary_mechanisms = [brief.mechanism_class for brief in search_plan.briefs]
    candidate_mechanisms = set(mechanism_counts)
    requested_brief_ids = {brief.brief_id for brief in search_plan.briefs}
    returned_brief_ids = {
        str(item.get("experiment_id") or "")
        for item in raw_experiments
        if isinstance(item, dict)
    }
    brief_by_id = {brief.brief_id: brief for brief in search_plan.briefs}
    mechanism_mismatch_ids = sorted(
        str(item.get("experiment_id") or "")
        for item in raw_experiments
        if isinstance(item, dict)
        and str(item.get("experiment_id") or "") in brief_by_id
        and str(item.get("mechanism_class") or "")
        and str(item.get("mechanism_class") or "")
        != brief_by_id[str(item.get("experiment_id") or "")].mechanism_class
    )
    missing_primary = [
        mechanism for mechanism in primary_mechanisms if mechanism not in candidate_mechanisms
    ]
    warnings: list[str] = []
    if proposal_budget > 0 and experiment_count == 0:
        warnings.append("no experiments returned")
    if experiment_count > 0 and not parsed_candidates:
        warnings.append("experiments contained no parseable candidates")
    distinct_primary_mechanisms = set(primary_mechanisms)
    if parsed_candidates and missing_primary and len(missing_primary) == len(primary_mechanisms):
        warnings.append("no candidate used a primary mechanism from planner guidance")
    if parsed_candidates and len(distinct_primary_mechanisms) > 1 and missing_primary:
        warnings.append("candidate portfolio missed requested mechanism diversity")
    missing_briefs = sorted(requested_brief_ids - returned_brief_ids)
    if requested_brief_ids and not (requested_brief_ids & returned_brief_ids):
        warnings.append("candidate implementer did not return any requested search brief IDs")
    if mechanism_mismatch_ids:
        warnings.append("returned experiment mechanism differed from requested search brief mechanism")
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
        "requested_brief_ids": sorted(requested_brief_ids),
        "returned_brief_ids": sorted(item for item in returned_brief_ids if item),
        "missing_brief_ids": missing_briefs,
        "mechanism_mismatch_brief_ids": mechanism_mismatch_ids,
        "primary_mechanisms": primary_mechanisms,
        "missing_primary_mechanisms": missing_primary,
        "warnings": warnings,
    }


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
            "tools": [_compact_tool_spec(tool) for tool in surface.tools.tools],
            "tool_description_rewrite_allowed": surface.tools.tool_description_rewrite_allowed,
            "tool_call_interception_allowed": surface.tools.tool_call_interception_allowed,
            "tool_metadata_allowed": surface.tools.tool_metadata_allowed,
        },
        "model": surface.model.to_dict(),
        "response": surface.response.to_dict(),
        "immutable_boundaries": list(surface.immutable_boundaries),
        "safety_constraints": list(surface.safety_constraints),
        "affordances": [dict(item) for item in surface.affordances],
    }


def _compact_tool_spec(tool: Any) -> dict[str, Any]:
    schema = dict(getattr(tool, "schema", {}) or {})
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    metadata = dict(getattr(tool, "metadata", {}) or {})
    return {
        "name": tool.name,
        "description": tool.description,
        "args": {
            "properties": sorted(str(key) for key in properties),
            "required": list(schema.get("required") or []),
        },
        "metadata": {
            key: value
            for key, value in metadata.items()
            if key in {"side_effect", "risk", "result_paths", "source"}
        },
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
            "proposal-safe train examples. Surface-example candidates may cite source_case_ids; "
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
                "surface_mechanism": row.get("surface_mechanism"),
                "mechanism_class": row.get("mechanism_class"),
                "experiment_id": row.get("experiment_id"),
                "candidate_role": row.get("candidate_role"),
                "comparison_group": row.get("comparison_group"),
                "transform_instance": row.get("transform_instance"),
                "transform_parameters": _value_summary(
                    (row.get("proposal_candidate") or row.get("candidate") or {}).get("transform_parameters")
                    or row.get("transform_parameters")
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
        "Allowed mechanisms: surface_context, surface_tool_loop, surface_state, surface_response, "
        "surface_output, surface_runtime, surface_model, surface_examples.\n\n"
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
        "proposal_candidate": candidate.to_dict(),
        "candidate": candidate.to_dict(),
        "applications": [application.to_dict() for application in candidate.applications],
        "surface_mechanism": candidate.surface_mechanism,
        "mechanism_class": candidate.mechanism_class,
        "experiment_id": candidate.experiment_id,
        "candidate_role": candidate.candidate_role,
        "comparison_group": candidate.comparison_group,
        "surface_opportunity_ids": list(candidate.surface_opportunity_ids),
        "transform_instance": candidate.transform_instance,
        "transform_parameters": candidate.transform_parameters,
        "target_slice": candidate.target_slice,
        "hypothesis": candidate.hypothesis,
        "evaluation_plan": candidate.evaluation_plan,
        "materialization": materialization or {},
        "valid": False,
        "invalid_reason": reason,
    }


def _invalid_raw_candidate_row(
    raw_candidate: Any,
    reason: str,
    *,
    experiment: ExperimentSpec | None = None,
    raw_experiment: dict[str, Any] | None = None,
    candidate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidate_payload = dict(candidate_payload or {}) if isinstance(candidate_payload, dict) else {}
    raw_experiment = dict(raw_experiment or {}) if isinstance(raw_experiment, dict) else {}
    experiment_id = (
        candidate_payload.get("experiment_id")
        or (experiment.experiment_id if experiment is not None else None)
        or raw_experiment.get("experiment_id")
    )
    mechanism_class = (
        experiment.mechanism
        if experiment is not None
        else raw_experiment.get("mechanism_class") or raw_experiment.get("mechanism")
    )
    raw_applications = candidate_payload.get("applications")
    surface_opportunity_ids = [
        str(application.get("surface_opportunity_id"))
        for application in (raw_applications if isinstance(raw_applications, list) else [])
        if isinstance(application, dict) and application.get("surface_opportunity_id")
    ]
    return {
        "proposal_program_hash": None,
        "proposal": dict(candidate_payload.get("program") or {}),
        "candidate": dict(candidate_payload) if candidate_payload else {},
        "raw_candidate": _value_summary(raw_candidate),
        "surface_mechanism": None,
        "mechanism_class": str(mechanism_class or ""),
        "experiment_id": str(experiment_id or ""),
        "candidate_role": str(candidate_payload.get("candidate_role") or ""),
        "comparison_group": str(candidate_payload.get("comparison_group") or ""),
        "surface_opportunity_ids": surface_opportunity_ids,
        "transform_instance": None,
        "transform_parameters": {},
        "target_slice": candidate_payload.get("target_slice"),
        "hypothesis": str(candidate_payload.get("hypothesis") or raw_experiment.get("hypothesis") or ""),
        "evaluation_plan": str(candidate_payload.get("evaluation_plan") or ""),
        "applications": list(raw_applications or []) if isinstance(raw_applications, list) else [],
        "materialization": {},
        "valid": False,
        "invalid_reason": reason,
    }


def _repair_targets_for_uncovered_briefs(
    invalid_rows: list[dict[str, Any]],
    *,
    valid_candidates: list[CandidateProposal],
    search_plan: SearchPlan,
) -> list[dict[str, Any]]:
    if not invalid_rows:
        return []
    valid_brief_ids = {
        candidate.experiment_id
        for candidate in valid_candidates
        if candidate.experiment_id
    }
    requested_brief_ids = {brief.brief_id for brief in search_plan.briefs}
    if not valid_brief_ids:
        return invalid_rows
    targets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in invalid_rows:
        brief_id = str(row.get("experiment_id") or "")
        if brief_id not in requested_brief_ids or brief_id in valid_brief_ids:
            continue
        key = (brief_id, str(row.get("invalid_reason") or ""))
        if key in seen:
            continue
        seen.add(key)
        targets.append(row)
    return targets


def _filter_repaired_proposals(
    proposals: list[CandidateProposal],
    *,
    repair_targets: list[dict[str, Any]],
) -> tuple[list[CandidateProposal], list[dict[str, Any]], Counter[str]]:
    allowed_opportunities_by_brief: dict[str, set[str]] = {}
    for row in repair_targets:
        brief_id = str(row.get("experiment_id") or "")
        if not brief_id:
            continue
        allowed = allowed_opportunities_by_brief.setdefault(brief_id, set())
        allowed.update(str(item) for item in row.get("surface_opportunity_ids") or [] if item)
        applications = row.get("applications") if isinstance(row.get("applications"), list) else []
        for application in applications:
            if isinstance(application, dict) and application.get("surface_opportunity_id"):
                allowed.add(str(application["surface_opportunity_id"]))
    valid: list[CandidateProposal] = []
    invalid_rows: list[dict[str, Any]] = []
    invalid_reasons: Counter[str] = Counter()
    for proposal in proposals:
        allowed = allowed_opportunities_by_brief.get(proposal.experiment_id)
        if allowed is None:
            reason = "repair changed experiment_id outside requested invalid brief"
        elif allowed and not set(proposal.surface_opportunity_ids).issubset(allowed):
            reason = "repair changed surface opportunity outside requested invalid candidate"
        else:
            valid.append(proposal)
            continue
        invalid_reasons[reason] += 1
        invalid_rows.append(_invalid_candidate_row(proposal, reason))
    return valid, invalid_rows, invalid_reasons


def _coverage_audit(
    *,
    search_plan: SearchPlan,
    valid_rows: list[dict[str, Any]],
    invalid_rows: list[dict[str, Any]],
    base_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    audit = dict(base_audit or {})
    requested_brief_ids = {brief.brief_id for brief in search_plan.briefs}
    valid_brief_ids = {
        str(row.get("experiment_id"))
        for row in valid_rows
        if row.get("experiment_id")
    }
    invalid_brief_ids = {
        str(row.get("experiment_id"))
        for row in invalid_rows
        if row.get("experiment_id")
    }
    invalid_only = (requested_brief_ids & invalid_brief_ids) - valid_brief_ids
    missed = requested_brief_ids - valid_brief_ids - invalid_brief_ids
    lost_mechanisms = sorted(
        {
            brief.mechanism_class
            for brief in search_plan.briefs
            if brief.brief_id in invalid_only
        }
    )
    invalid_reasons_by_brief: dict[str, list[str]] = {}
    for row in invalid_rows:
        brief_id = str(row.get("experiment_id") or "")
        if not brief_id:
            continue
        invalid_reasons_by_brief.setdefault(brief_id, [])
        reason = str(row.get("invalid_reason") or "unknown")
        if reason not in invalid_reasons_by_brief[brief_id]:
            invalid_reasons_by_brief[brief_id].append(reason)
    warnings = list(audit.get("warnings") or [])
    if invalid_only:
        warnings.append("candidate implementer attempted mechanisms that failed transform contract validation")
    if missed:
        warnings.append("candidate implementer did not attempt some requested search briefs")
    audit.update(
        {
            "valid_covered_brief_ids": sorted(requested_brief_ids & valid_brief_ids),
            "invalid_covered_brief_ids": sorted(requested_brief_ids & invalid_brief_ids),
            "unrepaired_invalid_brief_ids": sorted(invalid_only),
            "unattempted_brief_ids": sorted(missed),
            "mechanisms_lost_to_transform_errors": lost_mechanisms,
            "invalid_reasons_by_brief": {
                key: sorted(value)
                for key, value in sorted(invalid_reasons_by_brief.items())
            },
            "warnings": sorted(set(warnings)),
        }
    )
    return audit


def _repair_feedback_rows(rows: list[dict[str, Any]], *, limit: int = 4, max_chars: int = 5000) -> list[dict[str, Any]]:
    feedback: list[dict[str, Any]] = []
    for row in rows[:limit]:
        candidate = row.get("proposal_candidate") if isinstance(row.get("proposal_candidate"), dict) else {}
        if not candidate:
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
            candidate.surface_mechanism,
            str(comparison_group),
            str(target_slice),
        ]
    )


def _materialize_candidate_references(
    candidate: CandidateProposal,
    proposal_example_bank: ProposalExampleBank | None,
) -> tuple[CandidateProposal, dict[str, Any]]:
    raw_parameter_source_ids = _example_selection_source_ids(candidate)
    parameter_source_ids = (
        [str(item) for item in raw_parameter_source_ids if isinstance(item, str) and item]
        if isinstance(raw_parameter_source_ids, list)
        else []
    )
    if proposal_example_bank is None:
        if parameter_source_ids:
            return (
                candidate,
                {
                    "type": "few_shot_reference_expansion",
                    "materialized": False,
                    "source_case_ids": parameter_source_ids,
                    "error": "surface-example source_case_ids require a proposal example bank",
                },
            )
        return candidate, {}
    example_by_id = {example.case_id: example for example in proposal_example_bank.examples}
    materialized_rows: list[dict[str, Any]] = []
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
