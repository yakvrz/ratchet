from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any

from ratchet.evidence import ProposalExampleBank, build_behavior_diagnostics
from ratchet.errors import OptimizerModelError
from ratchet.io import patch_hash
from ratchet.model_client import ResponsesModelClient
from ratchet.patches import compose_patches
from ratchet.results import PatchSummary
from ratchet.transforms import (
    CandidateProposal,
    SearchHypothesis,
    build_search_hypothesis,
    transform_registry,
    validate_candidate_transform,
)
from ratchet.types import AgentPatch, AgentSpec, EditableTarget, FailureDiagnosis, OptimizationObjective, PatchOperation
from ratchet.types import EvalCase
from ratchet.validation import PatchValidator


MAX_PROPOSALS_PER_ITERATION = 8


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
    raw_output_text: str = ""

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
            "raw_output_text": self.raw_output_text,
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
        self._last_raw_output_text = ""
        self.last_stats = ProposalStats()
        self.last_candidate_rows: list[dict[str, Any]] = []
        self.last_invalid_candidate_rows: list[dict[str, Any]] = []

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
        llm_proposals, target_considerations = self._llm_proposals(
            summary,
            surface,
            objective=objective,
            diagnoses=diagnosis_context,
            history=history,
            search_hypothesis=search_hypothesis,
            proposal_example_bank=proposal_example_bank,
            proposal_budget=proposal_budget,
        )
        proposals.extend(llm_proposals)
        analysis_parts.append("LLM proposer returned transform candidate proposals.")
        validator = PatchValidator()
        valid: list[CandidateProposal] = []
        local_seen: set[str] = set()
        candidate_rows: list[dict[str, Any]] = []
        invalid_candidate_rows: list[dict[str, Any]] = []
        for raw_candidate in proposals:
            candidate, materialization = _materialize_candidate_references(raw_candidate, proposal_example_bank)
            family_error = validate_candidate_transform(
                candidate,
                surface=surface,
                search_hypothesis=search_hypothesis,
            )
            if family_error is not None:
                invalid_reasons[family_error] += 1
                invalid_candidate_rows.append(_invalid_candidate_row(candidate, family_error, materialization=materialization))
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
            valid.append(candidate)
            candidate_rows.append(
                {
                    "rank": len(candidate_rows) + 1,
                    "proposal_patch_hash": patch_hash(candidate.patch),
                    "patch_hash": digest,
                    "proposal": candidate.patch.to_dict(),
                    "candidate": candidate.to_dict(),
                    "transform_family": candidate.transform_family,
                    "transform_instance": candidate.transform_instance,
                    "target_slice": candidate.target_slice,
                    "hypothesis": candidate.hypothesis,
                    "evaluation_plan": candidate.evaluation_plan,
                    "materialization": materialization,
                    "scheduled": len(candidate_rows) < proposal_budget,
                }
            )
        returned = valid[:proposal_budget]
        self.last_candidate_rows = candidate_rows
        self.last_invalid_candidate_rows = invalid_candidate_rows
        self.last_stats = ProposalStats(
            raw_count=len(proposals),
            valid_count=len(valid),
            returned_count=len(returned),
            invalid_count=sum(count for reason, count in invalid_reasons.items() if reason != "duplicate patch"),
            duplicate_count=invalid_reasons.get("duplicate patch", 0),
            error=None,
            invalid_reasons=dict(sorted(invalid_reasons.items())),
            target_considerations=target_considerations,
            raw_output_text=self._last_raw_output_text,
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
        proposal_example_bank: ProposalExampleBank | None,
        proposal_budget: int,
    ) -> tuple[list[CandidateProposal], list[dict[str, Any]]]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        target_kinds = sorted({target.kind for target in surface})
        registry = transform_registry()
        active_family_rows = [
            registry[name].to_dict()
            for name in search_hypothesis.active_families
            if name in registry
        ]
        behavior_diagnostics = build_behavior_diagnostics(summary)
        prompt = {
            "objective": objective.to_dict(),
            "proposal_budget": proposal_budget,
            "target_kinds": target_kinds,
            "transform_library": active_family_rows,
            "search_hypothesis": search_hypothesis.to_prompt_dict(),
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
            "current_patch": summary.patch.to_dict(),
            "behavior": {
                "mean_score": summary.mean_score,
                "pass_count": summary.pass_count,
                "pass_rate": summary.pass_rate,
                "failure_labels": summary.failure_labels,
            },
            "behavior_diagnostics": behavior_diagnostics,
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "diagnoses": [diagnosis.to_dict() for diagnosis in diagnoses],
            "primary_diagnosis": diagnoses[0].to_dict() if diagnoses else None,
            "editable_targets": [target.to_dict() for target in surface],
            "diagnostic_only_examples": {
                "usage": "dev examples for diagnosis only. Do not copy their case IDs, inputs, or expected outputs into patches.",
                "examples": summary.failed_examples(limit=8, max_text_chars=900),
            },
            "proposal_example_bank": (
                proposal_example_bank.to_prompt_dict(
                    target_labels=_target_labels_for_examples(behavior_diagnostics),
                    max_examples=24,
                )
                if proposal_example_bank is not None
                else {
                    "usage": "no proposal-safe train examples available",
                    "example_count": 0,
                    "examples": [],
                }
            ),
            "recent_history": _compact_recent_history(history, limit=10),
        }
        try:
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
                                "candidates": {
                                    "type": "array",
                                    "maxItems": max(proposal_budget, 0),
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "transform_family": {"type": "string", "maxLength": 80},
                                            "transform_instance": {"type": "string", "maxLength": 160},
                                            "transform_parameters": {
                                                "type": "object",
                                                "additionalProperties": True,
                                                "maxProperties": 8,
                                            },
                                            "target_slice": {"type": "string", "maxLength": 160},
                                            "hypothesis": {"type": "string", "maxLength": 360},
                                            "expected_effects": {"type": "object"},
                                            "evaluation_plan": {"type": "string", "maxLength": 240},
                                            "patch": _patch_schema(),
                                        },
                                        "required": ["transform_family", "hypothesis", "patch"],
                                    },
                                },
                            },
                            "required": ["candidates"],
                        },
                    }
                },
                input=(
                    "You are Ratchet's patch proposer inside a task-agnostic agent optimizer. "
                    "Return JSON with a candidates array and, when possible, a target_considerations array. "
                    f"The candidates array may contain at most {proposal_budget} candidates. "
                    "Keep all rationale, hypothesis, transform_instance, and evaluation_plan strings concise. "
                    "Do not include long examples, full transcripts, or repeated label lists in candidate fields. "
                    "Each candidate must name one active transform_family from search_hypothesis.active_families, "
                    "state a hypothesis, include one patch object, and fill transform_parameters when the active "
                    "transform family's parameter_contract requests it. Prefer the listed active_contexts. "
                    "If a family's lifecycle state is constrained, it remains eligible, but only for candidates that are "
                    "materially distinct from constrained_or_paused_contexts; use different targets, operations, slices, "
                    "or concrete mechanism classes, and explain that distinction in transform_instance and hypothesis. "
                    "Each patch must have operations, rationale, expected_effect, and optional metadata. "
                    "Use only editable_targets listed in the prompt. Use only operations allowed by each target. "
                    "The patch operations must be compatible with the declared transform_family's supported_ops and "
                    "supported_edit_kinds. "
                    "Each operation value must satisfy the target value_schema exactly: respect type, enum, and maxLength. "
                    "Few-shot operation values must be arrays of objects shaped by the few_shot target schema. "
                    "Every few-shot item must cite source_case_id from proposal_example_bank, and may copy only those "
                    "train example inputs/outputs. Do not invent few-shot examples when proposal_example_bank is empty. "
                    "Allowed operations are add_instruction, revise_instruction, add_output_constraint, "
                    "revise_tool_description, revise_tool_policy, set_retrieval_param, set_runtime_param, "
                    "change_model, add_few_shot, and add_verifier_retry. Touch at most two targets per patch. "
                    "Prefer minimal, independently evaluable patches; do not combine unrelated ideas. "
                    "Before choosing patches, inspect the generated editable target kinds and record whether each plausible "
                    "kind was proposed or skipped. Treat model changes as a normal optimizer action when a model target "
                    "is present: compare the current model against allowed alternatives and either include a change_model "
                    "patch when it plausibly improves the objective, or explain why the model target was skipped. "
                    "For correctness mode, consider every diagnosis in the prompt and propose patches that address the "
                    "largest or most actionable failure clusters while preserving cost and latency constraints. "
                    "When failed examples mention invalid_output, malformed or empty raw output, parser fallback, or output "
                    "contract labels, prefer output_contract/output instruction fixes over loosening semantic grounding or "
                    "unknown-answer policy. Only loosen grounding/fallback behavior when the evidence shows the model returned "
                    "a valid output object with a semantically over-cautious answer. "
                    "For cost or latency mode, preserve correctness within constraints and change only model, runtime, retrieval, "
                    "or similarly relevant targets when the surface allows them. In cost or latency mode, do not require failing "
                    "examples before proposing patches; saturated correctness is a reason to explore cheaper or faster variants, "
                    "not a reason to return an empty patch list. "
                    "Return an empty candidates array only after considering every active transform family and concluding "
                    "that no safe, evaluable candidate exists. "
                    "Do not alter, describe, or route around the eval scorer; it is frozen. "
                    "Do not memorize diagnostic-only examples: no literal case IDs, user inputs, private names, or expected "
                    "answers from diagnostic_only_examples may appear in patch values unless the same text is already part "
                    "of the editable target's current_value. Proposal-safe train examples are the only examples that may be "
                    "copied, and only through few-shot items with source_case_id.\n\n"
                    f"{json.dumps(prompt, indent=2, default=str)}"
                ),
                max_output_tokens=6000,
            )
        except Exception as exc:
            raise OptimizerModelError(f"Optimizer proposer failed: {exc}") from exc
        self._last_raw_output_text = response.output_text
        try:
            payload = self._extract_json_object(response.output_text)
        except Exception as exc:
            raise OptimizerModelError(f"Optimizer proposer returned invalid JSON: {exc}") from exc
        candidates: list[CandidateProposal] = []
        raw_candidates = payload.get("candidates")
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, dict):
                continue
            try:
                candidates.append(CandidateProposal.from_dict(raw_candidate))
            except Exception:
                continue
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

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        last_error: Exception | None = None
        for match in re.finditer(r"\{", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(payload, dict):
                return payload
        if last_error is not None:
            raise last_error
        raise ValueError("No JSON object found in proposer response.")


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


def _materialize_candidate_references(
    candidate: CandidateProposal,
    proposal_example_bank: ProposalExampleBank | None,
) -> tuple[CandidateProposal, dict[str, Any]]:
    if proposal_example_bank is None or not candidate.patch.operations:
        return candidate, {}
    example_by_id = {example.case_id: example for example in proposal_example_bank.examples}
    operations: list[PatchOperation] = []
    changed = False
    materialized_rows: list[dict[str, Any]] = []
    transform_parameters = dict(candidate.transform_parameters)
    raw_parameter_source_ids = transform_parameters.get("source_case_ids", [])
    parameter_source_ids = (
        [str(item) for item in raw_parameter_source_ids if isinstance(item, str) and item]
        if isinstance(raw_parameter_source_ids, list)
        else []
    )
    for operation in candidate.patch.operations:
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
        transform_parameters.setdefault("source_case_ids", source_ids)
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
            transform_instance=candidate.transform_instance,
            transform_parameters=transform_parameters,
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
