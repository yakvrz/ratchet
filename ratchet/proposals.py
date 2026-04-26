from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from typing import Any

from ratchet.errors import OptimizerModelError
from ratchet.io import patch_hash
from ratchet.model_client import ResponsesModelClient
from ratchet.patches import compose_patches
from ratchet.results import PatchSummary
from ratchet.types import AgentPatch, AgentSpec, EditableTarget, FailureDiagnosis, OptimizationObjective
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

    def propose(
        self,
        summary: PatchSummary,
        surface: list[EditableTarget],
        *,
        objective: OptimizationObjective,
        seen_hashes: set[str],
        current_spec: AgentSpec | None,
        history: list[dict[str, Any]],
        diagnosis: FailureDiagnosis | None = None,
        diagnoses: list[FailureDiagnosis] | None = None,
        proposal_budget: int = MAX_PROPOSALS_PER_ITERATION,
    ) -> tuple[list[AgentPatch], str]:
        proposals: list[AgentPatch] = []
        analysis_parts: list[str] = []
        invalid_reasons: Counter[str] = Counter()
        proposal_budget = max(0, proposal_budget)
        diagnosis_context = list(diagnoses or ([] if diagnosis is None else [diagnosis]))
        llm_proposals, target_considerations = self._llm_proposals(
            summary,
            surface,
            objective=objective,
            diagnoses=diagnosis_context,
            history=history,
            proposal_budget=proposal_budget,
        )
        proposals.extend(llm_proposals)
        analysis_parts.append("LLM proposer returned patch proposals.")
        validator = PatchValidator()
        valid: list[AgentPatch] = []
        local_seen: set[str] = set()
        candidate_rows: list[dict[str, Any]] = []
        for patch in proposals:
            is_valid, invalid_reason = validator.validate_with_reason(
                patch,
                current_spec=current_spec,
                surface=surface,
                objective=objective,
                evidence_cases=[evaluation.case for evaluation in summary.evaluations],
            )
            if not is_valid:
                invalid_reasons[invalid_reason or "invalid patch"] += 1
                continue
            digest = patch_hash(compose_patches(summary.patch, patch))
            if digest in seen_hashes or digest in local_seen:
                invalid_reasons["duplicate patch"] += 1
                continue
            local_seen.add(digest)
            valid.append(patch)
            candidate_rows.append(
                {
                    "rank": len(candidate_rows) + 1,
                    "proposal_patch_hash": patch_hash(patch),
                    "patch_hash": digest,
                    "proposal": patch.to_dict(),
                    "scheduled": len(candidate_rows) < proposal_budget,
                }
            )
        returned = valid[:proposal_budget]
        self.last_candidate_rows = candidate_rows
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
            analysis_parts.append("Validated LLM patch proposals.")
        else:
            analysis_parts.append("No valid LLM patch proposals.")
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
        proposal_budget: int,
    ) -> tuple[list[AgentPatch], list[dict[str, Any]]]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        target_kinds = sorted({target.kind for target in surface})
        prompt = {
            "objective": objective.to_dict(),
            "proposal_budget": proposal_budget,
            "target_kinds": target_kinds,
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
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "diagnoses": [diagnosis.to_dict() for diagnosis in diagnoses],
            "primary_diagnosis": diagnoses[0].to_dict() if diagnoses else None,
            "editable_targets": [target.to_dict() for target in surface],
            "failed_examples": summary.failed_examples(limit=8, max_text_chars=900),
            "recent_history": history[-10:],
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
                                            "target_kind": {"type": "string"},
                                            "decision": {"type": "string"},
                                            "rationale": {"type": "string"},
                                        },
                                        "required": ["target_kind", "decision", "rationale"],
                                    },
                                },
                                "patches": {
                                    "type": "array",
                                    "maxItems": max(proposal_budget, 0),
                                    "items": {
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
                                                                {"type": "string"},
                                                                {"type": "number"},
                                                                {"type": "integer"},
                                                                {"type": "boolean"},
                                                                {
                                                                    "type": "object",
                                                                    "additionalProperties": True,
                                                                },
                                                                {"type": "array", "items": {}},
                                                                {"type": "null"},
                                                            ]
                                                        },
                                                        "rationale": {"type": "string"},
                                                    },
                                                    "required": ["op", "target", "value"],
                                                },
                                            },
                                            "rationale": {"type": "string"},
                                            "expected_effect": {"type": "string"},
                                            "metadata": {"type": "object"},
                                        },
                                        "required": ["operations", "rationale", "expected_effect"],
                                    },
                                }
                            },
                            "required": ["patches"],
                        },
                    }
                },
                input=(
                    "You are Ratchet's patch proposer inside a task-agnostic agent optimizer. "
                    "Return JSON with a patches array and, when possible, a target_considerations array. "
                    f"The patches array may contain at most {proposal_budget} patches. "
                    "Each patch must have operations, rationale, expected_effect, and optional metadata. "
                    "Use only editable_targets listed in the prompt. Use only operations allowed by each target. "
                    "Each operation value must satisfy the target value_schema exactly: respect type, enum, and maxLength. "
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
                    "Return an empty patches array only after considering every relevant editable target and concluding that no "
                    "safe, evaluable patch exists. "
                    "Do not alter, describe, or route around the eval scorer; it is frozen. "
                    "Do not memorize examples: no literal case IDs, user inputs, private names, or expected answers may appear "
                    "in patch values unless the same text is already part of the editable target's current_value.\n\n"
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
        patches: list[AgentPatch] = []
        for raw_patch in payload.get("patches", []):
            if not isinstance(raw_patch, dict):
                continue
            try:
                patches.append(AgentPatch.from_dict(raw_patch))
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
        return patches, considerations

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in proposer response.")
        return json.loads(match.group(0))
