from __future__ import annotations

import json
import time
from typing import Any

from ratchet.evidence import build_behavior_diagnostics
from ratchet.errors import OptimizerModelError
from ratchet.io import extract_json_object
from ratchet.model_client import (
    ResponsesModelClient,
    combine_response_diagnostics,
    error_response_diagnostics,
    response_diagnostics,
)
from ratchet.results import PatchSummary
from ratchet.types import EditableTarget, FailureDiagnosis, OptimizationObjective


class FailureDiagnoser:
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
        self.last_call_diagnostics: dict[str, Any] | None = None

    def diagnose(
        self,
        summary: PatchSummary,
        surface: list[EditableTarget],
        objective: OptimizationObjective | None = None,
    ) -> tuple[list[FailureDiagnosis], str]:
        failed_examples = summary.failed_examples(limit=12, max_text_chars=900)
        if not failed_examples:
            self.last_call_diagnostics = None
            return [], "No failing cases on the current eval set."
        diagnoses = self._llm_diagnoses(summary, surface, failed_examples, objective or OptimizationObjective())
        if not diagnoses:
            return [], "No valid LLM diagnoses."
        diagnoses.sort(key=lambda item: (-len(item.case_ids), item.category))
        return diagnoses, "LLM diagnoser returned structured failure diagnoses."

    def _llm_diagnoses(
        self,
        summary: PatchSummary,
        surface: list[EditableTarget],
        failed_examples: list[dict[str, Any]],
        objective: OptimizationObjective,
    ) -> list[FailureDiagnosis]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "objective": objective.to_dict(),
            "current_patch": summary.patch.to_dict(),
            "behavior": {
                "mean_score": summary.mean_score,
                "pass_count": summary.pass_count,
                "pass_rate": summary.pass_rate,
                "failure_labels": summary.failure_labels,
            },
            "behavior_diagnostics": build_behavior_diagnostics(summary),
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "editable_targets": [target.to_dict() for target in surface],
            "failed_examples": failed_examples,
        }
        response_format = {
            "format": {
                "type": "json_schema",
                "name": "ratchet_failure_diagnoses",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "diagnoses": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "case_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "category": {"type": "string"},
                                    "root_cause": {"type": "string"},
                                    "target_names": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "evidence": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": True,
                                        },
                                    },
                                },
                                "required": ["case_ids", "category", "root_cause", "target_names", "evidence"],
                            },
                        }
                    },
                    "required": ["diagnoses"],
                },
            }
        }
        try:
            started_at = time.perf_counter()
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text=response_format,
                input=(
                    "You are Ratchet's failure diagnoser inside a task-agnostic agent optimizer. "
                    "Return strict JSON with exactly one top-level key, diagnoses, whose value is an array. "
                    "Each diagnosis object must include case_ids, category, root_cause, target_names, and evidence. "
                    "case_ids must come from failed_examples. target_names must come from editable_targets. "
                    "evidence must be an array of compact objects that cite case_id plus the observed output, expected signal, "
                    "behavior_diagnostics may include label confusion, weak-label, and slice summaries; use those summaries "
                    "to group related failures when they are clearer than individual rows. If labels, notes, output fields, or raw_output_text indicate "
                    "malformed output, empty output, parser fallback, or an invalid output contract, diagnose that as an "
                    "output-format/contract failure before attributing it to semantic caution or grader behavior. "
                    "Do not propose patches. Do not mention hidden evaluator or judge changes; the scorer is frozen and "
                    "outside the editable surface. Generalize root causes from evidence "
                    "without copying case-specific inputs, IDs, or expected answers into future behavior.\n\n"
                    f"{json.dumps(prompt, indent=2, default=str)}"
                ),
                max_output_tokens=5000,
            )
            self.last_call_diagnostics = {
                "component": "diagnoser",
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
        except Exception as exc:
            self.last_call_diagnostics = {
                "component": "diagnoser",
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Optimizer diagnoser failed: {exc}") from exc
        try:
            payload = extract_json_object(response.output_text)
        except Exception as exc:
            primary_diagnostics = self.last_call_diagnostics or {}
            repair_started_at = time.perf_counter()
            try:
                repair_response = self._client.create_response(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    text=response_format,
                    input=(
                        "The previous diagnoser response was invalid JSON. "
                        "Return only a valid JSON object matching the requested schema. "
                        "Preserve the intended diagnoses where possible and do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:12000]}"
                    ),
                    max_output_tokens=5000,
                )
                repair_diagnostics = {
                    **response_diagnostics(
                        repair_response,
                        model=self.model,
                        elapsed_s=time.perf_counter() - repair_started_at,
                    )
                }
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="diagnoser",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                payload = extract_json_object(repair_response.output_text)
            except Exception as repair_exc:
                self.last_call_diagnostics = {
                    **primary_diagnostics,
                    "component": "diagnoser",
                    "repair_attempted": True,
                    "repair_error": str(repair_exc),
                }
                raise OptimizerModelError(
                    f"Optimizer diagnoser returned invalid JSON: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        case_ids = {str(item["case_id"]) for item in failed_examples}
        target_names = {target.name for target in surface}
        diagnoses: list[FailureDiagnosis] = []
        for raw in payload.get("diagnoses", []):
            if not isinstance(raw, dict):
                continue
            try:
                diagnosis = FailureDiagnosis.from_dict(raw)
            except Exception:
                continue
            valid_case_ids = [case_id for case_id in diagnosis.case_ids if case_id in case_ids]
            valid_target_names = [target for target in diagnosis.target_names if target in target_names]
            if not valid_case_ids or not diagnosis.category or not diagnosis.root_cause:
                continue
            diagnoses.append(
                FailureDiagnosis(
                    case_ids=valid_case_ids,
                    category=diagnosis.category,
                    root_cause=diagnosis.root_cause,
                    target_names=valid_target_names,
                    evidence=diagnosis.evidence,
                )
            )
        return diagnoses
