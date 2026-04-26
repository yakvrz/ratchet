from __future__ import annotations

import json
import re
from typing import Any

from ratchet.errors import OptimizerModelError
from ratchet.model_client import ResponsesModelClient
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

    def diagnose(
        self,
        summary: PatchSummary,
        surface: list[EditableTarget],
        objective: OptimizationObjective | None = None,
    ) -> tuple[list[FailureDiagnosis], str]:
        failed_examples = summary.failed_examples(limit=12, max_text_chars=900)
        if not failed_examples:
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
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "editable_targets": [target.to_dict() for target in surface],
            "failed_examples": failed_examples,
        }
        try:
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text={
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
                },
                input=(
                    "You are Ratchet's failure diagnoser inside a task-agnostic agent optimizer. "
                    "Return strict JSON with exactly one top-level key, diagnoses, whose value is an array. "
                    "Each diagnosis object must include case_ids, category, root_cause, target_names, and evidence. "
                    "case_ids must come from failed_examples. target_names must come from editable_targets. "
                    "evidence must be an array of compact objects that cite case_id plus the observed output, expected signal, "
                    "or grader note that supports the diagnosis. If labels, notes, output fields, or raw_output_text indicate "
                    "malformed output, empty output, parser fallback, or an invalid output contract, diagnose that as an "
                    "output-format/contract failure before attributing it to semantic caution or grader behavior. "
                    "Do not propose patches. Do not mention hidden evaluator or judge changes; the scorer is frozen and "
                    "outside the editable surface. Generalize root causes from evidence "
                    "without copying case-specific inputs, IDs, or expected answers into future behavior.\n\n"
                    f"{json.dumps(prompt, indent=2, default=str)}"
                ),
                max_output_tokens=5000,
            )
        except Exception as exc:
            raise OptimizerModelError(f"Optimizer diagnoser failed: {exc}") from exc
        try:
            payload = self._extract_json_object(response.output_text)
        except Exception as exc:
            raise OptimizerModelError(f"Optimizer diagnoser returned invalid JSON: {exc}") from exc
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

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in diagnoser response.")
        return json.loads(match.group(0))
