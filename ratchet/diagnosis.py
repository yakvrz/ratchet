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
from ratchet.surfaces import SurfaceSpec
from ratchet.types import EditableTarget, FailureDiagnosis, OptimizationObjective


DIAGNOSIS_FAILED_EXAMPLE_LIMIT = 8
DIAGNOSIS_FAILED_EXAMPLE_MAX_CHARS = 420
DIAGNOSIS_MAX_OUTPUT_TOKENS = 4000


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
        surface: list[EditableTarget] | SurfaceSpec,
        objective: OptimizationObjective | None = None,
    ) -> tuple[list[FailureDiagnosis], str]:
        failed_examples = summary.failed_examples(
            limit=DIAGNOSIS_FAILED_EXAMPLE_LIMIT,
            max_text_chars=DIAGNOSIS_FAILED_EXAMPLE_MAX_CHARS,
        )
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
        surface: list[EditableTarget] | SurfaceSpec,
        failed_examples: list[dict[str, Any]],
        objective: OptimizationObjective,
    ) -> list[FailureDiagnosis]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "objective": objective.to_dict(),
            "current_candidate": summary.patch.to_dict() if summary.patch is not None else None,
            "behavior": {
                "mean_score": summary.mean_score,
                "pass_count": summary.pass_count,
                "pass_rate": summary.pass_rate,
                "failure_labels": _top_mapping(summary.failure_labels, limit=12),
            },
            "behavior_diagnostics": _compact_behavior_diagnostics(build_behavior_diagnostics(summary)),
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "optimization_surface": _compact_surface(surface),
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
            prompt_input = (
                "You are Ratchet's failure diagnoser inside a task-agnostic agent optimizer. "
                "Return strict JSON with exactly one top-level key, diagnoses, whose value is an array. "
                "Each diagnosis object must include case_ids, category, root_cause, target_names, and evidence. "
                "case_ids must come from failed_examples. target_names must come from optimization_surface. "
                "evidence must be an array of compact objects that cite case_id plus the observed output, expected signal, "
                "behavior_diagnostics may include label confusion, weak-label, and slice summaries; use those summaries "
                "to group related failures when they are clearer than individual rows. If labels, notes, output fields, or raw_output_text indicate "
                "malformed output, empty output, parser fallback, or an invalid output contract, diagnose that as an "
                "output-format/contract failure before attributing it to semantic caution or grader behavior. "
                "Do not propose patches. Do not mention hidden evaluator or judge changes; the scorer is frozen and "
                "outside the editable surface. Generalize root causes from evidence "
                "without copying case-specific inputs, IDs, or expected answers into future behavior.\n\n"
                f"{json.dumps(prompt, separators=(',', ':'), default=str)}"
            )
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text=response_format,
                input=prompt_input,
                max_output_tokens=DIAGNOSIS_MAX_OUTPUT_TOKENS,
            )
            self.last_call_diagnostics = {
                "component": "diagnoser",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": _approximate_prompt_tokens(prompt_input),
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
        except Exception as exc:
            self.last_call_diagnostics = {
                "component": "diagnoser",
                "prompt_chars": len(prompt_input) if "prompt_input" in locals() else None,
                "prompt_approx_tokens": _approximate_prompt_tokens(prompt_input) if "prompt_input" in locals() else None,
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
                    max_output_tokens=DIAGNOSIS_MAX_OUTPUT_TOKENS,
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
        target_names = _surface_target_names(surface)
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


def _compact_behavior_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    return {
        "label_field": diagnostics.get("label_field"),
        "weak_labels": list(diagnostics.get("weak_labels") or [])[:12],
        "confusions": list(diagnostics.get("confusions") or [])[:8],
        "invalid_output_case_ids": list(diagnostics.get("invalid_output_case_ids") or [])[:8],
        "runtime_reliability": diagnostics.get("runtime_reliability", {}),
        "category_metrics": _compact_category_metrics(diagnostics.get("category_metrics") or {}, limit=12),
    }


def _compact_category_metrics(metrics: dict[str, Any], *, limit: int) -> dict[str, Any]:
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


def _compact_editable_target(target: EditableTarget) -> dict[str, Any]:
    current_value = target.current_value
    if isinstance(current_value, str):
        compact_value: Any = current_value[:420]
    elif isinstance(current_value, list):
        compact_value = {"type": "list", "count": len(current_value), "sample": current_value[:2]}
    elif isinstance(current_value, dict):
        compact_value = {"type": "object", "keys": sorted(str(key) for key in current_value.keys())[:12]}
    else:
        compact_value = current_value
    return {
        "name": target.name,
        "kind": target.kind,
        "current_value": compact_value,
        "allowed_ops": list(target.allowed_ops),
        "description": target.description[:180],
        "choices": list(target.choices)[:12],
        "value_schema": dict(target.value_schema),
    }


def _compact_surface(surface: list[EditableTarget] | SurfaceSpec) -> Any:
    if isinstance(surface, SurfaceSpec):
        return {
            "agent_id": surface.agent_id,
            "context_sections": surface.context.graph.section_names(),
            "editable_sections": list(surface.context.editable_sections),
            "hooks": {
                name: {
                    "supported": hook.supported,
                    "allowed_ops": list(hook.allowed_ops),
                    "available_inputs": list(hook.available_inputs),
                }
                for name, hook in sorted(surface.hooks.items())
            },
            "state": surface.state.to_dict(),
            "model": surface.model.to_dict(),
            "response": surface.response.to_dict(),
            "immutable_boundaries": list(surface.immutable_boundaries),
        }
    return [_compact_editable_target(target) for target in surface]


def _surface_target_names(surface: list[EditableTarget] | SurfaceSpec) -> set[str]:
    if isinstance(surface, SurfaceSpec):
        names = set(surface.context.editable_sections)
        names.update(f"context.{name}" for name in surface.context.editable_sections)
        names.add("generated_context")
        if surface.response.draft_response_interception_allowed:
            names.add("draft_response")
        if surface.state.supports_persistent_state:
            names.add("state")
        if surface.model.model_name_configurable or surface.model.max_tokens_configurable:
            names.add("model_config")
        names.update(tool.name for tool in surface.tools.tools)
        return names
    return {target.name for target in surface}


def _compact_patch(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        "operations": [
            {
                "op": operation.get("op"),
                "target": operation.get("target"),
                "value": _compact_value(operation.get("value")),
            }
            for operation in patch.get("operations", [])
            if isinstance(operation, dict)
        ],
        "rationale": str(patch.get("rationale") or "")[:240],
        "expected_effect": str(patch.get("expected_effect") or "")[:240],
    }


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:320]
    if isinstance(value, list):
        return {"type": "list", "count": len(value), "sample": value[:2]}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value)[:12]}
    return value


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


def _approximate_prompt_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)
