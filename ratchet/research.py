from __future__ import annotations

import json
import time
from typing import Any

from ratchet.errors import OptimizerModelError
from ratchet.experiments import CANDIDATE_ROLES, MECHANISM_CLASSES, SearchBrief, SearchPlan
from ratchet.io import extract_json_object
from ratchet.model_client import (
    ResponsesModelClient,
    combine_response_diagnostics,
    error_response_diagnostics,
    response_diagnostics,
)


SEARCH_PLANNER_MAX_OUTPUT_TOKENS = 5000


class SearchPlanner:
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

    def plan(self, *, state: dict[str, Any], surface_opportunity_ids: set[str]) -> SearchPlan:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "role": "Ratchet Search Planner",
            "instruction": (
                "Return one search_plan only. Diagnose the parent branch, name concise hypotheses, "
                "choose target mechanisms, and emit candidate briefs that cite only supplied "
                "surface_opportunity_ids. Do not write transform programs, candidate IDs, or "
                "measurement selections; deterministic code handles compilation and staged evaluation."
            ),
            "state": state,
        }
        response_format = {
            "format": {
                "type": "json_schema",
                "name": "ratchet_search_plan",
                "strict": False,
                "schema": _search_plan_schema(),
            }
        }
        payload = self._call_json(
            prompt_prefix="You are Ratchet's search planner. Return only JSON matching the schema.",
            prompt=prompt,
            response_format=response_format,
            max_output_tokens=SEARCH_PLANNER_MAX_OUTPUT_TOKENS,
        )
        payload = _normalize_search_plan_payload(payload)
        try:
            plan = SearchPlan.from_dict(payload)
        except Exception as exc:
            repaired = self._repair_payload(
                payload=payload,
                validation_error=OptimizerModelError(f"Search planner returned malformed search plan: {exc}"),
                response_format=response_format,
                max_output_tokens=SEARCH_PLANNER_MAX_OUTPUT_TOKENS,
            )
            repaired = _normalize_search_plan_payload(repaired)
            try:
                plan = SearchPlan.from_dict(repaired)
            except Exception as repair_exc:
                raise OptimizerModelError(
                    f"Search planner returned malformed search plan: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        unknown = sorted(set(plan.surface_opportunity_ids) - surface_opportunity_ids)
        if unknown:
            raise OptimizerModelError(f"Search planner used unknown surface_opportunity_ids: {unknown}")
        return plan

    def _repair_payload(
        self,
        *,
        payload: dict[str, Any],
        validation_error: OptimizerModelError,
        response_format: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        if self._client is None:
            raise OptimizerModelError("Search planner repair requested before client initialization.")
        primary_diagnostics = self.last_call_diagnostics or {}
        repair_started_at = time.perf_counter()
        repair_response = self._client.create_response(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            text=response_format,
            input=(
                "The previous search-planner response was valid JSON but failed schema validation. "
                "Return only a valid search_plan JSON object. Every brief needs a non-empty brief_id, "
                "mechanism_class, hypothesis, and at least one known surface_opportunity_id. "
                "Do not write patches or prose.\n\n"
                f"Validation error: {validation_error}\n"
                f"Invalid JSON object:\n{json.dumps(payload, separators=(',', ':'), default=str)[:12000]}"
            ),
            max_output_tokens=max_output_tokens,
        )
        repair_diagnostics = response_diagnostics(
            repair_response,
            model=self.model,
            elapsed_s=time.perf_counter() - repair_started_at,
        )
        self.last_call_diagnostics = combine_response_diagnostics(
            component="search_planner",
            primary=primary_diagnostics,
            repair=repair_diagnostics,
        )
        try:
            return extract_json_object(repair_response.output_text)
        except Exception as repair_exc:
            self.last_call_diagnostics = {
                **self.last_call_diagnostics,
                "repair_error": str(repair_exc),
            }
            raise OptimizerModelError(
                f"Search planner returned schema-invalid JSON: {validation_error}; repair failed: {repair_exc}"
            ) from repair_exc

    def _call_json(
        self,
        *,
        prompt_prefix: str,
        prompt: dict[str, Any],
        response_format: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        prompt_input = f"{prompt_prefix}\n\n{json.dumps(prompt, separators=(',', ':'), default=str)}"
        started_at = time.perf_counter()
        try:
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text=response_format,
                input=prompt_input,
                max_output_tokens=max_output_tokens,
            )
            self.last_call_diagnostics = {
                "component": "search_planner",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            try:
                return extract_json_object(response.output_text)
            except Exception as parse_exc:
                primary_diagnostics = self.last_call_diagnostics or {}
                repair_started_at = time.perf_counter()
                repair_response = self._client.create_response(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    text=response_format,
                    input=(
                        "The previous search-planner response was invalid JSON. "
                        "Return only a valid JSON object matching the same schema. "
                        "Preserve the intended search plan; do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:12000]}"
                    ),
                    max_output_tokens=max_output_tokens,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="search_planner",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                try:
                    return extract_json_object(repair_response.output_text)
                except Exception as repair_exc:
                    self.last_call_diagnostics = {
                        **self.last_call_diagnostics,
                        "repair_error": str(repair_exc),
                    }
                    raise OptimizerModelError(
                        f"Search planner returned invalid JSON: {parse_exc}; repair failed: {repair_exc}"
                    ) from repair_exc
        except Exception as exc:
            if isinstance(exc, OptimizerModelError):
                raise
            self.last_call_diagnostics = {
                "component": "search_planner",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Search planner failed: {exc}") from exc


def _search_plan_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "plan_id": {"type": "string", "maxLength": 80},
            "diagnosis": {"type": "string", "maxLength": 1200},
            "hypotheses": {"type": "array", "items": {"type": "string", "maxLength": 500}},
            "target_mechanisms": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(MECHANISM_CLASSES)},
            },
            "briefs": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "brief_id": {"type": "string", "maxLength": 80},
                        "mechanism_class": {"type": "string", "enum": sorted(MECHANISM_CLASSES)},
                        "hypothesis": {"type": "string", "maxLength": 600},
                        "surface_opportunity_ids": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "maxLength": 180},
                        },
                        "target_slices": {"type": "array", "items": {"type": "string", "maxLength": 160}},
                        "candidate_roles": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(CANDIDATE_ROLES)},
                        },
                        "measurements": {"type": "array", "items": {"type": "string", "maxLength": 120}},
                        "success_criteria": {"type": "string", "maxLength": 300},
                        "disconfirming_result": {"type": "string", "maxLength": 300},
                        "priority": {"type": "integer"},
                    },
                    "required": ["brief_id", "mechanism_class", "hypothesis", "surface_opportunity_ids"],
                },
            },
            "confidence": {"type": "string", "maxLength": 40},
        },
        "required": ["plan_id", "diagnosis", "hypotheses", "target_mechanisms", "briefs"],
    }


def _normalize_search_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    if "search_plan" in normalized and isinstance(normalized["search_plan"], dict):
        normalized = dict(normalized["search_plan"])
    normalized["plan_id"] = _first_text(normalized.get("plan_id"), normalized.get("id")) or "P_001"
    normalized["diagnosis"] = _first_text(
        normalized.get("diagnosis"),
        normalized.get("diagnosis_summary"),
        normalized.get("summary"),
    ) or "Planner found no actionable diagnosis."
    normalized["hypotheses"] = _string_list(normalized.get("hypotheses"))
    raw_briefs = normalized.get("briefs", normalized.get("candidate_briefs", []))
    briefs: list[dict[str, Any]] = []
    for index, item in enumerate(raw_briefs if isinstance(raw_briefs, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        mechanism = _first_text(item.get("mechanism_class"), item.get("mechanism"))
        opportunity_ids = _string_list(item.get("surface_opportunity_ids"))
        hypothesis = _first_text(item.get("hypothesis"), item.get("rationale"), item.get("description"))
        if mechanism not in MECHANISM_CLASSES or not opportunity_ids or not hypothesis:
            continue
        briefs.append(
            {
                **item,
                "brief_id": _first_text(item.get("brief_id"), item.get("id")) or f"B_{index:03d}",
                "mechanism_class": mechanism,
                "hypothesis": hypothesis,
                "surface_opportunity_ids": opportunity_ids,
                "target_slices": _string_list(item.get("target_slices")),
                "candidate_roles": _string_list(item.get("candidate_roles")),
                "measurements": _string_list(item.get("measurements")),
            }
        )
    normalized["briefs"] = briefs
    target_mechanisms = _string_list(normalized.get("target_mechanisms", normalized.get("active_mechanisms")))
    if not target_mechanisms:
        target_mechanisms = sorted({brief["mechanism_class"] for brief in briefs})
    normalized["target_mechanisms"] = [item for item in target_mechanisms if item in MECHANISM_CLASSES]
    normalized["confidence"] = _first_text(normalized.get("confidence")) or "low"
    return normalized


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []
