from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any

from ratchet.errors import OptimizerModelError
from ratchet.experiments import CANDIDATE_ROLES, ExperimentIntent, MeasurementDecision, ResearchState
from ratchet.io import extract_json_object
from ratchet.model_client import (
    ResponsesModelClient,
    error_response_diagnostics,
    response_diagnostics,
)


RESEARCH_PLANNER_MAX_OUTPUT_TOKENS = 3500
MEASUREMENT_SELECTOR_MAX_OUTPUT_TOKENS = 2200


class ResearchPlanner:
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

    def plan(self, state: ResearchState) -> list[ExperimentIntent]:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "role": "Ratchet Research Planner",
            "instruction": (
                "Return experiment_intents only. Do not write patches, candidate IDs, or measurement selections. "
                "Each intent must choose mechanism_class from the listed affordances and cite concrete affordance_ids "
                "that a later implementer may use."
            ),
            "state": state.to_dict(),
        }
        response_format = {
            "format": {
                "type": "json_schema",
                "name": "ratchet_experiment_intents",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "experiment_intents": {
                            "type": "array",
                            "maxItems": 4,
                            "items": _experiment_intent_schema(),
                        }
                    },
                    "required": ["experiment_intents"],
                },
            }
        }
        payload = self._call_json(
            prompt_prefix="You are Ratchet's research planner. Return only JSON matching the schema.",
            prompt=prompt,
            response_format=response_format,
            max_output_tokens=RESEARCH_PLANNER_MAX_OUTPUT_TOKENS,
        )
        raw_intents = payload.get("experiment_intents")
        if not isinstance(raw_intents, list):
            raise OptimizerModelError("Research planner experiment_intents is not an array")
        affordance_ids = {
            str(affordance.get("affordance_id"))
            for affordance in state.affordances
            if affordance.get("affordance_id")
        }
        active_families = {
            str(affordance.get("transform_family"))
            for affordance in state.affordances
            if affordance.get("transform_family")
        }
        intents: list[ExperimentIntent] = []
        for index, raw_intent in enumerate(raw_intents, start=1):
            if not isinstance(raw_intent, dict):
                raise OptimizerModelError("Research planner intent entry is not an object")
            try:
                intent = ExperimentIntent.from_dict(raw_intent)
            except Exception as exc:
                raise OptimizerModelError(f"Research planner returned malformed experiment intent: {exc}") from exc
            unknown_affordances = sorted(set(intent.affordance_ids) - affordance_ids)
            if unknown_affordances:
                raise OptimizerModelError(
                    f"Research planner intent {intent.intent_id!r} used unknown affordance_ids: {unknown_affordances}"
                )
            unknown_families = sorted(set(intent.allowed_families) - active_families)
            if unknown_families:
                raise OptimizerModelError(
                    f"Research planner intent {intent.intent_id!r} used unknown allowed_families: {unknown_families}"
                )
            intents.append(intent)
        return intents

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
                "component": "research_planner",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            return extract_json_object(response.output_text)
        except Exception as exc:
            self.last_call_diagnostics = {
                "component": "research_planner",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Research planner failed: {exc}") from exc


class MeasurementSelector:
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

    def select(
        self,
        *,
        stage: str,
        state: dict[str, Any],
        candidate_ids: list[str],
        max_select: int,
        max_select_per_group: int = 0,
    ) -> MeasurementDecision:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "role": "Ratchet Measurement Selector",
            "instruction": (
                "Select which already-valid candidates should be measured next. "
                "Do not create candidates, alter patches, or revise experiment intents. "
                "Base the decision on observed stage metrics and expected information value."
            ),
            "stage": stage,
            "candidate_ids": candidate_ids,
            "max_select": max_select,
            "max_select_per_group": max_select_per_group,
            "state": state,
        }
        response_format = {
            "format": {
                "type": "json_schema",
                "name": "ratchet_measurement_decision",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "selected_candidate_ids": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 120},
                        },
                        "rationale": {"type": "string", "maxLength": 600},
                        "expected_information": {"type": "string", "maxLength": 600},
                        "risks": {"type": "string", "maxLength": 500},
                        "skipped_candidate_reasons": {
                            "type": "object",
                            "additionalProperties": {"type": "string", "maxLength": 400},
                        },
                    },
                    "required": [
                        "selected_candidate_ids",
                        "rationale",
                        "expected_information",
                        "risks",
                        "skipped_candidate_reasons",
                    ],
                },
            }
        }
        prompt_input = (
            "You are Ratchet's measurement selector. Return only JSON matching the schema.\n\n"
            f"{json.dumps(prompt, separators=(',', ':'), default=str)}"
        )
        started_at = time.perf_counter()
        try:
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text=response_format,
                input=prompt_input,
                max_output_tokens=MEASUREMENT_SELECTOR_MAX_OUTPUT_TOKENS,
            )
            self.last_call_diagnostics = {
                "component": "measurement_selector",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            payload = extract_json_object(response.output_text)
        except Exception as exc:
            self.last_call_diagnostics = {
                "component": "measurement_selector",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Measurement selector failed: {exc}") from exc
        decision = _measurement_decision_from_payload(stage=stage, payload=payload)
        _validate_measurement_decision(
            decision,
            candidate_ids=candidate_ids,
            max_select=max_select,
        )
        return decision


def _experiment_intent_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "intent_id": {"type": "string", "maxLength": 80},
            "mechanism_class": {"type": "string", "maxLength": 80},
            "hypothesis": {"type": "string", "maxLength": 360},
            "target_slices": {"type": "array", "items": {"type": "string", "maxLength": 160}},
            "candidate_roles": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(CANDIDATE_ROLES)},
            },
            "measurements": {"type": "array", "items": {"type": "string", "maxLength": 120}},
            "allowed_families": {"type": "array", "items": {"type": "string", "maxLength": 80}},
            "affordance_ids": {"type": "array", "items": {"type": "string", "maxLength": 80}},
            "success_criteria": {"type": "string", "maxLength": 300},
            "disconfirming_result": {"type": "string", "maxLength": 300},
            "priority": {"type": "integer"},
        },
        "required": ["intent_id", "mechanism_class", "hypothesis", "affordance_ids"],
    }


def _measurement_decision_from_payload(*, stage: str, payload: dict[str, Any]) -> MeasurementDecision:
    raw_selected = payload.get("selected_candidate_ids", [])
    if not isinstance(raw_selected, list):
        raise OptimizerModelError("Measurement selector selected_candidate_ids is not an array")
    raw_skipped = payload.get("skipped_candidate_reasons", {})
    if not isinstance(raw_skipped, dict):
        raise OptimizerModelError("Measurement selector skipped_candidate_reasons is not an object")
    return MeasurementDecision(
        stage=stage,
        selected_candidate_ids=[str(item) for item in raw_selected if isinstance(item, str)],
        rationale=str(payload.get("rationale") or ""),
        expected_information=str(payload.get("expected_information") or ""),
        risks=str(payload.get("risks") or ""),
        skipped_candidate_reasons={str(key): str(value) for key, value in raw_skipped.items()},
    )


def _validate_measurement_decision(
    decision: MeasurementDecision,
    *,
    candidate_ids: list[str],
    max_select: int,
) -> None:
    selected = list(decision.selected_candidate_ids)
    if len(selected) != len(set(selected)):
        raise OptimizerModelError("Measurement selector selected duplicate candidate IDs")
    allowed = set(candidate_ids)
    unknown = sorted(set(selected) - allowed)
    if unknown:
        raise OptimizerModelError(f"Measurement selector selected unknown candidate IDs: {unknown}")
    if max_select >= 0 and len(selected) > max_select:
        raise OptimizerModelError(
            f"Measurement selector selected {len(selected)} candidates, above max_select={max_select}"
        )
    missing_reasons = sorted(candidate_id for candidate_id in allowed - set(selected) if not decision.skipped_candidate_reasons.get(candidate_id))
    if missing_reasons:
        raise OptimizerModelError(
            "Measurement selector omitted skipped_candidate_reasons for "
            f"{missing_reasons}"
        )


@dataclass(frozen=True)
class MeasurementAction:
    action_id: str
    action_type: str
    stage: str = ""
    candidate_ids: list[str] = field(default_factory=list)
    max_select: int = 0
    max_select_per_group: int = 0
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
