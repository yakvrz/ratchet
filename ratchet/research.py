from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any

from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentIntent
from ratchet.io import extract_json_object
from ratchet.model_client import (
    ResponsesModelClient,
    combine_response_diagnostics,
    error_response_diagnostics,
    response_diagnostics,
)


RESEARCH_CONTROLLER_MAX_OUTPUT_TOKENS = 3000


@dataclass(frozen=True)
class ResearchAction:
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


@dataclass(frozen=True)
class ResearchDecision:
    action_id: str
    action_type: str
    selected_candidate_ids: list[str] = field(default_factory=list)
    experiment_intents: list[ExperimentIntent] = field(default_factory=list)
    rationale: str = ""
    expected_information: str = ""
    risks: str = ""
    skipped_candidate_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "selected_candidate_ids": list(self.selected_candidate_ids),
            "experiment_intents": [intent.to_dict() for intent in self.experiment_intents],
            "rationale": self.rationale,
            "expected_information": self.expected_information,
            "risks": self.risks,
            "skipped_candidate_reasons": dict(self.skipped_candidate_reasons),
        }


class ResearchController:
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

    def decide(
        self,
        *,
        state: dict[str, Any],
        allowed_actions: list[ResearchAction],
    ) -> ResearchDecision:
        if not allowed_actions:
            raise ValueError("allowed_actions must not be empty")
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "role": "Ratchet Research Controller",
                "instruction": (
                    "Choose exactly one allowed action. You do not write patches. "
                    "You decide what Ratchet should learn next under budget. For plan_experiments actions, "
                    "return typed experiment_intents that the proposer must implement. "
                    "Respect action limits, candidate IDs, split boundaries, and skipped-candidate accountability."
                ),
            "state": state,
            "allowed_actions": [action.to_dict() for action in allowed_actions],
        }
        response_format = {
            "format": {
                "type": "json_schema",
                "name": "ratchet_research_decision",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "action_id": {"type": "string", "maxLength": 120},
                        "action_type": {"type": "string", "maxLength": 80},
                        "selected_candidate_ids": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 120},
                        },
                        "experiment_intents": {
                            "type": "array",
                            "maxItems": 4,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "intent_id": {"type": "string", "maxLength": 80},
                                    "mechanism_class": {"type": "string", "maxLength": 80},
                                    "hypothesis": {"type": "string", "maxLength": 360},
                                    "target_slices": {"type": "array", "items": {"type": "string", "maxLength": 160}},
                                    "candidate_roles": {"type": "array", "items": {"type": "string", "maxLength": 40}},
                                    "measurements": {"type": "array", "items": {"type": "string", "maxLength": 120}},
                                    "allowed_families": {"type": "array", "items": {"type": "string", "maxLength": 80}},
                                    "success_criteria": {"type": "string", "maxLength": 300},
                                    "disconfirming_result": {"type": "string", "maxLength": 300},
                                    "priority": {"type": "integer"},
                                },
                                "required": ["intent_id", "mechanism_class", "hypothesis"],
                            },
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
                        "action_id",
                        "action_type",
                        "selected_candidate_ids",
                        "experiment_intents",
                        "rationale",
                        "expected_information",
                        "risks",
                        "skipped_candidate_reasons",
                    ],
                },
            }
        }
        prompt_input = (
            "You are Ratchet's research controller. Return only JSON matching the schema.\n\n"
            f"{json.dumps(prompt, separators=(',', ':'), default=str)}"
        )
        started_at = time.perf_counter()
        try:
            response = self._client.create_response(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                text=response_format,
                input=prompt_input,
                max_output_tokens=RESEARCH_CONTROLLER_MAX_OUTPUT_TOKENS,
            )
            self.last_call_diagnostics = {
                "component": "research_controller",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **response_diagnostics(
                    response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
        except Exception as exc:
            self.last_call_diagnostics = {
                "component": "research_controller",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Research controller failed: {exc}") from exc
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
                        "The previous research-controller response was invalid JSON. "
                        "Return only a valid JSON object matching the requested schema. "
                        "Preserve the same action choice, candidate IDs, experiment_intents, rationale, "
                        "expected_information, risks, and skipped_candidate_reasons where possible; do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:9000]}"
                    ),
                    max_output_tokens=RESEARCH_CONTROLLER_MAX_OUTPUT_TOKENS,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="research_controller",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                payload = extract_json_object(repair_response.output_text)
            except Exception as repair_exc:
                self.last_call_diagnostics = {
                    **primary_diagnostics,
                    "component": "research_controller",
                    "repair_attempted": True,
                    "repair_error": str(repair_exc),
                }
                raise OptimizerModelError(
                    f"Research controller returned invalid JSON: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        decision = _decision_from_payload(payload)
        try:
            validate_research_decision(decision, allowed_actions)
        except OptimizerModelError as exc:
            if (self.last_call_diagnostics or {}).get("repair_attempted"):
                raise
            primary_diagnostics = self.last_call_diagnostics or {}
            repair_started_at = time.perf_counter()
            try:
                repair_response = self._client.create_response(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    text=response_format,
                    input=(
                        "The previous research-controller response was valid JSON but violated the allowed action "
                        "contract. Return only a corrected JSON object matching the requested schema. "
                        "Use exactly one allowed action_id/action_type, select only allowed candidate IDs, respect "
                        "max_select, and include skipped_candidate_reasons for every unselected candidate.\n\n"
                        f"Validation error:\n{exc}\n\n"
                        f"Allowed actions:\n{json.dumps([action.to_dict() for action in allowed_actions], separators=(',', ':'), default=str)}\n\n"
                        f"Invalid JSON object:\n{json.dumps(payload, separators=(',', ':'), default=str)[:9000]}"
                    ),
                    max_output_tokens=RESEARCH_CONTROLLER_MAX_OUTPUT_TOKENS,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="research_controller",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                decision = _decision_from_payload(extract_json_object(repair_response.output_text))
                validate_research_decision(decision, allowed_actions)
            except Exception as repair_exc:
                self.last_call_diagnostics = {
                    **primary_diagnostics,
                    "component": "research_controller",
                    "repair_attempted": True,
                    "repair_error": str(repair_exc),
                }
                raise OptimizerModelError(
                    f"Research controller returned invalid decision: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        return decision


def _decision_from_payload(payload: dict[str, Any]) -> ResearchDecision:
    raw_selected = payload.get("selected_candidate_ids", [])
    if not isinstance(raw_selected, list):
        raise OptimizerModelError("Research controller selected_candidate_ids is not an array")
    raw_skipped = payload.get("skipped_candidate_reasons", {})
    if not isinstance(raw_skipped, dict):
        raise OptimizerModelError("Research controller skipped_candidate_reasons is not an object")
    raw_intents = payload.get("experiment_intents", [])
    if raw_intents is None:
        raw_intents = []
    if not isinstance(raw_intents, list):
        raise OptimizerModelError("Research controller experiment_intents is not an array")
    intents: list[ExperimentIntent] = []
    for index, raw_intent in enumerate(raw_intents, start=1):
        if not isinstance(raw_intent, dict):
            raise OptimizerModelError("Research controller experiment_intents entries must be objects")
        try:
            intents.append(ExperimentIntent.from_dict(raw_intent, fallback_id=f"intent_{index}"))
        except Exception as exc:
            raise OptimizerModelError(f"Research controller returned malformed experiment intent: {exc}") from exc
    return ResearchDecision(
        action_id=str(payload.get("action_id") or ""),
        action_type=str(payload.get("action_type") or ""),
        selected_candidate_ids=[
            str(item)
            for item in raw_selected
            if isinstance(item, str)
        ],
        experiment_intents=intents,
        rationale=str(payload.get("rationale") or ""),
        expected_information=str(payload.get("expected_information") or ""),
        risks=str(payload.get("risks") or ""),
        skipped_candidate_reasons={
            str(key): str(value)
            for key, value in raw_skipped.items()
        },
    )


def validate_research_decision(
    decision: ResearchDecision,
    allowed_actions: list[ResearchAction],
) -> ResearchAction:
    actions_by_id = {action.action_id: action for action in allowed_actions}
    action = actions_by_id.get(decision.action_id)
    if action is None:
        raise OptimizerModelError(f"Research controller selected unknown action_id {decision.action_id!r}")
    if decision.action_type != action.action_type:
        raise OptimizerModelError(
            f"Research controller action_type {decision.action_type!r} does not match allowed action "
            f"{action.action_type!r}"
        )
    selected = list(decision.selected_candidate_ids)
    if len(selected) != len(set(selected)):
        raise OptimizerModelError("Research controller selected duplicate candidate IDs")
    allowed_candidate_ids = set(action.candidate_ids)
    unknown = sorted(set(selected) - allowed_candidate_ids)
    if unknown:
        raise OptimizerModelError(f"Research controller selected unknown candidate IDs: {unknown}")
    if action.max_select >= 0 and len(selected) > action.max_select:
        raise OptimizerModelError(
            f"Research controller selected {len(selected)} candidates, above max_select={action.max_select}"
        )
    if action.action_type != "plan_experiments" and decision.experiment_intents:
        raise OptimizerModelError("Research controller returned experiment_intents for a non-planning action")
    if action.action_type == "plan_experiments" and decision.action_id != "stop_branch" and not decision.experiment_intents:
        raise OptimizerModelError("Research controller plan_experiments action must include experiment_intents")
    unselected = allowed_candidate_ids - set(selected)
    missing_reasons = sorted(candidate_id for candidate_id in unselected if not decision.skipped_candidate_reasons.get(candidate_id))
    if missing_reasons:
        raise OptimizerModelError(
            "Research controller omitted skipped_candidate_reasons for "
            f"{missing_reasons}"
        )
    return action
