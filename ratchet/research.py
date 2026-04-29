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
    combine_response_diagnostics,
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
                "that a later implementer may use. Treat task_theory experiment opportunities and high-suitability "
                "affordances as the research surface; preserve mechanism-distinct questions when residual failures "
                "could plausibly be instruction/example-limited or model-capability-limited."
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
            str(affordance.get("family") or affordance.get("transform_family"))
            for affordance in state.affordances
            if affordance.get("family") or affordance.get("transform_family")
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
                        "The previous research-planner response was invalid JSON. "
                        "Return only a valid JSON object matching the same schema. "
                        "Preserve the intended experiment_intents; do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:9000]}"
                    ),
                    max_output_tokens=max_output_tokens,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="research_planner",
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
                        f"Research planner returned invalid JSON: {parse_exc}; repair failed: {repair_exc}"
                    ) from repair_exc
        except Exception as exc:
            if isinstance(exc, OptimizerModelError):
                raise
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
                "Use evidence_ledger.candidate_evidence as the decision surface. "
                "Small-dev evidence is triage evidence, not a final ranking; preserve mechanism-distinct "
                "candidates when evidence is close or noisy. Base the decision on evidence confidence, "
                "mechanism diversity, baseline stability, remaining budget, and expected information value. "
                "Distinguish the cost of measuring a candidate from the candidate's deployed cost/latency tradeoff; "
                "high-cost candidates can still deserve measurement when they test a capability, efficiency, or "
                "quality-frontier question."
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
        _validate_selector_state_has_evidence(state=state, candidate_ids=candidate_ids)
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
            try:
                payload = extract_json_object(response.output_text)
            except Exception as parse_exc:
                primary_diagnostics = self.last_call_diagnostics or {}
                repair_started_at = time.perf_counter()
                repair_response = self._client.create_response(
                    model=self.model,
                    reasoning={"effort": self.reasoning_effort},
                    text=response_format,
                    input=(
                        "The previous measurement-selector response was invalid JSON. "
                        "Return only a valid JSON object matching the same schema. "
                        "Preserve the selected_candidate_ids and skipped_candidate_reasons where possible; do not add prose.\n\n"
                        f"Invalid response:\n{response.output_text[:9000]}"
                    ),
                    max_output_tokens=MEASUREMENT_SELECTOR_MAX_OUTPUT_TOKENS,
                )
                repair_diagnostics = response_diagnostics(
                    repair_response,
                    model=self.model,
                    elapsed_s=time.perf_counter() - repair_started_at,
                )
                self.last_call_diagnostics = combine_response_diagnostics(
                    component="measurement_selector",
                    primary=primary_diagnostics,
                    repair=repair_diagnostics,
                )
                try:
                    payload = extract_json_object(repair_response.output_text)
                except Exception as repair_exc:
                    self.last_call_diagnostics = {
                        **self.last_call_diagnostics,
                        "repair_error": str(repair_exc),
                    }
                    raise OptimizerModelError(
                        f"Measurement selector returned invalid JSON: {parse_exc}; repair failed: {repair_exc}"
                    ) from repair_exc
        except Exception as exc:
            if isinstance(exc, OptimizerModelError):
                raise
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
        decision = _with_default_skip_reasons(decision, candidate_ids=candidate_ids)
        return decision


def _validate_selector_state_has_evidence(*, state: dict[str, Any], candidate_ids: list[str]) -> None:
    ledger = state.get("evidence_ledger")
    if not isinstance(ledger, dict):
        raise OptimizerModelError("Measurement selector requires evidence_ledger in state.")
    rows = ledger.get("candidate_evidence")
    if not isinstance(rows, list):
        raise OptimizerModelError("Measurement selector requires evidence_ledger.candidate_evidence.")
    observed_ids = {str(row.get("candidate_id")) for row in rows if isinstance(row, dict)}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in observed_ids]
    if missing:
        raise OptimizerModelError(
            "Measurement selector requires ledger evidence for every candidate: "
            + ", ".join(missing[:8])
        )


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
            "affordance_ids": {"type": "array", "items": {"type": "string", "maxLength": 180}},
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


def _with_default_skip_reasons(
    decision: MeasurementDecision,
    *,
    candidate_ids: list[str],
) -> MeasurementDecision:
    skipped = dict(decision.skipped_candidate_reasons)
    selected = set(decision.selected_candidate_ids)
    for candidate_id in candidate_ids:
        if candidate_id not in selected and not skipped.get(candidate_id):
            skipped[candidate_id] = "not selected by measurement selector"
    return MeasurementDecision(
        stage=decision.stage,
        selected_candidate_ids=list(decision.selected_candidate_ids),
        rationale=decision.rationale,
        expected_information=decision.expected_information,
        risks=decision.risks,
        skipped_candidate_reasons=skipped,
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
