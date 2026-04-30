from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any

from ratchet.errors import OptimizerModelError
from ratchet.experiments import (
    CANDIDATE_ROLES,
    MECHANISM_CLASSES,
    EvidencePacket,
    ExperimentIntent,
    MeasurementDecision,
    ResearchState,
    ResearchTheory,
)
from ratchet.io import extract_json_object
from ratchet.model_client import (
    ResponsesModelClient,
    combine_response_diagnostics,
    error_response_diagnostics,
    response_diagnostics,
)


RESEARCH_PLANNER_MAX_OUTPUT_TOKENS = 3500
RESEARCH_THEORIST_MAX_OUTPUT_TOKENS = 5000
MEASUREMENT_SELECTOR_MAX_OUTPUT_TOKENS = 3000


class ResearchTheorist:
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

    def build_theory(
        self,
        *,
        state: dict[str, Any],
        affordance_ids: set[str],
    ) -> ResearchTheory:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        prompt = {
            "role": "Ratchet Research Theorist",
            "instruction": (
                "Build the causal research theory for this branch. Interpret evidence, preserve competing "
                "explanations, name what would disconfirm each hypothesis, and propose measurement-worthy "
                "experiment opportunities. Do not write patches. Do not choose measurements. Every opportunity "
                "must cite only surface_opportunity_ids from the supplied inferred optimization surface."
            ),
            "state": state,
        }
        response_format = {
            "format": {
                "type": "json_schema",
                "name": "ratchet_research_theory",
                "strict": False,
                "schema": _research_theory_schema(),
            }
        }
        payload = self._call_json(
            prompt_prefix="You are Ratchet's research theorist. Return only JSON matching the schema.",
            prompt=prompt,
            response_format=response_format,
            max_output_tokens=RESEARCH_THEORIST_MAX_OUTPUT_TOKENS,
        )
        payload = _normalize_research_theory_payload(payload)
        try:
            theory = ResearchTheory.from_dict(payload)
        except Exception as exc:
            repaired = self._repair_payload(
                payload=payload,
                validation_error=OptimizerModelError(f"Research theorist returned malformed research theory: {exc}"),
                response_format=response_format,
                max_output_tokens=RESEARCH_THEORIST_MAX_OUTPUT_TOKENS,
            )
            repaired = _normalize_research_theory_payload(repaired)
            try:
                theory = ResearchTheory.from_dict(repaired)
            except Exception as repair_exc:
                raise OptimizerModelError(
                    f"Research theorist returned malformed research theory: {exc}; repair failed: {repair_exc}"
                ) from repair_exc
        unknown_affordances = sorted(
            {
                affordance_id
                for opportunity in theory.experiment_opportunities
                for affordance_id in opportunity.affordance_ids
                if affordance_id not in affordance_ids
            }
        )
        if unknown_affordances:
            raise OptimizerModelError(
                f"Research theorist used unknown surface_opportunity_ids: {unknown_affordances}"
            )
        return theory

    def _repair_payload(
        self,
        *,
        payload: dict[str, Any],
        validation_error: OptimizerModelError,
        response_format: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        if self._client is None:
            raise OptimizerModelError("Research theorist repair requested before client initialization.")
        primary_diagnostics = self.last_call_diagnostics or {}
        repair_started_at = time.perf_counter()
        repair_response = self._client.create_response(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            text=response_format,
            input=(
                "The previous research-theorist response was valid JSON but failed schema validation. "
                "Return only a valid research theory JSON object. Every hypothesis needs a non-empty "
                "hypothesis_id; every experiment opportunity needs a non-empty opportunity_id, at least "
                "one known hypothesis_id, a surface mechanism_class, rationale, and only supplied surface_opportunity_ids. "
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
            component="research_theorist",
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
                f"Research theorist returned schema-invalid JSON: {validation_error}; repair failed: {repair_exc}"
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
                "component": "research_theorist",
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
                        "The previous research-theorist response was invalid JSON. "
                        "Return only a valid JSON object matching the same schema. "
                        "Preserve the intended research theory; do not add prose.\n\n"
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
                    component="research_theorist",
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
                        f"Research theorist returned invalid JSON: {parse_exc}; repair failed: {repair_exc}"
                    ) from repair_exc
        except Exception as exc:
            if isinstance(exc, OptimizerModelError):
                raise
            self.last_call_diagnostics = {
                "component": "research_theorist",
                "prompt_chars": len(prompt_input),
                "prompt_approx_tokens": max(1, len(prompt_input) // 4),
                **error_response_diagnostics(
                    exc,
                    model=self.model,
                    elapsed_s=time.perf_counter() - started_at,
                ),
            }
            raise OptimizerModelError(f"Research theorist failed: {exc}") from exc


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
                "Each intent must choose a surface mechanism_class and cite concrete surface_opportunity_ids "
                "that a later implementer may use. Treat research-theory experiment opportunities and "
                "high-suitability surface opportunities as the research surface. Preserve distinct hook/state/tool/"
                "context questions when the evidence supports them."
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
        try:
            return self._parse_intents(payload, state)
        except OptimizerModelError as validation_error:
            repaired = self._repair_payload(
                payload=payload,
                validation_error=validation_error,
                response_format=response_format,
                max_output_tokens=RESEARCH_PLANNER_MAX_OUTPUT_TOKENS,
            )
            return self._parse_intents(repaired, state)

    def _parse_intents(self, payload: dict[str, Any], state: ResearchState) -> list[ExperimentIntent]:
        raw_intents = payload.get("experiment_intents")
        if isinstance(raw_intents, dict):
            raw_intents = [raw_intents]
        if not isinstance(raw_intents, list):
            raise OptimizerModelError("Research planner experiment_intents is not an array")
        affordance_ids = {
            str(affordance.get("surface_opportunity_id") or affordance.get("affordance_id"))
            for affordance in state.affordances
            if affordance.get("surface_opportunity_id") or affordance.get("affordance_id")
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
                intent = _ground_unknown_surface_ids(intent, unknown_affordances, state.affordances)
                unknown_affordances = sorted(set(intent.affordance_ids) - affordance_ids)
                if unknown_affordances:
                    raise OptimizerModelError(
                        f"Research planner intent {intent.intent_id!r} used unknown surface_opportunity_ids: {unknown_affordances}"
                    )
            intents.append(intent)
        return intents

    def _repair_payload(
        self,
        *,
        payload: dict[str, Any],
        validation_error: OptimizerModelError,
        response_format: dict[str, Any],
        max_output_tokens: int,
    ) -> dict[str, Any]:
        if self._client is None:
            raise OptimizerModelError("Research planner repair requested before client initialization.")
        primary_diagnostics = self.last_call_diagnostics or {}
        repair_started_at = time.perf_counter()
        repair_response = self._client.create_response(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            text=response_format,
            input=(
                "The previous research-planner response was valid JSON but failed schema validation. "
                "Return only a JSON object with experiment_intents as an array of valid intent objects. "
                "Preserve the intended research mechanisms where possible; do not add prose.\n\n"
                f"Validation error: {validation_error}\n"
                f"Invalid JSON object:\n{json.dumps(payload, separators=(',', ':'), default=str)[:9000]}"
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
                f"Research planner returned schema-invalid JSON: {validation_error}; repair failed: {repair_exc}"
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
                "mechanism diversity, baseline stability, remaining measurement budget, and expected information "
                "value per measurement dollar. Distinguish the cost of measuring a candidate from the candidate's "
                "deployed cost/latency tradeoff; high-cost candidates can still deserve measurement when they test "
                "a capability, efficiency, or quality-frontier question."
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
            "surface_opportunity_ids": {"type": "array", "minItems": 1, "items": {"type": "string", "maxLength": 180}},
            "success_criteria": {"type": "string", "maxLength": 300},
            "disconfirming_result": {"type": "string", "maxLength": 300},
            "priority": {"type": "integer"},
        },
        "required": ["intent_id", "mechanism_class", "hypothesis", "surface_opportunity_ids"],
    }


def _research_theory_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "theory_id": {"type": "string", "maxLength": 80},
            "summary": {"type": "string", "maxLength": 900},
            "primary_hypothesis_id": {"type": "string", "maxLength": 80},
            "hypotheses": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "hypothesis_id": {"type": "string", "maxLength": 80},
                        "statement": {"type": "string", "maxLength": 700},
                        "mechanism_class": {"type": "string", "enum": sorted(MECHANISM_CLASSES)},
                        "target_slices": {"type": "array", "items": {"type": "string", "maxLength": 160}},
                        "supporting_evidence": {"type": "array", "items": {"type": "string", "maxLength": 260}},
                        "competing_evidence": {"type": "array", "items": {"type": "string", "maxLength": 260}},
                        "disconfirming_result": {"type": "string", "maxLength": 320},
                        "confidence": {"type": "string", "maxLength": 40},
                    },
                    "required": ["hypothesis_id", "statement", "mechanism_class"],
                },
            },
            "experiment_opportunities": {
                "type": "array",
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "properties": {
                        "opportunity_id": {"type": "string", "maxLength": 80},
                        "hypothesis_ids": {"type": "array", "items": {"type": "string", "maxLength": 80}},
                        "mechanism_class": {"type": "string", "enum": sorted(MECHANISM_CLASSES)},
                        "target_slices": {"type": "array", "items": {"type": "string", "maxLength": 160}},
                        "rationale": {"type": "string", "maxLength": 600},
                        "measurements": {"type": "array", "items": {"type": "string", "maxLength": 120}},
                        "disconfirming_result": {"type": "string", "maxLength": 320},
                        "candidate_roles": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(CANDIDATE_ROLES)},
                        },
                        "compatible_mechanisms": {
                            "type": "array",
                            "items": {"type": "string", "enum": sorted(MECHANISM_CLASSES)},
                        },
                        "surface_opportunity_ids": {"type": "array", "items": {"type": "string", "maxLength": 180}},
                        "priority": {"type": "integer"},
                    },
                    "required": ["opportunity_id", "hypothesis_ids", "mechanism_class", "rationale"],
                },
            },
            "disconfirmed_explanations": {"type": "array", "items": {"type": "string", "maxLength": 260}},
            "surprising_observations": {"type": "array", "items": {"type": "string", "maxLength": 260}},
            "prior_lessons": {"type": "array", "items": {"type": "string", "maxLength": 260}},
            "uncertainty": {"type": "string", "maxLength": 500},
            "confidence": {"type": "string", "maxLength": 40},
        },
        "required": ["theory_id", "summary", "primary_hypothesis_id", "hypotheses", "experiment_opportunities"],
    }


def _normalize_research_theory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider JSON quirks at the optimizer-role boundary.

    IDs are Ratchet-internal references, so assigning them preserves the model's
    substantive theory while keeping the typed research objects strict.
    """
    normalized = dict(payload)
    raw_hypotheses = normalized.get("hypotheses", normalized.get("causal_hypotheses", []))
    hypotheses: list[dict[str, Any]] = []
    for index, item in enumerate(raw_hypotheses if isinstance(raw_hypotheses, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        statement = _first_text(
            item.get("statement"),
            item.get("hypothesis"),
            item.get("description"),
            item.get("summary"),
            item.get("rationale"),
        )
        mechanism_class = _first_text(item.get("mechanism_class"), item.get("mechanism"))
        if not statement or mechanism_class not in MECHANISM_CLASSES:
            continue
        hypothesis_id = _first_text(item.get("hypothesis_id"), item.get("id")) or f"H_{index:03d}"
        hypotheses.append(
            {
                **item,
                "hypothesis_id": hypothesis_id,
                "statement": statement,
                "mechanism_class": mechanism_class,
                "target_slices": _string_list(item.get("target_slices")),
                "supporting_evidence": _string_list(
                    item.get("supporting_evidence", item.get("evidence", []))
                ),
                "competing_evidence": _string_list(item.get("competing_evidence")),
                "disconfirming_result": _first_text(item.get("disconfirming_result")),
                "confidence": _first_text(item.get("confidence")) or "low",
            }
        )
    normalized["hypotheses"] = hypotheses
    hypothesis_ids = [item["hypothesis_id"] for item in hypotheses]
    mechanism_by_hypothesis = {item["hypothesis_id"]: item["mechanism_class"] for item in hypotheses}

    normalized["theory_id"] = _first_text(normalized.get("theory_id"), normalized.get("id")) or "T_001"
    normalized["summary"] = _first_text(
        normalized.get("summary"),
        normalized.get("overview"),
        normalized.get("analysis"),
        normalized.get("rationale"),
    )
    primary_hypothesis_id = _first_text(normalized.get("primary_hypothesis_id"))
    if primary_hypothesis_id not in hypothesis_ids and hypothesis_ids:
        primary_hypothesis_id = hypothesis_ids[0]
    normalized["primary_hypothesis_id"] = primary_hypothesis_id

    raw_opportunities = normalized.get(
        "experiment_opportunities",
        normalized.get("opportunities", normalized.get("experiments", [])),
    )
    opportunities: list[dict[str, Any]] = []
    for index, item in enumerate(raw_opportunities if isinstance(raw_opportunities, list) else [], start=1):
        if not isinstance(item, dict):
            continue
        raw_ids = _string_list(item.get("hypothesis_ids"))
        if not raw_ids:
            single_id = _first_text(item.get("hypothesis_id"))
            raw_ids = [single_id] if single_id else []
        cited_ids = [hypothesis_id for hypothesis_id in raw_ids if hypothesis_id in hypothesis_ids]
        if not cited_ids and len(hypothesis_ids) == 1:
            cited_ids = list(hypothesis_ids)
        rationale = _first_text(
            item.get("rationale"),
            item.get("hypothesis"),
            item.get("description"),
            item.get("summary"),
        )
        mechanism_class = _first_text(item.get("mechanism_class"), item.get("mechanism"))
        if mechanism_class not in MECHANISM_CLASSES and len(cited_ids) == 1:
            mechanism_class = mechanism_by_hypothesis.get(cited_ids[0], "")
        if not cited_ids or not rationale or mechanism_class not in MECHANISM_CLASSES:
            continue
        opportunity_id = _first_text(item.get("opportunity_id"), item.get("id")) or f"O_{index:03d}"
        opportunities.append(
            {
                **item,
                "opportunity_id": opportunity_id,
                "hypothesis_ids": cited_ids,
                "mechanism_class": mechanism_class,
                "target_slices": _string_list(item.get("target_slices")),
                "rationale": rationale,
                "measurements": _string_list(item.get("measurements")),
                "disconfirming_result": _first_text(item.get("disconfirming_result")),
                "candidate_roles": _string_list(item.get("candidate_roles")),
                "compatible_mechanisms": _string_list(item.get("compatible_mechanisms")),
                "affordance_ids": _string_list(
                    item.get("surface_opportunity_ids", item.get("affordance_ids"))
                ),
                "priority": int(item.get("priority") or index),
            }
        )
    normalized["experiment_opportunities"] = opportunities
    normalized["disconfirmed_explanations"] = _string_list(normalized.get("disconfirmed_explanations"))
    normalized["surprising_observations"] = _string_list(normalized.get("surprising_observations"))
    normalized["prior_lessons"] = _string_list(normalized.get("prior_lessons"))
    normalized["uncertainty"] = _first_text(normalized.get("uncertainty"))
    normalized["confidence"] = _first_text(normalized.get("confidence")) or "low"
    return normalized


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _ground_unknown_surface_ids(
    intent: ExperimentIntent,
    unknown_ids: list[str],
    affordances: list[dict[str, Any]],
) -> ExperimentIntent:
    mechanism = intent.mechanism_class
    candidates: list[str] = []
    for affordance in affordances:
        surface_id = str(affordance.get("surface_opportunity_id") or affordance.get("affordance_id") or "")
        if not surface_id:
            continue
        surface = str(affordance.get("surface") or affordance.get("mechanism") or "")
        if not surface and surface_id.startswith("surface."):
            parts = surface_id.split(".")
            surface = parts[1] if len(parts) > 1 else ""
        if surface != mechanism:
            continue
        target = str(affordance.get("target") or affordance.get("target_name") or "")
        if any(target and target in unknown_id for unknown_id in unknown_ids):
            candidates.insert(0, surface_id)
        else:
            candidates.append(surface_id)
    grounded_ids = list(dict.fromkeys([item for item in intent.affordance_ids if item not in unknown_ids] + candidates[:2]))
    if not grounded_ids or grounded_ids == intent.affordance_ids:
        return intent
    return ExperimentIntent(
        intent_id=intent.intent_id,
        mechanism_class=intent.mechanism_class,
        hypothesis=intent.hypothesis,
        target_slices=list(intent.target_slices),
        candidate_roles=list(intent.candidate_roles),
        measurements=list(intent.measurements),
        affordance_ids=grounded_ids,
        success_criteria=intent.success_criteria,
        disconfirming_result=intent.disconfirming_result,
        priority=intent.priority,
    )


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
