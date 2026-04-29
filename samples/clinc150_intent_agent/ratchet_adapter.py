from __future__ import annotations

import json
import os
from pathlib import Path

from ratchet.model_client import ResponsesModelClient
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, RunRecord, TargetSemantics

try:
    from agent import CLINC150_LABELS, Clinc150IntentRunner
except ModuleNotFoundError:
    from .agent import CLINC150_LABELS, Clinc150IntentRunner


BASE_SPEC = AgentSpec(
    name="clinc150-intent-agent",
    model="gemini-2.5-flash-lite",
    model_options=[
        "gemini-3.1-flash-lite-preview",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ],
    instructions={
        "task_rule": "Classify a user assistant request into exactly one intent label.",
        "label_rule": "Allowed labels: " + ", ".join(CLINC150_LABELS) + ".",
        "label_descriptions": (
            "account_blocked: user says an account is blocked or inaccessible. "
            "application_status: user asks about the status of an application. "
            "balance: user asks for the current account balance. "
            "bill_balance: user asks how much is owed on a bill. "
            "bill_due: user asks when a bill is due. "
            "book_flight: user wants to book an airline flight. "
            "book_hotel: user wants to book a hotel. "
            "calendar: user asks about calendar events or schedule. "
            "calendar_update: user wants to add, remove, or change a calendar event. "
            "card_declined: user says a card payment was declined. "
            "credit_limit: user asks what their credit limit is. "
            "credit_limit_change: user wants to raise, lower, or change a credit limit. "
            "damaged_card: user says a card is physically damaged or not working. "
            "directions: user asks how to get somewhere. "
            "distance: user asks how far away something is. "
            "flight_status: user asks whether a flight is on time, delayed, or canceled. "
            "freeze_account: user wants to freeze or lock an account. "
            "lost_luggage: user asks about missing baggage. "
            "order_status: user asks where an order is or whether it shipped. "
            "pay_bill: user wants to pay a bill. "
            "pin_change: user wants to change a card or account PIN. "
            "restaurant_reservation: user wants to book a restaurant table. "
            "restaurant_reviews: user asks for reviews of a restaurant. "
            "restaurant_suggestion: user asks for restaurant recommendations. "
            "report_fraud: user wants to report fraud or suspicious account activity. "
            "report_lost_card: user says a card is lost or stolen and wants to report it. "
            "shopping_list: user asks what is on a shopping list. "
            "shopping_list_update: user wants to add, remove, or change shopping-list items. "
            "transactions: user asks about recent or past account transactions. "
            "transfer: user wants to transfer money. "
            "travel_alert: user asks to set or manage a bank travel alert. "
            "travel_notification: user asks about travel notifications or notices. "
            "weather: user asks for weather or forecast information. "
            "oos: the request is outside these supported intents."
        ),
        "label_aliases": (
            "Bill amount, amount owed, and bill total map to bill_balance. Payment deadline, due date, "
            "or when to pay maps to bill_due. Lock, freeze, or disable an account maps to freeze_account "
            "when the user requests the action. Itinerary booking maps to book_flight or book_hotel, while "
            "flight delays and cancellations map to flight_status. Directions means route instructions; distance "
            "means how far. Restaurant ratings map to restaurant_reviews; places to eat map to restaurant_suggestion. "
            "Shopping-list edits map to shopping_list_update; shopping-list lookup maps to shopping_list."
        ),
        "confusable_label_rules": (
            "Distinguish account balance from bill balance: choose balance for money currently in an account and "
            "bill_balance for amount owed on a bill. Distinguish bill_due from pay_bill: choose bill_due for due-date "
            "questions and pay_bill for payment actions. Distinguish credit_limit from credit_limit_change: choose "
            "credit_limit when asking the current limit and credit_limit_change when asking to modify it. Distinguish "
            "account_blocked from freeze_account: account_blocked is an access problem, freeze_account is a requested "
            "security action. Distinguish damaged_card from report_lost_card: damaged means broken or worn, lost means "
            "missing or stolen. Distinguish calendar from calendar_update, and shopping_list from shopping_list_update, "
            "by whether the user asks to inspect existing information or change it. Choose oos when the request is not "
            "one of the listed intents, even if it shares words with a supported label."
        ),
        "decision_rule": (
            "Choose the allowed label with the strongest literal word overlap with the user request. "
            "If several labels overlap, prefer the label naming the concrete account, bill, card, credit, "
            "transfer, or fraud action requested."
        ),
        "output_rule": "Return JSON with a single string field named label.",
    },
    output_contract="Return JSON with a single string field named label whose value is one allowed label.",
    runtime={"reasoning_effort": "low", "output_cap": 512},
    target_semantics={
        "task_rule": TargetSemantics(
            role="task_instructions",
            axes=["task_framing", "intent_classification"],
            scope="global",
            risks=["broad_behavior_shift"],
            measurement_hints=["score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "label_rule": TargetSemantics(
            role="label_space",
            axes=["classification_boundary", "label_validity"],
            scope="global",
            risks=["label_space_regression"],
            measurement_hints=["target_label_score_delta", "wrong_label_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "label_descriptions": TargetSemantics(
            role="label_description",
            axes=["classification_boundary", "semantic_grounding"],
            scope="slice",
            risks=["neighbor_label_regression"],
            measurement_hints=["target_label_score_delta", "confusion_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "label_aliases": TargetSemantics(
            role="label_alias_mapping",
            axes=["classification_boundary", "confusion_resolution"],
            scope="slice",
            risks=["neighbor_label_regression"],
            measurement_hints=["target_label_score_delta", "confusion_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "confusable_label_rules": TargetSemantics(
            role="confusable_label_policy",
            axes=["classification_boundary", "confusion_resolution", "tie_breaking"],
            scope="slice",
            risks=["neighbor_label_regression"],
            measurement_hints=["target_label_score_delta", "confusion_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "decision_rule": TargetSemantics(
            role="decision_policy",
            axes=["selection_policy", "tie_breaking"],
            scope="global",
            risks=["broad_behavior_shift"],
            measurement_hints=["score_delta", "confusion_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "output_rule": TargetSemantics(
            role="output_format_rule",
            axes=["format_validity", "parser_compatibility"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "output_contract": TargetSemantics(
            role="external_output_contract",
            axes=["format_validity", "parser_compatibility", "contract_preservation"],
            scope="global",
            risks=["contract_regression"],
            measurement_hints=["invalid_output_delta", "score_delta", "non_target_regression"],
            confidence=1.0,
            source="adapter",
        ),
        "few_shot": TargetSemantics(
            role="example_bank",
            axes=["example_anchoring", "target_slice_recall"],
            scope="slice",
            risks=["neighbor_label_regression", "example_overfit"],
            measurement_hints=["target_slice_score_delta", "non_target_regression", "example_token_delta"],
            confidence=1.0,
            source="adapter",
        ),
        "model": TargetSemantics(
            role="model_choice",
            axes=["model_capability", "cost_latency_tradeoff"],
            scope="global",
            risks=["cost_latency_regression", "quality_regression"],
            measurement_hints=["score_delta", "cost_delta", "latency_delta"],
            confidence=1.0,
            source="adapter",
        ),
        "output_cap": TargetSemantics(
            role="output_budget_control",
            axes=["completion_integrity", "cost_latency_tradeoff"],
            scope="global",
            risks=["truncation_regression", "cost_latency_regression"],
            measurement_hints=["finish_reason_delta", "invalid_output_delta", "score_delta", "latency_delta"],
            confidence=1.0,
            source="adapter",
        ),
        "reasoning_effort": TargetSemantics(
            role="reasoning_effort_control",
            axes=["reasoning_depth", "cost_latency_tradeoff"],
            scope="global",
            risks=["cost_latency_regression", "quality_regression"],
            measurement_hints=["score_delta", "cost_delta", "latency_delta"],
            confidence=1.0,
            source="adapter",
        ),
    },
)


def agent_config_from_spec(spec: AgentSpec) -> dict[str, str]:
    return {
        "model": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "low")),
        "output_cap": str(spec.runtime.get("output_cap", 512)),
        "task_rule": spec.instructions.get("task_rule", ""),
        "label_rule": spec.instructions.get("label_rule", ""),
        "label_descriptions": spec.instructions.get("label_descriptions", ""),
        "label_aliases": spec.instructions.get("label_aliases", ""),
        "confusable_label_rules": spec.instructions.get("confusable_label_rules", ""),
        "decision_rule": spec.instructions.get("decision_rule", ""),
        "output_rule": spec.instructions.get("output_rule", ""),
        "few_shot": render_few_shot_prompt(spec.few_shot),
    }


class Clinc150IntentAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: Clinc150IntentRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            client = ResponsesModelClient(env_path=self.env_path)
            self._runner = Clinc150IntentRunner(client=client)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case)

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        expected = case.expected
        if not isinstance(expected, dict):
            raise ValueError("CLINC150 grader requires dict expected payloads.")
        if not isinstance(output, dict) or "label" not in output:
            return GradeResult(score=0.0, passed=False, labels=["invalid_output"], notes=f"output={output!r}")
        if "invalid_output" in output:
            return GradeResult(
                score=0.0,
                passed=False,
                labels=["invalid_output"],
                notes=f"raw={output.get('invalid_output')!r}",
            )
        actual = str(output["label"])
        expected_label = str(expected["label"])
        if actual == expected_label:
            return GradeResult(score=1.0, passed=True, labels=[])
        return GradeResult(
            score=0.0,
            passed=False,
            labels=["wrong_label", f"expected:{expected_label}", f"actual:{actual}"],
            notes=f"actual={actual!r} expected={expected_label!r}",
        )

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


adapter = Clinc150IntentAdapter()
