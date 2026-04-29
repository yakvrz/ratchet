from __future__ import annotations

import os

from ratchet.adapter_generation import (
    GeneratedSingleCallAdapter,
    ModelRequest,
    context_graph_from_spec,
    model_config_from_spec,
)
from ratchet.grading import extract_json_payload
from ratchet.types import AgentSpec, EvalCase, GradeResult, TargetSemantics

try:
    from agent import BANKING77_LABELS, Banking77AgentConfig, _extract_label
except ModuleNotFoundError:
    from .agent import BANKING77_LABELS, Banking77AgentConfig, _extract_label


BASE_SPEC = AgentSpec(
    name="banking77-intent-agent",
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
        "task_rule": "Classify a customer banking support message into exactly one intent label.",
        "label_rule": "Allowed labels: " + ", ".join(BANKING77_LABELS) + ".",
        "label_descriptions": (
            "cash_withdrawal_charge: fee charged for withdrawing cash. "
            "cash_withdrawal_not_recognised: account shows a cash withdrawal the customer did not make. "
            "wrong_amount_of_cash_received: ATM dispensed less or more cash than requested. "
            "card_payment_fee_charged: fee charged for a card purchase. "
            "card_payment_not_recognised: account shows a card payment the customer did not make. "
            "extra_charge_on_statement: duplicate, unexpected, or excessive statement charge not clearly a card fee. "
            "transfer_fee_charged: fee charged for making a bank transfer. "
            "transfer_not_received_by_recipient: sent transfer has not arrived to the recipient. "
            "beneficiary_not_allowed: blocked or unsupported payee/beneficiary. "
            "verify_my_identity: customer needs to complete identity verification. "
            "why_verify_identity: customer asks why identity verification is required. "
            "unable_to_verify_identity: customer tried verification and cannot complete it."
        ),
        "label_aliases": (
            "cash withdrawal means ATM/cashpoint/cash machine. Card payment means purchase or transaction made by card. "
            "Transfer means bank transfer to another account or recipient. Beneficiary means payee or recipient setup."
        ),
        "confusable_label_rules": (
            "Distinguish fees from unrecognized activity: choose *_fee_charged or *_charge only when the customer recognizes "
            "the underlying transaction but disputes the fee or amount. Choose *_not_recognised when the transaction itself "
            "is unfamiliar. Distinguish transfer_not_received_by_recipient from beneficiary_not_allowed: choose transfer_not_received_by_recipient "
            "when money was sent but did not arrive; choose beneficiary_not_allowed when the payee cannot be added or used."
        ),
        "decision_rule": (
            "Choose the allowed label with the strongest literal word overlap with the customer message. "
            "If several labels overlap, prefer the label mentioning the concrete payment, card, transfer, "
            "cash, or identity object named in the message."
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


class Banking77IntentHarness:
    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def build_model_request(self, spec: AgentSpec, case: EvalCase) -> ModelRequest:
        config = Banking77AgentConfig.from_agent_config(_agent_config_from_spec(spec))
        return ModelRequest(
            context=context_graph_from_spec(
                spec,
                section_order=[
                    "task_rule",
                    "label_rule",
                    "label_descriptions",
                    "label_aliases",
                    "confusable_label_rules",
                    "decision_rule",
                    "output_rule",
                ],
            ),
            input=str(case.input),
            model_config=model_config_from_spec(spec),
            text=config.text_config(),
        )

    def parse_output(self, raw_output_text: str) -> object:
        payload = extract_json_payload(raw_output_text)
        if isinstance(payload, dict):
            return payload
        label = _extract_label(raw_output_text)
        return {"label": label} if label is not None else {"label": "invalid", "invalid_output": raw_output_text}

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        expected = case.expected
        if not isinstance(expected, dict):
            raise ValueError("BANKING77 grader requires dict expected payloads.")
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

class Banking77IntentAdapter(GeneratedSingleCallAdapter):
    def __init__(self, env_path: str | None = None, client: object | None = None, runner: object | None = None) -> None:
        if client is None and runner is not None:
            client = getattr(runner, "client", None)
        super().__init__(
            harness=Banking77IntentHarness(),
            env_path=env_path or os.environ.get("RATCHET_ENV_FILE", ".env"),
            client=client,
        )


def _agent_config_from_spec(spec: AgentSpec) -> dict[str, str]:
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
        "few_shot": "",
    }


adapter = Banking77IntentAdapter()
