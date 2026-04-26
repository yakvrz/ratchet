from __future__ import annotations

import json
import os
from pathlib import Path

from ratchet.model_client import ResponsesModelClient
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, RunRecord

try:
    from agent import BANKING77_LABELS, Banking77IntentRunner
except ModuleNotFoundError:
    from .agent import BANKING77_LABELS, Banking77IntentRunner


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


class Banking77IntentAdapter:
    def __init__(
        self,
        env_path: str | None = None,
        runner: Banking77IntentRunner | None = None,
    ) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self._runner = runner

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def run_case(self, case: EvalCase, patch: AgentPatch | None = None) -> RunRecord:
        if self._runner is None:
            client = ResponsesModelClient(env_path=self.env_path)
            self._runner = Banking77IntentRunner(client=client)
        return self._runner.run_case(agent_config_from_spec(BASE_SPEC.apply_patch(patch)), case)

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

    def export(self, patch: AgentPatch, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        spec = BASE_SPEC.apply_patch(patch)
        (out_dir / "patch.json").write_text(json.dumps(patch.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_config.json").write_text(json.dumps(agent_config_from_spec(spec), indent=2, sort_keys=True))


adapter = Banking77IntentAdapter()
