from __future__ import annotations

import json
import os
from pathlib import Path

from ratchet.model_client import ResponsesModelClient
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, RunRecord

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
            "balance: user asks for the current account balance. "
            "transactions: user asks about recent or past account transactions. "
            "transfer: user wants to transfer money. "
            "pay_bill: user wants to pay a bill. "
            "bill_balance: user asks how much is owed on a bill. "
            "bill_due: user asks when a bill is due. "
            "credit_limit: user asks what their credit limit is. "
            "credit_limit_change: user wants to raise, lower, or change a credit limit. "
            "card_declined: user says a card payment was declined. "
            "freeze_account: user wants to freeze or lock an account. "
            "account_blocked: user says the account is blocked or inaccessible. "
            "report_fraud: user wants to report fraud or suspicious account activity. "
            "oos: the request is outside these supported intents."
        ),
        "label_aliases": (
            "Bill amount, amount owed, and bill total map to bill_balance. Payment deadline, due date, "
            "or when to pay maps to bill_due. Lock, freeze, or disable an account maps to freeze_account "
            "when the user requests the action."
        ),
        "confusable_label_rules": (
            "Distinguish account balance from bill balance: choose balance for money currently in an account and "
            "bill_balance for amount owed on a bill. Distinguish bill_due from pay_bill: choose bill_due for due-date "
            "questions and pay_bill for payment actions. Distinguish credit_limit from credit_limit_change: choose "
            "credit_limit when asking the current limit and credit_limit_change when asking to modify it. Choose oos "
            "when the request is not one of the listed intents, even if it mentions banking words."
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
