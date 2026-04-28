from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.grading import extract_json_payload
from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord


CLINC150_LABELS = [
    "balance",
    "transactions",
    "transfer",
    "pay_bill",
    "bill_balance",
    "bill_due",
    "credit_limit",
    "credit_limit_change",
    "card_declined",
    "freeze_account",
    "account_blocked",
    "report_fraud",
    "oos",
]


@dataclass(frozen=True)
class Clinc150AgentConfig:
    model: str
    reasoning_effort: str
    output_cap: int
    task_rule: str
    label_rule: str
    label_descriptions: str
    label_aliases: str
    confusable_label_rules: str
    decision_rule: str
    output_rule: str
    few_shot: str

    @classmethod
    def from_agent_config(cls, payload: dict[str, str]) -> "Clinc150AgentConfig":
        return cls(
            model=payload["model"],
            reasoning_effort=payload["reasoning_effort"],
            output_cap=int(payload["output_cap"]),
            task_rule=payload["task_rule"],
            label_rule=payload["label_rule"],
            label_descriptions=payload.get("label_descriptions", ""),
            label_aliases=payload.get("label_aliases", ""),
            confusable_label_rules=payload.get("confusable_label_rules", ""),
            decision_rule=payload["decision_rule"],
            output_rule=payload["output_rule"],
            few_shot=payload.get("few_shot", ""),
        )

    def instructions(self) -> str:
        return " ".join(
            item
            for item in [
                self.task_rule,
                self.label_rule,
                self.label_descriptions,
                self.label_aliases,
                self.confusable_label_rules,
                self.decision_rule,
                self.output_rule,
                self.few_shot,
            ]
            if item
        )

    def text_config(self) -> dict[str, Any]:
        return {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "clinc150_intent",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"label": {"type": "string", "enum": CLINC150_LABELS}},
                    "required": ["label"],
                    "additionalProperties": False,
                },
            },
        }


class Clinc150IntentRunner:
    def __init__(self, *, client: ResponsesModelClient) -> None:
        self.client = client

    def run_case(self, agent_config: dict[str, str], case: EvalCase) -> RunRecord:
        config = Clinc150AgentConfig.from_agent_config(agent_config)
        started_at = time.perf_counter()
        response = self.client.create_response(
            model=config.model,
            reasoning={"effort": config.reasoning_effort},
            instructions=config.instructions(),
            input=str(case.input),
            max_output_tokens=config.output_cap,
            text=config.text_config(),
        )
        latency_s = time.perf_counter() - started_at
        raw_output_text = response.output_text.strip()
        payload = extract_json_payload(raw_output_text)
        parser_fallback = not isinstance(payload, dict)
        if isinstance(payload, dict):
            output = payload
        else:
            label = _extract_label(raw_output_text)
            output = {"label": label} if label is not None else {"label": "invalid", "invalid_output": raw_output_text}
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                cost_usd=estimate_cost_usd(config.model, response.usage.input_tokens, response.usage.output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                raw_output_text=raw_output_text,
                metadata={
                    "model": config.model,
                    "finish_reason": str(getattr(response, "finish_reason", "") or ""),
                    "requested_output_cap": config.output_cap,
                    "raw_output_length": len(raw_output_text),
                    "parser_fallback": parser_fallback,
                    "invalid_output": isinstance(output, dict) and "invalid_output" in output,
                    "output_tokens": response.usage.output_tokens,
                    "output_item_types": [item.type for item in response.output],
                },
            ),
        )


def _extract_label(text: str) -> str | None:
    normalized = text.strip().strip("`").strip()
    if normalized in CLINC150_LABELS:
        return normalized
    lowered = normalized.lower()
    matches = [label for label in CLINC150_LABELS if label.lower() in lowered]
    if len(matches) == 1:
        return matches[0]
    return None
