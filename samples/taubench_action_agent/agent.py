from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.grading import extract_json_payload
from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord


@dataclass(frozen=True)
class TauBenchActionConfig:
    model: str
    reasoning_effort: str
    output_cap: int
    task_rule: str
    policy_rule: str
    action_rule: str
    sequencing_rule: str
    output_rule: str
    few_shot: str

    @classmethod
    def from_agent_config(cls, payload: dict[str, str]) -> "TauBenchActionConfig":
        return cls(
            model=payload["model"],
            reasoning_effort=payload["reasoning_effort"],
            output_cap=int(payload["output_cap"]),
            task_rule=payload["task_rule"],
            policy_rule=payload["policy_rule"],
            action_rule=payload["action_rule"],
            sequencing_rule=payload["sequencing_rule"],
            output_rule=payload["output_rule"],
            few_shot=payload.get("few_shot", ""),
        )

    def instructions(self) -> str:
        return "\n".join(
            line
            for line in [
                self.task_rule,
                self.policy_rule,
                self.action_rule,
                self.sequencing_rule,
                self.output_rule,
                self.few_shot,
            ]
            if line
        )

    def text_config(self) -> dict[str, Any]:
        return {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "taubench_action_plan",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                },
                                "required": ["name"],
                                "additionalProperties": False,
                            },
                        },
                        "message": {"type": "string"},
                    },
                    "required": ["actions", "message"],
                    "additionalProperties": False,
                },
            },
        }


class TauBenchActionRunner:
    def __init__(self, *, client: ResponsesModelClient) -> None:
        self.client = client

    def run_case(self, agent_config: dict[str, str], case: EvalCase) -> RunRecord:
        config = TauBenchActionConfig.from_agent_config(agent_config)
        started_at = time.perf_counter()
        response = self.client.create_response(
            model=config.model,
            reasoning={"effort": config.reasoning_effort},
            instructions=config.instructions(),
            input=case.input,
            max_output_tokens=config.output_cap,
            text=config.text_config(),
        )
        latency_s = time.perf_counter() - started_at
        raw_output_text = response.output_text.strip()
        payload = extract_json_payload(raw_output_text)
        parser_fallback = not _is_action_payload(payload)
        output = payload if _is_action_payload(payload) else {"actions": [], "message": "", "invalid_output": raw_output_text}
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


def _is_action_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    actions = value.get("actions")
    if not isinstance(actions, list):
        return False
    return all(
        isinstance(action, dict)
        and isinstance(action.get("name"), str)
        for action in actions
    )
