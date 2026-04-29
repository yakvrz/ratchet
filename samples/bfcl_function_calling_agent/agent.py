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
class BfclAgentConfig:
    model: str
    reasoning_effort: str
    output_cap: int
    task_rule: str
    schema_rule: str
    argument_rule: str
    no_call_rule: str
    decision_rule: str
    output_rule: str
    few_shot: str

    @classmethod
    def from_agent_config(cls, payload: dict[str, str]) -> "BfclAgentConfig":
        return cls(
            model=payload["model"],
            reasoning_effort=payload["reasoning_effort"],
            output_cap=int(payload["output_cap"]),
            task_rule=payload["task_rule"],
            schema_rule=payload["schema_rule"],
            argument_rule=payload["argument_rule"],
            no_call_rule=payload["no_call_rule"],
            decision_rule=payload["decision_rule"],
            output_rule=payload["output_rule"],
            few_shot=payload.get("few_shot", ""),
        )

    def instructions(self) -> str:
        return "\n".join(
            item
            for item in [
                self.task_rule,
                self.schema_rule,
                self.argument_rule,
                self.no_call_rule,
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
                "name": "bfcl_function_calls",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "calls": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "arguments": {"type": "object", "additionalProperties": True},
                                },
                                "required": ["name", "arguments"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["calls"],
                    "additionalProperties": False,
                },
            },
        }


class BfclFunctionCallingRunner:
    def __init__(self, *, client: ResponsesModelClient) -> None:
        self.client = client

    def run_case(self, agent_config: dict[str, str], case: EvalCase) -> RunRecord:
        config = BfclAgentConfig.from_agent_config(agent_config)
        started_at = time.perf_counter()
        response = self.client.create_response(
            model=config.model,
            reasoning={"effort": config.reasoning_effort},
            instructions=config.instructions(),
            input=_case_prompt(case),
            max_output_tokens=config.output_cap,
            text=config.text_config(),
        )
        latency_s = time.perf_counter() - started_at
        raw_output_text = response.output_text.strip()
        payload = extract_json_payload(raw_output_text)
        parser_fallback = not _is_call_payload(payload)
        output = payload if _is_call_payload(payload) else _fallback_output(raw_output_text)
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


def _case_prompt(case: EvalCase) -> str:
    try:
        payload = json.loads(case.input)
    except json.JSONDecodeError as exc:
        raise ValueError("BFCL case input must be a JSON object string.") from exc
    if not isinstance(payload, dict):
        raise ValueError("BFCL case input must be a JSON object string.")
    return json.dumps(
        {
            "user_request": payload.get("question"),
            "available_functions": payload.get("functions", []),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _is_call_payload(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    calls = value.get("calls")
    if not isinstance(calls, list):
        return False
    return all(isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("arguments"), dict) for item in calls)


def _fallback_output(text: str) -> dict[str, Any]:
    payload = extract_json_payload(text)
    if _is_call_payload(payload):
        return payload
    if isinstance(payload, list):
        calls = []
        for item in payload:
            if isinstance(item, dict) and len(item) == 1:
                name, arguments = next(iter(item.items()))
                if isinstance(arguments, dict):
                    calls.append({"name": str(name), "arguments": arguments})
        if calls:
            return {"calls": calls}
    return {"calls": [], "invalid_output": text}
