from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.rendering import render_few_shot_prompt
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord


@dataclass(frozen=True)
class KashiAgentConfig:
    model: str
    reasoning_effort: str
    instructions: list[str]
    output_cap: int

    @classmethod
    def from_spec(cls, spec: AgentSpec) -> "KashiAgentConfig":
        few_shot_prompt = render_few_shot_prompt(spec.few_shot)
        return cls(
            model=spec.model,
            reasoning_effort=str(spec.runtime.get("reasoning_effort", "none")),
            instructions=[
                *[text for _, text in sorted(spec.instructions.items()) if text],
                *([few_shot_prompt] if few_shot_prompt else []),
            ],
            output_cap=int(spec.runtime.get("output_cap", 160)),
        )


class KashiAgentRunner:
    def __init__(self, env_path: str | None = None, client: ResponsesModelClient | None = None) -> None:
        resolved_env = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self.client = client or ResponsesModelClient(env_path=resolved_env)

    def run_case(self, spec: AgentSpec, case: EvalCase) -> RunRecord:
        config = KashiAgentConfig.from_spec(spec)
        messages = json.loads(case.input)
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError("Kashi eval input must be a JSON array with instructions, mappings, and conversation.")

        mapping_message = str(messages[1].get("content", ""))
        conversation = [
            {
                "role": "assistant" if item.get("role") == "assistant" else "user",
                "content": str(item.get("content", "")),
            }
            for item in messages[2:]
            if item.get("role") in {"assistant", "user"}
        ]
        instructions = "\n\n".join(part for part in [*config.instructions, mapping_message] if part)

        started_at = time.perf_counter()
        request: dict[str, Any] = {
            "model": config.model,
            "instructions": instructions,
            "input": conversation,
            "max_output_tokens": config.output_cap,
        }
        if config.model.startswith("gpt-5"):
            request["reasoning"] = {"effort": config.reasoning_effort}
            request["text"] = {"verbosity": "low"}
        response = self.client.create_response(**request)
        raw_output_text = response.output_text.strip()
        usage = response.usage
        latency_s = time.perf_counter() - started_at
        total_tokens = usage.input_tokens + usage.output_tokens
        return RunRecord(
            output=raw_output_text,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=total_tokens,
                cost_usd=estimate_cost_usd(config.model, usage.input_tokens, usage.output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                raw_output_text=raw_output_text,
                metadata={
                    "model": config.model,
                    "response_ids": [response.id],
                },
            ),
        )
