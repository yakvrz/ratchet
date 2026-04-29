from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from ratchet.context_graph import ContextGraph, ContextSection
from ratchet.grading import extract_json_payload
from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.runtime import RuntimeContext, TransformRuntime
from ratchet.transform_program import CompiledCandidate
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
        return self.context_graph().render_text()

    def context_graph(self) -> ContextGraph:
        rows = [
            ("task_rule", self.task_rule),
            ("policy_rule", self.policy_rule),
            ("action_rule", self.action_rule),
            ("sequencing_rule", self.sequencing_rule),
            ("output_rule", self.output_rule),
            ("few_shot", self.few_shot),
        ]
        return ContextGraph(
            tuple(
                ContextSection(name=name, role="system", content=text, required=name != "few_shot")
                for name, text in rows
                if text
            )
        )

    def model_config(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.output_cap,
        }

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

    def run_case(
        self,
        agent_config: dict[str, str],
        case: EvalCase,
        candidate: CompiledCandidate | None = None,
    ) -> RunRecord:
        config = TauBenchActionConfig.from_agent_config(agent_config)
        runtime = TransformRuntime(candidate)
        ctx = RuntimeContext(
            case=case,
            context=config.context_graph(),
            model_config=config.model_config(),
        )
        runtime.run_hook("on_task_start", ctx)
        runtime.run_hook("before_model_call", ctx)
        started_at = time.perf_counter()
        response = self.client.create_response(
            model=str(ctx.model_config.get("model", config.model)),
            reasoning={"effort": str(ctx.model_config.get("reasoning_effort", config.reasoning_effort))},
            instructions=ctx.context.render_text(),
            input=case.input,
            max_output_tokens=int(ctx.model_config.get("max_tokens", config.output_cap)),
            text=config.text_config(),
        )
        latency_s = time.perf_counter() - started_at
        raw_output_text = response.output_text.strip()
        ctx.raw_response = raw_output_text
        runtime.run_hook("after_model_call", ctx)
        payload = extract_json_payload(raw_output_text)
        parser_fallback = not _is_action_payload(payload)
        output = payload if _is_action_payload(payload) else {"actions": [], "message": "", "invalid_output": raw_output_text}
        ctx.draft_response = output
        ctx.output = output
        runtime.run_hook("before_user_response", ctx)
        output = ctx.output
        runtime.run_hook("on_task_end", ctx)
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
                    "transform_candidate_id": candidate.program.candidate_id if candidate is not None else None,
                    "transform_compile_report": candidate.report.to_dict() if candidate is not None else None,
                    "transform_diff": candidate.diff.to_dict() if candidate is not None else None,
                    "transform_trace": list(ctx.trace_annotations),
                    "rendered_context_sections": ctx.context.section_names(),
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
