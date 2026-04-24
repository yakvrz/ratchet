from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import operator
import re
import time
from typing import Any

from ratchet.benchmark import KNOWLEDGE_BASE
from ratchet.openai_client import OpenAIResponsesClient
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord


MODEL_PRICING_USD_PER_TOKEN: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 2.50 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-5.4-mini": {"input": 0.75 / 1_000_000, "output": 4.50 / 1_000_000},
    "gpt-5.4-nano": {"input": 0.20 / 1_000_000, "output": 1.25 / 1_000_000},
}


@dataclass(frozen=True)
class HarnessConfig:
    model: str
    reasoning_effort: str = "none"
    kb_tool_enabled: str = "off"
    calculator_tool_enabled: str = "off"
    kb_tool_description: str = ""
    calculator_tool_description: str = ""
    knowledge_mode: str = "raw"
    prompt_identity_rule: str = ""
    prompt_answer_rule: str = ""
    prompt_kb_rule: str = ""
    prompt_calc_rule: str = ""
    prompt_fallback_rule: str = ""
    output_cap: int = 120
    max_tool_rounds: int = 4

    @classmethod
    def from_candidate(cls, candidate: dict[str, str]) -> "HarnessConfig":
        return cls(
            model=candidate["model"],
            reasoning_effort=candidate["reasoning_effort"],
            kb_tool_enabled=candidate["kb_tool_enabled"],
            calculator_tool_enabled=candidate["calculator_tool_enabled"],
            kb_tool_description=candidate["kb_tool_description"],
            calculator_tool_description=candidate["calculator_tool_description"],
            knowledge_mode=candidate["knowledge_mode"],
            prompt_identity_rule=candidate["prompt_identity_rule"],
            prompt_answer_rule=candidate["prompt_answer_rule"],
            prompt_kb_rule=candidate["prompt_kb_rule"],
            prompt_calc_rule=candidate["prompt_calc_rule"],
            prompt_fallback_rule=candidate["prompt_fallback_rule"],
            output_cap=int(candidate.get("output_cap", "120")),
            max_tool_rounds=int(candidate.get("max_tool_rounds", "4")),
        )

    @property
    def use_kb_tool(self) -> bool:
        return self.kb_tool_enabled == "on"

    @property
    def use_calculator(self) -> bool:
        return self.calculator_tool_enabled == "on"

    def system_prompt(self) -> str:
        lines = [
            "You answer benchmark questions about the fictional Northstar Fulfillment handbook.",
            self.prompt_identity_rule,
            self.prompt_answer_rule,
            self.prompt_fallback_rule,
        ]
        if self.use_kb_tool:
            lines.append(self.prompt_kb_rule)
        if self.use_calculator:
            lines.append(self.prompt_calc_rule)
        if self.use_kb_tool:
            if self.knowledge_mode == "distilled":
                lines.append("The knowledge tool searches distilled handbook cards; prefer exact card values.")
            elif self.knowledge_mode == "raw":
                lines.append("The knowledge tool searches the verbose handbook; use retrieved text, not outside knowledge.")
            else:
                raise ValueError(f"Unsupported knowledge mode: {self.knowledge_mode}")
        return " ".join(line for line in lines if line)


class SafeCalculator:
    _binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    _unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @classmethod
    def evaluate(cls, expression: str) -> str:
        tree = ast.parse(expression, mode="eval")
        value = cls._eval(tree.body)
        return cls._format(value)

    @classmethod
    def _eval(cls, node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in cls._binary_ops:
            left = cls._eval(node.left)
            right = cls._eval(node.right)
            return float(cls._binary_ops[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in cls._unary_ops:
            return float(cls._unary_ops[type(node.op)](cls._eval(node.operand)))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    @staticmethod
    def _format(value: float) -> str:
        rounded = round(value, 4)
        return f"{rounded:.4f}".rstrip("0").rstrip(".")


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_TOKEN[model]
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def search_knowledge(query: str, mode: str) -> str:
    query_terms = tokenize(query)
    scored_docs: list[tuple[int, str, str]] = []
    use_distilled = mode == "distilled"
    for doc in KNOWLEDGE_BASE:
        candidate_text = doc.distilled if use_distilled else doc.body
        score = len(query_terms & tokenize(f"{doc.title} {candidate_text}"))
        scored_docs.append((score, doc.title, candidate_text))
    scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_k = 3 if use_distilled else 2
    return "\n\n".join(f"[{title}]\n{text}" for _, title, text in scored_docs[:top_k])


class NorthstarHarnessRunner:
    def __init__(self, env_path: str = ".env", client: OpenAIResponsesClient | None = None) -> None:
        self.client = client or OpenAIResponsesClient(env_path=env_path)

    def _build_tools(self, config: HarnessConfig) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = []
        if config.use_kb_tool:
            tools.append(
                {
                    "type": "function",
                    "name": "kb_lookup",
                    "description": config.kb_tool_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                }
            )
        if config.use_calculator:
            tools.append(
                {
                    "type": "function",
                    "name": "calculator",
                    "description": config.calculator_tool_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"},
                        },
                        "required": ["expression"],
                        "additionalProperties": False,
                    },
                }
            )
        return tools

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        config = HarnessConfig.from_candidate(candidate)
        tools = self._build_tools(config)
        tool_calls: list[str] = []
        response_ids: list[str] = []
        output_item_types: list[list[str]] = []
        total_input_tokens = 0
        total_output_tokens = 0

        started_at = time.perf_counter()
        response = self.client.create_response(
            model=config.model,
            reasoning={"effort": config.reasoning_effort},
            instructions=config.system_prompt(),
            input=case.input,
            max_output_tokens=config.output_cap,
            tools=tools or [],
        )

        for _ in range(config.max_tool_rounds + 1):
            response_ids.append(response.id)
            output_item_types.append([item.type for item in response.output])
            usage = response.usage
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens

            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                break

            tool_outputs: list[dict[str, str]] = []
            for function_call in function_calls:
                arguments = json.loads(function_call.arguments)
                if function_call.name == "kb_lookup":
                    output = search_knowledge(arguments["query"], config.knowledge_mode)
                elif function_call.name == "calculator":
                    output = SafeCalculator.evaluate(arguments["expression"])
                else:
                    raise ValueError(f"Unsupported tool: {function_call.name}")
                tool_calls.append(function_call.name)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": output,
                    }
                )

            response = self.client.create_response(
                model=config.model,
                reasoning={"effort": config.reasoning_effort},
                previous_response_id=response.id,
                input=tool_outputs,
                max_output_tokens=config.output_cap,
                tools=tools or [],
            )

        raw_output_text = response.output_text.strip()
        latency_s = time.perf_counter() - started_at
        total_tokens = total_input_tokens + total_output_tokens
        return RunRecord(
            output=raw_output_text,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_tokens,
                cost_usd=estimate_cost_usd(config.model, total_input_tokens, total_output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=raw_output_text,
                metadata={
                    "model": config.model,
                    "response_ids": response_ids,
                    "output_item_types": output_item_types,
                },
            ),
        )
