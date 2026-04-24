from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

from ratchet.grading import extract_json_payload
from ratchet.harness import estimate_cost_usd
from ratchet.openai_client import OpenAIResponsesClient
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord
from docs_corpus import PUBLIC_DOCS, tokenize


@dataclass(frozen=True)
class DocsHarnessConfig:
    model: str
    reasoning_effort: str
    prompt_output_rule: str
    prompt_grounding_rule: str
    prompt_tool_rule: str
    prompt_fallback_rule: str
    docs_search_enabled: str
    docs_search_description: str
    knowledge_mode: str
    retrieval_top_k: int
    output_cap: int
    max_tool_rounds: int

    @classmethod
    def from_candidate(cls, candidate: dict[str, str]) -> "DocsHarnessConfig":
        return cls(
            model=candidate["model"],
            reasoning_effort=candidate["reasoning_effort"],
            prompt_output_rule=candidate["prompt_output_rule"],
            prompt_grounding_rule=candidate["prompt_grounding_rule"],
            prompt_tool_rule=candidate["prompt_tool_rule"],
            prompt_fallback_rule=candidate["prompt_fallback_rule"],
            docs_search_enabled=candidate["docs_search_enabled"],
            docs_search_description=candidate["docs_search_description"],
            knowledge_mode=candidate["knowledge_mode"],
            retrieval_top_k=int(candidate["retrieval_top_k"]),
            output_cap=int(candidate["output_cap"]),
            max_tool_rounds=int(candidate.get("max_tool_rounds", "4")),
        )

    @property
    def use_docs_search(self) -> bool:
        return self.docs_search_enabled == "on"

    def system_prompt(self) -> str:
        lines = [
            "You answer questions about a frozen public Python docs snapshot.",
            self.prompt_output_rule,
            self.prompt_grounding_rule,
            self.prompt_fallback_rule,
        ]
        if self.use_docs_search:
            lines.append(self.prompt_tool_rule)
        if self.knowledge_mode == "distilled":
            lines.append("The search tool returns short distilled cards; prefer exact literals from the card text.")
        elif self.knowledge_mode == "raw":
            lines.append("The search tool returns fuller docs snippets; copy exact literals from those snippets.")
        else:
            raise ValueError(f"Unsupported knowledge mode: {self.knowledge_mode}")
        return " ".join(line for line in lines if line)

    def text_config(self) -> dict[str, Any]:
        return {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "docs_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                    },
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }


def search_docs(query: str, mode: str, top_k: int) -> str:
    query_terms = tokenize(query)
    use_distilled = mode == "distilled"
    scored_docs: list[tuple[int, str, str, str]] = []
    for doc in PUBLIC_DOCS:
        text = doc.distilled if use_distilled else doc.body
        score = len(query_terms & tokenize(f"{doc.doc_id} {doc.title} {text}"))
        scored_docs.append((score, doc.doc_id, doc.title, text))
    scored_docs.sort(key=lambda item: (item[0], item[1]), reverse=True)
    payload = [
        {"doc_id": doc_id, "title": title, "text": text}
        for _, doc_id, title, text in scored_docs[:top_k]
    ]
    return json.dumps(payload, indent=2, sort_keys=True)


class PublicDocsAgentRunner:
    def __init__(self, env_path: str | None = None, client: OpenAIResponsesClient | None = None) -> None:
        resolved_env = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self.client = client or OpenAIResponsesClient(env_path=resolved_env)

    def _build_tools(self, config: DocsHarnessConfig) -> list[dict[str, Any]]:
        if not config.use_docs_search:
            return []
        return [
            {
                "type": "function",
                "name": "docs_search",
                "description": config.docs_search_description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        config = DocsHarnessConfig.from_candidate(candidate)
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
            text=config.text_config(),
            tools=tools,
        )

        for _ in range(config.max_tool_rounds + 1):
            response_ids.append(response.id)
            output_item_types.append([item.type for item in response.output])
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                break
            tool_outputs: list[dict[str, str]] = []
            for function_call in function_calls:
                arguments = json.loads(function_call.arguments)
                if function_call.name != "docs_search":
                    raise ValueError(f"Unsupported tool: {function_call.name}")
                tool_calls.append(function_call.name)
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": search_docs(
                            query=str(arguments["query"]),
                            mode=config.knowledge_mode,
                            top_k=config.retrieval_top_k,
                        ),
                    }
                )
            response = self.client.create_response(
                model=config.model,
                reasoning={"effort": config.reasoning_effort},
                previous_response_id=response.id,
                input=tool_outputs,
                max_output_tokens=config.output_cap,
                text=config.text_config(),
                tools=tools,
            )

        raw_output_text = response.output_text.strip()
        payload = extract_json_payload(raw_output_text)
        output: Any = payload if payload is not None else {"answer": "unknown", "invalid_output": raw_output_text}
        latency_s = time.perf_counter() - started_at
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=estimate_cost_usd(config.model, total_input_tokens, total_output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=raw_output_text,
                metadata={
                    "response_ids": response_ids,
                    "output_item_types": output_item_types,
                    "model": config.model,
                },
            ),
        )
