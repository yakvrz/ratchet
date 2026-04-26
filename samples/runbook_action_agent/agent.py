from __future__ import annotations

import json
import os
import time
from typing import Any

from ratchet.grading import extract_json_payload
from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord

try:
    from docs_corpus import RUNBOOK_DOCS, tokenize
except ModuleNotFoundError:
    from .docs_corpus import RUNBOOK_DOCS, tokenize


class RunbookActionRunner:
    def __init__(self, env_path: str | None = None, client: ResponsesModelClient | None = None) -> None:
        resolved_env = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self.client = client or ResponsesModelClient(env_path=resolved_env)

    def _build_tools(self, agent_config: dict[str, str]) -> list[dict[str, Any]]:
        if agent_config["runbook_search_enabled"] != "on":
            return []
        return [
            {
                "type": "function",
                "name": "runbook_search",
                "description": agent_config["runbook_search_description"],
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    def _system_prompt(self, agent_config: dict[str, str]) -> str:
        lines = [
            "You choose the next runbook action from a frozen incident-response snapshot.",
            agent_config["prompt_output_rule"],
            agent_config["prompt_grounding_rule"],
            agent_config["prompt_fallback_rule"],
            agent_config.get("prompt_few_shot", ""),
        ]
        if agent_config["runbook_search_enabled"] == "on":
            lines.append(agent_config["prompt_tool_rule"])
        if agent_config["answer_validator_enabled"] == "on":
            lines.append("If an answer is not directly supported by retrieved runbook evidence, the agent may replace it with unknown.")
        if agent_config["knowledge_mode"] == "distilled":
            lines.append("The search tool returns short distilled runbook cards.")
        else:
            lines.append("The search tool returns fuller runbook snippets.")
        return " ".join(line for line in lines if line)

    def _text_config(self) -> dict[str, Any]:
        return {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "runbook_action_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
        }

    def run_case(
        self,
        agent_config: dict[str, str],
        case: EvalCase,
        hooks: dict[str, Any] | None = None,
    ) -> RunRecord:
        if os.environ.get("RATCHET_OFFLINE_MODE") == "1":
            return self._run_case_offline(agent_config, case, hooks=hooks)
        hooks = hooks or {}
        tools = self._build_tools(agent_config)
        tool_calls: list[str] = []
        retrieved_cards: list[dict[str, str]] = []
        response_ids: list[str] = []
        output_item_types: list[list[str]] = []
        total_input_tokens = 0
        total_output_tokens = 0
        started_at = time.perf_counter()

        response = self.client.create_response(
            model=agent_config["model"],
            reasoning={"effort": agent_config["reasoning_effort"]},
            instructions=self._system_prompt(agent_config),
            input=case.input,
            max_output_tokens=int(agent_config["output_cap"]),
            text=self._text_config(),
            tools=tools,
        )

        max_tool_rounds = int(agent_config.get("max_tool_rounds", "4"))
        tool_rounds = 0
        while True:
            response_ids.append(response.id)
            output_item_types.append([item.type for item in response.output])
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                break
            if tool_rounds >= max_tool_rounds:
                raise RuntimeError("runbook_search tool round budget exhausted before a final answer.")
            tool_rounds += 1
            tool_outputs: list[dict[str, str]] = []
            for function_call in function_calls:
                arguments = json.loads(function_call.arguments)
                if function_call.name != "runbook_search":
                    raise ValueError(f"Unsupported tool: {function_call.name}")
                tool_calls.append(function_call.name)
                query = str(arguments["query"])
                if "pre_tool_query_hook" in hooks:
                    query = str(
                        hooks["pre_tool_query_hook"](
                            query,
                            {"case_input": case.input, "agent_config": agent_config, "tool_name": function_call.name},
                        )
                    )
                search_output = search_runbooks(
                    query=query,
                    mode=agent_config["knowledge_mode"],
                    top_k=int(agent_config["retrieval_top_k"]),
                )
                try:
                    cards = json.loads(search_output)
                    if isinstance(cards, list):
                        retrieved_cards.extend(
                            {
                                "doc_id": str(card.get("doc_id", "")),
                                "title": str(card.get("title", "")),
                                "text": str(card.get("text", "")),
                            }
                            for card in cards
                            if isinstance(card, dict)
                        )
                        if "post_tool_context_hook" in hooks:
                            hook_result = hooks["post_tool_context_hook"](
                                list(retrieved_cards),
                                {"case_input": case.input, "agent_config": agent_config, "query": query},
                            )
                            if isinstance(hook_result, list):
                                retrieved_cards = [
                                    {
                                        "doc_id": str(card.get("doc_id", "")),
                                        "title": str(card.get("title", "")),
                                        "text": str(card.get("text", "")),
                                    }
                                    for card in hook_result
                                    if isinstance(card, dict)
                                ]
                                search_output = json.dumps(retrieved_cards, indent=2, sort_keys=True)
                except json.JSONDecodeError:
                    pass
                tool_outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": search_output,
                    }
                )
            response = self.client.create_response(
                model=agent_config["model"],
                reasoning={"effort": agent_config["reasoning_effort"]},
                previous_response_id=response.id,
                input=tool_outputs,
                max_output_tokens=int(agent_config["output_cap"]),
                text=self._text_config(),
                tools=tools,
            )

        raw_output_text = response.output_text.strip()
        payload = extract_json_payload(raw_output_text)
        output: Any = payload if payload is not None else {"answer": "unknown", "invalid_output": raw_output_text}
        if agent_config["answer_validator_enabled"] == "on" and "post_answer_validator_hook" in hooks:
            hook_output = hooks["post_answer_validator_hook"](
                output,
                {
                    "case_input": case.input,
                    "agent_config": agent_config,
                    "retrieved_cards": list(retrieved_cards),
                    "option_literals": list(case.metadata.get("options", [])),
                    "validator_rule": agent_config["answer_validator_rule"],
                },
            )
            if hook_output is not None:
                output = hook_output

        latency_s = time.perf_counter() - started_at
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_tokens=total_input_tokens + total_output_tokens,
                cost_usd=estimate_cost_usd(agent_config["model"], total_input_tokens, total_output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=raw_output_text,
                metadata={
                    "model": agent_config["model"],
                    "response_ids": response_ids,
                    "output_item_types": output_item_types,
                    "retrieved_doc_ids": [card.get("doc_id", "") for card in retrieved_cards],
                },
            ),
        )

    def _run_case_offline(
        self,
        agent_config: dict[str, str],
        case: EvalCase,
        hooks: dict[str, Any] | None = None,
    ) -> RunRecord:
        hooks = hooks or {}
        started_at = time.perf_counter()
        tool_calls: list[str] = []
        retrieved_cards: list[dict[str, str]] = []
        if agent_config["runbook_search_enabled"] == "on":
            tool_calls.append("runbook_search")
            query = case.input
            if "pre_tool_query_hook" in hooks:
                query = str(
                    hooks["pre_tool_query_hook"](
                        query,
                        {"case_input": case.input, "agent_config": agent_config, "tool_name": "runbook_search"},
                    )
                )
            search_output = search_runbooks(
                query=query,
                mode=agent_config["knowledge_mode"],
                top_k=int(agent_config["retrieval_top_k"]),
            )
            retrieved_cards = json.loads(search_output)
            if "post_tool_context_hook" in hooks:
                hook_result = hooks["post_tool_context_hook"](
                    list(retrieved_cards),
                    {"case_input": case.input, "agent_config": agent_config, "query": query},
                )
                if isinstance(hook_result, list):
                    retrieved_cards = [
                        {
                            "doc_id": str(card.get("doc_id", "")),
                            "title": str(card.get("title", "")),
                            "text": str(card.get("text", "")),
                        }
                        for card in hook_result
                        if isinstance(card, dict)
                    ]

        output = self._offline_answer(case, agent_config, retrieved_cards)
        if agent_config["answer_validator_enabled"] == "on" and "post_answer_validator_hook" in hooks:
            hook_output = hooks["post_answer_validator_hook"](
                output,
                {
                    "case_input": case.input,
                    "agent_config": agent_config,
                    "retrieved_cards": list(retrieved_cards),
                    "option_literals": list(case.metadata.get("options", [])),
                    "validator_rule": agent_config["answer_validator_rule"],
                },
            )
            if hook_output is not None:
                output = hook_output

        total_tokens = self._offline_total_tokens(agent_config, retrieved_cards)
        latency_s = time.perf_counter() - started_at
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=total_tokens // 2,
                output_tokens=total_tokens - (total_tokens // 2),
                total_tokens=total_tokens,
                cost_usd=estimate_cost_usd(agent_config["model"], total_tokens // 2, total_tokens - (total_tokens // 2)),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=json.dumps(output, sort_keys=True),
                metadata={
                    "mode": "offline",
                    "model": agent_config["model"],
                    "retrieved_doc_ids": [card.get("doc_id", "") for card in retrieved_cards],
                },
            ),
        )

    @staticmethod
    def _offline_total_tokens(agent_config: dict[str, str], retrieved_cards: list[dict[str, str]]) -> int:
        base = 240
        model_delta = 90 if agent_config["model"] == "gpt-5.4" else 20
        reasoning_delta = 45 if agent_config["reasoning_effort"] == "low" else 0
        retrieval_delta = 28 * len(retrieved_cards)
        output_delta = max(int(agent_config["output_cap"]) // 2, 20)
        return base + model_delta + reasoning_delta + retrieval_delta + output_delta

    @staticmethod
    def _offline_answer(
        case: EvalCase,
        agent_config: dict[str, str],
        retrieved_cards: list[dict[str, str]],
    ) -> dict[str, str]:
        options = [str(option) for option in case.metadata.get("options", [])]
        haystack = "\n".join(
            f"{card.get('doc_id', '')} {card.get('title', '')} {card.get('text', '')}".lower()
            for card in retrieved_cards
        )
        grounded = [option for option in options if option.lower() != "unknown" and option.lower() in haystack]
        if grounded:
            return {"answer": grounded[0]}
        if "best action" in agent_config["prompt_fallback_rule"].lower():
            for option in options:
                if option.lower() != "unknown":
                    return {"answer": option}
        return {"answer": "unknown"}


def search_runbooks(query: str, mode: str, top_k: int) -> str:
    query_terms = tokenize(query)
    use_distilled = mode == "distilled"
    scored: list[tuple[int, str, str, str]] = []
    for doc in RUNBOOK_DOCS:
        text = doc.distilled if use_distilled else doc.body
        score = len(query_terms & tokenize(f"{doc.doc_id} {doc.title} {text}"))
        if score > 0:
            scored.append((score, doc.doc_id, doc.title, text))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    payload = [
        {"doc_id": doc_id, "title": title, "text": text}
        for _, doc_id, title, text in scored[:top_k]
    ]
    return json.dumps(payload, indent=2, sort_keys=True)
