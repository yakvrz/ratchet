from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any

from ratchet.grading import extract_json_payload
from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord

try:
    from docs_corpus import POLICY_DOCS, tokenize
except ModuleNotFoundError:
    from .docs_corpus import POLICY_DOCS, tokenize


@dataclass(frozen=True)
class PolicyTriageConfig:
    model: str
    reasoning_effort: str
    prompt_output_rule: str
    prompt_grounding_rule: str
    prompt_tool_rule: str
    prompt_fallback_rule: str
    prompt_few_shot: str
    decision_validator_enabled: str
    decision_validator_rule: str
    policy_search_enabled: str
    policy_search_description: str
    knowledge_mode: str
    retrieval_top_k: int
    output_cap: int
    max_tool_rounds: int

    @classmethod
    def from_agent_config(cls, agent_config: dict[str, str]) -> "PolicyTriageConfig":
        return cls(
            model=agent_config["model"],
            reasoning_effort=agent_config["reasoning_effort"],
            prompt_output_rule=agent_config["prompt_output_rule"],
            prompt_grounding_rule=agent_config["prompt_grounding_rule"],
            prompt_tool_rule=agent_config["prompt_tool_rule"],
            prompt_fallback_rule=agent_config["prompt_fallback_rule"],
            prompt_few_shot=agent_config.get("prompt_few_shot", ""),
            decision_validator_enabled=agent_config["decision_validator_enabled"],
            decision_validator_rule=agent_config["decision_validator_rule"],
            policy_search_enabled=agent_config["policy_search_enabled"],
            policy_search_description=agent_config["policy_search_description"],
            knowledge_mode=agent_config["knowledge_mode"],
            retrieval_top_k=int(agent_config["retrieval_top_k"]),
            output_cap=int(agent_config["output_cap"]),
            max_tool_rounds=int(agent_config.get("max_tool_rounds", "1")),
        )

    @property
    def use_policy_search(self) -> bool:
        return self.policy_search_enabled == "on"

    @property
    def use_validator(self) -> bool:
        return self.decision_validator_enabled == "on"

    def system_prompt(self) -> str:
        lines = [
            "You triage reimbursement requests against a frozen internal expense policy snapshot.",
            self.prompt_output_rule,
            self.prompt_grounding_rule,
            self.prompt_fallback_rule,
            self.prompt_few_shot,
        ]
        if self.use_policy_search:
            lines.append(self.prompt_tool_rule)
        if self.use_validator:
            lines.append("The agent may apply a post-answer validation rule before returning the final decision.")
        return " ".join(line for line in lines if line)

    def text_config(self) -> dict[str, Any]:
        return {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "policy_triage_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "decision": {
                            "type": "string",
                            "enum": ["approve", "deny", "escalate"],
                        },
                        "amount": {"type": "integer"},
                    },
                    "required": ["decision", "amount"],
                    "additionalProperties": False,
                },
            },
        }


def search_policy(query: str, mode: str, top_k: int) -> str:
    query_terms = tokenize(query)
    use_distilled = mode == "distilled"
    scored: list[tuple[int, str, str, str]] = []
    for doc in POLICY_DOCS:
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


class PolicyTriageRunner:
    def __init__(self, env_path: str | None = None, client: ResponsesModelClient | None = None) -> None:
        resolved_env = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self.client = client or ResponsesModelClient(env_path=resolved_env)

    def _build_tools(self, config: PolicyTriageConfig) -> list[dict[str, Any]]:
        if not config.use_policy_search:
            return []
        return [
            {
                "type": "function",
                "name": "policy_search",
                "description": config.policy_search_description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        ]

    def run_case(
        self,
        agent_config: dict[str, str],
        case: EvalCase,
        hooks: dict[str, Any] | None = None,
    ) -> RunRecord:
        if os.environ.get("RATCHET_OFFLINE_MODE") == "1":
            return self._run_case_offline(agent_config, case, hooks=hooks)
        config = PolicyTriageConfig.from_agent_config(agent_config)
        hooks = hooks or {}
        tools = self._build_tools(config)
        tool_calls: list[str] = []
        retrieved_cards: list[dict[str, str]] = []
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

        tool_rounds = 0
        while True:
            response_ids.append(response.id)
            output_item_types.append([item.type for item in response.output])
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
            function_calls = [item for item in response.output if item.type == "function_call"]
            if not function_calls:
                break
            if tool_rounds >= config.max_tool_rounds:
                raise RuntimeError("policy_search tool round budget exhausted before a final answer.")
            tool_rounds += 1
            tool_outputs: list[dict[str, str]] = []
            for function_call in function_calls:
                arguments = json.loads(function_call.arguments)
                if function_call.name != "policy_search":
                    raise ValueError(f"Unsupported tool: {function_call.name}")
                tool_calls.append(function_call.name)
                query = str(arguments["query"])
                if "pre_tool_query_hook" in hooks:
                    query = str(
                        hooks["pre_tool_query_hook"](
                            query,
                            {
                                "case_input": case.input,
                                "agent_config": agent_config,
                                "tool_name": function_call.name,
                            },
                        )
                    )
                search_output = search_policy(query=query, mode=config.knowledge_mode, top_k=config.retrieval_top_k)
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
                                {
                                    "case_input": case.input,
                                    "agent_config": agent_config,
                                    "query": query,
                                },
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
        output: Any = payload if payload is not None else {"decision": "escalate", "amount": 0}
        if config.use_validator and "post_answer_validator_hook" in hooks:
            hook_output = hooks["post_answer_validator_hook"](
                output,
                {
                    "case_input": case.input,
                    "agent_config": agent_config,
                    "retrieved_cards": list(retrieved_cards),
                    "validator_rule": config.decision_validator_rule,
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
                cost_usd=estimate_cost_usd(config.model, total_input_tokens, total_output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=raw_output_text,
                metadata={
                    "model": config.model,
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
        config = PolicyTriageConfig.from_agent_config(agent_config)
        hooks = hooks or {}
        started_at = time.perf_counter()
        tool_calls: list[str] = []
        retrieved_cards: list[dict[str, str]] = []
        if config.use_policy_search:
            tool_calls.append("policy_search")
            query = case.input
            if "pre_tool_query_hook" in hooks:
                query = str(
                    hooks["pre_tool_query_hook"](
                        query,
                        {"case_input": case.input, "agent_config": agent_config, "tool_name": "policy_search"},
                    )
                )
            search_output = search_policy(query=query, mode=config.knowledge_mode, top_k=config.retrieval_top_k)
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

        output = self._offline_decide(case.input, retrieved_cards)
        if config.use_validator and "post_answer_validator_hook" in hooks:
            hook_output = hooks["post_answer_validator_hook"](
                output,
                {
                    "case_input": case.input,
                    "agent_config": agent_config,
                    "retrieved_cards": list(retrieved_cards),
                    "validator_rule": config.decision_validator_rule,
                },
            )
            if hook_output is not None:
                output = hook_output

        total_tokens = self._offline_total_tokens(config, retrieved_cards)
        latency_s = time.perf_counter() - started_at
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=total_tokens // 2,
                output_tokens=total_tokens - (total_tokens // 2),
                total_tokens=total_tokens,
                cost_usd=estimate_cost_usd(config.model, total_tokens // 2, total_tokens - (total_tokens // 2)),
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=json.dumps(output, sort_keys=True),
                metadata={
                    "mode": "offline",
                    "model": config.model,
                    "retrieved_doc_ids": [card.get("doc_id", "") for card in retrieved_cards],
                },
            ),
        )

    @staticmethod
    def _offline_total_tokens(config: PolicyTriageConfig, retrieved_cards: list[dict[str, str]]) -> int:
        base = 220
        model_delta = 80 if config.model == "gpt-5.4" else 0
        reasoning_delta = 40 if config.reasoning_effort == "low" else 0
        retrieval_delta = 30 * len(retrieved_cards)
        output_delta = max(config.output_cap // 2, 20)
        return base + model_delta + reasoning_delta + retrieval_delta + output_delta

    @staticmethod
    def _extract_amount(case_input: str) -> int:
        match = re.search(r"Amount:\s*(\d+)\s*USD", case_input)
        if not match:
            return 0
        return int(match.group(1))

    def _offline_decide(self, case_input: str, retrieved_cards: list[dict[str, str]]) -> dict[str, Any]:
        lowered = case_input.lower()
        amount = self._extract_amount(case_input)
        doc_ids = {card.get("doc_id", "") for card in retrieved_cards}
        if "gift_cards" in doc_ids or "gift card" in lowered:
            return {"decision": "deny", "amount": 0}
        if "commuting" in doc_ids or "commute" in lowered:
            return {"decision": "deny", "amount": 0}
        if "training" in doc_ids:
            if "without preapproval" in lowered:
                return {"decision": "escalate", "amount": 0}
            return {"decision": "approve", "amount": amount}
        if "lodging" in doc_ids:
            if amount > 220:
                return {"decision": "escalate", "amount": 0}
            return {"decision": "approve", "amount": amount}
        if "saas_tools" in doc_ids:
            return {"decision": "escalate", "amount": 0}
        if "home_office" in doc_ids:
            if amount <= 300 and ("after 26 months" in lowered or "after 30 months" in lowered):
                return {"decision": "approve", "amount": amount}
            return {"decision": "escalate", "amount": 0}
        if "rideshare" in doc_ids:
            if "airport" in lowered or "client" in lowered:
                return {"decision": "approve", "amount": amount}
            return {"decision": "deny", "amount": 0}
        if "travel_meals" in doc_ids:
            return {"decision": "approve", "amount": amount}
        return {"decision": "escalate", "amount": 0}
