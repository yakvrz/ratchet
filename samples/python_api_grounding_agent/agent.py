from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any

from ratchet.grading import extract_json_payload
from ratchet.harness import estimate_cost_usd
from ratchet.openai_client import OpenAIResponsesClient
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord

from docs_corpus import PUBLIC_DOCS, tokenize


@dataclass(frozen=True)
class GroundingHarnessConfig:
    model: str
    reasoning_effort: str
    prompt_output_rule: str
    prompt_grounding_rule: str
    prompt_tool_rule: str
    prompt_fallback_rule: str
    answer_validator_enabled: str
    answer_validator_rule: str
    docs_search_enabled: str
    docs_search_description: str
    knowledge_mode: str
    retrieval_top_k: int
    output_cap: int
    max_tool_rounds: int

    @classmethod
    def from_candidate(cls, candidate: dict[str, str]) -> "GroundingHarnessConfig":
        return cls(
            model=candidate["model"],
            reasoning_effort=candidate["reasoning_effort"],
            prompt_output_rule=candidate["prompt_output_rule"],
            prompt_grounding_rule=candidate["prompt_grounding_rule"],
            prompt_tool_rule=candidate["prompt_tool_rule"],
            prompt_fallback_rule=candidate["prompt_fallback_rule"],
            answer_validator_enabled=candidate["answer_validator_enabled"],
            answer_validator_rule=candidate["answer_validator_rule"],
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

    @property
    def use_answer_validator(self) -> bool:
        return self.answer_validator_enabled == "on"

    def system_prompt(self) -> str:
        lines = [
            "You answer questions about a frozen public Python API snapshot.",
            self.prompt_output_rule,
            self.prompt_grounding_rule,
            self.prompt_fallback_rule,
        ]
        if self.use_docs_search:
            lines.append(self.prompt_tool_rule)
        if self.use_answer_validator:
            lines.append("If an answer is not directly supported by retrieved evidence, the harness may replace it with unknown.")
        if self.knowledge_mode == "distilled":
            lines.append("The search tool returns short distilled cards. Prefer exact literals from those cards only.")
        elif self.knowledge_mode == "raw":
            lines.append("The search tool returns fuller snippets. Prefer exact literals from those snippets only.")
        else:
            raise ValueError(f"Unsupported knowledge mode: {self.knowledge_mode}")
        return " ".join(line for line in lines if line)

    def text_config(self) -> dict[str, Any]:
        return {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "grounded_api_answer",
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


ANSWER_TOKEN_PATTERN = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*\.)+[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?|"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\(\)|=True|=False)?"
)


def canonicalize_answer_token(text: str) -> str:
    stripped = text.strip().replace("`", "")
    lowered = stripped.lower()
    if lowered == "unknown":
        return "unknown"
    match = ANSWER_TOKEN_PATTERN.search(stripped)
    if match:
        return match.group(0)
    return stripped.rstrip(".")


def extract_case_options(case_input: str) -> list[str]:
    if ":" not in case_input:
        return []
    raw_options = case_input.rsplit(":", 1)[-1]
    return [item.strip().rstrip(".") for item in raw_options.split(",") if item.strip()]


def evidence_contains_option(option: str, retrieved_cards: list[dict[str, str]]) -> bool:
    lowered = canonicalize_answer_token(option).lower()
    if not lowered or lowered == "unknown":
        return False
    haystack = "\n".join(
        f"{card.get('doc_id', '')} {card.get('title', '')} {card.get('text', '')}".lower()
        for card in retrieved_cards
    )
    return lowered in haystack


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


class PythonApiGroundingRunner:
    def __init__(self, env_path: str | None = None, client: OpenAIResponsesClient | None = None) -> None:
        resolved_env = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self.client = client or OpenAIResponsesClient(env_path=resolved_env)

    def _build_tools(self, config: GroundingHarnessConfig) -> list[dict[str, Any]]:
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

    def run_case(
        self,
        candidate: dict[str, str],
        case: EvalCase,
        hooks: dict[str, Any] | None = None,
    ) -> RunRecord:
        config = GroundingHarnessConfig.from_candidate(candidate)
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
                query = str(arguments["query"])
                if config.use_docs_search and "pre_tool_query_hook" in hooks:
                    query = str(
                        hooks["pre_tool_query_hook"](
                            query,
                            {
                                "case_input": case.input,
                                "candidate": candidate,
                                "tool_name": function_call.name,
                            },
                        )
                    )
                search_output = search_docs(
                    query=query,
                    mode=config.knowledge_mode,
                    top_k=config.retrieval_top_k,
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
                        if config.use_docs_search and "post_tool_context_hook" in hooks:
                            hook_result = hooks["post_tool_context_hook"](
                                list(retrieved_cards),
                                {
                                    "case_input": case.input,
                                    "candidate": candidate,
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
        output: Any = payload if payload is not None else {"answer": "unknown", "invalid_output": raw_output_text}
        validator_action = "not_enabled"
        if config.use_answer_validator:
            validator_action = "passed"
            option_literals = extract_case_options(case.input)
            if "post_answer_validator_hook" in hooks:
                hook_output = hooks["post_answer_validator_hook"](
                    output,
                    {
                        "case_input": case.input,
                        "candidate": candidate,
                        "retrieved_cards": list(retrieved_cards),
                        "option_literals": option_literals,
                        "validator_rule": config.answer_validator_rule,
                    },
                )
                if hook_output is not None:
                    output = hook_output
                if isinstance(output, dict) and str(output.get("answer", "")).strip().lower() == "unknown":
                    validator_action = "forced_unknown"
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
                    "validator_action": validator_action,
                    "validator_rule": config.answer_validator_rule if config.use_answer_validator else None,
                    "retrieved_doc_ids": [card.get("doc_id", "") for card in retrieved_cards],
                },
            ),
        )
