from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any, Iterator

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from ratchet.pricing import estimate_cost_usd

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL_TIMEOUT_S = 90.0
DEFAULT_MODEL_MAX_ATTEMPTS = 8
_REQUEST_TIMEOUT_S: ContextVar[float | None] = ContextVar("ratchet_request_timeout_s", default=None)
_REQUEST_MAX_ATTEMPTS: ContextVar[int | None] = ContextVar("ratchet_request_max_attempts", default=None)


@contextmanager
def model_request_limits(
    *,
    timeout_s: float | int | None = None,
    max_attempts: int | None = None,
) -> Iterator[None]:
    """Temporarily constrain model-client requests made in this context.

    Ratchet's optimizer uses this while evaluating target-agent cases. The
    optimizer already owns case-level retries, so target-agent model calls
    should not perform a long nested retry loop inside a single case attempt.
    """
    timeout_token = _REQUEST_TIMEOUT_S.set(float(timeout_s) if timeout_s is not None and timeout_s > 0 else None)
    attempts_token = _REQUEST_MAX_ATTEMPTS.set(max_attempts if max_attempts is not None and max_attempts > 0 else None)
    try:
        yield
    finally:
        _REQUEST_TIMEOUT_S.reset(timeout_token)
        _REQUEST_MAX_ATTEMPTS.reset(attempts_token)


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


@dataclass(frozen=True)
class CompatUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class CompatOutputItem:
    type: str
    text: str = ""
    name: str = ""
    arguments: str = ""
    call_id: str = ""


@dataclass(frozen=True)
class CompatResponse:
    id: str
    output: list[CompatOutputItem]
    output_text: str
    usage: CompatUsage
    finish_reason: str = ""


def response_diagnostics(response: Any, *, model: str, elapsed_s: float) -> dict[str, Any]:
    input_tokens, output_tokens = response_token_usage(response)
    try:
        cost_usd = estimate_cost_usd(model, input_tokens, output_tokens)
    except KeyError:
        cost_usd = None
    return {
        "model": model,
        "elapsed_s": elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost_usd,
        "finish_reason": str(getattr(response, "finish_reason", "") or _first_choice_finish_reason(response)),
    }


def error_response_diagnostics(error: Exception, *, model: str, elapsed_s: float) -> dict[str, Any]:
    return {
        "model": model,
        "elapsed_s": elapsed_s,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": None,
        "finish_reason": "",
        "error": str(error),
    }


def combine_response_diagnostics(
    *,
    component: str,
    primary: dict[str, Any],
    repair: dict[str, Any],
) -> dict[str, Any]:
    cost_values = [item.get("cost_usd") for item in (primary, repair)]
    cost_usd = None
    if any(value is not None for value in cost_values):
        cost_usd = sum(float(value or 0.0) for value in cost_values)
    return {
        "component": component,
        "model": repair.get("model") or primary.get("model"),
        "elapsed_s": float(primary.get("elapsed_s") or 0.0) + float(repair.get("elapsed_s") or 0.0),
        "input_tokens": int(primary.get("input_tokens") or 0) + int(repair.get("input_tokens") or 0),
        "output_tokens": int(primary.get("output_tokens") or 0) + int(repair.get("output_tokens") or 0),
        "total_tokens": int(primary.get("total_tokens") or 0) + int(repair.get("total_tokens") or 0),
        "cost_usd": cost_usd,
        "finish_reason": repair.get("finish_reason") or primary.get("finish_reason") or "",
        "repair_attempted": True,
        "primary_call": primary,
        "repair_call": repair,
    }


def validate_optimizer_model_access(
    *,
    env_path: str,
    model: str,
    client: ResponsesModelClient | None = None,
) -> dict[str, Any]:
    probe_client = client or ResponsesModelClient(env_path=env_path)
    started_at = time.perf_counter()
    response = probe_client.create_response(
        model=model,
        input="Return OK.",
        max_output_tokens=1,
        timeout=20,
    )
    diagnostics = response_diagnostics(
        response,
        model=model,
        elapsed_s=time.perf_counter() - started_at,
    )
    return {"checked": True, **diagnostics}


def response_token_usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    input_tokens = int(
        getattr(usage, "input_tokens", 0)
        or getattr(usage, "prompt_tokens", 0)
        or 0
    )
    output_tokens = int(
        getattr(usage, "output_tokens", 0)
        or getattr(usage, "completion_tokens", 0)
        or 0
    )
    return input_tokens, output_tokens


def _first_choice_finish_reason(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    return str(getattr(choices[0], "finish_reason", "") or "")


class ResponsesModelClient:
    def __init__(self, env_path: str = ".env") -> None:
        self.env_path = env_path
        load_env_file(env_path)
        self.client: OpenAI | None = None
        self.gemini_client: OpenAI | None = None
        self._gemini_messages_by_response_id: dict[str, list[dict[str, Any]]] = {}

    def create_response(self, **kwargs: Any) -> Any:
        model = str(kwargs.get("model", ""))
        if _is_gemini_model(model):
            return self._create_gemini_chat_response(**kwargs)
        return self._create_openai_response(**kwargs)

    def _create_openai_response(self, **kwargs: Any) -> Any:
        if self.client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is missing from the environment or .env")
            self.client = OpenAI(api_key=api_key)
        kwargs.setdefault("timeout", _current_request_timeout())
        return _with_retries(
            lambda: self.client.responses.create(**kwargs),
            max_attempts=_current_request_max_attempts(),
        )

    def _create_gemini_chat_response(self, **kwargs: Any) -> CompatResponse:
        if self.gemini_client is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("GEMINI_API_KEY is missing from the environment or .env")
            self.gemini_client = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("GEMINI_OPENAI_BASE_URL", GEMINI_OPENAI_BASE_URL),
            )
        messages = self._gemini_messages(kwargs)
        request: dict[str, Any] = {
            "model": kwargs["model"],
            "messages": messages,
        }
        if kwargs.get("max_output_tokens") is not None:
            request["max_tokens"] = int(kwargs["max_output_tokens"])
        reasoning = kwargs.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort"):
            request["reasoning_effort"] = str(reasoning["effort"])
        tools = _chat_tools(kwargs.get("tools") or [])
        response_format = _gemini_response_format(kwargs.get("text"))
        # Gemini's OpenAI-compatible endpoint rejects function calling combined
        # with JSON response mode. After tool outputs, prefer the structured
        # final answer over allowing another tool round.
        include_tools = bool(tools) and not (kwargs.get("previous_response_id") and response_format)
        if include_tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        if response_format and not include_tools:
            request["response_format"] = response_format
        request["timeout"] = kwargs.get("timeout") or _current_request_timeout()
        completion = _with_retries(
            lambda: self.gemini_client.chat.completions.create(**request),
            max_attempts=_current_request_max_attempts(),
        )
        return self._compat_response(completion, messages)

    def _gemini_messages(self, kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        previous_response_id = kwargs.get("previous_response_id")
        if previous_response_id is not None:
            try:
                messages = list(self._gemini_messages_by_response_id[str(previous_response_id)])
            except KeyError as exc:
                raise RuntimeError(f"Unknown Gemini previous_response_id: {previous_response_id}") from exc
            for item in kwargs.get("input") or []:
                if not isinstance(item, dict) or item.get("type") != "function_call_output":
                    raise ValueError("Gemini continuation input must contain function_call_output items.")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(item["call_id"]),
                        "content": str(item.get("output", "")),
                    }
                )
            return messages

        messages = []
        instructions = kwargs.get("instructions")
        if instructions:
            messages.append({"role": "system", "content": str(instructions)})
        raw_input = kwargs.get("input", "")
        if isinstance(raw_input, str):
            messages.append({"role": "user", "content": raw_input})
        else:
            messages.append({"role": "user", "content": str(raw_input)})
        return messages

    def _compat_response(self, completion: Any, request_messages: list[dict[str, Any]]) -> CompatResponse:
        choice = completion.choices[0]
        message = choice.message
        text = _message_text(getattr(message, "content", ""))
        output_items: list[CompatOutputItem] = []
        chat_tool_calls = list(getattr(message, "tool_calls", None) or [])
        if chat_tool_calls:
            for tool_call in chat_tool_calls:
                function = tool_call.function
                output_items.append(
                    CompatOutputItem(
                        type="function_call",
                        name=str(function.name),
                        arguments=str(function.arguments or "{}"),
                        call_id=str(tool_call.id),
                    )
                )
        else:
            output_items.append(CompatOutputItem(type="message", text=text))
        usage = getattr(completion, "usage", None)
        response = CompatResponse(
            id=str(getattr(completion, "id", "")),
            output=output_items,
            output_text=text,
            finish_reason=str(getattr(choice, "finish_reason", "") or ""),
            usage=CompatUsage(
                input_tokens=int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(
                    getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
                ),
            ),
        )
        self._gemini_messages_by_response_id[response.id] = [
            *request_messages,
            _assistant_message(message, text, chat_tool_calls),
        ]
        return response


def _current_request_timeout() -> float:
    return float(_REQUEST_TIMEOUT_S.get() or DEFAULT_MODEL_TIMEOUT_S)


def _current_request_max_attempts() -> int:
    return int(_REQUEST_MAX_ATTEMPTS.get() or DEFAULT_MODEL_MAX_ATTEMPTS)


def _with_retries(create: Any, *, max_attempts: int | None = None) -> Any:
    attempts = max(1, int(max_attempts or DEFAULT_MODEL_MAX_ATTEMPTS))
    delay_s = 1.0
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return create()
        except RateLimitError as error:
            if _is_insufficient_quota(error):
                raise
            last_error = error
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, 20)
        except (APIConnectionError, APITimeoutError, InternalServerError) as error:
            last_error = error
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, 20)
    assert last_error is not None
    raise last_error


def _is_gemini_model(model: str) -> bool:
    return model.startswith("gemini-")


def _chat_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chat_tools: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        if "function" in tool:
            chat_tools.append(tool)
            continue
        chat_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
        )
    return chat_tools


def _gemini_response_format(text_config: Any) -> dict[str, Any] | None:
    if not isinstance(text_config, dict):
        return None
    text_format = text_config.get("format")
    if not isinstance(text_format, dict):
        return None
    if text_format.get("type") == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {
                "name": str(text_format.get("name", "ratchet_json")),
                "schema": _gemini_safe_json_schema(text_format.get("schema", {"type": "object"})),
                "strict": False,
            },
        }
    if text_format.get("type") == "json_object":
        return {"type": "json_object"}
    return None


def _gemini_safe_json_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object"}
    if "anyOf" in schema or "oneOf" in schema or "allOf" in schema:
        raw_options = schema.get("anyOf") or schema.get("oneOf") or schema.get("allOf") or []
        options = [_gemini_safe_json_schema(item) for item in raw_options if isinstance(item, dict)]
        return {"anyOf": options} if options else {}
    result: dict[str, Any] = {}
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        result["type"] = schema_type
    elif isinstance(schema_type, list) and schema_type:
        result["type"] = [str(item) for item in schema_type]
    if "enum" in schema and isinstance(schema["enum"], list) and len(schema["enum"]) <= 12:
        result["enum"] = list(schema["enum"])
    if isinstance(schema.get("properties"), dict):
        result["properties"] = {
            str(name): _gemini_safe_json_schema(value)
            for name, value in schema["properties"].items()
            if isinstance(value, dict)
        }
    if isinstance(schema.get("required"), list):
        result["required"] = [str(item) for item in schema["required"]]
    if isinstance(schema.get("items"), dict):
        result["items"] = _gemini_safe_json_schema(schema["items"])
    if isinstance(schema.get("additionalProperties"), bool):
        result["additionalProperties"] = bool(schema["additionalProperties"])
    elif isinstance(schema.get("additionalProperties"), dict):
        result["additionalProperties"] = _gemini_safe_json_schema(schema["additionalProperties"])
    for key in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        value = schema.get(key)
        if isinstance(value, int):
            result[key] = value
    return result or {"type": "object"}


def _message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _assistant_message(message: Any, text: str, tool_calls: list[Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": str(tool_call.id),
                "type": "function",
                "function": {
                    "name": str(tool_call.function.name),
                    "arguments": str(tool_call.function.arguments or "{}"),
                },
            }
            for tool_call in tool_calls
        ]
    return payload


def _is_insufficient_quota(error: RateLimitError) -> bool:
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("code") == "insufficient_quota":
        return True
    return getattr(error, "code", None) == "insufficient_quota"
