from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from ratchet.model_client import (
    GEMINI_OPENAI_BASE_URL,
    ResponsesModelClient,
    error_response_diagnostics,
    model_request_limits,
    validate_optimizer_model_access,
)


class FakeChatCompletions:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self._count = 0

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        self._count += 1
        if self._count == 1 and kwargs.get("tools"):
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        id="call-1",
                        function=SimpleNamespace(name="docs_search", arguments='{"query":"Path.cwd"}'),
                    )
                ],
            )
        else:
            message = SimpleNamespace(content='{"answer":"Path.cwd()"}', tool_calls=None)
        return SimpleNamespace(
            id=f"chatcmpl-{self._count}",
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class FakeOpenAI:
    instances: list["FakeOpenAI"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.chat_completions = FakeChatCompletions()
        self.chat = SimpleNamespace(completions=self.chat_completions)
        FakeOpenAI.instances.append(self)


class ProbeClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create_response(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            id="resp-probe",
            output_text="OK",
            usage=SimpleNamespace(input_tokens=3, output_tokens=1),
        )


class GeminiCompatClientTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeOpenAI.instances.clear()

    def test_gemini_model_uses_gemini_key_without_openai_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GEMINI_API_KEY=test-gemini-key\n")

            with patch.dict("os.environ", {}, clear=True), patch("ratchet.model_client.OpenAI", FakeOpenAI):
                client = ResponsesModelClient(env_path=str(env_path))
                response = client.create_response(
                    model="gemini-2.5-flash",
                    instructions="Return JSON.",
                    input="hello",
                    max_output_tokens=20,
                    reasoning={"effort": "low"},
                    text={"format": {"type": "json_schema", "schema": {"type": "object"}}},
                )

            self.assertEqual(response.output_text, '{"answer":"Path.cwd()"}')
            self.assertEqual(response.usage.input_tokens, 11)
            self.assertEqual(response.usage.output_tokens, 7)
            self.assertEqual(FakeOpenAI.instances[0].kwargs["api_key"], "test-gemini-key")
            self.assertEqual(FakeOpenAI.instances[0].kwargs["base_url"], GEMINI_OPENAI_BASE_URL)
            request = FakeOpenAI.instances[0].chat_completions.requests[0]
            self.assertEqual(request["response_format"]["type"], "json_schema")
            self.assertEqual(request["response_format"]["json_schema"]["name"], "ratchet_json")
            self.assertEqual(request["response_format"]["json_schema"]["schema"], {"type": "object"})
            self.assertFalse(request["response_format"]["json_schema"]["strict"])
            self.assertEqual(request["reasoning_effort"], "low")

    def test_gemini_json_schema_preserves_shape_but_drops_complex_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GEMINI_API_KEY=test-gemini-key\n")

            with patch.dict("os.environ", {}, clear=True), patch("ratchet.model_client.OpenAI", FakeOpenAI):
                client = ResponsesModelClient(env_path=str(env_path))
                client.create_response(
                    model="gemini-2.5-flash",
                    input="hello",
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "shape_test",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "patches": {
                                        "type": "array",
                                        "maxItems": 8,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "value": {
                                                    "anyOf": [
                                                        {"type": "string"},
                                                        {"type": "boolean"},
                                                    ]
                                                }
                                            },
                                            "required": ["value"],
                                        },
                                    }
                                },
                                "required": ["patches"],
                            },
                        }
                    },
                )

            schema = FakeOpenAI.instances[0].chat_completions.requests[0]["response_format"]["json_schema"]["schema"]
            self.assertEqual(schema["required"], ["patches"])
            self.assertNotIn("maxItems", schema["properties"]["patches"])
            value_schema = schema["properties"]["patches"]["items"]["properties"]["value"]
            self.assertEqual(value_schema["anyOf"], [{"type": "string"}, {"type": "boolean"}])

    def test_model_request_limits_apply_to_gemini_compat_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GEMINI_API_KEY=test-gemini-key\n")

            with patch.dict("os.environ", {}, clear=True), patch("ratchet.model_client.OpenAI", FakeOpenAI):
                client = ResponsesModelClient(env_path=str(env_path))
                with model_request_limits(timeout_s=7, max_attempts=1):
                    client.create_response(
                        model="gemini-2.5-flash",
                        input="hello",
                    )

            request = FakeOpenAI.instances[0].chat_completions.requests[0]
            self.assertEqual(request["timeout"], 7.0)

    def test_gemini_tool_calls_are_resumed_as_chat_tool_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("GEMINI_API_KEY=test-gemini-key\n")

            with patch.dict("os.environ", {}, clear=True), patch("ratchet.model_client.OpenAI", FakeOpenAI):
                client = ResponsesModelClient(env_path=str(env_path))
                first = client.create_response(
                    model="gemini-2.5-flash",
                    instructions="Use tools.",
                    input="Where am I?",
                    text={"format": {"type": "json_schema", "schema": {"type": "object"}}},
                    tools=[
                        {
                            "type": "function",
                            "name": "docs_search",
                            "description": "Search docs.",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        }
                    ],
                )
                second = client.create_response(
                    model="gemini-2.5-flash",
                    previous_response_id=first.id,
                    text={"format": {"type": "json_schema", "schema": {"type": "object"}}},
                    tools=[
                        {
                            "type": "function",
                            "name": "docs_search",
                            "description": "Search docs.",
                            "parameters": {"type": "object"},
                        }
                    ],
                    input=[
                        {
                            "type": "function_call_output",
                            "call_id": first.output[0].call_id,
                            "output": "Path.cwd(): current working directory.",
                        }
                    ],
                )

            self.assertEqual(first.output[0].type, "function_call")
            self.assertEqual(second.output_text, '{"answer":"Path.cwd()"}')
            requests = FakeOpenAI.instances[0].chat_completions.requests
            self.assertEqual(requests[0]["tools"][0]["function"]["name"], "docs_search")
            self.assertNotIn("response_format", requests[0])
            self.assertEqual(requests[1]["messages"][-1]["role"], "tool")
            self.assertEqual(requests[1]["messages"][-1]["tool_call_id"], "call-1")
            self.assertNotIn("tools", requests[1])
            self.assertEqual(requests[1]["response_format"]["type"], "json_schema")

    def test_validate_optimizer_model_access_uses_one_token_probe(self) -> None:
        client = ProbeClient()

        diagnostics = validate_optimizer_model_access(
            env_path=".env",
            model="gpt-5.4-mini",
            client=client,
        )

        self.assertTrue(diagnostics["checked"])
        self.assertEqual(client.requests[0]["model"], "gpt-5.4-mini")
        self.assertEqual(client.requests[0]["max_output_tokens"], 1)
        self.assertEqual(diagnostics["input_tokens"], 3)
        self.assertEqual(diagnostics["output_tokens"], 1)

    def test_error_response_diagnostics_uses_none_cost(self) -> None:
        diagnostics = error_response_diagnostics(
            RuntimeError("boom"),
            model="gpt-5.4-mini",
            elapsed_s=0.25,
        )

        self.assertEqual(diagnostics["error"], "boom")
        self.assertIsNone(diagnostics["cost_usd"])
        self.assertEqual(diagnostics["total_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
