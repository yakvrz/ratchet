from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError


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


class OpenAIResponsesClient:
    def __init__(self, env_path: str = ".env") -> None:
        self.env_path = env_path
        load_env_file(env_path)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing from the environment or .env")
        self.client = OpenAI(api_key=api_key)

    def create_response(self, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", 90)
        delay_s = 1.0
        last_error: Exception | None = None
        for _ in range(8):
            try:
                return self.client.responses.create(**kwargs)
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as error:
                last_error = error
                time.sleep(delay_s)
                delay_s = min(delay_s * 2, 20)
        assert last_error is not None
        raise last_error
