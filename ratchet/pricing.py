from __future__ import annotations


MODEL_PRICING_USD_PER_TOKEN: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 2.50 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-5.4-mini": {"input": 0.75 / 1_000_000, "output": 4.50 / 1_000_000},
    "gpt-5.4-nano": {"input": 0.20 / 1_000_000, "output": 1.25 / 1_000_000},
    "gpt-5.2": {"input": 1.75 / 1_000_000, "output": 14.00 / 1_000_000},
    "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-2024-08-06": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gemini-2.5-flash-lite": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-2.5-flash": {"input": 0.30 / 1_000_000, "output": 2.50 / 1_000_000},
    "gemini-2.5-pro": {"input": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_USD_PER_TOKEN[model]
    return (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
