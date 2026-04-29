from __future__ import annotations

from typing import Any

from ratchet.interactive import InteractionRecorder
from ratchet.types import RunRecord


class TauBenchRunner:
    """Thin optional bridge for original tau-bench retail/airline simulations."""

    def __init__(
        self,
        *,
        env: str = "retail",
        agent_strategy: str = "tool-calling",
        user_strategy: str = "llm",
        task_split: str = "test",
        max_concurrency: int = 1,
        seed: int = 10,
    ) -> None:
        if env not in {"retail", "airline"}:
            raise ValueError("original tau-bench supports env='retail' or env='airline'.")
        self.env = env
        self.agent_strategy = agent_strategy
        self.user_strategy = user_strategy
        self.task_split = task_split
        self.max_concurrency = max_concurrency
        self.seed = seed

    def run_task(
        self,
        *,
        task_id: int,
        model: str,
        model_provider: str,
        user_model: str,
        user_model_provider: str,
        temperature: float = 0.0,
        log_dir: str = "results",
    ) -> RunRecord:
        try:
            from tau_bench.run import run
            from tau_bench.types import RunConfig
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tau-bench is not installed. Install the original sierra-research/tau-bench "
                "package to use TauBenchRunner."
            ) from exc

        results = run(
            RunConfig(
                model_provider=model_provider,
                user_model_provider=user_model_provider,
                model=model,
                user_model=user_model,
                env=self.env,
                agent_strategy=self.agent_strategy,
                temperature=temperature,
                task_split=self.task_split,
                task_ids=[task_id],
                log_dir=log_dir,
                max_concurrency=self.max_concurrency,
                seed=self.seed,
                user_strategy=self.user_strategy,
            )
        )
        if len(results) != 1:
            raise RuntimeError(f"Expected one tau-bench result for task {task_id}, got {len(results)}.")
        return taubench_result_to_run_record(results[0])


def taubench_result_to_run_record(result: Any) -> RunRecord:
    recorder = InteractionRecorder()
    input_tokens = 0
    output_tokens = 0
    cost_usd = 0.0
    model_calls = 0
    traj = list(getattr(result, "traj", []) or [])
    for index, message in enumerate(traj):
        if not isinstance(message, dict):
            recorder.add_turn(actor="unknown", message=str(message))
            continue
        role = str(message.get("role") or message.get("actor") or "unknown")
        content = message.get("content")
        usage = message.get("usage") or {}
        if isinstance(usage, dict):
            input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        if message.get("cost") is not None:
            cost_usd += float(message["cost"])
        if role in {"assistant", "user"} and (usage or message.get("cost") is not None):
            model_calls += 1
        turn = recorder.add_turn(actor=role, message=content, metadata={"trajectory_index": index})
        for tool_call in _tool_calls_from_message(message):
            recorder.add_tool_call(
                name=tool_call["name"],
                arguments=tool_call.get("arguments"),
                result=tool_call.get("result"),
                status=tool_call.get("status", "ok"),
                error=tool_call.get("error"),
                turn_index=turn,
                metadata={"source": "tau-bench"},
            )

    reward = float(getattr(result, "reward", 0.0) or 0.0)
    info = getattr(result, "info", {}) or {}
    output = {
        "benchmark": "tau-bench",
        "task_id": getattr(result, "task_id", None),
        "trial": getattr(result, "trial", None),
        "reward": reward,
        "info": _jsonable(info),
    }
    metrics = recorder.metrics(
        latency_s=0.0,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        model_calls=max(1, model_calls),
    )
    return RunRecord(
        output=output,
        metrics=metrics,
        diagnostics=recorder.diagnostics(
            raw_output_text=str(output),
            terminal_state={"reward": reward, "info": _jsonable(info)},
            terminal_reason="success" if reward >= 1.0 else "failed",
        ),
    )


def _tool_calls_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    if message.get("role") == "tool":
        return calls
    raw_calls = message.get("tool_calls") or message.get("tools") or []
    if isinstance(raw_calls, list):
        for item in raw_calls:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or (item.get("function") or {}).get("name")
            if not name:
                continue
            arguments = item.get("arguments") or (item.get("function") or {}).get("arguments")
            calls.append({"name": str(name), "arguments": arguments, "status": "ok"})
    if message.get("tool_name"):
        calls.append(
            {
                "name": str(message["tool_name"]),
                "arguments": message.get("tool_args"),
                "result": message.get("observation"),
                "status": "error" if message.get("error") else "ok",
                "error": str(message.get("error")) if message.get("error") else None,
            }
        )
    return calls


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)
