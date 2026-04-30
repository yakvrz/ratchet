from __future__ import annotations

import os
import time
from typing import Any

from ratchet.tool_loop import GeneratedToolLoopAdapter, ToolLoopRunConfig
from ratchet.types import AgentSpec, EvalCase, GradeResult


BASE_SPEC = AgentSpec(
    name="taubench-tool-loop",
    model="gemini-2.5-flash-lite",
    model_options=[
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    ],
    runtime={
        "model_provider": "gemini",
        "user_model": "gemini-2.5-flash-lite",
        "user_model_provider": "gemini",
        "user_strategy": "llm",
        "temperature": 0.0,
        "max_steps": 30,
        "model_provider_by_name": {
            "gemini-2.5-flash-lite": "gemini",
            "gemini-2.5-flash": "gemini",
            "gemini-2.5-pro": "gemini",
            "gemini-3-flash-preview": "gemini",
            "gemini-3-pro-preview": "gemini",
        },
    },
    metadata={"benchmark": "tau-bench", "benchmark_fidelity": "tau_bench_simulator"},
)


def _make_environment(case: EvalCase, config: ToolLoopRunConfig) -> Any:
    try:
        from tau_bench.envs import get_env
        from tau_bench.envs import user as user_module
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tau-bench is not installed. Install sierra-research/tau-bench to run this assessment."
        ) from exc
    _install_taubench_user_retries(user_module)
    metadata = dict(case.metadata)
    return get_env(
        env_name=str(metadata.get("env") or "retail"),
        user_strategy=str(metadata.get("user_strategy") or BASE_SPEC.runtime.get("user_strategy") or "llm"),
        user_model=str(metadata.get("user_model") or BASE_SPEC.runtime.get("user_model") or BASE_SPEC.model),
        user_provider=str(
            metadata.get("user_model_provider") or BASE_SPEC.runtime.get("user_model_provider") or "gemini"
        ),
        task_split=str(metadata.get("task_split") or "test"),
        task_index=_task_id(metadata),
    )


def _make_action(name: str, args: dict[str, Any]) -> Any:
    try:
        from tau_bench.types import Action
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tau-bench is not installed. Install sierra-research/tau-bench to run this assessment."
        ) from exc
    return Action(name=name, kwargs=args)


def _case_config(spec: AgentSpec, case: EvalCase) -> ToolLoopRunConfig:
    runtime = dict(spec.runtime)
    metadata = dict(case.metadata)
    return ToolLoopRunConfig(
        provider=str(runtime.get("model_provider", "gemini")),
        temperature=float(runtime.get("temperature", 0.0)),
        max_steps=int(metadata.get("max_steps") or runtime.get("max_steps") or 30),
        log_dir=str(metadata.get("log_dir") or runtime.get("log_dir") or "samples/taubench_agent/results/raw"),
        metadata={
            "benchmark": "tau-bench",
            "benchmark_fidelity": "tau_bench_simulator",
            "env": str(metadata.get("env") or "retail"),
            "task_id": _task_id(metadata),
            "task_split": str(metadata.get("task_split") or "test"),
        },
    )


def _grade(case: EvalCase, output: object) -> GradeResult:
    reward = 0.0
    if isinstance(output, dict):
        reward = float(output.get("reward") or 0.0)
    passed = reward >= 1.0
    return GradeResult(
        score=reward,
        passed=passed,
        labels=[] if passed else ["tau_bench_reward_failed"],
        notes=f"reward={reward:.3f}",
    )


def _task_id(metadata: dict[str, Any]) -> int:
    task_id = metadata.get("task_id")
    if not isinstance(task_id, int):
        raise ValueError("tau-bench cases require integer metadata.task_id.")
    return task_id


def _install_taubench_user_retries(user_module: Any) -> None:
    if getattr(user_module.completion, "_ratchet_retry_wrapped", False):
        return
    raw_completion = user_module.completion

    def completion_with_retries(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(4):
            try:
                return raw_completion(*args, **kwargs)
            except Exception as exc:
                if attempt >= 3 or not _is_transient_provider_error(exc):
                    raise
                time.sleep(2.0 * (attempt + 1))

    completion_with_retries._ratchet_retry_wrapped = True  # type: ignore[attr-defined]
    user_module.completion = completion_with_retries


def _is_transient_provider_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("503", "serviceunavailable", "unavailable", "rate limit", "429"))


adapter = GeneratedToolLoopAdapter(
    agent_spec=BASE_SPEC,
    environment_factory=_make_environment,
    action_factory=_make_action,
    respond_action_name="respond",
    case_config=_case_config,
    grade=_grade,
    env_path=os.environ.get("RATCHET_ENV_FILE", ".env"),
)
