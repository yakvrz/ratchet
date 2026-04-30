from __future__ import annotations

import os
import signal
import threading
import time
from typing import Any

from ratchet.tool_loop import GeneratedToolLoopAdapter, ToolLoopRunConfig
from ratchet.types import AgentSpec, EvalCase, GradeResult


BASE_SPEC = AgentSpec(
    name="taubench-tool-loop",
    model="gemini-2.5-flash",
    model_options=[
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
    ],
    runtime={
        "model_provider": "gemini",
        "user_model": "gemini-2.5-flash",
        "user_model_provider": "gemini",
        "user_strategy": "llm",
        "temperature": 0.0,
        "max_steps": 30,
        "request_timeout_s": 45.0,
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
    config_metadata = dict(config.metadata or {})
    surface_probe = bool(config_metadata.get("surface_probe"))
    return get_env(
        env_name=str(metadata.get("env") or "retail"),
        user_strategy="human"
        if surface_probe
        else str(metadata.get("user_strategy") or BASE_SPEC.runtime.get("user_strategy") or "llm"),
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
        request_timeout_s=float(metadata.get("request_timeout_s") or runtime.get("request_timeout_s") or 45.0),
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
        timeout_s = float(os.environ.get("RATCHET_TAUBENCH_USER_TIMEOUT_S", "45"))
        kwargs.setdefault("timeout", timeout_s)
        return _with_hard_timeout(lambda: raw_completion(*args, **kwargs), timeout_s=timeout_s)

    completion_with_retries._ratchet_retry_wrapped = True  # type: ignore[attr-defined]
    user_module.completion = completion_with_retries


def _with_hard_timeout(call: Any, *, timeout_s: float) -> Any:
    if timeout_s <= 0 or threading.current_thread() is not threading.main_thread():
        return call()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(signum: int, frame: object) -> None:
        raise TimeoutError(f"tau-bench user model request exceeded {timeout_s:.1f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


adapter = GeneratedToolLoopAdapter(
    agent_spec=BASE_SPEC,
    environment_factory=_make_environment,
    action_factory=_make_action,
    respond_action_name="respond",
    case_config=_case_config,
    grade=_grade,
    env_path=os.environ.get("RATCHET_ENV_FILE", ".env"),
)
