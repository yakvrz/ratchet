from __future__ import annotations

import os
from typing import Any

from ratchet.tool_loop import GeneratedToolLoopAdapter, ToolLoopRunConfig
from ratchet.types import AgentSpec, EvalCase, GradeResult


BASE_SPEC = AgentSpec(
    name="taubench-tool-loop",
    model="gpt-4o",
    model_options=[
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
    ],
    runtime={
        "model_provider": "openai",
        "user_model": "gpt-4o",
        "user_model_provider": "openai",
        "user_strategy": "llm",
        "temperature": 0.0,
        "max_steps": 30,
        "model_provider_by_name": {
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "claude-3-5-sonnet-20241022": "anthropic",
        },
    },
    metadata={"benchmark": "tau-bench", "benchmark_fidelity": "tau_bench_simulator"},
)


def _make_environment(case: EvalCase, config: ToolLoopRunConfig) -> Any:
    try:
        from tau_bench.envs import get_env
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tau-bench is not installed. Install sierra-research/tau-bench to run this assessment."
        ) from exc
    metadata = dict(case.metadata)
    return get_env(
        env_name=str(metadata.get("env") or "retail"),
        user_strategy=str(metadata.get("user_strategy") or BASE_SPEC.runtime.get("user_strategy") or "llm"),
        user_model=str(metadata.get("user_model") or BASE_SPEC.runtime.get("user_model") or "gpt-4o"),
        user_provider=str(
            metadata.get("user_model_provider") or BASE_SPEC.runtime.get("user_model_provider") or "openai"
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
        provider=str(runtime.get("model_provider", "openai")),
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


adapter = GeneratedToolLoopAdapter(
    agent_spec=BASE_SPEC,
    environment_factory=_make_environment,
    action_factory=_make_action,
    respond_action_name="respond",
    case_config=_case_config,
    grade=_grade,
    env_path=os.environ.get("RATCHET_ENV_FILE", ".env"),
)
