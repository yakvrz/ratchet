from __future__ import annotations

import json
import os
from typing import Any

from order_desk_env import OrderDeskEnv, make_action
from ratchet.tool_loop import GeneratedToolLoopAdapter, ToolLoopRunConfig
from ratchet.types import AgentSpec, AgentTool, EvalCase, GradeResult


MODEL_OPTIONS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]


BASE_SPEC = AgentSpec(
    name="order-desk-tool-loop-agent",
    model="gemini-2.5-flash",
    model_options=MODEL_OPTIONS,
    runtime={
        "model_provider": "gemini",
        "temperature": 0.0,
        "max_steps": 7,
        "request_timeout_s": 25.0,
        "model_provider_by_name": {model: "gemini" for model in MODEL_OPTIONS},
    },
    tools={
        "find_user_by_email": AgentTool(
            name="find_user_by_email",
            description="Authenticate by email.",
            metadata={"side_effect": "read", "risk": "low"},
        ),
        "find_user_by_name_zip": AgentTool(
            name="find_user_by_name_zip",
            description="Authenticate by name and zip.",
            metadata={"side_effect": "read", "risk": "low"},
        ),
        "list_orders": AgentTool(
            name="list_orders",
            description="List customer orders.",
            metadata={"side_effect": "read", "risk": "low"},
        ),
        "get_order": AgentTool(
            name="get_order",
            description="Inspect order details.",
            metadata={"side_effect": "read", "risk": "low"},
        ),
        "cancel_order": AgentTool(
            name="cancel_order",
            description="Cancel a pending order.",
            metadata={"side_effect": "mutating", "risk": "medium"},
        ),
        "modify_address": AgentTool(
            name="modify_address",
            description="Modify pending-order shipping address.",
            metadata={"side_effect": "mutating", "risk": "medium"},
        ),
        "return_item": AgentTool(
            name="return_item",
            description="Request return of a delivered item.",
            metadata={"side_effect": "mutating", "risk": "medium"},
        ),
    },
    metadata={"benchmark": "order-desk-tool-loop", "benchmark_fidelity": "local_deterministic"},
)


def _make_environment(case: EvalCase, config: ToolLoopRunConfig) -> Any:
    task_id = case.metadata.get("task_id")
    if not isinstance(task_id, int):
        raise ValueError("order desk cases require integer metadata.task_id")
    return OrderDeskEnv(task_id=task_id)


def _case_config(spec: AgentSpec, case: EvalCase) -> ToolLoopRunConfig:
    runtime = dict(spec.runtime)
    metadata = dict(case.metadata)
    return ToolLoopRunConfig(
        provider=str(runtime.get("model_provider", "gemini")),
        temperature=float(runtime.get("temperature", 0.0)),
        max_steps=int(metadata.get("max_steps") or runtime.get("max_steps") or 7),
        request_timeout_s=float(metadata.get("request_timeout_s") or runtime.get("request_timeout_s") or 25.0),
        log_dir=str(metadata.get("log_dir") or runtime.get("log_dir") or "demo/results/raw"),
        metadata={
            "benchmark": "order-desk-tool-loop",
            "benchmark_fidelity": "local_deterministic",
            "category": str(metadata.get("category") or "unknown"),
            "task_id": task_id_from_case(case),
        },
    )


def _grade(case: EvalCase, output: object) -> GradeResult:
    reward = 0.0
    labels: list[str] = []
    notes = ""
    if isinstance(output, dict):
        reward = float(output.get("reward") or 0.0)
        raw_observation = str(output.get("last_observation") or "")
        notes = raw_observation[:500]
        try:
            payload = json.loads(raw_observation)
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            raw_labels = payload.get("labels")
            if isinstance(raw_labels, list):
                labels = [str(item) for item in raw_labels if item]
    passed = reward >= 1.0
    if not passed and not labels:
        labels = ["order_desk_reward_failed"]
    return GradeResult(score=reward, passed=passed, labels=[] if passed else labels, notes=notes)


def task_id_from_case(case: EvalCase) -> int:
    task_id = case.metadata.get("task_id")
    if not isinstance(task_id, int):
        raise ValueError("order desk cases require integer metadata.task_id")
    return task_id


adapter = GeneratedToolLoopAdapter(
    agent_spec=BASE_SPEC,
    environment_factory=_make_environment,
    action_factory=make_action,
    respond_action_name="respond",
    case_config=_case_config,
    grade=_grade,
    env_path=os.environ.get("RATCHET_ENV_FILE", ".env"),
)
