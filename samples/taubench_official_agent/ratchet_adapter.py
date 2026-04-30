from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ratchet.benchmarks import TauBenchRunner
from ratchet.context_graph import ContextGraph
from ratchet.runtime import RuntimeContext, TransformRuntime
from ratchet.surfaces import SurfaceSpec, surface_from_agent_spec
from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, RunRecord


BASE_SPEC = AgentSpec(
    name="official-taubench-agent",
    model="gpt-4o",
    model_options=[
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20241022",
    ],
    instructions={
        "benchmark_contract": (
            "Run the original tau-bench simulator. The benchmark owns the domain policy, "
            "tool schemas, user simulator, environment state, and reward."
        )
    },
    output_contract="Reward is produced by the official tau-bench evaluator.",
    runtime={
        "model_provider": "openai",
        "user_model": "gpt-4o",
        "user_model_provider": "openai",
        "user_strategy": "llm",
        "agent_strategy": "tool-calling",
        "temperature": 0.0,
        "model_provider_by_name": {
            "gpt-4o": "openai",
            "gpt-4o-mini": "openai",
            "claude-3-5-sonnet-20241022": "anthropic",
        },
    },
    metadata={"benchmark": "tau-bench", "benchmark_fidelity": "official_tau_bench_simulator"},
)


class OfficialTauBenchAdapter:
    def __init__(self, env_path: str | None = None, runner_factory: Any | None = None) -> None:
        self.env_path = env_path or os.environ.get("RATCHET_ENV_FILE", ".env")
        self.runner_factory = runner_factory or TauBenchRunner
        self._surface: SurfaceSpec | None = None

    def agent_spec(self) -> AgentSpec:
        return BASE_SPEC

    def surface_spec(self) -> SurfaceSpec:
        if self._surface is None:
            self._surface = surface_from_agent_spec(self.agent_spec())
        return self._surface

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        _load_env_file(self.env_path)
        config = _runtime_config(self.agent_spec(), case)
        runtime = TransformRuntime(candidate)
        ctx = RuntimeContext(
            case=case,
            context=ContextGraph(()),
            model_config={
                "model_name": self.agent_spec().model,
                "temperature": config["temperature"],
            },
        )
        runtime.run_hook("on_task_start", ctx)
        runtime.run_hook("before_model_call", ctx)
        model = str(ctx.model_config["model_name"])
        provider = _provider_for_model(model, self.agent_spec())
        runner = self.runner_factory(
            env=config["env"],
            agent_strategy=config["agent_strategy"],
            user_strategy=config["user_strategy"],
            task_split=config["task_split"],
            max_concurrency=1,
            seed=config["seed"],
        )
        record = runner.run_task(
            task_id=config["task_id"],
            model=model,
            model_provider=provider,
            user_model=config["user_model"],
            user_model_provider=config["user_model_provider"],
            temperature=float(ctx.model_config.get("temperature", config["temperature"])),
            log_dir=config["log_dir"],
        )
        metadata = dict(record.diagnostics.metadata)
        metadata.update(
            {
                "benchmark": "tau-bench",
                "benchmark_fidelity": "official_tau_bench_simulator",
                "env": config["env"],
                "task_id": config["task_id"],
                "task_split": config["task_split"],
                "model": model,
                "model_provider": provider,
                "transform_candidate_id": candidate.program.candidate_id if candidate is not None else None,
                "transform_compile_report": candidate.report.to_dict() if candidate is not None else None,
                "transform_diff": candidate.diff.to_dict() if candidate is not None else None,
                "transform_trace": list(ctx.trace_annotations),
            }
        )
        return RunRecord(
            output=record.output,
            metrics=record.metrics,
            diagnostics=DiagnosticTrace(
                tool_calls=list(record.diagnostics.tool_calls),
                raw_output_text=record.diagnostics.raw_output_text,
                turns=list(record.diagnostics.turns),
                terminal_state=dict(record.diagnostics.terminal_state),
                terminal_reason=record.diagnostics.terminal_reason,
                metadata=metadata,
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
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

    def export(self, candidate: CompiledCandidate | None, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if candidate is not None:
            (out_dir / "compiled_candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True))
        (out_dir / "surface_spec.json").write_text(json.dumps(self.surface_spec().to_dict(), indent=2, sort_keys=True))


def _runtime_config(spec: AgentSpec, case: EvalCase) -> dict[str, Any]:
    metadata = dict(case.metadata)
    runtime = dict(spec.runtime)
    task_id = metadata.get("task_id")
    if not isinstance(task_id, int):
        raise ValueError("official tau-bench cases require integer metadata.task_id.")
    return {
        "env": str(metadata.get("env") or runtime.get("env") or "retail"),
        "task_split": str(metadata.get("task_split") or runtime.get("task_split") or "test"),
        "task_id": task_id,
        "agent_strategy": str(runtime.get("agent_strategy", "tool-calling")),
        "user_strategy": str(runtime.get("user_strategy", "llm")),
        "user_model": str(runtime.get("user_model", "gpt-4o")),
        "user_model_provider": str(runtime.get("user_model_provider", "openai")),
        "temperature": float(runtime.get("temperature", 0.0)),
        "seed": int(metadata.get("seed") or runtime.get("seed") or 10),
        "log_dir": str(metadata.get("log_dir") or runtime.get("log_dir") or "samples/taubench_official_agent/results/raw"),
    }


def _provider_for_model(model: str, spec: AgentSpec) -> str:
    mapping = spec.runtime.get("model_provider_by_name")
    if isinstance(mapping, dict) and model in mapping:
        return str(mapping[model])
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-"):
        return "openai"
    return str(spec.runtime.get("model_provider", "openai"))


def _load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key, value)


adapter = OfficialTauBenchAdapter()
