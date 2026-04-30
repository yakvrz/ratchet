from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any, Callable, Protocol

from ratchet.adapter_generation import context_graph_from_spec, model_config_from_spec
from ratchet.interactive import InteractionRecorder
from ratchet.runtime import RuntimeContext, TransformRuntime
from ratchet.surfaces import SurfaceSpec, tool_loop_surface_from_agent_spec
from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, RunRecord


class ToolLoopModelClient(Protocol):
    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        provider: str,
        tools: list[dict[str, Any]],
        temperature: float,
        timeout_s: float | None = None,
    ) -> "ToolLoopModelResponse":
        ...


@dataclass(frozen=True)
class ToolLoopModelResponse:
    message: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class ToolLoopRunConfig:
    provider: str
    temperature: float = 0.0
    max_steps: int = 30
    request_timeout_s: float | None = None
    log_dir: str = "results"
    metadata: dict[str, Any] | None = None


EnvironmentFactory = Callable[[EvalCase, ToolLoopRunConfig], Any]
ActionFactory = Callable[[str, dict[str, Any]], Any]
CaseConfigFactory = Callable[[AgentSpec, EvalCase], ToolLoopRunConfig]
GradeFactory = Callable[[EvalCase, object], GradeResult]


def _probe_tool_loop_surface(
    *,
    cases: tuple[EvalCase, ...],
    agent_spec: AgentSpec,
    environment_factory: EnvironmentFactory,
    case_config: CaseConfigFactory,
) -> dict[str, Any]:
    if not cases:
        raise ValueError("tool-loop surface inference requires at least one proposal-safe case.")
    errors: list[str] = []
    domain_policy = ""
    tools: list[dict[str, Any]] = []
    for case in cases[:3]:
        try:
            config = case_config(agent_spec, case)
            config = ToolLoopRunConfig(
                provider=config.provider,
                temperature=config.temperature,
                max_steps=config.max_steps,
                request_timeout_s=config.request_timeout_s,
                log_dir=config.log_dir,
                metadata={**dict(config.metadata or {}), "surface_probe": True},
            )
            env = environment_factory(case, config)
        except Exception as exc:
            errors.append(f"{case.id}: {type(exc).__name__}: {exc}")
            continue
        if not domain_policy:
            domain_policy = str(getattr(env, "wiki", "") or getattr(env, "policy", "") or "").strip()
        raw_tools = getattr(env, "tools_info", None)
        if isinstance(raw_tools, list) and raw_tools:
            tools = [dict(item) for item in raw_tools if isinstance(item, dict)]
            break
    if not domain_policy:
        raise RuntimeError(
            "tool-loop surface inference requires a pre-trajectory domain policy/wiki exposed by the environment."
        )
    if not tools:
        detail = f" Probe errors: {'; '.join(errors)}" if errors else ""
        raise RuntimeError(
            "tool-loop surface inference requires pre-trajectory tool schemas exposed as env.tools_info."
            + detail
        )
    return {"domain_policy": domain_policy, "tools": tools}


class LiteLLMToolLoopClient:
    def __init__(self, *, max_retries: int = 0, retry_delay_s: float = 2.0) -> None:
        self.max_retries = max_retries
        self.retry_delay_s = retry_delay_s

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        model: str,
        provider: str,
        tools: list[dict[str, Any]],
        temperature: float,
        timeout_s: float | None = None,
    ) -> ToolLoopModelResponse:
        try:
            from litellm import completion
        except ModuleNotFoundError as exc:
            raise RuntimeError("litellm is required to run the generic tool-loop adapter.") from exc
        response = _with_retries(
            lambda: completion(
                messages=messages,
                model=model,
                custom_llm_provider=provider,
                tools=tools,
                temperature=temperature,
                timeout=timeout_s,
            ),
            max_retries=self.max_retries,
            retry_delay_s=self.retry_delay_s,
            timeout_s=timeout_s,
        )
        message = response.choices[0].message
        if hasattr(message, "model_dump"):
            raw_message = message.model_dump()
        elif isinstance(message, dict):
            raw_message = dict(message)
        else:
            raw_message = dict(message)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0)
        hidden = getattr(response, "_hidden_params", {}) or {}
        return ToolLoopModelResponse(
            message=raw_message,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=float(hidden.get("response_cost") or 0.0),
        )


class GeneratedToolLoopAdapter:
    """Generic interactive tool-agent adapter.

    Ratchet owns the model/tool loop and the external benchmark or task supplies
    only an environment with reset/step semantics and tool schemas.
    """

    def __init__(
        self,
        *,
        agent_spec: AgentSpec,
        environment_factory: EnvironmentFactory,
        action_factory: ActionFactory,
        respond_action_name: str = "respond",
        case_config: CaseConfigFactory | None = None,
        grade: GradeFactory | None = None,
        env_path: str = ".env",
        client: ToolLoopModelClient | None = None,
    ) -> None:
        self._agent_spec = agent_spec
        self._environment_factory = environment_factory
        self._action_factory = action_factory
        self._respond_action_name = respond_action_name
        self._case_config = case_config or _default_case_config
        self._grade = grade or _reward_grade
        self.env_path = env_path
        self._client = client or LiteLLMToolLoopClient()
        self._surface: SurfaceSpec | None = None

    def agent_spec(self) -> AgentSpec:
        return self._agent_spec

    def surface_spec(self, cases: tuple[EvalCase, ...]) -> SurfaceSpec:
        if self._surface is None:
            _load_env_file(self.env_path)
            self._surface = tool_loop_surface_from_agent_spec(
                self._agent_spec,
                probe=_probe_tool_loop_surface(
                    cases=cases,
                    agent_spec=self._agent_spec,
                    environment_factory=self._environment_factory,
                    case_config=self._case_config,
                ),
            )
        return self._surface

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        _load_env_file(self.env_path)
        config = self._case_config(self._agent_spec, case)
        env = self._environment_factory(case, config)
        tools = [dict(item) for item in getattr(env, "tools_info", [])]
        schema_by_name = _schema_by_tool_name(tools)
        metadata_by_name = _metadata_by_tool_name(self._agent_spec)
        base_context = context_graph_from_spec(self._agent_spec)
        ctx = RuntimeContext(
            case=case,
            context=base_context,
            model_config=model_config_from_spec(self._agent_spec),
        )
        ctx.model_config["temperature"] = config.temperature
        runtime = TransformRuntime(candidate)
        recorder = InteractionRecorder()
        started_at = time.perf_counter()
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0
        model_calls = 0
        reward = 0.0
        done = False
        last_observation: Any = None
        runtime.run_hook("on_task_start", ctx)

        reset_response = env.reset(task_index=_task_index(case))
        last_observation = getattr(reset_response, "observation", "")
        messages = [_system_message(env, base_context), {"role": "user", "content": str(last_observation)}]
        ctx.message_history = messages
        recorder.add_turn(actor="user", message=last_observation, metadata=_info_dict(getattr(reset_response, "info", None)))
        runtime.run_hook("after_user_message", ctx)

        for step in range(config.max_steps):
            ctx.context = base_context
            ctx.message_history = messages
            ctx.tools = copy.deepcopy(tools)
            runtime.run_hook("before_model_call", ctx)
            prompt = _system_prompt(env, ctx.context)
            call_messages = [{"role": "system", "content": prompt}, *messages[1:]]
            response = self._client.complete(
                messages=call_messages,
                model=str(ctx.model_config["model_name"]),
                provider=_provider_for_model(str(ctx.model_config["model_name"]), self._agent_spec, config),
                tools=ctx.tools,
                temperature=float(ctx.model_config.get("temperature", config.temperature)),
                timeout_s=config.request_timeout_s,
            )
            model_calls += 1
            input_tokens += response.input_tokens
            output_tokens += response.output_tokens
            cost_usd += response.cost_usd
            next_message = dict(response.message)
            ctx.raw_response = next_message
            runtime.run_hook("after_model_call", ctx)
            action_name, action_args, tool_call_id = _action_from_message(next_message, self._respond_action_name)

            if action_name == self._respond_action_name:
                content = str(action_args.get("content", ""))
                ctx.draft_response = content
                ctx.output = content
                runtime.run_hook("before_user_response", ctx)
                content = str(ctx.output)
                next_message["content"] = content
                messages.append(next_message)
                turn_index = recorder.add_turn(actor="assistant", message=content, metadata={"step": step})
                env_response = env.step(self._action_factory(self._respond_action_name, {"content": content}))
                last_observation = getattr(env_response, "observation", "")
                reward = float(getattr(env_response, "reward", 0.0) or 0.0)
                done = bool(getattr(env_response, "done", False))
                messages.append({"role": "user", "content": str(last_observation)})
                recorder.turns[turn_index].metadata.update({"responded_to_user": True})
                recorder.add_turn(
                    actor="user",
                    message=last_observation,
                    metadata=_info_dict(getattr(env_response, "info", None)),
                )
                runtime.run_hook("after_user_message", ctx)
            else:
                ctx.tool_call = {"name": action_name, "args": action_args}
                ctx.tool_schema = schema_by_name.get(action_name, {})
                ctx.tool_metadata = metadata_by_name.get(action_name, {})
                runtime.run_hook("before_tool_call", ctx)
                control = ctx.state.pop("_control", None)
                if isinstance(control, dict) and control.get("op") in {"block", "replan", "ask_user", "terminate", "retry"}:
                    content = str(control.get("message") or "The proposed tool call was blocked by a transform.")
                    messages.append({"role": "user", "content": content})
                    recorder.add_turn(actor="system", message=content, outcome=str(control.get("op")))
                    if control.get("op") == "terminate":
                        done = True
                        break
                    continue
                action_name = str(ctx.tool_call["name"])
                action_args = dict(ctx.tool_call.get("args") or {})
                action = self._action_factory(action_name, action_args)
                env_response = env.step(action)
                observation = getattr(env_response, "observation", "")
                reward = float(getattr(env_response, "reward", 0.0) or 0.0)
                done = bool(getattr(env_response, "done", False))
                status = "error" if str(observation).startswith("Error:") else "ok"
                ctx.tool_result = {
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "info": _info_dict(getattr(env_response, "info", None)),
                    "status": status,
                }
                messages.append(_assistant_tool_call_message(next_message, action_name, action_args, tool_call_id))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": action_name,
                        "content": str(observation),
                    }
                )
                turn_index = recorder.add_turn(actor="assistant", message=next_message, metadata={"step": step})
                recorder.add_tool_call(
                    name=action_name,
                    arguments=action_args,
                    result=observation,
                    status=status,
                    turn_index=turn_index,
                    metadata={"source": "tool_loop"},
                )
                if status == "error":
                    ctx.tool_error = ctx.tool_result
                    runtime.run_hook("on_tool_error", ctx)
                runtime.run_hook("after_tool_result", ctx)
            if done:
                break

        ctx.output = {"reward": reward, "done": done, "last_observation": last_observation}
        runtime.run_hook("on_task_end", ctx)
        latency_s = time.perf_counter() - started_at
        output = {
            "reward": reward,
            "done": done,
            "last_observation": last_observation,
            "steps": step + 1 if "step" in locals() else 0,
        }
        diagnostics = recorder.diagnostics(
            raw_output_text=json.dumps(output, sort_keys=True, default=str),
            terminal_state={"reward": reward, "done": done},
            terminal_reason="success" if reward >= 1.0 else "failed" if done else "step_limit",
            metadata={
                **dict(config.metadata or {}),
                "model": str(ctx.model_config["model_name"]),
                "transform_candidate_id": candidate.program.candidate_id if candidate is not None else None,
                "transform_compile_report": candidate.report.to_dict() if candidate is not None else None,
                "transform_diff": candidate.diff.to_dict() if candidate is not None else None,
                "transform_trace": list(ctx.trace_annotations),
            },
        )
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost_usd,
                model_calls=model_calls,
                tool_calls=sum(len(turn.tool_calls) for turn in diagnostics.turns),
                turns=max(1, len(diagnostics.turns)),
            ),
            diagnostics=diagnostics,
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return self._grade(case, output)

    def export(self, candidate: CompiledCandidate | None, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if candidate is not None:
            (out_dir / "compiled_candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True))
        if self._surface is None:
            raise RuntimeError("surface_spec(cases) must be inferred before exporting a tool-loop adapter.")
        (out_dir / "surface_spec.json").write_text(json.dumps(self._surface.to_dict(), indent=2, sort_keys=True))


def _default_case_config(spec: AgentSpec, case: EvalCase) -> ToolLoopRunConfig:
    runtime = dict(spec.runtime)
    metadata = dict(case.metadata)
    provider = str(runtime.get("model_provider") or _provider_for_model(spec.model, spec, ToolLoopRunConfig(provider="")))
    return ToolLoopRunConfig(
        provider=provider,
        temperature=float(runtime.get("temperature", 0.0)),
        max_steps=int(metadata.get("max_steps") or runtime.get("max_steps") or 30),
        request_timeout_s=_optional_float(metadata.get("request_timeout_s", runtime.get("request_timeout_s"))),
        log_dir=str(metadata.get("log_dir") or runtime.get("log_dir") or "results"),
        metadata=dict(metadata),
    )


def _reward_grade(case: EvalCase, output: object) -> GradeResult:
    reward = 0.0
    if isinstance(output, dict):
        reward = float(output.get("reward") or 0.0)
    return GradeResult(
        score=reward,
        passed=reward >= 1.0,
        labels=[] if reward >= 1.0 else ["reward_failed"],
        notes=f"reward={reward:.3f}",
    )


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


def _task_index(case: EvalCase) -> int | None:
    value = case.metadata.get("task_id", case.metadata.get("task_index"))
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError("interactive tool-loop cases require integer metadata.task_id/task_index when provided.")
    return value


def _provider_for_model(model: str, spec: AgentSpec, config: ToolLoopRunConfig) -> str:
    mapping = spec.runtime.get("model_provider_by_name")
    if isinstance(mapping, dict) and model in mapping:
        return str(mapping[model])
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    return config.provider


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _system_message(env: Any, context: Any) -> dict[str, Any]:
    return {"role": "system", "content": _system_prompt(env, context)}


def _system_prompt(env: Any, context: Any) -> str:
    wiki = str(getattr(env, "wiki", "") or "").strip()
    extra = context.render_text()
    if wiki and extra:
        return f"{wiki}\n\n{extra}"
    return wiki or extra


def _action_from_message(message: dict[str, Any], respond_action_name: str) -> tuple[str, dict[str, Any], str]:
    calls = message.get("tool_calls")
    if isinstance(calls, list) and calls:
        call = calls[0]
        function = call.get("function") if isinstance(call, dict) else {}
        name = str(function.get("name") or "")
        raw_args = function.get("arguments") or "{}"
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args or {})
        return name, args, str(call.get("id") or "tool_call_0")
    return respond_action_name, {"content": str(message.get("content") or "")}, "respond"


def _assistant_tool_call_message(message: dict[str, Any], name: str, args: dict[str, Any], tool_call_id: str) -> dict[str, Any]:
    row = dict(message)
    row["tool_calls"] = [
        {
            "id": tool_call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, sort_keys=True)},
        }
    ]
    return row


def _schema_by_tool_name(tools: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    schemas = {}
    for tool in tools:
        function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
        name = function.get("name")
        if name:
            schemas[str(name)] = dict(function.get("parameters") or {})
    return schemas


def _metadata_by_tool_name(spec: AgentSpec) -> dict[str, dict[str, Any]]:
    return {name: dict(tool.metadata) for name, tool in spec.tools.items()}


def _info_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def _with_retries(
    call: Callable[[], Any],
    *,
    max_retries: int,
    retry_delay_s: float,
    timeout_s: float | None = None,
) -> Any:
    attempt = 0
    while True:
        try:
            return _with_hard_timeout(call, timeout_s=timeout_s)
        except Exception as exc:
            attempt += 1
            if attempt > max_retries or not _is_transient_provider_error(exc):
                raise
            time.sleep(retry_delay_s * attempt)


def _with_hard_timeout(call: Callable[[], Any], *, timeout_s: float | None) -> Any:
    if timeout_s is None or timeout_s <= 0 or threading.current_thread() is not threading.main_thread():
        return call()
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def _raise_timeout(signum: int, frame: object) -> None:
        raise TimeoutError(f"tool-loop model request exceeded {timeout_s:.1f}s")

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return call()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


def _is_transient_provider_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in ("503", "serviceunavailable", "unavailable", "rate limit", "429"))
