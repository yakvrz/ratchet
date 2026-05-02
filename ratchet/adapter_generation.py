from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol

from ratchet.context_graph import ContextGraph, ContextSection
from ratchet.model_client import ResponsesModelClient
from ratchet.pricing import estimate_cost_usd
from ratchet.rendering import render_few_shot_prompt
from ratchet.runtime import RuntimeContext, TransformRuntime
from ratchet.surfaces import SurfaceSpec, surface_from_agent_spec
from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, EvalCase, GradeResult, OperationalMetrics, RunRecord, DiagnosticTrace


@dataclass(frozen=True)
class ModelRequest:
    context: ContextGraph
    input: Any
    model_config: dict[str, Any]
    text: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SingleCallHarness(Protocol):
    def agent_spec(self) -> AgentSpec:
        ...

    def build_model_request(self, spec: AgentSpec, case: EvalCase) -> ModelRequest:
        ...

    def parse_output(self, raw_output_text: str) -> object:
        ...

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        ...


class AdapterGenerator:
    def infer_surface(self, harness: SingleCallHarness, cases: tuple[EvalCase, ...]) -> SurfaceSpec:
        if not cases:
            raise ValueError("single-call surface inference requires at least one proposal-safe case.")
        return surface_from_agent_spec(harness.agent_spec())

    def build_runtime_adapter(
        self,
        harness: SingleCallHarness,
        *,
        env_path: str = ".env",
        client: ResponsesModelClient | None = None,
    ) -> "GeneratedSingleCallAdapter":
        return GeneratedSingleCallAdapter(harness=harness, env_path=env_path, client=client)


class GeneratedSingleCallAdapter:
    def __init__(
        self,
        *,
        harness: SingleCallHarness,
        env_path: str = ".env",
        client: ResponsesModelClient | None = None,
    ) -> None:
        self.harness = harness
        self.env_path = env_path
        self._client = client
        self._surface: SurfaceSpec | None = None

    def agent_spec(self) -> AgentSpec:
        return self.harness.agent_spec()

    def surface_spec(self, cases: tuple[EvalCase, ...]) -> SurfaceSpec:
        if self._surface is None:
            self._surface = AdapterGenerator().infer_surface(self.harness, cases)
        return self._surface

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        if candidate is not None and not isinstance(candidate, CompiledCandidate):
            raise TypeError(
                f"Generated adapters execute CompiledCandidate instances, got {type(candidate).__name__}."
            )
        base_spec = self.harness.agent_spec()
        compiled = candidate
        request = self.harness.build_model_request(base_spec, case)
        runtime = TransformRuntime(compiled)
        ctx = RuntimeContext(
            case=case,
            context=request.context,
            model_config=dict(request.model_config),
        )
        runtime.run_hook("on_task_start", ctx)
        runtime.run_hook("before_model_call", ctx)
        started_at = time.perf_counter()
        response = self._client_or_create().create_response(
            model=str(ctx.model_config["model_name"]),
            reasoning={"effort": str(ctx.model_config.get("reasoning_effort", "low"))},
            instructions=ctx.context.render_text(),
            input=request.input,
            max_output_tokens=int(ctx.model_config.get("max_tokens", ctx.model_config.get("output_cap", 512))),
            text=request.text,
        )
        latency_s = time.perf_counter() - started_at
        raw_output_text = str(response.output_text).strip()
        ctx.raw_response = raw_output_text
        runtime.run_hook("after_model_call", ctx)
        output = self.harness.parse_output(raw_output_text)
        ctx.draft_response = output
        ctx.output = output
        runtime.run_hook("before_user_response", ctx)
        output = ctx.output
        runtime.run_hook("on_task_end", ctx)
        input_tokens = int(getattr(response.usage, "input_tokens", 0))
        output_tokens = int(getattr(response.usage, "output_tokens", 0))
        model = str(ctx.model_config["model_name"])
        return RunRecord(
            output=output,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=estimate_cost_usd(model, input_tokens, output_tokens),
            ),
            diagnostics=DiagnosticTrace(
                raw_output_text=raw_output_text,
                metadata={
                    **dict(request.metadata or {}),
                    "model": model,
                    "finish_reason": str(getattr(response, "finish_reason", "") or ""),
                    "requested_output_cap": int(ctx.model_config.get("max_tokens", ctx.model_config.get("output_cap", 512))),
                    "raw_output_length": len(raw_output_text),
                    "invalid_output": isinstance(output, dict) and "invalid_output" in output,
                    "output_tokens": output_tokens,
                    "output_item_types": [item.type for item in getattr(response, "output", [])],
                    "transform_candidate_id": compiled.program.candidate_id if compiled is not None else None,
                    "transform_compile_report": compiled.report.to_dict() if compiled is not None else None,
                    "transform_diff": compiled.diff.to_dict() if compiled is not None else None,
                    "transform_trace": list(ctx.trace_annotations),
                    "rendered_context_sections": ctx.context.section_names(),
                },
            ),
        )

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        return self.harness.grade(case, output)

    def export(self, candidate: CompiledCandidate | None, out_dir: Path) -> None:
        if candidate is not None and not isinstance(candidate, CompiledCandidate):
            raise TypeError(
                f"Generated adapters export CompiledCandidate instances, got {type(candidate).__name__}."
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        if candidate is not None:
            (out_dir / "compiled_candidate.json").write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True))
        if self._surface is None:
            raise RuntimeError("surface_spec(cases) must be inferred before exporting a generated adapter.")
        (out_dir / "surface_spec.json").write_text(json.dumps(self._surface.to_dict(), indent=2, sort_keys=True))

    def _client_or_create(self) -> ResponsesModelClient:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        return self._client


def context_graph_from_spec(
    spec: AgentSpec,
    *,
    section_order: list[str] | None = None,
    include_output_contract: bool = False,
    include_few_shot: bool = True,
) -> ContextGraph:
    sections: list[ContextSection] = []
    order = section_order or list(spec.instructions)
    for name in order:
        text = spec.instructions.get(name)
        if text:
            sections.append(ContextSection(name=name, role="system", content=text, required=True))
    for name, text in spec.instructions.items():
        if name not in set(order) and text:
            sections.append(ContextSection(name=name, role="system", content=text, required=True))
    if include_few_shot:
        few_shot = render_few_shot_prompt(spec.few_shot)
        if few_shot:
            sections.append(ContextSection(name="few_shot", role="system", content=few_shot))
    if include_output_contract and spec.output_contract:
        sections.append(ContextSection(name="output_contract", role="system", content=spec.output_contract, required=True))
    return ContextGraph(tuple(sections))


def model_config_from_spec(spec: AgentSpec) -> dict[str, Any]:
    return {
        "model_name": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "low")),
        "max_tokens": int(spec.runtime.get("output_cap", spec.runtime.get("max_tokens", 512))),
    }
