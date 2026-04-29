from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
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
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import CompiledCandidate, TransformPatch, TransformProgram
from ratchet.types import AgentPatch, AgentSpec, EvalCase, GradeResult, OperationalMetrics, RunRecord, DiagnosticTrace


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
    def infer_surface(self, harness: SingleCallHarness) -> SurfaceSpec:
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

    def surface_spec(self) -> SurfaceSpec:
        if self._surface is None:
            self._surface = AdapterGenerator().infer_surface(self.harness)
        return self._surface

    def run_case(self, case: EvalCase, candidate: AgentPatch | CompiledCandidate | None = None) -> RunRecord:
        base_spec = self.harness.agent_spec()
        compiled = self._compiled_candidate(candidate)
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
            model=str(ctx.model_config["model"]),
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
        model = str(ctx.model_config["model"])
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

    def export(self, candidate: AgentPatch | CompiledCandidate, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        compiled = self._compiled_candidate(candidate)
        if compiled is not None:
            (out_dir / "compiled_candidate.json").write_text(json.dumps(compiled.to_dict(), indent=2, sort_keys=True))
            (out_dir / "surface_spec.json").write_text(json.dumps(self.surface_spec().to_dict(), indent=2, sort_keys=True))
        if isinstance(candidate, CompiledCandidate):
            return
        spec = self.harness.agent_spec().apply_patch(candidate)
        (out_dir / "patch.json").write_text(json.dumps(candidate.to_dict(), indent=2, sort_keys=True))
        (out_dir / "agent_spec.json").write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True))

    def _client_or_create(self) -> ResponsesModelClient:
        if self._client is None:
            self._client = ResponsesModelClient(env_path=self.env_path)
        return self._client

    def _compiled_candidate(self, candidate: AgentPatch | CompiledCandidate | None) -> CompiledCandidate | None:
        if candidate is None:
            return None
        if isinstance(candidate, CompiledCandidate):
            return candidate
        if candidate.is_empty:
            return None
        program = transform_program_from_agent_patch(candidate, self.surface_spec())
        return TransformCompiler().compile_or_raise(program, self.surface_spec())


def transform_program_from_agent_patch(patch: AgentPatch, surface: SurfaceSpec) -> TransformProgram:
    patches: list[TransformPatch] = []
    for index, operation in enumerate(patch.operations):
        target = operation.target
        value = operation.value
        if operation.op == "add_instruction":
            section = _strip_prefix(target, "instructions.")
            current = _surface_section_content(surface, section)
            if current is None:
                patches.append(
                    TransformPatch.from_dict(
                        {
                            "hook": "before_model_call",
                            "op": "add_context_section",
                            "section": section,
                            "content": str(value),
                            "position": "end",
                            "required": True,
                        }
                    )
                )
            else:
                text = str(value).strip()
                content = f"{current.rstrip()}\n\n{text}" if text and text not in current else current
                patches.append(
                    TransformPatch.from_dict(
                        {
                            "hook": "before_model_call",
                            "op": "replace_context_section",
                            "section": section,
                            "content": content,
                        }
                    )
                )
            continue
        if operation.op == "revise_instruction":
            patches.append(
                TransformPatch.from_dict(
                    {
                        "hook": "before_model_call",
                        "op": "replace_context_section",
                        "section": _strip_prefix(target, "instructions."),
                        "content": str(value),
                    }
                )
            )
            continue
        if operation.op == "add_output_constraint":
            patches.append(
                TransformPatch.from_dict(
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": f"output_constraint_patch_{index}",
                        "content": str(value),
                        "position": "end",
                        "required": True,
                    }
                )
            )
            continue
        if operation.op == "change_model":
            patches.append(
                TransformPatch.from_dict(
                    {
                        "hook": "before_model_call",
                        "op": "set_model_config",
                        "field": "model",
                        "value": str(value),
                    }
                )
            )
            continue
        if operation.op == "set_runtime_param":
            field = _runtime_field(target)
            patches.append(
                TransformPatch.from_dict(
                    {
                        "hook": "before_model_call",
                        "op": "set_model_config",
                        "field": field,
                        "value": value,
                    }
                )
            )
            continue
        if operation.op == "add_few_shot":
            examples = value if isinstance(value, list) else [value]
            patches.append(
                TransformPatch.from_dict(
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": f"few_shot_patch_{index}",
                        "content": render_few_shot_prompt(
                            [dict(item) if isinstance(item, dict) else {"text": str(item)} for item in examples]
                        ),
                        "position": "end",
                    }
                )
            )
            continue
        if operation.op in {"revise_tool_description", "revise_tool_policy", "add_verifier_retry"}:
            raise ValueError(f"Patch operation {operation.op!r} is not expressible on generated single-call adapters.")
        raise ValueError(f"Unsupported patch operation: {operation.op}")
    digest = sha256(json.dumps(patch.to_dict(), sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]
    return TransformProgram(
        candidate_id=str(patch.metadata.get("candidate_id") or f"agent_patch_{digest}"),
        patches=tuple(patches),
        metadata={
            **dict(patch.metadata),
            "source_patch": patch.to_dict(),
            "lowered_from": "AgentPatch",
            "rationale": patch.rationale,
            "expected_effect": patch.expected_effect,
        },
    )


def _strip_prefix(value: str, prefix: str) -> str:
    return value[len(prefix) :] if value.startswith(prefix) else value


def _surface_section_content(surface: SurfaceSpec, section_name: str) -> str | None:
    for section in surface.context.graph.sections:
        if section.name == section_name:
            return str(section.content)
    return None


def _runtime_field(target: str) -> str:
    field = _strip_prefix(target, "runtime.")
    if field == "output_cap":
        return "max_tokens"
    return field


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
        "model": spec.model,
        "reasoning_effort": str(spec.runtime.get("reasoning_effort", "low")),
        "max_tokens": int(spec.runtime.get("output_cap", spec.runtime.get("max_tokens", 512))),
    }
