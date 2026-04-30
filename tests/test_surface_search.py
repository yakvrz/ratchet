from __future__ import annotations

from dataclasses import replace
import unittest

from ratchet.context_graph import ContextGraph
from ratchet.runtime import RuntimeContext, TransformRuntime
from ratchet.results import CaseEvaluation, CandidateSummary
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.candidates import CandidateProposal, CandidateSurfaceApplication
from ratchet.surface_search import (
    TransformContextKey,
    TransformContextState,
    build_search_hypothesis,
)
from ratchet.transform_validation import (
    validate_candidate_transform,
)
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord
from ratchet.surface_opportunities import SurfaceOpportunity, validate_candidate_surface_applications


def _summary(labels: list[list[str]]) -> CandidateSummary:
    evaluations: list[CaseEvaluation] = []
    for index, case_labels in enumerate(labels, start=1):
        passed = not case_labels
        evaluations.append(
            CaseEvaluation(
                case=EvalCase(id=f"case-{index}", split="dev", input=f"input {index}"),
                record=RunRecord(
                    output="ok" if passed else "wrong",
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=5,
                        total_tokens=15,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(metadata={"finish_reason": "stop"}),
                ),
                grade=GradeResult(score=1.0 if passed else 0.0, passed=passed, labels=case_labels),
            )
        )
    return CandidateSummary(candidate_id="baseline", candidate=None, split="dev", evaluations=evaluations)


def _context_candidate() -> CandidateProposal:
    return CandidateProposal(
        program=TransformProgram.from_dict(
            {
                "candidate_id": "context",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": "extra_rule",
                        "content": "Answer with the requested format.",
                    }
                ],
            }
        ),
        applications=[
            CandidateSurfaceApplication(
                surface_opportunity_id="surface.surface_context.system_prompt",
                rationale="extra_rule",
            )
        ],
        experiment_id="intent-1",
        hypothesis="Add a concise formatting rule.",
    )


class TransformLibraryTests(unittest.TestCase):
    def test_search_hypothesis_uses_surface_spec(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="large", instructions={"system_prompt": "Answer."})
        )

        hypothesis = build_search_hypothesis(
            summary=_summary([["wrong_label"], []]),
            surface=surface,
            objective=OptimizationObjective(),
            history=[],
        )

        self.assertIn("surface_context", hypothesis.active_mechanisms)
        self.assertTrue(hypothesis.context_states)

    def test_compiler_rejects_unsupported_hook(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad",
                "patches": [
                    {"hook": "before_tool_call", "op": "normalize_tool_args", "target": "tool_call"}
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "unsupported_hook")

    def test_compiler_rejects_context_patch_without_content(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="base", instructions={"system_prompt": "Answer."})
        )
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-context",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": "empty_rule",
                        "value": "This is the wrong field.",
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "context_content_required")
        self.assertIn("content", compiled.report.rejection.message)

    def test_compiler_rejects_empty_context_replacement(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="base", instructions={"system_prompt": "Answer."})
        )
        program = TransformProgram.from_dict(
            {
                "candidate_id": "empty-replace",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "replace_context_section",
                        "section": "system_prompt",
                        "content": {},
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "context_content_required")

    def test_compiler_rejects_simulator_stop_marker_in_candidate_content(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="base", instructions={"system_prompt": "Answer."})
        )
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-boundary",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "add_context_section",
                        "section": "bad_rule",
                        "content": "Never emit ###STOP### until the task is complete.",
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "immutable_boundary_violation")

    def test_compiler_rejects_model_name_outside_surface_options(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="base", model_options=["base", "larger"])
        )
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-model",
                "patches": [
                    {
                        "hook": "before_model_call",
                        "op": "set_model_config",
                        "field": "model_name",
                        "value": "unavailable",
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "model_name_not_allowed")

    def test_compiler_rejects_prose_only_validation(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-validation",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "validate",
                        "content": "Reject unsupported claims.",
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "validation_checks_required")

    def test_compiler_accepts_structured_validation_check_from_surface_registry(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "structured-validation",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "validate",
                        "target": "draft_response",
                        "checks": [{"type": "json_object"}],
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "compiled")

    def test_candidate_validation_rejects_log_only_control_candidate(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        candidate = CandidateProposal(
            program=TransformProgram.from_dict(
                {
                    "candidate_id": "control",
                    "patches": [{"hook": "on_task_end", "op": "log_event", "content": "baseline_control"}],
                }
            ),
            applications=[
                CandidateSurfaceApplication(
                    surface_opportunity_id="surface.surface_tool_loop.tool_loop",
                    rationale="baseline comparator",
                )
            ],
            experiment_id="intent-1",
            candidate_role="control",
        )

        error = validate_candidate_transform(candidate, surface=surface)

        self.assertEqual(error, "control candidates are measurement infrastructure, not optimizer candidates")

    def test_empty_source_case_ids_do_not_make_surface_application_an_example_selection(self) -> None:
        error = validate_candidate_surface_applications(
            applications=[
                CandidateSurfaceApplication(
                    surface_opportunity_id="surface.surface_tool_loop.before_tool_call",
                    selection={"source_case_ids": [], "selection_strategy": "global"},
                )
            ],
            surface_opportunities=[
                SurfaceOpportunity(
                    surface_opportunity_id="surface.surface_tool_loop.before_tool_call",
                    label="before tool call",
                    family="surface",
                    mechanism="surface_tool_loop",
                    target_name="before_tool_call",
                    target_kind="tool",
                    target_path="hooks.before_tool_call",
                    ops=["validate"],
                    value_schema={},
                    semantic_role="tool_loop",
                    behavioral_axes=[],
                    expected_scope="global",
                    risk="low",
                    measurements=[],
                    description="before tool call",
                )
            ],
        )

        self.assertIsNone(error)

    def test_candidate_validation_rejects_inactive_surface_family(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="base", instructions={"system_prompt": "Answer."})
        )
        hypothesis = build_search_hypothesis(
            summary=_summary([["wrong_label"], []]),
            surface=surface,
            objective=OptimizationObjective(),
            history=[],
        )
        family_state = hypothesis.mechanism_states["surface_context"]
        paused_hypothesis = replace(
            hypothesis,
            mechanism_states={
                **hypothesis.mechanism_states,
                "surface_context": replace(family_state, state="paused"),
            },
        )

        error = validate_candidate_transform(
            _context_candidate(),
            surface=surface,
            search_hypothesis=paused_hypothesis,
        )

        self.assertEqual(error, "inactive surface mechanism 'surface_context'")

    def test_candidate_validation_rejects_constrained_exact_context(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(name="sample", model="base", instructions={"system_prompt": "Answer."})
        )
        hypothesis = build_search_hypothesis(
            summary=_summary([["wrong_label"], []]),
            surface=surface,
            objective=OptimizationObjective(),
            history=[],
        )
        candidate = _context_candidate()
        context_key = TransformContextKey.from_candidate(candidate)
        constrained_hypothesis = replace(
            hypothesis,
            context_states={
                **hypothesis.context_states,
                context_key.id: TransformContextState(
                    key=context_key,
                    state="constrained",
                    suitability=0.8,
                    reason="Previous near-duplicate failed.",
                ),
            },
        )

        error = validate_candidate_transform(
            candidate,
            surface=surface,
            search_hypothesis=constrained_hypothesis,
        )

        self.assertIn("constrained transform context", str(error))

    def test_generated_surface_no_longer_compiles_extract_claims(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "claims",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "extract_claims",
                        "target": "draft_response",
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "unsupported_operation")

    def test_runtime_resolves_trace_reference_at_task_end(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "trace-ref",
                    "patches": [
                        {
                            "hook": "on_task_end",
                            "op": "trace_annotation",
                            "fields": {"prior": {"$ref": "trace"}},
                        }
                    ],
                }
            ),
            surface,
        )
        ctx = RuntimeContext(
            case=EvalCase(id="case-1", split="dev", input="x"),
            context=ContextGraph(),
            model_config={},
            trace_annotations=[{"hook": "before_user_response", "op": "rewrite_response"}],
        )

        TransformRuntime(candidate).run_hook("on_task_end", ctx)

        self.assertEqual(ctx.trace_annotations[-1]["fields"]["prior"][0]["op"], "rewrite_response")


if __name__ == "__main__":
    unittest.main()
