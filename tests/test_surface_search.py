from __future__ import annotations

import unittest

from ratchet.context_graph import ContextGraph
from ratchet.optimizer import compose_transform_candidate
from ratchet.runtime import RuntimeContext, TransformRuntime
from ratchet.surfaces import surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import TransformProgram
from ratchet.candidates import CandidateProposal, CandidateSurfaceApplication
from ratchet.transform_validation import (
    validate_candidate_transform,
)
from ratchet.types import AgentSpec, EvalCase
from ratchet.surface_opportunities import SurfaceOpportunity, validate_candidate_surface_applications


class TransformLibraryTests(unittest.TestCase):
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

    def test_compiler_rejects_operator_tree_conditions_before_runtime(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-condition",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "validate",
                        "target": "draft_response",
                        "checks": [{"type": "json_object"}],
                        "when": {"==": [{"$ref": "draft_response"}, "lookup_order"]},
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "invalid_condition")

    def test_compiler_rejects_unavailable_condition_references(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-condition-ref",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "validate",
                        "target": "draft_response",
                        "checks": [{"type": "json_object"}],
                        "when": {"tool_call.name": "lookup_order"},
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "unavailable_reference")

    def test_compiler_validates_on_fail_runtime_operation_shape(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        program = TransformProgram.from_dict(
            {
                "candidate_id": "bad-on-fail",
                "patches": [
                    {
                        "hook": "before_user_response",
                        "op": "validate",
                        "target": "draft_response",
                        "checks": [{"type": "clarification_response"}],
                        "on_fail": {"op": "replan", "message": "Try again."},
                    }
                ],
            }
        )

        compiled = TransformCompiler().compile(program, surface)

        self.assertEqual(compiled.report.status, "rejected")
        self.assertEqual(compiled.report.rejection.code, "invalid_on_fail")

    def test_response_clarification_guard_can_rewrite_implicit_choice_prompt(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        candidate = TransformCompiler().compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "clarify-response",
                    "patches": [
                        {
                            "hook": "before_user_response",
                            "op": "validate",
                            "target": "draft_response",
                            "checks": [{"type": "clarification_response"}],
                            "on_fail": {
                                "op": "rewrite_response",
                                "message": "Please clarify which option you want me to use before I continue.",
                            },
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
            draft_response="Do you want the mug or the lamp?",
            output="Do you want the mug or the lamp?",
        )

        TransformRuntime(candidate).run_hook("before_user_response", ctx)

        self.assertIn("Please clarify which option", ctx.output)
        self.assertTrue(any(item["op"] == "rewrite_response" for item in ctx.trace_annotations))

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

    def test_composed_candidate_deduplicates_parent_state_and_context_definitions(self) -> None:
        surface = surface_from_agent_spec(AgentSpec(name="sample", model="base"))
        compiler = TransformCompiler()
        parent = compiler.compile_or_raise(
            TransformProgram.from_dict(
                {
                    "candidate_id": "parent",
                    "patches": [
                        {"op": "define_state", "field": "observed_ids", "type": "list[string]", "initial": []},
                        {
                            "hook": "before_model_call",
                            "op": "render_state_section",
                            "section": "observed_identifiers",
                            "fields": ["observed_ids"],
                        },
                    ],
                }
            ),
            surface,
        )
        child = TransformProgram.from_dict(
            {
                "candidate_id": "child",
                "patches": [
                    {"op": "define_state", "field": "observed_ids", "type": "list[string]", "initial": []},
                    {"op": "define_state", "field": "observed_item_ids", "type": "list[string]", "initial": []},
                    {
                        "hook": "before_model_call",
                        "op": "render_state_section",
                        "section": "observed_identifiers",
                        "fields": ["observed_item_ids"],
                    },
                ],
            }
        )

        composed = compose_transform_candidate(parent, child, compiler=compiler, surface=surface)

        self.assertEqual(composed.report.status, "compiled")
        define_fields = [
            patch.op.params["field"]
            for patch in composed.program.patches
            if patch.op.op == "define_state"
        ]
        self.assertEqual(define_fields, ["observed_ids", "observed_item_ids"])
        render_patches = [patch for patch in composed.program.patches if patch.op.op == "render_state_section"]
        self.assertEqual(len(render_patches), 1)
        self.assertEqual(render_patches[0].op.params["fields"], ["observed_ids", "observed_item_ids"])


if __name__ == "__main__":
    unittest.main()
