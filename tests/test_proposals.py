from __future__ import annotations

import json
import unittest

from ratchet.experiments import SearchBrief, SearchPlan
from ratchet.model_client import CompatResponse, CompatUsage
from ratchet.proposals import CandidateImplementer, _surface_affordance_proposals
from ratchet.results import CandidateSummary, CaseEvaluation
from ratchet.surface_opportunities import generate_surface_opportunities
from ratchet.surfaces import surface_from_agent_spec, tool_loop_surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.types import AgentSpec, DiagnosticTrace, EvalCase, GradeResult, OperationalMetrics, OptimizationObjective, RunRecord


class _FakeProposalClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def create_response(self, **kwargs: object) -> CompatResponse:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return CompatResponse(
            id=f"fake-{len(self.calls)}",
            output=[],
            output_text=json.dumps(payload),
            usage=CompatUsage(input_tokens=10, output_tokens=10),
            finish_reason="stop",
        )


def _summary(candidate=None) -> CandidateSummary:
    return CandidateSummary(
        candidate_id="baseline",
        candidate=candidate,
        split="dev",
        evaluations=[
            CaseEvaluation(
                case=EvalCase(id="case-1", split="dev", input="x"),
                record=RunRecord(
                    output="ok",
                    metrics=OperationalMetrics(
                        latency_s=1.0,
                        input_tokens=10,
                        output_tokens=10,
                        total_tokens=20,
                        cost_usd=0.001,
                    ),
                    diagnostics=DiagnosticTrace(),
                ),
                grade=GradeResult(score=0.0, passed=False, labels=["needs_contract"]),
            )
        ],
    )


class CandidateImplementerContractTests(unittest.TestCase):
    def test_repairs_invalid_uncovered_brief_even_when_other_candidate_is_valid(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                model_options=["base", "larger"],
                instructions={"system_prompt": "Answer."},
            )
        )
        opportunities = generate_surface_opportunities(surface)
        context_opp = next(item for item in opportunities if item.mechanism == "surface_context")
        model_opp = next(item for item in opportunities if item.mechanism == "surface_model")
        search_plan = SearchPlan(
            plan_id="plan",
            diagnosis="Need context and model candidates.",
            hypotheses=["Cover both mechanisms."],
            target_mechanisms=["surface_context", "surface_model"],
            briefs=[
                SearchBrief(
                    brief_id="context_brief",
                    mechanism_class="surface_context",
                    hypothesis="Improve context.",
                    surface_opportunity_ids=[context_opp.surface_opportunity_id],
                ),
                SearchBrief(
                    brief_id="model_brief",
                    mechanism_class="surface_model",
                    hypothesis="Try a stronger allowed model.",
                    surface_opportunity_ids=[model_opp.surface_opportunity_id],
                ),
            ],
        )
        first_payload = {
            "experiments": [
                {
                    "experiment_id": "context_brief",
                    "mechanism_class": "surface_context",
                    "hypothesis": "Improve context.",
                    "candidates": [
                        {
                            "candidate_role": "atomic",
                            "hypothesis": "Wrong hook but right mechanism.",
                            "applications": [{"surface_opportunity_id": context_opp.surface_opportunity_id}],
                            "program": {
                                "candidate_id": "bad_context",
                                "patches": [
                                    {
                                        "hook": "on_task_start",
                                        "op": "replace_context_section",
                                        "section": "system_prompt",
                                        "content": "Clarify constraints.",
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "experiment_id": "model_brief",
                    "mechanism_class": "surface_model",
                    "hypothesis": "Try a stronger allowed model.",
                    "candidates": [
                        {
                            "candidate_role": "atomic",
                            "hypothesis": "Use allowed model.",
                            "applications": [{"surface_opportunity_id": model_opp.surface_opportunity_id}],
                            "program": {
                                "candidate_id": "model_candidate",
                                "patches": [
                                    {
                                        "hook": "before_model_call",
                                        "op": "set_model_config",
                                        "field": "model_name",
                                        "value": "larger",
                                    }
                                ],
                            },
                        }
                    ],
                },
            ]
        }
        repair_payload = {
            "experiments": [
                {
                    "experiment_id": "context_brief",
                    "mechanism_class": "surface_context",
                    "hypothesis": "Improve context.",
                    "candidates": [
                        {
                            "candidate_role": "atomic",
                            "hypothesis": "Legal context edit.",
                            "applications": [{"surface_opportunity_id": context_opp.surface_opportunity_id}],
                            "program": {
                                "candidate_id": "fixed_context",
                                "patches": [
                                    {
                                        "hook": "before_model_call",
                                        "op": "replace_context_section",
                                        "section": "system_prompt",
                                        "content": "Clarify constraints.",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        }
        implementer = CandidateImplementer(env_path="", model="fake-model", reasoning_effort="low")
        implementer._client = _FakeProposalClient([first_payload, repair_payload])

        proposals, _analysis = implementer.propose(
            _summary(),
            surface,
            objective=OptimizationObjective(),
            seen_hashes=set(),
            current_spec=None,
            history=[],
            search_plan=search_plan,
            proposal_budget=2,
            surface_opportunities=opportunities,
        )

        self.assertEqual({proposal.experiment_id for proposal in proposals}, {"context_brief", "model_brief"})
        self.assertEqual(len(implementer._client.calls), 2)
        audit = implementer.last_stats.plan_audit or {}
        self.assertEqual(audit["valid_covered_brief_ids"], ["context_brief", "model_brief"])
        self.assertEqual(audit["unrepaired_invalid_brief_ids"], [])
        self.assertIn("context_brief", audit["invalid_covered_brief_ids"])
        self.assertTrue(
            any(
                "unsupported_operation" in row.get("invalid_reason", "")
                for row in implementer.last_invalid_candidate_rows
            )
        )

    def test_repair_cannot_escape_to_different_brief(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                model_options=["base", "larger"],
                instructions={"system_prompt": "Answer."},
            )
        )
        opportunities = generate_surface_opportunities(surface)
        context_opp = next(item for item in opportunities if item.mechanism == "surface_context")
        model_opp = next(item for item in opportunities if item.mechanism == "surface_model")
        search_plan = SearchPlan(
            plan_id="plan",
            diagnosis="Need context candidate.",
            hypotheses=["Cover context mechanism."],
            target_mechanisms=["surface_context", "surface_model"],
            briefs=[
                SearchBrief(
                    brief_id="context_brief",
                    mechanism_class="surface_context",
                    hypothesis="Improve context.",
                    surface_opportunity_ids=[context_opp.surface_opportunity_id],
                ),
                SearchBrief(
                    brief_id="model_brief",
                    mechanism_class="surface_model",
                    hypothesis="Try a stronger allowed model.",
                    surface_opportunity_ids=[model_opp.surface_opportunity_id],
                ),
            ],
        )
        first_payload = {
            "experiments": [
                {
                    "experiment_id": "context_brief",
                    "mechanism_class": "surface_context",
                    "hypothesis": "Improve context.",
                    "candidates": [
                        {
                            "candidate_role": "atomic",
                            "hypothesis": "Wrong hook.",
                            "applications": [{"surface_opportunity_id": context_opp.surface_opportunity_id}],
                            "program": {
                                "candidate_id": "bad_context",
                                "patches": [
                                    {
                                        "hook": "on_task_start",
                                        "op": "replace_context_section",
                                        "section": "system_prompt",
                                        "content": "Clarify constraints.",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        }
        escaping_repair_payload = {
            "experiments": [
                {
                    "experiment_id": "model_brief",
                    "mechanism_class": "surface_model",
                    "hypothesis": "Escape repair.",
                    "candidates": [
                        {
                            "candidate_role": "atomic",
                            "hypothesis": "Wrong brief.",
                            "applications": [{"surface_opportunity_id": model_opp.surface_opportunity_id}],
                            "program": {
                                "candidate_id": "escaped_model",
                                "patches": [
                                    {
                                        "hook": "before_model_call",
                                        "op": "set_model_config",
                                        "field": "model_name",
                                        "value": "larger",
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        }
        implementer = CandidateImplementer(env_path="", model="fake-model", reasoning_effort="low")
        implementer._client = _FakeProposalClient([first_payload, escaping_repair_payload])

        proposals, _analysis = implementer.propose(
            _summary(),
            surface,
            objective=OptimizationObjective(),
            seen_hashes=set(),
            current_spec=None,
            history=[],
            search_plan=search_plan,
            proposal_budget=1,
            surface_opportunities=opportunities,
        )

        self.assertEqual(proposals, [])
        self.assertTrue(
            any(
                row.get("invalid_reason") == "repair changed experiment_id outside requested invalid brief"
                for row in implementer.last_invalid_candidate_rows
            )
        )


class SurfaceAffordanceProposalTests(unittest.TestCase):
    def test_identifier_flow_affordance_emits_composed_state_guard_candidate(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Inspect records before mutation.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_orders",
                            "description": "List orders.",
                            "parameters": {"type": "object"},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "description": "Inspect one order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "cancel_order",
                            "description": "Cancel an order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        },
                    },
                ],
                "tool_result_schemas": {
                    "list_orders": {
                        "type": "object",
                        "properties": {
                            "orders": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"order_id": {"type": "string"}},
                                },
                            }
                        },
                    },
                    "get_order": {
                        "type": "object",
                        "properties": {
                            "order": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            }
                        },
                    },
                },
            },
        )
        opportunities = generate_surface_opportunities(surface, active_mechanisms=["surface_tool_loop"])
        affordance_id = "surface.surface_tool_loop.inspect_before_mutate_order_id"
        search_plan = SearchPlan(
            plan_id="plan_tool_loop",
            diagnosis="Tool calls are not grounded in inspected identifiers.",
            hypotheses=["Ground mutating calls in inspected identifiers."],
            target_mechanisms=["surface_tool_loop"],
            briefs=[
                SearchBrief(
                    brief_id="brief_tool_loop",
                    mechanism_class="surface_tool_loop",
                    hypothesis="Ground mutating calls in inspected identifiers.",
                    surface_opportunity_ids=[affordance_id],
                    candidate_roles=["composed"],
                )
            ],
        )

        proposals = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=opportunities,
            search_plan=search_plan,
            proposal_budget=1,
        )

        self.assertEqual(len(proposals), 1)
        candidate = proposals[0]
        self.assertEqual(candidate.experiment_id, "brief_tool_loop")
        self.assertEqual(candidate.candidate_role, "composed")
        patches = [patch.to_dict() for patch in candidate.program.patches]
        self.assertIn(
            {
                "hook": "after_tool_result",
                "op": "append_state",
                "field": "inspected_order_ids",
                "value": {"$ref": "tool_result.parsed.order.order_id"},
                "extend": False,
                "when": {"tool_call.name": "get_order"},
            },
            patches,
        )
        self.assertIn(
            {
                "hook": "after_tool_result",
                "op": "append_state",
                "field": "listed_order_ids",
                "value": {"$ref": "tool_result.parsed.orders[].order_id"},
                "extend": True,
                "when": {"tool_call.name": "list_orders"},
            },
            patches,
        )
        self.assertTrue(
            any(
                patch.get("op") == "validate"
                and patch.get("tool") == "cancel_order"
                and patch.get("checks") == [
                    {"type": "tool_arg_in_state", "state_field": "inspected_order_ids", "arg": "order_id"}
                ]
                for patch in patches
            )
        )
        self.assertTrue(
            any(
                patch.get("op") == "render_state_section"
                and patch.get("fields") == ["inspected_order_ids", "listed_order_ids"]
                for patch in patches
            )
        )

    def test_spare_budget_adds_context_ablation_after_primary_scaffold(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Inspect records before mutation.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_orders",
                            "description": "List orders.",
                            "parameters": {"type": "object"},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order",
                            "description": "Inspect one order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "cancel_order",
                            "description": "Cancel an order.",
                            "parameters": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            },
                        },
                    },
                ],
                "tool_result_schemas": {
                    "list_orders": {
                        "type": "object",
                        "properties": {
                            "orders": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"order_id": {"type": "string"}},
                                },
                            }
                        },
                    },
                    "get_order": {
                        "type": "object",
                        "properties": {
                            "order": {
                                "type": "object",
                                "properties": {"order_id": {"type": "string"}},
                            }
                        },
                    },
                },
            },
        )
        opportunities = generate_surface_opportunities(surface, active_mechanisms=["surface_tool_loop"])
        search_plan = SearchPlan(
            plan_id="plan_tool_loop",
            diagnosis="Tool calls are not grounded in inspected identifiers.",
            hypotheses=["Ground mutating calls in inspected identifiers."],
            target_mechanisms=["surface_tool_loop"],
            briefs=[
                SearchBrief(
                    brief_id="brief_tool_loop",
                    mechanism_class="surface_tool_loop",
                    hypothesis="Ground mutating calls in inspected identifiers.",
                    surface_opportunity_ids=["surface.surface_tool_loop.inspect_before_mutate_order_id"],
                    candidate_roles=["composed", "ablation"],
                )
            ],
        )

        proposals = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=opportunities,
            search_plan=search_plan,
            proposal_budget=2,
        )

        self.assertEqual([proposal.candidate_role for proposal in proposals], ["composed", "ablation"])
        ablation_ops = [patch.op.op for patch in proposals[1].program.patches]
        self.assertNotIn("render_state_section", ablation_ops)
        self.assertIn("validate", ablation_ops)

    def test_identifier_flow_affordance_requires_inspection_producer(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Inspect records before mutation.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "list_records",
                            "description": "List records.",
                            "parameters": {"type": "object"},
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_record",
                            "description": "Update a record.",
                            "parameters": {
                                "type": "object",
                                "properties": {"record_id": {"type": "string"}},
                            },
                        },
                    },
                ],
                "tool_result_schemas": {
                    "list_records": {
                        "type": "object",
                        "properties": {
                            "records": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {"record_id": {"type": "string"}},
                                },
                            }
                        },
                    }
                },
            },
        )
        opportunities = generate_surface_opportunities(surface, active_mechanisms=["surface_tool_loop"])
        search_plan = SearchPlan(
            plan_id="plan_tool_loop",
            diagnosis="Mutating calls need stronger identifier grounding.",
            hypotheses=["Validate mutations only against inspected identifiers."],
            target_mechanisms=["surface_tool_loop"],
            briefs=[
                SearchBrief(
                    brief_id="brief_tool_loop",
                    mechanism_class="surface_tool_loop",
                    hypothesis="Validate mutations only against inspected identifiers.",
                    surface_opportunity_ids=["surface.surface_tool_loop.inspect_before_mutate_record_id"],
                    candidate_roles=["composed"],
                )
            ],
        )

        proposals = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=opportunities,
            search_plan=search_plan,
            proposal_budget=1,
        )

        self.assertEqual(proposals, [])

    def test_structural_affordance_skips_identifier_already_present_in_parent(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Inspect records before mutation.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_record",
                            "description": "Inspect one record.",
                            "parameters": {
                                "type": "object",
                                "properties": {"record_id": {"type": "string"}},
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_record",
                            "description": "Update a record.",
                            "parameters": {
                                "type": "object",
                                "properties": {"record_id": {"type": "string"}},
                            },
                        },
                    },
                ],
                "tool_result_schemas": {
                    "get_record": {
                        "type": "object",
                        "properties": {
                            "record": {
                                "type": "object",
                                "properties": {"record_id": {"type": "string"}},
                            }
                        },
                    }
                },
            },
        )
        opportunities = generate_surface_opportunities(surface, active_mechanisms=["surface_tool_loop"])
        search_plan = SearchPlan(
            plan_id="plan_tool_loop",
            diagnosis="Mutating calls need inspected identifier grounding.",
            hypotheses=["Validate mutations only against inspected identifiers."],
            target_mechanisms=["surface_tool_loop"],
            briefs=[
                SearchBrief(
                    brief_id="brief_tool_loop",
                    mechanism_class="surface_tool_loop",
                    hypothesis="Validate mutations only against inspected identifiers.",
                    surface_opportunity_ids=["surface.surface_tool_loop.inspect_before_mutate_record_id"],
                    candidate_roles=["composed"],
                )
            ],
        )
        first_round = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=opportunities,
            search_plan=search_plan,
            proposal_budget=1,
        )
        parent = TransformCompiler().compile_or_raise(first_round[0].program, surface)

        second_round = _surface_affordance_proposals(
            surface=surface,
            surface_opportunities=opportunities,
            search_plan=search_plan,
            proposal_budget=1,
            parent_candidate=parent,
        )

        self.assertEqual(second_round, [])


if __name__ == "__main__":
    unittest.main()
