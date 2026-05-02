from __future__ import annotations

import unittest

from ratchet.surface_opportunities import generate_surface_opportunities
from ratchet.surfaces import surface_from_agent_spec, tool_loop_surface_from_agent_spec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_contract import build_transform_contract, contract_example_programs
from ratchet.types import AgentSpec


class TransformContractTests(unittest.TestCase):
    def test_contract_uses_only_surface_allowed_hook_ops(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                model_options=["base", "larger"],
                instructions={"system_prompt": "Answer."},
            )
        )
        opportunities = generate_surface_opportunities(surface)
        contract = build_transform_contract(surface, opportunities)

        self.assertIn("before_model_call", contract.hook_ops)
        self.assertIn("replace_context_section", contract.hook_ops["before_model_call"])
        self.assertNotIn("replace_context_section", contract.hook_ops["on_task_start"])
        self.assertEqual(contract.op_hooks["replace_context_section"], ["before_model_call"])

    def test_contract_examples_compile(self) -> None:
        surface = surface_from_agent_spec(
            AgentSpec(
                name="sample",
                model="base",
                model_options=["base", "larger"],
                instructions={"system_prompt": "Answer."},
            )
        )
        opportunities = generate_surface_opportunities(surface)
        contract = build_transform_contract(surface, opportunities)

        for program in contract_example_programs(contract):
            compiled = TransformCompiler().compile(program, surface)
            self.assertEqual(compiled.report.status, "compiled", program.to_dict())

    def test_tool_loop_contract_includes_validation_checks_and_excludes_unsupported_ops(self) -> None:
        surface = tool_loop_surface_from_agent_spec(
            AgentSpec(name="interactive", model="base"),
            probe={
                "domain_policy": "Use tools safely.",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_record",
                            "description": "Lookup a record.",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )
        opportunities = generate_surface_opportunities(surface)
        contract = build_transform_contract(surface, opportunities)

        self.assertIn("before_tool_call", contract.hook_ops)
        self.assertIn("validate", contract.hook_ops["before_tool_call"])
        self.assertNotIn("rewrite_tool_result", contract.allowed_ops)
        check_types = {check["type"] for check in contract.validation_checks["before_tool_call"]}
        self.assertIn("args_schema_valid", check_types)


if __name__ == "__main__":
    unittest.main()
