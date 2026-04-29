from __future__ import annotations

import unittest

from ratchet.evidence import build_behavior_diagnostics
from ratchet.interactive import InteractionRecorder
from ratchet.results import CaseEvaluation, PatchSummary
from ratchet.types import AgentPatch, DiagnosticTrace, EvalCase, GradeResult, InteractionTurn, OperationalMetrics, RunRecord, ToolCallTrace


class InteractiveTraceTests(unittest.TestCase):
    def test_diagnostic_trace_round_trips_structured_tool_turns(self) -> None:
        trace = DiagnosticTrace(
            raw_output_text="done",
            turns=[
                InteractionTurn(
                    index=0,
                    actor="agent",
                    message="I will look that up.",
                    tool_calls=[
                        ToolCallTrace(
                            name="lookup_order",
                            arguments={"order_id": "O-1"},
                            result={"status": "delivered"},
                        )
                    ],
                )
            ],
            terminal_state={"resolved": True},
            terminal_reason="success",
        )

        restored = DiagnosticTrace.from_dict(trace.to_dict())

        self.assertEqual(restored.tool_calls, ["lookup_order"])
        self.assertEqual(restored.turns[0].tool_calls[0].arguments, {"order_id": "O-1"})
        self.assertEqual(restored.terminal_state, {"resolved": True})
        self.assertEqual(restored.terminal_reason, "success")

    def test_run_record_derives_tool_and_turn_counts_from_trace(self) -> None:
        trace = DiagnosticTrace(
            turns=[
                InteractionTurn(index=0, actor="user", message="Need help."),
                InteractionTurn(
                    index=1,
                    actor="agent",
                    tool_calls=[ToolCallTrace(name="lookup_customer")],
                ),
            ]
        )

        record = RunRecord(
            output={"ok": True},
            metrics=OperationalMetrics(
                latency_s=1.0,
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
                cost_usd=0.001,
            ),
            diagnostics=trace,
        )

        self.assertEqual(record.metrics.tool_calls, 1)
        self.assertEqual(record.metrics.turns, 2)

    def test_recorder_builds_metrics_and_diagnostics(self) -> None:
        recorder = InteractionRecorder()
        turn = recorder.add_turn(actor="agent", message="Checking policy")
        recorder.add_tool_call(name="get_policy", arguments={"topic": "refund"}, turn_index=turn)

        metrics = recorder.metrics(
            latency_s=2.0,
            input_tokens=100,
            output_tokens=40,
            cost_usd=0.01,
            model_calls=2,
        )
        diagnostics = recorder.diagnostics(terminal_reason="success")

        self.assertEqual(metrics.model_calls, 2)
        self.assertEqual(metrics.tool_calls, 1)
        self.assertEqual(metrics.turns, 1)
        self.assertEqual(diagnostics.tool_calls, ["get_policy"])

    def test_behavior_diagnostics_summarizes_tool_failures(self) -> None:
        evaluation = CaseEvaluation(
            case=EvalCase(id="case-1", split="dev", input="x", expected="y"),
            record=RunRecord(
                output="wrong",
                metrics=OperationalMetrics(
                    latency_s=1.0,
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    cost_usd=0.001,
                ),
                diagnostics=DiagnosticTrace(
                    turns=[
                        InteractionTurn(
                            index=0,
                            actor="agent",
                            outcome="bad_tool_arguments",
                            tool_calls=[
                                ToolCallTrace(
                                    name="refund_order",
                                    arguments={"order_id": None},
                                    status="invalid",
                                )
                            ],
                        )
                    ],
                    terminal_reason="premature_stop",
                ),
            ),
            grade=GradeResult(score=0.0, passed=False, labels=["bad_tool_arguments"]),
        )
        summary = PatchSummary(
            patch_hash="baseline",
            patch=AgentPatch.empty(),
            split="dev",
            evaluations=[evaluation],
        )

        diagnostics = build_behavior_diagnostics(summary)

        self.assertEqual(diagnostics["tool_interaction"]["tool_call_counts"], {"refund_order": 1})
        self.assertEqual(diagnostics["tool_interaction"]["tool_status_counts"], {"invalid": 1})
        self.assertEqual(diagnostics["tool_interaction"]["turn_outcome_counts"], {"bad_tool_arguments": 1})
        self.assertEqual(diagnostics["tool_interaction"]["premature_stop_case_ids"], ["case-1"])


if __name__ == "__main__":
    unittest.main()
