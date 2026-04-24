from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from examples.northstar.adapter import NorthstarAdapter
from ratchet.io import load_eval_cases
from ratchet.optimizer import RatchetOptimizer
from ratchet.types import DiagnosticTrace, EvalCase, OperationalMetrics, RunRecord


class ScriptedNorthstarRunner:
    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        category = str(case.metadata.get("category"))
        kb_enabled = candidate["kb_tool_enabled"] == "on"
        calc_enabled = candidate["calculator_tool_enabled"] == "on"
        model = candidate["model"]
        knowledge_mode = candidate["knowledge_mode"]
        solved = False
        tool_calls: list[str] = []

        if kb_enabled:
            tool_calls.append("kb_lookup")
            solved = category != "math"
        if category == "math" and calc_enabled:
            tool_calls.append("calculator")
            solved = True

        answer = str(case.expected) if solved else "unknown"
        base_tokens = 170 if not kb_enabled else 110
        if knowledge_mode == "distilled":
            base_tokens -= 20
        if model == "gpt-5.4-nano":
            base_tokens -= 25
        elif model == "gpt-5.4":
            base_tokens += 40
        cost_usd = 0.004 if not kb_enabled else 0.0022
        if knowledge_mode == "distilled":
            cost_usd -= 0.0003
        if model == "gpt-5.4-nano":
            cost_usd -= 0.0008
        elif model == "gpt-5.4":
            cost_usd += 0.0015
        latency_s = 1.0 if model != "gpt-5.4" else 1.2
        if model == "gpt-5.4-nano":
            latency_s = 1.08
        return RunRecord(
            output=answer,
            metrics=OperationalMetrics(
                latency_s=latency_s,
                input_tokens=base_tokens // 2,
                output_tokens=base_tokens // 2,
                total_tokens=base_tokens,
                cost_usd=cost_usd,
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=tool_calls,
                raw_output_text=answer,
            ),
        )


class NorthstarAdapterIntegrationTests(unittest.TestCase):
    def test_search_space_exposes_tool_kb_and_model_knobs(self) -> None:
        adapter = NorthstarAdapter(runner=ScriptedNorthstarRunner())
        search_space = adapter.search_space()
        knob_names = {spec.name for spec in search_space.all_specs()}
        self.assertTrue({"kb_tool_enabled", "knowledge_mode", "model"}.issubset(knob_names))

    def test_northstar_adapter_can_clear_final_gate_on_fixture_subset(self) -> None:
        adapter = NorthstarAdapter(runner=ScriptedNorthstarRunner())
        evals_path = Path(__file__).resolve().parents[1] / "examples" / "northstar" / "evals.jsonl"
        all_cases = load_eval_cases(evals_path)
        selected_ids = {"dev-01", "dev-13", "holdout-01", "holdout-11"}
        cases = tuple(
            case
            for case in all_cases
            if case.id in {"dev-01", "dev-13", "test-01", "test-11"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            optimizer = RatchetOptimizer(
                adapter=adapter,
                search_space=adapter.search_space(),
                out_dir=Path(tmp) / "run",
                env_path=".env",
                dev_budget=12,
                holdout_top_k=4,
                harnesser_enabled=False,
            )
            result = optimizer.run(cases)
            self.assertTrue(result.promoted)
            self.assertGreaterEqual(
                result.selected_holdout.mean_score,
                result.baseline_holdout.mean_score,
            )
            self.assertLess(
                result.selected_holdout.mean_cost_usd,
                result.baseline_holdout.mean_cost_usd,
            )
            self.assertLess(
                result.selected_holdout.mean_total_tokens,
                result.baseline_holdout.mean_total_tokens,
            )


if __name__ == "__main__":
    unittest.main()
