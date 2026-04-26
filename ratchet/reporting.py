from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from typing import Any

from ratchet.adapters import AdapterProtocol
from ratchet.io import write_json, write_jsonl
from ratchet.objectives import compare_summaries
from ratchet.results import PatchSummary, OptimizerStats, RatchetResult
from ratchet.types import AgentPatch, OptimizationObjective


def build_outcome_analysis(
    *,
    objective: OptimizationObjective,
    promoted: bool,
    baseline_dev: PatchSummary,
    accepted_dev_patches: list[PatchSummary],
    holdout_patches: list[PatchSummary],
    decision_log: list[dict[str, Any]],
) -> dict[str, Any]:
    proposal_iterations = [event for event in decision_log if event.get("type") == "proposal_iteration"]
    proposal_evaluations = [event for event in decision_log if event.get("type") == "proposal_evaluation"]
    holdout_validations = [event for event in decision_log if event.get("type") == "holdout_validation"]
    dev_rejection_reasons: Counter[str] = Counter(
        str(event.get("rejection_reason"))
        for event in proposal_evaluations
        if event.get("rejection_reason")
    )
    holdout_rejection_reasons: Counter[str] = Counter(
        str(event.get("rejection_reason"))
        for event in holdout_validations
        if event.get("rejection_reason")
    )
    rejection_reasons: Counter[str] = Counter(
        str(event.get("rejection_reason"))
        for event in [*proposal_evaluations, *holdout_validations]
        if event.get("rejection_reason")
    )
    latest_iteration = proposal_iterations[-1] if proposal_iterations else {}
    latest_stats = dict(latest_iteration.get("proposal_stats") or {})
    diagnosis_analysis = str(latest_iteration.get("diagnosis_analysis", ""))
    proposal_analysis = str(latest_iteration.get("proposal_analysis", ""))

    status = "promoted"
    summary = "Promoted an optimized patch after holdout validation."
    if not promoted:
        if (
            objective.mode == "correctness"
            and baseline_dev.pass_count == baseline_dev.case_count
            and not proposal_evaluations
        ):
            status = "no_failures"
            summary = "Baseline had no dev failures under the correctness objective."
        elif latest_stats.get("raw_count", 0) > 0 and latest_stats.get("valid_count", 0) == 0:
            status = "proposals_invalid"
            summary = "The optimizing model returned patches, but none satisfied the generated surface schema."
        elif proposal_iterations and not proposal_evaluations:
            status = "no_valid_model_proposals"
            if "failed" in proposal_analysis.lower():
                summary = "The optimizing model proposal call failed and Ratchet did not use a fallback."
            elif "No failing cases" in diagnosis_analysis and objective.mode == "correctness":
                status = "no_failures"
                summary = "Baseline had no dev failures under the correctness objective."
            else:
                summary = "The optimizing model produced no valid patches."
        elif proposal_evaluations and not accepted_dev_patches:
            if any("tradeoff" in reason or "constraint rejected" in reason for reason in dev_rejection_reasons):
                status = "objective_tradeoff_rejected"
                summary = "Patch proposals were evaluated but rejected by objective constraints or tradeoff guards."
            else:
                status = "proposals_evaluated_no_dev_gain"
                summary = "Patch proposals ran on dev but did not improve the configured objective."
        elif accepted_dev_patches and not holdout_patches:
            status = "holdout_not_run_budget_exhausted"
            summary = "At least one dev patch improved, but holdout validation budget was zero."
        elif holdout_validations and not any(event.get("passed_final_gate") for event in holdout_validations):
            if any("uncertainty rejected" in reason for reason in holdout_rejection_reasons):
                status = "holdout_gain_uncertain"
                summary = "A dev finalist reached holdout, but its measured gain was not statistically supported."
            elif any("tradeoff" in reason or "constraint rejected" in reason for reason in holdout_rejection_reasons):
                status = "objective_tradeoff_rejected"
                summary = "A dev finalist reached holdout but was rejected by objective constraints or tradeoff guards."
            else:
                status = "dev_gain_failed_holdout"
                summary = "A dev finalist reached holdout but did not beat the immutable baseline there."
        else:
            status = "baseline_kept_no_finalist"
            summary = "No finalist was available for promotion; kept the immutable baseline."

    return {
        "status": status,
        "summary": summary,
        "proposal_iterations": len(proposal_iterations),
        "proposal_evaluations": len(proposal_evaluations),
        "accepted_dev_patches": len(accepted_dev_patches),
        "holdout_validations": len(holdout_validations),
        "latest_diagnosis_analysis": diagnosis_analysis,
        "latest_proposal_analysis": proposal_analysis,
        "latest_proposal_stats": latest_stats,
        "rejection_reasons": dict(sorted(rejection_reasons.items(), key=lambda item: (-item[1], item[0]))),
    }


class RatchetReporter:
    def __init__(
        self,
        *,
        adapter: AdapterProtocol,
        out_dir: Path,
        objective: OptimizationObjective,
        stats: OptimizerStats,
    ) -> None:
        self.adapter = adapter
        self.out_dir = out_dir
        self.objective = objective
        self.stats = stats

    def write_outputs(self, result: RatchetResult) -> None:
        selected_comparison = compare_summaries(result.baseline_holdout, result.selected_holdout)
        write_json(self.out_dir / "run_manifest.json", result.manifest)
        write_json(self.out_dir / "decision_log.json", result.decision_log)
        write_json(self.out_dir / "outcome_analysis.json", result.outcome_analysis)
        write_jsonl(self.out_dir / "diagnoses.jsonl", result.diagnoses)
        write_jsonl(self.out_dir / "proposals.jsonl", result.proposals)
        write_json(
            self.out_dir / "patch_metrics.json",
            {
                "baseline_dev": result.baseline_dev.to_dict(),
                "baseline_holdout": result.baseline_holdout.to_dict(),
                "best_dev_patch": result.best_dev_patch.to_dict(),
                "selected_holdout": result.selected_holdout.to_dict(),
                "accepted_dev_patches": [summary.to_dict() for summary in result.accepted_dev_patches],
                "holdout_patches": [summary.to_dict() for summary in result.holdout_patches],
                "pareto_frontier": result.pareto_frontier,
                "generated_surface": result.generated_surface,
            },
        )
        write_json(
            self.out_dir / "selected_patch.json",
            {
                "promoted": result.promoted,
                "selected_patch_hash": result.selected_patch_hash,
                "patch": result.selected_patch.to_dict(),
                "objective": self.objective.to_dict(),
                "selection_reason": result.selection_reason,
                "outcome_analysis": result.outcome_analysis,
                "holdout_comparison_to_baseline": selected_comparison.to_dict(),
                "baseline": result.baseline_holdout.to_dict(),
                "selected": result.selected_holdout.to_dict(),
            },
        )
        export_dir = self.out_dir / "exported_patch"
        self.adapter.export(result.selected_patch, export_dir)
        self._write_report(result)
        self._write_summary_html(result)
        self._write_plots(result)

    def _write_report(self, result: RatchetResult) -> None:
        changes = self._patch_change_rows(result.selected_patch)
        comparison = compare_summaries(result.baseline_holdout, result.selected_holdout)
        lines = [
            "# Ratchet Report",
            "",
            f"Outcome: {'promoted optimized patch' if result.promoted else 'kept original baseline'}",
            f"Objective: `{self.objective.mode}`",
            f"Selected patch: `{result.selected_patch_hash}`",
            f"Outcome status: `{result.outcome_analysis['status']}`",
            f"Outcome summary: {result.outcome_analysis['summary']}",
            "",
            "## Baseline vs Selected Holdout",
            "",
            "| Metric | Baseline | Selected |",
            "| --- | ---: | ---: |",
            f"| Mean score | {result.baseline_holdout.mean_score:.3f} | {result.selected_holdout.mean_score:.3f} |",
            f"| Pass count | {result.baseline_holdout.pass_count} | {result.selected_holdout.pass_count} |",
            f"| Avg cost | ${result.baseline_holdout.mean_cost_usd:.6f} | ${result.selected_holdout.mean_cost_usd:.6f} |",
            f"| Median latency | {result.baseline_holdout.median_latency_s:.2f}s | {result.selected_holdout.median_latency_s:.2f}s |",
            f"| Samples | {result.baseline_holdout.sample_count} over {result.baseline_holdout.case_count} cases | {result.selected_holdout.sample_count} over {result.selected_holdout.case_count} cases |",
            f"| Split-vote cases | {len(result.baseline_holdout.split_vote_case_ids)} | {len(result.selected_holdout.split_vote_case_ids)} |",
            "",
            "## Holdout Uncertainty",
            "",
            f"- Score delta: {comparison.score_delta:.4f} CI [{comparison.score_ci[0]:.4f}, {comparison.score_ci[1]:.4f}]",
            f"- Cost delta: ${comparison.cost_delta:.6f} CI [${comparison.cost_ci[0]:.6f}, ${comparison.cost_ci[1]:.6f}]",
            f"- Token delta: {comparison.token_delta:.1f} CI [{comparison.token_ci[0]:.1f}, {comparison.token_ci[1]:.1f}]",
            f"- Latency delta: {comparison.latency_delta:.3f}s CI [{comparison.latency_ci[0]:.3f}s, {comparison.latency_ci[1]:.3f}s]",
            "",
            "## Selected Patch",
            "",
            *(changes or ["No patch operations; original baseline selected."]),
            "",
            "## Generated Surface",
            "",
            *[
                f"- `{target['name']}` ({target['kind']}): {', '.join(target['allowed_ops'])}; schema={json.dumps(target.get('value_schema', {}), sort_keys=True)}"
                for target in result.generated_surface
            ],
            "",
            "## Optimization Trace",
            "",
            f"- Proposal iterations: {result.outcome_analysis['proposal_iterations']}",
            f"- Proposal evaluations: {result.outcome_analysis['proposal_evaluations']}",
            f"- Accepted dev patches: {result.outcome_analysis['accepted_dev_patches']}",
            f"- Holdout validations: {result.outcome_analysis['holdout_validations']}",
            f"- Latest diagnosis: {result.outcome_analysis['latest_diagnosis_analysis'] or 'n/a'}",
            f"- Latest proposal: {result.outcome_analysis['latest_proposal_analysis'] or 'n/a'}",
            f"- Rejection reasons: {json.dumps(result.outcome_analysis['rejection_reasons'], sort_keys=True)}",
            "",
            "## Run Health",
            "",
            f"- Cache hits: {self.stats.cache_hits}",
            f"- Fresh case evaluations: {self.stats.fresh_case_evaluations}",
            f"- Runtime errors: {self.stats.runtime_errors}",
            f"- Grader errors: {self.stats.grader_errors}",
            f"- Timeouts: {self.stats.timeouts}",
            f"- Samples per case: {result.baseline_holdout.samples_per_case:g}",
        ]
        (self.out_dir / "report.md").write_text("\n".join(lines) + "\n")

    def _write_summary_html(self, result: RatchetResult) -> None:
        rows = self._patch_change_rows(result.selected_patch)
        html_rows = "".join(f"<li>{escape(row)}</li>" for row in rows) or "<li>Original baseline kept.</li>"
        html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Ratchet Summary</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .metric {{ display: inline-block; margin: 8px 18px 8px 0; }}
    .metric strong {{ display: block; font-size: 24px; }}
    code {{ background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }}
    img {{ max-width: 780px; display: block; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>{'Promoted patch' if result.promoted else 'Baseline kept'}</h1>
  <p>{escape(result.selection_reason)}</p>
  <p><strong>{escape(str(result.outcome_analysis['status']))}</strong>: {escape(str(result.outcome_analysis['summary']))}</p>
  <h2>Outcome</h2>
  <div class="metric"><span>Baseline score</span><strong>{result.baseline_holdout.mean_score:.3f}</strong></div>
  <div class="metric"><span>Selected score</span><strong>{result.selected_holdout.mean_score:.3f}</strong></div>
  <div class="metric"><span>Selected cost</span><strong>${result.selected_holdout.mean_cost_usd:.6f}</strong></div>
  <h2>What Changed</h2>
  <ul>{html_rows}</ul>
  <h2>Progress</h2>
  <img src="plots/scorecard.svg" alt="Scorecard">
  <img src="plots/progress.svg" alt="Progress">
</body>
</html>
"""
        (self.out_dir / "summary.html").write_text(html)

    def _write_plots(self, result: RatchetResult) -> None:
        plots = self.out_dir / "plots"
        plots.mkdir(parents=True, exist_ok=True)
        self._write_scorecard_svg(plots / "scorecard.svg", result)
        self._write_progress_svg(plots / "progress.svg", result)
        self._write_progress_svg(plots / "efficiency_progress.svg", result)

    def _write_scorecard_svg(self, path: Path, result: RatchetResult) -> None:
        metrics = [
            ("Score", result.baseline_holdout.mean_score, result.selected_holdout.mean_score),
            ("Cost", result.baseline_holdout.mean_cost_usd, result.selected_holdout.mean_cost_usd),
            ("Latency", result.baseline_holdout.median_latency_s, result.selected_holdout.median_latency_s),
        ]
        rows = []
        for index, (label, baseline, selected) in enumerate(metrics):
            y = 80 + index * 54
            max_value = max(baseline, selected, 1e-9)
            rows.append(f'<text x="32" y="{y}">{escape(label)}</text>')
            rows.append(f'<rect x="150" y="{y - 16}" width="{baseline / max_value * 220:.1f}" height="14" fill="#94a3b8"/>')
            rows.append(f'<rect x="150" y="{y + 2}" width="{selected / max_value * 220:.1f}" height="14" fill="#0f766e"/>')
            rows.append(f'<text x="390" y="{y}">{baseline:.3f} -> {selected:.3f}</text>')
        svg = "\n".join(
            [
                '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="260">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                '<text x="32" y="36" font-size="22" font-family="Arial">Baseline vs selected</text>',
                *rows,
                "</svg>",
            ]
        )
        path.write_text(svg)

    def _write_progress_svg(self, path: Path, result: RatchetResult) -> None:
        summaries = [result.baseline_dev, *result.accepted_dev_patches]
        points = []
        for index, summary in enumerate(summaries):
            x = 60 + index * 110
            y = 190 - summary.mean_score * 140
            points.append((x, y, summary.mean_score))
        circles = "\n".join(
            f'<circle cx="{x}" cy="{y}" r="6" fill="#0f766e"/><text x="{x - 18}" y="220" font-size="11">{score:.2f}</text>'
            for x, y, score in points
        )
        if len(points) > 1:
            polyline = " ".join(f"{x},{y}" for x, y, _ in points)
            line = f'<polyline points="{polyline}" fill="none" stroke="#0f766e" stroke-width="2"/>'
        else:
            line = ""
        path.write_text(
            "\n".join(
                [
                    '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="250">',
                    '<rect width="100%" height="100%" fill="#ffffff"/>',
                    '<text x="32" y="36" font-size="22" font-family="Arial">Dev score progress</text>',
                    '<line x1="50" y1="190" x2="680" y2="190" stroke="#cbd5e1"/>',
                    '<line x1="50" y1="50" x2="50" y2="190" stroke="#cbd5e1"/>',
                    line,
                    circles,
                    "</svg>",
                ]
            )
        )

    @staticmethod
    def _patch_change_rows(patch: AgentPatch) -> list[str]:
        return [
            f"`{operation.op}` on `{operation.target}`: {RatchetReporter._format_value(operation.value)}"
            for operation in patch.operations
        ]

    @staticmethod
    def _format_value(value: Any) -> str:
        text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        return text if len(text) <= 160 else text[:157] + "..."
