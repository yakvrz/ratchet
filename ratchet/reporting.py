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
    finalist_statuses: list[dict[str, Any]] | None = None,
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
    finalist_status_rows = list(finalist_statuses or [])
    if not finalist_status_rows:
        finalist_status_rows = [
            {
                "status": event.get("finalist_status"),
                "reason": event.get("rejection_reason"),
                "patch_hash": event.get("patch_hash"),
            }
            for event in holdout_validations
            if event.get("finalist_status")
        ]
    finalist_status_counts: Counter[str] = Counter(
        str(row.get("status")) for row in finalist_status_rows if row.get("status")
    )

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
        elif (
            latest_stats.get("raw_count", 0) > 0
            and latest_stats.get("valid_count", 0) == 0
            and not accepted_dev_patches
            and not holdout_patches
        ):
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
            if finalist_status_counts.get("failed", 0) > 0:
                status = "finalists_failed_confirmation"
                summary = "At least one dev patch improved, but finalist confirmation rejected all candidates before holdout."
            else:
                status = "holdout_not_run_budget_exhausted"
                summary = "At least one dev patch improved, but holdout validation budget was zero."
        elif holdout_validations and not any(event.get("passed_final_gate") for event in holdout_validations):
            if finalist_status_counts.get("directional", 0) > 0:
                status = "directional_holdout_gain"
                summary = "A dev finalist improved holdout directionally, but the uncertainty gate did not validate it."
            elif any("uncertainty rejected" in reason for reason in holdout_rejection_reasons):
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
        "finalist_status_counts": dict(sorted(finalist_status_counts.items())),
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
        write_jsonl(self.out_dir / "task_theories.jsonl", result.task_theories)
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
                "task_theories": result.task_theories,
                "frontier_status_summaries": _frontier_status_summaries(result.proposals),
                "proposal_example_bank": result.manifest.get("proposal_example_bank", {}),
                "transform_summaries": result.transform_summaries,
                "transform_context_summaries": result.transform_context_summaries,
                "finalist_statuses": result.finalist_statuses,
                "runtime_reliability_diagnostics": result.runtime_reliability_diagnostics,
                "confirmation_results": result.confirmation_results,
                "simplification_results": result.simplification_results,
                "frontier_recommendation": result.frontier_recommendation,
                "run_profile": result.run_profile,
                "run_cost": (result.run_profile or {}).get("run_cost", {}),
                "quality_cost_tradeoffs": result.quality_cost_tradeoffs,
                "optimizer_call_diagnostics": result.optimizer_call_diagnostics,
            },
        )
        write_json(
            self.out_dir / "selected_patch.json",
            {
                "promoted": result.promoted,
                "selected_patch_hash": result.selected_patch_hash,
                "selected_finalist_status": _selected_finalist_status(
                    result.finalist_statuses,
                    result.selected_patch_hash,
                ),
                "patch": result.selected_patch.to_dict(),
                "objective": self.objective.to_dict(),
                "selection_reason": result.selection_reason,
                "outcome_analysis": result.outcome_analysis,
                "task_theories": result.task_theories,
                "frontier_status_summaries": _frontier_status_summaries(result.proposals),
                "transform_summaries": result.transform_summaries,
                "transform_context_summaries": result.transform_context_summaries,
                "finalist_statuses": result.finalist_statuses,
                "runtime_reliability_diagnostics": result.runtime_reliability_diagnostics,
                "confirmation_results": result.confirmation_results,
                "simplification_results": result.simplification_results,
                "frontier_recommendation": result.frontier_recommendation,
                "run_profile": result.run_profile,
                "quality_cost_tradeoffs": result.quality_cost_tradeoffs,
                "optimizer_call_diagnostics": result.optimizer_call_diagnostics,
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
        transform_narrative = self._transform_narrative(result)
        lines = [
            "# Ratchet Report",
            "",
            f"Outcome: {'promoted optimized patch' if result.promoted else 'kept original baseline'}",
            f"Objective: `{self.objective.mode}`",
            f"Selected patch: `{result.selected_patch_hash}`",
            f"Outcome status: `{result.outcome_analysis['status']}`",
            f"Outcome summary: {result.outcome_analysis['summary']}",
            f"Recommendation: {result.frontier_recommendation.get('reason', result.selection_reason)}",
            f"Recommendation policy: `{result.frontier_recommendation.get('recommendation_policy', 'n/a')}`",
            "",
            "## Task Theory",
            "",
            *self._task_theory_rows(result),
            "",
            "## Frontier Categories",
            "",
            *self._frontier_status_rows(result),
            "",
            "## Holdout Frontier",
            "",
            *self._frontier_variant_rows(result),
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
            f"- Finalist status counts: {json.dumps(result.outcome_analysis.get('finalist_status_counts', {}), sort_keys=True)}",
            "",
            "## Finalist Statuses",
            "",
            *self._finalist_status_rows(result),
            "",
            "## Simplification",
            "",
            *self._simplification_rows(result),
            "",
            "## Few-Shot Compression",
            "",
            *self._few_shot_compression_rows(result),
            "",
            "## Runtime Reliability",
            "",
            *self._runtime_reliability_rows(result),
            "",
            "## Quality/Cost Tradeoffs",
            "",
            *self._quality_cost_tradeoff_rows(result),
            "",
            "## Run Profile",
            "",
            *self._run_profile_rows(result),
            "",
            "## Run Cost",
            "",
            *self._run_cost_rows(result),
            "",
            "## Optimizer Overhead",
            "",
            *self._optimizer_overhead_rows(result),
            "",
            "## Search Narrative",
            "",
            transform_narrative,
            "",
            "## Transform Families",
            "",
            *[
                "- "
                f"`{name}`: state={summary.get('state')}, "
                f"proposed={summary.get('proposed_count')}, "
                f"evaluated={summary.get('evaluated_count')}, "
                f"accepted={summary.get('accepted_count')}, "
                f"final={json.dumps((result.manifest.get('transform_final_statuses') or {}).get(name, {}), sort_keys=True)}; "
                f"{summary.get('reason')}"
                for name, summary in sorted(result.transform_summaries.items())
            ],
            "",
            "## Transform Contexts",
            "",
            *[
                "- "
                f"`{summary.get('key', {}).get('family')}` "
                f"targets={','.join(summary.get('key', {}).get('target_names', [])) or 'global'} "
                f"instance={summary.get('key', {}).get('transform_instance')} "
                f"slice={summary.get('key', {}).get('target_slice')}: "
                f"state={summary.get('state')}, evaluated={summary.get('evaluated_count')}, "
                f"accepted={summary.get('accepted_count')}; {summary.get('reason')}"
                for _, summary in sorted(result.transform_context_summaries.items())
            ],
            "",
            "## Run Health",
            "",
            f"- Cache hits: {self.stats.cache_hits}",
            f"- Fresh case evaluations: {self.stats.fresh_case_evaluations}",
            f"- Runtime errors: {self.stats.runtime_errors}",
            f"- Grader errors: {self.stats.grader_errors}",
            f"- Timeouts: {self.stats.timeouts}",
            f"- Samples per case: {result.baseline_holdout.samples_per_case:g}",
            f"- Proposal-safe train examples: {(result.manifest.get('proposal_example_bank') or {}).get('example_count', 0)}",
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
    def _finalist_status_rows(result: RatchetResult) -> list[str]:
        if not result.finalist_statuses:
            return ["No finalist status rows were recorded."]
        rows = []
        for item in result.finalist_statuses:
            status = str(item.get("status", "unknown"))
            patch_hash_value = str(item.get("patch_hash", "unknown"))
            reason = str(item.get("reason") or "passed")
            stage = str(item.get("stage") or "unknown")
            holdout_metrics = item.get("holdout_metrics") or {}
            pass_count = holdout_metrics.get("pass_count")
            case_count = holdout_metrics.get("case_count")
            metric_text = f"; holdout={pass_count}/{case_count}" if pass_count is not None and case_count is not None else ""
            rows.append(f"- `{patch_hash_value}`: `{status}` at `{stage}`{metric_text}; {reason}")
        return rows

    @staticmethod
    def _task_theory_rows(result: RatchetResult) -> list[str]:
        if not result.task_theories:
            return ["No task theory snapshots were recorded."]
        rows = []
        for item in result.task_theories[-5:]:
            theory = item.get("task_theory") or {}
            opportunities = [
                str(row.get("mechanism_class"))
                for row in theory.get("experiment_opportunities", [])[:3]
                if isinstance(row, dict) and row.get("mechanism_class")
            ]
            rows.append(
                "- "
                f"iteration={item.get('iteration')} parent=`{item.get('parent_patch_hash')}` "
                f"bottleneck=`{theory.get('bottleneck_class')}` "
                f"residual={json.dumps(theory.get('residual_failure_modes', []))} "
                f"weak={json.dumps(theory.get('weak_slices', [])[:6])} "
                f"opportunities={json.dumps(opportunities)} "
                f"confidence={theory.get('confidence')}"
            )
        return rows

    @staticmethod
    def _frontier_status_rows(result: RatchetResult) -> list[str]:
        summaries = _frontier_status_summaries(result.proposals)
        if not summaries:
            return ["No evaluated candidate frontier categories were recorded."]
        rows = [
            "| Status | Count | Best score delta | Best cost delta | Examples |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for status, item in sorted(summaries.items()):
            rows.append(
                "| "
                f"`{status}` | "
                f"{item.get('count', 0)} | "
                f"{float(item.get('best_score_delta') or 0.0):+.3f} | "
                f"${float(item.get('best_cost_delta') or 0.0):+.6f} | "
                f"{', '.join(str(value) for value in item.get('patch_hashes', [])[:4])} |"
            )
        return rows

    @staticmethod
    def _frontier_variant_rows(result: RatchetResult) -> list[str]:
        variants = result.frontier_recommendation.get("frontier_variants") or []
        if not variants:
            if result.holdout_patches:
                statuses = {
                    str(item.get("patch_hash")): item
                    for item in result.finalist_statuses
                    if item.get("patch_hash")
                }
                variants = [
                    {
                        "role": "holdout_candidate"
                        + (
                            f" ({statuses.get(item.patch_hash, {}).get('status')})"
                            if statuses.get(item.patch_hash, {}).get("status")
                            else ""
                        ),
                        "patch_hash": item.patch_hash,
                        "pass_count": item.pass_count,
                        "case_count": item.case_count,
                        "mean_score": item.mean_score,
                        "mean_cost_usd": item.mean_cost_usd,
                        "mean_total_tokens": item.mean_total_tokens,
                        "median_latency_s": item.median_latency_s,
                        "operation_count": item.operation_count,
                        "operations": [
                            {
                                "op": operation.op,
                                "target": operation.target,
                                "value_summary": RatchetReporter._format_value(operation.value),
                            }
                            for operation in item.patch.operations
                        ],
                    }
                    for item in result.holdout_patches
                ]
            else:
                return ["No holdout finalist candidates were available."]
        rows = [
            "| Role | Patch | Holdout | Score | Cost | Tokens | Latency | Ops |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        selected_hash = result.selected_patch_hash
        for item in variants:
            role = str(item.get("role", "candidate"))
            if item.get("patch_hash") == selected_hash:
                role = f"{role} (selected)"
            rows.append(
                "| "
                f"{role} | "
                f"`{item.get('patch_hash')}` | "
                f"{int(item.get('pass_count') or 0)}/{int(item.get('case_count') or 0)} | "
                f"{float(item.get('mean_score') or 0.0):.3f} | "
                f"${float(item.get('mean_cost_usd') or 0.0):.6f} | "
                f"{float(item.get('mean_total_tokens') or 0.0):.1f} | "
                f"{float(item.get('median_latency_s') or 0.0):.2f}s | "
                f"{RatchetReporter._format_frontier_ops(item.get('operations') or [])} |"
            )
        return rows

    @staticmethod
    def _format_frontier_ops(operations: list[dict[str, Any]]) -> str:
        if not operations:
            return "baseline"
        rows = []
        for operation in operations:
            op = operation.get("op")
            target = operation.get("target")
            value = operation.get("value_summary")
            if op == "add_few_shot" and isinstance(value, list):
                rows.append(f"{op}:{target}({len(value)} examples)")
            else:
                rows.append(f"{op}:{target}")
        return ", ".join(rows)

    @staticmethod
    def _runtime_reliability_rows(result: RatchetResult) -> list[str]:
        findings = [
            item
            for item in result.runtime_reliability_diagnostics
            if item.get("runtime_finding") or item.get("suspicious")
        ]
        if not findings:
            return ["No runtime reliability findings were flagged."]
        rows = []
        for item in findings:
            rows.append(
                "- "
                f"`{item.get('patch_hash')}` ({item.get('diagnostic_class', 'runtime_finding')}): {item.get('reason')} "
                f"fixed_invalid={item.get('fixed_invalid_output_case_ids', [])}, "
                f"low_token_fixed={item.get('low_token_fixed_case_ids', [])}, "
                f"finish_reasons={json.dumps(item.get('finish_reasons_by_case', {}), sort_keys=True)}"
            )
        if result.confirmation_results:
            for item in result.confirmation_results:
                rows.append(
                    "- "
                    f"confirmation `{item.get('patch_hash')}`: "
                    f"passed={item.get('passed')}; {item.get('reason')}"
                )
        return rows

    @staticmethod
    def _few_shot_compression_rows(result: RatchetResult) -> list[str]:
        rows = [
            row
            for row in result.proposals
            if row.get("transform_family") == "targeted_few_shot" and row.get("metrics")
        ]
        if not rows:
            return ["No evaluated few-shot variants were recorded."]
        table = [
            "| Patch | Status | Examples | Strategy | Score Delta | Token Delta | Tokens / Example | Source IDs |",
            "| --- | --- | ---: | --- | ---: | ---: | ---: | --- |",
        ]
        for row in rows[:12]:
            parameters = row.get("transform_parameters") or {}
            materialization = row.get("materialization") or {}
            example_count = int(parameters.get("few_shot_example_count") or len(materialization.get("source_case_ids") or []) or 0)
            comparison = row.get("comparison_to_parent") or {}
            token_delta = float(comparison.get("token_delta") or 0.0)
            score_delta = float(comparison.get("score_delta") or 0.0)
            token_per_example = token_delta / example_count if example_count > 0 else 0.0
            source_ids = materialization.get("source_case_ids") or parameters.get("source_case_ids") or []
            status = "accepted" if row.get("accepted") else "rejected"
            table.append(
                "| "
                f"`{row.get('patch_hash')}` | "
                f"{status} | "
                f"{example_count} | "
                f"{parameters.get('selection_strategy', 'unspecified')} | "
                f"{score_delta:+.3f} | "
                f"{token_delta:+.1f} | "
                f"{token_per_example:+.1f} | "
                f"{', '.join(str(item) for item in source_ids[:4])} |"
            )
        if len(rows) > 12:
            table.append(f"- {len(rows) - 12} additional few-shot variants omitted from this table.")
        return table

    @staticmethod
    def _simplification_rows(result: RatchetResult) -> list[str]:
        if not result.simplification_results:
            return ["No finalist simplification variants were evaluated."]
        accepted = [row for row in result.simplification_results if row.get("accepted")]
        rejected = [row for row in result.simplification_results if not row.get("accepted")]
        rows = [
            f"- Evaluated {len(result.simplification_results)} simplification variants: {len(accepted)} accepted, {len(rejected)} rejected."
        ]
        for item in result.simplification_results[:8]:
            simplification = item.get("simplification") or {}
            rows.append(
                "- "
                f"`{item.get('patch_hash')}` from `{item.get('parent_patch_hash')}`: "
                f"{simplification.get('type', 'simplification')} accepted={item.get('accepted')}; "
                f"{item.get('rejection_reason') or 'passed dev gate'}"
            )
        return rows

    @staticmethod
    def _quality_cost_tradeoff_rows(result: RatchetResult) -> list[str]:
        if not result.quality_cost_tradeoffs:
            return ["No model substitutions were rejected by the cost guard."]
        return [
            "- "
            f"`{item.get('patch_hash')}`: {item.get('rejection_reason')} "
            f"metrics={json.dumps(item.get('metrics', {}), sort_keys=True)}"
            for item in result.quality_cost_tradeoffs
        ]

    @staticmethod
    def _run_profile_rows(result: RatchetResult) -> list[str]:
        profile = result.run_profile or {}
        rows = [
            f"- Elapsed: {float(profile.get('elapsed_s') or 0.0):.3f}s",
            f"- Phase wall durations: {json.dumps(profile.get('phase_durations_s', {}), sort_keys=True)}",
            f"- Phase attempt durations: {json.dumps(profile.get('phase_attempt_durations_s', {}), sort_keys=True)}",
            f"- Cache events: {json.dumps(profile.get('cache_events', {}), sort_keys=True)}",
            f"- Cache hit rate: {float(profile.get('cache_hit_rate') or 0.0):.3f}",
        ]
        slowest = profile.get("slowest_cases") or []
        if slowest:
            rows.append("- Slowest cases: " + RatchetReporter._compact_profile_rows(slowest, "latency_s"))
        token_heavy = profile.get("highest_token_cases") or []
        if token_heavy:
            rows.append("- Highest token cases: " + RatchetReporter._compact_profile_rows(token_heavy, "total_tokens"))
        patch_profiles = profile.get("patch_profiles") or []
        if patch_profiles:
            rows.append("- Patch profiles: " + RatchetReporter._compact_patch_profile_rows(patch_profiles))
        patch_deltas = profile.get("patch_deltas_vs_baseline") or []
        if patch_deltas:
            rows.append("- Patch deltas vs baseline: " + RatchetReporter._compact_patch_delta_rows(patch_deltas))
        return rows

    @staticmethod
    def _run_cost_rows(result: RatchetResult) -> list[str]:
        cost = ((result.run_profile or {}).get("run_cost") or {})
        if not cost:
            return ["No run cost profile was recorded."]
        return [
            "- "
            f"Total cost=${float(cost.get('total_cost_usd') or 0.0):.6f} "
            f"(eval=${float(cost.get('eval_cost_usd') or 0.0):.6f}, "
            f"optimizer=${float(cost.get('optimizer_cost_usd') or 0.0):.6f})",
            "- "
            f"Total tokens={int(cost.get('total_tokens') or 0)} "
            f"(eval={int(cost.get('eval_tokens') or 0)}, "
            f"optimizer={int(cost.get('optimizer_tokens') or 0)})",
            "- "
            f"Eval case evaluations={int(cost.get('eval_case_evaluations') or 0)}, "
            f"input_tokens={int(cost.get('total_input_tokens') or 0)}, "
            f"output_tokens={int(cost.get('total_output_tokens') or 0)}",
        ]

    @staticmethod
    def _optimizer_overhead_rows(result: RatchetResult) -> list[str]:
        profile = ((result.run_profile or {}).get("optimizer_calls") or {})
        totals = profile.get("totals") or {}
        if not totals:
            return ["No optimizer call diagnostics were recorded."]
        rows = [
            "- "
            f"Calls={totals.get('call_count', 0)}, elapsed={float(totals.get('elapsed_s') or 0.0):.3f}s, "
            f"tokens={int(totals.get('total_tokens') or 0)}, cost=${float(totals.get('cost_usd') or 0.0):.6f}"
        ]
        by_component = profile.get("by_component") or {}
        for name, item in sorted(by_component.items()):
            rows.append(
                "- "
                f"`{name}`: calls={item.get('call_count', 0)}, "
                f"elapsed={float(item.get('elapsed_s') or 0.0):.3f}s, "
                f"tokens={int(item.get('total_tokens') or 0)}, "
                f"cost=${float(item.get('cost_usd') or 0.0):.6f}"
            )
        return rows

    @staticmethod
    def _compact_profile_rows(rows: list[dict[str, Any]], metric: str, *, limit: int = 5) -> str:
        return "; ".join(
            f"{row.get('split_group')}:{row.get('case_id')}@{row.get('patch_hash')}={row.get(metric)}"
            for row in rows[:limit]
        )

    @staticmethod
    def _compact_patch_profile_rows(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
        return "; ".join(
            f"{row.get('split')}:{row.get('patch_hash')} score={float(row.get('mean_score') or 0.0):.3f} "
            f"cost=${float(row.get('mean_cost_usd') or 0.0):.6f} "
            f"tokens={float(row.get('mean_total_tokens') or 0.0):.1f} "
            f"lat={float(row.get('median_latency_s') or 0.0):.2f}s"
            for row in rows[:limit]
        )

    @staticmethod
    def _compact_patch_delta_rows(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
        return "; ".join(
            f"{row.get('split')}:{row.get('patch_hash')} "
            f"score_delta={float(row.get('score_delta') or 0.0):+.3f} "
            f"cost_delta=${float(row.get('cost_delta') or 0.0):+.6f} "
            f"token_delta={float(row.get('token_delta') or 0.0):+.1f} "
            f"lat_delta={float(row.get('latency_delta') or 0.0):+.2f}s"
            for row in rows[:limit]
        )

    @staticmethod
    def _patch_change_rows(patch: AgentPatch) -> list[str]:
        return [
            f"`{operation.op}` on `{operation.target}`: {RatchetReporter._format_value(operation.value)}"
            for operation in patch.operations
        ]

    @staticmethod
    def _transform_narrative(result: RatchetResult) -> str:
        tested = [
            name
            for name, summary in sorted(result.transform_summaries.items())
            if int(summary.get("evaluated_count") or 0) > 0
        ]
        promoted = [
            name
            for name, summary in sorted(result.transform_summaries.items())
            if summary.get("state") == "promoted"
        ]
        constrained = [
            name
            for name, summary in sorted(result.transform_summaries.items())
            if summary.get("state") == "constrained"
        ]
        paused = [
            name
            for name, summary in sorted(result.transform_summaries.items())
            if summary.get("state") == "paused"
        ]
        if not tested:
            return "Ratchet did not evaluate any transform-family candidates after profiling the current branch."
        parts = [f"Ratchet evaluated transform families: {', '.join(f'`{name}`' for name in tested)}."]
        if promoted:
            parts.append(f"Promoted families: {', '.join(f'`{name}`' for name in promoted)}.")
        if constrained:
            parts.append(f"Constrained families after regressions or repeated failed gates: {', '.join(f'`{name}`' for name in constrained)}.")
        if paused:
            parts.append(f"Paused families pending stronger evidence: {', '.join(f'`{name}`' for name in paused)}.")
        if not result.promoted:
            parts.append("No candidate cleared the configured dev and holdout gates, so Ratchet kept the baseline.")
        return " ".join(parts)

    @staticmethod
    def _format_value(value: Any) -> str:
        text = json.dumps(value, sort_keys=True) if not isinstance(value, str) else value
        return text if len(text) <= 160 else text[:157] + "..."


def _frontier_status_summaries(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in proposals:
        status = row.get("frontier_status")
        if not status:
            continue
        item = summaries.setdefault(
            str(status),
            {"count": 0, "best_score_delta": None, "best_cost_delta": None, "patch_hashes": []},
        )
        item["count"] += 1
        comparison = row.get("comparison_to_parent") or {}
        score_delta = comparison.get("score_delta")
        cost_delta = comparison.get("cost_delta")
        if score_delta is not None and (
            item["best_score_delta"] is None or float(score_delta) > float(item["best_score_delta"])
        ):
            item["best_score_delta"] = float(score_delta)
        if cost_delta is not None and (
            item["best_cost_delta"] is None or float(cost_delta) < float(item["best_cost_delta"])
        ):
            item["best_cost_delta"] = float(cost_delta)
        patch_hash_value = row.get("patch_hash")
        if patch_hash_value and patch_hash_value not in item["patch_hashes"]:
            item["patch_hashes"].append(str(patch_hash_value))
    return summaries


def _selected_finalist_status(
    finalist_statuses: list[dict[str, Any]],
    selected_patch_hash: str,
) -> dict[str, Any] | None:
    for row in finalist_statuses:
        if row.get("patch_hash") == selected_patch_hash:
            return {
                "patch_hash": row.get("patch_hash"),
                "status": row.get("status"),
                "stage": row.get("stage"),
                "reason": row.get("reason"),
                "passed_final_gate": row.get("passed_final_gate"),
                "comparison_to_baseline": row.get("comparison_to_baseline"),
            }
    return None
