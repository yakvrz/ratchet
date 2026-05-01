from __future__ import annotations

from collections import Counter
from html import escape
import json
from pathlib import Path
from typing import Any

from ratchet.adapters import AdapterProtocol
from ratchet.io import write_json, write_jsonl
from ratchet.objectives import compare_summaries
from ratchet.results import CandidateSummary, OptimizerStats, RatchetResult
from ratchet.transform_program import CompiledCandidate
from ratchet.types import OptimizationObjective


def _summary_dict(summary: CandidateSummary | None) -> dict[str, Any] | None:
    return summary.to_dict() if summary is not None else None


def _metric_text(summary: CandidateSummary | None, field: str, *, currency: bool = False) -> str:
    if summary is None:
        return "not measured"
    value = float(getattr(summary, field))
    if currency:
        return f"${value:.6f}"
    return f"{value:.3f}"


def build_outcome_analysis(
    *,
    objective: OptimizationObjective,
    promoted: bool,
    baseline_dev: CandidateSummary,
    accepted_dev_candidates: list[CandidateSummary],
    holdout_candidates: list[CandidateSummary],
    events: list[dict[str, Any]],
    finalist_statuses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    proposal_iterations = [event for event in events if event.get("type") == "proposal_iteration"]
    proposal_evaluations = [event for event in events if event.get("type") == "proposal_evaluation"]
    holdout_validations = [event for event in events if event.get("type") == "holdout_validation"]
    search_plan_events = [event for event in events if event.get("type") == "search_plan"]
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
    latest_search_plan = search_plan_events[-1].get("search_plan", {}) if search_plan_events else {}
    search_plan_diagnosis = str(latest_search_plan.get("diagnosis", ""))
    proposal_analysis = str(latest_iteration.get("proposal_analysis", ""))
    finalist_status_rows = list(finalist_statuses or [])
    if not finalist_status_rows:
        finalist_status_rows = [
            {
                "status": event.get("finalist_status"),
                "reason": event.get("rejection_reason"),
                "candidate_id": event.get("candidate_id"),
            }
            for event in holdout_validations
            if event.get("finalist_status")
        ]
    finalist_status_counts: Counter[str] = Counter(
        str(row.get("status")) for row in finalist_status_rows if row.get("status")
    )

    status = "promoted"
    summary = "Promoted an optimized candidate after holdout validation."
    if not promoted:
        if (
            objective.mode == "correctness"
            and baseline_dev.pass_count == baseline_dev.case_count
            and not proposal_evaluations
        ):
            status = "no_failures"
            summary = "Baseline had no dev failures under the correctness objective."
        elif proposal_evaluations and not accepted_dev_candidates:
            if any("tradeoff" in reason or "constraint rejected" in reason for reason in dev_rejection_reasons):
                status = "objective_tradeoff_rejected"
                summary = "Candidate proposals were evaluated but rejected by objective constraints or tradeoff guards."
            else:
                status = "proposals_evaluated_no_dev_gain"
                summary = "Candidate proposals ran on dev but did not improve the configured objective."
        elif (
            latest_stats.get("raw_count", 0) > 0
            and latest_stats.get("valid_count", 0) == 0
            and not accepted_dev_candidates
            and not holdout_candidates
        ):
            status = "proposals_invalid"
            summary = "The optimizing model returned candidates, but none satisfied the generated surface schema."
        elif proposal_iterations and not proposal_evaluations:
            status = "no_valid_model_proposals"
            if "failed" in proposal_analysis.lower():
                summary = "The optimizing model proposal call failed and Ratchet did not use a fallback."
            elif "No failing cases" in search_plan_diagnosis and objective.mode == "correctness":
                status = "no_failures"
                summary = "Baseline had no dev failures under the correctness objective."
            else:
                summary = "The optimizing model produced no valid candidates."
        elif accepted_dev_candidates and not holdout_candidates:
            if finalist_status_counts.get("unstable", 0) > 0:
                status = "runtime_baseline_unstable"
                summary = "At least one dev finalist improved, but paired stability checks showed runtime/baseline instability."
            elif finalist_status_counts.get("failed", 0) > 0:
                status = "finalists_failed_confirmation"
                summary = "At least one dev candidate improved, but finalist confirmation rejected all candidates before holdout."
            else:
                status = "holdout_not_run_budget_exhausted"
                summary = "At least one dev candidate improved, but holdout validation budget was zero."
        elif holdout_validations and not any(event.get("passed_final_gate") for event in holdout_validations):
            if any("tradeoff" in reason or "constraint rejected" in reason for reason in holdout_rejection_reasons):
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
        "accepted_dev_candidates": len(accepted_dev_candidates),
        "holdout_validations": len(holdout_validations),
        "latest_search_plan_diagnosis": search_plan_diagnosis,
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
        selected_comparison = (
            compare_summaries(result.baseline_holdout, result.selected_holdout)
            if result.baseline_holdout is not None and result.selected_holdout is not None
            else None
        )
        write_json(self.out_dir / "run_manifest.json", result.manifest)
        write_jsonl(self.out_dir / "events.jsonl", result.events)
        write_json(
            self.out_dir / "run_summary.json",
            {
                "selected_candidate_id": result.selected_candidate_id,
                "promoted": result.promoted,
                "outcome_analysis": result.outcome_analysis,
                "search_plan_count": len(result.search_plans),
                "proposal_count": len(result.proposals),
                "accepted_dev_count": len(result.accepted_dev_candidates),
                "holdout_candidate_count": len(result.holdout_candidates),
            },
        )
        write_json(self.out_dir / "outcome_analysis.json", result.outcome_analysis)
        write_json(self.out_dir / "evidence_ledger.json", result.evidence_ledger)
        write_jsonl(self.out_dir / "search_plans.jsonl", result.search_plans)
        write_jsonl(self.out_dir / "proposals.jsonl", result.proposals)
        write_json(
            self.out_dir / "candidate_metrics.json",
            {
                "baseline_dev": result.baseline_dev.to_dict(),
                "baseline_holdout": _summary_dict(result.baseline_holdout),
                "best_dev_candidate": result.best_dev_candidate.to_dict(),
                "selected_holdout": _summary_dict(result.selected_holdout),
                "accepted_dev_candidates": [summary.to_dict() for summary in result.accepted_dev_candidates],
                "holdout_candidates": [summary.to_dict() for summary in result.holdout_candidates],
                "pareto_frontier": result.pareto_frontier,
                "generated_surface": result.generated_surface,
                "search_plans": result.search_plans,
                "frontier_status_summaries": _frontier_status_summaries(result.proposals),
                "proposal_example_bank": result.manifest.get("proposal_example_bank", {}),
                "transform_summaries": result.transform_summaries,
                "transform_context_summaries": result.transform_context_summaries,
                "surface_opportunity_summaries": result.surface_opportunity_summaries,
                "finalist_statuses": result.finalist_statuses,
                "runtime_reliability_diagnostics": result.runtime_reliability_diagnostics,
                "confirmation_results": result.confirmation_results,
                "simplification_results": result.simplification_results,
                "frontier_recommendation": result.frontier_recommendation,
                "run_profile": result.run_profile,
                "run_cost": (result.run_profile or {}).get("run_cost", {}),
                "quality_cost_tradeoffs": result.quality_cost_tradeoffs,
                "ideation_metrics": result.ideation_metrics,
                "transform_trace_attribution": _transform_trace_attribution(result),
                "evidence_ledger": result.evidence_ledger,
                "optimizer_call_diagnostics": result.optimizer_call_diagnostics,
            },
        )
        write_json(self.out_dir / "ideation_metrics.json", result.ideation_metrics)
        write_json(
            self.out_dir / "selected_candidate.json",
            {
                "promoted": result.promoted,
                "selected_candidate_id": result.selected_candidate_id,
                "selected_finalist_status": _selected_finalist_status(
                    result.finalist_statuses,
                    result.selected_candidate_id,
                ),
                "candidate": result.selected_candidate.to_dict() if result.selected_candidate is not None else None,
                "objective": self.objective.to_dict(),
                "selection_reason": result.selection_reason,
                "outcome_analysis": result.outcome_analysis,
                "search_plans": result.search_plans,
                "frontier_status_summaries": _frontier_status_summaries(result.proposals),
                "transform_summaries": result.transform_summaries,
                "transform_context_summaries": result.transform_context_summaries,
                "surface_opportunity_summaries": result.surface_opportunity_summaries,
                "finalist_statuses": result.finalist_statuses,
                "runtime_reliability_diagnostics": result.runtime_reliability_diagnostics,
                "confirmation_results": result.confirmation_results,
                "simplification_results": result.simplification_results,
                "frontier_recommendation": result.frontier_recommendation,
                "run_profile": result.run_profile,
                "quality_cost_tradeoffs": result.quality_cost_tradeoffs,
                "ideation_metrics": result.ideation_metrics,
                "evidence_ledger": result.evidence_ledger,
                "optimizer_call_diagnostics": result.optimizer_call_diagnostics,
                "holdout_comparison_to_baseline": selected_comparison.to_dict()
                if selected_comparison is not None
                else None,
                "baseline": _summary_dict(result.baseline_holdout),
                "selected": _summary_dict(result.selected_holdout),
            },
        )
        export_dir = self.out_dir / "exported_candidate"
        self.adapter.export(result.selected_candidate, export_dir)
        self._write_report(result)
        self._write_summary_html(result)
        self._write_plots(result)

    def _write_report(self, result: RatchetResult) -> None:
        changes = self._candidate_change_rows(result.selected_candidate)
        comparison = (
            compare_summaries(result.baseline_holdout, result.selected_holdout)
            if result.baseline_holdout is not None and result.selected_holdout is not None
            else None
        )
        transform_narrative = self._transform_narrative(result)
        holdout_rows = self._holdout_comparison_rows(result, comparison)
        samples_per_case = (
            result.baseline_holdout.samples_per_case
            if result.baseline_holdout is not None
            else result.baseline_dev.samples_per_case
        )
        lines = [
            "# Ratchet Report",
            "",
            f"Outcome: {'promoted optimized candidate' if result.promoted else 'kept original baseline'}",
            f"Objective: `{self.objective.mode}`",
            f"Selected candidate: `{result.selected_candidate_id}`",
            f"Outcome status: `{result.outcome_analysis['status']}`",
            f"Outcome summary: {result.outcome_analysis['summary']}",
            f"Recommendation: {result.frontier_recommendation.get('reason', result.selection_reason)}",
            f"Recommendation policy: `{result.frontier_recommendation.get('recommendation_policy', 'n/a')}`",
            "",
            "## Task Theory",
            "",
            *self._search_plan_rows(result),
            "",
            "## Frontier Categories",
            "",
            *self._frontier_status_rows(result),
            "",
            "## Early Evidence Quality",
            "",
            *self._early_evidence_quality_rows(result),
            "",
            "## Small-Dev Triage",
            "",
            *self._small_dev_screening_rows(result),
            "",
            "## Runtime Baseline Stability",
            "",
            *self._runtime_baseline_stability_rows(result),
            "",
            "## Measurement Efficiency",
            "",
            *self._measurement_efficiency_rows(result),
            "",
            "## Holdout Frontier",
            "",
            *self._frontier_variant_rows(result),
            "",
            "## Baseline vs Selected Holdout",
            "",
            *holdout_rows,
            "",
            "## Holdout Paired Evidence",
            "",
            *self._holdout_paired_evidence_rows(comparison),
            "",
            "## Selected Candidate",
            "",
            *(changes or ["No transform operations; original baseline selected."]),
            "",
            "## Transform Attribution",
            "",
            *self._transform_attribution_rows(result),
            "",
            "## Generated Surface",
            "",
            *_surface_rows(result.generated_surface),
            "",
            "## Optimization Trace",
            "",
            f"- Proposal iterations: {result.outcome_analysis['proposal_iterations']}",
            f"- Proposal evaluations: {result.outcome_analysis['proposal_evaluations']}",
            f"- Accepted dev candidates: {result.outcome_analysis['accepted_dev_candidates']}",
            f"- Holdout validations: {result.outcome_analysis['holdout_validations']}",
            f"- Latest search-plan diagnosis: {result.outcome_analysis['latest_search_plan_diagnosis'] or 'n/a'}",
            f"- Latest proposal: {result.outcome_analysis['latest_proposal_analysis'] or 'n/a'}",
            f"- Rejection reasons: {json.dumps(result.outcome_analysis['rejection_reasons'], sort_keys=True)}",
            f"- Finalist status counts: {json.dumps(result.outcome_analysis.get('finalist_status_counts', {}), sort_keys=True)}",
            "",
            "## Search Steps",
            "",
            *self._search_step_rows(result),
            "",
            "## Ideation Quality",
            "",
            *self._ideation_quality_rows(result),
            "",
            "## Finalist Statuses",
            "",
            *self._finalist_status_rows(result),
            "",
            "## Simplification",
            "",
            *self._simplification_rows(result),
            "",
            "## Few-Shot Examples",
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
            "## Surface Mechanisms",
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
            "## Surface Opportunities",
            "",
            *self._surface_opportunity_summary_rows(result),
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
            f"- Local cache hits: {self.stats.local_cache_hits}",
            f"- Shared cache hits: {self.stats.shared_cache_hits}",
            f"- Fresh case evaluations: {self.stats.fresh_case_evaluations}",
            f"- Runtime errors: {self.stats.runtime_errors}",
            f"- Grader errors: {self.stats.grader_errors}",
            f"- Timeouts: {self.stats.timeouts}",
            f"- Samples per case: {samples_per_case:g}",
            f"- Proposal-safe train examples: {(result.manifest.get('proposal_example_bank') or {}).get('example_count', 0)}",
        ]
        (self.out_dir / "report.md").write_text("\n".join(lines) + "\n")

    def _write_summary_html(self, result: RatchetResult) -> None:
        rows = self._candidate_change_rows(result.selected_candidate)
        html_rows = "".join(f"<li>{escape(row)}</li>" for row in rows) or "<li>Original baseline kept.</li>"
        baseline_score = _metric_text(result.baseline_holdout, "mean_score")
        selected_score = _metric_text(result.selected_holdout, "mean_score")
        selected_cost = _metric_text(result.selected_holdout, "mean_cost_usd", currency=True)
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
  <h1>{'Promoted candidate' if result.promoted else 'Baseline kept'}</h1>
  <p>{escape(result.selection_reason)}</p>
  <p><strong>{escape(str(result.outcome_analysis['status']))}</strong>: {escape(str(result.outcome_analysis['summary']))}</p>
  <h2>Outcome</h2>
  <div class="metric"><span>Baseline holdout score</span><strong>{baseline_score}</strong></div>
  <div class="metric"><span>Selected holdout score</span><strong>{selected_score}</strong></div>
  <div class="metric"><span>Selected holdout cost</span><strong>{selected_cost}</strong></div>
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

    @staticmethod
    def _holdout_comparison_rows(result: RatchetResult, comparison: Any | None) -> list[str]:
        if result.baseline_holdout is None or result.selected_holdout is None or comparison is None:
            return ["Holdout was not measured because no candidate reached final validation."]
        return [
            "| Metric | Baseline | Selected |",
            "| --- | ---: | ---: |",
            f"| Mean score | {result.baseline_holdout.mean_score:.3f} | {result.selected_holdout.mean_score:.3f} |",
            f"| Pass count | {result.baseline_holdout.pass_count} | {result.selected_holdout.pass_count} |",
            f"| Avg cost | ${result.baseline_holdout.mean_cost_usd:.6f} | ${result.selected_holdout.mean_cost_usd:.6f} |",
            f"| Median latency | {result.baseline_holdout.median_latency_s:.2f}s | {result.selected_holdout.median_latency_s:.2f}s |",
            f"| Samples | {result.baseline_holdout.sample_count} over {result.baseline_holdout.case_count} cases | {result.selected_holdout.sample_count} over {result.selected_holdout.case_count} cases |",
            f"| Split-vote cases | {len(result.baseline_holdout.split_vote_case_ids)} | {len(result.selected_holdout.split_vote_case_ids)} |",
        ]

    @staticmethod
    def _holdout_paired_evidence_rows(comparison: Any | None) -> list[str]:
        if comparison is None:
            return ["Holdout paired evidence was not computed."]
        rows = [
            f"- Score delta: {comparison.score_delta:.4f} CI [{comparison.score_ci[0]:.4f}, {comparison.score_ci[1]:.4f}]",
            f"- Cost delta: ${comparison.cost_delta:.6f} CI [${comparison.cost_ci[0]:.6f}, ${comparison.cost_ci[1]:.6f}]",
            f"- Token delta: {comparison.token_delta:.1f} CI [{comparison.token_ci[0]:.1f}, {comparison.token_ci[1]:.1f}]",
            f"- Latency delta: {comparison.latency_delta:.3f}s CI [{comparison.latency_ci[0]:.3f}s, {comparison.latency_ci[1]:.3f}s]",
        ]
        if comparison.pass_significance is not None:
            sig = comparison.pass_significance
            rows.append(
                f"- Pass flips: fixed={sig.fixed_count}, regressed={sig.regressed_count}, paired p={sig.p_value:.4f}"
            )
        return rows

    @staticmethod
    def _transform_attribution_rows(result: RatchetResult) -> list[str]:
        attribution = _transform_trace_attribution(result)
        candidates = attribution.get("candidates") or {}
        if not candidates:
            return ["No transform trace attribution was recorded."]
        rows = [
            "| Candidate | Split | Trace events | Top operations | Skipped-tool mismatches |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for candidate_id_value, item in sorted(candidates.items()):
            operation_counts = item.get("operation_counts") or {}
            skipped_tools = item.get("skipped_tool_counts") or {}
            top_ops = ", ".join(
                f"`{name}` x{count}" for name, count in list(operation_counts.items())[:4]
            ) or "none"
            skipped = ", ".join(
                f"`{name}` x{count}" for name, count in list(skipped_tools.items())[:3]
            ) or "none"
            rows.append(
                "| "
                f"`{candidate_id_value}` | "
                f"{', '.join(item.get('splits') or []) or 'n/a'} | "
                f"{int(item.get('trace_event_count') or 0)} | "
                f"{top_ops} | "
                f"{skipped} |"
            )
        return rows

    def _write_scorecard_svg(self, path: Path, result: RatchetResult) -> None:
        if result.baseline_holdout is None or result.selected_holdout is None:
            path.write_text(
                "\n".join(
                    [
                        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="180">',
                        '<rect width="100%" height="100%" fill="#ffffff"/>',
                        '<text x="32" y="36" font-size="22" font-family="Arial">Holdout not measured</text>',
                        '<text x="32" y="78" font-size="14" font-family="Arial">No candidate reached final holdout validation.</text>',
                        "</svg>",
                    ]
                )
            )
            return
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
        summaries = [result.baseline_dev, *result.accepted_dev_candidates]
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
            candidate_id_value = str(item.get("candidate_id", "unknown"))
            reason = str(item.get("reason") or "passed")
            stage = str(item.get("stage") or "unknown")
            holdout_metrics = item.get("holdout_metrics") or {}
            pass_count = holdout_metrics.get("pass_count")
            case_count = holdout_metrics.get("case_count")
            metric_text = f"; holdout={pass_count}/{case_count}" if pass_count is not None and case_count is not None else ""
            rows.append(f"- `{candidate_id_value}`: `{status}` at `{stage}`{metric_text}; {reason}")
        return rows

    @staticmethod
    def _search_step_rows(result: RatchetResult) -> list[str]:
        rows = [
            event
            for event in result.events
            if event.get("type") in {"search_plan_call", "search_plan"}
        ]
        if not rows:
            return ["No search planner decisions were recorded."]
        table = [
            "| Iteration | Type | Output | Rationale |",
            "| ---: | --- | --- | --- | --- |",
        ]
        for row in rows[-12:]:
            row_type = str(row.get("type") or "")
            plan = row.get("search_plan") or {}
            briefs = plan.get("briefs") or []
            output = ", ".join(str(item.get("brief_id", ""))[:16] for item in briefs if isinstance(item, dict))
            rationale = f"{len(briefs)} brief(s)"
            table.append(
                "| "
                f"{row.get('iteration')} | "
                f"`{row_type}` | "
                f"{output or 'none'} | "
                f"{rationale} |"
            )
        if len(rows) > 12:
            table.append(f"- {len(rows) - 12} earlier research decisions omitted from this table.")
        return table

    @staticmethod
    def _ideation_quality_rows(result: RatchetResult) -> list[str]:
        metrics = result.ideation_metrics or {}
        if not metrics:
            return ["No ideation telemetry was recorded."]
        planner = metrics.get("planner") or {}
        implementer = metrics.get("implementer") or {}
        discovery = metrics.get("discovery") or {}
        stage_counts = discovery.get("stage_counts") or {}
        invalid_reasons = implementer.get("invalid_reason_counts") or {}
        rows = [
            f"- Planner briefs: {planner.get('brief_count', 0)} across {planner.get('plan_count', 0)} plan call(s).",
            f"- Briefs citing surface opportunities: {planner.get('brief_with_surface_opportunity_ids', 0)}.",
            f"- Implementer candidates: valid={implementer.get('valid_candidate_count', 0)} invalid={implementer.get('invalid_candidate_count', 0)}.",
            f"- Discovery stages: {json.dumps(stage_counts, sort_keys=True)}",
        ]
        if invalid_reasons:
            rows.append(f"- Invalid implementation reasons: {json.dumps(invalid_reasons, sort_keys=True)}")
        missing = planner.get("missing_opportunity_mechanisms") or []
        if missing:
            rows.append(f"- Planner mechanisms not covered by candidates: {json.dumps(missing[:8])}")
        return rows

    @staticmethod
    def _search_plan_rows(result: RatchetResult) -> list[str]:
        if not result.search_plans:
            return ["No search plan snapshots were recorded."]
        rows = []
        for item in result.search_plans[-5:]:
            plan = item.get("search_plan") or {}
            mechanisms = list(plan.get("target_mechanisms") or plan.get("active_mechanisms") or [])[:3]
            briefs = [
                str(row.get("brief_id"))
                for row in plan.get("briefs", [])[:3]
                if isinstance(row, dict) and row.get("brief_id")
            ]
            rows.append(
                "- "
                f"iteration={item.get('iteration')} parent=`{item.get('parent_candidate_id')}` "
                f"plan=`{plan.get('plan_id')}` "
                f"briefs={json.dumps(briefs)} "
                f"mechanisms={json.dumps(mechanisms)} "
                f"confidence={plan.get('confidence')}"
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
                f"{', '.join(str(value) for value in item.get('candidate_ids', [])[:4])} |"
            )
        return rows

    @staticmethod
    def _small_dev_screening_rows(result: RatchetResult) -> list[str]:
        rows = [
            row
            for row in result.proposals
            if row.get("frontier_status") == "screened_out" and row.get("evaluation_stages")
        ]
        if not rows:
            return ["No candidates were screened after small-dev triage."]
        table = [
            "| Candidate | Reason | Small-dev cases | Score delta | Pass gain | Failure-label delta | Mechanism |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
        for row in rows[:12]:
            signal = _small_dev_signal(row)
            table.append(
                "| "
                f"`{row.get('candidate_id')}` | "
                f"{_short_reason(row.get('rejection_reason'))} | "
                f"{signal['case_count']} | "
                f"{signal['score_delta']:+.3f} | "
                f"{signal['pass_gain']:+d} | "
                f"{signal['failure_label_delta']} | "
                f"`{row.get('mechanism_class') or row.get('surface_mechanism')}` |"
            )
        if len(rows) > 12:
            table.append(f"- {len(rows) - 12} additional screened candidates omitted from this table.")
        return table

    @staticmethod
    def _early_evidence_quality_rows(result: RatchetResult) -> list[str]:
        ledger = result.evidence_ledger or {}
        records = list(ledger.get("records") or [])
        if not records:
            return ["No staged evidence records were written."]
        summary = dict(ledger.get("summary") or {})
        rows = [
            f"- Evidence records: {summary.get('record_count', len(records))}",
            f"- Stage counts: `{json.dumps(summary.get('stage_counts', {}), sort_keys=True)}`",
            f"- Confidence tiers: `{json.dumps(summary.get('confidence_counts', {}), sort_keys=True)}`",
            "| Candidate | Stage | Cases | Confidence | Pass gain | Score delta | Instability | Mechanism |",
            "| --- | --- | ---: | --- | ---: | ---: | --- | --- |",
        ]
        ranked = sorted(
            records,
            key=lambda row: (
                str(row.get("candidate_id") or ""),
                str(row.get("stage") or ""),
            ),
        )
        for row in ranked[:12]:
            comparison = row.get("comparison_to_reference") or {}
            rows.append(
                "| "
                f"`{row.get('candidate_id')}` | "
                f"`{row.get('stage')}` | "
                f"{row.get('case_count', 0)} | "
                f"`{row.get('confidence_tier', 'n/a')}` | "
                f"{int(row.get('pass_gain') or 0):+d} | "
                f"{float(comparison.get('score_delta') or 0.0):+.3f} | "
                f"{', '.join(row.get('baseline_instability_flags') or []) or 'none'} | "
                f"`{row.get('mechanism_class') or 'n/a'}` |"
            )
        if len(records) > 12:
            rows.append(f"- {len(records) - 12} additional evidence records omitted from this table.")
        return rows

    @staticmethod
    def _runtime_baseline_stability_rows(result: RatchetResult) -> list[str]:
        rows: list[str] = []
        baseline_stability = result.manifest.get("baseline_stability") or {}
        instability_counts = baseline_stability.get("instability_counts") or {}
        rows.append(f"- Early-stage instability flags: `{json.dumps(instability_counts, sort_keys=True)}`")
        rows.append(f"- Runtime repeat required: {bool(baseline_stability.get('requires_runtime_repeat'))}")
        confirmations = result.confirmation_results or []
        if not confirmations:
            rows.append("- No runtime/output paired stability checks were required.")
            return rows
        rows.extend(
            [
                "| Candidate | Status | Passed | Reason |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in confirmations:
            rows.append(
                "| "
                f"`{item.get('candidate_id')}` | "
                f"`{item.get('status') or item.get('diagnostic_class')}` | "
                f"{bool(item.get('passed'))} | "
                f"{_short_reason(item.get('reason'))} |"
            )
        return rows

    @staticmethod
    def _measurement_efficiency_rows(result: RatchetResult) -> list[str]:
        summary = (result.evidence_ledger or {}).get("summary") or {}
        measurement_cost = summary.get("measurement_cost") or {}
        run_cost = (result.run_profile or {}).get("run_cost", {})
        return [
            f"- Ledger-estimated candidate measurement cost: `${float(measurement_cost.get('estimated_total_cost_usd') or 0.0):.6f}`",
            f"- Ledger-estimated candidate measurement tokens: {int(measurement_cost.get('estimated_total_tokens') or 0)}",
            f"- Ledger-estimated model calls: {int(measurement_cost.get('estimated_model_calls') or 0)}",
            f"- Ledger-estimated tool calls: {int(measurement_cost.get('estimated_tool_calls') or 0)}",
            f"- Ledger-estimated interaction turns: {int(measurement_cost.get('estimated_turns') or 0)}",
            f"- Dev measurement budget used: `${float(result.manifest.get('dev_measurement_cost_used_usd') or 0.0):.6f}`"
            f" / `{result.manifest.get('max_dev_measurement_cost_usd')}`",
            f"- Holdout measurement budget used: `${float(result.manifest.get('holdout_measurement_cost_used_usd') or 0.0):.6f}`"
            f" / `{result.manifest.get('max_holdout_measurement_cost_usd')}`",
            f"- Dev tool-call budget used: `{float(result.manifest.get('dev_measurement_tool_calls_used') or 0.0):.1f}`"
            f" / `{result.manifest.get('max_dev_measurement_tool_calls')}`",
            f"- Holdout tool-call budget used: `{float(result.manifest.get('holdout_measurement_tool_calls_used') or 0.0):.1f}`"
            f" / `{result.manifest.get('max_holdout_measurement_tool_calls')}`",
            f"- Dev turn budget used: `{float(result.manifest.get('dev_measurement_turns_used') or 0.0):.1f}`"
            f" / `{result.manifest.get('max_dev_measurement_turns')}`",
            f"- Holdout turn budget used: `{float(result.manifest.get('holdout_measurement_turns_used') or 0.0):.1f}`"
            f" / `{result.manifest.get('max_holdout_measurement_turns')}`",
            f"- Expensive deployed-policy reporting threshold: `{result.manifest.get('expensive_candidate_cost_ratio')}x` baseline cost",
            f"- Total eval cost: `${float(run_cost.get('eval_cost_usd') or 0.0):.6f}`",
            f"- Fresh case evaluations: {((result.manifest.get('stats') or {}).get('fresh_case_evaluations', 0))}",
            f"- Local case cache hits: {((result.manifest.get('stats') or {}).get('local_cache_hits', 0))}",
            f"- Shared case cache hits: {((result.manifest.get('stats') or {}).get('shared_cache_hits', 0))}",
        ]

    @staticmethod
    def _surface_opportunity_summary_rows(result: RatchetResult) -> list[str]:
        if not result.surface_opportunity_summaries:
            return ["No surface opportunity applications were recorded."]
        rows = [
            "| Surface opportunity | State | Proposed | Evaluated | Accepted | Best score delta | Notes |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        ranked = sorted(
            result.surface_opportunity_summaries.values(),
            key=lambda item: (
                str(item.get("state") or ""),
                str(item.get("surface_opportunity_id") or ""),
            ),
        )
        for item in ranked[:20]:
            rows.append(
                "| "
                f"`{item.get('surface_opportunity_id')}` | "
                f"`{item.get('state')}` | "
                f"{item.get('proposed_count', 0)} | "
                f"{item.get('evaluated_count', 0)} | "
                f"{item.get('accepted_count', 0)} | "
                f"{float(item.get('best_score_delta') or 0.0):+.3f} | "
                f"{_short_reason(item.get('reason'))} |"
            )
        if len(ranked) > 20:
            rows.append(f"- {len(ranked) - 20} additional surface opportunity summaries omitted from this table.")
        return rows

    @staticmethod
    def _frontier_variant_rows(result: RatchetResult) -> list[str]:
        variants = result.frontier_recommendation.get("frontier_variants") or []
        if not variants:
            if result.holdout_candidates:
                statuses = {
                    str(item.get("candidate_id")): item
                    for item in result.finalist_statuses
                    if item.get("candidate_id")
                }
                variants = [
                    {
                        "role": "holdout_candidate"
                        + (
                            f" ({statuses.get(item.candidate_id, {}).get('status')})"
                            if statuses.get(item.candidate_id, {}).get("status")
                            else ""
                        ),
                        "candidate_id": item.candidate_id,
                        "pass_count": item.pass_count,
                        "case_count": item.case_count,
                        "mean_score": item.mean_score,
                        "mean_cost_usd": item.mean_cost_usd,
                        "mean_total_tokens": item.mean_total_tokens,
                        "median_latency_s": item.median_latency_s,
                        "operation_count": item.operation_count,
                        "operations": [
                            {
                                "op": operation.op.op,
                                "target": operation.hook or "on_task_start",
                                "value_summary": RatchetReporter._format_value(operation.op.params),
                            }
                            for operation in (item.candidate.program.patches if item.candidate is not None else ())
                        ],
                    }
                    for item in result.holdout_candidates
                ]
            else:
                return ["No holdout finalist candidates were available."]
        rows = [
            "| Role | Candidate | Holdout | Score | Cost | Tokens | Latency | Ops |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        selected_hash = result.selected_candidate_id
        for item in variants:
            role = str(item.get("role", "candidate"))
            if item.get("candidate_id") == selected_hash:
                role = f"{role} (selected)"
            rows.append(
                "| "
                f"{role} | "
                f"`{item.get('candidate_id')}` | "
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
                f"`{item.get('candidate_id')}` ({item.get('diagnostic_class', 'runtime_finding')}): {item.get('reason')} "
                f"fixed_invalid={item.get('fixed_invalid_output_case_ids', [])}, "
                f"low_token_fixed={item.get('low_token_fixed_case_ids', [])}, "
                f"finish_changed={item.get('finish_reason_changed_case_ids', [])}, "
                f"finish_reasons={json.dumps(item.get('finish_reasons_by_case', {}), sort_keys=True)}"
            )
        if result.confirmation_results:
            for item in result.confirmation_results:
                rows.append(
                    "- "
                    f"confirmation `{item.get('candidate_id')}`: "
                    f"passed={item.get('passed')}; {item.get('reason')}"
                )
        return rows

    @staticmethod
    def _few_shot_compression_rows(result: RatchetResult) -> list[str]:
        rows = [
            row
            for row in result.proposals
            if row.get("surface_mechanism") == "surface_examples" and row.get("metrics")
        ]
        if not rows:
            return ["No evaluated few-shot example candidates were recorded."]
        table = [
            "| Candidate | Status | Examples | Strategy | Score Delta | Token Delta | Tokens / Example | Source IDs |",
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
                f"`{row.get('candidate_id')}` | "
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
            return ["Finalist simplification is not enabled in this Ratchet run."]
        evaluated = [row for row in result.simplification_results if row.get("type") == "simplification_evaluation"]
        skipped = [row for row in result.simplification_results if row.get("type") == "simplification_skipped"]
        accepted = [row for row in evaluated if row.get("accepted")]
        rejected = [row for row in evaluated if not row.get("accepted")]
        rows = [
            f"- Evaluated {len(evaluated)} simplification variants: {len(accepted)} accepted, {len(rejected)} rejected; {len(skipped)} skipped."
        ]
        for item in result.simplification_results[:8]:
            if item.get("type") == "simplification_skipped":
                rows.append(
                    "- "
                    f"skipped parent `{item.get('parent_candidate_id')}`: "
                    f"{item.get('reason') or 'simplification skipped'}"
                )
                continue
            simplification = item.get("simplification") or {}
            rows.append(
                "- "
                f"`{item.get('candidate_id')}` from `{item.get('parent_candidate_id')}`: "
                f"{simplification.get('type', 'simplification')} accepted={item.get('accepted')}; "
                f"{item.get('rejection_reason') or 'passed dev gate'}"
            )
        return rows

    @staticmethod
    def _quality_cost_tradeoff_rows(result: RatchetResult) -> list[str]:
        if not result.quality_cost_tradeoffs:
            return ["No model substitutions failed promotion solely because of deployed cost/latency constraints."]
        return [
            "- "
            f"`{item.get('candidate_id')}`: {item.get('rejection_reason')} "
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
        candidate_profiles = profile.get("candidate_profiles") or []
        if candidate_profiles:
            rows.append("- Candidate profiles: " + RatchetReporter._compact_candidate_profile_rows(candidate_profiles))
        candidate_deltas = profile.get("candidate_deltas_vs_baseline") or []
        if candidate_deltas:
            rows.append("- Candidate deltas vs baseline: " + RatchetReporter._compact_candidate_delta_rows(candidate_deltas))
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
            f"{row.get('split_group')}:{row.get('case_id')}@{row.get('candidate_id')}={row.get(metric)}"
            for row in rows[:limit]
        )

    @staticmethod
    def _compact_candidate_profile_rows(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
        return "; ".join(
            f"{row.get('split')}:{row.get('candidate_id')} score={float(row.get('mean_score') or 0.0):.3f} "
            f"cost=${float(row.get('mean_cost_usd') or 0.0):.6f} "
            f"tokens={float(row.get('mean_total_tokens') or 0.0):.1f} "
            f"lat={float(row.get('median_latency_s') or 0.0):.2f}s"
            for row in rows[:limit]
        )

    @staticmethod
    def _compact_candidate_delta_rows(rows: list[dict[str, Any]], *, limit: int = 6) -> str:
        return "; ".join(
            f"{row.get('split')}:{row.get('candidate_id')} "
            f"score_delta={float(row.get('score_delta') or 0.0):+.3f} "
            f"cost_delta=${float(row.get('cost_delta') or 0.0):+.6f} "
            f"token_delta={float(row.get('token_delta') or 0.0):+.1f} "
            f"lat_delta={float(row.get('latency_delta') or 0.0):+.2f}s"
            for row in rows[:limit]
        )

    @staticmethod
    def _candidate_change_rows(candidate: CompiledCandidate | None) -> list[str]:
        if candidate is None:
            return []
        return [
            f"`{operation.op.op}` at `{operation.hook or 'on_task_start'}`: {RatchetReporter._format_value(operation.op.params)}"
            for operation in candidate.program.patches
        ]

    @staticmethod
    def _transform_narrative(result: RatchetResult) -> str:
        tested = [
            name
            for name, summary in sorted(result.transform_summaries.items())
            if int(summary.get("evaluated_count") or 0) > 0
        ]
        dev_eligible = [
            name
            for name, summary in sorted(result.transform_summaries.items())
            if summary.get("state") == "promotable_dev"
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
            return "Ratchet did not evaluate any surface-program candidates after profiling the current branch."
        parts = [f"Ratchet evaluated surface mechanisms: {', '.join(f'`{name}`' for name in tested)}."]
        if dev_eligible:
            parts.append(f"Dev-eligible surface mechanisms: {', '.join(f'`{name}`' for name in dev_eligible)}.")
        if constrained:
            parts.append(f"Constrained surface mechanisms after regressions or repeated failed gates: {', '.join(f'`{name}`' for name in constrained)}.")
        if paused:
            parts.append(f"Paused surface mechanisms pending stronger evidence: {', '.join(f'`{name}`' for name in paused)}.")
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
            {"count": 0, "best_score_delta": None, "best_cost_delta": None, "candidate_ids": []},
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
        candidate_id_value = row.get("candidate_id")
        if candidate_id_value and candidate_id_value not in item["candidate_ids"]:
            item["candidate_ids"].append(str(candidate_id_value))
    return summaries


def _transform_trace_attribution(result: RatchetResult) -> dict[str, Any]:
    summaries = [*result.accepted_dev_candidates, *result.holdout_candidates]
    by_candidate: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        if summary.candidate is None:
            continue
        item = by_candidate.setdefault(
            summary.candidate_id,
            {
                "candidate_id": summary.candidate_id,
                "splits": set(),
                "trace_event_count": 0,
                "operation_counts": Counter(),
                "skipped_tool_counts": Counter(),
            },
        )
        item["splits"].add(summary.split)
        for evaluation in summary.evaluations:
            trace = evaluation.record.diagnostics.metadata.get("transform_trace", [])
            if not isinstance(trace, list):
                continue
            for event in trace:
                if not isinstance(event, dict):
                    continue
                hook = str(event.get("hook") or "unknown")
                op = str(event.get("op") or "unknown")
                trace_result = str(event.get("result") or "applied")
                item["trace_event_count"] += 1
                item["operation_counts"][f"{hook}:{op}:{trace_result}"] += 1
                fields = event.get("fields") if isinstance(event.get("fields"), dict) else {}
                if trace_result == "skipped_tool":
                    expected = str(fields.get("tool") or "unknown")
                    actual = str(fields.get("actual") or "unknown")
                    item["skipped_tool_counts"][f"{expected}->{actual}"] += 1
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_id_value, item in by_candidate.items():
        candidates[candidate_id_value] = {
            "candidate_id": candidate_id_value,
            "splits": sorted(item["splits"]),
            "trace_event_count": int(item["trace_event_count"]),
            "operation_counts": dict(sorted(item["operation_counts"].items(), key=lambda pair: (-pair[1], pair[0]))),
            "skipped_tool_counts": dict(sorted(item["skipped_tool_counts"].items(), key=lambda pair: (-pair[1], pair[0]))),
        }
    return {"candidates": candidates}


def _small_dev_signal(row: dict[str, Any]) -> dict[str, Any]:
    stages = row.get("evaluation_stages") or []
    stage = next((item for item in reversed(stages) if item.get("stage") == "small_dev"), None)
    if stage is None:
        stage = stages[-1] if stages else {}
    comparison = stage.get("comparison_to_parent") or {}
    flips = stage.get("behavior_flip_summary") or {}
    try:
        pass_gain = int(flips.get("fixed_count") or 0) - int(flips.get("regressed_count") or 0)
    except (TypeError, ValueError):
        pass_gain = 0
    return {
        "case_count": int(stage.get("case_count") or 0),
        "score_delta": float(comparison.get("score_delta") or 0.0),
        "pass_gain": pass_gain,
        "failure_label_delta": _top_failure_label_delta(stage.get("failure_label_delta") or {}),
    }


def _top_failure_label_delta(delta: Any) -> str:
    if not isinstance(delta, dict) or not delta:
        return "none"
    rows = []
    for label, payload in delta.items():
        if not isinstance(payload, dict):
            continue
        change = int(payload.get("delta") or 0)
        if change == 0:
            continue
        rows.append((abs(change), label, change))
    if not rows:
        return "none"
    rows.sort(reverse=True)
    return ", ".join(f"`{label}` {change:+d}" for _magnitude, label, change in rows[:3])


def _short_reason(reason: Any) -> str:
    text = str(reason or "screened")
    return text.split(":", 1)[0] if ":" in text else text


def _selected_finalist_status(
    finalist_statuses: list[dict[str, Any]],
    selected_candidate_id: str,
) -> dict[str, Any] | None:
    for row in finalist_statuses:
        if row.get("candidate_id") == selected_candidate_id:
            return {
                "candidate_id": row.get("candidate_id"),
                "status": row.get("status"),
                "stage": row.get("stage"),
                "reason": row.get("reason"),
                "passed_final_gate": row.get("passed_final_gate"),
                "comparison_to_baseline": row.get("comparison_to_baseline"),
            }
    return None


def _surface_rows(surface_rows: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    for row in surface_rows:
        if "agent_id" in row:
            rows.append(
                f"- `{row['agent_id']}`: context_sections={len((row.get('context') or {}).get('graph', {}).get('sections', []))}; "
                f"hooks={len(row.get('hooks') or {})}"
            )
            continue
        rows.append(
            f"- `{row.get('name', 'surface')}` ({row.get('kind', 'unknown')}): "
            f"{', '.join(row.get('allowed_ops') or [])}; schema={json.dumps(row.get('value_schema', {}), sort_keys=True)}"
        )
    return rows
