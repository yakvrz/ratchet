from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from ratchet.results import Comparison
from ratchet.surface_search import TransformContextKey, _context_lifecycle_state, _context_summary_reason


@dataclass(frozen=True)
class TransformResultSummary:
    family: str
    proposed_count: int = 0
    evaluated_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    best_score_delta: float | None = None
    best_cost_delta: float | None = None
    best_latency_delta: float | None = None
    state: str = "available"
    reason: str = "No candidates evaluated for this surface mechanism."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_transform_results(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    context_summaries = summarize_transform_context_results(proposals)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    proposed_counts: Counter[str] = Counter()
    for row in proposals:
        family = str(row.get("surface_mechanism") or "unknown")
        if row.get("type") == "candidate_proposal":
            proposed_counts[family] += 1
        else:
            grouped[family].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for family in sorted(set(grouped) | set(proposed_counts)):
        rows = grouped.get(family, [])
        evaluated_count = len(rows)
        proposed_count = max(proposed_counts.get(family, 0), evaluated_count)
        accepted_rows = [row for row in rows if row.get("accepted")]
        comparisons = [row.get("comparison_to_parent") or {} for row in rows]
        score_deltas = [float(item["score_delta"]) for item in comparisons if "score_delta" in item]
        cost_deltas = [float(item["cost_delta"]) for item in comparisons if "cost_delta" in item]
        latency_deltas = [float(item["latency_delta"]) for item in comparisons if "latency_delta" in item]
        score_regressed = any(delta < 0 for delta in score_deltas)
        if accepted_rows:
            state = "promotable_dev"
            reason = "At least one candidate from this surface mechanism earned finalist eligibility on dev."
        elif evaluated_count >= 2 or score_regressed:
            state = "constrained"
            reason = (
                "At least one candidate from this surface mechanism regressed score; future attempts should use a distinct target, slice, or instance."
                if score_regressed
                else "Multiple evaluated candidates failed the configured objective gate; future attempts should avoid near-duplicate instances."
            )
        elif evaluated_count == 1:
            state = "paused"
            reason = "The evaluated candidate failed the configured objective gate."
        else:
            state = "available"
            reason = "No candidates evaluated for this surface mechanism."
        summaries[family] = TransformResultSummary(
            family=family,
            proposed_count=proposed_count,
            evaluated_count=evaluated_count,
            accepted_count=len(accepted_rows),
            rejected_count=max(evaluated_count - len(accepted_rows), 0),
            best_score_delta=max(score_deltas) if score_deltas else None,
            best_cost_delta=min(cost_deltas) if cost_deltas else None,
            best_latency_delta=min(latency_deltas) if latency_deltas else None,
            state=state,
            reason=reason,
        ).to_dict()
        family_contexts = [
            summary
            for summary in context_summaries.values()
            if ((summary.get("key") or {}).get("family") == family)
        ]
        if family_contexts:
            if any(summary.get("state") == "promotable_dev" for summary in family_contexts):
                summaries[family]["state"] = "promotable_dev"
            elif any(summary.get("state") == "active" for summary in family_contexts):
                summaries[family]["state"] = "active"
            elif any(summary.get("state") == "constrained" for summary in family_contexts):
                summaries[family]["state"] = "constrained"
            elif any(summary.get("state") == "paused" for summary in family_contexts):
                summaries[family]["state"] = "paused"
    return summaries


def summarize_transform_context_results(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        if "accepted" not in row:
            continue
        key = TransformContextKey.from_row(row)
        grouped[key.id].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for context_id, rows in sorted(grouped.items()):
        key = TransformContextKey.from_row(rows[-1])
        comparisons = [row.get("comparison_to_parent") or {} for row in rows]
        score_deltas = [float(item["score_delta"]) for item in comparisons if "score_delta" in item]
        cost_deltas = [float(item["cost_delta"]) for item in comparisons if "cost_delta" in item]
        latency_deltas = [float(item["latency_delta"]) for item in comparisons if "latency_delta" in item]
        accepted_rows = [row for row in rows if row.get("accepted")]
        state = _context_lifecycle_state(
            key=key,
            rows=rows,
            suitability=0.0,
            evidence=[],
        ).state
        reason = _context_summary_reason(state)
        summaries[context_id] = {
            "key": key.to_dict(),
            "state": state,
            "proposed_count": len(rows),
            "evaluated_count": len(rows),
            "accepted_count": len(accepted_rows),
            "rejected_count": max(len(rows) - len(accepted_rows), 0),
            "best_score_delta": max(score_deltas) if score_deltas else None,
            "best_cost_delta": min(cost_deltas) if cost_deltas else None,
            "best_latency_delta": min(latency_deltas) if latency_deltas else None,
            "reason": reason,
        }
    return summaries


def summarize_surface_opportunity_results(proposals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in proposals:
        candidate = row.get("proposal_candidate") if isinstance(row.get("proposal_candidate"), dict) else {}
        if not candidate:
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        applications = row.get("applications") or candidate.get("applications") if isinstance(candidate, dict) else []
        if not isinstance(applications, list):
            continue
        for application in applications:
            if not isinstance(application, dict):
                continue
            surface_opportunity_id = str(application.get("surface_opportunity_id") or "")
            if surface_opportunity_id:
                grouped[surface_opportunity_id].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for surface_opportunity_id, rows in sorted(grouped.items()):
        evaluated_rows = [row for row in rows if "accepted" in row]
        valid_rows = [row for row in rows if row.get("valid") is not False]
        accepted_rows = [row for row in evaluated_rows if row.get("accepted")]
        comparisons = [row.get("comparison_to_parent") or {} for row in evaluated_rows]
        score_deltas = [float(item["score_delta"]) for item in comparisons if "score_delta" in item]
        cost_deltas = [float(item["cost_delta"]) for item in comparisons if "cost_delta" in item]
        latency_deltas = [float(item["latency_delta"]) for item in comparisons if "latency_delta" in item]
        invalid_reasons = Counter(
            str(row.get("invalid_reason"))
            for row in rows
            if row.get("valid") is False and row.get("invalid_reason")
        )
        if accepted_rows:
            state = "promotable_dev"
            reason = "At least one application of this surface opportunity earned finalist eligibility on dev."
        elif score_deltas and any(delta < 0 for delta in score_deltas):
            state = "constrained"
            reason = "At least one evaluated application regressed score."
        elif len(evaluated_rows) >= 2:
            state = "constrained"
            reason = "Multiple evaluated applications failed the configured objective gate."
        elif evaluated_rows:
            state = "paused"
            reason = "The evaluated application did not improve the configured objective."
        elif invalid_reasons:
            state = "invalid"
            reason = "No application reached evaluation because implementation validation failed."
        else:
            state = "proposed"
            reason = "Surface opportunity was proposed but not evaluated."
        summaries[surface_opportunity_id] = {
            "surface_opportunity_id": surface_opportunity_id,
            "family": _surface_opportunity_id_part(surface_opportunity_id, 0),
            "mechanism": _surface_opportunity_id_part(surface_opportunity_id, 1),
            "state": state,
            "proposed_count": len(rows),
            "valid_count": len(valid_rows),
            "evaluated_count": len(evaluated_rows),
            "accepted_count": len(accepted_rows),
            "rejected_count": max(len(evaluated_rows) - len(accepted_rows), 0),
            "invalid_count": max(len(rows) - len(valid_rows), 0),
            "best_score_delta": max(score_deltas) if score_deltas else None,
            "best_cost_delta": min(cost_deltas) if cost_deltas else None,
            "best_latency_delta": min(latency_deltas) if latency_deltas else None,
            "invalid_reasons": dict(sorted(invalid_reasons.items())),
            "candidate_ids": [
                str(row.get("candidate_id"))
                for row in evaluated_rows
                if row.get("candidate_id")
            ][:8],
            "reason": reason,
        }
    return summaries


def _surface_opportunity_id_part(surface_opportunity_id: str, index: int) -> str:
    parts = surface_opportunity_id.split(".")
    return parts[index] if len(parts) > index else ""


def observe_transform_result(
    *,
    family: str,
    context_key: TransformContextKey | None = None,
    accepted: bool,
    comparison: Comparison,
    rejection_reason: str | None,
) -> dict[str, Any]:
    if accepted:
        state = "promotable_dev"
        reason = "Candidate earned finalist eligibility on dev."
    elif comparison.score_delta < 0:
        state = "constrained"
        reason = rejection_reason or "Candidate regressed score; future attempts should be materially distinct."
    else:
        state = "paused"
        reason = rejection_reason or "Candidate did not improve the configured objective."
    return {
        "type": "transform_observation",
        "surface_mechanism": family,
        "transform_context": context_key.to_dict() if context_key else None,
        "state": state,
        "reason": reason,
        "comparison_to_parent": comparison.to_dict(),
    }
