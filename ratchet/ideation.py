from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


DISCOVERY_STAGES = {
    "no_intent",
    "no_valid_implementation",
    "planned_not_attempted",
    "screened_at_smoke",
    "lost_at_small_dev",
    "failed_full_dev",
    "failed_confirmation",
    "unstable_confirmation",
    "failed_holdout",
    "validated",
    "directional_holdout",
    "promotable_dev",
}


def build_ideation_metrics(
    *,
    decision_log: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    finalist_statuses: list[dict[str, Any]],
) -> dict[str, Any]:
    plans = [row for row in decision_log if row.get("type") == "research_plan"]
    intents = [
        intent
        for plan in plans
        for intent in plan.get("experiment_intents", [])
        if isinstance(intent, dict)
    ]
    proposal_rows = [row for row in proposals if _is_candidate_row(row)]
    valid_rows = [row for row in proposal_rows if row.get("valid") is not False]
    evaluated_rows = [row for row in proposal_rows if "accepted" in row]
    invalid_rows = [row for row in proposal_rows if row.get("valid") is False]
    stage_counts = Counter(_discovery_stage(row) for row in evaluated_rows)
    finalist_counts = Counter(str(row.get("status") or "unknown") for row in finalist_statuses)
    for row in finalist_statuses:
        status = str(row.get("status") or "")
        if status == "validated":
            stage_counts["validated"] += 1
        elif status == "directional":
            stage_counts["directional_holdout"] += 1
        elif status == "failed":
            stage_counts["failed_holdout"] += 1
        elif status == "unstable":
            stage_counts["unstable_confirmation"] += 1
    intent_ids = {str(intent.get("intent_id")) for intent in intents if intent.get("intent_id")}
    implemented_intent_ids = {
        str(_proposal_candidate(row).get("experiment_id") or row.get("experiment_id"))
        for row in valid_rows
        if _proposal_candidate(row).get("experiment_id") or row.get("experiment_id")
    }
    mechanism_counts = Counter(str(row.get("mechanism_class") or "unknown") for row in valid_rows)
    family_counts = Counter(str(row.get("surface_mechanism") or "unknown") for row in valid_rows)
    invalid_reasons = Counter(str(row.get("reason") or row.get("invalid_reason") or "unknown") for row in invalid_rows)
    by_intent: dict[str, dict[str, Any]] = {}
    for intent_id in sorted(intent_ids):
        rows = [
            row
            for row in proposal_rows
            if str(_proposal_candidate(row).get("experiment_id") or row.get("experiment_id")) == intent_id
        ]
        by_intent[intent_id] = {
            "candidate_count": len(rows),
            "valid_candidate_count": sum(1 for row in rows if row.get("valid") is not False),
            "evaluated_candidate_count": sum(1 for row in rows if "accepted" in row),
            "best_stage": _best_intent_stage(rows),
        }
    return {
        "planner": {
            "plan_count": len(plans),
            "intent_count": len(intents),
            "intent_mechanisms": dict(Counter(str(intent.get("mechanism_class") or "unknown") for intent in intents)),
            "intent_with_surface_opportunity_ids": sum(1 for intent in intents if intent.get("surface_opportunity_ids")),
        },
        "implementer": {
            "raw_candidate_count": len(proposal_rows),
            "valid_candidate_count": len(valid_rows),
            "invalid_candidate_count": len(invalid_rows),
            "valid_implementation_rate": (len(valid_rows) / len(proposal_rows) if proposal_rows else 0.0),
            "implemented_intent_count": len(intent_ids & implemented_intent_ids),
            "missing_intent_count": len(intent_ids - implemented_intent_ids),
            "candidate_mechanisms": dict(sorted(mechanism_counts.items())),
            "candidate_families": dict(sorted(family_counts.items())),
            "invalid_reasons": dict(invalid_reasons.most_common(12)),
        },
        "discovery": {
            "stage_counts": dict(sorted(stage_counts.items())),
            "finalist_status_counts": dict(sorted(finalist_counts.items())),
            "by_intent": by_intent,
        },
    }


def _discovery_stage(row: dict[str, Any]) -> str:
    if row.get("accepted"):
        return "promotable_dev"
    if row.get("frontier_status") in {"screened_out", "failed"}:
        stages = [stage.get("stage") for stage in row.get("evaluation_stages", []) if isinstance(stage, dict)]
        if "small_dev" in stages:
            return "lost_at_small_dev"
        return "screened_at_smoke"
    if row.get("full_dev_evaluated"):
        return "failed_full_dev"
    return "no_valid_implementation"


def _is_candidate_row(row: dict[str, Any]) -> bool:
    if row.get("type") == "candidate_proposal":
        return True
    return bool(
        (row.get("proposal_candidate") or row.get("compiled_candidate") or row.get("candidate"))
        and (row.get("surface_mechanism") or row.get("mechanism_class"))
    )


def _proposal_candidate(row: dict[str, Any]) -> dict[str, Any]:
    candidate = row.get("proposal_candidate")
    if isinstance(candidate, dict):
        return candidate
    candidate = row.get("candidate")
    return candidate if isinstance(candidate, dict) else {}


def _best_intent_stage(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "planned_not_attempted"
    ordered = [
        "validated",
        "directional_holdout",
        "unstable_confirmation",
        "promotable_dev",
        "failed_full_dev",
        "lost_at_small_dev",
        "screened_at_smoke",
        "no_valid_implementation",
    ]
    stages = {_discovery_stage(row) for row in rows if "accepted" in row}
    if not stages:
        if any(row.get("valid") is not False for row in rows):
            return "no_valid_implementation"
        return "no_valid_implementation"
    for stage in ordered:
        if stage in stages:
            return stage
    return "no_valid_implementation"
