from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IdeationAssessmentSpec:
    task_id: str = ""
    mechanisms_of_interest: list[str] = field(default_factory=list)
    pivotal_mechanisms: list[str] = field(default_factory=list)
    min_valid_implementation_rate: float = 0.0
    min_holdout_score_delta: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "IdeationAssessmentSpec":
        frontier = payload.get("frontier_quality") or {}
        return cls(
            task_id=str(payload.get("task_id") or ""),
            mechanisms_of_interest=[str(item) for item in payload.get("mechanisms_of_interest", []) if item],
            pivotal_mechanisms=[str(item) for item in payload.get("pivotal_mechanisms", []) if item],
            min_valid_implementation_rate=float(payload.get("min_valid_implementation_rate") or 0.0),
            min_holdout_score_delta=(
                float(frontier["min_holdout_score_delta"])
                if isinstance(frontier, dict) and frontier.get("min_holdout_score_delta") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_ideation_assessment_spec(path: Path | str | None) -> IdeationAssessmentSpec:
    if path is None:
        return IdeationAssessmentSpec()
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("Ideation assessment spec must be a JSON object.")
    return IdeationAssessmentSpec.from_dict(payload)


def assess_ideation_run(
    run_dir: Path | str,
    *,
    spec: IdeationAssessmentSpec | None = None,
) -> dict[str, Any]:
    root = Path(run_dir)
    spec = spec or IdeationAssessmentSpec()
    manifest = _read_json(root / "run_manifest.json")
    candidate_metrics = _read_json(root / "candidate_metrics.json")
    ideation_metrics = _read_json(root / "ideation_metrics.json")
    proposals = _read_jsonl(root / "proposals.jsonl")
    search_plans = _read_jsonl(root / "search_plans.jsonl")

    candidate_rows = [row for row in proposals if _is_candidate_row(row)]
    valid_rows = [row for row in candidate_rows if row.get("valid") is not False]
    evaluated_rows = [row for row in candidate_rows if "accepted" in row]
    full_dev_rows = [row for row in evaluated_rows if row.get("full_dev_evaluated")]
    finalist_statuses = manifest.get("finalist_statuses") or candidate_metrics.get("finalist_statuses") or []
    holdout_hashes = {str(row.get("candidate_id")) for row in finalist_statuses if row.get("candidate_id")}
    validated_hashes = {
        str(row.get("candidate_id"))
        for row in finalist_statuses
        if str(row.get("status") or "") == "validated" and row.get("candidate_id")
    }
    candidate_mechanisms = Counter(str(row.get("mechanism_class") or "unknown") for row in valid_rows)
    opportunity_mechanisms = _opportunity_mechanisms(search_plans)
    expected_mechanisms = set(spec.mechanisms_of_interest or sorted(opportunity_mechanisms))
    pivotal_mechanisms = set(spec.pivotal_mechanisms)
    mechanisms_discovered = set(candidate_mechanisms)
    pivotal_candidate_hashes = {
        str(row.get("candidate_id"))
        for row in valid_rows
        if str(row.get("mechanism_class") or "") in pivotal_mechanisms and row.get("candidate_id")
    }
    best_dev_delta = max((_score_delta(row) for row in evaluated_rows), default=0.0)
    holdout_delta = _holdout_delta(candidate_metrics)
    valid_rate = float((ideation_metrics.get("implementer") or {}).get("valid_implementation_rate") or 0.0)
    checks = {
        "intent_relevance": not expected_mechanisms or bool(expected_mechanisms & set((ideation_metrics.get("planner") or {}).get("brief_mechanisms") or {})),
        "valid_implementation_rate": valid_rate >= spec.min_valid_implementation_rate,
        "pivotal_mechanism_discovered": not pivotal_mechanisms or bool(pivotal_candidate_hashes),
        "pivotal_reached_full_dev": not pivotal_mechanisms
        or bool(pivotal_candidate_hashes & {str(row.get("candidate_id")) for row in full_dev_rows}),
        "pivotal_reached_holdout": not pivotal_mechanisms or bool(pivotal_candidate_hashes & holdout_hashes),
        "pivotal_validated": not pivotal_mechanisms or bool(pivotal_candidate_hashes & validated_hashes),
        "holdout_frontier_quality": spec.min_holdout_score_delta is None or holdout_delta >= spec.min_holdout_score_delta,
    }
    return {
        "task_id": spec.task_id,
        "run_dir": str(root),
        "selected_candidate_id": manifest.get("selected_candidate_id"),
        "promoted": bool(manifest.get("promoted")),
        "checks": checks,
        "summary": {
            "passed_checks": sum(1 for value in checks.values() if value),
            "total_checks": len(checks),
            "valid_implementation_rate": valid_rate,
            "raw_candidate_count": (ideation_metrics.get("implementer") or {}).get("raw_candidate_count", len(candidate_rows)),
            "valid_candidate_count": (ideation_metrics.get("implementer") or {}).get("valid_candidate_count", len(valid_rows)),
            "full_dev_candidate_count": len(full_dev_rows),
            "holdout_candidate_count": len(holdout_hashes),
            "validated_candidate_count": len(validated_hashes),
            "best_dev_score_delta": best_dev_delta,
            "selected_holdout_score_delta": holdout_delta,
        },
        "mechanisms": {
            "expected": sorted(expected_mechanisms),
            "opportunities": sorted(opportunity_mechanisms),
            "implemented": dict(sorted(candidate_mechanisms.items())),
            "missing_expected": sorted(expected_mechanisms - mechanisms_discovered),
            "pivotal": sorted(pivotal_mechanisms),
        },
        "cost": (manifest.get("run_cost") or candidate_metrics.get("run_cost") or {}),
        "ideation_metrics": ideation_metrics,
    }


def write_ideation_assessment(
    run_dir: Path | str,
    *,
    spec_path: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    spec = load_ideation_assessment_spec(spec_path)
    assessment = assess_ideation_run(run_dir, spec=spec)
    path = Path(out_path) if out_path is not None else Path(run_dir) / "ideation_assessment.json"
    path.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n")
    return assessment


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _is_candidate_row(row: dict[str, Any]) -> bool:
    if row.get("type") == "candidate_proposal":
        return True
    return bool(
        (row.get("proposal_candidate") or row.get("compiled_candidate") or row.get("candidate"))
        and (row.get("surface_mechanism") or row.get("mechanism_class"))
    )


def _opportunity_mechanisms(search_plans: list[dict[str, Any]]) -> set[str]:
    mechanisms: set[str] = set()
    for row in search_plans:
        plan = row.get("search_plan") if "search_plan" in row else row
        if not isinstance(plan, dict):
            continue
        for brief in plan.get("briefs") or []:
            if isinstance(brief, dict) and brief.get("mechanism_class"):
                mechanisms.add(str(brief["mechanism_class"]))
    return mechanisms


def _score_delta(row: dict[str, Any]) -> float:
    comparison = row.get("comparison_to_parent") or {}
    value = comparison.get("score_delta")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _holdout_delta(candidate_metrics: dict[str, Any]) -> float:
    baseline = ((candidate_metrics.get("baseline_holdout") or {}).get("behavioral") or {}).get("mean_score")
    selected = ((candidate_metrics.get("selected_holdout") or {}).get("behavioral") or {}).get("mean_score")
    try:
        return float(selected) - float(baseline)
    except (TypeError, ValueError):
        return 0.0
