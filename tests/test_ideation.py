from __future__ import annotations

from ratchet.ideation import build_ideation_metrics


def test_counts_valid_current_proposal_rows_without_type_marker() -> None:
    metrics = build_ideation_metrics(
        events=[
            {
                "type": "search_plan",
                "search_plan": {
                    "briefs": [
                    {
                        "brief_id": "brief_1",
                        "mechanism_class": "surface_context",
                        "surface_opportunity_ids": ["surface.surface_context.system_prompt"],
                    }
                    ]
                },
            }
        ],
        proposals=[
            {
                "proposal_candidate": {"experiment_id": "brief_1"},
                "compiled_candidate": {"program": {"candidate_id": "compiled"}},
                "candidate": {"program": {"candidate_id": "compiled"}},
                "surface_mechanism": "surface_context",
                "mechanism_class": "surface_context",
                "accepted": True,
                "frontier_status": "promotable",
            },
            {
                "type": "candidate_proposal",
                "valid": False,
                "invalid_reason": "unknown surface_opportunity_id",
            },
        ],
        finalist_statuses=[],
    )

    assert metrics["implementer"]["raw_candidate_count"] == 2
    assert metrics["implementer"]["valid_candidate_count"] == 1
    assert metrics["implementer"]["invalid_candidate_count"] == 1
    assert metrics["implementer"]["implemented_brief_count"] == 1
    assert metrics["discovery"]["stage_counts"]["promotable_dev"] == 1


def test_failed_smoke_candidate_reports_screened_at_smoke() -> None:
    metrics = build_ideation_metrics(
        events=[
            {
                "type": "search_plan",
                "search_plan": {"briefs": [{"brief_id": "B1", "mechanism_class": "surface_context"}]},
            }
        ],
        proposals=[
            {
                "type": "candidate_proposal",
                "valid": True,
                "experiment_id": "B1",
                "mechanism_class": "surface_context",
                "surface_mechanism": "surface_context",
                "accepted": False,
                "frontier_status": "failed",
                "evaluation_stages": [{"stage": "smoke"}],
            }
        ],
        finalist_statuses=[],
    )

    assert metrics["discovery"]["stage_counts"] == {"screened_at_smoke": 1}
    assert metrics["discovery"]["by_brief"]["B1"]["best_stage"] == "screened_at_smoke"


def test_unreturned_planned_intent_reports_planned_not_attempted() -> None:
    metrics = build_ideation_metrics(
        events=[
            {
                "type": "search_plan",
                "search_plan": {"briefs": [{"brief_id": "B1", "mechanism_class": "surface_tool_loop"}]},
            }
        ],
        proposals=[],
        finalist_statuses=[],
    )

    assert metrics["discovery"]["by_brief"]["B1"]["best_stage"] == "planned_not_attempted"
