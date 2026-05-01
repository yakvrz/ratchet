from __future__ import annotations

from typing import Any

from ratchet.experiments import EvidencePacket
from ratchet.surface_opportunities import SurfaceOpportunity
from ratchet.surfaces import SurfaceSpec


def top_counter_dict(values: dict[str, int], *, limit: int) -> dict[str, int]:
    return dict(sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit])


def truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def planner_evidence_packet(packet: EvidencePacket) -> dict[str, Any]:
    raw = packet.to_dict()
    diagnostics = raw.get("behavior_diagnostics") or {}
    runtime = raw.get("runtime_defects") or {}
    output = raw.get("output_defects") or {}
    tool = raw.get("tool_defects") or {}
    category_metrics = diagnostics.get("category_metrics") or {}
    per_label = diagnostics.get("per_label") or []
    return {
        "residual_failure_modes": list(raw.get("residual_failure_modes") or [])[:8],
        "diagnosis_categories": list(raw.get("diagnosis_categories") or [])[:8],
        "evidence": list(raw.get("evidence") or [])[:8],
        "confidence": raw.get("confidence"),
        "weak_slices": list(raw.get("weak_slices") or [])[:8],
        "label_confusions": [
            {
                "expected": row.get("expected"),
                "actual": row.get("actual"),
                "count": row.get("count"),
                "case_ids": list(row.get("case_ids") or [])[:3],
            }
            for row in list(raw.get("label_confusions") or [])[:8]
            if isinstance(row, dict)
        ],
        "runtime_defects": {
            "finish_reason_counts": runtime.get("finish_reason_counts", {}),
            "length_finish_case_ids": list(runtime.get("length_finish_case_ids") or [])[:8],
            "parser_fallback_case_ids": list(runtime.get("parser_fallback_case_ids") or [])[:8],
            "low_output_token_length_case_ids": list(runtime.get("low_output_token_length_case_ids") or [])[:8],
        },
        "output_defects": {
            "invalid_output_count": output.get("invalid_output_count", 0),
            "invalid_output_case_ids": list(output.get("invalid_output_case_ids") or [])[:10],
        },
        "tool_defects": {
            "tool_call_counts": tool.get("tool_call_counts", {}),
            "tool_status_counts": tool.get("tool_status_counts", {}),
            "turn_outcome_counts": tool.get("turn_outcome_counts", {}),
            "terminal_reason_counts": tool.get("terminal_reason_counts", {}),
            "tool_error_case_ids": list(tool.get("tool_error_case_ids") or [])[:8],
            "invalid_tool_call_case_ids": list(tool.get("invalid_tool_call_case_ids") or [])[:8],
            "premature_stop_case_ids": list(tool.get("premature_stop_case_ids") or [])[:8],
        },
        "example_coverage": raw.get("example_coverage") or {},
        "cost_latency_profile": raw.get("cost_latency_profile") or {},
        "behavior_summary": {
            "category_metrics": category_metrics,
            "weakest_labels": [
                {
                    "label": row.get("label"),
                    "support": row.get("support"),
                    "pass_rate": row.get("pass_rate"),
                    "case_ids": list(row.get("case_ids") or [])[:4],
                }
                for row in list(per_label)[:8]
                if isinstance(row, dict)
            ],
        },
    }


def planner_surface_opportunities(
    surface_opportunities: list[SurfaceOpportunity],
    *,
    limit: int = 36,
) -> list[dict[str, Any]]:
    ranked = sorted(surface_opportunities, key=lambda item: (-item.suitability, item.surface_opportunity_id))
    selected: list[SurfaceOpportunity] = []
    seen: set[str] = set()
    for key_fn, per_group in (
        (lambda item: item.mechanism, 2),
        (lambda item: item.family, 1),
    ):
        counts: dict[str, int] = {}
        for surface_opportunity in ranked:
            group = str(key_fn(surface_opportunity))
            if counts.get(group, 0) >= per_group or surface_opportunity.surface_opportunity_id in seen:
                continue
            selected.append(surface_opportunity)
            seen.add(surface_opportunity.surface_opportunity_id)
            counts[group] = counts.get(group, 0) + 1
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    for surface_opportunity in ranked:
        if len(selected) >= limit:
            break
        if surface_opportunity.surface_opportunity_id in seen:
            continue
        selected.append(surface_opportunity)
        seen.add(surface_opportunity.surface_opportunity_id)
    return [
        {
            "surface_opportunity_id": surface_opportunity.surface_opportunity_id,
            "surface": surface_opportunity.mechanism,
            "target": surface_opportunity.target_name,
            "target_kind": surface_opportunity.target_kind,
            "target_path": surface_opportunity.target_path,
            "ops": list(surface_opportunity.ops),
            "semantic_role": surface_opportunity.semantic_role,
            "behavioral_axes": list(surface_opportunity.behavioral_axes)[:4],
            "expected_scope": surface_opportunity.expected_scope,
            "risk": surface_opportunity.risk,
            "measurements": list(surface_opportunity.measurements)[:5],
            "suitability": surface_opportunity.suitability,
            "evidence": list(surface_opportunity.evidence)[:3],
            "expected_cost_impact": surface_opportunity.expected_cost_impact,
            "expected_latency_impact": surface_opportunity.expected_latency_impact,
        }
        for surface_opportunity in selected
    ]


def planner_surface_spec(surface: SurfaceSpec) -> dict[str, Any]:
    return {
        "agent_id": surface.agent_id,
        "context_sections": [
            {
                "name": section.name,
                "role": section.role,
                "required": section.required,
                "editable": section.name in surface.context.editable_sections,
                "value_shape": value_shape(section.content),
            }
            for section in surface.context.graph.sections
        ],
        "context_capabilities": {
            "generated_sections_allowed": surface.context.generated_sections_allowed,
            "removable_sections_allowed": surface.context.removable_sections_allowed,
            "reorderable_sections_allowed": surface.context.reorderable_sections_allowed,
        },
        "hooks": {
            name: {
                "available_inputs": list(hook.available_inputs),
                "allowed_ops": list(hook.allowed_ops),
                "method": hook.method,
            }
            for name, hook in sorted(surface.hooks.items())
            if hook.supported
        },
        "state": surface.state.to_dict(),
        "tools": surface.tools.to_dict(),
        "model": surface.model.to_dict(),
        "response": surface.response.to_dict(),
        "immutable_boundaries": list(surface.immutable_boundaries),
        "safety_constraints": list(surface.safety_constraints),
    }


def value_shape(value: Any) -> Any:
    if isinstance(value, str):
        return {"type": "string", "chars": len(value), "prefix": value[:240]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value), "sample": value[:3]}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value.keys())[:16]}
