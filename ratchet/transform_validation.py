from __future__ import annotations

from ratchet.candidates import CandidateProposal
from ratchet.experiments import CANDIDATE_ROLES
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_program import TransformProgram


def validate_candidate_transform(
    candidate: CandidateProposal,
    *,
    surface: SurfaceSpec,
) -> str | None:
    if not candidate.experiment_id:
        return "candidate must belong to an experiment"
    if candidate.candidate_role not in CANDIDATE_ROLES:
        return f"unknown candidate role {candidate.candidate_role!r}"
    if candidate.candidate_role == "control":
        return "control candidates are measurement infrastructure, not optimizer candidates"
    if not _has_behavioral_patch(candidate.program):
        return "candidate program must include at least one non-instrumentation transform operation"
    if not candidate.applications:
        return "candidate must include at least one surface opportunity application"
    for application in candidate.applications:
        if application.surface_opportunity_id.startswith("surface."):
            continue
        return f"candidate application must cite a surface_opportunity_id, got {application.surface_opportunity_id!r}"
    return None


def _has_behavioral_patch(program: TransformProgram) -> bool:
    instrumentation_ops = {"log_event", "trace_annotation", "metric_counter", "capture_snapshot"}
    return any(patch.op.op not in instrumentation_ops for patch in program.patches)
