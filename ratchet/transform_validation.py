from __future__ import annotations

from ratchet.candidates import CandidateProposal
from ratchet.experiments import CANDIDATE_ROLES
from ratchet.surface_search import SearchHypothesis, TransformContextKey, _operation_context_error
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_program import TransformProgram


def validate_candidate_transform(
    candidate: CandidateProposal,
    *,
    surface: SurfaceSpec,
    search_hypothesis: SearchHypothesis | None = None,
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
    if search_hypothesis is not None:
        eligibility_error = validate_candidate_context(candidate, search_hypothesis=search_hypothesis, surface=surface)
        if eligibility_error is not None:
            return eligibility_error
    return None


def _has_behavioral_patch(program: TransformProgram) -> bool:
    instrumentation_ops = {"log_event", "trace_annotation", "metric_counter", "capture_snapshot"}
    return any(patch.op.op not in instrumentation_ops for patch in program.patches)


def validate_candidate_context(
    candidate: CandidateProposal,
    *,
    search_hypothesis: SearchHypothesis,
    surface: SurfaceSpec | None = None,
) -> str | None:
    for family_name in sorted({application.family for application in candidate.applications}):
        family_state = search_hypothesis.mechanism_states.get(family_name)
        if family_state is None or family_name not in search_hypothesis.active_mechanisms:
            return f"inactive surface mechanism {family_name!r}"
    combined_key = TransformContextKey.from_candidate(candidate)
    exact_state = search_hypothesis.context_states.get(combined_key.id)
    if exact_state is not None and exact_state.state in {"paused", "available"}:
        return f"inactive transform context {combined_key.id!r}"
    if exact_state is not None and exact_state.state == "constrained":
        return f"constrained transform context {combined_key.id!r} requires a materially distinct mechanism"
    for application in candidate.applications:
        if application.selection:
            operation_key = TransformContextKey(
                family=application.family,
                target_names=("proposal_selected_examples",),
                ops=("add_context_section",),
                target_slice=candidate.target_slice,
                mechanism=(application.mechanism,),
                transform_instance=application.rationale or candidate.hypothesis or "candidate",
            )
            operation_error = _operation_context_error(operation_key, search_hypothesis)
            if operation_error is not None:
                return operation_error
            continue
    return None
