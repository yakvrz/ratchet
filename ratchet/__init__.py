"""Ratchet: attachable agent optimizer."""

from ratchet.adapters import AdapterProtocol, load_adapter
from ratchet.evidence import ProposalExample, ProposalExampleBank, build_behavior_diagnostics, build_proposal_example_bank
from ratchet.grading import exact_text_grade, json_field_grade, numeric_tolerance_grade
from ratchet.objectives import FinalGateResult, final_gate_status
from ratchet.optimizer import RatchetOptimizer
from ratchet.pricing import estimate_cost_usd
from ratchet.transforms import (
    BehaviorProfile,
    CandidateProposal,
    SearchHypothesis,
    TransformContextKey,
    TransformContextState,
    TransformFamily,
    TransformFamilyState,
    transform_registry,
)
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    AgentTool,
    DiagnosticTrace,
    EditableTarget,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OptimizationConstraints,
    OptimizationObjective,
    OperationalMetrics,
    PatchOperation,
    RunRecord,
)

__all__ = [
    "AdapterProtocol",
    "AgentPatch",
    "AgentSpec",
    "AgentTool",
    "BehaviorProfile",
    "CandidateProposal",
    "DiagnosticTrace",
    "EditableTarget",
    "EvalCase",
    "FailureDiagnosis",
    "FinalGateResult",
    "GradeResult",
    "OptimizationConstraints",
    "OptimizationObjective",
    "OperationalMetrics",
    "PatchOperation",
    "ProposalExample",
    "ProposalExampleBank",
    "RatchetOptimizer",
    "RunRecord",
    "SearchHypothesis",
    "TransformContextKey",
    "TransformContextState",
    "TransformFamily",
    "TransformFamilyState",
    "build_behavior_diagnostics",
    "build_proposal_example_bank",
    "exact_text_grade",
    "estimate_cost_usd",
    "final_gate_status",
    "json_field_grade",
    "load_adapter",
    "numeric_tolerance_grade",
    "transform_registry",
]
