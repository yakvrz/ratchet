"""Ratchet: attachable agent optimizer."""

from ratchet.adapters import AdapterProtocol, load_adapter
from ratchet.config import RatchetConfigError
from ratchet.evidence import ProposalExample, ProposalExampleBank, build_behavior_diagnostics, build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentSpec, TaskTheory, build_task_theory
from ratchet.grading import exact_text_grade, json_field_grade, numeric_tolerance_grade
from ratchet.objectives import FinalGateResult, GatePredicate, final_gate_status, select_recommended_patch
from ratchet.optimizer import RatchetOptimizer
from ratchet.pricing import estimate_cost_usd
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
    "DiagnosticTrace",
    "EditableTarget",
    "EvalCase",
    "FailureDiagnosis",
    "FinalGateResult",
    "GatePredicate",
    "GradeResult",
    "ExperimentSpec",
    "OptimizationConstraints",
    "OptimizationObjective",
    "OptimizerModelError",
    "OperationalMetrics",
    "PatchOperation",
    "ProposalExample",
    "ProposalExampleBank",
    "RatchetOptimizer",
    "RatchetConfigError",
    "RunRecord",
    "TaskTheory",
    "build_behavior_diagnostics",
    "build_proposal_example_bank",
    "build_task_theory",
    "exact_text_grade",
    "estimate_cost_usd",
    "final_gate_status",
    "json_field_grade",
    "load_adapter",
    "numeric_tolerance_grade",
    "select_recommended_patch",
]
