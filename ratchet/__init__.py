"""Ratchet: attachable agent optimizer."""

from ratchet.adapters import AdapterProtocol, load_adapter
from ratchet.grading import exact_text_grade, json_field_grade, numeric_tolerance_grade
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
    "GradeResult",
    "OptimizationConstraints",
    "OptimizationObjective",
    "OperationalMetrics",
    "PatchOperation",
    "RatchetOptimizer",
    "RunRecord",
    "exact_text_grade",
    "estimate_cost_usd",
    "json_field_grade",
    "load_adapter",
    "numeric_tolerance_grade",
]
