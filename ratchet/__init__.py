"""Ratchet: attachable agent optimizer."""

from ratchet.adapters import AdapterProtocol, load_adapter
from ratchet.affordances import OptimizationAffordance, generate_optimization_affordances
from ratchet.benchmarks import TauBenchRunner, taubench_result_to_run_record
from ratchet.config import RatchetConfigError
from ratchet.evidence import ProposalExample, ProposalExampleBank, build_behavior_diagnostics, build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.experiments import CandidateImplementation, ExperimentIntent, ExperimentSpec, MeasurementDecision, ResearchState, TaskTheory, build_task_theory
from ratchet.grading import exact_text_grade, json_field_grade, numeric_tolerance_grade
from ratchet.ideation_benchmark import IdeationAssessmentSpec, assess_ideation_run
from ratchet.interactive import InteractionRecorder
from ratchet.objectives import FinalGateResult, GatePredicate, final_gate_status, select_recommended_patch
from ratchet.optimizer import RatchetOptimizer
from ratchet.pricing import estimate_cost_usd
from ratchet.proposals import CandidateImplementer
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
    InteractionTurn,
    RunRecord,
    TargetSemantics,
    ToolCallTrace,
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
    "IdeationAssessmentSpec",
    "InteractionRecorder",
    "ExperimentSpec",
    "ExperimentIntent",
    "OptimizationConstraints",
    "OptimizationAffordance",
    "OptimizationObjective",
    "OptimizerModelError",
    "OperationalMetrics",
    "PatchOperation",
    "InteractionTurn",
    "ProposalExample",
    "ProposalExampleBank",
    "RatchetOptimizer",
    "RatchetConfigError",
    "ResearchState",
    "RunRecord",
    "TaskTheory",
    "TargetSemantics",
    "ToolCallTrace",
    "TauBenchRunner",
    "CandidateImplementation",
    "CandidateImplementer",
    "MeasurementDecision",
    "build_behavior_diagnostics",
    "build_proposal_example_bank",
    "build_task_theory",
    "assess_ideation_run",
    "exact_text_grade",
    "estimate_cost_usd",
    "final_gate_status",
    "generate_optimization_affordances",
    "json_field_grade",
    "load_adapter",
    "numeric_tolerance_grade",
    "select_recommended_patch",
    "taubench_result_to_run_record",
]
