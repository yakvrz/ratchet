"""Ratchet: attachable agent optimizer."""

from ratchet.adapters import AdapterProtocol, load_adapter
from ratchet.adapter_generation import AdapterGenerator, GeneratedSingleCallAdapter, ModelRequest
from ratchet.candidates import CandidateProposal, CandidateSurfaceApplication, Intervention
from ratchet.surface_opportunities import SurfaceOpportunity, generate_surface_opportunities
from ratchet.config import RatchetConfigError
from ratchet.evidence import ProposalExample, ProposalExampleBank, build_behavior_diagnostics, build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.experiments import ExperimentSpec, SearchBrief, SearchPlan
from ratchet.grading import exact_text_grade, json_field_grade, numeric_tolerance_grade
from ratchet.ideation_benchmark import IdeationAssessmentSpec, assess_ideation_run
from ratchet.interactive import InteractionRecorder
from ratchet.objectives import FinalGateResult, GatePredicate, final_gate_status, select_recommended_candidate
from ratchet.optimizer import RatchetOptimizer
from ratchet.pricing import estimate_cost_usd
from ratchet.proposals import CandidateImplementer
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import CompiledCandidate, TransformProgram
from ratchet.tool_loop import GeneratedToolLoopAdapter, ToolLoopModelResponse, ToolLoopRunConfig
from ratchet.types import (
    AgentSpec,
    AgentTool,
    DiagnosticTrace,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OptimizationConstraints,
    OptimizationObjective,
    OperationalMetrics,
    InteractionTurn,
    RunRecord,
    TargetSemantics,
    ToolCallTrace,
)

__all__ = [
    "AdapterProtocol",
    "AdapterGenerator",
    "AgentSpec",
    "AgentTool",
    "CandidateProposal",
    "CandidateSurfaceApplication",
    "DiagnosticTrace",
    "EvalCase",
    "FailureDiagnosis",
    "FinalGateResult",
    "GatePredicate",
    "GradeResult",
    "GeneratedSingleCallAdapter",
    "GeneratedToolLoopAdapter",
    "IdeationAssessmentSpec",
    "Intervention",
    "InteractionRecorder",
    "ExperimentSpec",
    "OptimizationConstraints",
    "SurfaceOpportunity",
    "OptimizationObjective",
    "OptimizerModelError",
    "OperationalMetrics",
    "InteractionTurn",
    "ProposalExample",
    "ProposalExampleBank",
    "RatchetOptimizer",
    "RatchetConfigError",
    "SearchBrief",
    "SearchPlan",
    "RunRecord",
    "SurfaceSpec",
    "TargetSemantics",
    "ToolCallTrace",
    "ToolLoopModelResponse",
    "ToolLoopRunConfig",
    "CandidateImplementer",
    "CompiledCandidate",
    "ModelRequest",
    "TransformCompiler",
    "TransformProgram",
    "build_behavior_diagnostics",
    "build_proposal_example_bank",
    "assess_ideation_run",
    "exact_text_grade",
    "estimate_cost_usd",
    "final_gate_status",
    "generate_surface_opportunities",
    "json_field_grade",
    "load_adapter",
    "numeric_tolerance_grade",
    "select_recommended_candidate",
]
