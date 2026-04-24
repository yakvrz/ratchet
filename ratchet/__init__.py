"""Ratchet: attachable agent optimizer."""

from ratchet.adapters import AdapterProtocol, load_adapter
from ratchet.code_artifacts import CodeArtifactLoader, compile_code_artifact, default_hook_source
from ratchet.grading import exact_text_grade, json_field_grade, numeric_tolerance_grade
from ratchet.optimizer import RatchetOptimizer
from ratchet.types import (
    ArchitectureProposal,
    CodeArtifactSpec,
    ComponentSpec,
    DiagnosticTrace,
    EnumKnobSpec,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OperationalMetrics,
    PatchChange,
    PatchProposal,
    RunRecord,
    SearchSpace,
    TextArtifactSpec,
)

__all__ = [
    "AdapterProtocol",
    "ArchitectureProposal",
    "CodeArtifactLoader",
    "CodeArtifactSpec",
    "ComponentSpec",
    "compile_code_artifact",
    "default_hook_source",
    "DiagnosticTrace",
    "EnumKnobSpec",
    "EvalCase",
    "FailureDiagnosis",
    "GradeResult",
    "OperationalMetrics",
    "PatchChange",
    "PatchProposal",
    "RatchetOptimizer",
    "RunRecord",
    "SearchSpace",
    "TextArtifactSpec",
    "exact_text_grade",
    "json_field_grade",
    "load_adapter",
    "numeric_tolerance_grade",
]
