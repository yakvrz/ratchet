from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ratchet.transform_program import TransformProgram


@dataclass(frozen=True)
class Intervention:
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "payload": dict(self.payload)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Intervention":
        return cls(
            kind=str(payload.get("kind", "")),
            payload=dict(payload.get("payload", {})),
        )


@dataclass(frozen=True)
class CandidateSurfaceApplication:
    surface_opportunity_id: str
    selection: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    @property
    def family(self) -> str:
        if self.surface_opportunity_id.startswith("surface."):
            return self.mechanism
        parts = self.surface_opportunity_id.split(".")
        return parts[0] if parts else ""

    @property
    def mechanism(self) -> str:
        if self.surface_opportunity_id.startswith("surface."):
            parts = self.surface_opportunity_id.split(".")
            return parts[1] if len(parts) > 1 else "surface"
        parts = self.surface_opportunity_id.split(".")
        return parts[1] if len(parts) > 1 else ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_opportunity_id": self.surface_opportunity_id,
            "surface": self.mechanism,
            "selection": dict(self.selection),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateSurfaceApplication":
        surface_opportunity_id = str(payload.get("surface_opportunity_id") or "")
        if not surface_opportunity_id:
            raise ValueError("application requires surface_opportunity_id")
        if isinstance(payload.get("operation"), dict):
            raise ValueError("applications cite surface opportunities; transform edits belong in program.patches")
        raw_selection = payload.get("selection")
        selection = dict(raw_selection) if isinstance(raw_selection, dict) else {}
        return cls(
            surface_opportunity_id=surface_opportunity_id,
            selection=selection,
            rationale=str(payload.get("rationale") or ""),
        )


@dataclass(frozen=True)
class CandidateProposal:
    program: TransformProgram
    applications: list[CandidateSurfaceApplication]
    experiment_id: str = ""
    candidate_role: str = "atomic"
    comparison_group: str = ""
    target_slice: str = "global"
    hypothesis: str = ""
    expected_effects: dict[str, Any] = field(default_factory=dict)
    evaluation_plan: str = "full_dev"

    @property
    def surface_opportunity_ids(self) -> list[str]:
        return [application.surface_opportunity_id for application in self.applications]

    @property
    def surface_mechanism(self) -> str:
        return self.applications[0].family if self.applications else ""

    @property
    def mechanism_class(self) -> str:
        return self.applications[0].mechanism if self.applications else ""

    @property
    def transform_instance(self) -> str:
        return "; ".join(application.rationale for application in self.applications if application.rationale)[:240]

    @property
    def transform_parameters(self) -> dict[str, Any]:
        source_ids: list[str] = []
        strategies: list[str] = []
        for application in self.applications:
            raw_ids = application.selection.get("source_case_ids")
            if isinstance(raw_ids, list):
                source_ids.extend(str(item) for item in raw_ids if isinstance(item, str) and item)
            if application.selection.get("selection_strategy"):
                strategies.append(str(application.selection["selection_strategy"]))
        row: dict[str, Any] = {}
        if source_ids:
            row["source_case_ids"] = source_ids
        if strategies:
            row["selection_strategies"] = sorted(set(strategies))
        if "few_shot_example_count" in self.program.metadata:
            row["few_shot_example_count"] = self.program.metadata["few_shot_example_count"]
        return row

    @property
    def intervention(self) -> Intervention:
        if self.program.patches:
            return Intervention(kind="transform_program", payload={"program": self.program.to_dict()})
        if self.applications and self.applications[0].selection:
            return Intervention(kind="example_selection", payload=dict(self.applications[0].selection))
        return Intervention(kind="transform_program", payload={"program": self.program.to_dict()})

    def to_dict(self) -> dict[str, Any]:
        from ratchet.surface_search import TransformContextKey

        return {
            "surface_mechanism": self.surface_mechanism,
            "intervention": self.intervention.to_dict(),
            "transform_instance": self.transform_instance,
            "transform_parameters": dict(self.transform_parameters),
            "mechanism_class": self.mechanism_class,
            "experiment_id": self.experiment_id,
            "candidate_role": self.candidate_role,
            "comparison_group": self.comparison_group,
            "surface_opportunity_ids": list(self.surface_opportunity_ids),
            "target_slice": self.target_slice,
            "transform_context": TransformContextKey.from_candidate(self).to_dict(),
            "hypothesis": self.hypothesis,
            "expected_effects": dict(self.expected_effects),
            "evaluation_plan": self.evaluation_plan,
            "applications": [application.to_dict() for application in self.applications],
            "program": self.program.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateProposal":
        if "patch" in payload:
            raise ValueError("candidate must emit a typed transform program, not an untyped patch")
        if payload.get("transform_parameters"):
            raise ValueError("candidate transform_parameters are derived; put candidate-specific data in applications[]")
        if payload.get("surface_mechanism") or payload.get("mechanism_class"):
            raise ValueError("candidate must cite applications[]; family and mechanism are derived")
        raw_program = payload.get("program") or payload.get("transform_program")
        if not isinstance(raw_program, dict):
            raw_patches = payload.get("patches")
            if isinstance(raw_patches, list):
                raw_program = {
                    "id": str(payload.get("candidate_id") or payload.get("experiment_id") or ""),
                    "hypothesis_id": str(payload.get("hypothesis_id") or ""),
                    "patches": raw_patches,
                    "metadata": dict(payload.get("metadata") or {}),
                }
        if not isinstance(raw_program, dict):
            raise ValueError("candidate requires program or patches[]")
        program_payload = dict(raw_program)
        if not program_payload.get("id") and not program_payload.get("candidate_id"):
            program_payload["id"] = str(payload.get("experiment_id") or payload.get("candidate_id") or "")
        metadata = {
            **dict(program_payload.get("metadata") or {}),
            "hypothesis": str(payload.get("hypothesis", "")),
            "expected_effects": dict(payload.get("expected_effects", {})),
        }
        program_payload["metadata"] = metadata
        program = TransformProgram.from_dict(program_payload)
        raw_applications = payload.get("applications")
        if not isinstance(raw_applications, list) or not raw_applications:
            raise ValueError("candidate requires non-empty applications[]")
        applications = [
            CandidateSurfaceApplication.from_dict(application)
            for application in raw_applications
            if isinstance(application, dict)
        ]
        if len(applications) != len(raw_applications):
            raise ValueError("candidate applications must be objects")
        return cls(
            program=program,
            applications=applications,
            experiment_id=str(payload.get("experiment_id", "")),
            candidate_role=str(payload.get("candidate_role", "atomic") or "atomic"),
            comparison_group=str(payload.get("comparison_group", "")),
            target_slice=str(payload.get("target_slice", "global") or "global"),
            hypothesis=str(payload.get("hypothesis", "")),
            expected_effects=dict(payload.get("expected_effects", {})),
            evaluation_plan=str(payload.get("evaluation_plan", "full_dev") or "full_dev"),
        )
