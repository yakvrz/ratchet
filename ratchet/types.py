from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DependencyMap = dict[str, list[str]]


@dataclass(frozen=True)
class EnumKnobSpec:
    name: str
    kind: str
    values: list[str]
    default: str
    depends_on: DependencyMap = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"model", "reasoning", "tool", "kb", "param"}:
            raise ValueError(f"Unsupported enum knob kind: {self.kind}")
        if not self.values:
            raise ValueError(f"Enum knob {self.name} must define at least one value.")
        if self.default not in self.values:
            raise ValueError(f"Enum knob {self.name} default {self.default!r} is not in values.")
        for dependency_name, allowed_values in self.depends_on.items():
            if not dependency_name:
                raise ValueError("Dependency names must be non-empty.")
            if not allowed_values:
                raise ValueError(
                    f"Dependency {dependency_name} for enum knob {self.name} must list values."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "enum"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EnumKnobSpec":
        depends_on = {
            key: [str(value) for value in values]
            for key, values in payload.get("depends_on", {}).items()
        }
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            values=[str(value) for value in payload["values"]],
            default=str(payload["default"]),
            depends_on=depends_on,
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class TextArtifactSpec:
    name: str
    kind: str
    default: str
    max_chars: int
    depends_on: DependencyMap = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"prompt", "tool", "component"}:
            raise ValueError(f"Unsupported text artifact kind: {self.kind}")
        if self.max_chars <= 0:
            raise ValueError(f"Text artifact {self.name} max_chars must be positive.")
        if len(self.default) > self.max_chars:
            raise ValueError(
                f"Text artifact {self.name} default exceeds max_chars ({self.max_chars})."
            )
        for dependency_name, allowed_values in self.depends_on.items():
            if not dependency_name:
                raise ValueError("Dependency names must be non-empty.")
            if not allowed_values:
                raise ValueError(
                    f"Dependency {dependency_name} for text artifact {self.name} must list values."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "text"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TextArtifactSpec":
        depends_on = {
            key: [str(value) for value in values]
            for key, values in payload.get("depends_on", {}).items()
        }
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            default=str(payload["default"]),
            max_chars=int(payload["max_chars"]),
            depends_on=depends_on,
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    kind: str
    values: list[str]
    default: str
    depends_on: DependencyMap = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("Component kind must be non-empty.")
        if not self.values:
            raise ValueError(f"Component {self.name} must define at least one value.")
        if self.default not in self.values:
            raise ValueError(f"Component {self.name} default {self.default!r} is not in values.")
        for dependency_name, allowed_values in self.depends_on.items():
            if not dependency_name:
                raise ValueError("Dependency names must be non-empty.")
            if not allowed_values:
                raise ValueError(
                    f"Dependency {dependency_name} for component {self.name} must list values."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "component"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ComponentSpec":
        depends_on = {
            key: [str(value) for value in values]
            for key, values in payload.get("depends_on", {}).items()
        }
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            values=[str(value) for value in payload["values"]],
            default=str(payload["default"]),
            depends_on=depends_on,
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class CodeArtifactSpec:
    name: str
    language: str
    callable_name: str
    signature: str
    default: str
    max_chars: int
    max_lines: int
    depends_on: DependencyMap = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        if self.language != "python":
            raise ValueError(f"Unsupported code artifact language: {self.language}")
        if not self.callable_name:
            raise ValueError("Code artifact callable_name must be non-empty.")
        if not self.signature:
            raise ValueError("Code artifact signature must be non-empty.")
        if self.max_chars <= 0:
            raise ValueError(f"Code artifact {self.name} max_chars must be positive.")
        if self.max_lines <= 0:
            raise ValueError(f"Code artifact {self.name} max_lines must be positive.")
        if len(self.default) > self.max_chars:
            raise ValueError(
                f"Code artifact {self.name} default exceeds max_chars ({self.max_chars})."
            )
        if len(self.default.splitlines()) > self.max_lines:
            raise ValueError(
                f"Code artifact {self.name} default exceeds max_lines ({self.max_lines})."
            )
        for dependency_name, allowed_values in self.depends_on.items():
            if not dependency_name:
                raise ValueError("Dependency names must be non-empty.")
            if not allowed_values:
                raise ValueError(
                    f"Dependency {dependency_name} for code artifact {self.name} must list values."
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["type"] = "code"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CodeArtifactSpec":
        depends_on = {
            key: [str(value) for value in values]
            for key, values in payload.get("depends_on", {}).items()
        }
        return cls(
            name=str(payload["name"]),
            language=str(payload["language"]),
            callable_name=str(payload["callable_name"]),
            signature=str(payload["signature"]),
            default=str(payload["default"]),
            max_chars=int(payload["max_chars"]),
            max_lines=int(payload["max_lines"]),
            depends_on=depends_on,
            description=str(payload.get("description", "")),
        )


@dataclass(frozen=True)
class SearchSpace:
    enum_knobs: list[EnumKnobSpec] = field(default_factory=list)
    text_artifacts: list[TextArtifactSpec] = field(default_factory=list)
    components: list[ComponentSpec] = field(default_factory=list)
    code_artifacts: list[CodeArtifactSpec] = field(default_factory=list)

    def all_specs(self) -> list[EnumKnobSpec | TextArtifactSpec | ComponentSpec | CodeArtifactSpec]:
        return [*self.enum_knobs, *self.text_artifacts, *self.components, *self.code_artifacts]

    def spec_names(self) -> list[str]:
        return [spec.name for spec in self.all_specs()]

    def enum_spec(self, name: str) -> EnumKnobSpec | None:
        for spec in self.enum_knobs:
            if spec.name == name:
                return spec
        return None

    def text_spec(self, name: str) -> TextArtifactSpec | None:
        for spec in self.text_artifacts:
            if spec.name == name:
                return spec
        return None

    def component_spec(self, name: str) -> ComponentSpec | None:
        for spec in self.components:
            if spec.name == name:
                return spec
        return None

    def code_spec(self, name: str) -> CodeArtifactSpec | None:
        for spec in self.code_artifacts:
            if spec.name == name:
                return spec
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "enum_knobs": [spec.to_dict() for spec in self.enum_knobs],
            "text_artifacts": [spec.to_dict() for spec in self.text_artifacts],
            "components": [spec.to_dict() for spec in self.components],
            "code_artifacts": [spec.to_dict() for spec in self.code_artifacts],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchSpace":
        return cls(
            enum_knobs=[EnumKnobSpec.from_dict(item) for item in payload.get("enum_knobs", [])],
            text_artifacts=[
                TextArtifactSpec.from_dict(item) for item in payload.get("text_artifacts", [])
            ],
            components=[ComponentSpec.from_dict(item) for item in payload.get("components", [])],
            code_artifacts=[
                CodeArtifactSpec.from_dict(item) for item in payload.get("code_artifacts", [])
            ],
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    split: str
    input: str
    expected: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in {"dev", "holdout"}:
            raise ValueError(f"Unsupported split: {self.split}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalCase":
        return cls(
            id=str(payload["id"]),
            split=str(payload["split"]),
            input=str(payload["input"]),
            expected=payload.get("expected"),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class OperationalMetrics:
    latency_s: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OperationalMetrics":
        return cls(
            latency_s=float(payload.get("latency_s", 0.0)),
            input_tokens=int(payload.get("input_tokens", 0)),
            output_tokens=int(payload.get("output_tokens", 0)),
            total_tokens=int(payload.get("total_tokens", 0)),
            cost_usd=float(payload.get("cost_usd", 0.0)),
            error=payload.get("error"),
        )


@dataclass(frozen=True)
class DiagnosticTrace:
    tool_calls: list[str] = field(default_factory=list)
    raw_output_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosticTrace":
        return cls(
            tool_calls=[str(item) for item in payload.get("tool_calls", [])],
            raw_output_text=str(payload.get("raw_output_text", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class RunRecord:
    output: Any
    metrics: OperationalMetrics
    diagnostics: DiagnosticTrace = field(default_factory=DiagnosticTrace)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": self.output,
            "metrics": self.metrics.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunRecord":
        return cls(
            output=payload.get("output"),
            metrics=OperationalMetrics.from_dict(payload.get("metrics", {})),
            diagnostics=DiagnosticTrace.from_dict(payload.get("diagnostics", {})),
        )


@dataclass(frozen=True)
class GradeResult:
    score: float
    passed: bool
    labels: list[str] = field(default_factory=list)
    notes: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Grade score must be between 0 and 1, got {self.score}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GradeResult":
        return cls(
            score=float(payload["score"]),
            passed=bool(payload["passed"]),
            labels=[str(label) for label in payload.get("labels", [])],
            notes=payload.get("notes"),
        )


@dataclass(frozen=True)
class FailureDiagnosis:
    case_ids: list[str]
    category: str
    root_cause: str
    target_keys: list[str]
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailureDiagnosis":
        return cls(
            case_ids=[str(item) for item in payload.get("case_ids", [])],
            category=str(payload["category"]),
            root_cause=str(payload["root_cause"]),
            target_keys=[str(item) for item in payload.get("target_keys", [])],
            evidence=[dict(item) for item in payload.get("evidence", [])],
        )


@dataclass(frozen=True)
class PatchChange:
    op: str
    name: str
    value: str

    def __post_init__(self) -> None:
        if self.op not in {"set_enum", "rewrite_text", "set_component", "rewrite_code"}:
            raise ValueError(f"Unsupported patch operation: {self.op}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchChange":
        if "new_text" in payload:
            value = payload["new_text"]
        else:
            value = payload.get("value")
        return cls(
            op=str(payload["op"]),
            name=str(payload["name"]),
            value=str(value),
        )


@dataclass(frozen=True)
class PatchProposal:
    proposal_id: str
    diagnosis_category: str
    changes: list[PatchChange]
    rationale: str
    expected_effect: str
    estimated_scope: str = "low"

    def __post_init__(self) -> None:
        if not self.changes:
            raise ValueError("PatchProposal must include at least one change.")
        if self.estimated_scope not in {"low", "medium", "high"}:
            raise ValueError(f"Unsupported proposal scope: {self.estimated_scope}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "diagnosis_category": self.diagnosis_category,
            "changes": [change.to_dict() for change in self.changes],
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "estimated_scope": self.estimated_scope,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PatchProposal":
        return cls(
            proposal_id=str(payload["proposal_id"]),
            diagnosis_category=str(payload["diagnosis_category"]),
            changes=[PatchChange.from_dict(item) for item in payload["changes"]],
            rationale=str(payload.get("rationale", "")),
            expected_effect=str(payload.get("expected_effect", "")),
            estimated_scope=str(payload.get("estimated_scope", "low")),
        )


@dataclass(frozen=True)
class ArchitectureProposal:
    proposal_id: str
    rationale: str
    expected_effect: str
    expected_failure_categories: list[str]
    estimated_blast_radius: str
    requires_rebaseline: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArchitectureProposal":
        return cls(
            proposal_id=str(payload["proposal_id"]),
            rationale=str(payload.get("rationale", "")),
            expected_effect=str(payload.get("expected_effect", "")),
            expected_failure_categories=[
                str(category) for category in payload.get("expected_failure_categories", [])
            ],
            estimated_blast_radius=str(payload.get("estimated_blast_radius", "unknown")),
            requires_rebaseline=bool(payload.get("requires_rebaseline", True)),
        )
