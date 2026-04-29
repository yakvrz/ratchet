from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class ContextSection:
    name: str
    role: str
    content: Any
    required: bool = False
    visibility: str = "model_visible"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ContextSection name must be non-empty.")
        if not self.role:
            raise ValueError("ContextSection role must be non-empty.")
        if self.visibility not in {"model_visible", "internal_only"}:
            raise ValueError(f"Unsupported context visibility: {self.visibility}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextSection":
        return cls(
            name=str(payload["name"]),
            role=str(payload.get("role", "system")),
            content=payload.get("content", ""),
            required=bool(payload.get("required", False)),
            visibility=str(payload.get("visibility", "model_visible")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class ContextGraph:
    sections: tuple[ContextSection, ...] = ()

    def __post_init__(self) -> None:
        names = [section.name for section in self.sections]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate context sections: {', '.join(duplicates)}")

    def to_dict(self) -> dict[str, Any]:
        return {"sections": [section.to_dict() for section in self.sections]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContextGraph":
        return cls(
            sections=tuple(
                ContextSection.from_dict(dict(item))
                for item in payload.get("sections", [])
                if isinstance(item, dict)
            )
        )

    def section_names(self) -> list[str]:
        return [section.name for section in self.sections]

    def has_section(self, name: str) -> bool:
        return any(section.name == name for section in self.sections)

    def add_section(self, section: ContextSection, *, position: str | None = None) -> "ContextGraph":
        if self.has_section(section.name):
            raise ValueError(f"Context section {section.name!r} already exists.")
        sections = list(self.sections)
        index = _position_index(sections, position)
        sections.insert(index, section)
        return ContextGraph(tuple(sections))

    def remove_section(self, name: str) -> "ContextGraph":
        found = False
        sections: list[ContextSection] = []
        for section in self.sections:
            if section.name != name:
                sections.append(section)
                continue
            found = True
            if section.required:
                raise ValueError(f"Cannot remove required context section {name!r}.")
        if not found:
            raise ValueError(f"Unknown context section {name!r}.")
        return ContextGraph(tuple(sections))

    def replace_section(self, name: str, content: Any) -> "ContextGraph":
        sections: list[ContextSection] = []
        found = False
        for section in self.sections:
            if section.name == name:
                found = True
                sections.append(
                    ContextSection(
                        name=section.name,
                        role=section.role,
                        content=content,
                        required=section.required,
                        visibility=section.visibility,
                        metadata=dict(section.metadata),
                    )
                )
            else:
                sections.append(section)
        if not found:
            raise ValueError(f"Unknown context section {name!r}.")
        return ContextGraph(tuple(sections))

    def move_section(self, name: str, *, position: str) -> "ContextGraph":
        moving = None
        remaining: list[ContextSection] = []
        for section in self.sections:
            if section.name == name:
                moving = section
            else:
                remaining.append(section)
        if moving is None:
            raise ValueError(f"Unknown context section {name!r}.")
        index = _position_index(remaining, position)
        remaining.insert(index, moving)
        return ContextGraph(tuple(remaining))

    def reorder(self, names: list[str]) -> "ContextGraph":
        by_name = {section.name: section for section in self.sections}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"Unknown context sections in order: {', '.join(missing)}")
        required_missing = [section.name for section in self.sections if section.required and section.name not in names]
        if required_missing:
            raise ValueError(f"Order omits required sections: {', '.join(required_missing)}")
        ordered = [by_name[name] for name in names]
        ordered.extend(section for section in self.sections if section.name not in set(names))
        return ContextGraph(tuple(ordered))

    def render_text(self) -> str:
        rendered = []
        for section in self.sections:
            if section.visibility != "model_visible":
                continue
            text = _render_content(section.content)
            if not text:
                continue
            rendered.append(f"[{section.name}]\n{text}")
        return "\n\n".join(rendered)


def _position_index(sections: list[ContextSection], position: str | None) -> int:
    if not position:
        return len(sections)
    if position == "start":
        return 0
    if position == "end":
        return len(sections)
    if position.startswith("before:"):
        target = position.split(":", 1)[1]
        for index, section in enumerate(sections):
            if section.name == target:
                return index
        raise ValueError(f"Unknown context section position target {target!r}.")
    if position.startswith("after:"):
        target = position.split(":", 1)[1]
        for index, section in enumerate(sections):
            if section.name == target:
                return index + 1
        raise ValueError(f"Unknown context section position target {target!r}.")
    raise ValueError(f"Unsupported context section position {position!r}.")


def _render_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, indent=2, sort_keys=True, default=str)
    return str(content)
