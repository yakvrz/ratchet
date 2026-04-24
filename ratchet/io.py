from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ratchet.types import (
    CodeArtifactSpec,
    ComponentSpec,
    EnumKnobSpec,
    EvalCase,
    SearchSpace,
    TextArtifactSpec,
)


def load_eval_cases(path: str | Path) -> tuple[EvalCase, ...]:
    eval_path = Path(path)
    cases: list[EvalCase] = []
    for raw_line in eval_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        cases.append(EvalCase.from_dict(payload))
    return tuple(cases)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def write_jsonl(path: str | Path, payloads: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads)
        + ("\n" if payloads else "")
    )


def depends_on_satisfied(
    candidate: dict[str, str],
    spec: EnumKnobSpec | TextArtifactSpec | ComponentSpec | CodeArtifactSpec,
) -> bool:
    for dependency_name, allowed_values in spec.depends_on.items():
        if candidate.get(dependency_name) not in allowed_values:
            return False
    return True


def normalize_candidate(candidate: dict[str, str], search_space: SearchSpace) -> dict[str, str]:
    normalized = {
        spec.name: spec.default
        for spec in [
            *search_space.enum_knobs,
            *search_space.text_artifacts,
            *search_space.components,
            *search_space.code_artifacts,
        ]
    }
    normalized.update({str(key): str(value) for key, value in candidate.items()})

    spec_by_name = {spec.name: spec for spec in search_space.all_specs()}
    unknown = sorted(set(normalized) - set(spec_by_name))
    if unknown:
        raise ValueError(f"Unknown candidate keys: {', '.join(unknown)}")

    for spec in search_space.enum_knobs:
        value = normalized[spec.name]
        if value not in spec.values:
            raise ValueError(f"Candidate value {value!r} is invalid for enum knob {spec.name}")

    for spec in search_space.components:
        value = normalized[spec.name]
        if value not in spec.values:
            raise ValueError(f"Candidate value {value!r} is invalid for component {spec.name}")

    for spec in search_space.text_artifacts:
        value = normalized[spec.name]
        if len(value) > spec.max_chars:
            raise ValueError(
                f"Candidate value for text artifact {spec.name} exceeds max_chars {spec.max_chars}"
            )

    for spec in search_space.code_artifacts:
        value = normalized[spec.name]
        if len(value) > spec.max_chars:
            raise ValueError(
                f"Candidate value for code artifact {spec.name} exceeds max_chars {spec.max_chars}"
            )
        if len(value.splitlines()) > spec.max_lines:
            raise ValueError(
                f"Candidate value for code artifact {spec.name} exceeds max_lines {spec.max_lines}"
            )

    changed = True
    while changed:
        changed = False
        for spec in search_space.all_specs():
            if depends_on_satisfied(normalized, spec):
                continue
            if normalized[spec.name] != spec.default:
                normalized[spec.name] = spec.default
                changed = True
    return normalized


def candidate_hash(candidate: dict[str, str]) -> str:
    payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()[:12]


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
