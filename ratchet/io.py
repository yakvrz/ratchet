from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from ratchet.types import AgentPatch, AgentSpec, EvalCase


def load_eval_cases(path: str | Path) -> tuple[EvalCase, ...]:
    eval_path = Path(path)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for raw_line in eval_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        case = EvalCase.from_dict(payload)
        if case.id in seen_ids:
            raise ValueError(f"Eval file contains duplicate case id: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)
    return tuple(cases)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str))
        handle.write("\n")


def write_jsonl(path: str | Path, payloads: list[dict[str, Any]]) -> None:
    Path(path).write_text(
        "\n".join(json.dumps(payload, sort_keys=True, default=str) for payload in payloads)
        + ("\n" if payloads else "")
    )


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def short_digest(payload: Any) -> str:
    return stable_digest(payload)[:12]


def patch_hash(patch: AgentPatch | None) -> str:
    return short_digest((patch or AgentPatch.empty()).to_dict())


def agent_spec_hash(spec: AgentSpec | None) -> str:
    return short_digest(spec.to_dict() if spec is not None else {"agent_spec": None})


def case_digest(case: EvalCase) -> str:
    return stable_digest(case.to_dict())


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()
