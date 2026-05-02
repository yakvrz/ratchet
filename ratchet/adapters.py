from __future__ import annotations

from hashlib import sha256
from importlib import import_module
import inspect
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ratchet.surfaces import SurfaceSpec, surface_from_agent_spec
from ratchet.transform_program import CompiledCandidate
from ratchet.types import AgentSpec, EvalCase, GradeResult, RunRecord


FINGERPRINTED_SOURCE_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

IGNORED_FINGERPRINT_DIRS = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "results",
}


@runtime_checkable
class AdapterProtocol(Protocol):
    def agent_spec(self) -> AgentSpec:
        ...

    def surface_spec(self, cases: tuple[EvalCase, ...]) -> SurfaceSpec:
        ...

    def run_case(self, case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord:
        ...

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        ...

    def export(self, candidate: CompiledCandidate, out_dir: Path) -> None:
        ...


def load_adapter(spec: str) -> AdapterProtocol:
    if ":" not in spec:
        raise ValueError("Adapter spec must use the form package.module:adapter_object")
    module_name, attribute = spec.split(":", 1)
    module = import_module(module_name)
    adapter = getattr(module, attribute)
    missing = [
        method_name
        for method_name in ("agent_spec", "run_case", "grade", "export")
        if not callable(getattr(adapter, method_name, None))
    ]
    if missing:
        raise TypeError(f"Adapter {spec} is missing required methods: {', '.join(missing)}")
    return adapter


def checked_agent_spec(adapter: AdapterProtocol, *, adapter_spec: str = "adapter") -> AgentSpec:
    # Optimizer internals still derive task evidence from AgentSpec while the new
    # transform path uses surface_spec() as the executable optimization contract.
    method = getattr(adapter, "agent_spec", None)
    if not callable(method):
        raise TypeError(f"Adapter {adapter_spec} is missing agent_spec().")
    try:
        spec = method()
    except Exception as exc:
        raise TypeError(f"Adapter {adapter_spec} agent_spec() failed: {exc}") from exc
    if spec is None:
        raise TypeError(f"Adapter {adapter_spec} agent_spec() returned None, expected AgentSpec.")
    if not isinstance(spec, AgentSpec):
        raise TypeError(
            f"Adapter {adapter_spec} agent_spec() returned {type(spec).__name__}, expected AgentSpec."
        )
    return spec


def checked_surface_spec(
    adapter: AdapterProtocol,
    *,
    adapter_spec: str = "adapter",
    cases: tuple[EvalCase, ...],
) -> SurfaceSpec:
    if not cases:
        raise ValueError(f"Adapter {adapter_spec} surface inference requires at least one proposal-safe case.")
    method = getattr(adapter, "surface_spec", None)
    if callable(method):
        try:
            surface = method(cases)
        except Exception as exc:
            raise TypeError(f"Adapter {adapter_spec} surface_spec() failed: {exc}") from exc
        if not isinstance(surface, SurfaceSpec):
            raise TypeError(
                f"Adapter {adapter_spec} surface_spec() returned {type(surface).__name__}, expected SurfaceSpec."
            )
        return surface
    return surface_from_agent_spec(checked_agent_spec(adapter, adapter_spec=adapter_spec))


def adapter_fingerprint(spec: str) -> dict[str, Any]:
    if ":" not in spec:
        raise ValueError("Adapter spec must use the form package.module:adapter_object")
    module_name, attribute = spec.split(":", 1)
    module = import_module(module_name)
    adapter = getattr(module, attribute)
    source_path = None
    for source_object in (adapter, type(adapter), module):
        try:
            source_path = inspect.getsourcefile(source_object)
        except TypeError:
            source_path = None
        if source_path:
            break
    digest = None
    source_tree_digest = None
    if source_path is not None:
        path = Path(source_path)
        if path.exists():
            digest = sha256(path.read_bytes()).hexdigest()
            source_tree_digest = _source_tree_digest(path.parent)
            source_path = str(path.resolve())
    custom_fingerprint = _custom_adapter_fingerprint(adapter)
    return {
        "spec": spec,
        "module": module_name,
        "attribute": attribute,
        "source_path": source_path,
        "source_sha256": digest,
        "source_tree_sha256": source_tree_digest,
        "custom_fingerprint": custom_fingerprint,
        "custom_fingerprint_sha256": _stable_digest(custom_fingerprint)
        if custom_fingerprint is not None
        else None,
    }


def _source_tree_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if _should_fingerprint_source_file(root, item)):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _should_fingerprint_source_file(root: Path, path: Path) -> bool:
    if not path.is_file() or path.name.startswith(".") or path.suffix not in FINGERPRINTED_SOURCE_SUFFIXES:
        return False
    relative = path.relative_to(root)
    for part in relative.parts:
        if part.startswith(".") or part in IGNORED_FINGERPRINT_DIRS:
            return False
    return True


def _custom_adapter_fingerprint(adapter: object) -> Any:
    for method_name in ("fingerprint", "cache_fingerprint"):
        method = getattr(adapter, method_name, None)
        if callable(method):
            return method()
    return None


def _stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()
