from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, runtime_checkable

from ratchet.types import EvalCase, GradeResult, RunRecord, SearchSpace


@runtime_checkable
class AdapterProtocol(Protocol):
    def baseline(self) -> dict[str, str]:
        ...

    def search_space(self) -> SearchSpace:
        ...

    def run_case(self, candidate: dict[str, str], case: EvalCase) -> RunRecord:
        ...

    def grade(self, case: EvalCase, output: object) -> GradeResult:
        ...

    def export(self, candidate: dict[str, str], out_dir: Path) -> None:
        ...


def load_adapter(spec: str) -> AdapterProtocol:
    if ":" not in spec:
        raise ValueError("Adapter spec must use the form package.module:adapter_object")
    module_name, attribute = spec.split(":", 1)
    module = import_module(module_name)
    adapter = getattr(module, attribute)
    if not isinstance(adapter, AdapterProtocol):
        missing = [
            method_name
            for method_name in ("baseline", "search_space", "run_case", "grade", "export")
            if not callable(getattr(adapter, method_name, None))
        ]
        if missing:
            raise TypeError(f"Adapter {spec} is missing required methods: {', '.join(missing)}")
    return adapter
