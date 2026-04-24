from __future__ import annotations

import ast
import inspect
from types import FunctionType
from typing import Any, Callable

from ratchet.io import depends_on_satisfied
from ratchet.types import CodeArtifactSpec


SAFE_BUILTINS: dict[str, object] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def parse_signature(signature: str) -> tuple[str, ...]:
    cleaned = signature.strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    return tuple(part.strip() for part in cleaned.split(",") if part.strip())


def default_hook_source(spec: CodeArtifactSpec) -> str:
    parameter_names = parse_signature(spec.signature)
    if not parameter_names:
        raise ValueError(f"Code artifact {spec.name} has no callable parameters.")
    passthrough = parameter_names[0]
    args = ", ".join(parameter_names)
    return (
        f"def {spec.callable_name}({args}):\n"
        f"    return {passthrough}\n"
    )


def _validate_ast(tree: ast.AST, spec: CodeArtifactSpec) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(f"Code artifact {spec.name} may not import modules.")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)):
            raise ValueError(f"Code artifact {spec.name} uses an unsupported construct.")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"Code artifact {spec.name} may not access dunder names.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Code artifact {spec.name} may not access dunder attributes.")


def validate_code_artifact_source(spec: CodeArtifactSpec, source: str) -> None:
    if len(source) > spec.max_chars:
        raise ValueError(f"Code artifact {spec.name} exceeds max_chars {spec.max_chars}.")
    if len(source.splitlines()) > spec.max_lines:
        raise ValueError(f"Code artifact {spec.name} exceeds max_lines {spec.max_lines}.")
    try:
        tree = ast.parse(source, filename=f"<code-artifact:{spec.name}>", mode="exec")
    except SyntaxError as error:
        raise ValueError(f"Code artifact {spec.name} failed to parse: {error.msg}") from error
    _validate_ast(tree, spec)


def compile_code_artifact(spec: CodeArtifactSpec, source: str) -> Callable[..., Any]:
    validate_code_artifact_source(spec, source)
    environment: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "Any": Any,
    }
    compiled = compile(source, filename=f"<code-artifact:{spec.name}>", mode="exec")
    exec(compiled, environment, environment)
    hook = environment.get(spec.callable_name)
    if not isinstance(hook, FunctionType):
        raise ValueError(
            f"Code artifact {spec.name} must define callable {spec.callable_name!r}."
        )
    actual_parameters = tuple(inspect.signature(hook).parameters.keys())
    expected_parameters = parse_signature(spec.signature)
    if actual_parameters != expected_parameters:
        raise ValueError(
            f"Code artifact {spec.name} must match signature ({', '.join(expected_parameters)})."
        )
    return hook


class CodeArtifactLoader:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], Callable[..., Any]] = {}

    def compile(self, spec: CodeArtifactSpec, source: str) -> Callable[..., Any]:
        key = (spec.name, source)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = compile_code_artifact(spec, source)
            self._cache[key] = compiled
        return compiled

    def build_hooks(
        self,
        candidate: dict[str, str],
        code_artifacts: list[CodeArtifactSpec],
    ) -> dict[str, Callable[..., Any]]:
        hooks: dict[str, Callable[..., Any]] = {}
        for spec in code_artifacts:
            if not depends_on_satisfied(candidate, spec):
                continue
            hooks[spec.name] = self.compile(spec, candidate[spec.name])
        return hooks
