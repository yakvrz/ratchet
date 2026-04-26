from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import tomllib
from typing import Any

from ratchet.types import OptimizationConstraints, OptimizationObjective


class RatchetConfigError(ValueError):
    """Raised when a Ratchet run config is missing required fields or has unknown keys."""


RUN_CONFIG_KEYS = {
    "adapter",
    "evals",
    "out",
    "env_file",
    "dev_budget",
    "holdout_budget",
    "holdout_top_k",
    "objective",
    "mode",
    "allowed_models",
    "allowed_edits",
    "sanitize_examples",
    "optimizer_model",
    "optimizer_reasoning",
    "samples_per_case",
    "case_concurrency",
    "max_case_retries",
    "case_timeout_s",
    "fail_fast",
}
OBJECTIVE_KEYS = {"mode", "constraints", "tie_breakers"}
CONSTRAINT_KEYS = {
    "allowed_edits",
    "allowed_models",
    "max_cost_ratio",
    "max_latency_ratio",
    "min_correctness_delta",
    "max_patch_operations",
    "sanitize_examples",
}
REQUIRED_RUN_KEYS = {"adapter", "evals", "out"}


@dataclass(frozen=True)
class RatchetRunConfig:
    adapter: str
    evals: Path
    out: Path
    env_file: str = ".env"
    dev_budget: int = 20
    holdout_budget: int = 5
    objective: OptimizationObjective = field(default_factory=OptimizationObjective)
    optimizer_model: str = "gpt-5.4"
    optimizer_reasoning: str = "medium"
    samples_per_case: int = 1
    case_concurrency: int = 1
    max_case_retries: int = 2
    case_timeout_s: int = 180
    fail_fast: bool = False
    config_path: Path | None = None

    @property
    def search_path(self) -> Path | None:
        if self.config_path is None:
            return None
        return self.config_path.parent

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "evals": str(self.evals),
            "out": str(self.out),
            "env_file": self.env_file,
            "dev_budget": self.dev_budget,
            "holdout_budget": self.holdout_budget,
            "objective": self.objective.to_dict(),
            "optimizer_model": self.optimizer_model,
            "optimizer_reasoning": self.optimizer_reasoning,
            "samples_per_case": self.samples_per_case,
            "case_concurrency": self.case_concurrency,
            "max_case_retries": self.max_case_retries,
            "case_timeout_s": self.case_timeout_s,
            "fail_fast": self.fail_fast,
            "config_path": str(self.config_path) if self.config_path else None,
        }


def _resolve_path(raw_value: str, root: Path) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _as_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    if key not in payload:
        return default
    return bool(payload[key])


def _objective_from_payload(payload: dict[str, Any]) -> OptimizationObjective:
    raw_objective = dict(payload.get("objective", {}))
    _reject_unknown_keys(raw_objective, OBJECTIVE_KEYS, "ratchet.objective")
    if "mode" not in raw_objective and "mode" in payload:
        raw_objective["mode"] = payload["mode"]
    constraints = dict(raw_objective.get("constraints", {}))
    _reject_unknown_keys(constraints, CONSTRAINT_KEYS, "ratchet.objective.constraints")
    if "allowed_models" in payload:
        constraints["allowed_models"] = payload["allowed_models"]
    if "allowed_edits" in payload:
        constraints["allowed_edits"] = payload["allowed_edits"]
    if "sanitize_examples" in payload:
        constraints["sanitize_examples"] = payload["sanitize_examples"]
    raw_objective["constraints"] = constraints
    return OptimizationObjective.from_dict(raw_objective)


def load_run_config(path: str | Path) -> RatchetRunConfig:
    config_path = Path(path).resolve()
    try:
        payload = tomllib.loads(config_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise RatchetConfigError(f"Invalid TOML in {config_path}: {exc}") from exc
    if "ratchet" in payload:
        extra_top_level = sorted(set(payload) - {"ratchet"})
        if extra_top_level:
            raise RatchetConfigError(
                f"Unknown top-level config key(s): {', '.join(extra_top_level)}"
            )
        if not isinstance(payload["ratchet"], dict):
            raise RatchetConfigError("Config key 'ratchet' must be a table.")
        payload = dict(payload["ratchet"])
    _validate_run_payload(payload)
    root = config_path.parent
    return RatchetRunConfig(
        adapter=str(payload["adapter"]),
        evals=_resolve_path(str(payload["evals"]), root),
        out=_resolve_path(str(payload["out"]), root),
        env_file=str(_resolve_path(str(payload.get("env_file", ".env")), root)),
        dev_budget=int(payload.get("dev_budget", 20)),
        holdout_budget=int(payload.get("holdout_budget", payload.get("holdout_top_k", 5))),
        objective=_objective_from_payload(payload),
        optimizer_model=str(payload.get("optimizer_model", "gpt-5.4")),
        optimizer_reasoning=str(payload.get("optimizer_reasoning", "medium")),
        samples_per_case=int(payload.get("samples_per_case", 1)),
        case_concurrency=int(payload.get("case_concurrency", 1)),
        max_case_retries=int(payload.get("max_case_retries", 2)),
        case_timeout_s=int(payload.get("case_timeout_s", 180)),
        fail_fast=_as_bool(payload, "fail_fast", False),
        config_path=config_path,
    )


def _validate_run_payload(payload: dict[str, Any]) -> None:
    _reject_unknown_keys(payload, RUN_CONFIG_KEYS, "ratchet")
    missing = sorted(key for key in REQUIRED_RUN_KEYS if key not in payload)
    if missing:
        raise RatchetConfigError(f"Missing required config key(s): {', '.join(missing)}")
    raw_objective = payload.get("objective", {})
    if raw_objective is not None and not isinstance(raw_objective, dict):
        raise RatchetConfigError("Config key 'ratchet.objective' must be a table.")
    raw_constraints = dict(raw_objective or {}).get("constraints", {})
    if raw_constraints is not None and not isinstance(raw_constraints, dict):
        raise RatchetConfigError("Config key 'ratchet.objective.constraints' must be a table.")


def _reject_unknown_keys(payload: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RatchetConfigError(
            f"Unknown config key(s) in {context}: {', '.join(unknown)}"
        )


def resolve_run_config(
    *,
    config_path: str | Path | None,
    adapter: str | None,
    evals_path: str | Path | None,
    out_dir: str | Path | None,
    env_file: str | None,
    dev_budget: int | None,
    holdout_budget: int | None,
    objective_mode: str | None,
    allowed_models: list[str] | None,
    allowed_edits: list[str] | None,
    optimizer_model: str | None,
    optimizer_reasoning: str | None,
    samples_per_case: int | None,
    case_concurrency: int | None,
    max_case_retries: int | None,
    case_timeout_s: int | None,
    fail_fast: bool | None,
    sanitize_examples: bool | None = None,
) -> RatchetRunConfig:
    if config_path is not None:
        base = load_run_config(config_path)
    else:
        if adapter is None or evals_path is None or out_dir is None:
            raise ValueError("Either --config or --adapter/--evals/--out must be provided.")
        base = RatchetRunConfig(
            adapter=adapter,
            evals=Path(evals_path).resolve(),
            out=Path(out_dir).resolve(),
        )

    constraints_payload = base.objective.constraints.to_dict()
    if allowed_models is not None:
        constraints_payload["allowed_models"] = allowed_models
    if allowed_edits is not None:
        constraints_payload["allowed_edits"] = allowed_edits
    if sanitize_examples is not None:
        constraints_payload["sanitize_examples"] = sanitize_examples
    objective = OptimizationObjective(
        mode=objective_mode or base.objective.mode,
        constraints=OptimizationConstraints.from_dict(constraints_payload),
        tie_breakers=list(base.objective.tie_breakers),
    )
    return RatchetRunConfig(
        adapter=adapter or base.adapter,
        evals=Path(evals_path).resolve() if evals_path is not None else base.evals,
        out=Path(out_dir).resolve() if out_dir is not None else base.out,
        env_file=env_file or base.env_file,
        dev_budget=dev_budget if dev_budget is not None else base.dev_budget,
        holdout_budget=holdout_budget if holdout_budget is not None else base.holdout_budget,
        objective=objective,
        optimizer_model=optimizer_model or base.optimizer_model,
        optimizer_reasoning=optimizer_reasoning or base.optimizer_reasoning,
        samples_per_case=samples_per_case if samples_per_case is not None else base.samples_per_case,
        case_concurrency=case_concurrency if case_concurrency is not None else base.case_concurrency,
        max_case_retries=max_case_retries if max_case_retries is not None else base.max_case_retries,
        case_timeout_s=case_timeout_s if case_timeout_s is not None else base.case_timeout_s,
        fail_fast=fail_fast if fail_fast is not None else base.fail_fast,
        config_path=base.config_path,
    )


def ensure_search_path(config: RatchetRunConfig) -> None:
    search_path = config.search_path
    if search_path is None:
        return
    path_str = str(search_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
