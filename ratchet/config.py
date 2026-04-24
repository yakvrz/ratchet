from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tomllib
from typing import Any


@dataclass(frozen=True)
class RatchetRunConfig:
    adapter: str
    evals: Path
    out: Path
    env_file: str = ".env"
    dev_budget: int = 20
    holdout_top_k: int = 5
    harnesser_model: str = "gpt-5.4"
    harnesser_reasoning: str = "medium"
    harnesser_enabled: bool = True
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
            "holdout_top_k": self.holdout_top_k,
            "harnesser_model": self.harnesser_model,
            "harnesser_reasoning": self.harnesser_reasoning,
            "harnesser_enabled": self.harnesser_enabled,
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


def load_run_config(path: str | Path) -> RatchetRunConfig:
    config_path = Path(path).resolve()
    payload = tomllib.loads(config_path.read_text())
    if "ratchet" in payload:
        payload = dict(payload["ratchet"])
    root = config_path.parent
    return RatchetRunConfig(
        adapter=str(payload["adapter"]),
        evals=_resolve_path(str(payload["evals"]), root),
        out=_resolve_path(str(payload["out"]), root),
        env_file=str(_resolve_path(str(payload.get("env_file", ".env")), root)),
        dev_budget=int(payload.get("dev_budget", 20)),
        holdout_top_k=int(payload.get("holdout_top_k", 5)),
        harnesser_model=str(payload.get("harnesser_model", "gpt-5.4")),
        harnesser_reasoning=str(payload.get("harnesser_reasoning", "medium")),
        harnesser_enabled=_as_bool(payload, "harnesser_enabled", True),
        max_case_retries=int(payload.get("max_case_retries", 2)),
        case_timeout_s=int(payload.get("case_timeout_s", 180)),
        fail_fast=_as_bool(payload, "fail_fast", False),
        config_path=config_path,
    )


def resolve_run_config(
    *,
    config_path: str | Path | None,
    adapter: str | None,
    evals_path: str | Path | None,
    out_dir: str | Path | None,
    env_file: str | None,
    dev_budget: int | None,
    holdout_top_k: int | None,
    harnesser_model: str | None,
    harnesser_reasoning: str | None,
    harnesser_enabled: bool | None,
    max_case_retries: int | None,
    case_timeout_s: int | None,
    fail_fast: bool | None,
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
    return RatchetRunConfig(
        adapter=adapter or base.adapter,
        evals=Path(evals_path).resolve() if evals_path is not None else base.evals,
        out=Path(out_dir).resolve() if out_dir is not None else base.out,
        env_file=env_file or base.env_file,
        dev_budget=dev_budget if dev_budget is not None else base.dev_budget,
        holdout_top_k=holdout_top_k if holdout_top_k is not None else base.holdout_top_k,
        harnesser_model=harnesser_model or base.harnesser_model,
        harnesser_reasoning=harnesser_reasoning or base.harnesser_reasoning,
        harnesser_enabled=harnesser_enabled if harnesser_enabled is not None else base.harnesser_enabled,
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
