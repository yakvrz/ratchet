from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any


DEFAULT_DOMAINS = ("retail", "airline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tau-bench Ratchet eval cases.")
    parser.add_argument("--out", default="samples/taubench_agent/evals.assessment.jsonl")
    parser.add_argument("--per-domain-dev", type=int, default=8)
    parser.add_argument("--per-domain-holdout", type=int, default=8)
    parser.add_argument("--domains", nargs="*", default=list(DEFAULT_DOMAINS))
    args = parser.parse_args()
    rows = generate_cases(
        domains=tuple(args.domains),
        per_domain_dev=args.per_domain_dev,
        per_domain_holdout=args.per_domain_holdout,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"Wrote {len(rows)} tau-bench cases to {out}")


def generate_cases(
    *,
    domains: tuple[str, ...],
    per_domain_dev: int,
    per_domain_holdout: int,
) -> list[dict[str, Any]]:
    if per_domain_dev <= 0 or per_domain_holdout <= 0:
        raise ValueError("per-domain dev and holdout counts must be positive.")
    rows: list[dict[str, Any]] = []
    for domain in domains:
        task_count = _task_count(domain)
        required = per_domain_dev + per_domain_holdout
        if task_count < required:
            raise ValueError(f"tau-bench domain {domain!r} has {task_count} tasks, need {required}.")
        for index in range(per_domain_dev):
            rows.append(_case_row(domain=domain, split="dev", task_index=index))
        for index in range(per_domain_dev, required):
            rows.append(_case_row(domain=domain, split="holdout", task_index=index))
    dev_count = sum(1 for row in rows if row["split"] == "dev")
    holdout_count = sum(1 for row in rows if row["split"] == "holdout")
    if dev_count < 16 or holdout_count < 16:
        raise ValueError("representative tau assessment requires at least 16 dev and 16 holdout cases.")
    return rows


def _task_count(domain: str) -> int:
    try:
        module = importlib.import_module(f"tau_bench.envs.{domain}.tasks_test")
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"tau-bench domain {domain!r} is not installed.") from exc
    for name in ("TASKS_TEST", "TASKS"):
        tasks = getattr(module, name, None)
        if isinstance(tasks, list):
            return len(tasks)
    raise RuntimeError(f"tau-bench domain {domain!r} does not expose TASKS_TEST or TASKS.")


def _case_row(*, domain: str, split: str, task_index: int) -> dict[str, Any]:
    return {
        "id": f"{split}-{domain}-{task_index}",
        "split": split,
        "input": f"tau-bench {domain} task {task_index}.",
        "expected": {"reward": 1.0},
        "metadata": {
            "benchmark_fidelity": "tau_bench_simulator",
            "category": domain,
            "env": domain,
            "task_id": task_index,
            "task_split": "test",
        },
    }


if __name__ == "__main__":
    main()
