from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path

from ratchet.tool_loop import GeneratedToolLoopAdapter
from ratchet.types import EvalCase
from ratchet_adapter import BASE_SPEC, _case_config, _grade, _make_environment
from order_desk_env import make_action


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live baseline calibration pass for Order Desk.")
    parser.add_argument("--evals", default=str(ROOT / "evals.assessment.jsonl"))
    parser.add_argument("--split", default="dev", choices=["train", "dev", "holdout"])
    parser.add_argument("--model", default=BASE_SPEC.model)
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()

    spec = replace(BASE_SPEC, model=args.model)
    adapter = GeneratedToolLoopAdapter(
        agent_spec=spec,
        environment_factory=_make_environment,
        action_factory=make_action,
        respond_action_name="respond",
        case_config=_case_config,
        grade=_grade,
    )
    cases = [
        EvalCase.from_dict(json.loads(line))
        for line in Path(args.evals).read_text().splitlines()
        if json.loads(line)["split"] == args.split
    ]
    adapter.surface_spec(tuple(cases[: min(3, len(cases))]))

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [executor.submit(_run_case, adapter, case) for case in cases]
        for future in as_completed(futures):
            rows.append(future.result())

    rows.sort(key=lambda row: row["case_id"])
    labels = Counter(label for row in rows for label in row["labels"])
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)

    print(f"model={args.model} split={args.split} pass={sum(row['passed'] for row in rows)}/{len(rows)}")
    print("by_category:")
    for category, category_rows in sorted(by_category.items()):
        print(f"  {category}: {sum(row['passed'] for row in category_rows)}/{len(category_rows)}")
    print("labels:")
    for label, count in labels.most_common():
        print(f"  {label}: {count}")
    print("cases:")
    for row in rows:
        print(json.dumps(row, sort_keys=True))


def _run_case(adapter: GeneratedToolLoopAdapter, case: EvalCase) -> dict[str, object]:
    record = adapter.run_case(case)
    grade = adapter.grade(case, record.output)
    return {
        "case_id": case.id,
        "category": case.metadata.get("category"),
        "passed": grade.passed,
        "labels": list(grade.labels),
        "model_calls": record.metrics.model_calls,
        "tool_calls": record.metrics.tool_calls,
        "turns": record.metrics.turns,
    }


if __name__ == "__main__":
    main()
