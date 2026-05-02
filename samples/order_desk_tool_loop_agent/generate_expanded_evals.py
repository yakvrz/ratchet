from __future__ import annotations

import json
from pathlib import Path

from expanded_tasks import SPLITS, install_expanded_tasks
from order_desk_env import TASKS


ROOT = Path(__file__).resolve().parent


def main() -> None:
    install_expanded_tasks()
    rows = []
    for split, task_ids in SPLITS.items():
        for task_id in task_ids:
            task = TASKS[task_id]
            rows.append(
                {
                    "id": f"{split}-{task['category']}-{task_id}",
                    "split": split,
                    "input": f"order desk expanded task {task_id}",
                    "expected": {"reward": 1.0},
                    "metadata": {
                        "benchmark_fidelity": "local_deterministic_expanded",
                        "category": task["category"],
                        "task_id": task_id,
                    },
                }
            )
    out_path = ROOT / "evals.diagnostic_expanded.jsonl"
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"Wrote {len(rows)} cases to {out_path}")


if __name__ == "__main__":
    main()
