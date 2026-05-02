from __future__ import annotations

import json
from pathlib import Path

from order_desk_env import TASKS


ROOT = Path(__file__).resolve().parent


SPLITS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7],
    "dev": [100, 101, 102, 103, 104, 105, 106, 107, 200, 201, 202, 203, 204, 205, 206, 207],
    "holdout": [300, 301, 302, 303, 304, 305, 306, 307, 400, 401, 402, 403, 404, 405, 406, 407],
}


def main() -> None:
    rows = []
    for split, task_ids in SPLITS.items():
        for task_id in task_ids:
            task = TASKS[task_id]
            rows.append(
                {
                    "id": f"{split}-{task['category']}-{task_id}",
                    "split": split,
                    "input": f"order desk task {task_id}",
                    "expected": {"reward": 1.0},
                    "metadata": {
                        "benchmark_fidelity": "local_deterministic",
                        "category": task["category"],
                        "task_id": task_id,
                    },
                }
            )
    out_path = ROOT / "evals.assessment.jsonl"
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"Wrote {len(rows)} cases to {out_path}")


if __name__ == "__main__":
    main()
