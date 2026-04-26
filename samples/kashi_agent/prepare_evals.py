from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "eval.csv"
OUTPUT_PATH = ROOT / "evals.jsonl"
QUICK_OUTPUT_PATH = ROOT / "evals.quick.jsonl"
QUICK_DEV_CORRECT = 4
QUICK_DEV_INCORRECT = 8
QUICK_HOLDOUT_CORRECT = 4
QUICK_HOLDOUT_INCORRECT = 4


def split_for_index(index: int) -> str:
    return "holdout" if index % 5 == 0 else "dev"


def quick_payloads(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    quotas = {
        ("dev", "correct"): QUICK_DEV_CORRECT,
        ("dev", "incorrect"): QUICK_DEV_INCORRECT,
        ("holdout", "correct"): QUICK_HOLDOUT_CORRECT,
        ("holdout", "incorrect"): QUICK_HOLDOUT_INCORRECT,
    }
    selected: list[dict[str, object]] = []
    counts = {key: 0 for key in quotas}
    for payload in payloads:
        expected = payload.get("expected", {})
        if not isinstance(expected, dict):
            continue
        key = (str(payload.get("split")), str(expected.get("label")))
        if counts.get(key, 0) >= quotas.get(key, 0):
            continue
        selected.append(payload)
        counts[key] += 1
        if all(counts[key] >= quotas[key] for key in quotas):
            break
    return selected


def main() -> None:
    with INPUT_PATH.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    payloads = []
    for index, row in enumerate(rows):
        case_id = (row.get("Example ID") or row.get("example_id") or f"case-{index}").strip()
        expected_score = int(row.get("ground_truth.score") or 0)
        expected_label = (row.get("ground_truth.label") or "").strip().lower()
        if expected_label not in {"correct", "incorrect"}:
            expected_label = "correct" if expected_score == 1 else "incorrect"

        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"input_messages", "ground_truth.score", "ground_truth.label"}
        }
        metadata["historical_label"] = expected_label
        metadata["historical_score"] = expected_score

        payloads.append(
            {
                "id": case_id,
                "split": split_for_index(index),
                "input": row["input_messages"],
                "expected": {
                    "label": expected_label,
                    "score": expected_score,
                },
                "metadata": metadata,
            }
        )

    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in payloads) + "\n"
    )
    quick_rows = quick_payloads(payloads)
    QUICK_OUTPUT_PATH.write_text(
        "\n".join(json.dumps(payload, sort_keys=True) for payload in quick_rows) + "\n"
    )
    split_counts = {
        split: sum(1 for payload in payloads if payload["split"] == split)
        for split in ("dev", "holdout")
    }
    quick_split_counts = {
        split: sum(1 for payload in quick_rows if payload["split"] == split)
        for split in ("dev", "holdout")
    }
    print(f"Wrote {len(payloads)} cases to {OUTPUT_PATH}")
    print(f"Wrote {len(quick_rows)} stratified cases to {QUICK_OUTPUT_PATH}")
    print(split_counts)
    print(quick_split_counts)


if __name__ == "__main__":
    main()
