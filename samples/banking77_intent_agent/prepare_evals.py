from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agent import BANKING77_LABELS
except ModuleNotFoundError:
    from .agent import BANKING77_LABELS


DATA_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small BANKING77 Ratchet eval split.")
    parser.add_argument("--out", default="evals.sanity.jsonl")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--train-per-label", type=int, default=3)
    parser.add_argument("--dev-per-label", type=int, default=3)
    parser.add_argument("--holdout-per-label", type=int, default=2)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    train_by_label = _rows_by_label(_load_csv(cache_dir, "train"))
    test_by_label = _rows_by_label(_load_csv(cache_dir, "test"))
    for label in BANKING77_LABELS:
        train_rows = train_by_label.get(label, [])
        test_rows = test_by_label.get(label, [])
        train_needed = args.train_per_label + args.dev_per_label
        if len(train_rows) < train_needed:
            raise ValueError(f"Not enough train examples for label {label!r}.")
        if len(test_rows) < args.holdout_per_label:
            raise ValueError(f"Not enough test examples for label {label!r}.")
        rows.extend(
            _case_rows(
                split_name="train",
                source_split="train",
                label=label,
                examples=train_rows[: args.train_per_label],
            )
        )
        rows.extend(
            _case_rows(
                split_name="dev",
                source_split="train",
                label=label,
                examples=train_rows[args.train_per_label : train_needed],
            )
        )
        rows.extend(
            _case_rows(
                split_name="holdout",
                source_split="test",
                label=label,
                examples=test_rows[: args.holdout_per_label],
            )
        )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"Wrote {len(rows)} eval cases to {out_path}")


def _load_csv(cache_dir: Path, split: str) -> list[dict[str, str]]:
    path = cache_dir / f"{split}.csv"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(DATA_URLS[split], timeout=30).read())
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _rows_by_label(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {label: [] for label in BANKING77_LABELS}
    for row in rows:
        label = row.get("category", "")
        if label in grouped:
            grouped[label].append(row)
    return grouped


def _case_rows(
    *,
    split_name: str,
    source_split: str,
    label: str,
    examples: list[dict[str, str]],
) -> list[dict[str, object]]:
    return [
        {
            "id": f"{split_name}-{label}-{index}",
            "split": split_name,
            "input": row["text"],
            "expected": {"label": label},
            "metadata": {
                "category": label,
                "source": f"banking77_{source_split}",
            },
        }
        for index, row in enumerate(examples, start=1)
    ]


if __name__ == "__main__":
    main()
