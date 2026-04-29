from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from agent import CLINC150_LABELS
except ModuleNotFoundError:
    from .agent import CLINC150_LABELS


DATA_URL = "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CLINC150 Ratchet assessment split.")
    parser.add_argument("--out", default="evals.assessment.jsonl")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--train-total", type=int, default=96)
    parser.add_argument("--dev-total", type=int, default=96)
    parser.add_argument("--holdout-total", type=int, default=96)
    parser.add_argument("--train-per-label", type=int)
    parser.add_argument("--dev-per-label", type=int)
    parser.add_argument("--holdout-per-label", type=int)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(cache_dir)

    rows = []
    train_by_label = _rows_by_label(data["train"] + data["oos_train"])
    dev_by_label = _rows_by_label(data["val"] + data["oos_val"])
    holdout_by_label = _rows_by_label(data["test"] + data["oos_test"])
    train_counts = _counts_by_label(
        CLINC150_LABELS,
        total=args.train_total,
        per_label=args.train_per_label,
    )
    dev_counts = _counts_by_label(
        CLINC150_LABELS,
        total=args.dev_total,
        per_label=args.dev_per_label,
    )
    holdout_counts = _counts_by_label(
        CLINC150_LABELS,
        total=args.holdout_total,
        per_label=args.holdout_per_label,
    )
    for label in CLINC150_LABELS:
        train_rows = train_by_label.get(label, [])
        dev_rows = dev_by_label.get(label, [])
        holdout_rows = holdout_by_label.get(label, [])
        train_count = train_counts[label]
        dev_count = dev_counts[label]
        holdout_count = holdout_counts[label]
        if len(train_rows) < train_count:
            raise ValueError(f"Not enough train examples for label {label!r}.")
        if len(dev_rows) < dev_count:
            raise ValueError(f"Not enough dev examples for label {label!r}.")
        if len(holdout_rows) < holdout_count:
            raise ValueError(f"Not enough holdout examples for label {label!r}.")
        train_rows = _rank_examples(label, train_rows)
        dev_rows = _rank_examples(label, dev_rows)
        holdout_rows = _rank_examples(label, holdout_rows)
        rows.extend(
            _case_rows(
                split_name="train",
                source_split="train" if label != "oos" else "oos_train",
                label=label,
                examples=train_rows[:train_count],
            )
        )
        rows.extend(
            _case_rows(
                split_name="dev",
                source_split="val" if label != "oos" else "oos_val",
                label=label,
                examples=dev_rows[:dev_count],
            )
        )
        rows.extend(
            _case_rows(
                split_name="holdout",
                source_split="test" if label != "oos" else "oos_test",
                label=label,
                examples=holdout_rows[:holdout_count],
            )
        )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")
    print(f"Wrote {len(rows)} eval cases to {out_path}")


def _load_data(cache_dir: Path) -> dict[str, list[list[str]]]:
    path = cache_dir / "data_full.json"
    if not path.exists():
        path.write_bytes(urllib.request.urlopen(DATA_URL, timeout=30).read())
    payload = json.loads(path.read_text())
    required_splits = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    missing = sorted(required_splits - set(payload))
    if missing:
        raise ValueError(f"CLINC150 data is missing required splits: {missing}")
    return payload


def _rows_by_label(rows: list[list[str]]) -> dict[str, list[list[str]]]:
    grouped = {label: [] for label in CLINC150_LABELS}
    for row in rows:
        if len(row) != 2:
            raise ValueError(f"Expected CLINC150 row with text and label, got {row!r}.")
        label = row[1]
        if label in grouped:
            grouped[label].append(row)
    return grouped


def _counts_by_label(labels: list[str], *, total: int, per_label: int | None) -> dict[str, int]:
    if per_label is not None:
        if per_label <= 0:
            raise ValueError("per-label counts must be positive.")
        return {label: per_label for label in labels}
    if total < len(labels):
        raise ValueError(f"Total count {total} is smaller than label count {len(labels)}.")
    base, remainder = divmod(total, len(labels))
    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }


def _rank_examples(label: str, examples: list[list[str]]) -> list[list[str]]:
    if label == "oos":
        return examples
    label_terms = {term for term in label.split("_") if term}
    return sorted(
        examples,
        key=lambda row: (-_label_overlap(row[0], label_terms), len(row[0]), row[0]),
    )


def _label_overlap(text: str, label_terms: set[str]) -> int:
    normalized = "".join(character if character.isalnum() else " " for character in text.lower())
    text_terms = set(normalized.split())
    return len(label_terms & text_terms)


def _case_rows(
    *,
    split_name: str,
    source_split: str,
    label: str,
    examples: list[list[str]],
) -> list[dict[str, object]]:
    return [
        {
            "id": f"{split_name}-{label}-{index}",
            "split": split_name,
            "input": row[0],
            "expected": {"label": label},
            "metadata": {
                "category": label,
                "source": f"clinc150_{source_split}",
            },
        }
        for index, row in enumerate(examples, start=1)
    ]


if __name__ == "__main__":
    main()
