from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request


BASE_URL = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"
DEFAULT_CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BFCL Ratchet assessment split.")
    parser.add_argument("--out", default="evals.assessment.jsonl")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--train-total", type=int, default=96)
    parser.add_argument("--dev-total", type=int, default=96)
    parser.add_argument("--holdout-total", type=int, default=96)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases_by_category = {
        category: _load_category(cache_dir, category)
        for category in args.categories
    }
    for split, total in [
        ("train", args.train_total),
        ("dev", args.dev_total),
        ("holdout", args.holdout_total),
    ]:
        counts = _counts_by_category(args.categories, total)
        for category in args.categories:
            offset = _split_offset(split, counts[category])
            selected = cases_by_category[category][offset: offset + counts[category]]
            if len(selected) < counts[category]:
                raise ValueError(f"Not enough BFCL {category!r} cases for {split}.")
            rows.extend(
                _case_row(
                    split=split,
                    category=category,
                    index=index + 1,
                    source=row,
                )
                for index, row in enumerate(selected)
            )
    out_path = root / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} cases to {out_path}")


def _load_category(cache_dir: Path, category: str) -> list[dict[str, object]]:
    questions = _load_jsonl(cache_dir, f"BFCL_v3_{category}.json", f"{BASE_URL}/BFCL_v3_{category}.json")
    answers = _load_jsonl(
        cache_dir,
        f"possible_answer_BFCL_v3_{category}.json",
        f"{BASE_URL}/possible_answer/BFCL_v3_{category}.json",
    )
    answers_by_id = {str(row["id"]): row for row in answers}
    paired = []
    for question in questions:
        case_id = str(question.get("id") or "")
        answer = answers_by_id.get(case_id)
        if not answer:
            continue
        paired.append(
            {
                "id": case_id,
                "question": _question_text(question.get("question")),
                "functions": question.get("function", []),
                "ground_truth": answer.get("ground_truth", []),
            }
        )
    return paired


def _load_jsonl(cache_dir: Path, filename: str, url: str) -> list[dict[str, object]]:
    path = cache_dir / filename
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _question_text(raw_question: object) -> str:
    if not isinstance(raw_question, list):
        return str(raw_question or "")
    messages = []
    for turn in raw_question:
        if isinstance(turn, list):
            for message in turn:
                if isinstance(message, dict) and message.get("role") == "user":
                    messages.append(str(message.get("content") or ""))
    return "\n".join(messages)


def _counts_by_category(categories: list[str], total: int) -> dict[str, int]:
    base = total // len(categories)
    remainder = total % len(categories)
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(categories)
    }


def _split_offset(split: str, count: int) -> int:
    if split == "train":
        return 0
    if split == "dev":
        return count
    if split == "holdout":
        return count * 2
    raise ValueError(f"Unknown split {split!r}.")


def _case_row(*, split: str, category: str, index: int, source: dict[str, object]) -> dict[str, object]:
    return {
        "id": f"{split}-{category}-{index}",
        "split": split,
        "input": json.dumps(
            {
                "question": source["question"],
                "functions": source["functions"],
            },
            sort_keys=True,
        ),
        "expected": {
            "ground_truth": source["ground_truth"],
        },
        "metadata": {
            "category": category,
            "source": "bfcl_v3",
            "source_id": source["id"],
        },
    }


if __name__ == "__main__":
    main()
