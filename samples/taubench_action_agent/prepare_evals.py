from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import urllib.request


BASE_URL = "https://raw.githubusercontent.com/sierra-research/tau-bench/main"
DOMAINS = ("retail", "airline")
DEFAULT_COUNTS = {
    "retail_train": 48,
    "retail_dev": 48,
    "airline_holdout": 48,
}
TOOL_FILES = {
    "retail": [
        "calculate",
        "cancel_pending_order",
        "exchange_delivered_order_items",
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
        "get_order_details",
        "get_product_details",
        "get_user_details",
        "list_all_product_types",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
        "return_delivered_order_items",
        "think",
        "transfer_to_human_agents",
    ],
    "airline": [
        "book_reservation",
        "calculate",
        "cancel_reservation",
        "get_reservation_details",
        "get_user_details",
        "list_all_airports",
        "search_direct_flight",
        "search_onestop_flight",
        "send_certificate",
        "think",
        "transfer_to_human_agents",
        "update_reservation_baggages",
        "update_reservation_flights",
        "update_reservation_passengers",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an original tau-bench action-policy Ratchet assessment split.")
    parser.add_argument("--out", default="evals.assessment.jsonl")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--retail-train", type=int, default=DEFAULT_COUNTS["retail_train"])
    parser.add_argument("--retail-dev", type=int, default=DEFAULT_COUNTS["retail_dev"])
    parser.add_argument("--airline-holdout", type=int, default=DEFAULT_COUNTS["airline_holdout"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    retail_train = _load_tasks(cache_dir, "retail", "tasks_train.py", "TASKS_TRAIN")[: args.retail_train]
    retail_test = _load_tasks(cache_dir, "retail", "tasks_test.py", "TASKS_TEST")[: args.retail_dev]
    airline_test = _load_tasks(cache_dir, "airline", "tasks_test.py", "TASKS")[: args.airline_holdout]
    rows: list[dict[str, object]] = []
    rows.extend(
        _case_row(
            domain="retail",
            split="train",
            index=index + 1,
            task=task,
            tool_catalog=_tool_catalog(cache_dir, "retail"),
            policy=_load_text(cache_dir, "retail", "wiki.md"),
        )
        for index, task in enumerate(retail_train)
    )
    rows.extend(
        _case_row(
            domain="retail",
            split="dev",
            index=index + 1,
            task=task,
            tool_catalog=_tool_catalog(cache_dir, "retail"),
            policy=_load_text(cache_dir, "retail", "wiki.md"),
        )
        for index, task in enumerate(retail_test)
    )
    rows.extend(
        _case_row(
            domain="airline",
            split="holdout",
            index=index + 1,
            task=task,
            tool_catalog=_tool_catalog(cache_dir, "airline"),
            policy=_load_text(cache_dir, "airline", "wiki.md"),
        )
        for index, task in enumerate(airline_test)
    )

    out_path = root / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} cases to {out_path}")


def _load_tasks(cache_dir: Path, domain: str, filename: str, symbol: str) -> list[dict[str, object]]:
    source = _load_source(cache_dir, f"tau_bench/envs/{domain}/{filename}")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == symbol for target in node.targets):
            return [_task_from_call(item) for item in node.value.elts if isinstance(item, ast.Call)]
    raise ValueError(f"Could not find {symbol} in {domain}/{filename}")


def _task_from_call(node: ast.Call) -> dict[str, object]:
    values = {keyword.arg: _literal_or_actions(keyword.value) for keyword in node.keywords if keyword.arg}
    return {
        "user_id": values.get("user_id", ""),
        "instruction": values.get("instruction", ""),
        "actions": values.get("actions", []),
        "outputs": values.get("outputs", []),
    }


def _literal_or_actions(node: ast.AST) -> object:
    if isinstance(node, ast.List):
        return [_literal_or_actions(item) for item in node.elts]
    if isinstance(node, ast.Call):
        values = {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords if keyword.arg}
        if getattr(node.func, "id", "") == "Action":
            return {"name": values.get("name", ""), "arguments": values.get("kwargs", {})}
        return values
    return ast.literal_eval(node)


def _tool_catalog(cache_dir: Path, domain: str) -> list[dict[str, str]]:
    rows = []
    for name in TOOL_FILES[domain]:
        source = _load_source(cache_dir, f"tau_bench/envs/{domain}/tools/{name}.py")
        tree = ast.parse(source)
        description = ""
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                description = ast.get_docstring(node) or ""
                break
        rows.append({"name": name, "description": " ".join(description.split())[:500]})
    return rows


def _load_text(cache_dir: Path, domain: str, filename: str) -> str:
    return _load_source(cache_dir, f"tau_bench/envs/{domain}/{filename}")


def _load_source(cache_dir: Path, path: str) -> str:
    cached = cache_dir / path
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{path}"
        cached.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return cached.read_text()


def _case_row(
    *,
    domain: str,
    split: str,
    index: int,
    task: dict[str, object],
    tool_catalog: list[dict[str, str]],
    policy: str,
) -> dict[str, object]:
    actions = task["actions"] if isinstance(task.get("actions"), list) else []
    expected_actions = [
        {"name": str(action.get("name")), "arguments": action.get("arguments") or {}}
        for action in actions
        if isinstance(action, dict) and action.get("name")
    ]
    return {
        "id": f"{split}-{domain}-{index}",
        "split": split,
        "input": json.dumps(
            {
                "domain": domain,
                "task": {
                    "user_id": task.get("user_id"),
                    "instruction": task.get("instruction"),
                },
                "available_tools": tool_catalog,
                "policy_excerpt": _compact_policy(policy),
            },
            sort_keys=True,
        ),
        "expected": {
            "actions": expected_actions,
            "outputs": task.get("outputs") if isinstance(task.get("outputs"), list) else [],
        },
        "metadata": {
            "category": domain,
            "source": "tau_bench",
            "source_id": f"{domain}:{split}:{index}",
            "expected_action_count": len(expected_actions),
        },
    }


def _compact_policy(policy: str) -> str:
    return " ".join(policy.split())[:3500]


if __name__ == "__main__":
    main()
