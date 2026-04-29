from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import urllib.request


BASE_URL = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main"
DOMAINS = ("airline", "retail", "telecom")
DEFAULT_COUNTS = {
    "airline": 12,
    "retail": 20,
    "telecom": 64,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tau-bench action-policy Ratchet assessment split.")
    parser.add_argument("--out", default="evals.assessment.jsonl")
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--airline-per-split", type=int, default=DEFAULT_COUNTS["airline"])
    parser.add_argument("--retail-per-split", type=int, default=DEFAULT_COUNTS["retail"])
    parser.add_argument("--telecom-per-split", type=int, default=DEFAULT_COUNTS["telecom"])
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "airline": args.airline_per_split,
        "retail": args.retail_per_split,
        "telecom": args.telecom_per_split,
    }
    rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        tasks = _load_json(cache_dir, domain, "tasks.json")
        tools_source = _load_text(cache_dir, domain, "tools.py", source=True)
        policy = _load_text(cache_dir, domain, "policy.md", required=False)
        tasks_by_id = {str(task["id"]): task for task in tasks}
        ordered_tasks = [tasks_by_id[task_id] for task_id in sorted(tasks_by_id, key=_task_sort_key)]
        required = counts[domain] * 3
        if len(ordered_tasks) < required:
            raise ValueError(f"Not enough tau-bench {domain} tasks for requested split counts.")
        tool_catalog = _tool_catalog(tools_source)
        for split_index, split in enumerate(("train", "dev", "holdout")):
            start = split_index * counts[domain]
            selected = ordered_tasks[start: start + counts[domain]]
            rows.extend(
                _case_row(
                    domain=domain,
                    split=split,
                    index=index + 1,
                    task=task,
                    tool_catalog=tool_catalog,
                    policy=policy,
                )
                for index, task in enumerate(selected)
            )
    out_path = root / args.out
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} cases to {out_path}")


def _load_json(cache_dir: Path, domain: str, name: str) -> object:
    path = cache_dir / domain / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/data/tau2/domains/{domain}/{name}"
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return json.loads(path.read_text())


def _load_text(cache_dir: Path, domain: str, name: str, *, source: bool = False, required: bool = True) -> str:
    path = cache_dir / domain / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if source:
            url = f"{BASE_URL}/src/tau2/domains/{domain}/{name}"
        else:
            url = f"{BASE_URL}/data/tau2/domains/{domain}/{name}"
        try:
            path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        except Exception:
            if required:
                raise
            return ""
    return path.read_text()


def _task_sort_key(task_id: str) -> tuple[int, str]:
    if task_id.isdigit():
        return (int(task_id), task_id)
    return (10_000, task_id)


def _tool_catalog(source: str) -> list[dict[str, str]]:
    tree = ast.parse(source)
    tools = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(_decorator_name(decorator) == "is_tool" for decorator in node.decorator_list):
            continue
        doc = ast.get_docstring(node) or ""
        summary = " ".join(line.strip() for line in doc.splitlines() if line.strip())
        tools.append({"name": node.name, "description": summary[:500]})
    return sorted(tools, key=lambda item: item["name"])


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _case_row(
    *,
    domain: str,
    split: str,
    index: int,
    task: dict[str, object],
    tool_catalog: list[dict[str, str]],
    policy: str,
) -> dict[str, object]:
    criteria = task.get("evaluation_criteria") if isinstance(task.get("evaluation_criteria"), dict) else {}
    expected_actions = [
        {"name": str(action.get("name")), "arguments": action.get("arguments") or {}}
        for action in criteria.get("actions") or []
        if isinstance(action, dict) and action.get("name")
    ]
    return {
        "id": f"{split}-{domain}-{index}",
        "split": split,
        "input": json.dumps(
            {
                "domain": domain,
                "task": _task_context(task),
                "available_tools": tool_catalog,
                "policy_excerpt": _compact_policy(policy),
            },
            sort_keys=True,
        ),
        "expected": {
            "actions": expected_actions,
            "communicate_info": criteria.get("communicate_info") or [],
            "nl_assertions": criteria.get("nl_assertions") or [],
            "reward_basis": criteria.get("reward_basis") or [],
        },
        "metadata": {
            "category": domain,
            "source": "tau2_bench",
            "source_id": str(task.get("id")),
            "expected_action_count": len(expected_actions),
        },
    }


def _task_context(task: dict[str, object]) -> dict[str, object]:
    scenario = task.get("user_scenario") if isinstance(task.get("user_scenario"), dict) else {}
    instructions = scenario.get("instructions") if isinstance(scenario.get("instructions"), dict) else {}
    description = task.get("description") if isinstance(task.get("description"), dict) else {}
    return {
        "purpose": description.get("purpose"),
        "reason_for_call": instructions.get("reason_for_call"),
        "known_info": instructions.get("known_info"),
        "unknown_info": instructions.get("unknown_info"),
        "task_instructions": instructions.get("task_instructions"),
        "ticket": task.get("ticket"),
    }


def _compact_policy(policy: str) -> str:
    text = re.sub(r"\s+", " ", policy).strip()
    return text[:3500]


if __name__ == "__main__":
    main()
