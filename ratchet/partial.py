from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ratchet.io import write_json


def write_partial_run_outputs(
    out_dir: Path,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    """Write lightweight diagnostics for an interrupted or failed run.

    Full Ratchet artifacts require completed baseline, frontier, and holdout
    summaries. When a run aborts earlier, progress.jsonl and case_results.jsonl
    are still enough to explain where it stopped and which cases were pending.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_rows = _read_jsonl(out_dir / "progress.jsonl")
    case_rows = _read_jsonl(out_dir / "case_results.jsonl")
    manifest = {
        "status": status,
        "reason": reason,
        "written_at": datetime.now(timezone.utc).isoformat(),
        "progress_path": str(out_dir / "progress.jsonl"),
        "case_results_path": str(out_dir / "case_results.jsonl"),
        "elapsed_s": max((float(row.get("elapsed_s") or 0.0) for row in progress_rows), default=0.0),
        "event_counts": dict(Counter(str(row.get("event") or "") for row in progress_rows)),
        "case_result_count": len(case_rows),
        "incomplete_cases": _incomplete_cases(progress_rows),
        "latest_events": progress_rows[-20:],
    }
    write_json(out_dir / "partial_run_manifest.json", manifest)
    (out_dir / "partial_report.md").write_text(_partial_report(manifest))
    return manifest


def _incomplete_cases(progress_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    started: dict[tuple[str, str, int], dict[str, Any]] = {}
    completed: set[tuple[str, str, int]] = set()
    for row in progress_rows:
        event = row.get("event")
        if event not in {"case_started", "case_completed"}:
            continue
        key = (
            str(row.get("patch_hash") or ""),
            str(row.get("case_id") or ""),
            int(row.get("sample_index") or 0),
        )
        if event == "case_started":
            started[key] = row
        else:
            completed.add(key)
    incomplete = []
    for key, row in started.items():
        if key in completed:
            continue
        incomplete.append(
            {
                "patch_hash": key[0],
                "case_id": key[1],
                "sample_index": key[2],
                "split": row.get("split"),
                "started_elapsed_s": row.get("elapsed_s"),
                "started_at": row.get("timestamp"),
            }
        )
    return sorted(
        incomplete,
        key=lambda item: (
            str(item.get("patch_hash") or ""),
            str(item.get("case_id") or ""),
            int(item.get("sample_index") or 0),
        ),
    )


def _partial_report(manifest: dict[str, Any]) -> str:
    event_counts = manifest.get("event_counts") or {}
    incomplete_cases = manifest.get("incomplete_cases") or []
    rows = [
        "# Ratchet Partial Run Report",
        "",
        f"Status: `{manifest.get('status')}`",
        f"Reason: {manifest.get('reason')}",
        f"Elapsed: {float(manifest.get('elapsed_s') or 0.0):.3f}s",
        f"Case results written: {int(manifest.get('case_result_count') or 0)}",
        "",
        "## Progress Events",
        "",
    ]
    for event, count in sorted(event_counts.items()):
        rows.append(f"- `{event}`: {count}")
    rows.extend(["", "## Incomplete Cases", ""])
    if not incomplete_cases:
        rows.append("No incomplete case evaluations were recorded.")
    else:
        for item in incomplete_cases[:50]:
            rows.append(
                "- "
                f"patch=`{item.get('patch_hash')}` "
                f"case=`{item.get('case_id')}` "
                f"sample={item.get('sample_index')} "
                f"started_elapsed={item.get('started_elapsed_s')}s"
            )
        if len(incomplete_cases) > 50:
            rows.append(f"- ... {len(incomplete_cases) - 50} more")
    rows.extend(
        [
            "",
            "## Files",
            "",
            f"- Progress: `{manifest.get('progress_path')}`",
            f"- Case results: `{manifest.get('case_results_path')}`",
        ]
    )
    return "\n".join(rows) + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows
