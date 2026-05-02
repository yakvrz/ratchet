# Initial Release Gate

This gate is for the core product, not packaging polish. A release candidate must prove that Ratchet can run the full optimizer loop honestly and leave enough evidence to inspect its behavior.

## Required Checks

Run these before cutting a release:

```bash
python -m pytest -q
python -m ratchet release-check --config demo/ratchet.diagnostic_expanded.toml
python -m ratchet optimize --config demo/ratchet.diagnostic_expanded.toml
python -m ratchet assess-ideation --run-dir demo/results/diagnostic-expanded
```

Use a timestamped `--out` for release-candidate optimizer runs when preserving prior results matters.

## Pass Criteria

- Preflight passes with generated surfaces, materialization checks, and at least one dev and holdout case.
- Eval health is `healthy` under strict mode: no fatal issues and no warnings.
- Optimizer run completes without runtime, grader, compiler, or artifact-writing errors.
- `events.jsonl`, `progress.jsonl`, `search_plans.jsonl`, `proposals.jsonl`, `evidence_ledger.json`, `run_summary.json`, `candidate_metrics.json`, `report.md`, and `summary.html` are written.
- The report explains what Ratchet observed, planned, proposed, evaluated, rejected, and selected.
- If Ratchet promotes a candidate, the candidate must clear confirmation when required and protected holdout validation.
- If Ratchet keeps the baseline, the outcome must explain why, such as no valid candidates, no dev lift, confirmation regressions, holdout regression, or measurement budget exhaustion.

## Current Core-Product Standard

Order Desk is the release-candidate inner-loop benchmark because it is fast enough to iterate and still exercises tool calls, tool-result state, context transforms, response guarding, confirmation, and holdout protection.

tau-bench remains a secondary external benchmark. Raw tau-bench simulator runs are too slow for every release-candidate iteration; use them only after the Order Desk gate is healthy or after building a smaller tau-derived diagnostic harness.
