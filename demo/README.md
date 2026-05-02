# Ratchet Demo: Order Desk

This is Ratchet's maintained demo benchmark and release-gate workload. It is a local deterministic tool-loop task that is small enough to run during development while still exercising the core optimizer loop.

The environment exposes:

- a domain policy
- real tool schemas
- read-only and mutating tools
- hidden order state
- deterministic state-changing tool semantics
- local grading from final environment state

The optimizer must not contain Order Desk-specific logic. Candidate programs use the same surface primitives as any other tool-loop task: context sections, state, before-tool validation, after-tool-result state updates, response guards, and model config. The environment exposes generic tool result schemas so surface inference can derive identifier flows such as read-tool `order_id` observations feeding mutating-tool `order_id` arguments.

## Quick Start

Run the user-facing release check:

```bash
python -m ratchet release-check --config demo/ratchet.diagnostic_expanded.toml
```

Run the full diagnostic optimizer loop:

```bash
python -m ratchet optimize --config demo/ratchet.diagnostic_expanded.toml
```

Review the generated artifacts in `demo/results/diagnostic-expanded/`, especially `report.md`, `summary.html`, `events.jsonl`, `search_plans.jsonl`, `proposals.jsonl`, `evidence_ledger.json`, and `run_summary.json`.

The diagnostic config expects model credentials in the repository `.env` file. It uses the stronger optimizer model roles and a `gemini-2.5-flash` baseline agent.

## Maintenance

Generate evals:

```bash
python demo/generate_evals.py
```

Calibrate the live baseline:

```bash
python demo/calibrate_baseline.py --split dev
python demo/calibrate_baseline.py --split holdout
```

Run the smaller assessment config for cheaper development checks:

```bash
python -m ratchet check --config demo/ratchet.assessment.toml
python -m ratchet optimize --config demo/ratchet.assessment.toml
```

Run the larger diagnostic assessment when judging optimizer capability:

```bash
python demo/generate_expanded_evals.py
python -m ratchet eval-health --config demo/ratchet.diagnostic_expanded.toml --strict
python -m ratchet optimize --config demo/ratchet.diagnostic_expanded.toml
```

The expanded diagnostic set uses distinct generated tasks rather than repeated samples: 24 train, 48 dev, and 48 holdout cases, balanced across cancel, address, return, and ambiguity.

Development target:

- full reduced run under 5-10 minutes
- baseline success in a nontrivial range, not near 0% or 100%
- promoted candidates should be general surface programs, not task-id behavior
- reports should show failure-mode deltas for premature completion, wrong or uninspected objects, missing confirmation, unresolved ambiguity, and unsupported completion claims

Current calibration note:

- With `gemini-2.5-flash`, the development split sits in a useful middle band: the baseline can solve some trajectories but still fails on missing inspection, identifier grounding, and ambiguity.
- `gemini-2.5-flash-lite` is too weak for routine optimizer development on this sample; it currently behaves like a stress test rather than a calibration target.
- A useful Ratchet run should improve through general surface operations. For example, a valid result is a policy or tool-loop scaffold that forces authentication, list, inspect, then mutate, without encoding task IDs or hidden expected answers.
- If the baseline falls near 0% or 100% across repeated runs, recalibrate the task mix or baseline model before treating the assessment as a fine-grained optimizer comparison. It is still useful as an architecture regression test.
