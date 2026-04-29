# tau-bench Action Assessment

This sample is Ratchet's tau-bench-style assessment task for workflow/action-policy planning.

It builds a fixed assessment split from the original public `sierra-research/tau-bench` task files across:

- `airline`
- `retail`

The adapter asks the agent to predict the required workflow action names from the public task context, tool catalog, and policy excerpt. It is not the full tau-bench interactive simulator; it is an affordable Ratchet development assessment focused on action-policy reasoning, tool choice, output contract, model capability, examples, and runtime tradeoffs.

Build the local eval file. The default assessment split writes 48 proposal-safe train examples, 48 protected dev cases, and 48 protected holdout cases:

```bash
python3 samples/taubench_action_agent/prepare_evals.py
```

Run a preflight check:

```bash
python3 -m ratchet check --config samples/taubench_action_agent/ratchet.assessment.toml
```

Run the optimizer assessment:

```bash
python3 -m ratchet optimize --config samples/taubench_action_agent/ratchet.assessment.toml
```

This sample should be interpreted as a public workflow/action-policy probe, not as a replacement for official tau-bench leaderboard evaluation.

Ratchet's tool-call substrate now supports real interactive adapters through structured `DiagnosticTrace` trajectories and the optional original tau-bench bridge in `ratchet.benchmarks.taubench`. The static action probe remains useful for cheap optimizer development, but official tau-bench work should run the simulator and record tool/environment traces rather than flattening tasks into final action lists.
