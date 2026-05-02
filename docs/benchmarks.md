# Demo Benchmark

Ratchet ships one maintained demo benchmark: [demo/](../demo/).

The demo is a local deterministic order-desk tool-loop environment. It includes a model-visible policy, tool schemas, read tools, mutating tools, hidden order state, deterministic state-changing tool semantics, and grading from final environment state.

This is the benchmark we use for the core product release gate because it is small enough to run during development but still exercises the behavior Ratchet is supposed to optimize:

- context changes
- tool-call sequencing
- tool-result state tracking
- inspect-before-mutate behavior
- response guarding
- confirmation and holdout validation
- cost, token, turn, and latency accounting

The demo is not a public leaderboard claim. It is a product-quality diagnostic benchmark for Ratchet itself. A credible run should show Ratchet discovering task-agnostic transform programs through declared surfaces, not hardcoded task IDs, hidden answers, or benchmark recipes.

## Why Only One Demo Ships

The repository intentionally does not ship a broad examples directory. Earlier internal probes were useful during development, but shipping many half-maintained examples makes the product harder to evaluate and support.

Additional benchmarks should be added only when they become first-class release gates with:

- a clear user-facing purpose
- maintained eval data or reproducible generation
- documented run commands
- preflight and eval-health coverage
- optimizer artifact review expectations
- no task-specific optimizer shortcuts

Until then, keep them outside the shipped tree.
