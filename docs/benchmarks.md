# Benchmarks

Ratchet's sample suite is intentionally limited to public, trusted assessment vehicles. Samples should be useful for optimizer development, reproducible from public sources, and honest about what behavior they test.

Current samples:

- `samples/bfcl_function_calling_agent/`
- `samples/taubench_official_agent/`
- `samples/taubench_action_agent/`
- `samples/banking77_intent_agent/`
- `samples/clinc150_intent_agent/`

## BFCL Function Calling

BFCL is the primary Ratchet development benchmark.

It is useful because it exercises behavior that a policy optimizer should actually improve:

- output contract and JSON structure
- function and argument selection
- model capability probes
- runtime/output-cap defects
- few-shot examples
- cost and latency tradeoffs

Good Ratchet behavior on BFCL means discovering mechanism-relevant candidates, preserving useful quality-frontier information even when deployed cost is high, and validating real holdout gains without using holdout feedback during search.

BFCL is not a full leaderboard run in this repo. The sample is a fixed development assessment split large enough to inspect optimizer behavior while remaining affordable and reproducible.

## Official tau-bench

The official tau-bench sample uses `GeneratedToolLoopAdapter`, so Ratchet owns the agent loop: model calls, tool-call parsing, `before_tool_call`, `after_tool_result`, response interception, state, traces, and transform instrumentation. The external `sierra-research/tau-bench` package supplies the user simulator, environment state, tool schemas, and official reward.

That split is intentional. The benchmark connector may know how to create a tau-bench environment, but the optimizer does not get tau-specific candidate logic. Candidate programs still compile against the same hook-based surface used by any other interactive tool environment.

A credible tau-bench result should compare baseline and Ratchet-optimized runs with the same agent model, user simulator, task set, trial count, and inference budget. The report should include held-out success, failure-mode deltas, tool/model/turn cost deltas, promoted transform diffs, and immutable-boundary evidence.

## tau-bench Static Action Proxy

The tau-bench action sample is currently a workflow/action-policy development probe, not the official interactive benchmark.

It is built from the original public `sierra-research/tau-bench` retail and airline task files. The adapter does not run the full interactive tau-bench simulator. Instead, it turns each public task into an action-policy planning case: the agent sees task context, a compact tool catalog, and a policy excerpt, then predicts the required workflow action names.

This is useful for:

- tool/action selection
- policy-constrained workflow planning
- multi-domain customer-service tasks
- output-contract behavior
- model capability probes
- few-shot and instruction improvements

It should be interpreted as a Ratchet development assessment, not an official tau-bench leaderboard result. Official tau-bench evaluation requires the simulator, user model, domain state, and environment dynamics.

Ratchet now has the core substrate needed for faithful tau-style adapters:

- `GeneratedToolLoopAdapter` executes the real model/tool/environment loop under Ratchet-owned middleware hooks.
- `DiagnosticTrace` can record multi-turn trajectories, tool calls, terminal state, and terminal reason.
- `InteractionRecorder` helps adapters produce those traces without custom bookkeeping.
- tool/action affordances can be generated from observed trajectory failures.
- measurement budgets can cap candidate dollars, tool calls, and turns.
- `ratchet.benchmarks.taubench` provides a legacy bridge for converting original tau-bench retail/airline results into `RunRecord`s when the external `tau-bench` package is installed.

## BANKING77

BANKING77 is a secondary classifier-style probe.

It is useful for:

- label-boundary failures
- few-shot behavior
- semantic rewrite behavior
- model-capability comparisons
- eval stability under a stable taxonomy

It is not a flagship Ratchet benchmark. Single-label intent classification over a fixed taxonomy can often be better served by a fine-tuned encoder model than by an LLM agent. BANKING77 should not be overinterpreted as proof that Ratchet is useful for agent policy optimization.

Good Ratchet behavior on BANKING77 means clear evidence accounting, cautious promotion, and honest reporting when gains are directional, unstable, or too sample-sensitive.

## CLINC150

CLINC150 is also a secondary classifier-style probe, with broader label diversity and out-of-scope behavior.

It is useful for:

- out-of-scope boundary handling
- confusable intent clusters
- few-shot and prompt-boundary experiments
- comparing behavior against BANKING77 when classification evidence is noisy

Like BANKING77, CLINC150 should not be treated as the central product demonstration. It is a stability and classifier-behavior probe.

## Benchmark Policy

Keep benchmarks public, trusted, and reproducible. Do not add private local demos, synthetic one-off tasks, or sample-specific "success stories" to the main sample suite.

Removed samples should stay removed unless they are replaced by a public benchmark with a clear role. A benchmark should be added only if it tests a capability Ratchet is meant to optimize:

- configurable behavior surface
- output contract
- tool policy
- stateful or workflow decisions
- cost/latency tradeoffs
- multi-mechanism failures
