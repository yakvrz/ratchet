# Benchmarks

Ratchet's sample suite is intentionally limited to public, trusted assessment vehicles. Samples should be useful for optimizer development, reproducible from public sources, and honest about what behavior they test.

Current samples:

- `samples/bfcl_function_calling_agent/`
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
- retrieval policy
- stateful or workflow decisions
- cost/latency tradeoffs
- multi-mechanism failures

## Open Benchmark Gaps

Ratchet still needs stronger public benchmarks for:

- grounded retrieval QA
- workflow/action agents

For grounded retrieval QA, candidates worth evaluating include KILT-style tasks and RAG-focused benchmarks with public corpora and clear provenance requirements.

For workflow/action agents, candidates worth evaluating include tau-bench style retail/airline tasks and browser/workflow environments such as WorkArena, depending on how much environment complexity Ratchet should support.

These should be added only after the adapter/eval shape is clear enough to avoid reintroducing synthetic samples under a different name.
