# Ratchet Alpha Whitepaper

## Abstract

Ratchet is an evaluation-backed optimizer for Python agent harnesses. Given a Python agent, a current eval set, and a declared search surface, Ratchet diagnoses failures, proposes bounded harness changes, and promotes only candidates that clear a strict holdout gate. If no candidate safely improves on the baseline, Ratchet keeps the baseline unchanged.

The alpha release focuses on a narrow but defensible claim: Ratchet can optimize Python agent harnesses within a bounded executable mutation surface that includes prompts, tool descriptions, components, retrieval settings, model choice, and declared source-level Python hooks. It does not rewrite arbitrary repositories or change the benchmark's external contract.

This whitepaper documents the alpha system design, optimization loop, release scope, and live proof suite.

## 1. Problem

Agent tuning is usually manual, expensive, and difficult to audit. Teams change prompts, tools, retrieval settings, and model configurations by hand, then inspect a handful of examples and hope the new harness is better. That workflow has three common failure modes:

- silent regressions on cases that are no longer sampled
- lower-quality “optimizations” that save tokens by degrading behavior
- improvements that are hard to attribute because multiple parts changed at once

Ratchet addresses this by treating harness optimization as a guarded search problem:

- the external contract is fixed by the eval harness
- the internal harness surface is explicit and machine-editable
- every accepted change must survive full evaluation against the current holdout gate

## 2. Product Definition

Ratchet is a Python-first harness optimizer with the following contract:

- input: a Python agent harness exposed through an adapter, plus evals
- editable surface: only adapter-declared knobs and artifacts
- output: either a promoted optimized candidate or the original baseline

Ratchet is intentionally conservative. It is designed to prove safe wins, not to maximize search freedom.

## 3. Alpha Scope

The alpha release supports:

- Python agents only
- adapter-defined evals and grading
- optimization over declared enum knobs, text artifacts, components, and bounded code artifacts
- a separate optimizing model from the model being optimized
- strict holdout-backed promotion

The alpha release does not support:

- arbitrary repo-wide code rewriting
- language-agnostic attachment
- benchmark-free optimization
- production-trace ingestion as a required input

## 4. Contract Model

Ratchet separates three concerns:

1. External contract  
   The benchmark defines the agent’s task in terms of inputs, externally visible outputs, and scoring.

2. Internal harness contract  
   The adapter defines the editable harness surface: prompts, tool toggles, tool descriptions, retrieval settings, model choice, output caps, components, and bounded hook code.

3. Diagnostic trace  
   Tool calls, raw outputs, and metadata are used for diagnosis and proposal generation, but not for grading.

This separation matters because Ratchet’s graders evaluate external behavior, while the optimizer mutates only the internal harness surface.

## 5. System Architecture

Ratchet’s alpha architecture has five main layers.

### 5.1 Adapter Layer

Each agent is attached through a Python adapter that implements:

- `baseline()`
- `search_space()`
- `run_case(candidate, case)`
- `grade(case, output)`
- `export(candidate, out_dir)`

This keeps Ratchet neutral with respect to the harness implementation while still requiring an explicit optimization boundary.

### 5.2 Search Surface

The alpha search surface includes four artifact classes:

- `EnumKnobSpec`
- `TextArtifactSpec`
- `ComponentSpec`
- `CodeArtifactSpec`

The key alpha expansion over earlier versions is bounded source-level code mutation. `CodeArtifactSpec` exposes named Python hook points with:

- fixed callable name and signature
- bounded size
- dependency rules
- restricted execution via compilation and validation

This allows Ratchet to mutate harness logic without becoming an unconstrained code-rewriting system.

### 5.3 Diagnoser

Ratchet diagnoses failures into structured categories such as:

- `fallback_behavior`
- `retrieval_scope`
- `missing_tool`
- `output_contract`
- `arithmetic`

The diagnoser uses rule-based handling first and a stronger optimizing model when needed. The optimizing side is configured separately from the optimized harness and defaults to `gpt-5.4` with `medium` reasoning.

### 5.4 Proposer

The proposer emits bounded patch proposals using operations such as:

- `set_enum`
- `rewrite_text`
- `set_component`
- `rewrite_code`

The proposal loop is budgeted and validated. Ratchet rejects malformed or over-broad proposals before evaluation.

### 5.5 Gate

Ratchet’s gate is the core safety mechanism.

On `dev`, Ratchet can accept:

- behavior-improving candidates
- efficiency-improving candidates that preserve behavior

On `holdout`, Ratchet promotes only candidates that satisfy the final contract:

- quality non-inferior or better
- efficiency improved
- latency within guard

If no candidate clears that gate, Ratchet keeps baseline.

## 6. Optimization Loop

The alpha loop is:

1. Evaluate the current baseline on the current eval set.
2. Cluster failures into diagnosis buckets.
3. Generate bounded structural proposals.
4. Evaluate proposals on `dev`.
5. Accept only monotone `dev` improvements.
6. Validate accepted candidates on `holdout`.
7. Promote only holdout-clearing candidates.

This gives Ratchet a simple property:

- accepted `dev` improvements shape the search path
- the final shipped result is still constrained by the stricter holdout gate

That is why Ratchet can both discover useful stepping stones and still refuse to overclaim.

## 7. Proof Suite

The alpha proof suite consists of three external Python-agent integrations plus one smaller smoke benchmark.

### 7.1 Flagship: Python API Grounding

Task shape:

- grounded QA / retrieval
- includes supported and unsupported questions
- requires explicit `unknown` behavior on unsupported cases

Live result:

- baseline holdout mean score: `0.800`
- selected holdout mean score: `1.000`
- baseline holdout avg cost: `$0.004296`
- selected holdout avg cost: `$0.000908`
- baseline holdout avg tokens: `1250.0`
- selected holdout avg tokens: `940.6`
- baseline holdout median latency: `3.60s`
- selected holdout median latency: `2.11s`

This benchmark is the strongest alpha proof because it includes a successful bounded code-artifact behavior fix. Ratchet enabled a validator component and rewrote the post-answer validator hook to force `unknown` unless the chosen answer was grounded in retrieved evidence.

### 7.2 Structured Decision: Policy Triage

Task shape:

- structured decision over a frozen policy corpus
- objective grading on decision and amount fields

Live result:

- Ratchet improved `dev` from `4` passing cases to `6`
- holdout remained unchanged
- final decision: keep baseline

This benchmark is important because it demonstrates a non-promoted run with real search activity. Ratchet did work, found `dev` gains, and still refused to promote because holdout did not improve.

### 7.3 Procedural Agent: Runbook Action

Task shape:

- tool-using procedural next-step selection
- grounded action choice over a frozen runbook corpus

Live result:

- baseline holdout mean score: `1.000`
- selected holdout mean score: `1.000`
- baseline holdout avg cost: `$0.003118`
- selected holdout avg cost: `$0.001905`
- baseline holdout avg tokens: `764.7`
- selected holdout avg tokens: `478.7`
- baseline holdout median latency: `2.75s`
- selected holdout median latency: `2.72s`

This benchmark demonstrates a second live promoted win, this time as an efficiency-preserving optimization.

### 7.4 Smoke Benchmark: Public Docs QA

This remains a secondary smoke / efficiency benchmark rather than the flagship proof.

## 8. Alpha Results Summary

The alpha proof suite now demonstrates the pattern Ratchet needed:

- two live promoted wins
- one live honest baseline-kept outcome
- three external Python-agent task shapes
- at least one successful bounded code-artifact proposal on a non-toy agent

That is enough to support the alpha claim:

> Ratchet is an evaluation-backed optimizer for Python agent harnesses.

It is also strong evidence toward the broader claim:

> Ratchet can safely self-optimize Python agent harnesses within a bounded executable mutation surface.

## 9. Why This Matters

Ratchet’s novelty is not a radically new search algorithm. Its value is in product discipline:

- explicit contract separation
- bounded executable proposal surface
- separate optimizing model and optimized model
- strict holdout-backed promotion
- auditable artifacts for every run

The result is a system that can say one of two things with evidence:

- “this harness is now better”
- “this change did not generalize, so baseline is safer”

Both outcomes are useful.

## 10. Limitations

The alpha release still has clear boundaries.

- Ratchet is limited to Python-agent harnesses with source access or an adapter.
- The executable mutation surface is bounded; arbitrary repo rewriting is out of scope.
- The broader “any Python agent” claim is valid only within that bounded surface.
- Benchmarks still define what kinds of improvements are visible.

These are deliberate constraints, not accidental omissions. They keep Ratchet attributable, reproducible, and safe enough to ship.

## 11. Conclusion

Ratchet alpha is ready as a focused product:

- a Python-agent harness optimizer
- evaluation-backed and holdout-gated
- capable of prompt, tool, component, and bounded code mutation
- proven on a live external proof suite with both promoted and non-promoted outcomes

The alpha should be positioned narrowly and confidently. It already does the important thing: it turns agent optimization from informal craft into a controlled, evidence-backed loop.
