# Ratchet Architecture

Ratchet is an eval-grounded optimizer for Python agents. It treats the eval as the specification and the agent policy as the artifact under optimization.

The core loop is:

```text
SurfaceSpec
  -> TransformProgram
  -> TransformCompiler
  -> CompiledCandidate
  -> ResearchState
  -> ExperimentIntent[]
  -> CandidateProposal[]
  -> EvidenceLedger
  -> MeasurementDecision
  -> FrontierUpdate
  -> HoldoutValidation
```

## Boundaries

Ratchet is deliberately not a repo-wide coding agent. It does not rewrite arbitrary source files. The adapter exposes a `SurfaceSpec`, and candidates are expressed as typed `TransformProgram`s compiled against that surface. The compiler, not the optimizer model, decides whether a candidate is legal, executable, and inside immutable boundaries.

The adapter owns:

- running the baseline and compiled candidate
- grading externally visible outputs
- describing executable optimization surfaces through `SurfaceSpec`
- exporting a selected compiled candidate into an inspectable artifact

Ratchet owns:

- eval loading and split protection
- baseline measurement
- surface validation
- transform compilation
- research planning
- candidate implementation validation and boundary enforcement
- staged measurement
- evidence accounting
- frontier and holdout promotion gates
- reporting

## Core Artifacts

`SurfaceSpec` is the executable optimization contract. It describes context sections, lifecycle hooks, typed state support, tool interaction capabilities, model-call controls, response interception, immutable boundaries, and safety constraints.

`TransformProgram` is the candidate artifact. It is a typed, hook-based program over the inferred surface, with operations for context construction, state updates, tool-call validation, model configuration, response handling, control flow, and instrumentation.

`TransformCompiler` is the deterministic safety boundary. It parses transform programs, validates hooks and capabilities, type-checks references, enforces immutable boundaries, lowers operations into executable runtime middleware, and emits an inspectable compile report and candidate diff.

`CompiledCandidate` is what adapters execute. It contains the original program, operations grouped by hook, compiler report, candidate diff, and runtime instrumentation plan.

`EvidencePacket` is deterministic symptom evidence extracted from eval results and diagnostics. It records observed runtime, output, tool, label, example-coverage, and cost/latency signals without deciding the causal theory.

`ResearchTheory` is model-authored causal state for the branch. It preserves the primary hypothesis, competing hypotheses, disconfirmed explanations, surprising observations, experiment opportunities, and falsification criteria.

`ResearchState` is the branch-local planning packet. It includes research theory, behavior profile, active affordances, budget state, prior experiment outcomes, and frontier context.

`ExperimentIntent` is planner output. It defines a research question, mechanism, target slices, required surfaces, measurements, success criteria, and disconfirming result. It must not contain transform program content.

`CandidateProposal` is implementer output. It contains a proposed `TransformProgram` plus metadata about hypothesis, expected cost, risk, and targeted failures.

`EvidenceLedger` is the measurement source of truth. It records paired candidate-vs-reference deltas, pass flips, invalid-output changes, cost/latency/token deltas, sample sizes, reliability signals, measurement cost, and baseline-instability flags.

`MeasurementDecision` is selector output. It chooses which already-valid candidates receive more measurement. It must not create candidates or alter experiment intents.

`DiagnosticTrace` is the adapter-owned behavior trace. For single-call tasks it may contain only raw output text and metadata. For interactive tasks it should contain `InteractionTurn`, `ToolCallTrace`, terminal state, and terminal reason. Ratchet treats these traces as evidence; it does not infer tool behavior from final text alone.

## Model Roles

The optimizer uses separate model roles even when they share the same configured model:

- diagnoser: labels failure modes from eval traces
- research theorist: turns deterministic evidence into causal hypotheses and experiment opportunities
- research planner: emits experiment intents only
- candidate implementer: emits candidate affordance applications only
- measurement selector: chooses measurements from evidence summaries

Role separation is a design invariant. If a role starts needing retries, repairs, or prompt patches to do another role's job, the architecture should be reconsidered rather than patched around.

## Deterministic Code vs Model Judgment

Deterministic code should enforce invariants:

- train/dev/holdout boundaries
- budget ceilings
- adapter and eval fingerprints
- schema validation
- transform compile validity
- surface capability validity
- output-contract preservation
- no holdout-guided search
- no task-specific proposal recipes

Model calls should provide judgment:

- failure interpretation
- research questions worth testing
- concrete transform programs within legal surfaces
- measurement value tradeoffs under evidence uncertainty

## Tool-Call Tasks

Tool calls are handled as an extension of Ratchet's evidence loop, not as a tau-bench-specific recipe.

- adapters execute the real environment loop and return structured trajectories
- behavior diagnostics summarize tool status, tool errors, premature stopping, turn counts, and tool-call counts
- surface specs expose meaningful tool/action capabilities such as tool selection policy, argument grounding, precondition policy, and interaction completion
- candidates compile into hook-based runtime middleware rather than source rewrites
- evidence and reports distinguish task score gains from extra model calls, tool calls, turns, latency, and measurement spend

Known public benchmark integrations should use the official simulator when available. The optional tau-bench bridge converts original `tau-bench` retail/airline results into Ratchet `RunRecord`s; static action-list proxies are useful only as development probes, not leaderboard-comparable tau-bench evaluation.

Hard-coded task recipes, fallback proposal generators, or model-bypass switches violate the architecture. Tests may use fakes, but production optimization should fail visibly when a model role cannot produce valid output.

## Measurement Semantics

Ratchet separates deployed-policy tradeoffs from measurement spend.

Deployed-policy metrics describe what a selected candidate would cost or how fast it would run per case.

Measurement budgets control development spend while evaluating candidates. Expensive model probes may still be measured when they provide useful frontier evidence, but deterministic code must not exceed configured measurement budgets.

For interactive agents, measurement budgets may also cap candidate tool calls and interaction turns. These caps control development spend and runaway trajectories; they do not replace deployed-policy cost, latency, or quality constraints.

Staged evaluation has distinct roles:

- smoke: reject crashes and contract violations
- small dev: triage whether more measurement is worthwhile
- full dev: first selection-quality comparison
- confirmation: stability check for suspicious or runtime-sensitive finalists
- holdout: protected final validation only

Directional or unstable candidates can be reported as frontier evidence, but only holdout-validated candidates are promoted as selected artifacts.

## Failure Policy

Ratchet should prefer fail-fast behavior over hidden compatibility paths. Minimal JSON syntax repair is acceptable for model truncation or formatting noise. Semantic contract violations should be logged as role failures, not silently converted into fallback candidates.
