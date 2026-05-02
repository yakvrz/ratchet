# Ratchet Architecture

Ratchet is an eval-grounded optimizer for Python agents. It treats the eval as the specification and the agent policy as the artifact under optimization.

The core loop is:

```text
AgentHarness
  -> AdapterGenerator
  -> SurfaceSpec
  -> SurfaceOpportunity[]
  -> BaselineEvaluation
  -> EvidencePacket
  -> SearchPlan
  -> CandidateProposal[]
  -> TransformProgram
  -> TransformCompiler
  -> CompiledCandidate
  -> EvidenceLedger
  -> FrontierUpdate
  -> HoldoutValidation
```

## Boundaries

Ratchet is deliberately not a repo-wide coding agent. It does not rewrite arbitrary source files. A task-specific harness exposes model-request construction, output parsing, and grading. Ratchet's adapter generator turns that harness into an executable `SurfaceSpec` and runtime adapter. Candidates are expressed as typed `TransformProgram`s compiled against that surface. The compiler, not the optimizer model, decides whether a candidate is legal, executable, and inside immutable boundaries.

The harness owns:

- declaring task-specific model request construction
- parsing externally visible outputs
- grading externally visible outputs

The generated adapter owns:

- inferring and exporting `SurfaceSpec`
- applying compiled hook middleware
- executing model/tool loops through declared surfaces
- recording runtime diagnostics and transform instrumentation

Ratchet owns:

- adapter generation from the harness
- running the baseline and compiled candidate
- describing executable optimization surfaces through `SurfaceSpec`
- exporting a selected compiled candidate into an inspectable artifact
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

`AgentHarness` is the ingestion boundary. For single-call agents it declares how to build a `ModelRequest`, parse raw model output, and grade the resulting external output. It should not contain optimizer logic.

`AdapterGenerator` turns a harness into a runtime adapter. It infers a `SurfaceSpec`, invokes transform hooks, executes model calls, records instrumentation, and exports compiled artifacts.

`SurfaceSpec` is the executable optimization contract. It describes context sections, lifecycle hooks, typed state support, tool interaction capabilities, tool result shapes when adapters expose them, model-call controls, response interception, immutable boundaries, and safety constraints. For tool-loop agents it may also expose structural affordances, such as identifier flows from read-tool results into mutating-tool arguments.

`SurfaceOpportunity` is the deterministic planning view of `SurfaceSpec`. It names an editable target or structural affordance, the DSL operations legal on that target, expected measurement axes, suitability evidence, and cost/risk hints. It is not a benchmark recipe and it does not duplicate the surface contract.

`TransformProgram` is the candidate artifact. It is a typed, hook-based program over the inferred surface, with operations for context construction, state updates, tool-call validation, model configuration, response handling, control flow, and instrumentation.

`TransformCompiler` is the deterministic safety boundary. It parses transform programs, validates hooks and capabilities, type-checks references, enforces immutable boundaries, lowers operations into executable runtime middleware, and emits an inspectable compile report and candidate diff.

`CompiledCandidate` is what adapters execute. It contains the original program, operations grouped by hook, compiler report, candidate diff, and runtime instrumentation plan.

`EvidencePacket` is deterministic symptom evidence extracted from eval results and diagnostics. It records observed runtime, output, tool, label, example-coverage, and cost/latency signals without deciding the search plan.

`SearchPlan` is the single model-authored planning artifact for a parent branch. It contains diagnosis summary, hypotheses, target mechanisms, cited surface opportunity IDs, prior evidence context, and candidate briefs. It must not contain transform program content.

`SearchBrief` is a typed candidate brief inside a `SearchPlan`. It names the mechanism, target slice, required surfaces, expected measurements, success criteria, and disconfirming result for one proposal path.

`CandidateProposal` is implementer output. It contains a proposed `TransformProgram`, citations to the surface opportunities it uses, and metadata about hypothesis, expected cost, risk, and targeted failures.

`EvidenceLedger` is the measurement source of truth. It records paired candidate-vs-reference deltas, pass flips, invalid-output changes, cost/latency/token deltas, sample sizes, reliability signals, measurement cost, and baseline-instability flags.

`DiagnosticTrace` is the adapter-owned behavior trace. For single-call tasks it may contain only raw output text and metadata. For interactive tasks it should contain `InteractionTurn`, `ToolCallTrace`, terminal state, and terminal reason. Ratchet treats these traces as evidence; it does not infer tool behavior from final text alone.

## Division of Labor

The architecture has one ownership chain: surface inference defines what can be changed, search planning decides what is worth trying, candidate implementation expresses the change as a typed program, compilation decides legality, runtime executes only compiled middleware, and deterministic staged evaluation decides whether evidence justifies more budget or holdout validation. Components should not cross those boundaries.

The main modules follow that split:

- `surfaces.py`, `surface_opportunities.py`: inferred editable substrate and deterministic opportunity view
- `experiments.py`, `research.py`, `research_payloads.py`: evidence packets, search plans, search briefs, and model-facing planner payloads
- `candidates.py`, `proposals.py`: candidate data model and implementer role
- `transform_program.py`, `transform_compiler.py`, `transform_validation.py`, `runtime.py`: typed DSL, legality checks, compilation, and hook execution
- `surface_search.py`, `transform_results.py`, `evidence_ledger.py`, `objectives.py`: transform context identity, result summaries, paired evidence, and promotion gates
- `optimizer.py`, `reporting.py`, `results.py`: orchestration, artifacts, and reporting

## Model Roles

The optimizer uses two model roles even when they share the same configured model:

- search planner: reads objective, parent summary, evidence packet, surface opportunities, prior proposal/evidence summaries, and remaining budget, then emits a typed `SearchPlan`
- candidate implementer: emits typed transform programs that cite `SearchPlan` briefs and surface opportunities

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

- failure interpretation and search questions worth testing
- concrete transform programs within legal surfaces

Measurement selection is deterministic rather than model-authored. `smoke` evaluates compiled candidates that fit budget, `small_dev` screens in proposal order with one candidate per comparison group, `full_dev` requires positive objective signal from small-dev, `confirmation` checks unstable or runtime-sensitive finalists, and `holdout` is protected validation for selected dev finalists only.

## Tool-Call Tasks

Tool calls are handled as an extension of Ratchet's evidence loop, not as a tau-bench-specific recipe.

- `GeneratedToolLoopAdapter` executes the real environment loop and returns structured trajectories when Ratchet needs to optimize tool-use behavior
- behavior diagnostics summarize tool status, tool errors, premature stopping, turn counts, and tool-call counts
- surface specs expose meaningful tool/action capabilities such as tool selection policy, argument grounding, precondition policy, and interaction completion
- candidates compile into hook-based runtime middleware rather than source rewrites
- evidence and reports distinguish task score gains from extra model calls, tool calls, turns, latency, and measurement spend

Benchmark integrations should use the real evaluator and environment interface. If Ratchet is expected to optimize the harness, it must own the agent loop through a general adapter surface; black-box benchmark runners are only measurement bridges.

Hard-coded task recipes, fallback proposal generators, or model-bypass switches violate the architecture. Tests may use fakes, but production optimization should fail visibly when a model role cannot produce valid output.

## Measurement Semantics

Ratchet separates deployed-policy tradeoffs from measurement spend.

Deployed-policy metrics describe what a selected candidate would cost or how fast it would run per case.

Measurement budgets control development spend while evaluating candidates. Expensive model probes may still be measured when they provide useful frontier evidence, but deterministic code must not exceed configured measurement budgets.

For interactive agents, measurement budgets may also cap candidate tool calls and interaction turns. These caps control development spend and runaway trajectories; they do not replace deployed-policy cost, latency, or quality constraints.

Staged evaluation has distinct roles:

- smoke: reject crashes and contract violations
- small dev: triage whether more measurement is worthwhile; correctness runs require positive score/pass signal before full dev
- full dev: first selection-quality comparison
- confirmation: stability check for suspicious or runtime-sensitive finalists
- holdout: protected final validation only

Directional or unstable candidates can be reported as frontier evidence, but only holdout-validated candidates are promoted as selected artifacts.

Composed structural candidates may include ablations when proposal budget remains after primary candidates are queued. Ablations are measured as comparison evidence; they should not displace the primary composed scaffold in a constrained proposal batch.

## Failure Policy

Ratchet should prefer fail-fast behavior over hidden compatibility paths. Minimal JSON syntax repair is acceptable for model truncation or formatting noise. Semantic contract violations should be logged as role failures, not silently converted into fallback candidates.
