# Ratchet Architecture

Ratchet is an eval-grounded optimizer for Python agents. It treats the eval as the specification and the agent policy as the artifact under optimization.

The core loop is:

```text
AgentSpec
  -> EditableTarget[]
  -> OptimizationAffordance[]
  -> ResearchState
  -> ExperimentIntent[]
  -> CandidateProposal[]
  -> EvidenceLedger
  -> MeasurementDecision
  -> FrontierUpdate
  -> HoldoutValidation
```

## Boundaries

Ratchet is deliberately not a repo-wide coding agent. It does not rewrite arbitrary source files. The adapter exposes an `AgentSpec`, Ratchet derives a bounded optimization surface from that spec, and candidates are expressed as `AgentPatch` operations against that surface.

The adapter owns:

- running the baseline and patched agent
- grading externally visible outputs
- describing the current agent through `AgentSpec`
- exporting a selected patch into an inspectable artifact

Ratchet owns:

- eval loading and split protection
- baseline measurement
- editable target generation
- optimization affordance generation
- research planning
- candidate implementation validation
- staged measurement
- evidence accounting
- frontier and holdout promotion gates
- reporting

## Core Artifacts

`EditableTarget` is the low-level edit handle generated from `AgentSpec`: an instruction, model choice, runtime setting, output contract, tool policy, or few-shot bank.

`OptimizationAffordance` is the primary optimizer surface. It names one meaningful legal move, including the family, mechanism, target, allowed operations, expected measurements, risk, composition guidance, suitability, and evidence. Planner and implementer prompts should reason over affordances, not raw source files or arbitrary string targets.

`ResearchState` is the branch-local evidence packet. It includes task theory, behavior profile, active affordances, budget state, prior experiment outcomes, and frontier context.

`ExperimentIntent` is planner output. It defines a research question, mechanism, target slices, allowed affordance IDs, measurements, success criteria, and disconfirming result. It must not contain patch content.

`CandidateProposal` is implementer output. It applies one or more cited affordances through concrete operations or proposal-safe few-shot selections.

`EvidenceLedger` is the measurement source of truth. It records paired candidate-vs-reference deltas, pass flips, invalid-output changes, cost/latency/token deltas, sample sizes, reliability signals, measurement cost, and baseline-instability flags.

`MeasurementDecision` is selector output. It chooses which already-valid candidates receive more measurement. It must not create candidates or alter experiment intents.

`DiagnosticTrace` is the adapter-owned behavior trace. For single-call tasks it may contain only raw output text and metadata. For interactive tasks it should contain `InteractionTurn`, `ToolCallTrace`, terminal state, and terminal reason. Ratchet treats these traces as evidence; it does not infer tool behavior from final text alone.

## Model Roles

The optimizer uses separate model roles even when they share the same configured model:

- diagnoser: labels failure modes from eval traces
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
- patch compatibility
- affordance ID validity
- output-contract preservation
- no holdout-guided search
- no task-specific proposal recipes

Model calls should provide judgment:

- failure interpretation
- research questions worth testing
- concrete candidate content within legal affordances
- measurement value tradeoffs under evidence uncertainty

## Tool-Call Tasks

Tool calls are handled as an extension of Ratchet's evidence loop, not as a tau-bench-specific recipe.

- adapters execute the real environment loop and return structured trajectories
- behavior diagnostics summarize tool status, tool errors, premature stopping, turn counts, and tool-call counts
- affordance providers expose meaningful tool/action moves such as tool selection policy, argument grounding, precondition policy, and interaction completion
- candidates still apply legal affordances through normal `AgentPatch` operations
- evidence and reports distinguish task score gains from extra model calls, tool calls, turns, latency, and measurement spend

Known public benchmark integrations should use the official simulator when available. The optional tau-bench bridge converts original `tau-bench` retail/airline results into Ratchet `RunRecord`s; static action-list proxies are useful only as development probes, not leaderboard-comparable tau-bench evaluation.

Hard-coded task recipes, fallback proposal generators, or model-bypass switches violate the architecture. Tests may use fakes, but production optimization should fail visibly when a model role cannot produce valid output.

## Measurement Semantics

Ratchet separates deployed-policy tradeoffs from measurement spend.

Deployed-policy metrics describe what a selected patch would cost or how fast it would run per case.

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
