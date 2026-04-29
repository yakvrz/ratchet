# Ratchet

Ratchet is a Python-first optimizer for agents.

Bring your Python agent and evals. Ratchet runs the original agent as an immutable baseline, builds branch-local evidence, plans controlled optimization experiments, implements legal candidate changes through the configured optimizer model, and validates finalists against protected holdout before promoting a patch.

The adapter is intentionally minimal. It runs the agent, grades outputs, optionally exposes a descriptive `AgentSpec`, and exports the selected patch. Ratchet owns the optimization surface, research loop, objective handling, measurement decisions, evidence ledger, and promotion gates.

## Scope

- Python agents only
- evals are required
- grading is adapter-owned over externally visible outputs
- optimization is patch-based over Ratchet-generated optimization affordances
- supported objective modes: correctness, cost, and latency
- arbitrary repo-wide source mutation is out of scope

## Optimization Architecture

Ratchet's core loop is research-oriented rather than recipe-oriented:

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

The important artifacts are:

- `EditableTarget`: low-level adapter-exposed edit handle, such as an instruction, model choice, runtime setting, output contract, retrieval policy, tool policy, or few-shot bank.
- `OptimizationAffordance`: the primary optimizer surface. It names a meaningful legal move, its mechanism, target, operations, expected measurements, risks, composition guidance, suitability, and evidence.
- `ExperimentIntent`: planner output describing a research question and the affordances that may be used to test it. It contains no patch content.
- `CandidateProposal`: implementer output that applies one or more affordances through concrete operations or proposal-safe few-shot selections.
- `EvidenceLedger`: paired candidate-vs-reference evidence used by the measurement selector, reports, and promotion gates.

Model roles are split deliberately:

- diagnoser: labels failure modes from eval traces
- research planner: emits experiment intents only
- candidate implementer: emits candidate affordance applications only
- measurement selector: chooses which already-valid candidates receive more measurement

Ratchet validates every optimizer output. There are no hand-authored proposal recipes, candidate generators, or task-specific rule profiles in the core loop.

## Quickstart

Create a scaffold:

```bash
python3 -m ratchet init --template python_function --out my-agent-ratchet
```

Wire your agent into the generated scaffold, then run a preflight check:

```bash
python3 -m ratchet check --config my-agent-ratchet/ratchet.toml
```

Run the optimizer:

```bash
python3 -m ratchet optimize --config my-agent-ratchet/ratchet.toml
```

Optionally check eval-set and grader health before optimizing:

```bash
python3 -m ratchet eval-health --config my-agent-ratchet/ratchet.toml
```

The eval health command writes a readable `eval_health.md` and a complete ordered `eval_health.json`
under `<out>/eval_health/`.

You can still run with explicit flags instead of a config file:

```bash
python3 -m ratchet optimize \
  --adapter package.module:adapter \
  --evals path/to/evals.jsonl \
  --out results/run \
  --mode correctness
```

## Adapter Contract

An adapter object must implement:

- `agent_spec() -> AgentSpec | None`
- `run_case(case: EvalCase, patch: AgentPatch | None = None) -> RunRecord`
- `grade(case: EvalCase, output: object) -> GradeResult`
- `export(patch: AgentPatch, out_dir: Path) -> None`

Public serializable types:

- `AgentSpec`
- `AgentTool`
- `EditableTarget`
- `AgentPatch`
- `PatchOperation`
- `OptimizationObjective`
- `OptimizationConstraints`
- `EvalCase`
- `OperationalMetrics`
- `DiagnosticTrace`
- `RunRecord`
- `GradeResult`
- `FailureDiagnosis`

Internal optimization artifacts also appear in run outputs:

- `OptimizationAffordance`
- `TaskTheory`
- `ExperimentIntent`
- `CandidateProposal`
- `EvidenceLedger`
- `MeasurementDecision`

Helper utilities:

- `exact_text_grade(...)`
- `numeric_tolerance_grade(...)`
- `json_field_grade(...)`
- `estimate_cost_usd(...)` is available in `ratchet.pricing`

## Contract Model

- The eval set scores the agent's external contract: inputs, externally visible outputs, and success criteria.
- The adapter describes the current agent and scorer; it does not choose the optimization strategy.
- Ratchet generates editable targets from `AgentSpec`, then derives ranked optimization affordances from the surface, traces, failures, objective mode, and constraints.
- The research planner sees affordances, not raw source files or task-specific recipes.
- Candidate implementations must cite concrete affordance IDs; family and mechanism metadata are derived from those affordances.
- The scorer, including any LLM judge used by an eval, is frozen and outside the optimization surface.
- `patch=None` always means the original user-provided agent.

## Config

`ratchet.toml` supports:

- `adapter`
- `evals`
- `out`
- `env_file`
- `dev_budget`
- `holdout_budget`
- `optimizer_model`
- `optimizer_reasoning`
- `diagnoser_model`
- `diagnoser_reasoning`
- `research_planner_model`
- `research_planner_reasoning`
- `candidate_implementer_model`
- `candidate_implementer_reasoning`
- `measurement_selector_model`
- `measurement_selector_reasoning`
- `samples_per_case`
- `max_case_retries`
- `case_timeout_s`
- `case_concurrency`
- `stage_case_concurrency`
- `expensive_candidate_cost_ratio`
- `max_expensive_full_dev_candidates`
- `max_expensive_holdout_candidates`
- `fail_fast`

Optional eval health config:

```toml
[ratchet.eval_health]
sample_limit = 8
repeats = 2
min_holdout_cases = 5
max_runtime_error_rate = 0.05
max_unstable_case_rate = 0.2
max_mean_latency_s = 30.0
max_p95_latency_s = 60.0
max_mean_cost_usd = 0.25
max_estimated_eval_cost_usd = 25.0
max_estimated_eval_wall_time_s = 3600.0
max_estimated_eval_tokens = 5000000
```

Objective config:

```toml
[ratchet.objective]
mode = "correctness" # correctness | cost | latency

[ratchet.objective.constraints]
allowed_edits = ["instruction", "tool", "retrieval", "runtime", "model", "output"]
allowed_models = ["gpt-4o-2024-08-06", "gpt-5.4-mini"]
max_cost_ratio = 1.0
max_latency_ratio = 1.1
min_correctness_delta = 0.0 # optional; defaults to strict improvement for correctness and non-inferiority for cost/latency
```

Relative paths in `ratchet.toml` are resolved relative to the config file itself.
Set `samples_per_case > 1` for noisy agents or stochastic graders; Ratchet repeats every baseline and patch case with separate cache entries and aggregates case outcomes by majority vote / mean score.

## Commands

- `python3 -m ratchet init --template python_function|python_cli --out <dir>`
- `python3 -m ratchet check --config ratchet.toml`
- `python3 -m ratchet eval-health --config ratchet.toml`
- `python3 -m ratchet optimize --config ratchet.toml`

`run` remains as an alias for `optimize`.

## Outputs

Each run writes:

- `case_results.jsonl`: resumable per-case cache keyed by patch, case digest, eval digest, adapter fingerprint, objective, and baseline `AgentSpec`
- `patch_metrics.json`: true baseline, best dev patch, selected holdout patch, accepted dev patches, holdout validations, typed generated surface, and Pareto frontier
- `decision_log.json`: research state, task theory, planning, implementation, measurement, holdout validation, and final selection
- `outcome_analysis.json`: explicit reason for promotion or baseline retention
- `diagnoses.jsonl`: structured diagnosis buckets per iteration
- `proposals.jsonl`: candidate affordance applications with acceptance/rejection outcomes
- `evidence_ledger.json`: paired candidate evidence, reliability signals, and measurement history
- `ideation_metrics.json`: planner/implementer/measurement discovery quality
- `selected_patch.json`: selected patch and promotion status
- `run_manifest.json`: config, timestamps, cache stats, retries, and runtime-error counts
- `summary.html`: user-facing run summary
- `plots/`: SVG plots embedded by `summary.html`
- `report.md`: human-readable report
- `exported_patch/`: adapter-materialized patch bundle

## Samples

- `samples/python_api_grounding_agent/`
- `samples/banking77_intent_agent/`
- `samples/clinc150_intent_agent/`
- `samples/bfcl_function_calling_agent/`
- `samples/policy_triage_agent/`
- `samples/runbook_action_agent/`
- `samples/public_docs_agent/`
- `samples/kashi_agent/`

The primary optimizer-development assessment vehicles are BANKING77, CLINC150, and BFCL. They use `ratchet.assessment.toml`, protected dev/holdout splits, and optional ideation assessment specs. Smaller sample configs remain for smoke checks and adapter examples.

For live runs, copy `.env.example` to `.env` and set the API key required by your configured models, for example `OPENAI_API_KEY` for OpenAI models or `GEMINI_API_KEY` for Gemini models.

Ratchet's optimizer model is separate from the optimized agent. Configure `optimizer_model` and `optimizer_reasoning` as defaults for the research loop; override individual roles with `diagnoser_*`, `research_planner_*`, `candidate_implementer_*`, or `measurement_selector_*` when a run should use different optimizer models per role. The agent may move to allowed models through generated `change_model` patches.
