# Ratchet

Ratchet is a Python-first optimizer for agents.

Bring your Python agent and evals. Ratchet runs the original agent as an immutable baseline, diagnoses failures, generates proposed `AgentPatch` changes, validates them against dev and holdout splits, and either promotes a patch or keeps the original baseline.

The adapter is intentionally minimal. It runs the agent, grades outputs, optionally exposes a descriptive `AgentSpec`, and exports the selected patch. Ratchet owns search-surface generation, patch proposal, objective handling, and promotion.

## Scope

- Python agents only
- evals are required
- grading is adapter-owned over externally visible outputs
- optimization is patch-based over a Ratchet-generated surface
- supported objective modes: correctness, cost, and latency
- arbitrary repo-wide source mutation is out of scope

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

Helper utilities:

- `exact_text_grade(...)`
- `numeric_tolerance_grade(...)`
- `json_field_grade(...)`
- `estimate_cost_usd(...)` is available in `ratchet.pricing`

## Contract Model

- The eval set scores the agent's external contract: inputs, externally visible outputs, and success criteria.
- The adapter describes the current agent and scorer; it does not choose the optimization strategy.
- Ratchet generates editable targets from `AgentSpec`, traces, failures, objective mode, and constraints.
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
- `decision_log.json`: diagnosis, planning, implementation, measurement, holdout validation, and final selection
- `outcome_analysis.json`: explicit reason for promotion or baseline retention
- `diagnoses.jsonl`: structured diagnosis buckets per iteration
- `proposals.jsonl`: proposed patches with acceptance/rejection outcomes
- `selected_patch.json`: selected patch and promotion status
- `run_manifest.json`: config, timestamps, cache stats, retries, and runtime-error counts
- `summary.html`: user-facing run summary
- `plots/`: SVG plots embedded by `summary.html`
- `report.md`: human-readable report
- `exported_patch/`: adapter-materialized patch bundle

## Samples

- `samples/python_api_grounding_agent/`
- `samples/policy_triage_agent/`
- `samples/runbook_action_agent/`
- `samples/public_docs_agent/`
- `samples/kashi_agent/`

For live runs, copy `.env.example` to `.env` and set the API key required by your configured models, for example `OPENAI_API_KEY` for OpenAI models or `GEMINI_API_KEY` for Gemini models.

Ratchet's optimizer model is separate from the optimized agent. Configure `optimizer_model` and `optimizer_reasoning` as defaults for the research loop; override individual roles with `diagnoser_*`, `research_planner_*`, `candidate_implementer_*`, or `measurement_selector_*` when a run should use different optimizer models per role. The agent may move to allowed models through generated `change_model` patches.
