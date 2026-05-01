# Ratchet

Ratchet is a Python-first optimizer for agents.

Bring your Python agent and evals. Ratchet runs the original agent as an immutable baseline, builds branch-local evidence, plans controlled optimization experiments, compiles legal transform programs against the adapter's declared surface, and validates finalists against protected holdout before promoting a candidate.

The adapter is intentionally generated from a small harness. The harness declares how to build a model request, parse output, and grade externally visible behavior. Ratchet generates the executable `SurfaceSpec`, runtime hook wrapper, compiled-candidate execution, and export path.

For interactive agent benchmarks, one eval case may contain a full conversation with model turns, tool/environment calls, and terminal state. Adapters should return these trajectories through `DiagnosticTrace`; Ratchet uses them as evidence rather than flattening the case into one final string.

## Scope

- Python agents only
- evals are required
- grading is adapter-owned over externally visible outputs
- optimization is transform-program based over adapter-declared surfaces
- supported objective modes: correctness, cost, and latency
- arbitrary repo-wide source mutation is out of scope

## Optimization Architecture

Ratchet's core loop is research-oriented rather than recipe-oriented:

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

Model roles are intentionally narrow:

- search planner: reads objective, evidence, surface opportunities, prior evidence, and remaining budget, then emits a typed `SearchPlan`
- candidate implementer: emits typed transform programs citing search briefs and surface opportunities

Measurement selection is deterministic. Smoke evaluates compiled candidates that fit budget, small-dev screens in proposal order by comparison group, full-dev requires positive objective signal, confirmation checks unstable/runtime-sensitive finalists, and holdout is reserved for selected dev finalists.

Ratchet validates every optimizer output. There are no hand-authored proposal recipes, candidate generators, or task-specific rule profiles in the core loop.

See [docs/architecture.md](docs/architecture.md) for artifact definitions, role boundaries, measurement semantics, and failure policy.

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

- `surface_spec(cases: tuple[EvalCase, ...]) -> SurfaceSpec`
- `agent_spec() -> AgentSpec`
- `run_case(case: EvalCase, candidate: CompiledCandidate | None = None) -> RunRecord`
- `grade(case: EvalCase, output: object) -> GradeResult`
- `export(candidate: CompiledCandidate, out_dir: Path) -> None`

For single-call agents, prefer building a small harness and letting Ratchet generate the adapter:

```python
adapter = AdapterGenerator().build_runtime_adapter(harness)
```

The harness owns task-specific request construction, output parsing, and grading. Ratchet owns hook execution, transform compilation, instrumentation, surface export, and model-call runtime mechanics.

Public serializable types:

- `AgentSpec`
- `AgentTool`
- `SurfaceSpec`
- `TransformProgram`
- `CompiledCandidate`
- `AdapterGenerator`
- `GeneratedSingleCallAdapter`
- `OptimizationObjective`
- `OptimizationConstraints`
- `EvalCase`
- `OperationalMetrics`
- `DiagnosticTrace`
- `InteractionTurn`
- `ToolCallTrace`
- `RunRecord`
- `GradeResult`
- `FailureDiagnosis`

Internal optimization artifacts also appear in run outputs:

- `SurfaceOpportunity`
- `EvidencePacket`
- `SearchPlan`
- `SearchBrief`
- `CandidateProposal`
- `EvidenceLedger`

Helper utilities:

- `InteractionRecorder` for adapters that execute multi-turn tool/environment cases
- `GeneratedToolLoopAdapter` for interactive tool environments where Ratchet should own the model/tool loop
- `exact_text_grade(...)`
- `numeric_tolerance_grade(...)`
- `json_field_grade(...)`
- `estimate_cost_usd(...)` is available in `ratchet.pricing`

## Contract Model

- The eval set scores the agent's external contract: inputs, externally visible outputs, and success criteria.
- The adapter describes the current agent and scorer; it does not choose the optimization strategy.
- Ratchet compiles typed `TransformProgram` candidates against `SurfaceSpec`, then evaluates compiled candidates under the normal evidence and budget loop.
- Tool/environment traces are evidence. Tool-related surface opportunities are legal moves derived from the inferred agent surface plus observed trajectory failures.
- The search planner sees surface opportunities, not raw source files or task-specific recipes.
- Candidate implementations must compile against declared surfaces; unsupported hooks, state references, and boundary violations are rejected before eval.
- The scorer, including any LLM judge used by an eval, is frozen and outside the optimization surface.
- `candidate=None` always means the original user-provided agent.

## Config

`ratchet.toml` supports:

- `adapter`
- `evals`
- `out`
- `env_file`
- `dev_budget`
- `holdout_budget`
- `holdout_top_k`
- `optimizer_model`
- `optimizer_reasoning`
- `search_planner_model`
- `search_planner_reasoning`
- `candidate_implementer_model`
- `candidate_implementer_reasoning`
- `samples_per_case`
- `max_case_retries`
- `case_timeout_s`
- `case_concurrency`
- `stage_case_concurrency`
- `expensive_candidate_cost_ratio`
- `max_dev_measurement_cost_usd`
- `max_holdout_measurement_cost_usd`
- `max_dev_measurement_tool_calls`
- `max_holdout_measurement_tool_calls`
- `max_dev_measurement_turns`
- `max_holdout_measurement_turns`
- `fail_fast`
- `sanitize_examples`

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
allowed_models = ["gpt-4o-2024-08-06", "gpt-5.4-mini"]
max_cost_ratio = 1.0
max_latency_ratio = 1.1
max_transform_operations = 8
min_correctness_delta = 0.0 # optional; defaults to strict improvement for correctness and non-inferiority for cost/latency
```

Relative paths in `ratchet.toml` are resolved relative to the config file itself.
Set `samples_per_case > 1` for noisy agents or stochastic graders; Ratchet repeats every baseline and candidate case with separate cache entries and aggregates case outcomes by majority vote / mean score.
Per-case hard timeouts require serial case execution. Keep `case_concurrency = 1` and `stage_case_concurrency = 1` when `case_timeout_s > 0`; set `case_timeout_s = 0` for threaded case concurrency.

`max_dev_measurement_cost_usd` and `max_holdout_measurement_cost_usd` bound candidate evaluation spend. Interactive runs can also set tool-call and turn ceilings. These are separate from deployed-policy constraints: an expensive or long-horizon candidate may still be measured when it is useful frontier evidence, but deterministic code will not exceed configured measurement budgets.

## Commands

- `python3 -m ratchet init --template python_function|python_cli --out <dir>`
- `python3 -m ratchet check --config ratchet.toml`
- `python3 -m ratchet eval-health --config ratchet.toml`
- `python3 -m ratchet optimize --config ratchet.toml`
- `python3 -m ratchet assess-ideation --run-dir results/run --spec ideation_assessment.json`

`run` remains as an alias for `optimize`.

## Outputs

Each run writes:

- `case_results.jsonl`: resumable per-case cache keyed by candidate, case digest, eval digest, adapter fingerprint, objective, and surface spec
- `progress.jsonl`: chronological run progress, including model-role calls, case execution, cache hits, and stage decisions
- `events.jsonl`: typed optimizer events for planning, proposals, staged evaluation, frontier updates, holdout validation, and final selection
- `run_summary.json`: compact selected-candidate and run outcome summary
- `candidate_metrics.json`: true baseline, best dev candidate, selected holdout candidate, accepted dev candidates, holdout validations, typed surface, and Pareto frontier
- `outcome_analysis.json`: explicit reason for promotion or baseline retention
- `search_plans.jsonl`: typed search plans produced from deterministic evidence and surface opportunities
- `proposals.jsonl`: candidate transform programs, cited surface opportunities, and acceptance/rejection outcomes
- `evidence_ledger.json`: paired candidate evidence, reliability signals, and measurement history
- `ideation_metrics.json`: search-planner and implementer discovery quality
- `selected_candidate.json`: selected compiled candidate and promotion status
- `run_manifest.json`: config, timestamps, cache stats, retries, and runtime-error counts
- `summary.html`: user-facing run summary
- `plots/`: SVG plots embedded by `summary.html`
- `report.md`: human-readable report
- `exported_candidate/`: adapter-materialized candidate bundle

Interrupted runs write `partial_run_manifest.json` and `partial_report.md` with the last progress events and incomplete case evaluations. Shared per-case cache rows live outside run directories under `.ratchet/cache/`, which is intentionally git-ignored.

## Samples

- `samples/bfcl_function_calling_agent/`
- `samples/taubench_agent/`
- `samples/banking77_intent_agent/`
- `samples/clinc150_intent_agent/`

The sample suite is intentionally limited to public, trusted assessment vehicles. BFCL is the primary single-call function/tool contract benchmark. The tau-bench sample is the primary interactive tool-loop benchmark. BANKING77 and CLINC150 remain secondary classification probes for label-boundary, few-shot, and eval-stability behavior.

See [docs/benchmarks.md](docs/benchmarks.md) for benchmark roles, limitations, and criteria for adding new benchmarks.

For live runs, copy `.env.example` to `.env` and set the API key required by your configured models, for example `OPENAI_API_KEY` for OpenAI models or `GEMINI_API_KEY` for Gemini models.

Ratchet's optimizer model is separate from the optimized agent. Configure `optimizer_model` and `optimizer_reasoning` as defaults for the research loop; override `search_planner_*` or `candidate_implementer_*` when a run should use different optimizer models per role. The agent may move to allowed models through compiled model-configuration transforms.
