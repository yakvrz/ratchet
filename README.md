# Ratchet

![Ratchet](docs/assets/logo.png)

A Python-first optimizer for agents.

You provide an agent and an eval set. Ratchet keeps the original agent as a frozen baseline, proposes typed program transforms against the surface your adapter declares, evaluates them under a fixed budget, and promotes a candidate only if it beats the baseline on a protected holdout. The search planner and candidate implementer are LLMs, but their outputs are constrained to typed schemas and validated before they touch eval. The core loop has no hand-authored proposal templates or task-specific rules.

## Demo results

The bundled demo is a 48-case order-desk tool-loop benchmark with authentication, read/mutate tools, hidden state, deterministic grading, and a protected holdout. On the release run with a `gemini-2.5-flash` baseline:

| | Baseline | After Ratchet |
|---|---|---|
| Holdout pass rate | 62.5% | 81.25% (+30% rel.) |
| `cancel` slice | 58% | 100% |
| `address` slice | 92% | 100% |
| `ambiguity` slice | 0% | 25% |
| Mean cost / case | $0.0018 | $0.0016 |
| Mean tool calls / case | 3.04 | 2.79 |

Total optimizer spend for the run: $0.88. The winning candidate was a `before_tool_call` precondition that tracks inspected `order_id`s in agent state and forces `get_order` before any mutation, plus a `domain_policy` patch for ambiguous requests.

To reproduce:

```
python3 -m ratchet optimize --config demo/ratchet.diagnostic_expanded.toml
```

## Scope

- Python agents only.
- Evals are required. Grading is owned by the adapter.
- Optimization works through typed `TransformProgram`s applied to an adapter-declared `SurfaceSpec`. Arbitrary repo-wide source mutation is out of scope.
- Objective modes: correctness, cost, latency.

## Pipeline

```
AgentHarness -> AdapterGenerator -> SurfaceSpec -> SurfaceOpportunity[]
  -> BaselineEvaluation -> EvidencePacket -> SearchPlan
  -> CandidateProposal[] -> TransformProgram -> CompiledCandidate
  -> EvidenceLedger -> FrontierUpdate -> HoldoutValidation
```

Measurement is staged: smoke (does it run), small-dev (screen by comparison group), full-dev (require objective signal), confirmation (re-check unstable finalists), holdout (selected finalists only).

See [docs/architecture.md](docs/architecture.md) and [docs/release.md](docs/release.md) for details.

## Quickstart

```
python3 -m ratchet init --template python_function --out my-agent-ratchet
# wire your agent into the scaffold, then:
python3 -m ratchet check    --config my-agent-ratchet/ratchet.toml
python3 -m ratchet optimize --config my-agent-ratchet/ratchet.toml
```

Other commands: `eval-health`, `release-check`, `assess-ideation`. `run` is an alias for `optimize`.

For live runs, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

## Adapter contract

An adapter implements:

```
surface_spec(cases) -> SurfaceSpec
agent_spec()        -> AgentSpec
run_case(case, candidate=None) -> RunRecord
grade(case, output) -> GradeResult
export(candidate, out_dir) -> None
```

For single-call agents, write a small harness and let Ratchet generate the adapter:

```
adapter = AdapterGenerator().build_runtime_adapter(harness)
```

The harness owns request construction, output parsing, and grading. Ratchet owns hook execution, transform compilation, instrumentation, and the model-call runtime. `candidate=None` means the original user-provided agent.

For multi-turn tool-using agents, return the full trajectory through `DiagnosticTrace`. Use `GeneratedToolLoopAdapter` when you want Ratchet to own the model/tool loop.

## Config

`ratchet.toml` covers adapter wiring, budgets, optimizer-role models, and execution knobs. The optimizer model is separate from the optimized agent; the agent can only move to models in `objective.constraints.allowed_models` via compiled model-configuration transforms.

```
[ratchet.objective]
mode = "correctness"  # correctness | cost | latency

[ratchet.objective.constraints]
allowed_models = ["gpt-4o-2024-08-06", "gpt-5.4-mini"]
max_cost_ratio = 1.0
max_latency_ratio = 1.1
max_transform_operations = 8
```

Notes:

- Per-case hard timeouts (`case_timeout_s > 0`) require serial execution. Set `case_timeout_s = 0` to enable threaded concurrency.
- For noisy agents or stochastic graders, set `samples_per_case > 1`. Ratchet aggregates by majority vote / mean score.
- `max_dev_measurement_cost_usd` and friends cap evaluation spend during a run. `max_cost_ratio` is separate and constrains the deployed candidate.

`ratchet init` writes a populated `ratchet.toml`. The demo configs in [demo/](demo/) are working references.

## Outputs

Each run writes `run_summary.json`, `candidate_metrics.json`, `outcome_analysis.json`, a resumable per-case cache (`case_results.jsonl`), event logs (`events.jsonl`, `progress.jsonl`, `search_plans.jsonl`, `proposals.jsonl`, `evidence_ledger.json`), an exported candidate (`exported_candidate/`), and rendered reports (`summary.html`, `report.md`). Interrupted runs leave `partial_*` artifacts. The cross-run per-case cache lives at `.ratchet/cache/` (git-ignored).
