# Ratchet Alpha

Ratchet is a Python-first optimizer for agent harnesses.

Bring your Python agent and evals, declare the knobs Ratchet is allowed to change, and Ratchet will either:

- promote a holdout-validated candidate that is quality-non-inferior and more efficient, or
- keep your baseline unchanged.

Ratchet does not rewrite arbitrary source code. It diagnoses failures, proposes bounded harness changes, and validates those changes against the current eval set. The editable surface comes from the adapter: prompt artifacts, tool descriptions, tool availability, retrieval settings, model choice, output caps, bounded source-level hook logic, and similar harness controls.

Ratchet uses a separate optimizing model for diagnosis and proposal generation. By default, the optimized harness may search over smaller or cheaper agent models, while the optimizing side runs on `gpt-5.4` with `medium` reasoning effort unless you override `harnesser_model` / `harnesser_reasoning`.

For the alpha release overview and proof summary, see [ALPHA_WHITEPAPER.md](ALPHA_WHITEPAPER.md).

## Alpha scope

- Python agents only
- evals are required
- grading is adapter-defined over externally visible outputs
- optimization is over declared knobs and bounded declared artifacts
- arbitrary repo-wide code rewriting is out of scope

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
python3 -m ratchet run --config my-agent-ratchet/ratchet.toml
```

You can still run with explicit flags instead of a config file:

```bash
python3 -m ratchet run \
  --adapter package.module:adapter \
  --evals path/to/evals.jsonl \
  --out results/run
```

## Adapter contract

An adapter object must implement:

- `baseline() -> dict[str, str]`
- `search_space() -> SearchSpace`
- `run_case(candidate: dict[str, str], case: EvalCase) -> RunRecord`
- `grade(case: EvalCase, output: object) -> GradeResult`
- `export(candidate: dict[str, str], out_dir: Path) -> None`

Public serializable types:

- `EnumKnobSpec`
- `TextArtifactSpec`
- `CodeArtifactSpec`
- `SearchSpace`
- `EvalCase`
- `OperationalMetrics`
- `DiagnosticTrace`
- `RunRecord`
- `GradeResult`
- `FailureDiagnosis`
- `PatchProposal`

Helper graders are available in `ratchet.grading`:

- `exact_text_grade(...)`
- `numeric_tolerance_grade(...)`
- `json_field_grade(...)`

## Contract Model

- The current eval set scores the agent's external contract: inputs, externally visible outputs, and success criteria.
- Ratchet mutates the internal harness contract: prompt text, tool availability, tool descriptions, retrieval policy, model choice, bounded source-level hook logic, and similar artifacts.
- Diagnostic traces are used for diagnosis, proposal generation, and debugging only. Graders should depend on external outputs, not internal tool traces.

## Config

`ratchet.toml` supports:

- `adapter`
- `evals`
- `out`
- `env_file`
- `dev_budget`
- `holdout_top_k`
- `harnesser_model`
- `harnesser_reasoning`
- `harnesser_enabled`
- `max_case_retries`
- `case_timeout_s`
- `fail_fast`

Relative paths in `ratchet.toml` are resolved relative to the config file itself.

## Commands

- `python3 -m ratchet init --template python_function|python_cli --out <dir>`
- `python3 -m ratchet check --config ratchet.toml`
- `python3 -m ratchet run --config ratchet.toml`
- `python3 -m ratchet paired-demo`

## Outputs

Each run writes:

- `case_results.jsonl`: resumable per-case cache
- `candidate_metrics.json`: baseline, accepted dev incumbents, holdout validations, and promotable frontier
- `decision_log.json`: diagnosis/proposal iterations, holdout validation, and final selection
- `diagnoses.jsonl`: structured diagnosis buckets per iteration
- `proposals.jsonl`: proposed structural patches with acceptance/rejection outcomes
- `optimized_candidate.json`: selected candidate and promotion status
- `run_manifest.json`: config, timestamps, cache stats, retries, and runtime-error counts
- `report.md`: human-readable report with behavior history, diagnosis breakdown, proposal outcomes, and the gate decision
- `exported_candidate/`: adapter-materialized artifact bundle

## Included examples

Built-in regression fixtures:

- `examples/northstar/`

External proof suite:

- `samples/python_api_grounding_agent/`: flagship grounded-QA benchmark and the strongest current promoted result
- `samples/policy_triage_agent/`: structured-decision benchmark used as an honest baseline-kept proof
- `samples/runbook_action_agent/`: procedural tool-using benchmark that exercises diagnosis-driven behavior fixes and strict holdout rejection
- `samples/public_docs_agent/`: secondary smoke / efficiency benchmark

The flagship Python API grounding sample is intentionally outside the `ratchet/` core package. It is a standalone Python harness plus a thin adapter, and it exercises both grounded lookup and `unknown` behavior rather than only easy symbol retrieval.

Proof artifacts are reproducible from the included sample configs. The alpha whitepaper is the canonical release summary; rerun the sample benchmarks locally to regenerate fresh result bundles under `results/`.

For live runs, copy `.env.example` to `.env` and set `OPENAI_API_KEY`.

Ratchet's optimizing agent is separate from the optimized harness. By default the diagnoser/proposer loop runs on `gpt-5.4` with `medium` reasoning, while the searched harness may move to smaller or cheaper models.
