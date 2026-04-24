# Python API Grounding Sample

This is the flagship external sample for Ratchet.

It uses a frozen public Python API snapshot, but the task is not simple lookup. The benchmark mixes:
- supported in-snapshot questions
- out-of-snapshot questions that should return `unknown`
- near-neighbor API confusions where grounding discipline matters

That makes it a better fit for Ratchet's diagnosis/proposal loop than the simpler `public_docs_agent` smoke benchmark.

Files:
- `agent.py`: standalone harness runner using the OpenAI Responses API
- `docs_corpus.py`: frozen API snapshot
- `ratchet_adapter.py`: thin Ratchet adapter
- `evals.jsonl`: main dev/holdout benchmark
- `evals.quick.jsonl`: smaller calibration subset

Run a preflight check:

```bash
python3 -m ratchet check --config samples/python_api_grounding_agent/ratchet.toml
```

Run the full optimizer:

```bash
python3 -m ratchet run --config samples/python_api_grounding_agent/ratchet.toml
```

This sample keeps the optimizing side separate from the optimized harness: the harness baseline starts at `gpt-5.4`, while the diagnoser/proposer loop is configured independently through `harnesser_model` and `harnesser_reasoning` in `ratchet.toml`.

It also exposes bounded source-level hook artifacts in addition to prompts, tools, and components. Ratchet can rewrite those declared Python hooks during optimization, but it still does not perform arbitrary repo-wide code patching.

Run the smaller subset:

```bash
python3 -m ratchet run --config samples/python_api_grounding_agent/ratchet.quick.toml
```

For the alpha proof summary and headline metrics, see the repo-level `ALPHA_WHITEPAPER.md`.

To regenerate local artifacts, run the benchmark above after setting `OPENAI_API_KEY` in `.env`.
