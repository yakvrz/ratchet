# Python API Grounding Sample

This is the flagship external sample for Ratchet.

It uses a frozen public Python API snapshot, but the task is not simple lookup. The benchmark mixes:
- supported in-snapshot questions
- out-of-snapshot questions that should return `unknown`
- near-neighbor API confusions where grounding discipline matters

That makes it a better fit for Ratchet's research loop than the simpler `public_docs_agent` smoke benchmark.

Files:
- `agent.py`: standalone agent runner using Ratchet's Responses-compatible model client
- `docs_corpus.py`: frozen API snapshot
- `ratchet_adapter.py`: thin Ratchet adapter
- `evals.jsonl`: main dev/holdout benchmark
- `evals.quick.jsonl`: smaller calibration subset
- `expected_outcome.md`: what a healthy run should show

Run a preflight check:

```bash
python3 -m ratchet check --config samples/python_api_grounding_agent/ratchet.toml
```

Run the full optimizer:

```bash
python3 -m ratchet optimize --config samples/python_api_grounding_agent/ratchet.toml
```

Run the flagship demo configuration:

```bash
python3 -m ratchet optimize --config samples/python_api_grounding_agent/ratchet.demo.toml
```

This sample keeps Ratchet's optimizer model separate from the optimized agent: the agent baseline starts at `gemini-2.5-flash`, while the research loop is configured independently through `optimizer_model` and `optimizer_reasoning` in `ratchet.toml`. The Gemini configs use `gemini-3.1-pro-preview` for Ratchet's research loop, and the agent model surface includes current Gemini 3.x preview alternatives such as `gemini-3.1-flash-lite-preview` and `gemini-3-flash-preview`.

The adapter exposes a descriptive `AgentSpec`; Ratchet generates the editable surface and validates `AgentPatch` proposals itself.

Run the smaller subset:

```bash
python3 -m ratchet optimize --config samples/python_api_grounding_agent/ratchet.quick.toml
```

Assess ideation quality from a completed run:

```bash
python3 -m ratchet assess-ideation \
  --run-dir samples/python_api_grounding_agent/results/quick \
  --spec samples/python_api_grounding_agent/ideation_assessment.json
```

To regenerate local artifacts with the default config, run the benchmark above after setting `GEMINI_API_KEY` in `.env`.
