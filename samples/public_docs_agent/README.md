# Public Docs QA Sample

This is a standalone Python agent harness that Ratchet can optimize through the normal adapter interface.

Files:
- `agent.py`: the harnessed docs QA agent using the OpenAI Responses API
- `docs_corpus.py`: a frozen public-docs snapshot
- `ratchet_adapter.py`: the thin Ratchet adapter
- `evals.jsonl`: objective dev/holdout evals
- `evals.quick.jsonl`: smaller live-proof subset
- `ratchet.toml`: config for `python3 -m ratchet check/run`
- `ratchet.quick.toml`: faster bounded live-run config

Run it:

```bash
cd /Users/yakvrz/Projects/ratchet
python3 -m ratchet check --config samples/public_docs_agent/ratchet.toml
python3 -m ratchet run --config samples/public_docs_agent/ratchet.toml
```

Fast live proof:

```bash
cd /Users/yakvrz/Projects/ratchet
python3 -m ratchet run --config samples/public_docs_agent/ratchet.quick.toml
```

For the alpha proof summary, see `ALPHA_WHITEPAPER.md`. This sample is intentionally secondary to the flagship `python_api_grounding_agent`.

The optimization surface is intentionally bounded but meaningful: model, reasoning effort, prompt clauses, tool availability, tool descriptions, knowledge-card mode, retrieval top-k, and output cap. This sample is kept as a smoke / efficiency benchmark; the flagship code-artifact path lives in `python_api_grounding_agent`.
