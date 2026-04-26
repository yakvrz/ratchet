# Public Docs QA Sample

This is a standalone Python agent that Ratchet can optimize through the normal adapter interface.

Files:
- `agent.py`: the docs QA agent using the OpenAI Responses API
- `docs_corpus.py`: a frozen public-docs snapshot
- `ratchet_adapter.py`: the thin Ratchet adapter
- `evals.jsonl`: objective dev/holdout evals
- `evals.quick.jsonl`: smaller live-proof subset
- `ratchet.toml`: config for `python3 -m ratchet check/optimize`
- `ratchet.quick.toml`: faster bounded live-run config

Run it:

```bash
cd /Users/yakvrz/Projects/ratchet
python3 -m ratchet check --config samples/public_docs_agent/ratchet.toml
python3 -m ratchet optimize --config samples/public_docs_agent/ratchet.toml
```

Fast live proof:

```bash
cd /Users/yakvrz/Projects/ratchet
python3 -m ratchet optimize --config samples/public_docs_agent/ratchet.quick.toml
```

The adapter exposes a descriptive `AgentSpec`; Ratchet generates the bounded editable surface from that spec, objective mode, and constraints. This sample is kept as a smoke / efficiency benchmark.
