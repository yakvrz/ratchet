# tau-bench Assessment

This sample runs the `sierra-research/tau-bench` simulator through Ratchet's generic `GeneratedToolLoopAdapter`. Tau-bench supplies the user simulator, domain tools, environment state, and benchmark reward; Ratchet owns the model/tool loop and transform hooks.

Pass rates are comparable to tau-bench leaderboard numbers when model/provider/user settings match.

Install the upstream benchmark before running:

```bash
pip install "git+https://github.com/sierra-research/tau-bench.git"
```

Generate a representative non-full-scale assessment from installed tau-bench tasks:

```bash
python samples/taubench_agent/generate_evals.py
```

Then run the assessment:

```bash
python -m ratchet check --config samples/taubench_agent/ratchet.assessment.toml
python -m ratchet optimize --config samples/taubench_agent/ratchet.assessment.toml
```

The default config uses conservative per-case parallelism. tau-bench is multi-turn and substantially more expensive than single-call samples.
