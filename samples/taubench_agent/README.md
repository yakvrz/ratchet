# tau-bench Assessment

This sample runs the `sierra-research/tau-bench` simulator through Ratchet's generic `GeneratedToolLoopAdapter`. Tau-bench supplies the user simulator, domain tools, environment state, and benchmark reward; Ratchet owns the model/tool loop and transform hooks.

Pass rates are comparable to tau-bench leaderboard numbers when model/provider/user settings match.

Install the upstream benchmark before running:

```bash
pip install "git+https://github.com/sierra-research/tau-bench.git"
```

Then run a small smoke assessment:

```bash
python -m ratchet check --config samples/taubench_agent/ratchet.assessment.toml
python -m ratchet optimize --config samples/taubench_agent/ratchet.assessment.toml
```

Keep `case_concurrency = 1` unless provider limits are known. tau-bench is multi-turn and substantially more expensive than single-call samples.
