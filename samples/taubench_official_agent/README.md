# Official tau-bench Assessment

This sample is the Ratchet adapter for the original `sierra-research/tau-bench` simulator. It runs the real user simulator, domain tools, environment state, and official reward, so its pass rates are comparable to tau-bench leaderboard numbers when model/provider/user settings match.

Install the upstream benchmark before running:

```bash
pip install "git+https://github.com/sierra-research/tau-bench.git"
```

Then run a small smoke assessment:

```bash
python -m ratchet check --config samples/taubench_official_agent/ratchet.assessment.toml
python -m ratchet optimize --config samples/taubench_official_agent/ratchet.assessment.toml
```

Keep `case_concurrency = 1` unless provider limits are known. Official tau-bench is multi-turn and substantially more expensive than the static action proxy.
