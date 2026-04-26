# Expected Outcome

This sample is a live optimization demo, not a golden deterministic fixture.

With the default Gemini-backed `ratchet.demo.toml`, the baseline should be imperfect on both dev and holdout because it sometimes returns bare strings or conversational fragments instead of the required JSON object. Ratchet should diagnose those failures as output-contract failures, propose general `AgentPatch` changes such as output-instruction revisions or verifier retry, and evaluate them on dev before spending holdout budget.

The conservative final gate may still keep the baseline when holdout gains are small or noisy. In that case, inspect:

- `report.md` for the outcome status
- `diagnoses.jsonl` for the diagnosed failure clusters
- `proposals.jsonl` for accepted/rejected candidate patches
- `patch_metrics.json` for dev and holdout deltas

A healthy run is one where Ratchet produces task-agnostic diagnoses, validates concrete patches, and explains promotion or baseline retention. It should not use hand-authored proposal fallbacks or sample-specific rule profiles.
