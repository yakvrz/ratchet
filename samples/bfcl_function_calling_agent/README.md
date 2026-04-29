# BFCL Function Calling Assessment

This sample is Ratchet's BFCL assessment task for single-turn function-call extraction.

It builds a local subset of the Berkeley Function Calling Leaderboard V3 dataset from the public Hugging Face mirror. The assessment covers four single-turn categories:

- `simple`
- `multiple`
- `parallel`
- `parallel_multiple`

The baseline is deliberately simple and cheap:

- `gemini-2.5-flash-lite`
- no examples
- schema and argument instructions only
- JSON output emulating function calls rather than executing tools

Build the local eval file. The default assessment split writes 96 proposal-safe train examples, 96 protected dev cases, and 96 protected holdout cases:

```bash
python3 samples/bfcl_function_calling_agent/prepare_evals.py
```

Run a preflight check:

```bash
python3 -m ratchet check --config samples/bfcl_function_calling_agent/ratchet.assessment.toml
```

Run the optimizer assessment:

```bash
python3 -m ratchet optimize --config samples/bfcl_function_calling_agent/ratchet.assessment.toml
```

This is not a full BFCL leaderboard run. It is Ratchet's development assessment for tool/function-call policy optimization: large enough to exercise output-contract, argument-selection, model-capability, few-shot, and runtime interventions while remaining inspectable.
