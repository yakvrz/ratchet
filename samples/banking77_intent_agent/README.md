# BANKING77 Intent Sample

This sample is a small Ratchet sanity check for fine-grained banking intent classification.

It builds a focused subset of BANKING77 from the public PolyAI task-specific dataset repository. The sanity subset uses 12 of the 77 BANKING77 labels, rather than the full BANKING77 label set. The selected labels are intentionally confusable: cash withdrawal fees vs. unrecognized withdrawals, card payment fees vs. unrecognized card payments, transfer issues, and identity verification intents.

The baseline is deliberately simple and cheap:

- `gemini-2.5-flash-lite`
- no examples
- literal-overlap-oriented decision rule
- sufficient output cap, so the interesting failures should mostly be semantic label confusions

Build the local eval file. The default sanity split writes 36 proposal-safe train examples, 36 protected dev cases, and 24 protected holdout cases:

```bash
python3 samples/banking77_intent_agent/prepare_evals.py
```

Run a preflight check:

```bash
python3 -m ratchet check --config samples/banking77_intent_agent/ratchet.sanity.toml
```

Run the short optimizer sanity check:

```bash
python3 -m ratchet optimize --config samples/banking77_intent_agent/ratchet.sanity.toml
```

This is not a full BANKING77 benchmark. It is meant to inspect Ratchet behavior: whether it diagnoses weak labels, proposes distinct transform families, accepts dev improvements cautiously, and validates finalists on holdout.
