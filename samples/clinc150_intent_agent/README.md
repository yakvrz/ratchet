# CLINC150 Intent Sample

This sample is a small Ratchet sanity check for CLINC150-style intent classification with out-of-scope detection.

It builds a focused subset of CLINC150 from the official CLINC OOS evaluation repository. The sanity subset uses 12 of the 150 in-scope labels plus the `oos` label, rather than the full CLINC150 label set. The selected labels are intentionally confusable financial intents: balance vs. transactions, bill amount vs. bill due vs. bill payment, credit limit lookup vs. credit limit change, card/account access problems, and fraud reporting.

The baseline is deliberately simple and cheap:

- `gemini-2.5-flash-lite`
- no examples
- literal-overlap-oriented decision rule
- sufficient output cap, so the interesting failures should mostly be semantic label confusions and out-of-scope boundaries

Build the local eval file. The default sanity split writes 39 proposal-safe train examples, 39 protected dev cases, and 26 protected holdout cases:

```bash
python3 samples/clinc150_intent_agent/prepare_evals.py
```

Run a preflight check:

```bash
python3 -m ratchet check --config samples/clinc150_intent_agent/ratchet.sanity.toml
```

Run the short optimizer sanity check:

```bash
python3 -m ratchet optimize --config samples/clinc150_intent_agent/ratchet.sanity.toml
```

This is not a full CLINC150 benchmark. It is meant to inspect Ratchet behavior on a second intent-classification dataset: whether it diagnoses weak labels, proposes distinct transform families, accepts dev improvements cautiously, and validates finalists on holdout.
