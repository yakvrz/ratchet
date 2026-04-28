# CLINC150 Intent Assessment

This sample is Ratchet's CLINC150-style assessment task for intent classification with out-of-scope detection.

It builds a focused subset of CLINC150 from the official CLINC OOS evaluation repository. The selected labels cover several confusable clusters: banking and card support, bills and credit limits, travel booking and travel status, calendar and list update intents, restaurant lookup intents, shopping-list intents, directions vs. distance, weather, and out-of-scope requests.

The baseline is deliberately simple and cheap:

- `gemini-2.5-flash-lite`
- no examples
- literal-overlap-oriented decision rule
- deliberately tight output cap, so Ratchet has to separate runtime/output defects from semantic label confusions and out-of-scope boundaries

Build the local eval file. The default assessment split writes 96 proposal-safe train examples, 96 protected dev cases, and 96 protected holdout cases:

```bash
python3 samples/clinc150_intent_agent/prepare_evals.py
```

Run a preflight check:

```bash
python3 -m ratchet check --config samples/clinc150_intent_agent/ratchet.assessment.toml
```

Run the optimizer assessment:

```bash
python3 -m ratchet optimize --config samples/clinc150_intent_agent/ratchet.assessment.toml
```

Assess ideation quality from a completed run:

```bash
python3 -m ratchet assess-ideation \
  --run-dir samples/clinc150_intent_agent/results/assessment \
  --spec samples/clinc150_intent_agent/ideation_assessment.json
```

This is not the full CLINC150 benchmark. It is the default CLINC assessment we use for optimizer development: large enough to make holdout validation meaningful, while still focused enough to inspect whether Ratchet diagnoses weak labels, proposes distinct mechanisms, accepts dev improvements cautiously, and validates finalists on protected holdout.
