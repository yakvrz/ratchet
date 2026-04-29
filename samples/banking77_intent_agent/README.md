# BANKING77 Intent Assessment

This sample is Ratchet's BANKING77 assessment task for fine-grained banking intent classification.

It builds a focused subset of BANKING77 from the public PolyAI task-specific dataset repository. The selected labels are intentionally confusable: cash withdrawal fees vs. unrecognized withdrawals, card payment fees vs. unrecognized card payments, transfer issues, and identity verification intents.

The baseline is deliberately simple and cheap:

- `gemini-2.5-flash-lite`
- no examples
- literal-overlap-oriented decision rule
- deliberately tight output cap, so Ratchet has to separate runtime/output defects from semantic label confusions

Build the local eval file. The default assessment split writes 96 proposal-safe train examples, 96 protected dev cases, and 96 protected holdout cases:

```bash
python3 samples/banking77_intent_agent/prepare_evals.py
```

Run a preflight check:

```bash
python3 -m ratchet check --config samples/banking77_intent_agent/ratchet.assessment.toml
```

Run the optimizer assessment:

```bash
python3 -m ratchet optimize --config samples/banking77_intent_agent/ratchet.assessment.toml
```

Assess ideation quality from a completed run:

```bash
python3 -m ratchet assess-ideation \
  --run-dir samples/banking77_intent_agent/results/assessment \
  --spec samples/banking77_intent_agent/ideation_assessment.json
```

This is not the full 77-label BANKING77 benchmark. It is the default BANKING77 assessment we use for optimizer development: large enough to make holdout validation meaningful, while still focused enough to inspect whether Ratchet diagnoses weak labels, proposes distinct mechanisms, accepts dev improvements cautiously, and validates finalists on protected holdout.
