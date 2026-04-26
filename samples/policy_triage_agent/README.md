# Policy Triage Sample

This is a structured decision sample for Ratchet.

It uses a frozen reimbursement-policy snapshot plus objective JSON grading over:
- `decision`
- `amount`

The baseline is intentionally already tight and cheap, so this sample is useful as an honest baseline-kept proof rather than a flagship optimization story.

Run it:

```bash
python3 -m ratchet check --config samples/policy_triage_agent/ratchet.toml
python3 -m ratchet optimize --config samples/policy_triage_agent/ratchet.toml
```

Like the flagship sample, this adapter exposes only minimal integration plus a descriptive `AgentSpec`. The expected outcome here is usually that Ratchet keeps the baseline unless it finds a clearly safe improvement.

For local/reference runs without API budget, set `RATCHET_OFFLINE_MODE=1` before `check` or `optimize`.

To regenerate local artifacts, run the benchmark above after setting `OPENAI_API_KEY` in `.env`.
