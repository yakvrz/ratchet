# Runbook Action Sample

This is a procedural tool-using sample for Ratchet.

It asks the agent to choose the exact next runbook action from a frozen incident-response snapshot. The benchmark mixes:
- supported incident scenarios with grounded actions
- unsupported scenarios that should return `unknown`

That makes it a good second proof integration for Ratchet's research loop.

Run it:

```bash
python3 -m ratchet check --config samples/runbook_action_agent/ratchet.toml
python3 -m ratchet optimize --config samples/runbook_action_agent/ratchet.toml
```

Like the flagship sample, this adapter exposes minimal integration plus a descriptive `AgentSpec`; Ratchet generates the editable surface and patches.

For local/reference runs without API budget, set `RATCHET_OFFLINE_MODE=1` before `check` or `optimize`.

To regenerate local artifacts, run the benchmark above after setting `OPENAI_API_KEY` in `.env`.
