# Project Instructions

- Make system changes general and root-cause oriented. Avoid case-specific patches for individual examples.
- Prefer failing fast over defensive code with unnecessary fallbacks.
- Do not add hand-authored optimization recipes, task-specific rule profiles, fallback proposal generators, or switches that bypass the model-driven optimizer. Ratchet should discover patches through the task-agnostic eval loop using the configured optimizing model.
- When proposing multiple next steps, maintain continuity explicitly: identify the single active focus, keep deferred items in a visible backlog with status, and update that backlog after implementation instead of letting unimplemented suggestions disappear.
