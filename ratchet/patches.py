from __future__ import annotations

from ratchet.io import patch_hash
from ratchet.types import AgentPatch


def compose_patches(parent: AgentPatch, child: AgentPatch) -> AgentPatch:
    return AgentPatch(
        operations=[*parent.operations, *child.operations],
        rationale=child.rationale or parent.rationale,
        expected_effect=child.expected_effect or parent.expected_effect,
        metadata={
            **parent.metadata,
            **child.metadata,
            "parent_patch_hash": patch_hash(parent),
        },
    )
