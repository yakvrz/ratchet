from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ratchet.types import DiagnosticTrace, InteractionTurn, OperationalMetrics, ToolCallTrace


@dataclass
class InteractionRecorder:
    """Small helper for adapters that execute multi-turn tool/environment cases."""

    turns: list[InteractionTurn] = field(default_factory=list)
    def add_turn(
        self,
        *,
        actor: str,
        message: Any = None,
        outcome: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        index = len(self.turns)
        self.turns.append(
            InteractionTurn(
                index=index,
                actor=actor,
                message=message,
                outcome=outcome,
                metadata=dict(metadata or {}),
            )
        )
        return index

    def add_tool_call(
        self,
        *,
        name: str,
        arguments: Any = None,
        result: Any = None,
        status: str = "ok",
        latency_s: float | None = None,
        error: str | None = None,
        turn_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        target_index = self._target_turn_index(turn_index)
        turn = self.turns[target_index]
        tool_call = ToolCallTrace(
            name=name,
            arguments=arguments,
            result=result,
            status=status,
            latency_s=latency_s,
            error=error,
            metadata=dict(metadata or {}),
        )
        self.turns[target_index] = InteractionTurn(
            index=turn.index,
            actor=turn.actor,
            message=turn.message,
            tool_calls=[*turn.tool_calls, tool_call],
            outcome=turn.outcome,
            metadata=dict(turn.metadata),
        )

    def diagnostics(
        self,
        *,
        raw_output_text: str = "",
        terminal_state: dict[str, Any] | None = None,
        terminal_reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> DiagnosticTrace:
        return DiagnosticTrace(
            raw_output_text=raw_output_text,
            turns=list(self.turns),
            terminal_state=dict(terminal_state or {}),
            terminal_reason=terminal_reason,
            metadata=dict(metadata or {}),
        )

    def metrics(
        self,
        *,
        latency_s: float,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        model_calls: int = 1,
        total_tokens: int | None = None,
        error: str | None = None,
    ) -> OperationalMetrics:
        return OperationalMetrics(
            latency_s=latency_s,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens if total_tokens is not None else input_tokens + output_tokens,
            cost_usd=cost_usd,
            model_calls=model_calls,
            tool_calls=sum(len(turn.tool_calls) for turn in self.turns),
            turns=max(1, len(self.turns)),
            error=error,
        )

    def _target_turn_index(self, turn_index: int | None) -> int:
        if turn_index is None:
            if not self.turns:
                return self.add_turn(actor="agent")
            return len(self.turns) - 1
        if turn_index < 0 or turn_index >= len(self.turns):
            raise IndexError(f"Unknown interaction turn index: {turn_index}")
        return turn_index
