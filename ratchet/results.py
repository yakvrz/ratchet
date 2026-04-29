from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from ratchet.io import append_jsonl, case_digest, stable_digest
from ratchet.transform_program import CompiledCandidate
from ratchet.types import (
    AgentSpec,
    EvalCase,
    GradeResult,
    OptimizationObjective,
    RunRecord,
)


@dataclass
class OptimizerStats:
    cache_hits: int = 0
    fresh_case_evaluations: int = 0
    retries: int = 0
    runtime_errors: int = 0
    timeouts: int = 0
    grader_errors: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class CaseEvaluation:
    case: EvalCase
    record: RunRecord
    grade: GradeResult
    cached: bool = False
    sample_index: int = 0

    def to_record(
        self,
        candidate_id_value: str,
        candidate: CompiledCandidate | None,
        *,
        cache_namespace: str,
    ) -> dict[str, Any]:
        return {
            "cache_namespace": cache_namespace,
            "candidate_id": candidate_id_value,
            "sample_index": self.sample_index,
            "candidate": candidate.to_dict() if candidate is not None else None,
            "case_digest": case_digest(self.case),
            "case": self.case.to_dict(),
            "record": self.record.to_dict(),
            "grade": self.grade.to_dict(),
        }

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> "CaseEvaluation":
        return cls(
            case=EvalCase.from_dict(payload["case"]),
            record=RunRecord.from_dict(payload["record"]),
            grade=GradeResult.from_dict(payload["grade"]),
            cached=True,
            sample_index=int(payload.get("sample_index", 0)),
        )


@dataclass
class CandidateSummary:
    candidate_id: str
    candidate: CompiledCandidate | None
    split: str
    evaluations: list[CaseEvaluation]

    @property
    def grouped_evaluations(self) -> dict[str, list[CaseEvaluation]]:
        grouped: dict[str, list[CaseEvaluation]] = defaultdict(list)
        for evaluation in self.evaluations:
            grouped[evaluation.case.id].append(evaluation)
        return {
            case_id: sorted(items, key=lambda item: item.sample_index)
            for case_id, items in sorted(grouped.items())
        }

    @property
    def case_count(self) -> int:
        return len(self.grouped_evaluations)

    @property
    def sample_count(self) -> int:
        return len(self.evaluations)

    @property
    def samples_per_case(self) -> float:
        if self.case_count == 0:
            return 0.0
        return self.sample_count / self.case_count

    def _case_rows(self) -> list[tuple[str, list[CaseEvaluation], float, float, bool]]:
        rows: list[tuple[str, list[CaseEvaluation], float, float, bool]] = []
        for case_id, evaluations in self.grouped_evaluations.items():
            mean_score = statistics.fmean(evaluation.grade.score for evaluation in evaluations)
            pass_vote_rate = statistics.fmean(float(evaluation.grade.passed) for evaluation in evaluations)
            rows.append((case_id, evaluations, mean_score, pass_vote_rate, pass_vote_rate > 0.5))
        return rows

    def _failed_case_rows(self) -> list[tuple[str, list[CaseEvaluation], float, float, bool]]:
        return [row for row in self._case_rows() if not row[4]]

    @property
    def mean_score(self) -> float:
        return statistics.fmean(row[2] for row in self._case_rows())

    @property
    def pass_rate(self) -> float:
        return statistics.fmean(float(row[4]) for row in self._case_rows())

    @property
    def pass_count(self) -> int:
        return sum(1 for row in self._case_rows() if row[4])

    @property
    def mean_cost_usd(self) -> float:
        return statistics.fmean(evaluation.record.metrics.cost_usd for evaluation in self.evaluations)

    @property
    def mean_total_tokens(self) -> float:
        return statistics.fmean(evaluation.record.metrics.total_tokens for evaluation in self.evaluations)

    @property
    def mean_model_calls(self) -> float:
        return statistics.fmean(evaluation.record.metrics.model_calls for evaluation in self.evaluations)

    @property
    def mean_tool_calls(self) -> float:
        return statistics.fmean(evaluation.record.metrics.tool_calls for evaluation in self.evaluations)

    @property
    def mean_turns(self) -> float:
        return statistics.fmean(evaluation.record.metrics.turns for evaluation in self.evaluations)

    @property
    def median_latency_s(self) -> float:
        return statistics.median(evaluation.record.metrics.latency_s for evaluation in self.evaluations)

    @property
    def runtime_error_count(self) -> int:
        return sum(1 for evaluation in self.evaluations if evaluation.record.metrics.error)

    @property
    def split_vote_case_ids(self) -> list[str]:
        return [
            case_id
            for case_id, _, _, pass_vote_rate, _ in self._case_rows()
            if 0.0 < pass_vote_rate < 1.0
        ]

    @property
    def operation_count(self) -> int:
        return len(self.candidate.program.patches) if self.candidate is not None else 0

    @property
    def failure_labels(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for _, evaluations, _, _, _ in self._failed_case_rows():
            for evaluation in evaluations:
                if not evaluation.grade.passed:
                    counts.update(evaluation.grade.labels or ["failed"])
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def category_metrics(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
        for _, evaluations, mean_score, _, case_passed in self._case_rows():
            category = str(evaluations[0].case.metadata.get("category", "uncategorized"))
            grouped[category].append((mean_score, case_passed))
        metrics: dict[str, dict[str, float | int]] = {}
        for category, rows in sorted(grouped.items()):
            metrics[category] = {
                "count": len(rows),
                "pass_count": sum(1 for _, passed in rows if passed),
                "mean_score": round(statistics.fmean(score for score, _ in rows), 4),
                "pass_rate": round(statistics.fmean(float(passed) for _, passed in rows), 4),
            }
        return metrics

    def failed_examples(
        self,
        limit: int = 10,
        *,
        max_text_chars: int | None = 1200,
        sanitize_text: bool = False,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for _, evaluations, mean_score, pass_vote_rate, _ in self._failed_case_rows():
            evaluation = next((item for item in evaluations if not item.grade.passed), evaluations[0])
            redacted = "[redacted by sanitize_examples]"
            rows.append(
                {
                    "case_id": evaluation.case.id,
                    "sample_index": evaluation.sample_index,
                    "case_mean_score": mean_score,
                    "case_pass_vote_rate": pass_vote_rate,
                    "input": redacted
                    if sanitize_text
                    else _compact_prompt_value(evaluation.case.input, max_text_chars=max_text_chars),
                    "expected": redacted
                    if sanitize_text
                    else _compact_prompt_value(evaluation.case.expected, max_text_chars=max_text_chars),
                    "score": evaluation.grade.score,
                    "labels": evaluation.grade.labels,
                    "notes": redacted
                    if sanitize_text
                    else _compact_prompt_value(evaluation.grade.notes, max_text_chars=max_text_chars),
                    "output": redacted
                    if sanitize_text
                    else _compact_prompt_value(evaluation.record.output, max_text_chars=max_text_chars),
                    "error": evaluation.record.metrics.error,
                    "tool_calls": evaluation.record.diagnostics.tool_calls,
                    "turns": [
                        {
                            "index": turn.index,
                            "actor": turn.actor,
                            "tool_calls": [tool_call.name for tool_call in turn.tool_calls],
                            "outcome": turn.outcome,
                        }
                        for turn in evaluation.record.diagnostics.turns[:8]
                    ],
                    "terminal_reason": evaluation.record.diagnostics.terminal_reason,
                    "raw_output_text": redacted
                    if sanitize_text
                    else _compact_prompt_value(
                        evaluation.record.diagnostics.raw_output_text,
                        max_text_chars=max_text_chars,
                    ),
                }
            )
        return rows[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate": self.candidate.to_dict() if self.candidate is not None else None,
            "split": self.split,
            "case_count": self.case_count,
            "sample_count": self.sample_count,
            "samples_per_case": self.samples_per_case,
            "behavioral": {
                "mean_score": self.mean_score,
                "pass_rate": self.pass_rate,
                "pass_count": self.pass_count,
                "failure_labels": self.failure_labels,
                "category_metrics": self.category_metrics,
                "split_vote_case_count": len(self.split_vote_case_ids),
                "split_vote_case_ids": self.split_vote_case_ids,
            },
            "operational": {
                "mean_cost_usd": self.mean_cost_usd,
                "mean_total_tokens": self.mean_total_tokens,
                "mean_model_calls": self.mean_model_calls,
                "mean_tool_calls": self.mean_tool_calls,
                "mean_turns": self.mean_turns,
                "median_latency_s": self.median_latency_s,
                "runtime_error_count": self.runtime_error_count,
            },
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "pass_count": self.pass_count,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_total_tokens": self.mean_total_tokens,
            "mean_model_calls": self.mean_model_calls,
            "mean_tool_calls": self.mean_tool_calls,
            "mean_turns": self.mean_turns,
            "median_latency_s": self.median_latency_s,
            "operation_count": self.operation_count,
            "evaluations": [
                {
                    "case": evaluation.case.to_dict(),
                    "sample_index": evaluation.sample_index,
                    "record": evaluation.record.to_dict(),
                    "grade": evaluation.grade.to_dict(),
                    "cached": evaluation.cached,
                }
                for evaluation in self.evaluations
            ],
        }


def _compact_prompt_value(value: Any, *, max_text_chars: int | None) -> Any:
    if max_text_chars is None:
        return value
    if isinstance(value, str):
        return _truncate_middle(value, max_text_chars)
    if isinstance(value, list):
        compacted = [_compact_prompt_value(item, max_text_chars=max_text_chars) for item in value]
        return _compact_container(compacted, max_text_chars=max_text_chars)
    if isinstance(value, dict):
        compacted = {
            str(key): _compact_prompt_value(item, max_text_chars=max_text_chars)
            for key, item in value.items()
        }
        return _compact_container(compacted, max_text_chars=max_text_chars)
    return value


def _compact_container(value: Any, *, max_text_chars: int) -> Any:
    try:
        rendered = json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return _truncate_middle(str(value), max_text_chars)
    if len(rendered) <= max_text_chars * 2:
        return value
    return _truncate_middle(rendered, max_text_chars)


def _truncate_middle(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars < 40:
        return text[:max_chars]
    omitted = len(text) - max_chars
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return f"{text[:head_chars]}... [truncated {omitted} chars] ...{text[-tail_chars:]}"


@dataclass(frozen=True)
class PassSignificance:
    fixed_count: int
    regressed_count: int
    n_discordant: int
    n_cases: int
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Comparison:
    score_delta: float
    score_ci: tuple[float, float]
    cost_delta: float
    cost_ci: tuple[float, float]
    token_delta: float
    token_ci: tuple[float, float]
    latency_delta: float
    latency_ci: tuple[float, float]
    model_call_delta: float = 0.0
    tool_call_delta: float = 0.0
    turn_delta: float = 0.0
    pass_significance: PassSignificance | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RatchetResult:
    baseline_candidate: CompiledCandidate | None
    selected_candidate: CompiledCandidate | None
    selected_candidate_id: str
    promoted: bool
    baseline_dev: CandidateSummary
    baseline_holdout: CandidateSummary
    best_dev_candidate: CandidateSummary
    selected_holdout: CandidateSummary
    accepted_dev_candidates: list[CandidateSummary]
    holdout_candidates: list[CandidateSummary]
    pareto_frontier: list[dict[str, Any]]
    decision_log: list[dict[str, Any]]
    diagnoses: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    generated_surface: list[dict[str, Any]]
    task_theories: list[dict[str, Any]]
    transform_summaries: dict[str, dict[str, Any]]
    transform_context_summaries: dict[str, dict[str, Any]]
    affordance_summaries: dict[str, dict[str, Any]]
    finalist_statuses: list[dict[str, Any]]
    runtime_reliability_diagnostics: list[dict[str, Any]]
    confirmation_results: list[dict[str, Any]]
    simplification_results: list[dict[str, Any]]
    frontier_recommendation: dict[str, Any]
    run_profile: dict[str, Any]
    quality_cost_tradeoffs: list[dict[str, Any]]
    optimizer_call_diagnostics: list[dict[str, Any]]
    ideation_metrics: dict[str, Any]
    evidence_ledger: dict[str, Any]
    selection_reason: str
    outcome_analysis: dict[str, Any]
    manifest: dict[str, Any]


class ResultStore:
    def __init__(self, out_dir: Path, *, cache_namespace: str) -> None:
        self.out_dir = out_dir
        self.case_results_path = out_dir / "case_results.jsonl"
        self.cache_namespace = cache_namespace
        self.records: dict[tuple[str, str, int], CaseEvaluation] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.case_results_path.exists():
            return
        for raw_line in self.case_results_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("cache_namespace") != self.cache_namespace:
                continue
            evaluation = CaseEvaluation.from_record(payload)
            if evaluation.record.metrics.error:
                continue
            stored_digest = payload.get("case_digest")
            if stored_digest != case_digest(evaluation.case):
                continue
            self.records[(payload["candidate_id"], stored_digest, evaluation.sample_index)] = evaluation

    def get(self, candidate_id_value: str, case: EvalCase, sample_index: int = 0) -> CaseEvaluation | None:
        return self.records.get((candidate_id_value, case_digest(case), sample_index))

    def put(self, candidate_id_value: str, candidate: CompiledCandidate | None, evaluation: CaseEvaluation) -> None:
        key = (candidate_id_value, case_digest(evaluation.case), evaluation.sample_index)
        if key in self.records:
            return
        if not evaluation.record.metrics.error:
            self.records[key] = evaluation
        append_jsonl(
            self.case_results_path,
            evaluation.to_record(candidate_id_value, candidate, cache_namespace=self.cache_namespace),
        )


def build_cache_namespace(
    *,
    agent_spec: AgentSpec | None,
    objective: OptimizationObjective,
    run_metadata: dict[str, Any],
) -> str:
    return stable_digest(
        {
            "cache_version": 4,
            "adapter": run_metadata.get("adapter"),
            "adapter_fingerprint": run_metadata.get("adapter_fingerprint"),
            "evals_sha256": run_metadata.get("evals_sha256"),
            "agent_spec_sha256": stable_digest(agent_spec.to_dict() if agent_spec else None),
            "objective": objective.to_dict(),
        }
    )


def split_cases(cases: Iterable[EvalCase]) -> tuple[tuple[EvalCase, ...], tuple[EvalCase, ...]]:
    dev = tuple(case for case in cases if case.split == "dev")
    holdout = tuple(case for case in cases if case.split == "holdout")
    if not dev:
        raise ValueError("Eval file must include at least one dev case.")
    if not holdout:
        raise ValueError("Eval file must include at least one holdout case.")
    return dev, holdout


def split_train_dev_holdout(
    cases: Iterable[EvalCase],
) -> tuple[tuple[EvalCase, ...], tuple[EvalCase, ...], tuple[EvalCase, ...]]:
    rows = tuple(cases)
    train = tuple(case for case in rows if case.split == "train")
    dev, holdout = split_cases(rows)
    return train, dev, holdout
