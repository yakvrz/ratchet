from __future__ import annotations

from collections import Counter, defaultdict
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import signal
import statistics
import time
from typing import Any, Iterable

from ratchet.adapters import AdapterProtocol
from ratchet.code_artifacts import validate_code_artifact_source
from ratchet.io import (
    append_jsonl,
    candidate_hash,
    depends_on_satisfied,
    normalize_candidate,
    write_json,
    write_jsonl,
)
from ratchet.openai_client import OpenAIResponsesClient
from ratchet.types import (
    ComponentSpec,
    CodeArtifactSpec,
    DiagnosticTrace,
    EnumKnobSpec,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OperationalMetrics,
    PatchChange,
    PatchProposal,
    RunRecord,
    SearchSpace,
    TextArtifactSpec,
)


NON_INFERIORITY_MARGIN = 0.01
LATENCY_GUARD_MULTIPLIER = 1.15
BEHAVIOR_PHASE_GUARD_MULTIPLIER = 3.0
MAX_PROPOSALS_PER_ITERATION = 3
DIAGNOSIS_CATEGORIES = {
    "output_contract",
    "missing_tool",
    "tool_misuse",
    "retrieval_scope",
    "grounding",
    "arithmetic",
    "fallback_behavior",
    "prompt_ambiguity",
    "unknown",
}


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

    def to_record(self, candidate_hash_value: str, candidate: dict[str, str]) -> dict[str, Any]:
        return {
            "candidate_hash": candidate_hash_value,
            "candidate": dict(candidate),
            "case": self.case.to_dict(),
            "record": self.record.to_dict(),
            "grade": self.grade.to_dict(),
        }

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> "CaseEvaluation":
        if "record" in payload:
            record = RunRecord.from_dict(payload["record"])
        else:
            legacy_trace = payload["trace"]
            record = RunRecord(
                output=legacy_trace.get("answer", ""),
                metrics=OperationalMetrics(
                    latency_s=float(legacy_trace.get("latency_s", 0.0)),
                    input_tokens=int(legacy_trace.get("input_tokens", 0)),
                    output_tokens=int(legacy_trace.get("output_tokens", 0)),
                    total_tokens=int(legacy_trace.get("total_tokens", 0)),
                    cost_usd=float(legacy_trace.get("cost_usd", 0.0)),
                    error=legacy_trace.get("error"),
                ),
                diagnostics=DiagnosticTrace(
                    tool_calls=[str(item) for item in legacy_trace.get("tool_calls", [])],
                    raw_output_text=str(legacy_trace.get("answer", "")),
                    metadata=dict(legacy_trace.get("metadata", {})),
                ),
            )
        return cls(
            case=EvalCase.from_dict(payload["case"]),
            record=record,
            grade=GradeResult.from_dict(payload["grade"]),
            cached=True,
        )


@dataclass
class CandidateSummary:
    candidate_hash: str
    candidate: dict[str, str]
    split: str
    evaluations: list[CaseEvaluation]

    @property
    def mean_score(self) -> float:
        return statistics.fmean(evaluation.grade.score for evaluation in self.evaluations)

    @property
    def pass_rate(self) -> float:
        return statistics.fmean(float(evaluation.grade.passed) for evaluation in self.evaluations)

    @property
    def pass_count(self) -> int:
        return sum(1 for evaluation in self.evaluations if evaluation.grade.passed)

    @property
    def mean_cost_usd(self) -> float:
        return statistics.fmean(evaluation.record.metrics.cost_usd for evaluation in self.evaluations)

    @property
    def mean_total_tokens(self) -> float:
        return statistics.fmean(evaluation.record.metrics.total_tokens for evaluation in self.evaluations)

    @property
    def median_latency_s(self) -> float:
        return statistics.median(evaluation.record.metrics.latency_s for evaluation in self.evaluations)

    @property
    def runtime_error_count(self) -> int:
        return sum(1 for evaluation in self.evaluations if evaluation.record.metrics.error)

    @property
    def failure_labels(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for evaluation in self.evaluations:
            if evaluation.grade.passed:
                continue
            counts.update(evaluation.grade.labels or ["failed"])
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    @property
    def category_metrics(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[CaseEvaluation]] = defaultdict(list)
        for evaluation in self.evaluations:
            category = str(evaluation.case.metadata.get("category", "uncategorized"))
            grouped[category].append(evaluation)
        metrics: dict[str, dict[str, float | int]] = {}
        for category, evaluations in sorted(grouped.items()):
            metrics[category] = {
                "count": len(evaluations),
                "pass_count": sum(1 for item in evaluations if item.grade.passed),
                "mean_score": round(statistics.fmean(item.grade.score for item in evaluations), 4),
                "pass_rate": round(statistics.fmean(float(item.grade.passed) for item in evaluations), 4),
            }
        return metrics

    def failed_examples(self, limit: int = 10) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for evaluation in self.evaluations:
            if evaluation.grade.passed:
                continue
            rows.append(
                {
                    "case_id": evaluation.case.id,
                    "input": evaluation.case.input,
                    "expected": evaluation.case.expected,
                    "score": evaluation.grade.score,
                    "labels": evaluation.grade.labels,
                    "notes": evaluation.grade.notes,
                    "output": evaluation.record.output,
                    "error": evaluation.record.metrics.error,
                    "tool_calls": evaluation.record.diagnostics.tool_calls,
                    "raw_output_text": evaluation.record.diagnostics.raw_output_text,
                }
            )
        return rows[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "candidate": dict(self.candidate),
            "split": self.split,
            "behavioral": {
                "mean_score": self.mean_score,
                "pass_rate": self.pass_rate,
                "pass_count": self.pass_count,
                "failure_labels": self.failure_labels,
                "category_metrics": self.category_metrics,
            },
            "operational": {
                "mean_cost_usd": self.mean_cost_usd,
                "mean_total_tokens": self.mean_total_tokens,
                "median_latency_s": self.median_latency_s,
                "runtime_error_count": self.runtime_error_count,
            },
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "pass_count": self.pass_count,
            "mean_cost_usd": self.mean_cost_usd,
            "mean_total_tokens": self.mean_total_tokens,
            "median_latency_s": self.median_latency_s,
            "failure_labels": self.failure_labels,
            "category_metrics": self.category_metrics,
            "evaluations": [
                {
                    "case": evaluation.case.to_dict(),
                    "record": evaluation.record.to_dict(),
                    "grade": evaluation.grade.to_dict(),
                    "cached": evaluation.cached,
                }
                for evaluation in self.evaluations
            ],
        }


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RatchetResult:
    baseline_candidate: dict[str, str]
    selected_candidate: dict[str, str]
    selected_candidate_hash: str
    promoted: bool
    baseline_dev: CandidateSummary
    baseline_holdout: CandidateSummary
    best_dev_candidate: CandidateSummary
    selected_holdout: CandidateSummary
    accepted_dev_candidates: list[CandidateSummary]
    holdout_candidates: list[CandidateSummary]
    promotable_frontier: list[dict[str, Any]]
    decision_log: list[dict[str, Any]]
    diagnoses: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    selection_reason: str
    manifest: dict[str, Any]


class ResultStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.case_results_path = out_dir / "case_results.jsonl"
        self.records: dict[tuple[str, str], CaseEvaluation] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.case_results_path.exists():
            return
        for raw_line in self.case_results_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            evaluation = CaseEvaluation.from_record(payload)
            if evaluation.record.metrics.error:
                continue
            key = (payload["candidate_hash"], payload["case"]["id"])
            self.records[key] = evaluation

    def get(self, candidate_hash_value: str, case_id: str) -> CaseEvaluation | None:
        return self.records.get((candidate_hash_value, case_id))

    def put(self, candidate_hash_value: str, candidate: dict[str, str], evaluation: CaseEvaluation) -> None:
        key = (candidate_hash_value, evaluation.case.id)
        if key in self.records:
            return
        self.records[key] = evaluation
        append_jsonl(self.case_results_path, evaluation.to_record(candidate_hash_value, candidate))


def bootstrap_mean_ci(values: list[float], iterations: int = 4000, seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        boot = [values[rng.randrange(len(values))] for _ in range(len(values))]
        samples.append(statistics.fmean(boot))
    samples.sort()
    lower_index = int(0.025 * iterations)
    upper_index = int(0.975 * iterations)
    return samples[lower_index], samples[upper_index]


def split_cases(cases: Iterable[EvalCase]) -> tuple[tuple[EvalCase, ...], tuple[EvalCase, ...]]:
    dev = tuple(case for case in cases if case.split == "dev")
    holdout = tuple(case for case in cases if case.split == "holdout")
    if not dev:
        raise ValueError("Eval file must include at least one dev case.")
    if not holdout:
        raise ValueError("Eval file must include at least one holdout case.")
    return dev, holdout


@contextlib.contextmanager
def case_timeout(timeout_s: int) -> Iterable[None]:
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Case exceeded {timeout_s} second timeout.")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def build_candidate(candidate: dict[str, str], search_space: SearchSpace) -> dict[str, str]:
    return normalize_candidate(candidate, search_space)


def apply_patch_proposal(
    parent_candidate: dict[str, str],
    proposal: PatchProposal,
    search_space: SearchSpace,
) -> dict[str, str]:
    candidate = dict(parent_candidate)
    for change in proposal.changes:
        candidate[change.name] = change.value
    return normalize_candidate(candidate, search_space)


def generate_enum_mutation_proposals(
    parent_candidate: dict[str, str],
    search_space: SearchSpace,
    *,
    phase: str,
    target_names: set[str] | None = None,
    seen_hashes: set[str] | None = None,
) -> list[PatchProposal]:
    seen_hashes = seen_hashes or set()
    proposals: list[PatchProposal] = []
    enum_knobs = search_space.enum_knobs
    if target_names is not None:
        enum_knobs = [spec for spec in enum_knobs if spec.name in target_names]
    for spec in enum_knobs:
        if not depends_on_satisfied(parent_candidate, spec):
            continue
        current_value = parent_candidate[spec.name]
        current_index = spec.values.index(current_value)
        if phase == "efficiency":
            ordered_values = spec.values[current_index + 1 :] + spec.values[:current_index]
        else:
            ordered_values = spec.values[:current_index] + spec.values[current_index + 1 :]
            if not ordered_values:
                ordered_values = spec.values[current_index + 1 :]
        for value in ordered_values:
            if value == current_value:
                continue
            proposal = PatchProposal(
                proposal_id=f"enum::{spec.name}::{value}",
                diagnosis_category="enum_mutation",
                changes=[PatchChange(op="set_enum", name=spec.name, value=value)],
                rationale=spec.description or f"Change {spec.name} to {value}.",
                expected_effect=f"Adjust {spec.name} from {current_value} to {value}.",
            )
            candidate = apply_patch_proposal(parent_candidate, proposal, search_space)
            if candidate == parent_candidate:
                continue
            if candidate_hash(candidate) in seen_hashes:
                continue
            proposals.append(proposal)
    return proposals


def generate_component_mutation_proposals(
    parent_candidate: dict[str, str],
    search_space: SearchSpace,
    *,
    phase: str,
    target_names: set[str] | None = None,
    seen_hashes: set[str] | None = None,
) -> list[PatchProposal]:
    seen_hashes = seen_hashes or set()
    proposals: list[PatchProposal] = []
    components = search_space.components
    if target_names is not None:
        components = [spec for spec in components if spec.name in target_names]
    for spec in components:
        if not depends_on_satisfied(parent_candidate, spec):
            continue
        current_value = parent_candidate[spec.name]
        current_index = spec.values.index(current_value)
        if phase == "efficiency":
            ordered_values = spec.values[current_index + 1 :] + spec.values[:current_index]
        else:
            ordered_values = spec.values[:current_index] + spec.values[current_index + 1 :]
            if not ordered_values:
                ordered_values = spec.values[current_index + 1 :]
        for value in ordered_values:
            if value == current_value:
                continue
            proposal = PatchProposal(
                proposal_id=f"component::{spec.name}::{value}",
                diagnosis_category="component_mutation",
                changes=[PatchChange(op="set_component", name=spec.name, value=value)],
                rationale=spec.description or f"Change component {spec.name} to {value}.",
                expected_effect=f"Adjust component {spec.name} from {current_value} to {value}.",
            )
            candidate = apply_patch_proposal(parent_candidate, proposal, search_space)
            if candidate == parent_candidate:
                continue
            if candidate_hash(candidate) in seen_hashes:
                continue
            proposals.append(proposal)
    return proposals


def compare_summaries(reference: CandidateSummary, candidate: CandidateSummary) -> Comparison:
    reference_by_id = {evaluation.case.id: evaluation for evaluation in reference.evaluations}
    candidate_by_id = {evaluation.case.id: evaluation for evaluation in candidate.evaluations}
    if set(reference_by_id) != set(candidate_by_id):
        raise ValueError("Candidate summaries must cover the same cases for paired comparison.")
    case_ids = [evaluation.case.id for evaluation in reference.evaluations]
    score_deltas = [
        candidate_by_id[case_id].grade.score - reference_by_id[case_id].grade.score
        for case_id in case_ids
    ]
    cost_deltas = [
        candidate_by_id[case_id].record.metrics.cost_usd - reference_by_id[case_id].record.metrics.cost_usd
        for case_id in case_ids
    ]
    token_deltas = [
        float(
            candidate_by_id[case_id].record.metrics.total_tokens
            - reference_by_id[case_id].record.metrics.total_tokens
        )
        for case_id in case_ids
    ]
    latency_deltas = [
        candidate_by_id[case_id].record.metrics.latency_s - reference_by_id[case_id].record.metrics.latency_s
        for case_id in case_ids
    ]
    return Comparison(
        score_delta=statistics.fmean(score_deltas),
        score_ci=bootstrap_mean_ci(score_deltas),
        cost_delta=statistics.fmean(cost_deltas),
        cost_ci=bootstrap_mean_ci(cost_deltas),
        token_delta=statistics.fmean(token_deltas),
        token_ci=bootstrap_mean_ci(token_deltas),
        latency_delta=statistics.fmean(latency_deltas),
        latency_ci=bootstrap_mean_ci(latency_deltas),
    )


def behavior_flip_summary(reference: CandidateSummary, candidate: CandidateSummary) -> dict[str, Any]:
    reference_by_id = {evaluation.case.id: evaluation for evaluation in reference.evaluations}
    candidate_by_id = {evaluation.case.id: evaluation for evaluation in candidate.evaluations}
    if set(reference_by_id) != set(candidate_by_id):
        raise ValueError("Candidate summaries must cover the same cases for flip comparison.")
    fixed: list[str] = []
    regressed: list[str] = []
    changed_by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"fixed": 0, "regressed": 0})
    for case_id, reference_eval in reference_by_id.items():
        candidate_eval = candidate_by_id[case_id]
        if reference_eval.grade.passed == candidate_eval.grade.passed:
            continue
        category = str(reference_eval.case.metadata.get("category", "uncategorized"))
        if not reference_eval.grade.passed and candidate_eval.grade.passed:
            fixed.append(case_id)
            changed_by_category[category]["fixed"] += 1
        elif reference_eval.grade.passed and not candidate_eval.grade.passed:
            regressed.append(case_id)
            changed_by_category[category]["regressed"] += 1
    return {
        "fixed_case_ids": fixed,
        "regressed_case_ids": regressed,
        "fixed_count": len(fixed),
        "regressed_count": len(regressed),
        "by_category": dict(sorted(changed_by_category.items())),
    }


def final_gate(baseline: CandidateSummary, candidate: CandidateSummary) -> tuple[bool, Comparison]:
    comparison = compare_summaries(baseline, candidate)
    quality_pass = comparison.score_ci[0] >= -NON_INFERIORITY_MARGIN
    efficiency_pass = comparison.cost_ci[1] < 0 and comparison.token_ci[1] < 0
    latency_pass = candidate.median_latency_s <= baseline.median_latency_s * LATENCY_GUARD_MULTIPLIER
    return quality_pass and efficiency_pass and latency_pass, comparison


class FailureDiagnoser:
    def __init__(
        self,
        *,
        env_path: str,
        model: str,
        reasoning_effort: str,
        enabled: bool,
    ) -> None:
        self.env_path = env_path
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.enabled = enabled
        self._client: OpenAIResponsesClient | None = None

    def diagnose(
        self,
        summary: CandidateSummary,
        search_space: SearchSpace,
    ) -> tuple[list[FailureDiagnosis], str]:
        failures = [evaluation for evaluation in summary.evaluations if not evaluation.grade.passed]
        if not failures:
            return [], "No failing cases on the current eval set."

        diagnosed: list[FailureDiagnosis] = []
        unresolved: list[CaseEvaluation] = []
        for evaluation in failures:
            diagnosis = self._rule_based(summary.candidate, evaluation, search_space)
            if diagnosis is None:
                unresolved.append(evaluation)
            else:
                diagnosed.append(diagnosis)

        analysis = "Used rule-based diagnosis."
        if unresolved and self.enabled:
            try:
                llm_diagnoses = self._llm_diagnose(summary, unresolved, search_space)
                if llm_diagnoses:
                    diagnosed.extend(llm_diagnoses)
                    analysis = "Used rule-based diagnosis with LLM fallback."
                else:
                    analysis = "LLM diagnoser returned no valid diagnoses; fell back to unknown."
            except Exception:
                analysis = "LLM diagnoser failed; fell back to unknown."

        diagnosed_case_ids = {case_id for diagnosis in diagnosed for case_id in diagnosis.case_ids}
        for evaluation in unresolved:
            if evaluation.case.id in diagnosed_case_ids:
                continue
            diagnosed.append(
                FailureDiagnosis(
                    case_ids=[evaluation.case.id],
                    category="unknown",
                    root_cause="Could not classify the failure confidently.",
                    target_keys=self._default_prompt_targets(search_space),
                    evidence=[self._evidence_row(evaluation)],
                )
            )

        grouped: dict[str, FailureDiagnosis] = {}
        for diagnosis in diagnosed:
            existing = grouped.get(diagnosis.category)
            if existing is None:
                grouped[diagnosis.category] = diagnosis
                continue
            case_ids = sorted(set(existing.case_ids) | set(diagnosis.case_ids))
            target_keys = list(dict.fromkeys([*existing.target_keys, *diagnosis.target_keys]))
            evidence = [*existing.evidence, *diagnosis.evidence]
            grouped[diagnosis.category] = FailureDiagnosis(
                case_ids=case_ids,
                category=diagnosis.category,
                root_cause=existing.root_cause,
                target_keys=target_keys,
                evidence=evidence,
            )

        ordered = sorted(
            grouped.values(),
            key=lambda item: (-len(item.case_ids), item.category),
        )
        return ordered, analysis

    def _rule_based(
        self,
        candidate: dict[str, str],
        evaluation: CaseEvaluation,
        search_space: SearchSpace,
    ) -> FailureDiagnosis | None:
        labels = set(evaluation.grade.labels)
        metrics_error = evaluation.record.metrics.error or ""
        output = evaluation.record.output
        output_text = json.dumps(output, sort_keys=True) if output is not None else ""
        tool_calls = evaluation.record.diagnostics.tool_calls
        prompt_targets = self._default_prompt_targets(search_space)

        if "timeout" in labels or "TimeoutError" in metrics_error:
            return self._single_case_diagnosis(
                evaluation,
                category="unknown",
                root_cause="The case timed out during execution.",
                target_keys=[],
            )
        if "runtime_error" in labels or "grader_error" in labels:
            return self._single_case_diagnosis(
                evaluation,
                category="unknown",
                root_cause="The case raised a runtime or grading error.",
                target_keys=[],
            )
        if "invalid_output" in labels or (
            isinstance(output, dict) and "invalid_output" in output
        ):
            return self._single_case_diagnosis(
                evaluation,
                category="output_contract",
                root_cause="The harness returned malformed or externally invalid output.",
                target_keys=[
                    name
                    for name in prompt_targets
                    if "output" in name.lower() or "answer" in name.lower()
                ]
                or prompt_targets,
            )
        if "numeric_mismatch" in labels or "wrong_math_answer" in labels:
            return self._single_case_diagnosis(
                evaluation,
                category="arithmetic",
                root_cause="The answer failed an arithmetic or numeric check.",
                target_keys=self._calculator_targets(search_space),
            )

        expected_payload = evaluation.case.expected
        if (
            isinstance(expected_payload, dict)
            and str(expected_payload.get("answer", "")).strip().lower() == "unknown"
            and not self._is_unknown_output(output_text)
        ):
            validator_targets = self._validator_targets(search_space)
            fallback_targets = [
                name
                for name in prompt_targets
                if (
                    "fallback" in name.lower()
                    or "ground" in name.lower()
                    or "unknown" in name.lower()
                    or "output" in name.lower()
                    or "answer" in name.lower()
                )
            ]
            return self._single_case_diagnosis(
                evaluation,
                category="fallback_behavior",
                root_cause="The harness inferred an unsupported answer instead of returning unknown.",
                target_keys=[*validator_targets, *(fallback_targets or prompt_targets)],
            )

        off_tools = [
            spec.name
            for spec in search_space.enum_knobs
            if spec.kind == "tool" and candidate.get(spec.name) == "off"
        ]
        on_tools = [
            spec.name
            for spec in search_space.enum_knobs
            if spec.kind == "tool" and candidate.get(spec.name) == "on"
        ]
        if evaluation.case.metadata.get("needs_tool") and off_tools:
            return self._single_case_diagnosis(
                evaluation,
                category="missing_tool",
                root_cause="The case likely requires a tool that is currently disabled.",
                target_keys=[off_tools[0], *self._tool_prompt_targets(search_space)],
            )
        if evaluation.case.metadata.get("category") == "math":
            calculator_off = [
                name for name in off_tools if "calc" in name.lower() or "calculator" in name.lower()
            ]
            if calculator_off:
                return self._single_case_diagnosis(
                    evaluation,
                    category="arithmetic",
                    root_cause="A calculator-like tool is disabled on a math case.",
                    target_keys=[calculator_off[0], *self._calculator_targets(search_space)],
                )
        non_math_off_tools = [
            name
            for name in off_tools
            if "calc" not in name.lower() and "calculator" not in name.lower()
        ]
        if non_math_off_tools and evaluation.case.metadata.get("category") != "math":
            return self._single_case_diagnosis(
                evaluation,
                category="missing_tool",
                root_cause="A likely retrieval or lookup tool is disabled on a knowledge-seeking case.",
                target_keys=[non_math_off_tools[0], *self._tool_prompt_targets(search_space)],
            )
        if on_tools and not tool_calls:
            return self._single_case_diagnosis(
                evaluation,
                category="missing_tool",
                root_cause="A relevant tool was available but never called.",
                target_keys=[*self._tool_prompt_targets(search_space), *on_tools],
            )
        if self._is_unknown_output(output_text):
            fallback_targets = [
                name for name in prompt_targets if "fallback" in name.lower() or "unknown" in name.lower()
            ]
            return self._single_case_diagnosis(
                evaluation,
                category="fallback_behavior",
                root_cause="The harness fell back instead of producing a grounded answer.",
                target_keys=[*self._validator_targets(search_space), *(fallback_targets or prompt_targets)],
            )
        if tool_calls:
            target_keys = [
                spec.name
                for spec in search_space.enum_knobs
                if spec.kind in {"kb", "param"} and ("retrieval" in spec.name or "knowledge" in spec.name)
            ]
            target_keys.extend(
                name
                for name in prompt_targets
                if "ground" in name.lower() or "tool" in name.lower()
            )
            target_keys.extend(
                spec.name
                for spec in search_space.text_artifacts
                if spec.kind == "tool"
            )
            return self._single_case_diagnosis(
                evaluation,
                category="retrieval_scope",
                root_cause="The tool was used, but the retrieved or grounded context appears insufficient.",
                target_keys=list(dict.fromkeys(target_keys)),
            )

        if prompt_targets:
            return self._single_case_diagnosis(
                evaluation,
                category="prompt_ambiguity",
                root_cause="The current prompt or tool guidance is not steering the model to the right behavior.",
                target_keys=prompt_targets,
            )
        return None

    def _llm_diagnose(
        self,
        summary: CandidateSummary,
        unresolved: list[CaseEvaluation],
        search_space: SearchSpace,
    ) -> list[FailureDiagnosis]:
        if self._client is None:
            self._client = OpenAIResponsesClient(env_path=self.env_path)
        prompt = {
            "candidate": summary.candidate,
            "allowed_categories": sorted(DIAGNOSIS_CATEGORIES),
            "enum_knobs": [spec.to_dict() for spec in search_space.enum_knobs],
            "text_artifacts": [spec.to_dict() for spec in search_space.text_artifacts],
            "components": [spec.to_dict() for spec in search_space.components],
            "cases": [
                {
                    "case_id": evaluation.case.id,
                    "input": evaluation.case.input,
                    "expected": evaluation.case.expected,
                    "labels": evaluation.grade.labels,
                    "notes": evaluation.grade.notes,
                    "output": evaluation.record.output,
                    "error": evaluation.record.metrics.error,
                    "tool_calls": evaluation.record.diagnostics.tool_calls,
                    "raw_output_text": evaluation.record.diagnostics.raw_output_text,
                }
                for evaluation in unresolved
            ],
        }
        response = self._client.create_response(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=(
                "You are Ratchet's diagnoser. Classify each failed case into one allowed category and suggest "
                "the most relevant target keys in the declared harness search space. Return strict JSON with key "
                "diagnoses, an array of objects with case_ids, category, root_cause, and target_keys.\n\n"
                f"{json.dumps(prompt, indent=2)}"
            ),
            max_output_tokens=700,
        )
        payload = self._extract_json_object(response.output_text)
        diagnoses: list[FailureDiagnosis] = []
        for raw_diagnosis in payload.get("diagnoses", []):
            category = str(raw_diagnosis.get("category", "unknown"))
            if category not in DIAGNOSIS_CATEGORIES:
                category = "unknown"
            target_keys = [
                key
                for key in [str(item) for item in raw_diagnosis.get("target_keys", [])]
                if key in search_space.spec_names()
            ]
            diagnoses.append(
                FailureDiagnosis(
                    case_ids=[str(item) for item in raw_diagnosis.get("case_ids", [])],
                    category=category,
                    root_cause=str(raw_diagnosis.get("root_cause", "")) or "LLM diagnosis",
                    target_keys=target_keys or self._default_prompt_targets(search_space),
                    evidence=[],
                )
            )
        return diagnoses

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in diagnoser response.")
        return json.loads(match.group(0))

    @staticmethod
    def _evidence_row(evaluation: CaseEvaluation) -> dict[str, Any]:
        return {
            "case_id": evaluation.case.id,
            "input": evaluation.case.input,
            "expected": evaluation.case.expected,
            "labels": list(evaluation.grade.labels),
            "notes": evaluation.grade.notes,
            "output": evaluation.record.output,
            "error": evaluation.record.metrics.error,
            "tool_calls": list(evaluation.record.diagnostics.tool_calls),
            "raw_output_text": evaluation.record.diagnostics.raw_output_text,
        }

    def _single_case_diagnosis(
        self,
        evaluation: CaseEvaluation,
        *,
        category: str,
        root_cause: str,
        target_keys: list[str],
    ) -> FailureDiagnosis:
        return FailureDiagnosis(
            case_ids=[evaluation.case.id],
            category=category,
            root_cause=root_cause,
            target_keys=target_keys,
            evidence=[self._evidence_row(evaluation)],
        )

    @staticmethod
    def _is_unknown_output(output_text: str) -> bool:
        lowered = output_text.strip().lower()
        return lowered == "unknown" or '"answer": "unknown"' in lowered

    @staticmethod
    def _default_prompt_targets(search_space: SearchSpace) -> list[str]:
        return [spec.name for spec in search_space.text_artifacts if spec.kind == "prompt"]

    @staticmethod
    def _tool_prompt_targets(search_space: SearchSpace) -> list[str]:
        return [
            spec.name
            for spec in search_space.text_artifacts
            if spec.kind == "tool" or "tool" in spec.name.lower()
        ] + [
            spec.name
            for spec in search_space.text_artifacts
            if spec.kind == "prompt" and ("tool" in spec.name.lower() or "search" in spec.name.lower())
        ]

    @staticmethod
    def _calculator_targets(search_space: SearchSpace) -> list[str]:
        targets: list[str] = []
        for spec in search_space.enum_knobs:
            if "calc" in spec.name.lower() or "calculator" in spec.name.lower():
                targets.append(spec.name)
        for spec in search_space.components:
            if "calc" in spec.name.lower() or "calculator" in spec.name.lower():
                targets.append(spec.name)
        for spec in search_space.text_artifacts:
            if "calc" in spec.name.lower() or "calculator" in spec.name.lower():
                targets.append(spec.name)
        return targets

    @staticmethod
    def _validator_targets(search_space: SearchSpace) -> list[str]:
        targets: list[str] = []
        for spec in search_space.components:
            if spec.kind == "validator" or "validator" in spec.name.lower():
                targets.append(spec.name)
        for spec in search_space.text_artifacts:
            if spec.kind == "component" and ("validator" in spec.name.lower() or "ground" in spec.name.lower()):
                targets.append(spec.name)
        for spec in search_space.code_artifacts:
            lowered = spec.name.lower()
            if "validator" in lowered or "ground" in lowered:
                targets.append(spec.name)
        return targets


class ProposalEngine:
    def __init__(
        self,
        *,
        env_path: str,
        model: str,
        reasoning_effort: str,
        enabled: bool,
    ) -> None:
        self.env_path = env_path
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.enabled = enabled
        self._client: OpenAIResponsesClient | None = None

    def propose(
        self,
        summary: CandidateSummary,
        search_space: SearchSpace,
        *,
        phase: str,
        diagnosis: FailureDiagnosis | None,
        seen_hashes: set[str],
        history: list[dict[str, Any]],
    ) -> tuple[list[PatchProposal], str]:
        llm_validated: list[PatchProposal] = []
        analysis_parts: list[str] = []
        if self.enabled:
            try:
                proposals = self._llm_proposals(
                    summary,
                    search_space,
                    phase=phase,
                    diagnosis=diagnosis,
                    history=history,
                )
                validated = self._validate_and_filter(
                    proposals,
                    summary,
                    search_space,
                    diagnosis=diagnosis,
                    seen_hashes=seen_hashes,
                )
                if validated:
                    llm_validated = validated
                    analysis_parts.append("LLM proposer generated structural proposals.")
            except Exception:
                analysis_parts.append("LLM proposer failed.")

        heuristic = self._heuristic_proposals(summary, search_space, phase=phase, diagnosis=diagnosis)
        heuristic_validated = self._validate_and_filter(
            heuristic,
            summary,
            search_space,
            diagnosis=diagnosis,
            seen_hashes=seen_hashes,
        )
        merged: list[PatchProposal] = []
        seen_ids: set[str] = set()
        for proposal in [*llm_validated, *heuristic_validated]:
            if proposal.proposal_id in seen_ids:
                continue
            seen_ids.add(proposal.proposal_id)
            merged.append(proposal)
        if phase == "efficiency":
            merged.sort(
                key=lambda proposal: (
                    self._scope_rank(proposal.estimated_scope),
                    self._efficiency_priority(proposal),
                    len(proposal.changes),
                    proposal.proposal_id,
                )
            )
        else:
            merged.sort(
                key=lambda proposal: (
                    self._behavior_priority(proposal),
                    self._scope_rank(proposal.estimated_scope),
                    len(proposal.changes),
                    proposal.proposal_id,
                )
            )
        if heuristic_validated:
            if llm_validated:
                analysis_parts.append("Merged heuristic structural proposals.")
            else:
                analysis_parts.append("Fell back to heuristic structural proposals.")
        elif not llm_validated:
            analysis_parts.append("No valid proposals.")
        return merged[:MAX_PROPOSALS_PER_ITERATION], " ".join(analysis_parts)

    def _llm_proposals(
        self,
        summary: CandidateSummary,
        search_space: SearchSpace,
        *,
        phase: str,
        diagnosis: FailureDiagnosis | None,
        history: list[dict[str, Any]],
    ) -> list[PatchProposal]:
        if self._client is None:
            self._client = OpenAIResponsesClient(env_path=self.env_path)
        history_summary = self._history_summary(history)
        prompt = {
            "phase": phase,
            "candidate": summary.candidate,
            "behavior": {
                "mean_score": summary.mean_score,
                "pass_count": summary.pass_count,
                "pass_rate": summary.pass_rate,
                "failure_labels": summary.failure_labels,
            },
            "operational": {
                "mean_cost_usd": summary.mean_cost_usd,
                "mean_total_tokens": summary.mean_total_tokens,
                "median_latency_s": summary.median_latency_s,
            },
            "diagnosis": diagnosis.to_dict() if diagnosis is not None else None,
            "enum_knobs": [
                {
                    **spec.to_dict(),
                    "current": summary.candidate[spec.name],
                }
                for spec in search_space.enum_knobs
            ],
            "text_artifacts": [
                {
                    **spec.to_dict(),
                    "current": summary.candidate[spec.name],
                }
                for spec in search_space.text_artifacts
            ],
            "components": [
                {
                    **spec.to_dict(),
                    "current": summary.candidate[spec.name],
                }
                for spec in search_space.components
            ],
            "code_artifacts": [
                {
                    **spec.to_dict(),
                    "current": summary.candidate[spec.name],
                }
                for spec in search_space.code_artifacts
            ],
            "failed_examples": summary.failed_examples(),
            "history": history_summary,
        }
        response = self._client.create_response(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=(
                "You are Ratchet's proposer. Produce up to three bounded harness proposals. "
                "Only use operations set_enum(name, value), set_component(name, value), rewrite_text(name, new_text), "
                "and rewrite_code(name, new_code). "
                "Touch at most 2 keys per proposal. Do not mention case IDs or copy case-specific inputs. "
                "Generalize the fix so it addresses the diagnosed failure category rather than a single example. "
                "Return strict JSON with key proposals, an array of objects with changes, rationale, and expected_effect.\n\n"
                f"{json.dumps(prompt, indent=2)}"
            ),
            max_output_tokens=900,
        )
        payload = FailureDiagnoser._extract_json_object(response.output_text)
        proposals: list[PatchProposal] = []
        for index, raw_proposal in enumerate(payload.get("proposals", []), start=1):
            changes = [PatchChange.from_dict(item) for item in raw_proposal.get("changes", [])]
            proposals.append(
                PatchProposal(
                    proposal_id=str(raw_proposal.get("proposal_id", f"llm-{index}")),
                    diagnosis_category=(
                        diagnosis.category if diagnosis is not None else str(raw_proposal.get("diagnosis_category", phase))
                    ),
                    changes=changes,
                    rationale=str(raw_proposal.get("rationale", "")),
                    expected_effect=str(raw_proposal.get("expected_effect", "")),
                    estimated_scope=str(raw_proposal.get("estimated_scope", "low")),
                )
            )
        return proposals

    @staticmethod
    def _history_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
        accepted_categories: Counter[str] = Counter()
        rejected_categories: Counter[str] = Counter()
        rejection_reasons: Counter[str] = Counter()
        for row in history[-20:]:
            category = str(row.get("diagnosis_category", "unknown"))
            if row.get("accepted"):
                accepted_categories[category] += 1
            else:
                rejected_categories[category] += 1
                reason = str(row.get("rejection_reason") or "rejected")
                rejection_reasons[reason] += 1
        return {
            "accepted_categories": dict(accepted_categories),
            "rejected_categories": dict(rejected_categories),
            "common_rejection_reasons": dict(rejection_reasons.most_common(5)),
        }

    def _validate_and_filter(
        self,
        proposals: list[PatchProposal],
        summary: CandidateSummary,
        search_space: SearchSpace,
        *,
        diagnosis: FailureDiagnosis | None,
        seen_hashes: set[str],
    ) -> list[PatchProposal]:
        validated: list[PatchProposal] = []
        seen_candidates: set[str] = set()
        for proposal in proposals:
            if len(proposal.changes) > 2:
                continue
            change_names = [change.name for change in proposal.changes]
            if len(change_names) != len(set(change_names)):
                continue
            if diagnosis is not None and self._looks_overfit(proposal, diagnosis):
                continue
            try:
                candidate = apply_patch_proposal(summary.candidate, proposal, search_space)
            except Exception:
                continue
            if candidate == summary.candidate:
                continue
            candidate_hash_value = candidate_hash(candidate)
            if candidate_hash_value in seen_hashes or candidate_hash_value in seen_candidates:
                continue
            if not self._proposal_matches_specs(proposal, summary.candidate, search_space):
                continue
            validated.append(
                PatchProposal(
                    proposal_id=candidate_hash_value,
                    diagnosis_category=proposal.diagnosis_category,
                    changes=proposal.changes,
                    rationale=proposal.rationale,
                    expected_effect=proposal.expected_effect,
                    estimated_scope=proposal.estimated_scope,
                )
            )
            seen_candidates.add(candidate_hash_value)
        validated.sort(key=lambda proposal: (self._scope_rank(proposal.estimated_scope), len(proposal.changes), proposal.proposal_id))
        return validated

    def _proposal_matches_specs(
        self,
        proposal: PatchProposal,
        current_candidate: dict[str, str],
        search_space: SearchSpace,
    ) -> bool:
        for change in proposal.changes:
            enum_spec = search_space.enum_spec(change.name)
            text_spec = search_space.text_spec(change.name)
            component_spec = search_space.component_spec(change.name)
            code_spec = search_space.code_spec(change.name)
            if change.op == "set_enum":
                if enum_spec is None:
                    return False
                if change.value not in enum_spec.values:
                    return False
                if current_candidate.get(change.name) == change.value:
                    return False
            elif change.op == "set_component":
                if component_spec is None:
                    return False
                if change.value not in component_spec.values:
                    return False
                if current_candidate.get(change.name) == change.value:
                    return False
            elif change.op == "rewrite_text":
                if text_spec is None:
                    return False
                if len(change.value) > text_spec.max_chars:
                    return False
                if current_candidate.get(change.name) == change.value:
                    return False
            elif change.op == "rewrite_code":
                if code_spec is None:
                    return False
                if current_candidate.get(change.name) == change.value:
                    return False
                try:
                    validate_code_artifact_source(code_spec, change.value)
                except Exception:
                    return False
        return True

    @staticmethod
    def _scope_rank(scope: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(scope, 3)

    @staticmethod
    def _behavior_priority(proposal: PatchProposal) -> int:
        ops = [change.op for change in proposal.changes]
        if "rewrite_code" in ops and "set_component" in ops:
            return 0
        if "rewrite_code" in ops:
            return 1
        if "set_component" in ops:
            return 2
        if "set_enum" in ops:
            return 3
        return 4

    def _looks_overfit(self, proposal: PatchProposal, diagnosis: FailureDiagnosis) -> bool:
        haystack = " ".join(
            [proposal.rationale, proposal.expected_effect, *(change.value for change in proposal.changes)]
        ).lower()
        for case_id in diagnosis.case_ids:
            if case_id.lower() in haystack:
                return True
        for evidence in diagnosis.evidence:
            snippet = str(evidence.get("input", "")).strip().lower()
            if snippet and snippet in haystack:
                return True
        return False

    def _heuristic_proposals(
        self,
        summary: CandidateSummary,
        search_space: SearchSpace,
        *,
        phase: str,
        diagnosis: FailureDiagnosis | None,
    ) -> list[PatchProposal]:
        if phase == "behavior":
            return self._behavior_proposals(summary, search_space, diagnosis)
        return self._efficiency_proposals(summary, search_space)

    def _behavior_proposals(
        self,
        summary: CandidateSummary,
        search_space: SearchSpace,
        diagnosis: FailureDiagnosis | None,
    ) -> list[PatchProposal]:
        diagnosis = diagnosis or FailureDiagnosis([], "prompt_ambiguity", "", [], [])
        current = summary.candidate
        proposals: list[PatchProposal] = []
        target_names = set(diagnosis.target_keys)

        if diagnosis.category == "missing_tool":
            off_tools = [
                spec
                for spec in search_space.enum_knobs
                if spec.kind == "tool" and current.get(spec.name) == "off"
            ]
            for tool_spec in off_tools[:1]:
                related_text = self._best_tool_text_artifact(search_space, tool_spec.name)
                changes = [PatchChange(op="set_enum", name=tool_spec.name, value="on")]
                if related_text is not None:
                    changes.append(
                        PatchChange(
                            op="rewrite_text",
                            name=related_text.name,
                            value=self._rewrite_text(
                                related_text,
                                current[related_text.name],
                                category="missing_tool",
                            ),
                        )
                    )
                proposals.append(
                    PatchProposal(
                        proposal_id=f"missing-tool::{tool_spec.name}",
                        diagnosis_category=diagnosis.category,
                        changes=changes[:2],
                        rationale="Enable a currently disabled tool and sharpen the instructions that tell the model when to use it.",
                        expected_effect="Increase grounded tool usage on cases that require retrieval or computation.",
                    )
                )

        if diagnosis.category == "arithmetic":
            calc_targets = [
                spec
                for spec in search_space.enum_knobs
                if "calc" in spec.name.lower() or "calculator" in spec.name.lower()
            ]
            for spec in calc_targets[:1]:
                if current.get(spec.name) != "on":
                    proposals.append(
                        PatchProposal(
                            proposal_id=f"arithmetic::{spec.name}",
                            diagnosis_category=diagnosis.category,
                            changes=[PatchChange(op="set_enum", name=spec.name, value="on")],
                            rationale="Enable the calculator tool for arithmetic-sensitive cases.",
                            expected_effect="Avoid mental-math errors on numeric questions.",
                        )
                    )

        if diagnosis.category == "fallback_behavior":
            validator_components = [
                spec
                for spec in search_space.components
                if (spec.kind == "validator" or "validator" in spec.name.lower())
                and current.get(spec.name) != "on"
                and "on" in spec.values
            ]
            for spec in validator_components[:1]:
                changes = [PatchChange(op="set_component", name=spec.name, value="on")]
                related_code_artifacts = [
                    code_spec
                    for code_spec in search_space.code_artifacts
                    if "validator" in code_spec.name.lower() or "ground" in code_spec.name.lower()
                ]
                related_code_artifacts.sort(
                    key=lambda item: (
                        "post_answer_validator_hook" not in item.name.lower(),
                        "validator" not in item.name.lower(),
                        item.name,
                    )
                )
                if related_code_artifacts:
                    code_spec = related_code_artifacts[0]
                    changes.append(
                        PatchChange(
                            op="rewrite_code",
                            name=code_spec.name,
                            value=self._rewrite_code(code_spec, current[code_spec.name], diagnosis.category),
                        )
                    )
                proposals.append(
                    PatchProposal(
                        proposal_id=f"fallback-validator::{spec.name}",
                        diagnosis_category=diagnosis.category,
                        changes=changes[:2],
                        rationale="Enable a grounded-answer validator so unsupported answers are converted to unknown instead of guessed.",
                        expected_effect="Reduce unsupported-answer failures without changing the external contract.",
                        estimated_scope="medium",
                    )
                )

        code_targets = [
            search_space.code_spec(name)
            for name in diagnosis.target_keys
            if search_space.code_spec(name) is not None
        ]
        for code_spec in code_targets[:1]:
            proposals.append(
                PatchProposal(
                    proposal_id=f"code::{diagnosis.category}::{code_spec.name}",
                    diagnosis_category=diagnosis.category,
                    changes=[
                        PatchChange(
                            op="rewrite_code",
                            name=code_spec.name,
                            value=self._rewrite_code(code_spec, current[code_spec.name], diagnosis.category),
                        )
                    ],
                    rationale=f"Strengthen {code_spec.name} to directly address {diagnosis.category}.",
                    expected_effect="Reduce repeat failures in this diagnosis bucket with a bounded hook rewrite.",
                    estimated_scope="medium",
                )
            )

        prompt_targets = [
            search_space.text_spec(name)
            for name in diagnosis.target_keys
            if search_space.text_spec(name) is not None
        ]
        if not prompt_targets:
            prompt_targets = [
                spec for spec in search_space.text_artifacts if spec.kind == "prompt"
            ]

        for text_spec in prompt_targets[:2]:
            proposals.append(
                PatchProposal(
                    proposal_id=f"text::{diagnosis.category}::{text_spec.name}",
                    diagnosis_category=diagnosis.category,
                    changes=[
                        PatchChange(
                            op="rewrite_text",
                            name=text_spec.name,
                            value=self._rewrite_text(text_spec, current[text_spec.name], diagnosis.category),
                        )
                    ],
                    rationale=f"Strengthen {text_spec.name} to directly address {diagnosis.category}.",
                    expected_effect="Reduce repeat failures in this diagnosis bucket.",
                    estimated_scope="low",
                )
            )

        enum_fallback = generate_enum_mutation_proposals(
            current,
            search_space,
            phase="behavior",
            target_names=target_names or None,
        )
        proposals.extend(enum_fallback)
        component_fallback = generate_component_mutation_proposals(
            current,
            search_space,
            phase="behavior",
            target_names=target_names or None,
        )
        proposals.extend(component_fallback)
        return proposals

    def _efficiency_proposals(
        self,
        summary: CandidateSummary,
        search_space: SearchSpace,
    ) -> list[PatchProposal]:
        current = summary.candidate
        proposals: list[PatchProposal] = []
        enum_candidates = generate_enum_mutation_proposals(
            current,
            search_space,
            phase="efficiency",
        )
        enum_candidates.sort(key=lambda proposal: self._efficiency_priority(proposal))
        seen_names: set[str] = set()
        for proposal in enum_candidates:
            change_name = proposal.changes[0].name
            if change_name in seen_names:
                continue
            seen_names.add(change_name)
            proposals.append(proposal)
            if len(proposals) >= 4:
                break
        component_candidates = generate_component_mutation_proposals(
            current,
            search_space,
            phase="efficiency",
        )
        component_candidates.sort(key=lambda proposal: self._efficiency_priority(proposal))
        proposals.extend(component_candidates[:1])

        prompt_artifacts = [spec for spec in search_space.text_artifacts if spec.kind == "prompt"]
        for spec in prompt_artifacts[:1]:
            rewritten = self._rewrite_text(spec, current[spec.name], "grounding")
            if rewritten != current[spec.name]:
                proposals.append(
                    PatchProposal(
                        proposal_id=f"tighten::{spec.name}",
                        diagnosis_category="efficiency",
                        changes=[PatchChange(op="rewrite_text", name=spec.name, value=rewritten)],
                        rationale=f"Tighten {spec.name} so the harness stays grounded with fewer wasted tokens.",
                        expected_effect="Reduce verbosity and improve grounded behavior without changing the external contract.",
                    )
                )

        longest_text = sorted(
            search_space.text_artifacts,
            key=lambda spec: len(current[spec.name]),
            reverse=True,
        )
        for spec in longest_text[:1]:
            compressed = self._compress_text(current[spec.name], spec.max_chars)
            if compressed != current[spec.name]:
                proposals.append(
                    PatchProposal(
                        proposal_id=f"compress::{spec.name}",
                        diagnosis_category="efficiency",
                        changes=[PatchChange(op="rewrite_text", name=spec.name, value=compressed)],
                        rationale=f"Shorten {spec.name} to reduce prompt overhead without changing the external contract.",
                        expected_effect="Reduce prompt tokens and latency while preserving behavior.",
                    )
                )
        return proposals

    @staticmethod
    def _efficiency_priority(proposal: PatchProposal) -> tuple[int, str]:
        change = proposal.changes[0]
        name = change.name.lower()
        if "model" in name:
            bucket = 0
        elif "retrieval" in name or "top_k" in name:
            bucket = 1
        elif "reasoning" in name:
            bucket = 2
        elif "output" in name or "cap" in name:
            bucket = 3
        elif "knowledge" in name:
            bucket = 4
        elif change.op == "rewrite_text":
            bucket = 5
        else:
            bucket = 6
        return bucket, proposal.proposal_id

    @staticmethod
    def _best_tool_text_artifact(
        search_space: SearchSpace,
        tool_name: str,
    ) -> TextArtifactSpec | None:
        for spec in search_space.text_artifacts:
            lowered = spec.name.lower()
            if tool_name.replace("_enabled", "").replace("_tool", "") in lowered:
                return spec
        for spec in search_space.text_artifacts:
            if spec.kind == "tool":
                return spec
        for spec in search_space.text_artifacts:
            lowered = spec.name.lower()
            if "tool" in lowered or "search" in lowered or "lookup" in lowered:
                return spec
        return None

    def _rewrite_text(self, spec: TextArtifactSpec, current_value: str, category: str) -> str:
        lowered_name = spec.name.lower()
        if category == "output_contract":
            if "json" in current_value.lower() or "output" in lowered_name or "answer" in lowered_name:
                return self._fit_text(
                    spec,
                    "Return strict JSON with exactly the required fields and no extra text.",
                )
            return self._fit_text(spec, "Return only the exact externally required output.")
        if category == "missing_tool":
            if spec.kind == "tool" or "description" in lowered_name:
                return self._fit_text(
                    spec,
                    "Search the available knowledge source and return exact grounded facts needed to answer.",
                )
            return self._fit_text(
                spec,
                "Always call the available tool before answering when it can ground the result.",
            )
        if category == "arithmetic":
            return self._fit_text(
                spec,
                "Use the calculator for arithmetic instead of mental math, then return the exact result.",
            )
        if category == "fallback_behavior":
            return self._fit_text(spec, "If the answer is not grounded, return exactly unknown.")
        if category == "retrieval_scope":
            return self._fit_text(
                spec,
                "Use retrieved evidence only and prefer exact grounded literals from the returned results.",
            )
        if "output" in lowered_name and "json" in current_value.lower():
            return self._fit_text(
                spec,
                "Return strict JSON with exactly one answer field and no extra text.",
            )
        if "ground" in lowered_name:
            return self._fit_text(spec, "Use only grounded facts and return the exact answer.")
        if "fallback" in lowered_name:
            return self._fit_text(spec, "If the answer is not grounded, return exactly unknown.")
        if "tool" in lowered_name or "search" in lowered_name:
            return self._fit_text(
                spec,
                "Always use the relevant tool first, then answer with grounded results only.",
            )
        return self._fit_text(spec, "Answer with the exact grounded result only.")

    def _rewrite_code(self, spec: CodeArtifactSpec, current_value: str, category: str) -> str:
        lowered_name = spec.name.lower()
        if "post_answer_validator_hook" in lowered_name or "validator" in lowered_name:
            if category == "fallback_behavior":
                return self._fit_code(
                    spec,
                    """def post_answer_validator_hook(output, context):
    if not isinstance(output, dict):
        return output
    answer = str(output.get("answer", "unknown")).strip().lower()
    if answer == "unknown":
        return {"answer": "unknown"}
    option_literals = [str(item).strip() for item in context.get("option_literals", [])]
    retrieved_cards = context.get("retrieved_cards", [])
    haystack = "\\n".join(
        f"{card.get('doc_id', '')} {card.get('title', '')} {card.get('text', '')}".lower()
        for card in retrieved_cards
        if isinstance(card, dict)
    )
    grounded = {item.lower() for item in option_literals if item and item.lower() in haystack}
    if grounded and answer in grounded:
        return {"answer": output.get("answer", "unknown")}
    return {"answer": "unknown"}
""",
                )
            return self._fit_code(
                spec,
                """def post_answer_validator_hook(output, context):
    return output
""",
            )
        if "pre_tool_query_hook" in lowered_name:
            return self._fit_code(
                spec,
                """def pre_tool_query_hook(query, context):
    case_input = str(context.get("case_input", "")).strip()
    if not case_input:
        return query
    return f"{query}\\nQuestion: {case_input}"
""",
            )
        if "post_tool_context_hook" in lowered_name:
            return self._fit_code(
                spec,
                """def post_tool_context_hook(cards, context):
    if not isinstance(cards, list):
        return cards
    return cards[:2]
""",
            )
        return current_value

    @staticmethod
    def _compress_text(text: str, max_chars: int) -> str:
        compressed = re.sub(r"\s+", " ", text).strip()
        compressed = compressed.replace("the available ", "").replace("frozen ", "")
        return compressed[:max_chars].rstrip()

    @staticmethod
    def _fit_text(spec: TextArtifactSpec, text: str) -> str:
        if len(text) <= spec.max_chars:
            return text
        return text[: spec.max_chars].rstrip()

    @staticmethod
    def _fit_code(spec: CodeArtifactSpec, text: str) -> str:
        lines = text.rstrip().splitlines()
        if spec.max_lines and len(lines) > spec.max_lines:
            lines = lines[: spec.max_lines]
        fitted = "\n".join(lines).rstrip() + "\n"
        if len(fitted) <= spec.max_chars:
            return fitted
        return fitted[: spec.max_chars].rstrip() + "\n"


class RatchetOptimizer:
    def __init__(
        self,
        adapter: AdapterProtocol,
        search_space: SearchSpace,
        out_dir: Path,
        env_path: str = ".env",
        dev_budget: int = 20,
        holdout_top_k: int = 5,
        harnesser_model: str = "gpt-5.4",
        harnesser_reasoning: str = "medium",
        harnesser_enabled: bool = True,
        max_case_retries: int = 2,
        case_timeout_s: int = 180,
        fail_fast: bool = False,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.search_space = search_space
        self.out_dir = out_dir
        self.env_path = env_path
        self.dev_budget = dev_budget
        self.holdout_top_k = holdout_top_k
        self.diagnoser = FailureDiagnoser(
            env_path=env_path,
            model=harnesser_model,
            reasoning_effort=harnesser_reasoning,
            enabled=harnesser_enabled,
        )
        self.proposer = ProposalEngine(
            env_path=env_path,
            model=harnesser_model,
            reasoning_effort=harnesser_reasoning,
            enabled=harnesser_enabled,
        )
        self.store = ResultStore(out_dir)
        self.max_case_retries = max_case_retries
        self.case_timeout_s = case_timeout_s
        self.fail_fast = fail_fast
        self.run_metadata = dict(run_metadata or {})
        self.stats = OptimizerStats()
        self.started_at: datetime | None = None

    def run(self, cases: tuple[EvalCase, ...]) -> RatchetResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc)
        dev_cases, holdout_cases = split_cases(cases)

        baseline_candidate = build_candidate(self.adapter.baseline(), self.search_space)
        baseline_hash = candidate_hash(baseline_candidate)
        baseline_dev = self.evaluate_candidate(baseline_candidate, dev_cases)
        baseline_holdout = self.evaluate_candidate(baseline_candidate, holdout_cases)

        current_dev = baseline_dev
        accepted_dev_candidates: list[CandidateSummary] = [baseline_dev]
        holdout_candidates: list[CandidateSummary] = []
        promotable_frontier: list[dict[str, Any]] = []
        decision_log: list[dict[str, Any]] = []
        diagnoses_log: list[dict[str, Any]] = []
        proposals_log: list[dict[str, Any]] = []
        evaluated_candidate_hashes = {baseline_hash}
        dev_evaluations = 0
        phase = "behavior"
        iteration = 0

        while dev_evaluations < self.dev_budget:
            iteration += 1
            if phase == "behavior" and current_dev.pass_count == len(dev_cases):
                phase = "efficiency"

            diagnoses, diagnosis_analysis = self.diagnoser.diagnose(current_dev, self.search_space)
            for diagnosis in diagnoses:
                diagnoses_log.append(
                    {
                        "iteration": iteration,
                        "phase": phase,
                        "candidate_hash": current_dev.candidate_hash,
                        **diagnosis.to_dict(),
                    }
                )

            target_diagnosis = diagnoses[0] if phase == "behavior" and diagnoses else None
            if phase == "behavior" and target_diagnosis is None:
                phase = "efficiency"
                continue

            proposals, proposal_analysis = self.proposer.propose(
                current_dev,
                self.search_space,
                phase=phase,
                diagnosis=target_diagnosis,
                seen_hashes=evaluated_candidate_hashes,
                history=proposals_log,
            )
            decision_log.append(
                {
                    "type": "proposal_iteration",
                    "iteration": iteration,
                    "phase": phase,
                    "candidate_hash": current_dev.candidate_hash,
                    "diagnosis_analysis": diagnosis_analysis,
                    "proposal_analysis": proposal_analysis,
                    "diagnosis": target_diagnosis.to_dict() if target_diagnosis else None,
                    "proposal_ids": [proposal.proposal_id for proposal in proposals],
                }
            )
            if not proposals:
                if phase == "behavior":
                    phase = "efficiency"
                    continue
                break

            accepted_rows: list[tuple[PatchProposal, CandidateSummary, Comparison | None]] = []
            for proposal in proposals[: min(MAX_PROPOSALS_PER_ITERATION, self.dev_budget - dev_evaluations)]:
                candidate = apply_patch_proposal(current_dev.candidate, proposal, self.search_space)
                candidate_hash_value = candidate_hash(candidate)
                if candidate_hash_value in evaluated_candidate_hashes:
                    continue
                summary = self.evaluate_candidate(candidate, dev_cases)
                dev_evaluations += 1
                evaluated_candidate_hashes.add(candidate_hash_value)
                comparison = compare_summaries(current_dev, summary)
                flip_summary = behavior_flip_summary(current_dev, summary)
                accepted = False
                rejection_reason = ""
                if phase == "behavior":
                    within_guard = (
                        summary.mean_cost_usd <= baseline_dev.mean_cost_usd * BEHAVIOR_PHASE_GUARD_MULTIPLIER
                        and summary.mean_total_tokens
                        <= baseline_dev.mean_total_tokens * BEHAVIOR_PHASE_GUARD_MULTIPLIER
                        and summary.median_latency_s
                        <= baseline_dev.median_latency_s * BEHAVIOR_PHASE_GUARD_MULTIPLIER
                    )
                    accepted = summary.pass_count > current_dev.pass_count and within_guard
                    if not accepted:
                        rejection_reason = (
                            "dev pass count did not increase"
                            if summary.pass_count <= current_dev.pass_count
                            else "behavior-phase operational guard exceeded"
                        )
                else:
                    accepted = (
                        summary.pass_count == current_dev.pass_count
                        and comparison.score_ci[0] >= -NON_INFERIORITY_MARGIN
                        and comparison.cost_ci[1] < 0
                        and comparison.token_ci[1] < 0
                        and summary.median_latency_s
                        <= current_dev.median_latency_s * LATENCY_GUARD_MULTIPLIER
                    )
                    if not accepted:
                        rejection_reason = "behavior regressed or efficiency did not improve under the dev gate"

                proposal_row = {
                    "iteration": iteration,
                    "phase": phase,
                    "parent_hash": current_dev.candidate_hash,
                    "proposal_id": proposal.proposal_id,
                    "diagnosis_category": proposal.diagnosis_category,
                    "proposal": proposal.to_dict(),
                    "candidate_hash": candidate_hash_value,
                    "candidate": candidate,
                    "comparison_to_parent": comparison.to_dict(),
                    "behavior_flip_summary": flip_summary,
                    "metrics": summary.to_dict(),
                    "accepted": accepted,
                    "rejection_reason": rejection_reason or None,
                }
                proposals_log.append(proposal_row)
                decision_log.append({"type": "proposal_evaluation", **proposal_row})
                if accepted:
                    accepted_rows.append((proposal, summary, comparison))

            if accepted_rows:
                if phase == "behavior":
                    accepted_rows.sort(
                        key=lambda item: (
                            -item[1].pass_count,
                            -item[1].mean_score,
                            item[1].mean_cost_usd,
                            item[1].mean_total_tokens,
                            item[1].median_latency_s,
                        )
                    )
                else:
                    accepted_rows.sort(
                        key=lambda item: (
                            item[1].mean_cost_usd,
                            item[1].mean_total_tokens,
                            item[1].median_latency_s,
                            -item[1].mean_score,
                        )
                    )

                chosen_proposal, chosen_dev, _ = accepted_rows[0]
                current_dev = chosen_dev
                accepted_dev_candidates.append(chosen_dev)
                holdout_summary = self.evaluate_candidate(chosen_dev.candidate, holdout_cases)
                holdout_candidates.append(holdout_summary)
                passed_final_gate, holdout_comparison = final_gate(baseline_holdout, holdout_summary)
                holdout_flips = behavior_flip_summary(baseline_holdout, holdout_summary)
                decision_log.append(
                    {
                        "type": "accepted_proposal",
                        "iteration": iteration,
                        "phase": phase,
                        "proposal_id": chosen_proposal.proposal_id,
                        "candidate_hash": chosen_dev.candidate_hash,
                        "metrics": chosen_dev.to_dict(),
                    }
                )
                decision_log.append(
                    {
                        "type": "holdout_validation",
                        "iteration": iteration,
                        "candidate_hash": holdout_summary.candidate_hash,
                        "metrics": holdout_summary.to_dict(),
                        "comparison_to_baseline": holdout_comparison.to_dict(),
                        "behavior_flip_summary": holdout_flips,
                        "passed_final_gate": passed_final_gate,
                    }
                )
                if passed_final_gate:
                    promotable_frontier.append(
                        {
                            "candidate_hash": holdout_summary.candidate_hash,
                            "candidate": holdout_summary.candidate,
                            "metrics": holdout_summary.to_dict(),
                            "comparison_to_baseline": holdout_comparison.to_dict(),
                            "behavior_flip_summary": holdout_flips,
                        }
                    )
                continue

            if phase == "behavior":
                phase = "efficiency"
            else:
                break

        best_dev_candidate = max(
            accepted_dev_candidates,
            key=lambda summary: (
                summary.pass_count,
                summary.mean_score,
                -summary.mean_cost_usd,
                -summary.mean_total_tokens,
                -summary.median_latency_s,
            ),
        )

        if promotable_frontier:
            promotable_frontier.sort(
                key=lambda row: (
                    row["metrics"]["mean_cost_usd"],
                    row["metrics"]["mean_total_tokens"],
                    row["metrics"]["median_latency_s"],
                )
            )
            selected_frontier = promotable_frontier[0]
            selected_holdout = next(
                summary
                for summary in holdout_candidates
                if summary.candidate_hash == selected_frontier["candidate_hash"]
            )
            promoted = True
            selection_reason = (
                "Promoted the cheapest current-holdout candidate that preserved quality and improved efficiency."
            )
        else:
            selected_holdout = baseline_holdout
            promoted = False
            selection_reason = (
                "No accepted candidate cleared the holdout quality, efficiency, and latency gates; kept baseline."
            )

        selected_candidate = dict(selected_holdout.candidate)
        selected_candidate_hash = selected_holdout.candidate_hash
        decision_log.append(
            {
                "type": "final_selection",
                "selected_candidate_hash": selected_candidate_hash,
                "promoted": promoted,
                "reason": selection_reason,
                "best_dev_candidate_hash": best_dev_candidate.candidate_hash,
            }
        )

        manifest = self.build_manifest(
            total_cases=len(cases),
            selected_candidate_hash=selected_candidate_hash,
            promoted=promoted,
        )
        result = RatchetResult(
            baseline_candidate=baseline_candidate,
            selected_candidate=selected_candidate,
            selected_candidate_hash=selected_candidate_hash,
            promoted=promoted,
            baseline_dev=baseline_dev,
            baseline_holdout=baseline_holdout,
            best_dev_candidate=best_dev_candidate,
            selected_holdout=selected_holdout,
            accepted_dev_candidates=accepted_dev_candidates[1:],
            holdout_candidates=holdout_candidates,
            promotable_frontier=promotable_frontier[: self.holdout_top_k],
            decision_log=decision_log,
            diagnoses=diagnoses_log,
            proposals=proposals_log,
            selection_reason=selection_reason,
            manifest=manifest,
        )
        self.write_outputs(result)
        return result

    def evaluate_candidate(
        self,
        candidate: dict[str, str],
        cases: tuple[EvalCase, ...],
    ) -> CandidateSummary:
        normalized_candidate = build_candidate(candidate, self.search_space)
        candidate_hash_value = candidate_hash(normalized_candidate)
        evaluations: list[CaseEvaluation] = []
        for case in cases:
            cached = self.store.get(candidate_hash_value, case.id)
            if cached is not None:
                self.stats.cache_hits += 1
                evaluations.append(cached)
                continue
            evaluation = self._execute_case(normalized_candidate, case)
            self.store.put(candidate_hash_value, normalized_candidate, evaluation)
            if self.fail_fast and evaluation.record.metrics.error:
                raise RuntimeError(
                    f"Fail-fast stopping after case {case.id}: {evaluation.record.metrics.error}"
                )
            evaluations.append(evaluation)
        return CandidateSummary(
            candidate_hash=candidate_hash_value,
            candidate=normalized_candidate,
            split=cases[0].split,
            evaluations=evaluations,
        )

    def _execute_case(self, candidate: dict[str, str], case: EvalCase) -> CaseEvaluation:
        total_attempts = self.max_case_retries + 1
        started_at = time.perf_counter()
        last_error: Exception | None = None
        last_phase = "run_case"
        for attempt in range(1, total_attempts + 1):
            try:
                last_phase = "run_case"
                with case_timeout(self.case_timeout_s):
                    record = self.adapter.run_case(candidate, case)
                if not isinstance(record, RunRecord):
                    raise TypeError(f"run_case returned {type(record).__name__}, expected RunRecord.")
                try:
                    json.dumps(record.output, sort_keys=True)
                except TypeError as error:
                    raise TypeError("run_case returned a non-JSON-serializable output.") from error
                last_phase = "grade"
                with case_timeout(self.case_timeout_s):
                    grade = self.adapter.grade(case, record.output)
                if not isinstance(grade, GradeResult):
                    raise TypeError(f"grade returned {type(grade).__name__}, expected GradeResult.")
                diagnostic_metadata = dict(record.diagnostics.metadata)
                diagnostic_metadata.setdefault("attempts", attempt)
                if diagnostic_metadata != record.diagnostics.metadata:
                    record = RunRecord(
                        output=record.output,
                        metrics=record.metrics,
                        diagnostics=DiagnosticTrace(
                            tool_calls=list(record.diagnostics.tool_calls),
                            raw_output_text=record.diagnostics.raw_output_text,
                            metadata=diagnostic_metadata,
                        ),
                    )
                self.stats.fresh_case_evaluations += 1
                return CaseEvaluation(case=case, record=record, grade=grade)
            except Exception as error:
                last_error = error
                if attempt < total_attempts:
                    self.stats.retries += 1
                    continue

        assert last_error is not None
        elapsed = time.perf_counter() - started_at
        error_type = type(last_error).__name__
        labels = ["runtime_error"]
        notes_prefix = "Runtime error during agent execution."
        if isinstance(last_error, TimeoutError):
            labels.append("timeout")
            self.stats.timeouts += 1
            notes_prefix = "Timed out while evaluating the case."
        if last_phase == "grade":
            labels.append("grader_error")
            self.stats.grader_errors += 1
            notes_prefix = "Adapter grader raised an error."
        self.stats.runtime_errors += 1
        self.stats.fresh_case_evaluations += 1
        record = RunRecord(
            output=None,
            metrics=OperationalMetrics(
                latency_s=elapsed,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=f"{error_type}: {last_error}",
            ),
            diagnostics=DiagnosticTrace(
                tool_calls=[],
                raw_output_text="",
                metadata={
                    "attempts": total_attempts,
                    "error_type": error_type,
                    "error_phase": last_phase,
                },
            ),
        )
        grade = GradeResult(
            score=0.0,
            passed=False,
            labels=labels,
            notes=f"{notes_prefix} {error_type}: {last_error}",
        )
        return CaseEvaluation(case=case, record=record, grade=grade)

    def build_manifest(
        self,
        *,
        total_cases: int,
        selected_candidate_hash: str,
        promoted: bool,
    ) -> dict[str, Any]:
        finished_at = datetime.now(timezone.utc)
        duration_s = None
        if self.started_at is not None:
            duration_s = round((finished_at - self.started_at).total_seconds(), 3)
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": finished_at.isoformat(),
            "duration_s": duration_s,
            "total_cases": total_cases,
            "selected_candidate_hash": selected_candidate_hash,
            "promoted": promoted,
            "stats": self.stats.to_dict(),
            "config": dict(self.run_metadata),
        }

    def write_outputs(self, result: RatchetResult) -> None:
        candidate_metrics = {
            "baseline_dev": result.baseline_dev.to_dict(),
            "baseline_holdout": result.baseline_holdout.to_dict(),
            "best_dev_candidate": result.best_dev_candidate.to_dict(),
            "accepted_dev_candidates": [summary.to_dict() for summary in result.accepted_dev_candidates],
            "holdout_candidates": [summary.to_dict() for summary in result.holdout_candidates],
            "promotable_frontier": result.promotable_frontier,
            "selected_candidate_hash": result.selected_candidate_hash,
            "promoted": result.promoted,
        }
        write_json(self.out_dir / "candidate_metrics.json", candidate_metrics)
        write_json(self.out_dir / "decision_log.json", result.decision_log)
        write_json(self.out_dir / "run_manifest.json", result.manifest)
        write_jsonl(self.out_dir / "diagnoses.jsonl", result.diagnoses)
        write_jsonl(self.out_dir / "proposals.jsonl", result.proposals)
        write_json(
            self.out_dir / "optimized_candidate.json",
            {
                "candidate": result.selected_candidate,
                "candidate_hash": result.selected_candidate_hash,
                "promoted": result.promoted,
                "baseline_candidate": result.baseline_candidate,
                "best_dev_candidate": result.best_dev_candidate.candidate,
                "best_dev_candidate_hash": result.best_dev_candidate.candidate_hash,
            },
        )
        export_dir = self.out_dir / "exported_candidate"
        export_dir.mkdir(parents=True, exist_ok=True)
        self.adapter.export(result.selected_candidate, export_dir)
        (self.out_dir / "report.md").write_text(self.render_report(result))

    def render_report(self, result: RatchetResult) -> str:
        comparison = compare_summaries(result.baseline_holdout, result.selected_holdout)
        holdout_flips = behavior_flip_summary(result.baseline_holdout, result.selected_holdout)
        best_dev_flips = behavior_flip_summary(result.baseline_dev, result.best_dev_candidate)
        rejected_rows = self._rejected_candidates(result)
        baseline_latency = result.baseline_holdout.median_latency_s
        latency_ratio = (
            result.selected_holdout.median_latency_s / baseline_latency
            if baseline_latency > 0
            else 0.0
        )
        dev_history_rows = [
            f"- `{summary.candidate_hash}` pass_count={summary.pass_count} score={summary.mean_score:.3f} cost=${summary.mean_cost_usd:.6f} tokens={summary.mean_total_tokens:.1f}"
            for summary in [result.baseline_dev, *result.accepted_dev_candidates]
        ]
        diagnosis_counts: Counter[str] = Counter(row["category"] for row in result.diagnoses)
        accepted_proposals = [row for row in result.proposals if row["accepted"]]
        rejected_proposals = [row for row in result.proposals if not row["accepted"]]
        lines = [
            "# Ratchet Run Report",
            "",
            f"Outcome: {'promoted optimized candidate' if result.promoted else 'kept baseline'}",
            f"Decision: {result.selection_reason}",
            "",
                "## Behavioral Summary",
            "",
            "| Candidate | Mean score | Pass rate | Pass count |",
            "| --- | --- | --- | --- |",
            f"| baseline | {result.baseline_holdout.mean_score:.3f} | {result.baseline_holdout.pass_rate:.3f} | {result.baseline_holdout.pass_count} |",
                f"| selected | {result.selected_holdout.mean_score:.3f} | {result.selected_holdout.pass_rate:.3f} | {result.selected_holdout.pass_count} |",
                "",
                "## Behavior Flips",
                "",
                f"- Best dev incumbent fixed {best_dev_flips['fixed_count']} cases and regressed {best_dev_flips['regressed_count']} vs baseline dev.",
                f"- Selected holdout candidate fixed {holdout_flips['fixed_count']} cases and regressed {holdout_flips['regressed_count']} vs baseline holdout.",
                "",
                "## Operational Summary",
            "",
            "| Candidate | Avg cost | Avg tokens | Median latency |",
            "| --- | --- | --- | --- |",
            f"| baseline | ${result.baseline_holdout.mean_cost_usd:.6f} | {result.baseline_holdout.mean_total_tokens:.1f} | {result.baseline_holdout.median_latency_s:.2f}s |",
            f"| selected | ${result.selected_holdout.mean_cost_usd:.6f} | {result.selected_holdout.mean_total_tokens:.1f} | {result.selected_holdout.median_latency_s:.2f}s |",
            "",
            "## Dev Behavior History",
            "",
            *dev_history_rows,
            "",
            "## Diagnosis Breakdown",
            "",
        ]
        if diagnosis_counts:
            for category, count in sorted(diagnosis_counts.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- {category}: {count}")
        else:
            lines.append("- No diagnoses recorded")
        lines.extend(
            [
                "",
                "## Run Health",
                "",
                f"- Cache hits: {result.manifest['stats']['cache_hits']}",
                f"- Fresh case evaluations: {result.manifest['stats']['fresh_case_evaluations']}",
                f"- Retries: {result.manifest['stats']['retries']}",
                f"- Runtime errors: {result.manifest['stats']['runtime_errors']}",
                f"- Timeouts: {result.manifest['stats']['timeouts']}",
                f"- Grader errors: {result.manifest['stats']['grader_errors']}",
                f"- Eval SHA-256: {result.manifest['config'].get('evals_sha256', 'unknown')}",
                "",
                "## Proposal Summary",
                "",
                f"- Accepted proposals: {len(accepted_proposals)}",
                f"- Rejected proposals: {len(rejected_proposals)}",
                "",
            ]
        )
        if accepted_proposals:
            lines.append("### Accepted Proposals")
            lines.append("")
            for row in accepted_proposals[:5]:
                flip_summary = row.get("behavior_flip_summary", {})
                lines.append(
                    f"- `{row['proposal_id']}` phase={row['phase']} fixed={flip_summary.get('fixed_count', 0)} "
                    f"regressed={flip_summary.get('regressed_count', 0)} "
                    f"candidate={json.dumps(row['candidate'], sort_keys=True)}"
                )
            lines.append("")
        if rejected_rows:
            lines.extend(["## Top Rejected Proposals", "", *rejected_rows, ""])
        lines.extend(
            [
                "## Current Holdout Gate Check",
                "",
                f"- Score delta mean: {comparison.score_delta:+.3f} (95% CI {comparison.score_ci[0]:+.3f} to {comparison.score_ci[1]:+.3f})",
                f"- Cost delta mean: {comparison.cost_delta:+.6f} (95% CI {comparison.cost_ci[0]:+.6f} to {comparison.cost_ci[1]:+.6f})",
                f"- Token delta mean: {comparison.token_delta:+.1f} (95% CI {comparison.token_ci[0]:+.1f} to {comparison.token_ci[1]:+.1f})",
                f"- Latency ratio: {latency_ratio:.3f}x",
                "",
                "## Current Holdout Category Breakdown",
                "",
                "| Category | Baseline pass | Selected pass | Baseline score | Selected score |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for category in sorted(
            set(result.baseline_holdout.category_metrics) | set(result.selected_holdout.category_metrics)
        ):
            baseline_metrics = result.baseline_holdout.category_metrics.get(category, {})
            selected_metrics = result.selected_holdout.category_metrics.get(category, {})
            lines.append(
                f"| {category} | {float(baseline_metrics.get('pass_rate', 0.0)):.3f} | "
                f"{float(selected_metrics.get('pass_rate', 0.0)):.3f} | "
                f"{float(baseline_metrics.get('mean_score', 0.0)):.3f} | "
                f"{float(selected_metrics.get('mean_score', 0.0)):.3f} |"
            )
        lines.extend(["", "## Holdout Flip Breakdown", ""])
        if holdout_flips["fixed_count"] or holdout_flips["regressed_count"]:
            lines.append(
                f"- Fixed cases: {', '.join(holdout_flips['fixed_case_ids']) if holdout_flips['fixed_case_ids'] else 'none'}"
            )
            lines.append(
                f"- Regressed cases: {', '.join(holdout_flips['regressed_case_ids']) if holdout_flips['regressed_case_ids'] else 'none'}"
            )
            for category, counts in holdout_flips["by_category"].items():
                lines.append(
                    f"- {category}: fixed={counts.get('fixed', 0)} regressed={counts.get('regressed', 0)}"
                )
        else:
            lines.append("- No holdout case flips")
        lines.extend(["", "## Promotable Frontier", ""])
        if result.promotable_frontier:
            for row in result.promotable_frontier:
                lines.append(
                    f"- `{row['candidate_hash']}` cost=${row['metrics']['mean_cost_usd']:.6f} "
                    f"tokens={row['metrics']['mean_total_tokens']:.1f} "
                    f"candidate={json.dumps(row['candidate'], sort_keys=True)}"
                )
        else:
            lines.append("- None")
        lines.extend(
            [
                "",
                "## Selected Candidate",
                "",
                "```json",
                json.dumps(result.selected_candidate, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
        if result.best_dev_candidate.candidate_hash != result.selected_candidate_hash:
            lines.extend(
                [
                    "## Best Dev Incumbent",
                    "",
                    "```json",
                    json.dumps(result.best_dev_candidate.candidate, indent=2, sort_keys=True),
                    "```",
                    "",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _rejected_candidates(result: RatchetResult) -> list[str]:
        rejected_events = [row for row in result.proposals if not row["accepted"]]
        rejected_events.sort(
            key=lambda row: (
                -row["metrics"]["pass_count"],
                -row["metrics"]["mean_score"],
                row["metrics"]["mean_cost_usd"],
                row["metrics"]["mean_total_tokens"],
            )
        )
        rows: list[str] = []
        for row in rejected_events[:5]:
            rows.append(
                "- "
                f"`{row['proposal_id']}` "
                f"phase={row['phase']} "
                f"pass_count={row['metrics']['pass_count']} "
                f"score={row['metrics']['mean_score']:.3f} "
                f"reason={row['rejection_reason']}"
            )
        return rows
