from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from ratchet.results import CandidateSummary, _compact_prompt_value
from ratchet.types import EvalCase


LABEL_FIELD_CANDIDATES = ("label", "intent", "class", "category")


@dataclass(frozen=True)
class ProposalExample:
    case_id: str
    input: Any
    expected: Any
    metadata: dict[str, Any]
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalExampleBank:
    examples: list[ProposalExample]
    label_counts: dict[str, int]
    metadata_categories: dict[str, int]
    label_field: str | None = None

    @property
    def case_ids(self) -> set[str]:
        return {example.case_id for example in self.examples}

    def to_dict(self) -> dict[str, Any]:
        return {
            "usage": (
                "proposal-safe train examples. Candidate patches may copy these inputs/expected outputs "
                "only when they reference source_case_id."
            ),
            "label_field": self.label_field,
            "example_count": len(self.examples),
            "label_counts": dict(self.label_counts),
            "metadata_categories": dict(self.metadata_categories),
            "examples": [example.to_dict() for example in self.examples],
        }

    def to_prompt_dict(
        self,
        *,
        target_labels: set[str] | None = None,
        max_examples: int = 24,
    ) -> dict[str, Any]:
        labels = target_labels or set()
        selected = sorted(
            self.examples,
            key=lambda example: (
                0 if example.label in labels else 1,
                example.label or "",
                example.case_id,
            ),
        )[:max_examples]
        return {
            "usage": (
                "proposal-safe train examples. Candidate few-shot patches may copy these inputs/expected outputs "
                "only when each item references source_case_id."
            ),
            "label_field": self.label_field,
            "example_count": len(self.examples),
            "included_example_count": len(selected),
            "label_counts": dict(self.label_counts),
            "metadata_categories": dict(self.metadata_categories),
            "examples": [example.to_dict() for example in selected],
        }


def build_proposal_example_bank(
    cases: tuple[EvalCase, ...],
    *,
    limit: int = 80,
    max_text_chars: int = 600,
) -> ProposalExampleBank:
    label_field = infer_label_field_from_cases(cases)
    selected = _balanced_examples(cases, label_field=label_field, limit=limit)
    examples = [
        ProposalExample(
            case_id=case.id,
            input=_compact_prompt_value(case.input, max_text_chars=max_text_chars),
            expected=_compact_prompt_value(case.expected, max_text_chars=max_text_chars),
            metadata=_compact_prompt_value(case.metadata, max_text_chars=max_text_chars),
            label=_label_from_case(case, label_field=label_field),
        )
        for case in selected
    ]
    label_counts = Counter(example.label for example in examples if example.label)
    category_counts = Counter(
        str(example.metadata.get("category"))
        for example in examples
        if isinstance(example.metadata, dict) and example.metadata.get("category") is not None
    )
    return ProposalExampleBank(
        examples=examples,
        label_counts=dict(sorted(label_counts.items())),
        metadata_categories=dict(sorted(category_counts.items())),
        label_field=label_field,
    )


def build_behavior_diagnostics(summary: CandidateSummary, *, max_case_ids: int = 8) -> dict[str, Any]:
    label_field = infer_label_field_from_cases(tuple(evaluation.case for evaluation in summary.evaluations))
    per_label: dict[str, dict[str, Any]] = {}
    confusion_counts: Counter[tuple[str, str]] = Counter()
    confusion_case_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    actual_counts: Counter[str] = Counter()
    invalid_case_ids: list[str] = []
    length_finish_case_ids: list[str] = []
    parser_fallback_case_ids: list[str] = []
    low_output_token_length_case_ids: list[str] = []
    finish_reason_counts: Counter[str] = Counter()
    tool_call_counts: Counter[str] = Counter()
    tool_status_counts: Counter[str] = Counter()
    turn_outcome_counts: Counter[str] = Counter()
    terminal_reason_counts: Counter[str] = Counter()
    tool_error_case_ids: list[str] = []
    invalid_tool_call_case_ids: list[str] = []
    no_tool_call_case_ids: list[str] = []
    premature_stop_case_ids: list[str] = []

    for case_id, evaluations, mean_score, _, case_passed in summary._case_rows():
        evaluation = next((item for item in evaluations if not item.grade.passed), evaluations[0])
        metadata = evaluation.record.diagnostics.metadata
        diagnostics = evaluation.record.diagnostics
        finish_reason = str(metadata.get("finish_reason") or "")
        if finish_reason:
            finish_reason_counts[finish_reason] += 1
        terminal_reason = diagnostics.terminal_reason or str(metadata.get("terminal_reason") or "")
        if terminal_reason:
            terminal_reason_counts[terminal_reason] += 1
            if terminal_reason in {"premature_stop", "max_turns", "stopped_before_required_action"}:
                premature_stop_case_ids.append(case_id)
        case_tool_call_count = 0
        case_has_tool_error = False
        case_has_invalid_tool_call = False
        for turn in diagnostics.turns:
            if turn.outcome:
                turn_outcome_counts[turn.outcome] += 1
            for tool_call in turn.tool_calls:
                case_tool_call_count += 1
                tool_call_counts[tool_call.name] += 1
                tool_status_counts[tool_call.status] += 1
                case_has_tool_error = case_has_tool_error or bool(tool_call.error) or tool_call.status == "error"
                case_has_invalid_tool_call = case_has_invalid_tool_call or tool_call.status == "invalid"
        if not diagnostics.turns:
            case_tool_call_count = len(diagnostics.tool_calls)
            for tool_name in diagnostics.tool_calls:
                tool_call_counts[tool_name] += 1
        if case_tool_call_count == 0:
            no_tool_call_case_ids.append(case_id)
        if case_has_tool_error:
            tool_error_case_ids.append(case_id)
        if case_has_invalid_tool_call:
            invalid_tool_call_case_ids.append(case_id)
        if finish_reason == "length":
            length_finish_case_ids.append(case_id)
            cap = _safe_int(metadata.get("requested_output_cap"))
            if cap and evaluation.record.metrics.output_tokens <= max(1, int(cap * 0.25)):
                low_output_token_length_case_ids.append(case_id)
        if metadata.get("parser_fallback"):
            parser_fallback_case_ids.append(case_id)
        expected_label = _label_from_case(evaluation.case, label_field=label_field)
        actual_label = _actual_label(evaluation.record.output, evaluation.grade.labels, label_field=label_field)
        if expected_label:
            row = per_label.setdefault(
                expected_label,
                {"support": 0, "pass_count": 0, "score_sum": 0.0, "case_ids": []},
            )
            row["support"] += 1
            row["pass_count"] += int(case_passed)
            row["score_sum"] += mean_score
            if len(row["case_ids"]) < max_case_ids:
                row["case_ids"].append(case_id)
        if actual_label:
            actual_counts[actual_label] += 1
        if not case_passed and expected_label and actual_label:
            key = (expected_label, actual_label)
            confusion_counts[key] += 1
            if len(confusion_case_ids[key]) < max_case_ids:
                confusion_case_ids[key].append(case_id)
        if not case_passed and any("invalid_output" in label for label in evaluation.grade.labels):
            invalid_case_ids.append(case_id)

    label_metrics = []
    for label, row in per_label.items():
        support = int(row["support"])
        pass_count = int(row["pass_count"])
        mean_score = float(row["score_sum"]) / max(support, 1)
        label_metrics.append(
            {
                "label": label,
                "support": support,
                "pass_count": pass_count,
                "pass_rate": round(pass_count / max(support, 1), 4),
                "mean_score": round(mean_score, 4),
                "case_ids": list(row["case_ids"]),
            }
        )
    label_metrics.sort(key=lambda item: (float(item["pass_rate"]), -int(item["support"]), str(item["label"])))
    global_pass_rate = summary.pass_rate
    weak_labels = [
        str(row["label"])
        for row in label_metrics
        if int(row["support"]) > int(row["pass_count"]) and float(row["pass_rate"]) <= global_pass_rate
    ]
    confusions = [
        {
            "expected": expected,
            "actual": actual,
            "count": count,
            "case_ids": confusion_case_ids[(expected, actual)],
        }
        for (expected, actual), count in confusion_counts.most_common(12)
        if expected != actual
    ]
    overpredicted = [
        {"label": label, "count": count}
        for label, count in actual_counts.most_common(12)
        if count > 0
    ]
    return {
        "label_field": label_field,
        "per_label": label_metrics[:30],
        "weak_labels": weak_labels[:20],
        "confusions": confusions,
        "overpredicted_labels": overpredicted,
        "invalid_output_case_ids": invalid_case_ids[:max_case_ids],
        "runtime_reliability": {
            "finish_reason_counts": dict(sorted(finish_reason_counts.items())),
            "length_finish_case_ids": length_finish_case_ids[:max_case_ids],
            "parser_fallback_case_ids": parser_fallback_case_ids[:max_case_ids],
            "low_output_token_length_case_ids": low_output_token_length_case_ids[:max_case_ids],
        },
        "tool_interaction": {
            "tool_call_counts": dict(sorted(tool_call_counts.items())),
            "tool_status_counts": dict(sorted(tool_status_counts.items())),
            "turn_outcome_counts": dict(sorted(turn_outcome_counts.items())),
            "terminal_reason_counts": dict(sorted(terminal_reason_counts.items())),
            "tool_error_case_ids": tool_error_case_ids[:max_case_ids],
            "invalid_tool_call_case_ids": invalid_tool_call_case_ids[:max_case_ids],
            "no_tool_call_case_ids": no_tool_call_case_ids[:max_case_ids],
            "premature_stop_case_ids": premature_stop_case_ids[:max_case_ids],
            "mean_turns": round(summary.mean_turns, 4),
            "mean_tool_calls": round(summary.mean_tool_calls, 4),
            "mean_model_calls": round(summary.mean_model_calls, 4),
        },
        "category_metrics": summary.category_metrics,
    }


def infer_label_field_from_cases(cases: tuple[EvalCase, ...]) -> str | None:
    counts: Counter[str] = Counter()
    for case in cases:
        if isinstance(case.expected, dict):
            for key in LABEL_FIELD_CANDIDATES:
                if key in case.expected and isinstance(case.expected[key], str):
                    counts[key] += 1
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _balanced_examples(cases: tuple[EvalCase, ...], *, label_field: str | None, limit: int) -> list[EvalCase]:
    if limit <= 0:
        return []
    if label_field is None:
        return list(cases[:limit])
    grouped: dict[str, list[EvalCase]] = defaultdict(list)
    for case in cases:
        grouped[_label_from_case(case, label_field=label_field) or "unlabeled"].append(case)
    rows: list[EvalCase] = []
    labels = sorted(grouped)
    index = 0
    while len(rows) < limit:
        added = False
        for label in labels:
            bucket = grouped[label]
            if index < len(bucket):
                rows.append(bucket[index])
                added = True
                if len(rows) >= limit:
                    break
        if not added:
            break
        index += 1
    return rows


def _label_from_case(case: EvalCase, *, label_field: str | None) -> str | None:
    if label_field and isinstance(case.expected, dict) and isinstance(case.expected.get(label_field), str):
        return str(case.expected[label_field])
    category = case.metadata.get("category")
    if isinstance(category, str):
        return category
    return None


def _actual_label(output: Any, grade_labels: list[str], *, label_field: str | None) -> str | None:
    if label_field and isinstance(output, dict) and output.get(label_field) is not None:
        return str(output[label_field])
    for label in grade_labels:
        if label.startswith("actual:"):
            return label.split(":", 1)[1]
    return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
