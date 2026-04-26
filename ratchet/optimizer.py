from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time
from typing import Any, Iterable

from ratchet.adapters import AdapterProtocol
from ratchet.diagnosis import FailureDiagnoser
from ratchet.io import agent_spec_hash, patch_hash
from ratchet.objectives import (
    behavior_flip_summary,
    patch_rejection_reason,
    compare_summaries,
    final_gate_rejection_reason,
    objective_sort_key,
    pareto_frontier,
)
from ratchet.patches import compose_patches
from ratchet.proposals import ProposalEngine
from ratchet.reporting import RatchetReporter, build_outcome_analysis
from ratchet.results import (
    PatchSummary,
    CaseEvaluation,
    Comparison,
    OptimizerStats,
    RatchetResult,
    ResultStore,
    build_cache_namespace,
    split_cases,
)
from ratchet.surface import SurfaceGenerator
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    DiagnosticTrace,
    EditableTarget,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OperationalMetrics,
    OptimizationObjective,
    RunRecord,
)


SEARCH_FRONTIER_WIDTH = 2
PROPOSAL_RETRY_BUDGET = 1


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


class RatchetOptimizer:
    def __init__(
        self,
        adapter: AdapterProtocol,
        out_dir: Path,
        env_path: str = ".env",
        dev_budget: int = 20,
        holdout_budget: int = 5,
        objective: OptimizationObjective | None = None,
        optimizer_model: str = "gpt-5.4",
        optimizer_reasoning: str = "medium",
        samples_per_case: int = 1,
        max_case_retries: int = 2,
        case_timeout_s: int = 180,
        fail_fast: bool = False,
        run_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.out_dir = out_dir
        self.env_path = env_path
        self.dev_budget = dev_budget
        self.holdout_budget = holdout_budget
        self.objective = objective or OptimizationObjective()
        self.agent_spec = adapter.agent_spec()
        self.surface_generator = SurfaceGenerator()
        self.diagnoser = FailureDiagnoser(
            env_path=env_path,
            model=optimizer_model,
            reasoning_effort=optimizer_reasoning,
        )
        self.proposer = ProposalEngine(
            env_path=env_path,
            model=optimizer_model,
            reasoning_effort=optimizer_reasoning,
        )
        if samples_per_case <= 0:
            raise ValueError("samples_per_case must be positive.")
        self.samples_per_case = samples_per_case
        self.max_case_retries = max_case_retries
        self.case_timeout_s = case_timeout_s
        self.fail_fast = fail_fast
        self.run_metadata = dict(run_metadata or {})
        self.cache_namespace = build_cache_namespace(
            agent_spec=self.agent_spec,
            objective=self.objective,
            run_metadata=self.run_metadata,
        )
        self.store = ResultStore(out_dir, cache_namespace=self.cache_namespace)
        self.stats = OptimizerStats()
        self.started_at: datetime | None = None

    def run(self, cases: tuple[EvalCase, ...]) -> RatchetResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc)
        dev_cases, holdout_cases = split_cases(cases)

        baseline_patch = AgentPatch.empty()
        baseline_dev = self.evaluate_patch(baseline_patch, dev_cases)
        baseline_holdout = self.evaluate_patch(baseline_patch, holdout_cases)

        accepted_dev_patches: list[PatchSummary] = []
        accepted_dev_hashes: set[str] = set()
        frontier: list[PatchSummary] = [baseline_dev]
        decision_log: list[dict[str, Any]] = []
        diagnoses_log: list[dict[str, Any]] = []
        proposals_log: list[dict[str, Any]] = []
        evaluated_patch_hashes = {baseline_dev.patch_hash}
        generated_surface_rows: list[dict[str, Any]] = [
            target.to_dict() for target in self.surface_generator.generate(self.agent_spec, self.objective)
        ]
        dev_evaluations = 0
        iteration = 0

        while dev_evaluations < self.dev_budget and frontier:
            iteration += 1
            parent_summaries = sorted(frontier, key=lambda summary: objective_sort_key(summary, self.objective))[
                : SEARCH_FRONTIER_WIDTH
            ]
            next_frontier_by_hash: dict[str, PatchSummary] = {}
            search_complete = False

            for parent_index, current_dev in enumerate(parent_summaries):
                if dev_evaluations >= self.dev_budget:
                    break
                remaining_parents = len(parent_summaries) - parent_index
                remaining_budget = self.dev_budget - dev_evaluations
                proposal_budget = max(1, (remaining_budget + remaining_parents - 1) // remaining_parents)
                current_spec = self.agent_spec.apply_patch(current_dev.patch) if self.agent_spec else None
                surface = self.surface_generator.generate(current_spec, self.objective)
                generated_surface_rows = [target.to_dict() for target in surface]
                diagnoses, diagnosis_analysis = self.diagnoser.diagnose(current_dev, surface, self.objective)
                for diagnosis in diagnoses:
                    diagnoses_log.append(
                        {
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "patch_hash": current_dev.patch_hash,
                            **diagnosis.to_dict(),
                        }
                    )
                if (
                    self.objective.mode == "correctness"
                    and not diagnoses
                    and current_dev.pass_count == current_dev.case_count
                ):
                    decision_log.append(
                        {
                            "type": "search_stopped",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "patch_hash": current_dev.patch_hash,
                            "reason": "current dev branch has no correctness failures",
                        }
                    )
                    search_complete = True
                    break
                accepted_rows, evaluations_used = self._propose_and_evaluate_parent(
                    current_dev=current_dev,
                    baseline_dev=baseline_dev,
                    dev_cases=dev_cases,
                    surface=surface,
                    diagnoses=diagnoses,
                    diagnosis_analysis=diagnosis_analysis,
                    current_spec=current_spec,
                    evaluated_patch_hashes=evaluated_patch_hashes,
                    proposals_log=proposals_log,
                    decision_log=decision_log,
                    iteration=iteration,
                    parent_index=parent_index,
                    parent_summaries=parent_summaries,
                    proposal_budget=proposal_budget,
                )
                dev_evaluations += evaluations_used
                if not accepted_rows and evaluations_used > 0 and dev_evaluations < self.dev_budget:
                    retry_rows, retry_evaluations_used = self._propose_and_evaluate_parent(
                        current_dev=current_dev,
                        baseline_dev=baseline_dev,
                        dev_cases=dev_cases,
                        surface=surface,
                        diagnoses=diagnoses,
                        diagnosis_analysis=diagnosis_analysis,
                        current_spec=current_spec,
                        evaluated_patch_hashes=evaluated_patch_hashes,
                        proposals_log=proposals_log,
                        decision_log=decision_log,
                        iteration=iteration,
                        parent_index=parent_index,
                        parent_summaries=parent_summaries,
                        proposal_budget=min(PROPOSAL_RETRY_BUDGET, self.dev_budget - dev_evaluations),
                        proposal_retry=True,
                        retry_reason="no_accepted_candidates_from_parent",
                    )
                    dev_evaluations += retry_evaluations_used
                    accepted_rows.extend(retry_rows)

                accepted_rows.sort(key=lambda item: objective_sort_key(item[1], self.objective))
                for _, accepted_summary, _ in accepted_rows:
                    if accepted_summary.patch_hash not in accepted_dev_hashes:
                        accepted_dev_hashes.add(accepted_summary.patch_hash)
                        accepted_dev_patches.append(accepted_summary)
                    next_frontier_by_hash.setdefault(accepted_summary.patch_hash, accepted_summary)
                if accepted_rows:
                    chosen_proposal, chosen_dev, _ = accepted_rows[0]
                    decision_log.append(
                        {
                            "type": "accepted_proposal",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "proposal_patch_hash": patch_hash(chosen_proposal),
                            "patch_hash": chosen_dev.patch_hash,
                            "metrics": chosen_dev.to_dict(),
                        }
                    )

            if search_complete:
                break
            if not next_frontier_by_hash:
                break
            frontier = sorted(
                next_frontier_by_hash.values(),
                key=lambda summary: objective_sort_key(summary, self.objective),
            )[: SEARCH_FRONTIER_WIDTH]
            decision_log.append(
                {
                    "type": "frontier_update",
                    "iteration": iteration,
                    "frontier_width": SEARCH_FRONTIER_WIDTH,
                    "frontier_patch_hashes": [summary.patch_hash for summary in frontier],
                    "accepted_patch_hashes": [
                        summary.patch_hash
                        for summary in sorted(
                            next_frontier_by_hash.values(),
                            key=lambda summary: objective_sort_key(summary, self.objective),
                        )
                    ],
                }
            )

        best_dev_patch = min(
            [baseline_dev, *accepted_dev_patches],
            key=lambda summary: objective_sort_key(summary, self.objective),
        )
        finalist_dev_patches = sorted(
            accepted_dev_patches,
            key=lambda summary: objective_sort_key(summary, self.objective),
        )[: self.holdout_budget]

        holdout_patches: list[PatchSummary] = []
        promotable: list[tuple[PatchSummary, Comparison]] = []
        if self.holdout_budget <= 0 and accepted_dev_patches:
            decision_log.append(
                {
                    "type": "holdout_validation_skipped",
                    "reason": "holdout_budget validation budget exhausted",
                    "holdout_budget": self.holdout_budget,
                }
            )
        for dev_summary in finalist_dev_patches:
            holdout_summary = self.evaluate_patch(dev_summary.patch, holdout_cases)
            holdout_patches.append(holdout_summary)
            final_rejection_reason, comparison = final_gate_rejection_reason(
                baseline_holdout,
                holdout_summary,
                self.objective,
            )
            passed_gate = final_rejection_reason is None
            flip_summary = behavior_flip_summary(baseline_holdout, holdout_summary)
            decision_log.append(
                {
                    "type": "holdout_validation",
                    "patch_hash": holdout_summary.patch_hash,
                    "metrics": holdout_summary.to_dict(),
                    "comparison_to_baseline": comparison.to_dict(),
                    "behavior_flip_summary": flip_summary,
                    "passed_final_gate": passed_gate,
                    "rejection_reason": final_rejection_reason,
                }
            )
            if passed_gate:
                promotable.append((holdout_summary, comparison))

        if promotable:
            promotable.sort(key=lambda item: objective_sort_key(item[0], self.objective))
            selected_holdout = promotable[0][0]
            promoted = True
            selection_reason = f"Promoted best holdout patch for {self.objective.mode} objective."
        else:
            selected_holdout = baseline_holdout
            promoted = False
            selection_reason = "No finalist cleared the holdout objective gate; kept original baseline."

        selected_patch = selected_holdout.patch
        selected_patch_hash = selected_holdout.patch_hash
        decision_log.append(
            {
                "type": "final_selection",
                "selected_patch_hash": selected_patch_hash,
                "promoted": promoted,
                "reason": selection_reason,
                "best_dev_patch_hash": best_dev_patch.patch_hash,
            }
        )

        outcome_analysis = build_outcome_analysis(
            objective=self.objective,
            promoted=promoted,
            baseline_dev=baseline_dev,
            accepted_dev_patches=accepted_dev_patches,
            holdout_patches=holdout_patches,
            decision_log=decision_log,
        )
        manifest = self.build_manifest(
            total_cases=len(cases),
            selected_patch_hash=selected_patch_hash,
            promoted=promoted,
            generated_surface=generated_surface_rows,
            outcome_analysis=outcome_analysis,
        )
        result = RatchetResult(
            baseline_patch=baseline_patch,
            selected_patch=selected_patch,
            selected_patch_hash=selected_patch_hash,
            promoted=promoted,
            baseline_dev=baseline_dev,
            baseline_holdout=baseline_holdout,
            best_dev_patch=best_dev_patch,
            selected_holdout=selected_holdout,
            accepted_dev_patches=accepted_dev_patches,
            holdout_patches=holdout_patches,
            pareto_frontier=pareto_frontier([baseline_holdout, *holdout_patches]),
            decision_log=decision_log,
            diagnoses=diagnoses_log,
            proposals=proposals_log,
            generated_surface=generated_surface_rows,
            selection_reason=selection_reason,
            outcome_analysis=outcome_analysis,
            manifest=manifest,
        )
        self.write_outputs(result)
        return result

    def _propose_and_evaluate_parent(
        self,
        *,
        current_dev: PatchSummary,
        baseline_dev: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
        surface: list[EditableTarget],
        diagnoses: list[FailureDiagnosis],
        diagnosis_analysis: str,
        current_spec: AgentSpec | None,
        evaluated_patch_hashes: set[str],
        proposals_log: list[dict[str, Any]],
        decision_log: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        parent_summaries: list[PatchSummary],
        proposal_budget: int,
        proposal_retry: bool = False,
        retry_reason: str | None = None,
    ) -> tuple[list[tuple[AgentPatch, PatchSummary, Comparison]], int]:
        if proposal_budget <= 0:
            return [], 0
        target_diagnosis = diagnoses[0] if diagnoses else None
        proposals, proposal_analysis = self.proposer.propose(
            current_dev,
            surface,
            objective=self.objective,
            diagnosis=target_diagnosis,
            diagnoses=diagnoses,
            seen_hashes=evaluated_patch_hashes,
            current_spec=current_spec,
            history=proposals_log,
            proposal_budget=proposal_budget,
        )
        attempt = 2 if proposal_retry else 1
        decision_log.append(
            {
                "type": "proposal_iteration",
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "retry_reason": retry_reason,
                "parent_rank": parent_index + 1,
                "parent_patch_hash": current_dev.patch_hash,
                "patch_hash": current_dev.patch_hash,
                "frontier_width": SEARCH_FRONTIER_WIDTH,
                "active_frontier": [summary.patch_hash for summary in parent_summaries],
                "diagnosis_analysis": diagnosis_analysis,
                "proposal_analysis": proposal_analysis,
                "proposal_stats": self.proposer.last_stats.to_dict(),
                "diagnoses": [diagnosis.to_dict() for diagnosis in diagnoses],
                "diagnosis": target_diagnosis.to_dict() if target_diagnosis else None,
                "proposal_hashes": [patch_hash(proposal) for proposal in proposals],
                "candidate_proposals": self.proposer.last_candidate_rows,
            }
        )
        if not proposals:
            return [], 0

        accepted_rows: list[tuple[AgentPatch, PatchSummary, Comparison]] = []
        evaluations_used = 0
        for proposal in proposals:
            if evaluations_used >= proposal_budget:
                break
            patch = compose_patches(current_dev.patch, proposal)
            digest = patch_hash(patch)
            if digest in evaluated_patch_hashes:
                continue
            summary = self.evaluate_patch(patch, dev_cases)
            evaluations_used += 1
            evaluated_patch_hashes.add(digest)
            comparison = compare_summaries(current_dev, summary)
            flip_summary = behavior_flip_summary(current_dev, summary)
            rejection_reason = patch_rejection_reason(
                baseline=baseline_dev,
                reference=current_dev,
                patch_summary=summary,
                objective=self.objective,
            )
            accepted = rejection_reason is None
            proposal_row = {
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "retry_reason": retry_reason,
                "parent_rank": parent_index + 1,
                "parent_patch_hash": current_dev.patch_hash,
                "proposal_patch_hash": patch_hash(proposal),
                "proposal": proposal.to_dict(),
                "patch_hash": digest,
                "patch": patch.to_dict(),
                "comparison_to_parent": comparison.to_dict(),
                "behavior_flip_summary": flip_summary,
                "metrics": summary.to_dict(),
                "accepted": accepted,
                "rejection_reason": rejection_reason,
                "diagnosis_category": proposal.metadata.get("diagnosis_category"),
            }
            proposals_log.append(proposal_row)
            decision_log.append({"type": "proposal_evaluation", **proposal_row})
            if accepted:
                accepted_rows.append((proposal, summary, comparison))
        return accepted_rows, evaluations_used

    def evaluate_patch(self, patch: AgentPatch, cases: tuple[EvalCase, ...]) -> PatchSummary:
        digest = patch_hash(patch)
        evaluations: list[CaseEvaluation] = []
        for case in cases:
            for sample_index in range(self.samples_per_case):
                cached = self.store.get(digest, case, sample_index=sample_index)
                if cached is not None:
                    self.stats.cache_hits += 1
                    evaluations.append(cached)
                    continue
                evaluation = self._execute_case(patch, case, sample_index=sample_index)
                self.store.put(digest, patch, evaluation)
                if self.fail_fast and evaluation.record.metrics.error:
                    raise RuntimeError(
                        f"Fail-fast stopping after case {case.id}: {evaluation.record.metrics.error}"
                    )
                evaluations.append(evaluation)
        return PatchSummary(
            patch_hash=digest,
            patch=patch,
            split=cases[0].split,
            evaluations=evaluations,
        )

    def _execute_case(self, patch: AgentPatch, case: EvalCase, *, sample_index: int = 0) -> CaseEvaluation:
        total_attempts = self.max_case_retries + 1
        started_at = time.perf_counter()
        last_error: Exception | None = None
        last_phase = "run_case"
        effective_patch = None if patch.is_empty else patch
        for attempt in range(1, total_attempts + 1):
            try:
                last_phase = "run_case"
                with case_timeout(self.case_timeout_s):
                    record = self.adapter.run_case(case, effective_patch)
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
                return CaseEvaluation(case=case, record=record, grade=grade, sample_index=sample_index)
            except Exception as error:
                last_error = error
                if attempt < total_attempts:
                    self.stats.retries += 1
                    continue

        assert last_error is not None
        elapsed = time.perf_counter() - started_at
        message = f"{type(last_error).__name__}: {last_error}"
        if isinstance(last_error, TimeoutError):
            self.stats.timeouts += 1
            labels = ["timeout"]
        elif last_phase == "grade":
            self.stats.grader_errors += 1
            labels = ["grader_error"]
        else:
            self.stats.runtime_errors += 1
            labels = ["runtime_error"]
        record = RunRecord(
            output=None,
            metrics=OperationalMetrics(
                latency_s=elapsed,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                error=message,
            ),
            diagnostics=DiagnosticTrace(metadata={"phase": last_phase}),
        )
        grade = GradeResult(score=0.0, passed=False, labels=labels, notes=message)
        self.stats.fresh_case_evaluations += 1
        return CaseEvaluation(case=case, record=record, grade=grade, sample_index=sample_index)

    def build_manifest(
        self,
        *,
        total_cases: int,
        selected_patch_hash: str,
        promoted: bool,
        generated_surface: list[dict[str, Any]],
        outcome_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        ended_at = datetime.now(timezone.utc)
        return {
            **self.run_metadata,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": ended_at.isoformat(),
            "duration_s": (
                (ended_at - self.started_at).total_seconds() if self.started_at else None
            ),
            "total_cases": total_cases,
            "agent_spec_hash": agent_spec_hash(self.agent_spec),
            "objective": self.objective.to_dict(),
            "generated_surface_count": len(generated_surface),
            "samples_per_case": self.samples_per_case,
            "selected_patch_hash": selected_patch_hash,
            "promoted": promoted,
            "outcome": outcome_analysis,
            "cache_namespace": self.cache_namespace,
            "stats": self.stats.to_dict(),
        }

    def write_outputs(self, result: RatchetResult) -> None:
        RatchetReporter(
            adapter=self.adapter,
            out_dir=self.out_dir,
            objective=self.objective,
            stats=self.stats,
        ).write_outputs(result)
