from __future__ import annotations

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import threading
import time
from typing import Any, Callable, Iterable

from ratchet.adapters import AdapterProtocol, checked_agent_spec
from ratchet.affordances import OptimizationAffordance, generate_optimization_affordances
from ratchet.diagnosis import FailureDiagnoser
from ratchet.evidence import ProposalExampleBank, build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.evidence_ledger import EvidenceLedger
from ratchet.experiments import ExperimentIntent, ResearchState, TaskTheory, build_task_theory
from ratchet.ideation import build_ideation_metrics
from ratchet.io import agent_spec_hash, append_jsonl, patch_hash
from ratchet.model_client import model_request_limits
from ratchet.objectives import (
    behavior_flip_summary,
    constraint_rejection_reason,
    patch_rejection_reason,
    compare_summaries,
    final_gate_status,
    objective_rejection_reason,
    objective_sort_key,
    pareto_frontier,
    select_recommended_patch,
)
from ratchet.patches import compose_patches
from ratchet.profiling import (
    build_run_profile,
    confirmation_case_subset,
    confirmation_result,
    quality_cost_tradeoffs,
    runtime_reliability_diagnostics,
)
from ratchet.proposals import CandidateImplementer
from ratchet.research import MeasurementSelector, MeasurementAction, ResearchPlanner
from ratchet.reporting import RatchetReporter, build_outcome_analysis
from ratchet.results import (
    PatchSummary,
    CaseEvaluation,
    Comparison,
    OptimizerStats,
    RatchetResult,
    ResultStore,
    build_cache_namespace,
    split_train_dev_holdout,
)
from ratchet.surface import SurfaceGenerator
from ratchet.transforms import (
    CandidateProposal,
    TransformContextKey,
    build_search_hypothesis,
    observe_transform_result,
    summarize_affordance_results,
    summarize_transform_context_results,
    summarize_transform_results,
)
from ratchet.types import (
    AgentPatch,
    AgentSpec,
    DiagnosticTrace,
    EditableTarget,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OperationalMetrics,
    PatchOperation,
    OptimizationObjective,
    RunRecord,
)


SEARCH_FRONTIER_WIDTH = 2
PROPOSAL_RETRY_BUDGET = 1
FINALIST_CONFIRMATION_SAMPLES = 2
FRONTIER_PARENT_STALL_LIMIT = 2
MAX_SIMPLIFICATION_VARIANTS_PER_FINALIST = 2
MAX_SIMPLIFICATION_PARENT_COUNT = 2
MAX_SIMPLIFICATION_FULL_DEV_PER_PARENT = 1
MAX_SIMPLIFICATION_FULL_DEV_PER_RUN = 2
MIN_REMAINING_DEV_EVALS_FOR_NEW_ROUND = 2
MAX_CONSECUTIVE_ZERO_EVAL_PARENT_ATTEMPTS = 3
MAX_FULL_DEV_EXPERIMENT_CANDIDATES_PER_GROUP = 3
MAX_LATE_FULL_DEV_EXPERIMENT_CANDIDATES_PER_GROUP = 2
MAX_LATE_FULL_DEV_CANDIDATES_PER_ACTION = 1
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class FrontierParentState:
    visits: int = 0
    consecutive_stalls: int = 0
    accepted_child_count: int = 0
    last_selected_iteration: int = 0
    exhausted: bool = False


@dataclass
class CandidateEvaluationState:
    candidate: CandidateProposal
    patch: AgentPatch
    patch_hash: str
    proposal_patch_hash: str
    transform_context: TransformContextKey
    stage_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: PatchSummary | None = None
    comparison: Comparison | None = None
    flip_summary: dict[str, Any] | None = None
    rejection_reason: str | None = None
    constraint_warning: str | None = None
    frontier_status: str = "pending"
    accepted: bool = False
    full_dev_evaluated: bool = False


@contextlib.contextmanager
def case_timeout(timeout_s: int) -> Iterable[None]:
    if timeout_s <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        # SIGALRM is process-wide and cannot safely enforce per-case deadlines in worker threads.
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
        diagnoser_model: str | None = None,
        diagnoser_reasoning: str | None = None,
        research_planner_model: str | None = None,
        research_planner_reasoning: str | None = None,
        candidate_implementer_model: str | None = None,
        candidate_implementer_reasoning: str | None = None,
        measurement_selector_model: str | None = None,
        measurement_selector_reasoning: str | None = None,
        samples_per_case: int = 1,
        case_concurrency: int = 1,
        stage_case_concurrency: int | None = None,
        max_case_retries: int = 2,
        case_timeout_s: int = 180,
        fail_fast: bool = False,
        expensive_candidate_cost_ratio: float = 10.0,
        max_dev_measurement_cost_usd: float | None = None,
        max_holdout_measurement_cost_usd: float | None = None,
        max_dev_measurement_tool_calls: int | None = None,
        max_holdout_measurement_tool_calls: int | None = None,
        max_dev_measurement_turns: int | None = None,
        max_holdout_measurement_turns: int | None = None,
        run_metadata: dict[str, Any] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.adapter = adapter
        self.out_dir = out_dir
        self.env_path = env_path
        self.dev_budget = dev_budget
        self.holdout_budget = holdout_budget
        self.objective = objective or OptimizationObjective()
        self.agent_spec = checked_agent_spec(adapter)
        self.surface_generator = SurfaceGenerator()
        self.optimizer_role_models = {
            "diagnoser": diagnoser_model or optimizer_model,
            "research_planner": research_planner_model or optimizer_model,
            "candidate_implementer": candidate_implementer_model or optimizer_model,
            "measurement_selector": measurement_selector_model or optimizer_model,
        }
        self.optimizer_role_reasoning = {
            "diagnoser": diagnoser_reasoning or optimizer_reasoning,
            "research_planner": research_planner_reasoning or optimizer_reasoning,
            "candidate_implementer": candidate_implementer_reasoning or optimizer_reasoning,
            "measurement_selector": measurement_selector_reasoning or optimizer_reasoning,
        }
        self.diagnoser = FailureDiagnoser(
            env_path=env_path,
            model=self.optimizer_role_models["diagnoser"],
            reasoning_effort=self.optimizer_role_reasoning["diagnoser"],
        )
        self.candidate_implementer = CandidateImplementer(
            env_path=env_path,
            model=self.optimizer_role_models["candidate_implementer"],
            reasoning_effort=self.optimizer_role_reasoning["candidate_implementer"],
        )
        self.research_planner = ResearchPlanner(
            env_path=env_path,
            model=self.optimizer_role_models["research_planner"],
            reasoning_effort=self.optimizer_role_reasoning["research_planner"],
        )
        self.measurement_selector = MeasurementSelector(
            env_path=env_path,
            model=self.optimizer_role_models["measurement_selector"],
            reasoning_effort=self.optimizer_role_reasoning["measurement_selector"],
        )
        if samples_per_case <= 0:
            raise ValueError("samples_per_case must be positive.")
        self.samples_per_case = samples_per_case
        if case_concurrency <= 0:
            raise ValueError("case_concurrency must be positive.")
        self.case_concurrency = case_concurrency
        if stage_case_concurrency is not None and stage_case_concurrency <= 0:
            raise ValueError("stage_case_concurrency must be positive when set.")
        self.stage_case_concurrency = stage_case_concurrency or case_concurrency
        self.max_case_retries = max_case_retries
        self.case_timeout_s = case_timeout_s
        self.fail_fast = fail_fast
        if expensive_candidate_cost_ratio <= 0:
            raise ValueError("expensive_candidate_cost_ratio must be positive.")
        self.expensive_candidate_cost_ratio = expensive_candidate_cost_ratio
        if max_dev_measurement_cost_usd is not None and max_dev_measurement_cost_usd < 0:
            raise ValueError("max_dev_measurement_cost_usd must be non-negative when set.")
        if max_holdout_measurement_cost_usd is not None and max_holdout_measurement_cost_usd < 0:
            raise ValueError("max_holdout_measurement_cost_usd must be non-negative when set.")
        for name, value in {
            "max_dev_measurement_tool_calls": max_dev_measurement_tool_calls,
            "max_holdout_measurement_tool_calls": max_holdout_measurement_tool_calls,
            "max_dev_measurement_turns": max_dev_measurement_turns,
            "max_holdout_measurement_turns": max_holdout_measurement_turns,
        }.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when set.")
        self.max_dev_measurement_cost_usd = max_dev_measurement_cost_usd
        self.max_holdout_measurement_cost_usd = max_holdout_measurement_cost_usd
        self.max_dev_measurement_tool_calls = max_dev_measurement_tool_calls
        self.max_holdout_measurement_tool_calls = max_holdout_measurement_tool_calls
        self.max_dev_measurement_turns = max_dev_measurement_turns
        self.max_holdout_measurement_turns = max_holdout_measurement_turns
        self._dev_measurement_cost_usd = 0.0
        self._holdout_measurement_cost_usd = 0.0
        self._dev_measurement_tool_calls = 0.0
        self._holdout_measurement_tool_calls = 0.0
        self._dev_measurement_turns = 0.0
        self._holdout_measurement_turns = 0.0
        self.run_metadata = dict(run_metadata or {})
        self.cache_namespace = build_cache_namespace(
            agent_spec=self.agent_spec,
            objective=self.objective,
            run_metadata=self.run_metadata,
        )
        self.store = ResultStore(out_dir, cache_namespace=self.cache_namespace)
        self.stats = OptimizerStats()
        self.started_at: datetime | None = None
        self.progress_callback = progress_callback
        self._progress_started_at: float | None = None
        self._progress_path: Path | None = None
        self._progress_lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.optimizer_call_diagnostics: list[dict[str, Any]] = []

    def run(self, cases: tuple[EvalCase, ...]) -> RatchetResult:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = datetime.now(timezone.utc)
        self._progress_started_at = time.perf_counter()
        self._progress_path = self.out_dir / "progress.jsonl"
        self._progress_path.write_text("")
        self.optimizer_call_diagnostics = []
        self._dev_measurement_cost_usd = 0.0
        self._holdout_measurement_cost_usd = 0.0
        self._dev_measurement_tool_calls = 0.0
        self._holdout_measurement_tool_calls = 0.0
        self._dev_measurement_turns = 0.0
        self._holdout_measurement_turns = 0.0
        train_cases, dev_cases, holdout_cases = split_train_dev_holdout(cases)
        proposal_example_bank = build_proposal_example_bank(train_cases)
        self._emit_progress(
            "run_started",
            total_cases=len(cases),
            train_cases=len(train_cases),
            dev_cases=len(dev_cases),
            holdout_cases=len(holdout_cases),
            proposal_example_count=len(proposal_example_bank.examples),
            dev_budget=self.dev_budget,
            holdout_budget=self.holdout_budget,
            case_concurrency=self.case_concurrency,
            stage_case_concurrency=self.stage_case_concurrency,
            objective=self.objective.mode,
        )

        baseline_patch = AgentPatch.empty()
        self._emit_progress("baseline_dev_started", case_count=len(dev_cases))
        baseline_dev = self.evaluate_patch(baseline_patch, dev_cases)
        self._emit_progress("baseline_dev_completed", **_summary_progress_fields(baseline_dev))
        self._emit_progress("baseline_holdout_started", case_count=len(holdout_cases))
        baseline_holdout = self.evaluate_patch(baseline_patch, holdout_cases)
        self._emit_progress("baseline_holdout_completed", **_summary_progress_fields(baseline_holdout))

        accepted_dev_patches: list[PatchSummary] = []
        accepted_dev_hashes: set[str] = set()
        parent_pool_by_hash: dict[str, PatchSummary] = {baseline_dev.patch_hash: baseline_dev}
        frontier_states: dict[str, FrontierParentState] = {
            baseline_dev.patch_hash: FrontierParentState(),
        }
        decision_log: list[dict[str, Any]] = []
        diagnoses_log: list[dict[str, Any]] = []
        proposals_log: list[dict[str, Any]] = []
        task_theory_log: list[dict[str, Any]] = []
        evidence_ledger = EvidenceLedger()
        diagnosis_cache: dict[str, tuple[list[FailureDiagnosis], str]] = {}
        task_theory_cache: dict[str, TaskTheory] = {}
        evaluated_patch_hashes = {baseline_dev.patch_hash}
        generated_surface = self.surface_generator.generate(self.agent_spec, self.objective)
        generated_surface_rows: list[dict[str, Any]] = [target.to_dict() for target in generated_surface]
        dev_evaluations = 0
        iteration = 0
        consecutive_zero_eval_parent_attempts = 0

        while dev_evaluations < self.dev_budget and _has_selectable_frontier_parent(frontier_states):
            if (
                accepted_dev_patches
                and self.dev_budget - dev_evaluations < MIN_REMAINING_DEV_EVALS_FOR_NEW_ROUND
            ):
                decision_log.append(
                    {
                        "type": "search_stopped",
                        "iteration": iteration + 1,
                        "reason": "remaining dev budget too small for another informative proposal round",
                        "dev_evaluations": dev_evaluations,
                        "dev_budget": self.dev_budget,
                    }
                )
                self._emit_progress(
                    "search_stopped",
                    iteration=iteration + 1,
                    reason="remaining dev budget too small for another informative proposal round",
                    dev_evaluations=dev_evaluations,
                    dev_budget=self.dev_budget,
                )
                break
            iteration += 1
            parent_summaries = _select_frontier_parents(
                parent_pool_by_hash.values(),
                frontier_states=frontier_states,
                objective=self.objective,
                width=SEARCH_FRONTIER_WIDTH,
            )
            if not parent_summaries:
                break
            self._emit_progress(
                "iteration_started",
                iteration=iteration,
                frontier_width=len(parent_summaries),
                dev_evaluations=dev_evaluations,
                dev_budget=self.dev_budget,
            )
            next_frontier_by_hash: dict[str, PatchSummary] = {}
            search_complete = False

            for parent_index, current_dev in enumerate(parent_summaries):
                if dev_evaluations >= self.dev_budget:
                    break
                remaining_parents = len(parent_summaries) - parent_index
                remaining_budget = self.dev_budget - dev_evaluations
                proposal_budget = max(1, (remaining_budget + remaining_parents - 1) // remaining_parents)
                parent_state = frontier_states.setdefault(current_dev.patch_hash, FrontierParentState())
                parent_state.visits += 1
                parent_state.last_selected_iteration = iteration
                current_spec = self.agent_spec.apply_patch(current_dev.patch) if self.agent_spec else None
                surface = self.surface_generator.generate(current_spec, self.objective)
                generated_surface_rows = [target.to_dict() for target in surface]
                self._emit_progress(
                    "parent_started",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    parent_patch_hash=current_dev.patch_hash,
                    **_summary_progress_fields(current_dev),
                )
                self._emit_progress(
                    "diagnosis_started",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    failure_count=current_dev.case_count - current_dev.pass_count,
                )
                diagnosis_cached = current_dev.patch_hash in diagnosis_cache
                if diagnosis_cached:
                    diagnoses, diagnosis_analysis = diagnosis_cache[current_dev.patch_hash]
                    diagnosis_call_diagnostics: dict[str, Any] = {}
                    decision_log.append(
                        {
                            "type": "optimizer_cache_hit",
                            "cache": "diagnosis",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "patch_hash": current_dev.patch_hash,
                        }
                    )
                else:
                    diagnoses, diagnosis_analysis = self.diagnoser.diagnose(current_dev, surface, self.objective)
                    diagnosis_cache[current_dev.patch_hash] = (diagnoses, diagnosis_analysis)
                    diagnosis_call_diagnostics = self.diagnoser.last_call_diagnostics or {}
                    if self.diagnoser.last_call_diagnostics is not None:
                        self.optimizer_call_diagnostics.append(
                            {
                                "iteration": iteration,
                                "parent_rank": parent_index + 1,
                                "parent_patch_hash": current_dev.patch_hash,
                                **self.diagnoser.last_call_diagnostics,
                            }
                        )
                self._emit_progress(
                    "diagnosis_completed",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    diagnosis_count=len(diagnoses),
                    analysis=diagnosis_analysis,
                    cached=diagnosis_cached,
                    call_diagnostics=diagnosis_call_diagnostics,
                )
                task_theory_cached = current_dev.patch_hash in task_theory_cache
                if task_theory_cached:
                    task_theory = task_theory_cache[current_dev.patch_hash]
                    decision_log.append(
                        {
                            "type": "optimizer_cache_hit",
                            "cache": "task_theory",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "patch_hash": current_dev.patch_hash,
                        }
                    )
                else:
                    task_theory = build_task_theory(
                        summary=current_dev,
                        diagnoses=diagnoses,
                        objective=self.objective,
                        proposal_example_bank=proposal_example_bank,
                    )
                    task_theory_cache[current_dev.patch_hash] = task_theory
                task_theory_row = {
                    "type": "task_theory",
                    "iteration": iteration,
                    "parent_rank": parent_index + 1,
                    "parent_patch_hash": current_dev.patch_hash,
                    "patch_hash": current_dev.patch_hash,
                    "cached": task_theory_cached,
                    "task_theory": task_theory.to_dict(),
                }
                decision_log.append(task_theory_row)
                task_theory_log.append(task_theory_row)
                self._emit_progress(
                    "task_theory_ready",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    bottleneck_class=task_theory.bottleneck_class,
                    residual_failure_modes=task_theory.residual_failure_modes,
                    confidence=task_theory.confidence,
                    cached=task_theory_cached,
                )
                search_hypothesis = build_search_hypothesis(
                    summary=current_dev,
                    surface=surface,
                    objective=self.objective,
                    history=proposals_log,
                    parent_patch_hash=current_dev.patch_hash,
                    diagnoses=diagnoses,
                    proposal_example_count=len(proposal_example_bank.examples),
                )
                search_hypothesis_row = {
                    "type": "search_hypothesis",
                    "iteration": iteration,
                    "parent_rank": parent_index + 1,
                    "parent_patch_hash": current_dev.patch_hash,
                    "patch_hash": current_dev.patch_hash,
                    "search_hypothesis": search_hypothesis.to_dict(),
                }
                decision_log.append(search_hypothesis_row)
                self._emit_progress(
                    "search_hypothesis_ready",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    active_families=search_hypothesis.active_families,
                    active_context_count=len(search_hypothesis.active_contexts),
                )
                affordances = generate_optimization_affordances(
                    surface,
                    objective=self.objective,
                    active_families=search_hypothesis.active_families,
                    evidence=_affordance_evidence_from_theory(task_theory, diagnoses),
                )
                decision_log.append(
                    {
                        "type": "optimization_affordances",
                        "iteration": iteration,
                        "parent_rank": parent_index + 1,
                        "parent_patch_hash": current_dev.patch_hash,
                        "affordances": [affordance.to_dict() for affordance in affordances],
                    }
                )
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
                    self._emit_progress(
                        "search_stopped",
                        iteration=iteration,
                        parent_rank=parent_index + 1,
                        reason="current dev branch has no correctness failures",
                    )
                    search_complete = True
                    break
                experiment_intents = self._plan_parent_research_action(
                    current_dev=current_dev,
                    task_theory=task_theory,
                    search_hypothesis=search_hypothesis,
                    affordances=affordances,
                    proposals_log=proposals_log,
                    decision_log=decision_log,
                    iteration=iteration,
                    parent_index=parent_index,
                    proposal_budget=proposal_budget,
                    dev_evaluations_used=dev_evaluations,
                )
                if not experiment_intents:
                    parent_state.exhausted = True
                    parent_state.consecutive_stalls += 1
                    continue
                accepted_rows, evaluations_used = self._propose_and_evaluate_parent(
                    current_dev=current_dev,
                    baseline_dev=baseline_dev,
                    dev_cases=dev_cases,
                    surface=surface,
                    diagnoses=diagnoses,
                    task_theory=task_theory,
                    diagnosis_analysis=diagnosis_analysis,
                    search_hypothesis=search_hypothesis,
                    current_spec=current_spec,
                    proposal_example_bank=proposal_example_bank,
                    proposal_example_cases=train_cases,
                    evaluated_patch_hashes=evaluated_patch_hashes,
                    proposals_log=proposals_log,
                    decision_log=decision_log,
                    iteration=iteration,
                    parent_index=parent_index,
                    parent_summaries=parent_summaries,
                    proposal_budget=proposal_budget,
                    dev_evaluations_used=dev_evaluations,
                    experiment_intents=experiment_intents,
                    affordances=affordances,
                    evidence_ledger=evidence_ledger,
                )
                dev_evaluations += evaluations_used
                parent_evaluations_used = evaluations_used
                if evaluations_used == 0 and not accepted_rows:
                    consecutive_zero_eval_parent_attempts += 1
                else:
                    consecutive_zero_eval_parent_attempts = 0
                if (
                    consecutive_zero_eval_parent_attempts >= MAX_CONSECUTIVE_ZERO_EVAL_PARENT_ATTEMPTS
                    and accepted_dev_patches
                ):
                    reason = "repeated proposal rounds produced no valid evaluable candidates"
                    decision_log.append(
                        {
                            "type": "search_stopped",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "patch_hash": current_dev.patch_hash,
                            "reason": reason,
                            "consecutive_zero_eval_parent_attempts": consecutive_zero_eval_parent_attempts,
                        }
                    )
                    self._emit_progress(
                        "search_stopped",
                        iteration=iteration,
                        parent_rank=parent_index + 1,
                        reason=reason,
                        consecutive_zero_eval_parent_attempts=consecutive_zero_eval_parent_attempts,
                    )
                    search_complete = True
                    break
                if not accepted_rows and evaluations_used > 0 and dev_evaluations < self.dev_budget:
                    self._emit_progress(
                        "retry_started",
                        iteration=iteration,
                        parent_rank=parent_index + 1,
                        reason="no_accepted_candidates_from_parent",
                        dev_evaluations=dev_evaluations,
                        dev_budget=self.dev_budget,
                    )
                    retry_search_hypothesis = build_search_hypothesis(
                        summary=current_dev,
                        surface=surface,
                        objective=self.objective,
                        history=proposals_log,
                        parent_patch_hash=current_dev.patch_hash,
                        diagnoses=diagnoses,
                        proposal_example_count=len(proposal_example_bank.examples),
                    )
                    decision_log.append(
                        {
                            "type": "search_hypothesis",
                            "iteration": iteration,
                            "attempt": 2,
                            "proposal_retry": True,
                            "retry_reason": "no_accepted_candidates_from_parent",
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "patch_hash": current_dev.patch_hash,
                            "search_hypothesis": retry_search_hypothesis.to_dict(),
                        }
                    )
                    self._emit_progress(
                        "search_hypothesis_ready",
                        iteration=iteration,
                        parent_rank=parent_index + 1,
                        proposal_retry=True,
                        active_families=retry_search_hypothesis.active_families,
                        active_context_count=len(retry_search_hypothesis.active_contexts),
                    )
                    retry_affordances = generate_optimization_affordances(
                        surface,
                        objective=self.objective,
                        active_families=retry_search_hypothesis.active_families,
                        evidence=_affordance_evidence_from_theory(task_theory, diagnoses),
                    )
                    retry_experiment_intents = self._plan_parent_research_action(
                        current_dev=current_dev,
                        task_theory=task_theory,
                        search_hypothesis=retry_search_hypothesis,
                        affordances=retry_affordances,
                        proposals_log=proposals_log,
                        decision_log=decision_log,
                        iteration=iteration,
                        parent_index=parent_index,
                        proposal_budget=min(PROPOSAL_RETRY_BUDGET, self.dev_budget - dev_evaluations),
                        dev_evaluations_used=dev_evaluations,
                        proposal_retry=True,
                    )
                    if not retry_experiment_intents:
                        retry_rows, retry_evaluations_used = [], 0
                    else:
                        retry_rows, retry_evaluations_used = self._propose_and_evaluate_parent(
                            current_dev=current_dev,
                            baseline_dev=baseline_dev,
                            dev_cases=dev_cases,
                            surface=surface,
                            diagnoses=diagnoses,
                            task_theory=task_theory,
                            diagnosis_analysis=diagnosis_analysis,
                            search_hypothesis=retry_search_hypothesis,
                            current_spec=current_spec,
                            proposal_example_bank=proposal_example_bank,
                            proposal_example_cases=train_cases,
                            evaluated_patch_hashes=evaluated_patch_hashes,
                            proposals_log=proposals_log,
                            decision_log=decision_log,
                            iteration=iteration,
                            parent_index=parent_index,
                            parent_summaries=parent_summaries,
                            proposal_budget=min(PROPOSAL_RETRY_BUDGET, self.dev_budget - dev_evaluations),
                            dev_evaluations_used=dev_evaluations,
                            experiment_intents=retry_experiment_intents,
                            affordances=retry_affordances,
                            evidence_ledger=evidence_ledger,
                            proposal_retry=True,
                            retry_reason="no_accepted_candidates_from_parent",
                        )
                    dev_evaluations += retry_evaluations_used
                    parent_evaluations_used += retry_evaluations_used
                    accepted_rows.extend(retry_rows)

                accepted_rows.sort(key=lambda item: objective_sort_key(item[1], self.objective))
                for _, accepted_summary, _ in accepted_rows:
                    if accepted_summary.patch_hash not in accepted_dev_hashes:
                        accepted_dev_hashes.add(accepted_summary.patch_hash)
                        accepted_dev_patches.append(accepted_summary)
                    parent_pool_by_hash.setdefault(accepted_summary.patch_hash, accepted_summary)
                    frontier_states.setdefault(accepted_summary.patch_hash, FrontierParentState())
                    next_frontier_by_hash.setdefault(accepted_summary.patch_hash, accepted_summary)
                if accepted_rows:
                    parent_state.consecutive_stalls = 0
                    parent_state.accepted_child_count += len(accepted_rows)
                else:
                    parent_state.consecutive_stalls += 1
                    if parent_evaluations_used == 0 or parent_state.consecutive_stalls >= FRONTIER_PARENT_STALL_LIMIT:
                        parent_state.exhausted = True
                if accepted_rows:
                    chosen_proposal, chosen_dev, _ = accepted_rows[0]
                    decision_log.append(
                        {
                            "type": "accepted_proposal",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_patch_hash": current_dev.patch_hash,
                            "proposal_patch_hash": patch_hash(chosen_proposal.patch),
                            "transform_family": chosen_proposal.transform_family,
                            "transform_context": TransformContextKey.from_candidate(chosen_proposal).to_dict(),
                            "patch_hash": chosen_dev.patch_hash,
                            "metrics": chosen_dev.to_dict(),
                        }
                    )

            if search_complete:
                break
            if not next_frontier_by_hash and not _has_selectable_frontier_parent(frontier_states):
                break
            frontier = _select_frontier_parents(
                parent_pool_by_hash.values(),
                frontier_states=frontier_states,
                objective=self.objective,
                width=SEARCH_FRONTIER_WIDTH,
            )
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
                    "parent_pool_patch_hashes": [
                        summary.patch_hash
                        for summary in sorted(
                            parent_pool_by_hash.values(),
                            key=lambda summary: objective_sort_key(summary, self.objective),
                        )
                    ],
                    "frontier_parent_states": {
                        patch_hash_value: _frontier_state_dict(state)
                        for patch_hash_value, state in sorted(frontier_states.items())
                    },
                }
            )
            self._emit_progress(
                "frontier_updated",
                iteration=iteration,
                frontier_patch_hashes=[summary.patch_hash for summary in frontier],
                accepted_count=len(next_frontier_by_hash),
                selectable_parent_count=sum(1 for state in frontier_states.values() if not state.exhausted),
            )

        best_dev_patch = min(
            [baseline_dev, *accepted_dev_patches],
            key=lambda summary: objective_sort_key(summary, self.objective),
        )
        finalist_dev_patches = sorted(
            accepted_dev_patches,
            key=lambda summary: objective_sort_key(summary, self.objective),
        )[: self.holdout_budget]
        simplification_results: list[dict[str, Any]] = []
        if finalist_dev_patches:
            finalist_dev_patches = self._augment_finalists_with_simplifications(
                finalist_dev_patches=finalist_dev_patches,
                accepted_dev_patches=accepted_dev_patches,
                accepted_dev_hashes=accepted_dev_hashes,
                evaluated_patch_hashes=evaluated_patch_hashes,
                baseline_dev=baseline_dev,
                dev_cases=dev_cases,
                decision_log=decision_log,
                simplification_results=simplification_results,
            )

        holdout_patches: list[PatchSummary] = []
        finalist_statuses: list[dict[str, Any]] = []
        runtime_diagnostics: list[dict[str, Any]] = []
        confirmation_results: list[dict[str, Any]] = []
        promotable: list[tuple[PatchSummary, Comparison]] = []
        holdout_ready: list[tuple[PatchSummary, dict[str, Any]]] = []
        if self.holdout_budget <= 0 and accepted_dev_patches:
            decision_log.append(
                {
                    "type": "holdout_validation_skipped",
                    "reason": "holdout_budget validation budget exhausted",
                    "holdout_budget": self.holdout_budget,
                }
            )
            self._emit_progress(
                "holdout_validation_skipped",
                reason="holdout_budget validation budget exhausted",
                finalist_count=len(accepted_dev_patches),
            )
        for dev_summary in finalist_dev_patches:
            holdout_measurement_cost = dev_summary.mean_cost_usd * len(holdout_cases) * self.samples_per_case
            holdout_measurement_tool_calls = dev_summary.mean_tool_calls * len(holdout_cases) * self.samples_per_case
            holdout_measurement_turns = dev_summary.mean_turns * len(holdout_cases) * self.samples_per_case
            budget_reason = _measurement_budget_reason(
                used_usd=self._holdout_measurement_cost_usd,
                marginal_usd=holdout_measurement_cost,
                max_usd=self.max_holdout_measurement_cost_usd,
                used_tool_calls=self._holdout_measurement_tool_calls,
                marginal_tool_calls=holdout_measurement_tool_calls,
                max_tool_calls=self.max_holdout_measurement_tool_calls,
                used_turns=self._holdout_measurement_turns,
                marginal_turns=holdout_measurement_turns,
                max_turns=self.max_holdout_measurement_turns,
                stage="holdout",
            )
            if budget_reason is not None:
                reason = (
                    "measurement_budget_exhausted: "
                    f"{budget_reason}"
                )
                finalist_statuses.append(
                    {
                        "patch_hash": dev_summary.patch_hash,
                        "status": "deferred",
                        "stage": "holdout_skipped",
                        "reason": reason,
                        "dev_transform_families": _transform_lineage_families(dev_summary.patch_hash, proposals_log),
                        "dev_metrics": dev_summary.to_dict(),
                        "measurement_budget": {
                            "marginal_measurement_cost_usd": holdout_measurement_cost,
                            "marginal_measurement_tool_calls": holdout_measurement_tool_calls,
                            "marginal_measurement_turns": holdout_measurement_turns,
                            "measurement_cost_used_usd": self._holdout_measurement_cost_usd,
                            "max_measurement_cost_usd": self.max_holdout_measurement_cost_usd,
                            "measurement_tool_calls_used": self._holdout_measurement_tool_calls,
                            "max_measurement_tool_calls": self.max_holdout_measurement_tool_calls,
                            "measurement_turns_used": self._holdout_measurement_turns,
                            "max_measurement_turns": self.max_holdout_measurement_turns,
                        },
                        "passed_final_gate": False,
                    }
                )
                decision_log.append(
                    {
                        "type": "holdout_validation_skipped",
                        "patch_hash": dev_summary.patch_hash,
                        "reason": reason,
                        "dev_metrics": dev_summary.to_dict(),
                    }
                )
                self._emit_progress(
                    "holdout_validation_skipped",
                    patch_hash=dev_summary.patch_hash,
                    reason=reason,
                )
                continue
            self._holdout_measurement_cost_usd += holdout_measurement_cost
            self._holdout_measurement_tool_calls += holdout_measurement_tool_calls
            self._holdout_measurement_turns += holdout_measurement_turns
            runtime_diagnostic = runtime_reliability_diagnostics(baseline_dev, dev_summary)
            runtime_diagnostics.append(runtime_diagnostic)
            if _requires_finalist_confirmation(dev_summary.patch, runtime_diagnostic):
                confirmation_cases = confirmation_case_subset(baseline_dev, dev_summary, dev_cases)
                self._emit_progress(
                    "confirmation_started",
                    patch_hash=dev_summary.patch_hash,
                    case_count=len(confirmation_cases),
                    sample_count=FINALIST_CONFIRMATION_SAMPLES,
                    reason=runtime_diagnostic.get("reason"),
                )
                sample_start = 1000 + len(confirmation_results) * 100
                sample_indices = tuple(range(sample_start, sample_start + FINALIST_CONFIRMATION_SAMPLES))
                confirmation_summaries = self.evaluate_patches(
                    [baseline_patch, dev_summary.patch],
                    confirmation_cases,
                    sample_indices=sample_indices,
                )
                confirmation_baseline = confirmation_summaries[patch_hash(baseline_patch)]
                confirmation_candidate = confirmation_summaries[dev_summary.patch_hash]
                confirmation = confirmation_result(
                    reference=baseline_dev,
                    candidate=dev_summary,
                    confirmation_reference=confirmation_baseline,
                    confirmation_candidate=confirmation_candidate,
                    objective=self.objective,
                )
                confirmation_results.append(confirmation)
                decision_log.append(
                    {
                        "type": "finalist_confirmation",
                        "patch_hash": dev_summary.patch_hash,
                        "runtime_reliability_diagnostics": runtime_diagnostic,
                        "confirmation": confirmation,
                    }
                )
                self._emit_progress(
                    "confirmation_completed",
                    patch_hash=dev_summary.patch_hash,
                    passed=confirmation.get("passed"),
                    stability_status=confirmation.get("status"),
                    reason=confirmation.get("reason"),
                )
                if not confirmation.get("passed"):
                    confirmation_status = str(confirmation.get("status") or "failed")
                    finalist_status = "unstable" if confirmation_status == "runtime_instability" else "failed"
                    finalist_statuses.append(
                        {
                            "patch_hash": dev_summary.patch_hash,
                            "status": finalist_status,
                            "stage": "confirmation",
                            "reason": confirmation.get("reason"),
                            "dev_transform_families": _transform_lineage_families(dev_summary.patch_hash, proposals_log),
                            "dev_metrics": dev_summary.to_dict(),
                            "runtime_reliability_diagnostics": runtime_diagnostic,
                            "confirmation": confirmation,
                            "passed_final_gate": False,
                        }
                    )
                    continue
            else:
                decision_log.append(
                    {
                        "type": "finalist_confirmation_skipped",
                        "patch_hash": dev_summary.patch_hash,
                        "reason": "no runtime/output reliability suspicion; holdout is the validation gate",
                        "runtime_reliability_diagnostics": runtime_diagnostic,
                    }
                )
                self._emit_progress(
                    "confirmation_skipped",
                    patch_hash=dev_summary.patch_hash,
                    reason="no runtime/output reliability suspicion",
                )
            self._emit_progress(
                "holdout_candidate_started",
                patch_hash=dev_summary.patch_hash,
                case_count=len(holdout_cases),
            )
            holdout_ready.append((dev_summary, runtime_diagnostic))
        if holdout_ready:
            holdout_summaries = self.evaluate_patches(
                [dev_summary.patch for dev_summary, _ in holdout_ready],
                holdout_cases,
            )
        else:
            holdout_summaries = {}
        for dev_summary, runtime_diagnostic in holdout_ready:
            holdout_summary = holdout_summaries[dev_summary.patch_hash]
            holdout_patches.append(holdout_summary)
            gate = final_gate_status(
                baseline_holdout,
                holdout_summary,
                self.objective,
            )
            comparison = gate.comparison
            passed_gate = gate.validated
            flip_summary = behavior_flip_summary(baseline_holdout, holdout_summary)
            finalist_status = {
                "patch_hash": holdout_summary.patch_hash,
                "status": gate.status,
                "stage": "holdout",
                "reason": gate.reason,
                "dev_transform_families": _transform_lineage_families(dev_summary.patch_hash, proposals_log),
                "comparison_to_baseline": comparison.to_dict(),
                "behavior_flip_summary": flip_summary,
                "passed_final_gate": passed_gate,
                "dev_metrics": dev_summary.to_dict(),
                "holdout_metrics": holdout_summary.to_dict(),
                "runtime_reliability_diagnostics": runtime_diagnostic,
            }
            finalist_statuses.append(finalist_status)
            decision_log.append(
                {
                    "type": "holdout_validation",
                    "patch_hash": holdout_summary.patch_hash,
                    "metrics": holdout_summary.to_dict(),
                    "comparison_to_baseline": comparison.to_dict(),
                    "behavior_flip_summary": flip_summary,
                    "finalist_status": gate.status,
                    "final_gate": gate.to_dict(),
                    "passed_final_gate": passed_gate,
                    "rejection_reason": gate.reason,
                }
            )
            self._emit_progress(
                "holdout_candidate_completed",
                passed_final_gate=passed_gate,
                finalist_status=gate.status,
                rejection_reason=gate.reason,
                **_summary_progress_fields(holdout_summary),
            )
            if passed_gate:
                promotable.append((holdout_summary, comparison))

        if promotable:
            promotable_summaries = [summary for summary, _ in promotable]
            selected_holdout, frontier_recommendation = select_recommended_patch(
                promotable_summaries,
                self.objective,
            )
            promoted = True
            selection_reason = str(frontier_recommendation.get("reason") or f"Promoted validated patch for {self.objective.mode} objective.")
        else:
            selected_holdout = baseline_holdout
            promoted = False
            selection_reason = "No finalist cleared the holdout objective gate; kept original baseline."
            frontier_recommendation = {
                "recommended_patch_hash": selected_holdout.patch_hash,
                "highest_quality_patch_hash": selected_holdout.patch_hash,
                "reason": selection_reason,
                "validated_candidate_count": 0,
            }

        selected_patch = selected_holdout.patch
        selected_patch_hash = selected_holdout.patch_hash
        decision_log.append(
            {
                "type": "final_selection",
                "selected_patch_hash": selected_patch_hash,
                "promoted": promoted,
                "reason": selection_reason,
                "best_dev_patch_hash": best_dev_patch.patch_hash,
                "frontier_recommendation": frontier_recommendation,
            }
        )
        self._emit_progress(
            "run_completed",
            selected_patch_hash=selected_patch_hash,
            promoted=promoted,
            accepted_dev_patches=len(accepted_dev_patches),
            holdout_validations=len(holdout_patches),
            selection_reason=selection_reason,
        )

        transform_summaries = summarize_transform_results(proposals_log)
        transform_context_summaries = summarize_transform_context_results(proposals_log)
        affordance_summaries = summarize_affordance_results(proposals_log)
        transform_final_statuses = _transform_final_status_summaries(finalist_statuses)
        cost_tradeoffs = quality_cost_tradeoffs(proposals_log)
        ideation_metrics = build_ideation_metrics(
            decision_log=decision_log,
            proposals=proposals_log,
            finalist_statuses=finalist_statuses,
        )
        outcome_analysis = build_outcome_analysis(
            objective=self.objective,
            promoted=promoted,
            baseline_dev=baseline_dev,
            accepted_dev_patches=accepted_dev_patches,
            holdout_patches=holdout_patches,
            decision_log=decision_log,
            finalist_statuses=finalist_statuses,
        )
        manifest = self.build_manifest(
            total_cases=len(cases),
            train_case_count=len(train_cases),
            proposal_example_bank=proposal_example_bank,
            selected_patch_hash=selected_patch_hash,
            promoted=promoted,
            generated_surface=generated_surface_rows,
            task_theories=task_theory_log,
            transform_summaries=transform_summaries,
            transform_context_summaries=transform_context_summaries,
            affordance_summaries=affordance_summaries,
            transform_final_statuses=transform_final_statuses,
            finalist_statuses=finalist_statuses,
            runtime_reliability_diagnostics=runtime_diagnostics,
            confirmation_results=confirmation_results,
            simplification_results=simplification_results,
            frontier_recommendation=frontier_recommendation,
            optimizer_call_diagnostics=self.optimizer_call_diagnostics,
            quality_cost_tradeoffs=cost_tradeoffs,
            measurement_decisions=[
                row
                for row in decision_log
                if row.get("type") in {"research_plan", "measurement_decision"}
            ],
            ideation_metrics=ideation_metrics,
            evidence_ledger=evidence_ledger.to_dict(),
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
            task_theories=task_theory_log,
            transform_summaries=transform_summaries,
            transform_context_summaries=transform_context_summaries,
            affordance_summaries=affordance_summaries,
            finalist_statuses=finalist_statuses,
            runtime_reliability_diagnostics=runtime_diagnostics,
            confirmation_results=confirmation_results,
            simplification_results=simplification_results,
            frontier_recommendation=frontier_recommendation,
            run_profile={},
            quality_cost_tradeoffs=cost_tradeoffs,
            optimizer_call_diagnostics=self.optimizer_call_diagnostics,
            ideation_metrics=ideation_metrics,
            evidence_ledger=evidence_ledger.to_dict(),
            selection_reason=selection_reason,
            outcome_analysis=outcome_analysis,
            manifest=manifest,
        )
        result.run_profile.update(build_run_profile(result, self.out_dir))
        result.manifest["run_profile"] = result.run_profile
        result.manifest["run_cost"] = result.run_profile.get("run_cost", {})
        self.write_outputs(result)
        return result

    def _augment_finalists_with_simplifications(
        self,
        *,
        finalist_dev_patches: list[PatchSummary],
        accepted_dev_patches: list[PatchSummary],
        accepted_dev_hashes: set[str],
        evaluated_patch_hashes: set[str],
        baseline_dev: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
        decision_log: list[dict[str, Any]],
        simplification_results: list[dict[str, Any]],
    ) -> list[PatchSummary]:
        known_by_hash = {summary.patch_hash: summary for summary in [baseline_dev, *accepted_dev_patches]}
        augmented_by_hash = {summary.patch_hash: summary for summary in finalist_dev_patches}
        simplification_parents = list(finalist_dev_patches)[:MAX_SIMPLIFICATION_PARENT_COUNT]
        for skipped_parent in list(finalist_dev_patches)[MAX_SIMPLIFICATION_PARENT_COUNT:]:
            row = {
                "type": "simplification_skipped",
                "parent_patch_hash": skipped_parent.patch_hash,
                "reason": "simplification parent cap reached",
                "max_simplification_parent_count": MAX_SIMPLIFICATION_PARENT_COUNT,
            }
            simplification_results.append(row)
            decision_log.append(row)
        simplification_full_dev_count = 0
        for parent_index, parent in enumerate(simplification_parents, start=1):
            parent_full_dev_count = 0
            candidate_variants = [
                variant
                for variant in _simplification_variants(parent.patch)[:MAX_SIMPLIFICATION_VARIANTS_PER_FINALIST]
                if patch_hash(variant) != parent.patch_hash
            ]
            selected_variant_hashes = {patch_hash(variant) for variant in candidate_variants}
            skipped_variant_reasons: dict[str, str] = {}
            for variant in candidate_variants:
                digest = patch_hash(variant)
                if digest not in selected_variant_hashes:
                    row = {
                        "type": "simplification_skipped",
                        "parent_patch_hash": parent.patch_hash,
                        "patch_hash": digest,
                        "patch": variant.to_dict(),
                        "simplification": variant.metadata.get("simplification"),
                        "reason": skipped_variant_reasons.get(
                            digest,
                            "research controller did not select simplification variant",
                        ),
                    }
                    simplification_results.append(row)
                    decision_log.append(row)
                    continue
                summary = known_by_hash.get(digest)
                reused = summary is not None
                if summary is None:
                    if digest in evaluated_patch_hashes:
                        continue
                    self._emit_progress(
                        "simplification_started",
                        parent_patch_hash=parent.patch_hash,
                        patch_hash=digest,
                        simplification=variant.metadata.get("simplification"),
                    )
                    summary, rejection_reason, stage_rows = self._evaluate_simplification_variant(
                        patch=variant,
                        parent=parent,
                        baseline=baseline_dev,
                        dev_cases=dev_cases,
                        allow_full_dev=(
                            parent_full_dev_count < MAX_SIMPLIFICATION_FULL_DEV_PER_PARENT
                            and simplification_full_dev_count < MAX_SIMPLIFICATION_FULL_DEV_PER_RUN
                        ),
                    )
                    reached_full_dev = any(row.get("stage") == "full_dev" for row in stage_rows)
                    if reached_full_dev:
                        parent_full_dev_count += 1
                        simplification_full_dev_count += 1
                    evaluated_patch_hashes.add(digest)
                    known_by_hash[digest] = summary
                else:
                    rejection_reason = None
                    stage_rows = []
                summary_cases = tuple(evaluation.case for evaluation in summary.evaluations)
                comparable_baseline = _summary_for_cases(baseline_dev, summary_cases) or baseline_dev
                comparable_parent = _summary_for_cases(parent, summary_cases) or parent
                comparison_to_baseline = compare_summaries(comparable_baseline, summary)
                comparison_to_parent = compare_summaries(comparable_parent, summary)
                if rejection_reason is None:
                    rejection_reason = patch_rejection_reason(
                        baseline=baseline_dev,
                        reference=baseline_dev,
                        patch_summary=summary,
                        objective=self.objective,
                    )
                accepted = rejection_reason is None
                row = {
                    "type": "simplification_evaluation",
                    "parent_patch_hash": parent.patch_hash,
                    "patch_hash": digest,
                    "patch": variant.to_dict(),
                    "simplification": variant.metadata.get("simplification"),
                    "reused_existing_summary": reused,
                    "accepted": accepted,
                    "rejection_reason": rejection_reason,
                    "evaluation_stages": stage_rows,
                    "metrics": summary.to_dict(),
                    "comparison_to_baseline": comparison_to_baseline.to_dict(),
                    "comparison_to_parent": comparison_to_parent.to_dict(),
                }
                simplification_results.append(row)
                decision_log.append(row)
                self._emit_progress(
                    "simplification_completed",
                    parent_patch_hash=parent.patch_hash,
                    variant_patch_hash=digest,
                    accepted=accepted,
                    rejection_reason=rejection_reason,
                    **_summary_progress_fields(summary),
                )
                if accepted:
                    augmented_by_hash.setdefault(digest, summary)
                    if digest not in accepted_dev_hashes:
                        accepted_dev_hashes.add(digest)
                        accepted_dev_patches.append(summary)
        return sorted(
            augmented_by_hash.values(),
            key=lambda summary: objective_sort_key(summary, self.objective),
        )[: self.holdout_budget]

    def _evaluate_simplification_variant(
        self,
        *,
        patch: AgentPatch,
        parent: PatchSummary,
        baseline: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
        allow_full_dev: bool = True,
    ) -> tuple[PatchSummary, str | None, list[dict[str, Any]]]:
        stage_rows: list[dict[str, Any]] = []
        latest_summary: PatchSummary | None = None
        latest_rejection: str | None = None
        for stage_name, stage_cases in self._progressive_eval_stages(parent, dev_cases):
            if stage_name == "full_dev" and not allow_full_dev and latest_summary is not None:
                return (
                    latest_summary,
                    (
                        "simplification_full_dev_cap: screened out before full_dev "
                        f"(max {MAX_SIMPLIFICATION_FULL_DEV_PER_PARENT} per parent, "
                        f"{MAX_SIMPLIFICATION_FULL_DEV_PER_RUN} per run)"
                    ),
                    stage_rows,
                )
            parent_stage = _summary_for_cases(parent, stage_cases) or self.evaluate_patch(parent.patch, stage_cases)
            baseline_stage = _summary_for_cases(baseline, stage_cases) or self.evaluate_patch(baseline.patch, stage_cases)
            candidate_stage = self.evaluate_patch(patch, stage_cases)
            latest_summary = candidate_stage
            comparison_to_parent = compare_summaries(parent_stage, candidate_stage)
            comparison_to_baseline = compare_summaries(baseline_stage, candidate_stage)
            flip_summary = behavior_flip_summary(baseline_stage, candidate_stage)
            if stage_name == "smoke":
                rejection = _smoke_rejection_reason(parent_stage, candidate_stage)
            else:
                rejection = patch_rejection_reason(
                    baseline=baseline_stage,
                    reference=baseline_stage,
                    patch_summary=candidate_stage,
                    objective=self.objective,
                )
            latest_rejection = rejection
            stage_rows.append(
                {
                    "stage": stage_name,
                    "case_ids": [case.id for case in stage_cases],
                    "case_count": len(stage_cases),
                    "patch_hash": candidate_stage.patch_hash,
                    "metrics": candidate_stage.to_dict(),
                    "comparison_to_parent": comparison_to_parent.to_dict(),
                    "comparison_to_baseline": comparison_to_baseline.to_dict(),
                    "behavior_flip_summary": flip_summary,
                    "rejection_reason": rejection,
                    "passed": rejection is None,
                }
            )
            if rejection is not None:
                return candidate_stage, f"simplification {stage_name} gate rejected variant: {rejection}", stage_rows
            if stage_name == "small_dev" and candidate_stage.pass_count < parent_stage.pass_count:
                return (
                    candidate_stage,
                    (
                        "simplification small_dev gate rejected variant: "
                        f"pass count regressed versus parent ({candidate_stage.pass_count} < {parent_stage.pass_count})"
                    ),
                    stage_rows,
                )
            if stage_name == "full_dev":
                return candidate_stage, None, stage_rows
        if latest_summary is None:
            latest_summary = self.evaluate_patch(patch, dev_cases)
        return latest_summary, latest_rejection, stage_rows

    def _propose_and_evaluate_parent(
        self,
        *,
        current_dev: PatchSummary,
        baseline_dev: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
        surface: list[EditableTarget],
        diagnoses: list[FailureDiagnosis],
        task_theory: TaskTheory,
        diagnosis_analysis: str,
        search_hypothesis: Any,
        current_spec: AgentSpec | None,
        proposal_example_bank: ProposalExampleBank,
        proposal_example_cases: tuple[EvalCase, ...],
        evaluated_patch_hashes: set[str],
        proposals_log: list[dict[str, Any]],
        decision_log: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        parent_summaries: list[PatchSummary],
        proposal_budget: int,
        dev_evaluations_used: int,
        evidence_ledger: EvidenceLedger,
        experiment_intents: list[Any] | None = None,
        affordances: list[OptimizationAffordance] | None = None,
        proposal_retry: bool = False,
        retry_reason: str | None = None,
    ) -> tuple[list[tuple[CandidateProposal, PatchSummary, Comparison]], int]:
        if proposal_budget <= 0:
            return [], 0
        target_diagnosis = diagnoses[0] if diagnoses else None
        attempt = 2 if proposal_retry else 1
        self._emit_progress(
            "proposal_started",
            iteration=iteration,
            attempt=attempt,
            proposal_retry=proposal_retry,
            parent_rank=parent_index + 1,
            proposal_budget=proposal_budget,
            active_families=search_hypothesis.active_families,
        )
        proposals, proposal_analysis = self.candidate_implementer.propose(
            current_dev,
            surface,
            objective=self.objective,
            diagnosis=target_diagnosis,
            diagnoses=diagnoses,
            task_theory=task_theory,
            seen_hashes=evaluated_patch_hashes,
            current_spec=current_spec,
            history=proposals_log,
            search_hypothesis=search_hypothesis,
            proposal_example_bank=proposal_example_bank,
            proposal_example_cases=proposal_example_cases,
            proposal_budget=proposal_budget,
            experiment_intents=experiment_intents or [],
            affordances=affordances or [],
        )
        if self.candidate_implementer.last_call_diagnostics is not None:
            self.optimizer_call_diagnostics.append(
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "proposal_retry": proposal_retry,
                    "parent_rank": parent_index + 1,
                    "parent_patch_hash": current_dev.patch_hash,
                    **self.candidate_implementer.last_call_diagnostics,
                }
            )
        self._emit_progress(
            "proposal_completed",
            iteration=iteration,
            attempt=attempt,
            proposal_retry=proposal_retry,
            parent_rank=parent_index + 1,
            raw_count=self.candidate_implementer.last_stats.raw_count,
            valid_count=self.candidate_implementer.last_stats.valid_count,
            returned_count=self.candidate_implementer.last_stats.returned_count,
            invalid_count=self.candidate_implementer.last_stats.invalid_count,
            duplicate_count=self.candidate_implementer.last_stats.duplicate_count,
            call_diagnostics=self.candidate_implementer.last_call_diagnostics or {},
        )
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
                "proposal_stats": self.candidate_implementer.last_stats.to_dict(),
                "search_hypothesis": search_hypothesis.to_dict(),
                "diagnoses": [diagnosis.to_dict() for diagnosis in diagnoses],
                "diagnosis": target_diagnosis.to_dict() if target_diagnosis else None,
                "proposal_hashes": [patch_hash(proposal.patch) for proposal in proposals],
                "candidate_proposals": self.candidate_implementer.last_candidate_rows,
                "invalid_candidate_proposals": self.candidate_implementer.last_invalid_candidate_rows,
            }
        )
        for invalid_row in self.candidate_implementer.last_invalid_candidate_rows:
            proposal_row = {
                "type": "candidate_proposal",
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "retry_reason": retry_reason,
                "parent_rank": parent_index + 1,
                "parent_patch_hash": current_dev.patch_hash,
                "patch_hash": current_dev.patch_hash,
                "valid": False,
                **invalid_row,
            }
            proposals_log.append(proposal_row)
        if not proposals:
            return [], 0

        materialization_by_proposal_hash = {
            str(row.get("proposal_patch_hash")): dict(row.get("materialization") or {})
            for row in self.candidate_implementer.last_candidate_rows
            if row.get("proposal_patch_hash")
        }
        accepted_rows: list[tuple[CandidateProposal, PatchSummary, Comparison]] = []
        evaluation_states: list[CandidateEvaluationState] = []
        for candidate in proposals[:proposal_budget]:
            patch = compose_patches(current_dev.patch, candidate.patch)
            digest = patch_hash(patch)
            if digest in evaluated_patch_hashes:
                continue
            transform_context = TransformContextKey.from_candidate(candidate)
            self._emit_progress(
                "candidate_evaluation_started",
                iteration=iteration,
                attempt=attempt,
                parent_rank=parent_index + 1,
                transform_family=candidate.transform_family,
                transform_context=transform_context.to_dict(),
                patch_hash=digest,
                proposal_patch_hash=patch_hash(candidate.patch),
            )
            evaluation_states.append(
                CandidateEvaluationState(
                    candidate=candidate,
                    patch=patch,
                    patch_hash=digest,
                    proposal_patch_hash=patch_hash(candidate.patch),
                    transform_context=transform_context,
                )
            )
        if not evaluation_states:
            return [], 0

        self._evaluate_candidate_batch_progressively(
            states=evaluation_states,
            reference=current_dev,
            baseline=baseline_dev,
            dev_cases=dev_cases,
            proposals_log=proposals_log,
            decision_log=decision_log,
            dev_evaluations_used=dev_evaluations_used,
            evidence_ledger=evidence_ledger,
            iteration=iteration,
            attempt=attempt,
            parent_index=parent_index,
        )
        evaluations_used = len(evaluation_states)
        for state in evaluation_states:
            candidate = state.candidate
            summary = state.summary
            comparison = state.comparison
            flip_summary = state.flip_summary
            if summary is None or comparison is None or flip_summary is None:
                continue
            evaluated_patch_hashes.add(state.patch_hash)
            proposal_row = {
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "retry_reason": retry_reason,
                "parent_rank": parent_index + 1,
                "parent_patch_hash": current_dev.patch_hash,
                "proposal_patch_hash": state.proposal_patch_hash,
                "proposal": candidate.patch.to_dict(),
                "candidate": candidate.to_dict(),
                "materialization": materialization_by_proposal_hash.get(state.proposal_patch_hash, {}),
                "applications": [application.to_dict() for application in candidate.applications],
                "affordance_ids": list(candidate.affordance_ids),
                "transform_family": candidate.transform_family,
                "mechanism_class": candidate.mechanism_class,
                "experiment_id": candidate.experiment_id,
                "candidate_role": candidate.candidate_role,
                "comparison_group": candidate.comparison_group,
                "transform_instance": candidate.transform_instance,
                "transform_parameters": candidate.transform_parameters,
                "transform_context": state.transform_context.to_dict(),
                "target_slice": candidate.target_slice,
                "hypothesis": candidate.hypothesis,
                "expected_effects": candidate.expected_effects,
                "evaluation_plan": candidate.evaluation_plan,
                "evaluation_stages": state.stage_rows,
                "evidence_summary": (
                    evidence_ledger.latest(state.patch_hash).to_dict()
                    if evidence_ledger.latest(state.patch_hash)
                    else {}
                ),
                "evidence_history": [
                    record.to_dict() for record in evidence_ledger.by_candidate(state.patch_hash)
                ],
                "patch_hash": state.patch_hash,
                "patch": state.patch.to_dict(),
                "comparison_to_parent": comparison.to_dict(),
                "behavior_flip_summary": flip_summary,
                "metrics": summary.to_dict(),
                "accepted": state.accepted,
                "frontier_status": state.frontier_status,
                "rejection_reason": state.rejection_reason,
                "constraint_warning": state.constraint_warning,
                "full_dev_evaluated": state.full_dev_evaluated,
                "diagnosis_category": candidate.patch.metadata.get("diagnosis_category"),
            }
            proposals_log.append(proposal_row)
            decision_log.append({"type": "proposal_evaluation", **proposal_row})
            decision_log.append(
                observe_transform_result(
                    family=candidate.transform_family,
                    context_key=state.transform_context,
                    accepted=state.accepted,
                    comparison=comparison,
                    rejection_reason=state.rejection_reason,
                )
            )
            self._emit_progress(
                "candidate_evaluated",
                iteration=iteration,
                attempt=attempt,
                parent_rank=parent_index + 1,
                transform_family=candidate.transform_family,
                transform_context=state.transform_context.to_dict(),
                patch_hash=state.patch_hash,
                accepted=state.accepted,
                frontier_status=state.frontier_status,
                rejection_reason=state.rejection_reason,
                constraint_warning=state.constraint_warning,
                score_delta=comparison.score_delta,
                cost_delta=comparison.cost_delta,
                latency_delta=comparison.latency_delta,
                stage_count=len(state.stage_rows),
                full_dev_evaluated=state.full_dev_evaluated,
            )
            if state.accepted:
                accepted_rows.append((candidate, summary, comparison))
                if candidate.mechanism_class in {"runtime_defect_fix", "output_contract_fix"}:
                    decision_log.append(
                        {
                            "type": "residual_rediagnosis_triggered",
                            "patch_hash": state.patch_hash,
                            "parent_patch_hash": current_dev.patch_hash,
                            "mechanism_class": candidate.mechanism_class,
                            "reason": "structural/runtime fix accepted; child branch should be rediagnosed for residual failures",
                        }
                    )
        return accepted_rows, evaluations_used

    def _plan_parent_research_action(
        self,
        *,
        current_dev: PatchSummary,
        task_theory: TaskTheory,
        search_hypothesis: Any,
        affordances: list[OptimizationAffordance],
        proposals_log: list[dict[str, Any]],
        decision_log: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        proposal_budget: int,
        dev_evaluations_used: int,
        proposal_retry: bool = False,
    ) -> list[ExperimentIntent]:
        attempt = 2 if proposal_retry else 1
        task_theory_payload = _task_theory_with_affordance_opportunities(
            task_theory=task_theory,
            affordances=affordances,
            current_dev=current_dev,
            proposals_log=proposals_log,
            objective=self.objective,
        )
        state = ResearchState(
            objective=self.objective.to_dict(),
            budget={
                "proposal_budget": proposal_budget,
                "dev_evaluations_used": dev_evaluations_used,
                "dev_budget": self.dev_budget,
                "remaining_dev_budget": max(0, self.dev_budget - dev_evaluations_used),
            },
            parent={
                "patch_hash": current_dev.patch_hash,
                "score": current_dev.mean_score,
                "pass_count": current_dev.pass_count,
                "case_count": current_dev.case_count,
                "failure_labels": _top_counter_dict(current_dev.failure_labels, limit=8),
                "cost_usd": current_dev.mean_cost_usd,
                "latency_s": current_dev.median_latency_s,
            },
            task_theory=task_theory_payload,
            behavior_profile=search_hypothesis.profile.to_dict(),
            affordances=[affordance.to_dict() for affordance in affordances],
            prior_experiment_outcomes=_compact_prior_stage_results(proposals_log, stage=None, limit=8),
            frontier={
                "active_families": list(search_hypothesis.active_families),
                "active_context_count": len(search_hypothesis.active_contexts),
                "budget_allocation": dict(search_hypothesis.budget_allocation),
            },
        )
        self._emit_progress(
            "research_planner_started",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            stage="plan_experiments",
            opportunity_count=len(task_theory.experiment_opportunities),
            affordance_count=len(affordances),
        )
        intents = self.research_planner.plan(state)
        if self.research_planner.last_call_diagnostics is not None:
            self.optimizer_call_diagnostics.append(
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "parent_rank": parent_index + 1,
                    "stage": "plan_experiments",
                    **self.research_planner.last_call_diagnostics,
                }
            )
        decision_log.append(
            {
                "type": "research_plan",
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "parent_rank": parent_index + 1,
                "stage": "plan_experiments",
                "research_state": state.to_dict(),
                "experiment_intents": [intent.to_dict() for intent in intents],
            }
        )
        self._emit_progress(
            "research_planner_completed",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            stage="plan_experiments",
            intent_count=len(intents),
            mechanisms=[intent.mechanism_class for intent in intents],
            experiment_intents=[intent.to_dict() for intent in intents],
            call_diagnostics=self.research_planner.last_call_diagnostics or {},
        )
        return intents

    def _evaluate_candidate_batch_progressively(
        self,
        *,
        states: list[CandidateEvaluationState],
        reference: PatchSummary,
        baseline: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
        proposals_log: list[dict[str, Any]],
        decision_log: list[dict[str, Any]],
        dev_evaluations_used: int,
        evidence_ledger: EvidenceLedger,
        iteration: int,
        attempt: int,
        parent_index: int,
    ) -> None:
        active = list(states)
        for stage_name, stage_cases in self._progressive_eval_stages(reference, dev_cases):
            if not active:
                break
            if stage_name == "full_dev":
                if _has_evidence_for_selector(evidence_ledger, active):
                    active = self._select_candidate_stage_with_measurement_selector(
                        active,
                        baseline,
                        reference=reference,
                        stage_name=stage_name,
                        stage_cases=stage_cases,
                        proposals_log=proposals_log,
                        decision_log=decision_log,
                        dev_evaluations_used=dev_evaluations_used,
                        evidence_ledger=evidence_ledger,
                        iteration=iteration,
                        attempt=attempt,
                        parent_index=parent_index,
                    )
                    if not active:
                        break
            elif stage_name == "small_dev":
                active = self._select_candidate_stage_with_measurement_selector(
                    active,
                    baseline,
                    reference=reference,
                    stage_name=stage_name,
                    stage_cases=stage_cases,
                    proposals_log=proposals_log,
                    decision_log=decision_log,
                    dev_evaluations_used=dev_evaluations_used,
                    evidence_ledger=evidence_ledger,
                    iteration=iteration,
                    attempt=attempt,
                    parent_index=parent_index,
                )
                if not active:
                    break
            self._emit_progress(
                "candidate_stage_started",
                stage=stage_name,
                candidate_count=len(active),
                case_count=len(stage_cases),
                patch_hashes=[state.patch_hash for state in active],
            )
            reference_summary = _summary_for_cases(reference, stage_cases) or self.evaluate_patch(reference.patch, stage_cases)
            baseline_summary = _summary_for_cases(baseline, stage_cases) or self.evaluate_patch(baseline.patch, stage_cases)
            candidate_summaries = self.evaluate_patches(
                [state.patch for state in active],
                stage_cases,
            )
            next_active: list[CandidateEvaluationState] = []
            for state in active:
                candidate_summary = candidate_summaries[state.patch_hash]
                comparison = compare_summaries(reference_summary, candidate_summary)
                flip_summary = behavior_flip_summary(reference_summary, candidate_summary)
                constraint_warning = None
                if stage_name == "smoke":
                    rejection_reason = _smoke_rejection_reason(reference_summary, candidate_summary)
                else:
                    rejection_reason = objective_rejection_reason(
                        reference_summary,
                        candidate_summary,
                        self.objective,
                    )
                    constraint_warning = constraint_rejection_reason(
                        baseline_summary,
                        candidate_summary,
                        self.objective,
                    )
                evidence_summary = evidence_ledger.add(
                    candidate_id=state.patch_hash,
                    stage=stage_name,
                    reference=reference_summary,
                    baseline=baseline_summary,
                    candidate=candidate_summary,
                    mechanism_class=state.candidate.mechanism_class,
                    affordance_ids=list(state.candidate.affordance_ids),
                    comparison_group=state.candidate.comparison_group,
                    candidate_role=state.candidate.candidate_role,
                    rejection_reason=rejection_reason,
                    constraint_warning=constraint_warning,
                )
                if stage_name in {"smoke", "small_dev", "full_dev"}:
                    self._dev_measurement_cost_usd += float(
                        evidence_summary.measurement_cost.get("estimated_total_cost_usd") or 0.0
                    )
                    self._dev_measurement_tool_calls += float(
                        evidence_summary.measurement_cost.get("estimated_tool_calls") or 0.0
                    )
                    self._dev_measurement_turns += float(
                        evidence_summary.measurement_cost.get("estimated_turns") or 0.0
                    )
                state.stage_rows.append(
                    {
                        "stage": stage_name,
                        "case_ids": [case.id for case in stage_cases],
                        "case_count": len(stage_cases),
                        "patch_hash": candidate_summary.patch_hash,
                        "metrics": candidate_summary.to_dict(),
                        "comparison_to_parent": comparison.to_dict(),
                        "behavior_flip_summary": flip_summary,
                        "rejection_reason": rejection_reason,
                        "constraint_warning": constraint_warning,
                        "passed": rejection_reason is None,
                        "evidence_summary": evidence_summary.to_dict(),
                    }
                )
                state.summary = candidate_summary
                state.comparison = comparison
                state.flip_summary = flip_summary
                state.rejection_reason = rejection_reason
                state.constraint_warning = constraint_warning
                if stage_name == "full_dev":
                    state.full_dev_evaluated = True
                    _finalize_candidate_state(state, reference)
                    continue
                if rejection_reason is None:
                    next_active.append(state)
                else:
                    state.frontier_status = "failed"
                    state.accepted = False
            self._emit_progress(
                "candidate_stage_completed",
                stage=stage_name,
                candidate_count=len(active),
                advanced_count=len(next_active) if stage_name != "full_dev" else 0,
                accepted_count=sum(1 for state in active if state.accepted),
                rejected_count=sum(1 for state in active if state.frontier_status == "failed"),
                screened_count=sum(1 for state in active if state.frontier_status == "screened_out"),
                case_count=len(stage_cases),
            )
            active = next_active
        for state in states:
            if state.summary is None:
                continue
            if state.full_dev_evaluated or state.frontier_status != "pending":
                continue
            state.rejection_reason = "budget_not_worth_information"
            state.frontier_status = "screened_out"
            state.accepted = False

    def _select_candidate_stage_with_measurement_selector(
        self,
        states: list[CandidateEvaluationState],
        baseline: PatchSummary,
        *,
        reference: PatchSummary,
        stage_name: str,
        stage_cases: tuple[EvalCase, ...],
        proposals_log: list[dict[str, Any]],
        decision_log: list[dict[str, Any]],
        dev_evaluations_used: int,
        evidence_ledger: EvidenceLedger,
        iteration: int,
        attempt: int,
        parent_index: int,
    ) -> list[CandidateEvaluationState]:
        action = _measurement_action(
            stage_name=stage_name,
            states=states,
            dev_evaluations_used=dev_evaluations_used,
            dev_budget=self.dev_budget,
        )
        state_packet = _research_state_packet(
            objective=self.objective,
            stage_name=stage_name,
            reference=reference,
            baseline=baseline,
            states=states,
            proposals_log=proposals_log,
            dev_evaluations_used=dev_evaluations_used,
            dev_budget=self.dev_budget,
            evidence_ledger=evidence_ledger,
            stage_cases=stage_cases,
            samples_per_case=self.samples_per_case,
            measurement_cost_used_usd=self._dev_measurement_cost_usd,
            max_measurement_cost_usd=self.max_dev_measurement_cost_usd,
            measurement_tool_calls_used=self._dev_measurement_tool_calls,
            max_measurement_tool_calls=self.max_dev_measurement_tool_calls,
            measurement_turns_used=self._dev_measurement_turns,
            max_measurement_turns=self.max_dev_measurement_turns,
        )
        self._emit_progress(
            "measurement_selector_started",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            stage=stage_name,
            candidate_count=len(states),
            max_select=action.max_select,
        )
        decision = self.measurement_selector.select(
            stage=stage_name,
            state=state_packet,
            candidate_ids=action.candidate_ids,
            max_select=action.max_select,
            max_select_per_group=action.max_select_per_group,
        )
        if self.measurement_selector.last_call_diagnostics is not None:
            self.optimizer_call_diagnostics.append(
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "parent_rank": parent_index + 1,
                    "stage": stage_name,
                    **self.measurement_selector.last_call_diagnostics,
                }
            )
        self._validate_research_candidate_groups(decision, action, states)
        decision_row = {
            "type": "measurement_decision",
            "iteration": iteration,
            "attempt": attempt,
            "parent_rank": parent_index + 1,
            "stage": stage_name,
            "action": action.to_dict(),
            "decision": decision.to_dict(),
        }
        decision_log.append(decision_row)
        self._emit_progress(
            "measurement_selector_completed",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            stage=stage_name,
            selected_candidate_ids=decision.selected_candidate_ids,
            rationale=decision.rationale,
            call_diagnostics=self.measurement_selector.last_call_diagnostics or {},
        )
        selected_ids = set(decision.selected_candidate_ids)
        selected = [state for state in states if state.patch_hash in selected_ids]
        for state in states:
            if state.patch_hash in selected_ids:
                continue
            state.rejection_reason = (
                decision.skipped_candidate_reasons.get(state.patch_hash)
                or f"measurement_selector_skipped_{stage_name}"
            )
            state.frontier_status = "screened_out"
            state.accepted = False
        kept: list[CandidateEvaluationState] = []
        selected_measurement_cost = 0.0
        selected_measurement_tool_calls = 0.0
        selected_measurement_turns = 0.0
        for state in selected:
            marginal_cost = _estimated_marginal_measurement_cost_usd(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
                samples_per_case=self.samples_per_case,
            )
            marginal_tool_calls = _estimated_marginal_measurement_units(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
                samples_per_case=self.samples_per_case,
                unit="tool_calls",
            )
            marginal_turns = _estimated_marginal_measurement_units(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
                samples_per_case=self.samples_per_case,
                unit="turns",
            )
            budget_reason = _measurement_budget_reason(
                used_usd=self._dev_measurement_cost_usd + selected_measurement_cost,
                marginal_usd=marginal_cost,
                max_usd=self.max_dev_measurement_cost_usd,
                used_tool_calls=self._dev_measurement_tool_calls + selected_measurement_tool_calls,
                marginal_tool_calls=marginal_tool_calls,
                max_tool_calls=self.max_dev_measurement_tool_calls,
                used_turns=self._dev_measurement_turns + selected_measurement_turns,
                marginal_turns=marginal_turns,
                max_turns=self.max_dev_measurement_turns,
                stage=stage_name,
            )
            if budget_reason is not None:
                state.rejection_reason = (
                    "measurement_budget_exhausted: "
                    f"{budget_reason}"
                )
                state.frontier_status = "screened_out"
                state.accepted = False
                continue
            selected_measurement_cost += marginal_cost
            selected_measurement_tool_calls += marginal_tool_calls
            selected_measurement_turns += marginal_turns
            kept.append(state)
        return kept

    def _validate_research_candidate_groups(
        self,
        decision: Any,
        action: MeasurementAction,
        states: list[CandidateEvaluationState],
    ) -> None:
        if action.max_select_per_group <= 0:
            return
        group_counts: Counter[str] = Counter()
        state_by_id = {state.patch_hash: state for state in states}
        for candidate_id in decision.selected_candidate_ids:
            state = state_by_id.get(candidate_id)
            if state is None:
                continue
            group_counts[_candidate_research_group(state)] += 1
        over = {
            group: count
            for group, count in group_counts.items()
            if count > action.max_select_per_group
        }
        if over:
            raise OptimizerModelError(
                "Measurement selector exceeded max_select_per_group "
                f"{action.max_select_per_group}: {over}"
            )

    def _evaluate_candidate_progressively(
        self,
        *,
        patch: AgentPatch,
        reference: PatchSummary,
        baseline: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
    ) -> tuple[PatchSummary, Comparison, dict[str, Any], str | None, list[dict[str, Any]]]:
        stage_rows: list[dict[str, Any]] = []
        final_summary: PatchSummary | None = None
        final_comparison: Comparison | None = None
        final_flip_summary: dict[str, Any] | None = None
        final_rejection_reason: str | None = None
        for stage_name, stage_cases in self._progressive_eval_stages(reference, dev_cases):
            reference_summary = _summary_for_cases(reference, stage_cases) or self.evaluate_patch(reference.patch, stage_cases)
            baseline_summary = _summary_for_cases(baseline, stage_cases) or self.evaluate_patch(baseline.patch, stage_cases)
            candidate_summary = self.evaluate_patch(patch, stage_cases)
            comparison = compare_summaries(reference_summary, candidate_summary)
            flip_summary = behavior_flip_summary(reference_summary, candidate_summary)
            constraint_warning = None
            if stage_name == "smoke":
                rejection_reason = _smoke_rejection_reason(reference_summary, candidate_summary)
            else:
                rejection_reason = objective_rejection_reason(
                    reference_summary,
                    candidate_summary,
                    self.objective,
                )
                constraint_warning = constraint_rejection_reason(
                    baseline_summary,
                    candidate_summary,
                    self.objective,
                )
            stage_rows.append(
                {
                    "stage": stage_name,
                    "case_ids": [case.id for case in stage_cases],
                    "case_count": len(stage_cases),
                    "patch_hash": candidate_summary.patch_hash,
                    "metrics": candidate_summary.to_dict(),
                    "comparison_to_parent": comparison.to_dict(),
                    "behavior_flip_summary": flip_summary,
                    "rejection_reason": rejection_reason,
                    "constraint_warning": constraint_warning,
                    "passed": rejection_reason is None,
                }
            )
            final_summary = candidate_summary
            final_comparison = comparison
            final_flip_summary = flip_summary
            final_rejection_reason = rejection_reason
            if rejection_reason is not None or stage_name == "full_dev":
                break
        assert final_summary is not None
        assert final_comparison is not None
        assert final_flip_summary is not None
        return final_summary, final_comparison, final_flip_summary, final_rejection_reason, stage_rows

    def _progressive_eval_stages(
        self,
        reference: PatchSummary,
        dev_cases: tuple[EvalCase, ...],
    ) -> list[tuple[str, tuple[EvalCase, ...]]]:
        if len(dev_cases) <= 2:
            return [("full_dev", dev_cases)]
        case_by_id = {case.id: case for case in dev_cases}
        failed_ids = [
            case_id
            for case_id, _, _, _, case_passed in reference._case_rows()
            if not case_passed and case_id in case_by_id
        ]
        passed_ids = [
            case_id
            for case_id, _, _, _, case_passed in reference._case_rows()
            if case_passed and case_id in case_by_id
        ]
        smoke_ids = _ordered_unique([*(failed_ids[:1]), *(passed_ids[:1]), dev_cases[0].id])
        # Keep exploration cheap: the small stage should cover failures and a
        # representative stability sample without becoming an accidental full run.
        small_target = min(len(dev_cases), max(6, min(24, len(failed_ids) + len(smoke_ids) + 4)))
        small_ids = _ordered_unique([*smoke_ids, *failed_ids, *(case.id for case in dev_cases)])[:small_target]
        raw_stages = [
            ("smoke", tuple(case_by_id[case_id] for case_id in smoke_ids)),
            ("small_dev", tuple(case_by_id[case_id] for case_id in small_ids)),
            ("full_dev", dev_cases),
        ]
        stages: list[tuple[str, tuple[EvalCase, ...]]] = []
        seen_case_sets: set[tuple[str, ...]] = set()
        for name, cases in raw_stages:
            key = tuple(case.id for case in cases)
            if not cases or key in seen_case_sets:
                continue
            seen_case_sets.add(key)
            if len(cases) == len(dev_cases):
                stages.append(("full_dev", dev_cases))
                break
            stages.append((name, cases))
        return stages

    def evaluate_patch(
        self,
        patch: AgentPatch,
        cases: tuple[EvalCase, ...],
        *,
        sample_indices: Iterable[int] | None = None,
    ) -> PatchSummary:
        return self.evaluate_patches([patch], cases, sample_indices=sample_indices)[patch_hash(patch)]

    def evaluate_patches(
        self,
        patches: Iterable[AgentPatch],
        cases: tuple[EvalCase, ...],
        *,
        sample_indices: Iterable[int] | None = None,
    ) -> dict[str, PatchSummary]:
        patch_by_digest: dict[str, AgentPatch] = {}
        for patch in patches:
            patch_by_digest.setdefault(patch_hash(patch), patch)
        if not patch_by_digest:
            return {}
        indices = tuple(sample_indices) if sample_indices is not None else tuple(range(self.samples_per_case))
        if not indices:
            raise ValueError("sample_indices must not be empty.")
        ordered_by_digest: dict[str, list[CaseEvaluation | None]] = {
            digest: [None] * (len(cases) * len(indices))
            for digest in patch_by_digest
        }
        uncached: list[tuple[str, AgentPatch, int, EvalCase, int]] = []
        fresh_by_digest: Counter[str] = Counter()
        for digest, patch in patch_by_digest.items():
            order = 0
            ordered = ordered_by_digest[digest]
            for case in cases:
                for sample_index in indices:
                    with self._store_lock:
                        cached = self.store.get(digest, case, sample_index=sample_index)
                    if cached is not None:
                        with self._stats_lock:
                            self.stats.cache_hits += 1
                        ordered[order] = cached
                        self._emit_progress(
                            "case_cache_hit",
                            patch_hash=digest,
                            case_id=case.id,
                            split=case.split,
                            sample_index=sample_index,
                        )
                    else:
                        fresh_by_digest[digest] += 1
                        uncached.append((digest, patch, order, case, sample_index))
                    order += 1
        if uncached:
            concurrency_limit = self.stage_case_concurrency if len(patch_by_digest) > 1 else self.case_concurrency
            effective_concurrency = 1 if self.fail_fast else min(concurrency_limit, len(uncached))
            for digest, fresh_count in fresh_by_digest.items():
                self._emit_progress(
                    "case_batch_started",
                    patch_hash=digest,
                    split=cases[0].split,
                    case_count=len(cases),
                    sample_count=len(indices),
                    fresh_count=fresh_count,
                    concurrency=effective_concurrency,
                    parallel_patch_count=len(patch_by_digest),
                )
            if effective_concurrency == 1:
                for digest, patch, item_order, case, sample_index in uncached:
                    ordered_by_digest[digest][item_order] = self._run_uncached_case(
                        digest,
                        patch,
                        case,
                        sample_index=sample_index,
                    )
            else:
                futures: dict[Future[CaseEvaluation], tuple[str, AgentPatch, int, EvalCase, int]] = {}
                with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
                    for digest, patch, item_order, case, sample_index in uncached:
                        self._emit_progress(
                            "case_started",
                            patch_hash=digest,
                            case_id=case.id,
                            split=case.split,
                            sample_index=sample_index,
                        )
                        future = executor.submit(self._execute_case, patch, case, sample_index=sample_index)
                        futures[future] = (digest, patch, item_order, case, sample_index)
                    for future in as_completed(futures):
                        digest, patch, item_order, case, sample_index = futures[future]
                        evaluation = future.result()
                        with self._store_lock:
                            self.store.put(digest, patch, evaluation)
                        self._emit_case_completed(digest, evaluation)
                        ordered_by_digest[digest][item_order] = evaluation
            for digest, fresh_count in fresh_by_digest.items():
                self._emit_progress(
                    "case_batch_completed",
                    patch_hash=digest,
                    split=cases[0].split,
                    fresh_count=fresh_count,
                    concurrency=effective_concurrency,
                    parallel_patch_count=len(patch_by_digest),
                )
        summaries: dict[str, PatchSummary] = {}
        for digest, patch in patch_by_digest.items():
            ordered = ordered_by_digest[digest]
            evaluations = [evaluation for evaluation in ordered if evaluation is not None]
            if len(evaluations) != len(ordered):
                raise RuntimeError("internal evaluation error: missing case evaluation result")
            summaries[digest] = PatchSummary(
                patch_hash=digest,
                patch=patch,
                split=cases[0].split,
                evaluations=evaluations,
            )
        return summaries

    def _run_uncached_case(
        self,
        digest: str,
        patch: AgentPatch,
        case: EvalCase,
        *,
        sample_index: int,
    ) -> CaseEvaluation:
        self._emit_progress(
            "case_started",
            patch_hash=digest,
            case_id=case.id,
            split=case.split,
            sample_index=sample_index,
        )
        evaluation = self._execute_case(patch, case, sample_index=sample_index)
        with self._store_lock:
            self.store.put(digest, patch, evaluation)
        self._emit_case_completed(digest, evaluation)
        if self.fail_fast and evaluation.record.metrics.error:
            raise RuntimeError(
                f"Fail-fast stopping after case {case.id}: {evaluation.record.metrics.error}"
            )
        return evaluation

    def _emit_case_completed(self, digest: str, evaluation: CaseEvaluation) -> None:
        self._emit_progress(
            "case_completed",
            patch_hash=digest,
            case_id=evaluation.case.id,
            split=evaluation.case.split,
            sample_index=evaluation.sample_index,
            passed=evaluation.grade.passed,
            score=evaluation.grade.score,
            error=evaluation.record.metrics.error,
            latency_s=evaluation.record.metrics.latency_s,
            cost_usd=evaluation.record.metrics.cost_usd,
            model_calls=evaluation.record.metrics.model_calls,
            tool_calls=evaluation.record.metrics.tool_calls,
            turns=evaluation.record.metrics.turns,
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
                with case_timeout(self.case_timeout_s), model_request_limits(
                    timeout_s=self.case_timeout_s,
                    max_attempts=1,
                ):
                    record = self.adapter.run_case(case, effective_patch)
                if not isinstance(record, RunRecord):
                    raise TypeError(f"run_case returned {type(record).__name__}, expected RunRecord.")
                try:
                    json.dumps(record.output, sort_keys=True)
                except TypeError as error:
                    raise TypeError("run_case returned a non-JSON-serializable output.") from error
                last_phase = "grade"
                with case_timeout(self.case_timeout_s), model_request_limits(
                    timeout_s=self.case_timeout_s,
                    max_attempts=1,
                ):
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
                            turns=list(record.diagnostics.turns),
                            terminal_state=dict(record.diagnostics.terminal_state),
                            terminal_reason=record.diagnostics.terminal_reason,
                            metadata=diagnostic_metadata,
                        ),
                    )
                with self._stats_lock:
                    self.stats.fresh_case_evaluations += 1
                return CaseEvaluation(case=case, record=record, grade=grade, sample_index=sample_index)
            except Exception as error:
                last_error = error
                if attempt < total_attempts:
                    with self._stats_lock:
                        self.stats.retries += 1
                    continue

        assert last_error is not None
        elapsed = time.perf_counter() - started_at
        message = f"{type(last_error).__name__}: {last_error}"
        if _is_timeout_error(last_error):
            with self._stats_lock:
                self.stats.timeouts += 1
            labels = ["timeout"]
        elif last_phase == "grade":
            with self._stats_lock:
                self.stats.grader_errors += 1
            labels = ["grader_error"]
        else:
            with self._stats_lock:
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
        with self._stats_lock:
            self.stats.fresh_case_evaluations += 1
        return CaseEvaluation(case=case, record=record, grade=grade, sample_index=sample_index)

    def build_manifest(
        self,
        *,
        total_cases: int,
        train_case_count: int,
        proposal_example_bank: ProposalExampleBank,
        selected_patch_hash: str,
        promoted: bool,
        generated_surface: list[dict[str, Any]],
        task_theories: list[dict[str, Any]],
        transform_summaries: dict[str, dict[str, Any]],
        transform_context_summaries: dict[str, dict[str, Any]],
        affordance_summaries: dict[str, dict[str, Any]],
        transform_final_statuses: dict[str, dict[str, Any]],
        outcome_analysis: dict[str, Any],
        finalist_statuses: list[dict[str, Any]],
        runtime_reliability_diagnostics: list[dict[str, Any]],
        confirmation_results: list[dict[str, Any]],
        simplification_results: list[dict[str, Any]],
        frontier_recommendation: dict[str, Any],
        optimizer_call_diagnostics: list[dict[str, Any]],
        quality_cost_tradeoffs: list[dict[str, Any]],
        measurement_decisions: list[dict[str, Any]],
        ideation_metrics: dict[str, Any],
        evidence_ledger: dict[str, Any],
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
            "train_case_count": train_case_count,
            "proposal_example_bank": {
                "example_count": len(proposal_example_bank.examples),
                "label_counts": proposal_example_bank.label_counts,
                "metadata_categories": proposal_example_bank.metadata_categories,
                "label_field": proposal_example_bank.label_field,
            },
            "agent_spec_hash": agent_spec_hash(self.agent_spec),
            "objective": self.objective.to_dict(),
            "generated_surface_count": len(generated_surface),
            "task_theories": task_theories,
            "transform_summaries": transform_summaries,
            "transform_context_summaries": transform_context_summaries,
            "affordance_summaries": affordance_summaries,
            "transform_final_statuses": transform_final_statuses,
            "finalist_statuses": finalist_statuses,
            "runtime_reliability_diagnostics": runtime_reliability_diagnostics,
            "confirmation_results": confirmation_results,
            "simplification_results": simplification_results,
            "frontier_recommendation": frontier_recommendation,
            "optimizer_call_diagnostics": optimizer_call_diagnostics,
            "optimizer_role_models": self.optimizer_role_models,
            "optimizer_role_reasoning": self.optimizer_role_reasoning,
            "measurement_decisions": measurement_decisions,
            "quality_cost_tradeoffs": quality_cost_tradeoffs,
            "ideation_metrics": ideation_metrics,
            "evidence_ledger": evidence_ledger,
            "baseline_stability": _baseline_stability_from_evidence(evidence_ledger),
            "samples_per_case": self.samples_per_case,
            "case_concurrency": self.case_concurrency,
            "stage_case_concurrency": self.stage_case_concurrency,
            "expensive_candidate_cost_ratio": self.expensive_candidate_cost_ratio,
            "max_dev_measurement_cost_usd": self.max_dev_measurement_cost_usd,
            "max_holdout_measurement_cost_usd": self.max_holdout_measurement_cost_usd,
            "max_dev_measurement_tool_calls": self.max_dev_measurement_tool_calls,
            "max_holdout_measurement_tool_calls": self.max_holdout_measurement_tool_calls,
            "max_dev_measurement_turns": self.max_dev_measurement_turns,
            "max_holdout_measurement_turns": self.max_holdout_measurement_turns,
            "dev_measurement_cost_used_usd": self._dev_measurement_cost_usd,
            "holdout_measurement_cost_used_usd": self._holdout_measurement_cost_usd,
            "dev_measurement_tool_calls_used": self._dev_measurement_tool_calls,
            "holdout_measurement_tool_calls_used": self._holdout_measurement_tool_calls,
            "dev_measurement_turns_used": self._dev_measurement_turns,
            "holdout_measurement_turns_used": self._holdout_measurement_turns,
            "selected_patch_hash": selected_patch_hash,
            "promoted": promoted,
            "progress_path": str(self.out_dir / "progress.jsonl"),
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

    def _emit_progress(self, event: str, **fields: Any) -> None:
        started_at = self._progress_started_at or time.perf_counter()
        row = {
            "event": event,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with self._progress_lock:
            if self._progress_path is not None:
                append_jsonl(self._progress_path, row)
            if self.progress_callback is not None:
                self.progress_callback(row)


def _same_cases(summary: PatchSummary, cases: tuple[EvalCase, ...]) -> bool:
    return tuple(summary.grouped_evaluations) == tuple(case.id for case in cases)


def _summary_for_cases(summary: PatchSummary, cases: tuple[EvalCase, ...]) -> PatchSummary | None:
    grouped = summary.grouped_evaluations
    selected: list[CaseEvaluation] = []
    for case in cases:
        evaluations = grouped.get(case.id)
        if not evaluations:
            return None
        selected.extend(evaluations)
    return PatchSummary(
        patch_hash=summary.patch_hash,
        patch=summary.patch,
        split=cases[0].split,
        evaluations=selected,
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    rows: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        rows.append(value)
    return rows


def _top_counter_dict(values: dict[str, int], *, limit: int) -> dict[str, int]:
    return dict(sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit])


def _smoke_rejection_reason(reference: PatchSummary, candidate: PatchSummary) -> str | None:
    if candidate.runtime_error_count > reference.runtime_error_count:
        return "smoke rejected candidate because runtime errors increased"
    if candidate.pass_count < reference.pass_count:
        return "smoke rejected candidate because pass count regressed"
    return None


def _measurement_action(
    *,
    stage_name: str,
    states: list[CandidateEvaluationState],
    dev_evaluations_used: int,
    dev_budget: int | None,
) -> MeasurementAction:
    if stage_name == "full_dev":
        late_budget = dev_budget is not None and dev_budget > 0 and dev_evaluations_used >= dev_budget / 2
        group_cap = (
            MAX_LATE_FULL_DEV_EXPERIMENT_CANDIDATES_PER_GROUP
            if late_budget
            else MAX_FULL_DEV_EXPERIMENT_CANDIDATES_PER_GROUP
        )
        group_count = len({_candidate_research_group(state) for state in states})
        raw_max_select = sum(
            min(group_cap, len(group_states))
            for group_states in _states_by_research_group(states).values()
        )
        max_select = min(raw_max_select, MAX_LATE_FULL_DEV_CANDIDATES_PER_ACTION) if late_budget else raw_max_select
        rationale = (
            "Choose which smoke/small-dev survivors deserve full-dev measurement. "
            "Prefer experiments that best resolve the current task theory under remaining budget."
        )
    elif stage_name == "small_dev":
        group_cap = 0
        group_count = len({_candidate_research_group(state) for state in states})
        max_select = len(states)
        rationale = (
            "Choose which smoke-passing candidates deserve small-dev measurement. "
            "Reject duplicates or experiments that no longer teach useful information."
        )
    else:
        group_cap = 0
        group_count = len({_candidate_research_group(state) for state in states})
        max_select = len(states)
        rationale = f"Choose candidates for {stage_name} measurement."
    return MeasurementAction(
        action_id=f"evaluate_{stage_name}",
        action_type="evaluate_candidates",
        stage=stage_name,
        candidate_ids=[state.patch_hash for state in states],
        max_select=max_select,
        max_select_per_group=group_cap,
        rationale=rationale,
        metadata={
            "dev_evaluations_used": dev_evaluations_used,
            "dev_budget": dev_budget,
            "comparison_group_count": group_count,
            "late_budget": late_budget if stage_name == "full_dev" else False,
            "raw_max_select": raw_max_select if stage_name == "full_dev" else max_select,
        },
    )


def _has_evidence_for_selector(
    evidence_ledger: EvidenceLedger,
    states: list[CandidateEvaluationState],
) -> bool:
    return all(evidence_ledger.latest(state.patch_hash) is not None for state in states)


def _research_state_packet(
    *,
    objective: OptimizationObjective,
    stage_name: str,
    reference: PatchSummary,
    baseline: PatchSummary,
    states: list[CandidateEvaluationState],
    proposals_log: list[dict[str, Any]],
    dev_evaluations_used: int,
    dev_budget: int,
    evidence_ledger: EvidenceLedger,
    stage_cases: tuple[EvalCase, ...],
    samples_per_case: int,
    measurement_cost_used_usd: float,
    max_measurement_cost_usd: float | None,
    measurement_tool_calls_used: float,
    max_measurement_tool_calls: int | None,
    measurement_turns_used: float,
    max_measurement_turns: int | None,
) -> dict[str, Any]:
    candidate_ids = [state.patch_hash for state in states]
    candidate_evidence = _selector_rows_with_measurement_context(
        evidence_ledger=evidence_ledger,
        states=states,
        stage_cases=stage_cases,
        reference=reference,
        baseline=baseline,
        samples_per_case=samples_per_case,
        measurement_cost_used_usd=measurement_cost_used_usd,
        max_measurement_cost_usd=max_measurement_cost_usd,
        measurement_tool_calls_used=measurement_tool_calls_used,
        max_measurement_tool_calls=max_measurement_tool_calls,
        measurement_turns_used=measurement_turns_used,
        max_measurement_turns=max_measurement_turns,
    )
    remaining_measurement_budget = (
        None
        if max_measurement_cost_usd is None
        else max(0.0, max_measurement_cost_usd - measurement_cost_used_usd)
    )
    return {
        "objective": objective.to_dict(),
        "decision_point": stage_name,
        "budget": {
            "dev_evaluations_used": dev_evaluations_used,
            "dev_budget": dev_budget,
            "remaining_dev_budget": max(0, dev_budget - dev_evaluations_used),
            "measurement_cost_used_usd": measurement_cost_used_usd,
            "max_measurement_cost_usd": max_measurement_cost_usd,
            "remaining_measurement_budget_usd": remaining_measurement_budget,
            "measurement_tool_calls_used": measurement_tool_calls_used,
            "max_measurement_tool_calls": max_measurement_tool_calls,
            "remaining_measurement_tool_calls": (
                None
                if max_measurement_tool_calls is None
                else max(0.0, max_measurement_tool_calls - measurement_tool_calls_used)
            ),
            "measurement_turns_used": measurement_turns_used,
            "max_measurement_turns": max_measurement_turns,
            "remaining_measurement_turns": (
                None
                if max_measurement_turns is None
                else max(0.0, max_measurement_turns - measurement_turns_used)
            ),
        },
        "measurement_policy": {
            "small_dev": "Triage only; use it to decide whether more measurement is worth buying, not as final ranking.",
            "full_dev": "First selection-quality comparison; preserve mechanism-distinct high-signal candidates when budget permits.",
            "candidate_cost": (
                "Candidate cost_delta and latency_delta describe the deployed policy tradeoff. "
                "They are not the same as the cost of one more measurement. Expensive candidates may still be worth "
                "measuring when they test capability, efficiency, or quality-frontier hypotheses."
            ),
            "measurement_budget": (
                "Use marginal_measurement_cost_usd and remaining_measurement_budget_usd to decide whether the "
                "expected information is worth buying. For interactive tasks, also consider marginal tool calls and "
                "turns. Deterministic code enforces hard resource ceilings after selection."
            ),
            "quality_frontier": (
                "For correctness objectives, a high-quality candidate that violates cost or latency constraints can still "
                "be informative as a quality frontier. Do not skip it solely because promotion may later fail."
            ),
        },
        "reference": {
            "patch_hash": reference.patch_hash,
            "score": reference.mean_score,
            "pass_count": reference.pass_count,
            "case_count": reference.case_count,
            "cost_usd": reference.mean_cost_usd,
            "latency_s": reference.median_latency_s,
        },
        "baseline": {
            "patch_hash": baseline.patch_hash,
            "score": baseline.mean_score,
            "pass_count": baseline.pass_count,
            "case_count": baseline.case_count,
            "cost_usd": baseline.mean_cost_usd,
            "latency_s": baseline.median_latency_s,
        },
        "evidence_ledger": {
            "candidate_evidence": candidate_evidence,
            "summary": evidence_ledger.to_dict()["summary"],
        },
        "candidate_metadata": [_research_candidate_metadata(state) for state in states],
        "prior_full_dev_results": _compact_prior_stage_results(proposals_log, stage="full_dev", limit=8),
        "recent_candidate_history": _compact_prior_stage_results(proposals_log, stage=None, limit=8),
    }


def _selector_rows_with_measurement_context(
    *,
    evidence_ledger: EvidenceLedger,
    states: list[CandidateEvaluationState],
    stage_cases: tuple[EvalCase, ...],
    reference: PatchSummary,
    baseline: PatchSummary,
    samples_per_case: int,
    measurement_cost_used_usd: float,
    max_measurement_cost_usd: float | None,
    measurement_tool_calls_used: float,
    max_measurement_tool_calls: int | None,
    measurement_turns_used: float,
    max_measurement_turns: int | None,
) -> list[dict[str, Any]]:
    state_by_id = {state.patch_hash: state for state in states}
    rows: list[dict[str, Any]] = []
    remaining_budget = (
        None
        if max_measurement_cost_usd is None
        else max(0.0, max_measurement_cost_usd - measurement_cost_used_usd)
    )
    for raw_row in evidence_ledger.selector_rows(state_by_id):
        row = _compact_selector_evidence_row(raw_row)
        candidate_id = str(row.get("candidate_id") or "")
        state = state_by_id.get(candidate_id)
        if state is None:
            continue
        row["marginal_measurement_cost_usd"] = _estimated_marginal_measurement_cost_usd(
            state=state,
            stage_cases=stage_cases,
            evidence_ledger=evidence_ledger,
            samples_per_case=samples_per_case,
        )
        row["marginal_measurement_model_calls"] = _estimated_marginal_measurement_units(
            state=state,
            stage_cases=stage_cases,
            evidence_ledger=evidence_ledger,
            samples_per_case=samples_per_case,
            unit="model_calls",
        )
        row["marginal_measurement_tool_calls"] = _estimated_marginal_measurement_units(
            state=state,
            stage_cases=stage_cases,
            evidence_ledger=evidence_ledger,
            samples_per_case=samples_per_case,
            unit="tool_calls",
        )
        row["marginal_measurement_turns"] = _estimated_marginal_measurement_units(
            state=state,
            stage_cases=stage_cases,
            evidence_ledger=evidence_ledger,
            samples_per_case=samples_per_case,
            unit="turns",
        )
        row["remaining_measurement_budget_usd"] = remaining_budget
        row["remaining_measurement_tool_calls"] = (
            None
            if max_measurement_tool_calls is None
            else max(0.0, max_measurement_tool_calls - measurement_tool_calls_used)
        )
        row["remaining_measurement_turns"] = (
            None
            if max_measurement_turns is None
            else max(0.0, max_measurement_turns - measurement_turns_used)
        )
        row["deployed_cost_ratio"] = _safe_ratio(
            (state.summary.mean_cost_usd if state.summary is not None else 0.0),
            baseline.mean_cost_usd,
        )
        row["deployed_latency_ratio"] = _safe_ratio(
            (state.summary.median_latency_s if state.summary is not None else 0.0),
            baseline.median_latency_s,
        )
        row["reference_cost_ratio"] = _safe_ratio(
            (state.summary.mean_cost_usd if state.summary is not None else 0.0),
            reference.mean_cost_usd,
        )
        rows.append(row)
    return rows


def _compact_selector_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    comparison = row.get("comparison_to_reference") or {}
    measurement_cost = row.get("measurement_cost") or {}
    return {
        "candidate_id": row.get("candidate_id"),
        "stage": row.get("stage"),
        "case_count": row.get("case_count"),
        "effect_size": row.get("effect_size"),
        "pass_gain": row.get("pass_gain"),
        "fixed_count": row.get("fixed_count"),
        "regressed_count": row.get("regressed_count"),
        "invalid_output_delta": row.get("invalid_output_delta"),
        "finish_reason_delta": row.get("finish_reason_delta"),
        "token_delta": row.get("token_delta"),
        "cost_delta": row.get("cost_delta"),
        "latency_delta": row.get("latency_delta"),
        "model_call_delta": row.get("model_call_delta"),
        "tool_call_delta": row.get("tool_call_delta"),
        "turn_delta": row.get("turn_delta"),
        "sign_consistency": row.get("sign_consistency"),
        "confidence_tier": row.get("confidence_tier"),
        "baseline_instability_flags": list(row.get("baseline_instability_flags") or []),
        "measurement_cost": {
            "fresh_candidate_samples": measurement_cost.get("fresh_candidate_samples"),
            "estimated_total_cost_usd": measurement_cost.get("estimated_total_cost_usd"),
            "estimated_total_tokens": measurement_cost.get("estimated_total_tokens"),
            "estimated_model_calls": measurement_cost.get("estimated_model_calls"),
            "estimated_tool_calls": measurement_cost.get("estimated_tool_calls"),
            "estimated_turns": measurement_cost.get("estimated_turns"),
            "candidate_mean_cost_usd": measurement_cost.get("candidate_mean_cost_usd"),
            "candidate_mean_model_calls": measurement_cost.get("candidate_mean_model_calls"),
            "candidate_mean_tool_calls": measurement_cost.get("candidate_mean_tool_calls"),
            "candidate_mean_turns": measurement_cost.get("candidate_mean_turns"),
        },
        "mechanism_class": row.get("mechanism_class"),
        "affordance_ids": list(row.get("affordance_ids") or [])[:6],
        "comparison_group": row.get("comparison_group"),
        "candidate_role": row.get("candidate_role"),
        "rejection_reason": row.get("rejection_reason"),
        "constraint_warning": row.get("constraint_warning"),
        "passed_stage": row.get("passed_stage"),
        "comparison_to_reference": {
            "score_delta": comparison.get("score_delta"),
            "pass_rate_delta": comparison.get("pass_rate_delta"),
            "cost_delta": comparison.get("cost_delta"),
            "latency_delta": comparison.get("latency_delta"),
            "token_delta": comparison.get("token_delta"),
            "model_call_delta": comparison.get("model_call_delta"),
            "tool_call_delta": comparison.get("tool_call_delta"),
            "turn_delta": comparison.get("turn_delta"),
        },
        "stage_history": [
            _compact_selector_history_row(history)
            for history in list(row.get("stage_history") or [])[-3:]
            if isinstance(history, dict)
        ],
    }


def _compact_selector_history_row(row: dict[str, Any]) -> dict[str, Any]:
    comparison = row.get("comparison_to_reference") or {}
    return {
        "stage": row.get("stage"),
        "case_count": row.get("case_count"),
        "confidence_tier": row.get("confidence_tier"),
        "pass_gain": row.get("pass_gain"),
        "effect_size": row.get("effect_size"),
        "score_delta": comparison.get("score_delta"),
        "tool_call_delta": comparison.get("tool_call_delta"),
        "turn_delta": comparison.get("turn_delta"),
        "rejection_reason": row.get("rejection_reason"),
        "constraint_warning": row.get("constraint_warning"),
    }


def _research_candidate_metadata(state: CandidateEvaluationState) -> dict[str, Any]:
    return {
        "candidate_id": state.patch_hash,
        "transform_family": state.candidate.transform_family,
        "mechanism_class": state.candidate.mechanism_class,
        "candidate_role": state.candidate.candidate_role,
        "comparison_group": state.candidate.comparison_group,
        "target_slice": state.candidate.target_slice,
        "transform_instance": state.candidate.transform_instance,
        "hypothesis": state.candidate.hypothesis,
        "operation_count": len(state.patch.operations),
        "operations": [
            {"op": operation.op, "target": operation.target}
            for operation in state.patch.operations
        ],
        "comparison_group_key": _candidate_research_group(state),
    }


def _compact_prior_stage_results(
    proposals_log: list[dict[str, Any]],
    *,
    stage: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in reversed(proposals_log):
        stages = item.get("evaluation_stages") or []
        selected_stage = None
        if stage is None:
            selected_stage = stages[-1] if stages else None
        else:
            selected_stage = next((row for row in reversed(stages) if row.get("stage") == stage), None)
        if selected_stage is None:
            continue
        comparison = selected_stage.get("comparison_to_parent") or {}
        rows.append(
            {
                "candidate_id": item.get("patch_hash"),
                "parent_patch_hash": item.get("parent_patch_hash"),
                "transform_family": item.get("transform_family"),
                "mechanism_class": item.get("mechanism_class"),
                "candidate_role": item.get("candidate_role"),
                "target_slice": item.get("target_slice"),
                "frontier_status": item.get("frontier_status"),
                "accepted": item.get("accepted"),
                "stage": selected_stage.get("stage"),
                "case_count": selected_stage.get("case_count"),
                "score_delta": comparison.get("score_delta"),
                "cost_delta": comparison.get("cost_delta"),
                "latency_delta": comparison.get("latency_delta"),
                "rejection_reason": item.get("rejection_reason"),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _candidate_research_group(state: CandidateEvaluationState) -> str:
    comparison_group = (
        state.candidate.comparison_group
        or state.candidate.experiment_id
        or state.candidate.transform_family
    )
    return f"{comparison_group}|{state.candidate.target_slice or 'global'}"


def _states_by_research_group(
    states: list[CandidateEvaluationState],
) -> dict[str, list[CandidateEvaluationState]]:
    by_group: dict[str, list[CandidateEvaluationState]] = {}
    for state in states:
        by_group.setdefault(_candidate_research_group(state), []).append(state)
    return by_group


def _finalize_candidate_state(state: CandidateEvaluationState, reference: PatchSummary) -> None:
    if state.summary is None:
        return
    if state.rejection_reason is None and state.constraint_warning is None:
        state.frontier_status = "promotable"
    elif state.rejection_reason is None and state.constraint_warning is not None:
        state.frontier_status = "quality_frontier"
    elif _efficiency_improved(reference, state.summary):
        state.frontier_status = "efficiency_frontier"
    else:
        state.frontier_status = "failed"
    state.accepted = state.frontier_status in {"promotable", "quality_frontier", "efficiency_frontier"}


def _efficiency_improved(reference: PatchSummary, candidate: PatchSummary) -> bool:
    score_noninferior = candidate.mean_score >= reference.mean_score - 0.01
    cheaper = candidate.mean_cost_usd < reference.mean_cost_usd
    faster = candidate.median_latency_s < reference.median_latency_s
    return score_noninferior and (cheaper or faster)


def _estimated_marginal_measurement_cost_usd(
    *,
    state: CandidateEvaluationState,
    stage_cases: tuple[EvalCase, ...],
    evidence_ledger: EvidenceLedger,
    samples_per_case: int,
) -> float:
    if state.summary is None:
        return 0.0
    latest = evidence_ledger.latest(state.patch_hash)
    previously_measured = set(latest.case_ids) if latest is not None else set()
    marginal_case_count = sum(1 for case in stage_cases if case.id not in previously_measured)
    return max(0.0, state.summary.mean_cost_usd * marginal_case_count * max(1, samples_per_case))


def _estimated_marginal_measurement_units(
    *,
    state: CandidateEvaluationState,
    stage_cases: tuple[EvalCase, ...],
    evidence_ledger: EvidenceLedger,
    samples_per_case: int,
    unit: str,
) -> float:
    if state.summary is None:
        return 0.0
    latest = evidence_ledger.latest(state.patch_hash)
    previously_measured = set(latest.case_ids) if latest is not None else set()
    marginal_case_count = sum(1 for case in stage_cases if case.id not in previously_measured)
    multiplier = max(1, samples_per_case) * marginal_case_count
    if unit == "model_calls":
        return max(0.0, state.summary.mean_model_calls * multiplier)
    if unit == "tool_calls":
        return max(0.0, state.summary.mean_tool_calls * multiplier)
    if unit == "turns":
        return max(0.0, state.summary.mean_turns * multiplier)
    raise ValueError(f"Unsupported marginal measurement unit: {unit}")


def _measurement_budget_exhausted(
    *,
    used_usd: float,
    marginal_usd: float,
    max_usd: float | None,
) -> bool:
    if max_usd is None:
        return False
    return used_usd + marginal_usd > max_usd + 1e-12


def _measurement_budget_reason(
    *,
    used_usd: float,
    marginal_usd: float,
    max_usd: float | None,
    used_tool_calls: float,
    marginal_tool_calls: float,
    max_tool_calls: int | None,
    used_turns: float,
    marginal_turns: float,
    max_turns: int | None,
    stage: str,
) -> str | None:
    if max_usd is not None and used_usd + marginal_usd > max_usd + 1e-12:
        return (
            f"marginal {stage} measurement cost ${marginal_usd:.6f} exceeds remaining "
            "measurement dollar budget"
        )
    if max_tool_calls is not None and used_tool_calls + marginal_tool_calls > max_tool_calls + 1e-12:
        return (
            f"marginal {stage} measurement tool calls {marginal_tool_calls:.1f} exceed remaining "
            "measurement tool-call budget"
        )
    if max_turns is not None and used_turns + marginal_turns > max_turns + 1e-12:
        return (
            f"marginal {stage} measurement turns {marginal_turns:.1f} exceed remaining "
            "measurement turn budget"
        )
    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _is_timeout_error(error: Exception) -> bool:
    if isinstance(error, TimeoutError):
        return True
    error_type = type(error).__name__.lower()
    if "timeout" in error_type:
        return True
    message = str(error).lower()
    return "timed out" in message or "timeout" in message


def _requires_finalist_confirmation(patch: AgentPatch, runtime_diagnostic: dict[str, Any]) -> bool:
    if runtime_diagnostic.get("baseline_runtime_defect_fixed"):
        return True
    if runtime_diagnostic.get("fixed_invalid_output_case_ids") and _touches_output_or_runtime(patch):
        return True
    if runtime_diagnostic.get("finish_reason_changed_case_ids") and _touches_output_or_runtime(patch):
        return True
    return False


def _touches_output_or_runtime(patch: AgentPatch) -> bool:
    for operation in patch.operations:
        target = operation.target
        if operation.op == "set_runtime_param" and target.startswith("runtime."):
            return True
        if operation.op == "add_output_constraint" or target == "output_contract":
            return True
        if target.startswith("instructions.output"):
            return True
    return False


def _simplification_variants(patch: AgentPatch) -> list[AgentPatch]:
    variants: list[AgentPatch] = []
    operations = list(patch.operations)
    if len(operations) > 1:
        for index, operation in enumerate(operations):
            simplified = [item for item_index, item in enumerate(operations) if item_index != index]
            variants.append(
                AgentPatch(
                    operations=simplified,
                    rationale=f"Simplification removing operation {index + 1} ({operation.op} on {operation.target}).",
                    expected_effect="Preserve measured gain with less policy complexity.",
                    metadata={
                        **patch.metadata,
                        "simplification": {
                            "type": "remove_operation",
                            "removed_index": index,
                            "removed_op": operation.op,
                            "removed_target": operation.target,
                        },
                    },
                )
            )
    for op_index, operation in enumerate(operations):
        if operation.op != "add_few_shot" or not isinstance(operation.value, list) or len(operation.value) <= 1:
            continue
        for keep_count in sorted({1, max(1, len(operation.value) // 2)}):
            if keep_count >= len(operation.value):
                continue
            reduced_operations = list(operations)
            reduced_operations[op_index] = PatchOperation(
                op=operation.op,
                target=operation.target,
                value=operation.value[:keep_count],
                rationale=operation.rationale,
            )
            variants.append(
                AgentPatch(
                    operations=reduced_operations,
                    rationale=f"Simplification reducing few-shot examples from {len(operation.value)} to {keep_count}.",
                    expected_effect="Preserve measured gain with fewer prompt tokens.",
                    metadata={
                        **patch.metadata,
                        "simplification": {
                            "type": "reduce_few_shot",
                            "operation_index": op_index,
                            "original_count": len(operation.value),
                            "kept_count": keep_count,
                        },
                    },
                )
            )
    unique: dict[str, AgentPatch] = {}
    for variant in variants:
        if variant.operations:
            unique.setdefault(patch_hash(variant), variant)
    return list(unique.values())


def _transform_lineage_families(patch_hash_value: str, proposals: list[dict[str, Any]]) -> list[str]:
    row_by_patch = {
        str(row.get("patch_hash")): row
        for row in proposals
        if row.get("accepted") is not None and row.get("patch_hash")
    }
    families: list[str] = []
    seen: set[str] = set()
    cursor = patch_hash_value
    while cursor and cursor not in seen:
        seen.add(cursor)
        row = row_by_patch.get(cursor)
        if row is None:
            break
        family = row.get("transform_family")
        if isinstance(family, str) and family and family not in families:
            families.append(family)
        parent = row.get("parent_patch_hash")
        cursor = str(parent) if isinstance(parent, str) else ""
    return list(reversed(families))


def _transform_final_status_summaries(finalist_statuses: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for row in finalist_statuses:
        families = row.get("dev_transform_families") or ["unknown"]
        for family in families:
            family_name = str(family)
            summary = summaries.setdefault(
                family_name,
                {"validated": 0, "directional": 0, "failed": 0, "unstable": 0, "finalist_count": 0, "patch_hashes": []},
            )
            status = str(row.get("status") or "failed")
            if status not in {"validated", "directional", "failed", "unstable"}:
                status = "failed"
            summary[status] += 1
            summary["finalist_count"] += 1
            if row.get("patch_hash"):
                summary["patch_hashes"].append(row.get("patch_hash"))
    return summaries


def _baseline_stability_from_evidence(evidence_ledger: dict[str, Any]) -> dict[str, Any]:
    records = [row for row in evidence_ledger.get("records", []) if isinstance(row, dict)]
    unstable_records = [
        row
        for row in records
        if row.get("confidence_tier") == "unstable" or row.get("baseline_instability_flags")
    ]
    instability_counts = (evidence_ledger.get("summary") or {}).get("baseline_instability_counts") or {}
    return {
        "unstable_candidate_evidence_count": len(unstable_records),
        "instability_counts": dict(instability_counts),
        "requires_runtime_repeat": bool(instability_counts.get("runtime_repeat_required")),
    }


def _task_theory_with_affordance_opportunities(
    *,
    task_theory: TaskTheory,
    affordances: list[OptimizationAffordance],
    current_dev: PatchSummary,
    proposals_log: list[dict[str, Any]],
    objective: OptimizationObjective,
) -> dict[str, Any]:
    payload = task_theory.to_dict()
    opportunities = list(payload.get("experiment_opportunities") or [])
    existing_mechanisms = {
        str(item.get("mechanism_class"))
        for item in opportunities
        if isinstance(item, dict) and item.get("mechanism_class")
    }
    capability_affordances = [
        affordance
        for affordance in affordances
        if affordance.family == "model_substitution"
        and affordance.mechanism == "model_capability_probe"
    ]
    if (
        capability_affordances
        and "model_capability_probe" not in existing_mechanisms
        and current_dev.pass_count < current_dev.case_count
        and _residual_quality_signal_remains(task_theory, proposals_log, current_dev.patch_hash)
    ):
        opportunities.insert(
            _capability_probe_insert_index(opportunities),
            {
                "mechanism_class": "model_capability_probe",
                "target_slices": ["global", "failed_cases"],
                "candidate_roles": ["atomic", "control"],
                "measurements": ["score_delta", "fixed_case_ids", "regressed_case_ids", "cost_delta", "latency_delta"],
                "affordance_ids": [affordance.affordance_id for affordance in capability_affordances[:3]],
                "rationale": (
                    "Residual correctness failures remain and model-choice affordances are available; "
                    "test whether failures are capability-limited rather than instruction/example-limited."
                ),
                "disconfirming_result": "A stronger allowed model does not fix residual failures or causes regressions/cost that dominate the quality gain.",
            },
        )
        payload["evidence"] = [
            *list(payload.get("evidence") or []),
            "model capability affordance available for residual correctness failures",
        ]
    efficiency_affordances = [
        affordance
        for affordance in affordances
        if affordance.family == "model_substitution"
        and affordance.mechanism == "efficiency_probe"
    ]
    if efficiency_affordances and _model_efficiency_probe_is_relevant(objective):
        if "efficiency_probe" in existing_mechanisms:
            for opportunity in opportunities:
                if isinstance(opportunity, dict) and opportunity.get("mechanism_class") == "efficiency_probe":
                    existing_ids = [
                        str(item)
                        for item in opportunity.get("affordance_ids", [])
                        if item
                    ]
                    opportunity["affordance_ids"] = list(
                        dict.fromkeys(
                            [
                                *existing_ids,
                                *[affordance.affordance_id for affordance in efficiency_affordances[:3]],
                            ]
                        )
                    )
                    opportunity["rationale"] = (
                        str(opportunity.get("rationale") or "")
                        + " Model-choice affordances can also test cheaper or faster equivalent policies."
                    ).strip()
                    break
        else:
            opportunities.append(
                {
                    "mechanism_class": "efficiency_probe",
                    "target_slices": ["global"],
                    "candidate_roles": ["atomic", "control"],
                    "measurements": ["score_delta", "cost_delta", "latency_delta", "correctness_guard"],
                    "affordance_ids": [affordance.affordance_id for affordance in efficiency_affordances[:3]],
                    "rationale": (
                        "Model-choice affordances are available; test whether a different allowed model "
                        "can preserve quality while reducing cost or latency."
                    ),
                    "disconfirming_result": "The model change reduces cost or latency only by causing correctness regressions.",
                }
            )
        payload["evidence"] = [
            *list(payload.get("evidence") or []),
            "model efficiency affordance available for cost/latency tradeoff testing",
        ]
    payload["experiment_opportunities"] = opportunities[:8]
    return payload


def _model_efficiency_probe_is_relevant(objective: OptimizationObjective) -> bool:
    constraints = objective.constraints
    return (
        objective.mode in {"cost", "latency"}
        or constraints.max_cost_ratio is not None
        or constraints.max_latency_ratio is not None
        or "lower_cost" in objective.tie_breakers
        or "lower_latency" in objective.tie_breakers
    )


def _capability_probe_insert_index(opportunities: list[dict[str, Any]]) -> int:
    for index, row in enumerate(opportunities):
        if not isinstance(row, dict):
            continue
        if row.get("mechanism_class") in {"representative_examples", "contrastive_examples", "efficiency_probe"}:
            return index
    return len(opportunities)


def _residual_quality_signal_remains(
    task_theory: TaskTheory,
    proposals_log: list[dict[str, Any]],
    parent_patch_hash: str,
) -> bool:
    if task_theory.bottleneck_class in {"runtime_or_output_defect", "output_contract", "no_observed_failures"}:
        return False
    branch_rows = [
        row
        for row in proposals_log
        if row.get("parent_patch_hash") == parent_patch_hash and "accepted" in row
    ]
    if not branch_rows:
        return True
    local_mechanism_rows = [
        row
        for row in branch_rows
        if row.get("mechanism_class") in {
            "semantic_boundary_rewrite",
            "representative_examples",
            "contrastive_examples",
        }
    ]
    if not local_mechanism_rows:
        return True
    return any(not row.get("accepted") for row in local_mechanism_rows[-4:])


def _has_selectable_frontier_parent(frontier_states: dict[str, FrontierParentState]) -> bool:
    return any(not state.exhausted for state in frontier_states.values())


def _select_frontier_parents(
    summaries: Iterable[PatchSummary],
    *,
    frontier_states: dict[str, FrontierParentState],
    objective: OptimizationObjective,
    width: int,
) -> list[PatchSummary]:
    selectable = [
        summary
        for summary in summaries
        if not frontier_states.setdefault(summary.patch_hash, FrontierParentState()).exhausted
    ]
    ranked = sorted(
        selectable,
        key=lambda summary: _frontier_parent_sort_key(summary, frontier_states[summary.patch_hash], objective),
    )
    return ranked[: max(0, width)]


def _frontier_parent_sort_key(
    summary: PatchSummary,
    state: FrontierParentState,
    objective: OptimizationObjective,
) -> tuple[Any, ...]:
    return (
        state.consecutive_stalls,
        state.visits,
        objective_sort_key(summary, objective),
    )


def _frontier_state_dict(state: FrontierParentState) -> dict[str, Any]:
    return {
        "visits": state.visits,
        "consecutive_stalls": state.consecutive_stalls,
        "accepted_child_count": state.accepted_child_count,
        "last_selected_iteration": state.last_selected_iteration,
        "exhausted": state.exhausted,
    }


def _summary_progress_fields(summary: PatchSummary) -> dict[str, Any]:
    return {
        "patch_hash": summary.patch_hash,
        "case_count": summary.case_count,
        "pass_count": summary.pass_count,
        "mean_score": round(summary.mean_score, 4),
        "mean_cost_usd": summary.mean_cost_usd,
        "mean_model_calls": summary.mean_model_calls,
        "mean_tool_calls": summary.mean_tool_calls,
        "mean_turns": summary.mean_turns,
        "median_latency_s": summary.median_latency_s,
    }


def _affordance_evidence_from_theory(task_theory: TaskTheory, diagnoses: list[FailureDiagnosis]) -> dict[str, Any]:
    row = task_theory.to_dict()
    runtime = row.get("runtime_defects") or {}
    output = row.get("output_defects") or {}
    tool = row.get("tool_defects") or {}
    return {
        "bottleneck_class": row.get("bottleneck_class"),
        "runtime_defect": bool(runtime.get("length_finish_case_ids") or runtime.get("parser_fallback_case_ids")),
        "invalid_output": bool(output.get("invalid_output_count")),
        "tool_trajectory_defect": bool(
            tool.get("tool_error_case_ids")
            or tool.get("invalid_tool_call_case_ids")
            or tool.get("premature_stop_case_ids")
            or tool.get("turn_outcome_counts")
        ),
        "example_coverage": bool((row.get("example_coverage") or {}).get("example_count")),
        "diagnosis_target_names": sorted({target for diagnosis in diagnoses for target in diagnosis.target_names}),
    }
