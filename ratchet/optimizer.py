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

from ratchet.adapters import AdapterProtocol, checked_agent_spec, checked_surface_spec
from ratchet.affordances import OptimizationAffordance, generate_optimization_affordances
from ratchet.diagnosis import FailureDiagnoser
from ratchet.evidence import ProposalExampleBank, build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.evidence_ledger import EvidenceLedger
from ratchet.experiments import (
    EvidencePacket,
    ExperimentIntent,
    ResearchState,
    ResearchTheory,
    build_evidence_packet,
)
from ratchet.ideation import build_ideation_metrics
from ratchet.io import agent_spec_hash, append_jsonl, compiled_candidate_id, transform_program_hash
from ratchet.model_client import model_request_limits
from ratchet.objectives import (
    behavior_flip_summary,
    constraint_rejection_reason,
    candidate_rejection_reason,
    compare_summaries,
    final_gate_status,
    objective_rejection_reason,
    objective_sort_key,
    pareto_frontier,
    select_recommended_candidate,
)
from ratchet.profiling import (
    build_run_profile,
    confirmation_case_subset,
    confirmation_result,
    quality_cost_tradeoffs,
    runtime_reliability_diagnostics,
)
from ratchet.proposals import CandidateImplementer
from ratchet.research import MeasurementSelector, MeasurementAction, ResearchPlanner, ResearchTheorist
from ratchet.reporting import RatchetReporter, build_outcome_analysis
from ratchet.results import (
    CandidateSummary,
    CaseEvaluation,
    Comparison,
    OptimizerStats,
    RatchetResult,
    ResultStore,
    build_cache_namespace,
    split_train_dev_holdout,
)
from ratchet.surfaces import SurfaceSpec
from ratchet.transform_compiler import TransformCompiler
from ratchet.transform_program import CompiledCandidate, TransformPatch, TransformProgram
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
    AgentSpec,
    DiagnosticTrace,
    EvalCase,
    FailureDiagnosis,
    GradeResult,
    OperationalMetrics,
    OptimizationObjective,
    RunRecord,
)


SEARCH_FRONTIER_WIDTH = 1
PROPOSAL_RETRY_BUDGET = 1
FINALIST_CONFIRMATION_SAMPLES = 1
FRONTIER_PARENT_STALL_LIMIT = 2
MIN_REMAINING_DEV_EVALS_FOR_NEW_ROUND = 2
MAX_CONSECUTIVE_ZERO_EVAL_PARENT_ATTEMPTS = 3
MAX_FULL_DEV_EXPERIMENT_CANDIDATES_PER_GROUP = 1
MAX_LATE_FULL_DEV_EXPERIMENT_CANDIDATES_PER_GROUP = 1
MAX_LATE_FULL_DEV_CANDIDATES_PER_ACTION = 1
ProgressCallback = Callable[[dict[str, Any]], None]


def compose_transform_candidate(
    parent: CompiledCandidate | None,
    child: TransformProgram,
    *,
    compiler: TransformCompiler,
    surface: SurfaceSpec,
) -> CompiledCandidate:
    patches = (*((parent.program.patches if parent is not None else ())), *child.patches)
    metadata = {
        **(dict(parent.program.metadata) if parent is not None else {}),
        **dict(child.metadata),
        "parent_candidate_id": parent.program.candidate_id if parent is not None else None,
    }
    program = TransformProgram(
        candidate_id=child.candidate_id,
        hypothesis_id=child.hypothesis_id,
        patches=patches,
        metadata=metadata,
    )
    return compiler.compile_or_raise(program, surface)


@dataclass
class FrontierParentState:
    visits: int = 0
    consecutive_stalls: int = 0
    accepted_child_count: int = 0
    last_selected_iteration: int = 0
    exhausted: bool = False


@dataclass
class CandidateEvaluationState:
    proposal: CandidateProposal
    compiled_candidate: CompiledCandidate
    candidate_id: str
    proposal_candidate_id: str
    transform_context: TransformContextKey
    stage_rows: list[dict[str, Any]] = field(default_factory=list)
    summary: CandidateSummary | None = None
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
        research_theorist_model: str | None = None,
        research_theorist_reasoning: str | None = None,
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
        self.surface_spec: SurfaceSpec | None = None
        self.transform_compiler = TransformCompiler()
        self.optimizer_role_models = {
            "diagnoser": diagnoser_model or optimizer_model,
            "research_theorist": research_theorist_model or optimizer_model,
            "research_planner": research_planner_model or optimizer_model,
            "candidate_implementer": candidate_implementer_model or optimizer_model,
            "measurement_selector": measurement_selector_model or optimizer_model,
        }
        self.optimizer_role_reasoning = {
            "diagnoser": diagnoser_reasoning or optimizer_reasoning,
            "research_theorist": research_theorist_reasoning or optimizer_reasoning,
            "research_planner": research_planner_reasoning or optimizer_reasoning,
            "candidate_implementer": candidate_implementer_reasoning or optimizer_reasoning,
            "measurement_selector": measurement_selector_reasoning or optimizer_reasoning,
        }
        self.diagnoser = FailureDiagnoser(
            env_path=env_path,
            model=self.optimizer_role_models["diagnoser"],
            reasoning_effort=self.optimizer_role_reasoning["diagnoser"],
        )
        self.research_theorist = ResearchTheorist(
            env_path=env_path,
            model=self.optimizer_role_models["research_theorist"],
            reasoning_effort=self.optimizer_role_reasoning["research_theorist"],
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
        if self.case_timeout_s > 0 and (self.case_concurrency > 1 or self.stage_case_concurrency > 1):
            raise ValueError(
                "case_timeout_s requires serial case execution; set case_timeout_s=0 to use "
                "case_concurrency or stage_case_concurrency above 1."
            )
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
        self.store = ResultStore(
            out_dir,
            cache_namespace=self.cache_namespace,
            shared_cache_path=Path(".ratchet/cache/case_results.jsonl"),
        )
        self.stats = OptimizerStats()
        self.started_at: datetime | None = None
        self.progress_callback = progress_callback
        self._progress_started_at: float | None = None
        self._progress_path: Path | None = None
        self._progress_lock = threading.Lock()
        self._store_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self.optimizer_call_diagnostics: list[dict[str, Any]] = []

    def _surface(self) -> SurfaceSpec:
        if self.surface_spec is None:
            raise RuntimeError("surface_spec has not been inferred for this optimizer run.")
        return self.surface_spec

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
        surface_cases = train_cases or dev_cases
        self.surface_spec = checked_surface_spec(self.adapter, cases=surface_cases)
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

        baseline_candidate = None
        self._emit_progress("baseline_dev_started", case_count=len(dev_cases))
        baseline_dev = self.evaluate_candidate(baseline_candidate, dev_cases)
        self._emit_progress("baseline_dev_completed", **_summary_progress_fields(baseline_dev))
        baseline_holdout: CandidateSummary | None = None

        accepted_dev_candidates: list[CandidateSummary] = []
        accepted_dev_ids: set[str] = set()
        parent_pool_by_id: dict[str, CandidateSummary] = {baseline_dev.candidate_id: baseline_dev}
        frontier_states: dict[str, FrontierParentState] = {
            baseline_dev.candidate_id: FrontierParentState(),
        }
        decision_log: list[dict[str, Any]] = []
        diagnoses_log: list[dict[str, Any]] = []
        proposals_log: list[dict[str, Any]] = []
        task_theory_log: list[dict[str, Any]] = []
        evidence_ledger = EvidenceLedger()
        diagnosis_cache: dict[str, tuple[list[FailureDiagnosis], str]] = {}
        evidence_packet_cache: dict[str, EvidencePacket] = {}
        evaluated_candidate_ids = {baseline_dev.candidate_id}
        generated_surface_rows: list[dict[str, Any]] = [self._surface().to_dict()]
        dev_evaluations = 0
        iteration = 0
        consecutive_zero_eval_parent_attempts = 0

        while dev_evaluations < self.dev_budget and _has_selectable_frontier_parent(frontier_states):
            if (
                accepted_dev_candidates
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
                parent_pool_by_id.values(),
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
            next_frontier_by_id: dict[str, CandidateSummary] = {}
            search_complete = False

            for parent_index, current_dev in enumerate(parent_summaries):
                if dev_evaluations >= self.dev_budget:
                    break
                remaining_parents = len(parent_summaries) - parent_index
                remaining_budget = self.dev_budget - dev_evaluations
                proposal_budget = max(1, (remaining_budget + remaining_parents - 1) // remaining_parents)
                parent_state = frontier_states.setdefault(current_dev.candidate_id, FrontierParentState())
                parent_state.visits += 1
                parent_state.last_selected_iteration = iteration
                current_spec = self.agent_spec
                surface = self._surface()
                generated_surface_rows = [surface.to_dict()]
                self._emit_progress(
                    "parent_started",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    parent_candidate_id=current_dev.candidate_id,
                    **_summary_progress_fields(current_dev),
                )
                self._emit_progress(
                    "diagnosis_started",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    failure_count=current_dev.case_count - current_dev.pass_count,
                )
                diagnosis_cached = current_dev.candidate_id in diagnosis_cache
                if diagnosis_cached:
                    diagnoses, diagnosis_analysis = diagnosis_cache[current_dev.candidate_id]
                    diagnosis_call_diagnostics: dict[str, Any] = {}
                    decision_log.append(
                        {
                            "type": "optimizer_cache_hit",
                            "cache": "diagnosis",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
                        }
                    )
                else:
                    diagnoses, diagnosis_analysis = self.diagnoser.diagnose(current_dev, surface, self.objective)
                    diagnosis_cache[current_dev.candidate_id] = (diagnoses, diagnosis_analysis)
                    diagnosis_call_diagnostics = self.diagnoser.last_call_diagnostics or {}
                    if self.diagnoser.last_call_diagnostics is not None:
                        self.optimizer_call_diagnostics.append(
                            {
                                "iteration": iteration,
                                "parent_rank": parent_index + 1,
                                "parent_candidate_id": current_dev.candidate_id,
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
                evidence_packet_cached = current_dev.candidate_id in evidence_packet_cache
                if evidence_packet_cached:
                    evidence_packet = evidence_packet_cache[current_dev.candidate_id]
                    decision_log.append(
                        {
                            "type": "optimizer_cache_hit",
                            "cache": "evidence_packet",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
                        }
                    )
                else:
                    evidence_packet = build_evidence_packet(
                        summary=current_dev,
                        diagnoses=diagnoses,
                        objective=self.objective,
                        proposal_example_bank=proposal_example_bank,
                    )
                    evidence_packet_cache[current_dev.candidate_id] = evidence_packet
                evidence_packet_row = {
                    "type": "evidence_packet",
                    "iteration": iteration,
                    "parent_rank": parent_index + 1,
                    "parent_candidate_id": current_dev.candidate_id,
                    "candidate_id": current_dev.candidate_id,
                    "cached": evidence_packet_cached,
                    "evidence_packet": evidence_packet.to_dict(),
                }
                decision_log.append(evidence_packet_row)
                self._emit_progress(
                    "evidence_packet_ready",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    residual_failure_modes=evidence_packet.residual_failure_modes,
                    confidence=evidence_packet.confidence,
                    cached=evidence_packet_cached,
                )
                search_hypothesis = build_search_hypothesis(
                    summary=current_dev,
                    surface=surface,
                    objective=self.objective,
                    history=proposals_log,
                    parent_candidate_id=current_dev.candidate_id,
                    diagnoses=diagnoses,
                    proposal_example_count=len(proposal_example_bank.examples),
                )
                search_hypothesis_row = {
                    "type": "search_hypothesis",
                    "iteration": iteration,
                    "parent_rank": parent_index + 1,
                    "parent_candidate_id": current_dev.candidate_id,
                    "candidate_id": current_dev.candidate_id,
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
                    evidence=_affordance_evidence_from_packet(evidence_packet, diagnoses),
                )
                decision_log.append(
                    {
                        "type": "surface_opportunities",
                        "iteration": iteration,
                        "parent_rank": parent_index + 1,
                        "parent_candidate_id": current_dev.candidate_id,
                        "surface_opportunities": [affordance.to_dict() for affordance in affordances],
                    }
                )
                research_theory = self._build_research_theory(
                    current_dev=current_dev,
                    evidence_packet=evidence_packet,
                    diagnoses=diagnoses,
                    surface=surface,
                    search_hypothesis=search_hypothesis,
                    affordances=affordances,
                    proposals_log=proposals_log,
                    evidence_ledger=evidence_ledger,
                    decision_log=decision_log,
                    iteration=iteration,
                    parent_index=parent_index,
                    dev_evaluations_used=dev_evaluations,
                    proposal_budget=proposal_budget,
                )
                task_theory_row = {
                    "type": "research_theory",
                    "iteration": iteration,
                    "parent_rank": parent_index + 1,
                    "parent_candidate_id": current_dev.candidate_id,
                    "candidate_id": current_dev.candidate_id,
                    "research_theory": research_theory.to_dict(),
                    "task_theory": research_theory.to_dict(),
                    "evidence_packet": evidence_packet.to_dict(),
                }
                decision_log.append(task_theory_row)
                task_theory_log.append(task_theory_row)
                self._emit_progress(
                    "research_theory_ready",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    primary_hypothesis_id=research_theory.primary_hypothesis_id,
                    bottleneck_class=research_theory.bottleneck_class,
                    opportunity_count=len(research_theory.experiment_opportunities),
                    confidence=research_theory.confidence,
                )
                for diagnosis in diagnoses:
                    diagnoses_log.append(
                        {
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
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
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
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
                    research_theory=research_theory,
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
                    research_theory=research_theory,
                    evidence_packet=evidence_packet,
                    diagnosis_analysis=diagnosis_analysis,
                    search_hypothesis=search_hypothesis,
                    current_spec=current_spec,
                    proposal_example_bank=proposal_example_bank,
                    proposal_example_cases=train_cases,
                    evaluated_candidate_ids=evaluated_candidate_ids,
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
                    and accepted_dev_candidates
                ):
                    reason = "repeated proposal rounds produced no valid evaluable candidates"
                    decision_log.append(
                        {
                            "type": "search_stopped",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
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
                        parent_candidate_id=current_dev.candidate_id,
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
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
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
                        evidence=_affordance_evidence_from_packet(evidence_packet, diagnoses),
                    )
                    retry_research_theory = self._build_research_theory(
                        current_dev=current_dev,
                        evidence_packet=evidence_packet,
                        diagnoses=diagnoses,
                        surface=surface,
                        search_hypothesis=retry_search_hypothesis,
                        affordances=retry_affordances,
                        proposals_log=proposals_log,
                        evidence_ledger=evidence_ledger,
                        decision_log=decision_log,
                        iteration=iteration,
                        parent_index=parent_index,
                        dev_evaluations_used=dev_evaluations,
                        proposal_budget=min(PROPOSAL_RETRY_BUDGET, self.dev_budget - dev_evaluations),
                        proposal_retry=True,
                    )
                    retry_experiment_intents = self._plan_parent_research_action(
                        current_dev=current_dev,
                        research_theory=retry_research_theory,
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
                            research_theory=retry_research_theory,
                            evidence_packet=evidence_packet,
                            diagnosis_analysis=diagnosis_analysis,
                            search_hypothesis=retry_search_hypothesis,
                            current_spec=current_spec,
                            proposal_example_bank=proposal_example_bank,
                            proposal_example_cases=train_cases,
                            evaluated_candidate_ids=evaluated_candidate_ids,
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
                    if accepted_summary.candidate_id not in accepted_dev_ids:
                        accepted_dev_ids.add(accepted_summary.candidate_id)
                        accepted_dev_candidates.append(accepted_summary)
                    parent_pool_by_id.setdefault(accepted_summary.candidate_id, accepted_summary)
                    frontier_states.setdefault(accepted_summary.candidate_id, FrontierParentState())
                    next_frontier_by_id.setdefault(accepted_summary.candidate_id, accepted_summary)
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
                            "parent_candidate_id": current_dev.candidate_id,
                            "proposal_candidate_id": transform_program_hash(chosen_proposal.program),
                            "transform_family": chosen_proposal.transform_family,
                            "transform_context": TransformContextKey.from_candidate(chosen_proposal).to_dict(),
                            "candidate_id": chosen_dev.candidate_id,
                            "metrics": chosen_dev.to_dict(),
                        }
                    )

            if search_complete:
                break
            if not next_frontier_by_id and not _has_selectable_frontier_parent(frontier_states):
                break
            frontier = _select_frontier_parents(
                parent_pool_by_id.values(),
                frontier_states=frontier_states,
                objective=self.objective,
                width=SEARCH_FRONTIER_WIDTH,
            )
            decision_log.append(
                {
                    "type": "frontier_update",
                    "iteration": iteration,
                    "frontier_width": SEARCH_FRONTIER_WIDTH,
                    "frontier_candidate_ids": [summary.candidate_id for summary in frontier],
                    "accepted_candidate_ids": [
                        summary.candidate_id
                        for summary in sorted(
                            next_frontier_by_id.values(),
                            key=lambda summary: objective_sort_key(summary, self.objective),
                        )
                    ],
                    "parent_pool_candidate_ids": [
                        summary.candidate_id
                        for summary in sorted(
                            parent_pool_by_id.values(),
                            key=lambda summary: objective_sort_key(summary, self.objective),
                        )
                    ],
                    "frontier_parent_states": {
                        candidate_id_value: _frontier_state_dict(state)
                        for candidate_id_value, state in sorted(frontier_states.items())
                    },
                }
            )
            self._emit_progress(
                "frontier_updated",
                iteration=iteration,
                frontier_candidate_ids=[summary.candidate_id for summary in frontier],
                accepted_count=len(next_frontier_by_id),
                selectable_parent_count=sum(1 for state in frontier_states.values() if not state.exhausted),
            )

        best_dev_candidate = min(
            [baseline_dev, *accepted_dev_candidates],
            key=lambda summary: objective_sort_key(summary, self.objective),
        )
        finalist_dev_candidates = sorted(
            accepted_dev_candidates,
            key=lambda summary: objective_sort_key(summary, self.objective),
        )[: self.holdout_budget]
        simplification_results: list[dict[str, Any]] = []

        holdout_candidates: list[CandidateSummary] = []
        finalist_statuses: list[dict[str, Any]] = []
        runtime_diagnostics: list[dict[str, Any]] = []
        confirmation_results: list[dict[str, Any]] = []
        promotable: list[tuple[CandidateSummary, Comparison]] = []
        holdout_ready: list[tuple[CandidateSummary, dict[str, Any]]] = []
        if self.holdout_budget <= 0 and accepted_dev_candidates:
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
                finalist_count=len(accepted_dev_candidates),
            )
        for dev_summary in finalist_dev_candidates:
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
                        "candidate_id": dev_summary.candidate_id,
                        "status": "deferred",
                        "stage": "holdout_skipped",
                        "reason": reason,
                        "dev_transform_families": _transform_lineage_families(dev_summary.candidate_id, proposals_log),
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
                        "candidate_id": dev_summary.candidate_id,
                        "reason": reason,
                        "dev_metrics": dev_summary.to_dict(),
                    }
                )
                self._emit_progress(
                    "holdout_validation_skipped",
                    candidate_id=dev_summary.candidate_id,
                    reason=reason,
                )
                continue
            self._holdout_measurement_cost_usd += holdout_measurement_cost
            self._holdout_measurement_tool_calls += holdout_measurement_tool_calls
            self._holdout_measurement_turns += holdout_measurement_turns
            runtime_diagnostic = runtime_reliability_diagnostics(baseline_dev, dev_summary)
            runtime_diagnostics.append(runtime_diagnostic)
            if _requires_finalist_confirmation(dev_summary.candidate, runtime_diagnostic):
                confirmation_cases = confirmation_case_subset(baseline_dev, dev_summary, dev_cases)
                self._emit_progress(
                    "confirmation_started",
                    candidate_id=dev_summary.candidate_id,
                    case_count=len(confirmation_cases),
                    sample_count=FINALIST_CONFIRMATION_SAMPLES,
                    reason=runtime_diagnostic.get("reason"),
                )
                sample_start = 1000 + len(confirmation_results) * 100
                sample_indices = tuple(range(sample_start, sample_start + FINALIST_CONFIRMATION_SAMPLES))
                confirmation_summaries = self.evaluate_candidates(
                    [baseline_candidate, dev_summary.candidate],
                    confirmation_cases,
                    sample_indices=sample_indices,
                )
                confirmation_baseline = confirmation_summaries[compiled_candidate_id(baseline_candidate)]
                confirmation_candidate = confirmation_summaries[dev_summary.candidate_id]
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
                        "candidate_id": dev_summary.candidate_id,
                        "runtime_reliability_diagnostics": runtime_diagnostic,
                        "confirmation": confirmation,
                    }
                )
                self._emit_progress(
                    "confirmation_completed",
                    candidate_id=dev_summary.candidate_id,
                    passed=confirmation.get("passed"),
                    stability_status=confirmation.get("status"),
                    reason=confirmation.get("reason"),
                )
                if not confirmation.get("passed"):
                    confirmation_status = str(confirmation.get("status") or "failed")
                    finalist_status = "unstable" if confirmation_status == "runtime_instability" else "failed"
                    finalist_statuses.append(
                        {
                            "candidate_id": dev_summary.candidate_id,
                            "status": finalist_status,
                            "stage": "confirmation",
                            "reason": confirmation.get("reason"),
                            "dev_transform_families": _transform_lineage_families(dev_summary.candidate_id, proposals_log),
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
                        "candidate_id": dev_summary.candidate_id,
                        "reason": "no runtime/output reliability suspicion; holdout is the validation gate",
                        "runtime_reliability_diagnostics": runtime_diagnostic,
                    }
                )
                self._emit_progress(
                    "confirmation_skipped",
                    candidate_id=dev_summary.candidate_id,
                    reason="no runtime/output reliability suspicion",
                )
            self._emit_progress(
                "holdout_candidate_started",
                candidate_id=dev_summary.candidate_id,
                case_count=len(holdout_cases),
            )
            holdout_ready.append((dev_summary, runtime_diagnostic))
        if holdout_ready:
            self._emit_progress(
                "baseline_holdout_started",
                case_count=len(holdout_cases),
                reason="finalist_validation",
            )
            holdout_summaries = self.evaluate_candidates(
                [baseline_candidate, *[dev_summary.candidate for dev_summary, _ in holdout_ready]],
                holdout_cases,
            )
            baseline_holdout = holdout_summaries[compiled_candidate_id(baseline_candidate)]
            self._emit_progress("baseline_holdout_completed", **_summary_progress_fields(baseline_holdout))
        else:
            holdout_summaries = {}
        for dev_summary, runtime_diagnostic in holdout_ready:
            if baseline_holdout is None:
                raise RuntimeError("baseline holdout must be measured before candidate holdout gating.")
            holdout_summary = holdout_summaries[dev_summary.candidate_id]
            holdout_candidates.append(holdout_summary)
            gate = final_gate_status(
                baseline_holdout,
                holdout_summary,
                self.objective,
            )
            comparison = gate.comparison
            passed_gate = gate.validated
            flip_summary = behavior_flip_summary(baseline_holdout, holdout_summary)
            finalist_status = {
                "candidate_id": holdout_summary.candidate_id,
                "status": gate.status,
                "stage": "holdout",
                "reason": gate.reason,
                "dev_transform_families": _transform_lineage_families(dev_summary.candidate_id, proposals_log),
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
                    "candidate_id": holdout_summary.candidate_id,
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
            selected_holdout, frontier_recommendation = select_recommended_candidate(
                promotable_summaries,
                self.objective,
            )
            promoted = True
            selection_reason = str(frontier_recommendation.get("reason") or f"Promoted validated candidate for {self.objective.mode} objective.")
        else:
            promoted = False
            selected_holdout = baseline_holdout
            if baseline_holdout is None:
                selection_reason = "No finalist reached holdout validation; kept original baseline."
                recommended_candidate_id = baseline_dev.candidate_id
            else:
                selection_reason = "No finalist cleared the holdout objective gate; kept original baseline."
                recommended_candidate_id = baseline_holdout.candidate_id
            frontier_recommendation = {
                "recommended_candidate_id": recommended_candidate_id,
                "highest_quality_candidate_id": recommended_candidate_id,
                "reason": selection_reason,
                "validated_candidate_count": 0,
            }

        selected_candidate = selected_holdout.candidate if selected_holdout is not None else baseline_candidate
        selected_candidate_id = selected_holdout.candidate_id if selected_holdout is not None else baseline_dev.candidate_id
        decision_log.append(
            {
                "type": "final_selection",
                "selected_candidate_id": selected_candidate_id,
                "promoted": promoted,
                "reason": selection_reason,
                "best_dev_candidate_id": best_dev_candidate.candidate_id,
                "frontier_recommendation": frontier_recommendation,
            }
        )
        self._emit_progress(
            "run_completed",
            selected_candidate_id=selected_candidate_id,
            promoted=promoted,
            accepted_dev_candidates=len(accepted_dev_candidates),
            holdout_validations=len(holdout_candidates),
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
            accepted_dev_candidates=accepted_dev_candidates,
            holdout_candidates=holdout_candidates,
            decision_log=decision_log,
            finalist_statuses=finalist_statuses,
        )
        manifest = self.build_manifest(
            total_cases=len(cases),
            train_case_count=len(train_cases),
            proposal_example_bank=proposal_example_bank,
            selected_candidate_id=selected_candidate_id,
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
            baseline_candidate=baseline_candidate,
            selected_candidate=selected_candidate,
            selected_candidate_id=selected_candidate_id,
            promoted=promoted,
            baseline_dev=baseline_dev,
            baseline_holdout=baseline_holdout,
            best_dev_candidate=best_dev_candidate,
            selected_holdout=selected_holdout,
            accepted_dev_candidates=accepted_dev_candidates,
            holdout_candidates=holdout_candidates,
            pareto_frontier=pareto_frontier([baseline_holdout, *holdout_candidates])
            if baseline_holdout is not None
            else [],
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

    def _build_research_theory(
        self,
        *,
        current_dev: CandidateSummary,
        evidence_packet: EvidencePacket,
        diagnoses: list[FailureDiagnosis],
        surface: SurfaceSpec,
        search_hypothesis: Any,
        affordances: list[OptimizationAffordance],
        proposals_log: list[dict[str, Any]],
        evidence_ledger: EvidenceLedger,
        decision_log: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        dev_evaluations_used: int,
        proposal_budget: int,
        proposal_retry: bool = False,
    ) -> ResearchTheory:
        attempt = 2 if proposal_retry else 1
        state = {
            "objective": self.objective.to_dict(),
            "budget": {
                "proposal_budget": proposal_budget,
                "dev_evaluations_used": dev_evaluations_used,
                "dev_budget": self.dev_budget,
                "remaining_dev_budget": max(0, self.dev_budget - dev_evaluations_used),
            },
            "parent": {
                "candidate_id": current_dev.candidate_id,
                "score": current_dev.mean_score,
                "pass_count": current_dev.pass_count,
                "case_count": current_dev.case_count,
                "failure_labels": _top_counter_dict(current_dev.failure_labels, limit=12),
                "cost_usd": current_dev.mean_cost_usd,
                "latency_s": current_dev.median_latency_s,
            },
            "evidence_packet": _theorist_evidence_packet(evidence_packet),
            "diagnoses": [_theorist_diagnosis(diagnosis) for diagnosis in diagnoses[:8]],
            "search_hypothesis": _theorist_search_hypothesis(search_hypothesis),
            "surface_spec": _theorist_surface_spec(surface),
            "surface_opportunities": _theorist_affordances(affordances),
            "prior_experiment_outcomes": _compact_prior_stage_results(proposals_log, stage=None, limit=8),
            "evidence_ledger_summary": evidence_ledger.to_dict()["summary"],
            "recent_candidate_history": _compact_recent_history_for_theory(proposals_log, limit=10),
        }
        self._emit_progress(
            "research_theorist_started",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            surface_opportunity_count=len(affordances),
            diagnosis_count=len(diagnoses),
        )
        theory = self.research_theorist.build_theory(
            state=state,
            affordance_ids={affordance.affordance_id for affordance in affordances},
        )
        if self.research_theorist.last_call_diagnostics is not None:
            self.optimizer_call_diagnostics.append(
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "parent_rank": parent_index + 1,
                    "stage": "build_research_theory",
                    **self.research_theorist.last_call_diagnostics,
                }
            )
        decision_log.append(
            {
                "type": "research_theory_call",
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "parent_rank": parent_index + 1,
                "parent_candidate_id": current_dev.candidate_id,
                "research_theory": theory.to_dict(),
                "research_state": state,
            }
        )
        self._emit_progress(
            "research_theorist_completed",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            primary_hypothesis_id=theory.primary_hypothesis_id,
            hypothesis_count=len(theory.hypotheses),
            opportunity_count=len(theory.experiment_opportunities),
            call_diagnostics=self.research_theorist.last_call_diagnostics or {},
        )
        return theory

    def _propose_and_evaluate_parent(
        self,
        *,
        current_dev: CandidateSummary,
        baseline_dev: CandidateSummary,
        dev_cases: tuple[EvalCase, ...],
        surface: SurfaceSpec,
        diagnoses: list[FailureDiagnosis],
        research_theory: ResearchTheory,
        evidence_packet: EvidencePacket,
        diagnosis_analysis: str,
        search_hypothesis: Any,
        current_spec: AgentSpec | None,
        proposal_example_bank: ProposalExampleBank,
        proposal_example_cases: tuple[EvalCase, ...],
        evaluated_candidate_ids: set[str],
        proposals_log: list[dict[str, Any]],
        decision_log: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        parent_summaries: list[CandidateSummary],
        proposal_budget: int,
        dev_evaluations_used: int,
        evidence_ledger: EvidenceLedger,
        experiment_intents: list[Any] | None = None,
        affordances: list[OptimizationAffordance] | None = None,
        proposal_retry: bool = False,
        retry_reason: str | None = None,
    ) -> tuple[list[tuple[CandidateProposal, CandidateSummary, Comparison]], int]:
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
            research_theory=research_theory,
            evidence_packet=evidence_packet,
            seen_hashes=evaluated_candidate_ids,
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
                    "parent_candidate_id": current_dev.candidate_id,
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
                "parent_candidate_id": current_dev.candidate_id,
                "candidate_id": current_dev.candidate_id,
                "frontier_width": SEARCH_FRONTIER_WIDTH,
                "active_frontier": [summary.candidate_id for summary in parent_summaries],
                "diagnosis_analysis": diagnosis_analysis,
                "proposal_analysis": proposal_analysis,
                "proposal_stats": self.candidate_implementer.last_stats.to_dict(),
                "search_hypothesis": search_hypothesis.to_dict(),
                "research_theory": research_theory.to_dict(),
                "evidence_packet": evidence_packet.to_dict(),
                "diagnoses": [diagnosis.to_dict() for diagnosis in diagnoses],
                "diagnosis": target_diagnosis.to_dict() if target_diagnosis else None,
                "proposal_hashes": [transform_program_hash(proposal.program) for proposal in proposals],
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
                "parent_candidate_id": current_dev.candidate_id,
                "candidate_id": current_dev.candidate_id,
                "valid": False,
                **invalid_row,
            }
            proposals_log.append(proposal_row)
        if not proposals:
            return [], 0

        materialization_by_proposal_hash = {
            str(row.get("proposal_program_hash")): dict(row.get("materialization") or {})
            for row in self.candidate_implementer.last_candidate_rows
            if row.get("proposal_program_hash")
        }
        accepted_rows: list[tuple[CandidateProposal, CandidateSummary, Comparison]] = []
        evaluation_states: list[CandidateEvaluationState] = []
        model_candidate_used = False
        model_candidates_allowed = _model_candidate_evidence_present(diagnoses, research_theory)
        for proposal in proposals[:proposal_budget]:
            if proposal.transform_family == "surface_model":
                if model_candidate_used or not model_candidates_allowed:
                    proposals_log.append(
                        {
                            "type": "candidate_proposal",
                            "iteration": iteration,
                            "attempt": attempt,
                            "proposal_retry": proposal_retry,
                            "retry_reason": retry_reason,
                            "parent_rank": parent_index + 1,
                            "parent_candidate_id": current_dev.candidate_id,
                            "candidate_id": current_dev.candidate_id,
                            "valid": True,
                            "proposal": proposal.program.to_dict(),
                            "candidate": proposal.to_dict(),
                            "transform_family": proposal.transform_family,
                            "mechanism_class": proposal.mechanism_class,
                            "accepted": False,
                            "frontier_status": "screened_out",
                            "rejection_reason": "model candidate skipped because no explicit model-capacity evidence remained in budget",
                        }
                    )
                    continue
                model_candidate_used = True
            compiled_candidate = compose_transform_candidate(
                current_dev.candidate,
                proposal.program,
                compiler=self.transform_compiler,
                surface=self._surface(),
            )
            digest = compiled_candidate_id(compiled_candidate)
            if digest in evaluated_candidate_ids:
                continue
            transform_context = TransformContextKey.from_candidate(proposal)
            proposal_digest = transform_program_hash(proposal.program)
            self._emit_progress(
                "candidate_evaluation_started",
                iteration=iteration,
                attempt=attempt,
                parent_rank=parent_index + 1,
                transform_family=proposal.transform_family,
                transform_context=transform_context.to_dict(),
                candidate_id=digest,
                proposal_candidate_id=proposal_digest,
            )
            evaluation_states.append(
                CandidateEvaluationState(
                    proposal=proposal,
                    compiled_candidate=compiled_candidate,
                    candidate_id=digest,
                    proposal_candidate_id=proposal_digest,
                    transform_context=transform_context,
                )
            )
        if not evaluation_states:
            return [], 0

        self._evaluate_candidate_batch_progressively(
            states=evaluation_states,
            reference=current_dev,
            baseline=baseline_dev,
            research_theory=research_theory,
            dev_cases=dev_cases,
            proposals_log=proposals_log,
            decision_log=decision_log,
            dev_evaluations_used=dev_evaluations_used,
            evidence_ledger=evidence_ledger,
            iteration=iteration,
            attempt=attempt,
            parent_index=parent_index,
        )
        evaluations_used = sum(1 for state in evaluation_states if state.summary is not None)
        for state in evaluation_states:
            candidate = state.proposal
            summary = state.summary
            comparison = state.comparison
            flip_summary = state.flip_summary
            if summary is None or comparison is None or flip_summary is None:
                continue
            evaluated_candidate_ids.add(state.candidate_id)
            proposal_row = {
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "retry_reason": retry_reason,
                "parent_rank": parent_index + 1,
                "parent_candidate_id": current_dev.candidate_id,
                "proposal_candidate_id": state.proposal_candidate_id,
                "proposal": candidate.program.to_dict(),
                "proposal_candidate": candidate.to_dict(),
                "materialization": materialization_by_proposal_hash.get(state.proposal_candidate_id, {}),
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
                    evidence_ledger.latest(state.candidate_id).to_dict()
                    if evidence_ledger.latest(state.candidate_id)
                    else {}
                ),
                "evidence_history": [
                    record.to_dict() for record in evidence_ledger.by_candidate(state.candidate_id)
                ],
                "candidate_id": state.candidate_id,
                "compiled_candidate": state.compiled_candidate.to_dict(),
                "candidate": state.compiled_candidate.to_dict(),
                "comparison_to_parent": comparison.to_dict(),
                "behavior_flip_summary": flip_summary,
                "metrics": summary.to_dict(),
                "accepted": state.accepted,
                "frontier_status": state.frontier_status,
                "rejection_reason": state.rejection_reason,
                "constraint_warning": state.constraint_warning,
                "full_dev_evaluated": state.full_dev_evaluated,
                "diagnosis_category": candidate.program.metadata.get("diagnosis_category"),
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
                candidate_id=state.candidate_id,
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
                if candidate.mechanism_class in {"surface_runtime", "surface_output", "surface_response"}:
                    decision_log.append(
                        {
                            "type": "residual_rediagnosis_triggered",
                            "candidate_id": state.candidate_id,
                            "parent_candidate_id": current_dev.candidate_id,
                            "mechanism_class": candidate.mechanism_class,
                            "reason": "structural/runtime fix accepted; child branch should be rediagnosed for residual failures",
                        }
                    )
        return accepted_rows, evaluations_used

    def _plan_parent_research_action(
        self,
        *,
        current_dev: CandidateSummary,
        research_theory: ResearchTheory,
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
        research_theory_payload = research_theory.to_dict()
        state = ResearchState(
            objective=self.objective.to_dict(),
            budget={
                "proposal_budget": proposal_budget,
                "dev_evaluations_used": dev_evaluations_used,
                "dev_budget": self.dev_budget,
                "remaining_dev_budget": max(0, self.dev_budget - dev_evaluations_used),
            },
            parent={
                "candidate_id": current_dev.candidate_id,
                "score": current_dev.mean_score,
                "pass_count": current_dev.pass_count,
                "case_count": current_dev.case_count,
                "failure_labels": _top_counter_dict(current_dev.failure_labels, limit=8),
                "cost_usd": current_dev.mean_cost_usd,
                "latency_s": current_dev.median_latency_s,
            },
            task_theory=research_theory_payload,
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
            opportunity_count=len(research_theory.experiment_opportunities),
            surface_opportunity_count=len(affordances),
        )
        try:
            intents = self.research_planner.plan(state)
        except OptimizerModelError as exc:
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
                    "experiment_intents": [],
                    "error": str(exc),
                }
            )
            self._emit_progress(
                "research_planner_completed",
                iteration=iteration,
                attempt=attempt,
                parent_rank=parent_index + 1,
                stage="plan_experiments",
                intent_count=0,
                mechanisms=[],
                experiment_intents=[],
                error=str(exc),
                call_diagnostics=self.research_planner.last_call_diagnostics or {},
            )
            return []
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
        reference: CandidateSummary,
        baseline: CandidateSummary,
        research_theory: ResearchTheory,
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
        stages = self._progressive_eval_stages(reference, dev_cases)
        has_small_dev_stage = any(stage_name == "small_dev" for stage_name, _ in stages)
        for stage_name, stage_cases in stages:
            if not active:
                break
            active = self._filter_candidate_stage_by_measurement_budget(
                active,
                reference=reference,
                stage_name=stage_name,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
            )
            if not active:
                break
            if stage_name == "full_dev":
                if has_small_dev_stage:
                    next_active = []
                    for state in active:
                        if _eligible_for_full_dev_from_small_signal(state):
                            next_active.append(state)
                            continue
                        state.rejection_reason = "full_dev skipped because small-dev evidence did not show objective gain"
                        state.frontier_status = "screened_out"
                        state.accepted = False
                    active = next_active
                    if not active:
                        break
                if _has_evidence_for_selector(evidence_ledger, active):
                    active = self._select_candidate_stage_with_measurement_selector(
                        active,
                        baseline,
                        research_theory=research_theory,
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
                    research_theory=research_theory,
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
                candidate_ids=[state.candidate_id for state in active],
            )
            reference_summary = _summary_for_cases(reference, stage_cases) or self.evaluate_candidate(reference.candidate, stage_cases)
            baseline_summary = _summary_for_cases(baseline, stage_cases) or self.evaluate_candidate(baseline.candidate, stage_cases)
            candidate_summaries = self.evaluate_candidates(
                [state.compiled_candidate for state in active],
                stage_cases,
            )
            next_active: list[CandidateEvaluationState] = []
            for state in active:
                candidate_summary = candidate_summaries[state.candidate_id]
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
                    candidate_id=state.candidate_id,
                    stage=stage_name,
                    reference=reference_summary,
                    baseline=baseline_summary,
                    candidate=candidate_summary,
                    mechanism_class=state.proposal.mechanism_class,
                    affordance_ids=list(state.proposal.affordance_ids),
                    comparison_group=state.proposal.comparison_group,
                    candidate_role=state.proposal.candidate_role,
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
                        "candidate_id": candidate_summary.candidate_id,
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
                    _finalize_candidate_state(state, reference, self.objective)
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

    def _filter_candidate_stage_by_measurement_budget(
        self,
        states: list[CandidateEvaluationState],
        *,
        reference: CandidateSummary,
        stage_name: str,
        stage_cases: tuple[EvalCase, ...],
        evidence_ledger: EvidenceLedger,
    ) -> list[CandidateEvaluationState]:
        kept: list[CandidateEvaluationState] = []
        selected_measurement_cost = 0.0
        selected_measurement_tool_calls = 0.0
        selected_measurement_turns = 0.0
        for state in states:
            marginal_case_count = _marginal_case_count(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
            )
            marginal_cost = _estimated_marginal_measurement_cost_usd(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
                samples_per_case=self.samples_per_case,
                reference=reference,
            )
            marginal_tool_calls = _estimated_marginal_measurement_units(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
                samples_per_case=self.samples_per_case,
                unit="tool_calls",
                reference=reference,
            )
            marginal_turns = _estimated_marginal_measurement_units(
                state=state,
                stage_cases=stage_cases,
                evidence_ledger=evidence_ledger,
                samples_per_case=self.samples_per_case,
                unit="turns",
                reference=reference,
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
            if budget_reason is None and marginal_case_count > 0:
                budget_reason = _closed_measurement_budget_reason(
                    used_usd=self._dev_measurement_cost_usd + selected_measurement_cost,
                    max_usd=self.max_dev_measurement_cost_usd,
                    used_tool_calls=self._dev_measurement_tool_calls + selected_measurement_tool_calls,
                    max_tool_calls=self.max_dev_measurement_tool_calls,
                    used_turns=self._dev_measurement_turns + selected_measurement_turns,
                    max_turns=self.max_dev_measurement_turns,
                    stage=stage_name,
            )
            if budget_reason is not None:
                state.rejection_reason = f"measurement_budget_exhausted: {budget_reason}"
                state.frontier_status = "screened_out"
                state.accepted = False
                continue
            selected_measurement_cost += marginal_cost
            selected_measurement_tool_calls += marginal_tool_calls
            selected_measurement_turns += marginal_turns
            kept.append(state)
        return kept

    def _select_candidate_stage_with_measurement_selector(
        self,
        states: list[CandidateEvaluationState],
        baseline: CandidateSummary,
        *,
        research_theory: ResearchTheory,
        reference: CandidateSummary,
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
            research_theory=research_theory,
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
        selected = [state for state in states if state.candidate_id in selected_ids]
        for state in states:
            if state.candidate_id in selected_ids:
                continue
            state.rejection_reason = (
                decision.skipped_candidate_reasons.get(state.candidate_id)
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
        state_by_id = {state.candidate_id: state for state in states}
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
        candidate: CompiledCandidate,
        reference: CandidateSummary,
        baseline: CandidateSummary,
        dev_cases: tuple[EvalCase, ...],
    ) -> tuple[CandidateSummary, Comparison, dict[str, Any], str | None, list[dict[str, Any]]]:
        stage_rows: list[dict[str, Any]] = []
        final_summary: CandidateSummary | None = None
        final_comparison: Comparison | None = None
        final_flip_summary: dict[str, Any] | None = None
        final_rejection_reason: str | None = None
        for stage_name, stage_cases in self._progressive_eval_stages(reference, dev_cases):
            reference_summary = _summary_for_cases(reference, stage_cases) or self.evaluate_candidate(reference.candidate, stage_cases)
            baseline_summary = _summary_for_cases(baseline, stage_cases) or self.evaluate_candidate(baseline.candidate, stage_cases)
            candidate_summary = self.evaluate_candidate(candidate, stage_cases)
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
                    "candidate_id": candidate_summary.candidate_id,
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
        reference: CandidateSummary,
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
        # Keep exploration cheap: small-dev must stay a triage slice even when
        # the baseline has many failures, otherwise failure-heavy suites jump
        # straight from smoke to full-dev.
        small_target = min(len(dev_cases) - 1, max(6, min(12, (len(dev_cases) + 1) // 2)))
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

    def evaluate_candidate(
        self,
        candidate: CompiledCandidate | None,
        cases: tuple[EvalCase, ...],
        *,
        sample_indices: Iterable[int] | None = None,
    ) -> CandidateSummary:
        candidate_id = compiled_candidate_id(candidate)
        return self.evaluate_candidates([candidate], cases, sample_indices=sample_indices)[candidate_id]

    def evaluate_candidates(
        self,
        candidates: Iterable[CompiledCandidate | None],
        cases: tuple[EvalCase, ...],
        *,
        sample_indices: Iterable[int] | None = None,
    ) -> dict[str, CandidateSummary]:
        candidate_by_id: dict[str, CompiledCandidate | None] = {}
        for candidate in candidates:
            candidate_by_id.setdefault(compiled_candidate_id(candidate), candidate)
        if not candidate_by_id:
            return {}
        indices = tuple(sample_indices) if sample_indices is not None else tuple(range(self.samples_per_case))
        if not indices:
            raise ValueError("sample_indices must not be empty.")
        ordered_by_digest: dict[str, list[CaseEvaluation | None]] = {
            digest: [None] * (len(cases) * len(indices))
            for digest in candidate_by_id
        }
        uncached: list[tuple[str, CompiledCandidate | None, int, EvalCase, int]] = []
        fresh_by_digest: Counter[str] = Counter()
        for digest, candidate in candidate_by_id.items():
            order = 0
            ordered = ordered_by_digest[digest]
            for case in cases:
                for sample_index in indices:
                    with self._store_lock:
                        cached = self.store.get(digest, case, sample_index=sample_index, candidate=candidate)
                    if cached is not None:
                        with self._stats_lock:
                            self.stats.cache_hits += 1
                            if cached.cache_source == "shared":
                                self.stats.shared_cache_hits += 1
                            else:
                                self.stats.local_cache_hits += 1
                        ordered[order] = cached
                        self._emit_progress(
                            "case_cache_hit",
                            candidate_id=digest,
                            case_id=case.id,
                            split=case.split,
                            sample_index=sample_index,
                            cache_source=cached.cache_source,
                        )
                    else:
                        fresh_by_digest[digest] += 1
                        uncached.append((digest, candidate, order, case, sample_index))
                    order += 1
        if uncached:
            concurrency_limit = self.stage_case_concurrency if len(candidate_by_id) > 1 else self.case_concurrency
            effective_concurrency = 1 if self.fail_fast else min(concurrency_limit, len(uncached))
            effective_concurrency = min(
                effective_concurrency,
                _candidate_batch_concurrency_limit(candidate_by_id.values()),
            )
            for digest, fresh_count in fresh_by_digest.items():
                self._emit_progress(
                    "case_batch_started",
                    candidate_id=digest,
                    split=cases[0].split,
                    case_count=len(cases),
                    sample_count=len(indices),
                    fresh_count=fresh_count,
                    concurrency=effective_concurrency,
                    parallel_candidate_count=len(candidate_by_id),
                )
            if effective_concurrency == 1:
                for digest, candidate, item_order, case, sample_index in uncached:
                    ordered_by_digest[digest][item_order] = self._run_uncached_case(
                        digest,
                        candidate,
                        case,
                        sample_index=sample_index,
                    )
            else:
                futures: dict[Future[CaseEvaluation], tuple[str, CompiledCandidate | None, int, EvalCase, int]] = {}
                with ThreadPoolExecutor(max_workers=effective_concurrency) as executor:
                    for digest, candidate, item_order, case, sample_index in uncached:
                        self._emit_progress(
                            "case_started",
                            candidate_id=digest,
                            case_id=case.id,
                            split=case.split,
                            sample_index=sample_index,
                        )
                        future = executor.submit(self._execute_case, candidate, case, sample_index=sample_index)
                        futures[future] = (digest, candidate, item_order, case, sample_index)
                    for future in as_completed(futures):
                        digest, candidate, item_order, case, sample_index = futures[future]
                        evaluation = future.result()
                        with self._store_lock:
                            self.store.put(digest, candidate, evaluation)
                        self._emit_case_completed(digest, evaluation)
                        ordered_by_digest[digest][item_order] = evaluation
            for digest, fresh_count in fresh_by_digest.items():
                self._emit_progress(
                    "case_batch_completed",
                    candidate_id=digest,
                    split=cases[0].split,
                    fresh_count=fresh_count,
                    concurrency=effective_concurrency,
                    parallel_candidate_count=len(candidate_by_id),
                )
        summaries: dict[str, CandidateSummary] = {}
        for digest, candidate in candidate_by_id.items():
            ordered = ordered_by_digest[digest]
            evaluations = [evaluation for evaluation in ordered if evaluation is not None]
            if len(evaluations) != len(ordered):
                raise RuntimeError("internal evaluation error: missing case evaluation result")
            summaries[digest] = CandidateSummary(
                candidate_id=digest,
                candidate=candidate,
                split=cases[0].split,
                evaluations=evaluations,
            )
        return summaries

    def _run_uncached_case(
        self,
        digest: str,
        candidate: CompiledCandidate | None,
        case: EvalCase,
        *,
        sample_index: int,
    ) -> CaseEvaluation:
        self._emit_progress(
            "case_started",
            candidate_id=digest,
            case_id=case.id,
            split=case.split,
            sample_index=sample_index,
        )
        evaluation = self._execute_case(candidate, case, sample_index=sample_index)
        with self._store_lock:
            self.store.put(digest, candidate, evaluation)
        self._emit_case_completed(digest, evaluation)
        if self.fail_fast and evaluation.record.metrics.error:
            raise RuntimeError(
                f"Fail-fast stopping after case {case.id}: {evaluation.record.metrics.error}"
            )
        return evaluation

    def _emit_case_completed(self, digest: str, evaluation: CaseEvaluation) -> None:
        self._emit_progress(
            "case_completed",
            candidate_id=digest,
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

    def _execute_case(self, candidate: CompiledCandidate | None, case: EvalCase, *, sample_index: int = 0) -> CaseEvaluation:
        total_attempts = self.max_case_retries + 1
        started_at = time.perf_counter()
        last_error: Exception | None = None
        last_phase = "run_case"
        for attempt in range(1, total_attempts + 1):
            try:
                last_phase = "run_case"
                with case_timeout(self.case_timeout_s), model_request_limits(
                    timeout_s=self.case_timeout_s,
                    max_attempts=1,
                ):
                    record = self.adapter.run_case(case, candidate)
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
        selected_candidate_id: str,
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
            "selected_candidate_id": selected_candidate_id,
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


def _same_cases(summary: CandidateSummary, cases: tuple[EvalCase, ...]) -> bool:
    return tuple(summary.grouped_evaluations) == tuple(case.id for case in cases)


def _summary_for_cases(summary: CandidateSummary, cases: tuple[EvalCase, ...]) -> CandidateSummary | None:
    grouped = summary.grouped_evaluations
    selected: list[CaseEvaluation] = []
    for case in cases:
        evaluations = grouped.get(case.id)
        if not evaluations:
            return None
        selected.extend(evaluations)
    return CandidateSummary(
        candidate_id=summary.candidate_id,
        candidate=summary.candidate,
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


def _truncate_text(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _theorist_evidence_packet(packet: EvidencePacket) -> dict[str, Any]:
    raw = packet.to_dict()
    diagnostics = raw.get("behavior_diagnostics") or {}
    runtime = raw.get("runtime_defects") or {}
    output = raw.get("output_defects") or {}
    tool = raw.get("tool_defects") or {}
    category_metrics = diagnostics.get("category_metrics") or {}
    per_label = diagnostics.get("per_label") or []
    return {
        "residual_failure_modes": list(raw.get("residual_failure_modes") or [])[:8],
        "diagnosis_categories": list(raw.get("diagnosis_categories") or [])[:8],
        "evidence": list(raw.get("evidence") or [])[:8],
        "confidence": raw.get("confidence"),
        "weak_slices": list(raw.get("weak_slices") or [])[:8],
        "label_confusions": [
            {
                "expected": row.get("expected"),
                "actual": row.get("actual"),
                "count": row.get("count"),
                "case_ids": list(row.get("case_ids") or [])[:3],
            }
            for row in list(raw.get("label_confusions") or [])[:8]
            if isinstance(row, dict)
        ],
        "runtime_defects": {
            "finish_reason_counts": runtime.get("finish_reason_counts", {}),
            "length_finish_case_ids": list(runtime.get("length_finish_case_ids") or [])[:8],
            "parser_fallback_case_ids": list(runtime.get("parser_fallback_case_ids") or [])[:8],
            "low_output_token_length_case_ids": list(runtime.get("low_output_token_length_case_ids") or [])[:8],
        },
        "output_defects": {
            "invalid_output_count": output.get("invalid_output_count", 0),
            "invalid_output_case_ids": list(output.get("invalid_output_case_ids") or [])[:10],
        },
        "tool_defects": {
            "tool_call_counts": tool.get("tool_call_counts", {}),
            "tool_status_counts": tool.get("tool_status_counts", {}),
            "turn_outcome_counts": tool.get("turn_outcome_counts", {}),
            "terminal_reason_counts": tool.get("terminal_reason_counts", {}),
            "tool_error_case_ids": list(tool.get("tool_error_case_ids") or [])[:8],
            "invalid_tool_call_case_ids": list(tool.get("invalid_tool_call_case_ids") or [])[:8],
            "premature_stop_case_ids": list(tool.get("premature_stop_case_ids") or [])[:8],
        },
        "example_coverage": raw.get("example_coverage") or {},
        "cost_latency_profile": raw.get("cost_latency_profile") or {},
        "behavior_summary": {
            "category_metrics": category_metrics,
            "weakest_labels": [
                {
                    "label": row.get("label"),
                    "support": row.get("support"),
                    "pass_rate": row.get("pass_rate"),
                    "case_ids": list(row.get("case_ids") or [])[:4],
                }
                for row in list(per_label)[:8]
                if isinstance(row, dict)
            ],
        },
    }


def _theorist_diagnosis(diagnosis: FailureDiagnosis) -> dict[str, Any]:
    return {
        "case_ids": list(diagnosis.case_ids)[:8],
        "category": diagnosis.category,
        "root_cause": _truncate_text(diagnosis.root_cause, limit=500),
        "target_names": list(diagnosis.target_names)[:8],
        "evidence": [
            {
                str(key): (
                    _truncate_text(value, limit=240)
                    if isinstance(value, str)
                    else value
                )
                for key, value in row.items()
                if key in {"case_id", "expected", "actual", "score", "passed", "notes", "labels"}
            }
            for row in diagnosis.evidence[:6]
            if isinstance(row, dict)
        ],
    }


def _theorist_search_hypothesis(search_hypothesis: Any) -> dict[str, Any]:
    prompt = search_hypothesis.to_prompt_dict(
        max_contexts_per_family=1,
        max_constrained_contexts=3,
    )
    family_states = {}
    for name, row in (prompt.get("family_states") or {}).items():
        if not isinstance(row, dict):
            continue
        family_states[name] = {
            "state": row.get("state"),
            "suitability": row.get("suitability"),
            "budget_share": row.get("budget_share"),
            "constraints": list(row.get("constraints") or [])[:3],
        }
    return {
        "active_families": list(prompt.get("active_families") or [])[:8],
        "target_slices": list(prompt.get("target_slices") or [])[:8],
        "family_states": family_states,
        "active_contexts": [
            _theorist_context_row(row)
            for row in list(prompt.get("active_contexts") or [])[:8]
            if isinstance(row, dict)
        ],
        "constrained_or_paused_contexts": [
            _theorist_context_row(row)
            for row in list(prompt.get("constrained_or_paused_contexts") or [])[:3]
            if isinstance(row, dict)
        ],
        "profile": prompt.get("profile") or {},
        "budget_allocation": prompt.get("budget_allocation") or {},
        "rationale": _truncate_text(prompt.get("rationale"), limit=500),
    }


def _theorist_context_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "family": row.get("family"),
        "state": row.get("state"),
        "target_names": list(row.get("target_names") or [])[:4],
        "target_slice": row.get("target_slice"),
        "ops": list(row.get("ops") or [])[:4],
        "suitability": row.get("suitability"),
        "accepted_count": row.get("accepted_count"),
        "rejected_count": row.get("rejected_count"),
        "constraints": list(row.get("constraints") or [])[:3],
    }


def _theorist_affordances(
    affordances: list[OptimizationAffordance],
    *,
    limit: int = 36,
) -> list[dict[str, Any]]:
    ranked = sorted(affordances, key=lambda item: (-item.suitability, item.affordance_id))
    selected: list[OptimizationAffordance] = []
    seen: set[str] = set()
    for key_fn, per_group in (
        (lambda item: item.mechanism, 2),
        (lambda item: item.family, 1),
    ):
        counts: dict[str, int] = {}
        for affordance in ranked:
            group = str(key_fn(affordance))
            if counts.get(group, 0) >= per_group or affordance.affordance_id in seen:
                continue
            selected.append(affordance)
            seen.add(affordance.affordance_id)
            counts[group] = counts.get(group, 0) + 1
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    for affordance in ranked:
        if len(selected) >= limit:
            break
        if affordance.affordance_id in seen:
            continue
        selected.append(affordance)
        seen.add(affordance.affordance_id)
    return [
        {
            "surface_opportunity_id": affordance.affordance_id,
            "surface": affordance.mechanism,
            "target": affordance.target_name,
            "target_kind": affordance.target_kind,
            "target_path": affordance.target_path,
            "ops": list(affordance.ops),
            "semantic_role": affordance.semantic_role,
            "behavioral_axes": list(affordance.behavioral_axes)[:4],
            "expected_scope": affordance.expected_scope,
            "risk": affordance.risk,
            "measurements": list(affordance.measurements)[:5],
            "suitability": affordance.suitability,
            "evidence": list(affordance.evidence)[:3],
            "expected_cost_impact": affordance.expected_cost_impact,
            "expected_latency_impact": affordance.expected_latency_impact,
        }
        for affordance in selected
    ]


def _theorist_surface_spec(surface: SurfaceSpec) -> dict[str, Any]:
    return {
        "agent_id": surface.agent_id,
        "context_sections": [
            {
                "name": section.name,
                "role": section.role,
                "required": section.required,
                "editable": section.name in surface.context.editable_sections,
                "value_shape": _value_shape(section.content),
            }
            for section in surface.context.graph.sections
        ],
        "context_capabilities": {
            "generated_sections_allowed": surface.context.generated_sections_allowed,
            "removable_sections_allowed": surface.context.removable_sections_allowed,
            "reorderable_sections_allowed": surface.context.reorderable_sections_allowed,
        },
        "hooks": {
            name: {
                "available_inputs": list(hook.available_inputs),
                "allowed_ops": list(hook.allowed_ops),
                "method": hook.method,
            }
            for name, hook in sorted(surface.hooks.items())
            if hook.supported
        },
        "state": surface.state.to_dict(),
        "tools": surface.tools.to_dict(),
        "model": surface.model.to_dict(),
        "response": surface.response.to_dict(),
        "immutable_boundaries": list(surface.immutable_boundaries),
        "safety_constraints": list(surface.safety_constraints),
    }


def _value_shape(value: Any) -> Any:
    if isinstance(value, str):
        return {"type": "string", "chars": len(value), "prefix": value[:240]}
    if isinstance(value, list):
        return {"type": "list", "count": len(value), "sample": value[:3]}
    if isinstance(value, dict):
        return {"type": "object", "keys": sorted(str(key) for key in value.keys())[:16]}
    return value


def _smoke_rejection_reason(reference: CandidateSummary, candidate: CandidateSummary) -> str | None:
    if candidate.runtime_error_count > reference.runtime_error_count:
        return "smoke rejected candidate because runtime errors increased"
    if candidate.pass_count < reference.pass_count:
        return "smoke rejected candidate because pass count regressed"
    return None


def _candidate_batch_concurrency_limit(candidates: Iterable[CompiledCandidate | None]) -> int:
    limits = [
        limit
        for candidate in candidates
        if (limit := _compiled_candidate_concurrency_limit(candidate)) is not None
    ]
    return min(limits) if limits else 10_000


def _compiled_candidate_concurrency_limit(candidate: CompiledCandidate | None) -> int | None:
    model_name = _compiled_candidate_model_name(candidate)
    if model_name is None:
        return None
    return _model_name_concurrency_limit(model_name)


def _compiled_candidate_model_name(candidate: CompiledCandidate | None) -> str | None:
    if candidate is None:
        return None
    for patch in candidate.program.patches:
        if patch.op.op != "set_model_config":
            continue
        if patch.op.params.get("field") != "model_name":
            continue
        value = patch.op.params.get("value")
        if isinstance(value, str) and value:
            return value
    return None


def _model_name_concurrency_limit(model_name: str) -> int | None:
    normalized = model_name.lower()
    if normalized.startswith("gemini-") and "pro" in normalized:
        return 1
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
        candidate_ids=[state.candidate_id for state in states],
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
    return all(evidence_ledger.latest(state.candidate_id) is not None for state in states)


def _research_state_packet(
    *,
    objective: OptimizationObjective,
    stage_name: str,
    reference: CandidateSummary,
    baseline: CandidateSummary,
    states: list[CandidateEvaluationState],
    research_theory: ResearchTheory | None = None,
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
    candidate_ids = [state.candidate_id for state in states]
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
            "candidate_id": reference.candidate_id,
            "score": reference.mean_score,
            "pass_count": reference.pass_count,
            "case_count": reference.case_count,
            "cost_usd": reference.mean_cost_usd,
            "latency_s": reference.median_latency_s,
        },
        "baseline": {
            "candidate_id": baseline.candidate_id,
            "score": baseline.mean_score,
            "pass_count": baseline.pass_count,
            "case_count": baseline.case_count,
            "cost_usd": baseline.mean_cost_usd,
            "latency_s": baseline.median_latency_s,
        },
        "research_theory": research_theory.to_dict() if research_theory is not None else {},
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
    reference: CandidateSummary,
    baseline: CandidateSummary,
    samples_per_case: int,
    measurement_cost_used_usd: float,
    max_measurement_cost_usd: float | None,
    measurement_tool_calls_used: float,
    max_measurement_tool_calls: int | None,
    measurement_turns_used: float,
    max_measurement_turns: int | None,
) -> list[dict[str, Any]]:
    state_by_id = {state.candidate_id: state for state in states}
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
        "candidate_id": state.candidate_id,
        "transform_family": state.proposal.transform_family,
        "mechanism_class": state.proposal.mechanism_class,
        "candidate_role": state.proposal.candidate_role,
        "comparison_group": state.proposal.comparison_group,
        "target_slice": state.proposal.target_slice,
        "transform_instance": state.proposal.transform_instance,
        "hypothesis": state.proposal.hypothesis,
        "operation_count": len(state.compiled_candidate.program.patches),
        "operations": [
            {"op": operation.op.op, "target": operation.hook or "on_task_start"}
            for operation in state.compiled_candidate.program.patches
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
                "candidate_id": item.get("candidate_id"),
                "parent_candidate_id": item.get("parent_candidate_id"),
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


def _compact_recent_history_for_theory(
    proposals_log: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in reversed(proposals_log):
        if not item.get("candidate"):
            continue
        stages = item.get("evaluation_stages") or []
        latest_stage = stages[-1] if stages else {}
        if not isinstance(latest_stage, dict):
            latest_stage = {}
        comparison = latest_stage.get("comparison_to_parent") or {}
        if not isinstance(comparison, dict):
            comparison = {}
        flip_summary = item.get("behavior_flip_summary") or {}
        if not isinstance(flip_summary, dict):
            flip_summary = {}
        rows.append(
            {
                "candidate_id": item.get("candidate_id"),
                "parent_candidate_id": item.get("parent_candidate_id"),
                "hypothesis": _truncate_text(item.get("hypothesis"), limit=320),
                "expected_effects": _truncate_text(item.get("expected_effects"), limit=240),
                "mechanism_class": item.get("mechanism_class"),
                "transform_family": item.get("transform_family"),
                "candidate_role": item.get("candidate_role"),
                "target_slice": item.get("target_slice"),
                "accepted": item.get("accepted"),
                "frontier_status": item.get("frontier_status"),
                "rejection_reason": item.get("rejection_reason"),
                "latest_stage": {
                    "stage": latest_stage.get("stage"),
                    "case_count": latest_stage.get("case_count"),
                    "passed": latest_stage.get("passed"),
                    "score_delta": comparison.get("score_delta"),
                    "pass_rate_delta": comparison.get("pass_rate_delta"),
                    "cost_delta": comparison.get("cost_delta"),
                    "latency_delta": comparison.get("latency_delta"),
                    "token_delta": comparison.get("token_delta"),
                    "rejection_reason": latest_stage.get("rejection_reason"),
                },
                "behavior_flips": {
                    "fixed_count": flip_summary.get("fixed_count"),
                    "regressed_count": flip_summary.get("regressed_count"),
                    "invalid_output_delta": flip_summary.get("invalid_output_delta"),
                    "finish_reason_delta": flip_summary.get("finish_reason_delta"),
                },
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _candidate_research_group(state: CandidateEvaluationState) -> str:
    comparison_group = (
        state.proposal.comparison_group
        or state.proposal.experiment_id
        or state.proposal.transform_family
    )
    return f"{comparison_group}|{state.proposal.target_slice or 'global'}"


def _states_by_research_group(
    states: list[CandidateEvaluationState],
) -> dict[str, list[CandidateEvaluationState]]:
    by_group: dict[str, list[CandidateEvaluationState]] = {}
    for state in states:
        by_group.setdefault(_candidate_research_group(state), []).append(state)
    return by_group


def _finalize_candidate_state(
    state: CandidateEvaluationState,
    reference: CandidateSummary,
    objective: OptimizationObjective,
) -> None:
    if state.summary is None:
        return
    if state.rejection_reason is None and state.constraint_warning is None:
        state.frontier_status = "promotable_dev"
    elif state.rejection_reason is None and state.constraint_warning is not None:
        state.frontier_status = "quality_frontier"
    elif _efficiency_improved(reference, state.summary):
        state.frontier_status = "efficiency_frontier"
    else:
        state.frontier_status = "failed"
    state.accepted = state.frontier_status == "promotable_dev"


def _eligible_for_full_dev_from_small_signal(state: CandidateEvaluationState) -> bool:
    small_stage = next(
        (
            row
            for row in reversed(state.stage_rows)
            if isinstance(row, dict) and row.get("stage") == "small_dev"
        ),
        None,
    )
    if small_stage is None:
        return True
    comparison = small_stage.get("comparison_to_parent") or {}
    behavior = small_stage.get("behavior_flip_summary") or {}
    if float(comparison.get("score_delta") or 0.0) > 0:
        return True
    if int(behavior.get("fixed_count") or 0) > int(behavior.get("regressed_count") or 0):
        return True
    labels = ((small_stage.get("metrics") or {}).get("behavioral") or {}).get("failure_labels") or {}
    if isinstance(labels, dict) and int(labels.get("invalid_output") or 0) == 0 and int(behavior.get("regressed_count") or 0) == 0:
        return True
    return False


def _model_candidate_evidence_present(
    diagnoses: list[FailureDiagnosis],
    research_theory: ResearchTheory,
) -> bool:
    for hypothesis in research_theory.hypotheses:
        if hypothesis.mechanism_class == "surface_model":
            return True
        text = " ".join([hypothesis.hypothesis_id, hypothesis.statement, *hypothesis.supporting_evidence]).lower()
        if "model" in text and any(token in text for token in ("capacity", "reasoning", "capability")):
            return True
    for diagnosis in diagnoses:
        text = json.dumps(diagnosis.to_dict(), sort_keys=True, default=str).lower()
        if "model" in text and any(token in text for token in ("capacity", "reasoning", "capability")):
            return True
    return False


def _efficiency_improved(reference: CandidateSummary, candidate: CandidateSummary) -> bool:
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
    reference: CandidateSummary | None = None,
) -> float:
    summary = state.summary or reference
    if summary is None:
        return 0.0
    marginal_case_count = _marginal_case_count(
        state=state,
        stage_cases=stage_cases,
        evidence_ledger=evidence_ledger,
    )
    return max(0.0, summary.mean_cost_usd * marginal_case_count * max(1, samples_per_case))


def _estimated_marginal_measurement_units(
    *,
    state: CandidateEvaluationState,
    stage_cases: tuple[EvalCase, ...],
    evidence_ledger: EvidenceLedger,
    samples_per_case: int,
    unit: str,
    reference: CandidateSummary | None = None,
) -> float:
    summary = state.summary or reference
    if summary is None:
        return 0.0
    marginal_case_count = _marginal_case_count(
        state=state,
        stage_cases=stage_cases,
        evidence_ledger=evidence_ledger,
    )
    multiplier = max(1, samples_per_case) * marginal_case_count
    if unit == "model_calls":
        return max(0.0, summary.mean_model_calls * multiplier)
    if unit == "tool_calls":
        return max(0.0, summary.mean_tool_calls * multiplier)
    if unit == "turns":
        return max(0.0, summary.mean_turns * multiplier)
    raise ValueError(f"Unsupported marginal measurement unit: {unit}")


def _marginal_case_count(
    *,
    state: CandidateEvaluationState,
    stage_cases: tuple[EvalCase, ...],
    evidence_ledger: EvidenceLedger,
) -> int:
    latest = evidence_ledger.latest(state.candidate_id)
    previously_measured = set(latest.case_ids) if latest is not None else set()
    return sum(1 for case in stage_cases if case.id not in previously_measured)


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


def _closed_measurement_budget_reason(
    *,
    used_usd: float,
    max_usd: float | None,
    used_tool_calls: float,
    max_tool_calls: int | None,
    used_turns: float,
    max_turns: int | None,
    stage: str,
) -> str | None:
    if max_usd is not None and used_usd >= max_usd - 1e-12:
        return f"no remaining measurement dollar budget for {stage}"
    if max_tool_calls is not None and used_tool_calls >= max_tool_calls - 1e-12:
        return f"no remaining measurement tool-call budget for {stage}"
    if max_turns is not None and used_turns >= max_turns - 1e-12:
        return f"no remaining measurement turn budget for {stage}"
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


def _requires_finalist_confirmation(candidate: CompiledCandidate | None, runtime_diagnostic: dict[str, Any]) -> bool:
    if runtime_diagnostic.get("baseline_runtime_defect_fixed"):
        return True
    if runtime_diagnostic.get("fixed_invalid_output_case_ids") and _touches_output_or_runtime(candidate):
        return True
    if runtime_diagnostic.get("finish_reason_changed_case_ids") and _touches_output_or_runtime(candidate):
        return True
    return False


def _touches_output_or_runtime(candidate: CompiledCandidate | None) -> bool:
    if candidate is None:
        return False
    for operation in candidate.program.patches:
        target = str(operation.op.params.get("section") or operation.op.params.get("field") or "")
        if operation.op.op == "set_model_config" and target in {"max_tokens", "temperature", "reasoning_effort"}:
            return True
        if operation.op.op in {"rewrite_response", "block_response", "validate_claims"}:
            return True
        if target == "output_contract" or target.startswith("output"):
            return True
    return False


def _transform_lineage_families(candidate_id_value: str, proposals: list[dict[str, Any]]) -> list[str]:
    row_by_candidate = {
        str(row.get("candidate_id")): row
        for row in proposals
        if row.get("accepted") is not None and row.get("candidate_id")
    }
    families: list[str] = []
    seen: set[str] = set()
    cursor = candidate_id_value
    while cursor and cursor not in seen:
        seen.add(cursor)
        row = row_by_candidate.get(cursor)
        if row is None:
            break
        family = row.get("transform_family")
        if isinstance(family, str) and family and family not in families:
            families.append(family)
        parent = row.get("parent_candidate_id")
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
                {"validated": 0, "directional": 0, "failed": 0, "unstable": 0, "finalist_count": 0, "candidate_ids": []},
            )
            status = str(row.get("status") or "failed")
            if status not in {"validated", "directional", "failed", "unstable"}:
                status = "failed"
            summary[status] += 1
            summary["finalist_count"] += 1
            if row.get("candidate_id"):
                summary["candidate_ids"].append(row.get("candidate_id"))
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
    current_dev: CandidateSummary,
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
    model_affordances = [
        affordance
        for affordance in affordances
        if affordance.mechanism == "surface_model"
    ]
    if (
        model_affordances
        and "surface_model" not in existing_mechanisms
        and current_dev.pass_count < current_dev.case_count
        and _residual_quality_signal_remains(task_theory, proposals_log, current_dev.candidate_id)
    ):
        opportunities.insert(
            _capability_probe_insert_index(opportunities),
            {
                "mechanism_class": "surface_model",
                "target_slices": ["global", "failed_cases"],
                "candidate_roles": ["atomic", "control"],
                "measurements": ["score_delta", "fixed_case_ids", "regressed_case_ids", "cost_delta", "latency_delta"],
                "affordance_ids": [affordance.affordance_id for affordance in model_affordances[:3]],
                "rationale": (
                    "Residual correctness failures remain and model-choice surface opportunities are available; "
                    "test whether failures are capability-limited rather than instruction/example-limited."
                ),
                "disconfirming_result": "A stronger allowed model does not fix residual failures or causes regressions/cost that dominate the quality gain.",
            },
        )
        payload["evidence"] = [
            *list(payload.get("evidence") or []),
            "model capability surface opportunity available for residual correctness failures",
        ]
    efficiency_affordances = [
        affordance
        for affordance in model_affordances
    ]
    if efficiency_affordances and _model_efficiency_probe_is_relevant(objective):
        if "surface_model" in existing_mechanisms:
            for opportunity in opportunities:
                if isinstance(opportunity, dict) and opportunity.get("mechanism_class") == "surface_model":
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
                        + " Model-choice surface opportunities can also test cheaper or faster equivalent policies."
                    ).strip()
                    break
        else:
            opportunities.append(
                {
                    "mechanism_class": "surface_model",
                    "target_slices": ["global"],
                    "candidate_roles": ["atomic", "control"],
                    "measurements": ["score_delta", "cost_delta", "latency_delta", "correctness_guard"],
                    "affordance_ids": [affordance.affordance_id for affordance in efficiency_affordances[:3]],
                    "rationale": (
                        "Model-choice surface opportunities are available; test whether a different allowed model "
                        "can preserve quality while reducing cost or latency."
                    ),
                    "disconfirming_result": "The model change reduces cost or latency only by causing correctness regressions.",
                }
            )
        payload["evidence"] = [
            *list(payload.get("evidence") or []),
            "model efficiency surface opportunity available for cost/latency tradeoff testing",
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
        if row.get("mechanism_class") in {"surface_examples", "surface_model"}:
            return index
    return len(opportunities)


def _residual_quality_signal_remains(
    task_theory: TaskTheory,
    proposals_log: list[dict[str, Any]],
    parent_candidate_id: str,
) -> bool:
    if task_theory.bottleneck_class in {"runtime_or_output_defect", "output_contract", "no_observed_failures"}:
        return False
    branch_rows = [
        row
        for row in proposals_log
        if row.get("parent_candidate_id") == parent_candidate_id and "accepted" in row
    ]
    if not branch_rows:
        return True
    local_mechanism_rows = [
        row
        for row in branch_rows
        if row.get("mechanism_class") in {"surface_context", "surface_examples"}
    ]
    if not local_mechanism_rows:
        return True
    return any(not row.get("accepted") for row in local_mechanism_rows[-4:])


def _has_selectable_frontier_parent(frontier_states: dict[str, FrontierParentState]) -> bool:
    return any(not state.exhausted for state in frontier_states.values())


def _select_frontier_parents(
    summaries: Iterable[CandidateSummary],
    *,
    frontier_states: dict[str, FrontierParentState],
    objective: OptimizationObjective,
    width: int,
) -> list[CandidateSummary]:
    selectable = [
        summary
        for summary in summaries
        if not frontier_states.setdefault(summary.candidate_id, FrontierParentState()).exhausted
    ]
    ranked = sorted(
        selectable,
        key=lambda summary: _frontier_parent_sort_key(summary, frontier_states[summary.candidate_id], objective),
    )
    return ranked[: max(0, width)]


def _frontier_parent_sort_key(
    summary: CandidateSummary,
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


def _summary_progress_fields(summary: CandidateSummary) -> dict[str, Any]:
    return {
        "candidate_id": summary.candidate_id,
        "case_count": summary.case_count,
        "pass_count": summary.pass_count,
        "mean_score": round(summary.mean_score, 4),
        "mean_cost_usd": summary.mean_cost_usd,
        "mean_model_calls": summary.mean_model_calls,
        "mean_tool_calls": summary.mean_tool_calls,
        "mean_turns": summary.mean_turns,
        "median_latency_s": summary.median_latency_s,
    }


def _affordance_evidence_from_packet(evidence_packet: EvidencePacket, diagnoses: list[FailureDiagnosis]) -> dict[str, Any]:
    row = evidence_packet.to_dict()
    runtime = row.get("runtime_defects") or {}
    output = row.get("output_defects") or {}
    tool = row.get("tool_defects") or {}
    return {
        "bottleneck_class": ",".join(row.get("residual_failure_modes") or []),
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
