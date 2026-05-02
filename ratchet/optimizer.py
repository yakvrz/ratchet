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
from ratchet.surface_opportunities import SurfaceOpportunity, generate_surface_opportunities
from ratchet.evidence import ProposalExampleBank, build_proposal_example_bank
from ratchet.errors import OptimizerModelError
from ratchet.evidence_ledger import EvidenceLedger
from ratchet.experiments import (
    EvidencePacket,
    SearchPlan,
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
from ratchet.research import SearchPlanner
from ratchet.research_payloads import (
    planner_evidence_packet,
    planner_surface_opportunities,
    planner_surface_spec,
    top_counter_dict,
    truncate_text,
)
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
from ratchet.transform_program import CompiledCandidate, TransformOp, TransformPatch, TransformProgram
from ratchet.candidates import CandidateProposal
from ratchet.surface_search import TransformContextKey
from ratchet.transform_results import (
    observe_transform_result,
    summarize_surface_opportunity_results,
    summarize_transform_context_results,
    summarize_transform_results,
)
from ratchet.types import (
    AgentSpec,
    DiagnosticTrace,
    EvalCase,
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
MAX_SYSTEMIC_RUNTIME_ERROR_RATE = 0.2
MIN_SYSTEMIC_RUNTIME_ERRORS = 3
ProgressCallback = Callable[[dict[str, Any]], None]


def compose_transform_candidate(
    parent: CompiledCandidate | None,
    child: TransformProgram,
    *,
    compiler: TransformCompiler,
    surface: SurfaceSpec,
) -> CompiledCandidate:
    patches = _compose_transform_patches(parent.program.patches if parent is not None else (), child.patches)
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


def _compose_transform_patches(
    parent_patches: tuple[TransformPatch, ...],
    child_patches: tuple[TransformPatch, ...],
) -> tuple[TransformPatch, ...]:
    patches = list(parent_patches)
    state_fields = {
        str(patch.op.params.get("field"))
        for patch in parent_patches
        if patch.op.op == "define_state" and patch.op.params.get("field")
    }
    context_sections: dict[str, int] = {}
    for index, patch in enumerate(patches):
        if patch.op.op not in {"add_context_section", "render_state_section"}:
            continue
        section = patch.op.params.get("section")
        if section:
            context_sections[str(section)] = index
    for child_patch in child_patches:
        if child_patch.op.op == "define_state":
            field = str(child_patch.op.params.get("field") or "")
            if field in state_fields:
                continue
            state_fields.add(field)
        if child_patch.op.op in {"add_context_section", "render_state_section"}:
            section = str(child_patch.op.params.get("section") or "")
            existing_index = context_sections.get(section)
            if existing_index is not None:
                _merge_rendered_state_section(patches, existing_index, child_patch)
                continue
            if section:
                context_sections[section] = len(patches)
        patches.append(child_patch)
    return tuple(patches)


def _merge_rendered_state_section(patches: list[TransformPatch], existing_index: int, child_patch: TransformPatch) -> None:
    existing = patches[existing_index]
    if existing.op.op != "render_state_section" or child_patch.op.op != "render_state_section":
        return
    existing_fields = existing.op.params.get("fields")
    child_fields = child_patch.op.params.get("fields")
    if not isinstance(existing_fields, list) or not isinstance(child_fields, list):
        return
    fields = [str(field) for field in existing_fields]
    for field in child_fields:
        text = str(field)
        if text not in fields:
            fields.append(text)
    params = {**existing.op.params, "fields": fields}
    patches[existing_index] = TransformPatch(
        op=TransformOp(existing.op.op, params),
        hook=existing.hook,
        when=existing.when,
        unless=existing.unless,
    )


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


@dataclass
class RunRecorder:
    events: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    search_plans: list[dict[str, Any]] = field(default_factory=list)

    def record_event(self, row: dict[str, Any]) -> None:
        self.events.append(row)

    def record_search_plan(self, row: dict[str, Any]) -> None:
        self.search_plans.append(row)
        self.record_event(row)


class ProposalEngine:
    def __init__(self, implementer: CandidateImplementer) -> None:
        self.implementer = implementer

    def propose(
        self,
        summary: CandidateSummary,
        surface: SurfaceSpec,
        **kwargs: Any,
    ) -> tuple[list[CandidateProposal], str]:
        return self.implementer.propose(summary, surface, **kwargs)


class StageEvaluator:
    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer

    def evaluate_candidate_batch_progressively(self, **kwargs: Any) -> None:
        self.optimizer._evaluate_candidate_batch_progressively(**kwargs)


class FrontierController:
    def __init__(self, objective: OptimizationObjective, states: dict[str, FrontierParentState]) -> None:
        self.objective = objective
        self.states = states

    def has_selectable_parent(self) -> bool:
        return any(not state.exhausted for state in self.states.values())

    def select_parents(
        self,
        parent_pool: Iterable[CandidateSummary],
        *,
        width: int,
    ) -> list[CandidateSummary]:
        return _select_frontier_parents(
            parent_pool,
            frontier_states=self.states,
            objective=self.objective,
            width=width,
        )

    def start_parent(self, candidate_id: str, *, iteration: int) -> FrontierParentState:
        state = self.states.setdefault(candidate_id, FrontierParentState())
        state.visits += 1
        state.last_selected_iteration = iteration
        return state

    def track_child(self, candidate_id: str) -> None:
        self.states.setdefault(candidate_id, FrontierParentState())


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
        search_planner_model: str | None = None,
        search_planner_reasoning: str | None = None,
        candidate_implementer_model: str | None = None,
        candidate_implementer_reasoning: str | None = None,
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
            "search_planner": search_planner_model or optimizer_model,
            "candidate_implementer": candidate_implementer_model or optimizer_model,
        }
        self.optimizer_role_reasoning = {
            "search_planner": search_planner_reasoning or optimizer_reasoning,
            "candidate_implementer": candidate_implementer_reasoning or optimizer_reasoning,
        }
        self.search_planner = SearchPlanner(
            env_path=env_path,
            model=self.optimizer_role_models["search_planner"],
            reasoning_effort=self.optimizer_role_reasoning["search_planner"],
        )
        self.candidate_implementer = CandidateImplementer(
            env_path=env_path,
            model=self.optimizer_role_models["candidate_implementer"],
            reasoning_effort=self.optimizer_role_reasoning["candidate_implementer"],
        )
        self.proposal_engine = ProposalEngine(self.candidate_implementer)
        self.stage_evaluator = StageEvaluator(self)
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
        frontier_controller = FrontierController(self.objective, frontier_states)
        recorder = RunRecorder()
        events = recorder.events
        proposals_log = recorder.proposals
        search_plan_log = recorder.search_plans
        evidence_ledger = EvidenceLedger()
        evidence_packet_cache: dict[str, EvidencePacket] = {}
        evaluated_candidate_ids = {baseline_dev.candidate_id}
        generated_surface_rows: list[dict[str, Any]] = [self._surface().to_dict()]
        dev_evaluations = 0
        iteration = 0
        consecutive_zero_eval_parent_attempts = 0

        while dev_evaluations < self.dev_budget and frontier_controller.has_selectable_parent():
            if (
                accepted_dev_candidates
                and self.dev_budget - dev_evaluations < MIN_REMAINING_DEV_EVALS_FOR_NEW_ROUND
            ):
                events.append(
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
            parent_summaries = frontier_controller.select_parents(
                parent_pool_by_id.values(),
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
                parent_state = frontier_controller.start_parent(current_dev.candidate_id, iteration=iteration)
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
                evidence_packet_cached = current_dev.candidate_id in evidence_packet_cache
                if evidence_packet_cached:
                    evidence_packet = evidence_packet_cache[current_dev.candidate_id]
                    events.append(
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
                        diagnoses=[],
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
                events.append(evidence_packet_row)
                self._emit_progress(
                    "evidence_packet_ready",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    residual_failure_modes=evidence_packet.residual_failure_modes,
                    weak_slices=evidence_packet.weak_slices,
                    evidence=evidence_packet.evidence,
                    tool_error_case_count=len(evidence_packet.tool_defects.get("tool_error_case_ids", [])),
                    invalid_output_count=evidence_packet.output_defects.get("invalid_output_count", 0),
                    weak_labels_without_examples=evidence_packet.example_coverage.get("weak_labels_without_examples", []),
                    confidence=evidence_packet.confidence,
                    cached=evidence_packet_cached,
                )
                surface_opportunities = generate_surface_opportunities(
                    surface,
                    objective=self.objective,
                    active_mechanisms=None,
                    evidence=_surface_opportunity_evidence_from_packet(evidence_packet),
                )
                events.append(
                    {
                        "type": "surface_opportunities",
                        "iteration": iteration,
                        "parent_rank": parent_index + 1,
                        "parent_candidate_id": current_dev.candidate_id,
                        "surface_opportunities": [surface_opportunity.to_dict() for surface_opportunity in surface_opportunities],
                    }
                )
                search_plan = self._build_search_plan(
                    current_dev=current_dev,
                    evidence_packet=evidence_packet,
                    surface=surface,
                    surface_opportunities=surface_opportunities,
                    proposals_log=proposals_log,
                    evidence_ledger=evidence_ledger,
                    events=events,
                    iteration=iteration,
                    parent_index=parent_index,
                    dev_evaluations_used=dev_evaluations,
                    proposal_budget=proposal_budget,
                )
                search_plan_row = {
                    "type": "search_plan",
                    "iteration": iteration,
                    "parent_rank": parent_index + 1,
                    "parent_candidate_id": current_dev.candidate_id,
                    "candidate_id": current_dev.candidate_id,
                    "search_plan": search_plan.to_dict(),
                    "evidence_packet": evidence_packet.to_dict(),
                }
                recorder.record_search_plan(search_plan_row)
                self._emit_progress(
                    "search_plan_ready",
                    iteration=iteration,
                    parent_rank=parent_index + 1,
                    plan_id=search_plan.plan_id,
                    diagnosis=search_plan.diagnosis,
                    hypotheses=search_plan.hypotheses,
                    briefs=[
                        {
                            "brief_id": brief.brief_id,
                            "mechanism_class": brief.mechanism_class,
                            "target_slices": list(brief.target_slices),
                            "hypothesis": brief.hypothesis,
                            "priority": brief.priority,
                        }
                        for brief in search_plan.briefs
                    ],
                    target_mechanisms=search_plan.active_mechanisms,
                    brief_count=len(search_plan.briefs),
                    confidence=search_plan.confidence,
                )
                if (
                    self.objective.mode == "correctness"
                    and not search_plan.briefs
                    and current_dev.pass_count == current_dev.case_count
                ):
                    events.append(
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
                if not search_plan.briefs:
                    parent_state.exhausted = True
                    parent_state.consecutive_stalls += 1
                    continue
                accepted_rows, evaluations_used = self._propose_and_evaluate_parent(
                    current_dev=current_dev,
                    baseline_dev=baseline_dev,
                    dev_cases=dev_cases,
                    surface=surface,
                    search_plan=search_plan,
                    evidence_packet=evidence_packet,
                    current_spec=current_spec,
                    proposal_example_bank=proposal_example_bank,
                    proposal_example_cases=train_cases,
                    evaluated_candidate_ids=evaluated_candidate_ids,
                    proposals_log=proposals_log,
                    events=events,
                    iteration=iteration,
                    parent_index=parent_index,
                    parent_summaries=parent_summaries,
                    proposal_budget=proposal_budget,
                    dev_evaluations_used=dev_evaluations,
                    surface_opportunities=surface_opportunities,
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
                    events.append(
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
                accepted_rows.sort(key=lambda item: objective_sort_key(item[1], self.objective))
                for _, accepted_summary, _ in accepted_rows:
                    if accepted_summary.candidate_id not in accepted_dev_ids:
                        accepted_dev_ids.add(accepted_summary.candidate_id)
                        accepted_dev_candidates.append(accepted_summary)
                    parent_pool_by_id.setdefault(accepted_summary.candidate_id, accepted_summary)
                    frontier_controller.track_child(accepted_summary.candidate_id)
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
                    events.append(
                        {
                            "type": "accepted_proposal",
                            "iteration": iteration,
                            "parent_rank": parent_index + 1,
                            "parent_candidate_id": current_dev.candidate_id,
                            "proposal_candidate_id": transform_program_hash(chosen_proposal.program),
                            "surface_mechanism": chosen_proposal.surface_mechanism,
                            "transform_context": TransformContextKey.from_candidate(chosen_proposal).to_dict(),
                            "candidate_id": chosen_dev.candidate_id,
                            "metrics": chosen_dev.to_dict(),
                        }
                    )

            if search_complete:
                break
            if not next_frontier_by_id and not frontier_controller.has_selectable_parent():
                break
            frontier = frontier_controller.select_parents(
                parent_pool_by_id.values(),
                width=SEARCH_FRONTIER_WIDTH,
            )
            events.append(
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
            events.append(
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
                events.append(
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
                events.append(
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
                events.append(
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
            events.append(
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
        events.append(
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
        surface_opportunity_summaries = summarize_surface_opportunity_results(proposals_log)
        transform_final_statuses = _transform_final_status_summaries(finalist_statuses)
        cost_tradeoffs = quality_cost_tradeoffs(proposals_log)
        ideation_metrics = build_ideation_metrics(
            events=events,
            proposals=proposals_log,
            finalist_statuses=finalist_statuses,
        )
        outcome_analysis = build_outcome_analysis(
            objective=self.objective,
            promoted=promoted,
            baseline_dev=baseline_dev,
            accepted_dev_candidates=accepted_dev_candidates,
            holdout_candidates=holdout_candidates,
            events=events,
            finalist_statuses=finalist_statuses,
        )
        manifest = self.build_manifest(
            total_cases=len(cases),
            train_case_count=len(train_cases),
            proposal_example_bank=proposal_example_bank,
            selected_candidate_id=selected_candidate_id,
            promoted=promoted,
            generated_surface=generated_surface_rows,
            search_plans=search_plan_log,
            transform_summaries=transform_summaries,
            transform_context_summaries=transform_context_summaries,
            surface_opportunity_summaries=surface_opportunity_summaries,
            transform_final_statuses=transform_final_statuses,
            finalist_statuses=finalist_statuses,
            runtime_reliability_diagnostics=runtime_diagnostics,
            confirmation_results=confirmation_results,
            simplification_results=simplification_results,
            frontier_recommendation=frontier_recommendation,
            optimizer_call_diagnostics=self.optimizer_call_diagnostics,
            quality_cost_tradeoffs=cost_tradeoffs,
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
            events=events,
            proposals=proposals_log,
            generated_surface=generated_surface_rows,
            search_plans=search_plan_log,
            transform_summaries=transform_summaries,
            transform_context_summaries=transform_context_summaries,
            surface_opportunity_summaries=surface_opportunity_summaries,
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

    def _build_search_plan(
        self,
        *,
        current_dev: CandidateSummary,
        evidence_packet: EvidencePacket,
        surface: SurfaceSpec,
        surface_opportunities: list[SurfaceOpportunity],
        proposals_log: list[dict[str, Any]],
        evidence_ledger: EvidenceLedger,
        events: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        dev_evaluations_used: int,
        proposal_budget: int,
        proposal_retry: bool = False,
    ) -> SearchPlan:
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
                "failure_labels": top_counter_dict(current_dev.failure_labels, limit=12),
                "cost_usd": current_dev.mean_cost_usd,
                "latency_s": current_dev.median_latency_s,
            },
            "evidence_packet": planner_evidence_packet(evidence_packet),
            "surface_spec": planner_surface_spec(surface),
            "surface_opportunities": planner_surface_opportunities(surface_opportunities),
            "prior_candidate_results": _compact_prior_stage_results(proposals_log, stage=None, limit=8),
            "evidence_ledger_summary": evidence_ledger.to_dict()["summary"],
            "recent_candidate_history": _compact_recent_history_for_planner(proposals_log, limit=10),
        }
        self._emit_progress(
            "search_planner_started",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            surface_opportunity_count=len(surface_opportunities),
        )
        plan = self.search_planner.plan(
            state=state,
            surface_opportunity_ids={surface_opportunity.surface_opportunity_id for surface_opportunity in surface_opportunities},
        )
        if self.search_planner.last_call_diagnostics is not None:
            self.optimizer_call_diagnostics.append(
                {
                    "iteration": iteration,
                    "attempt": attempt,
                    "parent_rank": parent_index + 1,
                    "stage": "build_search_plan",
                    **self.search_planner.last_call_diagnostics,
                }
            )
        events.append(
            {
                "type": "search_plan_call",
                "iteration": iteration,
                "attempt": attempt,
                "proposal_retry": proposal_retry,
                "parent_rank": parent_index + 1,
                "parent_candidate_id": current_dev.candidate_id,
                "search_plan": plan.to_dict(),
                "planner_state": state,
            }
        )
        self._emit_progress(
            "search_planner_completed",
            iteration=iteration,
            attempt=attempt,
            parent_rank=parent_index + 1,
            plan_id=plan.plan_id,
            hypothesis_count=len(plan.hypotheses),
            brief_count=len(plan.briefs),
            call_diagnostics=self.search_planner.last_call_diagnostics or {},
        )
        return plan

    def _propose_and_evaluate_parent(
        self,
        *,
        current_dev: CandidateSummary,
        baseline_dev: CandidateSummary,
        dev_cases: tuple[EvalCase, ...],
        surface: SurfaceSpec,
        search_plan: SearchPlan,
        evidence_packet: EvidencePacket,
        current_spec: AgentSpec | None,
        proposal_example_bank: ProposalExampleBank,
        proposal_example_cases: tuple[EvalCase, ...],
        evaluated_candidate_ids: set[str],
        proposals_log: list[dict[str, Any]],
        events: list[dict[str, Any]],
        iteration: int,
        parent_index: int,
        parent_summaries: list[CandidateSummary],
        proposal_budget: int,
        dev_evaluations_used: int,
        evidence_ledger: EvidenceLedger,
        surface_opportunities: list[SurfaceOpportunity] | None = None,
        proposal_retry: bool = False,
        retry_reason: str | None = None,
    ) -> tuple[list[tuple[CandidateProposal, CandidateSummary, Comparison]], int]:
        if proposal_budget <= 0:
            return [], 0
        attempt = 2 if proposal_retry else 1
        self._emit_progress(
            "proposal_started",
            iteration=iteration,
            attempt=attempt,
            proposal_retry=proposal_retry,
            parent_rank=parent_index + 1,
            proposal_budget=proposal_budget,
            active_mechanisms=search_plan.active_mechanisms,
        )
        proposals, proposal_analysis = self.proposal_engine.propose(
            current_dev,
            surface,
            objective=self.objective,
            search_plan=search_plan,
            evidence_packet=evidence_packet,
            seen_hashes=evaluated_candidate_ids,
            current_spec=current_spec,
            history=proposals_log,
            proposal_example_bank=proposal_example_bank,
            proposal_example_cases=proposal_example_cases,
            proposal_budget=proposal_budget,
            surface_opportunities=surface_opportunities or [],
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
        events.append(
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
                "proposal_analysis": proposal_analysis,
                "proposal_stats": self.candidate_implementer.last_stats.to_dict(),
                "search_plan": search_plan.to_dict(),
                "evidence_packet": evidence_packet.to_dict(),
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
        model_candidates_allowed = _model_candidate_evidence_present(search_plan)
        for proposal in proposals[:proposal_budget]:
            if proposal.surface_mechanism == "surface_model":
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
                            "surface_mechanism": proposal.surface_mechanism,
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
                surface_mechanism=proposal.surface_mechanism,
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

        self.stage_evaluator.evaluate_candidate_batch_progressively(
            states=evaluation_states,
            reference=current_dev,
            baseline=baseline_dev,
            search_plan=search_plan,
            dev_cases=dev_cases,
            proposals_log=proposals_log,
            events=events,
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
                "surface_opportunity_ids": list(candidate.surface_opportunity_ids),
                "surface_mechanism": candidate.surface_mechanism,
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
            events.append({"type": "proposal_evaluation", **proposal_row})
            events.append(
                observe_transform_result(
                    family=candidate.surface_mechanism,
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
                surface_mechanism=candidate.surface_mechanism,
                transform_context=state.transform_context.to_dict(),
                candidate_id=state.candidate_id,
                accepted=state.accepted,
                frontier_status=state.frontier_status,
                rejection_reason=state.rejection_reason,
                constraint_warning=state.constraint_warning,
                score_delta=comparison.score_delta,
                cost_delta=comparison.cost_delta,
                latency_delta=comparison.latency_delta,
                fixed_count=flip_summary.get("fixed_count"),
                regressed_count=flip_summary.get("regressed_count"),
                stage_count=len(state.stage_rows),
                full_dev_evaluated=state.full_dev_evaluated,
            )
            if state.accepted:
                accepted_rows.append((candidate, summary, comparison))
                if candidate.mechanism_class in {"surface_runtime", "surface_output", "surface_response"}:
                    events.append(
                        {
                            "type": "residual_rediagnosis_triggered",
                            "candidate_id": state.candidate_id,
                            "parent_candidate_id": current_dev.candidate_id,
                            "mechanism_class": candidate.mechanism_class,
                            "reason": "structural/runtime fix accepted; child branch should be rediagnosed for residual failures",
                        }
                    )
        return accepted_rows, evaluations_used

    def _evaluate_candidate_batch_progressively(
        self,
        *,
        states: list[CandidateEvaluationState],
        reference: CandidateSummary,
        baseline: CandidateSummary,
        search_plan: SearchPlan,
        dev_cases: tuple[EvalCase, ...],
        proposals_log: list[dict[str, Any]],
        events: list[dict[str, Any]],
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
                        eligible, reason = _eligible_for_full_dev_from_small_signal(state, self.objective)
                        if eligible:
                            next_active.append(state)
                            continue
                        state.rejection_reason = reason
                        state.frontier_status = "screened_out"
                        state.accepted = False
                    active = next_active
                    if not active:
                        break
            elif stage_name == "small_dev":
                active = _select_small_dev_candidates(active)
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
                    surface_opportunity_ids=list(state.proposal.surface_opportunity_ids),
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
                        "failure_label_delta": _failure_label_delta(reference_summary, candidate_summary),
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
            summary = CandidateSummary(
                candidate_id=digest,
                candidate=candidate,
                split=cases[0].split,
                evaluations=evaluations,
            )
            _raise_on_systemic_runtime_errors(summary)
            summaries[digest] = summary
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
            input_tokens=evaluation.record.metrics.input_tokens,
            output_tokens=evaluation.record.metrics.output_tokens,
            total_tokens=evaluation.record.metrics.total_tokens,
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
        search_plans: list[dict[str, Any]],
        transform_summaries: dict[str, dict[str, Any]],
        transform_context_summaries: dict[str, dict[str, Any]],
        surface_opportunity_summaries: dict[str, dict[str, Any]],
        transform_final_statuses: dict[str, dict[str, Any]],
        outcome_analysis: dict[str, Any],
        finalist_statuses: list[dict[str, Any]],
        runtime_reliability_diagnostics: list[dict[str, Any]],
        confirmation_results: list[dict[str, Any]],
        simplification_results: list[dict[str, Any]],
        frontier_recommendation: dict[str, Any],
        optimizer_call_diagnostics: list[dict[str, Any]],
        quality_cost_tradeoffs: list[dict[str, Any]],
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
            "search_plans": search_plans,
            "transform_summaries": transform_summaries,
            "transform_context_summaries": transform_context_summaries,
            "surface_opportunity_summaries": surface_opportunity_summaries,
            "transform_final_statuses": transform_final_statuses,
            "finalist_statuses": finalist_statuses,
            "runtime_reliability_diagnostics": runtime_reliability_diagnostics,
            "confirmation_results": confirmation_results,
            "simplification_results": simplification_results,
            "frontier_recommendation": frontier_recommendation,
            "optimizer_call_diagnostics": optimizer_call_diagnostics,
            "optimizer_role_models": self.optimizer_role_models,
            "optimizer_role_reasoning": self.optimizer_role_reasoning,
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
                "surface_mechanism": item.get("surface_mechanism"),
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


def _compact_recent_history_for_planner(
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
                "hypothesis": truncate_text(item.get("hypothesis"), limit=320),
                "expected_effects": truncate_text(item.get("expected_effects"), limit=240),
                "mechanism_class": item.get("mechanism_class"),
                "surface_mechanism": item.get("surface_mechanism"),
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


def _candidate_comparison_group(state: CandidateEvaluationState) -> str:
    comparison_group = (
        state.proposal.comparison_group
        or state.proposal.experiment_id
        or state.proposal.surface_mechanism
    )
    return f"{comparison_group}|{state.proposal.target_slice or 'global'}"


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


def _select_small_dev_candidates(states: list[CandidateEvaluationState]) -> list[CandidateEvaluationState]:
    selected: list[CandidateEvaluationState] = []
    seen_groups: set[str] = set()
    for state in states:
        group = _candidate_comparison_group(state)
        if group in seen_groups:
            state.rejection_reason = "deterministic_small_dev_skipped_duplicate_comparison_group"
            state.frontier_status = "screened_out"
            state.accepted = False
            continue
        seen_groups.add(group)
        selected.append(state)
    return selected


def _eligible_for_full_dev_from_small_signal(
    state: CandidateEvaluationState,
    objective: OptimizationObjective,
) -> tuple[bool, str]:
    small_stage = next(
        (
            row
            for row in reversed(state.stage_rows)
            if isinstance(row, dict) and row.get("stage") == "small_dev"
        ),
        None,
    )
    if small_stage is None:
        return True, ""
    comparison = small_stage.get("comparison_to_parent") or {}
    behavior = small_stage.get("behavior_flip_summary") or {}
    score_delta = float(comparison.get("score_delta") or 0.0)
    fixed_count = int(behavior.get("fixed_count") or 0)
    regressed_count = int(behavior.get("regressed_count") or 0)
    pass_delta = fixed_count - regressed_count
    if objective.mode == "correctness":
        if score_delta > 0.0:
            return True, ""
        if pass_delta > 0:
            return True, ""
        return (
            False,
            "full_dev skipped because small-dev evidence had no positive correctness signal",
        )
    if objective.mode == "cost":
        cost_delta = float(comparison.get("cost_delta") or 0.0)
        if score_delta >= -0.01 and pass_delta >= 0 and cost_delta < 0.0:
            return True, ""
        return (
            False,
            "full_dev skipped because small-dev evidence did not reduce cost without correctness regression",
        )
    if objective.mode == "latency":
        latency_delta = float(comparison.get("latency_delta") or 0.0)
        if score_delta >= -0.01 and pass_delta >= 0 and latency_delta < 0.0:
            return True, ""
        return (
            False,
            "full_dev skipped because small-dev evidence did not reduce latency without correctness regression",
        )
    return False, f"full_dev skipped because objective mode {objective.mode!r} is unsupported"


def _failure_label_delta(reference: CandidateSummary, candidate: CandidateSummary) -> dict[str, dict[str, int]]:
    labels = sorted(set(reference.failure_labels) | set(candidate.failure_labels))
    return {
        label: {
            "reference": int(reference.failure_labels.get(label, 0)),
            "candidate": int(candidate.failure_labels.get(label, 0)),
            "delta": int(candidate.failure_labels.get(label, 0)) - int(reference.failure_labels.get(label, 0)),
        }
        for label in labels
    }


def _model_candidate_evidence_present(search_plan: SearchPlan) -> bool:
    for brief in search_plan.briefs:
        if brief.mechanism_class == "surface_model":
            return True
        text = " ".join([brief.brief_id, brief.hypothesis, *brief.measurements]).lower()
        if "model" in text and any(token in text for token in ("capacity", "reasoning", "capability")):
            return True
    for hypothesis in search_plan.hypotheses:
        text = hypothesis.lower()
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
    if runtime_diagnostic.get("tool_trajectory_defect_fixed") or runtime_diagnostic.get("fixed_tool_problem_case_ids"):
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
        family = row.get("surface_mechanism")
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
                {"validated": 0, "failed": 0, "unstable": 0, "finalist_count": 0, "candidate_ids": []},
            )
            status = str(row.get("status") or "failed")
            if status not in {"validated", "failed", "unstable"}:
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


def _raise_on_systemic_runtime_errors(summary: CandidateSummary) -> None:
    runtime_errors = summary.runtime_error_count
    if runtime_errors == 0:
        return
    sample_count = summary.sample_count
    error_rate = runtime_errors / max(1, sample_count)
    if runtime_errors < MIN_SYSTEMIC_RUNTIME_ERRORS and error_rate <= MAX_SYSTEMIC_RUNTIME_ERROR_RATE:
        return
    examples = [
        evaluation.case.id
        for evaluation in summary.evaluations
        if evaluation.record.metrics.error
    ][:3]
    raise RuntimeError(
        "Evaluation aborted because runtime errors made the measurement invalid: "
        f"candidate={summary.candidate_id} split={summary.split} "
        f"errors={runtime_errors}/{sample_count} ({error_rate:.1%}); "
        f"examples={examples}"
    )


def _summary_progress_fields(summary: CandidateSummary) -> dict[str, Any]:
    return {
        "candidate_id": summary.candidate_id,
        "case_count": summary.case_count,
        "pass_count": summary.pass_count,
        "mean_score": round(summary.mean_score, 4),
        "failure_labels": summary.failure_labels,
        "category_metrics": summary.category_metrics,
        "mean_cost_usd": summary.mean_cost_usd,
        "mean_model_calls": summary.mean_model_calls,
        "mean_tool_calls": summary.mean_tool_calls,
        "mean_turns": summary.mean_turns,
        "median_latency_s": summary.median_latency_s,
    }


def _surface_opportunity_evidence_from_packet(evidence_packet: EvidencePacket) -> dict[str, Any]:
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
    }
