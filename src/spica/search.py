# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The smart sweep: SearchSpace -> ranked candidates (best-first).

One Vizier study per ``deployment_mode`` branch searches the parallel-config + knob
space (backend is one of the knobs); each suggestion is unrolled, translated to a
deployment, evaluated by replay, scored, and fed back to the optimizer; feasible
candidates are ranked across branches.

Each round is a **barrier**: the study suggests trials until ``per_round`` unique full
samples complete successfully (ask), they are evaluated **in parallel across worker
processes** (``SweepConfig.parallel_evals``; ``<= 1`` runs sequentially), then their
scores are fed back (tell). Exact duplicates use a run-local result cache and trigger
replacement asks. Vizier ask/tell stay on the main process — workers run only the pure
unroll->deploy->replay->score and never touch the study (the Vizier trial handle never
crosses the process boundary). The load-predictor winner is resolved once and injected
into every unroll.

``evaluator`` and ``sampler_factory`` are injectable so the loop is unit-testable
without real replay / Vizier (use ``parallel_evals=1`` to avoid spawning processes).
"""

from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Protocol

from tqdm import tqdm

from .config import (
    AicResolutionPolicy,
    Candidate,
    MeasurementLease,
    OptimizationGoal,
    SmartSearchConfig,
)
from .deploy import build_deployment
from .evaluator import ReplayEvaluator, UnscorableCandidate
from .kv_estimate import resolve_backend_version
from .kv_load import InfeasibleKVCapacity, resolve_kv_load
from .load_predictor_sweep import LoadPredictorResult, sweep_load_predictor
from .planner import filter_scaling_policies, scaling_fields
from .sample import unroll_sample
from .sampler import BranchSampler, Suggestion, make_branch_sampler
from .score import is_feasible, make_candidate, pareto_front, rank
from .search_space import BranchSpace, enumerate_branches


class _Evaluator(Protocol):
    def evaluate(
        self,
        plan: Any,
        *,
        concurrency_override: int | None = None,
    ) -> dict[str, Any] | UnscorableCandidate: ...


# Result of evaluating one suggestion (no Vizier here): (candidate|None, observe_metrics|None,
# outcome, reason). observe_metrics is the dict fed to sampler.observe — {"objective": score}
# for a single-objective sweep, or {obj_name: raw_value, ...} under a pareto goal. outcome in
# {"feasible","infeasible","failed","unscorable"}. Every non-feasible outcome carries
# no metrics -> the loop tells the sampler observe_infeasible (a gated trial is never fed
# back as a high score). A typed AIC resolution rejection keeps its structured value here;
# only the sampler's string-only boundary receives its human-readable rendering.
_EvalResult = tuple[Candidate | None, dict[str, float] | None, str, str | UnscorableCandidate]


def _freeze(value: Any) -> Any:
    """Convert a nested suggestion/context value into a stable hashable key."""
    if is_dataclass(value) and not isinstance(value, type):
        return _freeze(asdict(value))
    if isinstance(value, dict):
        return tuple((str(key), _freeze(item)) for key, item in sorted(value.items(), key=lambda pair: str(pair[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _suggestion_cache_key(suggestion: Suggestion, context: Any) -> Any:
    """A run-local full-sample key; parallel equality alone is not a cache hit."""
    return (context, _freeze(suggestion.selection), _freeze(suggestion.parallel_config))


def _resolution_report_is_fully_exact(report: object) -> bool:
    """Whether cumulative resolution evidence proves an exact-only result.

    Resolving results fail closed for cache admission.  The V1.3 report always
    exposes complete final source counts; missing or malformed counts therefore
    cannot prove that a later exact overlay completion would be unobservable.
    """

    if not isinstance(report, dict):
        return False
    counts = report.get("final_source_counts")
    if not isinstance(counts, dict):
        return False
    values = tuple(counts.get(source) for source in ("overlay", "curated_exact", "fallback"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        return False
    return counts["fallback"] == 0


@dataclass(frozen=True, slots=True)
class UnscorableReasonCount:
    """One structured AIC resolution reason and its evaluated-candidate count."""

    code: str
    operation: str
    detail: str
    report: object | None
    count: int


class AllCandidatesUnscorableError(RuntimeError):
    """Raised when every replay attempt was rejected by typed AIC resolution."""

    def __init__(self, failures: list[UnscorableCandidate]) -> None:
        if not failures:
            raise ValueError("AllCandidatesUnscorableError requires at least one failure")
        grouped: dict[Any, tuple[UnscorableCandidate, int]] = {}
        for failure in failures:
            key = (failure.code, failure.operation, failure.detail, _freeze(failure.report))
            _, count = grouped.get(key, (failure, 0))
            grouped[key] = (failure, count + 1)
        self.total_count = len(failures)
        self.reasons = tuple(
            UnscorableReasonCount(
                code=failure.code,
                operation=failure.operation,
                detail=failure.detail,
                report=failure.report,
                count=count,
            )
            for failure, count in grouped.values()
        )
        summary = " | ".join(
            f"[{reason.code}] {reason.operation}: {reason.detail} (x{reason.count})" for reason in self.reasons
        )
        super().__init__(f"all {self.total_count} evaluated candidates were unscorable: {summary}")


def _evaluate_one(
    selection: dict[str, Any],
    parallel_config: Any,
    *,
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    load_predictor: LoadPredictorResult,
    evaluator: _Evaluator,
    measurement_lease: MeasurementLease | None = None,
) -> _EvalResult:
    """Pure unroll -> deploy -> replay -> score for one (already backend-supported)
    suggestion. No Vizier, no shared mutable state -> safe to run in a worker process."""
    try:
        sample = unroll_sample(
            search_space=config.search_space,
            selection=selection,
            parallel_config=parallel_config,
            load_predictor=load_predictor,
        )
        backend_version = resolve_backend_version(config.search_space.hardware_sku, selection["backend"])
        # The resolved perf-model version is part of the evaluated contract. Keep it
        # on the candidate so downstream artifact generation cannot independently
        # select a different backend version.
        sample["backend_version"] = backend_version
        concurrency = config.workload.concurrency
        if "kv_load_ratio" in selection:
            ratio = float(selection["kv_load_ratio"])
            resolution = resolve_kv_load(
                sample,
                workload=config.workload,
                parallel_config=parallel_config,
                ratio=ratio,
                backend_version=backend_version,
            )
            concurrency = resolution.concurrency
            sample["kv_load_ratio"] = resolution.ratio
            sample["kv_load_concurrency_capacity"] = resolution.concurrency_capacity
            load_role = "decode" if sample["deployment_mode"] == "disagg" else "agg"
            sample["kv_load_capacity_tokens"] = resolution.role_capacity_tokens[load_role]
            for role, tokens in resolution.role_capacity_tokens.items():
                sample[f"{role}_kv_capacity_tokens"] = tokens
        if concurrency is not None:
            # Preserve the concrete load on every candidate, including a fixed absolute
            # concurrency and one derived from kv_load_ratio.
            sample["concurrency"] = concurrency
        resolution = config.aic_resolution
        aic_resolution = None
        if resolution is not None:
            aic_resolution = resolution.runtime_payload(measurement_lease)
        plan = build_deployment(
            sample,
            backend_version=backend_version,
            optimization_target=goal.target.planner_optimization_target,
            planner_sla=goal.sla,
            aic_resolution=aic_resolution,
        )
    except InfeasibleKVCapacity as exc:
        return None, None, "infeasible", f"candidate KV capacity infeasible: {exc}"
    except Exception as exc:
        return None, None, "failed", f"candidate build failed: {type(exc).__name__}: {exc}"
    try:
        report = evaluator.evaluate(plan, concurrency_override=concurrency)
    except Exception as exc:  # one candidate failing must not abort the sweep
        return None, None, "failed", f"replay failed: {type(exc).__name__}: {exc}"
    if isinstance(report, UnscorableCandidate):
        return None, None, "unscorable", report
    if not is_feasible(int(sample["used_gpus"]), config.search_space.gpu_budget):
        # Over gpu_budget: report as infeasible to the optimizer (observe_infeasible, not
        # observe(metrics)) so a high score doesn't steer the sampler into the infeasible
        # region. The trial is gated, not ranked.
        return (
            None,
            None,
            "infeasible",
            f"over gpu_budget: used_gpus={int(sample['used_gpus'])} > gpu_budget={config.search_space.gpu_budget}",
        )
    if goal.is_pareto:
        candidate = make_candidate(sample, report, goal.target, pareto_objectives=goal.resolved_pareto_objectives)
        # Pareto objectives are reported raw (each metric carries its own MAXIMIZE/MINIMIZE goal).
        observe_metrics: dict[str, float] = dict(candidate.objectives or {})
    else:
        candidate = make_candidate(sample, report, goal.target)
        observe_metrics = {"objective": candidate.score}  # single metric, pre-signed higher-is-better
    return candidate, observe_metrics, "feasible", ""


# Worker-process plumbing: the shared read-only state (config/goal/load_predictor/
# evaluator) is sent once via the pool initializer and stashed as a module global, so
# each task only ships the per-suggestion (selection, parallel_config).
_WORKER_CTX: dict[str, Any] = {}


def _init_worker(
    config: SmartSearchConfig,
    goal: OptimizationGoal,
    load_predictor: LoadPredictorResult,
    evaluator: _Evaluator,
    measurement_lease: MeasurementLease | None,
) -> None:
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        config=config,
        goal=goal,
        load_predictor=load_predictor,
        evaluator=evaluator,
        measurement_lease=measurement_lease,
    )


def _worker_eval(selection: dict[str, Any], parallel_config: Any) -> _EvalResult:
    return _evaluate_one(selection, parallel_config, **_WORKER_CTX)


def run_smart_search(
    config: SmartSearchConfig,
    *,
    evaluator: _Evaluator | None = None,
    sampler_factory: Callable[..., BranchSampler] = make_branch_sampler,
    show_progress: bool = True,
    on_round: Callable[[int, list[Candidate]], None] | None = None,
) -> list[Candidate]:
    """Run the sweep and return feasible candidates sorted best-first.

    ``evaluator`` defaults to a :class:`ReplayEvaluator` over the workload+goal;
    inject a fake to test the loop without replay. ``sampler_factory`` defaults to
    the Vizier-backed sampler. Within a round, suggestions are evaluated across
    ``SweepConfig.parallel_evals`` **spawned** worker processes (``<= 1`` -> sequential,
    no pool). With ``parallel_evals > 1`` the caller must guard its entrypoint with
    ``if __name__ == "__main__":`` (spawn re-imports the module) — the ``python -m spica``
    CLI already does; ad-hoc scripts must too, or set ``parallel_evals=1``.
    ``show_progress`` draws a tqdm bar over the
    candidate evaluations (live feasible/failed tally + best score) and prints a
    one-line summary at the end; set False for quiet/non-interactive runs.

    Raises :class:`AllCandidatesUnscorableError` with counted structured reasons
    when every replay attempt is rejected by typed AIC resolution.
    """
    resolution = config.aic_resolution
    if (
        resolution is not None
        and resolution.policy is AicResolutionPolicy.MEASURE_ON_MISS
        and config.measurement_gpu_groups is None
        and config.sweep.parallel_evals > 1
    ):
        config = config.model_copy(update={"sweep": config.sweep.model_copy(update={"parallel_evals": 1})})
        if show_progress:
            tqdm.write(
                "smart-sweep: measure_on_miss without measurement_gpu_groups uses sequential "
                "candidate evaluation; each callback still schedules independent GPU work"
            )
    goal = config.goal
    # Predictive throughput scaling only works under the planner's "sla" target
    # (a goodput sweep). For throughput/latency sweeps, drop the throughput-scaling
    # policies up front so neither the sampler nor the load-predictor sub-sweep sees
    # them. (Disabled / load_* still run; static-path goodput is fine once the mocker
    # is SLA-aware.)
    kept, dropped = filter_scaling_policies(
        config.search_space.planner_scaling_policy,
        allow_throughput=(goal.target.planner_optimization_target == "sla"),
    )
    if dropped:
        if not kept:
            raise ValueError(
                f"every planner_scaling_policy enables throughput scaling, which a "
                f"'{goal.target.value}' sweep can't use (it has no SLA — use a goodput target, "
                f"or include 'disabled' / a load_* policy)"
            )
        if show_progress:
            tqdm.write(
                f"smart-sweep: dropped {len(dropped)} throughput-scaling policy option(s) "
                f"for target={goal.target.value} (needs SLA): {dropped}"
            )
        config = config.model_copy(
            update={"search_space": config.search_space.model_copy(update={"planner_scaling_policy": kept})}
        )

    # Goodput can be defined by an end-to-end SLA, but the planner's SLA scaling
    # target can only be seeded from ttft+itl. With e2e-only SLA, keep static
    # candidates and drop every scaling policy before Vizier can sample it.
    sla = goal.sla
    if (
        goal.target.planner_optimization_target == "sla"
        and sla is not None
        and (sla.ttft_ms is None or sla.itl_ms is None)
    ):
        kept = []
        dropped = []
        for policy in config.search_space.planner_scaling_policy:
            fields = scaling_fields(policy)
            target = dropped if (fields["enable_throughput_scaling"] or fields["enable_load_scaling"]) else kept
            target.append(policy)
        if dropped:
            if not kept:
                raise ValueError(
                    "every planner_scaling_policy enables planner scaling, but an e2e-only SLA "
                    "cannot seed the planner's ttft/itl scaling target; use ttft_ms+itl_ms, "
                    "or include 'disabled'"
                )
            if show_progress:
                tqdm.write(
                    f"smart-sweep: dropped {len(dropped)} planner-scaling policy option(s) "
                    f"for e2e-only SLA (planner needs ttft_ms+itl_ms): {dropped}"
                )
            config = config.model_copy(
                update={"search_space": config.search_space.model_copy(update={"planner_scaling_policy": kept})}
            )

    # Thread the configured context_length into KV feasibility so parallel configs that
    # can't fit the requested sequence length are dropped up front (None -> model max).
    branches: list[BranchSpace] = enumerate_branches(config, max_seq_len=config.search_space.context_length)
    load_predictor = sweep_load_predictor(config)
    if evaluator is None:
        evaluator = ReplayEvaluator(config.workload, goal)

    sweep = config.sweep
    per_round = sweep.candidates_per_round or sweep.parallel_evals
    # Target number of successful unique replay configurations across all rounds.
    total = len(branches) * sweep.max_rounds * per_round
    candidates: list[Candidate] = []
    tally = {
        "feasible": 0,
        "infeasible": 0,
        "failed": 0,
        "unscorable": 0,
        "unsupported": 0,
        "cache_hit": 0,
    }
    failure_reasons: dict[str, int] = {}
    unscorable_failures: list[UnscorableCandidate] = []
    # Unique per run: Vizier's datastore persists studies by id, so a fixed id would
    # make a later run inherit a stale study (and its old param space) -> decode crash.
    run_nonce = uuid.uuid4().hex[:8]
    # Multi-objective (pareto) -> one Vizier metric per objective (each with its own
    # direction); single-objective -> the sampler's default single maximized "objective".
    sampler_objectives = [(t.value, t.maximize) for t in goal.resolved_pareto_objectives] if goal.is_pareto else None
    cache_context = _freeze(
        {
            "search_space": config.search_space.model_dump(mode="python"),
            "workload": config.workload.model_dump(mode="python"),
            "goal": goal.model_dump(mode="python"),
            "load_predictor": load_predictor,
            "aic_resolution": (
                config.aic_resolution.model_dump(mode="python") if config.aic_resolution is not None else None
            ),
            "measurement_gpu_groups": (
                config.measurement_gpu_groups.model_dump(mode="python")
                if config.measurement_gpu_groups is not None
                else None
            ),
        }
    )
    replay_cache: dict[Any, tuple[Candidate, dict[str, float]]] = {}
    candidate_index_by_key: dict[Any, int] = {}

    def _best() -> float | None:
        return max((c.score for c in candidates), default=None)

    # Parallel across worker processes when parallel_evals > 1: spawn (not fork —
    # dynamo's tokio runtime isn't fork-safe); shared read-only state goes once via the
    # initializer; one pool for the whole run amortizes the per-worker dynamo import.
    use_pool = sweep.parallel_evals > 1 and per_round > 1
    max_eval_seconds = sweep.max_eval_seconds

    measurement_pool = config.measurement_gpu_groups
    pool_workers = min(sweep.parallel_evals, per_round)

    def _new_pool(
        measurement_lease: MeasurementLease | None,
        *,
        max_workers: int,
    ) -> ProcessPoolExecutor:
        # spawn (not fork — dynamo's tokio runtime isn't fork-safe); shared read-only state
        # goes once via the initializer; one pool amortizes the per-worker dynamo import.
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=_init_worker,
            initargs=(config, goal, load_predictor, evaluator, measurement_lease),
        )

    def _new_pools() -> list[ProcessPoolExecutor]:
        if not use_pool:
            return []
        if measurement_pool is not None:
            return [_new_pool(measurement_pool.lease(index), max_workers=1) for index in range(pool_workers)]
        return [_new_pool(None, max_workers=pool_workers)]

    # One-element box so a runtime timeout can kill the hung pool and swap in a fresh one
    # (the closures below read/replace pool_box[0]).
    pool_box: list[list[ProcessPoolExecutor]] = [_new_pools()]

    def _eval_batch(todo: list[Suggestion]):
        """Yield ``(suggestion, _EvalResult)`` for each supported suggestion — across worker
        processes when a pool is set, else sequentially in-process. On the pool path it
        enforces the per-candidate ``max_eval_seconds`` cap: a replay that overruns is
        reported infeasible ("exceed runtime") and its worker is force-killed (a shared pool
        can't cancel a running task), then a fresh pool is swapped in for the next rounds."""
        pools = pool_box[0]
        if not pools:
            measurement_lease = measurement_pool.lease(0) if measurement_pool is not None else None
            for s in todo:
                yield (
                    s,
                    _evaluate_one(
                        s.selection,
                        s.parallel_config,
                        config=config,
                        goal=goal,
                        load_predictor=load_predictor,
                        evaluator=evaluator,
                        measurement_lease=measurement_lease,
                    ),
                )
            return
        try:
            # submit() can also raise BrokenProcessPool (a worker died before/while tasks
            # were queued), so it must be inside the friendly-error wrapper too.
            futures = {}
            pool_indices = {}
            for index, suggestion in enumerate(todo):
                pool_index = index % len(pools)
                future = pools[pool_index].submit(_worker_eval, suggestion.selection, suggestion.parallel_config)
                futures[future] = suggestion
                pool_indices[future] = pool_index
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "smart-sweep worker pool died (parallel_evals>1 uses spawned processes). The "
                "usual cause is calling run_smart_search at a script's top level without guarding "
                "the entrypoint: spawn re-imports the module, so wrap it in `if __name__ == "
                '"__main__":`. Or set sweep.parallel_evals=1 to evaluate sequentially (no pool)."'
            ) from exc
        pending = set(futures)
        # Workers run concurrently (max_workers >= len(todo) here), so one wall-clock
        # deadline approximates the per-candidate cap. None -> no cap.
        deadline = (time.monotonic() + max_eval_seconds) if max_eval_seconds else None
        while pending:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done:
                break  # deadline hit with nothing newly finished -> the rest are hung
            for fut in done:
                yield futures[fut], fut.result()
        if pending:
            secs = max_eval_seconds or 0.0
            for fut in pending:
                yield futures[fut], (None, None, "infeasible", f"exceed runtime: replay > {secs:.0f}s")
            # Replace only the pools that own timed-out work. Under explicit GPU
            # groups this preserves every other worker's stable resource lease.
            for pool_index in sorted({pool_indices[future] for future in pending}):
                pool = pools[pool_index]
                for proc in list((getattr(pool, "_processes", None) or {}).values()):
                    proc.terminate()
                pool.shutdown(wait=False, cancel_futures=True)
                lease = measurement_pool.lease(pool_index) if measurement_pool is not None else None
                workers = 1 if measurement_pool is not None else pool_workers
                pools[pool_index] = _new_pool(lease, max_workers=workers)

    with tqdm(total=total, desc="smart-sweep", unit="eval", disable=not show_progress) as bar:

        def _record(outcome: str, candidate: Candidate | None, *, key: Any | None = None) -> bool:
            tally[outcome] += 1
            added = False
            if candidate is not None:
                if key is None:
                    raise ValueError("scored candidates require a cache key")
                candidate_index = candidate_index_by_key.get(key)
                if candidate_index is None:
                    candidate_index_by_key[key] = len(candidates)
                    candidates.append(candidate)
                    bar.update(1)
                    added = True
                else:
                    candidates[candidate_index] = candidate
            best = _best()
            bar.set_postfix(
                feasible=tally["feasible"],
                failed=tally["failed"],
                unscorable=tally["unscorable"],
                best=("-" if best is None else f"{best:.4g}"),
            )
            return added

        round_no = 0
        for branch in branches:
            branch_stalled = False
            sampler = sampler_factory(
                branch, study_id=f"spica_{branch.deployment_mode}_{run_nonce}", objectives=sampler_objectives
            )
            bar.set_description(f"smart-sweep {branch.deployment_mode}")
            for _ in range(sweep.max_rounds):
                unique_this_round = 0
                trial_attempts = 0
                max_trial_attempts = per_round * 11  # requested batch + at most 10x replacement trials
                while unique_this_round < per_round and trial_attempts < max_trial_attempts:
                    ask_count = min(per_round - unique_this_round, max_trial_attempts - trial_attempts)
                    suggestions = sampler.suggest(ask_count)  # ask stays on the main process
                    if not suggestions:
                        break
                    trial_attempts += len(suggestions)

                    # Deduplicate against completed exact-cache entries and nominate one
                    # primary per same-batch key. Duplicates reuse a result only after its
                    # cumulative report proves exact-only evidence; otherwise they are
                    # replayed deterministically below.
                    todo: list[Suggestion] = []
                    primary_by_key: dict[Any, Suggestion] = {}
                    duplicates_by_key: dict[Any, list[Suggestion]] = {}
                    for suggestion in suggestions:
                        backend = suggestion.selection["backend"]
                        if backend not in branch.supported_backends.get(suggestion.parallel_config, frozenset()):
                            sampler.observe_infeasible(
                                suggestion,
                                f"backend {backend!r} does not support this parallel config",
                            )
                            _record("unsupported", None)
                            continue

                        key = _suggestion_cache_key(suggestion, cache_context)
                        cached = replay_cache.get(key)
                        if cached is not None:
                            _, cached_metrics = cached
                            sampler.observe(suggestion, cached_metrics)
                            tally["cache_hit"] += 1
                            continue
                        if key in primary_by_key:
                            duplicates_by_key.setdefault(key, []).append(suggestion)
                            continue
                        primary_by_key[key] = suggestion
                        todo.append(suggestion)

                    deferred_duplicates: list[Suggestion] = []

                    def _accept_evaluation(
                        suggestion: Suggestion,
                        result: _EvalResult,
                        duplicates: list[Suggestion],
                    ) -> None:
                        nonlocal unique_this_round
                        candidate, observe_metrics, outcome, reason = result
                        key = _suggestion_cache_key(suggestion, cache_context)
                        if outcome in ("failed", "infeasible", "unscorable"):
                            sampler_reason = str(reason)
                            sampler.observe_infeasible(suggestion, sampler_reason)
                            if outcome == "unscorable" and resolution is not None:
                                # A typed resolution failure belongs to this evaluation.
                                # Its duplicate must get its own Replay lookup because
                                # exact evidence may have completed in the meantime.
                                deferred_duplicates.extend(duplicates)
                            else:
                                for duplicate in duplicates:
                                    sampler.observe_infeasible(duplicate, sampler_reason)
                            if outcome == "failed":
                                assert isinstance(reason, str)
                                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                            elif outcome == "unscorable":
                                assert isinstance(reason, UnscorableCandidate)
                                unscorable_failures.append(reason)
                            _record(outcome, None)
                            return

                        assert candidate is not None and observe_metrics is not None
                        sampler.observe(suggestion, observe_metrics)
                        cacheable = resolution is None or _resolution_report_is_fully_exact(
                            candidate.aic_resolution_report
                        )
                        if cacheable:
                            replay_cache[key] = (candidate, dict(observe_metrics))
                            for duplicate in duplicates:
                                sampler.observe(duplicate, observe_metrics)
                                tally["cache_hit"] += 1
                        else:
                            # Keep the degraded/mixed candidate scorable, but require
                            # each duplicate to re-enter Replay until one proves exact.
                            deferred_duplicates.extend(duplicates)
                        unique_this_round += int(_record(outcome, candidate, key=key))

                    for suggestion, result in _eval_batch(todo):
                        key = _suggestion_cache_key(suggestion, cache_context)
                        _accept_evaluation(suggestion, result, duplicates_by_key.get(key, []))

                    # Preserve ask ordering.  A newly exact result becomes cacheable
                    # for later duplicates; degraded or unscorable results advance to
                    # exactly one new Replay lookup at a time.
                    for duplicate in deferred_duplicates:
                        key = _suggestion_cache_key(duplicate, cache_context)
                        cached = replay_cache.get(key)
                        if cached is not None:
                            _, cached_metrics = cached
                            sampler.observe(duplicate, cached_metrics)
                            tally["cache_hit"] += 1
                            continue
                        evaluated = tuple(_eval_batch([duplicate]))
                        if len(evaluated) != 1 or evaluated[0][0] is not duplicate:
                            raise RuntimeError("single duplicate evaluation returned an unexpected suggestion")
                        _accept_evaluation(duplicate, evaluated[0][1], [])
                round_no += 1
                if on_round is not None:
                    on_round(round_no, list(candidates))
                if unique_this_round < per_round:
                    branch_stalled = True
                    if show_progress:
                        tqdm.write(
                            f"smart-sweep {branch.deployment_mode} stopped early: projection stalled after "
                            f"{trial_attempts} Vizier trial(s), with {unique_this_round}/{per_round} "
                            "new replay configuration(s) in the round"
                        )
                    break
            if branch_stalled:
                continue

    # Tear down the (possibly recreated) worker pool; force-kill any lingering workers.
    for final_pool in pool_box[0]:
        for proc in list((getattr(final_pool, "_processes", None) or {}).values()):
            proc.terminate()
        final_pool.shutdown(wait=False, cancel_futures=True)

    # Single-objective -> rank best-first by score; pareto -> the non-dominated front.
    result = pareto_front(candidates, goal.resolved_pareto_objectives) if goal.is_pareto else rank(candidates)
    replay_attempts = tally["feasible"] + tally["infeasible"] + tally["failed"] + tally["unscorable"]
    if show_progress:
        summary = (
            f"smart-sweep done: {tally['feasible']}/{replay_attempts} replay attempt(s) feasible, "
            f"{tally['infeasible']} gated, {tally['unsupported']} backend-unsupported, "
            f"{tally['failed']} replay-failed, {tally['unscorable']} resolution-unscorable, "
            f"{tally['cache_hit']} cache hit(s)"
        )
        if not candidates:
            summary += " — NO feasible candidate (check backends / SLA / gpu_budget / replay errors)"
        elif goal.is_pareto:
            summary += f"; pareto front: {len(result)} non-dominated candidate(s)"
        else:
            summary += f"; best {goal.target.value}={_best():.4g}"
        tqdm.write(summary)
        if failure_reasons:
            displayed = []
            for reason, count in list(failure_reasons.items())[:3]:
                displayed.append(f"{reason} (x{count})" if count > 1 else reason)
            remaining = len(failure_reasons) - len(displayed)
            suffix = f" | +{remaining} more distinct reason(s)" if remaining else ""
            tqdm.write(f"smart-sweep failure reason(s): {' | '.join(displayed)}{suffix}")
    if unscorable_failures and replay_attempts == tally["unscorable"]:
        raise AllCandidatesUnscorableError(unscorable_failures)
    return result
