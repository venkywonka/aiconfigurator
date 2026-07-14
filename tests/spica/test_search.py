# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Orchestration loop. enumerate_branches / sweep_load_predictor / backend-version
are stubbed and a fake sampler + evaluator are injected, so this exercises the
suggest->unroll->deploy->evaluate->score->observe->rank loop without Vizier or
real replay."""

import pytest

import spica.evaluator as evaluator_mod
import spica.search as search_mod
from spica.config import MeasurementLease, SmartSearchConfig
from spica.kv_load import KVLoadResolution
from spica.load_predictor_sweep import LoadPredictorResult
from spica.parallel_enum import ParallelShape, ReplicaParallelConfig
from spica.sampler import Suggestion
from spica.search import run_smart_search
from spica.search_space import BranchSpace


def _config(gpu_budget=32, *, backend="trtllm"):
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": [backend],
            "deployment_mode": ["agg"],
            "gpu_budget": gpu_budget,
        },
        workload={"trace_path": "/tmp/t.jsonl"},
        sweep={"max_rounds": 1, "candidates_per_round": 3, "parallel_evals": 1},  # sequential (fakes)
        goal={"target": "throughput"},
    )


class _FakeSampler:
    """Suggests `count` agg candidates with increasing agg_max_num_seqs."""

    backend = "trtllm"

    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch
        self.objectives = objectives
        self.scored: list = []

    def suggest(self, count):
        out = []
        for i in range(count):
            sel = {
                "deployment_mode": "agg",
                "backend": self.backend,
                "router_mode": "round_robin",
                "planner_scaling_policy": "disabled",
                "planner_fpm_sampling": "default",
                "planner_load_sensitivity": "default",
                "agg_max_num_batched_tokens": 8192,
                "agg_max_num_seqs": 256 * (i + 1),
            }
            out.append(Suggestion(selection=sel, parallel_config=self.branch.parallel_configs[0], handle=sel))
        return out

    def observe(self, suggestion, metrics):
        self.scored.append(metrics)

    def observe_infeasible(self, suggestion, reason):
        self.scored.append(("infeasible", reason))


class _SglangFakeSampler(_FakeSampler):
    backend = "sglang"


class _FakeEvaluator:
    """trace_report throughput == the plan's max_num_seqs (so higher seqs wins)."""

    def __init__(self):
        self.calls = 0
        self.last_plan = None
        self.plans = []

    def evaluate(self, plan, *, concurrency_override=None):
        self.calls += 1
        self.last_plan = plan
        self.plans.append(plan)
        return {"output_throughput_tok_s": float(plan.agg_engine_args["max_num_seqs"]), "gpu_hours": 1.0}


def _branch(parallel_config, *, backend="trtllm"):
    return BranchSpace(
        deployment_mode="agg",
        parallel_configs=(parallel_config,),
        supported_backends={parallel_config: frozenset({backend})},
        knob_choices={"backend": [backend]},
    )


def _stub(monkeypatch, branch):
    monkeypatch.setattr(search_mod, "enumerate_branches", lambda config, *, max_seq_len=None: [branch])
    monkeypatch.setattr(search_mod, "sweep_load_predictor", lambda config: LoadPredictorResult(reason="static"))
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10")


def _aic_resolution_config(tmp_path, *, on_measurement_failure="hybrid", candidates_per_round=2):
    base = _config(backend="sglang")
    payload = base.model_dump(mode="python")
    payload["sweep"]["candidates_per_round"] = candidates_per_round
    payload["aic_resolution"] = {
        "policy": "measure_on_miss",
        "on_measurement_failure": on_measurement_failure,
        "overlay_path": (tmp_path / "evidence.sqlite").resolve(),
    }
    return SmartSearchConfig.model_validate(payload)


def _aic_success_report(source, *, gpu_ids=(0, 1, 2, 3)):
    fallback_count = int(source == "fallback")
    report = {
        "final_source_counts": {
            "overlay": int(source == "overlay"),
            "curated_exact": int(source == "curated_exact"),
            "fallback": fallback_count,
        },
        "callbacks": [
            {
                "outcome": "resolved",
                "final_evidence": [
                    {
                        "key_digest": "physical-key",
                        "source": source,
                    }
                ],
            }
        ],
        # GPU ids belong to execution provenance. They do not change the
        # physical key or whether the final evidence is exact.
        "assignments": [{"gpu_ids": list(gpu_ids)}],
        "hybrid_fallbacks": [],
    }
    if source == "fallback":
        report["hybrid_fallbacks"] = [
            {
                "key_digest": "physical-key",
                "latency_ms": 4.25,
                "path": "/tmp/evidence.sqlite.live-fallbacks/physical-key.json",
                "hybrid_provenance": {
                    "source": "empirical",
                    "prediction_revision": "aic-v1.3-test",
                },
                "measurement_failure": {
                    "code": "timeout",
                    "operation": "attention.context",
                    "detail": "collector deadline expired",
                },
            }
        ]
    return report


class _DuplicateSglangSampler(_SglangFakeSampler):
    """Return two identical trials once, then stop replacement asks."""

    def __init__(self, branch, study_id, objectives=None):
        super().__init__(branch, study_id, objectives)
        self.ask_count = 0
        self.observed = []
        self.rejected = []

    def suggest(self, count):
        self.ask_count += 1
        if self.ask_count > 1:
            return []
        selection = {
            "deployment_mode": "agg",
            "backend": "sglang",
            "router_mode": "round_robin",
            "planner_scaling_policy": "disabled",
            "planner_fpm_sampling": "default",
            "planner_load_sensitivity": "default",
            "agg_max_num_batched_tokens": 8192,
            "agg_max_num_seqs": 256,
        }
        return [
            Suggestion(
                selection=dict(selection),
                parallel_config=self.branch.parallel_configs[0],
                handle=index,
            )
            for index in range(min(count, 2))
        ]

    def observe(self, suggestion, metrics):
        self.observed.append((suggestion.handle, metrics))

    def observe_infeasible(self, suggestion, reason):
        self.rejected.append((suggestion.handle, reason))


def test_ranks_feasible_best_first(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))  # 8 GPUs
    _stub(monkeypatch, branch)
    cands = run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler)
    assert [c.score for c in cands] == [768.0, 512.0, 256.0]  # throughput, best first
    assert all(c.used_gpus == 8 for c in cands)
    assert all(c.config["backend_version"] == "1.3.0rc10" for c in cands)
    assert cands[0].metrics["gpu_hours"] == 1.0


def test_ranked_candidates_preserve_replay_resolution_report(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    rich_report = {
        "status": "resolved",
        "callbacks": [{"operation": "gemm", "collection": {"new_keys_delta": 1}}],
    }

    class ResolutionReportEvaluator(_FakeEvaluator):
        def evaluate(self, plan, *, concurrency_override=None):
            report = super().evaluate(plan, concurrency_override=concurrency_override)
            report["aic_resolution_report"] = rich_report
            return report

    candidates = run_smart_search(
        _config(),
        evaluator=ResolutionReportEvaluator(),
        sampler_factory=_FakeSampler,
        show_progress=False,
    )

    assert len(candidates) == 3
    assert all(candidate.aic_resolution_report == rich_report for candidate in candidates)


def test_hybrid_degraded_candidate_remains_scorable_and_visible(monkeypatch, tmp_path):
    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    config = _aic_resolution_config(tmp_path, candidates_per_round=1)
    degraded = _aic_success_report("fallback", gpu_ids=(6,))

    class DegradedEvaluator:
        def evaluate(self, plan, *, concurrency_override=None):
            return {
                "output_throughput_tok_s": 123.0,
                "aic_resolution_report": degraded,
            }

    candidates = run_smart_search(
        config,
        evaluator=DegradedEvaluator(),
        sampler_factory=_SglangFakeSampler,
        show_progress=False,
    )

    assert len(candidates) == 1
    assert candidates[0].score == 123.0
    assert candidates[0].aic_resolution_report == degraded
    assert candidates[0].aic_resolution_report["hybrid_fallbacks"][0]["measurement_failure"]["code"] == "timeout"


def test_hybrid_degraded_duplicate_reenters_replay_and_late_exact_replaces_it(monkeypatch, tmp_path):
    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    config = _aic_resolution_config(tmp_path)
    degraded = _aic_success_report("fallback", gpu_ids=(0,))
    exact = _aic_success_report("overlay", gpu_ids=(7,))
    seen = {}

    class DegradedThenExactEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, plan, *, concurrency_override=None):
            self.calls += 1
            if self.calls == 1:
                return {
                    "output_throughput_tok_s": 100.0,
                    "aic_resolution_report": degraded,
                }
            return {
                "output_throughput_tok_s": 125.0,
                "aic_resolution_report": exact,
            }

    def factory(branch, study_id, objectives=None):
        sampler = _DuplicateSglangSampler(branch, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    evaluator = DegradedThenExactEvaluator()
    candidates = run_smart_search(
        config,
        evaluator=evaluator,
        sampler_factory=factory,
        show_progress=False,
    )

    assert evaluator.calls == 2
    assert [metrics["objective"] for _, metrics in seen["sampler"].observed] == [100.0, 125.0]
    assert seen["sampler"].rejected == []
    assert len(candidates) == 1
    assert candidates[0].score == 125.0
    assert candidates[0].aic_resolution_report == exact


def test_fully_exact_resolving_duplicate_uses_cache_even_with_gpu_provenance(monkeypatch, tmp_path):
    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    config = _aic_resolution_config(tmp_path)
    exact = _aic_success_report("overlay", gpu_ids=(5,))
    seen = {}

    class ExactEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, plan, *, concurrency_override=None):
            self.calls += 1
            return {
                "output_throughput_tok_s": 125.0,
                "aic_resolution_report": exact,
            }

    def factory(branch, study_id, objectives=None):
        sampler = _DuplicateSglangSampler(branch, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    evaluator = ExactEvaluator()
    candidates = run_smart_search(
        config,
        evaluator=evaluator,
        sampler_factory=factory,
        show_progress=False,
    )

    assert evaluator.calls == 1
    assert len(seen["sampler"].observed) == 2
    assert len(candidates) == 1
    assert candidates[0].aic_resolution_report == exact


@pytest.mark.parametrize("failure_policy", ["error", "hybrid"])
def test_resolution_failure_excludes_only_affected_duplicate_and_never_invents_metric(
    monkeypatch,
    tmp_path,
    failure_policy,
):
    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    config = _aic_resolution_config(tmp_path, on_measurement_failure=failure_policy)
    exact = _aic_success_report("overlay")
    seen = {}
    failure = evaluator_mod.UnscorableCandidate(
        code="collector_failed",
        operation="attention.context",
        detail=(
            "HYBRID fallback failed after collector_failed"
            if failure_policy == "hybrid"
            else "collector returned no valid samples"
        ),
        report={
            "callbacks": [{"outcome": "failed"}],
            "unresolved": [{"code": "collector_failed", "operation": "attention.context"}],
        },
    )

    class FailureThenExactEvaluator:
        def __init__(self):
            self.calls = 0

        def evaluate(self, plan, *, concurrency_override=None):
            self.calls += 1
            if self.calls == 1:
                return failure
            return {
                "output_throughput_tok_s": 125.0,
                "aic_resolution_report": exact,
            }

    def factory(branch, study_id, objectives=None):
        sampler = _DuplicateSglangSampler(branch, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    evaluator = FailureThenExactEvaluator()
    candidates = run_smart_search(
        config,
        evaluator=evaluator,
        sampler_factory=factory,
        show_progress=False,
    )

    assert evaluator.calls == 2
    assert seen["sampler"].rejected == [(0, str(failure))]
    assert seen["sampler"].observed == [(1, {"objective": 125.0})]
    assert [candidate.score for candidate in candidates] == [125.0]


def test_measure_on_miss_without_measurement_pool_forces_sequential_evaluation(monkeypatch, tmp_path, capsys):
    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    base = _config(backend="sglang")
    cfg = base.model_copy(
        update={
            "sweep": base.sweep.model_copy(update={"parallel_evals": 2, "candidates_per_round": 2}),
            "aic_resolution": {
                "policy": "measure_on_miss",
                "overlay_path": (tmp_path / "evidence.sqlite").resolve(),
            },
        }
    )
    cfg = SmartSearchConfig.model_validate(cfg.model_dump())

    def forbidden_pool(*args, **kwargs):
        raise AssertionError("measure-on-miss without disjoint groups must not create a process pool")

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", forbidden_pool)
    evaluator = _FakeEvaluator()

    candidates = run_smart_search(cfg, evaluator=evaluator, sampler_factory=_SglangFakeSampler, show_progress=True)

    assert len(candidates) == 2
    assert evaluator.calls == 2
    assert "sequential candidate evaluation" in capsys.readouterr().out


def test_resolution_payload_uses_concrete_measurement_lease(monkeypatch, tmp_path):
    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    cfg = _config(backend="sglang").model_copy(
        update={
            "aic_resolution": {
                "policy": "measure_on_miss",
                "overlay_path": (tmp_path / "evidence.sqlite").resolve(),
                "max_new_keys": 17,
                "max_wall_seconds": 45.0,
            },
            "measurement_gpu_groups": [[4, 5, 6, 7]],
        }
    )
    cfg = SmartSearchConfig.model_validate(cfg.model_dump())
    evaluator = _FakeEvaluator()

    result = search_mod._evaluate_one(
        _SglangFakeSampler(branch, "test").suggest(1)[0].selection,
        branch.parallel_configs[0],
        config=cfg,
        goal=cfg.goal,
        load_predictor=LoadPredictorResult(reason="static"),
        evaluator=evaluator,
        measurement_lease=MeasurementLease(gpu_ids=(4, 5, 6, 7)),
    )

    assert result[2] == "feasible"
    # The fake evaluator reports from the built plan, proving the exact payload reached Replay args.
    plan_payload = evaluator.last_plan.agg_engine_args["aic_resolution"]
    assert plan_payload == {
        "policy": "measure_on_miss",
        "on_measurement_failure": "error",
        "overlay_path": str((tmp_path / "evidence.sqlite").resolve()),
        "fallback_cache_dir": f"{(tmp_path / 'evidence.sqlite').resolve()}.live-fallbacks",
        "max_new_keys": 17,
        "max_wall_seconds": 45.0,
        "force_remeasure": False,
        "gpu_ids": [4, 5, 6, 7],
    }


def test_parallel_resolution_uses_one_single_worker_pool_per_disjoint_group(monkeypatch, tmp_path):
    from concurrent.futures import Future

    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    base = _config(backend="sglang")
    cfg = base.model_copy(
        update={
            "sweep": base.sweep.model_copy(update={"parallel_evals": 2, "candidates_per_round": 2}),
            "aic_resolution": {
                "policy": "measure_on_miss",
                "overlay_path": (tmp_path / "evidence.sqlite").resolve(),
            },
            "measurement_gpu_groups": [[0, 1, 2, 3], [4, 5, 6, 7]],
        }
    )
    cfg = SmartSearchConfig.model_validate(cfg.model_dump())
    created = []

    class ImmediatePool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            self.max_workers = max_workers
            self.initializer = initializer
            self.initargs = initargs
            self._processes = {}
            created.append(self)

        def submit(self, fn, selection, parallel_config):
            worker_config, goal, load_predictor, evaluator, lease = self.initargs
            future = Future()
            future.set_result(
                search_mod._evaluate_one(
                    selection,
                    parallel_config,
                    config=worker_config,
                    goal=goal,
                    load_predictor=load_predictor,
                    evaluator=evaluator,
                    measurement_lease=lease,
                )
            )
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            return None

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", ImmediatePool)
    evaluator = _FakeEvaluator()

    candidates = run_smart_search(cfg, evaluator=evaluator, sampler_factory=_SglangFakeSampler, show_progress=False)

    assert len(candidates) == 2
    assert [pool.max_workers for pool in created] == [1, 1]
    assert [pool.initargs[-1].gpu_ids for pool in created] == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert [plan.agg_engine_args["aic_resolution"]["gpu_ids"] for plan in evaluator.plans] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]


def test_timed_out_worker_pool_replacement_preserves_its_concrete_measurement_lease(monkeypatch, tmp_path):
    from concurrent.futures import Future

    branch = _branch(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2),
        backend="sglang",
    )
    _stub(monkeypatch, branch)
    base = _aic_resolution_config(tmp_path)
    payload = base.model_dump(mode="python")
    payload["sweep"].update(
        {
            "parallel_evals": 2,
            "candidates_per_round": 2,
            "max_eval_seconds": 0.01,
        }
    )
    payload["measurement_gpu_groups"] = [[0, 1, 2, 3], [4, 5, 6, 7]]
    config = SmartSearchConfig.model_validate(payload)
    created = []

    class ProcessStub:
        def terminate(self):
            return None

    class LeasePool:
        def __init__(self, *, max_workers, mp_context, initializer, initargs):
            self.initargs = initargs
            self.creation_index = len(created)
            self._processes = {0: ProcessStub()}
            created.append(self)

        def submit(self, fn, selection, parallel_config):
            # The original worker bound to group zero hangs. Its successor must
            # be recreated with the same lease and completes normally.
            if self.creation_index == 0:
                return Future()
            worker_config, goal, load_predictor, evaluator, lease = self.initargs
            future = Future()
            future.set_result(
                search_mod._evaluate_one(
                    selection,
                    parallel_config,
                    config=worker_config,
                    goal=goal,
                    load_predictor=load_predictor,
                    evaluator=evaluator,
                    measurement_lease=lease,
                )
            )
            return future

        def shutdown(self, wait=True, cancel_futures=False):
            return None

    def ready_only(futures, timeout=None, return_when=None):
        done = {future for future in futures if future.done()}
        return done, set(futures) - done

    class ReplacementSampler(_SglangFakeSampler):
        def __init__(self, branch, study_id, objectives=None):
            super().__init__(branch, study_id, objectives)
            self.ask_count = 0

        def suggest(self, count):
            self.ask_count += 1
            seqs = [256, 512] if self.ask_count == 1 else [256]
            return [
                Suggestion(
                    selection={
                        "deployment_mode": "agg",
                        "backend": "sglang",
                        "router_mode": "round_robin",
                        "planner_scaling_policy": "disabled",
                        "planner_fpm_sampling": "default",
                        "planner_load_sensitivity": "default",
                        "agg_max_num_batched_tokens": 8192,
                        "agg_max_num_seqs": seqs_value,
                    },
                    parallel_config=self.branch.parallel_configs[0],
                    handle=seqs_value,
                )
                for seqs_value in seqs[:count]
            ]

    class LeaseEvidenceEvaluator:
        def __init__(self):
            self.gpu_ids = []

        def evaluate(self, plan, *, concurrency_override=None):
            gpu_ids = tuple(plan.agg_engine_args["aic_resolution"]["gpu_ids"])
            self.gpu_ids.append(gpu_ids)
            return {
                "output_throughput_tok_s": float(plan.agg_engine_args["max_num_seqs"]),
                "aic_resolution_report": _aic_success_report("overlay", gpu_ids=gpu_ids),
            }

    monkeypatch.setattr(search_mod, "ProcessPoolExecutor", LeasePool)
    monkeypatch.setattr(search_mod, "wait", ready_only)
    evaluator = LeaseEvidenceEvaluator()

    candidates = run_smart_search(
        config,
        evaluator=evaluator,
        sampler_factory=ReplacementSampler,
        show_progress=False,
    )

    assert [pool.initargs[-1].gpu_ids for pool in created] == [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 2, 3),
    ]
    assert evaluator.gpu_ids == [(4, 5, 6, 7), (0, 1, 2, 3)]
    assert {candidate.config["agg_max_num_seqs"] for candidate in candidates} == {256, 512}


def test_over_budget_candidates_dropped(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=16, dp=1, moe_tp=1, moe_ep=16), replicas=4))  # 64 GPUs
    _stub(monkeypatch, branch)
    sampler_seen = {}

    def factory(b, study_id, objectives=None):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(gpu_budget=32), evaluator=_FakeEvaluator(), sampler_factory=factory)
    assert cands == []  # 64 GPUs > 32 budget -> all infeasible, dropped
    # Over-budget trials are told to the optimizer as INFEASIBLE (observe_infeasible),
    # never fed back as a high objective score (which would steer it into the infeasible
    # region). _FakeSampler records observe_infeasible as ("infeasible", reason) tuples.
    scored = sampler_seen["s"].scored
    # Failed candidates do not consume the unique-success budget, so the loop asks for
    # replacements until the fixed 11x safety cap is reached.
    assert len(scored) == 33
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in scored)
    assert all("over gpu_budget" in x[1] for x in scored)  # reason carries the budget breach


def test_eval_failure_marked_infeasible(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class _Boom:
        def evaluate(self, plan, *, concurrency_override=None):
            raise RuntimeError("replay blew up")

    def factory(b, study_id, objectives=None):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_Boom(), sampler_factory=factory)
    assert cands == []
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in sampler_seen["s"].scored)


def test_resolution_failure_does_not_score_candidate(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    seen = {}
    failure = evaluator_mod.UnscorableCandidate(
        code="measurement_failed",
        operation="gemm",
        detail="collector returned no valid samples",
        report={"status": "failed", "failed_keys": ["gemm:abc"]},
    )

    class OneResolutionFailureSampler(_FakeSampler):
        def __init__(self, branch, study_id, objectives=None):
            super().__init__(branch, study_id, objectives)
            self.next_seqs = 256
            self.observed = []
            self.unscorable = []

        def suggest(self, count):
            suggestions = []
            for _ in range(count):
                selection = {
                    "deployment_mode": "agg",
                    "backend": "trtllm",
                    "router_mode": "round_robin",
                    "planner_scaling_policy": "disabled",
                    "planner_fpm_sampling": "default",
                    "planner_load_sensitivity": "default",
                    "agg_max_num_batched_tokens": 8192,
                    "agg_max_num_seqs": self.next_seqs,
                }
                self.next_seqs += 256
                suggestions.append(
                    Suggestion(
                        selection=selection,
                        parallel_config=self.branch.parallel_configs[0],
                        handle=selection,
                    )
                )
            return suggestions

        def observe(self, suggestion, metrics):
            self.observed.append((suggestion.selection["agg_max_num_seqs"], metrics))

        def observe_infeasible(self, suggestion, reason):
            self.unscorable.append((suggestion.selection["agg_max_num_seqs"], reason))

    class OneResolutionFailureEvaluator(_FakeEvaluator):
        def evaluate(self, plan, *, concurrency_override=None):
            if plan.agg_engine_args["max_num_seqs"] == 256:
                return failure
            return super().evaluate(plan, concurrency_override=concurrency_override)

    def factory(branch, study_id, objectives=None):
        sampler = OneResolutionFailureSampler(branch, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    candidates = run_smart_search(
        _config(),
        evaluator=OneResolutionFailureEvaluator(),
        sampler_factory=factory,
        show_progress=False,
    )

    assert [candidate.score for candidate in candidates] == [1024.0, 768.0, 512.0]
    assert [seqs for seqs, _ in seen["sampler"].observed] == [512, 768, 1024]
    assert seen["sampler"].unscorable == [(256, str(failure))]


def test_all_resolution_failures_raise_reason_summary(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    failure = evaluator_mod.UnscorableCandidate(
        code="budget_exhausted",
        operation="attention",
        detail="max_new_keys=2 exhausted",
        report={"status": "failed", "unresolved_count": 3},
    )

    class AlwaysUnscorable:
        def evaluate(self, plan, *, concurrency_override=None):
            return failure

    config = _config().model_copy(update={"sweep": _config().sweep.model_copy(update={"candidates_per_round": 1})})

    with pytest.raises(search_mod.AllCandidatesUnscorableError) as raised:
        run_smart_search(
            config,
            evaluator=AlwaysUnscorable(),
            sampler_factory=_FakeSampler,
            show_progress=False,
        )

    assert raised.value.total_count == 11
    assert raised.value.reasons == (
        search_mod.UnscorableReasonCount(
            code="budget_exhausted",
            operation="attention",
            detail="max_new_keys=2 exhausted",
            report={"status": "failed", "unresolved_count": 3},
            count=11,
        ),
    )
    assert "all 11 evaluated candidates were unscorable" in str(raised.value)


def test_unsupported_backend_config_pair_marked_unsupported(monkeypatch):
    # A (backend, parallel_config) pair the backend can't run is split off on the main
    # process: it's told observe_infeasible ("does not support") and never evaluated.
    pc = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(pc,),
        supported_backends={pc: frozenset({"vllm"})},  # NOT trtllm (what _FakeSampler suggests)
        knob_choices={"backend": ["vllm", "trtllm"]},
    )
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class _NeverCalled:
        def evaluate(self, plan, *, concurrency_override=None):
            raise AssertionError("evaluator must not run for an unsupported (backend, config) pair")

    def factory(b, study_id, objectives=None):
        s = _FakeSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_NeverCalled(), sampler_factory=factory)
    assert cands == []  # nothing evaluated -> no feasible candidate
    scored = sampler_seen["s"].scored
    assert len(scored) == 33  # replacement asks stop at the fixed 11x safety cap
    assert all(isinstance(x, tuple) and x[0] == "infeasible" for x in scored)
    assert all("does not support" in x[1] for x in scored)


def test_study_id_unique_per_run(monkeypatch):
    # Vizier persists studies by id; a fixed id makes a later run inherit a stale
    # study (and its old param space). run_smart_search must use a fresh id per run.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    seen: list[str] = []

    def factory(b, study_id, objectives=None):
        seen.append(study_id)
        return _FakeSampler(b, study_id)

    run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=False)
    run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=False)
    assert len(seen) == 2 and seen[0] != seen[1]  # fresh study per run, no stale reuse
    assert all(s.startswith("spica_agg_") for s in seen)  # study id is per-mode (backend is a knob)


def _config_with_policies(policies, target="throughput"):
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
            "planner_scaling_policy": policies,
        },
        workload={"trace_path": "/tmp/t.jsonl"},
        sweep={"max_rounds": 1, "candidates_per_round": 2, "parallel_evals": 1},  # sequential (fakes)
        goal={"target": target},
    )


def test_non_goodput_sweep_rejects_all_throughput_scaling_policies(monkeypatch):
    # a throughput sweep can't use predictive throughput scaling (no SLA); if EVERY
    # policy enables it, there's nothing to search -> a clear error.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    cfg = _config_with_policies(["throughput_180_5", "hybrid_600_5"], target="throughput")
    with pytest.raises(ValueError, match="throughput scaling"):
        run_smart_search(cfg, evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler, show_progress=False)


def test_non_goodput_sweep_drops_throughput_scaling_and_proceeds(monkeypatch):
    # mixed list -> the throughput-scaling entry is dropped, the rest still run.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    cfg = _config_with_policies(["disabled", "throughput_180_5", "load_180_5"], target="throughput")
    cands = run_smart_search(cfg, evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler, show_progress=False)
    assert [c.score for c in cands] == [512.0, 256.0]  # ran fine (throughput == max_num_seqs)


def test_e2e_only_goodput_drops_planner_scaling_and_proceeds(monkeypatch):
    # e2e-only SLA is valid for goodput, but cannot seed the planner's ttft/itl target.
    # Scaling policies are filtered out before Vizier can sample a build-time-invalid plan.
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    seen = {}

    def fake_enumerate_branches(config, *, max_seq_len=None):
        seen["policies"] = list(config.search_space.planner_scaling_policy)
        return [branch]

    monkeypatch.setattr(search_mod, "enumerate_branches", fake_enumerate_branches)
    monkeypatch.setattr(search_mod, "sweep_load_predictor", lambda config: LoadPredictorResult(reason="static"))
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10")
    cfg = SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
            "planner_scaling_policy": ["disabled", "throughput_180_5", "load_180_5", "hybrid_180_5"],
        },
        workload={"trace_path": "/tmp/t.jsonl"},
        sweep={"max_rounds": 1, "candidates_per_round": 1, "parallel_evals": 1},
        goal={"target": "goodput_per_gpu", "sla": {"e2e_ms": 5000.0}},
    )

    cands = run_smart_search(cfg, evaluator=_FakeEvaluator(), sampler_factory=_FakeSampler, show_progress=False)

    assert seen["policies"] == ["disabled"]
    assert len(cands) == 1


def test_candidate_build_error_is_reported_not_raised(monkeypatch, capsys):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    sampler_seen = {}

    class BadSampler(_FakeSampler):
        def suggest(self, count):
            sel = {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "router_mode": "round_robin",
                "planner_scaling_policy": "disabled",
                "planner_fpm_sampling": "default",
                "planner_load_sensitivity": "default",
                "agg_max_num_batched_tokens": 8192,
                # missing agg_max_num_seqs -> unroll/build failure
            }
            return [Suggestion(selection=sel, parallel_config=self.branch.parallel_configs[0], handle=sel)]

    def factory(b, study_id, objectives=None):
        s = BadSampler(b, study_id)
        sampler_seen["s"] = s
        return s

    cands = run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=True)

    assert cands == []
    scored = sampler_seen["s"].scored
    assert len(scored) == 33  # one bad suggestion per replacement ask, up to the safety cap
    assert scored[0][0] == "infeasible"
    assert "candidate build failed" in scored[0][1]
    assert "smart-sweep failure reason(s): candidate build failed" in capsys.readouterr().out


def test_duplicate_full_samples_use_cache_and_are_replaced(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))
    _stub(monkeypatch, branch)
    seen = {}

    class DuplicateThenUniqueSampler(_FakeSampler):
        def __init__(self, branch, study_id, objectives=None):
            super().__init__(branch, study_id, objectives)
            self.ask_no = 0

        def suggest(self, count):
            self.ask_no += 1
            seqs = [256] * count if self.ask_no == 1 else [512, 768][:count]
            out = []
            for seqs_value in seqs:
                selection = {
                    "deployment_mode": "agg",
                    "backend": "trtllm",
                    "router_mode": "round_robin",
                    "planner_scaling_policy": "disabled",
                    "planner_fpm_sampling": "default",
                    "planner_load_sensitivity": "default",
                    "agg_max_num_batched_tokens": 8192,
                    "agg_max_num_seqs": seqs_value,
                }
                out.append(
                    Suggestion(selection=selection, parallel_config=self.branch.parallel_configs[0], handle=selection)
                )
            return out

    def factory(b, study_id, objectives=None):
        sampler = DuplicateThenUniqueSampler(b, study_id, objectives)
        seen["sampler"] = sampler
        return sampler

    evaluator = _FakeEvaluator()
    candidates = run_smart_search(_config(), evaluator=evaluator, sampler_factory=factory, show_progress=False)

    assert evaluator.calls == 3
    assert {candidate.config["agg_max_num_seqs"] for candidate in candidates} == {256, 512, 768}
    assert len(seen["sampler"].scored) == 5  # every Vizier trial, including duplicates, was told


def test_projection_stall_only_stops_current_branch(monkeypatch):
    parallel = ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2)
    agg = _branch(parallel)
    disagg = BranchSpace(
        deployment_mode="disagg",
        parallel_configs=(parallel,),
        supported_backends={parallel: frozenset({"trtllm"})},
        knob_choices={"backend": ["trtllm"]},
    )
    monkeypatch.setattr(search_mod, "enumerate_branches", lambda config, *, max_seq_len=None: [agg, disagg])
    monkeypatch.setattr(search_mod, "sweep_load_predictor", lambda config: LoadPredictorResult(reason="static"))
    monkeypatch.setattr(search_mod, "resolve_backend_version", lambda hw, be: "1.3.0rc10")
    seen = []

    class RepeatingSampler(_FakeSampler):
        def suggest(self, count):
            suggestions = super().suggest(1)
            return suggestions * count

    class EmptySampler(_FakeSampler):
        def suggest(self, count):
            return []

    def factory(branch, study_id, objectives=None):
        seen.append(branch.deployment_mode)
        sampler_type = RepeatingSampler if branch.deployment_mode == "agg" else EmptySampler
        return sampler_type(branch, study_id, objectives)

    run_smart_search(_config(), evaluator=_FakeEvaluator(), sampler_factory=factory, show_progress=False)

    assert seen == ["agg", "disagg"]


# --- pareto (multi-objective) sweep over candidate-relative KV load ---


def _pareto_config():
    return SmartSearchConfig(
        search_space={
            "model_name": "deepseek-ai/DeepSeek-V3",
            "hardware_sku": "gb200",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
            "gpu_budget": 32,
        },
        workload={"isl": 1024, "osl": 1024, "kv_load_ratio": [0.0, 1.0], "num_request_ratio": 10},
        sweep={"max_rounds": 1, "candidates_per_round": 3, "parallel_evals": 1},
        goal={"target": "pareto"},
    )


# per-concurrency (aggregate throughput, per-user throughput): higher concurrency trades
# more aggregate throughput for less per-user interactivity -> all three are non-dominated.
_PARETO_POINTS = {4: (100.0, 40.0), 8: (150.0, 25.0), 16: (180.0, 12.0)}


class _ParetoSampler:
    """Suggests three KV-load points and records the observed objective vectors."""

    def __init__(self, branch, study_id, objectives=None):
        self.branch = branch
        self.objectives = objectives
        self.observed: list = []

    def suggest(self, count):
        out = []
        for ratio in (0.25, 0.5, 1.0):
            sel = {
                "deployment_mode": "agg",
                "backend": "trtllm",
                "router_mode": "round_robin",
                "planner_scaling_policy": "disabled",
                "planner_fpm_sampling": "default",
                "planner_load_sensitivity": "default",
                "agg_max_num_batched_tokens": 8192,
                "agg_max_num_seqs": 256,
                "kv_load_ratio": ratio,
            }
            out.append(Suggestion(selection=sel, parallel_config=self.branch.parallel_configs[0], handle=sel))
        return out

    def observe(self, suggestion, metrics):
        self.observed.append(metrics)

    def observe_infeasible(self, suggestion, reason):
        self.observed.append(("infeasible", reason))


class _ParetoEvaluator:
    def evaluate(self, plan, *, concurrency_override=None):
        tput, user = _PARETO_POINTS[concurrency_override]
        # avg_gpu = gpu_hours / (duration_ms / 3.6e6) = 1.0 / 1.0 = 1.0 -> throughput_per_gpu == tput
        return {
            "output_throughput_tok_s": tput,
            "mean_output_token_throughput_per_user": user,
            "gpu_hours": 1.0,
            "duration_ms": 3_600_000.0,
        }


def test_pareto_sweep_returns_non_dominated_front(monkeypatch):
    branch = _branch(ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2))  # 8 GPUs
    _stub(monkeypatch, branch)
    concurrency_by_ratio = {0.25: 4, 0.5: 8, 1.0: 16}

    def fake_resolve(sample, *, workload, parallel_config, ratio, backend_version):
        concurrency = concurrency_by_ratio[ratio]
        return KVLoadResolution(
            ratio=ratio,
            concurrency=concurrency,
            concurrency_capacity=16,
            role_capacity_tokens={"agg": 24_576},
        )

    monkeypatch.setattr(search_mod, "resolve_kv_load", fake_resolve)
    seen = {}

    def factory(b, study_id, objectives=None):
        s = _ParetoSampler(b, study_id, objectives)
        seen["s"] = s
        return s

    front = run_smart_search(
        _pareto_config(), evaluator=_ParetoEvaluator(), sampler_factory=factory, show_progress=False
    )
    # all three load points are mutually non-dominated -> full front, sorted by the
    # x-axis (per-user throughput) ascending.
    assert [c.objectives["throughput_per_user"] for c in front] == [12.0, 25.0, 40.0]
    assert [c.objectives["throughput_per_gpu"] for c in front] == [180.0, 150.0, 100.0]
    assert {c.config["concurrency"] for c in front} == {4, 8, 16}  # each point recorded its concurrency
    assert {c.config["kv_load_ratio"] for c in front} == {0.25, 0.5, 1.0}
    # the sampler was built multi-objective (one (name, maximize) per objective) and fed raw vectors
    assert seen["s"].objectives == [("throughput_per_gpu", True), ("throughput_per_user", True)]
    assert all(set(m) == {"throughput_per_gpu", "throughput_per_user"} for m in seen["s"].observed)
