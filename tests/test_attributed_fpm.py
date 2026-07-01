# tests/test_attributed_fpm.py
import os
import sys
import pathlib
from unittest import mock

import pytest

pytestmark = pytest.mark.unit

# The step marker imports torch at module load and runs _install() as a side
# effect; stub torch and disable the install so the pure-logic helpers
# (_advance_profiler_window / _parse_profiler_window) are importable without a
# GPU/torch runtime. Mirrors tests/test_vllm_step_marker.py.
os.environ.setdefault("LAYERWISE_STEP_MARKER", "0")
sys.modules.setdefault("torch", mock.Mock())
sys.modules.setdefault("torch.cuda", mock.Mock())
sys.modules.setdefault("torch.cuda.nvtx", mock.Mock())


def test_filter_boundary_discards_chronological_per_measure_run():
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    # Discard is CHRONOLOGICAL per measure_run (NOT per (bs,past) shape): run_min[0]=0,
    # keep step>=3 for BOTH shapes regardless of their own first step.
    rows = [{"step": s, "batch_size": 32, "past_kv": 100, "measure_run": 0} for s in range(10)]
    rows += [{"step": s, "batch_size": 64, "past_kv": 200, "measure_run": 0} for s in range(5, 12)]
    kept = _filter_boundary_discards(rows, discard_first_n=3)
    assert all(r["step"] >= 3 for r in kept)
    assert len([r for r in kept if r["batch_size"] == 32]) == 7   # steps 3..9
    assert len([r for r in kept if r["batch_size"] == 64]) == 7   # steps 5..11 (all >=3)


def test_filter_boundary_discards_keeps_unique_per_step_shapes():
    # Regression for the conc>1 bug: continuous concurrent decode emits a UNIQUE
    # (bs, past) every step. Per-shape cohorting would make each a 1-step cohort and
    # wipe everything; chronological discard keeps all but the first N steps.
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    rows = [{"step": s, "batch_size": 16, "past_kv": 8000 + s, "measure_run": 0} for s in range(20)]
    kept = _filter_boundary_discards(rows, discard_first_n=5)
    assert len(kept) == 15 and min(r["step"] for r in kept) == 5


def test_filter_boundary_discards_zero_is_noop():
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    rows = [{"step": s, "batch_size": 1, "past_kv": 0, "measure_run": 0} for s in range(4)]
    assert _filter_boundary_discards(rows, discard_first_n=0) == rows


def test_split_latency_buckets_compute_comm_other():
    from collector.layerwise.diagnostics.aic_fpm_gap import _split_latency

    latency = {
        "generation_layerwise": 4.0,
        "generation_tp_allreduce": 1.0,
        "generation_moe_ep_alltoall": 0.25,
        "generation_moe_scheduler_overhead": 0.5,
    }
    compute, comm, other = _split_latency(latency)
    assert compute == 4.0
    assert comm == 1.25
    assert other == 0.5
    assert compute + comm + other == sum(latency.values())


def test_predict_decode_breakdown_preserves_total(monkeypatch):
    import collector.layerwise.diagnostics.aic_fpm_gap as G

    monkeypatch.setattr(G, "_classify_source", lambda s: "silicon")

    class FakeBackend:
        def _get_decode_step_latency(self, model, db, rc, *, batch_size, past_kv):
            return ({"generation_layerwise": 4.0, "generation_tp_allreduce": 1.0}, None, {})

    compute, comm, total, src, status = G.predict_decode_breakdown(
        FakeBackend(), None, None, None, batch_size=32, past_kv=100, api={}
    )
    assert (compute, comm, total) == (4.0, 1.0, 5.0)
    assert status == G.ST_OK


def test_decompose_shape_identity_and_overhead():
    from collector.layerwise.diagnostics.aic_fpm_attribute import decompose_shape

    # high-C context-like shape: wall (200) far exceeds GPU busy (48) -> overhead dominates
    r = decompose_shape(
        wall_ms=200.0,
        aic_compute_ms=50.0, aic_comm_ms=8.0, aic_other_ms=0.0,
        gpu_compute_ms=45.0, gpu_comm_ms=6.0, gpu_busy_ms=48.0,
    )
    # the five attributed terms reconstruct the gap exactly
    terms = (r["term_compute_err"] + r["term_comm_err"] + r["term_aic_other"]
             + r["term_overlap"] + r["term_neg_overhead"])
    assert abs(terms - r["gap_ms"]) < 1e-9
    assert r["overhead_ms"] == 152.0          # 200 - 48
    assert r["overlap_ms"] == 3.0             # (45+6) - 48
    assert r["gap_ms"] == 58.0 - 200.0        # aic_total 58 - wall 200


def test_aggregate_profiled_by_shape_us_to_ms_and_discard():
    from collector.layerwise.diagnostics.aic_fpm_attribute import aggregate_profiled_by_shape

    # 5 steps of one decode shape; us inputs; discard first 2 -> median over steps 2,3,4
    rows = [
        {"step": s, "batch_size": 32, "past_kv": 100, "measure_run": 0,
         "compute_gpu_us": 4000.0 + s, "comm_gpu_us": 1000.0, "total_union_us": 4500.0}
        for s in range(5)
    ]
    out = aggregate_profiled_by_shape(rows, discard_first_n=2, aggregate="median")
    key = (32, 100)
    assert key in out
    assert out[key]["gpu_comm_ms"] == 1.0          # 1000us -> 1.0ms
    assert out[key]["gpu_busy_ms"] == 4.5          # 4500us -> 4.5ms
    assert out[key]["gpu_compute_ms"] == 4.003     # median of steps 2,3,4 = 4003us


def test_run_decode_attribution_joins_three_lanes():
    from collector.layerwise.diagnostics.aic_fpm_attribute import run_decode_attribution

    profiled_rows = [
        {"step": s, "batch_size": 32, "past_kv": 100, "measure_run": 0,
         "compute_gpu_us": 4000.0, "comm_gpu_us": 1000.0, "total_union_us": 4500.0}
        for s in range(5)
    ]
    fpm_wall = {(32, 100): 6.0}  # ms
    # shape (32,100) present in all three lanes; (64,200) only in FPM -> skipped
    fpm_wall[(64, 200)] = 9.0
    aic = lambda bs, kv: (5.5, 0.8, 6.3) if (bs, kv) == (32, 100) else (None, None, None)

    rows = run_decode_attribution(
        sqlite_path="UNUSED", profiled_rows=profiled_rows,
        fpm_wall_by_shape=fpm_wall, aic_predict=aic, discard_first_n=2,
    )
    assert len(rows) == 1
    r = rows[0]
    assert (r["batch_size"], r["past_kv"]) == (32, 100)
    assert r["gpu_busy_ms"] == 4.5 and r["overhead_ms"] == 1.5  # 6.0 - 4.5
    terms = (r["term_compute_err"] + r["term_comm_err"] + r["term_aic_other"]
             + r["term_overlap"] + r["term_neg_overhead"])
    assert abs(terms - r["gap_ms"]) < 1e-9


def test_run_decode_attribution_rounds_float_fpm_kv_to_profiled_past_kv():
    """KEY-SPACE ALIGNMENT (TASK C step 2).

    Profiled lane keys decode rows by (batch_size, past_kv) where past_kv is the
    NVTX label = ``round(mean(num_computed_tokens))`` -> an INTEGER (see
    dynamo_step_marker._decode_batch_and_kv). The clean FPM lane (_load_fpm decode)
    keys by (batch_size, mean_decode_kv_tokens) where mean_kv is a RAW FLOAT. A naive
    ``set(profiled) & set(fpm)`` join therefore drops every FPM shape whose mean_kv is
    not an exact integer (e.g. 100.4), because (32, 100) != (32, 100.4).

    Aligning the two means binning the FPM float key with the SAME round() the NVTX
    marker uses, so a profiled row at past_kv=K joins to an FPM shape at mean_kv≈K.
    """
    from collector.layerwise.diagnostics.aic_fpm_attribute import run_decode_attribution

    profiled_rows = [
        {"step": s, "batch_size": 32, "past_kv": 100, "measure_run": 0,
         "compute_gpu_us": 4000.0, "comm_gpu_us": 1000.0, "total_union_us": 4500.0}
        for s in range(5)
    ]
    # FPM mean_kv is a non-integer float that rounds to the profiled past_kv (100).
    fpm_wall = {(32, 100.4): 6.0}
    aic = lambda bs, kv: (5.5, 0.8, 6.3) if (bs, kv) == (32, 100) else (None, None, None)

    rows = run_decode_attribution(
        sqlite_path="UNUSED", profiled_rows=profiled_rows,
        fpm_wall_by_shape=fpm_wall, aic_predict=aic, discard_first_n=2,
    )
    assert len(rows) == 1, "float FPM mean_kv must bin to the integer profiled past_kv"
    r = rows[0]
    assert (r["batch_size"], r["past_kv"]) == (32, 100)
    assert r["wall_ms"] == 6.0
    # aic_predict is invoked with the integer profiled past_kv (100), not the float.
    assert r["aic_total_ms"] == 6.3


def test_bin_fpm_wall_to_profiled_key_rounds_and_aggregates_collisions():
    """_bin_fpm_wall_to_profiled_key locks the FPM->profiled key transform: float
    mean_kv -> int(round(mean_kv)) (the NVTX marker's round), and two FPM floats that
    round to the same integer bin are aggregated (mean) into one wall."""
    from collector.layerwise.diagnostics.aic_fpm_attribute import _bin_fpm_wall_to_profiled_key

    binned = _bin_fpm_wall_to_profiled_key({
        (32, 100.4): 6.0,   # -> (32, 100)
        (32, 99.7): 8.0,    # -> (32, 100), collides with the above -> mean(6,8)=7
        (16, 4096.0): 3.0,  # exact integer float -> (16, 4096)
    })
    assert binned[(32, 100)] == 7.0
    assert binned[(16, 4096)] == 3.0
    # keys are pure ints (not floats), matching the profiled (batch_size, past_kv) space
    assert all(isinstance(b, int) and isinstance(k, int) for (b, k) in binned)


def test_advance_profiler_window_opens_once_closes_once():
    from collector.layerwise.vllm.vllm_step_marker import _advance_profiler_window

    calls = []
    state = {"active": False}
    spans = [(3, 6)]
    for step in range(0, 9):
        _advance_profiler_window(step, spans, state, lambda action: calls.append((step, action)))
    # start exactly at step 3, stop exactly at step 6, nothing else
    assert calls == [(3, "start"), (6, "stop")]
    assert state["active"] is False


def test_advance_profiler_window_two_windows():
    from collector.layerwise.vllm.vllm_step_marker import _advance_profiler_window, _parse_profiler_window

    spans = _parse_profiler_window("3-6,10-12")
    assert spans == [(3, 6), (10, 12)]
    calls = []
    state = {"active": False}
    for step in range(0, 14):
        _advance_profiler_window(step, spans, state, lambda a: calls.append((step, a)))
    assert calls == [(3, "start"), (6, "stop"), (10, "start"), (12, "stop")]


def test_collect_threads_nsys_flags_to_inner_shell():
    """stage_attribute -> collect.py -> docker.build_collect_command -> collect_fpm_metrics.sh.
    The nsys flags must be parsed by collect.py and threaded into the inner shell argv."""
    import types
    from collector.layerwise.fpm import collect as C
    from collector.layerwise.fpm import docker as D

    case = types.SimpleNamespace(tp_size=1, ep_size=1, decode_past_kv=4096)

    # with flags: collect.py must accept them, docker must forward them to the inner shell
    args = C._build_arg_parser().parse_args(
        ["--model", "Qwen/Qwen3-0.6B", "--nsys-profile-worker",
         "--nsys-cuda-profiler-window", "20-30"]
    )
    assert args.nsys_profile_worker is True
    assert args.nsys_cuda_profiler_window == "20-30"
    argv = D.build_collect_command(args, case, __import__("pathlib").Path("/tmp/x")).argv
    assert argv[1].endswith("collect_fpm_metrics.sh")
    assert "--nsys-profile-worker" in argv
    i = argv.index("--nsys-cuda-profiler-window")
    assert argv[i + 1] == "20-30"
    # the cuda-profiler-window must precede any '--' extra-vllm-arg separator
    if "--" in argv:
        assert i < argv.index("--")


def test_collect_omits_nsys_flags_when_unset():
    import types
    from collector.layerwise.fpm import collect as C
    from collector.layerwise.fpm import docker as D

    case = types.SimpleNamespace(tp_size=1, ep_size=1, decode_past_kv=4096)
    args = C._build_arg_parser().parse_args(["--model", "Qwen/Qwen3-0.6B"])
    assert args.nsys_profile_worker is False
    assert args.nsys_cuda_profiler_window is None
    argv = D.build_collect_command(args, case, __import__("pathlib").Path("/tmp/x")).argv
    assert "--nsys-profile-worker" not in argv
    assert "--nsys-cuda-profiler-window" not in argv


# ---------------------------------------------------------------------------
# TASK A: dynamo_step_marker pure-logic helpers
# ---------------------------------------------------------------------------
#
# These cover the DYNAMO-side per-step NVTX marker for the FPM/attribute path:
# a hook on InstrumentedScheduler.update_from_output that reads the REAL
# per-step batch state (decode_batch + mean past_kv) and emits a label in the
# EXACT format the existing nsys parser keys on. Pure helpers only; the
# monkeypatch effect needs torch+dynamo and is validated on hardware.


def _fake_scheduler_output(req_ids, num_computed_tokens, context_phase_ids):
    """Build a fake SchedulerOutput-like object.

    scheduled_cached_reqs mirrors vLLM's CachedRequestData: parallel lists
    .req_ids / .num_computed_tokens, plus an .is_context_phase(req_id) method.
    """
    import types

    ctx = set(context_phase_ids)
    cached = types.SimpleNamespace(
        req_ids=list(req_ids),
        num_computed_tokens=list(num_computed_tokens),
        is_context_phase=lambda req_id: req_id in ctx,
    )
    return types.SimpleNamespace(scheduled_cached_reqs=cached)


def test_decode_batch_and_kv_mixed_prefill_and_decode():
    from collector.layerwise.vllm.dynamo_step_marker import _decode_batch_and_kv

    # 4 cached reqs: r0 prefill (context phase), r1/r2/r3 decode.
    # decode num_computed = [100, 200, 300] -> mean 200.
    so = _fake_scheduler_output(
        req_ids=["r0", "r1", "r2", "r3"],
        num_computed_tokens=[5, 100, 200, 300],
        context_phase_ids=["r0"],
    )
    decode_batch, mean_kv = _decode_batch_and_kv(so)
    assert decode_batch == 3
    assert mean_kv == 200


def test_decode_batch_and_kv_all_decode_rounds_mean():
    from collector.layerwise.vllm.dynamo_step_marker import _decode_batch_and_kv

    # all-decode batch; num_computed = [100, 101] -> mean 100.5 -> round 100
    so = _fake_scheduler_output(
        req_ids=["a", "b"],
        num_computed_tokens=[100, 101],
        context_phase_ids=[],
    )
    decode_batch, mean_kv = _decode_batch_and_kv(so)
    assert decode_batch == 2
    assert mean_kv == 100


def test_decode_batch_and_kv_no_decode_reqs_is_zero():
    from collector.layerwise.vllm.dynamo_step_marker import _decode_batch_and_kv

    # all reqs in context/prefill phase -> no decode reqs -> (0, 0)
    so = _fake_scheduler_output(
        req_ids=["p0", "p1"],
        num_computed_tokens=[3, 7],
        context_phase_ids=["p0", "p1"],
    )
    decode_batch, mean_kv = _decode_batch_and_kv(so)
    assert decode_batch == 0
    assert mean_kv == 0


def test_decode_batch_and_kv_tolerates_missing_attrs():
    import types

    from collector.layerwise.vllm.dynamo_step_marker import _decode_batch_and_kv

    # scheduler_output with no scheduled_cached_reqs at all -> degrade to (0, 0)
    so = types.SimpleNamespace()
    assert _decode_batch_and_kv(so) == (0, 0)


def test_bench_step_label_exact_format():
    from collector.layerwise.vllm.dynamo_step_marker import _bench_step_label

    assert _bench_step_label(16, 128, 15) == "bench_step::N0000016::bs128::past000015"


def test_bench_step_label_roundtrips_through_parser_regex():
    from collector.layerwise.vllm.dynamo_step_marker import _bench_step_label
    from collector.layerwise.common.parse_nsys_step_sweep import _BENCH_STEP_RE

    label = _bench_step_label(42, 64, 4096)
    m = _BENCH_STEP_RE.search(label)
    assert m is not None
    assert int(m.group(1)) == 42
    assert int(m.group(2)) == 64
    assert int(m.group(3)) == 4096


def test_run_decode_attribution_divides_profiled_by_ranks():
    """TP>1: analyze_sqlite sums kernels across all ranks, but wall+AIC are single-rank.
    ranks= divides the profiled compute/comm SUMS to per-rank; gpu_busy (union) stays."""
    from collector.layerwise.diagnostics.aic_fpm_attribute import run_decode_attribution

    profiled_rows = [
        {"step": s, "batch_size": 4, "past_kv": 8000, "measure_run": 0,
         "compute_gpu_us": 40000.0, "comm_gpu_us": 8000.0, "total_union_us": 6500.0}
        for s in range(5)
    ]
    fpm_wall = {(4, 8000.0): 6.6}
    aic = lambda bs, kv: (6.9, 0.37, 7.27)
    rows = run_decode_attribution(
        sqlite_path="X", profiled_rows=profiled_rows, fpm_wall_by_shape=fpm_wall,
        aic_predict=aic, discard_first_n=2, ranks=8,
    )
    r = rows[0]
    assert r["gpu_compute_ms"] == 5.0   # 40000us -> 40.0ms / 8 ranks
    assert r["gpu_comm_ms"] == 1.0      # 8000us  -> 8.0ms  / 8 ranks
    assert r["gpu_busy_ms"] == 6.5      # union NOT divided
    terms = (r["term_compute_err"] + r["term_comm_err"] + r["term_aic_other"]
             + r["term_overlap"] + r["term_neg_overhead"])
    assert abs(terms - r["gap_ms"]) < 1e-9


# ---------------------------------------------------------------------------
# TASK A: CONTEXT-PHASE attribution (the high-C context puzzle).
# Context chunks are UNIFORM 2048-token prefill steps (FPM_MAX_NUM_BATCHED_TOKENS),
# so PURE-CONTEXT profiled steps (decode_batch==0 -> bs0 in the dynamo marker label)
# aggregate cleanly without per-shape keys.
# ---------------------------------------------------------------------------


def test_aggregate_profiled_context_selects_bs0_and_divides_by_ranks():
    """aggregate_profiled_context picks ONLY pure-context profiled steps (batch_size==0,
    the dynamo marker's bs0 pure-prefill label), converts us->ms, and divides
    compute/comm by ranks (per-rank, like decode); gpu_busy (union) is left as-is."""
    from collector.layerwise.diagnostics.aic_fpm_attribute import aggregate_profiled_context

    # mix: 5 pure-context steps (bs0) + 3 decode steps (bs>0) that MUST be ignored.
    rows = [
        {"step": s, "batch_size": 0, "past_kv": 0, "measure_run": 0,
         "compute_gpu_us": 320000.0, "comm_gpu_us": 16000.0, "total_union_us": 48000.0}
        for s in range(5)
    ]
    rows += [
        {"step": 100 + s, "batch_size": 32, "past_kv": 100, "measure_run": 0,
         "compute_gpu_us": 4000.0, "comm_gpu_us": 1000.0, "total_union_us": 4500.0}
        for s in range(3)
    ]
    out = aggregate_profiled_context(rows, discard_first_n=2, aggregate="median", ranks=8)
    # only the bs0 cohort, discard first 2 -> median over steps 2,3,4 (all identical).
    assert out["gpu_compute_ms"] == 40.0   # 320000us -> 320.0ms / 8 ranks
    assert out["gpu_comm_ms"] == 2.0       # 16000us  -> 16.0ms  / 8 ranks
    assert out["gpu_busy_ms"] == 48.0      # 48000us -> 48.0ms; union NOT divided


def test_run_decode_attribution_per_pid_preserves_each_rank_undivided():
    """per_pid=True: feed SYNTHETIC per-PID analyze_sqlite rows (8 ranks of ONE
    (bs,past_kv) with UNEQUAL compute to model TP skew) and assert:
      (i)  each rank's compute/comm is preserved UN-divided (NO /ranks),
      (ii) rows are keyed PER RANK -> cross-rank variance survives (not medianed
           to a single value),
      (iii) the existing per_pid=False aggregate for the SAME input is unchanged
            (cross-rank SUM /ranks, cross-rank-union gpu_busy).
    """
    from collector.layerwise.diagnostics.aic_fpm_attribute import run_decode_attribution

    # 8 ranks, one shape (4, 8000). Per-rank compute SKEWED: rank r contributes
    # (40000 + 1000*r) us. Repeat each rank over 5 steps (same value) so the
    # per-(rank,shape) median is well-defined. comm uniform 8000us, busy 6500us.
    profiled_rows = []
    for r in range(8):
        for s in range(5):
            profiled_rows.append({
                "step": s, "batch_size": 4, "past_kv": 8000, "measure_run": 0,
                "pid": r,
                "compute_gpu_us": 40000.0 + 1000.0 * r,
                "comm_gpu_us": 8000.0,
                "total_union_us": 6500.0,
            })
    fpm_wall = {(4, 8000.0): 6.6}
    aic = lambda bs, kv: (6.9, 0.37, 7.27)

    rows = run_decode_attribution(
        sqlite_path="X", profiled_rows=profiled_rows, fpm_wall_by_shape=fpm_wall,
        aic_predict=aic, discard_first_n=2, ranks=8, per_pid=True,
    )

    # one aggregate row (pid empty) + 8 per-rank rows.
    per_rank = sorted((r for r in rows if r.get("pid") not in (None, "")),
                      key=lambda r: r["pid"])
    assert len(per_rank) == 8, "must emit one row per captured rank"
    # (i) + (ii): each rank's compute is its OWN value (un-divided, not medianed).
    for r in per_rank:
        rank = int(r["pid"])
        assert r["gpu_compute_ms"] == 40.0 + rank, f"rank {rank} compute must be undivided"
        assert r["gpu_comm_ms"] == 8.0           # 8000us, undivided
        assert r["gpu_busy_ms"] == 6.5           # this rank's own union
        assert (r["batch_size"], r["past_kv"]) == (4, 8000)
        terms = (r["term_compute_err"] + r["term_comm_err"] + r["term_aic_other"]
                 + r["term_overlap"] + r["term_neg_overhead"])
        assert abs(terms - r["gap_ms"]) < 1e-9
    # variance survives: 8 distinct compute values, not a single medianed one.
    assert len({r["gpu_compute_ms"] for r in per_rank}) == 8

    # (iii) the aggregate row alongside MUST equal the old per_pid=False numbers.
    # per_pid=False sees the SAME per-rank rows as a single merged cohort, so the
    # cross-rank SUM /ranks reproduces today's behavior. Build the aggregate-only
    # input by summing per-rank compute/comm per step (the merged-lane semantics).
    agg = [r for r in rows if r.get("pid") in (None, "")]
    assert len(agg) == 1
    ar = agg[0]
    # aggregate is computed from the per_pid=False reduction (sum across ranks /ranks).
    # sum_r (40000+1000r) = 320000 + 1000*28 = 348000us -> 348.0ms /8 = 43.5ms
    assert ar["gpu_compute_ms"] == 43.5
    # comm sum = 8*8000 = 64000us -> 64.0ms /8 = 8.0ms
    assert ar["gpu_comm_ms"] == 8.0
    assert ar["pid"] in (None, "")
    aterms = (ar["term_compute_err"] + ar["term_comm_err"] + ar["term_aic_other"]
              + ar["term_overlap"] + ar["term_neg_overhead"])
    assert abs(aterms - ar["gap_ms"]) < 1e-9


def test_run_context_attribution_identity_holds():
    """run_context_attribution produces a single aggregate context decompose_shape row
    whose five attributed terms sum exactly to the gap, tagged phase='context'."""
    from collector.layerwise.diagnostics.aic_fpm_attribute import run_context_attribution

    profiled_rows = [
        {"step": s, "batch_size": 0, "past_kv": 0, "measure_run": 0,
         "compute_gpu_us": 320000.0, "comm_gpu_us": 16000.0, "total_union_us": 48000.0}
        for s in range(5)
    ]
    # high-C context puzzle: FPM wall (219) far exceeds GPU busy (48) -> overhead dominates.
    fpm_ctx_wall_ms = 219.0
    aic_ctx_predict = lambda: (50.0, 8.0, 58.0)  # (compute, comm, total)

    row = run_context_attribution(
        sqlite_path="UNUSED", profiled_rows=profiled_rows,
        fpm_ctx_wall_ms=fpm_ctx_wall_ms, aic_ctx_predict=aic_ctx_predict,
        discard_first_n=2, ranks=8,
    )
    assert row["phase"] == "context"
    assert row["wall_ms"] == 219.0
    assert row["aic_total_ms"] == 58.0
    assert row["gpu_compute_ms"] == 40.0   # 320.0ms / 8
    assert row["gpu_comm_ms"] == 2.0       # 16.0ms / 8
    assert row["gpu_busy_ms"] == 48.0      # union, undivided
    assert row["overhead_ms"] == 171.0     # 219 - 48
    terms = (row["term_compute_err"] + row["term_comm_err"] + row["term_aic_other"]
             + row["term_overlap"] + row["term_neg_overhead"])
    assert abs(terms - row["gap_ms"]) < 1e-9


def test_stage_attribute_decomposes_against_clean_fpm_run_not_profiled_attribute_run():
    script = pathlib.Path("collector/layerwise/reproduce_layerwise_fpm.sh").read_text()

    assert 'local clean_fpm_run; clean_fpm_run="$(fpm_run_dir "$slug" "$pname")"' in script
    assert '--fpm-run "$clean_fpm_run"' in script
    assert '--profiled-fpm-run "$rdir"' in script


def test_stage_attribute_reuses_clean_fpm_prompt_seed_for_matching_shapes():
    script = pathlib.Path("collector/layerwise/reproduce_layerwise_fpm.sh").read_text()

    assert 'local seed_env=(PROMPT_TOKEN_SEED="$i")' in script
    assert 'run_env "${seed_env[@]} MAX_NUM_SEQS=$FPM_MAX_NUM_SEQS' in script
