# tests/test_attributed_fpm.py
import os
import sys
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


def test_filter_boundary_discards_drops_first_n_per_cohort():
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    rows = [{"step": s, "batch_size": 32, "past_kv": 100, "measure_run": 0} for s in range(10)]
    rows += [{"step": s, "batch_size": 64, "past_kv": 200, "measure_run": 0} for s in range(5, 12)]
    kept = _filter_boundary_discards(rows, discard_first_n=3)
    a = [r for r in kept if r["batch_size"] == 32]
    b = [r for r in kept if r["batch_size"] == 64]
    # cohort A: min step 0 -> keep step>=3 (steps 3..9 = 7 rows)
    assert min(r["step"] for r in a) == 3 and len(a) == 7
    # cohort B: min step 5 -> keep step>=8 (steps 8..11 = 4 rows)
    assert min(r["step"] for r in b) == 8 and len(b) == 4


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
