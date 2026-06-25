# tests/test_attributed_fpm.py
import pytest

pytestmark = pytest.mark.unit


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
