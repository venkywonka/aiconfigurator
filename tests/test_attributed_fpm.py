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
