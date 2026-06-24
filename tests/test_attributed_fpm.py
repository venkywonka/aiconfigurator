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
