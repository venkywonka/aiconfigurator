# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-step composition report (comm-concurrency experiment).

Exercises the PURE-LOGIC join/composition functions of
``collector/layerwise/diagnostics/fpm_step_composition.py`` against synthetic
inputs -- no nsys sqlite, no torch, no GPU. The nsys lane is represented by the
dict rows ``analyze_sqlite(per_pid=False)`` produces (step/batch_size/past_kv/
measure_run + kernel timing), and the FPM lane by a temp ``fpm_metrics_phase.csv``
written in the exact schema ``summarize_fpm.py`` emits.
"""

import csv

import pytest

pytestmark = pytest.mark.unit

from collector.layerwise.diagnostics.fpm_step_composition import (  # noqa: E402
    OUTPUT_COLUMNS,
    build_composition_rows,
    load_fpm_phase_rows,
    select_nsys_window_steps,
    summarize,
    _parse_window,
    _shape_matches,
)

# Schema mirrored from collector/layerwise/fpm_ground_truth/summarize_fpm.py.
_PHASE_COLUMNS = [
    "phase", "workload_segment", "counter_id", "worker_id", "dp_rank",
    "ctx_tokens", "ctx_requests", "ctx_kv_tokens",
    "decode_tokens", "decode_requests", "decode_kv_tokens", "mean_decode_kv_tokens",
    "queued_ctx_tokens", "queued_ctx_requests",
    "queued_decode_requests", "queued_decode_kv_tokens", "latency_ms",
]


def _nsys_row(step, bs, past, run=0, **extra):
    row = {
        "step": step, "batch_size": bs, "past_kv": past, "measure_run": run,
        "pid": "", "compute_gpu_us": 100.0, "comm_gpu_us": 10.0, "comm_visible_us": 2.0,
    }
    row.update(extra)
    return row


def _write_phase_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PHASE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            full = {c: r.get(c, "") for c in _PHASE_COLUMNS}
            writer.writerow(full)


def test_parse_window_single_and_list_and_empty():
    assert _parse_window(None) is None
    assert _parse_window("") is None
    assert _parse_window("100-115") == (100, 115)
    # comma list -> first window only (matches --nsys-cuda-profiler-window first span)
    assert _parse_window("100-115,200-210") == (100, 115)
    with pytest.raises(ValueError):
        _parse_window("100")


def test_select_nsys_window_steps_filters_and_sorts():
    rows = [
        _nsys_row(120, 0, 2048),
        _nsys_row(101, 16, 4096),
        _nsys_row(100, 16, 4096),
        _nsys_row(99, 16, 4096),
    ]
    selected = select_nsys_window_steps(rows, window=(100, 115))
    assert [r["step"] for r in selected] == [100, 101]  # 99 and 120 excluded, sorted


def test_select_nsys_window_steps_sorts_by_step_then_run():
    rows = [_nsys_row(5, 4, 4096, run=2), _nsys_row(5, 4, 4096, run=1), _nsys_row(4, 4, 4096, run=0)]
    selected = select_nsys_window_steps(rows, window=None)
    assert [(r["step"], r["measure_run"]) for r in selected] == [(4, 0), (5, 1), (5, 2)]


def test_shape_matches_exact_decode_and_rounded_kv():
    fpm = {"decode_requests": "16", "mean_decode_kv_tokens": "4096.4"}
    assert _shape_matches(16, 4096, fpm) is True   # round(4096.4) == 4096
    assert _shape_matches(15, 4096, fpm) is False  # decode mismatch
    assert _shape_matches(16, 4097, fpm) is False  # kv mismatch


def test_load_fpm_phase_rows_filters_rank_and_idle(tmp_path):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, [
        {"phase": "decode", "counter_id": "12", "dp_rank": "0", "worker_id": "w0",
         "decode_requests": "4", "mean_decode_kv_tokens": "4096.0"},
        {"phase": "idle", "counter_id": "13", "dp_rank": "0", "worker_id": "w0"},
        {"phase": "decode", "counter_id": "14", "dp_rank": "1", "worker_id": "w0",
         "decode_requests": "4", "mean_decode_kv_tokens": "4096.0"},
        {"phase": "decode", "counter_id": "11", "dp_rank": "0", "worker_id": "w0",
         "decode_requests": "1", "mean_decode_kv_tokens": "4096.0"},
    ])
    rows = load_fpm_phase_rows(csv_path, dp_rank=0)
    # idle dropped, dp_rank=1 dropped, sorted by counter_id (11 before 12)
    assert [r["counter_id"] for r in rows] == ["11", "12"]


def test_build_composition_rows_sequence_join_and_shape_flag(tmp_path):
    nsys_steps = [
        _nsys_row(100, 4, 4096),
        _nsys_row(101, 16, 4096),
    ]
    fpm_rows = [
        {"phase": "decode", "counter_id": "50", "dp_rank": "0",
         "decode_requests": "4", "decode_kv_tokens": "16384",
         "mean_decode_kv_tokens": "4096.0", "ctx_tokens": "0", "latency_ms": "1.2"},
        {"phase": "decode", "counter_id": "51", "dp_rank": "0",
         "decode_requests": "99", "decode_kv_tokens": "405504",
         "mean_decode_kv_tokens": "4096.0", "ctx_tokens": "0", "latency_ms": "2.4"},
    ]
    rows = build_composition_rows(nsys_steps, fpm_rows)
    assert len(rows) == 2
    # seq 0: marker bs=4 vs fpm decode_requests=4 -> match
    assert rows[0]["seq"] == 0
    assert rows[0]["marker_step"] == 100
    assert rows[0]["fpm_counter_id"] == 50
    assert rows[0]["shape_match"] == 1
    assert rows[0]["phase"] == "decode"
    assert rows[0]["decode_requests"] == 4
    # seq 1: marker bs=16 vs fpm decode_requests=99 -> mismatch flagged
    assert rows[1]["shape_match"] == 0
    # output dicts cover exactly the declared columns
    assert set(rows[0].keys()) == set(OUTPUT_COLUMNS)


def test_build_composition_rows_length_mismatch_blanks_missing_side():
    nsys_steps = [_nsys_row(100, 4, 4096), _nsys_row(101, 4, 4096)]
    fpm_rows = [
        {"phase": "decode", "counter_id": "50", "dp_rank": "0",
         "decode_requests": "4", "mean_decode_kv_tokens": "4096.0"},
    ]
    rows = build_composition_rows(nsys_steps, fpm_rows)
    assert len(rows) == 2  # max(2, 1)
    # extra nsys step has no FPM pair -> blank fpm fields, shape_match=0
    assert rows[1]["marker_step"] == 101
    assert rows[1]["fpm_counter_id"] == ""
    assert rows[1]["phase"] == ""
    assert rows[1]["shape_match"] == 0


def test_summarize_counts_phases_and_match_rate():
    nsys_steps = [_nsys_row(1, 4, 4096), _nsys_row(2, 16, 4096), _nsys_row(3, 0, 2048)]
    fpm_rows = [
        {"phase": "decode", "counter_id": "1", "dp_rank": "0",
         "decode_requests": "4", "mean_decode_kv_tokens": "4096.0"},
        {"phase": "mixed", "counter_id": "2", "dp_rank": "0",
         "decode_requests": "16", "mean_decode_kv_tokens": "4096.0"},
        {"phase": "context", "counter_id": "3", "dp_rank": "0",
         "decode_requests": "7", "mean_decode_kv_tokens": "4096.0"},  # mismatch (0 vs 7)
    ]
    rows = build_composition_rows(nsys_steps, fpm_rows)
    summary = summarize(rows)
    assert summary["rows"] == 3
    assert summary["paired"] == 3
    assert summary["shape_matched"] == 2          # decode + mixed match, context mismatches
    assert summary["phase_counts"] == {"decode": 1, "mixed": 1, "context": 1}
    assert summary["shape_match_rate"] == pytest.approx(2 / 3)
