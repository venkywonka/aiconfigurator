# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the per-concurrency FPM data-profile plotter.

Exercises ``collector/layerwise/diagnostics/plot_fpm_distributions.py`` against a
synthetic ``fpm_metrics_phase.csv`` written in the exact schema
``summarize_fpm.py`` emits -- no GPU, no AIC lookup, no collector. Verifies phase
CSV resolution (flat + nested + explicit), phase normalisation, empty-CSV
handling, and that the three profile PNGs are produced with non-trivial content.
"""

from __future__ import annotations

import csv

import pytest

pytestmark = pytest.mark.unit

from collector.layerwise.diagnostics.plot_fpm_distributions import (
    generate_all,
    load_phase_frame,
    main,
    resolve_phase_csv,
)

# Schema mirrored from collector/layerwise/fpm_ground_truth/summarize_fpm.py.
_PHASE_COLUMNS = [
    "phase",
    "workload_segment",
    "counter_id",
    "worker_id",
    "dp_rank",
    "ctx_tokens",
    "ctx_requests",
    "ctx_kv_tokens",
    "decode_tokens",
    "decode_requests",
    "decode_kv_tokens",
    "mean_decode_kv_tokens",
    "queued_ctx_tokens",
    "queued_ctx_requests",
    "queued_decode_requests",
    "queued_decode_kv_tokens",
    "latency_ms",
]


def _row(phase, **extra):
    row = dict.fromkeys(_PHASE_COLUMNS, 0)
    row["phase"] = phase
    row["workload_segment"] = "real"
    row.update(extra)
    return row


def _write_phase_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PHASE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in _PHASE_COLUMNS})


def _mixed_rows():
    """A small mixed run: context, decode-batch, and mixed steps."""

    rows = []
    # A few context (prefill) steps of varying new-token counts.
    for tokens in (512, 2048, 8192):
        rows.append(_row("context", ctx_tokens=tokens, ctx_requests=1, latency_ms=tokens / 40.0))
    # Decode steps at a few batch sizes and KV lengths.
    for bs in (1, 4, 16, 64):
        for kv in (2048, 8192):
            rows.append(
                _row(
                    "decode",
                    decode_requests=bs,
                    decode_tokens=bs,
                    decode_kv_tokens=bs * kv,
                    mean_decode_kv_tokens=kv,
                    latency_ms=5.0 + bs * 0.1,
                )
            )
    # A couple of mixed steps.
    for _ in range(3):
        rows.append(
            _row(
                "mixed",
                ctx_tokens=1024,
                ctx_requests=1,
                decode_requests=8,
                decode_kv_tokens=8 * 4096,
                mean_decode_kv_tokens=4096,
                latency_ms=42.0,
            )
        )
    return rows


def test_resolve_phase_csv_explicit(tmp_path):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, [_row("decode", decode_requests=1)])
    assert resolve_phase_csv(csv_path, None) == csv_path


def test_resolve_phase_csv_flat_run_dir(tmp_path):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, [_row("decode", decode_requests=1)])
    assert resolve_phase_csv(None, tmp_path) == csv_path


def test_resolve_phase_csv_nested_run_dir(tmp_path):
    nested = tmp_path / "tp8_ep1_past4096"
    nested.mkdir()
    csv_path = nested / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, [_row("decode", decode_requests=1)])
    assert resolve_phase_csv(None, tmp_path) == csv_path


def test_resolve_phase_csv_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_phase_csv(None, tmp_path)
    with pytest.raises(ValueError):
        resolve_phase_csv(None, None)


def test_load_phase_frame_normalizes_phase(tmp_path):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    # Legacy emitters use ctx/gen; they must fold into context/decode.
    _write_phase_csv(csv_path, [_row("ctx", ctx_tokens=10), _row("gen", decode_requests=2)])
    frame = load_phase_frame(csv_path)
    assert set(frame["phase"]) == {"context", "decode"}
    # Numeric coercion keeps the columns numeric.
    assert frame["decode_requests"].max() == 2


def test_generate_all_writes_three_pngs(tmp_path):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, _mixed_rows())
    frame = load_phase_frame(csv_path)
    out_dir = tmp_path / "profile"
    paths = generate_all(frame, out_dir, "test c8")
    names = {p.name for p in paths}
    assert names == {
        "fpm_distribution_composition.png",
        "fpm_distribution_params.png",
        "fpm_distribution_scatter.png",
    }
    for p in paths:
        assert p.is_file()
        # A real figure is comfortably over a few KB; catch a blank/zero write.
        assert p.stat().st_size > 2000


def test_main_cli_smoke(tmp_path, capsys):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, _mixed_rows())
    out_dir = tmp_path / "profile"
    rc = main(["--fpm-csv", str(csv_path), "--out-dir", str(out_dir), "--concurrency", "8"])
    assert rc == 0
    assert (out_dir / "fpm_distribution_composition.png").is_file()


def test_main_empty_csv_is_fail_safe(tmp_path):
    csv_path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(csv_path, [])  # header only, zero rows
    out_dir = tmp_path / "profile"
    # An empty run must not crash the plotter (fail-safe): return 0, no PNGs.
    rc = main(["--fpm-csv", str(csv_path), "--out-dir", str(out_dir)])
    assert rc == 0


def test_main_missing_csv_returns_error(tmp_path):
    rc = main(["--fpm-csv", str(tmp_path / "nope.csv"), "--out-dir", str(tmp_path / "o")])
    assert rc == 1
