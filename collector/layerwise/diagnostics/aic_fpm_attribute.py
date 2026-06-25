# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attributed-FPM: join the profiled nsys lane (composition) to the clean FPM lane
(authoritative wall) and AIC's layerwise prediction, per shape, and decompose the
AIC-vs-FPM gap into named terms (spec slop/fpm-nsys-attribution/spec.md, section 2).

Two lanes, joined by shape:
  * clean lane  -> per-shape ``wall`` (FPM golden run; reused, authoritative timing)
  * profiled lane -> per-shape composition from analyze_sqlite (nsys; never timing)

Decomposition identity (exact):
  gap = aic_total - wall
      = (aic_compute - gpu_compute)  # AIC compute-model error
      + (aic_comm    - gpu_comm)      # AIC comm-model error
      + aic_other                     # AIC scheduler/residual (~0 for dense, TP)
      + overlap                       # compute-comm overlap AIC double-counts
      - overhead                      # real wall not explained by GPU kernels
  where overlap = (gpu_compute + gpu_comm) - gpu_busy and overhead = wall - gpu_busy.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any


def decompose_shape(
    *,
    wall_ms: float,
    aic_compute_ms: float,
    aic_comm_ms: float,
    aic_other_ms: float,
    gpu_compute_ms: float,
    gpu_comm_ms: float,
    gpu_busy_ms: float,
) -> dict[str, float]:
    """Two-lane gap decomposition for a single shape. The five ``term_*`` fields
    sum exactly to ``gap_ms`` by construction (the identity is algebraic, not fit)."""
    aic_total = aic_compute_ms + aic_comm_ms + aic_other_ms
    overlap = (gpu_compute_ms + gpu_comm_ms) - gpu_busy_ms
    overhead = wall_ms - gpu_busy_ms
    return {
        "wall_ms": wall_ms,
        "aic_total_ms": aic_total,
        "aic_compute_ms": aic_compute_ms,
        "aic_comm_ms": aic_comm_ms,
        "aic_other_ms": aic_other_ms,
        "gpu_compute_ms": gpu_compute_ms,
        "gpu_comm_ms": gpu_comm_ms,
        "gpu_busy_ms": gpu_busy_ms,
        "overlap_ms": overlap,
        "overhead_ms": overhead,
        "gap_ms": aic_total - wall_ms,
        "term_compute_err": aic_compute_ms - gpu_compute_ms,
        "term_comm_err": aic_comm_ms - gpu_comm_ms,
        "term_aic_other": aic_other_ms,
        "term_overlap": overlap,
        "term_neg_overhead": -overhead,
    }


def aggregate_profiled_by_shape(
    rows: list[dict[str, Any]],
    *,
    discard_first_n: int = 3,
    aggregate: str = "median",
) -> dict[tuple[int, int], dict[str, float]]:
    """Aggregate analyze_sqlite per-step rows into per-(batch_size, past_kv)
    composition, in milliseconds. Discards sync-drained boundary steps first."""
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    rows = _filter_boundary_discards(rows, discard_first_n=discard_first_n)
    buckets: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(
        lambda: {"compute": [], "comm": [], "busy": []}
    )
    for row in rows:
        key = (int(row["batch_size"]), int(row["past_kv"]))
        buckets[key]["compute"].append(float(row["compute_gpu_us"]) / 1000.0)
        buckets[key]["comm"].append(float(row["comm_gpu_us"]) / 1000.0)
        buckets[key]["busy"].append(float(row["total_union_us"]) / 1000.0)
    reduce = statistics.median if aggregate == "median" else statistics.fmean
    out: dict[tuple[int, int], dict[str, float]] = {}
    for key, series in buckets.items():
        out[key] = {
            "gpu_compute_ms": float(reduce(series["compute"])),
            "gpu_comm_ms": float(reduce(series["comm"])),
            "gpu_busy_ms": float(reduce(series["busy"])),
        }
    return out


def run_decode_attribution(
    *,
    sqlite_path: str,
    profiled_rows: list[dict[str, Any]] | None,
    fpm_wall_by_shape: dict[tuple[int, int], float],
    aic_predict,
    discard_first_n: int = 3,
    aggregate: str = "median",
) -> list[dict[str, Any]]:
    """Join profiled composition + clean FPM wall + AIC breakdown per decode shape.

    Args:
      sqlite_path: nsys .sqlite to reduce (ignored if profiled_rows is given).
      profiled_rows: pre-loaded analyze_sqlite rows (test seam); else analyze_sqlite is called.
      fpm_wall_by_shape: {(batch_size, past_kv): wall_ms} from the CLEAN golden run.
      aic_predict: callable(batch_size, past_kv) -> (compute_ms, comm_ms, total_ms)
                   (wrap aic_fpm_gap.predict_decode_breakdown with bound backend/model/db/rc).
    Returns one decompose_shape() dict per shape present in ALL THREE lanes; shapes
    missing from a lane are skipped and recorded in the 'skipped' list on each row's
    'join' field is omitted -- callers can diff keys to find one-lane-only shapes.
    """
    if profiled_rows is None:
        from collector.layerwise.diagnostics.analyze_nsys_comm_overlap import analyze_sqlite

        profiled_rows, _meta = analyze_sqlite(sqlite_path)
    profiled = aggregate_profiled_by_shape(
        profiled_rows, discard_first_n=discard_first_n, aggregate=aggregate
    )
    out: list[dict[str, Any]] = []
    for key in sorted(set(profiled) & set(fpm_wall_by_shape)):
        batch_size, past_kv = key
        aic_compute, aic_comm, aic_total = aic_predict(batch_size, past_kv)
        if aic_compute is None:
            continue
        comp = profiled[key]
        row = decompose_shape(
            wall_ms=fpm_wall_by_shape[key],
            aic_compute_ms=aic_compute,
            aic_comm_ms=aic_comm,
            aic_other_ms=aic_total - aic_compute - aic_comm,
            gpu_compute_ms=comp["gpu_compute_ms"],
            gpu_comm_ms=comp["gpu_comm_ms"],
            gpu_busy_ms=comp["gpu_busy_ms"],
        )
        row["phase"] = "decode"
        row["batch_size"] = batch_size
        row["past_kv"] = past_kv
        out.append(row)
    return out


def write_decomposition_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    """Write decomposition rows to CSV (stable column order)."""
    import csv

    if not rows:
        return
    cols = [
        "phase", "batch_size", "past_kv", "wall_ms", "aic_total_ms",
        "aic_compute_ms", "aic_comm_ms", "aic_other_ms",
        "gpu_compute_ms", "gpu_comm_ms", "gpu_busy_ms",
        "overlap_ms", "overhead_ms", "gap_ms",
        "term_compute_err", "term_comm_err", "term_aic_other",
        "term_overlap", "term_neg_overhead",
    ]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in cols})
