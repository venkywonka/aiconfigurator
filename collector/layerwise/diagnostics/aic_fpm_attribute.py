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


def _bin_fpm_wall_to_profiled_key(
    fpm_wall_by_shape: dict[tuple[int, float], float],
) -> dict[tuple[int, int], float]:
    """Re-key the clean FPM decode wall into the profiled-lane integer key space.

    The two decode lanes are keyed differently:
      * profiled (analyze_sqlite): (batch_size, past_kv) where past_kv is the NVTX
        ``bench_step`` label = ``round(mean(num_computed_tokens))`` -> an INTEGER
        (collector/layerwise/vllm/dynamo_step_marker._decode_batch_and_kv).
      * clean FPM (_load_fpm decode): (decode_requests, mean_decode_kv_tokens) where
        mean_kv is a RAW FLOAT straight off the CSV.

    A plain ``set(profiled) & set(fpm)`` only matches when the FPM mean_kv is an exact
    integer (100.0 == 100), silently dropping every fractional-mean shape (100.4).
    Bin the FPM float key with the SAME ``round()`` the marker uses so a profiled row
    at past_kv=K joins to the FPM shape whose mean_kv rounds to K. Collisions (two FPM
    floats rounding to the same int) are aggregated by mean so the wall stays a single
    representative value per integer shape.
    """
    from collections import defaultdict as _dd

    binned: dict[tuple[int, int], list[float]] = _dd(list)
    for (batch_size, mean_kv), wall_ms in fpm_wall_by_shape.items():
        binned[(int(batch_size), int(round(mean_kv)))].append(float(wall_ms))
    return {key: statistics.fmean(walls) for key, walls in binned.items()}


def run_decode_attribution(
    *,
    sqlite_path: str,
    profiled_rows: list[dict[str, Any]] | None,
    fpm_wall_by_shape: dict[tuple[int, float], float],
    aic_predict,
    discard_first_n: int = 3,
    aggregate: str = "median",
    ranks: int = 1,
) -> list[dict[str, Any]]:
    """Join profiled composition + clean FPM wall + AIC breakdown per decode shape.

    Args:
      sqlite_path: nsys .sqlite to reduce (ignored if profiled_rows is given).
      profiled_rows: pre-loaded analyze_sqlite rows (test seam); else analyze_sqlite is called.
      fpm_wall_by_shape: {(batch_size, mean_kv): wall_ms} from the CLEAN golden run.
        mean_kv is the raw FPM float; it is binned to the profiled integer past_kv via
        _bin_fpm_wall_to_profiled_key before joining (NVTX uses round(mean)).
      aic_predict: callable(batch_size, past_kv) -> (compute_ms, comm_ms, total_ms)
                   (wrap aic_fpm_gap.predict_decode_breakdown with bound backend/model/db/rc).
                   past_kv is the integer profiled key (post-binning), matching the NVTX label.
    Returns one decompose_shape() dict per shape present in ALL THREE lanes (profiled,
    binned-FPM, AIC); shapes missing from a lane (or AIC-unpredictable) are skipped.
    """
    if profiled_rows is None:
        from collector.layerwise.diagnostics.analyze_nsys_comm_overlap import analyze_sqlite

        profiled_rows, _meta = analyze_sqlite(sqlite_path)
    profiled = aggregate_profiled_by_shape(
        profiled_rows, discard_first_n=discard_first_n, aggregate=aggregate
    )
    # Bin the clean-lane float mean_kv onto the profiled integer past_kv key space so
    # the join below is over a single, consistent integer key space.
    fpm_wall_binned = _bin_fpm_wall_to_profiled_key(fpm_wall_by_shape)
    out: list[dict[str, Any]] = []
    for key in sorted(set(profiled) & set(fpm_wall_binned)):
        batch_size, past_kv = key
        aic_compute, aic_comm, aic_total = aic_predict(batch_size, past_kv)
        if aic_compute is None:
            continue
        comp = profiled[key]
        # TP>1: analyze_sqlite sums kernel durations across ALL ranks captured in the
        # single nsys report, but `wall` and the AIC layerwise prediction are single-rank.
        # For a balanced dense TP run each rank does ~1/ranks of the compute and one
        # allreduce per collective, so the per-rank value is the sum / ranks. gpu_busy
        # (interval union) is left as-is: ranks run in wall-clock lockstep, so the union
        # already collapses to ~one rank's wall (empirically gpu_busy ≈ wall at TP=8).
        row = decompose_shape(
            wall_ms=fpm_wall_binned[key],
            aic_compute_ms=aic_compute,
            aic_comm_ms=aic_comm,
            aic_other_ms=aic_total - aic_compute - aic_comm,
            gpu_compute_ms=comp["gpu_compute_ms"] / ranks,
            gpu_comm_ms=comp["gpu_comm_ms"] / ranks,
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


def _main(argv=None):
    """CLI entry for the `attribute` driver stage: build AIC's layerwise predictor,
    load the clean-lane FPM wall, reduce the profiled nsys lane, join + decompose,
    and write the per-shape decomposition CSV.

    All AIC wiring goes through aic_fpm_gap (symbols confirmed against the in-tree
    module + the working slop/h100-dense-gap-analysis/make_ctx_concurrency_chart.py
    usage): G._import_repo / G.DEFAULT_REPO_ROOT / G.build_model_and_db /
    G.predict_decode_breakdown / G.MODEL_NAME / G.ST_OK are module-level on G, while
    VLLMBackend / vllm_backend / RuntimeConfig / _load_fpm / _aggregate are in the
    api dict returned by _import_repo. _DECODE_COMPUTE_BATCH_CAL stays 0.0 (matches
    the headline 'layerwise' track), per spec.
    """
    import argparse
    from pathlib import Path
    import collector.layerwise.diagnostics.aic_fpm_gap as G

    p = argparse.ArgumentParser(description="Attributed-FPM gap decomposition")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--fpm-run", required=True, help="clean golden FPM run dir (authoritative wall)")
    p.add_argument("--system", default="h100_sxm")
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--discard-first-n", type=int, default=3)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    api = G._import_repo(Path(G.DEFAULT_REPO_ROOT))
    G.MODEL_NAME = args.model  # build_model_and_db + the decode KV-snap read this module global
    backend = api["VLLMBackend"]()
    api["vllm_backend"]._USE_LAYERWISE = True
    api["vllm_backend"]._DECODE_COMPUTE_BATCH_CAL = 0.0
    model, db, err = G.build_model_and_db(
        "layerwise", True, None, "0.20.1", args.tp,
        system=args.system, backend="vllm", comm_version="0.19.0",
        systems_root=str(Path(G.DEFAULT_REPO_ROOT) / "src/aiconfigurator/systems"),
        layerwise_csv=str(Path(G.DEFAULT_REPO_ROOT)
                          / f"src/aiconfigurator/systems/data/{args.system}/vllm/0.20.1/layerwise_perf.csv"),
        api=api,
    )
    if err:
        raise SystemExit(f"AIC model/db build failed: {err}")
    rc = api["RuntimeConfig"](vllm_max_num_batched_tokens=8192, vllm_max_num_seqs=None)

    # clean-lane FPM wall per decode shape (batch_size, mean_kv) from the golden run.
    # Per-pareto-point FPM runs land flat (fpm_metrics_phase.csv); fall back to the
    # H100 concurrency layout (fpm/qwen32/c*/) via _resolve_fpm_source if absent.
    fpm_run = Path(args.fpm_run)
    fpm_csv = fpm_run / "fpm_metrics_phase.csv"
    if not fpm_csv.exists():
        fpm_csv, _subdir = G._resolve_fpm_source(fpm_run, args.tp, Path(args.out).resolve().parent)
    # _load_fpm returns (context, decode, filtered_rows); we only need decode here.
    # decode keys are (batch_size, mean_kv) with a RAW FLOAT mean_kv; run_decode_attribution
    # bins them to the profiled integer past_kv (NVTX round(mean)) before joining.
    _ctx, decode, _filtered = api["_load_fpm"](fpm_csv, workload_segment="real")
    fpm_wall = {key: api["_aggregate"](samples, "median") for key, samples in decode.items()}

    def aic_predict(batch_size, past_kv):
        # Snap the (integer, profiled) past_kv to the nearest COLLECTED layerwise decode
        # KV before predicting -- the layerwise GEN grid is exact-lookup, so an off-grid
        # KV would raise/miss and drop the shape. This mirrors the headline 'layerwise'
        # track in aic_fpm_gap.run() (which snaps via _nearest_available_generation_kv).
        snapped = api["_nearest_available_generation_kv"](
            db.layerwise, model=G.MODEL_NAME, tp_size=args.tp,
            requested_kv=int(past_kv), max_distance=float("inf"),
        )
        if snapped is None:
            return (None, None, None)
        compute, comm, total, _src, status = G.predict_decode_breakdown(
            backend, model, db, rc, batch_size=batch_size, past_kv=snapped, api=api
        )
        return (compute, comm, total) if status == G.ST_OK else (None, None, None)

    rows = run_decode_attribution(
        sqlite_path=args.sqlite, profiled_rows=None,
        fpm_wall_by_shape=fpm_wall, aic_predict=aic_predict,
        discard_first_n=args.discard_first_n,
        ranks=args.tp,
    )
    write_decomposition_csv(rows, args.out)
    print(f"[attribute] wrote {len(rows)} shapes -> {args.out}")


if __name__ == "__main__":
    _main()
