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
    per_pid: bool = False,
) -> dict[tuple[int, ...], dict[str, float]]:
    """Aggregate analyze_sqlite per-step rows into per-shape composition, in ms.
    Discards sync-drained boundary steps first.

    per_pid=False (default): key on (batch_size, past_kv). Rows are the merged
    cross-rank analyze_sqlite output (compute/comm are SUMS over all ranks, busy is
    the cross-rank interval union); the median is over repeated STEPS of the shape.

    per_pid=True: rows carry a 'pid' (rank id) -- key on (batch_size, past_kv, pid)
    so each rank is its OWN bucket. The median is over repeated steps of the SAME
    (rank, shape) and NEVER across ranks, preserving per-rank variance. Values are
    already per-rank (analyze_sqlite per_pid=True does not sum across ranks), so the
    caller must NOT divide compute/comm by ranks; busy is that rank's own union."""
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    rows = _filter_boundary_discards(rows, discard_first_n=discard_first_n)
    buckets: dict[tuple[int, ...], dict[str, list[float]]] = defaultdict(
        lambda: {"compute": [], "comm": [], "busy": []}
    )
    for row in rows:
        if per_pid:
            key: tuple[int, ...] = (int(row["batch_size"]), int(row["past_kv"]), int(row["pid"]))
        else:
            key = (int(row["batch_size"]), int(row["past_kv"]))
        buckets[key]["compute"].append(float(row["compute_gpu_us"]) / 1000.0)
        buckets[key]["comm"].append(float(row["comm_gpu_us"]) / 1000.0)
        buckets[key]["busy"].append(float(row["total_union_us"]) / 1000.0)
    reduce = statistics.median if aggregate == "median" else statistics.fmean
    out: dict[tuple[int, ...], dict[str, float]] = {}
    for key, series in buckets.items():
        out[key] = {
            "gpu_compute_ms": float(reduce(series["compute"])),
            "gpu_comm_ms": float(reduce(series["comm"])),
            "gpu_busy_ms": float(reduce(series["busy"])),
        }
    return out


def _sum_per_pid_rows_to_merged(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-PID analyze_sqlite rows back to the merged (per_pid=False) shape.

    The aggregate (cross-rank) row must be computed from the SAME reduction the
    per_pid=False path uses: compute/comm SUMMED across ranks per step, busy the
    cross-rank interval union. We do not have the raw intervals here, so busy is the
    MAX rank union per step (ranks run in lockstep, so the cross-rank union ≈ the
    longest single-rank union -- the empirical gpu_busy ≈ wall identity). This keeps
    the aggregate identical to today for the genuine merged-lane input (one row per
    step, pid empty), where each step is its own group of size one and sum/max are
    no-ops."""
    by_step: dict[tuple[int, int, int, int], dict[str, float]] = {}
    for row in rows:
        gk = (int(row["step"]), int(row["batch_size"]), int(row["past_kv"]), int(row["measure_run"]))
        acc = by_step.setdefault(
            gk, {"compute_gpu_us": 0.0, "comm_gpu_us": 0.0, "total_union_us": 0.0}
        )
        acc["compute_gpu_us"] += float(row["compute_gpu_us"])
        acc["comm_gpu_us"] += float(row["comm_gpu_us"])
        acc["total_union_us"] = max(acc["total_union_us"], float(row["total_union_us"]))
    merged: list[dict[str, Any]] = []
    for (step, bs, past, run), acc in by_step.items():
        merged.append({
            "step": step, "batch_size": bs, "past_kv": past, "measure_run": run,
            "compute_gpu_us": acc["compute_gpu_us"],
            "comm_gpu_us": acc["comm_gpu_us"],
            "total_union_us": acc["total_union_us"],
        })
    return merged


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
    per_pid: bool = False,
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

        profiled_rows, _meta = analyze_sqlite(sqlite_path, per_pid=per_pid)
    # Bin the clean-lane float mean_kv onto the profiled integer past_kv key space so
    # the join below is over a single, consistent integer key space.
    fpm_wall_binned = _bin_fpm_wall_to_profiled_key(fpm_wall_by_shape)
    out: list[dict[str, Any]] = []

    # --- aggregate (cross-rank) rows: ALWAYS computed from the per_pid=False reduction
    # so they are IDENTICAL to today regardless of per_pid. When per_pid=True the input
    # rows are per-rank, so first sum them back to the merged (cross-rank) shape; when
    # per_pid=False the merged rows pass through _sum_per_pid_rows_to_merged as a no-op.
    merged_rows = _sum_per_pid_rows_to_merged(profiled_rows) if per_pid else profiled_rows
    profiled = aggregate_profiled_by_shape(
        merged_rows, discard_first_n=discard_first_n, aggregate=aggregate
    )
    for key in sorted(set(profiled) & set(fpm_wall_binned)):
        batch_size, past_kv = key
        aic_compute, aic_comm, aic_total = aic_predict(batch_size, past_kv)
        if aic_compute is None:
            continue
        comp = profiled[key]
        # TP>1: analyze_sqlite (merged) sums kernel durations across ALL ranks captured
        # in the single nsys report, but `wall` and the AIC layerwise prediction are
        # single-rank. For a balanced dense TP run each rank does ~1/ranks of the compute
        # and one allreduce per collective, so the per-rank value is the sum / ranks.
        # gpu_busy (interval union) is left as-is: ranks run in wall-clock lockstep, so
        # the union already collapses to ~one rank's wall (empirically gpu_busy ≈ wall
        # at TP=8).
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
        row["pid"] = ""
        out.append(row)

    # --- per-rank rows (per_pid=True only): keyed by (batch_size, past_kv, pid),
    # median over repeated steps of the SAME (rank, shape) -- NEVER across ranks, so
    # cross-rank variance is preserved. Values are ALREADY per-rank, so they are NOT
    # divided by ranks; gpu_busy is that rank's own interval union. These rows are
    # emitted ALONGSIDE the aggregate rows above and never feed the /ranks aggregate.
    if per_pid:
        per_rank = aggregate_profiled_by_shape(
            profiled_rows, discard_first_n=discard_first_n, aggregate=aggregate, per_pid=True
        )
        for key in sorted(per_rank):
            batch_size, past_kv, pid = key
            if (batch_size, past_kv) not in fpm_wall_binned:
                continue
            aic_compute, aic_comm, aic_total = aic_predict(batch_size, past_kv)
            if aic_compute is None:
                continue
            comp = per_rank[key]
            row = decompose_shape(
                wall_ms=fpm_wall_binned[(batch_size, past_kv)],
                aic_compute_ms=aic_compute,
                aic_comm_ms=aic_comm,
                aic_other_ms=aic_total - aic_compute - aic_comm,
                gpu_compute_ms=comp["gpu_compute_ms"],  # per-rank, NOT /ranks
                gpu_comm_ms=comp["gpu_comm_ms"],        # per-rank, NOT /ranks
                gpu_busy_ms=comp["gpu_busy_ms"],        # this rank's own union
            )
            row["phase"] = "decode"
            row["batch_size"] = batch_size
            row["past_kv"] = past_kv
            row["pid"] = pid
            out.append(row)
    return out


def aggregate_profiled_context(
    rows: list[dict[str, Any]],
    *,
    discard_first_n: int = 3,
    aggregate: str = "median",
    ranks: int = 1,
) -> dict[str, float]:
    """Aggregate the PURE-CONTEXT profiled steps into one composition, in ms.

    Context chunks are UNIFORM 2048-token prefill steps (FPM_MAX_NUM_BATCHED_TOKENS),
    so they aggregate cleanly without per-shape keys (unlike decode, which varies by
    (batch, past_kv)). PURE-CONTEXT steps are those the dynamo marker labels bs0 --
    a step with NO decode requests (decode_batch == 0 -> ``batch_size == 0``); any
    bs>0 row is a (possibly mixed) step with decode work and is excluded.

    us->ms; compute/comm are divided by ranks (per-rank, same TP rationale as decode --
    analyze_sqlite sums kernel durations across all captured ranks while wall + AIC are
    single-rank); gpu_busy (interval union) is left as-is (ranks run in lockstep)."""
    from collector.layerwise.vllm.nsys import _filter_boundary_discards

    ctx_rows = [row for row in rows if int(row["batch_size"]) == 0]
    ctx_rows = _filter_boundary_discards(ctx_rows, discard_first_n=discard_first_n)
    compute = [float(row["compute_gpu_us"]) / 1000.0 for row in ctx_rows]
    comm = [float(row["comm_gpu_us"]) / 1000.0 for row in ctx_rows]
    busy = [float(row["total_union_us"]) / 1000.0 for row in ctx_rows]
    reduce = statistics.median if aggregate == "median" else statistics.fmean
    return {
        "gpu_compute_ms": float(reduce(compute)) / ranks,
        "gpu_comm_ms": float(reduce(comm)) / ranks,
        "gpu_busy_ms": float(reduce(busy)),
    }


def run_context_attribution(
    *,
    sqlite_path: str,
    profiled_rows: list[dict[str, Any]] | None,
    fpm_ctx_wall_ms: float,
    aic_ctx_predict,
    discard_first_n: int = 3,
    aggregate: str = "median",
    ranks: int = 1,
) -> dict[str, Any]:
    """Join the profiled pure-context composition + clean FPM context wall + AIC ctx
    breakdown into ONE aggregate decomposition (the 2048-token chunk shape is uniform,
    so there is a single context decomposition, not a per-shape list like decode).

    Args:
      sqlite_path: nsys .sqlite to reduce (ignored if profiled_rows is given).
      profiled_rows: pre-loaded analyze_sqlite rows (test seam); else analyze_sqlite is called.
      fpm_ctx_wall_ms: the CLEAN FPM context-phase wall (median of the FPM 'context'
                       phase latency_ms for the 2048-token single-request chunk).
      aic_ctx_predict: callable() -> (compute_ms, comm_ms, total_ms) (caller wraps
                       aic_fpm_gap.predict_context_breakdown for the 2048-token ctx chunk).
    Returns one decompose_shape() dict tagged phase='context'.
    """
    if profiled_rows is None:
        from collector.layerwise.diagnostics.analyze_nsys_comm_overlap import analyze_sqlite

        profiled_rows, _meta = analyze_sqlite(sqlite_path)
    comp = aggregate_profiled_context(
        profiled_rows, discard_first_n=discard_first_n, aggregate=aggregate, ranks=ranks
    )
    aic_compute, aic_comm, aic_total = aic_ctx_predict()
    row = decompose_shape(
        wall_ms=fpm_ctx_wall_ms,
        aic_compute_ms=aic_compute,
        aic_comm_ms=aic_comm,
        aic_other_ms=aic_total - aic_compute - aic_comm,
        gpu_compute_ms=comp["gpu_compute_ms"],
        gpu_comm_ms=comp["gpu_comm_ms"],
        gpu_busy_ms=comp["gpu_busy_ms"],
    )
    row["phase"] = "context"
    return row


def write_decomposition_csv(rows: list[dict[str, Any]], out_path: str) -> None:
    """Write decomposition rows to CSV (stable column order).

    Raises ``ValueError`` on empty input so an empty join cannot pass silently
    (the ``.done`` gate / shell ``|| warn`` only fires on a real failure signal).
    """
    import csv

    if not rows:
        raise ValueError(
            f"write_decomposition_csv: no decomposition rows to write to {out_path} "
            "(empty join -- profiled/FPM/AIC lanes had no shape in common)."
        )
    cols = [
        "phase", "batch_size", "past_kv", "pid", "wall_ms", "aic_total_ms",
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
    import sys
    from pathlib import Path
    import collector.layerwise.diagnostics.aic_fpm_gap as G

    p = argparse.ArgumentParser(description="Attributed-FPM gap decomposition")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--fpm-run", required=True, help="clean golden FPM run dir (authoritative wall)")
    p.add_argument("--system", default="h100_sxm")
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--discard-first-n", type=int, default=3)
    p.add_argument(
        "--per-pid",
        action="store_true",
        help="ALSO emit accurate per-rank rows (pid column) alongside the cross-rank "
             "aggregate. Per-rank compute/comm are NOT divided by ranks and per-rank "
             "variance is preserved; the aggregate rows are unchanged.",
    )
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
    # Build the predictor's RuntimeConfig from the run's effective config rather
    # than a hardcoded literal (F3). vllm_max_num_seqs stays None: GEN rows carry
    # an empty max_num_seqs, so None selects the primary index on the layerwise track.
    run_rc = G._read_runtime_config(Path(args.fpm_run))
    rc = api["RuntimeConfig"](
        vllm_max_num_batched_tokens=run_rc["vllm_max_num_batched_tokens"],
        vllm_max_num_seqs=None,
    )

    # clean-lane FPM wall per decode shape (batch_size, mean_kv) from the golden run.
    # Per-pareto-point FPM runs land flat (fpm_metrics_phase.csv); fall back to the
    # H100 concurrency layout (fpm/qwen32/c*/) via _resolve_fpm_source if absent.
    fpm_run = Path(args.fpm_run)
    fpm_csv = fpm_run / "fpm_metrics_phase.csv"
    if not fpm_csv.exists():
        fpm_csv, _subdir = G._resolve_fpm_source(fpm_run, args.tp, Path(args.out).resolve().parent)
    # _load_fpm returns (context, decode, filtered_rows). decode keys are
    # (batch_size, mean_kv) with a RAW FLOAT mean_kv; run_decode_attribution bins them to
    # the profiled integer past_kv (NVTX round(mean)) before joining. context keys are
    # (ctx_requests, ctx_tokens, ctx_prefix) and feed the context-phase attribution below.
    context, decode, _filtered = api["_load_fpm"](fpm_csv, workload_segment="real")
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

    # Reduce the nsys lane once; both decode and context phases share the profiled rows
    # (decode keys on per-(batch,kv) rows, context aggregates the bs0 pure-prefill rows).
    from collector.layerwise.diagnostics.analyze_nsys_comm_overlap import analyze_sqlite
    profiled_rows, _meta = analyze_sqlite(args.sqlite, per_pid=args.per_pid)

    rows = run_decode_attribution(
        sqlite_path=args.sqlite, profiled_rows=profiled_rows,
        fpm_wall_by_shape=fpm_wall, aic_predict=aic_predict,
        discard_first_n=args.discard_first_n,
        ranks=args.tp, per_pid=args.per_pid,
    )

    # --- context phase (the high-C context puzzle) ---
    # Context chunks are UNIFORM 2048-token single-request prefill steps. Pick the
    # DOMINANT FPM context key (most-sampled) so the AIC ctx_prefix matches the real
    # chunk, median its latency for the clean wall, and decompose the single aggregate
    # context shape. Skip gracefully (warn, keep decode rows) if either side is missing.
    try:
        ctx_2048 = {k: v for k, v in context.items() if int(k[1]) == 2048}
        ctx_pool = ctx_2048 or context
        if not ctx_pool:
            raise ValueError("no FPM context-phase shapes found")
        dom_key = max(ctx_pool.items(), key=lambda kv: len(kv[1]))[0]
        ctx_requests, ctx_tokens, ctx_prefix = dom_key
        fpm_ctx_wall_ms = api["_aggregate"](ctx_pool[dom_key], "median")

        def aic_ctx_predict():
            compute, comm, total, _src, status = G.predict_context_breakdown(
                backend, model, db, rc,
                ctx_tokens=int(ctx_tokens), ctx_prefix_tokens=int(ctx_prefix), api=api,
            )
            if status != G.ST_OK or compute is None:
                raise ValueError(f"AIC context predict unavailable (status={status})")
            return (compute, comm, total)

        # Context is a single CROSS-RANK aggregate (no per-rank rows). When --per-pid
        # is set the profiled rows are per-rank, so sum them back to the merged shape
        # first; aggregate_profiled_context then divides by ranks exactly as today.
        ctx_profiled = _sum_per_pid_rows_to_merged(profiled_rows) if args.per_pid else profiled_rows
        ctx_row = run_context_attribution(
            sqlite_path=args.sqlite, profiled_rows=ctx_profiled,
            fpm_ctx_wall_ms=fpm_ctx_wall_ms, aic_ctx_predict=aic_ctx_predict,
            discard_first_n=args.discard_first_n, ranks=args.tp,
        )
        ctx_row["batch_size"] = ""
        ctx_row["past_kv"] = int(ctx_tokens)
        ctx_row["pid"] = ""
        rows.append(ctx_row)
    except Exception as exc:  # noqa: BLE001 - never crash _main; decode rows still written
        print(f"[attribute] context phase skipped: {exc}", file=sys.stderr)

    write_decomposition_csv(rows, args.out)
    print(f"[attribute] wrote {len(rows)} shapes -> {args.out}")


if __name__ == "__main__":
    _main()
