#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step composition report for the comm-concurrency experiment (design.md v3 §5).

Joins the DYNAMO marker step (the ``bench_step::N...`` NVTX ordinal captured in the
nsys sqlite) to the AUTHORITATIVE FPM scheduler composition row in
``fpm_metrics_phase.csv``, and emits one row per profiled step with its phase
(context / decode / mixed / idle) and scheduled token counts. This is the
"per-step composition for pre-registered inclusion" deliverable: it lets the
analysis pre-register which high-C mixed steps are included in the arrival-skew
decomposition, with a documented (auditable) join and a shape-consistency check.

Two independent lanes, joined by chronological step sequence:

  * nsys lane  -- ``analyze_nsys_comm_overlap.analyze_sqlite`` returns per-step rows
    keyed by the marker ordinal ``step`` plus the marker-derived ``batch_size``
    (decode_batch) and ``past_kv`` (mean decode KV). We take the per-aggregate rows
    (``per_pid=False``) so there is exactly one nsys row per (step, measure_run).
  * fpm lane   -- ``fpm_metrics_phase.csv`` (written by ``summarize_fpm.py``) holds one
    row per published FPM step with the scheduled ``phase`` / ``ctx_tokens`` /
    ``ctx_requests`` / ``decode_requests`` / ``decode_kv_tokens`` /
    ``mean_decode_kv_tokens``. Rows are filtered to a single ``dp_rank`` / worker so
    the sequence is monotonic.

JOIN MODEL (documented, not magical). Both lanes are emitted in step order. The
nsys marker ordinal ``N`` counts EVERY ``execute_model`` forward (1-indexed); the
FPM phase CSV drops idle steps and starts ``counter_id`` at an arbitrary offset, so
the two ordinals are NOT directly equal. We therefore join by CHRONOLOGICAL
SEQUENCE within the captured window: the i-th in-window nsys step is paired with
the i-th in-window FPM phase row, after both are restricted to the same window and
sorted. The pairing is cross-validated by a shape check
(``marker batch_size == fpm decode_requests`` and
``round(fpm mean_decode_kv_tokens) == marker past_kv``); rows that fail the check
are still emitted but flagged ``shape_match=0`` so the pre-registration step can see
(and exclude) any misaligned steps. This module performs NO attribution and does NOT
modify analyze_nsys_comm_overlap.py or aic_fpm_attribute.py.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Import the per-step nsys reducer WITHOUT modifying it.
try:
    from collector.layerwise.diagnostics.analyze_nsys_comm_overlap import analyze_sqlite
except ModuleNotFoundError:  # pragma: no cover - direct script compatibility
    _REPO_ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(_REPO_ROOT))
    from collector.layerwise.diagnostics.analyze_nsys_comm_overlap import analyze_sqlite


OUTPUT_COLUMNS = [
    # join + identity
    "seq",              # 0-based chronological pairing index within the window
    "marker_step",      # nsys bench_step ordinal N (1-indexed execute_model count)
    "measure_run",
    "fpm_counter_id",   # FPM per-worker step counter (gapped; idle steps dropped)
    "shape_match",      # 1 if marker (bs,past) matches fpm (decode_requests, mean_kv)
    # authoritative scheduler composition (from fpm_metrics_phase.csv)
    "phase",
    "ctx_tokens",       # scheduled sum_prefill_tokens (freshly computed prefill)
    "ctx_requests",
    "ctx_kv_tokens",
    "decode_requests",
    "decode_kv_tokens",
    "mean_decode_kv_tokens",
    "queued_ctx_tokens",
    "queued_ctx_requests",
    "queued_decode_requests",
    "fpm_latency_ms",
    # marker-derived composition (from the nsys NVTX label)
    "marker_decode_batch",   # bs
    "marker_mean_kv",        # past
    # nsys kernel timing (per-step aggregate, microseconds)
    "compute_gpu_us",
    "comm_gpu_us",
    "comm_visible_us",
]


def _as_int(row: dict, key: str) -> int:
    """Parse ``row[key]`` as an int, tolerating floats / blanks (-> 0)."""
    value = row.get(key, "")
    if value in ("", None):
        return 0
    return int(float(value))


def _as_float(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def load_fpm_phase_rows(
    phase_csv: str | Path,
    *,
    dp_rank: int | None = 0,
    worker_id: str | None = None,
    include_idle: bool = False,
) -> list[dict]:
    """Load ``fpm_metrics_phase.csv`` rows in publish (step) order.

    Restricts to one ``dp_rank`` (default 0) and optionally one ``worker_id`` so the
    sequence is a single monotonic per-worker step stream (the phase CSV interleaves
    workers/ranks). ``idle`` rows are dropped unless ``include_idle``.
    """
    path = Path(phase_csv)
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if dp_rank is not None and _as_int(row, "dp_rank") != dp_rank:
                continue
            if worker_id is not None and str(row.get("worker_id", "")) != str(worker_id):
                continue
            if not include_idle and row.get("phase") == "idle":
                continue
            rows.append(row)
    # Preserve publish order; counter_id is monotonic within a worker.
    rows.sort(key=lambda r: _as_int(r, "counter_id"))
    return rows


def select_nsys_window_steps(
    nsys_rows: list[dict],
    *,
    window: tuple[int, int] | None = None,
) -> list[dict]:
    """Collapse the (already aggregate) nsys rows to one row per marker step, in order.

    ``analyze_sqlite(per_pid=False)`` returns one row per (step, measure_run); we keep
    that grain and sort by (marker_step, measure_run). When ``window=(lo, hi)`` is
    given, only marker steps with ``lo <= step <= hi`` are kept (matching the
    ``--nsys-cuda-profiler-window`` capture window semantics).
    """
    out = []
    for r in nsys_rows:
        step_n = int(r["step"])
        if window is not None and not (window[0] <= step_n <= window[1]):
            continue
        out.append(r)
    out.sort(key=lambda r: (int(r["step"]), int(r["measure_run"])))
    return out


def _shape_matches(marker_bs: int, marker_past: int, fpm_row: dict) -> bool:
    """True if the marker (decode_batch, mean_kv) matches the FPM scheduler row.

    decode_batch <-> decode_requests (exact); mean_kv <-> round(mean_decode_kv_tokens)
    (the marker rounds the mean KV to an int, so compare on the rounded FPM mean).
    """
    fpm_decode = _as_int(fpm_row, "decode_requests")
    fpm_mean_kv = round(_as_float(fpm_row, "mean_decode_kv_tokens"))
    return marker_bs == fpm_decode and marker_past == fpm_mean_kv


def build_composition_rows(nsys_steps: list[dict], fpm_rows: list[dict]) -> list[dict]:
    """Pair in-window nsys marker steps with FPM phase rows by chronological sequence.

    The pairing is positional (i-th nsys step <-> i-th FPM row); each pair carries a
    ``shape_match`` flag from the (decode_batch, mean_kv) cross-check. Extra rows on
    either side (length mismatch) are emitted with the missing side blank and
    ``shape_match=0`` so the count discrepancy is visible to the pre-registration step.
    """
    rows: list[dict] = []
    n = max(len(nsys_steps), len(fpm_rows))
    for i in range(n):
        nsys = nsys_steps[i] if i < len(nsys_steps) else None
        fpm = fpm_rows[i] if i < len(fpm_rows) else None

        marker_bs = int(nsys["batch_size"]) if nsys is not None else 0
        marker_past = int(nsys["past_kv"]) if nsys is not None else 0
        match = 1 if (nsys is not None and fpm is not None
                      and _shape_matches(marker_bs, marker_past, fpm)) else 0

        rows.append({
            "seq": i,
            "marker_step": int(nsys["step"]) if nsys is not None else "",
            "measure_run": int(nsys["measure_run"]) if nsys is not None else "",
            "fpm_counter_id": _as_int(fpm, "counter_id") if fpm is not None else "",
            "shape_match": match,
            "phase": fpm.get("phase", "") if fpm is not None else "",
            "ctx_tokens": _as_int(fpm, "ctx_tokens") if fpm is not None else "",
            "ctx_requests": _as_int(fpm, "ctx_requests") if fpm is not None else "",
            "ctx_kv_tokens": _as_int(fpm, "ctx_kv_tokens") if fpm is not None else "",
            "decode_requests": _as_int(fpm, "decode_requests") if fpm is not None else "",
            "decode_kv_tokens": _as_int(fpm, "decode_kv_tokens") if fpm is not None else "",
            "mean_decode_kv_tokens": (f'{_as_float(fpm, "mean_decode_kv_tokens"):.3f}'
                                      if fpm is not None else ""),
            "queued_ctx_tokens": _as_int(fpm, "queued_ctx_tokens") if fpm is not None else "",
            "queued_ctx_requests": _as_int(fpm, "queued_ctx_requests") if fpm is not None else "",
            "queued_decode_requests": _as_int(fpm, "queued_decode_requests") if fpm is not None else "",
            "fpm_latency_ms": fpm.get("latency_ms", "") if fpm is not None else "",
            "marker_decode_batch": marker_bs if nsys is not None else "",
            "marker_mean_kv": marker_past if nsys is not None else "",
            "compute_gpu_us": nsys.get("compute_gpu_us", "") if nsys is not None else "",
            "comm_gpu_us": nsys.get("comm_gpu_us", "") if nsys is not None else "",
            "comm_visible_us": nsys.get("comm_visible_us", "") if nsys is not None else "",
        })
    return rows


def write_composition_csv(rows: list[dict], out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    """Return a small dict summary (phase counts, match rate) for stderr/logging."""
    phase_counts: dict[str, int] = {}
    matched = 0
    paired = 0
    for r in rows:
        phase = r.get("phase") or "(no-fpm)"
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        if r.get("marker_step") != "" and r.get("fpm_counter_id") != "":
            paired += 1
            if r.get("shape_match") == 1:
                matched += 1
    return {
        "rows": len(rows),
        "paired": paired,
        "shape_matched": matched,
        "shape_match_rate": (matched / paired) if paired else 0.0,
        "phase_counts": phase_counts,
    }


def _parse_window(spec: str | None) -> tuple[int, int] | None:
    """Parse a single ``lo-hi`` window (the comm-concurrency capture uses one window).

    Accepts the same ``lo-hi`` form as ``--nsys-cuda-profiler-window`` (first window
    only if a comma list is given). Returns None for an empty/unset spec.
    """
    if not spec:
        return None
    first = spec.split(",")[0].strip()
    if "-" not in first:
        raise ValueError(f"invalid window spec {spec!r}; expected 'lo-hi'")
    lo_s, hi_s = first.split("-", 1)
    return int(lo_s), int(hi_s)


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sqlite", required=True, help="nsys sqlite export with bench_step NVTX ranges.")
    p.add_argument("--fpm-phase-csv", required=True, help="fpm_metrics_phase.csv from the same run.")
    p.add_argument("--out", required=True, help="Output per-step composition CSV path.")
    p.add_argument(
        "--window",
        default=None,
        help="Optional 'lo-hi' marker-step window (match --nsys-cuda-profiler-window). "
        "Restricts the nsys lane to that step range before pairing.",
    )
    p.add_argument("--dp-rank", type=int, default=0, help="FPM dp_rank to keep (default 0).")
    p.add_argument("--worker-id", default=None, help="Optional FPM worker_id to keep.")
    p.add_argument("--include-idle", action="store_true", help="Keep FPM idle steps.")
    args = p.parse_args(argv)

    window = _parse_window(args.window)
    nsys_rows, meta = analyze_sqlite(args.sqlite, per_pid=False)
    nsys_steps = select_nsys_window_steps(nsys_rows, window=window)
    fpm_rows = load_fpm_phase_rows(
        args.fpm_phase_csv,
        dp_rank=args.dp_rank,
        worker_id=args.worker_id,
        include_idle=args.include_idle,
    )
    rows = build_composition_rows(nsys_steps, fpm_rows)
    write_composition_csv(rows, args.out)

    summary = summarize(rows)
    if len(nsys_steps) != len(fpm_rows):
        print(
            f"[fpm-step-composition] WARNING: lane length mismatch "
            f"(nsys in-window={len(nsys_steps)} vs fpm rows={len(fpm_rows)}); "
            f"extra rows emitted with a blank side and shape_match=0.",
            file=sys.stderr,
        )
    print(
        f"[fpm-step-composition] wrote {summary['rows']} rows to {args.out}; "
        f"paired={summary['paired']} shape_matched={summary['shape_matched']} "
        f"({summary['shape_match_rate'] * 100:.1f}%); phases={summary['phase_counts']}; "
        f"nsys_meta_groups={meta.get('groups')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
