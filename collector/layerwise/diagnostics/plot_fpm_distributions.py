#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-concurrency data-profile plots for a single FPM ground-truth run.

Each row in ``fpm_metrics_phase.csv`` is one *scheduled forward-pass step*: the
scheduler decided which requests to run together, and this file records the
resulting batch composition (how many context/prefill requests, how many decode
requests, their token counts) plus the measured step ``latency_ms``. This module
turns one such CSV into a small set of PNGs that let a human grasp -- in seconds
-- *what workload shapes the FPM run actually scheduled*, so a degenerate run
(e.g. every step was a batch-size-1 decode) is obvious at a glance rather than
buried in 4000 rows.

Diagnostic-only: it never re-runs AIC or the collector, and it is deliberately
dependency-light so it can run offline against any collected CSV. Three figures
are written per concurrency point:

  fpm_distribution_composition.png
      Batch-composition landscape: how many prefill vs decode requests were in
      each scheduled step. Overlaid per-request-count histograms + a 2-D
      ``ctx_requests`` x ``decode_requests`` hexbin/heatmap, split by phase.

  fpm_distribution_params.png
      A multi-panel "data profile": histograms of the key per-step columns
      (ctx_tokens, decode_requests, decode_kv_tokens, mean_decode_kv_tokens,
      latency_ms, queued_* fields) plus a phase-count bar. One glance answers
      "what token/KV/latency ranges did this run cover".

  fpm_distribution_scatter.png
      Relationship scatters that expose the latency drivers: latency_ms vs
      ctx_tokens (context steps), latency_ms vs decode_requests (decode batch
      size), latency_ms vs decode_kv_tokens, and a batch-composition-vs-latency
      view coloured by phase.

Usage (offline, no GPU)::

    python3 -m collector.layerwise.diagnostics.plot_fpm_distributions \\
        --fpm-csv results/validate2/review_bundle/fpm_metrics_phase.csv \\
        --out-dir /tmp/fpm_profile --concurrency 1 --title "Qwen3-32B c1"

    # or point at an FPM run dir; the phase CSV is resolved beneath it:
    python3 -m collector.layerwise.diagnostics.plot_fpm_distributions \\
        --fpm-run results/out/fpm/Qwen-Qwen3-32B/c1 --out-dir /tmp/fpm_profile
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Matplotlib needs a writable config dir (the default ~/.config may be read-only
# under sandboxing / CI). Set it before importing matplotlib. Mirrors
# tools/plot_fpm_vs_aic.py.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))

# Phase display order and colours reused across every figure.
_PHASES = ("context", "decode", "mixed")
_PHASE_COLORS = {"context": "#4c78a8", "decode": "#e45756", "mixed": "#54a24b"}

# Per-step columns worth profiling as histograms, with human labels. Only those
# actually present in the CSV are plotted, so this survives schema drift.
_PARAM_COLUMNS: tuple[tuple[str, str], ...] = (
    ("ctx_tokens", "context (new) tokens / step"),
    ("ctx_requests", "context requests / step"),
    ("ctx_kv_tokens", "context cached-KV tokens / step"),
    ("decode_requests", "decode requests / step (decode batch size)"),
    ("decode_kv_tokens", "decode KV tokens / step (sum)"),
    ("mean_decode_kv_tokens", "mean decode KV tokens / request"),
    ("latency_ms", "step latency (ms)"),
    ("queued_ctx_requests", "queued context requests"),
    ("queued_decode_requests", "queued decode requests"),
)


def resolve_phase_csv(fpm_csv: Path | None, fpm_run: Path | None) -> Path:
    """Return the ``fpm_metrics_phase.csv`` to plot from the two CLI options.

    ``--fpm-csv`` wins when given. Otherwise ``--fpm-run`` is treated as an FPM
    run directory and searched for a phase CSV: first at the top level, then
    (matching the driver's nested ``tp{T}_ep{E}_past{K}/`` layout and the
    ``review_bundle/`` copy) recursively. The first match wins.
    """

    if fpm_csv is not None:
        if not fpm_csv.is_file():
            raise FileNotFoundError(f"FPM phase CSV not found: {fpm_csv}")
        return fpm_csv
    if fpm_run is None:
        raise ValueError("one of --fpm-csv or --fpm-run is required")
    if not fpm_run.exists():
        raise FileNotFoundError(f"FPM run dir not found: {fpm_run}")
    flat = fpm_run / "fpm_metrics_phase.csv"
    if flat.is_file():
        return flat
    matches = sorted(fpm_run.rglob("fpm_metrics_phase.csv"))
    if not matches:
        raise FileNotFoundError(f"no fpm_metrics_phase.csv found under {fpm_run}")
    return matches[0]


def load_phase_frame(path: Path):
    """Load the phase CSV, coercing numeric columns and normalising ``phase``.

    Numeric columns are parsed with ``errors='coerce'`` so a stray blank never
    aborts the plot. The ``phase`` string is lower-cased and mapped so ``ctx``
    and ``gen`` (older emitters) fold into ``context`` / ``decode``.
    """

    import pandas as pd

    frame = pd.read_csv(path)
    if "phase" in frame.columns:
        phase = frame["phase"].astype(str).str.lower().str.strip()
        frame["phase"] = phase.replace({"ctx": "context", "gen": "decode"})
    for column, _label in _PARAM_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _phase_subset(frame, phase: str):
    """Rows whose normalised ``phase`` equals ``phase`` (empty frame if absent)."""

    if "phase" not in frame.columns:
        return frame.iloc[0:0]
    return frame[frame["phase"] == phase]


def _annotate_empty(ax, message: str) -> None:
    """Draw a centred placeholder so an empty panel still reads clearly."""

    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#888888")
    ax.set_xticks([])
    ax.set_yticks([])


def _int_hist_bins(values):
    """Integer-aligned bin edges for small-count columns, else count of bins.

    Request counts and small batch sizes read best as one bar per integer; large
    token counts fall back to a fixed bin count so the histogram stays legible.
    Accepts a pandas Series or a numpy array.
    """

    import numpy as np

    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return 10
    hi = float(clean.max())
    if hi <= 64:
        return np.arange(-0.5, hi + 1.5, 1.0)
    return min(50, max(10, int(np.sqrt(clean.size) * 2)))


def plot_composition(frame, out_path: Path, title: str) -> Path:
    """Batch-composition landscape: prefill vs decode request counts per step.

    Top-left overlays the ``ctx_requests`` and ``decode_requests`` distributions
    (all steps) so the split between prefill-heavy and decode-heavy batches is
    immediate. Top-right is a log-count 2-D histogram of
    (ctx_requests, decode_requests) -- the batch-composition heatmap. The bottom
    row breaks the two request-count distributions out per phase so a mixed run's
    structure is visible even when decode dominates the totals.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ctx_req = frame.get("ctx_requests")
    dec_req = frame.get("decode_requests")

    # (0,0) overlaid per-request-count histograms across all steps.
    ax = axes[0, 0]
    if ctx_req is not None and dec_req is not None and not frame.empty:
        combined = np.concatenate([ctx_req.dropna().to_numpy(dtype=float), dec_req.dropna().to_numpy(dtype=float)])
        bins = _int_hist_bins(combined) if combined.size else 10
        ax.hist(ctx_req.dropna(), bins=bins, color=_PHASE_COLORS["context"], alpha=0.6, label="ctx_requests")
        ax.hist(dec_req.dropna(), bins=bins, color=_PHASE_COLORS["decode"], alpha=0.6, label="decode_requests")
        ax.set_yscale("log")
        ax.set_xlabel("requests in scheduled step")
        ax.set_ylabel("scheduled steps (log)")
        ax.set_title("Requests per step: prefill vs decode")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.25)
    else:
        _annotate_empty(ax, "no ctx/decode request columns")

    # (0,1) 2-D composition heatmap.
    ax = axes[0, 1]
    if ctx_req is not None and dec_req is not None and not frame.empty:
        pair = frame[["ctx_requests", "decode_requests"]].dropna()
        if not pair.empty:
            hb = ax.hexbin(
                pair["ctx_requests"],
                pair["decode_requests"],
                gridsize=30,
                cmap="viridis",
                bins="log",
                mincnt=1,
            )
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("scheduled steps (log)")
            ax.set_xlabel("ctx_requests (prefill)")
            ax.set_ylabel("decode_requests")
            ax.set_title("Batch composition landscape")
        else:
            _annotate_empty(ax, "no (ctx, decode) request pairs")
    else:
        _annotate_empty(ax, "no ctx/decode request columns")

    # (1,0)/(1,1) per-phase request-count breakdowns.
    for ax, column, label in (
        (axes[1, 0], "ctx_requests", "ctx_requests / step"),
        (axes[1, 1], "decode_requests", "decode_requests / step"),
    ):
        if column not in frame.columns or frame.empty:
            _annotate_empty(ax, f"no {column} column")
            continue
        drew = False
        bins = _int_hist_bins(frame[column])
        for phase in _PHASES:
            sub = _phase_subset(frame, phase)[column].dropna()
            if sub.empty:
                continue
            ax.hist(sub, bins=bins, color=_PHASE_COLORS[phase], alpha=0.55, label=f"{phase} (n={len(sub)})")
            drew = True
        if drew:
            ax.set_yscale("log")
            ax.set_xlabel(label)
            ax.set_ylabel("scheduled steps (log)")
            ax.set_title(f"{label} by phase")
            ax.legend()
            ax.grid(True, axis="y", alpha=0.25)
        else:
            _annotate_empty(ax, f"no rows with {column}")

    fig.suptitle(f"{title}  |  batch composition", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_params(frame, out_path: Path, title: str) -> Path:
    """Multi-panel data profile: histograms of the key per-step columns.

    One histogram panel per available column in ``_PARAM_COLUMNS`` (small counts
    get integer-aligned bars, large token counts fall to a fixed bin count) plus
    a final phase-count bar so the run's phase mix is explicit. This is the
    "scan in seconds" figure.
    """

    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [(c, lbl) for c, lbl in _PARAM_COLUMNS if c in frame.columns]
    n_panels = len(available) + 1  # +1 for the phase-count bar
    ncol = 3
    nrow = math.ceil(n_panels / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow), constrained_layout=True)
    flat = axes.ravel()

    for idx, (column, label) in enumerate(available):
        ax = flat[idx]
        values = frame[column].dropna().astype(float)
        if values.empty:
            _annotate_empty(ax, f"{column}: no data")
            continue
        ax.hist(values, bins=_int_hist_bins(values), color="#4c78a8", alpha=0.85)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel(column, fontsize=8)
        ax.set_ylabel("steps", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(labelsize=7)
        # A one-line stats caption makes the panel quantitative at a glance.
        ax.text(
            0.98,
            0.95,
            f"n={len(values)}\nmin={values.min():.0f}\nmax={values.max():.0f}\nmed={values.median():.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#333333",
            bbox={"boxstyle": "round", "fc": "white", "ec": "#cccccc", "alpha": 0.7},
        )

    # Final panel: phase counts.
    ax = flat[len(available)]
    if "phase" in frame.columns and not frame.empty:
        counts = frame["phase"].value_counts()
        ordered = [p for p in _PHASES if p in counts.index] + [p for p in counts.index if p not in _PHASES]
        heights = [int(counts[p]) for p in ordered]
        colors = [_PHASE_COLORS.get(p, "#888888") for p in ordered]
        ax.bar(ordered, heights, color=colors, alpha=0.85)
        ax.set_title("scheduled steps by phase", fontsize=10)
        ax.set_ylabel("steps", fontsize=8)
        ax.tick_params(labelsize=8)
        for x, h in enumerate(heights):
            ax.text(x, h, str(h), ha="center", va="bottom", fontsize=8)
    else:
        _annotate_empty(ax, "no phase column")

    for idx in range(n_panels, nrow * ncol):
        flat[idx].axis("off")

    fig.suptitle(f"{title}  |  FPM parameter distributions", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def _scatter(ax, xs, ys, title: str, xlabel: str, ylabel: str, *, logx: bool = False, color: str = "#4c78a8"):
    """Draw one labelled scatter panel (placeholder if no data)."""

    xs = xs.dropna()
    ys = ys.reindex(xs.index).dropna()
    xs = xs.reindex(ys.index)
    if xs.empty:
        _annotate_empty(ax, "no rows")
        ax.set_title(title, fontsize=10)
        return
    ax.scatter(xs, ys, s=14, alpha=0.5, color=color, edgecolors="none")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(f"{title} (n={len(xs)})", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)


def plot_scatter(frame, out_path: Path, title: str) -> Path:
    """Relationship scatters exposing what drives step latency.

    Panels: latency vs ctx_tokens (context steps), latency vs decode_requests
    (decode batch size), latency vs decode_kv_tokens, and an all-step
    ctx_requests-vs-decode_requests view coloured by latency so batch composition
    and cost are visible together.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    have_lat = "latency_ms" in frame.columns

    ctx = _phase_subset(frame, "context")
    dec = _phase_subset(frame, "decode")

    if have_lat and "ctx_tokens" in frame.columns and not ctx.empty:
        _scatter(
            axes[0, 0],
            ctx["ctx_tokens"],
            ctx["latency_ms"],
            "Context: latency vs new tokens",
            "ctx_tokens",
            "latency_ms",
            logx=True,
            color=_PHASE_COLORS["context"],
        )
    else:
        _annotate_empty(axes[0, 0], "no context latency rows")
        axes[0, 0].set_title("Context: latency vs new tokens", fontsize=10)

    if have_lat and "decode_requests" in frame.columns and not dec.empty:
        _scatter(
            axes[0, 1],
            dec["decode_requests"],
            dec["latency_ms"],
            "Decode: latency vs batch size",
            "decode_requests",
            "latency_ms",
            color=_PHASE_COLORS["decode"],
        )
    else:
        _annotate_empty(axes[0, 1], "no decode latency rows")
        axes[0, 1].set_title("Decode: latency vs batch size", fontsize=10)

    if have_lat and "decode_kv_tokens" in frame.columns and not dec.empty:
        _scatter(
            axes[1, 0],
            dec["decode_kv_tokens"],
            dec["latency_ms"],
            "Decode: latency vs KV tokens",
            "decode_kv_tokens",
            "latency_ms",
            logx=True,
            color=_PHASE_COLORS["decode"],
        )
    else:
        _annotate_empty(axes[1, 0], "no decode KV rows")
        axes[1, 0].set_title("Decode: latency vs KV tokens", fontsize=10)

    # (1,1) composition coloured by latency, all steps.
    ax = axes[1, 1]
    cols = {"ctx_requests", "decode_requests"}
    if cols.issubset(frame.columns) and not frame.empty:
        sub = frame[["ctx_requests", "decode_requests"] + (["latency_ms"] if have_lat else [])].dropna()
        if not sub.empty:
            c = sub["latency_ms"] if have_lat else None
            sc = ax.scatter(
                sub["ctx_requests"],
                sub["decode_requests"],
                c=c,
                cmap="magma",
                s=16,
                alpha=0.7,
                edgecolors="none",
            )
            if have_lat:
                cb = fig.colorbar(sc, ax=ax)
                cb.set_label("latency_ms")
            ax.set_xlabel("ctx_requests (prefill)", fontsize=8)
            ax.set_ylabel("decode_requests", fontsize=8)
            ax.set_title("Batch composition coloured by latency", fontsize=10)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)
        else:
            _annotate_empty(ax, "no composition rows")
    else:
        _annotate_empty(ax, "no ctx/decode request columns")

    fig.suptitle(f"{title}  |  latency relationships", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def generate_all(frame, out_dir: Path, title: str) -> list[Path]:
    """Write the three profile figures and return their paths."""

    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_composition(frame, out_dir / "fpm_distribution_composition.png", title),
        plot_params(frame, out_dir / "fpm_distribution_params.png", title),
        plot_scatter(frame, out_dir / "fpm_distribution_scatter.png", title),
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, non-zero on hard failure."""

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--fpm-csv", type=Path, default=None, help="Path to fpm_metrics_phase.csv.")
    src.add_argument(
        "--fpm-run",
        type=Path,
        default=None,
        help="FPM run dir; fpm_metrics_phase.csv is resolved beneath it.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to write PNGs into.")
    parser.add_argument(
        "--concurrency",
        default=None,
        help="Concurrency (pareto point) value for the figure titles.",
    )
    parser.add_argument("--title", default=None, help="Explicit figure title prefix (overrides --concurrency).")
    args = parser.parse_args(argv)

    try:
        csv_path = resolve_phase_csv(args.fpm_csv, args.fpm_run)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[fpm-distributions] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.title is not None:
        title = args.title
    elif args.concurrency is not None:
        title = f"FPM data profile  |  concurrency={args.concurrency}"
    else:
        title = f"FPM data profile  |  {csv_path.parent.name}"

    frame = load_phase_frame(csv_path)
    if frame.empty:
        print(f"[fpm-distributions] WARNING: {csv_path} has no rows; nothing to plot", file=sys.stderr)
        return 0

    paths = generate_all(frame, args.out_dir, title)
    for path in paths:
        print(f"[fpm-distributions] wrote {path}")
    print(f"[fpm-distributions] {len(frame)} scheduled steps from {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
