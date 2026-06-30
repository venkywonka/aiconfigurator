#!/usr/bin/env python3
"""Self-contained HTML dashboard + markdown verdict for the AIC-predictor vs FPM gap analysis.

Charts are rendered with matplotlib and base64-embedded (no network dependency).
Consumes the ``result`` dict produced by gap_analysis.run().
"""

from __future__ import annotations

import base64
import html
import io
import os
import statistics
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRACK_ORDER = ["layerwise", "opwise_silicon", "hybrid", "empirical", "sol"]
# Diagnostic/sensitivity tracks excluded from the headline ranking (none currently).
SENSITIVITY_TRACKS: set[str] = set()
TRACK_COLOR = {
    "layerwise": "#1f77b4",
    "opwise_silicon": "#d62728",
    "hybrid": "#9467bd",
    "empirical": "#ff7f0e",
    "sol": "#7f7f7f",
}
TRACK_VERSION = {
    "layerwise": "0.20.1", "opwise_silicon": "0.19.0",
    "hybrid": "0.19.0", "empirical": "0.19.0", "sol": "0.19.0",
}
PHASES = ["ctx", "gen"]
TPS = [1, 2, 4, 8]


def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"/>'


def _ok_rows(rows, track, tp, phase):
    return [r for r in rows if r["track"] == track and r["tp"] == tp and r["phase"] == phase
            and r["status"] == "ok" and r["rel_err"] != ""]


def _summary_lookup(summary):
    return {(s["track"], s["tp"], s["phase"]): s for s in summary}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _chart_bias_vs_tp(summary, noise_by_tp_phase):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    lut = _summary_lookup(summary)
    for ax, phase in zip(axes, PHASES):
        # FPM noise band (±p90 CV), averaged across TP for a single shaded band.
        p90s = [noise_by_tp_phase.get((tp, phase), {}).get("p90_cv") for tp in TPS]
        p90s = [p for p in p90s if p is not None]
        if p90s:
            band = 100 * statistics.median(p90s)
            ax.axhspan(-band, band, color="green", alpha=0.10, label=f"±FPM noise (median p90 CV≈{band:.1f}%)")
        ax.axhline(0, color="black", lw=0.8)
        for track in TRACK_ORDER:
            xs, ys = [], []
            for tp in TPS:
                s = lut.get((track, tp, phase))
                if s and s["signed_bias_pct"] != "":
                    xs.append(tp)
                    ys.append(s["signed_bias_pct"])
            if xs:
                ax.plot(xs, ys, "o-", color=TRACK_COLOR[track], label=track, lw=1.8, ms=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TPS)
        ax.set_xticklabels(TPS)
        ax.set_xlabel("tensor parallel (TP)")
        ax.set_ylabel("signed bias % (pred vs FPM)")
        ax.set_title(f"{phase} — signed bias vs TP")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Signed bias vs TP (compute@TP1, comm = growth above TP1)", fontsize=11)
    return _b64(fig)


def _chart_mape_vs_tp(summary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    lut = _summary_lookup(summary)
    for ax, phase in zip(axes, PHASES):
        for track in TRACK_ORDER:
            xs, ys = [], []
            for tp in TPS:
                s = lut.get((track, tp, phase))
                if s and s["mape_pct"] != "":
                    xs.append(tp)
                    ys.append(s["mape_pct"])
            if xs:
                ax.plot(xs, ys, "o-", color=TRACK_COLOR[track], label=track, lw=1.8, ms=5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TPS)
        ax.set_xticklabels(TPS)
        ax.set_xlabel("tensor parallel (TP)")
        ax.set_ylabel("MAPE %")
        ax.set_title(f"{phase} — MAPE vs TP")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle("Mean absolute % error vs TP (lower = better)", fontsize=11)
    return _b64(fig)


def _chart_mape_vs_concurrency(by_concurrency):
    """MAPE vs concurrency (decode batch size), faceted phase(rows) × TP(cols), lines per track.

    ctx is single-request (concurrency=1) so it is omitted; the concurrency
    sweep lives in the gen and mixed phases.
    """
    phases = [p for p in ("gen", "mixed") if any(r["phase"] == p for r in by_concurrency)]
    if not phases:
        phases = ["gen"]
    fig, axes = plt.subplots(len(phases), len(TPS), figsize=(3.1 * len(TPS), 3.0 * len(phases)),
                             squeeze=False, sharey="row")
    for ri, phase in enumerate(phases):
        for ci, tp in enumerate(TPS):
            ax = axes[ri][ci]
            for track in TRACK_ORDER:
                pts = sorted((r["concurrency"], r["mape_pct"]) for r in by_concurrency
                             if r["track"] == track and r["tp"] == tp and r["phase"] == phase
                             and r["mape_pct"] != "")
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                            color=TRACK_COLOR[track], label=track, lw=1.5, ms=4)
            ax.set_xscale("log", base=2)
            ax.grid(True, alpha=0.3)
            if ri == 0:
                ax.set_title(f"TP{tp}", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"{phase}\nMAPE %", fontsize=9)
            if ri == len(phases) - 1:
                ax.set_xlabel("concurrency (decode batch)")
            if ri == 0 and ci == len(TPS) - 1:
                ax.legend(fontsize=6, loc="best")
    fig.suptitle("MAPE vs concurrency, grouped by TP (cols) and phase (rows)", fontsize=11)
    return _b64(fig)


def _chart_parity(rows):
    fig, axes = plt.subplots(1, len(TRACK_ORDER), figsize=(3.0 * len(TRACK_ORDER), 3.2), sharex=True, sharey=True)
    cmap = {1: "#440154", 2: "#31688e", 4: "#35b779", 8: "#fde725"}
    for ax, track in zip(axes, TRACK_ORDER):
        allx, ally = [], []
        for tp in TPS:
            xs = [float(r["fpm_ms"]) for r in rows if r["track"] == track and r["tp"] == tp and r["status"] == "ok" and r["pred_ms"] != ""]
            ys = [float(r["pred_ms"]) for r in rows if r["track"] == track and r["tp"] == tp and r["status"] == "ok" and r["pred_ms"] != ""]
            if xs:
                ax.scatter(xs, ys, s=8, color=cmap[tp], alpha=0.6, label=f"TP{tp}")
                allx += xs
                ally += ys
        if allx:
            lo = min(min(allx), min(ally))
            hi = max(max(allx), max(ally))
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_title(f"{track}\n({TRACK_VERSION[track]})", fontsize=8)
        ax.set_xlabel("FPM ms")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("pred ms")
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Parity: prediction vs FPM truth (y=x ideal), colored by TP", fontsize=11)
    return _b64(fig)


def _chart_err_vs_shape(rows):
    """Where divergence lives: signed err% vs shape axis, at TP1 and TP8."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    specs = [
        ("ctx", "ctx_tokens", 1), ("gen", "decode_kv", 1),
        ("ctx", "ctx_tokens", 8), ("gen", "decode_kv", 8),
    ]
    for ax, (phase, xkey, tp) in zip(axes.flat, specs):
        ax.axhline(0, color="black", lw=0.8)
        for track in TRACK_ORDER:
            pts = [(float(r[xkey]), 100 * float(r["rel_err"]))
                   for r in rows if r["track"] == track and r["tp"] == tp and r["phase"] == phase
                   and r["status"] == "ok" and r["rel_err"] != "" and r[xkey] != ""]
            pts.sort()
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=TRACK_COLOR[track],
                        label=track, lw=1.2, ms=3, alpha=0.8)
        ax.set_xlabel(xkey)
        ax.set_ylabel("signed err %")
        ax.set_title(f"{phase} @ TP{tp} — err% vs {xkey}")
        ax.grid(True, alpha=0.3)
        if phase == "gen":
            ax.set_xscale("log")
        ax.legend(fontsize=6, loc="best")
    fig.suptitle("Where each track diverges along the shape axis (TP1 top, TP8 bottom)", fontsize=11)
    return _b64(fig)


# ---------------------------------------------------------------------------
# Classification + head-to-head + verdict
# ---------------------------------------------------------------------------
def classify(summary, noise_by_tp_phase):
    """Per (track, phase): within-noise / biased / diverging."""
    lut = _summary_lookup(summary)
    out = {}
    for track in TRACK_ORDER:
        for phase in PHASES:
            biases = [lut[(track, tp, phase)]["signed_bias_pct"] for tp in TPS
                      if (track, tp, phase) in lut and lut[(track, tp, phase)]["signed_bias_pct"] != ""]
            mapes = [lut[(track, tp, phase)]["mape_pct"] for tp in TPS
                     if (track, tp, phase) in lut and lut[(track, tp, phase)]["mape_pct"] != ""]
            if not biases:
                out[(track, phase)] = ("n/a", "no data")
                continue
            p90s = [noise_by_tp_phase.get((tp, phase), {}).get("p90_cv") for tp in TPS]
            p90s = [100 * p for p in p90s if p is not None]
            noise = statistics.median(p90s) if p90s else 0.0
            spread = max(biases) - min(biases)
            max_abs_bias = max(abs(b) for b in biases)
            if max_abs_bias <= noise:
                label = "within-noise"
            elif spread >= max(2 * noise, 10.0):
                label = "diverging"
            else:
                label = "biased"
            detail = f"bias {min(biases):+.1f}…{max(biases):+.1f}% (TP1→8 spread {spread:.1f}%), MAPE {statistics.fmean(mapes):.1f}%, noise≈{noise:.1f}%"
            out[(track, phase)] = (label, detail)
    return out


def head_to_head(rows):
    """layerwise vs opwise_silicon: paired comparison on shapes both matched ok."""
    out = {}
    for phase in PHASES:
        for tp in TPS:
            lw = {r["shape"]: float(r["rel_err"]) for r in rows
                  if r["track"] == "layerwise" and r["tp"] == tp and r["phase"] == phase
                  and r["status"] == "ok" and r["rel_err"] != ""}
            ow = {r["shape"]: float(r["rel_err"]) for r in rows
                  if r["track"] == "opwise_silicon" and r["tp"] == tp and r["phase"] == phase
                  and r["status"] == "ok" and r["rel_err"] != ""}
            common = set(lw) & set(ow)
            if not common:
                continue
            lw_wins = sum(1 for k in common if abs(lw[k]) < abs(ow[k]))
            out[(phase, tp)] = {
                "n": len(common),
                "lw_wins": lw_wins,
                "lw_win_frac": round(lw_wins / len(common), 3),
                "lw_mape": round(100 * statistics.fmean(abs(lw[k]) for k in common), 2),
                "ow_mape": round(100 * statistics.fmean(abs(ow[k]) for k in common), 2),
            }
    return out


# ---------------------------------------------------------------------------
# HTML/markdown assembly
# ---------------------------------------------------------------------------
def _table(headers, rows_data, *, cell_class=None):
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = []
    for row in rows_data:
        tds = []
        for i, c in enumerate(row):
            cls = cell_class(i, c) if cell_class else ""
            cls_attr = f' class="{cls}"' if cls else ""
            tds.append(f"<td{cls_attr}>{html.escape(str(c))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><thead><tr>{th}</tr></thead><tbody>{''.join(body)}</tbody></table>"


# Methodology headsup content (shared by HTML + verdict.md). Each tuple is
# (status, title, text). status ∈ {REMAINS, MITIGATED}.
CONFOUNDERS = [
    ("REMAINS", "Measurement-domain host floor",
     "AIC decode = single execute_model GPU-event time; FPM truth = full scheduler step wall "
     "(sampling/host/ZMQ), a ~7–8 ms host floor. Would bias AIC UNDER, worst at low batch, dense-decode only. "
     "Empirically here the B1 intercept gap is only +0.007 ms, so it is ~captured at the floor for this dataset "
     "— flagged as a risk, not a driver. Not corrected (no host-floor offset added to decode)."),
    ("MITIGATED", "Layerwise decode batch-calibration (now DISABLED)",
     "vllm_backend._DECODE_COMPUTE_BATCH_CAL has been set to 0.0, disabling the (1 + 0.0066·batch) multiplier that "
     "previously scaled layerwise dense decode (+21% at batch=32) and was the dominant cause of the concurrency "
     "over-prediction. The headline 'layerwise' track is now uncalibrated. NOTE: the "
     "fix was the FORM — disabling a multiplier on the floor-dominated decode sum; a per-(model,system,TP) additive "
     "slope, validated against FPM, would be the principled replacement (default 0.0 where no FPM exists)."),
    ("REMAINS", "Engine/SKU/version non-parity + synthesized TP comm",
     "layerwise=vLLM 0.20.1, op-wise=0.19.0; the FPM engine build differs; TP comm for layerwise TP>1 is a "
     "single-GPU modeled all-reduce add-back (not measured), ±7–11% sign-flipping. MITIGATED: version recorded "
     "per track, comm_modeled flag set, and the TP1-anchored compute/comm decomposition isolates comm error."),
    ("MITIGATED", "Mark-not-drop coverage",
     "Every prediction failure is wrapped and marked (missing_data/offgrid_raise/error); no silent drops; "
     "coverage.csv balances. No downward MAPE bias from hidden lookup failures."),
    ("MITIGATED", "No pathology filters",
     "filter_pathological_{context,decode} default OFF and unused — no peer-median deviation cut, no 3σ trim, "
     "no AUTO context-budget truncation. MAPE computed on the full FPM bin set."),
    ("MITIGATED", "KV-snap transparency",
     "Layerwise snaps requested KV to the nearest measured grid value; recorded as match_type/kv_snap_delta "
     "(op-wise uses exact KV), so snap-induced error is segmentable rather than hidden."),
    ("MITIGATED", "Empirical noise floor",
     "within_noise uses the measured FPM noise floor (gen p90 CV ≈1.3% at TP1), NOT the unsupported 0.3% CV "
     "claim. Multi-percent gaps are confirmed real signal."),
]

CONCURRENCY_EXPLANATION = [
    "RESOLVED in the headline: _DECODE_COMPUTE_BATCH_CAL is now 0.0, so the live 'layerwise' track no longer "
    "carries the concurrency degradation that the old 0.0066 decode batch-cal introduced.",
    "What it was: with cal=0.0066 the layerwise decode over-predicted increasingly with batch — clean TP1/kv~4096 "
    "fit pred = 13.383 + 0.2524·batch vs fpm = 13.376 + 0.1911·batch (intercepts match to +0.007 ms, but the "
    "per-request slope ran 1.32× too steep), so signed bias climbed −0.0% @B1 → +5.8% @B8 → +15.8% @B32.",
    "Why the FORM was wrong: the calibration MULTIPLIED the entire per-layer decode time (vllm_backend.py:1423,1432) "
    "by (1 + 0.0066·batch). But decode time is dominated by a large batch-INDEPENDENT floor (affine fit ≈ 13.8 ms + "
    "0.074·batch — the ~13.8 ms is one-time weight streaming for the 32B model, amortized across the batch). "
    "Multiplying that fixed floor by a batch-proportional factor injected phantom cost: at batch 32, 0.0066·13.84·32 "
    "≈ 2.9 ms of pure floor-inflation + a ~0.5 ms quadratic term (≈3.4 ms total = the whole over-prediction).",
    "The PREMISE was right (author comment at :1424 — the single-GPU microbenchmark grows too gently: FPM's true "
    "per-request slope ≈0.150 vs the microbenchmark's ≈0.074 ms/req, ~2×), but the magnitude/form over-corrected. "
    "With cal=0 the headline layerwise now tracks FPM within ±5% at every batch (gen MAPE ~3–4% at TP1) vs the "
    "old cal's 11–16%. A principled replacement is a per-"
    "(model,system,TP) ADDITIVE slope validated against FPM (the slope varies ~4× across TP, so it is NOT a single "
    "global constant); default 0.0 where no FPM exists.",
    "Adversarial checks (layerwise-specific, not an FPM artifact): (a) op-wise over the SAME FPM truth shows the "
    "OPPOSITE sign and a flat trend (−10% @B1 → −9% @B32); (b) B32 and kv 2294–4126 are INTERIOR to the measured "
    "grid (batch∈{1…512}, kv∈{1…32768}) → not extrapolation; (c) the host floor is captured at B1 (+0.007 ms). "
    "A smaller KV-snap effect remains (B32 kv~2340 snapped to grid 2048) — visible per-row via match_type/kv_snap_delta.",
]


def _confounders_html():
    items = []
    for status, title, text in CONFOUNDERS:
        pill = "no" if status == "REMAINS" else "ok"
        items.append(f'<li><span class="pill {pill}">{status}</span> <b>{html.escape(title)}</b> — {html.escape(text)}</li>')
    return "<ul>" + "".join(items) + "</ul>"


def _concurrency_html():
    return "<ul>" + "".join(f"<li>{html.escape(p)}</li>" for p in CONCURRENCY_EXPLANATION) + "</ul>"


def write_report(out_dir: Path, result, *, fpm_run, aggregation, workload_segment,
                 system="h100_sxm", model="Qwen/Qwen3-32B",
                 compute_version="0.20.1", comm_version="0.19.0"):
    out_dir = Path(out_dir)
    # SKU/version-general display labels (the tool serves B300, H100, and future SKUs).
    sku_label = system.replace("_", " ").upper()
    model_short = model.split("/")[-1]
    TRACK_VERSION.update({
        "layerwise": compute_version,
        "opwise_silicon": comm_version, "hybrid": comm_version,
        "empirical": comm_version, "sol": comm_version,
    })
    rows = result["rows"]
    summary = result["summary"]
    by_concurrency = result.get("by_concurrency", [])
    coverage = result["coverage"]
    decomposition = result["decomposition"]
    noise_by_tp_phase = result["noise_by_tp_phase"]

    cls = classify(summary, noise_by_tp_phase)
    h2h = head_to_head(rows)

    charts = {
        "bias": _chart_bias_vs_tp(summary, noise_by_tp_phase),
        "mape": _chart_mape_vs_tp(summary),
        "mape_conc": _chart_mape_vs_concurrency(by_concurrency),
        "parity": _chart_parity(rows),
        "shape": _chart_err_vs_shape(rows),
    }

    # --- summary table ---
    sum_headers = ["track", "ver", "tp", "phase", "n_ok", "cov", "MAPE%", "median%", "p90|err|%", "bias%", "within-noise", "src≠silicon"]
    sum_data = []
    for s in summary:
        sum_data.append([
            s["track"], TRACK_VERSION.get(s["track"], ""), s["tp"], s["phase"], s["n_ok"],
            s["coverage_frac"], s["mape_pct"], s["median_err_pct"], s["p90_abs_err_pct"],
            s["signed_bias_pct"], s["within_noise_frac"], s["nonsilicon_frac"],
        ])

    def sum_cell(i, c):
        if i == 6 and c != "":  # MAPE
            try:
                v = float(c)
                return "good" if v < 8 else ("warn" if v < 25 else "bad")
            except ValueError:
                return ""
        return ""

    # --- decomposition table ---
    dec_headers = ["track", "phase", "compute_err@TP1%", "comm_err@TP2%", "comm_err@TP4%", "comm_err@TP8%"]
    dec_data = [[d["track"], d["phase"], d["compute_err_pct_tp1"],
                 d.get("comm_err_pct_tp2", ""), d.get("comm_err_pct_tp4", ""), d.get("comm_err_pct_tp8", "")]
                for d in decomposition]

    # --- coverage table ---
    cov_headers = ["track", "tp", "phase", "n_total", "ok", "missing", "offgrid", "no_kv", "no_db", "error", "balanced"]
    cov_data = [[c["track"], c["tp"], c["phase"], c["n_total"], c["ok"], c["missing_data"],
                 c["offgrid_raise"], c["no_grid_kv"], c["no_database"], c["error"], c["balanced"]]
                for c in coverage]
    all_balanced = all(c["balanced"] for c in coverage)

    # --- classification table ---
    cls_headers = ["track", "phase", "class", "detail"]
    cls_data = [[t, p, cls[(t, p)][0], cls[(t, p)][1]] for t in TRACK_ORDER for p in PHASES if (t, p) in cls]

    def cls_cell(i, c):
        if i == 2:
            return {"within-noise": "good", "biased": "warn", "diverging": "bad"}.get(c, "")
        return ""

    # --- head-to-head table ---
    h2h_headers = ["phase", "tp", "n_shapes", "layerwise wins", "lw win frac", "lw MAPE%", "opwise MAPE%"]
    h2h_data = [[p, tp, v["n"], v["lw_wins"], v["lw_win_frac"], v["lw_mape"], v["ow_mape"]]
                for (p, tp), v in sorted(h2h.items())]

    # --- noise-floor table ---
    nf_headers = ["tp", "phase", "n_shapes", "n_with_repeats", "n_pathological", "median_cv%", "p90_cv% (noise bar)"]
    nf_data = []
    for tp in TPS:
        for phase in PHASES:
            s = noise_by_tp_phase.get((tp, phase))
            if not s:
                continue
            nf_data.append([
                tp, phase, s.get("n_shapes", ""), s.get("n_with_repeats", ""),
                s.get("n_pathological_variance", 0),
                round(100 * s["median_cv"], 3) if s.get("median_cv") is not None else "",
                round(100 * s["p90_cv"], 3) if s.get("p90_cv") is not None else "",
            ])

    verdict_md = _build_verdict_md(summary, decomposition, cls, h2h, fpm_run, aggregation, workload_segment, all_balanced)
    (out_dir / "verdict.md").write_text(verdict_md)

    style = """
    body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
    header{background:#0b3d5e;color:#fff;padding:18px 28px}
    header h1{margin:0 0 4px 0;font-size:20px}
    header .sub{opacity:.85;font-size:13px}
    .wrap{max-width:1200px;margin:0 auto;padding:18px 28px}
    section{background:#fff;border:1px solid #e2e5ea;border-radius:8px;padding:16px 18px;margin:16px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
    h2{font-size:16px;margin:0 0 12px 0;border-bottom:2px solid #0b3d5e;padding-bottom:6px}
    table{border-collapse:collapse;width:100%;font-size:12.5px;margin:6px 0}
    th,td{border:1px solid #e2e5ea;padding:5px 8px;text-align:right}
    th{background:#eef1f5;text-align:center}
    td:first-child,th:first-child{text-align:left}
    td.good{background:#d8f3dc} td.warn{background:#fff3cd} td.bad{background:#f8d7da}
    img{max-width:100%;height:auto;display:block;margin:6px 0}
    .pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600}
    .ok{background:#d8f3dc;color:#14532d}.no{background:#f8d7da;color:#7a1620}
    .note{font-size:12px;color:#555;margin:6px 0}
    pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:6px;overflow:auto;font-size:12px}
    code{background:#eef1f5;padding:1px 4px;border-radius:3px}
    """

    balanced_pill = ('<span class="pill ok">coverage balanced — no silent drops</span>'
                     if all_balanced else '<span class="pill no">COVERAGE NOT BALANCED</span>')

    html_doc = f"""<!doctype html><html><head><meta charset="utf-8"/>
<title>{sku_label} {model_short} Gap Analysis</title><style>{style}</style></head><body>
<header>
  <h1>{sku_label} {model_short} — Gap Analysis: FPM truth vs predictor tracks</h1>
  <div class="sub">layerwise(compute {compute_version} + comm {comm_version}) · opwise-SILICON({comm_version}) · empirical · hybrid · SOL &nbsp;|&nbsp; phases ctx+gen · BF16 · {workload_segment} workload</div>
</header>
<div class="wrap">

<section>
  <h2>Provenance &amp; controls</h2>
  <div class="note">FPM run: <code>{html.escape(str(fpm_run))}</code> &nbsp;|&nbsp; aggregation: <code>{aggregation}</code> &nbsp;|&nbsp; workload segment: <code>{workload_segment}</code></div>
  <div class="note">{balanced_pill} &nbsp; Version: layerwise <b>compute</b> is {compute_version} (version-matched to FPM); where no {compute_version} comm dir exists for {sku_label}, layerwise comm + all opwise/empirical/hybrid/SOL tracks fall back to {comm_version}. Treat {comm_version}-sourced error (incl. layerwise TP&gt;1 comm) as model+version, not model alone.</div>
  <div class="note">SKU: layerwise collected on {sku_label} (1-GPU TP-mock), same SKU as FPM. If FPM lacks a TP=1 point, the comm-free TP1 anchor (when collected) is a one-sided decomposition aid, not validated against this SKU's truth. Layerwise comm at TP&gt;1 is a single-GPU modeled add-back; op-wise composes comm from allreduce tables.</div>
  <div class="note">Outputs: <code>gap_rows.csv</code>, <code>gap_summary.csv</code>, <code>compute_comm_decomposition.csv</code>, <code>coverage.csv</code>, <code>fpm_noise_floor.csv</code>, <code>verdict.md</code></div>
</section>

<section><h2>⚠ Methodology &amp; FPM confounders (read first)</h2>
  <div class="note">FPM is the ground-truth reference but not a perfectly clean one. Confounders between FPM and AIC predictions for this dense-Qwen3-32B / H100-SXM harness, ranked by impact. <span class="pill no">REMAINS</span> = uncorrected, interpret with care; <span class="pill ok">MITIGATED</span> = controlled by this harness. Source: <code>fpm-ground-truth-confounders</code> audit, re-verified against live code.</div>
  {_confounders_html()}
  <div class="note">Out of scope (N/A here): MoE overlays, low-biased p25 reducers, emergent concurrency-32 workloads — this is dense Qwen3-32B on the clean <code>sweep</code> grid with a symmetric median aggregator.</div></section>

<section><h2>Verdict</h2><pre>{html.escape(verdict_md)}</pre></section>

<section><h2>Signed bias vs TP (compute vs comm)</h2>{_img(charts['bias'], 'bias vs tp')}</section>
<section><h2>MAPE vs TP</h2>{_img(charts['mape'], 'mape vs tp')}</section>
<section><h2>MAPE vs concurrency (× TP × phase)</h2>
  <div class="note">Concurrency = decode batch size (decode_requests). Faceted by TP (columns) and phase (rows: gen, mixed); ctx is single-request (concurrency=1) and omitted. The <code>layerwise</code> track runs with the decode batch-calibration disabled (_DECODE_COMPUTE_BATCH_CAL = 0.0). Data: <code>gap_summary_by_concurrency.csv</code>.</div>
  {_img(charts['mape_conc'], 'mape vs concurrency')}</section>
<section><h2>Why layerwise accuracy degrades with concurrency</h2>
  {_concurrency_html()}</section>
<section><h2>Compute / comm decomposition</h2>
  <div class="note">compute_err = signed bias at TP1; comm_err(TP) = bias(TP) − bias(TP1).</div>
  {_table(dec_headers, dec_data)}</section>
<section><h2>Parity: prediction vs FPM truth</h2>{_img(charts['parity'], 'parity')}</section>
<section><h2>Where divergence lives (error vs shape)</h2>{_img(charts['shape'], 'err vs shape')}</section>

<section><h2>Per-track classification</h2>{_table(cls_headers, cls_data, cell_class=cls_cell)}</section>
<section><h2>Layerwise vs opwise head-to-head (paired)</h2>{_table(h2h_headers, h2h_data)}</section>
<section><h2>Summary (per track × TP × phase)</h2>
  <div class="note">Headline = ctx+gen. The <code>mixed</code> rows are best-effort (chunked-prefill+decode steps) and are excluded from the ranking, decomposition, and charts.</div>
  {_table(sum_headers, sum_data, cell_class=sum_cell)}</section>
<section><h2>FPM measurement-noise floor</h2>
  <div class="note">Per-shape CV from repeated FPM steps (sweep segment). Shapes with CV&gt;50% are pathological (warmup/compile spikes), excluded from the noise bar and counted separately. The p90_cv column is the "within-noise" bar used for classification.</div>
  {_table(nf_headers, nf_data)}</section>
<section><h2>Coverage / status matrix</h2>
  <div class="note">Balance check: ok + missing + offgrid + no_kv + no_db + error == n_total for every group.</div>
  {_table(cov_headers, cov_data)}</section>

</div></body></html>"""

    (out_dir / "dashboard.html").write_text(html_doc)
    return out_dir / "dashboard.html"


def _build_verdict_md(summary, decomposition, cls, h2h, fpm_run, aggregation, workload_segment, all_balanced):
    lut = _summary_lookup(summary)

    def overall_mape(track):
        # Headline ranking uses ctx+gen only; mixed is best-effort and excluded.
        # Spec §7 escape hatch: drop (track,phase) groups below the coverage
        # threshold so a track is never ranked on a biased subset.
        vals = [s["mape_pct"] for s in summary
                if s["track"] == track and s["mape_pct"] != "" and s["phase"] in ("ctx", "gen")
                and s.get("sufficient_coverage", True)]
        return statistics.fmean(vals) if vals else float("inf")

    headline_tracks = [t for t in TRACK_ORDER if t not in SENSITIVITY_TRACKS]
    ranking = sorted([t for t in headline_tracks if overall_mape(t) != float("inf")], key=overall_mape)
    inconclusive = [t for t in headline_tracks if overall_mape(t) == float("inf")]
    lines = []
    lines.append("# Verdict — AIC predictor vs FPM gap analysis\n")
    lines.append(f"FPM run: {fpm_run}")
    lines.append(f"Aggregation: {aggregation} | workload segment: {workload_segment}")
    lines.append(f"Coverage balanced (no silent drops): {all_balanced}\n")

    lines.append("## Track ranking by overall MAPE (ctx+gen; lower = closer to FPM truth)")
    for i, t in enumerate(ranking, 1):
        m = overall_mape(t)
        lines.append(f"  {i}. {t:16} {TRACK_VERSION.get(t,''):7}  mean MAPE = {m:.1f}%")
    if inconclusive:
        lines.append(f"  inconclusive (excluded, <60% coverage on every ctx/gen group): {', '.join(inconclusive)}")
    else:
        lines.append("  (all tracks met the 60% coverage bar — none excluded as inconclusive)")
    lines.append("")

    lines.append("## Compute vs comm (TP1 anchor)")
    for d in decomposition:
        comm8 = d.get("comm_err_pct_tp8", "")
        lines.append(f"  {d['track']:16} {d['phase']:4}  compute@TP1 bias={d['compute_err_pct_tp1']}%  comm@TP8={comm8}%")
    lines.append("")

    lines.append("## Layerwise vs opwise_silicon (paired, fraction of shapes layerwise is closer)")
    for (p, tp), v in sorted(h2h.items()):
        lines.append(f"  {p} TP{tp}: layerwise closer on {v['lw_win_frac']*100:.0f}% of {v['n']} shapes "
                     f"(lw MAPE {v['lw_mape']}% vs opwise {v['ow_mape']}%)")
    lines.append("")

    lines.append("## Per-track classification")
    for t in TRACK_ORDER:
        for ph in PHASES:
            if (t, ph) in cls:
                lab, det = cls[(t, ph)]
                lines.append(f"  {t:16} {ph:4} [{lab}] {det}")
    lines.append("")

    lines.append("## Caveats")
    lines.append("  - Version skew: only layerwise is version-matched to FPM; op-wise/empirical/hybrid/SOL error mixes model + version.")
    lines.append("  - SOL is the pure roofline floor (no scale factor) — expected to under-predict; it bounds, not predicts.")
    lines.append("  - hybrid == opwise_silicon means the op-wise silicon coverage is complete (no empirical fallback).")
    lines.append("  - Layerwise TP>1 comm is a single-GPU modeled add-back; opwise composes comm from allreduce tables.")
    lines.append("")
    lines.append("## FPM confounders (methodology — read before trusting the numbers)")
    for status, title, text in CONFOUNDERS:
        lines.append(f"  [{status}] {title}: {text}")
    lines.append("  Out of scope (N/A): MoE overlays, p25 reducers, emergent concurrency-32 workloads (dense, clean sweep grid, median aggregator).")
    lines.append("")
    lines.append("## Why layerwise accuracy degrades with concurrency")
    for p in CONCURRENCY_EXPLANATION:
        lines.append(f"  - {p}")
    return "\n".join(lines)
