#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone analyzer for the mixed-step overlap experiment (Task 10).

Consumes a ``mixed_steps.jsonl`` side-artifact written by the worker mixed driver
(one line per measured forward, role in {M, C, D_BK, D_B1}, keyed by
``work_unit_id``) and derives, per cell ``(P, B, K)``::

    M        = measured fused mixed forward (P prefill tokens + B decodes @ KV K)
    C        = context-only reference, P prefill tokens
    D_BK     = decode-only reference, B requests @ KV K
    D_B1     = decode-only reference, B requests @ KV 1
    S        = C + D_BK                              # additive (no-overlap) model
    Overlap  = M - S                                 # < 0 => fused step is faster
    F        = D_B1
    R        = Overlap + min(D_BK, F)                # AIC_error reconstruction
    clamp    = "clamped" if D_BK < F else "ok"       # min() clamp diagnostic

Across a 2x2 (P_lo/hi x B_lo/hi) corner set at fixed K it reports a
difference-in-differences contrast::

    DiD(R) = [R(Phi,Bhi) - R(Phi,Blo)] - [R(Plo,Bhi) - R(Plo,Blo)]

(and the same on Overlap). Repeats of the same (cell, role) are median-reduced.

This analyzer is intentionally standalone: it does NOT import or route through the
FPM comparator (``compare_aic_layerwise_fpm.py``) or the single-population
aggregation pipeline. Pure stdlib (json/csv/statistics) so it runs without
vLLM/torch installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_ROLES = ("M", "C", "D_BK", "D_B1")


def _load_steps(jsonl_path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(jsonl_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _cv(values: list[float]) -> float:
    """Coefficient of variation; 0.0 when fewer than two samples or zero mean."""
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    if mean == 0.0:
        return 0.0
    return float(statistics.stdev(values) / mean)


def _reduce_cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group rows by (P, B, K), median-reduce repeats per role, derive metrics."""
    # (P, B, K) -> role -> list[gpu_ms]
    grouped: dict[tuple[int, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (int(row["P"]), int(row["B"]), int(row["K"]))
        grouped[key][str(row["role"])].append(float(row["gpu_ms"]))

    cells: list[dict[str, Any]] = []
    for (P, B, K), per_role in sorted(grouped.items()):
        reduced = {
            role: _median(per_role[role]) for role in _ROLES if per_role.get(role)
        }
        # CV across ALL measured forwards in this cell (noise-floor input).
        all_samples = [v for samples in per_role.values() for v in samples]
        cell: dict[str, Any] = {
            "P": P,
            "B": B,
            "K": K,
            "M": reduced.get("M", math.nan),
            "C": reduced.get("C", math.nan),
            "D_BK": reduced.get("D_BK", math.nan),
            "D_B1": reduced.get("D_B1", math.nan),
            "repeats": max((len(v) for v in per_role.values()), default=0),
            "cv": _cv(all_samples),
        }
        M, C, D_BK, D_B1 = cell["M"], cell["C"], cell["D_BK"], cell["D_B1"]
        if any(math.isnan(x) for x in (M, C, D_BK, D_B1)):
            cell.update(
                {"S": math.nan, "overlap": math.nan, "F": math.nan, "R": math.nan,
                 "clamp_status": "incomplete"}
            )
        else:
            S = C + D_BK
            overlap = M - S
            F = D_B1
            R = overlap + min(D_BK, F)
            cell.update(
                {
                    "S": S,
                    "overlap": overlap,
                    "F": F,
                    "R": R,
                    "clamp_status": "clamped" if D_BK < F else "ok",
                }
            )
        cells.append(cell)
    return cells


def _corner_grid(cells: list[dict[str, Any]]) -> dict[str, float] | None:
    """Locate a 2x2 P_lo/hi x B_lo/hi grid at a single fixed K.

    Returns None when no fixed-K plane has exactly the four (lo/hi, lo/hi) corners.
    """
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        by_k[cell["K"]].append(cell)
    for K, plane in by_k.items():
        ps = sorted({c["P"] for c in plane})
        bs = sorted({c["B"] for c in plane})
        if len(ps) < 2 or len(bs) < 2:
            continue
        p_lo, p_hi = ps[0], ps[-1]
        b_lo, b_hi = bs[0], bs[-1]
        lookup = {(c["P"], c["B"]): c for c in plane}
        corners = {
            "lo_lo": lookup.get((p_lo, b_lo)),
            "lo_hi": lookup.get((p_lo, b_hi)),
            "hi_lo": lookup.get((p_hi, b_lo)),
            "hi_hi": lookup.get((p_hi, b_hi)),
        }
        return {"K": K, "corners": corners}
    return None


def _did(corners: dict[str, dict[str, Any] | None], metric: str) -> float:
    """Difference-in-differences on `metric`; NaN if any corner is missing."""
    needed = ("lo_lo", "lo_hi", "hi_lo", "hi_hi")
    if any(corners.get(name) is None for name in needed):
        return math.nan
    vals = {name: corners[name][metric] for name in needed}
    if any(isinstance(v, float) and math.isnan(v) for v in vals.values()):
        return math.nan
    return (vals["hi_hi"] - vals["hi_lo"]) - (vals["lo_hi"] - vals["lo_lo"])


def _write_csv(cells: list[dict[str, Any]], out_csv: str) -> None:
    columns = [
        "P", "B", "K", "M", "C", "D_BK", "D_B1",
        "S", "overlap", "F", "R", "clamp_status", "repeats", "cv",
    ]
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for cell in cells:
            writer.writerow({col: cell.get(col, "") for col in columns})


def analyze(mixed_steps_jsonl: str, out_csv: str) -> dict[str, Any]:
    """Compute per-cell Overlap/R/clamp + DiD; write per-cell CSV; return summary.

    Returns a summary dict with keys:
        ``cells``       list of per-cell metric dicts
        ``did``         {"R": float, "overlap": float} (NaN if a corner is missing)
        ``noise_floor`` sqrt(3) * (max per-cell CV) -- propagated DiD noise floor
        ``verdict``     "fused_faster" / "additive" / "inconclusive"
    """
    rows = _load_steps(mixed_steps_jsonl)
    cells = _reduce_cells(rows)
    _write_csv(cells, out_csv)

    grid = _corner_grid(cells)
    if grid is None:
        did = {"R": math.nan, "overlap": math.nan}
    else:
        corners = grid["corners"]
        did = {
            "R": _did(corners, "R"),
            "overlap": _did(corners, "overlap"),
        }

    cvs = [c["cv"] for c in cells if not math.isnan(c.get("cv", math.nan))]
    max_cv = max(cvs) if cvs else 0.0
    # DiD is a sum/difference of 4 corner values -> error propagates as sqrt(4)=2;
    # but each corner R = overlap + min(...) folds in ~3 measured forwards, so the
    # conservative per-corner floor is sqrt(3)*CV (spec); the DiD floor uses that
    # times 2 for the four-corner combination.
    noise_floor = math.sqrt(3.0) * max_cv

    did_r = did["R"]
    if math.isnan(did_r):
        verdict = "inconclusive"
    else:
        overlaps = [c["overlap"] for c in cells if not math.isnan(c.get("overlap", math.nan))]
        median_overlap = _median(overlaps) if overlaps else math.nan
        if not math.isnan(median_overlap) and median_overlap < 0.0:
            verdict = "fused_faster"
        elif abs(did_r) <= noise_floor:
            verdict = "additive"
        else:
            verdict = "interaction_present"

    return {
        "cells": cells,
        "did": did,
        "noise_floor": noise_floor,
        "verdict": verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mixed_steps_jsonl", help="path to mixed_steps.jsonl")
    parser.add_argument(
        "--out-csv",
        default="mixed_overlap.csv",
        help="per-cell CSV output path (default: mixed_overlap.csv)",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="optional path to write the summary JSON (did/noise_floor/verdict)",
    )
    args = parser.parse_args(argv)

    summary = analyze(args.mixed_steps_jsonl, args.out_csv)

    if args.summary_json:
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))
    for cell in summary["cells"]:
        print(
            f"  P{cell['P']} B{cell['B']} K{cell['K']}: "
            f"M={cell['M']} S={cell['S']} overlap={cell['overlap']} "
            f"R={cell['R']} [{cell['clamp_status']}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
