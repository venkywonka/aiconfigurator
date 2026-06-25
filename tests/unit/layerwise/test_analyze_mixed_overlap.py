# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the standalone mixed-overlap analyzer (Task 10).

The analyzer consumes a ``mixed_steps.jsonl`` side-artifact (one line per measured
forward, role in {M, C, D_BK, D_B1}) and derives, per cell ``(P, B, K)``::

    S        = C + D_BK
    Overlap  = M - S
    F        = D_B1
    R        = Overlap + min(D_BK, F)            # AIC_error reconstruction
    clamp    = "clamped" if D_BK < F else "ok"

plus a difference-in-differences contrast on R across a 2x2 (P_lo/hi x B_lo/hi)
corner set at fixed K. It never touches the FPM comparator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from collector.layerwise.diagnostics.analyze_mixed_overlap import analyze


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _cell_rows(work_unit_id, P, B, K, M, C, D_BK, D_B1):
    return [
        {"work_unit_id": work_unit_id, "role": "M", "P": P, "B": B, "K": K, "gpu_ms": M, "enforce_eager": True},
        {"work_unit_id": work_unit_id, "role": "C", "P": P, "B": B, "K": K, "gpu_ms": C, "enforce_eager": True},
        {"work_unit_id": work_unit_id, "role": "D_BK", "P": P, "B": B, "K": K, "gpu_ms": D_BK, "enforce_eager": True},
        {"work_unit_id": work_unit_id, "role": "D_B1", "P": P, "B": B, "K": K, "gpu_ms": D_B1, "enforce_eager": True},
    ]


# A synthetic 2x2 corner grid: P in {512, 2048} x B in {4, 64} at fixed K=1024.
#                                   M     C   D_BK  D_B1
_GRID = {
    ("wu_lo_lo", 512, 4, 1024): (10.0, 4.0, 5.0, 2.0),    # Overlap=1, R=1+min(5,2)=3
    ("wu_lo_hi", 512, 64, 1024): (20.0, 4.0, 14.0, 3.0),  # Overlap=2, R=2+min(14,3)=5
    ("wu_hi_lo", 2048, 4, 1024): (18.0, 12.0, 5.0, 2.0),  # Overlap=1, R=1+min(5,2)=3
    ("wu_hi_hi", 2048, 64, 1024): (30.0, 12.0, 14.0, 3.0),  # Overlap=4, R=4+min(14,3)=7
}


def _build_grid_jsonl(path: Path) -> None:
    rows = []
    for (wu, P, B, K), (M, C, D_BK, D_B1) in _GRID.items():
        rows.extend(_cell_rows(wu, P, B, K, M, C, D_BK, D_B1))
    _write_jsonl(path, rows)


def test_per_cell_overlap_r_clamp(tmp_path: Path):
    jsonl = tmp_path / "mixed_steps.jsonl"
    out_csv = tmp_path / "mixed_overlap.csv"
    _build_grid_jsonl(jsonl)

    summary = analyze(str(jsonl), str(out_csv))
    cells = {(c["P"], c["B"], c["K"]): c for c in summary["cells"]}

    lo_lo = cells[(512, 4, 1024)]
    assert lo_lo["M"] == 10.0
    assert lo_lo["S"] == 9.0  # C + D_BK = 4 + 5
    assert lo_lo["overlap"] == 1.0  # M - S
    assert lo_lo["F"] == 2.0
    assert lo_lo["R"] == 3.0  # overlap + min(D_BK, F) = 1 + min(5,2)
    assert lo_lo["clamp_status"] == "ok"  # D_BK(5) >= F(2)

    hi_hi = cells[(2048, 64, 1024)]
    assert hi_hi["S"] == 26.0
    assert hi_hi["overlap"] == 4.0
    assert hi_hi["R"] == 7.0
    assert hi_hi["clamp_status"] == "ok"


def test_clamped_status_when_dbk_below_f(tmp_path: Path):
    jsonl = tmp_path / "mixed_steps.jsonl"
    out_csv = tmp_path / "out.csv"
    # D_BK (1.0) < F (4.0) -> clamped; R = overlap + min(1,4) = overlap + 1
    _write_jsonl(jsonl, _cell_rows("wu_clamp", 256, 2, 512, M=8.0, C=3.0, D_BK=1.0, D_B1=4.0))

    summary = analyze(str(jsonl), str(out_csv))
    cell = summary["cells"][0]
    assert cell["clamp_status"] == "clamped"
    assert cell["overlap"] == 4.0  # 8 - (3 + 1)
    assert cell["R"] == 5.0  # 4 + min(1, 4)


def test_did_contrast_on_r(tmp_path: Path):
    jsonl = tmp_path / "mixed_steps.jsonl"
    out_csv = tmp_path / "out.csv"
    _build_grid_jsonl(jsonl)

    summary = analyze(str(jsonl), str(out_csv))
    # Delta = [R(Phi,Bhi) - R(Phi,Blo)] - [R(Plo,Bhi) - R(Plo,Blo)]
    #       = [7 - 3] - [5 - 3] = 4 - 2 = 2
    assert summary["did"]["R"] == 2.0


def test_did_nan_when_corner_missing(tmp_path: Path):
    jsonl = tmp_path / "mixed_steps.jsonl"
    out_csv = tmp_path / "out.csv"
    # drop the hi_hi corner
    rows = []
    for (wu, P, B, K), (M, C, D_BK, D_B1) in _GRID.items():
        if wu == "wu_hi_hi":
            continue
        rows.extend(_cell_rows(wu, P, B, K, M, C, D_BK, D_B1))
    _write_jsonl(jsonl, rows)

    summary = analyze(str(jsonl), str(out_csv))
    assert math.isnan(summary["did"]["R"])


def test_median_reduce_repeats(tmp_path: Path):
    jsonl = tmp_path / "mixed_steps.jsonl"
    out_csv = tmp_path / "out.csv"
    # three repeats of the M role: 9, 10, 11 -> median 10
    rows = _cell_rows("wu_rep", 512, 4, 1024, M=10.0, C=4.0, D_BK=5.0, D_B1=2.0)
    rows.append({"work_unit_id": "wu_rep", "role": "M", "P": 512, "B": 4, "K": 1024, "gpu_ms": 9.0, "enforce_eager": True})
    rows.append({"work_unit_id": "wu_rep", "role": "M", "P": 512, "B": 4, "K": 1024, "gpu_ms": 11.0, "enforce_eager": True})
    _write_jsonl(jsonl, rows)

    summary = analyze(str(jsonl), str(out_csv))
    cell = summary["cells"][0]
    assert cell["M"] == 10.0  # median(9, 10, 11)
    assert cell["overlap"] == 1.0


def test_writes_per_cell_csv(tmp_path: Path):
    import csv as _csv

    jsonl = tmp_path / "mixed_steps.jsonl"
    out_csv = tmp_path / "mixed_overlap.csv"
    _build_grid_jsonl(jsonl)

    analyze(str(jsonl), str(out_csv))
    assert out_csv.exists()
    with out_csv.open(newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 4
    # required columns present
    for col in ("P", "B", "K", "M", "S", "overlap", "F", "R", "clamp_status"):
        assert col in rows[0]
