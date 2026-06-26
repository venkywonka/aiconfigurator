# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""F2 (RED): the ``.done`` fail-closed attribution gate.

Spec: docs/superpowers/specs/2026-06-26-aic-fpm-integrity-design.md, "Fix 2".

A NOT-YET-EXISTENT module ``collector.layerwise.diagnostics.assert_attribution_valid``
must expose ``assert_attribution_valid(sqlite_path, decomposition_csv_path)`` that
RAISES (or otherwise signals failure) unless ALL of these hold:
  (a) the nsys sqlite has >= 1 CUPTI kernel row,
  (b) the nsys sqlite has >= 1 NVTX range whose ``text`` starts ``bench_step::``,
  (c) the decomposition CSV has >= 1 data row.

The synthetic sqlite fixtures mirror the EXACT table/column names the real
pipeline queries (collector/layerwise/common/parse_nsys_step_sweep.py):
  * CUPTI kernels: ``CUPTI_ACTIVITY_KIND_KERNEL`` (correlationId, graphNodeId,
    start, end, shortName, globalPid) joined with ``CUPTI_ACTIVITY_KIND_RUNTIME``
    (correlationId, globalTid, start, end); names resolve through ``StringIds``
    (id, value).
  * NVTX ranges: ``NVTX_EVENTS`` (text, start, end, globalTid) with bench_step
    text format ``bench_step::N<n>::bs<B>::past<p>``.

The module does not exist yet, so importing it is expected to raise ImportError
(feature missing) -- that is the correct RED signal. This file writes ONLY the
test; no production code.
"""

import csv
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# nsys globalTid with the low 24 bits cleared == globalPid (see
# parse_nsys_step_sweep._GLOBAL_PID_MASK). The CUPTI join is
# ``K.globalPid = (R.globalTid & mask)``, so pick a tid whose masked value is a
# clean pid the kernel row can reference.
_GLOBAL_PID_MASK = -16777216
_TID = 4242
_PID = _TID & _GLOBAL_PID_MASK


def _create_schema(con: sqlite3.Connection) -> None:
    """Create the exact tables/columns analyze_sqlite + _query_kernels query."""
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)"
    )
    cur.execute(
        "CREATE TABLE NVTX_EVENTS ("
        "text TEXT, start INTEGER, end INTEGER, globalTid INTEGER)"
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
        "correlationId INTEGER, graphNodeId INTEGER, start INTEGER, end INTEGER, "
        "shortName INTEGER, globalPid INTEGER)"
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME ("
        "correlationId INTEGER, globalTid INTEGER, start INTEGER, end INTEGER)"
    )
    con.commit()


def _insert_bench_step(con: sqlite3.Connection, *, start: int, end: int) -> None:
    con.execute(
        "INSERT INTO NVTX_EVENTS (text, start, end, globalTid) VALUES (?, ?, ?, ?)",
        (f"bench_step::N0000003::bs1::past100", start, end, _TID),
    )
    con.commit()


def _insert_kernel(
    con: sqlite3.Connection,
    *,
    correlation_id: int,
    name: str,
    kernel_start: int,
    kernel_end: int,
    runtime_start: int,
) -> None:
    cur = con.cursor()
    # Register the kernel's short name in StringIds (CUPTI shortName -> id).
    cur.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM StringIds")
    short_id = cur.fetchone()[0]
    cur.execute("INSERT INTO StringIds (id, value) VALUES (?, ?)", (short_id, name))
    cur.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
        "(correlationId, graphNodeId, start, end, shortName, globalPid) "
        "VALUES (?, NULL, ?, ?, ?, ?)",
        (correlation_id, kernel_start, kernel_end, short_id, _PID),
    )
    cur.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
        "(correlationId, globalTid, start, end) VALUES (?, ?, ?, ?)",
        (correlation_id, _TID, runtime_start, runtime_start + 5),
    )
    con.commit()


def _write_sqlite(
    path: Path,
    *,
    with_kernel: bool = True,
    with_bench_step: bool = True,
) -> None:
    con = sqlite3.connect(str(path))
    try:
        _create_schema(con)
        if with_bench_step:
            _insert_bench_step(con, start=1000, end=2000)
        if with_kernel:
            # runtime_start falls inside the bench_step window [1000, 2000) so the
            # kernel attributes to a step when one is present.
            _insert_kernel(
                con,
                correlation_id=1,
                name="sm90_gemm_kernel",
                kernel_start=1100,
                kernel_end=1500,
                runtime_start=1100,
            )
    finally:
        con.close()


# Stable header the real write_decomposition_csv emits (aic_fpm_attribute.py).
_DECOMP_COLS = [
    "phase", "batch_size", "past_kv", "wall_ms", "aic_total_ms",
    "aic_compute_ms", "aic_comm_ms", "aic_other_ms",
    "gpu_compute_ms", "gpu_comm_ms", "gpu_busy_ms",
    "overlap_ms", "overhead_ms", "gap_ms",
    "term_compute_err", "term_comm_err", "term_aic_other",
    "term_overlap", "term_neg_overhead",
]


def _write_decomposition_csv(path: Path, *, with_row: bool = True) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_DECOMP_COLS)
        writer.writeheader()
        if with_row:
            row = {col: 0.0 for col in _DECOMP_COLS}
            row["phase"] = "decode"
            row["batch_size"] = 1
            row["past_kv"] = 100
            writer.writerow(row)


class AssertAttributionValidTests(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _gate(self):
        # The module does not exist yet -> ImportError is the correct RED.
        from collector.layerwise.diagnostics.assert_attribution_valid import (
            assert_attribution_valid,
        )

        return assert_attribution_valid

    def _build(self, *, with_kernel=True, with_bench_step=True, with_csv_row=True):
        sqlite_path = self.tmp / "trace.sqlite"
        csv_path = self.tmp / "decomposition.csv"
        _write_sqlite(
            sqlite_path, with_kernel=with_kernel, with_bench_step=with_bench_step
        )
        _write_decomposition_csv(csv_path, with_row=with_csv_row)
        return sqlite_path, csv_path

    # (a) valid triple -> passes (no raise)
    def test_valid_triple_passes(self):
        gate = self._gate()
        sqlite_path, csv_path = self._build()
        # Must not raise. A truthy/zero "success" return is acceptable.
        result = gate(str(sqlite_path), str(csv_path))
        if result is not None:
            self.assertIn(result, (0, True), f"unexpected gate return {result!r}")

    # (b) zero CUPTI kernel rows -> raises
    def test_zero_kernels_raises(self):
        gate = self._gate()
        sqlite_path, csv_path = self._build(with_kernel=False)
        with self.assertRaises(Exception):
            gate(str(sqlite_path), str(csv_path))

    # (c) zero bench_step:: NVTX rows -> raises
    def test_zero_bench_step_rows_raises(self):
        gate = self._gate()
        sqlite_path, csv_path = self._build(with_bench_step=False)
        with self.assertRaises(Exception):
            gate(str(sqlite_path), str(csv_path))

    # (d) empty decomposition.csv (header only, no data rows) -> raises
    def test_empty_decomposition_csv_raises(self):
        gate = self._gate()
        sqlite_path, csv_path = self._build(with_csv_row=False)
        with self.assertRaises(Exception):
            gate(str(sqlite_path), str(csv_path))


class WriteDecompositionCsvFailClosedTests(unittest.TestCase):
    """defect 3b: write_decomposition_csv([]) must SIGNAL failure (raise), not
    silently return -- otherwise the gate / ``|| warn`` in the shell never fires."""

    def test_write_decomposition_csv_empty_signals_failure(self):
        from collector.layerwise.diagnostics.aic_fpm_attribute import (
            write_decomposition_csv,
        )

        with TemporaryDirectory() as d:
            out_path = str(Path(d) / "decomposition.csv")
            with self.assertRaises(Exception):
                write_decomposition_csv([], out_path)


if __name__ == "__main__":
    unittest.main()
