# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed attribution-validity gate (spec "Fix 2").

A run's ``.done`` marker must only be stamped once attribution is *real*. This
diagnostic gate asserts three independent facts about a finished attribute unit
and names which one failed (so the failure pinpoints the broken stage):

  (a) the nsys sqlite has >= 1 CUPTI kernel row   -> no kernels = profiler window
  (b) the nsys sqlite has >= 1 ``bench_step::`` NVTX row -> no ranges = step marker
  (c) ``decomposition.csv`` has >= 1 data row     -> rows-but-empty = join/config

Any miss raises :class:`AttributionInvalidError`; the ``main()`` entry point
exits non-zero with the same message so the driver shell can ``die`` on it.
"""

from __future__ import annotations

import csv
import sqlite3
import sys


class AttributionInvalidError(RuntimeError):
    """Raised when an attribute unit's artifacts fail a validity check."""


def _table_exists(cur: sqlite3.Cursor, name: str) -> bool:
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def _count_cupti_kernels(cur: sqlite3.Cursor) -> int:
    if not _table_exists(cur, "CUPTI_ACTIVITY_KIND_KERNEL"):
        return 0
    cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    return int(cur.fetchone()[0] or 0)


def _count_bench_step_rows(cur: sqlite3.Cursor) -> int:
    if not _table_exists(cur, "NVTX_EVENTS"):
        return 0
    cur.execute(
        "SELECT COUNT(*) FROM NVTX_EVENTS "
        "WHERE text IS NOT NULL AND text LIKE 'bench_step::%'"
    )
    return int(cur.fetchone()[0] or 0)


def _count_decomposition_rows(decomposition_csv_path: str) -> int:
    with open(decomposition_csv_path, newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)  # header
        except StopIteration:
            return 0
        return sum(1 for row in reader if any(cell.strip() for cell in row))


def assert_attribution_valid(
    sqlite_path: str,
    decomposition_csv_path: str,
    *,
    allow_empty_decomposition: bool = False,
) -> int:
    """Raise :class:`AttributionInvalidError` unless validity checks hold.

    By default all three checks hold: kernels, bench_step ranges, and >=1 decomposition
    row. ``allow_empty_decomposition`` keeps the profiler/marker checks fail-closed while
    allowing a header-only decomposition CSV for no-overlap diagnostic buckets.

    Returns ``0`` on success so callers may use the return value as an exit code.
    """
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = con.cursor()
        n_kernels = _count_cupti_kernels(cur)
        n_bench_steps = _count_bench_step_rows(cur)
    finally:
        con.close()

    if n_kernels < 1:
        raise AttributionInvalidError(
            f"check (a) FAILED: no CUPTI kernel rows in {sqlite_path} "
            "(profiler window captured no kernels)."
        )
    if n_bench_steps < 1:
        raise AttributionInvalidError(
            f"check (b) FAILED: no 'bench_step::' NVTX rows in {sqlite_path} "
            "(step marker did not emit ranges)."
        )

    n_rows = _count_decomposition_rows(decomposition_csv_path)
    if n_rows < 1 and not allow_empty_decomposition:
        raise AttributionInvalidError(
            f"check (c) FAILED: no data rows in {decomposition_csv_path} "
            "(shape join / config produced an empty decomposition)."
        )
    return 0


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Fail-closed attribution-validity gate (spec Fix 2)."
    )
    p.add_argument("--sqlite", required=True, help="nsys .sqlite for the attribute unit")
    p.add_argument(
        "--decomposition",
        required=True,
        help="decomposition.csv emitted by aic_fpm_attribute",
    )
    p.add_argument(
        "--allow-empty-decomposition",
        action="store_true",
        help="allow a header-only decomposition CSV while still requiring kernels and bench_step rows",
    )
    args = p.parse_args(argv)

    try:
        assert_attribution_valid(
            args.sqlite,
            args.decomposition,
            allow_empty_decomposition=args.allow_empty_decomposition,
        )
    except (AttributionInvalidError, OSError, sqlite3.Error) as exc:
        print(f"[assert_attribution_valid] {exc}", file=sys.stderr)
        return 1
    if args.allow_empty_decomposition and _count_decomposition_rows(args.decomposition) < 1:
        print(
            "[assert_attribution_valid] OK: kernels + bench_step ranges present; "
            "decomposition empty by explicit allowance"
        )
    else:
        print("[assert_attribution_valid] OK: kernels + bench_step ranges + decomposition rows present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
