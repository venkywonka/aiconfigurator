#!/usr/bin/env python3
"""Analyze communication/compute overlap in an Nsight Systems sqlite export.

The preferred input is a trace with `bench_step::*` NVTX ranges from
`vllm_step_marker.py`; use `--whole-trace` for a coarse trace-wide summary when
those markers are unavailable.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
sys.path.insert(0, str(_COMMON_DIR))

from parse_nsys_step_sweep import (
    _DEFAULT_KERNEL_DROP,
    _GLOBAL_PID_MASK,
    _build_nvtx_lookups,
    _query_kernels,
    _step_of_with_starts,
)


_WHOLE_TRACE_STEP = (-1, 0, 0, 0)
_DEFAULT_COMM_RE = re.compile(
    _DEFAULT_KERNEL_DROP.pattern + r"|all[_-]?reduce|allreduce|nccl",
    re.IGNORECASE,
)


def _merge_intervals_ns(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted((int(s), int(e)) for s, e in intervals if e > s):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals_ns(intervals))


def _overlap_ns(left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]) -> int:
    left_merged = _merge_intervals_ns(left)
    right_merged = _merge_intervals_ns(right)
    i = j = 0
    total = 0
    while i < len(left_merged) and j < len(right_merged):
        ls, le = left_merged[i]
        rs, re = right_merged[j]
        total += max(0, min(le, re) - max(ls, rs))
        if le <= re:
            i += 1
        else:
            j += 1
    return total


def _span_ns(intervals: Iterable[tuple[int, int]]) -> int:
    values = [(s, e) for s, e in intervals if e > s]
    if not values:
        return 0
    return max(e for _, e in values) - min(s for s, _ in values)


def _format_us(ns: int | float) -> float:
    return float(ns) / 1000.0


def _build_step_indexes(step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]]):
    step_wins_by_pid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]] = defaultdict(list)
    all_step_wins: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for tid, wins in step_wins_by_tid.items():
        step_wins_by_pid[tid & _GLOBAL_PID_MASK].extend(wins)
        all_step_wins.extend(wins)
    for wins in step_wins_by_pid.values():
        wins.sort()
    all_step_wins.sort()
    return (
        step_wins_by_pid,
        all_step_wins,
        {tid: [s for s, _, _ in wins] for tid, wins in step_wins_by_tid.items()},
        {pid: [s for s, _, _ in wins] for pid, wins in step_wins_by_pid.items()},
        [s for s, _, _ in all_step_wins],
    )


def _find_step(
    tid: int,
    runtime_start: int,
    *,
    step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]],
    step_wins_by_pid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]],
    all_step_wins: list[tuple[int, int, tuple[int, int, int, int]]],
    step_starts_by_tid: dict[int, list[int]],
    step_starts_by_pid: dict[int, list[int]],
    all_step_starts: list[int],
) -> tuple[int, int, int, int] | None:
    step = _step_of_with_starts(
        step_wins_by_tid.get(tid, []),
        step_starts_by_tid.get(tid, []),
        runtime_start,
    )
    if step is not None:
        return step
    pid = tid & _GLOBAL_PID_MASK
    step = _step_of_with_starts(
        step_wins_by_pid.get(pid, []),
        step_starts_by_pid.get(pid, []),
        runtime_start,
    )
    if step is not None:
        return step
    return _step_of_with_starts(all_step_wins, all_step_starts, runtime_start)


def analyze_sqlite(
    sqlite_path: str | Path,
    *,
    comm_re: re.Pattern[str] = _DEFAULT_COMM_RE,
    batch_size: int | None = None,
    past_kv: int | None = None,
    per_pid: bool = False,
    whole_trace: bool = False,
) -> tuple[list[dict], dict]:
    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(path)

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]] = defaultdict(list)
    if not whole_trace:
        step_wins_by_tid, _module_intervals = _build_nvtx_lookups(cur)
        if not step_wins_by_tid:
            raise RuntimeError("no bench_step NVTX ranges found; retry with --whole-trace for a coarse summary")
    (
        step_wins_by_pid,
        all_step_wins,
        step_starts_by_tid,
        step_starts_by_pid,
        all_step_starts,
    ) = _build_step_indexes(step_wins_by_tid)

    cur.execute("SELECT id, value FROM StringIds")
    string_ids = dict(cur.fetchall())

    groups: dict[tuple[tuple[int, int, int, int], int | None], dict[str, list]] = defaultdict(
        lambda: {"compute": [], "comm": []}
    )
    name_totals: dict[tuple[tuple[int, int, int, int], int | None], Counter[str]] = defaultdict(Counter)
    seen = set()
    outside_step = 0
    for row in _query_kernels(cur):
        cid, graph_node_id, kernel_start, kernel_end, short_name_id, tid, runtime_start, _cap_s, _cap_e = row
        key = (tid, cid, graph_node_id)
        if key in seen:
            continue
        seen.add(key)
        if whole_trace:
            step = _WHOLE_TRACE_STEP
        else:
            step = _find_step(
                tid,
                runtime_start,
                step_wins_by_tid=step_wins_by_tid,
                step_wins_by_pid=step_wins_by_pid,
                all_step_wins=all_step_wins,
                step_starts_by_tid=step_starts_by_tid,
                step_starts_by_pid=step_starts_by_pid,
                all_step_starts=all_step_starts,
            )
            if step is None:
                outside_step += 1
                continue
        step_n, bs, past, _run = step
        if batch_size is not None and bs != batch_size:
            continue
        if past_kv is not None and past != past_kv:
            continue
        name = string_ids.get(short_name_id, str(short_name_id))
        kind = "comm" if comm_re.search(name) else "compute"
        group_key = (step, (tid & _GLOBAL_PID_MASK) if per_pid else None)
        groups[group_key][kind].append((int(kernel_start), int(kernel_end), name))
        name_totals[group_key][name] += int(kernel_end) - int(kernel_start)
    con.close()

    rows = []
    for (step, pid), kernels_by_kind in groups.items():
        compute = kernels_by_kind["compute"]
        comm = kernels_by_kind["comm"]
        compute_intervals = [(s, e) for s, e, _ in compute]
        comm_intervals = [(s, e) for s, e, _ in comm]
        all_intervals = compute_intervals + comm_intervals
        compute_union = _union_ns(compute_intervals)
        comm_union = _union_ns(comm_intervals)
        total_union = _union_ns(all_intervals)
        overlap = _overlap_ns(compute_intervals, comm_intervals)
        comm_visible = max(0, total_union - compute_union)
        step_n, bs, past, run = step
        top_comm = Counter({name: ns for name, ns in name_totals[(step, pid)].items() if comm_re.search(name)})
        rows.append(
            {
                "step": step_n,
                "batch_size": bs,
                "past_kv": past,
                "measure_run": run,
                "pid": "" if pid is None else pid,
                "kernel_count": len(compute) + len(comm),
                "compute_kernels": len(compute),
                "comm_kernels": len(comm),
                "compute_gpu_us": _format_us(sum(e - s for s, e in compute_intervals)),
                "comm_gpu_us": _format_us(sum(e - s for s, e in comm_intervals)),
                "compute_union_us": _format_us(compute_union),
                "comm_union_us": _format_us(comm_union),
                "total_union_us": _format_us(total_union),
                "total_span_us": _format_us(_span_ns(all_intervals)),
                "comm_compute_overlap_us": _format_us(overlap),
                "comm_visible_us": _format_us(comm_visible),
                "comm_overlap_pct": (100.0 * overlap / comm_union) if comm_union else 0.0,
                "comm_visible_pct": (100.0 * comm_visible / comm_union) if comm_union else 0.0,
                "top_comm_kernels": ";".join(
                    f"{name}:{_format_us(ns):.3f}us" for name, ns in top_comm.most_common(5)
                ),
            }
        )
    rows.sort(key=lambda r: (r["step"], r["measure_run"], r["batch_size"], r["past_kv"], str(r["pid"])))
    return rows, {
        "sqlite": str(path),
        "groups": len(rows),
        "outside_step": outside_step,
        "deduped_kernels": len(seen),
        "whole_trace": whole_trace,
    }


def _load_kernel_rows(
    sqlite_path: str | Path,
    *,
    batch_size: int | None = None,
    past_kv: int | None = None,
    whole_trace: bool = False,
) -> list[dict]:
    """Load per-kernel rows for the per-barrier extractor.

    Returns one dict per (step, pid) kernel with keys:
      step (the (n, bs, past, run) tuple), pid (rank = tid & _GLOBAL_PID_MASK),
      kernel_start, kernel_end, runtime_start (host cudaLaunchKernel = launch_ts),
      name. Step attribution + dedup mirror analyze_sqlite exactly.
    """
    path = Path(sqlite_path)
    if not path.exists():
        raise FileNotFoundError(path)

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = con.cursor()
    step_wins_by_tid: dict[int, list[tuple[int, int, tuple[int, int, int, int]]]] = defaultdict(list)
    if not whole_trace:
        step_wins_by_tid, _module_intervals = _build_nvtx_lookups(cur)
        if not step_wins_by_tid:
            raise RuntimeError("no bench_step NVTX ranges found; retry with --whole-trace for a coarse summary")
    (
        step_wins_by_pid,
        all_step_wins,
        step_starts_by_tid,
        step_starts_by_pid,
        all_step_starts,
    ) = _build_step_indexes(step_wins_by_tid)

    cur.execute("SELECT id, value FROM StringIds")
    string_ids = dict(cur.fetchall())

    seen = set()
    out: list[dict] = []
    for row in _query_kernels(cur):
        cid, graph_node_id, kernel_start, kernel_end, short_name_id, tid, runtime_start, _cap_s, _cap_e = row
        key = (tid, cid, graph_node_id)
        if key in seen:
            continue
        seen.add(key)
        if whole_trace:
            step = _WHOLE_TRACE_STEP
        else:
            step = _find_step(
                tid,
                runtime_start,
                step_wins_by_tid=step_wins_by_tid,
                step_wins_by_pid=step_wins_by_pid,
                all_step_wins=all_step_wins,
                step_starts_by_tid=step_starts_by_tid,
                step_starts_by_pid=step_starts_by_pid,
                all_step_starts=all_step_starts,
            )
            if step is None:
                continue
        _step_n, bs, past, _run = step
        if batch_size is not None and bs != batch_size:
            continue
        if past_kv is not None and past != past_kv:
            continue
        name = string_ids.get(short_name_id, str(short_name_id))
        out.append(
            {
                "step": step,
                "pid": tid & _GLOBAL_PID_MASK,
                "kernel_start": int(kernel_start),
                "kernel_end": int(kernel_end),
                "runtime_start": int(runtime_start),
                "name": name,
            }
        )
    con.close()
    return out


def extract_per_barrier(
    sqlite_path: str | Path,
    *,
    comm_re: re.Pattern[str] = _DEFAULT_COMM_RE,
    batch_size: int | None = None,
    past_kv: int | None = None,
    whole_trace: bool = False,
    kernel_rows: list[dict] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Per-barrier arrival-skew decomposition (design.md §0, the decisive instrument).

    For each (step, pid=rank): order kernels by kernel_start, classify comm via
    comm_re; the i-th comm kernel (0-based) is barrier_index i for that rank.
      arrival_ts(i) = kernel_end of the last COMPUTE kernel before that comm
                      kernel within (step, pid); if none precedes, arrival =
                      the comm kernel's runtime_start (launch) and the row is
                      flagged arrival_is_fallback=True.
      launch_ts(i)  = runtime_start of the comm kernel.
      ar_start(i)   = kernel_start; ar_end(i) = kernel_end.

    A logical barrier = (step, barrier_index) across all ranks. A barrier with
    fewer than the step's max rank-count instances is matched=False and EXCLUDED
    from spread aggregates (spreads/transfer/spin set to None).

    Returns (per_rank_rows, per_barrier_rows, meta). `kernel_rows` is a test seam
    (pre-loaded rows in the _load_kernel_rows shape); if None, the sqlite is read.
    """
    if kernel_rows is None:
        kernel_rows = _load_kernel_rows(
            sqlite_path,
            batch_size=batch_size,
            past_kv=past_kv,
            whole_trace=whole_trace,
        )

    # group kernels by (step, pid)
    by_step_pid: dict[tuple, list[dict]] = defaultdict(list)
    for kr in kernel_rows:
        by_step_pid[(kr["step"], kr["pid"])].append(kr)

    per_rank_rows: list[dict] = []
    arrival_fallbacks = 0
    for (step, pid), kerns in by_step_pid.items():
        kerns_sorted = sorted(kerns, key=lambda k: (k["kernel_start"], k["kernel_end"]))
        last_compute_end: int | None = None
        barrier_index = 0
        for k in kerns_sorted:
            is_comm = bool(comm_re.search(k["name"]))
            if not is_comm:
                last_compute_end = k["kernel_end"]
                continue
            if last_compute_end is None:
                arrival_ts = k["runtime_start"]
                arrival_is_fallback = True
                arrival_fallbacks += 1
            else:
                arrival_ts = last_compute_end
                arrival_is_fallback = False
            per_rank_rows.append(
                {
                    "step": step[0] if isinstance(step, tuple) else step,
                    "step_key": step,
                    "barrier_index": barrier_index,
                    "rank": pid,
                    "arrival_ts": arrival_ts,
                    "launch_ts": k["runtime_start"],
                    "ar_start_ts": k["kernel_start"],
                    "ar_end_ts": k["kernel_end"],
                    "arrival_is_fallback": arrival_is_fallback,
                }
            )
            barrier_index += 1

    # expected n_ranks per step = max distinct ranks observed in that step
    ranks_per_step: dict[object, set] = defaultdict(set)
    for r in per_rank_rows:
        ranks_per_step[r["step_key"]].add(r["rank"])
    expected_ranks = {sk: len(ranks) for sk, ranks in ranks_per_step.items()}

    # group per-rank rows into logical barriers (step_key, barrier_index)
    by_barrier: dict[tuple, list[dict]] = defaultdict(list)
    for r in per_rank_rows:
        by_barrier[(r["step_key"], r["barrier_index"])].append(r)

    per_barrier_rows: list[dict] = []
    unmatched_barriers = 0
    spin_by_rankrow: dict[int, int] = {}
    for (step_key, bidx), members in by_barrier.items():
        n_ranks = len(members)
        expected = expected_ranks.get(step_key, n_ranks)
        matched = n_ranks == expected
        barrier_complete = max(m["ar_end_ts"] for m in members)
        row = {
            "step": step_key[0] if isinstance(step_key, tuple) else step_key,
            "step_key": step_key,
            "barrier_index": bidx,
            "n_ranks": n_ranks,
            "matched": matched,
        }
        if matched:
            launches = [m["launch_ts"] for m in members]
            arrivals = [m["arrival_ts"] for m in members]
            transfers = [m["ar_end_ts"] - m["ar_start_ts"] for m in members]
            spins = []
            for m in members:
                spin = barrier_complete - m["arrival_ts"]
                spins.append(spin)
                spin_by_rankrow[id(m)] = spin
            row.update(
                {
                    "launch_spread": max(launches) - min(launches),
                    "arrival_spread": max(arrivals) - min(arrivals),
                    "transfer": min(transfers),
                    "barrier_complete": barrier_complete,
                    "spin_mean": sum(spins) / len(spins),
                    "spin_max": max(spins),
                    "spin_per_rank": sorted(
                        ((m["rank"], barrier_complete - m["arrival_ts"]) for m in members),
                        key=lambda t: t[0],
                    ),
                }
            )
        else:
            unmatched_barriers += 1
            row.update(
                {
                    "launch_spread": None,
                    "arrival_spread": None,
                    "transfer": None,
                    "barrier_complete": barrier_complete,
                    "spin_mean": None,
                    "spin_max": None,
                    "spin_per_rank": None,
                }
            )
        per_barrier_rows.append(row)

    # annotate per-rank rows with spin (None for unmatched barriers)
    for r in per_rank_rows:
        r["spin"] = spin_by_rankrow.get(id(r))

    # per-rank comm-kernel counts per step: a detectable signal that ranks
    # disagree on how many barriers they hit (ordinal mismatch). Without this
    # a missing/extra allreduce on one rank would silently mis-align logical
    # barriers across ranks instead of being caught.
    comm_counts_per_rank: dict[object, dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for r in per_rank_rows:
        comm_counts_per_rank[r["step_key"]][r["rank"]] += 1
    comm_counts_per_rank = {
        sk: dict(counts) for sk, counts in comm_counts_per_rank.items()
    }
    comm_count_mismatch_steps = [
        sk
        for sk, counts in comm_counts_per_rank.items()
        if len(set(counts.values())) > 1
    ]

    per_rank_rows.sort(
        key=lambda r: (str(r["step_key"]), r["barrier_index"], r["rank"])
    )
    per_barrier_rows.sort(
        key=lambda r: (str(r["step_key"]), r["barrier_index"])
    )
    meta = {
        "sqlite": str(sqlite_path),
        "barriers": len(per_barrier_rows),
        "matched_barriers": len(per_barrier_rows) - unmatched_barriers,
        "unmatched_barriers": unmatched_barriers,
        "arrival_fallbacks": arrival_fallbacks,
        "per_rank_rows": len(per_rank_rows),
        "comm_counts_per_rank": comm_counts_per_rank,
        "comm_count_mismatch_steps": comm_count_mismatch_steps,
    }
    return per_rank_rows, per_barrier_rows, meta


def _print_table(rows: list[dict], metadata: dict) -> None:
    print(
        f"[overlap] groups={metadata['groups']} kernels={metadata['deduped_kernels']} "
        f"outside_step={metadata['outside_step']} whole_trace={metadata['whole_trace']}",
        file=sys.stderr,
    )
    if not rows:
        print("No matching kernel groups.", file=sys.stderr)
        return
    print(
        f"{'step':>8} {'bs':>5} {'past':>8} {'run':>4} {'pid':>12} "
        f"{'comp_us':>10} {'comm_us':>10} {'total_us':>10} "
        f"{'overlap_us':>11} {'visible_us':>11} {'visible%':>9} {'comm_k':>7}"
    )
    for row in rows:
        print(
            f"{row['step']:8d} {row['batch_size']:5d} {row['past_kv']:8d} {row['measure_run']:4d} "
            f"{str(row['pid']):>12} "
            f"{row['compute_union_us']:10.3f} {row['comm_union_us']:10.3f} "
            f"{row['total_union_us']:10.3f} {row['comm_compute_overlap_us']:11.3f} "
            f"{row['comm_visible_us']:11.3f} {row['comm_visible_pct']:8.1f}% "
            f"{row['comm_kernels']:7d}"
        )


_PER_RANK_FIELDS = [
    "step",
    "barrier_index",
    "rank",
    "arrival_ts",
    "launch_ts",
    "ar_start_ts",
    "ar_end_ts",
    "arrival_is_fallback",
    "spin",
]
_PER_BARRIER_FIELDS = [
    "step",
    "barrier_index",
    "n_ranks",
    "matched",
    "launch_spread",
    "arrival_spread",
    "transfer",
    "barrier_complete",
    "spin_mean",
    "spin_max",
]


def _write_csv(path: str | None, fieldnames: list[str], rows: list[dict]) -> None:
    fh = open(path, "w", newline="") if path else sys.stdout
    try:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if path:
            fh.close()


def _run_per_barrier(args) -> int:
    per_rank_rows, per_barrier_rows, meta = extract_per_barrier(
        args.sqlite_path,
        comm_re=re.compile(args.comm_regex, re.IGNORECASE),
        batch_size=args.batch_size,
        past_kv=args.past_kv,
        whole_trace=args.whole_trace,
    )

    # meta sanity to stderr
    print(
        f"[per-barrier] matched_barriers={meta['matched_barriers']}/{meta['barriers']} "
        f"arrival_fallbacks={meta['arrival_fallbacks']} "
        f"unmatched_barriers={meta['unmatched_barriers']} "
        f"per_rank_rows={meta['per_rank_rows']}",
        file=sys.stderr,
    )
    mismatch_steps = meta.get("comm_count_mismatch_steps") or []
    if mismatch_steps:
        print(
            f"[per-barrier] WARNING: per-rank comm-count mismatch in "
            f"{len(mismatch_steps)} step(s) (ordinal mis-alignment); affected "
            f"barriers are matched=False and excluded from spread aggregates. "
            f"steps={mismatch_steps}",
            file=sys.stderr,
        )

    # per-rank CSV defaults to stdout when no explicit path is given.
    _write_csv(args.per_rank_csv, _PER_RANK_FIELDS, per_rank_rows)
    if args.per_barrier_csv:
        _write_csv(args.per_barrier_csv, _PER_BARRIER_FIELDS, per_barrier_rows)
    else:
        # no separate barrier path: emit a delimiter + barrier table to stderr
        print("[per-barrier] per-barrier rows (no --per-barrier-csv given):", file=sys.stderr)
        _write_csv(None, _PER_BARRIER_FIELDS, per_barrier_rows)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_path")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--past-kv", type=int)
    parser.add_argument("--per-pid", action="store_true")
    parser.add_argument("--whole-trace", action="store_true")
    parser.add_argument("--comm-regex", default=_DEFAULT_COMM_RE.pattern)
    parser.add_argument("--format", choices=("table", "csv"), default="table")
    parser.add_argument(
        "--per-barrier",
        action="store_true",
        help="Per-barrier arrival-skew decomposition (writes per-rank + "
        "per-barrier CSVs) instead of the overlap summary.",
    )
    parser.add_argument(
        "--per-rank-csv",
        help="Output path for per-rank rows (--per-barrier mode); "
        "defaults to stdout.",
    )
    parser.add_argument(
        "--per-barrier-csv",
        help="Output path for per-barrier rows (--per-barrier mode); "
        "if omitted, printed to stdout after the per-rank CSV.",
    )
    args = parser.parse_args(argv)

    if args.per_barrier:
        return _run_per_barrier(args)

    rows, metadata = analyze_sqlite(
        args.sqlite_path,
        comm_re=re.compile(args.comm_regex, re.IGNORECASE),
        batch_size=args.batch_size,
        past_kv=args.past_kv,
        per_pid=args.per_pid,
        whole_trace=args.whole_trace,
    )
    if args.format == "csv":
        fieldnames = list(rows[0].keys()) if rows else [
            "step",
            "batch_size",
            "past_kv",
            "measure_run",
            "pid",
            "kernel_count",
            "compute_kernels",
            "comm_kernels",
            "compute_gpu_us",
            "comm_gpu_us",
            "compute_union_us",
            "comm_union_us",
            "total_union_us",
            "total_span_us",
            "comm_compute_overlap_us",
            "comm_visible_us",
            "comm_overlap_pct",
            "comm_visible_pct",
            "top_comm_kernels",
        ]
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    else:
        _print_table(rows, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
