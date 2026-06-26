"""TDD for the per-barrier extractor (decisive instrument, design.md §0).

Mirrors tests/test_layerwise_nsys_comm_overlap.py (import-from-path style) and
the aic_fpm_attribute `profiled_rows` test seam: feed SYNTHETIC pre-loaded
kernel rows via the `kernel_rows=` seam so no sqlite is needed.

Row seam shape (one dict per kernel):
  {step, pid, kernel_start, kernel_end, runtime_start, name}
  - step: logical step id (scalar or tuple)
  - pid: rank id
  - kernel_start/kernel_end: device kernel span (ns); kernel_start == ar_start,
    kernel_end == ar_end for comm kernels
  - runtime_start: host cudaLaunchKernel ts (ns) == launch_ts
  - name: kernel short name (classified comm vs compute by comm_re)
"""
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "collector" / "layerwise" / "diagnostics"))

from analyze_nsys_comm_overlap import extract_per_barrier  # noqa: E402

# _GLOBAL_PID_MASK = -16777216 clears the low 24 bits of a globalTid, so the
# high bits act as the process/rank id. Build rank-distinct globalTids by
# shifting (rank+1) into bits >= 24 and using a fixed thread id in the low bits.
_PID_MASK = -16777216


def _gtid(rank: int) -> int:
    return ((rank + 1) << 24) | 0x1234  # thread bits in the masked-off region


def _gpid(rank: int) -> int:
    return _gtid(rank) & _PID_MASK


def _build_sqlite(path, kernels, *, step_n=0, bs=8, past=128):
    """Construct the minimal nsys-sqlite tables that _load_kernel_rows reads.

    `kernels` is a list of dicts:
      {rank, name, runtime_start, kernel_start, kernel_end}
    where runtime_start is the HOST cudaLaunchKernel time (== launch_ts) and
    kernel_start/kernel_end are the DEVICE span (kernel_start == ar_start).
    These are intentionally allowed to differ and be skewed independently per
    rank, so a test can prove launch_ts is wired to runtime_start (host) and
    NOT to kernel_start (device).

    No CUDA_GRAPH_NODE_EVENTS table is created, so _query_kernels takes the
    eager (graphNodeId IS NULL) branch only.
    """
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(correlationId INTEGER, graphNodeId INTEGER, start INTEGER, end INTEGER, "
        " shortName INTEGER, globalPid INTEGER)"
    )
    cur.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
        "(correlationId INTEGER, globalTid INTEGER, start INTEGER, end INTEGER)"
    )
    cur.execute(
        "CREATE TABLE NVTX_EVENTS "
        "(text TEXT, start INTEGER, end INTEGER, globalTid INTEGER)"
    )

    # intern kernel names
    name_ids = {}
    for k in kernels:
        if k["name"] not in name_ids:
            sid = len(name_ids) + 1
            name_ids[k["name"]] = sid
            cur.execute("INSERT INTO StringIds (id, value) VALUES (?, ?)", (sid, k["name"]))

    ranks = sorted({k["rank"] for k in kernels})
    # one bench_step NVTX window per rank, wide enough to enclose every launch
    all_launch = [k["runtime_start"] for k in kernels]
    win_s = min(all_launch) - 10_000
    win_e = max(all_launch) + 10_000
    step_text = f"bench_step::N{step_n:07d}::bs{bs}::past{past:06d}::run0"
    for rank in ranks:
        cur.execute(
            "INSERT INTO NVTX_EVENTS (text, start, end, globalTid) VALUES (?, ?, ?, ?)",
            (step_text, win_s, win_e, _gtid(rank)),
        )

    cid = 1
    for k in kernels:
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL "
            "(correlationId, graphNodeId, start, end, shortName, globalPid) "
            "VALUES (?, NULL, ?, ?, ?, ?)",
            (cid, k["kernel_start"], k["kernel_end"], name_ids[k["name"]], _gpid(k["rank"])),
        )
        cur.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME "
            "(correlationId, globalTid, start, end) VALUES (?, ?, ?, ?)",
            (cid, _gtid(k["rank"]), k["runtime_start"], k["runtime_start"] + 5),
        )
        cid += 1
    con.commit()
    con.close()


def _rank_barrier_rows(
    *,
    step,
    n_ranks,
    n_barriers,
    delta,
    base=1_000_000,
    period=100_000,
    ar_dur=2_000,
    compute_dur=500,
    launch_lead=300,
):
    """Build synthetic kernels for n_ranks x n_barriers in one step.

    For barrier b at rank r:
      - a compute kernel that ENDS at T_b + r*delta (compute-finish skew).
      - an allreduce kernel launched at a SYNCHRONIZED host time (launch_spread
        == 0 by construction), starting once the slowest rank could arrive,
        running for ar_dur.
    """
    rows = []
    for b in range(n_barriers):
        t_b = base + b * period
        # allreduce device start: after the latest arrival, common to all ranks
        ar_start = t_b + (n_ranks - 1) * delta + 50
        launch_ts = ar_start - launch_lead  # synchronized launch across ranks
        for r in range(n_ranks):
            arrival = t_b + r * delta  # compute-finish edge for this rank
            rows.append({
                "step": step,
                "pid": r,
                "kernel_start": arrival - compute_dur,
                "kernel_end": arrival,
                "runtime_start": arrival - compute_dur - 10,
                "name": "sm90_gemm_compute",
            })
            rows.append({
                "step": step,
                "pid": r,
                "kernel_start": ar_start,
                "kernel_end": ar_start + ar_dur,
                "runtime_start": launch_ts,
                "name": "multimem_all_reduce_kernel",
            })
    return rows


def test_compute_finish_skew_decomposed_launch_zero():
    n_ranks, n_barriers, delta = 4, 3, 1_000
    rows = _rank_barrier_rows(step=7, n_ranks=n_ranks, n_barriers=n_barriers, delta=delta)
    per_rank_rows, per_barrier_rows, meta = extract_per_barrier(
        "UNUSED", kernel_rows=rows
    )

    # one logical barrier per (step, barrier_index)
    assert len(per_barrier_rows) == n_barriers
    for bar in per_barrier_rows:
        assert bar["step"] == 7
        assert bar["n_ranks"] == n_ranks
        assert bar["matched"] is True
        # compute-finish skew is (n_ranks-1)*delta; launch synchronized -> 0
        assert bar["arrival_spread"] == (n_ranks - 1) * delta
        assert bar["launch_spread"] == 0
        # transfer floor = min over ranks of (ar_end - ar_start)
        assert bar["transfer"] == 2_000

    # per-rank rows: 4 ranks x 3 barriers
    assert len(per_rank_rows) == n_ranks * n_barriers
    # spin strictly decreasing in r: latest-finishing rank waits least
    for bidx in range(n_barriers):
        spins = [
            row["spin"]
            for row in per_rank_rows
            if row["barrier_index"] == bidx
        ]
        spins_by_rank = [
            row["spin"]
            for r in range(n_ranks)
            for row in per_rank_rows
            if row["barrier_index"] == bidx and row["rank"] == r
        ]
        assert spins_by_rank == sorted(spins_by_rank, reverse=True)
        # strictly decreasing
        assert all(
            spins_by_rank[i] > spins_by_rank[i + 1]
            for i in range(len(spins_by_rank) - 1)
        )

    assert meta["unmatched_barriers"] == 0


def test_arrival_ts_is_preceding_compute_finish():
    rows = _rank_barrier_rows(step=1, n_ranks=4, n_barriers=1, delta=1_000)
    per_rank_rows, _per_barrier, _meta = extract_per_barrier("UNUSED", kernel_rows=rows)
    by_rank = {row["rank"]: row for row in per_rank_rows}
    # rank r arrival = base + r*delta
    for r in range(4):
        assert by_rank[r]["arrival_ts"] == 1_000_000 + r * 1_000
        # launch synchronized across ranks
        assert by_rank[r]["launch_ts"] == by_rank[0]["launch_ts"]


def test_unmatched_barrier_excluded_from_aggregates():
    rows = _rank_barrier_rows(step=2, n_ranks=4, n_barriers=2, delta=1_000)
    # drop ONE rank's comm kernel for barrier_index 1 -> only 3 ranks present
    dropped = [
        row
        for row in rows
        if not (
            row["pid"] == 3
            and row["name"] == "multimem_all_reduce_kernel"
            and row["kernel_start"] >= 1_100_000  # second barrier window
        )
    ]
    per_rank_rows, per_barrier_rows, meta = extract_per_barrier(
        "UNUSED", kernel_rows=dropped
    )
    by_bidx = {bar["barrier_index"]: bar for bar in per_barrier_rows}
    assert by_bidx[0]["matched"] is True
    assert by_bidx[0]["n_ranks"] == 4
    assert by_bidx[1]["matched"] is False
    assert by_bidx[1]["n_ranks"] == 3
    # unmatched barrier excluded from spread aggregates
    assert by_bidx[1]["arrival_spread"] is None
    assert by_bidx[1]["launch_spread"] is None
    assert meta["unmatched_barriers"] == 1


def test_arrival_falls_back_to_launch_when_no_preceding_compute():
    # a comm kernel with NO compute kernel before it within (step, pid)
    rows = [
        {
            "step": 5,
            "pid": 0,
            "kernel_start": 2_000,
            "kernel_end": 4_000,
            "runtime_start": 1_500,
            "name": "multimem_all_reduce_kernel",
        }
    ]
    per_rank_rows, _per_barrier, meta = extract_per_barrier("UNUSED", kernel_rows=rows)
    assert len(per_rank_rows) == 1
    row = per_rank_rows[0]
    assert row["arrival_ts"] == 1_500  # falls back to runtime_start (launch)
    assert row["arrival_is_fallback"] is True
    assert meta["arrival_fallbacks"] == 1


# --- NEW: caveat-closing tests that exercise the REAL sqlite path -----------


def test_launch_spread_tracks_host_launch_not_device_start(tmp_path):
    """launch_ts must come from runtime_start (host cudaLaunchKernel), NOT
    kernel_start (device). Build a step where, per rank, the host launch and the
    device start are skewed in OPPOSITE directions:

      - device allreduce starts are SYNCHRONIZED across ranks (all == AR_START),
        so a launch_ts==kernel_start wiring would give launch_spread == 0.
      - host launches are skewed by rank: launch_r = AR_START - r*LAUNCH_SKEW,
        so the true (host) launch_spread == (n_ranks-1)*LAUNCH_SKEW.

    Conversely arrival_ts is the preceding compute-finish edge, which we skew
    yet differently. This exercises _load_kernel_rows (real sqlite), not the
    kernel_rows= seam.

    FAILS if launch_ts is wired to kernel_start: launch_spread would be 0.
    """
    n_ranks = 4
    base = 1_000_000
    compute_dur = 500
    ar_dur = 2_000
    arrival_skew = 1_000   # compute-finish edge skew per rank
    launch_skew = 7_000    # host-launch skew per rank (independent magnitude)

    # device allreduce start: identical across ranks (synchronized device start)
    ar_start = base + (n_ranks - 1) * arrival_skew + 50

    kernels = []
    for r in range(n_ranks):
        arrival = base + r * arrival_skew  # compute-finish edge (device end)
        # compute kernel (device span ends at `arrival`)
        kernels.append({
            "rank": r,
            "name": "sm90_gemm_compute",
            "runtime_start": arrival - compute_dur - 10,
            "kernel_start": arrival - compute_dur,
            "kernel_end": arrival,
        })
        # allreduce: device start SYNCHRONIZED, host launch SKEWED by rank
        kernels.append({
            "rank": r,
            "name": "multimem_all_reduce_kernel",
            "runtime_start": ar_start - r * launch_skew,  # host launch skew
            "kernel_start": ar_start,                      # device start synced
            "kernel_end": ar_start + ar_dur,
        })

    db = tmp_path / "trace.sqlite"
    _build_sqlite(db, kernels)

    per_rank_rows, per_barrier_rows, meta = extract_per_barrier(str(db))

    assert len(per_barrier_rows) == 1
    bar = per_barrier_rows[0]
    assert bar["matched"] is True
    assert bar["n_ranks"] == n_ranks
    # launch_spread reflects the HOST runtime_start spread, not the (zero)
    # device kernel_start spread.
    assert bar["launch_spread"] == (n_ranks - 1) * launch_skew
    # device starts are synchronized: if launch_ts were kernel_start this == 0
    assert bar["launch_spread"] != 0
    # arrival_spread reflects the compute-finish edge skew (independent value)
    assert bar["arrival_spread"] == (n_ranks - 1) * arrival_skew

    # ar_start_ts is the device start; confirm it is synchronized across ranks
    ar_starts = {row["ar_start_ts"] for row in per_rank_rows}
    assert ar_starts == {ar_start}
    # and launch_ts is genuinely per-rank distinct (the host skew)
    launches = sorted(row["launch_ts"] for row in per_rank_rows)
    assert launches == sorted(ar_start - r * launch_skew for r in range(n_ranks))


def test_ordinal_mismatch_flags_unmatched_barrier_on_real_sqlite(tmp_path):
    """A step where ranks have DIFFERENT comm-kernel COUNTS (one rank is missing
    an allreduce) must NOT silently mis-align/average. The affected logical
    barrier is matched=False and counted in meta.unmatched_barriers, and meta
    records the per-rank comm-count mismatch so the mis-alignment is detectable.
    """
    n_ranks = 4
    base = 2_000_000
    period = 100_000
    compute_dur = 500
    ar_dur = 2_000

    kernels = []
    for r in range(n_ranks):
        # every rank gets barrier 0
        for b in range(2):
            # rank 3 is MISSING its second allreduce (ordinal mismatch)
            if b == 1 and r == 3:
                continue
            t_b = base + b * period
            arrival = t_b
            kernels.append({
                "rank": r,
                "name": "sm90_gemm_compute",
                "runtime_start": arrival - compute_dur - 10,
                "kernel_start": arrival - compute_dur,
                "kernel_end": arrival,
            })
            ar_start = t_b + 50
            kernels.append({
                "rank": r,
                "name": "multimem_all_reduce_kernel",
                "runtime_start": ar_start - 300,
                "kernel_start": ar_start,
                "kernel_end": ar_start + ar_dur,
            })

    db = tmp_path / "trace.sqlite"
    _build_sqlite(db, kernels)

    per_rank_rows, per_barrier_rows, meta = extract_per_barrier(str(db))

    by_bidx = {bar["barrier_index"]: bar for bar in per_barrier_rows}
    # barrier 0 has all ranks -> matched
    assert by_bidx[0]["matched"] is True
    assert by_bidx[0]["n_ranks"] == n_ranks
    # barrier 1 is missing rank 3 -> NOT matched, excluded from aggregates
    assert by_bidx[1]["matched"] is False
    assert by_bidx[1]["n_ranks"] == n_ranks - 1
    assert by_bidx[1]["arrival_spread"] is None
    assert by_bidx[1]["launch_spread"] is None
    assert meta["unmatched_barriers"] == 1

    # a detectable signal of the per-rank comm-count mismatch must exist so the
    # mis-alignment is caught, not silently averaged. `rank` in the rows is the
    # nsys masked globalPid (rank == pid), so map back through _gpid.
    counts = meta.get("comm_counts_per_rank")
    assert counts is not None
    per_rank = counts[by_bidx[1]["step_key"]]
    # rank 3 has one fewer comm kernel (only barrier 0) than the others (two)
    assert per_rank[_gpid(3)] == 1
    assert all(per_rank[_gpid(r)] == 2 for r in range(3))
    assert meta.get("comm_count_mismatch_steps")
