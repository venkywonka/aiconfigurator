#!/usr/bin/env python3
"""Multi-track AIC-predictor vs FPM gap analysis (SKU/version-general).

Compares AIC's predictor tracks against Dynamo/vLLM ForwardPassMetrics (FPM) ground truth,
per scheduler-step shape, for any (system, backend, version, model). All tracks run through
ONE per-step entry point (flipping ``vllm_backend._USE_LAYERWISE`` + swapping the ``database``):

    layerwise        layerwise CSV (compute-version) + comm tables (comm-version), cal-off headline
    layerwise_cal0066 same + the old 0.0066 decode batch-cal (counterfactual sensitivity)
    opwise_silicon   measured op-wise PerfDatabase (SILICON)
    empirical        analytic SOL/scale_factor (version-skew-immune)
    hybrid           SILICON-with-empirical-fallback
    sol              pure roofline floor

The layerwise track is version-matched to FPM (compute-version, e.g. 0.20.1); op-wise tracks
use the op-DB/comm-version (e.g. 0.19.0 where no compute-version comm dir exists for the SKU).
Auto-detects both FPM layouts: B300 TP-sweep (tp{tp}_ep1_past4096/) and H100-SXM concurrency
sweep (fpm/qwen32/c{conc}/, points merged; decode bins span batches 1..conc, grouped by batch).

Read-only: no GPU, no new collection, no committed-data mutation.

Usage (H100 dense Qwen3-32B):
    python -m collector.layerwise.diagnostics.aic_fpm_gap \
        --system h100_sxm --model Qwen/Qwen3-32B \
        --compute-version 0.20.1 --comm-version 0.19.0 --workload-segment real \
        --fpm-run fpm_golden_runs/fpm_h100_qwen32_tp8_8k1k_pareto_20260624_011804 \
        --out-dir <out>
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo wiring. This tool lives in-repo at collector/layerwise/diagnostics/, so __file__
# locates the repo root (parents[3] = <repo>/). --repo-root overrides.
# ---------------------------------------------------------------------------
DEFAULT_REPO_ROOT = str(Path(__file__).resolve().parents[3])

# Defaults (all overridable via CLI). The tool is SKU/version-general: it serves both the
# B300 TP-sweep and the H100-SXM concurrency-sweep FPM layouts (auto-detected; _resolve_fpm_source).
DEFAULT_MODEL = "Qwen/Qwen3-32B"
DEFAULT_SYSTEM = "h100_sxm"
DEFAULT_BACKEND = "vllm"
DEFAULT_COMPUTE_VERSION = "0.20.1"   # layerwise CSV version (version-matched to FPM)
DEFAULT_COMM_VERSION = "0.19.0"      # comm + op-wise PerfDatabase version (e.g. H100 has no 0.20.1 comm dir)
DEFAULT_FPM_RUN = "fpm_golden_runs/fpm_h100_qwen32_tp8_8k1k_pareto_20260624_011804"

# Read by build_model_and_db + the decode KV-snap; main() overrides it from --model.
MODEL_NAME = DEFAULT_MODEL
TP_VALUES = (8,)  # concurrency-sweep layouts are TP=8-only (points merged); TP-sweep layouts iterate this.


def make_tracks(compute_version: str, comm_version: str):
    """Track table: (name, use_layerwise, database_mode, version, is_measured_track, decode_cal_override).

    Layerwise tracks use the COMPUTE version (the layerwise CSV, version-matched to FPM); op-wise
    tracks use the COMM/op-DB version. decode_cal_override forces vllm_backend._DECODE_COMPUTE_BATCH_CAL
    (None = live default, 0.0 = disable). The headline 'layerwise' forces 0.0 (the linear decode
    batch-cal is mis-tuned); 'layerwise_cal0066' forces the old 0.0066 as a counterfactual sensitivity track.
    """
    return [
        ("layerwise", True, None, compute_version, True, 0.0),
        ("layerwise_cal0066", True, None, compute_version, True, 0.0066),
        ("opwise_silicon", False, "SILICON", comm_version, True, None),
        ("empirical", False, "EMPIRICAL", comm_version, False, None),
        ("hybrid", False, "HYBRID", comm_version, True, None),
        ("sol", False, "SOL", comm_version, False, None),
    ]

# Tracks excluded from the headline ranking/decomposition (best-effort or sensitivity).
NON_HEADLINE_TRACKS = {"layerwise_cal0066"}

INSUFFICIENT_COVERAGE_FRACTION = 0.60  # spec §7 honest escape hatch

# Status enum (spec §5 Stage 2)
ST_OK = "ok"
ST_MISSING = "missing_data"      # PerfDataNotAvailableError / KeyError
ST_OFFGRID = "offgrid_raise"     # AssertionError / ValueError (divisibility, off-grid)
ST_NOTIMPL = "not_implemented"   # NotImplementedError (e.g. wideep empirical)
ST_NOKV = "no_grid_kv"           # layerwise: no nearby collected decode KV
ST_NODB = "no_database"          # get_database returned None
ST_ERROR = "error"               # anything else (incl. non-positive pred/fpm)

VALID_STATUSES = [ST_OK, ST_MISSING, ST_OFFGRID, ST_NOTIMPL, ST_NOKV, ST_NODB, ST_ERROR]


# ---------------------------------------------------------------------------
# Reusable machinery (imported after sys.path wiring in main()).
# ---------------------------------------------------------------------------
def _import_repo(repo_root: Path):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from collector.layerwise.diagnostics.compare_aic_layerwise_fpm import (
        _LayerwiseDatabase,
        _aggregate,
        _effective_moe_parallelism,
        _load_fpm,
        _model_defaults,
        _nearest_available_generation_kv,
        _read_fpm_rows,
    )
    from aiconfigurator.sdk.backends import vllm_backend
    from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend
    from aiconfigurator.sdk.config import RuntimeConfig, ModelConfig
    from aiconfigurator.sdk import models, common
    from aiconfigurator.sdk.perf_database import (
        PerfDataNotAvailableError,
        PerfDatabase,
        get_database,
    )

    return {
        "_LayerwiseDatabase": _LayerwiseDatabase,
        "_aggregate": _aggregate,
        "_effective_moe_parallelism": _effective_moe_parallelism,
        "_load_fpm": _load_fpm,
        "_model_defaults": _model_defaults,
        "_nearest_available_generation_kv": _nearest_available_generation_kv,
        "_read_fpm_rows": _read_fpm_rows,
        "vllm_backend": vllm_backend,
        "VLLMBackend": VLLMBackend,
        "RuntimeConfig": RuntimeConfig,
        "ModelConfig": ModelConfig,
        "models": models,
        "common": common,
        "PerfDataNotAvailableError": PerfDataNotAvailableError,
        "PerfDatabase": PerfDatabase,
        "get_database": get_database,
    }


def _classify_source(sources: dict[str, str]) -> str:
    """Collapse a per-op source dict into one label."""
    vals = {str(v).lower() for v in sources.values()}
    if not vals:
        return "n/a"
    if vals <= {"silicon"}:
        return "silicon"
    if vals <= {"empirical"}:
        return "empirical"
    if vals <= {"sol"}:
        return "sol"
    if vals <= {"empirical", "sol", "estimated", "analytic"}:
        return "analytic"
    return "mixed"


def _status_for_exception(exc: BaseException, api) -> str:
    if isinstance(exc, api["PerfDataNotAvailableError"]):
        return ST_MISSING
    if isinstance(exc, KeyError):
        return ST_MISSING
    if isinstance(exc, NotImplementedError):
        return ST_NOTIMPL
    if isinstance(exc, (AssertionError, ValueError)):
        return ST_OFFGRID
    return ST_ERROR


def build_model_and_db(track_name, use_layerwise, mode, version, tp, *, system, backend, comm_version, systems_root, layerwise_csv, api):
    """Construct (model, database) for a track. Returns (model|None, db|None, err_status|None).

    layerwise track: diagnostic ``_Model`` + ``_LayerwiseDatabase`` (the layerwise
        backend path uses ``query_layerwise_detail``, not ``model.context_ops``).
    op-wise tracks: a REAL SDK model (``get_model``, which populates context_ops/
        generation_ops) + a ``PerfDatabase`` whose query mode is set via
        ``set_default_database_mode`` (the ``database_mode`` ctor arg only toggles the
        HYBRID shared-layer load, NOT the query mode — and SILICON/EMPIRICAL/SOL share
        one cached instance, so the mode must be set explicitly per track).
    """
    common = api["common"]
    try:
        if use_layerwise:
            model = api["_model_defaults"](MODEL_NAME, tp, 1, 1)
            real_db = api["PerfDatabase"](system, backend, comm_version, systems_root=systems_root)  # comm tables (may differ from the layerwise CSV's compute version, e.g. H100 0.19.0 comm + 0.20.1 compute)
            db = api["_LayerwiseDatabase"](Path(layerwise_csv), real_db)
            if MODEL_NAME.lower() not in db.layerwise:
                return None, None, ST_MISSING
            return model, db, None
        # op-wise tracks: real SDK model (dense Qwen3-32B, BF16 to match FPM).
        model_config = api["ModelConfig"](
            tp_size=tp,
            pp_size=1,
            gemm_quant_mode=common.GEMMQuantMode.bfloat16,
            kvcache_quant_mode=common.KVCacheQuantMode.bfloat16,
            fmha_quant_mode=common.FMHAQuantMode.bfloat16,
            moe_tp_size=1,
            moe_ep_size=1,
            attention_dp_size=1,
        )
        model = api["models"].get_model(MODEL_NAME, model_config, "vllm")
        db = api["PerfDatabase"](system, backend, version, systems_root=systems_root, database_mode=mode)
        db.set_default_database_mode(common.DatabaseMode[mode])
        return model, db, None
    except Exception as exc:  # noqa: BLE001 - mark, never silently drop
        return None, None, _status_for_exception(exc, api)


def predict_context(backend, model, database, rc, *, ctx_tokens, ctx_prefix_tokens, api):
    """Return (pred_ms|None, pred_source, status)."""
    try:
        latency, _, sources = backend._get_context_step_latency(
            model,
            database,
            rc,
            ctx_tokens=ctx_tokens,
            ctx_kv_tokens=ctx_prefix_tokens,  # batch=1
            ctx_requests=1,
        )
    except Exception as exc:  # noqa: BLE001 - mapped to status, never silently dropped
        return None, "n/a", _status_for_exception(exc, api)
    return float(sum(latency.values())), _classify_source(sources), ST_OK


def predict_decode(backend, model, database, rc, *, batch_size, past_kv, api):
    """Return (pred_ms|None, pred_source, status)."""
    try:
        latency, _, sources = backend._get_decode_step_latency(
            model,
            database,
            rc,
            batch_size=batch_size,
            past_kv=past_kv,
        )
    except Exception as exc:  # noqa: BLE001
        return None, "n/a", _status_for_exception(exc, api)
    return float(sum(latency.values())), _classify_source(sources), ST_OK


# Comm kernels in AIC's layerwise latency_dict are collective ops: TP allreduce and
# MoE expert all-to-all. Compute is the dense per-layer forward (`*_layerwise`).
# Everything else (scheduler overhead/residual) is "other". See public layerwise op
# naming in src/aiconfigurator/sdk/operations.
_AIC_COMM_KEY_RE = re.compile(r"(allreduce|alltoall|all_to_all|all_gather|reduce_scatter)")


def _split_latency(latency: dict[str, float]) -> tuple[float, float, float]:
    """Split an AIC layerwise latency_dict into (compute_ms, comm_ms, other_ms).

    compute + comm + other == sum(latency.values()) == predict_*()'s total, so the
    split is loss-free and the downstream decomposition identity stays exact.
    """
    compute = comm = other = 0.0
    for key, value in latency.items():
        val = float(value)
        if key.endswith("layerwise"):
            compute += val
        elif _AIC_COMM_KEY_RE.search(key):
            comm += val
        else:
            other += val
    return compute, comm, other


def predict_context_breakdown(backend, model, database, rc, *, ctx_tokens, ctx_prefix_tokens, api):
    """Like predict_context, but returns (compute_ms, comm_ms, total_ms, source, status)."""
    try:
        latency, _, sources = backend._get_context_step_latency(
            model,
            database,
            rc,
            ctx_tokens=ctx_tokens,
            ctx_kv_tokens=ctx_prefix_tokens,  # batch=1
            ctx_requests=1,
        )
    except Exception as exc:  # noqa: BLE001 - mapped to status, never silently dropped
        return None, None, None, "n/a", _status_for_exception(exc, api)
    compute, comm, other = _split_latency(latency)
    return compute, comm, compute + comm + other, _classify_source(sources), ST_OK


def predict_decode_breakdown(backend, model, database, rc, *, batch_size, past_kv, api):
    """Like predict_decode, but returns (compute_ms, comm_ms, total_ms, source, status)."""
    try:
        latency, _, sources = backend._get_decode_step_latency(
            model,
            database,
            rc,
            batch_size=batch_size,
            past_kv=past_kv,
        )
    except Exception as exc:  # noqa: BLE001
        return None, None, None, "n/a", _status_for_exception(exc, api)
    compute, comm, other = _split_latency(latency)
    return compute, comm, compute + comm + other, _classify_source(sources), ST_OK


# ---------------------------------------------------------------------------
# Noise floor (spec Stage 0): per-shape CV from repeated FPM steps.
# ---------------------------------------------------------------------------
def _r(x, n=5):
    """Round numerics; pass empties through as ''."""
    return round(x, n) if isinstance(x, (int, float)) else ""


def _cv(samples: list[float]) -> float | None:
    if len(samples) < 2:
        return None
    mean = statistics.fmean(samples)
    if mean <= 0:
        return None
    return statistics.pstdev(samples) / mean


PATHOLOGICAL_CV = 0.5  # a per-step CV above 50% is contamination (warmup/compile spike), not noise


def _load_fpm_mixed(read_fpm_rows, fpm_csv, workload_segment):
    """Bin mixed (chunked-prefill + decode) FPM steps by step shape.

    Best-effort phase per spec §3/§8: kept out of the headline. Key mirrors the
    mixed estimator args: (ctx_tokens, ctx_requests, prefix, gen_tokens, isl).
    """
    bins: dict[tuple, list[float]] = defaultdict(list)
    for row in read_fpm_rows(fpm_csv, workload_segment=workload_segment):
        if row.get("phase") != "mixed":
            continue
        try:
            ctx_tokens = int(row["ctx_tokens"])
            ctx_requests = int(row["ctx_requests"])
            ctx_kv = int(float(row.get("ctx_kv_tokens") or 0))
            gen_tokens = int(row["decode_requests"])
            isl = round(float(row["mean_decode_kv_tokens"]))
            latency = float(row["latency_ms"])
        except (KeyError, ValueError, TypeError):
            continue
        prefix = round(ctx_kv / max(ctx_requests, 1))
        bins[(ctx_tokens, ctx_requests, prefix, gen_tokens, isl)].append(latency)
    return bins


def predict_mixed(backend, model, database, rc, *, ctx_tokens, ctx_requests, prefix, gen_tokens, isl, api):
    """Return (pred_ms|None, pred_source, status). Mixed entry point returns ms directly."""
    try:
        aic_ms, _, per_ops, per_ops_source = backend._get_mix_step_latency(
            model, database, rc,
            ctx_tokens=ctx_tokens, gen_tokens=gen_tokens,
            isl=isl, osl=1, prefix=prefix, ctx_requests=ctx_requests,
        )
    except Exception as exc:  # noqa: BLE001
        return None, "n/a", _status_for_exception(exc, api)
    return float(aic_ms), _classify_source(per_ops_source), ST_OK


def compute_noise_floor(bins_by_phase):
    """Per-phase measurement-noise stats from repeated FPM steps.

    Shapes whose CV exceeds PATHOLOGICAL_CV are excluded from the noise band
    (and counted separately) so one warmup-contaminated shape cannot inflate the
    'within-noise' bar for the whole group.
    """
    out = {}
    for phase, bins in bins_by_phase.items():
        cvs, n_path = [], 0
        for samples in bins.values():
            cv = _cv(samples)
            if cv is None:
                continue
            if cv > PATHOLOGICAL_CV:
                n_path += 1
                continue
            cvs.append(cv)
        out[phase] = {
            "n_shapes": len(bins),
            "n_with_repeats": len(cvs) + n_path,
            "n_pathological_variance": n_path,
            "median_cv": (statistics.median(cvs) if cvs else None),
            "p90_cv": (_percentile(cvs, 90) if cvs else None),
        }
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


# ---------------------------------------------------------------------------
# FPM layout adapter (H100 concurrency sweep vs B300 TP sweep).
# ---------------------------------------------------------------------------
def _merge_fpm_phase_csvs(paths: list[Path], out_path: Path) -> Path:
    """Concatenate per-concurrency fpm_metrics_phase.csv files (identical schema) into one.

    The H100 FPM run is a concurrency sweep (fpm/qwen32/c{1,4,16,64,128}/) at fixed TP=8, vs
    B300's TP sweep (tp{tp}_ep1_past4096/). Per-step (batch, mean_kv) binning is concurrency-
    agnostic and build_summary_by_concurrency groups decode by batch (= concurrency), so merging
    all points into one csv lets the single TP=8 iteration see the full batch range. Header once.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = None
    with open(out_path, "w", newline="") as out:
        for p in paths:
            if not p.exists():
                continue
            with open(p, newline="") as f:
                lines = f.readlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
                out.write(header)
            elif lines[0] != header:
                raise ValueError(f"FPM schema mismatch in {p}")
            out.writelines(lines[1:])
    return out_path


def _resolve_fpm_source(fpm_run: Path, tp: int, out_dir: Path):
    """Return (fpm_csv_path, runtime_config_subdir) for one TP, handling both FPM layouts.

    H100: merge fpm/<model>/c*/fpm_metrics_phase.csv -> one csv (config from the first c-dir).
    B300: tp{tp}_ep1_past4096/fpm_metrics_phase.csv directly.

    The concurrency-layout model dir is discovered, not hardcoded: it is whatever single dir
    under fpm/ holds c*/fpm_metrics_phase.csv (e.g. 'qwen32' for the golden runs, or an
    autocollector model slug like 'Qwen-Qwen3-32B'). Hardcoding 'qwen32' silently produced a
    zero-row gap for any other model.
    """
    fpm_base = fpm_run / "fpm"
    conc_base = None
    if fpm_base.is_dir():
        model_dirs = sorted(
            d for d in fpm_base.iterdir()
            if d.is_dir() and any((c / "fpm_metrics_phase.csv").exists() for c in d.glob("c*"))
        )
        if model_dirs:
            conc_base = model_dirs[0]
    cdirs = ([d for d in conc_base.glob("c*") if (d / "fpm_metrics_phase.csv").exists()]
             if conc_base is not None else [])
    cdirs.sort(key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else 0)
    if cdirs:
        merged = _merge_fpm_phase_csvs([d / "fpm_metrics_phase.csv" for d in cdirs],
                                       out_dir / "merged_fpm_metrics_phase.csv")
        print(f"[fpm] concurrency layout ({conc_base.name}): merged {[d.name for d in cdirs]} (tp={tp})",
              file=sys.stderr)
        return merged, cdirs[0]
    subdir = fpm_run / f"tp{tp}_ep1_past4096"
    return subdir / "fpm_metrics_phase.csv", subdir


# ---------------------------------------------------------------------------
# Core driver.
# ---------------------------------------------------------------------------
def run(repo_root: Path, fpm_run: Path, out_dir: Path, *, system: str, backend_name: str,
        compute_version: str, comm_version: str, tracks: list, layerwise_csv: Path | None = None,
        aggregation: str, workload_segment: str, include_mixed: bool = True):
    # NOTE: `backend_name` is the backend STRING ("vllm"); the local `backend` below is the
    # VLLMBackend INSTANCE used for prediction — keep them distinct (do not rename to `backend`).
    api = _import_repo(repo_root)
    systems_root = str(repo_root / "src" / "aiconfigurator" / "systems")
    if layerwise_csv is None:
        layerwise_csv = (repo_root / "src" / "aiconfigurator" / "systems" / "data"
                         / system / backend_name / compute_version / "layerwise_perf.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []        # per (track, tp, phase, shape)
    noise_rows: list[dict[str, Any]] = []  # per (tp, phase)
    noise_by_tp_phase: dict[tuple[int, str], dict] = {}

    for tp in TP_VALUES:
        # FPM-FIXED: resolve the FPM source for this TP, handling H100 concurrency layout
        # (fpm/qwen32/c{conc}, merged) vs B300 TP layout (tp{tp}_ep1_past4096). decode bins are
        # keyed by (batch, mean_kv) and build_summary_by_concurrency groups by batch=concurrency.
        fpm_csv, subdir = _resolve_fpm_source(fpm_run, tp, out_dir)
        if not Path(fpm_csv).exists():
            print(f"[warn] missing FPM csv: {fpm_csv}", file=sys.stderr)
            continue
        rc_kwargs = _read_runtime_config(subdir)

        context, decode, _filtered = api["_load_fpm"](fpm_csv, workload_segment=workload_segment)
        # Restrict context to single-request prefill steps (the entry point's
        # well-exercised regime; matches the existing diagnostic).
        context = {k: v for k, v in context.items() if k[0] == 1}
        mixed = (_load_fpm_mixed(api["_read_fpm_rows"], fpm_csv, workload_segment)
                 if include_mixed else {})

        bins_by_phase = {"ctx": context, "gen": decode}
        if include_mixed:
            bins_by_phase["mixed"] = mixed
        nf = compute_noise_floor(bins_by_phase)
        for phase, stats in nf.items():
            noise_by_tp_phase[(tp, phase)] = stats
            noise_rows.append({"tp": tp, "phase": phase, **stats})

        backend = api["VLLMBackend"]()

        old_flag = api["vllm_backend"]._USE_LAYERWISE
        old_cal = api["vllm_backend"]._DECODE_COMPUTE_BATCH_CAL
        try:
            for (track_name, use_lw, mode, version, is_measured, cal_override) in tracks:
                api["vllm_backend"]._USE_LAYERWISE = use_lw
                # Force the decode batch-calibration for sensitivity tracks
                # (layerwise_nocal sets it to 0.0); else keep the live default.
                eff_cal = old_cal if cal_override is None else cal_override
                api["vllm_backend"]._DECODE_COMPUTE_BATCH_CAL = eff_cal
                # Per-track RuntimeConfig: the layerwise GEN data has no
                # max_num_seqs index for this model, so force None to use the
                # primary index; op-wise uses the FPM-measured value.
                rc = api["RuntimeConfig"](
                    vllm_max_num_batched_tokens=rc_kwargs["vllm_max_num_batched_tokens"],
                    vllm_max_num_seqs=(None if use_lw else rc_kwargs["vllm_max_num_seqs"]),
                )
                model, database, db_err = build_model_and_db(
                    track_name, use_lw, mode, version, tp,
                    system=system, backend=backend_name, comm_version=comm_version,
                    systems_root=systems_root, layerwise_csv=layerwise_csv, api=api,
                )

                # --- context shapes ---
                for (ctx_requests, ctx_tokens, ctx_prefix), samples in sorted(context.items()):
                    fpm_ms = api["_aggregate"](samples, aggregation)
                    shape_cv = _cv(samples)
                    base = _row_base(track_name, version, tp, "ctx",
                                     shape=f"ctx{ctx_tokens}" + (f"_prefix{ctx_prefix}" if ctx_prefix else ""),
                                     fpm_ms=fpm_ms, samples=samples,
                                     extra={"ctx_tokens": ctx_tokens, "ctx_prefix_tokens": ctx_prefix,
                                            "batch_size": "", "decode_kv": "", "aic_past_kv": "",
                                            "kv_snap_delta": "", "shape_cv": _r(shape_cv),
                                            "match_type": "exact", "representative_kv": ""})
                    if database is None:
                        rows.append(_mark(base, db_err, noise_by_tp_phase))
                        continue
                    pred, src, status = predict_context(
                        backend, model, database, rc,
                        ctx_tokens=ctx_tokens, ctx_prefix_tokens=ctx_prefix, api=api)
                    rows.append(_finish(base, pred, src, status, is_measured, noise_by_tp_phase))

                # --- decode shapes (iterate FPM (batch, mean_kv) bins directly) ---
                for (batch_size, mean_kv), samples in sorted(decode.items()):
                    fpm_ms = api["_aggregate"](samples, aggregation)
                    shape_cv = _cv(samples)
                    req_kv = round(mean_kv)
                    aic_past_kv = req_kv
                    kv_snap_delta = 0
                    if use_lw:
                        snapped = api["_nearest_available_generation_kv"](
                            database.layerwise, model=MODEL_NAME, tp_size=tp,
                            requested_kv=req_kv, max_distance=float("inf"),
                        ) if database is not None else None
                        if snapped is None and database is not None:
                            base = _row_base(track_name, version, tp, "gen",
                                             shape=f"bs{batch_size}_kv{req_kv}", fpm_ms=fpm_ms, samples=samples,
                                             extra={"ctx_tokens": "", "ctx_prefix_tokens": "",
                                                    "batch_size": batch_size, "decode_kv": req_kv,
                                                    "aic_past_kv": "", "kv_snap_delta": "", "shape_cv": _r(shape_cv),
                                                    "match_type": "no_grid_kv", "representative_kv": req_kv})
                            rows.append(_mark(base, ST_NOKV, noise_by_tp_phase))
                            continue
                        if snapped is not None:
                            aic_past_kv = snapped
                            kv_snap_delta = snapped - req_kv
                    base = _row_base(track_name, version, tp, "gen",
                                     shape=f"bs{batch_size}_kv{req_kv}", fpm_ms=fpm_ms, samples=samples,
                                     extra={"ctx_tokens": "", "ctx_prefix_tokens": "",
                                            "batch_size": batch_size, "decode_kv": req_kv,
                                            "aic_past_kv": aic_past_kv, "kv_snap_delta": kv_snap_delta,
                                            "shape_cv": _r(shape_cv), "representative_kv": req_kv,
                                            "match_type": ("exact" if (not use_lw or kv_snap_delta == 0) else "snapped"),
                                            # linear decode batch-cal factor applied (layerwise only)
                                            "decode_batch_cal": (round(1.0 + eff_cal * batch_size, 4) if use_lw else "")})
                    if database is None:
                        rows.append(_mark(base, db_err, noise_by_tp_phase))
                        continue
                    pred, src, status = predict_decode(
                        backend, model, database, rc,
                        batch_size=batch_size, past_kv=aic_past_kv, api=api)
                    rows.append(_finish(base, pred, src, status, is_measured, noise_by_tp_phase))

                # --- mixed shapes (best-effort; excluded from the headline) ---
                for (ctx_tokens, ctx_requests, prefix, gen_tokens, isl), samples in sorted(mixed.items()):
                    fpm_ms = api["_aggregate"](samples, aggregation)
                    shape_cv = _cv(samples)
                    base = _row_base(track_name, version, tp, "mixed",
                                     shape=f"mix_ctx{ctx_tokens}_req{ctx_requests}_gen{gen_tokens}_kv{isl}",
                                     fpm_ms=fpm_ms, samples=samples,
                                     extra={"ctx_tokens": ctx_tokens, "ctx_prefix_tokens": prefix,
                                            "batch_size": gen_tokens, "decode_kv": isl, "aic_past_kv": "",
                                            "kv_snap_delta": "", "shape_cv": _r(shape_cv),
                                            "match_type": "exact", "representative_kv": isl})
                    if database is None:
                        rows.append(_mark(base, db_err, noise_by_tp_phase))
                        continue
                    pred, src, status = predict_mixed(
                        backend, model, database, rc,
                        ctx_tokens=ctx_tokens, ctx_requests=ctx_requests, prefix=prefix,
                        gen_tokens=gen_tokens, isl=isl, api=api)
                    rows.append(_finish(base, pred, src, status, is_measured, noise_by_tp_phase))
        finally:
            api["vllm_backend"]._USE_LAYERWISE = old_flag
            api["vllm_backend"]._DECODE_COMPUTE_BATCH_CAL = old_cal

    summary = build_summary(rows)
    by_concurrency = build_summary_by_concurrency(rows)
    coverage = build_coverage(rows)
    decomposition = build_decomposition(summary)

    _write_csv(out_dir / "gap_rows.csv", rows)
    _write_csv(out_dir / "fpm_noise_floor.csv", noise_rows)
    _write_csv(out_dir / "coverage.csv", coverage)
    _write_csv(out_dir / "gap_summary.csv", summary)
    _write_csv(out_dir / "gap_summary_by_concurrency.csv", by_concurrency)
    _write_csv(out_dir / "compute_comm_decomposition.csv", decomposition)
    return {
        "rows": rows, "noise_rows": noise_rows, "coverage": coverage,
        "summary": summary, "by_concurrency": by_concurrency, "decomposition": decomposition,
        "noise_by_tp_phase": noise_by_tp_phase,
    }


def _read_runtime_config(subdir: Path) -> dict:
    """Pull max_num_batched_tokens / max_num_seqs from the FPM effective config."""
    import json
    cfg_path = subdir / "effective_vllm_config.json"
    mnbt, mns = 2048, 128
    try:
        cfg = json.loads(cfg_path.read_text())
        mnbt = int(cfg.get("max_num_batched_tokens") or cfg.get("scheduler_config", {}).get("max_num_batched_tokens") or mnbt)
        mns = int(cfg.get("max_num_seqs") or cfg.get("scheduler_config", {}).get("max_num_seqs") or mns)
    except Exception:  # noqa: BLE001 - fall back to known FPM-run constants
        pass
    return {"vllm_max_num_batched_tokens": mnbt, "vllm_max_num_seqs": mns}


def _row_base(track, version, tp, phase, *, shape, fpm_ms, samples, extra):
    base = {
        "track": track, "version": version, "tp": tp, "phase": phase, "shape": shape,
        "fpm_ms": round(fpm_ms, 5), "fpm_samples": len(samples),
        "pred_ms": "", "rel_err": "", "signed_log_err": "",
        "pred_source": "", "status": "", "is_measured_track": "", "within_noise": "",
        # §9 fairness: match_type/representative_kv (decode), comm_modeled flags
        # layerwise TP>1 (single-GPU modeled comm add-back, not measured).
        "match_type": "", "representative_kv": "",
        "comm_modeled": (str(track).startswith("layerwise") and tp > 1),
        # linear decode batch-calibration factor (filled for layerwise gen rows)
        "decode_batch_cal": "",
    }
    base.update(extra)
    return base


def _mark(base, status, noise_by_tp_phase):
    base["status"] = status
    return base


def _finish(base, pred, src, status, is_measured, noise_by_tp_phase):
    base["pred_source"] = src
    base["is_measured_track"] = is_measured
    if status != ST_OK:
        base["status"] = status
        return base
    # Non-positive prediction or FPM latency is not a valid datapoint — mark it
    # rather than emit an ST_OK row with asymmetric/blank metrics.
    fpm_ms = float(base["fpm_ms"]) if base["fpm_ms"] not in ("", None) else 0.0
    if pred is None or pred <= 0 or fpm_ms <= 0:
        base["status"] = ST_ERROR
        return base
    rel = (pred / fpm_ms) - 1.0
    base["status"] = ST_OK
    base["pred_ms"] = round(pred, 5)
    base["rel_err"] = round(rel, 6)
    base["signed_log_err"] = round(math.log(pred / fpm_ms), 6)
    nf = noise_by_tp_phase.get((base["tp"], base["phase"]), {})
    p90 = nf.get("p90_cv")
    base["within_noise"] = (abs(rel) <= p90) if p90 is not None else ""
    return base


# ---------------------------------------------------------------------------
# Aggregation: summary, coverage, decomposition.
# ---------------------------------------------------------------------------
def build_summary(rows):
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["track"], r["tp"], r["phase"])].append(r)
    summary = []
    for (track, tp, phase), grp in sorted(groups.items()):
        oks = [r for r in grp if r["status"] == ST_OK and r["rel_err"] != ""]
        errs = [float(r["rel_err"]) for r in oks]
        abs_errs = [abs(e) for e in errs]
        # Denominator = every FPM bin this track attempted (ok + marked). Each
        # track attempts the same bin set, so this is consistent across tracks
        # and never exceeds 1.0.
        n_total = len(grp)
        n_ok = len(oks)
        within = [r for r in oks if r["within_noise"] is True]
        nonsilicon = [r for r in oks if r["pred_source"] not in ("silicon", "")]
        summary.append({
            "track": track, "tp": tp, "phase": phase,
            "n_fpm_shapes": n_total, "n_ok": n_ok,
            "coverage_frac": round(n_ok / n_total, 4) if n_total else 0.0,
            "sufficient_coverage": (n_ok / n_total >= INSUFFICIENT_COVERAGE_FRACTION) if n_total else False,
            "mape_pct": round(100 * statistics.fmean(abs_errs), 3) if abs_errs else "",
            "median_err_pct": round(100 * statistics.median(errs), 3) if errs else "",
            "p90_abs_err_pct": round(100 * _percentile(abs_errs, 90), 3) if abs_errs else "",
            "max_abs_err_pct": round(100 * max(abs_errs), 3) if abs_errs else "",
            "signed_bias_pct": round(100 * statistics.median(errs), 3) if errs else "",
            "within_noise_frac": round(len(within) / n_ok, 4) if n_ok else "",
            "nonsilicon_frac": round(len(nonsilicon) / n_ok, 4) if n_ok else "",
        })
    return summary


def build_summary_by_concurrency(rows):
    """MAPE/bias grouped by (track, tp, phase, concurrency).

    Concurrency = number of concurrent requests in the step: decode batch size
    (`batch_size` = decode_requests) for gen/mixed; 1 for ctx (single-request
    prefill grid). Only ok rows contribute; multiple KV shapes at the same
    concurrency are aggregated.
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["status"] != ST_OK or r["rel_err"] == "":
            continue
        bs = r.get("batch_size")
        conc = int(bs) if bs not in ("", None) else 1
        groups[(r["track"], r["tp"], r["phase"], conc)].append(float(r["rel_err"]))
    out = []
    for (track, tp, phase, conc), errs in sorted(groups.items()):
        abs_errs = [abs(e) for e in errs]
        out.append({
            "track": track, "tp": tp, "phase": phase, "concurrency": conc,
            "n_ok": len(errs),
            "mape_pct": round(100 * statistics.fmean(abs_errs), 3),
            "median_err_pct": round(100 * statistics.median(errs), 3),
            "signed_bias_pct": round(100 * statistics.median(errs), 3),
            "p90_abs_err_pct": round(100 * _percentile(abs_errs, 90), 3),
        })
    return out


def build_coverage(rows):
    cov: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cov[(r["track"], r["tp"], r["phase"])][r["status"]] += 1
    out = []
    statuses = VALID_STATUSES
    for (track, tp, phase), counts in sorted(cov.items()):
        total = sum(counts.values())
        row = {"track": track, "tp": tp, "phase": phase, "n_total": total}
        for s in statuses:
            row[s] = counts.get(s, 0)
        row["balanced"] = (sum(counts.get(s, 0) for s in statuses) == total)
        out.append(row)
    return out


def build_decomposition(summary):
    """compute_err = bias@TP1; comm_err(tp) = bias@tp - bias@TP1, per (track, phase).

    Headline only (ctx, gen): mixed is excluded — its TP1 anchor is not a clean
    compute-only reference.
    """
    by_track_phase: dict[tuple, dict[int, float]] = defaultdict(dict)
    for s in summary:
        if s["phase"] not in ("ctx", "gen"):
            continue
        if s["signed_bias_pct"] != "":
            by_track_phase[(s["track"], s["phase"])][s["tp"]] = float(s["signed_bias_pct"])
    out = []
    for (track, phase), bias_by_tp in sorted(by_track_phase.items()):
        bias1 = bias_by_tp.get(1)
        row = {"track": track, "phase": phase,
               "compute_err_pct_tp1": (round(bias1, 3) if bias1 is not None else "")}
        for tp in (2, 4, 8):
            b = bias_by_tp.get(tp)
            row[f"comm_err_pct_tp{tp}"] = (round(b - bias1, 3) if (b is not None and bias1 is not None) else "")
            row[f"bias_pct_tp{tp}"] = (round(b, 3) if b is not None else "")
        out.append(row)
    return out


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    # union of keys (rows may differ slightly)
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model name (e.g. Qwen/Qwen3-32B).")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="systems-data SKU dir (e.g. h100_sxm, b300_sxm).")
    p.add_argument("--backend", default=DEFAULT_BACKEND, help="backend dir (vllm).")
    p.add_argument("--compute-version", default=DEFAULT_COMPUTE_VERSION,
                   help="layerwise CSV version (version-matched to FPM, e.g. 0.20.1).")
    p.add_argument("--comm-version", default=DEFAULT_COMM_VERSION,
                   help="comm + op-wise PerfDatabase version (e.g. 0.19.0 where no compute-version comm dir exists).")
    p.add_argument("--layerwise-csv", default=None,
                   help="explicit layerwise CSV path (default: data/<system>/<backend>/<compute-version>/layerwise_perf.csv).")
    p.add_argument("--fpm-run", default=None, help="FPM golden run dir (auto-detects TP-sweep vs concurrency-sweep layout).")
    p.add_argument("--out-dir", default=None, help="Output dir (default: <tool>/out).")
    p.add_argument("--aggregation", default="median", choices=["median", "mean", "trimmed_mean"])
    p.add_argument("--workload-segment", default="sweep",
                   help="FPM workload segment ('sweep' static grid [B300], 'real' [H100 concurrency sweep], or 'all').")
    p.add_argument("--no-mixed", action="store_true", help="Skip the best-effort mixed phase.")
    p.add_argument("--no-html", action="store_true")
    return p.parse_args()


def main():
    global MODEL_NAME
    args = _parse_args()
    MODEL_NAME = args.model  # read by build_model_and_db + the decode KV-snap
    repo_root = Path(args.repo_root).resolve()
    fpm_run = Path(args.fpm_run) if args.fpm_run else (repo_root / DEFAULT_FPM_RUN)
    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / "out")
    layerwise_csv = Path(args.layerwise_csv) if args.layerwise_csv else None
    tracks = make_tracks(args.compute_version, args.comm_version)

    result = run(repo_root, fpm_run, out_dir,
                 system=args.system, backend_name=args.backend,
                 compute_version=args.compute_version, comm_version=args.comm_version,
                 tracks=tracks, layerwise_csv=layerwise_csv,
                 aggregation=args.aggregation, workload_segment=args.workload_segment,
                 include_mixed=not args.no_mixed)

    if not args.no_html:
        try:
            from aic_fpm_gap_report import write_report  # local sibling module
        except Exception:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from aic_fpm_gap_report import write_report
        write_report(out_dir, result, fpm_run=fpm_run, aggregation=args.aggregation,
                     workload_segment=args.workload_segment,
                     system=args.system, model=args.model,
                     compute_version=args.compute_version, comm_version=args.comm_version)

    # console summary
    print(f"\nWrote outputs to {out_dir}")
    print("(headline = ctx+gen; mixed is best-effort and in the CSVs only)")
    print(f"{'track':16} {'tp':>2} {'phase':5} {'n_ok':>5} {'cov':>5} {'MAPE%':>7} {'bias%':>7} {'wn%':>5}")
    for s in result["summary"]:
        if s["phase"] not in ("ctx", "gen"):
            continue
        print(f"{s['track']:16} {s['tp']:>2} {s['phase']:5} {s['n_ok']:>5} "
              f"{s['coverage_frac']:>5} {str(s['mape_pct']):>7} {str(s['signed_bias_pct']):>7} "
              f"{str(s['within_noise_frac']):>5}")


if __name__ == "__main__":
    main()
