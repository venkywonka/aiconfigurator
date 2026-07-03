#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# reproduce_layerwise_fpm.sh
# =============================================================================
# One-shot driver to reproduce the full layerwise <-> FPM-ground-truth <-> align
# experiment, top to bottom, on arbitrary hardware. Defaults are tuned for the
# DENSE Qwen3-32B, TP=8 scenario on H100 (8xH100 for real FPM ground-truth,
# 1xH100 for single-GPU TP-mock layerwise collection), mirroring the committed
# B300 setup but with the current ("latest") code paths.
#
# Three stages (each independently selectable via STAGES=):
#   layerwise : 1xGPU single-GPU TP-mock per-layer compute timing (vLLM container)
#               -> <OUT_ROOT>/layerwise/<slug>/layerwise.csv
#   fpm       : 8xGPU real Dynamo/vLLM deployment, ForwardPassMetrics ground truth,
#               one run per [low,mid,high] pareto concurrency point
#               -> <OUT_ROOT>/fpm/<slug>/<pareto>/.../fpm_metrics_phase.csv
#   align     : tools/plot_fpm_vs_aic.py, one chart set per pareto point
#               -> <OUT_ROOT>/charts/<slug>/<pareto>/fpm_vs_aic_*.png
#               ALSO runs the profile figures below (so they arrive with align).
#   profile   : collector/layerwise/diagnostics/plot_fpm_distributions.py, one
#               data-profile figure set per pareto point (batch-composition +
#               param distributions + latency scatters). Pure CSV postprocessing
#               (no GPU/AIC); fail-safe. -> <OUT_ROOT>/charts/<slug>/<pareto>/
#               fpm_profile/fpm_distribution_*.png
#
# Default order is "layerwise fpm align" (fast stage first so a setup bug fails
# in minutes, not after the ~hours-long FPM sweep); align triggers profile too.
# Reorder via STAGES=; select "profile" alone to (re)plot only the distributions.
#
# DESIGN: fail-fast (set -euo pipefail). Idempotent: each unit drops a marker in
# <OUT_ROOT>/.done/ and is skipped on re-run unless FORCE=1 -> fix the failure,
# re-run, and completed work is skipped. DRY_RUN=1 prints every command without
# executing (the no-GPU verification path). SMOKE=1 runs a tiny end-to-end probe.
#
# BAKED-IN CORRECTNESS (do not "fix" these):
#   * NO --live-step-driver (deprecated; corrupts MoE decode). Decode uses the
#     default execute_model_gpu source.
#   * NSYS 2026.3.1 mounted read-only into the vLLM container.
#   * FPM --gpus quoted / inferred from TP; vLLM cache kept separate from $HOME.
#   * OUT_ROOT must be LOCAL ext4 -- nsys export to SMB/NFS dies "database is locked".
#
# H100-SPECIFIC CAVEATS (see slop/h100-dense-repro-driver/):
#   * AIC systems-data for H100 is vllm/0.19.0 (there is NO 0.20.1 dir). The align
#     stage points PerfDatabase at DATA_VERSION=0.19.0 while collection uses vLLM
#     0.20.1 -- comm is modeled from 0.19.0 tables, compute measured on 0.20.1.
#   * h100_sxm/vllm/0.19.0 has custom_allreduce_perf.parquet but NOT
#     allreduce_rms_perf.parquet -> the vLLM backend falls back to custom_allreduce
#     for the fused decode allreduce. Correct & expected; only the "custom vs fused"
#     comparison chart degrades to custom-only.
#   * The FPM shell HARD-FAILS if the image's vLLM != VLLM_VERSION. Either the image
#     genuinely ships that version, or set ALLOW_VERSION_MISMATCH=1.
#
# Requires patches (already applied on this branch):
#   M1/M2  tools/plot_fpm_vs_aic.py: --system/--backend/--version + --fpm-run-name
#   M3     collector/layerwise/fpm/{collect,docker}.py: --allow-version-mismatch passthrough
#
# USAGE
#   DRY_RUN=1 ./collector/layerwise/reproduce_layerwise_fpm.sh        # print all commands, no GPU
#   SMOKE=1   ./collector/layerwise/reproduce_layerwise_fpm.sh        # tiny end-to-end probe
#             ./collector/layerwise/reproduce_layerwise_fpm.sh        # full dense Qwen3-32B tp8 H100
#   STAGES="align" FORCE=1 ./...reproduce_layerwise_fpm.sh            # re-plot only
#   STAGES="layerwise" ./...reproduce_layerwise_fpm.sh               # just the 1xH100 collection
#
# DECODE-ONLY CONCURRENCY SWEEP ARM (design.md v3 §5 PRIMARY clean testbed)
#   Capture the clean decode sweep C in {1,4,16,64,128} at TP=8, fixed past_kv=4096,
#   under nsys windowed capture + per-rank decompose:
#
#     STAGES="fpm attribute" \
#     ATTRIBUTE_PHASES="decode" ATTRIBUTE_REAL_WORKLOAD=0 \
#     DECODE_BATCH_SIZES="1,4,16,64,128" DECODE_PAST_KV=4096 \
#     ATTRIBUTE_PER_PID=1 \
#     TP=8 FPM_TP_LIST=8 PARETO_CONCURRENCY="1" \
#     ./collector/layerwise/reproduce_layerwise_fpm.sh
#
#   The static decode sweep sends each DECODE_BATCH_SIZES value as BOTH the request
#   count and the concurrency (so the value IS the resident decode population for that
#   point); the whole {1,4,16,64,128} ladder runs inside ONE deployment, so pin
#   PARETO_CONCURRENCY to a single value (one pareto point) to avoid re-running the
#   identical sweep per pareto name. The windowed nsys/per-rank capture
#   applies to every decode point in the sweep.
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# Repo / paths
# ----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIC_REPO="${AIC_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT_ROOT="${OUT_ROOT:-/tmp/aic-h100-repro}"   # MUST be local ext4 (nsys lock trap on SMB/NFS)

# ----------------------------------------------------------------------------
# Mode toggles
# ----------------------------------------------------------------------------
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
STAGES="${STAGES:-layerwise fpm align}"
PREFLIGHT_IMAGE_CHECK="${PREFLIGHT_IMAGE_CHECK:-0}"   # 1 = docker-run the FPM image to verify vLLM version

# ----------------------------------------------------------------------------
# Hardware / systems-data identity
# ----------------------------------------------------------------------------
SYSTEM="${SYSTEM:-h100_sxm}"            # AIC systems-data SKU dir (comm/compute tables) used by align
BACKEND="${BACKEND:-vllm}"
DATA_VERSION="${DATA_VERSION:-0.19.0}"  # systems-data version dir for align; H100 has 0.19.0, NOT 0.20.1
VLLM_VERSION="${VLLM_VERSION:-0.20.1}"  # actual vLLM used for collection (CSV label + image gate)
MODEL_REVISION="${MODEL_REVISION:-main}" # exact Hugging Face revision used by the FPM deployment

# ----------------------------------------------------------------------------
# Containers
# ----------------------------------------------------------------------------
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.20.1}"                          # layerwise
DYNAMO_VLLM_IMAGE="${DYNAMO_VLLM_IMAGE:-nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0}"  # FPM
ALLOW_VERSION_MISMATCH="${ALLOW_VERSION_MISMATCH:-0}"

# ----------------------------------------------------------------------------
# Nsight Systems (host install mounted read-only into the vLLM container)
# ----------------------------------------------------------------------------
NSYS_VERSION_DIR="${NSYS_VERSION_DIR:-2026.3.1}"
NSYS_ROOT="${NSYS_ROOT:-$HOME/.local/opt/nsight-systems-cli-${NSYS_VERSION_DIR}/opt/nvidia/nsight-systems-cli/${NSYS_VERSION_DIR}}"
AIC_NODE_ARCH="${AIC_NODE_ARCH:-}"
NSYS_TARGET_HOST_DIR="${NSYS_TARGET_HOST_DIR:-}"
NSYS_IMPORTER_HOST_DIR="${NSYS_IMPORTER_HOST_DIR:-}"
NSYS_TARGET_CONTAINER_DIR=""
NSYS_IMPORTER_CONTAINER_DIR=""
NSYS_BIN_CONTAINER_DIR=""

# ----------------------------------------------------------------------------
# Caches / auth
# ----------------------------------------------------------------------------
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
HF_TOKEN="${HF_TOKEN:-}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-}"
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-$HOME/.cache/aic-vllm}"   # kept separate from $HOME/.cache/vllm

# ----------------------------------------------------------------------------
# Scenario: parallelism + pareto + shapes
# ----------------------------------------------------------------------------
TP="${TP:-8}"
EP="${EP:-1}"
LW_TP_LIST="${LW_TP_LIST:-$TP}"          # layerwise TP sizes (single-GPU mock); default = focus TP
FPM_TP_LIST="${FPM_TP_LIST:-$TP}"        # FPM real-deployment TP sizes
DECODE_PAST_KV="${DECODE_PAST_KV:-4096}" # -> FPM subdir tp{T}_ep{E}_past4096 (matches PRIMARY_CASES)
DECODE_OSL="${DECODE_OSL:-}"             # static decode-only arm: override decode OSL to SUSTAIN the target in-flight
                                         # batch (empty = collect.py default 8). With OSL=8 a "batch 64" run peaks at
                                         # in-flight bs~37 (requests finish before 64 accumulate) -> no bs=64 steps to
                                         # capture. Long OSL (e.g. 512) keeps all N requests resident -> batch reaches N.

# [low, mid, high] throughput-latency pareto = real-workload concurrency sweep.
PARETO_NAMES=(low mid high)
PARETO_CONCURRENCY=(${PARETO_CONCURRENCY:-8 16 32})
FPM_REQUESTS="${FPM_REQUESTS:-128}"     # legacy fixed default; superseded by P2 per-point scaling below
# P2: per-point request scaling -> requests(conc) = clamp(FPM_REQ_MULT*conc, FPM_REQ_MIN, FPM_REQ_MAX).
# At OSL=1024 a fixed high request count makes low-concurrency points run for tens of minutes; scaling
# keeps each point ~1-3 min while reaching steady state. {1,4,16,64,128} -> {6,8,32,128,256}.
FPM_REQ_MULT="${FPM_REQ_MULT:-2}"
FPM_REQ_MIN="${FPM_REQ_MIN:-6}"
FPM_REQ_MAX="${FPM_REQ_MAX:-256}"

# Large-ish ISL/OSL (FPM real-workload shape distribution).
ISL_MIN="${ISL_MIN:-100}";  ISL_MAX="${ISL_MAX:-16384}";  ISL_MEAN="${ISL_MEAN:-4096}"
OSL_MIN="${OSL_MIN:-100}";  OSL_MAX="${OSL_MAX:-4096}";   OSL_MEAN="${OSL_MEAN:-1024}"
FPM_DATASET="${FPM_DATASET:-OpenAssistant/oasst1}"
FPM_SHAPE_SOURCE="${FPM_SHAPE_SOURCE:-scaled_dataset}"
FPM_WARMUP_REQUESTS="${FPM_WARMUP_REQUESTS:-4}"

# Attribute stage: windowed nsys capture (cudaProfilerStart/Stop) + decomposition.
# ATTRIBUTE_WINDOW is "lo-hi[,lo-hi...]" step ordinals (the marker's NVTX bench_step::N
# label step) passed through to --nsys-cuda-profiler-window; ATTRIBUTE_DISCARD_N drops
# the first N sync-drained boundary steps of each cohort before reducing.
ATTRIBUTE_WINDOW="${ATTRIBUTE_WINDOW:-100-115}"
# BLOCKER-2 FIX (task #41 aws-dfw): full-worker nsys capture so CUPTI records real
# GPU kernels -> real decomposition gpu_*_ms. Windowed cudaProfilerApi capture yielded
# 0 CUPTI kernel rows (assert_attribution_valid failed). Default ON; set 0 to restore
# the legacy windowed capture. Post-hoc step-window slicing is done in analysis.
ATTRIBUTE_FULL_WORKER="${ATTRIBUTE_FULL_WORKER:-1}"
ATTRIBUTE_DISCARD_N="${ATTRIBUTE_DISCARD_N:-3}"
# ATTRIBUTE_PER_PID=1 passes --per-pid to aic_fpm_attribute so the decomposition CSV
# carries accurate per-rank rows (pid column) ALONGSIDE the cross-rank aggregate
# (per design.md v3 §0.2: per-rank variance is preserved for the arrival-skew
# decomposition). Set to 0 for the aggregate-only legacy behavior.
ATTRIBUTE_PER_PID="${ATTRIBUTE_PER_PID:-1}"

# Attribute-stage workload selection (design.md v3 §5 arms). Defaults reproduce the
# real-workload high-C mixed-step capture. The DECODE-ONLY SWEEP ARM (PRIMARY clean
# testbed) is selected by ATTRIBUTE_PHASES=decode + ATTRIBUTE_REAL_WORKLOAD=0, which
# drives the static decode sweep over DECODE_BATCH_SIZES (= concurrency ladder) at a
# fixed DECODE_PAST_KV. See the "Decode-only sweep arm" usage note in the header.
ATTRIBUTE_PHASES="${ATTRIBUTE_PHASES:-context,decode,mixed}"
ATTRIBUTE_REAL_WORKLOAD="${ATTRIBUTE_REAL_WORKLOAD:-1}"
# DECODE_BATCH_SIZES: static-sweep decode batch sizes; each value is BOTH the request
# count and the concurrency for that decode point (collect_fpm_metrics.sh send_sweep
# decode loop). For the v3 decode concurrency sweep set "1,4,16,64,128".
DECODE_BATCH_SIZES="${DECODE_BATCH_SIZES:-1,4,16,64,128}"

# Scheduler parity: forced via env so FPM shell + align agree (shell reads $MAX_NUM_SEQS).
FPM_MAX_NUM_SEQS="${FPM_MAX_NUM_SEQS:-256}"
FPM_MAX_NUM_BATCHED_TOKENS="${FPM_MAX_NUM_BATCHED_TOKENS:-2048}"
# Opt-in: enable chunked prefill on the FPM/context worker (both the clean FPM run and the
# nsys attribute capture) so context steps become uniform C-token chunks (C =
# FPM_MAX_NUM_BATCHED_TOKENS). Off by default -> the decode lane is unchanged. When on, the
# layerwise ctx grid also gains an exact new_tokens=C anchor (appended below) so the C-chunk
# query lands on a grid point instead of interpolating (the AIC CTX lookup interpolates
# within-grid but is exact at grid points, and raises when C is outside the grid range).
FPM_ENABLE_CHUNKED_PREFILL="${FPM_ENABLE_CHUNKED_PREFILL:-0}"

# Layerwise shapes (single-GPU TP-mock).
LW_PHASES="${LW_PHASES:-both}"
LW_CTX_NEW_TOKENS="${LW_CTX_NEW_TOKENS:-1,16,128,1024,4096}"
# Empty by default -> the collector uses the run-preset ctx_past_kv (full: 0,16,...,32768;
# smoke: singleton [0]). The chunked-prefill block below seeds {0,C} when unset so a
# continuation chunk (past_kv=C) is bracketed instead of hitting a singleton kv axis.
LW_CTX_PAST_KV="${LW_CTX_PAST_KV:-}"
LW_GEN_BATCH_SIZES="${LW_GEN_BATCH_SIZES:-1,2,4,8,16,32,64}"
LW_GEN_PAST_KV="${LW_GEN_PAST_KV:-1,4096,8192,16384,32768}"
LW_MAX_DECODE_BATCH_SIZE="${LW_MAX_DECODE_BATCH_SIZE:-256}"
LW_MAX_MODEL_LEN="${LW_MAX_MODEL_LEN:-40960}"   # >= max(LW_GEN_PAST_KV)+ctx margin
LW_GPU_MEM_UTIL="${LW_GPU_MEM_UTIL:-0.9}"
LW_GPUS="${LW_GPUS:-0}"                          # single GPU id for the TP-mock
LW_LATENCY_SOURCE="${LW_LATENCY_SOURCE:-auto}"  # context=schedule_to_update; decode=execute_model_gpu (predictor component basis)

# Align / plot.
PLOT_PHASES="${PLOT_PHASES:-ctx,gen,mixed,allreduce}"
PLOT_PARETO="${PLOT_PARETO:-${PARETO_NAMES[*]}}" # which pareto points to plot (default all)

# Quant legs (dense bf16; nvfp4 is Blackwell-only and intentionally excluded).
GEMM_QUANT="${GEMM_QUANT:-bf16}"; ATTN_QUANT="${ATTN_QUANT:-bf16}"
KV_QUANT="${KV_QUANT:-bf16}";     MOE_QUANT="${MOE_QUANT:-bf16}"

# ----------------------------------------------------------------------------
# Model matrix: "slug | hf_id | kind | moe_perf_file(optional, relative to repo)"
# Edit this array to add models. Dense models leave moe_perf_file empty.
# ----------------------------------------------------------------------------
MODELS=(
  "qwen32|Qwen/Qwen3-32B|dense|"
)

# SMOKE: tiny, fast, end-to-end plumbing probe.
if [[ "$SMOKE" == "1" ]]; then
  MODELS=("qwen0p6b|Qwen/Qwen3-0.6B|dense|")
  LW_TP_LIST="1"; FPM_TP_LIST="1"; TP="1"
  LW_PHASES="both"; LW_CTX_NEW_TOKENS="1,128"; LW_GEN_BATCH_SIZES="1,4"; LW_GEN_PAST_KV="1,4096"
  LW_MAX_MODEL_LEN="8192"; LW_MAX_DECODE_BATCH_SIZE="8"
  PARETO_NAMES=(c2); PARETO_CONCURRENCY=(2); FPM_REQ_MIN="2"; FPM_REQ_MAX="8"; FPM_WARMUP_REQUESTS="1"
  FPM_MAX_NUM_SEQS="8"; FPM_MAX_NUM_BATCHED_TOKENS="40960"; PLOT_PARETO="c2"
  LW_RUN_PRESET="smoke"
fi

# Env override: a caller-supplied MODEL (e.g. the autocollector's AIC_LAYERWISE_MODELS)
# wins over the hardcoded/smoke default so the model is actually selectable. SMOKE
# still scales the shapes; only the model identity is overridden here. Optional
# MODEL_SLUG / MODEL_KIND / MOE_PERF_FILE refine the single-entry matrix.
if [[ -n "${MODEL:-}" ]]; then
  _model_slug="${MODEL_SLUG:-${MODEL//\//-}}"
  MODELS=("${_model_slug}|${MODEL}|${MODEL_KIND:-dense}|${MOE_PERF_FILE:-}")
fi
LW_RUN_PRESET="${LW_RUN_PRESET:-full}"

# When chunked prefill is opted in, the FPM/context worker emits uniform C-token chunks
# (C = FPM_MAX_NUM_BATCHED_TOKENS) and the attribution filter selects context steps at
# ctx_tokens==C. The AIC CTX layerwise lookup keys on BOTH new_tokens AND past_kv: it is
# exact at grid points and 2-D linearly interpolates within the collected hull, raising
# only when a coordinate falls OUTSIDE the axis min/max range (or when the kv axis is a
# singleton, e.g. smoke's [0]). Appending new_tokens=C gives an exact anchor at the
# dominant chunk shape (avoiding interpolation error there) and guarantees coverage when
# C exceeds the current grid max. A continuation chunk also lands at past_kv=C, so the ctx
# past_kv grid must contain C (and 0 for the first chunk); when LW_CTX_PAST_KV is unset the
# smoke preset collapses to a singleton [0] kv axis, which would make the (C, C) query raise.
# We therefore also seed {0,C} on the past_kv axis when it is unset. All appends preserve
# existing shapes (append, not replace). Runs after the SMOKE/MODEL/preset blocks so their
# overrides win.
if [[ "$FPM_ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  case ",${LW_CTX_NEW_TOKENS}," in
    *",${FPM_MAX_NUM_BATCHED_TOKENS},"*) : ;;
    *) LW_CTX_NEW_TOKENS="${LW_CTX_NEW_TOKENS},${FPM_MAX_NUM_BATCHED_TOKENS}" ;;
  esac
  if [[ -z "${LW_CTX_PAST_KV}" ]]; then
    # Unset -> collector would use the preset default. Seed {0,C} so the past_kv axis is
    # non-singleton and brackets the continuation-chunk prefix=C exactly (fixes smoke's [0]).
    LW_CTX_PAST_KV="0,${FPM_MAX_NUM_BATCHED_TOKENS}"
  else
    case ",${LW_CTX_PAST_KV}," in
      *",${FPM_MAX_NUM_BATCHED_TOKENS},"*) : ;;
      *) LW_CTX_PAST_KV="${LW_CTX_PAST_KV},${FPM_MAX_NUM_BATCHED_TOKENS}" ;;
    esac
  fi
fi

# P1: normalize pareto point names to the concurrency ladder length (auto-derive c<conc>) so
# PARETO_CONCURRENCY can define any number of points. Previously PARETO_NAMES=(low mid high) was a
# literal and the fpm/align loops iterate ${!PARETO_NAMES[@]}, silently dropping points beyond 3.
if [[ "${#PARETO_NAMES[@]}" -ne "${#PARETO_CONCURRENCY[@]}" ]]; then
  PARETO_NAMES=()
  for _c in "${PARETO_CONCURRENCY[@]}"; do PARETO_NAMES+=("c${_c}"); done
  PLOT_PARETO="${PLOT_PARETO_OVERRIDE:-${PARETO_NAMES[*]}}"
fi

# ============================================================================
# Helpers
# ============================================================================
RUN_TS="${RUN_TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_DIR="$OUT_ROOT/logs"; DONE_DIR="$OUT_ROOT/.done"
C_BOLD=$'\033[1m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'; C_GRN=$'\033[32m'; C_YEL=$'\033[33m'; C_OFF=$'\033[0m'

log()  { printf '%s[driver]%s %s\n' "$C_BOLD" "$C_OFF" "$*"; }
warn() { printf '%s[driver WARN]%s %s\n' "$C_YEL" "$C_OFF" "$*" >&2; }
err()  { printf '%s[driver ERROR]%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; }
die()  { err "$*"; exit 1; }

layerwise_needs_nsight() {
  case "$LW_LATENCY_SOURCE" in
    auto|schedule_to_update|worker_wall|execute_model_gpu) return 1;;
    *) return 0;;
  esac
}

resolve_nsight_architecture() {
  local stage requires_nsight=0
  for stage in $STAGES; do
    if [[ "$stage" == "attribute" ]] || \
       { [[ "$stage" == "layerwise" ]] && layerwise_needs_nsight; }; then
      requires_nsight=1
    fi
  done
  [[ "$requires_nsight" == "1" ]] || return 0

  local observed expected layout_output
  observed="$(uname -m)" || die "failed to read allocated node architecture"
  expected="${AIC_NODE_ARCH:-$observed}"
  if ! layout_output="$(
    python3 "$SCRIPT_DIR/common/nsight_arch.py" \
      --expected "$expected" --observed "$observed" 2>&1
  )"; then
    die "Nsight architecture validation failed: $layout_output"
  fi
  local -a layout=()
  mapfile -t layout <<< "$layout_output"
  [[ "${#layout[@]}" == "2" && -n "${layout[0]}" && -n "${layout[1]}" ]] || \
    die "Nsight architecture resolver returned an invalid layout"

  AIC_NODE_ARCH="$expected"
  export AIC_NODE_ARCH
  NSYS_TARGET_HOST_DIR="${NSYS_TARGET_HOST_DIR:-$NSYS_ROOT/${layout[0]}}"
  NSYS_IMPORTER_HOST_DIR="${NSYS_IMPORTER_HOST_DIR:-$NSYS_ROOT/${layout[1]}}"
  NSYS_TARGET_CONTAINER_DIR="/opt/nvidia/nsight-systems/${NSYS_VERSION_DIR}/${layout[0]}"
  NSYS_IMPORTER_CONTAINER_DIR="/opt/nvidia/nsight-systems/${NSYS_VERSION_DIR}/${layout[1]}"
  NSYS_BIN_CONTAINER_DIR="/opt/nvidia/nsight-systems/${NSYS_VERSION_DIR}/bin"

  if [[ "$DRY_RUN" != "1" ]]; then
    [[ -d "$NSYS_TARGET_HOST_DIR" ]] || \
      die "Nsight target directory missing: $NSYS_TARGET_HOST_DIR"
    [[ -d "$NSYS_IMPORTER_HOST_DIR" ]] || \
      die "Nsight importer directory missing: $NSYS_IMPORTER_HOST_DIR"
    [[ -x "$NSYS_ROOT/bin/nsys" ]] || \
      die "Nsight nsys executable missing: $NSYS_ROOT/bin/nsys"
  fi
}

# run <logfile> <cmd...> : echo, then exec (or just echo under DRY_RUN), tee to log.
run() {
  local logf="$1"; shift
  printf '%s+ %s%s\n' "$C_DIM" "$*" "$C_OFF"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  mkdir -p "$(dirname "$logf")"
  local restore_errexit=0
  [[ $- == *e* ]] && { restore_errexit=1; set +e; }
  ( "$@" ) 2>&1 | tee "$logf"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  (( restore_errexit == 1 )) && set -e
  if (( pipeline_status[0] != 0 )); then
    return "${pipeline_status[0]}"
  fi
  return "${pipeline_status[1]}"
}

# run_env <logfile> VAR=val VAR2=val2 -- <cmd...> : like run but with extra env.
run_env() {
  local logf="$1"; shift
  local -a env_args=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    [[ "$1" == *=* ]] || die "run_env expected an environment assignment, got: $1"
    env_args+=("$1")
    shift
  done
  [[ $# -gt 0 && "$1" == "--" ]] || die "run_env missing -- command separator"
  shift
  [[ $# -gt 0 ]] || die "run_env missing command"
  printf '%s+ env' "$C_DIM"
  printf ' %s' "${env_args[@]}" "$@"
  printf '%s\n' "$C_OFF"
  if [[ "$DRY_RUN" == "1" ]]; then return 0; fi
  mkdir -p "$(dirname "$logf")"
  local restore_errexit=0
  [[ $- == *e* ]] && { restore_errexit=1; set +e; }
  ( env -- "${env_args[@]}" "$@" ) 2>&1 | tee "$logf"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  (( restore_errexit == 1 )) && set -e
  if (( pipeline_status[0] != 0 )); then
    return "${pipeline_status[0]}"
  fi
  return "${pipeline_status[1]}"
}

# The shared runtime backend normally calls run <cmd...>.  This outer driver
# has a log-file-first run helper, so inject the equivalent command invoker.
RUNTIME_LOG_FILE=""
runtime_invoke() {
  if [[ -z "${RUNTIME_LOG_FILE:-}" ]]; then
    err "runtime command log path is not configured"
    return 64
  fi
  run "$RUNTIME_LOG_FILE" "$@"
}

# shellcheck source=fpm_ground_truth/runtime.sh
source "$SCRIPT_DIR/fpm_ground_truth/runtime.sh"

reset_outer_runtime_state() {
  unset ENROOT_STATE_DIR ENROOT_PIDDIR ENROOT_CACHE_PATH ENROOT_TEMP_PATH \
    ENROOT_RUNTIME_PATH ENROOT_CONTAINER_NAME _ENROOT_SHARED_CONTAINER
  if [[ -z "${ENROOT_IMAGE_MAP:-}" ]]; then
    unset ENROOT_DATA_PATH
  fi
}

runtime_image_version() (
  local IMAGE="$1"
  local RUN_DIR="$OUT_ROOT/.runtime/image-version"
  local NAME_PREFIX="aic-image-version-${BASHPID}"
  local RUNTIME_LOG_FILE="$LOG_DIR/image-version.log"
  local -a version_runtime_opts=(--network none)
  if [[ "${RUNTIME}" == "enroot" ]]; then
    version_runtime_opts=(--network host)
  fi
  reset_outer_runtime_state
  trap 'runtime_teardown >/dev/null 2>&1 || true' EXIT
  runtime_prepare || return 1
  runtime_run_oneshot "" "" version_runtime_opts -- \
    python3 -c 'import vllm,sys; sys.stdout.write(vllm.__version__)'
)

done_marker() { echo "$DONE_DIR/$1.done"; }
is_done()     { [[ "$FORCE" != "1" && -f "$(done_marker "$1")" ]]; }
mark_done()   { [[ "$DRY_RUN" == "1" ]] || { mkdir -p "$DONE_DIR"; date -u +%Y-%m-%dT%H:%M:%SZ > "$(done_marker "$1")"; }; }

hf_token_value() {
  if [[ -n "$HF_TOKEN" ]]; then echo "$HF_TOKEN"; return; fi
  if [[ -n "$HF_TOKEN_FILE" && -f "$HF_TOKEN_FILE" ]]; then cat "$HF_TOKEN_FILE"; return; fi
  [[ -f "$HOME/hf.token" ]] && { cat "$HOME/hf.token"; return; }
  echo ""
}

PIPELINE_DUMMY_ENV=()
METADATA_MODEL_CONTAINER=""
apply_pipeline_policy() {
  if [[ "${AIC_MODEL_MODE:-}" != "metadata_dummy" ]]; then return; fi

  local resolved_model
  if ! resolved_model="$(
    python3 "$SCRIPT_DIR/common/pipeline_policy.py" \
      --model "${MODEL:-}" \
      --load-format "${LOAD_FORMAT:-}"
  )"; then
    die "metadata dummy policy validation failed"
  fi

  local metadata_hf_home
  mkdir -p "$OUT_ROOT"
  if ! metadata_hf_home="$(mktemp -d "$OUT_ROOT/metadata-hf-cache.XXXXXX")"; then
    die "failed to create metadata dummy cache"
  fi
  chmod 700 "$metadata_hf_home"

  MODEL="$resolved_model"
  AIC_MODEL_METADATA_DIR="$resolved_model"
  AIC_LOAD_FORMAT=dummy
  FPM_REAL_WORKLOAD_SHAPE_SOURCE=synthetic
  FPM_SHAPE_SOURCE=synthetic
  LOAD_FORMAT=dummy
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
  HF_HOME="$metadata_hf_home"
  HF_TOKEN=""
  HF_TOKEN_FILE=""
  METADATA_MODEL_CONTAINER=/aic-model-metadata
  MODELS=("${MODEL_SLUG:-metadata}|${MODEL}|${MODEL_KIND:-dense}|${MOE_PERF_FILE:-}")
  PIPELINE_DUMMY_ENV=(
    "AIC_LOAD_FORMAT=dummy"
    "AIC_MODEL_METADATA_DIR=${MODEL}"
    "AIC_MODEL_MODE=metadata_dummy"
    "FPM_REAL_WORKLOAD_SHAPE_SOURCE=synthetic"
    "HF_DATASETS_OFFLINE=1"
    "HF_HUB_OFFLINE=1"
    "TRANSFORMERS_OFFLINE=1"
  )
  export \
    AIC_LOAD_FORMAT \
    AIC_MODEL_METADATA_DIR \
    AIC_MODEL_MODE \
    FPM_REAL_WORKLOAD_SHAPE_SOURCE \
    HF_DATASETS_OFFLINE \
    HF_HUB_OFFLINE \
    TRANSFORMERS_OFFLINE
}

# ============================================================================
# Preflight
# ============================================================================
preflight() {
  log "Preflight (SYSTEM=$SYSTEM backend=$BACKEND data_version=$DATA_VERSION vllm=$VLLM_VERSION; SMOKE=$SMOKE DRY_RUN=$DRY_RUN)"
  mkdir -p "$OUT_ROOT" "$LOG_DIR" "$DONE_DIR" "$HF_HOME" "$VLLM_CACHE_HOST/tilelang/tmp"

  # GPU count
  if command -v nvidia-smi >/dev/null 2>&1; then
    local ngpu
    if ngpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"; then
      log "Visible GPUs: $ngpu"
      [[ " $STAGES " == *" fpm "* && "$ngpu" -lt "$TP" ]] && warn "fpm stage needs >= $TP GPUs (TP=$TP) but only $ngpu visible."
      [[ " $STAGES " == *" layerwise "* && "$ngpu" -lt 1 ]] && warn "layerwise stage needs >= 1 GPU."
    else
      warn "nvidia-smi is installed but unavailable -- cannot verify GPU count."
    fi
  else
    warn "nvidia-smi not found -- cannot verify GPU count."
  fi

  # OUT_ROOT filesystem type (nsys 'database is locked' on SMB/NFS/fuse)
  local fstype; fstype="$(stat -f -c %T "$OUT_ROOT" 2>/dev/null || echo unknown)"
  case "$fstype" in
    ext2/ext3|ext4|xfs|btrfs|tmpfs) log "OUT_ROOT=$OUT_ROOT fstype=$fstype (local OK)";;
    *) warn "OUT_ROOT=$OUT_ROOT fstype=$fstype -- nsys export may fail with 'database is locked'. Use local ext4.";;
  esac

  # NSYS host dirs (attribute or an explicitly nsys-backed layerwise source)
  if [[ " $STAGES " == *" attribute "* ]] || \
     { [[ " $STAGES " == *" layerwise "* ]] && layerwise_needs_nsight; }; then
    [[ -d "$NSYS_TARGET_HOST_DIR" ]]   || warn "NSYS target dir missing: $NSYS_TARGET_HOST_DIR (set NSYS_ROOT/NSYS_TARGET_HOST_DIR)."
    [[ -d "$NSYS_IMPORTER_HOST_DIR" ]] || warn "NSYS importer dir missing: $NSYS_IMPORTER_HOST_DIR."
  fi

  # HF token (fpm needs to pull/serve gated models)
  if [[ "${AIC_MODEL_MODE:-}" != "metadata_dummy" && " $STAGES " == *" fpm "* && -z "$(hf_token_value)" ]]; then
    warn "No HF token (HF_TOKEN / HF_TOKEN_FILE / ~/hf.token). FPM model serve may fail for gated models."
  fi

  # Align needs systems-data for the target SKU/version
  if [[ " $STAGES " == *" align "* ]]; then
    local dd="$AIC_REPO/src/aiconfigurator/systems/data/$SYSTEM/$BACKEND/$DATA_VERSION"
    if [[ -d "$dd" ]]; then
      log "Align systems-data: $dd"
      [[ -f "$dd/custom_allreduce_perf.parquet" ]] || warn "no custom_allreduce_perf.parquet in $dd -- comm term will be unavailable."
      [[ -f "$dd/allreduce_rms_perf.parquet" ]]    || warn "no allreduce_rms_perf.parquet in $dd -- fused decode allreduce falls back to custom_allreduce (expected on H100 0.19.0)."
    else
      die "Align systems-data dir not found: $dd (set SYSTEM/BACKEND/DATA_VERSION). H100 ships 0.19.0, not 0.20.1."
    fi
  fi

  # Optional: verify the FPM image's vLLM version (the shell hard-fails on mismatch)
  if [[ " $STAGES " == *" fpm "* ]]; then
    if [[ "$PREFLIGHT_IMAGE_CHECK" == "1" && "$DRY_RUN" != "1" ]]; then
      local v version_output
      if version_output="$(runtime_image_version "$DYNAMO_VLLM_IMAGE" 2>/dev/null)"; then
        v="${version_output##*$'\n'}"
      else
        v="?"
      fi
      log "FPM image vLLM version: $v (expected $VLLM_VERSION)"
      if [[ "$v" != "$VLLM_VERSION" && "$ALLOW_VERSION_MISMATCH" != "1" ]]; then
        die "FPM image vLLM=$v != $VLLM_VERSION and ALLOW_VERSION_MISMATCH!=1. The FPM shell will die. Set ALLOW_VERSION_MISMATCH=1 or use a matching image."
      fi
    else
      warn "FPM image vLLM version unverified. The FPM shell HARD-FAILS if image vLLM != $VLLM_VERSION. Set PREFLIGHT_IMAGE_CHECK=1, or ALLOW_VERSION_MISMATCH=1 to bypass the gate."
    fi
  fi
  log "Preflight done."
}

# ============================================================================
# Stage: layerwise (1xGPU single-GPU TP-mock, inside vLLM container)
# ============================================================================
lw_run_dir() { echo "$OUT_ROOT/layerwise/$1"; }
lw_csv()     { echo "$(lw_run_dir "$1")/layerwise.csv"; }

stage_layerwise() (
  local IMAGE="$VLLM_IMAGE"
  local RUN_DIR="$OUT_ROOT/.runtime/layerwise"
  local NAME_PREFIX="aic-layerwise-${BASHPID}"
  local RUNTIME_LOG_FILE="$LOG_DIR/layerwise-runtime.log"
  reset_outer_runtime_state
  trap 'runtime_teardown >/dev/null 2>&1 || true' EXIT
  runtime_prepare || die "runtime '${RUNTIME}' preparation failed for image '${IMAGE}'"

  local slug hf kind moe; local m
  for m in "${MODELS[@]}"; do
    IFS='|' read -r slug hf kind moe <<<"$m"
    local unit="layerwise_${slug}"
    if is_done "$unit"; then log "skip $unit (done; FORCE=1 to redo)"; continue; fi
    local rdir; rdir="$(lw_run_dir "$slug")"; mkdir -p "$rdir"
    log "Layerwise: $hf ($kind) tp=$LW_TP_LIST -> $rdir/layerwise.csv"

    local collect_model="$hf"
    local model_mount=()
    local model_policy_env=()
    local model_policy_args=()
    local auth_env=()
    local runtime_hf_token=""
    if [[ "${AIC_MODEL_MODE:-}" == "metadata_dummy" ]]; then
      collect_model="$METADATA_MODEL_CONTAINER"
      model_mount=(-v "$MODEL:$METADATA_MODEL_CONTAINER:ro")
      model_policy_env=(
        -e "AIC_LOAD_FORMAT=dummy"
        -e "AIC_MODEL_METADATA_DIR=$METADATA_MODEL_CONTAINER"
        -e "AIC_MODEL_MODE=metadata_dummy"
        -e "FPM_REAL_WORKLOAD_SHAPE_SOURCE=synthetic"
        -e "HF_DATASETS_OFFLINE=1"
        -e "HF_HUB_OFFLINE=1"
        -e "TRANSFORMERS_OFFLINE=1"
      )
      model_policy_args=(--extra-vllm-arg=--load-format=dummy)
    else
      runtime_hf_token="$(hf_token_value)"
      auth_env=(-e HF_TOKEN)
    fi

    # In-container collect command (modeled on the committed run_layerwise_smoke.sh).
    # Decode rows must carry the paired FPM scheduler surface; max-decode-batch-size
    # only bounds datapoint generation and does not set max_num_seqs.
    local incmd nsys_preamble=""
    local -a layerwise_nsys_args=(--nsys-capture none)
    local -a layerwise_nsys_mounts=()
    if layerwise_needs_nsight; then
      nsys_preamble=$(cat <<EOS
export PATH="${NSYS_BIN_CONTAINER_DIR}:${NSYS_TARGET_CONTAINER_DIR}:\$PATH"
if [[ -n "\${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${NSYS_TARGET_CONTAINER_DIR}:${NSYS_IMPORTER_CONTAINER_DIR}:\${LD_LIBRARY_PATH}"
else
  export LD_LIBRARY_PATH="${NSYS_TARGET_CONTAINER_DIR}:${NSYS_IMPORTER_CONTAINER_DIR}"
fi
${NSYS_BIN_CONTAINER_DIR}/nsys --version
EOS
)
      layerwise_nsys_args=(--nsys-capture full)
      layerwise_nsys_mounts=(
        -v "$NSYS_ROOT/bin:$NSYS_BIN_CONTAINER_DIR:ro"
        -v "$NSYS_TARGET_HOST_DIR:$NSYS_TARGET_CONTAINER_DIR:ro"
        -v "$NSYS_IMPORTER_HOST_DIR:$NSYS_IMPORTER_CONTAINER_DIR:ro"
      )
    fi
    incmd=$(cat <<EOS
set -euo pipefail
${nsys_preamble}
python3 -m collector.layerwise.vllm.collect \
  --run-dir /results \
  --model "${collect_model}" --model-kind "${kind}" \
  --tp-sizes ${LW_TP_LIST} --ep-sizes ${EP} \
  --phases ${LW_PHASES} --run-preset ${LW_RUN_PRESET} \
  --ctx-new-tokens ${LW_CTX_NEW_TOKENS} ${LW_CTX_PAST_KV:+--ctx-past-kv ${LW_CTX_PAST_KV}} --ctx-batch-sizes auto \
  --gen-batch-sizes ${LW_GEN_BATCH_SIZES} --gen-past-kv ${LW_GEN_PAST_KV} \
  --gen-max-num-seqs ${FPM_MAX_NUM_SEQS} \
  --max-decode-batch-size ${LW_MAX_DECODE_BATCH_SIZE} \
  --gemm-quant ${GEMM_QUANT} --attn-quant ${ATTN_QUANT} --kv-quant ${KV_QUANT} --moe-quant ${MOE_QUANT} \
  --system ${SYSTEM} --framework-version ${VLLM_VERSION} \
  --gpus ${LW_GPUS} --max-workers 1 \
  --max-model-len ${LW_MAX_MODEL_LEN} \
  --gpu-memory-utilization ${LW_GPU_MEM_UTIL} \
  --latency-source ${LW_LATENCY_SOURCE} ${layerwise_nsys_args[*]} ${model_policy_args[*]}
EOS
)
    local -a layerwise_runtime_opts=(
      --entrypoint bash --ipc=host --network=host \
        "${layerwise_nsys_mounts[@]}" \
        -v "$AIC_REPO:/workspace" \
        -v "$rdir:/results" \
        "${model_mount[@]}" \
        -v "$HF_HOME:/hf-cache" \
        -v "$VLLM_CACHE_HOST:/home/dynamo/.cache/vllm" \
        -v "$VLLM_CACHE_HOST:/root/.cache/vllm" \
        -e HF_HOME=/hf-cache -e HF_HUB_CACHE=/hf-cache/hub \
        "${auth_env[@]}" \
        "${model_policy_env[@]}" \
        -e TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang \
        -e TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp \
        -w /workspace
    )
    RUNTIME_LOG_FILE="$LOG_DIR/${unit}.log"
    HF_TOKEN="$runtime_hf_token" runtime_run_oneshot "" "\"device=${LW_GPUS}\"" layerwise_runtime_opts -- \
      -lc "$incmd"

    [[ "$DRY_RUN" == "1" || -f "$rdir/layerwise.csv" ]] || die "layerwise.csv not produced in $rdir"
    mark_done "$unit"
  done
)

# ============================================================================
# Stage: fpm (8xGPU real Dynamo deployment, one run per pareto point)
# ============================================================================
fpm_run_dir() { echo "$OUT_ROOT/fpm/$1/$2"; }   # <slug>/<pareto>

# Build the single workload contract shared by the clean and profiled lanes.
# Keeping this in one function prevents a real clean run from being paired with
# a static-sweep Nsight run (or vice versa).
FPM_WORKLOAD_ARGS=()
build_fpm_workload_args() {
  local req="$1" conc="$2"
  if [[ "$ATTRIBUTE_REAL_WORKLOAD" == "1" ]]; then
    FPM_WORKLOAD_ARGS=(
      --real-workload --real-workload-requests "$req" --real-workload-concurrency "$conc"
      --real-workload-dataset "$FPM_DATASET" --real-workload-shape-source "$FPM_SHAPE_SOURCE"
      --real-workload-isl-min "$ISL_MIN" --real-workload-isl-max "$ISL_MAX" --real-workload-isl-mean "$ISL_MEAN"
      --real-workload-osl-min "$OSL_MIN" --real-workload-osl-max "$OSL_MAX" --real-workload-osl-mean "$OSL_MEAN"
    )
  else
    FPM_WORKLOAD_ARGS=(--no-real-workload --decode-batches "$DECODE_BATCH_SIZES")
    if [[ -n "${DECODE_OSL:-}" ]]; then
      FPM_WORKLOAD_ARGS+=(--decode-osl "$DECODE_OSL")
    fi
  fi
  return 0
}

stage_fpm() {
  local slug hf kind moe; local m i
  for m in "${MODELS[@]}"; do
    IFS='|' read -r slug hf kind moe <<<"$m"
    for i in "${!PARETO_NAMES[@]}"; do
      local pname="${PARETO_NAMES[$i]}" conc="${PARETO_CONCURRENCY[$i]}"
      local seed_env=(PROMPT_TOKEN_SEED="$i")
      # P2: per-point request count = clamp(FPM_REQ_MULT*conc, FPM_REQ_MIN, FPM_REQ_MAX).
      local req=$(( FPM_REQ_MULT * conc ))
      (( req < FPM_REQ_MIN )) && req="$FPM_REQ_MIN"
      (( req > FPM_REQ_MAX )) && req="$FPM_REQ_MAX"
      local unit="fpm_${slug}_${pname}"
      if is_done "$unit"; then log "skip $unit (done; FORCE=1 to redo)"; continue; fi
      local rdir; rdir="$(fpm_run_dir "$slug" "$pname")"; mkdir -p "$rdir"
      log "FPM: $hf tp=$FPM_TP_LIST pareto=$pname concurrency=$conc requests=$req -> $rdir"

      local extra=()
      [[ "$ALLOW_VERSION_MISMATCH" == "1" ]] && extra+=(--allow-version-mismatch --expected-vllm-version "$VLLM_VERSION")
      build_fpm_workload_args "$req" "$conc"
      local workload=("${FPM_WORKLOAD_ARGS[@]}")
      local policy_extra=()
      local -a fpm_env_args=(
        "${seed_env[@]}"
        "MAX_NUM_SEQS=$FPM_MAX_NUM_SEQS"
        "MAX_NUM_BATCHED_TOKENS=$FPM_MAX_NUM_BATCHED_TOKENS"
        "ENABLE_CHUNKED_PREFILL=$FPM_ENABLE_CHUNKED_PREFILL"
      )
      if [[ "${AIC_MODEL_MODE:-}" == "metadata_dummy" ]]; then
        policy_extra=(--extra-vllm-arg=--load-format=dummy)
        fpm_env_args+=("${PIPELINE_DUMMY_ENV[@]}")
      else
        fpm_env_args+=("HF_TOKEN=$(hf_token_value)")
      fi

      # Scheduler parity forced via env (FPM shell reads $MAX_NUM_SEQS / $MAX_NUM_BATCHED_TOKENS;
      # the python wrapper inherits os.environ into the subprocess).
      run_env "$LOG_DIR/${unit}.log" \
        "${fpm_env_args[@]}" -- \
        python3 -m collector.layerwise.fpm.collect \
          --model "$hf" \
          --tp-sizes "$FPM_TP_LIST" --ep-sizes "$EP" \
          --phases "$ATTRIBUTE_PHASES" \
          --decode-past-kv "$DECODE_PAST_KV" \
          "${workload[@]}" \
          --prompt-token-mode safe_ascii \
          --warmup-requests "$FPM_WARMUP_REQUESTS" \
          --image "$DYNAMO_VLLM_IMAGE" \
          --run-dir "$rdir" \
          "${policy_extra[@]}" \
          "${extra[@]}"

      mark_done "$unit"
    done
  done
}

# ============================================================================
# Stage: align (plot_fpm_vs_aic.py, one chart set per pareto point)
# ============================================================================
# Ensure <fpm_run_dir>/tp{T}_ep{E}_past{K}/fpm_metrics_phase.csv exists for each
# collected TP. Single-TP runs land flat (case_run_dir nests only when >1 case),
# so symlink the flat phase CSV into the nested path the plot's --fpm-run-name expects.
normalize_fpm_layout() {
  local rdir="$1" tp
  for tp in ${FPM_TP_LIST//,/ }; do
    local nested="$rdir/tp${tp}_ep${EP}_past${DECODE_PAST_KV}/fpm_metrics_phase.csv"
    local flat="$rdir/fpm_metrics_phase.csv"
    if [[ ! -e "$nested" && -f "$flat" ]]; then
      printf '%s+ symlink %s -> %s%s\n' "$C_DIM" "$nested" "$flat" "$C_OFF"
      [[ "$DRY_RUN" == "1" ]] || { mkdir -p "$(dirname "$nested")"; ln -sf "$flat" "$nested"; }
    fi
  done
}

# DEPRECATED (2026-07-02): superseded by plot_fpm_vs_aic.py `--vllm-max-num-seqs auto`, which
# resolves mns from the run's own FPM metadata (via _load_fpm_max_num_seqs) AND applies the same
# to max_num_batched_tokens. stage_align no longer calls this; kept only for reference/back-compat.
# shellcheck disable=SC2329  # retained intentionally; not invoked after the auto-parity fix
plot_max_num_seqs() {
  local rdir="$1" cfg; cfg="$(ls "$rdir"/effective_vllm_config.json "$rdir"/*/effective_vllm_config.json 2>/dev/null | head -1 || true)"
  if [[ -n "$cfg" && -f "$cfg" && "$DRY_RUN" != "1" ]]; then
    python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('scheduler_config.max_num_seqs') or d.get('max_num_seqs') or $FPM_MAX_NUM_SEQS)" "$cfg" 2>/dev/null || echo "$FPM_MAX_NUM_SEQS"
  else
    echo "$FPM_MAX_NUM_SEQS"
  fi
}

stage_align() {
  local slug hf kind moe; local m pname
  for m in "${MODELS[@]}"; do
    IFS='|' read -r slug hf kind moe <<<"$m"
    local lwcsv; lwcsv="$(lw_csv "$slug")"
    if [[ "$DRY_RUN" != "1" && ! -f "$lwcsv" ]]; then die "layerwise CSV missing for align: $lwcsv (run layerwise stage first)"; fi
    for pname in $PLOT_PARETO; do
      local unit="align_${slug}_${pname}"
      if is_done "$unit"; then log "skip $unit (done; FORCE=1 to redo)"; continue; fi
      local fdir; fdir="$(fpm_run_dir "$slug" "$pname")"
      normalize_fpm_layout "$fdir"
      local cdir="$OUT_ROOT/charts/$slug/$pname"; mkdir -p "$cdir"
      log "Align: $hf pareto=$pname  layerwise=$lwcsv  fpm-root=$(dirname "$fdir")  -> $cdir"

      local moearg=()
      [[ -n "$moe" ]] && moearg=(--moe-perf-file "$AIC_REPO/$moe")

      # Config parity (load-bearing): pass `auto` so plot_fpm_vs_aic.py resolves
      # max_num_batched_tokens / max_num_seqs from THIS run's own FPM metadata (matching the
      # summary tool) instead of hardcoding 2048/256. A wrong mnbt off-parity-chunks any ctx
      # point with new_tokens>mnbt; a wrong mns mis-selects the decode row. Overridable via
      # PLOT_MAX_NUM_BATCHED_TOKENS / PLOT_MAX_NUM_SEQS for explicit control.
      local plot_mnbt="${PLOT_MAX_NUM_BATCHED_TOKENS:-auto}"
      local plot_mns="${PLOT_MAX_NUM_SEQS:-auto}"
      run "$LOG_DIR/${unit}.log" \
        python3 tools/plot_fpm_vs_aic.py \
          --layerwise "$lwcsv" \
          --model "$hf" \
          --system "$SYSTEM" --backend "$BACKEND" --version "$DATA_VERSION" \
          --systems-root "$AIC_REPO/src/aiconfigurator/systems" \
          --fpm-root "$(dirname "$fdir")" \
          --fpm-run-name "$(basename "$fdir")" \
          --vllm-max-num-seqs "$plot_mns" --vllm-max-num-batched-tokens "$plot_mnbt" \
          --phases "$PLOT_PHASES" \
          --out-dir "$cdir" \
          "${moearg[@]}"

      mark_done "$unit"
    done

    # Per-concurrency FPM data-profile figures (batch composition + param
    # distributions + latency scatters). Runs alongside align so the diagnostic
    # charts arrive with the AIC-vs-FPM charts. Fail-safe: a plotting failure
    # (missing matplotlib/pandas/CSV) warns and does not abort the pipeline.
    profile_fpm_run "$slug" "$hf"
  done
}

# ============================================================================
# Stage: profile (per-concurrency FPM data-distribution figures)
# ============================================================================
# For each pareto point, render fpm_distribution_{composition,params,scatter}.png
# from that point's fpm_metrics_phase.csv into
# <OUT_ROOT>/charts/<slug>/<pareto>/fpm_profile/. This is pure CSV postprocessing:
# no GPU, no AIC lookup. It is OPT-fail-safe -- if the phase CSV is absent or the
# plotter errors (e.g. matplotlib/pandas missing), it warns and continues so a
# diagnostics gap never fails the collection pipeline. Selectable via STAGES and
# also invoked automatically at the tail of stage_align.
profile_fpm_run() {
  local slug="$1" hf="$2" pname
  for pname in $PLOT_PARETO; do
    local unit="profile_${slug}_${pname}"
    if is_done "$unit"; then log "skip $unit (done; FORCE=1 to redo)"; continue; fi
    local fdir; fdir="$(fpm_run_dir "$slug" "$pname")"
    local pcsv="$fdir/fpm_metrics_phase.csv"
    if [[ "$DRY_RUN" != "1" && ! -f "$pcsv" ]]; then
      # Single-TP runs may nest under tp{T}_ep{E}_past{K}/; fall back to a search.
      pcsv="$(find "$fdir" -name fpm_metrics_phase.csv 2>/dev/null | head -1 || true)"
    fi
    if [[ "$DRY_RUN" != "1" && ( -z "$pcsv" || ! -f "$pcsv" ) ]]; then
      warn "profile: no fpm_metrics_phase.csv under $fdir; skipping $unit"
      continue
    fi
    local pdir="$OUT_ROOT/charts/$slug/$pname/fpm_profile"; mkdir -p "$pdir"
    local conc="$pname"
    local idx; for idx in "${!PARETO_NAMES[@]}"; do
      [[ "${PARETO_NAMES[$idx]}" == "$pname" ]] && conc="${PARETO_CONCURRENCY[$idx]}" && break
    done
    log "Profile: $hf pareto=$pname conc=$conc  csv=${pcsv:-<dry-run>}  -> $pdir"
    if run "$LOG_DIR/${unit}.log" \
        python3 -m collector.layerwise.diagnostics.plot_fpm_distributions \
          --fpm-csv "${pcsv:-$pcsv}" \
          --out-dir "$pdir" \
          --title "$hf  |  concurrency=$conc"; then
      mark_done "$unit"
    else
      warn "profile: plotter failed for $unit (see $LOG_DIR/${unit}.log); continuing"
    fi
  done
}

stage_profile() {
  local slug hf kind moe m
  for m in "${MODELS[@]}"; do
    IFS='|' read -r slug hf kind moe <<<"$m"
    profile_fpm_run "$slug" "$hf"
  done
}

# ============================================================================
# Report
# ============================================================================
report() {
  log "Done. STAGES='$STAGES'  OUT_ROOT=$OUT_ROOT"
  if [[ "$DRY_RUN" == "1" ]]; then log "(DRY_RUN: nothing executed; commands printed above)"; return; fi
  log "Artifacts:"
  [[ -d "$OUT_ROOT/layerwise" ]] && find "$OUT_ROOT/layerwise" -name layerwise.csv -printf '  layerwise: %p\n' 2>/dev/null || true
  [[ -d "$OUT_ROOT/fpm" ]]       && find "$OUT_ROOT/fpm" -name fpm_metrics_phase.csv -printf '  fpm:       %p\n' 2>/dev/null || true
  [[ -d "$OUT_ROOT/charts" ]]    && find "$OUT_ROOT/charts" -name '*.png' -printf '  chart:     %p\n' 2>/dev/null || true
  log "Logs: $LOG_DIR/"
}

# ============================================================================
# Stage: attribute (nsys windowed capture of the real FPM workload + decompose)
# ============================================================================
# Two lanes joined by shape: the profiled lane (nsys windowed capture -> per-step
# compute/comm/busy) supplies composition only; the clean lane (the existing FPM
# golden run dir) supplies the authoritative per-shape wall. aic_fpm_attribute.py
# joins {profiled composition + clean wall + AIC layerwise prediction} and writes
# the gap decomposition CSV.
stage_attribute() {
  local slug hf kind moe m i
  for m in "${MODELS[@]}"; do
    IFS='|' read -r slug hf kind moe <<<"$m"
    for i in "${!PARETO_NAMES[@]}"; do
      local pname="${PARETO_NAMES[$i]}" conc="${PARETO_CONCURRENCY[$i]}"
      local seed_env=(PROMPT_TOKEN_SEED="$i")
      local req=$(( FPM_REQ_MULT * conc ))
      (( req < FPM_REQ_MIN )) && req="$FPM_REQ_MIN"
      (( req > FPM_REQ_MAX )) && req="$FPM_REQ_MAX"
      local unit="attribute_${slug}_${pname}"
      if is_done "$unit"; then log "skip $unit (done; FORCE=1 to redo)"; continue; fi
      local rdir; rdir="$(fpm_run_dir "$slug" "$pname")/attribute"; mkdir -p "$rdir"
      log "Attribute: $hf tp=$FPM_TP_LIST pareto=$pname conc=$conc req=$req window=$ATTRIBUTE_WINDOW -> $rdir"

      # (a) FPM real workload under nsys windowed capture (reuses the FPM collector path).
      # NSYS_BIN/NSYS_HOST_DIR propagate (collect.py runs the inner shell with os.environ.copy()) so
      # collect_fpm_metrics.sh mounts the host Nsight install ro into the worker container and execs the
      # absolute nsys path there (the worker image has no nsys on PATH -> bare `nsys` exits 127).
      # The collector can exit nonzero (e.g. 2) on a non-fatal per-request case failure even when the
      # nsys capture succeeded (Phase-0 smoke: DRIVER_EXIT=2 but fpm_worker.nsys-rep was produced). So
      # capture the rc without letting `set -e` abort the driver, and gate continuation on a .nsys-rep
      # existing rather than on rc==0; only a total collection failure (NO .nsys-rep) fails the unit.
      # LOAD_FORMAT=dummy -> vLLM random weights (no checkpoint download). Valid for the
      # timing/comm mechanism study: kernels are shape-driven (value-independent), Qwen3-32B
      # is dense (no MoE routing), and IGNORE_EOS fixes decode length. Default empty = real weights.
      local extra=()
      [[ "$ALLOW_VERSION_MISMATCH" == "1" ]] && extra+=(--allow-version-mismatch --expected-vllm-version "$VLLM_VERSION")
      local extra_vllm=()
      [[ -n "${LOAD_FORMAT:-}" ]] && extra_vllm+=(--extra-vllm-arg="--load-format=${LOAD_FORMAT}")

      # Workload selection (design.md v3 §5). Default = real-workload high-C mixed-step
      # capture. ATTRIBUTE_REAL_WORKLOAD=0 selects the static decode sweep (decode-only
      # arm): DECODE_BATCH_SIZES drives both request count and concurrency per decode
      # point, at a fixed DECODE_PAST_KV; the real-workload-shape args are dropped.
      build_fpm_workload_args "$req" "$conc"
      local workload=("${FPM_WORKLOAD_ARGS[@]}")

      local attribute_nsys_flags=(--nsys-profile-worker)
      if [[ "$ATTRIBUTE_FULL_WORKER" == "1" ]]; then
        attribute_nsys_flags+=(--nsys-full-worker)
      else
        attribute_nsys_flags+=(--nsys-cuda-profiler-window "$ATTRIBUTE_WINDOW")
      fi

      local collect_rc=0
      local -a attribute_env_args=(
        "${seed_env[@]}"
        "MAX_NUM_SEQS=$FPM_MAX_NUM_SEQS"
        "MAX_NUM_BATCHED_TOKENS=$FPM_MAX_NUM_BATCHED_TOKENS"
        "ENABLE_CHUNKED_PREFILL=$FPM_ENABLE_CHUNKED_PREFILL"
      )
      if [[ "${AIC_MODEL_MODE:-}" == "metadata_dummy" ]]; then
        attribute_env_args+=("${PIPELINE_DUMMY_ENV[@]}")
      else
        attribute_env_args+=("HF_TOKEN=$(hf_token_value)")
      fi
      attribute_env_args+=(
        "NSYS_BIN=$NSYS_ROOT/bin/nsys"
        "NSYS_HOST_DIR=$NSYS_ROOT"
        "NSYS_CONTAINER_ROOT=/opt/nvidia/nsight-systems/${NSYS_VERSION_DIR}"
        "NSYS_TARGET_HOST_DIR=$NSYS_TARGET_HOST_DIR"
        "NSYS_IMPORTER_HOST_DIR=$NSYS_IMPORTER_HOST_DIR"
        "NSYS_TARGET_CONTAINER_DIR=$NSYS_TARGET_CONTAINER_DIR"
        "NSYS_IMPORTER_CONTAINER_DIR=$NSYS_IMPORTER_CONTAINER_DIR"
      )
      run_env "$LOG_DIR/${unit}.log" \
        "${attribute_env_args[@]}" -- \
        python3 -m collector.layerwise.fpm.collect \
          --model "$hf" --tp-sizes "$FPM_TP_LIST" --ep-sizes "$EP" \
          --phases "$ATTRIBUTE_PHASES" --decode-past-kv "$DECODE_PAST_KV" \
          "${workload[@]}" \
          --prompt-token-mode safe_ascii --warmup-requests "$FPM_WARMUP_REQUESTS" \
          --image "$DYNAMO_VLLM_IMAGE" --run-dir "$rdir" \
          "${extra[@]}" \
          "${extra_vllm[@]}" \
          "${attribute_nsys_flags[@]}" || collect_rc=$?

      local nsysrep; nsysrep="$(ls -1 "$rdir"/nsys/*.nsys-rep 2>/dev/null | head -1 || true)"
      if [[ "$DRY_RUN" != "1" && -z "$nsysrep" ]]; then
        die "collect produced no .nsys-rep under $rdir/nsys (rc=$collect_rc); attribute capture failed for $unit"
      fi
      if [[ "$DRY_RUN" != "1" && "$collect_rc" != "0" ]]; then
        warn "collect exited rc=$collect_rc but .nsys-rep was produced ($nsysrep); proceeding to decompose for $unit"
      fi

      # (b) reduce + decompose on the captured sqlite (clean wall from the existing fpm run dir).
      # If a .nsys-rep exists but no .sqlite yet, export it host-side before the sqlite lookup
      # (warn+continue on export failure so the lookup below can still find a pre-existing sqlite).
      local existing_sqlite; existing_sqlite="$(ls -1 "$rdir"/nsys/*.sqlite 2>/dev/null | head -1 || true)"
      if [ -z "$existing_sqlite" ]; then
        local sqlite_base="${nsysrep%.nsys-rep}"
        log "Exporting nsys report to sqlite: $nsysrep -> ${sqlite_base}.sqlite"
        run_env "$LOG_DIR/${unit}_export.log" -- \
          "$NSYS_ROOT/bin/nsys" export --type sqlite --force-overwrite true \
            -o "${sqlite_base}.sqlite" "$nsysrep" \
          || warn "nsys export failed for $nsysrep (rc=$?); decompose may have no sqlite to read"
      fi

      local sqlite; sqlite="$(ls -1 "$rdir"/nsys/*.sqlite 2>/dev/null | head -1 || true)"
      # Fail closed: a missing .sqlite means there is nothing to attribute, so the
      # unit must NOT be stamped .done (the .nsys-rep is retained on disk for manual
      # export+decompose). Aborting loudly forces the run to be fixed and re-attempted.
      if [ -z "$sqlite" ]; then die "no .sqlite under $rdir/nsys for $unit; capture retained, refusing to mark done"; fi
      # Fail locally when decomposition is empty or invalid. The .nsys-rep/.sqlite stay
      # under the node-local run directory for diagnosis, and no .done marker is written.
      # --fpm-run is the CLEAN non-profiled FPM run for this pareto point: its wall timing is
      # authoritative and is not perturbed by nsys. --profiled-fpm-run is this attribute run
      # ($rdir), used for runtime-config/provenance only; the sqlite supplies the profiled
      # composition lane.
      local clean_fpm_run; clean_fpm_run="$(fpm_run_dir "$slug" "$pname")"
      if [[ "$DRY_RUN" != "1" && ! -f "$clean_fpm_run/fpm_metrics_phase.csv" ]]; then
        die "clean FPM phase CSV missing for attribute: $clean_fpm_run/fpm_metrics_phase.csv (run fpm stage first)"
      fi
      # --per-pid (ATTRIBUTE_PER_PID=1, default) ALSO emits accurate per-rank rows
      # (pid column) alongside the aggregate -- the per-rank variance the v3
      # arrival-skew decomposition needs. The aggregate rows are unchanged.
      local perpid=()
      [[ "$ATTRIBUTE_PER_PID" == "1" ]] && perpid=(--per-pid)
      # Grade AIC's layerwise prediction against THIS run's freshly-collected layerwise.csv
      # (collected by the layerwise stage at $OUT_ROOT/layerwise/<slug>/layerwise.csv) rather
      # than the shipped systems-data file. If the layerwise stage was not run (no fresh CSV),
      # omit the flag so aic_fpm_attribute falls back to the shipped default.
      local lwarg=(); local lwcsv; lwcsv="$(lw_csv "$slug")"
      [[ -f "$lwcsv" ]] && lwarg=(--layerwise-csv "$lwcsv")
      # Decompose imports the aiconfigurator SDK (AIC predictions). On a source-checkout
      # box with no installed dist, put src/ on PYTHONPATH so the import resolves; the
      # __init__ version-fallback makes it work without dist metadata.
      run_env "$LOG_DIR/${unit}_decompose.log" \
        "PYTHONPATH=$AIC_REPO:$AIC_REPO/src${PYTHONPATH:+:$PYTHONPATH}" -- \
        python3 -m collector.layerwise.diagnostics.aic_fpm_attribute \
          --sqlite "$sqlite" \
          --fpm-run "$clean_fpm_run" \
          --profiled-fpm-run "$rdir" \
          --system "$SYSTEM" --model "$hf" --tp "$TP" \
          --step-window "$ATTRIBUTE_WINDOW" \
          --discard-first-n "$ATTRIBUTE_DISCARD_N" \
          "${perpid[@]}" \
          "${lwarg[@]}" \
          --out "$rdir/decomposition.csv" \
        || die "decompose failed for $unit; .nsys-rep + .sqlite retained under $rdir/nsys, refusing to mark done"

      # Fail-closed attribution-validity gate (spec Fix 2): a unit is only stamped
      # .done when attribution is real -- >=1 CUPTI kernel row, >=1 bench_step:: NVTX
      # range, and >=1 decomposition.csv data row. On any miss, die (the named check
      # pinpoints the broken stage); the .nsys-rep/.sqlite/csv stay on disk for manual
      # decompose, and the unit re-runs on the next pass (no FORCE=1 needed).
      if [[ "$DRY_RUN" != "1" ]]; then
        run_env "$LOG_DIR/${unit}_assert.log" \
          "PYTHONPATH=$AIC_REPO:$AIC_REPO/src${PYTHONPATH:+:$PYTHONPATH}" -- \
          python3 -m collector.layerwise.diagnostics.assert_attribution_valid \
            --sqlite "$sqlite" \
            --decomposition "$rdir/decomposition.csv" \
          || die "attribution-validity gate failed for $unit; artifacts retained under $rdir, refusing to mark done"
      fi

      mark_done "$unit"
    done

    # Mandatory Stage-1 tail: preserve the clean and profiled populations, align
    # every measured profiled row to its same-run Nsight marker, reduce intact
    # per-rank kernel tuples, and invoke AIC exactly once per measured clean bin.
    # This runs only after every requested cohort passed the per-unit attribution
    # gate above and before report()/the outer collector packages OUT_ROOT.
    local semantic_unit="attribute_semantic_${slug}"
    local semantic_out="$OUT_ROOT/semantic_insights"
    if is_done "$semantic_unit"; then
      [[ -f "$semantic_out/manifest.json" ]] \
        || die "$semantic_unit is marked done but $semantic_out/manifest.json is missing"
      log "skip $semantic_unit (done; FORCE=1 to redo)"
      continue
    fi
    if [[ -e "$semantic_out" ]]; then
      if [[ "$FORCE" == "1" ]]; then
        rm -rf "$semantic_out"
      else
        die "$semantic_out already exists without a done marker; inspect it or use FORCE=1"
      fi
    fi

    local source_job_id="${SEMANTIC_JOB_ID:-${CI_JOB_ID:-}}"
    local source_pipeline_id="${SEMANTIC_PIPELINE_ID:-${CI_PIPELINE_ID:-}}"
    local collector_commit="${AUTO_COLLECTOR_COMMIT:-${CI_COMMIT_SHA:-}}"
    if [[ "$DRY_RUN" != "1" ]]; then
      [[ -n "$source_job_id" ]] \
        || die "semantic reducer requires SEMANTIC_JOB_ID or CI_JOB_ID"
      [[ -n "$source_pipeline_id" ]] \
        || die "semantic reducer requires SEMANTIC_PIPELINE_ID or CI_PIPELINE_ID"
      [[ -n "$collector_commit" ]] \
        || die "semantic reducer requires AUTO_COLLECTOR_COMMIT or CI_COMMIT_SHA"
    fi

    local cohort_args=()
    for i in "${!PARETO_NAMES[@]}"; do
      local pname="${PARETO_NAMES[$i]}" conc="${PARETO_CONCURRENCY[$i]}"
      local clean_run; clean_run="$(fpm_run_dir "$slug" "$pname")"
      local profiled_run="$clean_run/attribute"
      local cohort_sqlite; cohort_sqlite="$(ls -1 "$profiled_run"/nsys/*.sqlite 2>/dev/null | head -1 || true)"
      if [[ "$DRY_RUN" != "1" ]]; then
        [[ -f "$clean_run/fpm_metrics_phase.csv" ]] \
          || die "semantic reducer clean CSV missing: $clean_run/fpm_metrics_phase.csv"
        [[ -f "$profiled_run/fpm_metrics_phase.csv" ]] \
          || die "semantic reducer profiled CSV missing: $profiled_run/fpm_metrics_phase.csv"
        [[ -n "$cohort_sqlite" ]] \
          || die "semantic reducer Nsight SQLite missing under $profiled_run/nsys"
      fi
      cohort_args+=(--cohort "${conc}:${clean_run}:${profiled_run}:${cohort_sqlite}")
    done
    local chunked_flag="--no-chunked-prefill"
    [[ "$FPM_ENABLE_CHUNKED_PREFILL" == "1" ]] && chunked_flag="--chunked-prefill"
    local measured_segment="real"
    [[ "$ATTRIBUTE_REAL_WORKLOAD" == "0" ]] && measured_segment="sweep"
    local lwcsv; lwcsv="$(lw_csv "$slug")"
    run_env "$LOG_DIR/${semantic_unit}.log" \
      "PYTHONPATH=$AIC_REPO:$AIC_REPO/src${PYTHONPATH:+:$PYTHONPATH}" -- \
      python3 -m collector.layerwise.diagnostics.semantic_fpm_stage1 \
        "${cohort_args[@]}" \
        --output-dir "$semantic_out" \
        --aic-repo "$AIC_REPO" \
        --auto-collector-commit "$collector_commit" \
        --layerwise-csv "$lwcsv" \
        --model "$hf" \
        --model-revision "$MODEL_REVISION" \
        --system "$SYSTEM" \
        --comm-version "$DATA_VERSION" \
        "$chunked_flag" \
        --ep-size "$EP" \
        --gemm-quant "$GEMM_QUANT" \
        --attention-quant "$ATTN_QUANT" \
        --kv-cache-quant "$KV_QUANT" \
        --moe-quant "$MOE_QUANT" \
        --job-id "$source_job_id" \
        --pipeline-id "$source_pipeline_id" \
        --measured-segment "$measured_segment" \
      || die "semantic Stage-1 reducer failed; refusing to mark $semantic_unit done"
    mark_done "$semantic_unit"
  done
}

# ============================================================================
# Main
# ============================================================================
main() {
  cd "$AIC_REPO"
  apply_pipeline_policy
  resolve_nsight_architecture
  log "repo=$AIC_REPO"
  preflight
  for stage in $STAGES; do
    case "$stage" in
      layerwise) log "== STAGE layerwise =="; stage_layerwise;;
      fpm)       log "== STAGE fpm =="; stage_fpm;;
      attribute) log "== STAGE attribute =="; stage_attribute;;
      align)     log "== STAGE align =="; stage_align;;
      profile)   log "== STAGE profile =="; stage_profile;;
      *) die "unknown stage: $stage (valid: layerwise fpm attribute align profile)";;
    esac
  done
  report
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
