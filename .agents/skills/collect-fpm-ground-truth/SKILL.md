---
name: collect-fpm-ground-truth
description: Collect Dynamo/vLLM ForwardPassMetrics ground-truth latency data for AIC/layerwise validation. Use when the user asks to run or update FPM ground truth, collect context/decode/mixed vLLM metrics, compare layerwise/AIC against real vLLM behavior, debug Dynamo FPM collection, or preserve mixed-step FPM latency rows.
---

# Collect FPM Ground Truth

## Core Workflow

Work from the `aiconfigurator` repo unless the user points elsewhere. Use:

- Collector: `python -m collector.layerwise.fpm.collect --model <model>`
- Internal shell wrapper: `collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh`
- Dynamo image: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0`
- Expected vLLM: `0.20.1`; do not allow version mismatch unless explicitly accepted.
- HF token: export `HF_TOKEN` directly, or set `HF_TOKEN_FILE` and export `HF_TOKEN="$(tr -d '\n' < "$HF_TOKEN_FILE")"`.
- Model cache: set `HF_HOME`; default to `$HOME/.cache/huggingface` when unspecified.
- vLLM compile/cache directory: set `VLLM_CACHE_HOST`; default to `$HOME/.cache/aic-vllm`. The wrapper mounts it to both `/home/dynamo/.cache/vllm` and `/root/.cache/vllm` so DeepGEMM, FlashInfer, TileLang, and torch compile artifacts survive worker restarts in Dynamo and root-run vLLM containers. For DeepSeek/TileLang paths, export `TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang` and `TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp`.
- Artifact directory: set `AIC_LAYERWISE_ARTIFACTS`; default to `$PWD/.tmp/layerwise-artifacts` when running from the repo.

Prefer one deployment per sweep. For layerwise gap closure, collect `context,decode` first and add `mixed` only after those phases are understood. Always preserve `*_phase.csv`, `*_detail.csv`, and `*_workload.csv`; partial runs can still be valid for phases that finished.

Read [references/fpm-commands.md](references/fpm-commands.md) for the smoke command, canonical TP=2 command, output interpretation, and failure handling.

The normal CLI should be compact: provide `--model`, optional `--run-dir`, and `--tp-sizes`/`--ep-sizes` when needed. The wrapper infers Docker GPUs from TP/EP unless `--gpus` is explicitly supplied. Standard output files are created under `--run-dir`; use explicit output-path flags only for unusual plumbing.

## Important Defaults

- Use random prompt token IDs through the completions API. Do not use constant token IDs.
- Keep `ignore_eos=true` unless the user asks for OSL-as-cap behavior.
- Use local file-discovery heartbeat. `--file-discovery-touch-seconds 2` fixed prior discovery expiry/503 failures.
- Omit `--gpus` for the common case. If overriding devices, quote Docker device selectors as `--gpus '"device=0,1"'`; unquoted `--gpus device=0,1` is rejected by Docker.
- GPT-OSS FPM synthetic sweeps use vLLM's recommended benchmark defaults automatically: FP8 KV cache when unset, prefix caching disabled unless explicitly overridden, `max-cudagraph-capture-size=2048`, and `stream-interval=20`. Treat prefix-cache disablement as a measurement consistency default, not a generic GPT-OSS correctness rule.
- For context repeats, use medians or trimmed means and inspect first-pass cold outliers before tuning against them.
- For decode, compare pure decode-only FPM rows by `decode_requests` and `mean_decode_kv_tokens`. Prefer the longest consecutive full-batch block where `decode_tokens == decode_requests`; exclude tail rows after requests finish.
- For 16k context with vLLM chunked prefill, FPM appears as `8192 @ ctx_kv=0` plus `8192 @ ctx_kv=8192`, not a single `ctx_tokens=16384` row.

## Smoke Validation

Use the reference smoke command after FPM wrapper changes. A passing Qwen3-32B TP1 smoke writes `fpm_metrics.csv`, `fpm_metrics_detail.csv`, `fpm_metrics_phase.csv`, request/warmup workload CSVs, and vLLM metadata/config JSON files. Expect both context and decode rows in `fpm_metrics_phase.csv`; cold first context rows can be much slower than subsequent rows.

## Attributed-FPM (STAGES=attribute)

A VALIDATED sibling collection mode (Linear AIC-1195) that runs the FPM real workload UNDER nsys with per-step NVTX, then decomposes the AIC-vs-FPM gap into compute/comm/overhead terms. Use it when the user asks to attribute the FPM gap, profile FPM steps under nsys, or produce a per-step gap decomposition. The driver stage lives in `collector/layerwise/reproduce_layerwise_fpm.sh:stage_attribute` (dispatched by `STAGES=attribute`).

**What it produces.** Per pareto point under `<OUT_ROOT>/fpm/<slug>/<pareto>/attribute/`: a windowed nsys capture (`nsys/*.nsys-rep`), its export (`nsys/*.sqlite`), the clean per-shape FPM wall (`fpm_metrics_phase.csv` from this same run dir), and the gap decomposition (`decomposition.csv`). Two lanes joined by per-step shape: the nsys lane supplies COMPOSITION only (never timing); the clean FPM wall is the authoritative per-shape latency.

**Per-step NVTX (DYNAMO-side, real-batch labels).** `collector/layerwise/vllm/dynamo_step_marker.py` monkeypatches `vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model` (the GPU FORWARD where kernels run — NOT `update_from_output`, which brackets only post-step CPU bookkeeping). It reads the REAL per-step batch from `scheduler_output` and emits `bench_step::N{step:07d}::bs{decode_batch}::past{mean_kv:06d}`. It is injected by `collector/layerwise/vllm/sitecustomize.py:_install` when `LAYERWISE_DYNAMO_STEP_MARKER=1` and the repo is on `PYTHONPATH`. Do NOT use `vllm_step_marker` here — its counter-mode labels assume single-stream `isl=1` and are wrong for a multi-request real workload.

**nsys flag threading + host-nsys mount.** `--nsys-profile-worker` / `--nsys-cuda-profiler-window` are threaded `fpm/collect.py:_build_arg_parser` -> `fpm/docker.py:build_collect_command` -> `collect_fpm_metrics.sh`. The worker image has NO nsys on PATH (bare `nsys` exits 127), so the host Nsight install is bind-mounted read-only into the worker: pass `NSYS_BIN=$NSYS_ROOT/bin/nsys` + `NSYS_HOST_DIR=$NSYS_ROOT` as env. `collect.py:main` runs the inner shell with `env={**os.environ.copy(), **cmd.env}`, so these propagate; the shell mounts `NSYS_HOST_DIR` ro at the same path (`collect_fpm_metrics.sh` ~line 765-772) and execs the absolute `NSYS_BIN` inside the worker. The shell also mounts the repo ro at `/aic-src`, sets `PYTHONPATH=/aic-src/collector/layerwise/vllm:/aic-src`, and `LAYERWISE_DYNAMO_STEP_MARKER=1` (gated on `--nsys-profile-worker`, so non-profiled runs are unchanged).

**Session-gating, not `-c cudaProfilerApi`.** Capture is SESSION-gated (`NSYS_PROFILE_TRAFFIC_ONLY=1` -> `nsys profile --session-new <name> --start-later`, `collect_fpm_metrics.sh` ~line 1359-1376). The window only LABELS steps via NVTX; ALL traffic steps are captured. So OSL/requests govern trace size — keep OSL SHORT (e.g. OSL_MEAN/MAX 256) to bound it. `aic_fpm_attribute` needs `.sqlite`, so nsys exports `.nsys-rep` -> `.sqlite`.

**Box invocation (8xh100-layerwise).** Source the host venv first (host deps numpy/pandas/transformers; the driver calls bare `python3`), point NSYS at the box's 2025.3.2 install, and keep OSL short:

```bash
source /home/ubuntu/.venv/bin/activate
NSYS_VERSION_DIR=2025.3.2 NSYS_ROOT=/opt/nvidia/nsight-systems/2025.3.2 \
STAGES=attribute MODEL=Qwen/Qwen3-32B TP=8 \
PARETO_CONCURRENCY=4 OSL_MEAN=256 OSL_MAX=256 \
ATTRIBUTE_WINDOW=100-115 ATTRIBUTE_DISCARD_N=3 \
./collector/layerwise/reproduce_layerwise_fpm.sh
```

`ATTRIBUTE_WINDOW` is `"lo-hi[,lo-hi...]"` step ordinals (the marker's `bench_step::N` step) passed to `--nsys-cuda-profiler-window`; `ATTRIBUTE_DISCARD_N` drops the first N sync-drained boundary steps before reducing. Note the driver default `NSYS_VERSION_DIR=2026.3.1` is for layerwise; the box ships 2025.3.2, so override it.

**Box gotchas.** Disk is the binding constraint (130MB+ per 32B capture; `OUT_ROOT` must be local ext4 — nsys export dies "database is locked" on SMB/NFS). `docker rm -f` is permission-gated on the box — use `docker stop` to clear leftover `dynamo-fpm-*` containers.

**4 stage-script robustness fixes (do NOT regress these in `stage_attribute`).**
1. Empty-glob + `pipefail` abort is guarded with `|| true` on `ls ... | head` lookups (`nsysrep`, `existing_sqlite`, `sqlite`).
2. Decompose is guarded with `|| warn` so a reduce-side failure (e.g. AIC has no layerwise data for the model) NEVER discards the expensive capture; `.nsys-rep` + `.sqlite` are retained for manual decompose.
3. `--fpm-run` points at THIS attribute run dir (`$rdir`), where the collector wrote `fpm_metrics_phase.csv` — NOT the parent fpm dir (which has no CSV and sends `_load_fpm` down a nonexistent nested `tp{T}_ep{E}_past{K}` fallback).
4. Boundary discard is CHRONOLOGICAL per `measure_run` (global step ordinal), NOT per `(bs,past)` shape — `collector/layerwise/vllm/nsys.py:_filter_boundary_discards`. Per-shape cohorting is wrong for continuous concurrent decode where every step is a unique `(decode_batch, mean_kv)`.

Also note: the collector can exit nonzero (e.g. 2) on a non-fatal per-request case failure even when the nsys capture succeeded, so the stage captures `collect_rc` without aborting and gates continuation on a `.nsys-rep` existing rather than on `rc==0`.

**Manual fallback (if the stage aborts post-capture).** The capture is the expensive part; recover it by hand:

```bash
"$NSYS_ROOT/bin/nsys" export --type sqlite --force-overwrite true \
  -o <run>/nsys/fpm_worker.sqlite <run>/nsys/fpm_worker.nsys-rep
python -m collector.layerwise.diagnostics.aic_fpm_attribute \
  --sqlite <run>/nsys/fpm_worker.sqlite --fpm-run <run> \
  --system h100_sxm --model Qwen/Qwen3-32B --tp 8 \
  --discard-first-n 3 --out <run>/decomposition.csv
```

For interpreting the resulting `decomposition.csv` (compute/comm/overhead terms, the cross-rank summed-vs-single-rank TP caveat, wall ≈ GPU-forward semantics), see the `close-layerwise-fpm-gap` analysis skill and AIC-1195.

## After Collection

Summarize `*_phase.csv` before comparing:

- `context`: group by `(ctx_tokens, ctx_kv_tokens)` for chunked-prefill correctness.
- `decode`: group by `decode_requests`, usually with mean KV near 1024 for the b1..64 decode sweep.
- `mixed`: keep rows as observed scheduler iterations; shapes are `(ctx_tokens, decode_requests, mean_decode_kv_tokens)`.

If collection fails after some traffic, still inspect copied partial outputs. The script should preserve collector CSVs before exiting.
