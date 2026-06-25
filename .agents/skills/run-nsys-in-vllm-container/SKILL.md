---
name: run-nsys-in-vllm-container
description: Run Nsight Systems inside a vLLM Docker container for layerwise or FPM profiling. Use when Docker image lacks nsys, when mounting host Nsight Systems into vllm/vllm-openai or Dynamo vLLM containers, or when validating that exported nsys sqlite files contain CUDA/CUPTI kernel traces rather than NVTX-only spans.
---

# Run Nsys In vLLM Container

## When The Image Lacks `nsys`

First check inside the target container:

```bash
docker run --rm --entrypoint /bin/bash IMAGE -lc 'command -v nsys || true; nsys --version || true'
```

If missing, mount host Nsight Systems and expose both the CLI and target libraries:

```bash
NSYS_HOME=/opt/nvidia/nsight-systems/2025.6.3
VLLM_CACHE_HOST="${VLLM_CACHE_HOST:-$HOME/.cache/aic-vllm}"
mkdir -p "$VLLM_CACHE_HOST/tilelang/tmp"
docker run --rm --gpus '"device=0"' --ipc=host \
  --entrypoint /bin/bash \
  -e PATH="$NSYS_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  -e LD_LIBRARY_PATH="$NSYS_HOME/target-linux-x64:${LD_LIBRARY_PATH:-}" \
  -e TILELANG_CACHE_DIR=/home/dynamo/.cache/vllm/tilelang \
  -e TILELANG_TMP_DIR=/home/dynamo/.cache/vllm/tilelang/tmp \
  -v "$NSYS_HOME:$NSYS_HOME:ro" \
  -v "$VLLM_CACHE_HOST:/home/dynamo/.cache/vllm" \
  -v "$VLLM_CACHE_HOST:/root/.cache/vllm" \
  -v "$PWD:/work/aiconfigurator" \
  -w /work/aiconfigurator \
  IMAGE \
  -lc 'nsys --version && python3 -m collector.layerwise.vllm.collect ...'
```

On this host, `/usr/local/cuda/bin/nsys` is only a wrapper. Prefer mounting the real Nsight tree under `/opt/nvidia/nsight-systems/<version>`.

## Validation

For AIC layerwise data, `nsys` must collect CUDA/CUPTI tables. NVTX-only traces are not valid even if the collector writes a CSV row.

After a run, inspect `profiles/status.jsonl`:

```bash
rg '"nsys_parse_succeeded"|"success"' RUN_DIR/profiles/status.jsonl
```

For default span-latency layerwise collection, valid traces should show:

- `meta.attribution_source` can be `nvtx_span`
- `meta.attributed_kernels` is nonzero
- success rows have nonzero `kernel_count`
- exported sqlite is usually much larger than an NVTX-only trace for the same run

For `--latency-source gpu` or `gpu_capped`, require CUPTI-backed attribution; if those modes fall back to NVTX spans, treat the row as invalid for GPU-kernel attribution.

On this host, prefer `/opt/nvidia/nsight-systems/2025.6.3` for CUDA 12.9 containers such as `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.2.0`. A smoke test with `/opt/nvidia/nsight-systems/2024.6.2` and `--capture-range=cudaProfilerApi` fired `cudaProfilerStart/Stop` but finalized with an empty `Generated:` line and no `.nsys-rep`; the same test with 2025.6.3 produced a valid report. Mounting 2024.6.2 into `vllm/vllm-openai:v0.20.1` made `nsys` runnable, but both with and without `LD_LIBRARY_PATH=$NSYS_HOME/target-linux-x64` the exported sqlite stayed NVTX-only. In these cases, do not use the CSV latency for layerwise data. Prefer a container/image with native Nsight/CUPTI support, or keep iterating on container privileges/CUPTI injection until the validation shows `attribution_source=cupti`.

## Attributed-FPM Path (Dynamo Worker Under Nsys)

The FPM real-workload attribute path (`STAGES=attribute` in `collector/layerwise/reproduce_layerwise_fpm.sh:stage_attribute`) runs the vLLM worker as `python3 -m dynamo.vllm` under nsys. It needs different nsys handling than the span-latency layerwise collector above.

### The worker image has no `nsys` on PATH -- bind-mount the host Nsight install

The Dynamo vLLM worker image does not ship `nsys`. Instead of relying on container PATH, the FPM shell bind-mounts the host Nsight tree into the worker and points at the host binary:

- `--nsys-profile-worker` / `--nsys-cuda-profiler-window` are threaded `collector/layerwise/fpm/collect.py` (arg parsing) -> `collector/layerwise/fpm/docker.py:build_collect_command` (appends `--nsys-profile-worker` and `--nsys-cuda-profiler-window <window>`) -> `collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh`.
- `NSYS_BIN` (the nsys binary path inside the worker, e.g. `$NSYS_ROOT/bin/nsys`) and `NSYS_HOST_DIR` (the Nsight root to mount, e.g. `$NSYS_ROOT`) are read by `collect_fpm_metrics.sh` (defaults: `NSYS_BIN=nsys`, `NSYS_HOST_DIR=`). When `NSYS_HOST_DIR` is set it is bind-mounted read-only into the worker (`-v "${NSYS_HOST_DIR}:${NSYS_HOST_DIR}:ro"`); if `NSYS_HOST_DIR` is empty and `NSYS_BIN` is an absolute path, the shell derives the host dir from `dirname "${NSYS_BIN}"`.
- The driver propagates these as plain env (`NSYS_BIN=$NSYS_ROOT/bin/nsys NSYS_HOST_DIR=$NSYS_ROOT`), and `collect.py` runs the inner shell with `os.environ.copy()` (`collector/layerwise/fpm/collect.py:144`), so the env reaches `collect_fpm_metrics.sh` without explicit flags.

### Per-step NVTX is a Dynamo-side `execute_model` wrapper, not the counter-mode marker

For real multi-request FPM traffic, use `collector/layerwise/vllm/dynamo_step_marker.py`, NOT `vllm_step_marker.py`:

- `dynamo_step_marker.py` monkeypatches `vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model` (the GPU forward) and reads the REAL per-step batch state from the `scheduler_output` arg, emitting labels `bench_step::N<step:07d>::bs<decode_batch>::past<mean_kv:06d>`.
- It is injected via `collector/layerwise/vllm/sitecustomize.py` when `LAYERWISE_DYNAMO_STEP_MARKER=1` and the repo dir is on `PYTHONPATH` (so the interpreter imports `sitecustomize` at startup inside the `python3 -m dynamo.vllm` worker).
- Do NOT use `vllm_step_marker.py` here: in the Dynamo launch context it is never imported, and its counter-mode label assumes `isl=1` single-stream (`past_kv = n - 1`), which is wrong for a real multi-request workload. (See the docstring in `dynamo_step_marker.py` for the Phase-0 finding.)

### Capture is session-gated, not `-c cudaProfilerApi`

`collect_fpm_metrics.sh` defaults `NSYS_PROFILE_TRAFFIC_ONLY=1` and gates the capture with `nsys start --session` / `nsys stop --session` around the real workload, NOT `--capture-range=cudaProfilerApi`. Consequences:

- The `--nsys-cuda-profiler-window` window only LABELS steps (via the NVTX marker); it does NOT bound capture. ALL traffic steps are captured.
- Trace size is therefore governed by OSL / number of requests. Keep OSL short to bound `.nsys-rep` / `.sqlite` size -- box disk is the binding constraint on 8xh100-layerwise.

### Export `.nsys-rep` -> `.sqlite` before decompose

The decompose reads a `.sqlite`, not the raw `.nsys-rep`. The driver (and the manual fallback) export with:

```bash
nsys export --type sqlite --force-overwrite true -o OUT.sqlite RUN.nsys-rep
python -m collector.layerwise.diagnostics.aic_fpm_attribute --sqlite OUT.sqlite --fpm-run <attribute dir> --tp 8 ...
```

`aic_fpm_attribute.py` loads the sqlite via `collector/layerwise/diagnostics/analyze_nsys_comm_overlap.py`. If `stage_attribute` aborts after a valid capture, run these two steps manually rather than re-collecting.

## Notes For This Repo

- The vLLM collector launches `nsys profile` from inside the scheduler process, so `nsys` must be visible inside the container that runs `python -m collector.layerwise.vllm.collect`.
- Keep the collector CLI short; common vLLM extras such as `--skip-mm-profiling`, `--limit-mm-per-prompt {"image":0,"video":0}`, and `--generation-config vllm` are added by the collector.
- Do not expose HF tokens in process listings. Prefer `HF_TOKEN_FILE=~/hf.token` in wrapper scripts, or pass token environment variables without printing `ps -ef` command lines.
