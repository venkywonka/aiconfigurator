---
name: close-layerwise-fpm-gap
description: Diagnose and close AIC/vLLM layerwise prediction gaps against Dynamo/vLLM ForwardPassMetrics ground truth. Use when the user asks why layerwise differs from FPM, asks to compare AIC layerwise vs FPM for context/decode/mixed phases, asks whether to recollect FPM/layerwise data, asks about compile/CUDA graph/deployment parity mismatches, or asks to debug TP1/TP2/TP8/Qwen3 latency accuracy.
---

# Close Layerwise-FPM Gap

## Core Rule

Treat FPM as the ground truth for complete vLLM forward-pass iterations. Treat layerwise as a decomposed approximation that is only valid after proving the collector and FPM deployment configs match.

Do not recollect FPM until existing artifacts have been checked. FPM is slow and partial runs can still contain useful `*_phase.csv`, `*_detail.csv`, and `*_workload.csv` rows.

Use raw AIC-vs-FPM error as the accuracy metric. Post-hoc scaled/multiplier columns are diagnostics only; do not report them as predictive accuracy, hide calibration inside a global multiplier, or add target-model FPM deltas as prediction inputs.

Read [references/today-learnings.md](references/today-learnings.md) when working on Qwen3-32B/B300/vLLM 0.20.1, FP8, TP1/TP2/TP8, fused allreduce RMS, or context/decode mismatch analysis.

## Workflow

1. **Inventory artifacts before running anything.**
   - Check active Docker/GPU state with `docker ps` and `nvidia-smi`.
   - Find FPM files: `*_fpm_*_phase.csv`, `*_detail.csv`, `*_workload.csv`, `*_effective_vllm_config.json`, `*_metadata.json`.
   - Find layerwise files: `*_vllm_context_*.csv`, `*_vllm_decode_*.csv`, profile `status.jsonl`, metadata JSON, and Nsight SQLite reports.
   - Report which TP sizes have both FPM and layerwise data. Do not imply a TP comparison exists when only layerwise was collected.

2. **Validate deployment parity before interpreting error.**
   Compare FPM and layerwise effective config metadata:
   - `vllm_version`
   - `parallel_config.tensor_parallel_size`
   - `model_config.dtype`
   - `cache_config.cache_dtype`
   - `scheduler_config.max_num_batched_tokens`
   - `scheduler_config.max_num_seqs`
   - `compilation_config.mode`
   - `compilation_config.cudagraph_mode`
   - `compilation_config.custom_ops`
   - key `compilation_config.pass_config` fusions

   Any mismatch can dominate the gap. Common culprits are compile mode, CUDA graph mode, max sequence/batch-token settings, default-vs-forced KV dtype, and FP8 fusion settings.

3. **Normalize layerwise rows exactly once.**
   - Map `ctx` to `CTX`, `gen` to `GEN`.
   - Map `attn_tp` to `tp_size`.
   - Map `new_tokens` to `seq_len_q`.
   - Map `past_kv` to `seq_len_kv_cache`.
   - Divide multi-layer context rows by the collected `target_layer_count` before loading into AIC, exactly once.
   - Do not divide one-layer decode rows.
   - Keep `rms_latency_ms` with the same division as `latency_ms`.

4. **Compare through AIC, not a hand formula, unless debugging a single component.**
   - Set `AIC_VLLM_USE_LAYERWISE=1` before importing/running AIC comparison code.
   - If the repo AIC CSV does not yet contain the target rows, create a temporary systems root under the artifact directory and load `PerfDatabase(..., systems_root=temp_root)`.
   - Use AIC’s vLLM layerwise backend so context chunking, per-layer scaling, and fused RMS logic follow the code under test.
   - For TP1, comm should be zero; any gap is compute/runtime/collector/deployment parity.
   - For TP>1 decode, do not add generic allreduce blindly. vLLM default-compile decode may use fused or overlapped paths; current AIC pure-decode modeling intentionally avoids generic allreduce unless calibrated data supports it.

5. **Summarize FPM by scheduled shape.**
   - Context: group by `(ctx_tokens, ctx_kv_tokens, ctx_requests)` and use medians or trimmed means. Exclude mixed/decode rows when comparing pure context.
   - vLLM chunked 16k context appears as `8192,past=0` plus `8192,past=8192`.
   - Decode: group by `decode_requests`; filter `mean_decode_kv_tokens` to the intended KV window, usually around 1024 for the b1..64 sweep. Prefer the longest consecutive full-batch block where `decode_tokens == decode_requests`, and exclude tail rows after requests finish.
   - Mixed: compare only after context/decode are understood; shape is scheduler-dependent.

6. **Break down the error before proposing collection.**
   - Context TP1: compare FPM vs `layerwise_context * num_layers`; no comm exists.
   - Decode TP1: compare FPM vs `layerwise_decode * num_layers`; no comm exists.
   - TP>1 context: show compute and generic TP allreduce separately.
   - TP>1 decode: show layerwise compute after subtracting measured RMS, fused allreduce RMS estimate, and total. Do not use the old `isl=1023` workaround; AIC GEN lookup should query the KV length at the start of the decode step.
   - For Nsight-backed checks, compare wrapper span vs kernel/GPU time before blaming `gpu_capped`.

7. **Choose the next action conservatively.**
   - If metadata differs, first run a small targeted layerwise sanity pass matching FPM defaults.
   - If multi-layer extrapolation is suspicious, run a small full-depth or larger-slice point before broad recollection.
   - If FPM failed after preserving output, use complete phases that finished; rerun only the missing phase if needed.
   - If FPM and layerwise configs match and the gap persists, update AIC modeling or collector normalization before collecting a large grid. Do not "fix" a target model by feeding its own FPM residual back into the prediction.

## Reporting

Give the user a compact table with:

- phase and shape
- FPM latency
- AIC/layerwise predicted latency
- signed error and MAPE
- compute vs comm breakdown when TP>1
- exact caveats, especially missing TP FPM rows or failed mixed runs

State whether the evidence points to FPM noise, deployment parity mismatch, layerwise collector error, AIC normalization error, or a real modeling gap.

## Multi-track gap analysis (op-wise + layerwise)

Beyond the single-track layerwise comparison above, the canonical multi-track analyzer compares
ALL of AIC's predictor tracks against FPM through ONE per-step entry point (flip
`vllm_backend._USE_LAYERWISE` + swap the `database`/mode):

`collector/layerwise/diagnostics/aic_fpm_gap.py`  (report: `aic_fpm_gap_report.py`)

| track | source | notes |
|---|---|---|
| `layerwise` | layerwise CSV (compute-version) + comm tables | version-matched to FPM; **cal-off headline** (`_DECODE_COMPUTE_BATCH_CAL=0`) |
| `layerwise_cal0066` | + old 0.0066 decode batch-cal | counterfactual sensitivity (shows the removed, mis-tuned cal) |
| `opwise_silicon` | measured op-wise `PerfDatabase` (SILICON) | the op-wise model |
| `hybrid` | SILICON-with-empirical-fallback | `hybrid == opwise_silicon` ⇒ op coverage complete |
| `empirical` | analytic SOL/scale_factor | version-skew-immune |
| `sol` | pure roofline floor | bounds, not predicts |

It is **SKU/version-general** (`--system`, `--model`, `--compute-version`, `--comm-version`) and
**auto-detects both FPM layouts**: B300 TP-sweep (`tp{tp}_ep1_past4096/`) and the H100-SXM
concurrency sweep (`fpm/qwen32/c{conc}/`, points merged; decode bins span batches 1..conc, grouped
by batch via `build_summary_by_concurrency`).

```bash
python -m collector.layerwise.diagnostics.aic_fpm_gap \
  --system h100_sxm --model Qwen/Qwen3-32B \
  --compute-version 0.20.1 --comm-version 0.19.0 --workload-segment real \
  --fpm-run fpm_golden_runs/fpm_h100_qwen32_tp8_8k1k_pareto_<ts> --out-dir <out>
```

Outputs: `gap_summary.csv`, `gap_summary_by_concurrency.csv` (per decode batch = concurrency),
`compute_comm_decomposition.csv`, `gap_rows.csv`, plus `dashboard.html` + `verdict.md` (track ranking).

Interpretation (carries the Core Rule + step 6):
- Only the **layerwise** track is version-matched to FPM; op-wise/empirical/hybrid/SOL use the
  comm/op-DB version, so their error mixes model + version skew — disclose it, don't bury it.
- The headline forces `_DECODE_COMPUTE_BATCH_CAL=0`: the linear decode batch-cal is mis-tuned; on
  H100 cal-on inflates dense decode to ~30% MAPE (→79% at batch 128). `layerwise_cal0066` exists
  only to show that counterfactual.
- H100 finding (2026-06-24, dense Qwen3-32B, TP=8 8k/1k): op-wise SILICON/hybrid **4.0%** gen MAPE
  < layerwise(0.20.1) **7.9%** < empirical 28.9% < SOL 48.4% — op-wise wins on H100 (opposite of
  B300), mainly because layerwise over-predicts low-batch decode via the H100 comm-table fallback
  (`custom_allreduce` substituting for the missing fused `allreduce_rms`).

## Attributed-FPM (nsys gap decomposition) — VALIDATED

The multi-track analyzer above tells you *which track* is closest to FPM, but FPM is one number per
step (`wall` ≈ GPU forward via CPU–GPU overlap; no internal breakdown — see memory
`fpm-walltime-semantics`), so it cannot tell you *which term* of a track is wrong: compute model, comm
model, or effects AIC structurally can't see (compute–comm overlap, exposed-CPU/host overhead).
Attributed-FPM re-runs the FPM input space under nsys, attributes each captured step's GPU activity,
and decomposes the layerwise-vs-FPM gap per shape. Spec: `slop/fpm-nsys-attribution/spec.md`; full
chronological findings (Phase-0/Task-7 — this section reflects them): `slop/fpm-nsys-attribution/log.md`.

This section documents what ACTUALLY works after end-to-end validation on real Qwen3-32B TP=8. Earlier
drafts (pre-Phase-0) were wrong about two things that broke in practice — keep these corrections in mind:
- **Per-step NVTX comes from a DYNAMO-side hook, NOT `vllm_step_marker` reuse.** The FPM worker is
  `python3 -m dynamo.vllm`; `vllm_step_marker`'s counter-mode label derives `past_kv = n−1` (assumes
  isl=1 single-stream), which is WRONG for real multi-request FPM steps. The working marker reads REAL
  batch state from the live forward (see Collection below).
- **Capture is SESSION-gated, NOT `-c cudaProfilerApi` windowed.** The harness uses
  `NSYS_PROFILE_TRAFFIC_ONLY` session bracketing, so ALL traffic steps are captured; OSL/requests
  govern trace size, and the window only LABELS which step ordinals to keep. The reducer/analysis
  selects steady state.

### 1. Two-lane model + the exact decomposition identity

**Two lanes, joined by shape. The nsys lane is composition only, NEVER a timing source:**
- **Clean lane (authoritative timing):** the golden FPM run = source of truth for per-shape `wall`.
  Reused, not re-run. (The first demonstrated run joined to the attribute run's own lightly-perturbed
  FPM wall; joining to the separate golden run is the two-lane-purity refinement — see Caveats.)
- **Profiled lane (composition only):** a new nsys capture. Supplies the granularity of GPU work —
  CUPTI kernel *durations* are accurate, but the profiled *wall/span* is perturbed and untrusted.

Per shape, the profiled lane maps directly to the existing `analyze_sqlite` per-step columns:
`gpu_compute = compute_gpu_us`, `gpu_comm = comm_gpu_us`, `gpu_busy = total_union_us` (union of kernel
intervals, accounts for overlap). Derived: `overlap = (gpu_compute + gpu_comm) − gpu_busy` and the
cross-lane residual `overhead = wall − gpu_busy` (real wall not explained by GPU kernels).

**Decomposition identity (exact, algebraic — not a fit):**
> aic_total − wall = (aic_compute − gpu_compute) + (aic_comm − gpu_comm) + overlap − overhead

`decompose_shape(...)` in `aic_fpm_attribute.py` emits these as `compute_err`, `comm_err`, `overlap`,
`-overhead`; they sum EXACTLY to `gap_ms = aic_total − wall`. (`aic_other` — scheduler/residual — is
≈0 for dense TP and folds into the compute split.) Each term names a cause: AIC compute-model error,
AIC comm-model error, the compute–comm overlap AIC double-counts (it sums them; the GPU overlaps), and
the engine overhead AIC structurally can't model.

### 2. Collection — the CORRECTED harness (what actually works)

**Per-step NVTX marker (DYNAMO-side hook on the forward).**
`collector/layerwise/vllm/dynamo_step_marker.py`:
- `_install()` monkeypatches `GPUModelRunner.execute_model` (the forward — NOT
  `InstrumentedScheduler.update_from_output`, which runs AFTER the forward and brackets only post-step
  CPU bookkeeping, capturing ~no kernels; that was the smoke-#5 failure). No dynamo source edit.
- `_decode_batch_and_kv(scheduler_output)` reads REAL per-step state: `decode_batch` = number of
  non-context (decode) scheduled requests; `mean_kv` = mean `num_computed_tokens`. `_bench_step_label`
  emits `bench_step::N{step:07d}::bs{decode_batch}::past{mean_kv:06d}` — round-trips against
  `parse_nsys_step_sweep._BENCH_STEP_RE`. (Verified on real capture: `past` GROWS with KV growth, never
  matches counter-mode `n−1`, so no contamination from the vLLM marker.)

**Injection.** `collector/layerwise/vllm/sitecustomize.py` fires in the spawned EngineCore subprocs
(patches the class pre-instantiation) when `LAYERWISE_DYNAMO_STEP_MARKER=1`. The repo is bind-mounted
ro at `/aic-src` and put on the worker `PYTHONPATH` (`/aic-src/collector/layerwise/vllm:/aic-src`).
`collect_fpm_metrics.sh` sets these when `NSYS_PROFILE_WORKER=1`. Do NOT also set `LAYERWISE_STEP_MARKER`
in the FPM worker env — that would make `vllm_step_marker` double-wrap `execute_model`.

**Host nsys mounted into the worker.** The ai-dynamo `vllm-runtime` image has NO nsys on PATH (a bare
`nsys` exits 127). The host Nsight install is bind-mounted ro into the worker at the same absolute path
and the worker execs the absolute binary. The driver exports `NSYS_BIN=$NSYS_ROOT/bin/nsys` +
`NSYS_HOST_DIR=$NSYS_ROOT`; `collect.py` forwards env (`os.environ.copy()`) to
`collect_fpm_metrics.sh`, which adds the bind mount.

**Flag threading (the chain that the driver actually walks).** The driver stage runs
`python3 -m collector.layerwise.fpm.collect` → `fpm/docker.py:build_collect_command` → `bash
collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh`. The nsys flags had to be added to
`fpm/collect.py` argparse and threaded through `fpm/docker.py:build_collect_command` (before the `--`
extra-vllm-arg separator) — not just to the shell. So `--nsys-profile-worker` /
`--nsys-cuda-profiler-window` exist on all three: `fpm/collect.py`, `fpm/docker.py`,
`collect_fpm_metrics.sh` (the last maps `--nsys-cuda-profiler-window` → `LAYERWISE_CUDA_PROFILER_WINDOW`
in the worker env).

**Capture gating = SESSION, not `-c cudaProfilerApi`.** `collect_fpm_metrics.sh` uses
`NSYS_PROFILE_TRAFFIC_ONLY` session bracketing, which captures ALL traffic steps; OSL and request count
govern trace size (keep OSL short to bound it). The `--nsys-cuda-profiler-window` value only LABELS the
step ordinals to keep — the reducer/analysis (below) selects steady state and discards drained boundary
steps. (Context: `_cuda_profiler_call` in `collector/layerwise/vllm/worker.py` does
`torch.cuda.synchronize()` before EVERY `cudaProfilerStart`/`Stop`, so per-step fencing would drain the
pipeline at each boundary and destroy the CPU–GPU overlap we measure — that is why per-step fencing is
the anti-pattern. The windowed `_parse_profiler_window` / `_advance_profiler_window` in
`vllm_step_marker.py` / `dynamo_step_marker.py` exist for the true `cudaProfilerApi` window, but the
validated FPM path runs session-gated.)

**nsys export.** nsys writes a `.nsys-rep`; `analyze_sqlite` needs a `.sqlite`. The `attribute` stage
exports `.nsys-rep` → `.sqlite` (`nsys export --type sqlite`) before reduction; the manual fallback
runs the same `nsys export`.

### 3. Reduce + decompose

- **Reduce: `collector/layerwise/diagnostics/analyze_nsys_comm_overlap.py:analyze_sqlite`** — per-step
  rows keyed `(step, batch_size, past_kv, measure_run)` with `compute_gpu_us` / `comm_gpu_us` /
  `total_union_us` (microseconds). It reads NVTX range text INLINE from `NVTX_EVENTS.text`
  (`SELECT text, start, end, globalTid FROM NVTX_EVENTS WHERE text IS NOT NULL`), NOT via a
  `textId → StringIds` join — a count query that assumes `textId` will wrongly return 0 ranges (the
  parser is right). Per-step NVTX correlation + the cuda-graph `originalGraphNodeId` JOIN for cudagraph
  decode live in `collector/layerwise/common/parse_nsys_step_sweep.py`.
- **AIC split: `collector/layerwise/diagnostics/aic_fpm_gap.py`** — `_split_latency` +
  `predict_context_breakdown` / `predict_decode_breakdown` expose AIC's layerwise compute/comm split
  (compute = `*_layerwise`, comm = allreduce/alltoall collectives, other = scheduler/residual);
  loss-free (`compute + comm + other == predict_*()`'s total).
- **Join/decompose: `collector/layerwise/diagnostics/aic_fpm_attribute.py`** (a thin 3-way joiner, NOT
  a reducer):
  - `aggregate_profiled_by_shape(..., ranks=tp)` reduces `analyze_sqlite` rows to per-shape composition
    (us→ms) after `nsys.py:_filter_boundary_discards` drops the first N steps of each
    `(batch_size, past_kv, measure_run)` cohort. **Per-rank normalization (HARD for TP>1):**
    `analyze_sqlite` SUMS kernel durations across ALL captured ranks, so `gpu_compute`/`gpu_comm` are
    divided by `ranks` (each rank does ~1/ranks of compute + one allreduce per collective), while
    `gpu_busy`/union is left as-is (ranks run in wall-clock lockstep, so the union ≈ one rank). Without
    `--tp`, per-term compute/comm errors are not apples-to-apples with the single-rank `wall`/`aic`.
  - `run_decode_attribution(...)` joins profiled composition ⊕ clean FPM `wall` ⊕ AIC breakdown per
    decode shape and emits one `decompose_shape` dict each; `write_decomposition_csv(...)` writes a
    stable-column CSV. Timing always comes from the clean lane; the profiled lane only supplies
    composition.
  - **KV off-grid snapping:** the layerwise GEN grid is exact-lookup, so before
    `predict_decode_breakdown` the integer `past_kv` is snapped to the nearest collected layerwise GEN
    KV via `_nearest_available_generation_kv(...)`; off-grid KV would otherwise miss and silently drop
    the shape.
  - **Decode shape-key binning:** the profiled lane keys decode by `(int batch, int past_kv)` where
    `past_kv = round(mean(num_computed_tokens))`; the clean FPM lane keys by
    `(decode_requests:int, mean_decode_kv_tokens:FLOAT)`. `_bin_fpm_wall_to_profiled_key()` re-keys the
    FPM float to `(int(batch), int(round(mean_kv)))` with the SAME `round()` the marker uses (collisions
    aggregated by mean) so the float-vs-int key spaces actually join.

### 4. The EXACT working invocation

On the box (`8xh100-layerwise`, real 8×H100 SXM, TP=8), inline pattern — activate the host venv (driver
calls bare `python3`), point at the box Nsight 2025.3.2, keep OSL short to bound the trace, run the
`attribute` stage. `OUT_ROOT` MUST be local ext4 (nsys lock-traps on SMB/NFS):

```bash
cd /home/gvenkatarama/.codex/worktrees/6c30/aiconfigurator
source /home/ubuntu/.venv/bin/activate
STAGES=attribute \
SYSTEM=h100_sxm MODEL=Qwen/Qwen3-32B TP=8 \
OUT_ROOT=/home/ubuntu/aic-h100-repro \
NSYS_VERSION_DIR=2025.3.2 NSYS_ROOT=/opt/nvidia/nsight-systems/2025.3.2 \
OSL_MEAN=256 OSL_MAX=256 \
PARETO_CONCURRENCY="4" \
ATTRIBUTE_WINDOW="60-90" ATTRIBUTE_DISCARD_N=3 \
bash collector/layerwise/reproduce_layerwise_fpm.sh
```

The stage, per model × pareto point: (a) runs the FPM real workload under nsys session capture (via
`collect_fpm_metrics.sh --nsys-profile-worker --nsys-cuda-profiler-window "$ATTRIBUTE_WINDOW"`); the
unit succeeds if a `.nsys-rep` was produced even when `collect.py` exits non-zero (capture-then-tolerate,
so a tiny-request abort under `set -e` doesn't lose the capture); then (b) exports `.nsys-rep` → `.sqlite`
and decomposes via `aic_fpm_attribute` (`--tp "$TP"` → per-rank normalization is automatic;
`--discard-first-n "$ATTRIBUTE_DISCARD_N"`; `--fpm-run` = the existing clean FPM run dir). A decompose
failure (e.g. AIC has no layerwise data for the model) warns + retains the `.nsys-rep`/`.sqlite` for
manual decompose instead of aborting. Knobs: `ATTRIBUTE_WINDOW` (default `100-115`, `lo-hi[,lo-hi...]`
step ordinals) and `ATTRIBUTE_DISCARD_N` (default `3`).

**Manual export + decompose fallback** (the documented path used for the first real 32B result, when the
stage aborted before part (b)):

```bash
"$NSYS_ROOT/bin/nsys" export --type sqlite --force-overwrite true \
  -o fpm_worker.sqlite fpm_worker.nsys-rep
python -m collector.layerwise.diagnostics.aic_fpm_attribute \
  --sqlite fpm_worker.sqlite \
  --fpm-run <clean_fpm_run_dir> \
  --system h100_sxm --model Qwen/Qwen3-32B --tp 8 \
  --discard-first-n 3 --out decomposition.csv
```

### 5. First result + interpretation (the payoff)

First real attribution — Qwen3-32B TP=8 decode, conc4, 249 decode shapes, per-rank normalized
(`slop/fpm-nsys-attribution/runs/decomposition_qwen32_c4_perrank.csv`):

- **HEADLINE: `aic_total − wall` = +10.5%** (mean signed / MAPE). AIC layerwise OVER-predicts decode
  (reproduces the known ~8% decode over-prediction direction).
- **Decomposition (ms/step, per-rank):**
  `compute_err = +1.81` (AIC over-estimates per-rank compute 6.91 vs measured 5.10, ~+35% — DOMINANT)
  · `comm_err = −0.61` (AIC UNDER-estimates comm 0.37 vs measured 0.98 — the `custom_allreduce`
  fallback undercounts the real `allreduce_rms` kernels) · `overlap = −0.46` · `−overhead = −0.04`.
  Identity exact (sum = +0.69 ms = `gap_ms`).
- **`overhead ≈ 0`** → `wall ≈ gpu_busy` for steady-state decode → empirically confirms the
  wall≈GPU-forward-proxy semantics (no exposed host residual at decode; memory `fpm-walltime-semantics`).

**Interpretation (overturns the prior hypothesis):** the +10.5% over-prediction is driven by the AIC
COMPUTE model over-estimate, PARTIALLY MASKED by a comm UNDER-estimate — NOT by the comm fallback
over-predicting. The net gap is the residual of two opposing errors, invisible without attribution. The
`allreduce_rms` parquet coverage gap (h100/0.19.0) is now quantified: −0.61 ms/step undercount = an AIC
data-coverage gap, not a structural comm model error.

**Caveats:** `/ranks` assumes balanced dense TP (valid for dense 32B); the demonstrated `wall` was this
run's own lightly-nsys-perturbed FPM, not the separate golden run (two-lane-purity refinement pending);
single concurrency point; comm = `custom_allreduce` fallback. A deviceId=0-exact profiled filter would
replace the `/ranks` approximation. Remaining research refinements (not blockers): multi-concurrency
sweep (c1..c128) for the low-C-vs-high-C decode comm-err trend + the high-C context overhead puzzle;
join to the golden run's clean wall; HYBRID-mode AIC comm.

### 6. Confidentiality + bi-repo

**Confidentiality (HARD).** aiconfigurator is PUBLIC Apache-2.0. The loi-deep submodules (`dlb`,
`dlb-ai-agent`, `trtllm-agent-toolkit`, loi skills, `aibroom`) are confidential and **learn-from-only**:
never copy their source/docstrings/schema into this repo and never take a runtime/build dependency on
them. Only public Nsight facts (nsys CLI flags, the CUPTI/`NVTX_EVENTS` table schema) may inform new
code, cited in comments. `dlb` is permitted ONLY as a dev-time validation oracle invoked from `slop/`
(`slop/fpm-nsys-attribution/dlb_oracle_check.py`), never entering the public dependency graph. A
confidentiality grep gate over `collector/ src/ tests/` for `loi-deep`/`dl-inference-bridge`/`import dlb`
must stay CLEAN.

**Bi-repo.** Real runs execute on the Brev box `8xh100-layerwise`; the worktree is push-only synced via
`slop/birepo-brev-8xh100-layerwise/sync_birepo.sh` (git-stash + namespace push-only; verify MATCH after
each sync). Box facts: Nsight 2025.3.2 at `/opt/nvidia/nsight-systems/2025.3.2`; host venv at
`/home/ubuntu/.venv` (activate first — driver calls bare `python3`); `OUT_ROOT` on local ext4 (nsys
lock-traps on SMB/NFS); clear leftover `dynamo-fpm-*` containers with `docker stop` (`docker rm -f` is
permission-gated).
