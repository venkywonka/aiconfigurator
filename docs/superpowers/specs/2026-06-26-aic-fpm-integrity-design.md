# AIC-FPM Attribution Integrity — Design

- **Date:** 2026-06-26
- **Branch (target):** `venky/aic-fpm-integrity` (off `venky/layerwise`)
- **Status:** Approved design — implementation pending
- **Author:** venky + Claude (council-verified: codex, opus, grok)

## Motivation

A council code-read of the layerwise attributed-FPM pipeline surfaced three alleged
correctness bugs. This design records the **investigation of their validity** (all three
confirmed against the current working tree, independently re-verified by 3 models) and the
**fixes**. These are attribution-pipeline correctness bugs — they bite on any runner (Brev,
K8s, local). They are independent of the teleport work and belong on their own branch.

Two of the three are live and **compound** into a silent-success failure mode. The third is
real-but-dormant (wrong-by-construction, no numeric effect on today's pareto). The council
also found **two additional defects** that feed the same failure modes.

## Validity & severity verdict

| # | Claim | Verdict | Severity |
|---|---|---|---|
| 1 | Markers fail open (should fail closed) | **VALID** | HIGH |
| 2 | `.done` stamped after a decompose warning | **VALID** | HIGH |
| 3 | Hardcoded `max_num_batched_tokens=8192` vs run's 2048 | **VALID defect** | LOW today / latent footgun |
| 3b (found) | `write_decomposition_csv` silently returns on empty rows | **VALID** | feeds #2 |
| 3c (found) | `_read_runtime_config` never reads the real config (key-format bug) | **VALID** | makes #3 fix cosmetic if unfixed |

All line references below are against the **current working tree** (which includes
substantial uncommitted WIP in `aic_fpm_attribute.py` / `reproduce_layerwise_fpm.sh`; the WIP
does not touch any defect site). Re-confirm line numbers at implementation time.

## Evidence (verified)

### Claim 1 — markers fail open
- `dynamo_step_marker._install()` (`collector/layerwise/vllm/dynamo_step_marker.py:169-173`):
  `except Exception` → `print(...)` → continue. Runs at import (`:176`), gated on
  `LAYERWISE_DYNAMO_STEP_MARKER=1` (`:124`).
- `sitecustomize._try_import()` (`collector/layerwise/vllm/sitecustomize.py:16-22`):
  `except Exception` → `print(...)` → continue. Invokes the dynamo marker at `:31-32`.
- `LAYERWISE_DYNAMO_STEP_MARKER=1` is set in production profiled workers
  (`collector/layerwise/fpm_ground_truth/collect_fpm_metrics.sh:731`).
- Result: when markers are explicitly required and the patch/import fails (vLLM API drift,
  wrong PYTHONPATH/version), you get a `.nsys-rep` with **zero `bench_step::` ranges** — an
  unattributable but successful-looking expensive run.

### Claim 2 — `.done` stamped after warn
`stage_attribute()` in `reproduce_layerwise_fpm.sh` reaches `mark_done` without a success
check via three paths:
- `:627` — no sqlite → `warn ...; mark_done "$unit"; continue`
- `:651` — decompose `|| warn ...` (defeats `set -e`)
- `:653` — **unconditional** `mark_done` after decompose
The only hard gate (`:606-607` `die`) fires only when **no** `.nsys-rep` exists.

### The compounding interaction (1 + 2)
A fail-open marker yields a `.nsys-rep` that **exists** → passes the `:606` gate → exports to
sqlite fine → decompose has no `bench_step::` ranges. Two observed sub-modes (both end at a
stamped `.done`, then `is_done` skips on re-run; only `FORCE=1` escapes):
- **Empty rep** (profiler-gated capture, marker never called `cudaProfilerStart`):
  zero attributable rows → `write_decomposition_csv` returns early (defect 3b) → CLI exits 0
  → `|| warn` never fires → `:653` stamps `.done`.
- **Rep with kernels but no ranges:** `analyze_nsys_comm_overlap.py:145` raises `RuntimeError`
  → decompose fails → `:651` warns → `:653` stamps `.done`.

### Claim 3 — hardcoded MNBT=8192
- `aic_fpm_attribute.py:418`: `rc = RuntimeConfig(vllm_max_num_batched_tokens=8192,
  vllm_max_num_seqs=None)`. Driver default is 2048 (`reproduce_layerwise_fpm.sh:188`).
- **Inert for decode (headline):** `_get_decode_step_latency`
  (`src/aiconfigurator/sdk/backends/vllm_backend.py:1383`) reads `vllm_max_num_seqs` (`:1424`)
  but **never** `vllm_max_num_batched_tokens`. Verified empirically by the council: decode
  predictions are bit-identical at 8192 vs 2048.
- **Inert for the 2048-context grid — but not for the reason "single chunk".** MNBT is also a
  **CTX row-lookup key** (`compare_aic_layerwise_fpm.py:1774-1778, 1860-1869`), not just a
  chunk divisor. The CSV has CTX rows only at `mnbt=2048`; an 8192 lookup *misses* the key →
  falls back to the non-MNBT index → resolves to the same 2048 rows. Inertness is a
  **data-contingent accident of the single-mnbt grid**, confirmed bit-identical by the
  council. Adding an 8192 CTX row would make it diverge **at ctx=2048** with no test catching
  it.
- **Where it bites:** for `ctx_tokens ∈ (2048, 8192]`, 8192 goes **off-grid and drops the
  shape** (returns None) while 2048 chunks it correctly. So the defect, where active, is worse
  than "wrong numbers" — it silently drops shapes.

### Defect 3c — `_read_runtime_config` never reads the real config
`aic_fpm_gap._read_runtime_config()` (`:614-625`) reads top-level `max_num_batched_tokens` or
a nested `scheduler_config` **object**. The real `effective_vllm_config.json` uses a
**flattened dotted key** `"scheduler_config.max_num_batched_tokens"` (verified against
`fpm_golden_runs/.../effective_vllm_config.json`). So the helper always falls through to its
`2048/128` constants and never actually reads the run config. Calling it as-is for the
claim-3 fix would be cosmetic.

## Design

Branch `venky/aic-fpm-integrity` off `venky/layerwise`. Three TDD commits (test → red → fix →
green). **Supersedes** `slop/teleport-fpm-feasibility/plan-1-runtime-process.md:326`, which
proposes a `raise RuntimeError` that the council proved is re-swallowed (see Fix 1).

### Fix 1 — markers fail closed when explicitly required
When `LAYERWISE_DYNAMO_STEP_MARKER=1`, a patch/import failure must **abort the worker**, not
warn-and-continue.

The two fail-open layers are **in series on one load path**: `_try_import("dynamo_step_marker")`
→ `dynamo_step_marker._install()` (at import). Because `_try_import` catches `except Exception`,
a plain `raise RuntimeError(...)` inside `_install` **is re-swallowed** and the worker still
runs unmarked. So:

- `dynamo_step_marker._install()`: on failure when env=1, raise a **`BaseException`** that
  escapes `except Exception` — `SystemExit` (preferred; testable) or `os._exit(1)`.
- `sitecustomize._try_import()`: add a `required: bool = False` param. The dynamo-marker call
  passes `required=True` and **re-raises** (does not swallow). The scheduler-timing and
  step-marker calls keep `required=False` (unchanged).
- Both layers cooperate so neither can silently re-open the hole.

**Files:** `dynamo_step_marker.py`, `sitecustomize.py`.

### Fix 2 — `.done` is a fail-closed gate (`die`)
- New `collector/layerwise/diagnostics/assert_attribution_valid.py`: a diagnostic gate that
  asserts (a) sqlite has ≥1 CUPTI kernel row, (b) sqlite has ≥1 `bench_step::` NVTX row, and
  (c) `decomposition.csv` has ≥1 data row. Diagnostic (3 checks) over a single
  `decomposition.csv`-nonempty check because it pinpoints which stage failed (no kernels =
  profiler window; no ranges = marker; rows-but-empty-csv = join/config). Non-zero exit / raise
  on failure.
- Invoke it before `mark_done` in `stage_attribute()`; on failure → **`die`** (chosen
  behavior: abort the whole campaign loudly; artifacts already on disk are retained for manual
  decompose).
- Change the `:627` no-sqlite path from `mark_done` to `die`.
- Make `write_decomposition_csv` signal failure on empty input (raise / non-zero) instead of
  returning silently (defect 3b), so the gate and `|| warn` actually fire.

**Files:** new `assert_attribution_valid.py`, `reproduce_layerwise_fpm.sh`,
`aic_fpm_attribute.py` (`write_decomposition_csv`).

### Fix 3 — read MNBT from the run's effective config
1. Fix `aic_fpm_gap._read_runtime_config()` to read the flattened
   `"scheduler_config.max_num_batched_tokens"` / `"scheduler_config.max_num_seqs"` keys
   (keep the existing top-level / nested-object reads as fallbacks), and **warn loudly** when
   it falls back to the `2048/128` constants.
2. In `aic_fpm_attribute._main()`, build `rc` from `_read_runtime_config(args.fpm_run)` for
   `vllm_max_num_batched_tokens` instead of the literal `8192`. Keep `vllm_max_num_seqs=None`
   (correct for the layerwise track). Warn-and-fallback only if the config is unreadable.

**Files:** `aic_fpm_gap.py`, `aic_fpm_attribute.py`.

## Test plan (one focused test per fix; TDD)

- **F1** (`tests/.../test_dynamo_step_marker_failclosed.py`): with `LAYERWISE_DYNAMO_STEP_MARKER=1`
  and a missing/broken patch target, assert the failure propagates **past an
  `except Exception`** (simulating `_try_import`) — i.e. the worker *process* would abort, not
  just that `_install` raises. Assert env-unset / `!=1` stays a no-op. Assert the `required=True`
  `_try_import` path re-raises while `required=False` still swallows.
- **F2** (`tests/.../test_assert_attribution_valid.py`): the gate returns non-zero / raises for
  (a) no CUPTI kernel rows, (b) no `bench_step::` rows, (c) empty `decomposition.csv`; passes on
  a valid triple. Tiny synthetic sqlite + csv fixtures. Also: `write_decomposition_csv([])`
  signals failure.
- **F3** (`tests/.../test_read_runtime_config.py`): `_read_runtime_config` returns 2048 from a
  flattened-key JSON and from a real golden fixture, 8192 from an `8192` JSON, and warns on a
  missing file.

## Out of scope (verified, not bugs)

- `vllm_max_num_seqs=None` (`aic_fpm_attribute.py:418`) is **correct** for the layerwise track
  (`aic_fpm_gap.py:500-502`); GEN rows carry empty `max_num_seqs`, so `None` selects the
  primary index. The nearby "no max_num_seqs index" comment is **stale** (the CSV does have the
  column, values 128/4) but behavior is right. Optional: a one-line comment-accuracy tweak.
- `--fpm-run` = the attribute run dir (`reproduce_layerwise_fpm.sh:646`) is **intentional**
  (docstring `:631-634`; that dir holds the clean-lane `fpm_metrics_phase.csv`). *Note:* the
  Python arg-help "clean golden FPM run dir" contradicts the shell usage — optional one-line
  help-text correction, no behavior change.

## Working-tree / branch sequencing risk

`venky/layerwise` has substantial uncommitted WIP, including in `aic_fpm_attribute.py` (+138)
and `reproduce_layerwise_fpm.sh` (+89) — the same files Fix 2 and Fix 3 edit. The WIP does
**not** touch any defect site, but it interleaves in those files. Before implementation we must
decide how to sequence: (a) commit/stash the WIP onto `venky/layerwise` first, then branch
clean; (b) branch carrying the WIP and stage integrity hunks selectively. This is the open
decision at the implementation gate.

## Acceptance criteria

- F1/F2/F3 tests pass (and demonstrably fail before their fixes).
- With `LAYERWISE_DYNAMO_STEP_MARKER=1` and a forced patch failure, the worker aborts (no
  silent unmarked `.nsys-rep`).
- A run that produces no kernels / no `bench_step` ranges / empty decomposition `die`s before
  `.done` is stamped; the unit re-runs on the next pass without `FORCE=1`.
- `aic_fpm_attribute` builds `rc` from the run's effective config; with a non-2048 MNBT config,
  the predictor uses that value (and warns on fallback).
- The standard 2048-context pareto decomposition is numerically unchanged by Fix 3 (inert
  today — regression guard).
