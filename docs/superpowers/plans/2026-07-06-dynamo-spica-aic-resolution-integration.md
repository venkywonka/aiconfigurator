# Dynamo Mocker and Spica AIC Resolution Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let opt-in Mocker/Spica evaluations pass the exact scheduled batch to AIC, resolve all missing perf points on demand, and reject unscorable candidates cleanly while preserving the current pure-Rust AIC path by default.

**Architecture:** Mocker carries a lossless `ConcreteBatchDescriptor` alongside its existing aggregate timing inputs. The ordinary AIC configuration still uses `RustAicCallback` and reproduces today's aggregate results. An explicit resolution configuration selects a dedicated Python-backed callback, whose `AicSession` performs the existing Python operation walk with concrete vectors in op kwargs and one callback-local resolution cycle. GPU collectors remain in subprocesses. Callback errors propagate as structured perf-model failures to replay and then Spica; they never become a fallback or mixed score.

**Tech Stack:** Rust, serde, anyhow/thiserror, PyO3, Python 3.10+, Pydantic, AIC resolution/collector runtime, Dynamo Mocker/Replay, Spica, pytest, cargo nextest.

---

## Source prerequisites and repository boundary

Read these first:

- AIC `docs/plans/2026-07-06-dynamic-lazy-perf-collection-design.md`
- AIC `docs/superpowers/plans/2026-07-06-aic-lazy-perf-core.md`
- AIC `docs/superpowers/plans/2026-07-06-aic-hardware-aware-collector-runtime.md`

Complete both AIC plans first. Perform the AIC integration on a branch containing `0828d6b7e4a7880079443b1c6f9c148d85bdbf54`, the upstream Spica tree, and the two completed AIC feature series. Perform Dynamo changes on a branch containing `1fe16eb6b2e5318d78f5ece7733e054bab7ef938`; the older local Dynamo checkout used during initial brainstorming is 185 commits behind and is not an implementation base. Keep commits repository-local and run each repository's checks before the cross-repository test.

### Task 0: Materialize and verify both reviewed source baselines

**Files:** no changes

- [ ] **Step 1: Create implementation branches containing the reviewed commits**

In AIC and Dynamo respectively, verify:

```bash
git merge-base --is-ancestor 0828d6b7e4a7880079443b1c6f9c148d85bdbf54 HEAD
test -f src/spica/evaluator.py
git -C ../dynamo merge-base --is-ancestor 1fe16eb6b2e5318d78f5ece7733e054bab7ef938 HEAD
test -f ../dynamo/lib/mocker/src/common/perf_model.rs
test -f ../dynamo/lib/bindings/python/src/dynamo/_internal/aic.py
```

If the repository layout differs, resolve the sibling path explicitly and record both absolute roots. If either ancestor check fails, rebase/cherry-pick onto the reviewed mainline before editing. A newer mainline is allowed only after updating every symbol anchor below and rerunning this baseline gate.

- [ ] **Step 2: Verify current symbols rather than stale line numbers**

Run focused `rg` checks for `AicCallback`, `predict_prefill_time`, `predict_decode_time`, `aic_per_rank_batch`, `MockEngineArgs`, Replay entrypoints, `AicSession`, `ReplayEvaluator`, `_evaluate_one`, and `ProcessPoolExecutor`. Record the resolved file:line map in the implementation PR. Stop and revise the plan if any symbol or error boundary has materially changed.

- [ ] **Step 3: Run pre-change behavior and packaging baselines**

Run the existing AIC Spica unit suite and the Dynamo Mocker/Replay/AIC callback suites. Install the completed AIC wheel into a clean environment and assert these imports succeed without a source checkout:

```bash
python -c "import aiconfigurator.collector, aiconfigurator.collector.trtllm.gemm_adapter, aiconfigurator.collector.network.nccl_adapter"
```

Also assert the wheel does not provide a generic top-level `collector` module. This wheel/import check is repeated in Task 7 after the cross-repository wiring; resolving integration does not proceed against an editable-only collector runtime.

## File map

### Dynamo repository

- Modify `lib/mocker/src/common/perf_model.rs` — exact descriptors, callback result, legacy aggregation.
- Modify `lib/mocker/src/common/protocols.rs` — serializable AIC resolution settings.
- Modify `lib/mocker/src/scheduler/mod.rs` — structured pass-level perf-model failure.
- Modify `lib/mocker/src/scheduler/sglang/core.rs` — exact scheduled prefill descriptor.
- Modify `lib/mocker/src/scheduler/sglang/decode.rs` — exact running decode descriptor.
- Modify `lib/mocker/src/scheduler/vllm/core.rs` — exact scheduled prefill/decode descriptors.
- Modify replay loops under `lib/mocker/src/replay/` — stop on pass-level perf-model failure.
- Modify `lib/bindings/python/rust/llm/aic_callback.rs` — pure/default and resolving callback selection.
- Modify `lib/bindings/python/rust/llm/entrypoint.rs` — keep live Mocker pure and reject replay-only resolution settings.
- Modify `lib/bindings/python/rust/llm/replay.rs` — replay config forwarding.
- Modify `lib/bindings/python/rust/errors.rs` — typed Python resolution exception.
- Modify `lib/bindings/python/src/dynamo/_internal/aic.py` — resolving AIC session and concrete op walk.
- Add inline Rust tests in the modified Rust modules.
- Create `lib/bindings/python/tests/test_aic_resolution.py`.
- Create `lib/bindings/python/tests/replay/test_replay_aic_resolution.py`.

### AIC repository (Spica tree from upstream/main)

- Modify `src/spica/config.py` — lazy collection policy and resource controls.
- Modify `src/spica/deploy.py` — emit resolution settings into Dynamo engine args.
- Modify `src/spica/evaluator.py` — preserve structured replay failure details.
- Modify `src/spica/search.py` — safe process-pool policy and resource leases.
- Modify `tests/spica/test_config.py`.
- Modify `tests/spica/test_deploy.py`.
- Modify `tests/spica/test_evaluator.py`.
- Modify `tests/spica/test_search.py`.
- Modify `tests/spica/test_replay_integration.py`.

### Task 1: Preserve exact scheduled batch shapes in Mocker

**Files:**
- Modify: `lib/mocker/src/common/perf_model.rs` (`PerfModel`, descriptors, and attention-DP projection)
- Modify: `lib/mocker/src/scheduler/sglang/core.rs` (finalized prefill batch)
- Modify: `lib/mocker/src/scheduler/sglang/decode.rs` (running decode batch)
- Modify: `lib/mocker/src/scheduler/vllm/core.rs` (scheduled prefill/decode batches)
- Modify: `lib/mocker/src/common/protocols.rs` (serializable descriptors/config)

- [ ] **Step 1: Add descriptor and legacy-parity tests**

In `perf_model.rs`, construct heterogeneous batches and assert:

```rust
let prefill = ConcreteBatchDescriptor::Prefill(PrefillBatch {
    requests: vec![
        PrefillRequestShape { prompt_tokens: 101, context_tokens: 65, prefix_tokens: 1, scheduled_tokens: 64 },
        PrefillRequestShape { prompt_tokens: 203, context_tokens: 103, prefix_tokens: 3, scheduled_tokens: 100 },
    ],
});
assert_eq!(prefill.legacy_prefill_args(), Some((2, 84, 2)));

let decode = ConcreteBatchDescriptor::Decode(DecodeBatch {
    requests: vec![
        DecodeRequestShape { context_tokens: 127, scheduled_tokens: 1 },
        DecodeRequestShape { context_tokens: 256, scheduled_tokens: 1 },
    ],
    active_kv_tokens: 383,
    total_kv_tokens: 4096,
});
assert_eq!(decode.legacy_decode_args(), Some((2, 383, 191, 4096)));
```

These tuples use the exact positional contracts of current mainline: prefill is `(batch_size, mean_isl, mean_prefix)` and decode is `(batch_size, active_kv_tokens, average_context, total_kv_tokens)`. Thus prefill has `mean_isl=(65+103)/2=84`, `mean_prefix=(1+3)/2=2`, and the existing formula derives `effective_isl=84-2=82`; decode keeps `383` in the active-KV slot and `191` in the average-context slot. Full prompt lengths and actual scheduled chunks remain lossless descriptor metadata but do not inflate the pure/default chunked-prefill timing path.

Add empty-batch validation and serde round-trip tests. For prefill, require `prefix_tokens <= context_tokens <= prompt_tokens` and `scheduled_tokens <= context_tokens - prefix_tokens`; decode scheduled tokens must be positive.

Implement `legacy_prefill_args()` from `context_tokens` and `prefix_tokens`, returning the current `predict_prefill_time(batch_size, isl, prefix)` signature values; never substitute full prompt length, return the already-subtracted effective ISL, or assume `context_tokens == prefix_tokens + scheduled_tokens`. Implement `legacy_decode_args()` in the current `predict_decode_time(batch_size, active_kv_tokens, context_length, total_kv_tokens)` order. Add parity tests with `attention_dp_size=2` and heterogeneous/chunked shapes. Pure/default AIC must still use the current ceil-divided batch size plus legacy scheduled-chunk means exactly. Do not invent a per-rank exact descriptor from the global vector; Task 5 rejects resolution mode for attention-DP greater than one.

- [ ] **Step 2: Define lossless descriptors and callback errors**

```rust
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefillRequestShape {
    pub prompt_tokens: usize,
    pub context_tokens: usize,
    pub prefix_tokens: usize,
    pub scheduled_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefillBatch {
    pub requests: Vec<PrefillRequestShape>,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecodeRequestShape {
    pub context_tokens: usize,
    pub scheduled_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DecodeBatch {
    pub requests: Vec<DecodeRequestShape>,
    pub active_kv_tokens: usize,
    pub total_kv_tokens: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "phase", content = "batch", rename_all = "snake_case")]
pub enum ConcreteBatchDescriptor {
    Prefill(PrefillBatch),
    Decode(DecodeBatch),
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum AicCallbackError {
    #[error("AIC resolution failed [{code}] for {operation}: {detail}")]
    Resolution {
        code: String,
        operation: String,
        detail: String,
        report_json: Option<String>,
    },
    #[error("AIC callback failed: {0}")]
    Runtime(String),
}
```

Implement validated constructors plus `legacy_prefill_args()` and `legacy_decode_args()`. Do not discard the request vectors after calculating aggregates.

- [ ] **Step 3: Build descriptors at the scheduler-owned source of truth**

For SGLang prefill, build `PrefillRequestShape` at the scheduler-owned point where `chunk_end`, `alloc.prefix_len`, `chunk_tokens`, and full `prompt_len` are simultaneously available; extend `PrefillFpmItem` if necessary rather than reconstructing `chunk_end` later. Pass the descriptor to `simulate_prefill_duration`. For SGLang decode, after memory preflight/retraction has finalized `running` but before timing or token mutation, map each request to `current_sequence_len()` and set `scheduled_tokens=min(max_burst, remaining_output_tokens)`. This is the forward-pass work reserved/planned at the timing boundary; stochastic speculative acceptance is sampled afterward and must not be retroactively substituted into the descriptor.

For vLLM prefill, build the vector from the finalized `scheduled: FxHashMap<Uuid, ScheduledWork>` after preemption has removed undone work: `context_tokens=prefix_tokens+prompt_tokens`, `scheduled_tokens=prompt_tokens`, retain full `prompt_len`, include only entries with `prompt_tokens > 0`, and sort by UUID bytes for deterministic keys. Do not substitute `total_tokens`, which may include non-prompt work. For both vLLM decode sites, build request shapes from the exact post-preemption `ready` UUIDs and their current sequence lengths; set `scheduled_tokens=1` for ordinary decode and `min(max_burst, remaining_generation_tokens)` for speculative decode after reservation succeeds but before acceptance sampling/mutation.

Change `PrefillCost::predict_prefill_compute()` to construct a one-request descriptor so router/admission estimates use the same API without inventing a mean.

- [ ] **Step 4: Keep non-AIC timing behavior identical**

Change `PerfModel::predict_prefill_time` and `predict_decode_time` to accept `&ConcreteBatchDescriptor`. Polynomial and interpolated variants unpack the descriptor's signature-ordered legacy aggregate methods and execute the existing formulas unchanged: prefill subtracts `mean_prefix` from `mean_isl` exactly once, and decode retains `(active_kv_tokens, average_context)` in that order. In this first task, keep the existing split `AicCallback::predict_prefill/predict_decode -> f64` interface: the AIC arm derives its current callback scalars from the same raw tuple, separately computes `projected_batch_size = aic_per_rank_batch(global_batch_size, attention_dp_size)`, and invokes those methods exactly as before. The immutable global descriptor is now available at the `PerfModel` boundary but does not cross the callback trait until Task 2. This preserves today's ceil-divided pure/default AIC behavior without pretending the global request vector can be partitioned exactly. Add before/after golden tests for all three existing variants, plus scheduler-boundary tests asserting the exact per-request vectors for heterogeneous, cached-prefix, chunked-prefill, preempted, and speculative-decode cases.

- [ ] **Step 5: Run and commit descriptor propagation**

Run from Dynamo:

```bash
cargo nextest run -p dynamo-mocker common::perf_model scheduler::sglang scheduler::vllm
cargo check -p dynamo-mocker
```

Expected: existing timing tests remain unchanged and new descriptor tests pass.

```bash
git add lib/mocker/src/common/perf_model.rs lib/mocker/src/common/protocols.rs lib/mocker/src/scheduler
git commit -m "feat: preserve concrete mocker batch shapes"
```

### Task 2: Replace the bare callback `f64` with a structured failure path

**Files:**
- Modify: `lib/mocker/src/common/perf_model.rs` (`AicCallback` result)
- Modify: `lib/mocker/src/scheduler/mod.rs` (`EnginePassResult`)
- Modify: `lib/mocker/src/scheduler/sglang/core.rs`
- Modify: `lib/mocker/src/scheduler/sglang/decode.rs`
- Modify: `lib/mocker/src/scheduler/vllm/core.rs`
- Modify: `lib/mocker/src/scheduler/live_boundary.rs` (preserve the live hard-failure boundary)
- Modify: replay loops under `lib/mocker/src/replay/`

- [ ] **Step 1: Write failing-callback tests**

Implement a test callback that returns:

```rust
Err(AicCallbackError::Resolution {
    code: "missing_adapter".into(),
    operation: "attention.context".into(),
    detail: "no lazy adapter for context attention".into(),
    report_json: None,
})
```

Assert `PerfModel` preserves all three fields, the current engine pass reports a perf-model error, and offline replay returns an error containing `candidate_unscorable` without emitting a trace report. Add a sibling test proving a valid callback still gets the decode minimum-latency clamp after success. Add a live-boundary test proving an unexpected `perf_model_error` aborts the live scheduler task before admissions, KV effects, timing waits, or outputs are published; resolution callbacks are forbidden live, so silently converting this field into a zero-progress pass would be a regression from today's hard failure.

- [ ] **Step 2: Change the callback contract**

```rust
pub trait AicCallback: Send + Sync {
    fn predict(
        &self,
        batch: &ConcreteBatchDescriptor,
        projected_batch_size: usize,
    ) -> Result<f64, AicCallbackError>;
}
```

`projected_batch_size` is the existing AIC-only attention-DP ceil division; it does not alter the descriptor and is not used by Polynomial or Interpolated. In the `PerfModel::Aiconfigurator` arm, replace the temporary Task-1 scalar callback dispatch with `callback.predict(batch, projected_batch_size)`. Return `Result<f64, AicCallbackError>` from both `PerfModel` prediction methods. Validate finiteness and non-negativity after the callback succeeds; turn NaN/infinity into `AicCallbackError::Runtime`.

- [ ] **Step 3: Carry one failure through an engine pass**

Add this defaulted field to `EnginePassResult`:

```rust
pub(crate) perf_model_error: Option<AicCallbackError>,
```

At each scheduler prediction boundary, match the result. On error, stop that pass before token emission, preserve the scheduler's already finalized admission state, set `end_ms=now_ms`, and return the error. Do not substitute zero, a polynomial estimate, or a curated-only retry.

Add a public `CandidateUnscorable(AicCallbackError)` error wrapper in Mocker. Update every replay loop that consumes `EnginePassResult` to check this field before advancing virtual time and return that typed error through `anyhow` without flattening it to text. V1 resolution is enabled only through Replay/Spica. Because current `SchedulerHandle` has no terminal-error channel, `LiveEffectsPublisher.capture_pass()` must check `perf_model_error` before draining or publishing any effects and preserve the existing hard-failure behavior (panic/terminate the live scheduler task with the structured error). Task 5 rejects `aic_resolution` on the live path, so a recoverable candidate-level resolution error can never reach this boundary; the guard prevents a pure callback error from becoming an endless zero-progress loop.

- [ ] **Step 4: Compile all callers and run replay tests**

Run:

```bash
cargo check -p dynamo-mocker --all-features
cargo nextest run -p dynamo-mocker replay scheduler live_boundary
```

Use compiler errors to update all direct test callbacks and prediction call sites; no call site may use `unwrap_or`, `unwrap_or_default`, or a latency fallback for `AicCallbackError`.

- [ ] **Step 5: Continue directly to the cross-crate callback update**

Do not commit the trait change yet: `dynamo-py3` implements `AicCallback` in another crate, so a mocker-only commit here would leave the workspace uncompilable. Keep the changes in the working tree and complete Task 3 as the same atomic cross-crate checkpoint.

### Task 3: Preserve the pure-Rust callback as the default

**Files:**
- Modify: `lib/bindings/python/rust/llm/aic_callback.rs` (callback implementations/factory)
- Modify: inline `#[cfg(test)]` module in `lib/bindings/python/rust/llm/aic_callback.rs`

- [ ] **Step 1: Add parity tests around heterogeneous descriptors**

Create a fake compiled engine and descriptors matching Task 1. For prefill, unpack `(global_batch_size, mean_isl, mean_prefix)`, derive `effective_isl = mean_isl.saturating_sub(mean_prefix)` exactly once, and assert `RustAicCallback.predict()` calls `prefill_latency_ms(projected_batch_size, effective_isl + mean_prefix, mean_prefix)` — `(2, 84, 2)` for the fixture when attention DP is one. For decode, unpack `(global_batch_size, active_kv_tokens, average_context, total_kv_tokens)` and assert it calls `decode_latency_ms(projected_batch_size, average_context, 2)` without swapping the active-KV and context fields. Add an attention-DP=2 assertion that only the projected callback batch changes from two to one; the descriptor and all means remain unchanged. Assert an engine error becomes `AicCallbackError::Runtime`; remove both current `panic!` branches.

- [ ] **Step 2: Implement descriptor dispatch**

Match `ConcreteBatchDescriptor` in `RustAicCallback`. Use only the signature-ordered `legacy_*_args()` methods plus the `projected_batch_size` supplied by `PerfModel`; never infer a second effective-ISL subtraction or swap decode tuple positions. Keep `create_aic_callback()` returning `RustAicCallback` whenever no resolution config is supplied, and keep the predict hot path GIL-free.

- [ ] **Step 3: Run and commit default-path parity**

Run:

```bash
cargo nextest run -p dynamo-py3 aic_callback
cargo check -p dynamo-py3 --features aic-forward-pass
```

```bash
git add lib/mocker/src lib/bindings/python/rust/llm/aic_callback.rs lib/bindings/python/tests
git commit -m "feat: propagate structured descriptor-aware AIC callbacks"
```

### Task 4: Add the opt-in Python resolving callback and concrete AIC op walk

**Files:**
- Modify: `lib/bindings/python/src/dynamo/_internal/aic.py` (`AicSession`)
- Modify: `lib/bindings/python/rust/llm/aic_callback.rs`
- Create: `lib/bindings/python/tests/test_aic_resolution.py`

- [ ] **Step 1: Add Python session tests with a fake model and executor**

Use one heterogeneous prefill descriptor with two requests. Assert:

- non-logits ops receive `x=sum(scheduled_tokens)`, not `batch_size * floor(mean)`;
- logits GEMM receives `x=batch_size`;
- every op receives immutable `scheduled_tokens`, `prompt_tokens`, `context_tokens`, and `prefix_tokens` tuples;
- the whole op list is invoked twice on a cold miss and once on a warm overlay hit;
- two ops sharing one key produce one executor request but both latencies in the second walk;
- a decode descriptor forwards exact context tuples and the speculative effective batch;
- setting resolution policy to pure still permits the compiled engine.
- the Rust-owned Python callback thread can create a spawn-context fake collector worker, wait without holding the GIL, receive its correlated reply, and close it cleanly; this catches embedded-Python/multiprocessing bootstrap failures before GPU testing.

- [ ] **Step 2: Construct the AIC resolution stack only when requested**

Extend `AicSession.__init__` with `resolution: dict | None = None`. When it is absent, preserve `_build_compiled_engine()`. When present:

1. require an absolute overlay path;
2. build `OverlayStore`, the complete `MeasurementProtocol`, `ResolutionBudget`, hardware inventory restricted to configured physical GPU ids, namespaced backend plus network `LazyAdapterRegistry`, and `ResourceAwareMeasurementExecutor`;
3. create one `ResolutionSession` owned by this `AicSession`;
4. set `_engine=None` deliberately, because the compiled Rust engine cannot emit a full `MissSet` yet;
5. register `close()` and context-manager cleanup for workers and SQLite.

The Python owner thread must wait through `multiprocessing.connection.wait`, whose OS poll releases the GIL while collector subprocesses run. Every CUDA/NCCL import and benchmark executes in those subprocesses; no runtime tensor crosses the callback boundary.

- [ ] **Step 3: Implement `predict_concrete` around one operation walk**

For prefill, derive `batch_size`, exact total scheduled tokens, integer legacy `s/prefix` for curated queries, and the three concrete tuples. For decode, derive exact context tuples and effective speculative batch. Call each operation's `query_with_resolution()` with those fields. Wrap the entire list traversal in exactly one `resolution_session.execute_callback(walk)` call; never wrap individual ops.

Catch AIC `ResolutionFailed` and return a JSON-safe payload:

```python
{
    "ok": False,
    "failure": {
        "kind": "resolution",
        "reasons": [
            {"code": reason.code.value, "operation": reason.operation, "detail": reason.detail}
            for reason in error.reasons
        ],
        "report": self._resolution_session.report.to_dict(),
    },
}
```

Success returns `{"ok": True, "latency_ms": total}`. This avoids parsing Python exception strings in Rust.

- [ ] **Step 4: Add a dedicated bridge thread in Rust**

Implement `PyResolvingAicCallback` with a bounded Rust channel and one named OS thread. Its `predict()` first requires `projected_batch_size == batch.request_count()`; resolution mode is attention-DP-one only, so inequality is an explicit `AicCallbackError::Runtime` rather than a fabricated per-rank descriptor. The thread owns the Python `AicSession`, acquires the GIL only to call `predict_concrete`, converts the returned dict to `Result<f64, AicCallbackError>`, and replies over a one-shot channel. `predict()` sends the cloned descriptor and blocks on the Rust receiver without holding the GIL. `Drop` sends shutdown and joins the thread.

Serialize concurrent callers through this owner thread because one `ResolutionSession` has callback-local mutable state. Map each structured Python reason directly to `AicCallbackError::Resolution`; if multiple reasons exist, retain the first in typed fields and store the complete JSON-safe resolution report in `report_json`.

- [ ] **Step 5: Run and commit the resolving bridge**

Run from Dynamo:

```bash
pytest -q lib/bindings/python/tests -k aic
cargo nextest run -p dynamo-py3 aic
cargo check -p dynamo-py3 --features aic-forward-pass
```

```bash
git add lib/bindings/python/src/dynamo/_internal/aic.py lib/bindings/python/rust/llm/aic_callback.rs lib/bindings/python/tests
git commit -m "feat: resolve AIC misses through a Python callback"
```

### Task 5: Expose an explicit replay resolution configuration and keep live Mocker pure

**Files:**
- Modify: `lib/mocker/src/common/protocols.rs` (`MockEngineArgs` and compatibility serde)
- Modify: `lib/bindings/python/rust/llm/entrypoint.rs` (live-path validation)
- Modify: `lib/bindings/python/rust/llm/replay.rs` (Replay config forwarding/error mapping)
- Modify: `lib/bindings/python/rust/llm/aic_callback.rs` (factory selection)
- Modify: `lib/bindings/python/rust/errors.rs`
- Modify: inline tests in `lib/mocker/src/common/protocols.rs`
- Create: `lib/bindings/python/tests/replay/test_replay_aic_resolution.py`

- [ ] **Step 1: Add strict serialization and validation tests**

Assert omitted config selects the pure Rust callback. Test JSON/Python construction for `observe_only` and `measure_on_miss`. Reject relative overlay paths, zero/negative budgets, duplicate GPU ids, resolution settings without `aic_backend`, and any resolution setting with `aic_attention_dp_size > 1`. Verify ordinary `MockEngineArgs` JSON remains byte-for-field compatible apart from the newly optional field and that pure prediction with attention DP remains numerically unchanged.

- [ ] **Step 2: Define one serializable Rust config**

```rust
#[derive(Debug, Clone, Serialize, Deserialize, Validate, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AicResolutionArgs {
    pub policy: AicResolutionPolicy,
    pub overlay_path: PathBuf,
    #[validate(range(min = 1))]
    pub max_new_keys: usize,
    #[validate(range(exclusive_min = 0.0))]
    pub max_wall_seconds: f64,
    pub gpu_ids: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AicResolutionPolicy {
    ObserveOnly,
    MeasureOnMiss,
}
```

Add `pub aic_resolution: Option<AicResolutionArgs>` to `MockEngineArgs`, its serde compatibility struct, builder defaults, `TryFrom` normalization, and validation. Do not add `pure` as a serialized mode: absence is the pure/default contract.

In cross-field validation, reject `aic_resolution.is_some()` when `aic_attention_dp_size.unwrap_or(1) > 1` with a message explaining that V1 requires scheduler-owned per-rank concrete descriptors. This restriction applies to observe-only as well as measure-on-miss; the pure/default path retains current attention-DP support.

- [ ] **Step 3: Wire Replay and reject the setting on the live path**

Expose `aic_resolution` through Replay's `MockEngineArgs.from_json()` and Python constructor, and pass it to Replay's `create_aic_callback()` call. Change `create_aic_callback()` to accept `Option<&AicResolutionArgs>` and branch once: absent builds `RustAicCallback`; present builds `PyResolvingAicCallback`.

Keep `AicPerfConfig` unchanged. In the live entrypoint path that consumes `RsMockEngineArgs`, reject `mocker_args.aic_resolution.is_some()` with `"AIC measure-on-miss is replay-only in V1"` before callback construction, and pass `None` to `create_aic_callback()`. `create_aic_prefill_load_estimator()` remains pure Rust. This makes the unsupported live boundary explicit instead of letting a background scheduler task lose a resolution error.

Define and register `AicResolutionError` in `errors.rs`. Add a replay-specific mapper that searches an `anyhow::Error` chain for `CandidateUnscorable`, constructs `AicResolutionError`, and sets Python attributes `code`, `operation`, `detail`, and parsed `report`; all other errors continue through the existing `to_pyerr`. Replace replay entrypoint `.map_err(to_pyerr)` calls with this mapper. This is the structured exception `ReplayEvaluator` catches in Task 6.

- [ ] **Step 4: Run and commit API wiring**

Run:

```bash
cargo nextest run -p dynamo-mocker common::protocols
cargo nextest run -p dynamo-py3 llm
pytest -q lib/bindings/python/tests -k "aic or replay"
```

```bash
git add lib/mocker/src/common/protocols.rs lib/bindings/python
git commit -m "feat: configure opt-in AIC miss resolution"
```

### Task 6: Add Spica policy, overlay, and safe parallel-evaluation controls

**Files:**
- Modify: AIC `src/spica/config.py` (`SmartSearchConfig`)
- Modify: AIC `src/spica/deploy.py`
- Modify: AIC `src/spica/evaluator.py` (`ReplayEvaluator`)
- Modify: AIC `src/spica/search.py` (`_evaluate_one` and process pools)
- Modify: AIC `tests/spica/test_config.py`
- Modify: AIC `tests/spica/test_deploy.py`
- Modify: AIC `tests/spica/test_search.py`

- [ ] **Step 1: Write config and deployment tests**

Add cases proving:

- no `lazy_collection` field emits no `aic_resolution` and preserves current Spica behavior;
- `observe_only` and `measure_on_miss` require an absolute overlay path;
- measure-on-miss defaults to safe sequential candidate evaluation unless explicit disjoint GPU groups are supplied;
- GPU groups contain no duplicate ids within or across groups;
- deployment emits the identical resolution payload into aggregated, prefill, and decode engine args;
- overlay path and budgets are pinned run context, never optimizer knobs or candidate-cache identity omissions.

- [ ] **Step 2: Add Pydantic policy models**

```python
class LazyCollectionPolicy(str, Enum):
    OBSERVE_ONLY = "observe_only"
    MEASURE_ON_MISS = "measure_on_miss"


class LazyCollectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: LazyCollectionPolicy
    overlay_path: Path
    max_new_keys: int = Field(default=256, ge=1)
    max_wall_seconds: float = Field(default=600.0, gt=0)
    gpu_groups: list[list[int]] | None = None
```

Add `lazy_collection: LazyCollectionConfig | None = None` to `SmartSearchConfig`. Its validator resolves `overlay_path`, requires it to be absolute, validates nonempty disjoint groups, and rejects group count smaller than `parallel_evals` when `parallel_evals > 1`.

- [ ] **Step 3: Emit Dynamo engine arguments**

Add `aic_resolution: dict | None = None` to `build_deployment()` and `_engine_args_payload()`. In `_evaluate_one`, build one payload from `config.lazy_collection` and that worker's assigned GPU group:

```python
lazy = config.lazy_collection
assert lazy is not None
resolution = {
    "policy": lazy.policy.value,
    "overlay_path": str(lazy.overlay_path),
    "max_new_keys": lazy.max_new_keys,
    "max_wall_seconds": lazy.max_wall_seconds,
    "gpu_ids": assigned_group,
}
```

Pass the payload as `build_deployment(sample, backend_version=backend_version, optimization_target=goal.target.planner_optimization_target, planner_sla=goal.sla, aic_resolution=resolution)`, and attach it as `aic_resolution` to every AIC-backed engine args dict. Keep it absent when lazy collection is absent. Update candidate cache context to include the full lazy configuration so a pure result cannot alias a resolution-enabled result.

- [ ] **Step 4: Make the safe execution rule explicit**

For the first integration, if measure-on-miss has no `gpu_groups`, copy the config with `parallel_evals=1` and emit one progress message explaining that hardware parallelism still occurs inside each callback. If disjoint groups are supplied, create one single-worker `ProcessPoolExecutor` per GPU group rather than one shared pool. Pass that group's ids in the pool initializer, submit suggestions round-robin across group pools, and retain the existing timeout/recreate behavior per pool. This gives every long-lived Spica worker one stable resource lease; `ProcessPoolExecutor`'s shared initializer cannot safely infer a unique group index on its own.

V1 deliberately adds no distributed per-key coordinator: session-local single-flight and SQLite's process-safe append/index operations remain the only deduplication boundary. Two evaluators on disjoint groups may measure the same simultaneously cold key; both records remain auditable and the latest compatible sequence wins. Users who require unique cold measurement cost use the default `parallel_evals=1` or run a sequential coverage preflight before read-only parallel scoring.

- [ ] **Step 5: Preserve structured candidate failures**

In `ReplayEvaluator`, catch Dynamo's typed `AicResolutionError` and raise a Spica `UnscorableCandidate` carrying its `code`, `operation`, `detail`, and `report` attributes. In `_evaluate_one`, map that exception to outcome `failed` with reason prefix `aic_resolution:`; do not create a `Candidate`, score, or sampler metric. If every evaluated suggestion is unscorable, `run_smart_search` raises a summary containing the top reason counts rather than returning an empty successful result.

- [ ] **Step 6: Run and commit Spica integration**

Run from AIC:

```bash
pytest -q tests/spica/test_config.py tests/spica/test_deploy.py tests/spica/test_evaluator.py tests/spica/test_search.py
git diff --check
```

```bash
git add src/spica tests/spica
git commit -m "feat: integrate lazy AIC collection with Spica"
```

### Task 7: Validate cold, warm, failure, and cross-process behavior

**Files:**
- Modify: AIC `tests/spica/test_replay_integration.py`
- Modify: Dynamo `lib/bindings/python/tests/replay/test_replay_aic_resolution.py`.
- Add: one opt-in multi-GPU end-to-end test in the AIC integration suite.

- [ ] **Step 1: Add a CPU fake-executor cross-repository test**

Run Mocker with a heterogeneous two-request batch and a fake AIC collector. Assert the cold callback:

1. receives the lossless descriptor;
2. discovers every op miss in one first walk;
3. dispatches one deduplicated request batch;
4. replays the AIC op walk exactly once;
5. advances Mocker virtual time only from the completed second-walk latency.

Run the identical replay again against the reopened overlay and assert zero executor calls and identical trace metrics.

- [ ] **Step 2: Add strict failure tests**

Test missing adapter, rejected measurement, collector crash, key budget, wall budget, transient retry exhaustion, and a second-walk miss. Every case must yield an unscorable Spica trial with a structured reason. A sibling valid candidate must still complete. Assert resolving config with attention DP greater than one fails validation while the identical pure config still scores with legacy behavior. When all candidates fail, assert the sweep raises rather than ranking a synthetic score.

- [ ] **Step 3: Add process-pool lease and duplicate-evidence tests**

Start two Spica worker processes with the same cold key and two declared disjoint GPU groups. Assert each worker sees only its assigned ids, both observations may append safely, and lookup returns the later valid sequence. With overlapping groups, config validation fails before ProcessPool creation. With no groups, assert Spica forces sequential evaluation and the repeated key measures once then hits the overlay.

- [ ] **Step 4: Add one real-GPU pilot**

On a compatible multi-GPU TensorRT-LLM host, search a tiny two-candidate fixture whose AIC operation list is deliberately limited to the two V1-supported pilots (or whose other operations all have literal curated exact rows): one missing BF16 GEMM and one missing NCCL all-reduce point. Run capability preflight first and assert exactly those two keys are collectable and no unsupported operation is hidden. Assert both records include real hardware/fabric provenance, the cold run reports collection wall time separately from simulated serving latency, and the warm run starts no collector worker commands. Do not assert a search-score improvement; this test validates causality and cache reuse.

- [ ] **Step 5: Run complete validation**

From Dynamo:

```bash
cargo check -p dynamo-mocker -p dynamo-py3 --all-features
cargo nextest run -p dynamo-mocker
pytest -q lib/bindings/python/tests -k "aic or replay"
```

From AIC:

```bash
pytest -m unit tests/unit/sdk/resolution tests/unit/collector/lazy -v
pytest -q tests/spica
pytest -m gpu tests/integration/collector tests/spica/test_replay_integration.py -v -s
git diff --check
```

Build the final non-editable AIC wheel, install it into a clean environment used by the Dynamo Python tests, rerun `test_import_surface.py`, and import `aiconfigurator.collector` plus both lightweight adapters. Assert the wheel contains the namespaced runtime and pilot worker modules, does not provide top-level `collector`, and the cold fake-executor Replay test passes without the AIC source checkout on `PYTHONPATH`. This repeats the packaging gate after all cross-repository wiring.

Expected: pure/default tests show no GPU discovery or Python callback; cold resolving tests collect once and requery once; warm tests hit the overlay; unresolvable candidates never receive scores; hardware tests keep compute and collectives within declared leases.

- [ ] **Step 6: Commit end-to-end coverage in each repository**

In Dynamo:

```bash
git add lib/mocker lib/bindings/python
git commit -m "test: validate Mocker AIC resolution lifecycle"
```

In AIC:

```bash
git add tests/spica tests/integration
git commit -m "test: validate Spica lazy collection lifecycle"
```
