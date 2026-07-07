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

Complete both AIC plans first. Spica currently exists on AIC `upstream/main`; perform this integration on an AIC branch containing that upstream Spica tree plus the two completed AIC feature series. Dynamo changes belong in the sibling Dynamo repository. Keep commits repository-local and run each repository's checks before the cross-repository test.

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
- Modify: `lib/mocker/src/common/perf_model.rs:25-45,211-280`
- Modify: `lib/mocker/src/scheduler/sglang/core.rs:363-378,527-548`
- Modify: `lib/mocker/src/scheduler/sglang/decode.rs:226-237`
- Modify: `lib/mocker/src/scheduler/vllm/core.rs:1003-1167,1290-1455,1499-1514,1672-1689,1817-1835`
- Modify: `lib/mocker/src/common/protocols.rs:256-265`

- [ ] **Step 1: Add descriptor and legacy-parity tests**

In `perf_model.rs`, construct heterogeneous batches and assert:

```rust
let prefill = ConcreteBatchDescriptor::Prefill(PrefillBatch {
    requests: vec![
        PrefillRequestShape { prompt_tokens: 101, cached_tokens: 1, scheduled_tokens: 100 },
        PrefillRequestShape { prompt_tokens: 203, cached_tokens: 3, scheduled_tokens: 200 },
    ],
});
assert_eq!(prefill.legacy_prefill_args(), Some((2, 150, 2)));

let decode = ConcreteBatchDescriptor::Decode(DecodeBatch {
    requests: vec![
        DecodeRequestShape { context_tokens: 127, scheduled_tokens: 1 },
        DecodeRequestShape { context_tokens: 256, scheduled_tokens: 1 },
    ],
    active_kv_tokens: 383,
    total_kv_tokens: 4096,
});
assert_eq!(decode.legacy_decode_args(), Some((2, 191, 383, 4096)));
```

The prefill result intentionally reproduces current integer truncation: `mean_isl=(101+203)/2=152`, `mean_prefix=(1+3)/2=2`, and `effective_isl=150`. This catches an accidental switch to averaging each request's uncached length independently.

Add empty-batch validation and serde round-trip tests. `scheduled_tokens` must not exceed `prompt_tokens - cached_tokens` for prefill and must be positive for decode.

- [ ] **Step 2: Define lossless descriptors and callback errors**

```rust
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PrefillRequestShape {
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
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

For SGLang prefill, build `PrefillRequestShape` from each `prefill_fpm` item using `prompt_len`, `prefix_tokens`, and `tokens_computed`; pass the descriptor to `simulate_prefill_duration`. For SGLang decode, map `running` to each `current_sequence_len()` before the step mutates requests.

For vLLM prefill, build the vector from the finalized `scheduled: FxHashMap<Uuid, ScheduledWork>` after preemption has removed undone work; include only entries with `prompt_tokens > 0` and sort by UUID bytes for deterministic keys. For both vLLM decode sites, build request shapes from the exact `ready` UUIDs and their current sequence lengths before sampling/mutation.

Change `PrefillCost::predict_prefill_compute()` to construct a one-request descriptor so router/admission estimates use the same API without inventing a mean.

- [ ] **Step 4: Keep non-AIC timing behavior identical**

Change `PerfModel::predict_prefill_time` and `predict_decode_time` to accept `&ConcreteBatchDescriptor`. Polynomial and interpolated variants call the descriptor's legacy aggregate methods and execute the existing formulas unchanged. Add recording callbacks in SGLang and vLLM tests to assert the exact per-request vectors received for heterogeneous, cached-prefix, chunked-prefill, preempted, and speculative-decode cases.

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
- Modify: `lib/mocker/src/common/perf_model.rs:30-39`
- Modify: `lib/mocker/src/scheduler/mod.rs:150-245`
- Modify: `lib/mocker/src/scheduler/sglang/core.rs`
- Modify: `lib/mocker/src/scheduler/sglang/decode.rs`
- Modify: `lib/mocker/src/scheduler/vllm/core.rs`
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

Assert `PerfModel` preserves all three fields, the current engine pass reports a perf-model error, and offline replay returns an error containing `candidate_unscorable` without emitting a trace report. Add a sibling test proving a valid callback still gets the decode minimum-latency clamp after success.

- [ ] **Step 2: Change the callback contract**

```rust
pub trait AicCallback: Send + Sync {
    fn predict(&self, batch: &ConcreteBatchDescriptor) -> Result<f64, AicCallbackError>;
}
```

Return `Result<f64, AicCallbackError>` from both `PerfModel` prediction methods. Validate finiteness and non-negativity after the callback succeeds; turn NaN/infinity into `AicCallbackError::Runtime`.

- [ ] **Step 3: Carry one failure through an engine pass**

Add this defaulted field to `EnginePassResult`:

```rust
pub(crate) perf_model_error: Option<AicCallbackError>,
```

At each scheduler prediction boundary, match the result. On error, stop that pass before token emission, preserve the scheduler's already finalized admission state, set `end_ms=now_ms`, and return the error. Do not substitute zero, a polynomial estimate, or a curated-only retry.

Add a public `CandidateUnscorable(AicCallbackError)` error wrapper in Mocker. Update every replay loop that consumes `EnginePassResult` to check this field before advancing virtual time and return that typed error through `anyhow` without flattening it to text. V1 resolution is enabled only through Replay/Spica; live scheduler construction rejects `aic_resolution` in Task 5, avoiding an error channel that current `SchedulerHandle` does not expose.

- [ ] **Step 4: Compile all callers and run replay tests**

Run:

```bash
cargo check -p dynamo-mocker --all-features
cargo nextest run -p dynamo-mocker replay scheduler
```

Use compiler errors to update all direct test callbacks and prediction call sites; no call site may use `unwrap_or`, `unwrap_or_default`, or a latency fallback for `AicCallbackError`.

- [ ] **Step 5: Commit structured failure propagation**

```bash
git add lib/mocker/src
git commit -m "feat: propagate unscorable AIC predictions"
```

### Task 3: Preserve the pure-Rust callback as the default

**Files:**
- Modify: `lib/bindings/python/rust/llm/aic_callback.rs:35-61,181-250`
- Modify: inline `#[cfg(test)]` module in `lib/bindings/python/rust/llm/aic_callback.rs`

- [ ] **Step 1: Add parity tests around heterogeneous descriptors**

Create a fake compiled engine and descriptors matching Task 1. Assert `RustAicCallback.predict()` calls `prefill_latency_ms(batch_size, effective_isl + prefix, prefix)` and `decode_latency_ms(batch_size, average_context, 2)` with exactly the old aggregate values. Assert an engine error becomes `AicCallbackError::Runtime`; remove both current `panic!` branches.

- [ ] **Step 2: Implement descriptor dispatch**

Match `ConcreteBatchDescriptor` in `RustAicCallback`. Use only the `legacy_*_args()` methods so this path's numerical behavior does not change. Keep `create_aic_callback()` returning `RustAicCallback` whenever no resolution config is supplied, and keep the predict hot path GIL-free.

- [ ] **Step 3: Run and commit default-path parity**

Run:

```bash
cargo nextest run -p dynamo-py3 aic_callback
cargo check -p dynamo-py3 --features aic-forward-pass
```

```bash
git add lib/bindings/python/rust/llm/aic_callback.rs lib/bindings/python/tests
git commit -m "refactor: make Rust AIC callback descriptor aware"
```

### Task 4: Add the opt-in Python resolving callback and concrete AIC op walk

**Files:**
- Modify: `lib/bindings/python/src/dynamo/_internal/aic.py:112-291`
- Modify: `lib/bindings/python/rust/llm/aic_callback.rs`
- Create: `lib/bindings/python/tests/test_aic_resolution.py`

- [ ] **Step 1: Add Python session tests with a fake model and executor**

Use one heterogeneous prefill descriptor with two requests. Assert:

- non-logits ops receive `x=sum(scheduled_tokens)`, not `batch_size * floor(mean)`;
- logits GEMM receives `x=batch_size`;
- every op receives immutable `scheduled_tokens`, `prompt_tokens`, and `cached_tokens` tuples;
- the whole op list is invoked twice on a cold miss and once on a warm overlay hit;
- two ops sharing one key produce one executor request but both latencies in the second walk;
- a decode descriptor forwards exact context tuples and the speculative effective batch;
- setting resolution policy to pure still permits the compiled engine.

- [ ] **Step 2: Construct the AIC resolution stack only when requested**

Extend `AicSession.__init__` with `resolution: dict | None = None`. When it is absent, preserve `_build_compiled_engine()`. When present:

1. require an absolute overlay path;
2. build `OverlayStore`, `MeasurementProtocol`, `ResolutionBudget`, hardware inventory restricted to configured physical GPU ids, backend plus network `LazyAdapterRegistry`, and `ResourceAwareMeasurementExecutor`;
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

Implement `PyResolvingAicCallback` with a bounded Rust channel and one named OS thread. The thread owns the Python `AicSession`, acquires the GIL only to call `predict_concrete`, converts the returned dict to `Result<f64, AicCallbackError>`, and replies over a one-shot channel. `predict()` sends the cloned descriptor and blocks on the Rust receiver without holding the GIL. `Drop` sends shutdown and joins the thread.

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
- Modify: `lib/mocker/src/common/protocols.rs:489-740,925-1138`
- Modify: `lib/bindings/python/rust/llm/entrypoint.rs:80-151,740-800`
- Modify: `lib/bindings/python/rust/llm/replay.rs:156-332,1300-1360`
- Modify: `lib/bindings/python/rust/llm/aic_callback.rs:181-250`
- Modify: `lib/bindings/python/rust/errors.rs`
- Modify: inline tests in `lib/mocker/src/common/protocols.rs`
- Create: `lib/bindings/python/tests/replay/test_replay_aic_resolution.py`

- [ ] **Step 1: Add strict serialization and validation tests**

Assert omitted config selects the pure Rust callback. Test JSON/Python construction for `observe_only` and `measure_on_miss`. Reject relative overlay paths, zero/negative budgets, duplicate GPU ids, and resolution settings without `aic_backend`. Verify ordinary `MockEngineArgs` JSON remains byte-for-field compatible apart from the newly optional field.

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
- Modify: AIC `src/spica/config.py:603-681`
- Modify: AIC `src/spica/deploy.py`
- Modify: AIC `src/spica/evaluator.py:130-260`
- Modify: AIC `src/spica/search.py:86-178,180-360`
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
resolution = {
    "policy": config.policy.value,
    "overlay_path": str(config.overlay_path),
    "max_new_keys": config.max_new_keys,
    "max_wall_seconds": config.max_wall_seconds,
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

Test missing adapter, rejected measurement, collector crash, key budget, wall budget, and a second-walk miss. Every case must yield an unscorable Spica trial with a structured reason. A sibling valid candidate must still complete. When all candidates fail, assert the sweep raises rather than ranking a synthetic score.

- [ ] **Step 3: Add process-pool lease and duplicate-evidence tests**

Start two Spica worker processes with the same cold key and two declared disjoint GPU groups. Assert each worker sees only its assigned ids, both observations may append safely, and lookup returns the later valid sequence. With overlapping groups, config validation fails before ProcessPool creation. With no groups, assert Spica forces sequential evaluation and the repeated key measures once then hits the overlay.

- [ ] **Step 4: Add one real-GPU pilot**

On a compatible multi-GPU TensorRT-LLM host, search a tiny two-candidate space whose workload reaches one missing BF16 GEMM and one missing NCCL all-reduce point. Assert both records include real hardware/fabric provenance, the cold run reports collection wall time separately from simulated serving latency, and the warm run starts no collector worker commands. Do not assert a search-score improvement; this test validates causality and cache reuse.

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
