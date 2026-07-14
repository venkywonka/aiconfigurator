---
name: aic-jitcollect-op-dev
description: Add, review, debug, or validate AIC online JIT performance collection for a pre-existing operation. Use whenever work mentions measure-on-miss, lazy collection, PerfKey or MeasurementRequest support, exact one-case adapters/runners, ResolutionSession wiring, cold/warm/reopened-overlay ablations, hardware-aware measurement, or Replay/Spica AIC callback integration—even if the request only says to make an existing operation collect missing performance data on demand.
---

# AIC JITCollect Op Development

Add the smallest operation-specific vertical slice that lets the existing AIC
operation walk discover an exact missing physical point, measure it safely,
persist validated evidence, and replay once without changing pure prediction.

## Scope and companion skills

This skill owns the online exact-shape path:

```text
existing operation query
  -> canonical PerfKey
  -> callback-local miss
  -> exact one-case adapter and runner
  -> hardware-safe measurement
  -> append-only overlay
  -> one exact replay
```

It does not create a new graph IR, forecast operation dependencies in a second
path, run an offline shape grid, or promote overlay rows into curated data.

- If the task changes offline Collector case population, registry case grids,
  deduplication, or curated perf files, also read
  `$aic-collector-op-development`.
- If the task runs a long GPU perf-data campaign, also use `$aic-auto-collect`.
- Absence of `aic_resolution` keeps Live Mocker collection-free. V1.3 may enable
  blocking resolution only for its approved frozen SGLang/DSv4 profile and only
  through the shared policy plus transactional Live boundary; an operation slice
  by itself does not authorize or claim Live support.

## Required reading and source reconnaissance

Before editing, read:

1. `docs/superpowers/specs/2026-07-07-aic-dsv4-online-collection-v1-2-design.md`
2. `src/aiconfigurator/sdk/resolution/types.py`
3. `src/aiconfigurator/sdk/resolution/session.py`
4. `src/aiconfigurator/sdk/resolution/overlay.py`
5. `src/aiconfigurator/collector/adapters.py`
6. `src/aiconfigurator/collector/executor.py`
7. `src/aiconfigurator/collector/scheduler.py`
8. The target operation, its database loader/query, backend registry, runner,
   and the closest existing `test_*_online_collection.py` vertical slice

Search from both the operation class and perf namespace through producers and
consumers:

```bash
rg -n "<Operation>|<perf_namespace>|query_<op>|measurement_request" \
  src collector tests
rg -n "LazyOpEntry|PreparedMeasurement|ResourceContract|query_with_resolution" \
  src collector tests
```

Record the existing ordinary-query behavior before changing it: exact key
fields, interpolation/fallback behavior, scaling, phase selection, backend and
version routing, and every compound parent that consumes the operation.

## Expected edit map

A normal operation slice changes only the layers that own its semantics:

| Layer | Typical change |
|---|---|
| `src/aiconfigurator/sdk/operations/<family>.py` | Shared normalization, exact request construction, resolution-aware lookup |
| `src/aiconfigurator/collector/<backend>/<op>_adapter.py` | Request-to-case/resource/record conversion |
| `src/aiconfigurator/collector/<backend>/<op>.py` | Exact one-case runtime runner |
| `src/aiconfigurator/collector/<backend>/registry.py` | One namespace/backend/version lazy route |
| `tests/unit/sdk/resolution/test_<op>_online_collection.py` | Pure, identity, exact-hit, cold/warm/reopen, failure ablations |
| `tests/unit/collector/lazy/` | Adapter, package, executor, resource, and optional GPU gates |

Keep installable runtime code under `src/aiconfigurator`. Add or update a
top-level `collector/` compatibility wrapper only when an existing source-mode
entry point requires it. A normal operation slice should not add
operation-specific logic to `session.py`, `overlay.py`, `executor.py`,
`scheduler.py`, or generic evidence types.

## Step 1: Classify the operation boundary

Choose exactly one contract:

1. **Deterministic** — no measured data; retain ordinary computation and add no
   JIT adapter.
2. **Composition-only** — trace the existing physical children and preserve the
   existing sum/max/fallback composition.
3. **Measured leaf** — emit one physical `PerfKey` and one exact benchmark case.
4. **Measured compound** — emit one full-module key and do not recurse into its
   internal kernels, because that would double count.

If later child selection depends on a provisional measured latency, generic
tracing is unsafe. Expose deterministic child traversal or leave the operation
structured-unsupported; do not return a partially exact callback.

## Step 2: Establish one normalization contract

Create or reuse one operation-local normalization helper. The same normalized
values must feed:

- the ordinary database lookup;
- `PerfKey.query` in resolving mode;
- `MeasurementRequest.query`;
- the adapter's exact benchmark case;
- the persisted row consumed on replay.

Do not create a parallel shape forecast, a second key builder, or a synthetic
aggregate key for a composition-only parent. Test boundary shapes, TP/EP/CP or
head sharding, context versus generation, dtype/quantization, and any correction
factor applied after the physical lookup.

Before coding, write a temporary contract table:

| Field | Runtime source | Ordinary query | PerfKey | Benchmark case | Persisted row |
|---|---|---|---|---|---|

Every row should have one authoritative conversion. If two columns disagree,
fix the normalization seam before adding collection.

## Step 3: Define exact identity and compatibility

Use the existing evidence types:

- `PerfKey.namespace` is the dataset discriminator.
- `PerfKey.query` contains the exact physical lookup fields.
- `PerfKey.environment` prevents incompatible system/backend/version/runtime,
  model-boundary, or topology reuse.
- `MeasurementProtocol` identifies timer, sample statistic, warmups/replays, and
  tuning revision independently from the physical key.
- Physical GPU IDs, rank PIDs, invocation IDs, and timestamps are provenance,
  not reusable identity.

An overlay hit or a literal compatible curated row may satisfy resolution.
Interpolation, extrapolation, empirical fallback, clamping, or a nearby bucket
may help discover later children but cannot satisfy an exact resolved callback.

## Step 4: Add the operation hook without changing pure prediction

Follow the closest existing operation implementation rather than inventing a
new session API. Preserve these semantics:

1. With no `ResolutionSession`, execute the original query path byte-for-byte
   behaviorally: same result, source, scaling, exceptions, and Rust fast path.
2. With a session, normalize once and build the exact request.
3. Probe the compatible overlay first, then literal curated exact evidence.
4. On a hit, convert the record to the normal `PerformanceResult` and apply the
   operation's existing scaling exactly once.
5. On a miss, record the `MeasurementRequest` and consumer in the callback-local
   `MissSet`, mark the callback tainted, and continue only far enough to expose
   deterministic remaining dependencies.
6. Never let the provisional value escape the callback, advance Replay time,
   become an overlay row, or satisfy a candidate score.

Repeated physical keys deduplicate in `MissSet`, but retain every consumer so
the one measured value contributes at every original replay position.

## Step 5: Add one import-light adapter route

Register exactly one unambiguous route for
`(namespace, backend, backend_version)`. The adapter should:

1. Validate the request environment and operation-specific supported domain.
2. Convert the exact normalized query into one benchmark case without snapping
   to a grid.
3. Return the real `ResourceContract`: GPU count, exclusivity, fabric
   requirement, and whether the whole fabric domain must be reserved.
4. Convert raw worker output into a `MeasurementRecord` with the identical
   `PerfKey` and bound protocol.
5. Reject invalid, mismatched, non-finite, zero, or incomplete evidence before
   the parent can append it.

The registry and adapter must remain import-light. Torch, SGLang, CUDA, and
other heavy runtime imports belong in worker processes.

## Step 6: Implement the exact one-case runner

Reuse the real framework implementation and execute the exact encountered
shape. The runner should:

- use deterministic inputs and explicit seeds where initialization matters;
- perform the protocol's warmups and independent timed samples;
- use the operation-appropriate device timer, normally CUDA events;
- synchronize at the documented timing boundary;
- prove which kernel/module path ran and reject unintended fallbacks;
- aggregate multi-rank samples according to the physical contract, such as the
  elementwise slowest rank for a collective;
- return samples plus framework, version, kernel, device, topology, rank, and
  invocation provenance;
- never write curated data or the overlay directly.

The parent `ResolutionSession` remains the authoritative validator and overlay
writer. A valid partial record may survive another request's failure, but the
current callback fails until all of its physical keys are exact.

## Step 7: Wire through the existing ResolutionSession

Operation support should normally require no operation-name branch in
`ResolutionSession`, the scheduler, executor, or overlay. Those components are
generic:

- `MissSet` deduplicates keys and retains consumers.
- `PersistentMeasurementExecutor` binds routes, plans waves, reuses compatible
  workers, correlates invocations, and restores request order.
- `HardwareAwareScheduler` packs disjoint one-GPU jobs and isolates conflicting
  device/fabric domains.
- `OverlayStore` validates append-only evidence and serves compatible hits.
- `ResolutionSession.execute_callback()` discards a tainted first result,
  resolves pending misses, and replays once.

If the operation appears to require a core resolver branch, first prove the
missing capability with a model-agnostic failing test. Do not add model names,
operation names, or perf filenames to generic resolution modules.

## Step 8: Develop in strict TDD using ablations

Add the smallest failing test at each boundary, then the smallest production
change. Keep the following ablations distinct so a broad passing test cannot
hide the broken layer.

### A. Pure/default path

- No session preserves the original result and exception behavior.
- Pure prediction imports no worker runtime and performs no GPU work.
- Live Mocker without `aic_resolution` remains resolution-free; supported V1.3
  Live tests must separately prove calibration/effect transactionality.

### B. Normalization and identity

- Ordinary lookup, `PerfKey`, request, case, and emitted row agree exactly.
- Boundary parallelism/dtype/phase shapes neither alias nor double-correct.
- Protocol or environment mismatch fails rather than reusing evidence.

### C. Exact-source precedence

- Compatible overlay hit performs no collection.
- Compatible literal curated hit performs no collection.
- An interpolated or empirical ordinary result still records an exact miss.

### D. Observe-only

- The callback reports the unique misses and consumers.
- No worker or overlay mutation occurs.
- A tainted result is not presented as fully resolved.

### E. Measure-on-miss with a fake executor

- Cold callback measures once per unique `PerfKey`, appends, and requeries once.
- Same-process warm callback issues zero worker commands.
- Close/reopen overlay callback issues zero worker commands.
- Cold, warm, and reopened final results and evidence identities match.

### F. Composition and deduplication

- Repeated children measure once but contribute at every replay position.
- Sum/max/fallback behavior is unchanged.
- Measured compounds do not also collect internal leaves.

### G. Failure and lifecycle

- Unsupported shapes/routes produce structured unresolved reasons.
- Invalid samples, protocol mismatch, stale replies, timeout, cancellation, and
  child death never become zero latency or poison the overlay.
- Budgets count unique physical keys, not consumers or transient attempts.
- Teardown is bounded, idempotent, and preserves the primary failure.

### H. Package and real-hardware gates

- Extracted-wheel route/adapter smoke succeeds without a source checkout.
- Adapter import does not import the heavy framework.
- The real GPU gate proves exact device/fabric placement, kernel route, positive
  samples, record provenance, cold one-measurement behavior, warm/reopen zero
  commands, and clean teardown.
- For persistent runtimes, compare repeated persistent calls with isolated
  setup controls; distinguish kernel latency from orchestration wall time.

## Replay and Spica integration gate

After the isolated operation vertical slice is green, exercise it from the
actual AIC callback driven by scheduled aggregate ForwardPassMetrics:

```text
num_prefill_requests
sum_prefill_tokens
sum_prefill_kv_tokens
num_decode_requests
sum_decode_kv_tokens
```

Do not key or estimate from queued metrics, wall time, or FPM variance. Preserve
the existing prefill-only, decode-only, and mixed projection paths.

Spica owns the explicit policy, overlay path, budget, GPU assignment, and safe
evaluator parallelism. Absence of resolution configuration preserves the pure
path. In resolving Replay:

1. Wrap one actual AIC callback in one Python `ResolutionSession`.
2. Do not advance virtual time after a tainted first walk.
3. Return latency only after the exact replay succeeds.
4. Map structured resolution failure to an unscorable candidate, never zero or
   mixed evidence.
5. Run a deterministic cold FPM stream containing prefill-only, mixed,
   decode-only, and a repeated shape.
6. Close/reopen the overlay and run the identical warm stream.
7. Assert every missing reachable key was measured once and the warm stream
   issued zero GPU commands with identical final latency/evidence identity.

## Validation commands

Adapt paths to the operation, then run at least:

```bash
.venv/bin/pytest -q tests/unit/sdk/resolution/test_<op>_online_collection.py
.venv/bin/pytest -q tests/unit/sdk/resolution tests/unit/collector/lazy
.venv/bin/ruff check <changed-python-files> <changed-tests>
.venv/bin/ruff format --check <changed-python-files> <changed-tests>
git diff --check
```

Run the repository's broader unit selection before freeze. Hardware-dependent
tests must be opt-in and ordinary CPU test discovery must skip them cleanly.

## Definition of done

Report all of the following before calling the operation supported:

- operation classification and why it does or does not recurse;
- canonical query/PerfKey/case/row mapping;
- namespace, environment, protocol, backend/version route, and resource contract;
- unchanged no-session behavior;
- exact curated and overlay hit behavior;
- cold command count, unique-key count, replay count, warm command count, and
  reopened-overlay command count;
- compound consumer multiplicity and no-double-count evidence;
- structured failure and bounded-teardown results;
- clean-wheel/import-light validation;
- exact GPU/framework/kernel/topology provenance when hardware support is claimed;
- Replay/Spica cold/warm FPM result, or an explicit statement that only the
  isolated AIC operation slice is complete;
- remaining unsupported shape or profile boundaries.

Do not describe package tests as physical validation, a successful timing as
consumer integration, or an isolated operation gate as full Replay release.
