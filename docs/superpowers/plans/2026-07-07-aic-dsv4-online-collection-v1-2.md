# AIC DSv4 Online Collection V1.2 Execution Plan

> **Status:** Active user-directed scope for AIC-1326, derived from the
> review-ready design at
> `docs/superpowers/specs/2026-07-07-aic-dsv4-online-collection-v1-2-design.md`.
> This plan supersedes conflicting shape-propagation details in the 2026-07-06
> plans; it is not yet a claim that the V1.2 design has passed its milestone
> review gate.

**Goal:** Complete the global lazy-resolution architecture for one frozen,
aggregated SGLang DeepSeek-V4 profile while keeping pure prediction unchanged
and keeping every resolver/runtime primitive model-agnostic.

**Current baseline:** Hardware discovery is committed as `badd73c5`. The
generic deterministic wave scheduler is the active milestone. The completed
lazy-resolution core remains the substrate; V1.2-covered operations will
converge their lookup and request construction onto one normalization path.

## Scope reconciliation

The following older-plan details remain valid:

- `2026-07-06-aic-lazy-perf-core.md`: `PerfKey`, `MissSet`,
  `MeasurementRequest`, `MeasurementRecord`, `ResolutionSession`, append-only
  overlay ownership, tainted first walk, and one replay;
- `2026-07-06-aic-hardware-aware-collector-runtime.md`: packaged registry
  metadata, hardware discovery, generic wave scheduling, adapter loading,
  persistent subprocess workers, BF16 GEMM, and NCCL as generic gates;
- `2026-07-06-dynamo-spica-aic-resolution-integration.md`: structured callback
  failure, an opt-in replay path, pure default behavior, Spica overlay policy,
  and cold/warm lifecycle validation.

The following details are superseded for V1.2:

- scheduled aggregate `ForwardPassMetrics`, not a
  `ConcreteBatchDescriptor`, are the runtime workload;
- covered operations do not maintain a forecast-only request graph or a second
  normalization path;
- `PerfKey.namespace` is the only persisted dataset discriminator; do not add
  `EvidenceQuery`, `dataset_id`, or `collector_ref`;
- registry routing is exactly `(namespace, backend, backend_version)`;
- the frozen profile is TP4/DP1/CP1/PP1, MTP disabled, aggregated serving;
- CustomAllReduce, not generic NCCL, is the profile's measured communication
  leaf. NCCL remains a non-DSv4 multi-GPU regression gate.

## Cross-cutting invariants

1. Generic resolver, evidence, registry, scheduler, and executor modules contain
   no model name, DSv4 operation name, or DSv4 perf-filename branch.
2. Every covered physical lookup has one normalization helper shared by
   ordinary lookup, literal-exact probing, and resolving-key construction.
3. The actual operation query path discovers dependencies. A provisional
   approximate value may expose later deterministic children, but it taints the
   callback and can never be returned, persisted, or used for scheduling.
4. Each reachable operation is exactly one of measured leaf, deterministic,
   composition-only, or structured-unsupported. A measured compound does not
   also recurse into implementation children.
5. Workers measure exact canonical cases independently. Grouping amortizes
   runtime setup; it never creates a fused latency record.
6. The parent ResolutionSession is the only overlay writer. Valid partial
   records survive sibling failures; the callback remains unresolved until all
   keys are exact.
7. Heavy Torch, SGLang, CUDA, and distributed imports occur only in workers.
8. Council review is one end-of-milestone gate. Implementation and review fixes
   inside a milestone use strict RED-GREEN-refactor tests and independent
   non-Council audits.

## Milestone 0: Freeze the generic scheduler

Finish only the already-active Task 3 scheduler before pivoting to the V1.2
actual-query convergence. Adapter routing and execution resume after that
convergence so they consume the final identity and capability contracts.

### 0A. Deterministic scheduler

- [x] Start from a missing-module import RED.
- [x] Sort jobs by `(-gpu_count, request_digest)` and enumerate ascending
  physical GPU combinations.
- [x] Require all four directed P2P read/write capabilities for every selected
  pair; require symmetric `NV#` paths as an additional NVLink gate.
- [x] Treat stable fabric-domain labels as contention identities, never as
  connectivity proof.
- [x] Track occupied domains separately from exclusive reservations so a
  collective excludes every other occupant regardless of job order.
- [x] Keep single-GPU reservations GPU-local, including when a caller sets the
  fabric reservation flag.
- [x] Pin the exact frozen-profile shape: a four-GPU NVLink collective owns the
  whole domain and cannot co-run with any one-GPU job; independent one-GPU jobs
  pack together.
- [ ] Run the focused, lazy-collector, and complete collector suites; then use
  one milestone Council and commit the exact reviewed patch.

## Milestone 1: Actual-query trace and key convergence

### 1A. Pin one normalization path

For a fake measured operation, begin with tests showing that ordinary lookup,
literal-exact lookup, and resolving lookup receive the same normalized mapping.
The ordinary table may interpolate an off-grid point, while resolving mode must
record an exact miss for that same normalized point.

Refactor covered operations so raw kwargs are normalized once beside their
existing query semantics. A temporary `measurement_request()` method may call
that helper during migration, but it may not normalize independently.

### 1B. Pin taint and compound boundaries

Add generic tests with no DSv4 imports:

- a miss taints the first callback and its provisional aggregate cannot escape;
- composition-only compounds traverse actual selected children;
- measured compounds emit one boundary key and do not recurse;
- repeated child keys measure once but contribute at every replay position;
- branch selection for covered compounds depends only on configuration/shape,
  never on provisional latency;
- one replay is the hard maximum and any approximate/unresolved replay source
  fails structurally.

### 1C. Capability preflight

Implement the generic classification/preflight contract first with fake routes.
The full profile pass runs only after Milestones 2-3 install every adapter.
That final pass builds the frozen model and checks one registry route, exact
capability, lightweight imports, resource-contract construction, GB200
topology, and protocol/timer/tuning compatibility. Runtime tracing remains
authoritative for the exact shape-local dependency set.

## Milestone 2: Adapter/executor runtime and frozen-profile one-GPU adapters

### 2A. Adapter reverse lookup

Write CPU-only RED tests before `aiconfigurator.collector.adapters` exists.
They must prove:

- one route for `(PerfKey.namespace, backend, backend_version)`;
- missing and ambiguous routes fail capability preflight;
- protocol, timer, tuning, system, GPU class, and topology mismatches fail
  before resource acquisition;
- the lightweight adapter may translate final canonical field names/order but
  cannot reinterpret raw runtime inputs or mutate the key;
- worker imports use installable namespaced modules and never the repository
  root `collector` package.

Implement the smallest reverse index over existing `OpEntry`/`LazyOpEntry`
metadata. `CollectionJob.adapter_namespace` is transient dispatch metadata, not
a persisted identity and not a replacement `collector_ref`.

### 2B. Persistent worker executor

Write fake-channel RED tests for submit-all-before-receive wave behavior,
request-order results, worker lease reuse, invocation correlation, deadline and
cancellation accounting, stale reply rejection, fail-closed eviction, partial
success preservation, and parent-only overlay ownership.

The worker sets `CUDA_VISIBLE_DEVICES` from assigned UUIDs before heavy imports.
Local worker ordinals are `0..N-1`. One-case runners return raw samples; the
parent validates records and the ResolutionSession appends them.

### 2C. Generic release gates first

- BF16 GEMM: exact one-case collection, literal curated reuse, interpolation as
  an exact miss, cold collection once, warm zero-command reuse.
- NCCL: a generic multi-GPU persistent-group test proving topology-aware
  scheduling and fabric isolation. It is not a DSv4 operation dependency.
- Mixed fake workload: disjoint one-GPU saturation plus collective isolation.
- Pure prediction: API/behavior compatible and GPU-free.

### 2D. Frozen-profile measured leaves

Add adapters in this order, each behind runtime-input -> normalized query ->
PerfKey -> case -> record -> identical PerfKey round-trip tests:

| Family | Required capability | Resource |
|---|---|---|
| GEMM | BF16 and `fp8_block` profile shapes | one GPU |
| DeepSeekV4MHCModule | BF16 pre/post full-module keys | one GPU |
| DSv4 CSA/HCA context/generation | four full-module namespaces, TP4 simulation, FP8 KV | one GPU |
| MoE | `fp8_block`, TP1/EP4, `power_law_1.01` synthetic local-rank runner | one GPU |

Embedding and ElementWise remain explicitly reviewed deterministic operations.
MoEDispatch and OverlapOp are composition-only. P2P is a deterministic no-op at
PP1. The measured profile uses `sgl-project/DeepSeek-V4-Flash-FP8`, bfloat16
module tables, FP8 KV cache, the existing non-DeepEP/non-MegaMoE dispatch path,
and SGLang 0.5.10. Tests keep scale/multiplicity out of physical identity and
include model-artifact compatibility only where a persisted module schema
cannot distinguish artifacts. Reject shapes outside TP4/DP1/CP1/PP1, MTP0,
aggregated SGLang 0.5.10, GB200 capability before launching a worker.

Offline sweeps may call the same exact runner. Lazy execution must never invoke
a grid sweep, append a curated file, snap an FPM shape, or use an offline
logging path.

## Milestone 3: Four-GPU CustomAllReduce

### 3A. CPU contract and placement REDs

Pin one canonical key for `half`, world size four, operation, and exact element
count. Its resource function returns
`ResourceContract(4, NVLINK, reserve_fabric_domain=True)`. Scale/multiplicity and
physical GPU IDs stay out of the key.

### 3B. Persistent four-rank runner

Factor an exact SGLang CustomAllReduce one-case runner. Spawn one rank process
per assigned UUID, initialize its four-rank runtime once, and reuse it for
compatible cases. Correlated commands
carry exact case fields, warmups, and samples. Every rank completes; only the
validated aggregate result becomes a record. Timeout, EOF, protocol mismatch,
or child death evicts the whole rank group.

### 3C. GB200 isolation gate

On one four-GPU NVLink-connected GB200 node, compare repeated isolated controls
with the persistent runner using a documented robust interval. Prove the
collective occupies the whole fabric domain, no one-GPU job overlaps it, and a
warm identical key sends no second collective command. The earlier H100/H200
runtime preflight does not satisfy this gate; survey or allocate a real GB200
lease and retain its raw discovery/provenance before claiming profile support.

## Milestone 4: Aggregate FPM, Mocker, and Spica end to end

### 4A. Re-anchor the Dynamo source boundary

Materialize a clean Dynamo worktree that contains the approved integration
baseline `1fe16eb6b2e5318d78f5ece7733e054bab7ef938` (or explicitly review and
record a newer replacement). Do not apply the integration plan to the current
local Dynamo head merely because it is nearby; verify ancestry, Mocker FPM
fields, Replay error propagation, and Spica configuration anchors first.

### 4B. Replace the superseded descriptor path

Do not add `ConcreteBatchDescriptor` or per-request vectors. Pass the scheduled
aggregate FPM fields already consumed by AIC:

- `num_prefill_requests`, `sum_prefill_tokens`, `sum_prefill_kv_tokens`;
- `num_decode_requests`, `sum_decode_kv_tokens`.

Preserve existing prefill-only, decode-only, and mixed three-pass projection
rules. Ignore queued metrics, wall time, and FPM variance for identity and
estimation. With MTP disabled, `num_decode_requests` is the scheduled decode
token count.

### 4C. Structured replay integration

Keep the pure Rust/default AIC path unchanged. The opt-in replay path activates
one Python ResolutionSession around the actual AIC operation callback. A tainted
first walk never advances virtual time. Structured unresolved failures make the
candidate unscorable; they never become zero latency or mixed evidence.

Spica owns explicit resolution policy, overlay path, budget, and safe evaluator
parallelism. Live Mocker remains collection-free in V1.2.

Extend the resolution report with FPM/rank/counter identity, selected phase
path, consumer counts, registry routes, assignments/waves, worker/invocation
identities, samples/protocol/provenance, partial failures, replay outcome, and
the exact overlay records used by the final score.

### 4D. Deterministic cold/warm release gate

Use an aggregated replay with prefill-only, mixed, decode-only, and one repeated
shape. Starting from an empty overlay plus normal curated data, prove:

1. compatible literal curated points are reused;
2. every missing reachable key is measured once per unique PerfKey;
3. no reachable operation is unsupported;
4. no provisional/interpolated value escapes;
5. the cold replay completes;
6. overlay close/reopen preserves evidence;
7. the identical warm replay issues zero GPU measurement commands;
8. cold/warm final latencies and evidence identities match.

## Milestone validation and handoff

Every milestone ends with:

1. the strict RED evidence and smallest GREEN change recorded in the scratch
   ledger and AIC-1326;
2. focused tests, broader relevant regression tests, Ruff/format/diff checks,
   and packaging/import smoke;
3. a stable exact patch hash;
4. one four-provider Council gate, with fixes folded back through tests;
5. an exact-scope commit and Linear/scratch/dashboard update.

The final release additionally requires a real four-GPU GB200 provenance bundle
containing raw hardware discovery, normalized assignments, worker/rank identity,
protocol/sample metadata, overlay records, and the cold/warm Mocker reports.
