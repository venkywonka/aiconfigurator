# AIC Offline-vs-Online Gap Analysis Skill

**Status:** Approved design pending written-spec review

**Date:** 2026-07-14

**Scope:** Add a repository-local, end-to-end skill that repeatedly compares
offline AIC prediction with online exact resolution over identical frozen
Mocker FPM streams, using real GPU collection and producing mechanically
validated evidence plus a self-contained analysis dashboard.

## Decision summary

The skill lives in AIC under
`.agents/skills/aic-offline-online-gap-analysis/`. It is a deterministic
campaign orchestrator over existing AIC and Dynamo entry points, not a third
estimator or a second FPM decomposition path.

Every authoritative campaign:

1. resolves and locks the AIC checkout, Dynamo checkout, coherent runtime
   artifact, hardware requirements, and ordered cluster pool;
2. freezes the exact scheduled Mocker FPM stream and records its SHA-256
   digest;
3. replays that stream through one canonical AIC-owned FPM walker, first with
   online resolution absent and then with a `ResolutionSession` enabled;
4. performs a campaign-wide cold, warm, and process-reopen lifecycle against
   one exact overlay;
5. fails closed on shape drift, unresolved operations, incoherent runtime
   provenance, lease violations, persistence failure, or repeated warm GPU
   work;
6. reports numerical offline-vs-online differences as ranked findings rather
   than automatically treating the existence of a gap as campaign failure;
7. reruns the most suspicious shapes against isolated empty overlays for
   shape-local attribution; and
8. preserves a machine-readable evidence bundle and a self-contained HTML
   dashboard.

The skill may allocate compute, submit jobs, and write scratch artifacts. It
does not edit AIC, Dynamo, SGLang, or other production source during a
campaign. If no compatible real GPU allocation is available after exhausting
the declared cluster pool, the campaign stops as blocked. CPU, fake-executor,
or historical evidence never substitutes for an authoritative run.

## Goals

1. Make the previously manual AIC offline-vs-online experiment repeatable from
   any compatible AIC worktree.
2. Compare identical scheduled FPMs through identical AIC composition
   semantics so resolution mode is the controlled variable.
3. Prove cold measurement, exact persistence, same-process warm reuse,
   process-reopen reuse, and zero repeated GPU work.
4. Show whether exact hit rate converges as the same Mocker FPM stream is
   replayed.
5. Attribute the end-to-end modeled-latency delta to the exact AIC operations
   and physical performance keys responsible for it.
6. Distinguish modeled latency, collection wall time, orchestration wall time,
   and lookup overhead.
7. Maximize compatible GPU use through AIC's existing hardware-aware leasing
   and scheduling rather than adding skill-local placement semantics.
8. Retain enough provenance and raw evidence for an independent reviewer to
   reproduce or challenge every conclusion.
9. Support multiple AIC models and backends through explicit campaign
   manifests, with DSV4 on SGLang/GB200 as the first reference campaign.
10. Stop with an honest blocked or invalid status when the required evidence
    cannot be produced.

## Non-goals

The skill does not:

- add a new prediction API, estimator, graph IR, FPM walker, or operation
  dependency forecast;
- independently implement Mocker scheduling or AIC operation composition;
- compare arbitrary commands whose shape and decomposition identity cannot be
  proven;
- use the legacy split prefill/decode estimator as a mandatory comparison
  lane;
- patch source code, repair gaps, recollect curated grids, or promote overlay
  rows into released perf data;
- modify SGLang internals, permit runtime source grafts, or assemble a hybrid
  framework tree;
- auto-discover and use arbitrary hosts from `~/.ssh/config`;
- accept GPU-less tests, fake executors, or one isolated microbenchmark as
  production evidence;
- impose one universal numerical accuracy threshold across models; or
- update Linear, GitHub, dashboards outside the campaign workspace, or other
  external systems unless separately requested.

## Ownership boundaries

| Concern | Owner |
|---|---|
| Request scheduling and scheduled FPM construction | Mocker/Dynamo |
| Frozen FPM replay and existing A/B lifecycle | Dynamo Replay `aic_ab.py` |
| FPM-to-operation walk and composition | AIC canonical forward-pass walker |
| Offline prediction behavior | AIC with no `ResolutionSession` |
| Miss discovery, exact collection, persistence, and replay | AIC online resolution |
| Physical benchmark implementation | AIC canonical backend runner using the coherent framework artifact |
| GPU placement within the declared lease | AIC hardware-aware scheduler and executor |
| Campaign policy and model-specific inputs | Campaign manifest |
| Preflight, orchestration, receipts, validation, analysis, and rendering | This skill |

The skill must invoke the existing Dynamo A/B harness. It must not copy that
harness's scheduled-shape gating, cold/reopen semantics, or prediction
composition into a new runner. Where additional observability is required,
the skill uses a recorder around the canonical AIC walker; it never enumerates
or predicts the operation graph independently.

## Skill package

The proposed repository-local package is:

```text
.agents/skills/aic-offline-online-gap-analysis/
├── SKILL.md
├── references/
│   ├── campaign-schema.md
│   ├── evidence-contract.md
│   └── cluster-policy.md
├── scripts/
│   ├── run_campaign.py
│   ├── preflight_campaign.py
│   ├── freeze_fpm.py
│   ├── validate_evidence.py
│   ├── analyze_deltas.py
│   └── render_dashboard.py
└── evals/
    └── evals.json
```

`SKILL.md` owns the orchestration sequence, decision rules, stop conditions,
and reporting language. `run_campaign.py` owns the resumable stage graph and
invokes existing entry points; it does not execute prediction logic itself.
The remaining scripts make schema validation, FPM freezing, evidence gates,
attribution, and rendering deterministic. They do not contain model-specific
shape logic or production prediction behavior.

The skill may call existing repository tools directly. A script is added only
where a mechanical contract needs stable enforcement across agents. In
particular, paired execution continues to use Dynamo's
`components/src/dynamo/replay/aic_ab.py` and its existing invariant helpers.

## Campaign manifest

Model and environment variation is expressed through a versioned manifest.
The conceptual schema is:

```yaml
schema_version: 1
campaign: dsv4-gb200-offline-online

repositories:
  aic:
    path: /path/to/aiconfigurator
    commit: <expected-sha>
  dynamo:
    path: /path/to/dynamo
    commit: <expected-sha>

runtime:
  artifact: <container-reference>
  digest: <immutable-digest>
  backend: sglang
  backend_version: 0.5.10
  model: deepseek-v4

hardware:
  gpu_models: [GB200]
  min_gpus: 4
  fabric: nvlink
  cluster_pool:
    - host: lyris
      scheduler: slurm
    - host: bia
      scheduler: slurm
    - host: polyphe
      scheduler: slurm

fpm:
  trace: frozen-fpm.jsonl
  digest: sha256:<digest>

execution:
  warm_iterations: 5
  reopen_iterations: 2
  isolated_outliers: 3
```

The committed DSV4 manifest is a reference fixture, not hard-coded skill
policy. A real invocation resolves the manifest into `manifest.lock.json`,
which includes:

- exact AIC and Dynamo commit hashes;
- whether each executed checkout is clean and import-safe;
- container reference and immutable digest;
- framework, Python, Torch, CUDA, driver, and NCCL identities when applicable;
- complete discovered hardware and fabric inventory;
- scheduler allocation and worker-visible leases;
- FPM trace path, digest, count, and ordering identity;
- resolution protocol, overlay schema, budgets, and iteration counts; and
- every optional diagnostic threshold.

Any source tree whose Python, Rust, native extension, or framework code is
executed must be clean and identified. An orchestration-only checkout may be
dirty only when it is not imported, bind-mounted, packaged, or otherwise used
as runtime source. Untracked files capable of affecting imports invalidate the
run.

## Frozen FPM contract

The exact scheduled FPM stream is the authoritative experiment input.

The freezer captures the ordered scheduled passes once and writes canonical
JSON Lines plus a SHA-256 digest. Canonicalization includes all fields consumed
by the AIC walker, request ordering, and any model/topology identity required
to interpret the pass. The trace is immutable after the first comparison arm
starts.

Every offline, cold-online, warm-online, reopened-online, and isolated-outlier
arm must prove:

- the same trace digest;
- the same pass count and ordering;
- the same scheduled-shape digest per pass;
- the same request identities and multiplicities; and
- the same model, topology, and AIC configuration.

An arm that regenerates an apparently equivalent FPM is not comparable. The
legacy Mocker split prefill/decode callback may be run as an explicitly labeled
compatibility diagnostic, but its result never enters the canonical
offline-vs-online delta.

## Canonical comparison lanes

The experiment has two required prediction lanes:

1. **Offline AIC:** invoke the canonical AIC forward-pass walker with no
   `ResolutionSession`. Existing SILICON/HYBRID behavior is unchanged.
2. **Online AIC:** invoke the same walker over the same frozen FPM with the
   configured `ResolutionSession`. Exact misses are measured, persisted, and
   replayed through the same operation composition.

The authoritative online lane uses exact, fail-closed resolution. A HYBRID
sidecar or other degraded result may be exercised as a separately labeled
diagnostic, but it cannot satisfy exact closure or enter the canonical
offline-vs-online delta.

The steady-state modeled-latency comparison uses offline AIC versus reopened
online AIC. Cold collection wall time is reported separately and is never
added to modeled GPU duration. The dashboard shows at least:

- modeled pass latency;
- paired absolute and relative offline-vs-online delta;
- cold collection wall time;
- end-to-end orchestration wall time;
- ordinary warm lookup overhead; and
- repeated-run variance for every wall-time metric.

## Campaign lifecycle

The resumable stage graph is:

```text
PREFLIGHT
  -> FREEZE_FPM
  -> ALLOCATE
  -> OFFLINE
  -> ONLINE_COLD
  -> ONLINE_WARM
  -> PROCESS_REOPEN
  -> VALIDATE
  -> ANALYZE
  -> ISOLATE_OUTLIERS
  -> RENDER
```

Both required lanes run in the same immutable runtime artifact and allocation.
The offline lane may execute before any measurement worker is started, but it
does not use a different host dependency closure merely because it is
collection-free.

The main lifecycle uses one campaign-wide overlay and the frozen trace's fixed
ordering:

1. Start with an empty exact overlay and fallback location.
2. Run the complete online stream once and capture pre-measurement exact hits,
   complete miss sets, measurement commands, accepted records, and final
   results.
3. Replay the identical stream for the configured same-process warm
   iterations.
4. Close every AIC/Dynamo owner, reopen the persisted overlay in a fresh
   process, and replay the identical stream for the configured reopen
   iterations.
5. Validate exact closure and zero repeated GPU measurement work.
6. Rank shape-level and operation-level deltas.
7. Rerun the top `isolated_outliers` shapes, plus any shape with an evidence
   anomaly, against a fresh isolated overlay.

The shared campaign overlay measures realistic cumulative reuse. The isolated
reruns distinguish shape-local prediction gaps from order-dependent reuse or
evidence inherited from earlier shapes.

## Hardware and cluster orchestration

The manifest provides an ordered cluster allowlist and capability constraints.
The skill does not scan arbitrary SSH hosts.

For each allowed cluster, preflight records reachability, authentication,
scheduler availability, compatible GPU type/count, fabric, runtime-artifact
availability, and allocation result. It selects the first compatible available
allocation and moves to the next allowed cluster only for capacity,
authentication, scheduler, or artifact-availability blockers.

Inside an allocation, AIC's existing coordinator and hardware-aware scheduler
remain authoritative:

- independent single-GPU requests may run concurrently on disjoint leases;
- multi-GPU operations reserve their required GPU subset and fabric domain;
- overlapping leases do not benchmark concurrently;
- duplicate exact keys coalesce into one measurement;
- complete inventory, scheduler lease, and worker-visible devices are distinct
  provenance fields; and
- accepted records are normalized and ordered by `PerfKey`, so concurrency
  cannot change the evidence bundle.

The skill may configure concurrency and resource limits. It must not perform
its own operation placement or bypass the AIC scheduler. If worker attestation
does not prove placement within the scheduler lease, the evidence is invalid.

## Runtime coherence

Authoritative evidence requires one coherent immutable runtime artifact. The
same artifact digest is used for both required comparison lanes.

AIC may configure public framework flags and invoke supported public or stable
APIs. The campaign must reject:

- partial source overlays;
- files copied from a different framework build;
- runtime monkey patches that alter framework implementation;
- bind mounts that replace installed framework modules;
- missing or ambiguous container digests; and
- observed framework, CUDA, Torch, or NCCL identities incompatible with the
  declared evidence environment.

Diagnostic experiments using patched runtimes may be preserved separately,
but they cannot satisfy the skill's authoritative evidence contract.

## Evidence bundle

Every campaign writes the following stable structure under its scratch
workspace:

```text
campaign/
├── manifest.lock.json
├── provenance.json
├── state.json
├── fpm/
│   ├── frozen.jsonl
│   └── digest.json
├── jobs/
│   ├── allocation-attempts.jsonl
│   └── worker-leases.jsonl
├── raw/
│   ├── offline/
│   ├── online-cold/
│   ├── online-warm/
│   ├── online-reopen/
│   └── isolated-outliers/
├── normalized/
│   ├── comparisons.json
│   ├── cache-trajectory.json
│   ├── op-deltas.json
│   └── isolated-outliers.json
├── validation-receipt.json
└── dashboard.html
```

Raw worker output is immutable. Normalization and reporting create new
artifacts without rewriting physical-run evidence.

Each stage receipt contains its input digest, output digests, resolved command,
environment identity, start/end timestamps, status, allocation, and worker
provenance. A completed stage is reusable only when every input identity still
matches. Changing the manifest, FPM trace, commit, image digest, protocol, or
runtime invalidates that stage and all descendants.

## Mechanical evidence gates

`validate_evidence.py` produces one `validation-receipt.json`. A campaign is
not authoritative unless all applicable gates pass:

1. **Shape identity:** all arms use the identical frozen FPM and per-pass
   scheduled-shape digests.
2. **Lane identity:** offline resolution is absent and online resolution is
   enabled with the locked policy.
3. **Exact closure:** the cold online stream finishes with no unresolved
   physical dependency when exact mode is required.
4. **Record validity:** accepted exact records have finite positive samples,
   the requested `PerfKey`, the bound protocol, and complete environment and
   hardware provenance.
5. **Persistence:** the reopened process reads the expected overlay identity
   and reproduces the cold final evidence identities and modeled results.
6. **Zero repeated work:** same-process warm and process-reopen iterations
   issue zero new GPU measurement commands, accept no new exact records, and
   consume no collection wall time.
7. **Lease isolation:** worker-visible devices and topology are compatible
   subsets of the scheduler lease and full discovered inventory.
8. **Runtime coherence:** observed runtime identities and image digest match
   the locked manifest, with no source grafts.
9. **Attribution reconciliation:** the recorded per-operation totals reconcile
   with the canonical end-to-end prediction within a documented floating-point
   tolerance.
10. **Raw-evidence preservation:** every normalized claim links to immutable
    raw artifacts and physical-run provenance.

Zero collection alone does not prove persistent exact reuse. A reopened pass
must also prove exact hits for the same physical identities and equal final
modeled results.

## Cache-convergence metrics

For every pass and lifecycle iteration, the skill reports:

- initial compatible exact hits and exact lookups;
- initial exact hit rate;
- required and resolved unique physical keys;
- miss-set size and consumer multiplicity;
- collection callbacks and GPU worker commands;
- accepted, rejected, and unresolved records;
- collection wall time; and
- final exact-evidence coverage.

The initial exact hit rate is:

```text
compatible exact hits before measurement / exact lookups before measurement
```

It is distinct from final exact coverage after a cold pass. A 100% lookup rate
does not qualify as warm reuse if the source is interpolated, HYBRID,
protocol-incompatible, or provenance-incompatible.

The dashboard plots the hit-rate and measurement-work trajectories over the
ordered campaign stream and repeated iterations. It must make the expected
cold-to-warm transition visually explicit without implying that later shapes
were independently cold.

## Op-wise delta attribution

The analyzer replays the locked FPM through the same canonical walker with an
observation recorder for each evidence lane. The recorder captures operation
name, occurrence/consumer identity, physical `PerfKey` where applicable,
evidence source, scale/composition contribution, and final contributed
latency. It does not forecast or independently enumerate operations.

For each shape, attribution reports:

- offline latency;
- reopened exact latency;
- absolute and relative delta;
- occurrence count;
- unique physical-key count;
- evidence source and exact identity;
- share of the total absolute gap; and
- whether the operation changed evidence, composition, or both.

The sum of recorded contributions must reconcile with the end-to-end result.
Failure to reconcile is `INVALID_EVIDENCE`, not an unattributed "other"
bucket.

Outlier selection is diagnostic. By default, the skill isolates the top `N`
shapes by absolute offline-vs-online delta plus any shape with an evidence
anomaly. A manifest may add model-specific absolute or relative triggers, but
those triggers choose additional investigation work; they do not determine
whether the evidence itself is valid.

## Dashboard contract

`dashboard.html` is self-contained and includes:

1. campaign status and a limitation-first executive summary;
2. exact AIC/Dynamo commits, container digest, framework/runtime identities,
   cluster, hardware inventory, leases, and FPM digest;
3. shape-level offline versus reopened-online modeled latency;
4. absolute and relative delta distributions;
5. cold collection wall time separated from modeled latency;
6. cache hit-rate, collection-command, and exact-coverage trajectories;
7. expandable per-shape operation attribution;
8. isolated-outlier results beside campaign-wide results;
9. every evidence-gate result and failure explanation;
10. links or relative paths to raw and normalized artifacts; and
11. explicit unsupported routes, missing experiments, and other limitations.

The dashboard may show valid campaigns with large red numerical gaps. Red
prediction error is a finding; red evidence integrity is an invalid campaign.
Those states must use different labels and visual treatment.

## Status and failure semantics

The campaign has five terminal states:

| Status | Meaning |
|---|---|
| `BLOCKED` | No compatible allocation or immutable runtime was available after the declared pool was exhausted |
| `EXECUTION_FAILED` | Workload or measurement execution failed before complete evidence was produced |
| `INVALID_EVIDENCE` | Execution completed but one or more structural evidence gates failed |
| `VALID_WITH_FINDINGS` | Evidence is authoritative and one or more diagnostic gap/outlier triggers fired |
| `VALID` | Evidence is authoritative and no configured diagnostic trigger fired |

Only bounded transient infrastructure failures are retried. Capacity,
authentication, scheduler, and artifact-availability failures may advance to
the next declared cluster. Shape drift, unsupported operations, source drift,
runtime incoherence, invalid records, and persistence or lease violations do
not become retries.

Partial evidence is retained for diagnosis, but blocked, failed, and invalid
campaigns must not publish numerical conclusions as authoritative. Historical
GPU artifacts may be compared as explicitly labeled context, never substituted
for the current campaign.

## Source-read-only contract

During a campaign, the skill may:

- inspect the declared repositories and runtime artifact;
- create a scratch campaign workspace;
- allocate declared compute;
- submit and monitor jobs;
- execute existing AIC and Dynamo entry points;
- create overlays, logs, receipts, normalized evidence, and dashboards in
  scratch; and
- terminate its own failed or completed jobs.

It may not:

- edit source, tests, generated code, dependency locks, or configuration in the
  declared repositories;
- commit, push, rebase, or otherwise mutate Git state;
- patch the runtime artifact;
- promote evidence into curated databases; or
- create external issues or comments without separate authorization.

The final report proposes ranked follow-up work items with exact files,
operations, shapes, and evidence paths. Fixing them is a separate task and
must use the appropriate development skill.

## Testing strategy

The skill is validated at four layers.

### 1. Script unit tests

Fixture-driven tests cover:

- manifest parsing, locking, and digest stability;
- dirty/importable-source rejection;
- stage invalidation and safe resume;
- cluster fallback classification;
- shape and request identity drift;
- missing, malformed, or incompatible runtime provenance;
- cold exact closure and partial unresolved outcomes;
- false warm runs that issue commands or use non-exact evidence;
- process-reopen overlay identity mismatch;
- lease and worker-device mismatch;
- cache trajectory calculations;
- op-delta reconciliation; and
- terminal status selection.

### 2. Existing-harness integration tests

Use small fake campaigns to prove the skill delegates paired execution to the
existing Dynamo A/B harness and consumes its reports without reimplementing
prediction. These tests verify command construction and evidence handling, not
GPU correctness.

### 3. Skill evals

`evals/evals.json` contains prompts and expected behaviors for at least:

- a normal DSV4 end-to-end request;
- multiple candidate clusters with the first unavailable;
- no compatible compute anywhere in the declared pool;
- mismatched offline and online FPM digests;
- a runtime graft or incoherent SGLang artifact;
- a valid campaign with a very large numerical gap;
- a warm pass that performs one new GPU measurement;
- a reopened pass with 100% lookup rate but incompatible evidence provenance;
- an interrupted campaign with reusable and invalidated stages; and
- a request to fix the discovered source bug, which the skill must decline and
  hand off as a separate task.

### 4. Real-GPU acceptance campaign

Before the skill is considered complete, run the committed DSV4 reference
campaign on compatible real hardware and prove:

- a coherent unchanged SGLang artifact;
- identical frozen FPMs through both canonical lanes;
- real GEMM, MoE, mHC, attention, NCCL, and CustomAllReduce evidence where the
  frozen stream requires them;
- disjoint single-GPU and isolated multi-GPU leases;
- campaign-wide cold population and hit-rate convergence;
- same-process warm and process-reopen zero-work reuse;
- isolated fresh-overlay reruns for the selected outliers;
- reconciled operation attribution; and
- a browser-validated self-contained dashboard.

Unit and fake-executor tests may validate plumbing while compute is blocked,
but they do not change the campaign status from `BLOCKED` or constitute this
acceptance gate.

## Definition of done

The skill implementation is complete when:

1. the repository-local `SKILL.md`, references, deterministic helper scripts,
   and evals are committed;
2. the campaign schema has a committed DSV4 reference manifest without
   hard-coded production credentials or allocation identifiers;
3. the skill calls the existing Dynamo A/B and canonical AIC paths rather than
   introducing new timing semantics;
4. all evidence gates and terminal states have fixture-based tests;
5. interrupted stages resume only under identical input digests;
6. cluster fallback is bounded to the explicit allowlist;
7. the real DSV4 GPU acceptance campaign reaches `VALID` or
   `VALID_WITH_FINDINGS` with complete cold/warm/reopen evidence;
8. cache convergence and per-operation deltas reconcile with the end-to-end
   comparison;
9. the generated dashboard is self-contained and browser-validated; and
10. a reviewer can trace every conclusion from the dashboard to normalized
    output, raw physical evidence, the frozen FPM, and the locked runtime.

## Implementation sequencing

The implementation plan should preserve this order:

1. schema and fixture contracts;
2. evidence validator and RED tests;
3. preflight, FPM freezer, and resumable campaign controller;
4. thin controller adapter to the existing Dynamo A/B harness;
5. cache-convergence and operation-attribution analysis;
6. dashboard renderer;
7. skill instructions and evals;
8. DSV4 reference campaign; and
9. real-GPU acceptance plus browser validation.

No implementation phase may claim success based only on generated fixtures or
GPU-less tests.
