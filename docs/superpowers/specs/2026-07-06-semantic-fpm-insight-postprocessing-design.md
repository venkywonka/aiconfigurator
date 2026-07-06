# Semantic FPM/Nsight Insight Post-processing — Design

- **Date:** 2026-07-06
- **Status:** Lean-v1 council-reviewed — approved for implementation
- **Normative owner:** Aiconfigurator
- **Aiconfigurator base:** `origin/agent/qwen3-32b-full-pipeline-goal-aic` at
  `1a8f4809fbb1953d51578fc1e4fcb30cb7f6c110`
- **Auto-collector base:** `fork/agent/qwen3-32b-full-pipeline-goal` at
  `4f27f96f2b9e63b6f96b79b9a21faf585f36ec0c`
- **Acceptance artifact:** GitLab job `354281415`, locally mounted at
  `/home/gvenkatarama/scratch/gitlab_ci/fpm_job_354281415/result.tar.gz`

## Decision summary

Build a two-stage post-processing path:

1. Aiconfigurator performs trace-adjacent reduction, same-run FPM-to-Nsight
   alignment, semantic normalization, independent-distribution aggregation, and
   AIC attribution. It emits exactly three small, versioned contract files:
   `samples.csv`, `bins.csv`, and `manifest.json`.
2. A CPU-only auto-collector renderer consumes only those contracts and adds
   `insights.json`, `report.md`, and four deterministic multi-panel chart families
   in both SVG and PNG form.

The report first establishes whether the analyzed population is trustworthy,
then identifies the worst per-bin relative errors across every clean bin with an
AIC total prediction, and finally decomposes the shared subset into compute,
communication, AIC-other, GPU-concurrency, profiled-overhead, and profile-wall-
delta terms.

No calibration, workload recollection, row imputation, or cross-run ordinal
pairing is part of this design.

## Problem statement

The clean FPM run and the Nsight-profiled FPM run execute the same macroscopic
workload but do not produce identical scheduler steps. Joining them by scheduler
ordinal or chronological position therefore creates false matches and false
attribution at the per-step level.

The existing pipeline also has three presentation problems:

- Overall coverage is decode-dominated and hides near-zero mixed or
  high-concurrency context overlap.
- Existing decomposition and report code collapses shapes coarsely, treats
  profiled and clean observations as though they were paired, and primarily
  reports decode means.
- Large Nsight SQLite files are too expensive and too coupled to reprocess in the
  CPU analysis job solely to draw charts.

The new post-processing contract must preserve every real-run observation while
making the narrower decomposition population explicit and auditable.

## Goals

- On successful reduction, retain every clean and profiled real-run FPM
  observation in the sample contract; mark structurally ineligible rows
  explicitly rather than silently deleting them.
- Align every profiled FPM observation to its own Nsight step mechanically and
  fail closed if that same-run mapping is incomplete or ambiguous.
- Compare clean and profiled lanes as independent distributions within a shared
  semantic bin, never as paired rows.
- Use every structurally valid shared bin regardless of sample count or runtime
  dispersion.
- Retain every clean bin with an AIC total prediction in overall gap/error
  analysis even when it has no profiled/Nsight counterpart.
- Invoke one versioned AIC predictor adapter for every measured clean semantic
  bin before computing clean/profiled overlap.
- Separate all-clean, overall-gap-eligible, FPM/Nsight-shared, and
  decomposition-eligible populations with explicit orthogonal status flags.
- Produce evidence-backed static charts and generated findings with stable,
  machine-readable source tables.
- Keep trace reduction beside the trace and keep rendering CPU-only,
  deterministic, and free of SQLite or GPU dependencies.
- Cover context, decode, and mixed phases honestly, including explicit zero
  decomposition coverage.
- Compare AIC and Nsight components on one documented trace-rank-key time basis;
  never combine cross-rank-key sums with a different key's busy-time union or
  label an Nsight process key as a logical TP ordinal without a validated map.

## Non-goals

- Calibration or supplemental controlled workload runs.
- Forcing missing clean/profiled overlap through wider approximate bins.
- Dropping singleton or high-dispersion bins from the primary analysis.
- Treating profiled FPM wall time as authoritative.
- Fitting correction factors into AIC from this single artifact.
- Building an interactive dashboard or requiring browser-side dependencies.
- Replacing existing general-purpose FPM charts that remain useful outside this
  semantic-attribution workflow.

## Branch and repository ownership

Implementation must remain a descendant of the two immutable bases listed in
the header. It must not silently rebase onto `upstream/main`, another layerwise
branch, or a different pipeline head.

Aiconfigurator owns the normative schema and scientific semantics:

- semantic-key normalization;
- same-run profiled-FPM-to-Nsight alignment;
- clean/profiled population construction;
- AIC prediction and additive decomposition;
- normalized CSV/JSON emission;
- the job-354281415 prototype adapter.

Auto-collector owns orchestration and presentation:

- artifact transport;
- schema/version compatibility checks;
- static chart and Markdown rendering;
- pipeline-level artifact completeness policy.

There is one normative specification and one set of golden vectors. The two
repositories must not carry independently worded versions of the semantic
contract.

## Architecture

### Stage 1: trace-adjacent reducer

The reducer runs on the collection node as a mandatory tail action of the
existing `attribute` stage, after every requested bucket has produced and passed
the Nsight/decomposition validity gate and before node-local result packaging.
It is not a new top-level stage name; the current drivers admit only
`layerwise`, `fpm`, `attribute`, `align`, and `profile`, so adding a silently
filtered sixth stage would be incorrect. Its inputs are:

- clean `fpm_metrics_phase.csv` observations;
- profiled `fpm_metrics_phase.csv` observations;
- same-run Nsight step composition;
- AIC predictor inputs/configuration and provenance;
- numeric offered concurrency and configuration fingerprint.

It performs:

1. strict same-run profiled-FPM-to-Nsight alignment;
2. phase classification and semantic-key normalization;
3. exact concurrency/configuration partitioning;
4. construction of the measured clean-bin set;
5. exactly one AIC predictor-adapter call per measured clean bin;
6. clean/profiled semantic support calculation after those calls;
7. per-sample and per-bin descriptive statistics;
8. additive gap decomposition;
9. normalized artifact and manifest emission.

The reducer is the only stage that may read Nsight-derived step composition or
construct/call the AIC predictor. A prototype adapter may load existing AIC
predictions only after normalizing them to the same internal prediction record
and validating their provenance.

On the collection node, the reducer atomically writes its three-file source
contract under `${OUT_ROOT}/semantic_insights/`; `gpu_node.sh` packages that
entire `OUT_ROOT`. After the CPU analysis job untars it into `out_root` and sets
`FPM_COMPARE_RUN_DIR` to that directory, the same files are visible at
`${FPM_COMPARE_RUN_DIR}/semantic_insights/`. The CPU renderer reads that consumer
path and publishes the complete review bundle under
`${FPM_RESULTS_DIR}/insight_report/`. Only `${FPM_RESULTS_DIR}/**` is uploaded by
the analysis job, so a final bundle left solely in the untarred tree is a hard
publication failure.

#### Same-run profiled alignment

Alignment version `profiled-monotonic-v1` treats the measured profiled-FPM
sequence as the required sequence and the Nsight marker sequence as a superset:

1. Select the declared measured FPM scheduler stream and sort it by exact
   `counter_id`. Independently sort every Nsight `rank_key` marker stream by
   marker start, then require identical `(marker_step, measure_run,
   decode_batch, mean_decode_kv)` identities across all keys. Canonicalize that
   time-ordered marker stream by retaining only strictly increasing
   `marker_step` values within each `measure_run`; a marker whose ordinal is less
   than or equal to the previous retained ordinal in the same run is a separately
   recorded nonmonotonic diagnostic marker, not an alignment candidate. Declared
   measure-run order permits an ordinal reset between runs.
2. A marker is compatible with an FPM row only when marker decode batch equals
   exact `decode_requests` and marker mean KV equals the versioned marker
   encoder's value from exact `decode_kv_tokens / decode_requests` (Python
   ties-to-even for job `354281415`); the absent decode axis is `(0, 0)`. This
   same-run anchor convention never changes the half-up cross-run semantic key.
3. Compute order-preserving mappings from every measured FPM row to a distinct
   increasing canonical marker index. Marker-only rows may be skipped; measured
   FPM rows may not. Rank complete mappings first by the minimum inclusive
   first-to-last marker span, equivalently the number of internal marker-only
   rows. Marker labels cannot distinguish pure context from idle `(0, 0)` steps,
   so remaining equal-span mappings are ranked by the maximum sum, over selected
   pure-context markers, of each marker's median integer-nanosecond NVTX span
   across expected rank keys. This tie-break is alignment-only and never
   substitutes profiled marker duration for FPM wall time. Count final optimum
   solutions only up to two: zero complete mappings is
   `same_run_alignment_missing`, and more than one final optimum mapping is
   `same_run_alignment_ambiguous`.
4. Hash the unique minimum-span row-to-marker mapping and record nonmonotonic
   diagnostic markers, unmatched prefix/suffix markers, internal skipped markers,
   and maximal contiguous mapped segments. An exact mapping has one segment and
   no internal skips. After nonmonotonic-marker canonicalization, the c1
   acceptance trace has six segments separated by five zero-decode gap groups.
   Each group has three canonical `(0, 0)` candidates: one mapped pure-context
   marker and two skipped marker-only candidates, for ten skipped markers total.
   The other cohorts each have one exact segment.

This algorithm attaches each authoritative profiled FPM shape to the same marker
on every rank key before kernel aggregation. The fixture freezes mapping hash,
segment boundaries, diagnostic-marker identities, and skipped marker identities,
not only the mapped-row count.

#### Predictor adapter contract

Aiconfigurator defines one reusable `SemanticBinPredictor` protocol at the
reducer/predictor seam in
`collector/layerwise/diagnostics/semantic_fpm_insights.py`:

```text
predict(query: SemanticBinQuery) -> AicPredictionRecord
```

`SemanticBinQuery` is an immutable record containing the configuration
fingerprint, offered concurrency, phase, canonical serialized semantic key, exact
request counts, all three normalized per-request token values, and the canonical
integer AIC query shape reconstructed from them. `AicPredictionRecord` contains:

- stable status and reason code;
- requested and actually evaluated AIC shapes, selected lookup-surface identity,
  and per-axis lookup mode, bounds, weights, and deltas;
- total milliseconds plus `total_basis` (`direct` or `operation_sum`), compute,
  communication, other, and signed component-sum milliseconds, each nullable;
- prediction source/match type, predictor/API version, component-classifier
  version, classified/unclassified operation counts plus inventory hash, and
  configuration provenance.

The reducer calls this protocol exactly once for every measured clean bin before
forming any clean/profiled intersection. It records one returned prediction
record on that bin. A pre-existing coarse attribution/decomposition CSV does not
satisfy this contract because it omits clean-only bins and the canonical
five-field identity. Tests use a spy adapter to prove call-count equality with the
measured clean-bin count and explicit invocation for clean-only bins.

Only a normal `AicPredictionRecord` whose stable reason is an expected lookup
miss may represent `aic_total_unavailable`. Expected misses are the explicit
missing-surface, out-of-cap, or unsupported-phase outcomes of the pinned lookup
policy. Any unexpected predictor exception, non-finite result, corrupt database,
or internal adapter error is the `predictor_error` process failure and aborts the
contract. The new adapter must not reuse the legacy broad exception-to-off-grid
mapping unchanged.

The reference `AiconfiguratorSemanticBinPredictor` in that module adapts the
existing vLLM backend APIs for context, decode, and mixed totals. Current helpers
that hard-code `ctx_requests=1` or intersect with the profiled lane before calling
AIC cannot be reused unchanged. Mixed bins may return a total-only record until a
lossless mixed component inventory exists.

### Stage 2: static insight renderer

The renderer runs in the CPU analysis job. It reads only the versioned normalized
contract. It calculates ranking, generated findings, and chart selections; it
applies the reducer-emitted support annotations and then emits the static bundle.

It must not:

- open `.sqlite`, `.nsys-rep`, or `.qdstrm` files;
- import or construct the AIC model;
- reinterpret the semantic key;
- repair missing decomposition fields;
- recompute medians, descriptive bands, or support classes;
- filter observations by statistical support.

### Prototype adapter

The first adapter applies the Stage 1 contract to cached job `354281415` inputs.
It exists to prove the schema, calculations, and chart bundle before pipeline
integration. It emits the same v1 files as the production reducer and does not
create a second proxy-only schema.

## Semantic identity

Offered concurrency is an exact outer cohort: `c1`, `c16`, `c64`, or `c128`.
Observations never join across cohorts.

Phase is derived from the exact request counts:

- context: `ctx_requests > 0` and `decode_requests == 0`;
- decode: `ctx_requests == 0` and `decode_requests > 0`;
- mixed: both counts are positive;
- idle: both counts are zero and therefore structurally ineligible.

Request counts and token totals must be non-negative integers. If
`ctx_requests == 0`, both context totals must be zero; if
`decode_requests == 0`, the decode-KV total must be zero. Any source phase label
must equal the count-derived phase. A zero-count/nonzero-total shape or phase
disagreement is a hard structural error, not an alternate phase convention.
Conversely, an active context phase requires `ctx_new_tokens > 0`, and an active
decode phase requires `decode_kv_tokens > 0`; zero context KV remains valid for
non-prefix context. The reducer performs these checks before predictor calls and
never relies on a backend `ValueError` to classify shape validity.

Within `(configuration_fingerprint, concurrency, phase)`, the semantic key is:

```text
(
  ctx_requests,
  decode_requests,
  round_half_up(ctx_new_tokens / ctx_requests),
  round_half_up(ctx_kv_tokens / ctx_requests),
  round_half_up(decode_kv_tokens / decode_requests),
)
```

Request counts are exact and never binned. Per-request token means use a
one-token, nearest-integer, half-up rule. An axis whose request count is zero is
absent (`null`), not numeric zero. Half-up normalization is distinct from the
legacy marker anchor's Python ties-to-even rounding.
This half-up rule governs the cross-run semantic key only. Same-run marker
compatibility uses the captured marker anchor's versioned ties-to-even integer as
specified by `profiled-monotonic-v1`; alignment identity and binning identity are
deliberately separate.

For non-negative integer `total` and positive integer `count`, the normative
calculation is integer-only:

```text
round_half_up(total / count) = (2 * total + count) // (2 * count)
```

The reducer never derives a key from a serialized floating-point mean or Python
`round()`.

`ctx_new_tokens` is mandatory. Omitting it would make unrelated pure-context
prompts appear semantically identical.

In the existing FPM CSV, source `ctx_tokens` maps to canonical
`ctx_new_tokens`, and source `latency_ms` maps to canonical `wall_ms`. The
adapter must use the integer total columns, not the serialized
`mean_decode_kv_tokens` float.

The canonical serialized semantic key is the whitespace-free JSON array:

```text
[ctx_requests,decode_requests,ctx_new_per_request,ctx_kv_per_request,decode_kv_per_request]
```

Values are base-10 integers or JSON `null`. The stable bin identifier is the
SHA-256 of a canonical JSON object containing schema version, configuration
fingerprint, numeric offered concurrency, phase, and that array.

The predictor query is deterministic even when multiple raw totals round into
one semantic bin. It reconstructs:

```text
query_ctx_new_total = ctx_requests * ctx_new_per_request
query_ctx_kv_total  = ctx_requests * ctx_kv_per_request
query_decode_kv     = decode_kv_per_request
```

Absent axes reconstruct to zero. Context, decode, and mixed adapters receive
these reconstructed integers plus the exact request counts. If AIC snaps or
falls back to another collected grid point, the requested and evaluated shapes,
per-axis deltas, match policy, and source are recorded; that lookup never changes
the semantic key or clean/profiled membership.

The job-354281415 adapter pins lookup policy `conservative-v1`:

- the full configuration, phase, and request-count/batch surface is selected
  before any value candidates; counts are exact, candidates are never unioned
  across batch or topology surfaces, and a multi-request context query cannot
  fall back to a generic or batch-one context surface;
- zero-prefix context requires an exact context-KV value of zero;
- decode-KV and nonzero context-KV-per-request select the nearest collected value with a
  stable lower-value tie-break and
  `abs(delta) <= max(1024, floor(requested_kv / 2))`;
- `query_ctx_new_total` and every remaining scheduler-shape axis must match
  exactly. The semantic scheduler-surface lookup never interpolates or
  extrapolates between shapes;
- after the evaluated scheduler shape is fixed, a lower-level AIC operation model
  may apply only its versioned deterministic interpolation within that selected
  operation surface. Its lower/upper bounds, weight, mode, topology, and content
  hash must be introspectable and recorded; otherwise the prediction is
  unavailable. This internal operation interpolation never changes the requested
  or evaluated scheduler shape;
- a missing or non-introspectable grid, or a value beyond that cap, returns an
  unavailable prediction rather than unbounded extrapolation;
- environment overrides cannot alter this acceptance policy;
- any mixed-shape fallback is separately versioned, cannot cross phase or
  configuration, and records the exact surface and evaluated shape; a total may
  remain available even when no lossless component split exists.

The first reviewed prototype run creates a candidate golden oracle containing
overall-gap eligibility counts by concurrency and phase. Pipeline integration is
blocked until that file is independently reviewed and frozen as a cross-repository
fixture; later drift is a hard failure, not a newly accepted derived value.

### Configuration parity identity

`configuration_fingerprint` is a SHA-256 over a versioned, sorted-key,
whitespace-free JSON parity record. The record includes model/revision, system
and GPU count, backend/version, TP/PP/DP/EP/attention-DP, numerical and KV-cache
dtypes/quantization, scheduler token/sequence limits, prefix-cache and chunked-
prefill settings, and other effective runtime flags consumed by the predictor.
The complete parity record is stored in `manifest.json`; paths and timestamps are
not hash inputs. Clean, profiled, and predictor parity records must agree on all
shared fields. Predictor database/layerwise content hashes and both repository
commits are recorded separately as prediction provenance.

## Population semantics

The contract records four analytic populations with orthogonal eligibility flags:

1. **All clean:** every structurally valid clean real-run observation. This is the
   authoritative workload and wall-time population.
2. **Overall-gap eligible:** clean bins for which AIC returns a total prediction.
   Profiled/Nsight overlap is not required. These bins support AIC-total versus
   clean-wall gap and relative-error analysis.
3. **Nsight-shared:** all clean and profiled observations whose semantic key
   exists in both lanes after complete same-run profiled-FPM-to-Nsight mapping.
   Sample multiplicities may differ and are retained independently.
4. **Decomposition eligible:** Nsight-shared bins for which every mapped profiled
   row has complete Nsight components and AIC returns all required prediction
   components.

Cross-run semantic overlap is set membership by bin, not row pairing. No
`min(n_clean, n_profiled)` truncation is applied to analysis samples. Row-mass
coverage and unique-bin coverage are reported separately. A clean-only bin may
participate in overall gap analysis but never receives Nsight-derived terms.

Bins missing from any downstream population remain in the output with explicit
status flags and reason codes. Their unavailable fields are blank, never zero.

### Measured rows and coverage denominators

The successful reducer contract retains every source row in `samples.csv`, but
analytic populations use only workload segments declared as measured in
`manifest.json`; there is no implicit segment-name default. For job `354281415`,
the measured segment is exactly `real`. Warmup or diagnostic rows remain in `samples.csv` with
`non_measured_segment` status and do not enter a numerator or denominator. Idle
rows are similarly retained but structurally ineligible.

Coverage is always lane-specific:

- the clean denominator is every structurally valid measured clean row;
- the profiled denominator is every structurally valid measured profiled FPM row
  before same-run Nsight alignment;
- mapped profiled mass is reported separately as successfully mapped rows divided
  by that pre-alignment denominator; a successful reducer requires 100% mapping;
- shared row mass in a lane counts that lane's rows whose semantic bin exists in
  both lanes;
- overall-gap row mass counts clean rows whose bin has an AIC total prediction;
- decomposition row mass in each lane counts that lane's rows whose bin is
  decomposition eligible.

Unique-bin coverage uses the analogous lane-specific bin sets. No coverage
metric uses `min(n_clean, n_profiled)`, pairs cross-run rows, or renormalizes away
unsupported clean mass.

## Normalized contract

The contract schema identifier is `fpm-semantic-insights/v1`. CSV blanks and JSON
`null` represent unavailable values. Numeric zero is always a measured or derived
value, never a missing-value sentinel.

### `samples.csv`

On successful reduction, one row per source clean or profiled observation,
including rows marked structurally ineligible. Required columns are:

- schema and provenance: `schema_version`, `configuration_fingerprint`,
  `concurrency`, `phase`, `lane`, `sample_id`, `workload_segment`;
- semantic inputs: exact request counts and token totals, all five normalized key
  fields, and a stable serialized `semantic_key`;
- timing: `wall_ms`; clean wall is authoritative, while profiled wall is retained
  only to separate within-profile overhead from the cross-lane wall delta and is
  never substituted for clean wall;
- profiled composition: `gpu_compute_ms`, `gpu_comm_ms`, `gpu_busy_ms` for mapped
  profiled rows, plus selected Nsight `rank_key`, rank-identity kind/mapping
  provenance, captured-rank-key count, and imbalance diagnostics; blank for clean
  rows. If an unknown positive-duration kernel makes compute/communication
  classification incomplete, busy time remains populated while compute and
  communication are blank;
- status: same-run mapping status, semantic support status, overall-gap and
  decomposition eligibility, AIC availability, and stable reason codes when
  ineligible.

The profiled compute/communication/busy tuple remains on the same row so neither
v1 nor later tuple-level analyses can accidentally destroy within-step
correlation.

### `bins.csv`

One row per key in the union of structurally valid measured clean and profiled
`(configuration, concurrency, phase, semantic_key)` values. Required fields
include:

- `n_clean`, `n_profiled`, clean and profiled row-mass weights;
- exact-raw-shape cardinality per lane and per-axis raw min/max/query deltas, so
  collisions introduced by one-token mean normalization remain auditable;
- structural status plus explicit all-clean, overall-gap, Nsight-shared, and
  decomposition-eligibility flags;
- clean support count/class and clean descriptive-band kind;
- profiled support count/class, decomposition support count/class (the weaker
  lane), and profiled descriptive-band kind;
- median, minimum, maximum, first quartile, and third quartile fields for clean
  wall, profiled wall, GPU compute, communication, and busy time; unavailable
  quartiles are blank rather than overloaded with a different statistic;
- the complete `SemanticBinQuery` and `AicPredictionRecord` fields, including AIC
  compute, communication, other, and total prediction;
- signed/absolute overall gap in milliseconds and signed/absolute relative error
  whenever clean wall and AIC total exist, including clean-only bins;
- GPU-concurrency credit, profiled overhead, profile-wall delta, and all six
  additive decomposition terms only when the bin is decomposition eligible;
- closure error and stable identifiers used by chart selections and findings.

The reducer emits closure-preserving point estimates and descriptive support
bounds. The renderer does not mutate this contract. It records selected stable
bin identifiers, ranks, dominant-term labels, and finding evidence in
`insights.json`.

Unsupported observations and bins stay in `samples.csv` and `bins.csv`; there is
no duplicate unmatched table. Successful-bundle eligibility reason codes are:

- `idle_step`;
- `non_measured_segment`;
- `clean_only_bin`;
- `profiled_only_bin`;
- `aic_total_unavailable`;
- `aic_components_unavailable`;
- `nsight_components_unavailable`.

The disjoint process-error vocabulary is:

- `schema_missing`;
- `schema_incompatible`;
- `same_run_alignment_missing`;
- `same_run_alignment_ambiguous`;
- `configuration_mismatch`;
- `repository_commit_mismatch`;
- `concurrency_mismatch`;
- `phase_mismatch`;
- `invalid_shape`;
- `invalid_wall_time`;
- `invalid_measurement`;
- `duplicate_identity`;
- `incomplete_rank_capture`;
- `rank_identity_unavailable`;
- `rank_tuple_identity_mismatch`;
- `predictor_call_identity_mismatch`;
- `predictor_error`;
- `aic_component_identity_mismatch`;
- `decomposition_closure_mismatch`;
- `manifest_integrity_mismatch`.

Process errors are emitted in the reducer failure diagnostic/job log and prevent
creation of a successful three-file contract. They never masquerade as row-level
missing coverage in a successful `samples.csv` or `bins.csv`.

### `manifest.json`

The manifest records:

- schema identifier and both repository commits;
- source job/model/system/runtime configuration;
- full parity record, configuration fingerprint, and exact concurrency cohorts;
- the explicit measured workload-segment set;
- pre-alignment input, mapped, predictor-call, overall-gap-eligible, shared,
  decomposition-eligible, and unsupported counts, including counts by reason
  code;
- row-mass and unique-bin coverage by concurrency and phase;
- exact-shape collision counts and raw-to-canonical query-delta summaries;
- same-run alignment method and result per cohort;
- semantic-key/query reconstruction version, AIC lookup/snap policy, predictor
  provenance, AIC component-classifier version, Nsight kernel-classifier version,
  AIC total basis, unknown-operation/kernel counts and signed/duration mass,
  expected/captured rank keys, rank-identity/mapping provenance, and critical-key
  selection policy;
- `samples.csv` and `bins.csv` filenames and content hashes (the manifest does
  not attempt to hash itself);
- validation outcome and stable failure reasons.

## Statistical contract

For one semantic bin, let clean wall observations be `Wc`, and, when a profiled
lane exists, let each profiled row be the intact tuple `(Wp, C, M, B)` for
profiled wall, GPU compute, GPU communication, and GPU busy time. AIC may supply
deterministic `(A_compute, A_comm, A_other, A_total)` for the bin.

### Nsight and AIC component basis

Nsight reduction is per `rank_key`. A key is either a logical rank obtained from
a validated launcher/trace map or the trace's numeric `globalPid` process key;
`rank_identity_kind` records which. A bare `globalPid` is never presented as a TP
ordinal. The runtime parity record supplies the expected rank count, and the
capture supplies the stable key set and optional logical-rank map. An absent
identity/count contract is `rank_identity_unavailable`; a missing expected key or
count mismatch is `incomplete_rank_capture`.

For each mapped profiled step and each expected key, `C` and `M` are summed
durations of kernels classified respectively as compute and communication, and
`B` is the interval union of every positive-duration GPU kernel on that same key.
The step representative is the complete tuple from the key with maximum `B`
(stable tie-break: lowest numeric key). The reducer never combines cross-key sums
with another key's busy union. Rank-key count, identity/map provenance, selected
key, busy spread, and classifier version are retained as diagnostics.

The Nsight kernel classifier and AIC operation-component classifier are
versioned registries with golden vectors; substring/suffix heuristics alone are
not normative classifiers. Every positive-duration Nsight kernel on every
expected rank key contributes to that key's `B`. If any such kernel on any key is
unknown, its name hash/count/duration remain auditable and that key's `C` and `M`
are incomplete. The emitted representative step row then retains `B` but leaves
`C` and `M` null regardless of which key was busiest, and the whole semantic bin
is `nsight_components_unavailable` rather than silently defaulting the kernel to
compute. The AIC registry must classify every finite nonzero signed operation
exactly once, including negative overlap credits. If AIC supplies a total but no
lossless compute/communication/other split, the bin remains overall-gap eligible
while its components are null and decomposition ineligible.

`A_total` comes from the adapter's declared `total_basis`. `direct` means a
distinct total API was evaluated independently of the operation inventory;
`operation_sum` means the current backend's total is the signed sum of that same
latency dictionary. When components are available, the registry sums
`A_compute_raw`, `A_comm_raw`, and `A_other_raw` and reconciles their signed sum
to `A_total` using:

```text
abs(component_sum - A_total) <= max(1e-6 ms, 1e-6 * abs(A_total))
```

Under `direct`, this is an independent total-versus-inventory gate;
under `operation_sum`, it is only an algebra/serialization check and must not be
described as independent evidence. Only after that validation does the reducer
store canonical `A_other = A_total - A_compute - A_comm`. An unknown operation
returns total-only with `aic_components_unavailable`; a record claiming complete
components but failing signed reconciliation is an
`aic_component_identity_mismatch` process error.

### Point estimate

- `wall = median(Wc)`;
- `gap = A_total - wall` and `relative_error = gap / wall` whenever clean wall
  and AIC total exist, whether or not the profiled lane overlaps;
- `profiled_wall = median(Wp)` for Nsight-shared bins, used only in the two bridge
  terms below and never as the authoritative gap denominator;
- `gpu_busy = median(B)` for Nsight-shared bins;
- `gpu_compute = median(C)` and `gpu_comm = median(M)` only when every mapped
  profiled row has complete Nsight components;
- `gpu_concurrency = gpu_compute + gpu_comm - gpu_busy` when those components are
  complete;
- `profiled_overhead = profiled_wall - gpu_busy`;
- `profile_wall_delta = profiled_wall - wall`.

`profiled_overhead` is within the profiled lane. `profile_wall_delta` isolates
the cross-lane wall shift needed to bridge back to authoritative clean wall; it
may contain both profiler perturbation and ordinary run-to-run distribution
difference and is not labeled as a purely causal profiler cost.

The additive terms are:

```text
term_compute_err = A_compute - gpu_compute
term_comm_err    = A_comm    - gpu_comm
term_aic_other   = A_other
term_gpu_concurrency = gpu_concurrency
term_neg_profiled_overhead = -profiled_overhead
term_profile_wall_delta = profile_wall_delta
```

They must satisfy:

```text
gap = term_compute_err + term_comm_err + term_aic_other
    + term_gpu_concurrency + term_neg_profiled_overhead
    + term_profile_wall_delta
```

This closure check is an algebra/serialization integrity assertion, not evidence
that component classification is scientifically correct. A direct-basis total-
versus-operation-inventory comparison, when available, is an independent gate;
operation-sum identity is not. Classifier golden vectors, complete signed AIC
inventory, complete Nsight kernel classification, and rank-key provenance are the
substantive component-validation gates.

Component medians define the deterministic point vector. Per-sample descriptive
distributions may also be reported but must not replace or silently alter this
closure-preserving point estimate. Decomposition fields remain blank unless all
required lanes, every mapped row's Nsight components, and all AIC components
exist.

### Sampling support

Overall-gap support is based on `n_clean`. Decomposition support is based on the
weaker lane, `min(n_clean, n_profiled)`. Both use the same classes:

- singleton: `n == 1`;
- sparse: `2 <= n <= 4`;
- repeated: `n >= 5`.

Support affects annotation only. It never excludes a structurally valid bin.

- Every nonempty metric stores median, minimum, and maximum in separately named
  columns.
- Singleton bins render the point plus an explicit `n=1` annotation.
- Sparse bins render the observed minimum-to-maximum range.
- Repeated bins additionally store and render the first-to-third-quartile range,
  with minimum/maximum retained as lighter outer whiskers.

These are descriptive support bands, not inferential confidence intervals. The
v1 contract does not bootstrap. Raw profiled compute/communication/busy values
remain row tuples in `samples.csv` so later analyses cannot accidentally destroy
their within-step relationship. Clean-wall bounds use the clean support class;
profiled-wall and composition bounds use the profiled support class. The
weaker-lane class controls only the annotation/style of decomposition results.
For sorted observations `x`, median and quartiles use the linear-interpolation
quantile at rank `r = p * (n - 1)`: `x[floor(r)] + (r - floor(r)) *
(x[ceil(r)] - x[floor(r)])`. The median uses `p=0.5` for every `n >= 1`; for
even `n` it is therefore the arithmetic mean of the two central observations,
never a lower or upper median. First and third quartiles use `p=0.25` and
`p=0.75` and remain blank when `n < 5`. The same median convention applies to
clean wall, profiled wall, GPU compute/communication/busy, and the cross-rank-key
NVTX-span tie-break in `profiled-monotonic-v1`.

### Ranking and aggregation

The primary ranking is worst absolute relative error:

```text
abs((A_total - median_clean_wall) / median_clean_wall)
```

The renderer ranks every overall-gap-eligible bin, including clean-only bins. It
emits a global top 20 and a top 5 within every concurrency-by-phase cohort. Stable
tie-breakers are absolute millisecond gap, then serialized semantic key. Labels
retain signed percentage, signed milliseconds, clean sample count, clean support
class, decomposition availability, phase, and concurrency.

The all-bin ranking remains primary so singleton observations are never silently
dropped. Because job `354281415` is singleton-dominated, the renderer also emits
a clearly secondary `n_clean >= 2` sensitivity ranking and the singleton fraction
for every leaderboard. Singleton entries say `one observation; no stability
estimate`; neither view claims an inferential confidence interval. An empty
repeated/sparse sensitivity view is rendered explicitly.

For decomposition-eligible bins, dominant cause is the additive term with the
largest absolute magnitude; its sign is retained. No dominant cause is assigned
to overall-gap-only bins.

Cohort summaries publish both:

- the bin-equal median; and
- the clean-row-frequency-weighted mean.

Neither is labeled simply "overall" without its weighting semantics. Gap
summaries use the overall-gap-eligible population; attribution summaries state
explicitly that they are conditional on decomposition eligibility.

## Insight and chart catalog

The bundle tells the same three-part diagnostic story with four multi-panel chart
families instead of eight single-purpose figures.

1. `01_trust`: a branched population summary showing all-clean to
   overall-gap-eligible and all-clean to Nsight-shared to decomposition-eligible;
   concurrency-by-phase row/bin coverage matrices; and the profiled-versus-clean
   wall delta distribution. The panel labels this as a profile-lane shift, not a
   purely causal profiler cost.
2. `02_worst_errors`: the global top 20 absolute relative errors plus top-five
   concurrency-by-phase facets over every overall-gap-eligible bin, with an
   adjacent `n_clean >= 2` sensitivity ranking and singleton-dominance banner.
   Bins without decomposition support remain in the primary ranking and are
   labeled accordingly.
3. `03_shape_map`: decode-request count versus mean decode KV, faceted by exact
   offered concurrency, colored by signed relative error and styled by clean
   support class. This family is decode-specific; context and mixed remain in
   trust, error-ranking, findings, and parity whenever an AIC total is available.
   Their missing component decomposition is shown explicitly rather than implied.
4. `04_attribution`: diverging six-term additive bars for the worst
   decomposition-eligible bins plus AIC-total versus clean-wall parity across
   every overall-gap-eligible bin. Decomposition-eligible parity points are
   colored by dominant term and styled by weaker-lane support; other points are
   labeled `attribution unavailable` rather than dropped.

Evidence-backed generated findings appear in `report.md` and `insights.json`.
Support styling is stable: singleton markers are hollow, sparse markers are
outlined, and repeated markers are filled. Static labels carry exact sample
counts. Every bin-level point or bar resolves to a stable row in `bins.csv`.
`insights.json` records bin identifiers for ranked/shape panels and explicit
population, cohort, phase, numerator, and denominator source keys for aggregate
trust panels.

## Generated findings

Generated prose is descriptive and evidence-backed. It may state:

- population coverage and the largest missing-mass phase/cohort;
- the worst absolute relative-error bin and signed millisecond error across the
  overall-gap-eligible population;
- the dominant additive term when that bin is decomposition eligible, or an
  explicit statement that attribution is unavailable otherwise;
- median signed bias by cohort;
- profile-lane wall-delta and within-profile overhead summaries;
- counts of singleton, sparse, and repeated bins;
- whether singleton observations dominate a reported ranking and how its
  `n_clean >= 2` sensitivity ordering differs;
- unsupported and AIC-unavailable mass by reason.

Every finding includes stable keys into `bins.csv`, `samples.csv`, or
`manifest.json`. The generator does not use subjective terms such as "low
confidence", does not infer causality beyond the additive identity, and does not
convert descriptive support bands into filters.

## Static artifact bundle

```text
${FPM_RESULTS_DIR}/insight_report/
├── report.md
├── insights.json
├── samples.csv
├── bins.csv
├── manifest.json
└── charts/
    ├── 01_trust.{svg,png}
    ├── 02_worst_errors.{svg,png}
    ├── 03_shape_map.{svg,png}
    └── 04_attribution.{svg,png}
```

`report.md` is the reviewer entry point. It contains the generated findings,
coverage caveats, PNG previews, and links to full-resolution SVG and contract
evidence. `insights.json` contains the generated findings, ranks, chart
selections, renderer version, input contract hashes, emitted chart hashes, and
stable references back to the three reducer files; it does not duplicate their
full row payloads.

Rendering is deterministic:

- stable sort and filenames;
- fixed palette, dimensions, fonts, and support styling;
- no timestamps embedded in figures;
- no network dependency;
- every chart backed by `samples.csv`, `bins.csv`, and/or `manifest.json` source
  keys selected explicitly in `insights.json`.

This bundle supplements rather than replaces every existing full-analysis
artifact. The collection-side Qwen3 FPM profile charts remain at:

```text
${FPM_COMPARE_RUN_DIR}/charts/Qwen-Qwen3-32B/c{1,16,64,128}/fpm_profile/
  fpm_distribution_{params,composition,scatter}.png
```

The current publication step copies those twelve files without flattening into:

```text
${FPM_RESULTS_DIR}/fpm_distributions/Qwen-Qwen3-32B/c{1,16,64,128}/
  fpm_distribution_{params,composition,scatter}.png
```

The semantic-insight gate extends the existing `analysis_artifact_gate.py`
requirements; it does not replace them. The published tree must still contain
nonempty `gap_summary.csv`, `by_bucket/c{1,16,64,128}/gap_summary.csv`,
`gap_mape_vs_concurrency.png`,
`layerwise_vs_opwise/high_concurrency_layerwise_vs_opwise_summary.csv`,
`layerwise_vs_opwise/high_concurrency_mape_vs_batch.png`, and
`layerwise_vs_opwise/high_concurrency_signed_error_vs_batch.png`. The attribution
gate must still find a nonempty Nsight SQLite and `decomposition.csv` for every
requested bucket in `out_root`. Thus integrating semantic insights cannot make
FPM distribution, gap, Nsight, layerwise, opwise, or any earlier
collection/analysis phase optional.

## Failure handling

### Hard failures

The reducer exits nonzero for:

- missing or incompatible required schemas (`schema_missing`,
  `schema_incompatible`);
- configuration, repository-commit, or concurrency mismatch
  (`configuration_mismatch`, `repository_commit_mismatch`,
  `concurrency_mismatch`);
- ambiguous or incomplete same-run FPM-to-Nsight alignment
  (`same_run_alignment_ambiguous`, `same_run_alignment_missing`);
- non-integer/negative counts or totals, zero-count/nonzero-total shapes, or a
  non-positive `ctx_new_tokens`/`decode_kv_tokens` total for its active request
  count, or source/count-derived phase mismatch (`invalid_shape`,
  `phase_mismatch`);
- non-finite measurements or non-positive authoritative clean wall
  (`invalid_measurement`, `invalid_wall_time`);
- duplicate identities that violate the schema (`duplicate_identity`);
- unavailable rank identity/count contract, missing expected Nsight rank keys, or
  mixed-key composition tuples (`rank_identity_unavailable`,
  `incomplete_rank_capture`, `rank_tuple_identity_mismatch`);
- predictor-call count or clean-bin identity mismatch
  (`predictor_call_identity_mismatch`);
- an unexpected predictor exception, corrupt predictor input, or non-finite
  predictor result (`predictor_error`);
- inconsistent AIC component identity (`aic_component_identity_mismatch`);
- decomposition closure outside tolerance (`decomposition_closure_mismatch`);
- manifest count or artifact-hash inconsistency
  (`manifest_integrity_mismatch`).

Closure tolerance is:

```text
abs(sum(terms) - gap) <= max(1e-6 ms, 1e-6 * abs(gap))
```

### Valid missing coverage

These are not processing failures:

- no clean/profiled overlap for a phase or cohort;
- unavailable AIC prediction;
- singleton or sparse support;
- high dispersion;
- negative profiled overhead or profile-wall delta;
- a negative algebraic GPU-concurrency point caused by separately taken component
  medians, provided every underlying same-rank-key tuple satisfies `C + M >= B`.

They remain visible with explicit status. Negative terms are not clipped. A chart
with no eligible rows is emitted as a labeled `No eligible bins` panel rather than
silently omitted.

The renderer reports coverage; it does not decide whether coverage satisfies a
particular pipeline policy. For the Qwen3 workflow, the separate artifact gate
requires at least one decomposition-eligible decode bin per requested concurrency
and does not require natural context or mixed overlap.

## Testing

### Unit tests

- integer-only half-up normalization, `.5` ties, and absent-axis encoding;
- canonical semantic-key serialization, stable bin ID, and canonical predictor-
  query reconstruction;
- phase derivation, zero-count/nonzero-total rejection, source-phase validation,
  positive active-token validation, and exact concurrency partitioning;
- explicit measured-segment selection and lane-specific coverage denominators;
- pre-alignment profiled denominator; unique full order-preserving mapping;
  pure-context versus idle duration tie-break; ambiguous/missing failure;
  multi-key marker identity; segment/skip/mapping-hash oracle; and exact
  mapped-mass reconciliation;
- sample multiplicity without Cartesian expansion or truncation;
- exact-shape collision cardinality and raw-to-canonical query deltas;
- clean and weaker-lane support classification;
- explicit min/max and quartile columns, linear quartile interpolation, and
  singleton/sparse/repeated rendering policy, including the even-count
  mean-of-two median convention;
- point-estimate and additive-term calculations;
- canonical AIC-other residual, signed negative-operation accounting, declared
  total basis, and direct-versus-operation-sum validation semantics;
- same-key critical-key selection, logical-rank versus `globalPid` identity,
  rank-key completeness, and rejection of merged cross-key-sum/other-key-union
  inputs;
- predictor spy proving one call per measured clean bin before intersection;
- explicit expected lookup misses versus fail-closed unexpected predictor errors;
- exhaustive stable process-error codes for every hard-failure branch;
- exact `query_ctx_new_total`, zero-prefix exactness, bounded nearest-KV snap,
  prohibition of scheduler-surface interpolation, lower-level operation-
  interpolation provenance, and rejection of cross-batch/topology fallback under
  `conservative-v1`;
- versioned Nsight/AIC classifier golden vectors, unknown-kernel busy retention,
  detection on non-selected rank keys, no unknown-to-compute default, and
  component-null behavior when a lossless split is unavailable;
- component-identity and closure tolerances plus negative-term preservation;
- worst-relative-error ranking and deterministic tie-breaking;
- all-bin versus `n_clean >= 2` sensitivity ranking, singleton-fraction warning,
  and empty-sensitivity rendering;
- overall-gap eligibility without profiled overlap and decomposition fail-closed
  behavior for those same bins;
- bin-equal versus clean-frequency-weighted summaries;
- dominant-term selection and generated-finding source keys;
- empty chart panels and stable artifact names;
- collection-side contract versus published-bundle path separation and extension
  of every pre-existing full-analysis artifact gate;
- `attribute`-stage tail invocation, atomic `${OUT_ROOT}/semantic_insights`
  creation, packaging, and post-untar consumer-path discovery.

### Synthetic integration fixture

The fixture contains context, decode, and mixed bins; singleton, sparse, and
repeated support; unequal lane multiplicities; an AIC-predicted clean-only bin;
multiple raw shapes collapsing to one semantic bin; missing AIC predictions;
positive/negative profile-wall deltas; a negative AIC overlap credit; direct and
operation-sum total bases; an unknown Nsight kernel; rank imbalance, `globalPid`
identity, and missing-rank-key failure; and each unsupported reason. Tests
validate normalized CSV/JSON contents and
renderer completeness. Plot tests assert source data, labels, output existence,
and dimensions rather than brittle pixel-perfect equality.

### Cross-repository contract fixture

Aiconfigurator emits a small three-file normalized fixture. Auto-collector
consumes it unchanged, validates the schema/manifest, and emits all expected
artifacts under a fixture `${FPM_RESULTS_DIR}/insight_report/` without opening
SQLite or importing AIC. The fixture also proves that adding this bundle does not
weaken the pre-existing FPM distribution, gap, Nsight, layerwise, or opwise gate.

### Visual baseline

The first accepted job-354281415 bundle receives a browser screenshot review for
layout, legibility, support styling, clipping, and color consistency. This is a
review baseline, not the primary numeric correctness oracle.

## Job 354281415 acceptance oracle

The prototype must reproduce the following artifact-backed counts.

### Same-run profiled FPM to Nsight

| Concurrency | Mapped | Profiled FPM | Required method |
|---|---:|---:|---|
| c1 | 4,612 | 4,612 | six-segment, skip-aware monotonic alignment |
| c16 | 4,099 | 4,099 | exact monotonic alignment |
| c64 | 4,200 | 4,200 | exact monotonic alignment |
| c128 | 4,242 | 4,242 | exact monotonic alignment |

Under `profiled-monotonic-v1`, c1 must additionally reproduce five internal idle
gap groups containing ten skipped canonical markers and six contiguous mapped
segments; c16, c64, and c128 have zero internal skips and one mapped segment each.
The reviewed fixture pins each mapping hash and the separately dropped
nonmonotonic diagnostic-marker list.

### Cross-run semantic support

All 34,311 source FPM rows in this artifact are in measured segment `real`
(17,158 clean and 17,153 profiled). The prototype must also reproduce zero
source/count-derived phase mismatches, zero zero-count/nonzero-total shapes, and
zero invalid authoritative clean walls before applying the overlap oracle.

| Population | Shared rows | Total rows | Coverage |
|---|---:|---:|---:|
| clean overall | 16,679 | 17,158 | 97.2083% |
| profiled overall | 16,646 | 17,153 | 97.0442% |
| clean decode | 16,671 | 17,022 | 97.9380% |
| profiled decode | 16,638 | 17,016 | 97.7786% |
| clean context | 6 | 9 | 66.6667% |
| profiled context | 6 | 9 | 66.6667% |
| clean mixed | 2 | 127 | 1.5748% |
| profiled mixed | 2 | 128 | 1.5625% |

The clean lane contains 17,009 unique semantic bins. Exactly 479 clean rows in
479 unique bins are clean-only and must remain visible in the contract and
coverage. Their structural concurrency/phase breakdown is enumerated below; the
AIC-total-eligible subset is a later derived oracle and is not asserted by this
table:

| Concurrency | Context | Decode | Mixed | Total clean-only bins/rows |
|---|---:|---:|---:|---:|
| c1 | 0 | 0 | 0 | 0 |
| c16 | 1 | 16 | 10 | 27 |
| c64 | 1 | 107 | 39 | 147 |
| c128 | 1 | 228 | 76 | 305 |
| total | 3 | 351 | 125 | 479 |

Clean support is overwhelmingly descriptive: 16,869 bins are singletons, 140
are sparse (`n=2..4`), no bin is repeated (`n>=5`), and the maximum clean count is
four. The report must reproduce those counts, display the singleton-dominance
warning, report exactly 140 `n_clean >= 2` clean bins before AIC-availability
filtering, and populate the sensitivity ranking from the subset of those bins
that is overall-gap eligible under the frozen adapter oracle.

The current conservative layerwise availability proxy supports 11,214 shared bins
and represents 11,315 clean rows and 11,314 profiled rows. This is reported as AIC
availability, not semantic-join failure. That proxy evaluates prediction support
only after the shared-bin intersection, so it is not an oracle for AIC-total
availability across all 17,009 clean bins or the 479 clean-only bins.

The first reviewed adapter run must make and reconcile exactly 17,009 predictor-
adapter calls before intersection, including all 479 clean-only bins, then pin
AIC-total-eligible row/bin counts for the full clean population and clean-only
subset plus the exact global/cohort top-error and additive-term selections. Those
values require the prototype predictor adapter and are derived outputs rather
than predeclared assumptions. Independent review freezes them in the shared
golden fixture `tests/fixtures/semantic_fpm_insights/job_354281415_expected.json`;
the production pipeline cannot accept the adapter before that fixture exists.

Artifact acceptance additionally requires:

- four SVG and four PNG multi-panel charts;
- publication of the full bundle at `${FPM_RESULTS_DIR}/insight_report/`;
- all twelve pre-existing Qwen3 per-concurrency FPM distribution PNGs at their
  published `${FPM_RESULTS_DIR}/fpm_distributions/...` paths;
- `samples.csv`, `bins.csv`, `manifest.json`, `insights.json`, and `report.md`;
- every pre-existing gap-summary, gap plot, layerwise/opwise summary/plot,
  per-bucket Nsight SQLite, and decomposition gate remains green;
- valid relative links from the report;
- exact reconciliation between report, chart selections, contract rows, and
  manifest counts;
- no SQLite access by the renderer.

## Implementation hygiene

The designated Aiconfigurator worktree currently exposes broad pre-existing
file-mode-only changes (`100644` to `100755`) across tracked files. These changes
are not part of this design. Spec and implementation commits must stage only
explicit task paths and must not normalize, revert, or include unrelated modes.

The auto-collector worktree contains an existing untracked `slop/` entry. It is
also outside this design and must remain untouched.

## Acceptance criteria

- Implementation commits descend from the exact paired bases in this document.
- The normalized v1 contract is produced for job `354281415` and passes all count
  oracles.
- The reviewed adapter pins and reconciles AIC-total eligibility for all 17,009
  clean bins and the 479 clean-only bins instead of substituting the shared-only
  layerwise proxy.
- All structurally valid shared bins are analyzed regardless of support band.
- Every clean bin with an AIC total prediction participates in overall gap/error
  analysis regardless of profiled/Nsight overlap.
- Predictor-call count equals the measured clean-bin count before intersection,
  including explicit calls for all 479 clean-only bins in the acceptance fixture.
- Every unavailable field is blank/null with a stable reason, never fabricated as
  zero.
- Every decomposition row closes algebraically within tolerance.
- Global and cohort worst-relative-error rankings are stable and auditable.
- The complete deterministic bundle (five core files and eight chart images) is
  published under `${FPM_RESULTS_DIR}/insight_report/` and visually reviewed.
- Existing FPM distribution, layerwise, and opwise deliverables remain gated and
  are not displaced by the new bundle.
- Auto-collector renders from normalized files without reading raw Nsight data.
- The reducer runs as the existing `attribute` stage's mandatory tail and its
  `${OUT_ROOT}/semantic_insights` contract survives packaging at the corresponding
  `${FPM_COMPARE_RUN_DIR}/semantic_insights` consumer path.
- Context and mixed zero-coverage panels remain explicit and do not cause a
  processing failure.
- The Qwen3 policy gate finds at least one decomposition-eligible decode bin in
  every requested concurrency cohort.
