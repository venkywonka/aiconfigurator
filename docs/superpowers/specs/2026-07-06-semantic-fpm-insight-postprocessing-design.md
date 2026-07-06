# Semantic FPM/Nsight Insight Post-processing — Design

- **Date:** 2026-07-06
- **Status:** Approved lean-v1 design — awaiting revised written-spec review
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
communication, AIC-other, overlap, and overhead terms.

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
- Separate all-clean, overall-gap-eligible, FPM/Nsight-shared, and
  decomposition-eligible populations with explicit orthogonal status flags.
- Produce evidence-backed static charts and generated findings with stable,
  machine-readable source tables.
- Keep trace reduction beside the trace and keep rendering CPU-only,
  deterministic, and free of SQLite or GPU dependencies.
- Cover context, decode, and mixed phases honestly, including explicit zero
  decomposition coverage.

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

The reducer runs on the collection node after the Nsight SQLite has been reduced
to per-step composition. Its inputs are:

- clean `fpm_metrics_phase.csv` observations;
- profiled `fpm_metrics_phase.csv` observations;
- same-run Nsight step composition;
- AIC predictor inputs/configuration and provenance;
- numeric offered concurrency and configuration fingerprint.

It performs:

1. strict same-run profiled-FPM-to-Nsight alignment;
2. phase classification and semantic-key normalization;
3. exact concurrency/configuration partitioning;
4. clean/profiled semantic support calculation;
5. AIC predictor construction/loading, prediction, and availability lookup;
6. per-sample and per-bin descriptive statistics;
7. additive gap decomposition;
8. normalized artifact and manifest emission.

The reducer is the only stage that may read Nsight-derived step composition or
construct/call the AIC predictor. A prototype adapter may load existing AIC
predictions only after normalizing them to the same internal prediction record
and validating their provenance.

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

For non-negative integer `total` and positive integer `count`, the normative
calculation is integer-only:

```text
round_half_up(total / count) = (2 * total + count) // (2 * count)
```

The reducer never derives a key from a serialized floating-point mean or Python
`round()`.

`ctx_new_tokens` is mandatory. Omitting it would make unrelated pure-context
prompts appear semantically identical.

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
4. **Decomposition eligible:** Nsight-shared bins for which AIC returns all
   required prediction components.

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
- the profiled denominator is every structurally valid measured profiled row
  after complete same-run Nsight alignment;
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
  only for perturbation diagnostics;
- profiled composition: `gpu_compute_ms`, `gpu_comm_ms`, `gpu_busy_ms` for mapped
  profiled rows, blank for clean rows;
- status: same-run mapping status, semantic support status, overall-gap and
  decomposition eligibility, AIC availability, and stable reason codes when
  ineligible.

The profiled compute/communication/busy tuple remains on the same row so neither
v1 nor later tuple-level analyses can accidentally destroy within-step
correlation.

### `bins.csv`

One row per key in the union of retained clean and profiled
`(configuration, concurrency, phase, semantic_key)` values. Required fields
include:

- `n_clean`, `n_profiled`, clean and profiled row-mass weights;
- structural status plus explicit all-clean, overall-gap, Nsight-shared, and
  decomposition-eligibility flags;
- clean support count/class and clean descriptive-band kind;
- profiled support count/class, decomposition support count/class (the weaker
  lane), and profiled descriptive-band kind;
- medians and descriptive bounds for clean wall, profiled wall, GPU compute,
  communication, and busy time;
- AIC compute, communication, other, and total prediction;
- signed/absolute overall gap in milliseconds and signed/absolute relative error
  whenever clean wall and AIC total exist, including clean-only bins;
- overlap, overhead, and all five additive decomposition terms only when the bin
  is decomposition eligible;
- closure error and stable identifiers used by chart selections and findings.

The reducer emits closure-preserving point estimates and descriptive support
bounds. The renderer does not mutate this contract. It records selected stable
bin identifiers, ranks, dominant-term labels, and finding evidence in
`insights.json`.

Unsupported observations and bins stay in `samples.csv` and `bins.csv`; there is
no duplicate unmatched table. Stable reason codes include:

- `idle_step`;
- `non_measured_segment`;
- `same_run_alignment_missing`;
- `same_run_alignment_ambiguous`;
- `configuration_mismatch`;
- `concurrency_mismatch`;
- `phase_mismatch`;
- `invalid_shape`;
- `clean_only_bin`;
- `profiled_only_bin`;
- `aic_total_unavailable`;
- `aic_components_unavailable`;
- `invalid_wall_time`.

`idle_step`, `non_measured_segment`, lane-only bins, and unavailable AIC values
are valid row/bin statuses in a successful bundle. Alignment, configuration,
concurrency, phase, shape, and wall-time errors are stable process-error codes;
they prevent a successful bundle rather than being treated as missing coverage.

### `manifest.json`

The manifest records:

- schema identifier and both repository commits;
- source job/model/system/runtime configuration;
- configuration fingerprint and exact concurrency cohorts;
- the explicit measured workload-segment set;
- input, mapped, overall-gap-eligible, shared, decomposition-eligible, and
  unsupported counts, including counts by reason code;
- row-mass and unique-bin coverage by concurrency and phase;
- same-run alignment method and result per cohort;
- `samples.csv` and `bins.csv` filenames and content hashes (the manifest does
  not attempt to hash itself);
- validation outcome and stable failure reasons.

## Statistical contract

For one semantic bin, let clean wall observations be `Wc`, and, when a profiled
lane exists, let each profiled row be the intact tuple `(Wp, C, M, B)` for
profiled wall, GPU compute, GPU communication, and GPU busy time. AIC may supply
deterministic `(A_compute, A_comm, A_other, A_total)` for the bin.

When component predictions are available, the canonical residual is
`A_other = A_total - A_compute - A_comm`. If an upstream predictor also emits an
`A_other` value, the reducer validates it against this identity within the closure
tolerance and fails on disagreement. This makes the later additive closure an
identity rather than an assumption about predictor internals.

### Point estimate

- `wall = median(Wc)`;
- `gap = A_total - wall` and `relative_error = gap / wall` whenever clean wall
  and AIC total exist, whether or not the profiled lane overlaps;
- `profiled_wall = median(Wp)`, diagnostic only, for Nsight-shared bins;
- `gpu_compute = median(C)`, `gpu_comm = median(M)`, and
  `gpu_busy = median(B)` for Nsight-shared bins;
- `overlap = gpu_compute + gpu_comm - gpu_busy`;
- `overhead = wall - gpu_busy`.

The additive terms are:

```text
term_compute_err = A_compute - gpu_compute
term_comm_err    = A_comm    - gpu_comm
term_aic_other   = A_other
term_overlap     = overlap
term_neg_overhead = -overhead
```

They must satisfy:

```text
gap = term_compute_err + term_comm_err + term_aic_other
    + term_overlap + term_neg_overhead
```

Component medians define the deterministic point vector. Per-sample descriptive
distributions may also be reported but must not replace or silently alter this
closure-preserving point estimate. Decomposition fields remain blank unless all
required lanes and AIC components exist.

### Sampling support

Overall-gap support is based on `n_clean`. Decomposition support is based on the
weaker lane, `min(n_clean, n_profiled)`. Both use the same classes:

- singleton: `n == 1`;
- sparse: `2 <= n <= 4`;
- repeated: `n >= 5`.

Support affects annotation only. It never excludes a structurally valid bin.

- Singleton bins receive the median point estimate, blank bounds, and an explicit
  `n=1` annotation.
- Sparse bins receive the median plus the observed minimum-to-maximum range.
- Repeated bins receive the median plus the first-to-third-quartile range.

These are descriptive support bands, not inferential confidence intervals. The
v1 contract does not bootstrap. Raw profiled compute/communication/busy values
remain row tuples in `samples.csv` so later analyses cannot accidentally destroy
their within-step relationship. Clean-wall bounds use the clean support class;
profiled-wall and composition bounds use the profiled support class. The
weaker-lane class controls only the annotation/style of decomposition results.

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
   concurrency-by-phase row/bin coverage matrices; and profiled-versus-clean
   wall perturbation diagnostics. Profiled wall remains diagnostic only.
2. `02_worst_errors`: the global top 20 absolute relative errors plus top-five
   concurrency-by-phase facets over every overall-gap-eligible bin. Bins without
   decomposition support remain in the ranking and are labeled accordingly.
3. `03_shape_map`: decode-request count versus mean decode KV, faceted by exact
   offered concurrency, colored by signed relative error and styled by clean
   support class. This family is decode-specific; context and mixed remain fully
   represented in trust, error-ranking, findings, and attribution outputs.
4. `04_attribution`: diverging additive-term bars for the worst
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
- profiler perturbation summaries;
- counts of singleton, sparse, and repeated bins;
- unsupported and AIC-unavailable mass by reason.

Every finding includes stable keys into `bins.csv`, `samples.csv`, or
`manifest.json`. The generator does not use subjective terms such as "low
confidence", does not infer causality beyond the additive identity, and does not
convert descriptive support bands into filters.

## Static artifact bundle

```text
insight_report/
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

## Failure handling

### Hard failures

The reducer exits nonzero for:

- missing or incompatible required schemas;
- configuration, repository-commit, or concurrency mismatch;
- ambiguous or incomplete same-run FPM-to-Nsight alignment;
- non-integer/negative counts or totals, zero-count/nonzero-total shapes, or a
  source/count-derived phase mismatch;
- non-finite measurements or non-positive authoritative clean wall;
- duplicate identities that violate the schema;
- inconsistent AIC component identity;
- decomposition closure outside tolerance;
- manifest count or artifact-hash inconsistency.

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
- negative overlap or overhead.

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
- phase derivation, zero-count/nonzero-total rejection, source-phase validation,
  and exact concurrency partitioning;
- explicit measured-segment selection and lane-specific coverage denominators;
- sample multiplicity without Cartesian expansion or truncation;
- clean and weaker-lane support classification;
- singleton/no-bound, sparse/min-max, and repeated/IQR descriptive bands;
- point-estimate and additive-term calculations;
- canonical AIC-other residual and upstream-component consistency validation;
- closure tolerance and negative-term preservation;
- worst-relative-error ranking and deterministic tie-breaking;
- overall-gap eligibility without profiled overlap and decomposition fail-closed
  behavior for those same bins;
- bin-equal versus clean-frequency-weighted summaries;
- dominant-term selection and generated-finding source keys;
- empty chart panels and stable artifact names.

### Synthetic integration fixture

The fixture contains context, decode, and mixed bins; singleton, sparse, and
repeated support; unequal lane multiplicities; an AIC-predicted clean-only bin;
missing AIC predictions; negative overlap/overhead; and each unsupported reason.
Tests validate normalized CSV/JSON contents and renderer completeness. Plot tests
assert source data, labels, output existence, and dimensions rather than brittle
pixel-perfect equality.

### Cross-repository contract fixture

Aiconfigurator emits a small three-file normalized fixture. Auto-collector
consumes it unchanged, validates the schema/manifest, and emits all expected
artifacts without opening SQLite or importing AIC.

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
479 unique bins are clean-only and must remain visible in overall gap analysis:

| Concurrency | Context | Decode | Mixed | Total clean-only bins/rows |
|---|---:|---:|---:|---:|
| c1 | 0 | 0 | 0 | 0 |
| c16 | 1 | 16 | 10 | 27 |
| c64 | 1 | 107 | 39 | 147 |
| c128 | 1 | 228 | 76 | 305 |
| total | 3 | 351 | 125 | 479 |

The current conservative layerwise availability proxy supports 11,214 shared bins
and represents 11,315 clean rows and 11,314 profiled rows. This is reported as AIC
availability, not semantic-join failure. That proxy evaluates prediction support
only after the shared-bin intersection, so it is not an oracle for AIC-total
availability across all 17,009 clean bins or the 479 clean-only bins.

The first reviewed adapter run must pin AIC-total-eligible row/bin counts for the
full clean population and the clean-only subset, plus the exact global/cohort
top-error and additive-term selections. Those values require the prototype
predictor adapter and are derived outputs rather than predeclared assumptions.

Artifact acceptance additionally requires:

- four SVG and four PNG multi-panel charts;
- `samples.csv`, `bins.csv`, `manifest.json`, `insights.json`, and `report.md`;
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
- Every unavailable field is blank/null with a stable reason, never fabricated as
  zero.
- Every decomposition row closes algebraically within tolerance.
- Global and cohort worst-relative-error rankings are stable and auditable.
- The complete deterministic bundle (five core files and eight chart images) is
  emitted and visually reviewed.
- Auto-collector renders from normalized files without reading raw Nsight data.
- Context and mixed zero-coverage panels remain explicit and do not cause a
  processing failure.
- The Qwen3 policy gate finds at least one decomposition-eligible decode bin in
  every requested concurrency cohort.
