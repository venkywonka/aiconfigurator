# Semantic FPM/Nsight Insight Post-processing — Design

- **Date:** 2026-07-06
- **Status:** Approved design — awaiting written-spec review
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
   AIC attribution. It emits small, versioned CSV/JSON contracts.
2. A CPU-only auto-collector renderer consumes only those contracts and emits a
   deterministic static review bundle: Markdown, SVG, PNG, CSV, and JSON.

The report first establishes whether the analyzed population is trustworthy,
then identifies the worst per-bin relative errors, and finally decomposes those
errors into compute, communication, AIC-other, overlap, and overhead terms.

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

- Retain 100% of valid clean and profiled real-run FPM observations.
- Align every profiled FPM observation to its own Nsight step mechanically and
  fail closed if that same-run mapping is incomplete or ambiguous.
- Compare clean and profiled lanes as independent distributions within a shared
  semantic bin, never as paired rows.
- Use every structurally valid shared bin regardless of sample count or runtime
  dispersion.
- Separate three nested populations: all clean observations, FPM/Nsight-shared
  bins, and AIC-supported shared bins.
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
- AIC predictions and provenance;
- numeric offered concurrency and configuration fingerprint.

It performs:

1. strict same-run profiled-FPM-to-Nsight alignment;
2. phase classification and semantic-key normalization;
3. exact concurrency/configuration partitioning;
4. clean/profiled semantic support calculation;
5. AIC availability lookup;
6. per-sample and per-bin descriptive statistics;
7. additive gap decomposition;
8. normalized artifact and manifest emission.

The reducer is the only stage that may read Nsight-derived step composition or
construct the AIC predictor.

### Stage 2: static insight renderer

The renderer runs in the CPU analysis job. It reads only the versioned normalized
contract. It calculates ranking, support annotations, bootstrap intervals,
generated findings, and chart-ready summaries, then emits the static bundle.

It must not:

- open `.sqlite`, `.nsys-rep`, or `.qdstrm` files;
- import or construct the AIC model;
- reinterpret the semantic key;
- repair missing decomposition fields;
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

`ctx_new_tokens` is mandatory. Omitting it would make unrelated pure-context
prompts appear semantically identical.

## Population semantics

The report carries three nested populations:

1. **All clean:** every structurally valid clean real-run observation. This is the
   authoritative workload and wall-time population.
2. **Nsight-shared:** all clean and profiled observations whose semantic key
   exists in both lanes after complete same-run profiled-FPM-to-Nsight mapping.
   Sample multiplicities may differ and are retained independently.
3. **AIC-supported:** Nsight-shared bins for which AIC returns all required
   prediction components.

Cross-run semantic overlap is set membership by bin, not row pairing. No
`min(n_clean, n_profiled)` truncation is applied to analysis samples. Row-mass
coverage and unique-bin coverage are reported separately.

Bins missing from one of the three populations remain in the output with an
explicit status. Their unavailable decomposition fields are blank, never zero.

## Normalized contract

The contract schema identifier is `fpm-semantic-insights/v1`. CSV blanks and JSON
`null` represent unavailable values. Numeric zero is always a measured or derived
value, never a missing-value sentinel.

### `semantic_step_samples.csv`

One row per retained clean or profiled observation. Required columns are:

- schema and provenance: `schema_version`, `configuration_fingerprint`,
  `concurrency`, `phase`, `lane`, `sample_id`, `workload_segment`;
- semantic inputs: exact request counts and token totals, all five normalized key
  fields, and a stable serialized `semantic_key`;
- timing: `wall_ms`; clean wall is authoritative, while profiled wall is retained
  only for perturbation diagnostics;
- profiled composition: `gpu_compute_ms`, `gpu_comm_ms`, `gpu_busy_ms` for mapped
  profiled rows, blank for clean rows;
- status: same-run mapping status, semantic support status, AIC availability, and
  a stable reason code when ineligible.

The profiled compute/communication/busy tuple remains on the same row so later
resampling preserves within-step correlation.

### `semantic_bin_summary.csv`

One row per `(configuration, concurrency, phase, semantic_key)`. Required fields
include:

- `n_clean`, `n_profiled`, clean and profiled row-mass weights;
- structural status and population-membership flags;
- medians and descriptive ranges for clean wall, profiled wall, GPU compute,
  communication, and busy time;
- AIC compute, communication, other, and total prediction;
- signed/absolute gap in milliseconds and signed/absolute relative error;
- overlap, overhead, and all five additive decomposition terms;
- closure error and stable identifiers needed by renderer-derived tables.

The reducer emits closure-preserving point estimates. The renderer does not
mutate this contract; it writes `tables/bin_insights.csv`, which adds support
class, bootstrap intervals, global/cohort rank, and dominant-term labels for
every bin.

### `semantic_unmatched.csv`

One row per structurally unsupported observation or bin, retaining its lane,
identity, raw shape, and stable reason code. Reasons include:

- `idle_step`;
- `same_run_alignment_missing`;
- `same_run_alignment_ambiguous`;
- `configuration_mismatch`;
- `concurrency_mismatch`;
- `phase_mismatch`;
- `clean_only_bin`;
- `profiled_only_bin`;
- `aic_prediction_unavailable`;
- `invalid_wall_time`.

### `semantic_join_manifest.json`

The manifest records:

- schema identifier and both repository commits;
- source job/model/system/runtime configuration;
- configuration fingerprint and exact concurrency cohorts;
- input, mapped, eligible, shared, AIC-supported, and unmatched counts;
- row-mass and unique-bin coverage by concurrency and phase;
- same-run alignment method and result per cohort;
- artifact filenames and content hashes;
- validation outcome and stable failure reasons.

## Statistical contract

For one semantic bin, let clean wall observations be `Wc`, and let each profiled
row be the intact tuple `(Wp, C, M, B)` for profiled wall, GPU compute, GPU
communication, and GPU busy time. AIC supplies deterministic
`(A_compute, A_comm, A_other, A_total)` for the bin.

### Point estimate

- `wall = median(Wc)`;
- `profiled_wall = median(Wp)`, diagnostic only;
- `gpu_compute = median(C)`;
- `gpu_comm = median(M)`;
- `gpu_busy = median(B)`;
- `overlap = gpu_compute + gpu_comm - gpu_busy`;
- `overhead = wall - gpu_busy`;
- `gap = A_total - wall`;
- `relative_error = gap / wall`.

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

Component medians define the deterministic point vector. Per-sample derived
distributions may also be reported but must not replace or silently alter this
closure-preserving point estimate.

### Sampling support

Support is based on `min(n_clean, n_profiled)`:

- singleton: `n == 1`;
- sparse: `2 <= n <= 4`;
- repeated: `n >= 5`.

Support affects annotation only. It never excludes a structurally valid bin.

- Singleton bins receive a point estimate and `uncertainty_unavailable`.
- Sparse bins receive point estimates plus descriptive observed ranges.
- Repeated bins receive deterministic 95% percentile-bootstrap intervals for the
  median using 1,000 replicates.

Bootstrap samples are drawn independently across clean and profiled lanes. The
profiled tuples are resampled as rows, not as independent component columns. The
random seed is derived from the schema version plus full semantic identity.

### Ranking and aggregation

The primary ranking is worst absolute relative error:

```text
abs((A_total - median_clean_wall) / median_clean_wall)
```

The renderer emits a global top 20 and a top 5 within every concurrency-by-phase
cohort. Stable tie-breakers are absolute millisecond gap, then serialized semantic
key. Labels retain signed percentage, signed milliseconds, sample counts, support
class, phase, and concurrency.

Dominant cause is the additive term with the largest absolute magnitude; its sign
is retained.

Cohort summaries publish both:

- the bin-equal median; and
- the clean-row-frequency-weighted mean.

Neither is labeled simply "overall" without its weighting semantics.

## Insight and chart catalog

The bundle tells a three-part diagnostic story.

### Can the analyzed population be trusted?

1. `01_coverage_funnel`: all-clean to Nsight-shared to AIC-supported row mass and
   unique bins, overall and by phase.
2. `02_coverage_matrix`: concurrency-by-phase semantic-overlap and AIC-availability
   matrices.
3. `03_profiler_perturbation`: profiled-versus-clean wall distributions within
   shared bins. Profiled wall remains diagnostic only.

### Where is AIC most wrong?

4. `04_global_top_errors`: horizontal ranking of the global top 20 absolute
   relative errors.
5. `05_cohort_top_errors`: small multiples containing the top five errors per
   concurrency and phase.
6. `06_decode_shape_map`: decode-request count versus mean decode KV, faceted by
   concurrency, colored by signed relative error and styled by support class.

### Why is it wrong?

7. `07_additive_terms`: diverging stacked additive terms for the globally worst
   bins.
8. `08_prediction_parity`: AIC total versus clean wall with the identity line,
   colored by dominant additive term and styled by support class.
9. Evidence-backed generated findings in `report.md` and
   `tables/generated_findings.json`.

Support styling is stable across charts: singleton markers are hollow, sparse
markers are outlined, and repeated markers are filled. Static labels and companion
tables carry the exact sample counts.

## Generated findings

Generated prose is descriptive and evidence-backed. It may state:

- population coverage and the largest missing-mass phase/cohort;
- the worst absolute relative-error bin and signed millisecond error;
- the dominant additive term for that bin;
- median signed bias by cohort;
- profiler perturbation summaries;
- counts of singleton, sparse, and repeated bins;
- unmatched and AIC-unavailable mass by reason.

Every finding includes stable source-table keys. The generator does not use
subjective terms such as "low confidence", does not infer causality beyond the
additive identity, and does not convert descriptive support bands into filters.

## Static artifact bundle

```text
insight_report/
├── report.md
├── summary.json
├── charts/
│   ├── 01_coverage_funnel.{svg,png}
│   ├── 02_coverage_matrix.{svg,png}
│   ├── 03_profiler_perturbation.{svg,png}
│   ├── 04_global_top_errors.{svg,png}
│   ├── 05_cohort_top_errors.{svg,png}
│   ├── 06_decode_shape_map.{svg,png}
│   ├── 07_additive_terms.{svg,png}
│   └── 08_prediction_parity.{svg,png}
└── tables/
    ├── bin_insights.csv
    ├── coverage.csv
    ├── top_errors.csv
    ├── cohort_summary.csv
    └── generated_findings.json
```

`report.md` is the reviewer entry point. It contains the generated findings,
coverage caveats, PNG previews, and links to full-resolution SVG and CSV evidence.

Rendering is deterministic:

- stable sort and filenames;
- fixed palette, dimensions, fonts, and support styling;
- seeded bootstrap;
- no timestamps embedded in figures;
- no network dependency;
- every chart backed by a companion table.

## Failure handling

### Hard failures

The reducer exits nonzero for:

- missing or incompatible required schemas;
- configuration, repository-commit, or concurrency mismatch;
- ambiguous or incomplete same-run FPM-to-Nsight alignment;
- non-finite measurements or non-positive authoritative clean wall;
- duplicate identities that violate the schema;
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
requires at least one AIC-supported decode bin per requested concurrency and does
not require natural context or mixed overlap.

## Testing

### Unit tests

- half-up normalization and absent-axis encoding;
- phase derivation and exact concurrency partitioning;
- sample multiplicity without Cartesian expansion or truncation;
- support-band classification;
- point-estimate and additive-term calculations;
- closure tolerance and negative-term preservation;
- worst-relative-error ranking and deterministic tie-breaking;
- tuple-preserving deterministic bootstrap;
- bin-equal versus clean-frequency-weighted summaries;
- dominant-term selection and generated-finding source keys;
- empty chart panels and stable artifact names.

### Synthetic integration fixture

The fixture contains context, decode, and mixed bins; singleton, sparse, and
repeated support; unequal lane multiplicities; missing AIC predictions; negative
overlap/overhead; and each unmatched reason. Tests validate normalized CSV/JSON
contents and renderer completeness. Plot tests assert source data, labels, output
existence, and dimensions rather than brittle pixel-perfect equality.

### Cross-repository contract fixture

Aiconfigurator emits a small normalized fixture. Auto-collector consumes it
unchanged, validates the schema/manifest, and emits all expected artifacts without
opening SQLite or importing AIC.

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

The current conservative layerwise availability proxy supports 11,214 shared bins
and represents 11,315 clean rows and 11,314 profiled rows. This is reported as AIC
availability, not semantic-join failure.

The first reviewed adapter run pins the exact global/cohort top-error and additive
term tables. Those values are derived outputs rather than predeclared assumptions.

Artifact acceptance additionally requires:

- eight SVG and eight PNG charts;
- `report.md`, `summary.json`, and all five evidence tables;
- valid relative links from the report;
- exact reconciliation between report, chart-source tables, and manifest counts;
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
- All structurally valid shared bins are analyzed regardless of support band.
- Every unavailable field is blank/null with a stable reason, never fabricated as
  zero.
- Every decomposition row closes algebraically within tolerance.
- Global and cohort worst-relative-error rankings are stable and auditable.
- The complete deterministic static bundle is emitted and visually reviewed.
- Auto-collector renders from normalized files without reading raw Nsight data.
- Context and mixed zero-coverage panels remain explicit and do not cause a
  processing failure.
- The Qwen3 policy gate finds at least one AIC-supported decode bin in every
  requested concurrency cohort.
