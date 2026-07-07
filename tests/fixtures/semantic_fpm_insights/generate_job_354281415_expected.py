#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the artifact-backed job-354281415 semantic/predictor oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collector.layerwise.diagnostics.semantic_fpm_aic import (
    RepositoryAicConfig,
    build_repository_predictor,
)
from collector.layerwise.diagnostics.semantic_fpm_insights import (
    MarkerObservation,
    ProfiledFpmObservation,
    align_profiled_fpm_to_nsys,
)
from collector.layerwise.diagnostics.semantic_fpm_predictor import (
    ARTIFACT_PROXY_LOOKUP_POLICY_VERSION,
    predict_clean_bins,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    build_semantic_population,
    load_fpm_phase_csv,
    support_class,
)

COHORTS = ("c1", "c16", "c64", "c128")
MARKER_RE = re.compile(r"bench_step::N(\d+)::bs(\d+)::past(\d+)(?:::run(\d+))?")
ARCHIVE_BYTES = 6_554_156_865
ARCHIVE_SHA256 = "de4bd64b5f327fa2e73e131d5e7ec3346367814ece0181fe305ec05a795b22be"
LAYERWISE_SHA256 = "17781fac0d806c3683642c85e0f5e76f6046ab8739ad154fc1e2db01646c18f9"
INPUT_HASHES = {
    "c1": {
        "clean_fpm_phase_sha256": "4bd2cad42947fb8c52d64ec22a4e579f2fce327fa309254672a127c140d790f9",
        "profiled_fpm_phase_sha256": "49a5ee2ca87f7c5b0f944cea86a01256c52c58973d72f861771e479ef436c644",
        "nsys_sqlite_sha256": "314b5d88c0504170b76f70bb996cbb29016c0715d6ed98c51af4e2200e121c0d",
    },
    "c16": {
        "clean_fpm_phase_sha256": "f9defa6e90a8be0b78d5fc6a597ad68f5dfd349d1e9879b56e4af120c0c404b9",
        "profiled_fpm_phase_sha256": "a87b56149255b21dc91ca13be2879edc0ba5d8877d8819931c285a32eaa43761",
        "nsys_sqlite_sha256": "26603e2aeffec5c1f41a3e90ba2d3c775def28ec93a5daf89e1f226814c5e57b",
    },
    "c64": {
        "clean_fpm_phase_sha256": "8b16f0655b93a78ca35444b54ca7ffbe4c4ea9208ba44014a8328c15a217d7ef",
        "profiled_fpm_phase_sha256": "daab99031476e1d2b5989f7355759c52c35c6a344e290c73a1e843514fda886c",
        "nsys_sqlite_sha256": "36e8d31f8ca574300f98e46c347f79b5db73bbf8ed9e9caf1b4c2fbef60ce3dc",
    },
    "c128": {
        "clean_fpm_phase_sha256": "4054e1203ffdad1fcbfd8becfd9ea89799fe14d4ad6a7bd198c0f558bdcd49d5",
        "profiled_fpm_phase_sha256": "9eff5bcd827601477fc35b73c298089f7a54c3865bc391c96c30e17c14e73368",
        "nsys_sqlite_sha256": "fefa00b10b44a4c3071da118aac0ebf012e7f026b9e8313fe0534cf058c7bf0a",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _input_paths(artifact_root: Path, sqlite_root: Path, cohort: str) -> dict[str, Path]:
    return {
        "clean_fpm_phase_sha256": artifact_root / cohort / "fpm_metrics_phase.csv",
        "profiled_fpm_phase_sha256": artifact_root / cohort / "attribute/fpm_metrics_phase.csv",
        "nsys_sqlite_sha256": sqlite_root / cohort / "attribute/nsys/fpm_worker.sqlite",
    }


def verify_source_hashes(
    archive: Path,
    artifact_root: Path,
    sqlite_root: Path,
    layerwise_csv: Path,
) -> None:
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"archive size mismatch: expected {ARCHIVE_BYTES}, got {archive.stat().st_size}")
    archive_hash = sha256_file(archive)
    if archive_hash != ARCHIVE_SHA256:
        raise ValueError(f"archive SHA-256 mismatch: expected {ARCHIVE_SHA256}, got {archive_hash}")
    layerwise_hash = sha256_file(layerwise_csv)
    if layerwise_hash != LAYERWISE_SHA256:
        raise ValueError(f"layerwise SHA-256 mismatch: expected {LAYERWISE_SHA256}, got {layerwise_hash}")
    for cohort in COHORTS:
        for hash_name, path in _input_paths(artifact_root, sqlite_root, cohort).items():
            actual = sha256_file(path)
            expected = INPUT_HASHES[cohort][hash_name]
            if actual != expected:
                raise ValueError(f"{cohort} {hash_name} mismatch: expected {expected}, got {actual} ({path})")


def load_markers(sqlite_path: Path) -> list[MarkerObservation]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro&immutable=1", uri=True)
    try:
        raw = connection.execute(
            "SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'bench_step::%' ORDER BY globalTid,start"
        ).fetchall()
    finally:
        connection.close()

    markers = []
    for text, start, end, global_tid in raw:
        match = MARKER_RE.fullmatch(text)
        if match is None:
            raise ValueError(f"unexpected marker label: {text}")
        step, batch, past, measure_run = match.groups()
        markers.append(
            MarkerObservation(
                rank_key=str(global_tid),
                marker_step=int(step),
                measure_run=int(measure_run or 0),
                decode_batch=int(batch),
                mean_decode_kv=int(past),
                start_ns=int(start),
                end_ns=int(end),
            )
        )
    return markers


def alignment_summary(artifact_root: Path, sqlite_root: Path, cohort: str) -> dict:
    concurrency = int(cohort.removeprefix("c"))
    samples = load_fpm_phase_csv(
        artifact_root / cohort / "attribute/fpm_metrics_phase.csv",
        lane="profiled",
        concurrency=concurrency,
    )
    fpm = [
        ProfiledFpmObservation(counter_id=sample.counter_id, phase=sample.phase, shape=sample.shape)
        for sample in samples
        if sample.workload_segment == "real" and sample.dp_rank == 0
    ]
    markers = load_markers(sqlite_root / cohort / "attribute/nsys/fpm_worker.sqlite")
    result = align_profiled_fpm_to_nsys(
        fpm,
        markers,
        expected_rank_keys=tuple(sorted({marker.rank_key for marker in markers})),
    )
    return {
        "profiled_fpm_rows": len(fpm),
        "mapped_rows": result.mapped_count,
        "rank_keys": list(result.rank_keys),
        "raw_markers_per_rank": result.raw_marker_count,
        "canonical_markers_per_rank": result.canonical_marker_count,
        "nonmonotonic": [
            [row.marker_step, row.decode_batch, row.mean_decode_kv, row.measure_run]
            for row in result.nonmonotonic_markers
        ],
        "mapping_hash": result.mapping_hash,
        "first_mapping": result.mapping[0].__dict__,
        "last_mapping": result.mapping[-1].__dict__,
        "prefix_markers": len(result.unmatched_prefix_markers),
        "suffix_markers": len(result.unmatched_suffix_markers),
        "skipped_markers": [row.marker_step for row in result.internal_skipped_markers],
        "segments": [row.__dict__ for row in result.segments],
    }


def _load_population(artifact_root: Path, *, configuration_fingerprint: str):
    all_samples = []
    for cohort in COHORTS:
        concurrency = int(cohort.removeprefix("c"))
        all_samples.extend(
            load_fpm_phase_csv(artifact_root / cohort / "fpm_metrics_phase.csv", lane="clean", concurrency=concurrency)
        )
        all_samples.extend(
            load_fpm_phase_csv(
                artifact_root / cohort / "attribute/fpm_metrics_phase.csv",
                lane="profiled",
                concurrency=concurrency,
            )
        )
    return build_semantic_population(
        all_samples,
        configuration_fingerprint=configuration_fingerprint,
        measured_segments=frozenset({"real"}),
    )


def population_summary(artifact_root: Path) -> dict:
    population = _load_population(
        artifact_root,
        configuration_fingerprint="job-354281415-placeholder",
    )
    clean_bins = [bin_ for bin_ in population.bins if bin_.n_clean]
    shared_bins = [bin_ for bin_ in population.bins if bin_.nsight_shared]
    clean_only_bins = [bin_ for bin_ in population.bins if bin_.clean_only]
    support_counts = dict.fromkeys(("singleton", "sparse", "repeated"), 0)
    for bin_ in clean_bins:
        support_counts[support_class(bin_.n_clean)] += 1

    by_cohort = {}
    for cohort in COHORTS:
        concurrency = int(cohort.removeprefix("c"))
        cohort_bins = [bin_ for bin_ in population.bins if bin_.concurrency == concurrency]
        cohort_samples = [sample for sample in population.samples if sample.concurrency == concurrency]
        by_cohort[cohort] = {
            "clean_rows": sum(sample.lane == "clean" for sample in cohort_samples),
            "profiled_rows": sum(sample.lane == "profiled" for sample in cohort_samples),
            "clean_bins": sum(bin_.n_clean > 0 for bin_ in cohort_bins),
            "profiled_bins": sum(bin_.n_profiled > 0 for bin_ in cohort_bins),
            "shared_bins": sum(bin_.nsight_shared for bin_ in cohort_bins),
            "shared_clean_rows": sum(bin_.n_clean for bin_ in cohort_bins if bin_.nsight_shared),
            "shared_profiled_rows": sum(bin_.n_profiled for bin_ in cohort_bins if bin_.nsight_shared),
            "clean_only_bins": sum(bin_.clean_only for bin_ in cohort_bins),
            "clean_collision_bins": sum(
                bin_.n_clean > 0 and bin_.clean_raw_shape_cardinality > 1 for bin_ in cohort_bins
            ),
        }

    return {
        "source_rows": len(population.samples),
        "clean_rows": sum(sample.lane == "clean" for sample in population.samples),
        "profiled_rows": sum(sample.lane == "profiled" for sample in population.samples),
        "clean_bins": len(clean_bins),
        "profiled_bins": sum(bin_.n_profiled > 0 for bin_ in population.bins),
        "shared_bins": len(shared_bins),
        "shared_clean_rows": sum(bin_.n_clean for bin_ in shared_bins),
        "shared_profiled_rows": sum(bin_.n_profiled for bin_ in shared_bins),
        "clean_only_bins": len(clean_only_bins),
        "clean_only_rows": sum(bin_.n_clean for bin_ in clean_only_bins),
        "clean_support": support_counts,
        "max_clean_count": max(bin_.n_clean for bin_ in clean_bins),
        "clean_only_by_cohort_phase": {
            cohort: {
                phase: sum(
                    bin_.clean_only and bin_.concurrency == int(cohort.removeprefix("c")) and bin_.phase == phase
                    for bin_ in clean_only_bins
                )
                for phase in ("context", "decode", "mixed")
            }
            for cohort in COHORTS
        },
        "by_cohort": by_cohort,
        "phase_row_coverage": {
            phase: {
                "clean_shared": sum(bin_.n_clean for bin_ in shared_bins if bin_.phase == phase),
                "clean_total": sum(sample.lane == "clean" and sample.phase == phase for sample in population.samples),
                "profiled_shared": sum(bin_.n_profiled for bin_ in shared_bins if bin_.phase == phase),
                "profiled_total": sum(
                    sample.lane == "profiled" and sample.phase == phase for sample in population.samples
                ),
            }
            for phase in ("context", "decode", "mixed")
        },
    }


def _artifact_proxy_parity_record() -> str:
    return json.dumps(
        {
            "attention_dp_size": 1,
            "attention_quant": "bf16",
            "backend": "vllm",
            "backend_version": "0.20.1",
            "chunked_prefill": True,
            "dp_size": 1,
            "ep_size": 1,
            "gemm_quant": "bf16",
            "gpu_count": 8,
            "kv_cache_dtype": "bf16",
            "kv_cache_quant": "bf16",
            "max_num_batched_tokens": 40960,
            "max_num_seqs": 256,
            "model": "Qwen/Qwen3-32B",
            "model_revision": "artifact-job-354281415",
            "moe_quant": "bf16",
            "numerical_dtype": "bf16",
            "pp_size": 1,
            "prefix_caching": False,
            "runtime_flags": {"source": "cached-artifact-proxy"},
            "schema_version": "aic-runtime-parity/v1",
            "system": "h100_sxm",
            "tp_size": 8,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def predictor_summary(artifact_root: Path, layerwise_csv: Path) -> dict:
    parity_record = _artifact_proxy_parity_record()
    configuration_fingerprint = hashlib.sha256(parity_record.encode()).hexdigest()
    population = _load_population(
        artifact_root,
        configuration_fingerprint=configuration_fingerprint,
    )
    config = RepositoryAicConfig(
        repo_root=REPO_ROOT,
        repo_commit="artifact-proxy-job-354281415",
        configuration_fingerprint=configuration_fingerprint,
        parity_record=parity_record,
        layerwise_csv=layerwise_csv,
        comm_version="0.19.0",
        context_vllm_config_hash="2862698e53019524",
        decode_vllm_config_hash="dda11a431c370f78",
        artifact_proxy_decode_max_num_seqs=64,
        verify_repository=False,
    )
    predictions = predict_clean_bins(
        population,
        build_repository_predictor(config),
        required_lookup_policy=ARTIFACT_PROXY_LOOKUP_POLICY_VERSION,
    )
    bin_by_id = {bin_.bin_id: bin_ for bin_ in population.bins}
    by_cohort_phase = {}
    for cohort in COHORTS:
        concurrency = int(cohort.removeprefix("c"))
        for phase in ("context", "decode", "mixed"):
            scope = [
                item for item in predictions if item.query.concurrency == concurrency and item.query.phase == phase
            ]
            by_cohort_phase[f"{cohort}/{phase}"] = {
                "attempted": len(scope),
                "eligible": sum(item.record.status == "ok" for item in scope),
                "reasons": dict(sorted(Counter(item.record.reason for item in scope).items())),
            }
    clean_only = [item for item in predictions if bin_by_id[item.bin_id].clean_only]
    evaluated_shapes = Counter(
        tuple(value for _, value in item.record.evaluated_shape.items())
        for item in predictions
        if item.record.status == "ok" and item.record.evaluated_shape is not None
    )
    prediction_vector = [
        {
            "bin_id": item.bin_id,
            "status": item.record.status,
            "reason": item.record.reason,
            "requested_shape": asdict(item.record.requested_shape),
            "evaluated_shape": (
                asdict(item.record.evaluated_shape) if item.record.evaluated_shape is not None else None
            ),
            "lookup_surface_id": item.record.lookup_surface_id,
            "scheduler_surface_content_hash": item.record.scheduler_surface_content_hash,
            "axis_lookups": [asdict(lookup) for lookup in item.record.axis_lookups],
            "total_ms": item.record.total_ms,
            "compute_ms": item.record.compute_ms,
            "communication_ms": item.record.communication_ms,
            "other_ms": item.record.other_ms,
            "operation_inventory_hash": item.record.operation_inventory_hash,
            "operation_values": item.record.operation_values,
            "operation_lookups": [asdict(lookup) for lookup in item.record.operation_lookups],
        }
        for item in sorted(predictions, key=lambda prediction: prediction.bin_id)
    ]
    prediction_vector_sha256 = hashlib.sha256(
        json.dumps(
            prediction_vector,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    top_errors = []
    for item in predictions:
        if item.record.total_ms is None:
            continue
        clean_walls = sorted(sample.wall_ms for sample in bin_by_id[item.bin_id].clean_samples)
        midpoint = len(clean_walls) // 2
        clean_median = (
            clean_walls[midpoint] if len(clean_walls) % 2 else (clean_walls[midpoint - 1] + clean_walls[midpoint]) / 2.0
        )
        relative_error = (item.record.total_ms - clean_median) / clean_median
        top_errors.append(
            {
                "bin_id": item.bin_id,
                "concurrency": item.query.concurrency,
                "phase": item.query.phase,
                "semantic_key": item.query.semantic_key,
                "relative_error": relative_error,
            }
        )
    top_errors.sort(
        key=lambda row: (
            -abs(row["relative_error"]),
            row["concurrency"],
            row["phase"],
            row["semantic_key"],
            row["bin_id"],
        )
    )
    return {
        "status": "frozen_artifact_proxy_v1",
        "policy": ARTIFACT_PROXY_LOOKUP_POLICY_VERSION,
        "production_exact_surface_eligible": 0,
        "production_exact_surface_reason": "cached_layerwise_decode_max_num_seqs_64_vs_runtime_256",
        "call_count": len(predictions),
        "status_reason": dict(
            sorted(Counter(f"{item.record.status}/{item.record.reason}" for item in predictions).items())
        ),
        "clean_only": {
            "attempted": len(clean_only),
            "eligible": sum(item.record.status == "ok" for item in clean_only),
            "reasons": dict(sorted(Counter(item.record.reason for item in clean_only).items())),
        },
        "by_cohort_phase": by_cohort_phase,
        "evaluated_shapes": {str(key): count for key, count in sorted(evaluated_shapes.items())},
        "prediction_vector_sha256": prediction_vector_sha256,
        "top_absolute_relative_errors": top_errors[:20],
        "layerwise_sha256": LAYERWISE_SHA256,
    }


def build_oracle(artifact_root: Path, sqlite_root: Path, layerwise_csv: Path) -> dict:
    return {
        "oracle_schema": "job-354281415-semantic-fpm/v1",
        "source": {
            "job_id": 354281415,
            "archive_basename": "result.tar.gz",
            "archive_bytes": ARCHIVE_BYTES,
            "archive_sha256": ARCHIVE_SHA256,
            "collection_pipeline_id": 56729332,
            "aiconfigurator_commit": "1a8f4809fbb1953d51578fc1e4fcb30cb7f6c110",
            "auto_collector_commit": "4f27f96f2b9e63b6f96b79b9a21faf585f36ec0c",
            "model": "Qwen/Qwen3-32B",
            "system": "h100_sxm",
            "tensor_parallel": 8,
            "measured_segments": ["real"],
        },
        "policies": {
            "schema_version": "fpm-semantic-insights/v1",
            "alignment_version": "profiled-monotonic-v1",
            "marker_encoder_version": "python-round-half-even-v1",
            "semantic_key_version": "one-token-half-up-v1",
            "mapping_hash_serialization": (
                "sha256(canonical-json[{counter_id,fpm_sequence_index,marker_canonical_index,marker_step,measure_run}])"
            ),
        },
        "input_hashes": INPUT_HASHES,
        "alignment": {cohort: alignment_summary(artifact_root, sqlite_root, cohort) for cohort in COHORTS},
        "population": population_summary(artifact_root),
        "witnesses": {
            "half_up_cross_run_wiggle": {
                "cohort": "c16",
                "clean_counter_id": 1390,
                "clean_decode_kv_tokens": 33802,
                "profiled_counter_id": 1392,
                "profiled_decode_kv_tokens": 33803,
                "decode_requests": 4,
                "semantic_key": [0, 4, None, None, 8451],
                "clean_marker_anchor": 8450,
            },
            "sparse_collision": {
                "cohort": "c64",
                "clean_counter_ids": [1173, 1226],
                "semantic_key": [0, 64, None, None, 9253],
                "n_clean": 2,
                "exact_shape_cardinality": 2,
            },
            "max_clean_multiplicity": {
                "cohort": "c128",
                "clean_counter_ids": [1116, 1200, 1207, 1242],
                "semantic_key": [0, 128, None, None, 8077],
                "n_clean": 4,
                "exact_shape_cardinality": 4,
            },
            "clean_only": [
                {"cohort": "c16", "phase": "context", "counter_id": 1050},
                {"cohort": "c64", "phase": "decode", "counter_id": 1052},
                {"cohort": "c128", "phase": "mixed", "counter_id": 1058},
            ],
        },
        "predictor_oracle": predictor_summary(artifact_root, layerwise_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--sqlite-root", type=Path, required=True)
    parser.add_argument("--layerwise-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verify_source_hashes(args.archive, args.artifact_root, args.sqlite_root, args.layerwise_csv)
    rendered = (
        json.dumps(
            build_oracle(args.artifact_root, args.sqlite_root, args.layerwise_csv),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()
