#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit the normalized semantic FPM source contract.

This module is the shipping Stage-1 boundary.  It preserves the independent
clean/profiled populations, invokes AIC exactly once per measured clean bin,
and writes the only three files consumed by the CPU renderer.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from collector.layerwise.diagnostics.semantic_fpm_aic import (
    RepositoryAicConfig,
    build_repository_predictor,
)
from collector.layerwise.diagnostics.semantic_fpm_insights import (
    ALIGNMENT_VERSION,
    MARKER_ENCODER_VERSION,
    SCHEMA_VERSION,
    AlignmentSegment,
    CanonicalMarker,
)
from collector.layerwise.diagnostics.semantic_fpm_predictor import (
    LOOKUP_POLICY_VERSION,
    SemanticBinPrediction,
    predict_clean_bins,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    ContractError,
    FpmSample,
    SemanticPopulation,
    SemanticPopulationBin,
    descriptive_stats,
    support_class,
)

CONTRACT_WRITER_VERSION = "semantic-contract-writer-v1"
SEMANTIC_KEY_VERSION = "one-token-half-up-v1"
QUERY_RECONSTRUCTION_VERSION = "semantic-key-product-v1"
CRITICAL_KEY_POLICY = "max-busy-lower-numeric-v1"
_EMPTY_UNKNOWN_KERNEL_HASH = hashlib.sha256(b"").hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_git_sha(value: str) -> bool:
    return len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _is_nonnegative_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.0


def _numeric_rank_keys(keys: tuple[str, ...]) -> tuple[str, ...]:
    if not keys or any(not key.isdecimal() for key in keys):
        raise ContractError(
            process_code="rank_identity_unavailable",
            detail="rank keys must be non-empty decimal identities",
        )
    ordered = tuple(sorted(keys, key=int))
    if keys != ordered or len(keys) != len(set(keys)):
        raise ContractError(
            process_code="rank_tuple_identity_mismatch",
            detail="rank keys must be unique and numerically ordered",
        )
    return ordered


@dataclass(frozen=True)
class AlignmentMarkerIdentity:
    """Named, canonical identity for one diagnostic Nsight marker."""

    marker_step: int
    measure_run: int
    decode_batch: int
    mean_decode_kv: int

    def __post_init__(self) -> None:
        values = (self.marker_step, self.measure_run, self.decode_batch, self.mean_decode_kv)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ContractError(
                process_code="same_run_alignment_ambiguous",
                detail="alignment marker identity fields must be non-negative integers",
            )

    @classmethod
    def from_canonical(cls, marker: CanonicalMarker) -> AlignmentMarkerIdentity:
        return cls(
            marker_step=marker.marker_step,
            measure_run=marker.measure_run,
            decode_batch=marker.decode_batch,
            mean_decode_kv=marker.mean_decode_kv,
        )


@dataclass(frozen=True)
class AlignmentSegmentProvenance:
    """Named boundaries for one maximal contiguous same-run mapping segment."""

    measure_run: int
    start_marker_canonical_index: int
    end_marker_canonical_index: int
    start_marker_step: int
    end_marker_step: int
    mapped_rows: int

    def __post_init__(self) -> None:
        values = (
            self.measure_run,
            self.start_marker_canonical_index,
            self.end_marker_canonical_index,
            self.start_marker_step,
            self.end_marker_step,
            self.mapped_rows,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise ContractError(
                process_code="same_run_alignment_missing",
                detail="alignment segment fields must be non-negative integers",
            )
        if (
            self.start_marker_canonical_index > self.end_marker_canonical_index
            or self.start_marker_step > self.end_marker_step
            or self.mapped_rows != self.end_marker_canonical_index - self.start_marker_canonical_index + 1
        ):
            raise ContractError(
                process_code="same_run_alignment_missing",
                detail="alignment segment boundaries do not reconcile mapped rows",
            )

    @classmethod
    def from_alignment(cls, segment: AlignmentSegment) -> AlignmentSegmentProvenance:
        return cls(
            measure_run=segment.measure_run,
            start_marker_canonical_index=segment.start_marker_canonical_index,
            end_marker_canonical_index=segment.end_marker_canonical_index,
            start_marker_step=segment.start_marker_step,
            end_marker_step=segment.end_marker_step,
            mapped_rows=segment.mapped_rows,
        )


@dataclass(frozen=True)
class CohortAlignmentProvenance:
    """Complete same-run mapping and rank identity for one concurrency cohort."""

    concurrency: int
    input_profiled_rows: int
    mapped_profiled_rows: int
    raw_marker_count: int
    canonical_marker_count: int
    mapping_hash: str
    expected_rank_keys: tuple[str, ...]
    captured_rank_keys: tuple[str, ...]
    rank_identity_kind: str
    rank_mapping_provenance: str
    nonmonotonic_marker_identities: tuple[AlignmentMarkerIdentity, ...]
    unmatched_prefix_marker_identities: tuple[AlignmentMarkerIdentity, ...]
    unmatched_suffix_marker_identities: tuple[AlignmentMarkerIdentity, ...]
    internal_skipped_marker_identities: tuple[AlignmentMarkerIdentity, ...]
    mapped_segments: tuple[AlignmentSegmentProvenance, ...]

    def __post_init__(self) -> None:
        counts = (
            self.concurrency,
            self.input_profiled_rows,
            self.mapped_profiled_rows,
            self.raw_marker_count,
            self.canonical_marker_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ContractError(
                process_code="invalid_measurement",
                detail="alignment concurrency and row counts must be integers",
            )
        if self.concurrency <= 0 or any(value < 0 for value in counts[1:]):
            raise ContractError(
                process_code="invalid_measurement",
                detail="alignment concurrency/counts must be non-negative with positive concurrency",
            )
        if self.input_profiled_rows != self.mapped_profiled_rows:
            raise ContractError(
                process_code="same_run_alignment_missing",
                detail="successful contract requires every profiled row to map",
            )
        if not _is_sha256(self.mapping_hash):
            raise ContractError(
                process_code="same_run_alignment_ambiguous",
                detail="alignment mapping hash must be a canonical SHA-256",
            )
        _numeric_rank_keys(self.expected_rank_keys)
        _numeric_rank_keys(self.captured_rank_keys)
        if self.expected_rank_keys != self.captured_rank_keys:
            raise ContractError(
                process_code="incomplete_rank_capture",
                detail="captured rank keys do not match expected rank keys",
            )
        if self.rank_identity_kind not in {"logical_rank", "global_pid"}:
            raise ContractError(
                process_code="rank_identity_unavailable",
                detail=f"unsupported rank identity kind {self.rank_identity_kind!r}",
            )
        if not self.rank_mapping_provenance:
            raise ContractError(
                process_code="rank_identity_unavailable",
                detail="rank mapping provenance must be explicit",
            )
        marker_groups = (
            self.nonmonotonic_marker_identities,
            self.unmatched_prefix_marker_identities,
            self.unmatched_suffix_marker_identities,
            self.internal_skipped_marker_identities,
        )
        if any(not isinstance(identity, AlignmentMarkerIdentity) for group in marker_groups for identity in group):
            raise ContractError(
                process_code="schema_incompatible",
                detail="alignment marker identities must use named canonical records",
            )
        if any(not isinstance(segment, AlignmentSegmentProvenance) for segment in self.mapped_segments):
            raise ContractError(
                process_code="schema_incompatible",
                detail="alignment segments must use named provenance records",
            )
        invalid_segment_order = False
        previous_end_index = -1
        internal_gap_count = 0
        for segment in self.mapped_segments:
            if segment.start_marker_canonical_index <= previous_end_index:
                invalid_segment_order = True
                break
            if previous_end_index >= 0:
                internal_gap_count += segment.start_marker_canonical_index - previous_end_index - 1
            previous_end_index = segment.end_marker_canonical_index
        if (
            invalid_segment_order
            or sum(segment.mapped_rows for segment in self.mapped_segments) != self.mapped_profiled_rows
            or internal_gap_count != len(self.internal_skipped_marker_identities)
            or self.raw_marker_count != self.canonical_marker_count + len(self.nonmonotonic_marker_identities)
            or self.canonical_marker_count
            != (
                self.mapped_profiled_rows
                + len(self.unmatched_prefix_marker_identities)
                + len(self.unmatched_suffix_marker_identities)
                + len(self.internal_skipped_marker_identities)
            )
            or (
                bool(self.mapped_segments)
                and self.mapped_segments[0].start_marker_canonical_index != len(self.unmatched_prefix_marker_identities)
            )
            or (
                bool(self.mapped_segments)
                and self.mapped_segments[-1].end_marker_canonical_index
                + len(self.unmatched_suffix_marker_identities)
                + 1
                != self.canonical_marker_count
            )
        ):
            raise ContractError(
                process_code="same_run_alignment_missing",
                detail="alignment segment boundaries do not reconcile mapped rows",
            )


@dataclass(frozen=True)
class ContractSourceMetadata:
    """Typed source/alignment manifest inputs; no opaque pass-through metadata."""

    job_id: int
    pipeline_id: int
    model: str
    system: str
    backend: str
    backend_version: str
    nsight_kernel_classifier_version: str
    alignments: tuple[CohortAlignmentProvenance, ...]
    alignment_version: str = ALIGNMENT_VERSION
    marker_encoder_version: str = MARKER_ENCODER_VERSION
    semantic_key_version: str = SEMANTIC_KEY_VERSION
    query_reconstruction_version: str = QUERY_RECONSTRUCTION_VERSION
    critical_key_policy: str = CRITICAL_KEY_POLICY

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.job_id, self.pipeline_id)
        ):
            raise ContractError(
                process_code="invalid_measurement",
                detail="job_id and pipeline_id must be positive integers",
            )
        for field_name in (
            "model",
            "system",
            "backend",
            "backend_version",
            "nsight_kernel_classifier_version",
        ):
            if not getattr(self, field_name):
                raise ContractError(
                    process_code="schema_missing",
                    detail=f"source metadata {field_name} must be non-empty",
                )
        if self.alignment_version != ALIGNMENT_VERSION:
            raise ContractError(process_code="schema_incompatible", detail="alignment version mismatch")
        if self.marker_encoder_version != MARKER_ENCODER_VERSION:
            raise ContractError(process_code="schema_incompatible", detail="marker encoder version mismatch")
        if self.semantic_key_version != SEMANTIC_KEY_VERSION:
            raise ContractError(process_code="schema_incompatible", detail="semantic-key version mismatch")
        if self.query_reconstruction_version != QUERY_RECONSTRUCTION_VERSION:
            raise ContractError(process_code="schema_incompatible", detail="query reconstruction version mismatch")
        if self.critical_key_policy != CRITICAL_KEY_POLICY:
            raise ContractError(process_code="schema_incompatible", detail="critical-key policy mismatch")
        concurrencies = tuple(alignment.concurrency for alignment in self.alignments)
        if not concurrencies or concurrencies != tuple(sorted(set(concurrencies))):
            raise ContractError(
                process_code="concurrency_mismatch",
                detail="alignment cohorts must be non-empty, unique, and ordered",
            )


@dataclass(frozen=True)
class RankStepComposition:
    """Intact compute/communication/busy tuple for one captured rank key."""

    rank_key: str
    gpu_compute_ms: float | None
    gpu_comm_ms: float | None
    gpu_busy_ms: float
    unknown_kernel_count: int = 0
    unknown_kernel_duration_ms: float = 0.0
    unknown_kernel_name_hash: str = _EMPTY_UNKNOWN_KERNEL_HASH

    def __post_init__(self) -> None:
        if not self.rank_key.isdecimal() or not _is_nonnegative_number(self.gpu_busy_ms):
            raise ContractError(
                process_code="rank_identity_unavailable",
                detail="rank tuple requires a numeric key and non-negative busy time",
            )
        components = (self.gpu_compute_ms, self.gpu_comm_ms)
        if (components[0] is None) != (components[1] is None) or any(
            value is not None and not _is_nonnegative_number(value) for value in components
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="rank compute and communication must be jointly valid or null",
            )
        if (
            isinstance(self.unknown_kernel_count, bool)
            or not isinstance(self.unknown_kernel_count, int)
            or self.unknown_kernel_count < 0
            or not _is_nonnegative_number(self.unknown_kernel_duration_ms)
        ):
            raise ContractError(
                process_code="invalid_measurement",
                detail="rank unknown-kernel count/duration must be non-negative and typed",
            )
        if self.unknown_kernel_count:
            if (
                self.unknown_kernel_duration_ms <= 0.0
                or not _is_sha256(self.unknown_kernel_name_hash)
                or components != (None, None)
            ):
                raise ContractError(
                    process_code="rank_tuple_identity_mismatch",
                    detail="unknown rank kernels require auditable mass and null components",
                )
        elif (
            self.unknown_kernel_duration_ms != 0.0
            or self.unknown_kernel_name_hash != _EMPTY_UNKNOWN_KERNEL_HASH
            or components[0] is None
            or float(components[0]) + float(components[1]) + 1e-9 < self.gpu_busy_ms
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="complete rank components must cover same-rank busy time",
            )


@dataclass(frozen=True)
class ProfiledStepComposition:
    """Same-run Nsight tuple selected for one profiled FPM observation."""

    sample_id: str
    gpu_compute_ms: float | None
    gpu_comm_ms: float | None
    gpu_busy_ms: float
    rank_key: str
    rank_identity_kind: str
    rank_mapping_provenance: str
    captured_rank_keys: tuple[str, ...]
    busy_by_rank_key_ms: tuple[tuple[str, float], ...]
    rank_compositions: tuple[RankStepComposition, ...]
    alignment_mapping_hash: str
    kernel_classifier_version: str
    unknown_kernel_count: int = 0
    unknown_kernel_duration_ms: float = 0.0
    unknown_kernel_name_hash: str = _EMPTY_UNKNOWN_KERNEL_HASH

    def __post_init__(self) -> None:
        if not self.sample_id or not self.rank_key:
            raise ContractError(
                process_code="rank_identity_unavailable",
                detail="profiled composition sample_id and rank_key must be non-empty",
            )
        if self.rank_identity_kind not in {"logical_rank", "global_pid"} or not self.rank_mapping_provenance:
            raise ContractError(
                process_code="rank_identity_unavailable",
                detail="profiled composition rank identity/provenance is invalid",
            )
        _numeric_rank_keys(self.captured_rank_keys)
        busy_keys = tuple(key for key, _ in self.busy_by_rank_key_ms)
        _numeric_rank_keys(busy_keys)
        if busy_keys != self.captured_rank_keys:
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="busy tuple keys do not match captured rank keys",
            )
        rank_tuple_keys = tuple(item.rank_key for item in self.rank_compositions)
        if rank_tuple_keys != self.captured_rank_keys:
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="intact rank tuple keys do not match captured rank keys",
            )
        if any(
            not math.isclose(item.gpu_busy_ms, busy, rel_tol=0.0, abs_tol=1e-9)
            for item, (_key, busy) in zip(
                self.rank_compositions,
                self.busy_by_rank_key_ms,
                strict=True,
            )
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="intact rank busy values disagree with rank diagnostics",
            )
        if not _is_sha256(self.alignment_mapping_hash):
            raise ContractError(
                process_code="same_run_alignment_ambiguous",
                detail="composition alignment hash must be a canonical SHA-256",
            )
        if not self.kernel_classifier_version:
            raise ContractError(
                process_code="schema_missing",
                detail="kernel classifier version must be explicit",
            )
        if not _is_nonnegative_number(self.gpu_busy_ms):
            raise ContractError(
                process_code="invalid_measurement",
                detail="gpu_busy_ms must be finite and non-negative",
            )
        if any(not _is_nonnegative_number(value) for _, value in self.busy_by_rank_key_ms):
            raise ContractError(
                process_code="invalid_measurement",
                detail="per-rank busy values must be finite and non-negative",
            )
        selected_key, selected_busy = max(
            self.busy_by_rank_key_ms,
            key=lambda item: (item[1], -int(item[0])),
        )
        if self.rank_key != selected_key or not math.isclose(
            self.gpu_busy_ms,
            selected_busy,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="selected rank tuple violates max-busy/lower-numeric policy",
            )
        selected_tuple = next(item for item in self.rank_compositions if item.rank_key == selected_key)
        components = (self.gpu_compute_ms, self.gpu_comm_ms)
        if (components[0] is None) != (components[1] is None):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="gpu_compute_ms and gpu_comm_ms must be jointly available",
            )
        for value in components:
            if value is not None and not _is_nonnegative_number(value):
                raise ContractError(
                    process_code="invalid_measurement",
                    detail="GPU components must be finite and non-negative",
                )
        if (
            self.gpu_compute_ms is not None
            and self.gpu_comm_ms is not None
            and self.gpu_compute_ms + self.gpu_comm_ms + 1e-9 < self.gpu_busy_ms
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="gpu_compute_ms + gpu_comm_ms must cover gpu_busy_ms",
            )
        if (
            isinstance(self.unknown_kernel_count, bool)
            or not isinstance(self.unknown_kernel_count, int)
            or self.unknown_kernel_count < 0
            or not _is_nonnegative_number(self.unknown_kernel_duration_ms)
        ):
            raise ContractError(
                process_code="invalid_measurement",
                detail="unknown-kernel count/duration must be non-negative and finite",
            )
        rank_unknown_count = sum(item.unknown_kernel_count for item in self.rank_compositions)
        rank_unknown_duration = sum(item.unknown_kernel_duration_ms for item in self.rank_compositions)
        if self.unknown_kernel_count != rank_unknown_count or not math.isclose(
            self.unknown_kernel_duration_ms,
            rank_unknown_duration,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="aggregate unknown-kernel mass disagrees with intact rank tuples",
            )
        if self.unknown_kernel_count:
            if (
                self.unknown_kernel_duration_ms <= 0.0
                or not _is_sha256(self.unknown_kernel_name_hash)
                or components != (None, None)
            ):
                raise ContractError(
                    process_code="rank_tuple_identity_mismatch",
                    detail="unknown kernels require auditable mass and unavailable components",
                )
        elif (
            self.unknown_kernel_duration_ms != 0.0
            or self.unknown_kernel_name_hash != _EMPTY_UNKNOWN_KERNEL_HASH
            or components[0] is None
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="complete classification requires zero unknown mass and complete components",
            )
        elif not math.isclose(
            float(self.gpu_compute_ms),
            float(selected_tuple.gpu_compute_ms),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            float(self.gpu_comm_ms),
            float(selected_tuple.gpu_comm_ms),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail="emitted components do not come from the selected same-rank tuple",
            )

    @property
    def captured_rank_key_count(self) -> int:
        return len(self.captured_rank_keys)

    @property
    def busy_min_ms(self) -> float:
        return min(value for _, value in self.busy_by_rank_key_ms)

    @property
    def busy_max_ms(self) -> float:
        return max(value for _, value in self.busy_by_rank_key_ms)

    @property
    def imbalance_ratio(self) -> float | None:
        minimum = self.busy_min_ms
        return self.busy_max_ms / minimum if minimum > 0.0 else None


@dataclass(frozen=True)
class ContractBundle:
    """Paths and hashes of one atomically published source contract."""

    root: Path
    samples_csv: Path
    bins_csv: Path
    manifest_json: Path
    samples_sha256: str
    bins_sha256: str


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _csv_bytes(rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _stats(prefix: str, values: list[float]) -> dict[str, Any]:
    keys = ("count", "support_class", "band_kind", "median", "minimum", "q1", "q3", "maximum")
    if not values:
        return {f"{prefix}_{key}": None for key in keys}
    stats = descriptive_stats(values)
    band_kind = {
        "singleton": "point",
        "sparse": "min_max",
        "repeated": "iqr_and_min_max",
    }[str(stats["support_class"])]
    return {
        f"{prefix}_count": stats["count"],
        f"{prefix}_support_class": stats["support_class"],
        f"{prefix}_band_kind": band_kind,
        f"{prefix}_median": stats["median"],
        f"{prefix}_minimum": stats["minimum"],
        f"{prefix}_q1": stats["q1"],
        f"{prefix}_q3": stats["q3"],
        f"{prefix}_maximum": stats["maximum"],
    }


def _prediction_json(prediction: SemanticBinPrediction | None) -> str | None:
    return _canonical_json(asdict(prediction.record)) if prediction is not None else None


def _query_json(prediction: SemanticBinPrediction | None) -> str | None:
    return _canonical_json(asdict(prediction.query)) if prediction is not None else None


def _composition_complete(
    bin_: SemanticPopulationBin,
    compositions: Mapping[str, ProfiledStepComposition],
) -> bool:
    return bool(bin_.profiled_samples) and all(
        compositions[sample.sample_id].gpu_compute_ms is not None
        and compositions[sample.sample_id].gpu_comm_ms is not None
        for sample in bin_.profiled_samples
    )


def _eligibility_reason(
    bin_: SemanticPopulationBin,
    prediction: SemanticBinPrediction | None,
    composition_complete: bool,
) -> str | None:
    if not bin_.clean_samples:
        return "profiled_only_bin"
    if not bin_.profiled_samples:
        return "clean_only_bin"
    if prediction is None or prediction.record.total_ms is None:
        return "aic_total_unavailable"
    if prediction.record.component_sum_ms is None:
        return "aic_components_unavailable"
    if not composition_complete:
        return "nsight_components_unavailable"
    return None


def _shape_audit(prefix: str, samples: tuple[FpmSample, ...]) -> dict[str, Any]:
    axes = (
        "ctx_requests",
        "decode_requests",
        "ctx_new_tokens",
        "ctx_kv_tokens",
        "decode_kv_tokens",
    )
    result: dict[str, Any] = {f"{prefix}_raw_shape_cardinality": len({s.raw_shape for s in samples})}
    for index, axis in enumerate(axes):
        values = [sample.raw_shape[index] for sample in samples]
        result[f"{prefix}_{axis}_raw_min"] = min(values) if values else None
        result[f"{prefix}_{axis}_raw_max"] = max(values) if values else None
    deltas = {
        "ctx_new_tokens": [
            (sample.semantic_key[2] or 0) * sample.shape.ctx_requests - sample.shape.ctx_new_tokens
            for sample in samples
        ],
        "ctx_kv_tokens": [
            (sample.semantic_key[3] or 0) * sample.shape.ctx_requests - sample.shape.ctx_kv_tokens for sample in samples
        ],
        "decode_kv_tokens": [
            (sample.semantic_key[4] or 0) * sample.shape.decode_requests - sample.shape.decode_kv_tokens
            for sample in samples
        ],
    }
    for axis, values in deltas.items():
        result[f"{prefix}_{axis}_query_delta_min"] = min(values) if values else None
        result[f"{prefix}_{axis}_query_delta_max"] = max(values) if values else None
    return result


def _decomposition(
    *,
    clean_wall: float,
    profiled_wall: float,
    gpu_compute: float,
    gpu_comm: float,
    gpu_busy: float,
    aic_compute: float,
    aic_comm: float,
    aic_other: float,
    aic_total: float,
) -> dict[str, float]:
    gap = aic_total - clean_wall
    gpu_concurrency = gpu_compute + gpu_comm - gpu_busy
    profiled_overhead = profiled_wall - gpu_busy
    profile_wall_delta = profiled_wall - clean_wall
    terms = {
        "term_compute_err_ms": aic_compute - gpu_compute,
        "term_comm_err_ms": aic_comm - gpu_comm,
        "term_aic_other_ms": aic_other,
        "term_gpu_concurrency_ms": gpu_concurrency,
        "term_neg_profiled_overhead_ms": -profiled_overhead,
        "term_profile_wall_delta_ms": profile_wall_delta,
    }
    closure_error = sum(terms.values()) - gap
    tolerance = max(1e-6, 1e-6 * abs(gap))
    if abs(closure_error) > tolerance:
        raise ContractError(
            process_code="decomposition_closure_mismatch",
            detail=f"decomposition closure error {closure_error} exceeds {tolerance}",
        )
    return {
        "gpu_concurrency_ms": gpu_concurrency,
        "profiled_overhead_ms": profiled_overhead,
        "profile_wall_delta_ms": profile_wall_delta,
        **terms,
        "decomposition_closure_error_ms": closure_error,
    }


def _build_bin_rows(
    population: SemanticPopulation,
    predictions: tuple[SemanticBinPrediction, ...],
    compositions: Mapping[str, ProfiledStepComposition],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prediction_by_bin = {prediction.bin_id: prediction for prediction in predictions}
    expected_prediction_ids = {bin_.bin_id for bin_ in population.bins if bin_.clean_samples}
    if set(prediction_by_bin) != expected_prediction_ids or len(predictions) != len(expected_prediction_ids):
        raise ContractError(
            process_code="predictor_call_identity_mismatch",
            detail="prediction identities do not exactly match measured clean bins",
        )

    total_clean = sum(bin_.n_clean for bin_ in population.bins)
    total_profiled = sum(bin_.n_profiled for bin_ in population.bins)
    bin_status: dict[str, dict[str, Any]] = {}
    rows = []
    for bin_ in population.bins:
        prediction = prediction_by_bin.get(bin_.bin_id)
        record = prediction.record if prediction is not None else None
        clean_walls = [sample.wall_ms for sample in bin_.clean_samples]
        profiled_walls = [sample.wall_ms for sample in bin_.profiled_samples]
        bin_compositions = [compositions[sample.sample_id] for sample in bin_.profiled_samples]
        complete = _composition_complete(bin_, compositions)
        overall_gap_eligible = bool(record is not None and record.total_ms is not None)
        decomposition_eligible = bool(
            bin_.nsight_shared
            and overall_gap_eligible
            and record is not None
            and record.compute_ms is not None
            and record.communication_ms is not None
            and record.other_ms is not None
            and complete
        )
        reason = _eligibility_reason(bin_, prediction, complete)
        clean_stats = _stats("clean_wall_ms", clean_walls)
        profiled_stats = _stats("profiled_wall_ms", profiled_walls)
        compute_values = [
            float(composition.gpu_compute_ms)
            for composition in bin_compositions
            if composition.gpu_compute_ms is not None
        ]
        comm_values = [
            float(composition.gpu_comm_ms) for composition in bin_compositions if composition.gpu_comm_ms is not None
        ]
        busy_values = [composition.gpu_busy_ms for composition in bin_compositions]
        row: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "bin_id": bin_.bin_id,
            "configuration_fingerprint": bin_.configuration_fingerprint,
            "concurrency": bin_.concurrency,
            "phase": bin_.phase,
            "semantic_key": bin_.serialized_semantic_key,
            "ctx_requests": bin_.semantic_key[0],
            "decode_requests": bin_.semantic_key[1],
            "ctx_new_per_request": bin_.semantic_key[2],
            "ctx_kv_per_request": bin_.semantic_key[3],
            "decode_kv_per_request": bin_.semantic_key[4],
            "n_clean": bin_.n_clean,
            "n_profiled": bin_.n_profiled,
            "structural_status": "valid",
            "clean_row_mass_weight": bin_.n_clean / total_clean if total_clean else None,
            "profiled_row_mass_weight": bin_.n_profiled / total_profiled if total_profiled else None,
            "all_clean": bool(bin_.clean_samples),
            "overall_gap_eligible": overall_gap_eligible,
            "nsight_shared": bin_.nsight_shared,
            "decomposition_eligible": decomposition_eligible,
            "eligibility_reason": reason,
            "decomposition_support_count": min(bin_.n_clean, bin_.n_profiled),
            "decomposition_support_class": (
                support_class(min(bin_.n_clean, bin_.n_profiled)) if bin_.nsight_shared else None
            ),
            "semantic_query_json": _query_json(prediction),
            "aic_prediction_json": _prediction_json(prediction),
            "aic_status": record.status if record is not None else None,
            "aic_reason": record.reason if record is not None else None,
            "aic_total_ms": record.total_ms if record is not None else None,
            "aic_compute_ms": record.compute_ms if record is not None else None,
            "aic_comm_ms": record.communication_ms if record is not None else None,
            "aic_other_ms": record.other_ms if record is not None else None,
            **clean_stats,
            **profiled_stats,
            **_stats("gpu_compute_ms", compute_values),
            **_stats("gpu_comm_ms", comm_values),
            **_stats("gpu_busy_ms", busy_values),
            **_shape_audit("clean", bin_.clean_samples),
            **_shape_audit("profiled", bin_.profiled_samples),
        }
        clean_median = clean_stats["clean_wall_ms_median"]
        if overall_gap_eligible and clean_median is not None and record is not None:
            gap = record.total_ms - float(clean_median)  # type: ignore[operator]
            row.update(
                {
                    "overall_gap_signed_ms": gap,
                    "overall_gap_abs_ms": abs(gap),
                    "relative_error_signed": gap / float(clean_median),
                    "relative_error_abs": abs(gap / float(clean_median)),
                }
            )
        else:
            row.update(
                {
                    "overall_gap_signed_ms": None,
                    "overall_gap_abs_ms": None,
                    "relative_error_signed": None,
                    "relative_error_abs": None,
                }
            )
        if decomposition_eligible and record is not None:
            row.update(
                _decomposition(
                    clean_wall=float(clean_median),
                    profiled_wall=float(profiled_stats["profiled_wall_ms_median"]),
                    gpu_compute=float(row["gpu_compute_ms_median"]),
                    gpu_comm=float(row["gpu_comm_ms_median"]),
                    gpu_busy=float(row["gpu_busy_ms_median"]),
                    aic_compute=float(record.compute_ms),
                    aic_comm=float(record.communication_ms),
                    aic_other=float(record.other_ms),
                    aic_total=float(record.total_ms),
                )
            )
        else:
            row.update(
                dict.fromkeys(
                    (
                        "gpu_concurrency_ms",
                        "profiled_overhead_ms",
                        "profile_wall_delta_ms",
                        "term_compute_err_ms",
                        "term_comm_err_ms",
                        "term_aic_other_ms",
                        "term_gpu_concurrency_ms",
                        "term_neg_profiled_overhead_ms",
                        "term_profile_wall_delta_ms",
                        "decomposition_closure_error_ms",
                    ),
                    None,
                )
            )
        rows.append(row)
        bin_status[bin_.bin_id] = {
            "overall_gap_eligible": overall_gap_eligible,
            "nsight_shared": bin_.nsight_shared,
            "decomposition_eligible": decomposition_eligible,
            "aic_status": row["aic_status"],
            "eligibility_reason": reason,
        }
    return rows, bin_status


def _sample_rows(
    population: SemanticPopulation,
    compositions: Mapping[str, ProfiledStepComposition],
    bin_status: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    bin_by_identity = {(bin_.concurrency, bin_.phase, bin_.semantic_key): bin_ for bin_ in population.bins}
    rows = []
    for sample in population.samples:
        bin_ = bin_by_identity.get((sample.concurrency, sample.phase, sample.semantic_key))
        status = bin_status.get(bin_.bin_id, {}) if bin_ is not None else {}
        composition = compositions.get(sample.sample_id)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "configuration_fingerprint": population.configuration_fingerprint,
                "concurrency": sample.concurrency,
                "phase": sample.phase,
                "lane": sample.lane,
                "sample_id": sample.sample_id,
                "workload_segment": sample.workload_segment,
                "counter_id": sample.counter_id,
                "worker_id": sample.worker_id,
                "dp_rank": sample.dp_rank,
                "ctx_requests": sample.shape.ctx_requests,
                "decode_requests": sample.shape.decode_requests,
                "ctx_new_tokens": sample.shape.ctx_new_tokens,
                "ctx_kv_tokens": sample.shape.ctx_kv_tokens,
                "decode_kv_tokens": sample.shape.decode_kv_tokens,
                "ctx_new_per_request": sample.semantic_key[2],
                "ctx_kv_per_request": sample.semantic_key[3],
                "decode_kv_per_request": sample.semantic_key[4],
                "semantic_key": sample.serialized_semantic_key,
                "bin_id": bin_.bin_id if bin_ is not None else None,
                "wall_ms": sample.wall_ms,
                "gpu_compute_ms": composition.gpu_compute_ms if composition else None,
                "gpu_comm_ms": composition.gpu_comm_ms if composition else None,
                "gpu_busy_ms": composition.gpu_busy_ms if composition else None,
                "rank_key": composition.rank_key if composition else None,
                "rank_identity_kind": composition.rank_identity_kind if composition else None,
                "rank_mapping_provenance": composition.rank_mapping_provenance if composition else None,
                "captured_rank_key_count": composition.captured_rank_key_count if composition else None,
                "captured_rank_keys_json": (_canonical_json(composition.captured_rank_keys) if composition else None),
                "rank_compositions_json": (
                    _canonical_json([asdict(item) for item in composition.rank_compositions]) if composition else None
                ),
                "rank_busy_min_ms": composition.busy_min_ms if composition else None,
                "rank_busy_max_ms": composition.busy_max_ms if composition else None,
                "imbalance_ratio": composition.imbalance_ratio if composition else None,
                "alignment_mapping_hash": composition.alignment_mapping_hash if composition else None,
                "kernel_classifier_version": composition.kernel_classifier_version if composition else None,
                "unknown_kernel_count": composition.unknown_kernel_count if composition else None,
                "unknown_kernel_duration_ms": (composition.unknown_kernel_duration_ms if composition else None),
                "unknown_kernel_name_hash": composition.unknown_kernel_name_hash if composition else None,
                "same_run_mapping_status": "mapped" if composition else None,
                "analytic_eligible": sample.analytic_eligible,
                "sample_eligibility_reason": sample.eligibility_reason,
                "semantic_support_status": (
                    "shared"
                    if bin_ is not None and bin_.nsight_shared
                    else "clean_only"
                    if bin_ is not None and bin_.clean_only
                    else "profiled_only"
                    if bin_ is not None and bin_.profiled_only
                    else None
                ),
                "overall_gap_eligible": status.get("overall_gap_eligible"),
                "decomposition_eligible": status.get("decomposition_eligible"),
                "aic_status": status.get("aic_status"),
                "eligibility_reason": sample.eligibility_reason or status.get("eligibility_reason"),
            }
        )
    return rows


def _validate_compositions(
    population: SemanticPopulation,
    compositions: Mapping[str, ProfiledStepComposition],
    *,
    source_metadata: ContractSourceMetadata,
    expected_rank_key_count: int,
) -> None:
    required_ids = {sample.sample_id for sample in population.samples if sample.lane == "profiled" and sample.measured}
    actual_ids = set(compositions)
    if actual_ids != required_ids:
        missing = sorted(required_ids - actual_ids)
        extra = sorted(actual_ids - required_ids)
        raise ContractError(
            process_code="same_run_alignment_missing" if missing else "duplicate_identity",
            detail=f"profiled composition identity mismatch: missing={missing[:5]!r}, extra={extra[:5]!r}",
        )
    samples_by_id = {sample.sample_id: sample for sample in population.samples}
    alignments = {alignment.concurrency: alignment for alignment in source_metadata.alignments}
    population_concurrencies = {sample.concurrency for sample in population.samples if sample.measured}
    if set(alignments) != population_concurrencies:
        raise ContractError(
            process_code="concurrency_mismatch",
            detail="alignment cohorts do not match measured population concurrencies",
        )
    for concurrency, alignment in alignments.items():
        input_rows = sum(
            sample.lane == "profiled" and sample.measured and sample.concurrency == concurrency
            for sample in population.samples
        )
        mapped_rows = sum(samples_by_id[sample_id].concurrency == concurrency for sample_id in compositions)
        if alignment.input_profiled_rows != input_rows or alignment.mapped_profiled_rows != mapped_rows:
            raise ContractError(
                process_code="same_run_alignment_missing",
                detail=f"alignment counts disagree for concurrency {concurrency}",
            )
        if len(alignment.expected_rank_keys) != expected_rank_key_count:
            raise ContractError(
                process_code="incomplete_rank_capture",
                detail=f"concurrency {concurrency} expected-rank count disagrees with runtime parity",
            )

    for sample_id, composition in compositions.items():
        if composition.sample_id != sample_id:
            raise ContractError(
                process_code="duplicate_identity",
                detail=f"composition key {sample_id!r} disagrees with payload {composition.sample_id!r}",
            )
        sample = samples_by_id[sample_id]
        alignment = alignments[sample.concurrency]
        if composition.captured_rank_keys != alignment.captured_rank_keys:
            raise ContractError(
                process_code="incomplete_rank_capture",
                detail=(f"composition {sample_id!r} captured rank keys disagree with its cohort"),
            )
        if (
            composition.rank_identity_kind != alignment.rank_identity_kind
            or composition.rank_mapping_provenance != alignment.rank_mapping_provenance
            or composition.alignment_mapping_hash != alignment.mapping_hash
        ):
            raise ContractError(
                process_code="rank_tuple_identity_mismatch",
                detail=f"composition {sample_id!r} rank/alignment provenance disagrees with its cohort",
            )
        if composition.kernel_classifier_version != source_metadata.nsight_kernel_classifier_version:
            raise ContractError(
                process_code="schema_incompatible",
                detail=f"composition {sample_id!r} kernel classifier version mismatch",
            )


def _coverage_rows(bin_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    coverage: dict[str, dict[str, int]] = {}
    for row in bin_rows:
        key = f"c{row['concurrency']}/{row['phase']}"
        bucket = coverage.setdefault(
            key,
            {
                "clean_bins": 0,
                "profiled_bins": 0,
                "overall_gap_eligible_bins": 0,
                "shared_bins": 0,
                "decomposition_eligible_bins": 0,
                "clean_rows": 0,
                "profiled_rows": 0,
                "overall_gap_eligible_clean_rows": 0,
                "shared_clean_rows": 0,
                "shared_profiled_rows": 0,
                "decomposition_eligible_clean_rows": 0,
                "decomposition_eligible_profiled_rows": 0,
            },
        )
        bucket["clean_bins"] += int(row["n_clean"] > 0)
        bucket["profiled_bins"] += int(row["n_profiled"] > 0)
        bucket["overall_gap_eligible_bins"] += int(row["overall_gap_eligible"])
        bucket["shared_bins"] += int(row["nsight_shared"])
        bucket["decomposition_eligible_bins"] += int(row["decomposition_eligible"])
        bucket["clean_rows"] += int(row["n_clean"])
        bucket["profiled_rows"] += int(row["n_profiled"])
        if row["overall_gap_eligible"]:
            bucket["overall_gap_eligible_clean_rows"] += int(row["n_clean"])
        if row["nsight_shared"]:
            bucket["shared_clean_rows"] += int(row["n_clean"])
            bucket["shared_profiled_rows"] += int(row["n_profiled"])
        if row["decomposition_eligible"]:
            bucket["decomposition_eligible_clean_rows"] += int(row["n_clean"])
            bucket["decomposition_eligible_profiled_rows"] += int(row["n_profiled"])
    return coverage


def _shape_audit_summary(bin_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for lane in ("clean", "profiled"):
        summary[f"{lane}_collision_bins"] = sum(int(row[f"{lane}_raw_shape_cardinality"] > 1) for row in bin_rows)
        for axis in ("ctx_new_tokens", "ctx_kv_tokens", "decode_kv_tokens"):
            values = [
                value
                for row in bin_rows
                for value in (
                    row[f"{lane}_{axis}_query_delta_min"],
                    row[f"{lane}_{axis}_query_delta_max"],
                )
                if value is not None
            ]
            summary[f"{lane}_{axis}_query_delta_min"] = min(values) if values else None
            summary[f"{lane}_{axis}_query_delta_max"] = max(values) if values else None
    return summary


def _predictor_provenance(predictions: tuple[SemanticBinPrediction, ...]) -> dict[str, Any]:
    unclassified_names = set()
    unclassified_signed_ms = 0.0
    provenance_records = set()
    for prediction in predictions:
        record = prediction.record
        component_by_name = dict(record.operation_inventory)
        for name, value, _source in record.operation_values:
            if component_by_name[name] == "unclassified":
                unclassified_names.add(name)
                unclassified_signed_ms += value
        provenance_records.add(record.configuration_provenance)
    return {
        "predictor_versions": sorted({item.record.predictor_version for item in predictions}),
        "api_versions": sorted({item.record.api_version for item in predictions}),
        "lookup_policies": sorted({item.record.lookup_policy for item in predictions}),
        "component_classifier_versions": sorted({item.record.component_classifier_version for item in predictions}),
        "total_bases": sorted({item.record.total_basis for item in predictions if item.record.total_basis is not None}),
        "operation_inventory_hashes": sorted({item.record.operation_inventory_hash for item in predictions}),
        "configuration_provenance": [json.loads(payload) for payload in sorted(provenance_records)],
        "configuration_provenance_hashes": [
            _sha256_bytes(payload.encode("utf-8")) for payload in sorted(provenance_records)
        ],
        "unknown_operation_names_sha256": _sha256_bytes(_canonical_json(sorted(unclassified_names)).encode("utf-8")),
        "unknown_operation_name_count": len(unclassified_names),
        "unknown_operation_signed_ms": unclassified_signed_ms,
    }


def _orthogonal_status_counts(bin_rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "clean_only_bin": sum(row["n_clean"] > 0 and row["n_profiled"] == 0 for row in bin_rows),
        "profiled_only_bin": sum(row["n_profiled"] > 0 and row["n_clean"] == 0 for row in bin_rows),
        "aic_total_unavailable": sum(row["n_clean"] > 0 and not row["overall_gap_eligible"] for row in bin_rows),
        "aic_components_unavailable": sum(
            row["overall_gap_eligible"] and (row["aic_compute_ms"] is None or row["aic_comm_ms"] is None)
            for row in bin_rows
        ),
        "nsight_components_unavailable": sum(
            row["nsight_shared"] and row["gpu_compute_ms_count"] != row["n_profiled"] for row in bin_rows
        ),
    }


def _write_atomic_bundle(
    output_dir: Path,
    *,
    samples_payload: bytes,
    bins_payload: bytes,
    manifest_payload: bytes,
) -> ContractBundle:
    if output_dir.exists():
        raise FileExistsError(f"semantic contract output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        (temporary / "samples.csv").write_bytes(samples_payload)
        (temporary / "bins.csv").write_bytes(bins_payload)
        (temporary / "manifest.json").write_bytes(manifest_payload)
        _rename_noreplace(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return ContractBundle(
        root=output_dir,
        samples_csv=output_dir / "samples.csv",
        bins_csv=output_dir / "bins.csv",
        manifest_json=output_dir / "manifest.json",
        samples_sha256=_sha256_bytes(samples_payload),
        bins_sha256=_sha256_bytes(bins_payload),
    )


def _rename_noreplace(source: Path, target: Path) -> None:
    """Atomically publish a directory without replacing a concurrent target."""

    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is required")
    result = renameat2(
        ctypes.c_int(-100),
        ctypes.c_char_p(os.fsencode(source)),
        ctypes.c_int(-100),
        ctypes.c_char_p(os.fsencode(target)),
        ctypes.c_uint(1),
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, f"semantic contract output already exists: {target}", target)
    raise OSError(error, os.strerror(error), target)


def reduce_and_write_semantic_contract(
    *,
    population: SemanticPopulation,
    repository_config: RepositoryAicConfig,
    compositions: Mapping[str, ProfiledStepComposition],
    output_dir: str | Path,
    aiconfigurator_commit: str,
    auto_collector_commit: str,
    source_metadata: ContractSourceMetadata,
    required_lookup_policy: str = LOOKUP_POLICY_VERSION,
) -> ContractBundle:
    """Run the Stage-1 predictor/reducer once and atomically publish its contract."""

    if population.schema_version != SCHEMA_VERSION:
        raise ContractError(
            process_code="schema_incompatible",
            detail=f"population schema {population.schema_version!r} is not {SCHEMA_VERSION!r}",
        )
    if population.configuration_fingerprint != repository_config.configuration_fingerprint:
        raise ContractError(
            process_code="configuration_mismatch",
            detail="population and predictor configuration fingerprints differ",
        )
    if (
        aiconfigurator_commit != repository_config.repo_commit
        or not _is_git_sha(aiconfigurator_commit)
        or not _is_git_sha(auto_collector_commit)
    ):
        raise ContractError(
            process_code="repository_commit_mismatch",
            detail="repository commits must be exact 40-character SHAs and AIC identities must agree",
        )
    parity = repository_config.parity
    source_identity = (
        source_metadata.model,
        source_metadata.system,
        source_metadata.backend,
        source_metadata.backend_version,
    )
    parity_identity = (
        parity["model"],
        parity["system"],
        parity["backend"],
        parity["backend_version"],
    )
    if source_identity != parity_identity:
        raise ContractError(
            process_code="configuration_mismatch",
            detail="source metadata model/system/backend identity disagrees with parity record",
        )
    _validate_compositions(
        population,
        compositions,
        source_metadata=source_metadata,
        expected_rank_key_count=int(repository_config.parity["gpu_count"]),
    )

    predictor = build_repository_predictor(repository_config)
    predictions = predict_clean_bins(
        population,
        predictor,
        required_lookup_policy=required_lookup_policy,
    )
    bin_rows, bin_status = _build_bin_rows(population, predictions, compositions)
    sample_rows = _sample_rows(population, compositions, bin_status)
    sample_fields = tuple(sample_rows[0]) if sample_rows else ()
    bin_fields = tuple(bin_rows[0]) if bin_rows else ()
    if not sample_fields or not bin_fields:
        raise ContractError(
            process_code="schema_missing",
            detail="semantic contract requires at least one source sample and one analytic bin",
        )
    samples_payload = _csv_bytes(sample_rows, sample_fields)
    bins_payload = _csv_bytes(bin_rows, bin_fields)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "writer_version": CONTRACT_WRITER_VERSION,
        "aiconfigurator_commit": aiconfigurator_commit,
        "auto_collector_commit": auto_collector_commit,
        "configuration_fingerprint": population.configuration_fingerprint,
        "parity_record": repository_config.parity,
        "measured_segments": sorted(population.measured_segments),
        "concurrencies": sorted({bin_.concurrency for bin_ in population.bins}),
        "source_metadata": asdict(source_metadata),
        "lookup_policy": required_lookup_policy,
        "counts": {
            "source_rows": len(population.samples),
            "bins": len(population.bins),
            "profiled_pre_alignment_rows": sum(
                alignment.input_profiled_rows for alignment in source_metadata.alignments
            ),
            "profiled_mapped_rows": sum(alignment.mapped_profiled_rows for alignment in source_metadata.alignments),
            "predictor_calls": len(predictions),
            "overall_gap_eligible_bins": sum(row["overall_gap_eligible"] for row in bin_rows),
            "shared_bins": sum(row["nsight_shared"] for row in bin_rows),
            "decomposition_eligible_bins": sum(row["decomposition_eligible"] for row in bin_rows),
            "unsupported_by_reason": {
                reason: sum(row["eligibility_reason"] == reason for row in bin_rows)
                for reason in sorted({row["eligibility_reason"] for row in bin_rows if row["eligibility_reason"]})
            },
            "orthogonal_status_counts": _orthogonal_status_counts(bin_rows),
            "excluded_source_rows_by_reason": dict(sorted(population.excluded_reason_counts.items())),
        },
        "coverage_by_concurrency_phase": _coverage_rows(bin_rows),
        "shape_audit": _shape_audit_summary(bin_rows),
        "alignment": {f"c{alignment.concurrency}": asdict(alignment) for alignment in source_metadata.alignments},
        "predictor_provenance": _predictor_provenance(predictions),
        "nsight_provenance": {
            "kernel_classifier_version": source_metadata.nsight_kernel_classifier_version,
            "critical_key_policy": source_metadata.critical_key_policy,
            "unknown_kernel_count": sum(composition.unknown_kernel_count for composition in compositions.values()),
            "unknown_kernel_duration_ms": sum(
                composition.unknown_kernel_duration_ms for composition in compositions.values()
            ),
            "unknown_kernel_name_hashes_sha256": _sha256_bytes(
                _canonical_json(
                    sorted(
                        {
                            composition.unknown_kernel_name_hash
                            for composition in compositions.values()
                            if composition.unknown_kernel_count
                        }
                    )
                ).encode("utf-8")
            ),
            "rank_identity_kinds": sorted({alignment.rank_identity_kind for alignment in source_metadata.alignments}),
            "rank_mapping_provenance": sorted(
                {alignment.rank_mapping_provenance for alignment in source_metadata.alignments}
            ),
        },
        "classifiers": {
            "aic_component": sorted({prediction.record.component_classifier_version for prediction in predictions}),
            "nsight_kernel": [source_metadata.nsight_kernel_classifier_version],
        },
        "files": {
            "samples.csv": {
                "rows": len(sample_rows),
                "sha256": _sha256_bytes(samples_payload),
            },
            "bins.csv": {
                "rows": len(bin_rows),
                "sha256": _sha256_bytes(bins_payload),
            },
        },
        "validation": {"status": "passed", "failure_reasons": []},
    }
    manifest_payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _write_atomic_bundle(
        Path(output_dir),
        samples_payload=samples_payload,
        bins_payload=bins_payload,
        manifest_payload=manifest_payload,
    )
