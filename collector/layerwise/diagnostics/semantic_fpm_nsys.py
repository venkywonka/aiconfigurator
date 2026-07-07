#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-run Nsight alignment and per-rank GPU composition for semantic FPM."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from collector.layerwise.common.parse_nsys_step_sweep import (
    _GLOBAL_PID_MASK,
    _extract_bench_step,
)
from collector.layerwise.diagnostics.semantic_fpm_contract import (
    AlignmentMarkerIdentity,
    AlignmentSegmentProvenance,
    CohortAlignmentProvenance,
    ProfiledStepComposition,
    RankStepComposition,
)
from collector.layerwise.diagnostics.semantic_fpm_insights import (
    AlignmentError,
    AlignmentResult,
    MarkerObservation,
    ProfiledFpmObservation,
    align_profiled_fpm_to_nsys,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import ContractError, FpmSample

KERNEL_CLASSIFIER_VERSION = "nsys-qwen3-vllm-exact-registry-v1"
RANK_MAPPING_PROVENANCE = "nsys.globalTid-high40-globalPid-v1"

# Registry entries are either exact exported short names or anchored namespace
# grammars.  Full-match semantics are deliberate: an incidental substring such
# as ``myncclDevKernel`` never becomes communication by accident.
_COMMUNICATION_EXACT = frozenset(
    {
        "allreduce_fusion_kernel_oneshot_lamport",
        "multimem_all_reduce_kernel",
    }
)
_COMMUNICATION_FULLMATCH = tuple(
    re.compile(pattern)
    for pattern in (r"ncclDevKernel_(?:AllGather|AllReduce|Broadcast|ReduceScatter|SendRecv)_[A-Za-z0-9_]+",)
)
_COMPUTE_EXACT = frozenset(
    {
        "CatArrayBatchedCopy_vectorized",
        "RadixTopKMaskLogitsKernel_MultiCTA",
        "TopPSamplingFromProbKernel",
        "_compute_slot_mapping_kernel",
        "_scatter_gather_elementwise_kernel",
        "cunn_SoftMaxForward",
        "device_kernel",
        "distribution_elementwise_grid_stride_kernel",
        "elementwise_kernel",
        "elementwise_kernel_with_index",
        "index_elementwise_kernel",
        "prepare_varlen_num_blocks_kernel",
        "reduce_kernel",
        "reshape_and_cache_flash_kernel",
        "splitKreduce_kernel",
        "unrolled_elementwise_kernel",
        "vectorized_elementwise_kernel",
        "vectorized_gather_kernel",
    }
)
_COMPUTE_FULLMATCH = tuple(
    re.compile(pattern)
    for pattern in (
        r"nvjet_tst_[A-Za-z0-9_.]+",
        r"triton_[A-Za-z0-9_.]*",
    )
)


@dataclass(frozen=True)
class AttributedKernel:
    """One positive-duration kernel attached to a canonical marker and rank key."""

    rank_key: str
    marker_canonical_index: int
    start_ns: int
    end_ns: int
    name: str

    def __post_init__(self) -> None:
        if not self.rank_key.isdecimal():
            raise ContractError(
                process_code="rank_identity_unavailable",
                detail=f"Nsight rank key must be numeric, got {self.rank_key!r}",
            )
        for field_name in ("marker_canonical_index", "start_ns", "end_ns"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ContractError(
                    process_code="invalid_measurement",
                    detail=f"kernel {field_name} must be a non-negative integer",
                )
        if self.end_ns <= self.start_ns:
            raise ContractError(
                process_code="invalid_measurement",
                detail="attributed kernels must have positive duration",
            )


def classify_kernel_name(name: str) -> str | None:
    """Classify one exact Nsight short name with the pinned registry."""

    if name in _COMMUNICATION_EXACT or any(pattern.fullmatch(name) for pattern in _COMMUNICATION_FULLMATCH):
        return "communication"
    if name in _COMPUTE_EXACT or any(pattern.fullmatch(name) for pattern in _COMPUTE_FULLMATCH):
        return "compute"
    return None


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _unknown_name_hash(names: Iterable[str]) -> str:
    unique = sorted(set(names))
    payload = _canonical_json(unique).encode("utf-8") if unique else b""
    return hashlib.sha256(payload).hexdigest()


def _union_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def load_nsys_markers(sqlite_path: str | Path) -> list[MarkerObservation]:
    """Load bench-step markers keyed by the trace's numeric globalPid."""

    source = Path(sqlite_path)
    if not source.is_file() or source.stat().st_size <= 0:
        raise ContractError(
            process_code="schema_missing",
            detail=f"Nsight SQLite is missing or empty: {source}",
        )
    connection = sqlite3.connect(f"file:{source}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute(
            "SELECT text,start,end,globalTid FROM NVTX_EVENTS WHERE text LIKE 'bench_step::%' ORDER BY globalTid,start"
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise ContractError(
            process_code="schema_incompatible",
            detail=f"cannot load bench_step markers from {source}: {exc}",
        ) from exc
    finally:
        connection.close()

    markers = []
    for text, start, end, global_tid in rows:
        parsed = _extract_bench_step(str(text))
        if parsed is None:
            raise ContractError(
                process_code="schema_incompatible",
                detail=f"unexpected bench_step marker label: {text!r}",
            )
        marker_step, decode_batch, mean_decode_kv, measure_run = parsed
        markers.append(
            MarkerObservation(
                rank_key=str(int(global_tid) & _GLOBAL_PID_MASK),
                marker_step=marker_step,
                measure_run=measure_run,
                decode_batch=decode_batch,
                mean_decode_kv=mean_decode_kv,
                start_ns=int(start),
                end_ns=int(end),
            )
        )
    if not markers:
        raise ContractError(
            process_code="same_run_alignment_missing",
            detail=f"Nsight SQLite contains no bench_step markers: {source}",
        )
    return markers


def captured_rank_keys(
    markers: list[MarkerObservation],
    *,
    expected_rank_count: int,
) -> tuple[str, ...]:
    """Validate the observed globalPid set against independent runtime parity."""

    if isinstance(expected_rank_count, bool) or not isinstance(expected_rank_count, int) or expected_rank_count <= 0:
        raise ContractError(
            process_code="rank_identity_unavailable",
            detail="expected rank count must be an independently supplied positive integer",
        )
    rank_keys = tuple(sorted({marker.rank_key for marker in markers}, key=int))
    if len(rank_keys) != expected_rank_count:
        raise ContractError(
            process_code="incomplete_rank_capture",
            detail=(
                f"runtime parity requires {expected_rank_count} ranks, but Nsight captured "
                f"{len(rank_keys)} keys: {rank_keys!r}"
            ),
        )
    return rank_keys


def _canonical_windows_by_rank(
    markers: list[MarkerObservation],
    rank_keys: tuple[str, ...],
) -> dict[str, tuple[MarkerObservation, ...]]:
    streams: dict[str, list[MarkerObservation]] = defaultdict(list)
    for marker in markers:
        streams[marker.rank_key].append(marker)
    for rank_key in rank_keys:
        streams[rank_key].sort(
            key=lambda marker: (
                marker.start_ns,
                marker.measure_run,
                marker.marker_step,
                marker.decode_batch,
                marker.mean_decode_kv,
                marker.end_ns,
            )
        )
    lengths = {len(streams[rank_key]) for rank_key in rank_keys}
    if len(lengths) != 1:
        raise AlignmentError(
            process_code="incomplete_rank_capture",
            detail="rank marker counts disagree while reconstructing canonical windows",
        )

    retained_raw_indices = []
    previous_step_by_run: dict[int, int] = {}
    for raw_index, marker_group in enumerate(zip(*(streams[key] for key in rank_keys), strict=True)):
        identities = {
            (
                marker.marker_step,
                marker.measure_run,
                marker.decode_batch,
                marker.mean_decode_kv,
            )
            for marker in marker_group
        }
        if len(identities) != 1:
            raise AlignmentError(
                process_code="rank_tuple_identity_mismatch",
                detail=f"rank marker identities disagree at raw index {raw_index}",
            )
        marker = marker_group[0]
        previous = previous_step_by_run.get(marker.measure_run)
        if previous is not None and marker.marker_step <= previous:
            continue
        retained_raw_indices.append(raw_index)
        previous_step_by_run[marker.measure_run] = marker.marker_step

    return {rank_key: tuple(streams[rank_key][index] for index in retained_raw_indices) for rank_key in rank_keys}


def _mapped_window_indexes(
    windows: dict[str, tuple[MarkerObservation, ...]],
) -> dict[str, tuple[list[int], tuple[MarkerObservation, ...]]]:
    return {rank_key: ([marker.start_ns for marker in stream], stream) for rank_key, stream in windows.items()}


def _window_for_timestamp(
    indexes: dict[str, tuple[list[int], tuple[MarkerObservation, ...]]],
    rank_key: str,
    timestamp_ns: int,
) -> int | None:
    rank_index = indexes.get(rank_key)
    if rank_index is None:
        return None
    starts, windows = rank_index
    offset = bisect.bisect_right(starts, timestamp_ns) - 1
    if offset < 0:
        return None
    marker = windows[offset]
    return offset if marker.start_ns <= timestamp_ns < marker.end_ns else None


def _profiled_runtime_launches(
    cursor: sqlite3.Cursor,
    indexes: dict[str, tuple[list[int], tuple[MarkerObservation, ...]]],
) -> dict[tuple[int, int], int]:
    """Index only runtime launches that fall inside a canonical marker window."""

    launches = {}
    cursor.execute(
        "SELECT correlationId,globalTid,start FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId IS NOT NULL"
    )
    for correlation_id, global_tid, runtime_start in cursor:
        global_pid = int(global_tid) & _GLOBAL_PID_MASK
        marker_index = _window_for_timestamp(indexes, str(global_pid), int(runtime_start))
        if marker_index is None:
            continue
        identity = (global_pid, int(correlation_id))
        previous = launches.setdefault(identity, marker_index)
        if previous != marker_index:
            raise ContractError(
                process_code="duplicate_identity",
                detail=(
                    "one runtime correlation identity maps to multiple canonical "
                    f"markers: {identity!r} -> {previous}, {marker_index}"
                ),
            )
    return launches


def load_attributed_kernels(
    sqlite_path: str | Path,
    *,
    windows: dict[str, tuple[MarkerObservation, ...]],
) -> tuple[AttributedKernel, ...]:
    """Attach deduplicated CUPTI kernels to canonical marker windows."""

    source = Path(sqlite_path)
    indexes = _mapped_window_indexes(windows)
    connection = sqlite3.connect(f"file:{source}?mode=ro&immutable=1", uri=True)
    try:
        cursor = connection.cursor()
        string_ids = dict(cursor.execute("SELECT id,value FROM StringIds").fetchall())
        launches = _profiled_runtime_launches(cursor, indexes)
        attributed = []
        seen = set()
        cursor.execute(
            "SELECT correlationId,graphNodeId,start,end,shortName,globalPid "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId IS NOT NULL"
        )
        for correlation_id, graph_node_id, start, end, short_name_id, global_pid in cursor:
            global_pid = int(global_pid)
            marker_index = launches.get((global_pid, int(correlation_id)))
            if marker_index is None:
                continue
            identity = (global_pid, int(correlation_id), graph_node_id)
            if identity in seen:
                continue
            seen.add(identity)
            start_ns, end_ns = int(start), int(end)
            if end_ns <= start_ns:
                continue
            attributed.append(
                AttributedKernel(
                    rank_key=str(global_pid),
                    marker_canonical_index=marker_index,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    name=str(string_ids.get(short_name_id, "")),
                )
            )
    except sqlite3.DatabaseError as exc:
        raise ContractError(
            process_code="schema_incompatible",
            detail=f"cannot load CUPTI kernels from {source}: {exc}",
        ) from exc
    finally:
        connection.close()
    return tuple(attributed)


def compose_profiled_steps(
    *,
    samples: tuple[FpmSample, ...],
    alignment: AlignmentResult,
    rank_keys: tuple[str, ...],
    kernels: Iterable[AttributedKernel],
) -> dict[str, ProfiledStepComposition]:
    """Reduce aligned kernels without ever mixing per-rank component tuples."""

    ordered_samples = tuple(sorted(samples, key=lambda sample: sample.counter_id))
    if len(ordered_samples) != len(alignment.mapping):
        raise ContractError(
            process_code="same_run_alignment_missing",
            detail="profiled sample count does not match alignment mapping",
        )
    sample_by_marker = {
        mapped.marker_canonical_index: ordered_samples[mapped.fpm_sequence_index] for mapped in alignment.mapping
    }
    if len(sample_by_marker) != len(ordered_samples):
        raise ContractError(
            process_code="duplicate_identity",
            detail="alignment mapping reuses a canonical marker",
        )
    ordered_rank_keys = tuple(sorted(rank_keys, key=int))
    grouped: dict[str, dict[str, list[AttributedKernel]]] = {
        sample.sample_id: {rank_key: [] for rank_key in ordered_rank_keys} for sample in ordered_samples
    }
    for kernel in kernels:
        if kernel.rank_key not in ordered_rank_keys:
            raise ContractError(
                process_code="incomplete_rank_capture",
                detail=f"kernel rank {kernel.rank_key!r} is outside the captured rank contract",
            )
        sample = sample_by_marker.get(kernel.marker_canonical_index)
        if sample is not None:
            grouped[sample.sample_id][kernel.rank_key].append(kernel)

    compositions = {}
    for sample in ordered_samples:
        by_rank = grouped[sample.sample_id]
        busy_by_rank_ns = {
            rank_key: _union_duration_ns((kernel.start_ns, kernel.end_ns) for kernel in rank_kernels)
            for rank_key, rank_kernels in by_rank.items()
        }
        selected_rank = max(
            ordered_rank_keys,
            key=lambda rank_key: (busy_by_rank_ns[rank_key], -int(rank_key)),
        )
        unknown = [
            kernel
            for rank_kernels in by_rank.values()
            for kernel in rank_kernels
            if classify_kernel_name(kernel.name) is None
        ]
        rank_compositions = []
        for rank_key in ordered_rank_keys:
            rank_kernels = by_rank[rank_key]
            rank_unknown = [kernel for kernel in rank_kernels if classify_kernel_name(kernel.name) is None]
            if rank_unknown:
                rank_compute_ms = rank_comm_ms = None
            else:
                rank_compute_ms = (
                    sum(
                        kernel.end_ns - kernel.start_ns
                        for kernel in rank_kernels
                        if classify_kernel_name(kernel.name) == "compute"
                    )
                    / 1e6
                )
                rank_comm_ms = (
                    sum(
                        kernel.end_ns - kernel.start_ns
                        for kernel in rank_kernels
                        if classify_kernel_name(kernel.name) == "communication"
                    )
                    / 1e6
                )
            rank_compositions.append(
                RankStepComposition(
                    rank_key=rank_key,
                    gpu_compute_ms=rank_compute_ms,
                    gpu_comm_ms=rank_comm_ms,
                    gpu_busy_ms=busy_by_rank_ns[rank_key] / 1e6,
                    unknown_kernel_count=len(rank_unknown),
                    unknown_kernel_duration_ms=sum(kernel.end_ns - kernel.start_ns for kernel in rank_unknown) / 1e6,
                    unknown_kernel_name_hash=_unknown_name_hash(kernel.name for kernel in rank_unknown),
                )
            )
        if unknown:
            compute_ms = comm_ms = None
        else:
            selected_tuple = next(item for item in rank_compositions if item.rank_key == selected_rank)
            compute_ms = selected_tuple.gpu_compute_ms
            comm_ms = selected_tuple.gpu_comm_ms
        compositions[sample.sample_id] = ProfiledStepComposition(
            sample_id=sample.sample_id,
            gpu_compute_ms=compute_ms,
            gpu_comm_ms=comm_ms,
            gpu_busy_ms=busy_by_rank_ns[selected_rank] / 1e6,
            rank_key=selected_rank,
            rank_identity_kind="global_pid",
            rank_mapping_provenance=RANK_MAPPING_PROVENANCE,
            captured_rank_keys=ordered_rank_keys,
            busy_by_rank_key_ms=tuple((rank_key, busy_by_rank_ns[rank_key] / 1e6) for rank_key in ordered_rank_keys),
            rank_compositions=tuple(rank_compositions),
            alignment_mapping_hash=alignment.mapping_hash,
            kernel_classifier_version=KERNEL_CLASSIFIER_VERSION,
            unknown_kernel_count=len(unknown),
            unknown_kernel_duration_ms=sum(kernel.end_ns - kernel.start_ns for kernel in unknown) / 1e6,
            unknown_kernel_name_hash=_unknown_name_hash(kernel.name for kernel in unknown),
        )
    return compositions


def build_cohort_nsight_contract(
    *,
    concurrency: int,
    profiled_samples: tuple[FpmSample, ...],
    sqlite_path: str | Path,
    expected_rank_count: int,
) -> tuple[CohortAlignmentProvenance, dict[str, ProfiledStepComposition]]:
    """Build typed alignment provenance and compositions for one concurrency."""

    if not profiled_samples:
        raise ContractError(
            process_code="same_run_alignment_missing",
            detail=f"concurrency {concurrency} has no measured profiled samples",
        )
    markers = load_nsys_markers(sqlite_path)
    rank_keys = captured_rank_keys(markers, expected_rank_count=expected_rank_count)
    observations = [
        ProfiledFpmObservation(
            counter_id=sample.counter_id,
            phase=sample.phase,
            shape=sample.shape,
        )
        for sample in profiled_samples
    ]
    alignment = align_profiled_fpm_to_nsys(
        observations,
        markers,
        expected_rank_keys=rank_keys,
    )
    windows = _canonical_windows_by_rank(markers, rank_keys)
    kernels = load_attributed_kernels(sqlite_path, windows=windows)
    compositions = compose_profiled_steps(
        samples=profiled_samples,
        alignment=alignment,
        rank_keys=rank_keys,
        kernels=kernels,
    )
    provenance = CohortAlignmentProvenance(
        concurrency=concurrency,
        input_profiled_rows=len(profiled_samples),
        mapped_profiled_rows=alignment.mapped_count,
        raw_marker_count=alignment.raw_marker_count,
        canonical_marker_count=alignment.canonical_marker_count,
        mapping_hash=alignment.mapping_hash,
        expected_rank_keys=rank_keys,
        captured_rank_keys=rank_keys,
        rank_identity_kind="global_pid",
        rank_mapping_provenance=RANK_MAPPING_PROVENANCE,
        nonmonotonic_marker_identities=tuple(
            AlignmentMarkerIdentity.from_canonical(marker) for marker in alignment.nonmonotonic_markers
        ),
        unmatched_prefix_marker_identities=tuple(
            AlignmentMarkerIdentity.from_canonical(marker) for marker in alignment.unmatched_prefix_markers
        ),
        unmatched_suffix_marker_identities=tuple(
            AlignmentMarkerIdentity.from_canonical(marker) for marker in alignment.unmatched_suffix_markers
        ),
        internal_skipped_marker_identities=tuple(
            AlignmentMarkerIdentity.from_canonical(marker) for marker in alignment.internal_skipped_markers
        ),
        mapped_segments=tuple(AlignmentSegmentProvenance.from_alignment(segment) for segment in alignment.segments),
    )
    return provenance, compositions
