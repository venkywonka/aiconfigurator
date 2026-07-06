#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized semantic identity for clean and profiled FPM observations.

This module owns the versioned, integer-only identity used to compare independent
FPM runs. Same-run FPM-to-Nsight alignment deliberately uses its captured marker
encoder instead; callers must not substitute Python ``round`` for this contract.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from typing import TypeAlias

SCHEMA_VERSION = "fpm-semantic-insights/v1"
ALIGNMENT_VERSION = "profiled-monotonic-v1"
MARKER_ENCODER_VERSION = "python-round-half-even-v1"

SemanticKey: TypeAlias = tuple[int, int, int | None, int | None, int | None]


class ShapeValidationError(ValueError):
    """A source scheduler shape violates the normalized structural contract."""

    def __init__(self, *, process_code: str, reason: str, detail: str):
        super().__init__(detail)
        self.process_code = process_code
        self.reason = reason


def _require_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShapeValidationError(
            process_code="invalid_shape",
            reason="negative_value" if isinstance(value, int) and value < 0 else "non_integer_value",
            detail=f"{name} must be a non-negative integer, got {value!r}",
        )


def round_half_up_ratio(total: int, count: int) -> int:
    """Round a non-negative integer ratio to nearest integer, with ties upward."""

    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError(f"total must be a non-negative integer, got {total!r}")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"count must be a positive integer, got {count!r}")
    return (2 * total + count) // (2 * count)


def derive_phase(ctx_requests: int, decode_requests: int) -> str:
    """Derive the scheduler phase from exact request counts."""

    _require_nonnegative_integer("ctx_requests", ctx_requests)
    _require_nonnegative_integer("decode_requests", decode_requests)
    if ctx_requests and decode_requests:
        return "mixed"
    if ctx_requests:
        return "context"
    if decode_requests:
        return "decode"
    return "idle"


@dataclass(frozen=True)
class SemanticShape:
    """Exact scheduler counts and token totals from one FPM observation."""

    ctx_requests: int
    decode_requests: int
    ctx_new_tokens: int
    ctx_kv_tokens: int
    decode_kv_tokens: int

    def validate(self, *, source_phase: str | None = None) -> str:
        """Validate structural invariants and return the count-derived phase."""

        for name in (
            "ctx_requests",
            "decode_requests",
            "ctx_new_tokens",
            "ctx_kv_tokens",
            "decode_kv_tokens",
        ):
            _require_nonnegative_integer(name, getattr(self, name))

        if self.ctx_requests == 0 and (self.ctx_new_tokens != 0 or self.ctx_kv_tokens != 0):
            raise ShapeValidationError(
                process_code="invalid_shape",
                reason="zero_count_nonzero_total",
                detail="context token totals must be zero when ctx_requests is zero",
            )
        if self.decode_requests == 0 and self.decode_kv_tokens != 0:
            raise ShapeValidationError(
                process_code="invalid_shape",
                reason="zero_count_nonzero_total",
                detail="decode_kv_tokens must be zero when decode_requests is zero",
            )
        if self.ctx_requests > 0 and self.ctx_new_tokens == 0:
            raise ShapeValidationError(
                process_code="invalid_shape",
                reason="active_context_without_new_tokens",
                detail="an active context axis requires positive ctx_new_tokens",
            )
        if self.decode_requests > 0 and self.decode_kv_tokens == 0:
            raise ShapeValidationError(
                process_code="invalid_shape",
                reason="active_decode_without_kv_tokens",
                detail="an active decode axis requires positive decode_kv_tokens",
            )

        phase = derive_phase(self.ctx_requests, self.decode_requests)
        if source_phase is not None and source_phase != phase:
            raise ShapeValidationError(
                process_code="phase_mismatch",
                reason="source_phase_disagrees_with_counts",
                detail=f"source phase {source_phase!r} disagrees with count-derived phase {phase!r}",
            )
        return phase

    @property
    def semantic_key(self) -> SemanticKey:
        self.validate()
        ctx_new = round_half_up_ratio(self.ctx_new_tokens, self.ctx_requests) if self.ctx_requests else None
        ctx_kv = round_half_up_ratio(self.ctx_kv_tokens, self.ctx_requests) if self.ctx_requests else None
        decode_kv = round_half_up_ratio(self.decode_kv_tokens, self.decode_requests) if self.decode_requests else None
        return (self.ctx_requests, self.decode_requests, ctx_new, ctx_kv, decode_kv)

    @property
    def serialized_key(self) -> str:
        return canonical_semantic_key(self.semantic_key)


def canonical_semantic_key(key: SemanticKey) -> str:
    """Serialize a semantic key as the normative whitespace-free JSON array."""

    if len(key) != 5:
        raise ValueError(f"semantic key must have five fields, got {len(key)}")
    return json.dumps(key, separators=(",", ":"), ensure_ascii=True)


def stable_bin_id(*, configuration_fingerprint: str, concurrency: int, phase: str, semantic_key: SemanticKey) -> str:
    """Return the stable SHA-256 identifier for a normalized semantic bin."""

    if not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be non-empty")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError(f"concurrency must be a positive integer, got {concurrency!r}")
    identity = {
        "concurrency": concurrency,
        "configuration_fingerprint": configuration_fingerprint,
        "phase": phase,
        "schema_version": SCHEMA_VERSION,
        "semantic_key": list(semantic_key),
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SemanticBinQuery:
    """Canonical, immutable query passed through the reducer/predictor seam."""

    schema_version: str
    configuration_fingerprint: str
    concurrency: int
    phase: str
    semantic_key: str
    ctx_requests: int
    decode_requests: int
    ctx_new_per_request: int | None
    ctx_kv_per_request: int | None
    decode_kv_per_request: int | None
    query_ctx_new_total: int
    query_ctx_kv_total: int
    query_decode_kv: int


def build_semantic_query(
    *,
    configuration_fingerprint: str,
    concurrency: int,
    shape: SemanticShape,
    source_phase: str | None = None,
) -> SemanticBinQuery:
    """Normalize a measured shape and reconstruct its deterministic AIC query."""

    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError(f"concurrency must be a positive integer, got {concurrency!r}")
    if not configuration_fingerprint:
        raise ValueError("configuration_fingerprint must be non-empty")
    phase = shape.validate(source_phase=source_phase)
    if phase == "idle":
        raise ValueError("idle shapes are structurally ineligible for predictor queries")
    key = shape.semantic_key
    ctx_new_per_request = key[2]
    ctx_kv_per_request = key[3]
    decode_kv_per_request = key[4]
    return SemanticBinQuery(
        schema_version=SCHEMA_VERSION,
        configuration_fingerprint=configuration_fingerprint,
        concurrency=concurrency,
        phase=phase,
        semantic_key=canonical_semantic_key(key),
        ctx_requests=shape.ctx_requests,
        decode_requests=shape.decode_requests,
        ctx_new_per_request=ctx_new_per_request,
        ctx_kv_per_request=ctx_kv_per_request,
        decode_kv_per_request=decode_kv_per_request,
        query_ctx_new_total=shape.ctx_requests * (ctx_new_per_request or 0),
        query_ctx_kv_total=shape.ctx_requests * (ctx_kv_per_request or 0),
        query_decode_kv=decode_kv_per_request or 0,
    )


class AlignmentError(RuntimeError):
    """The profiled FPM stream cannot be mapped uniquely to Nsight markers."""

    def __init__(self, *, process_code: str, detail: str):
        super().__init__(detail)
        self.process_code = process_code


@dataclass(frozen=True)
class ProfiledFpmObservation:
    """The scheduler fields required for same-run profiled alignment."""

    counter_id: int
    phase: str
    shape: SemanticShape


@dataclass(frozen=True)
class MarkerObservation:
    """One rank-key copy of a captured ``bench_step`` NVTX marker."""

    rank_key: str
    marker_step: int
    measure_run: int
    decode_batch: int
    mean_decode_kv: int
    start_ns: int
    end_ns: int


@dataclass(frozen=True)
class CanonicalMarker:
    """A rank-reconciled marker in chronological stream order."""

    canonical_index: int
    marker_step: int
    measure_run: int
    decode_batch: int
    mean_decode_kv: int
    median_span_ns: int | float


@dataclass(frozen=True)
class AlignedStep:
    """One profiled FPM row attached to one canonical marker."""

    fpm_sequence_index: int
    fpm_counter_id: int
    marker_canonical_index: int
    marker_step: int
    measure_run: int
    marker_decode_batch: int
    marker_mean_decode_kv: int


@dataclass(frozen=True)
class AlignmentSegment:
    """A maximal contiguous run of mapped canonical markers."""

    measure_run: int
    start_marker_canonical_index: int
    end_marker_canonical_index: int
    start_marker_step: int
    end_marker_step: int
    mapped_rows: int


@dataclass(frozen=True)
class AlignmentResult:
    """Auditable result of ``profiled-monotonic-v1`` alignment."""

    alignment_version: str
    marker_encoder_version: str
    mapping: tuple[AlignedStep, ...]
    mapping_hash: str
    segments: tuple[AlignmentSegment, ...]
    internal_skipped_markers: tuple[CanonicalMarker, ...]
    nonmonotonic_markers: tuple[CanonicalMarker, ...]
    unmatched_prefix_markers: tuple[CanonicalMarker, ...]
    unmatched_suffix_markers: tuple[CanonicalMarker, ...]
    rank_keys: tuple[str, ...]
    raw_marker_count: int
    canonical_marker_count: int

    @property
    def mapped_count(self) -> int:
        return len(self.mapping)


def _marker_identity(marker: MarkerObservation | CanonicalMarker) -> tuple[int, int, int, int]:
    return (marker.marker_step, marker.measure_run, marker.decode_batch, marker.mean_decode_kv)


def _round_half_even_ratio(total: int, count: int) -> int:
    """Exact non-negative equivalent of Python ``round(total / count)``."""

    quotient, remainder = divmod(total, count)
    doubled = 2 * remainder
    if doubled < count:
        return quotient
    if doubled > count:
        return quotient + 1
    return quotient if quotient % 2 == 0 else quotient + 1


def _fpm_marker_identity(row: ProfiledFpmObservation) -> tuple[int, int]:
    phase = row.shape.validate(source_phase=row.phase)
    if phase == "idle":
        return (0, 0)
    if row.shape.decode_requests == 0:
        return (0, 0)
    return (
        row.shape.decode_requests,
        _round_half_even_ratio(row.shape.decode_kv_tokens, row.shape.decode_requests),
    )


def _validate_marker(marker: MarkerObservation) -> None:
    for name in ("marker_step", "measure_run", "decode_batch", "mean_decode_kv", "start_ns", "end_ns"):
        value = getattr(marker, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AlignmentError(
                process_code="invalid_measurement",
                detail=f"marker {name} must be a non-negative integer, got {value!r}",
            )
    if marker.end_ns < marker.start_ns:
        raise AlignmentError(
            process_code="invalid_measurement",
            detail=f"marker end {marker.end_ns} precedes start {marker.start_ns}",
        )
    if not marker.rank_key:
        raise AlignmentError(process_code="rank_identity_unavailable", detail="marker rank_key is empty")


def _canonicalize_marker_streams(
    markers: list[MarkerObservation], expected_rank_keys: tuple[str, ...]
) -> tuple[list[CanonicalMarker], list[CanonicalMarker], tuple[str, ...], int]:
    if not expected_rank_keys or len(set(expected_rank_keys)) != len(expected_rank_keys):
        raise AlignmentError(
            process_code="rank_identity_unavailable",
            detail="expected_rank_keys must be a non-empty unique tuple",
        )
    for marker in markers:
        _validate_marker(marker)

    expected = tuple(sorted(expected_rank_keys))
    by_rank: dict[str, list[MarkerObservation]] = defaultdict(list)
    for marker in markers:
        by_rank[marker.rank_key].append(marker)
    actual = tuple(sorted(by_rank))
    if actual != expected:
        raise AlignmentError(
            process_code="incomplete_rank_capture",
            detail=f"expected rank keys {expected!r}, captured {actual!r}",
        )

    streams = []
    for rank_key in expected:
        stream = sorted(
            by_rank[rank_key],
            key=lambda marker: (
                marker.start_ns,
                marker.measure_run,
                marker.marker_step,
                marker.decode_batch,
                marker.mean_decode_kv,
                marker.end_ns,
            ),
        )
        streams.append(stream)
    lengths = {len(stream) for stream in streams}
    if len(lengths) != 1:
        raise AlignmentError(
            process_code="incomplete_rank_capture",
            detail=f"rank marker counts disagree: {[len(stream) for stream in streams]!r}",
        )

    raw: list[CanonicalMarker] = []
    for stream_index, rank_markers in enumerate(zip(*streams, strict=True)):
        identities = {_marker_identity(marker) for marker in rank_markers}
        if len(identities) != 1:
            raise AlignmentError(
                process_code="rank_tuple_identity_mismatch",
                detail=f"rank marker identities disagree at stream index {stream_index}: {sorted(identities)!r}",
            )
        first = rank_markers[0]
        raw.append(
            CanonicalMarker(
                canonical_index=stream_index,
                marker_step=first.marker_step,
                measure_run=first.measure_run,
                decode_batch=first.decode_batch,
                mean_decode_kv=first.mean_decode_kv,
                median_span_ns=statistics.median(marker.end_ns - marker.start_ns for marker in rank_markers),
            )
        )

    canonical: list[CanonicalMarker] = []
    nonmonotonic: list[CanonicalMarker] = []
    previous_step_by_run: dict[int, int] = {}
    for marker in raw:
        previous_step = previous_step_by_run.get(marker.measure_run)
        if previous_step is not None and marker.marker_step <= previous_step:
            nonmonotonic.append(marker)
            continue
        canonical.append(
            CanonicalMarker(
                canonical_index=len(canonical),
                marker_step=marker.marker_step,
                measure_run=marker.measure_run,
                decode_batch=marker.decode_batch,
                mean_decode_kv=marker.mean_decode_kv,
                median_span_ns=marker.median_span_ns,
            )
        )
        previous_step_by_run[marker.measure_run] = marker.marker_step
    return canonical, nonmonotonic, expected, len(raw)


def _minimum_span_pairs(
    fpm_identities: list[tuple[int, int]], candidates_by_identity: dict[tuple[int, int], list[int]]
) -> list[tuple[int, int]]:
    first_candidates = candidates_by_identity.get(fpm_identities[0], [])
    complete: list[tuple[int, int, int]] = []
    for start in first_candidates:
        position = start
        for identity in fpm_identities[1:]:
            candidates = candidates_by_identity.get(identity, [])
            next_offset = bisect_right(candidates, position)
            if next_offset == len(candidates):
                break
            position = candidates[next_offset]
        else:
            complete.append((position - start + 1, start, position))
    if not complete:
        return []
    minimum_span = min(span for span, _start, _end in complete)
    return [(start, end) for span, start, end in complete if span == minimum_span]


def _context_weight(row: ProfiledFpmObservation, marker: CanonicalMarker) -> int | float:
    return marker.median_span_ns if row.phase == "context" else 0


def _best_path_for_span(
    fpm_rows: list[ProfiledFpmObservation],
    candidates_by_identity: dict[tuple[int, int], list[int]],
    canonical_markers: list[CanonicalMarker],
    start: int,
    end: int,
) -> tuple[int | float, int, tuple[int, ...]] | None:
    identities = [_fpm_marker_identity(row) for row in fpm_rows]
    states: dict[int, tuple[int | float, int, tuple[int, ...]]] = {
        start: (_context_weight(fpm_rows[0], canonical_markers[start]), 1, (start,))
    }
    for row_index, (row, identity) in enumerate(zip(fpm_rows[1:], identities[1:], strict=True), start=1):
        if row_index == len(fpm_rows) - 1:
            current_candidates = [end]
        else:
            all_candidates = candidates_by_identity.get(identity, [])
            left = bisect_right(all_candidates, start)
            right = bisect_left(all_candidates, end)
            current_candidates = all_candidates[left:right]

        previous = sorted(states.items())
        previous_offset = 0
        best_score: int | float | None = None
        best_count = 0
        best_path: tuple[int, ...] = ()
        next_states: dict[int, tuple[int | float, int, tuple[int, ...]]] = {}
        for candidate in current_candidates:
            while previous_offset < len(previous) and previous[previous_offset][0] < candidate:
                _index, (score, count, path) = previous[previous_offset]
                if best_score is None or score > best_score:
                    best_score, best_count, best_path = score, count, path
                elif score == best_score:
                    best_count = min(2, best_count + count)
                    best_path = min(best_path, path)
                previous_offset += 1
            if best_score is None:
                continue
            next_states[candidate] = (
                best_score + _context_weight(row, canonical_markers[candidate]),
                best_count,
                best_path + (candidate,),
            )
        states = next_states
        if not states:
            return None
    return states.get(end)


def _alignment_segments(mapping: tuple[AlignedStep, ...]) -> tuple[AlignmentSegment, ...]:
    if not mapping:
        return ()
    segments: list[AlignmentSegment] = []
    segment_start = mapping[0]
    previous = mapping[0]
    count = 1
    for current in mapping[1:]:
        if (
            current.marker_canonical_index != previous.marker_canonical_index + 1
            or current.measure_run != previous.measure_run
        ):
            segments.append(
                AlignmentSegment(
                    measure_run=segment_start.measure_run,
                    start_marker_canonical_index=segment_start.marker_canonical_index,
                    end_marker_canonical_index=previous.marker_canonical_index,
                    start_marker_step=segment_start.marker_step,
                    end_marker_step=previous.marker_step,
                    mapped_rows=count,
                )
            )
            segment_start = current
            count = 0
        count += 1
        previous = current
    segments.append(
        AlignmentSegment(
            measure_run=segment_start.measure_run,
            start_marker_canonical_index=segment_start.marker_canonical_index,
            end_marker_canonical_index=previous.marker_canonical_index,
            start_marker_step=segment_start.marker_step,
            end_marker_step=previous.marker_step,
            mapped_rows=count,
        )
    )
    return tuple(segments)


def _mapping_hash(mapping: tuple[AlignedStep, ...]) -> str:
    records = [
        {
            "counter_id": row.fpm_counter_id,
            "fpm_sequence_index": row.fpm_sequence_index,
            "marker_canonical_index": row.marker_canonical_index,
            "marker_step": row.marker_step,
            "measure_run": row.measure_run,
        }
        for row in mapping
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def align_profiled_fpm_to_nsys(
    fpm_rows: list[ProfiledFpmObservation],
    marker_rows: list[MarkerObservation],
    *,
    expected_rank_keys: tuple[str, ...],
) -> AlignmentResult:
    """Apply the fail-closed ``profiled-monotonic-v1`` same-run alignment."""

    if not fpm_rows:
        raise AlignmentError(process_code="same_run_alignment_missing", detail="profiled FPM stream is empty")
    ordered_fpm = sorted(fpm_rows, key=lambda row: row.counter_id)
    if any(
        isinstance(row.counter_id, bool) or not isinstance(row.counter_id, int) or row.counter_id < 0
        for row in ordered_fpm
    ):
        raise AlignmentError(process_code="invalid_measurement", detail="FPM counter_id must be a non-negative integer")
    if len({row.counter_id for row in ordered_fpm}) != len(ordered_fpm):
        raise AlignmentError(process_code="duplicate_identity", detail="profiled FPM counter_id values are not unique")

    try:
        fpm_identities = [_fpm_marker_identity(row) for row in ordered_fpm]
    except ShapeValidationError as exc:
        raise AlignmentError(process_code=exc.process_code, detail=str(exc)) from exc

    canonical, nonmonotonic, rank_keys, raw_marker_count = _canonicalize_marker_streams(marker_rows, expected_rank_keys)
    candidates_by_identity: dict[tuple[int, int], list[int]] = defaultdict(list)
    for marker in canonical:
        candidates_by_identity[(marker.decode_batch, marker.mean_decode_kv)].append(marker.canonical_index)

    span_pairs = _minimum_span_pairs(fpm_identities, candidates_by_identity)
    if not span_pairs:
        raise AlignmentError(
            process_code="same_run_alignment_missing",
            detail="no complete order-preserving FPM-to-marker mapping exists",
        )

    best_score: int | float | None = None
    best_count = 0
    best_path: tuple[int, ...] = ()
    for start, end in span_pairs:
        candidate = _best_path_for_span(ordered_fpm, candidates_by_identity, canonical, start, end)
        if candidate is None:
            continue
        score, count, path = candidate
        if best_score is None or score > best_score:
            best_score, best_count, best_path = score, count, path
        elif score == best_score:
            best_count = min(2, best_count + count)
            best_path = min(best_path, path)

    if best_score is None:
        raise AlignmentError(
            process_code="same_run_alignment_missing",
            detail="minimum-span candidates did not yield a complete mapping",
        )
    if best_count > 1:
        raise AlignmentError(
            process_code="same_run_alignment_ambiguous",
            detail="multiple minimum-span, maximum-context-duration mappings remain",
        )

    mapping = tuple(
        AlignedStep(
            fpm_sequence_index=index,
            fpm_counter_id=row.counter_id,
            marker_canonical_index=marker_index,
            marker_step=canonical[marker_index].marker_step,
            measure_run=canonical[marker_index].measure_run,
            marker_decode_batch=canonical[marker_index].decode_batch,
            marker_mean_decode_kv=canonical[marker_index].mean_decode_kv,
        )
        for index, (row, marker_index) in enumerate(zip(ordered_fpm, best_path, strict=True))
    )
    mapped_indices = set(best_path)
    first_index = best_path[0]
    last_index = best_path[-1]
    return AlignmentResult(
        alignment_version=ALIGNMENT_VERSION,
        marker_encoder_version=MARKER_ENCODER_VERSION,
        mapping=mapping,
        mapping_hash=_mapping_hash(mapping),
        segments=_alignment_segments(mapping),
        internal_skipped_markers=tuple(
            marker for marker in canonical[first_index : last_index + 1] if marker.canonical_index not in mapped_indices
        ),
        nonmonotonic_markers=tuple(nonmonotonic),
        unmatched_prefix_markers=tuple(canonical[:first_index]),
        unmatched_suffix_markers=tuple(canonical[last_index + 1 :]),
        rank_keys=rank_keys,
        raw_marker_count=raw_marker_count,
        canonical_marker_count=len(canonical),
    )
