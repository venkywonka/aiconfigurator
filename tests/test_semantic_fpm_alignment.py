# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from collector.layerwise.diagnostics.semantic_fpm_insights import (
    ALIGNMENT_VERSION,
    MARKER_ENCODER_VERSION,
    AlignmentError,
    MarkerObservation,
    ProfiledFpmObservation,
    SemanticShape,
    align_profiled_fpm_to_nsys,
)

pytestmark = pytest.mark.unit


def _fpm(counter: int, shape: SemanticShape, phase: str) -> ProfiledFpmObservation:
    return ProfiledFpmObservation(counter_id=counter, phase=phase, shape=shape)


def _marker(
    rank_key: str,
    step: int,
    batch: int,
    past: int,
    *,
    start: int,
    duration: int = 10,
    run: int = 0,
) -> MarkerObservation:
    return MarkerObservation(
        rank_key=rank_key,
        marker_step=step,
        measure_run=run,
        decode_batch=batch,
        mean_decode_kv=past,
        start_ns=start,
        end_ns=start + duration,
    )


def _two_rank_stream(specs: list[tuple[int, int, int, int]]) -> list[MarkerObservation]:
    rows: list[MarkerObservation] = []
    for step, batch, past, duration in specs:
        rows.append(_marker("r0", step, batch, past, start=step * 100, duration=duration))
        rows.append(_marker("r1", step, batch, past, start=step * 100 + 1, duration=duration + 2))
    return rows


def test_exact_multi_rank_alignment_maps_every_fpm_row():
    fpm = [
        _fpm(10, SemanticShape(1, 0, 8, 0, 0), "context"),
        _fpm(11, SemanticShape(0, 2, 0, 0, 17), "decode"),
        _fpm(12, SemanticShape(1, 2, 4, 0, 19), "mixed"),
    ]
    markers = _two_rank_stream([(7, 0, 0, 20), (8, 2, 8, 10), (9, 2, 10, 10)])

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))

    assert [row.fpm_counter_id for row in result.mapping] == [10, 11, 12]
    assert [row.marker_step for row in result.mapping] == [7, 8, 9]
    assert result.mapped_count == len(fpm)
    assert result.alignment_version == ALIGNMENT_VERSION == "profiled-monotonic-v1"
    assert result.marker_encoder_version == MARKER_ENCODER_VERSION == "python-round-half-even-v1"
    assert result.internal_skipped_markers == ()
    assert [
        (segment.start_marker_step, segment.end_marker_step, segment.mapped_rows) for segment in result.segments
    ] == [(7, 9, 3)]
    segment = result.segments[0]
    assert (segment.measure_run, segment.start_marker_canonical_index, segment.end_marker_canonical_index) == (0, 0, 2)


def test_alignment_uses_legacy_ties_to_even_marker_anchor_not_semantic_half_up():
    fpm = [_fpm(1390, SemanticShape(0, 4, 0, 0, 33802), "decode")]
    markers = _two_rank_stream([(100, 4, 8450, 10)])

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))

    assert fpm[0].shape.semantic_key[-1] == 8451
    assert result.mapping[0].marker_mean_decode_kv == 8450


def test_alignment_canonicalizes_nonmonotonic_markers_per_measure_run():
    fpm = [
        _fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode"),
        _fpm(2, SemanticShape(0, 1, 0, 0, 11), "decode"),
        _fpm(3, SemanticShape(0, 1, 0, 0, 12), "decode"),
    ]
    markers = []
    for rank in ("r0", "r1"):
        markers.extend(
            [
                _marker(rank, 1, 1, 10, start=100),
                _marker(rank, 2, 1, 11, start=200),
                _marker(rank, 1, 2, 99, start=300),
                _marker(rank, 3, 1, 12, start=400),
            ]
        )

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))

    assert [row.marker_step for row in result.mapping] == [1, 2, 3]
    assert [(row.marker_step, row.decode_batch, row.mean_decode_kv) for row in result.nonmonotonic_markers] == [
        (1, 2, 99)
    ]


def test_alignment_allows_declared_measure_run_ordinal_reset():
    fpm = [
        _fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode"),
        _fpm(2, SemanticShape(0, 1, 0, 0, 20), "decode"),
    ]
    markers = []
    for rank in ("r0", "r1"):
        markers.extend(
            [
                _marker(rank, 7, 1, 10, start=100, run=0),
                _marker(rank, 1, 1, 20, start=200, run=1),
            ]
        )

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert [(row.measure_run, row.marker_step) for row in result.mapping] == [(0, 7), (1, 1)]
    assert [(segment.measure_run, segment.mapped_rows) for segment in result.segments] == [(0, 1), (1, 1)]


def test_equal_timestamp_markers_have_deterministic_identity_tie_break():
    fpm = [
        _fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode"),
        _fpm(2, SemanticShape(0, 1, 0, 0, 11), "decode"),
    ]
    markers = [
        _marker("r0", 2, 1, 11, start=100, duration=10),
        _marker("r0", 1, 1, 10, start=100, duration=20),
        _marker("r1", 1, 1, 10, start=100, duration=10),
        _marker("r1", 2, 1, 11, start=100, duration=20),
    ]

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert [row.marker_step for row in result.mapping] == [1, 2]


def test_nonmonotonicity_is_tracked_per_measure_run_when_run_reenters():
    fpm = [
        _fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode"),
        _fpm(2, SemanticShape(0, 1, 0, 0, 20), "decode"),
    ]
    markers = []
    for rank in ("r0", "r1"):
        markers.extend(
            [
                _marker(rank, 2, 1, 10, start=100, run=0),
                _marker(rank, 1, 1, 20, start=200, run=1),
                _marker(rank, 1, 9, 99, start=300, run=0),
            ]
        )

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert [(row.measure_run, row.marker_step) for row in result.mapping] == [(0, 2), (1, 1)]
    assert [(row.measure_run, row.marker_step) for row in result.nonmonotonic_markers] == [(0, 1)]


def test_context_duration_breaks_equal_span_mapping_tie():
    fpm = [
        _fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode"),
        _fpm(2, SemanticShape(1, 0, 4, 0, 0), "context"),
        _fpm(3, SemanticShape(0, 1, 0, 0, 11), "decode"),
    ]
    markers = _two_rank_stream(
        [
            (1, 1, 10, 10),
            (2, 0, 0, 10),
            (3, 0, 0, 30),
            (4, 1, 11, 10),
        ]
    )

    result = align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))

    assert [row.marker_step for row in result.mapping] == [1, 3, 4]
    assert [row.marker_step for row in result.internal_skipped_markers] == [2]
    assert [(segment.start_marker_step, segment.end_marker_step) for segment in result.segments] == [(1, 1), (3, 4)]


def test_equal_span_and_duration_is_ambiguous():
    fpm = [_fpm(1, SemanticShape(1, 0, 4, 0, 0), "context")]
    markers = _two_rank_stream([(1, 0, 0, 10), (2, 0, 0, 10)])

    with pytest.raises(AlignmentError) as exc_info:
        align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert exc_info.value.process_code == "same_run_alignment_ambiguous"


def test_missing_complete_mapping_fails_closed():
    fpm = [_fpm(1, SemanticShape(0, 4, 0, 0, 40), "decode")]
    markers = _two_rank_stream([(1, 3, 10, 10)])

    with pytest.raises(AlignmentError) as exc_info:
        align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert exc_info.value.process_code == "same_run_alignment_missing"


def test_incomplete_rank_capture_fails_before_alignment():
    markers = [_marker("r0", 1, 1, 10, start=100)]
    fpm = [_fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode")]

    with pytest.raises(AlignmentError) as exc_info:
        align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert exc_info.value.process_code == "incomplete_rank_capture"


@pytest.mark.parametrize("expected_rank_keys", [(), ("r0", "r0")])
def test_missing_or_duplicate_expected_rank_contract_has_stable_failure(expected_rank_keys):
    fpm = [_fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode")]
    markers = [_marker("r0", 1, 1, 10, start=100)]

    with pytest.raises(AlignmentError) as exc_info:
        align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=expected_rank_keys)
    assert exc_info.value.process_code == "rank_identity_unavailable"


def test_rank_tuple_identity_mismatch_fails_before_alignment():
    markers = [
        _marker("r0", 1, 1, 10, start=100),
        _marker("r1", 1, 2, 10, start=101),
    ]
    fpm = [_fpm(1, SemanticShape(0, 1, 0, 0, 10), "decode")]

    with pytest.raises(AlignmentError) as exc_info:
        align_profiled_fpm_to_nsys(fpm, markers, expected_rank_keys=("r0", "r1"))
    assert exc_info.value.process_code == "rank_tuple_identity_mismatch"
