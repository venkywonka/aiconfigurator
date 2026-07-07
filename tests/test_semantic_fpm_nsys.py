# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from collector.layerwise.diagnostics.semantic_fpm_insights import (
    ALIGNMENT_VERSION,
    MARKER_ENCODER_VERSION,
    AlignedStep,
    AlignmentResult,
    AlignmentSegment,
    MarkerObservation,
    SemanticShape,
)
from collector.layerwise.diagnostics.semantic_fpm_nsys import (
    AttributedKernel,
    captured_rank_keys,
    classify_kernel_name,
    compose_profiled_steps,
    load_attributed_kernels,
)
from collector.layerwise.diagnostics.semantic_fpm_reduction import ContractError, FpmSample


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "communication"),
        ("allreduce_fusion_kernel_oneshot_lamport", "communication"),
        ("triton_poi_fused_0", "compute"),
        ("nvjet_tst_256x128_64x4_2x4_h_b_10...", "compute"),
        ("myncclDevKernel_AllReduce_Sum_bf16_RING_LL", None),
        ("allreduce_mystery", None),
        ("", None),
    ),
)
def test_versioned_kernel_classifier_uses_full_registry_matches(name, expected):
    assert classify_kernel_name(name) == expected


def _sample() -> FpmSample:
    shape = SemanticShape(0, 2, 0, 0, 200)
    return FpmSample(
        lane="profiled",
        concurrency=16,
        sample_id="profiled-1",
        phase="decode",
        workload_segment="real",
        counter_id=1,
        worker_id="worker",
        dp_rank=0,
        shape=shape,
        wall_ms=1.0,
        measured=True,
        analytic_eligible=True,
    )


def _alignment() -> AlignmentResult:
    mapped = AlignedStep(
        fpm_sequence_index=0,
        fpm_counter_id=1,
        marker_canonical_index=0,
        marker_step=7,
        measure_run=0,
        marker_decode_batch=2,
        marker_mean_decode_kv=100,
    )
    return AlignmentResult(
        alignment_version=ALIGNMENT_VERSION,
        marker_encoder_version=MARKER_ENCODER_VERSION,
        mapping=(mapped,),
        mapping_hash="a" * 64,
        segments=(
            AlignmentSegment(
                measure_run=0,
                start_marker_canonical_index=0,
                end_marker_canonical_index=0,
                start_marker_step=7,
                end_marker_step=7,
                mapped_rows=1,
            ),
        ),
        internal_skipped_markers=(),
        nonmonotonic_markers=(),
        unmatched_prefix_markers=(),
        unmatched_suffix_markers=(),
        rank_keys=("0", "1"),
        raw_marker_count=1,
        canonical_marker_count=1,
    )


def _kernel(rank_key: str, start: int, end: int, name: str) -> AttributedKernel:
    return AttributedKernel(
        rank_key=rank_key,
        marker_canonical_index=0,
        start_ns=start,
        end_ns=end,
        name=name,
    )


def test_composition_selects_one_intact_max_busy_rank_tuple():
    composition = compose_profiled_steps(
        samples=(_sample(),),
        alignment=_alignment(),
        rank_keys=("0", "1"),
        kernels=(
            _kernel("0", 0, 10, "triton_poi_fused_0"),
            _kernel("0", 5, 15, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"),
            _kernel("1", 0, 20, "triton_poi_fused_0"),
        ),
    )["profiled-1"]

    assert composition.rank_key == "1"
    assert composition.gpu_busy_ms == pytest.approx(20 / 1e6)
    assert composition.gpu_compute_ms == pytest.approx(20 / 1e6)
    assert composition.gpu_comm_ms == pytest.approx(0.0)
    assert composition.rank_compositions[0].gpu_compute_ms == pytest.approx(10 / 1e6)
    assert composition.rank_compositions[0].gpu_comm_ms == pytest.approx(10 / 1e6)


def test_unknown_on_nonselected_rank_retains_busy_and_nulls_emitted_components():
    composition = compose_profiled_steps(
        samples=(_sample(),),
        alignment=_alignment(),
        rank_keys=("0", "1"),
        kernels=(
            _kernel("0", 0, 2, "mystery_kernel"),
            _kernel("1", 0, 20, "triton_poi_fused_0"),
        ),
    )["profiled-1"]

    assert composition.rank_key == "1"
    assert composition.gpu_busy_ms == pytest.approx(20 / 1e6)
    assert composition.gpu_compute_ms is None
    assert composition.gpu_comm_ms is None
    assert composition.unknown_kernel_count == 1
    assert composition.unknown_kernel_duration_ms == pytest.approx(2 / 1e6)
    assert composition.unknown_kernel_name_hash == hashlib.sha256(b'["mystery_kernel"]').hexdigest()
    assert composition.rank_compositions[0].gpu_compute_ms is None
    assert composition.rank_compositions[1].gpu_compute_ms == pytest.approx(20 / 1e6)


def test_max_busy_tie_uses_lower_numeric_rank_key():
    composition = compose_profiled_steps(
        samples=(_sample(),),
        alignment=_alignment(),
        rank_keys=("0", "1"),
        kernels=(
            _kernel("0", 0, 20, "triton_poi_fused_0"),
            _kernel("1", 0, 20, "triton_poi_fused_0"),
        ),
    )["profiled-1"]

    assert composition.rank_key == "0"


def test_observed_rank_keys_are_checked_against_independent_runtime_count():
    marker = MarkerObservation(
        rank_key="100",
        marker_step=1,
        measure_run=0,
        decode_batch=2,
        mean_decode_kv=100,
        start_ns=0,
        end_ns=10,
    )

    with pytest.raises(ContractError) as exc_info:
        captured_rank_keys([marker], expected_rank_count=2)

    assert exc_info.value.process_code == "incomplete_rank_capture"


def test_kernel_loader_uses_profiled_runtime_index_without_graph_join(tmp_path):
    sqlite_path = tmp_path / "trace.sqlite"
    connection = sqlite3.connect(sqlite_path)
    connection.executescript(
        """
        CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
            correlationId INTEGER, globalTid INTEGER, start INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
            correlationId INTEGER, graphNodeId INTEGER, start INTEGER,
            end INTEGER, shortName INTEGER, globalPid INTEGER
        );
        INSERT INTO StringIds VALUES(1, 'triton_poi_fused_0');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(7, 16777221, 150);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(8, 16777221, 250);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(7, 11, 160, 180, 1, 16777216);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(8, 12, 260, 280, 1, 16777216);
        """
    )
    connection.commit()
    connection.close()
    windows = {
        "16777216": (
            MarkerObservation(
                rank_key="16777216",
                marker_step=1,
                measure_run=0,
                decode_batch=2,
                mean_decode_kv=100,
                start_ns=100,
                end_ns=200,
            ),
        )
    }

    rows = load_attributed_kernels(sqlite_path, windows=windows)

    assert rows == (
        AttributedKernel(
            rank_key="16777216",
            marker_canonical_index=0,
            start_ns=160,
            end_ns=180,
            name="triton_poi_fused_0",
        ),
    )
