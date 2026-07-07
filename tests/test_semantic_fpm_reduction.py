# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv

import pytest

from collector.layerwise.diagnostics.semantic_fpm_reduction import (
    FPM_REQUIRED_COLUMNS,
    ContractError,
    build_semantic_population,
    descriptive_stats,
    load_fpm_phase_csv,
    support_class,
)

pytestmark = pytest.mark.unit

_LEGACY_PHASE_COLUMNS = (
    "phase",
    "workload_segment",
    "counter_id",
    "worker_id",
    "dp_rank",
    "ctx_tokens",
    "ctx_requests",
    "ctx_kv_tokens",
    "decode_tokens",
    "decode_requests",
    "decode_kv_tokens",
    "mean_decode_kv_tokens",
    "queued_ctx_tokens",
    "queued_ctx_requests",
    "queued_decode_requests",
    "queued_decode_kv_tokens",
    "latency_ms",
)


def _write_phase_csv(path, rows, *, columns=_LEGACY_PHASE_COLUMNS):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, overrides in enumerate(rows):
            row = {
                "phase": "decode",
                "workload_segment": "real",
                "counter_id": str(index + 100),
                "worker_id": "worker-0",
                "dp_rank": "0",
                "ctx_tokens": "0",
                "ctx_requests": "0",
                "ctx_kv_tokens": "0",
                "decode_tokens": "1",
                "decode_requests": "1",
                "decode_kv_tokens": str(index + 10),
                "mean_decode_kv_tokens": str(float(index + 10)),
                "queued_ctx_tokens": "0",
                "queued_ctx_requests": "0",
                "queued_decode_requests": "0",
                "queued_decode_kv_tokens": "0",
                "latency_ms": "2.5",
            }
            row.update(overrides)
            writer.writerow({column: row[column] for column in columns})


def test_load_fpm_phase_csv_preserves_all_segments_and_idle_rows(tmp_path):
    path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(
        path,
        [
            {},
            {"workload_segment": "warmup", "counter_id": "101"},
            {
                "phase": "idle",
                "counter_id": "102",
                "decode_tokens": "0",
                "decode_requests": "0",
                "decode_kv_tokens": "0",
                "mean_decode_kv_tokens": "0.0",
            },
        ],
    )

    rows = load_fpm_phase_csv(path, lane="profiled", concurrency=16)

    assert len(rows) == 3
    assert [row.workload_segment for row in rows] == ["real", "warmup", "real"]
    assert [row.phase for row in rows] == ["decode", "decode", "idle"]
    assert rows[0].lane == "profiled"
    assert rows[0].concurrency == 16
    assert rows[0].shape.decode_kv_tokens == 10
    assert rows[0].wall_ms == 2.5
    assert len({row.sample_id for row in rows}) == 3

    population = build_semantic_population(
        rows,
        configuration_fingerprint="cfg",
        measured_segments=frozenset({"real"}),
    )
    idle = next(sample for sample in population.samples if sample.phase == "idle")
    assert idle.measured is True
    assert idle.analytic_eligible is False
    assert idle.eligibility_reason == "idle_step"


def test_loader_uses_integer_totals_not_serialized_mean(tmp_path):
    path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(
        path,
        [
            {
                "decode_requests": "4",
                "decode_kv_tokens": "33802",
                "mean_decode_kv_tokens": "999999.0",
            }
        ],
    )

    row = load_fpm_phase_csv(path, lane="clean", concurrency=16)[0]
    assert row.shape.semantic_key == (0, 4, None, None, 8451)


def test_loader_accepts_minimal_consumed_contract_without_unused_legacy_columns(tmp_path):
    path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(path, [{}], columns=FPM_REQUIRED_COLUMNS)

    rows = load_fpm_phase_csv(path, lane="clean", concurrency=1)
    assert len(rows) == 1
    assert rows[0].shape.semantic_key == (0, 1, None, None, 10)


def test_loader_rejects_missing_required_schema(tmp_path):
    path = tmp_path / "fpm_metrics_phase.csv"
    columns = [column for column in FPM_REQUIRED_COLUMNS if column != "ctx_tokens"]
    _write_phase_csv(path, [{}], columns=columns)

    with pytest.raises(ContractError) as exc_info:
        load_fpm_phase_csv(path, lane="clean", concurrency=1)
    assert exc_info.value.process_code == "schema_missing"


@pytest.mark.parametrize(
    ("overrides", "process_code"),
    [
        ({"phase": "context"}, "phase_mismatch"),
        ({"ctx_tokens": "1", "ctx_requests": "0"}, "invalid_shape"),
        ({"decode_kv_tokens": "not-an-int"}, "schema_incompatible"),
        ({"latency_ms": "nan"}, "invalid_wall_time"),
        ({"latency_ms": "0"}, "invalid_wall_time"),
    ],
)
def test_loader_fails_closed_on_invalid_rows(tmp_path, overrides, process_code):
    path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(path, [overrides])

    with pytest.raises(ContractError) as exc_info:
        load_fpm_phase_csv(path, lane="clean", concurrency=1)
    assert exc_info.value.process_code == process_code


def test_population_uses_explicit_measured_segments_without_dropping_samples(tmp_path):
    path = tmp_path / "fpm_metrics_phase.csv"
    _write_phase_csv(path, [{}, {"workload_segment": "warmup"}])
    samples = load_fpm_phase_csv(path, lane="clean", concurrency=1)

    population = build_semantic_population(
        samples,
        configuration_fingerprint="cfg",
        measured_segments=frozenset({"real"}),
    )

    assert {sample.sample_id for sample in population.samples} == {sample.sample_id for sample in samples}
    assert len(population.bins) == 1
    assert population.excluded_reason_counts == {"non_measured_segment": 1}
    by_segment = {sample.workload_segment: sample for sample in population.samples}
    assert by_segment["real"].measured is True
    assert by_segment["real"].analytic_eligible is True
    assert by_segment["real"].eligibility_reason is None
    assert by_segment["warmup"].measured is False
    assert by_segment["warmup"].analytic_eligible is False
    assert by_segment["warmup"].eligibility_reason == "non_measured_segment"


def test_population_partitions_exact_concurrency_and_phase(tmp_path):
    c1_path = tmp_path / "c1.csv"
    c16_path = tmp_path / "c16.csv"
    _write_phase_csv(c1_path, [{"decode_kv_tokens": "100"}])
    _write_phase_csv(c16_path, [{"decode_kv_tokens": "100"}])
    samples = load_fpm_phase_csv(c1_path, lane="clean", concurrency=1)
    samples += load_fpm_phase_csv(c16_path, lane="profiled", concurrency=16)

    population = build_semantic_population(
        samples,
        configuration_fingerprint="cfg",
        measured_segments=frozenset({"real"}),
    )

    assert len(population.bins) == 2
    assert {(bin_.concurrency, bin_.phase, bin_.n_clean, bin_.n_profiled) for bin_ in population.bins} == {
        (1, "decode", 1, 0),
        (16, "decode", 0, 1),
    }


def test_population_is_set_membership_without_pairing_or_truncation(tmp_path):
    clean_path = tmp_path / "clean.csv"
    profiled_path = tmp_path / "profiled.csv"
    _write_phase_csv(
        clean_path,
        [
            {"counter_id": "1", "decode_requests": "4", "decode_kv_tokens": "33802"},
            {"counter_id": "2", "decode_requests": "4", "decode_kv_tokens": "33804"},
            {"counter_id": "3", "decode_requests": "1", "decode_kv_tokens": "99"},
        ],
    )
    _write_phase_csv(
        profiled_path,
        [{"counter_id": "9", "decode_requests": "4", "decode_kv_tokens": "33803"}],
    )
    samples = load_fpm_phase_csv(clean_path, lane="clean", concurrency=16)
    samples += load_fpm_phase_csv(profiled_path, lane="profiled", concurrency=16)

    population = build_semantic_population(
        samples,
        configuration_fingerprint="cfg",
        measured_segments=frozenset({"real"}),
    )
    shared = next(bin_ for bin_ in population.bins if bin_.nsight_shared)
    clean_only = next(bin_ for bin_ in population.bins if bin_.clean_only)

    assert (shared.n_clean, shared.n_profiled) == (2, 1)
    assert shared.clean_raw_shape_cardinality == 2
    assert shared.profiled_raw_shape_cardinality == 1
    assert [sample.counter_id for sample in shared.clean_samples] == [1, 2]
    assert [sample.counter_id for sample in shared.profiled_samples] == [9]
    assert (clean_only.n_clean, clean_only.n_profiled) == (1, 0)


def test_population_order_is_deterministic_for_permuted_equal_counter_inputs(tmp_path):
    worker_a = tmp_path / "worker_a.csv"
    worker_b = tmp_path / "worker_b.csv"
    _write_phase_csv(worker_a, [{"counter_id": "7", "worker_id": "a", "decode_kv_tokens": "100"}])
    _write_phase_csv(worker_b, [{"counter_id": "7", "worker_id": "b", "decode_kv_tokens": "100"}])
    samples = load_fpm_phase_csv(worker_b, lane="clean", concurrency=16)
    samples += load_fpm_phase_csv(worker_a, lane="clean", concurrency=16)

    forward = build_semantic_population(
        samples,
        configuration_fingerprint="cfg",
        measured_segments=frozenset({"real"}),
    )
    reverse = build_semantic_population(
        reversed(samples),
        configuration_fingerprint="cfg",
        measured_segments=frozenset({"real"}),
    )

    assert forward == reverse
    assert [sample.worker_id for sample in forward.samples] == ["a", "b"]
    assert [sample.worker_id for sample in forward.bins[0].clean_samples] == ["a", "b"]


@pytest.mark.parametrize(
    ("count", "expected"),
    [(1, "singleton"), (2, "sparse"), (4, "sparse"), (5, "repeated")],
)
def test_support_class(count, expected):
    assert support_class(count) == expected


def test_descriptive_stats_use_linear_quantiles_and_blank_sparse_quartiles():
    singleton = descriptive_stats([4.0])
    sparse = descriptive_stats([1.0, 3.0, 7.0, 9.0])
    repeated = descriptive_stats([1.0, 3.0, 5.0, 7.0, 9.0])

    assert singleton == {
        "count": 1,
        "support_class": "singleton",
        "minimum": 4.0,
        "q1": None,
        "median": 4.0,
        "q3": None,
        "maximum": 4.0,
    }
    assert sparse["median"] == 5.0
    assert sparse["q1"] is None and sparse["q3"] is None
    assert repeated["q1"] == 3.0
    assert repeated["median"] == 5.0
    assert repeated["q3"] == 7.0
