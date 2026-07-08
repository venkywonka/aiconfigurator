# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.dsv4 import DeepSeekV4MHCModule
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    PerfKey,
)

_PROFILE_COMPATIBILITY = {
    "model_artifact": "sgl-project/DeepSeek-V4-Flash-FP8",
    "serving_mode": "aggregated",
    "tp_size": 4,
    "attention_dp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 4,
    "nextn": 0,
}


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions={
            "cuda": "13.0",
            "model_profile": "dsv4-v1.2",
            "sglang": "0.5.10",
        },
        topology_schema="nvidia-smi-v1",
        topology_fingerprint="gb200-nvlink4",
        profile_compatibility=_PROFILE_COMPATIBILITY,
    )


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-mhc-v1",
    )


class _Database:
    def __init__(self) -> None:
        self.system = "gb200"
        self.backend = "sglang"
        self.version = "0.5.10"
        self.measurement_environment = _environment()
        self.queries: list[dict[str, Any]] = []

    def query_mhc_module(self, **query: Any) -> PerformanceResult:
        self.queries.append(query)
        return PerformanceResult(1.0, energy=2.0, source="silicon")


def _operation(
    *,
    op: str = "pre",
    scale_factor: float = 3.0,
    seq_split: int = 1,
) -> DeepSeekV4MHCModule:
    return DeepSeekV4MHCModule(
        "context_mhc",
        scale_factor,
        op,
        4096,
        4,
        20,
        common.GEMMQuantMode.bfloat16,
        seq_split=seq_split,
    )


@pytest.mark.parametrize("op_name", ["pre", "post"])
def test_one_mhc_normalization_feeds_ordinary_query_and_exact_request(op_name: str) -> None:
    database = _Database()
    operation = _operation(op=op_name)
    expected = {
        "op": op_name,
        "num_tokens": 17,
        "hidden_size": 4096,
        "hc_mult": 4,
        "sinkhorn_iters": 20,
        "quant_mode": "bfloat16",
    }

    normalized = operation.normalize_perf_query(x=17)
    ordinary = operation.query(database, x=17)
    request = operation.measurement_request(database, _protocol(), x=17)

    assert normalized == expected
    assert database.queries == [
        {
            "num_tokens": 17,
            "hidden_size": 4096,
            "hc_mult": 4,
            "sinkhorn_iters": 20,
            "op": op_name,
            "quant_mode": common.GEMMQuantMode.bfloat16,
        }
    ]
    assert float(ordinary) == pytest.approx(3.0)
    assert ordinary.energy == pytest.approx(6.0)
    assert request is not None
    assert request.query == expected
    assert request.key == PerfKey.build(f"{PerfFile.MHC_MODULE}/v1", expected, database.measurement_environment)
    assert request.semantic_descriptor == {
        "num_sites": 2,
        "tensor_generator": "normal-v1",
        "seed": 0,
    }


def test_mhc_scale_and_consumer_identity_are_not_physical_identity() -> None:
    database = _Database()
    first = _operation(scale_factor=1.0)
    second = _operation(scale_factor=43.0)
    second._name = "another_consumer"

    first_request = first.measurement_request(database, _protocol(), x=17)
    second_request = second.measurement_request(database, _protocol(), x=17)

    assert first_request is not None and second_request is not None
    assert first_request.op_id != second_request.op_id
    assert first_request.key == second_request.key


@pytest.mark.parametrize("x", [True, 0, -1, 1.5, "17", None])
def test_mhc_normalization_rejects_non_positive_integer_tokens(x: object) -> None:
    with pytest.raises((TypeError, ValueError), match="mHC x"):
        _operation().normalize_perf_query(x=x)


def test_combined_mhc_operation_has_no_single_physical_request() -> None:
    operation = _operation(op="both")

    assert operation.measurement_request(_Database(), _protocol(), x=17) is None


def test_cp_sharding_normalizes_for_pure_prediction_but_is_not_a_v1_2_measurement() -> None:
    operation = _operation(seq_split=4)

    assert operation.normalize_perf_query(x=17)["num_tokens"] == 5
    with pytest.raises(ValueError, match=r"CP1|seq_split|profile"):
        operation.measurement_request(_Database(), _protocol(), x=17)


def test_mhc_literal_curated_row_hits_but_unmeasured_point_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    database = _Database()
    database._mhc_module_data = {
        "pre": {
            4: {
                4096: {
                    17: {
                        "latency": 1.25,
                        "energy": 12.5,
                    }
                }
            }
        }
    }
    monkeypatch.setattr(DeepSeekV4MHCModule, "load_data", classmethod(lambda cls, database: None))
    operation = _operation(scale_factor=3.0)

    exact = operation.curated_exact_result(database, x=17)
    absent = operation.curated_exact_result(database, x=21)
    cp_sharded = _operation(seq_split=4).curated_exact_result(database, x=65)

    assert exact is not None
    assert float(exact) == pytest.approx(3.75)
    assert exact.energy == pytest.approx(37.5)
    assert exact.source == "curated_exact"
    assert absent is None
    assert cp_sharded is None


def test_mhc_missing_curated_dataset_is_an_exact_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    database = _Database()
    database._mhc_module_data = LoadedOpData(
        None,
        PerfDataFilename.mhc_module,
        "/missing/mhc_module_perf.txt",
    )
    monkeypatch.setattr(DeepSeekV4MHCModule, "load_data", classmethod(lambda cls, database: None))

    assert _operation().curated_exact_result(database, x=17) is None
