# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 physical MoE normalization and exact-evidence contracts."""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import Any

import pytest

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations.moe import MoE
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    PerfKey,
)

pytestmark = pytest.mark.unit

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_NAMESPACE = f"{PerfFile.MOE}/v1"
_PROFILE_COMPATIBILITY = {
    "model_artifact": _MODEL_ARTIFACT,
    "serving_mode": "aggregated",
    "tp_size": 4,
    "attention_dp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 4,
    "nextn": 0,
}
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
_SEMANTIC_DESCRIPTOR = {
    "workload_generator": "power_law_v3",
    "seed": 0,
    "rank_simulation": "single-gpu-ep4-rank0",
}


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions=_RUNTIME_VERSIONS,
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
        tuning_revision="sglang-moe-v1",
    )


def _expected_query(num_tokens: int = 19) -> dict[str, object]:
    return {
        "num_tokens": num_tokens,
        "hidden_size": 4096,
        "inter_size": 2048,
        "topk": 6,
        "num_experts": 256,
        "moe_tp_size": 1,
        "moe_ep_size": 4,
        "quant_mode": "fp8_block",
        "workload_distribution": "power_law_1.01",
    }


class _ProfileDatabase:
    def __init__(self) -> None:
        self.system = "gb200"
        self.backend = "sglang"
        self.version = "0.5.10"
        self.measurement_environment = _environment()
        self.queries: list[dict[str, Any]] = []

    def query_moe(self, **query: Any) -> PerformanceResult:
        self.queries.append(query)
        return PerformanceResult(1.0, energy=2.0, source="silicon")


@pytest.fixture(scope="module")
def dsv4_profile_moe() -> MoE:
    model_config = config.ModelConfig(
        tp_size=4,
        pp_size=1,
        attention_dp_size=1,
        cp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        nextn=0,
        workload_distribution="power_law",
        moe_backend=None,
    )
    model = get_model(_MODEL_ARTIFACT, model_config, backend_name="sglang")
    return next(operation for operation in model.context_ops if isinstance(operation, MoE))


def _nested_exact_data(num_tokens: int = 19) -> LoadedOpData:
    data = {
        common.MoEQuantMode.fp8_block: {
            "power_law_1.01": {
                6: {
                    256: {
                        4096: {
                            2048: {
                                1: {
                                    4: {
                                        num_tokens: {
                                            "latency": 1.25,
                                            "power": 10.0,
                                            "energy": 12.5,
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return LoadedOpData(data, PerfDataFilename.moe, "/frozen/gb200/sglang/0.5.10/moe_perf.parquet")


def test_actual_dsv4_moe_runtime_input_feeds_one_physical_query_and_request(dsv4_profile_moe: MoE) -> None:
    database = _ProfileDatabase()
    expected = _expected_query()

    normalized = dsv4_profile_moe.normalize_perf_query(x=19)
    ordinary = dsv4_profile_moe.query(database, x=19)
    request = dsv4_profile_moe.measurement_request(database, _protocol(), x=19)

    assert dsv4_profile_moe._scale_factor == 43
    assert normalized == expected
    assert database.queries == [
        {
            "num_tokens": 19,
            "hidden_size": 4096,
            "inter_size": 2048,
            "topk": 6,
            "num_experts": 256,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "quant_mode": common.MoEQuantMode.fp8_block,
            "workload_distribution": "power_law_1.01",
            "is_context": True,
            "moe_backend": None,
            "is_gated": True,
            "enable_eplb": False,
        }
    ]
    assert float(ordinary) == pytest.approx(43.0)
    assert ordinary.energy == pytest.approx(86.0)
    assert request is not None
    assert request.query == expected
    assert request.key == PerfKey.build(_NAMESPACE, expected, database.measurement_environment)
    assert request.semantic_descriptor == _SEMANTIC_DESCRIPTOR
    assert request.environment.profile_compatibility == _PROFILE_COMPATIBILITY


def test_moe_phase_scale_and_consumer_identity_are_not_physical_identity(dsv4_profile_moe: MoE) -> None:
    database = _ProfileDatabase()
    context = copy(dsv4_profile_moe)
    generation = copy(dsv4_profile_moe)
    generation._name = "generation_moe"
    generation._scale_factor = 1.0
    generation._is_context = False

    context_request = context.measurement_request(database, _protocol(), x=19)
    generation_request = generation.measurement_request(database, _protocol(), x=19)

    assert context_request is not None and generation_request is not None
    assert context_request.op_id != generation_request.op_id
    assert context_request.key == generation_request.key
    assert context_request.query == generation_request.query == _expected_query()
    assert {"phase", "is_context", "scale_factor", "num_layers", "model_artifact"}.isdisjoint(context_request.query)


@pytest.mark.parametrize("x", [True, 0, -1, 1.5, "19", None])
def test_moe_normalization_rejects_non_positive_integer_tokens(dsv4_profile_moe: MoE, x: object) -> None:
    with pytest.raises((TypeError, ValueError), match=r"MoE|token|positive integer"):
        dsv4_profile_moe.normalize_perf_query(x=x)


def test_literal_curated_moe_row_is_reused_without_phase_or_scale_in_identity(
    dsv4_profile_moe: MoE,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ProfileDatabase()
    database._moe_data = _nested_exact_data()
    database._moe_low_latency_data = LoadedOpData(None, PerfDataFilename.moe, "/missing/low_latency.parquet")
    monkeypatch.setattr(MoE, "load_data", classmethod(lambda cls, database: None))

    result = dsv4_profile_moe.curated_exact_result(database, x=19)

    assert result is not None
    assert float(result) == pytest.approx(1.25 * 43)
    assert result.energy == pytest.approx(12.5 * 43)
    assert result.source == "curated_exact"


@pytest.mark.parametrize(
    ("runtime_name", "runtime_version"),
    [
        pytest.param("cuda", "12.9", id="cuda"),
        pytest.param("sglang", "0.5.11", id="sglang"),
        pytest.param("model_profile", "dsv4-v1.3", id="model-profile"),
    ],
)
def test_literal_curated_moe_row_rejects_frozen_runtime_mismatch(
    dsv4_profile_moe: MoE,
    monkeypatch: pytest.MonkeyPatch,
    runtime_name: str,
    runtime_version: str,
) -> None:
    database = _ProfileDatabase()
    database._moe_data = _nested_exact_data()
    database._moe_low_latency_data = LoadedOpData(None, PerfDataFilename.moe, "/missing/low_latency.parquet")
    monkeypatch.setattr(MoE, "load_data", classmethod(lambda cls, database: None))
    if runtime_name == "sglang":
        database.version = runtime_version
        database.measurement_environment = replace(
            database.measurement_environment,
            backend_version=runtime_version,
            runtime_versions={**_RUNTIME_VERSIONS, runtime_name: runtime_version},
        )
    else:
        database.measurement_environment = replace(
            database.measurement_environment,
            runtime_versions={**_RUNTIME_VERSIONS, runtime_name: runtime_version},
        )

    assert dsv4_profile_moe.curated_exact_result(database, x=19) is None


def test_moe_stable_curated_profile_and_rc0_measurement_use_distinct_keys(
    dsv4_profile_moe: MoE,
) -> None:
    database = _ProfileDatabase()
    stable_request = dsv4_profile_moe.measurement_request(database, _protocol(), x=19)
    stable_environment = database.measurement_environment
    database.measurement_environment = replace(
        stable_environment,
        backend_version="0.5.10rc0",
        runtime_versions={**stable_environment.runtime_versions, "sglang": "0.5.10rc0"},
    )

    rc0_request = dsv4_profile_moe.measurement_request(database, _protocol(), x=19)

    assert database.version == "0.5.10"
    assert stable_request is not None and rc0_request is not None
    assert stable_request.key != rc0_request.key
    assert stable_request.environment.backend_version == "0.5.10"
    assert rc0_request.environment.backend_version == "0.5.10rc0"


@pytest.mark.parametrize(
    ("environment_overrides", "profile_overrides"),
    [
        pytest.param({"system": "h100_sxm"}, {}, id="system"),
        pytest.param({"gpu_class": "NVIDIA H100"}, {}, id="gpu-class"),
        pytest.param({}, {"serving_mode": "disaggregated"}, id="serving-mode"),
        pytest.param({}, {"tp_size": 8}, id="tp-size"),
        pytest.param({}, {"attention_dp_size": 2}, id="attention-dp-size"),
        pytest.param({}, {"moe_ep_size": 8}, id="moe-ep-size"),
    ],
)
def test_literal_curated_moe_row_rejects_frozen_hardware_or_profile_mismatch(
    dsv4_profile_moe: MoE,
    environment_overrides: dict[str, object],
    profile_overrides: dict[str, object],
) -> None:
    database = _ProfileDatabase()
    database._moe_data = _nested_exact_data()
    environment = database.measurement_environment
    if profile_overrides:
        environment = replace(
            environment,
            profile_compatibility={**_PROFILE_COMPATIBILITY, **profile_overrides},
        )
    database.measurement_environment = replace(environment, **environment_overrides)

    assert dsv4_profile_moe.curated_exact_result(database, x=19) is None


def test_curated_moe_probe_never_interpolates_or_falls_back_to_uniform(
    dsv4_profile_moe: MoE,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ProfileDatabase()
    database._moe_data = _nested_exact_data(num_tokens=18)
    database._moe_low_latency_data = LoadedOpData(None, PerfDataFilename.moe, "/missing/low_latency.parquet")
    monkeypatch.setattr(MoE, "load_data", classmethod(lambda cls, database: None))

    assert dsv4_profile_moe.curated_exact_result(database, x=19) is None

    uniform = _nested_exact_data().data[common.MoEQuantMode.fp8_block].pop("power_law_1.01")
    database._moe_data.data[common.MoEQuantMode.fp8_block]["uniform"] = uniform
    assert dsv4_profile_moe.curated_exact_result(database, x=19) is None
