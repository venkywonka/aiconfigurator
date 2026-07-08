# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import sys

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.trtllm import gemm_adapter
from aiconfigurator.collector.trtllm.gemm_adapter import (
    gemm_request_to_case,
    gemm_resource_for_request,
    gemm_result_to_record,
)
from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC, TRTLLM_LAZY_REGISTRY
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
)


def _protocol(*, samples: int = 3, statistic: str = "median") -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=samples,
        statistic=statistic,
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )


def _request(
    *,
    gemm_type: str = "bfloat16",
    samples: int = 3,
    statistic: str = "median",
) -> MeasurementRequest:
    environment = MeasurementEnvironment(
        system="gb200",
        backend="trtllm",
        backend_version="1.3.0rc10",
        gpu_class="NVIDIA GB200",
        runtime_versions={"cuda": "13.0", "tensorrt_llm": "1.3.0rc10"},
        topology_schema="nvidia-smi-v1",
        topology_fingerprint="topology-a",
    )
    query = {"gemm_type": gemm_type, "m": 8, "n": 16, "k": 32}
    return MeasurementRequest(
        op_id="router-gemm",
        key=PerfKey.build(f"{PerfFile.GEMM}/v1", query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"tensor_generator": "normal-v1", "seed": 0},
        protocol=_protocol(samples=samples, statistic=statistic),
    )


def _raw_result(request: MeasurementRequest) -> dict[str, object]:
    samples = (1.2, 1.25, 1.3)
    return {
        "latency_ms": 1.25,
        "energy_wms": 12.5,
        "samples_ms": samples,
        "statistic": "median",
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "gemm_dtype": request.query["gemm_type"],
            "m": request.query["m"],
            "n": request.query["n"],
            "k": request.query["k"],
            "latency": 1.25,
        },
        "provenance": {
            "framework": "TRTLLM",
            "framework_version": "1.3.0rc10",
            "device": "NVIDIA GB200",
            "used_cuda_graph": True,
        },
    }


def test_packaged_registry_uses_canonical_perf_namespace_and_lightweight_adapter() -> None:
    assert GEMM_LAZY_SPEC.namespace == f"{PerfFile.GEMM}/v1"
    assert GEMM_LAZY_SPEC.adapter_module == "aiconfigurator.collector.trtllm.gemm_adapter"
    assert GEMM_LAZY_SPEC.run_module == "aiconfigurator.collector.trtllm.gemm"
    assert len(TRTLLM_LAZY_REGISTRY) == 1
    assert TRTLLM_LAZY_REGISTRY[0].lazy is GEMM_LAZY_SPEC
    assert TRTLLM_LAZY_REGISTRY[0].perf_filename == PerfFile.GEMM
    assert "torch" not in sys.modules
    assert "tensorrt_llm" not in sys.modules


def test_bf16_request_maps_to_one_gpu_case_without_reinterpreting_shape() -> None:
    request = _request()

    case = gemm_request_to_case(request)

    assert case == {"gemm_type": "bfloat16", "m": 8, "n": 16, "k": 32}
    assert gemm_resource_for_request(request, case) == ResourceContract(
        gpu_count=1,
        fabric=FabricRequirement.NONE,
    )


@pytest.mark.parametrize("gemm_type", ["fp8", "fp8_block", "nvfp4", "int8_wo"])
def test_v1_generic_pilot_rejects_non_bf16_before_resource_acquisition(gemm_type: str) -> None:
    request = _request(gemm_type=gemm_type)

    with pytest.raises(ValueError, match=r"bfloat16|BF16"):
        gemm_request_to_case(request)


def test_raw_result_round_trips_identical_query_key_and_protocol() -> None:
    request = _request()
    case = gemm_request_to_case(request)

    record = gemm_result_to_record(request, case, _raw_result(request))

    assert record.status is RecordStatus.VALID
    assert record.key == request.key
    assert record.protocol == request.protocol
    assert record.latency_ms == 1.25
    assert record.energy_wms == 12.5
    assert record.samples_ms == (1.2, 1.25, 1.3)
    assert record.perf_row == {
        "gemm_dtype": "bfloat16",
        "m": 8,
        "n": 16,
        "k": 32,
        "latency": 1.25,
    }


@pytest.mark.parametrize("invalid_sample", (True, "1.25"))
def test_gemm_result_rejects_non_numeric_sample_types_before_coercion(invalid_sample: object) -> None:
    request = _request()
    case = gemm_request_to_case(request)
    raw = _raw_result(request)
    raw["samples_ms"] = (1.2, invalid_sample, 1.3)

    with pytest.raises(TypeError, match="samples"):
        gemm_result_to_record(request, case, raw)


def test_gemm_result_requires_latency_to_equal_sample_median() -> None:
    request = _request()
    case = gemm_request_to_case(request)
    raw = _raw_result(request)
    raw["latency_ms"] = 1.2
    raw["perf_row"] = dict(raw["perf_row"], latency=1.2)

    with pytest.raises(ValueError, match=r"median|samples"):
        gemm_result_to_record(request, case, raw)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("framework_version", "1.2.0", r"framework|version|runtime"),
        ("device", "NVIDIA H100", r"device|GPU|gpu"),
    ],
)
def test_gemm_result_provenance_must_match_request_environment(
    field: str,
    value: str,
    match: str,
) -> None:
    request = _request()
    case = gemm_request_to_case(request)
    raw = _raw_result(request)
    raw["provenance"] = dict(raw["provenance"], **{field: value})

    with pytest.raises(ValueError, match=match):
        gemm_result_to_record(request, case, raw)


def test_gemm_route_rejects_non_median_statistic_before_resource_construction(monkeypatch) -> None:
    resource_calls: list[object] = []

    def _resource(request, case):
        resource_calls.append((request, case))
        return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    monkeypatch.setattr(gemm_adapter, "gemm_resource_for_request", _resource)
    route = LazyAdapterIndex.from_registries({"trtllm": TRTLLM_LAZY_REGISTRY}).routes_for(
        (GEMM_LAZY_SPEC.namespace, "trtllm", "1.3.0rc10")
    )[0]

    with pytest.raises(ValueError, match=r"median|statistic"):
        route.prepare(_request(statistic="mean"))
    assert resource_calls == []


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"latency_ms": 0.0}, "latency"),
        ({"latency_ms": math.inf}, "latency"),
        ({"samples_ms": (1.2, 1.3)}, "sample"),
        ({"samples_ms": "111"}, "sequence"),
        ({"statistic": "mean"}, "statistic"),
        ({"protocol_digest": "wrong-protocol"}, "protocol"),
        ({"perf_row": {"gemm_dtype": "bfloat16", "m": 9, "n": 16, "k": 32, "latency": 1.25}}, "row|query"),
    ],
)
def test_result_adapter_rejects_invalid_measurement_identity(
    mutation: dict[str, object],
    match: str,
) -> None:
    request = _request()
    case = gemm_request_to_case(request)
    raw = _raw_result(request)
    raw.update(mutation)

    with pytest.raises((TypeError, ValueError), match=match):
        gemm_result_to_record(request, case, raw)
