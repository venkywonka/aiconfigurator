# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 physical GEMM adapter contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
from aiconfigurator.collector.trtllm import gemm_adapter
from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk import common, config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)
from collector.sglang.registry import REGISTRY as SGLANG_REGISTRY

pytestmark = pytest.mark.unit

_NAMESPACE = f"{PerfFile.GEMM}/v1"
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
    "tensorrt_llm": "1.3.0rc10",
}


class _ProfileDatabase:
    def __init__(self, environment: MeasurementEnvironment) -> None:
        self.system = environment.system
        self.backend = environment.backend
        self.version = environment.backend_version
        self.measurement_environment = environment


def _environment(**overrides: Any) -> MeasurementEnvironment:
    values: dict[str, Any] = {
        "system": "gb200",
        "backend": "sglang",
        "backend_version": "0.5.10",
        "gpu_class": "NVIDIA GB200",
        "runtime_versions": _RUNTIME_VERSIONS,
        "topology_schema": "nvidia-smi-v1",
        "topology_fingerprint": "gb200-nvlink4",
    }
    values.update(overrides)
    return MeasurementEnvironment(**values)


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )


@pytest.fixture(scope="module")
def dsv4_profile_model():
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
    return get_model(
        "sgl-project/DeepSeek-V4-Flash-FP8",
        model_config,
        backend_name="sglang",
    )


def _profile_operation(model, name: str) -> GEMM:
    operation = next(operation for operation in model.context_ops if operation._name == name)
    assert isinstance(operation, GEMM)
    return operation


def _request(
    operation: GEMM,
    *,
    environment: MeasurementEnvironment | None = None,
    x: int = 37,
) -> MeasurementRequest:
    request = operation.measurement_request(
        _ProfileDatabase(environment or _environment()),
        _protocol(),
        x=x,
    )
    assert request is not None
    return request


def _sglang_route():
    routes = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )
    assert len(routes) == 1
    return routes[0]


def _raw_result(request: MeasurementRequest, *, kernel_source: str) -> dict[str, object]:
    latency_ms = 1.25
    return {
        "latency_ms": latency_ms,
        "energy_wms": 12.5,
        "samples_ms": (1.2, latency_ms, 1.3),
        "statistic": request.protocol.statistic,
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "gemm_dtype": request.query["gemm_type"],
            "m": request.query["m"],
            "n": request.query["n"],
            "k": request.query["k"],
            "latency": latency_ms,
        },
        "provenance": {
            "framework": "TRTLLM",
            "framework_version": "1.3.0rc10",
            "kernel_source": kernel_source,
            "device": "NVIDIA GB200",
            "used_cuda_graph": True,
        },
    }


def test_sglang_registry_installs_one_packaged_gemm_route() -> None:
    gemm_entry = next(entry for entry in SGLANG_REGISTRY if entry.op == "gemm")

    assert gemm_entry.lazy is GEMM_LAZY_SPEC
    assert SGLANG_LAZY_REGISTRY[0].lazy is GEMM_LAZY_SPEC
    routes = LazyAdapterIndex.from_registries({"sglang": SGLANG_REGISTRY}).routes_for((_NAMESPACE, "sglang", "0.5.10"))
    assert len(routes) == 1
    assert routes[0].lazy.run_module == "aiconfigurator.collector.trtllm.gemm"


@pytest.mark.parametrize(
    ("operation_name", "expected_query", "kernel_source"),
    [
        (
            "context_router_gemm",
            {"gemm_type": "bfloat16", "m": 37, "n": 256, "k": 4096},
            "torch_flow",
        ),
        (
            "context_shared_gate_up_gemm",
            {"gemm_type": "fp8_block", "m": 37, "n": 1024, "k": 4096},
            "deepgemm",
        ),
        (
            "context_shared_ffn2_gemm",
            {"gemm_type": "fp8_block", "m": 37, "n": 4096, "k": 512},
            "deepgemm",
        ),
        (
            "context_logits_gemm",
            {"gemm_type": "bfloat16", "m": 37, "n": 32320, "k": 4096},
            "torch_flow",
        ),
    ],
)
def test_dsv4_runtime_query_round_trips_through_case_and_emitted_row(
    dsv4_profile_model,
    operation_name: str,
    expected_query: dict[str, object],
    kernel_source: str,
) -> None:
    operation = _profile_operation(dsv4_profile_model, operation_name)
    normalized = operation.normalize_perf_query(x=37)
    request = _request(operation)

    assert normalized == expected_query
    assert request.query == expected_query
    assert request.key == PerfKey.build(_NAMESPACE, expected_query, request.environment)

    route = _sglang_route()
    prepared = route.prepare(request)
    assert prepared.case == expected_query
    assert prepared.contract == ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    record = route.record(prepared, _raw_result(request, kernel_source=kernel_source))
    emitted_query = {
        "gemm_type": record.perf_row["gemm_dtype"],
        "m": record.perf_row["m"],
        "n": record.perf_row["n"],
        "k": record.perf_row["k"],
    }
    assert record.key == request.key == PerfKey.build(_NAMESPACE, emitted_query, request.environment)


def test_dsv4_scale_and_consumer_identity_do_not_change_physical_key(dsv4_profile_model) -> None:
    profile_operation = _profile_operation(dsv4_profile_model, "context_shared_gate_up_gemm")
    assert profile_operation._scale_factor == 43
    single_consumer = GEMM(
        "another_consumer",
        1,
        profile_operation._n,
        profile_operation._k,
        profile_operation._quant_mode,
    )

    profile_request = _request(profile_operation)
    single_request = _request(single_consumer)

    assert profile_request.op_id != single_request.op_id
    assert profile_request.key == single_request.key
    assert set(profile_request.query) == {"gemm_type", "m", "n", "k"}


@pytest.mark.parametrize(
    "mismatch",
    [
        "system",
        "gpu_class",
        "model_profile",
        "collector_runtime",
        "collector_runtime_version",
        "backend_version",
        "shape",
        "quantization",
    ],
)
def test_dsv4_capability_mismatches_fail_before_resource_acquisition(
    dsv4_profile_model,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
) -> None:
    environment = _environment()
    operation = _profile_operation(dsv4_profile_model, "context_router_gemm")
    if mismatch == "system":
        environment = replace(environment, system="h100_sxm")
    elif mismatch == "gpu_class":
        environment = replace(environment, gpu_class="NVIDIA H100")
    elif mismatch == "model_profile":
        environment = replace(
            environment,
            runtime_versions={**_RUNTIME_VERSIONS, "model_profile": "other-profile"},
        )
    elif mismatch == "collector_runtime":
        environment = replace(
            environment,
            runtime_versions={key: value for key, value in _RUNTIME_VERSIONS.items() if key != "tensorrt_llm"},
        )
    elif mismatch == "collector_runtime_version":
        environment = replace(
            environment,
            runtime_versions={**_RUNTIME_VERSIONS, "tensorrt_llm": "1.3.0rc9"},
        )
    elif mismatch == "backend_version":
        environment = replace(
            environment,
            backend_version="0.5.11",
            runtime_versions={**_RUNTIME_VERSIONS, "sglang": "0.5.11"},
        )
    elif mismatch == "shape":
        operation = GEMM("outside_profile", 1, 768, 4096, common.GEMMQuantMode.bfloat16)
    elif mismatch == "quantization":
        operation = GEMM("outside_profile", 1, 256, 4096, common.GEMMQuantMode.fp8)
    else:  # pragma: no cover - exhaustive over the parameter list
        raise AssertionError(mismatch)

    request = _request(operation, environment=environment)
    resource_calls: list[object] = []

    def _resource(request, case):
        resource_calls.append((request, case))
        return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    monkeypatch.setattr(gemm_adapter, "gemm_resource_for_request", _resource)
    route = _sglang_route()

    with pytest.raises(ValueError, match=r"capability|profile|backend|version|quantization|bfloat16"):
        route.prepare(request)
    assert resource_calls == []
