# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact DSv4 V1.2 SGLang CustomAllReduce request/result adapter."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.resolution.types import MeasurementRecord, MeasurementRequest, PerfKey

_NAMESPACE = f"{PerfFile.CUSTOM_ALLREDUCE}/v1"
_QUERY_FIELDS = ("dtype", "operation", "world_size", "elements")
_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
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
_SEMANTIC_DESCRIPTOR = {
    "operation": "all_reduce",
    "implementation": "sglang_custom_allreduce",
    "mode": "graph",
}
_PHYSICAL_BYTES_PER_ELEMENT = 2
_MAX_CUSTOM_ALLREDUCE_BYTES = 8 * 1024 * 1024


def _normalized_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _validate_capability(request: MeasurementRequest, case: Mapping[str, Any]) -> None:
    environment = request.environment
    if (
        environment.system != "gb200"
        or environment.backend != "sglang"
        or environment.backend_version != "0.5.10"
        or _normalized_label(environment.gpu_class) != "nvidia gb200"
        or any(environment.runtime_versions.get(name) != version for name, version in _RUNTIME_VERSIONS.items())
    ):
        raise ValueError("CustomAllReduce request is outside the frozen GB200/SGLang 0.5.10 runtime capability")
    if dict(environment.profile_compatibility or {}) != _PROFILE_COMPATIBILITY:
        raise ValueError("CustomAllReduce profile compatibility does not match the frozen DSv4 V1.2 deployment")
    if (
        not isinstance(environment.topology_schema, str)
        or not environment.topology_schema.strip()
        or not isinstance(environment.topology_fingerprint, str)
        or not environment.topology_fingerprint.strip()
    ):
        raise ValueError("CustomAllReduce requires an explicit NVLink topology identity")
    if dict(request.semantic_descriptor) != _SEMANTIC_DESCRIPTOR:
        raise ValueError("CustomAllReduce semantic descriptor must identify the SGLang graph implementation")
    if request.protocol.samples < 3:
        raise ValueError("CustomAllReduce measurement protocol samples must be at least three")
    if case["dtype"] != "half":
        raise ValueError("CustomAllReduce frozen capability requires half dtype")
    if case["world_size"] != 4:
        raise ValueError("CustomAllReduce frozen capability requires world size four (4)")
    physical_bytes = case["element_count"] * _PHYSICAL_BYTES_PER_ELEMENT
    if physical_bytes % 16:
        raise ValueError("CustomAllReduce physical byte size must be a multiple of 16")
    if physical_bytes > _MAX_CUSTOM_ALLREDUCE_BYTES:
        raise ValueError("CustomAllReduce element count exceeds the SGLang 8 MiB maximum")


def custom_allreduce_request_to_case(request: MeasurementRequest) -> dict[str, Any]:
    """Translate one canonical key to one exact four-rank runner case."""

    if not isinstance(request, MeasurementRequest):
        raise TypeError("request must be a MeasurementRequest")
    if request.key.namespace != _NAMESPACE:
        raise ValueError(f"CustomAllReduce request namespace must be {_NAMESPACE!r}")
    if set(request.query) != set(_QUERY_FIELDS):
        raise ValueError(f"CustomAllReduce query fields must be exactly {_QUERY_FIELDS!r}")
    dtype = request.query["dtype"]
    operation = request.query["operation"]
    world_size = request.query["world_size"]
    elements = request.query["elements"]
    if operation != "all_reduce":
        raise ValueError("CustomAllReduce operation must be all_reduce")
    for name, value in (("world_size", world_size), ("elements", elements)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"CustomAllReduce {name} must be a positive integer")
    if request.key != PerfKey.build(_NAMESPACE, dict(request.query), request.environment):
        raise ValueError("CustomAllReduce request PerfKey does not match its canonical query and environment")
    case = {
        "dtype": dtype,
        "world_size": world_size,
        "element_count": elements,
    }
    _validate_capability(request, case)
    return case


def custom_allreduce_resource_for_request(
    request: MeasurementRequest,
    case: Mapping[str, Any],
) -> ResourceContract:
    """Reserve the complete four-GPU NVLink fabric domain."""

    expected = custom_allreduce_request_to_case(request)
    if dict(case) != expected:
        raise ValueError("CustomAllReduce case does not match the canonical request query")
    return ResourceContract(
        gpu_count=4,
        fabric=FabricRequirement.NVLINK,
        reserve_fabric_domain=True,
    )


def _positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"CustomAllReduce {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"CustomAllReduce {field} must be positive and finite")
    return value


def custom_allreduce_result_to_record(
    request: MeasurementRequest,
    case: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> MeasurementRecord:
    """Validate one correlated four-rank result before overlay persistence."""

    expected = custom_allreduce_request_to_case(request)
    if dict(case) != expected:
        raise ValueError("CustomAllReduce case does not match the canonical request query")
    if not isinstance(raw_result, Mapping):
        raise TypeError("raw CustomAllReduce result must be a mapping")

    latency_ms = _positive_finite_number(raw_result.get("latency_ms"), field="latency")
    energy_value = raw_result.get("energy_wms", 0.0)
    if isinstance(energy_value, bool) or not isinstance(energy_value, (int, float)):
        raise TypeError("CustomAllReduce energy must be numeric")
    energy_wms = float(energy_value)
    if not math.isfinite(energy_wms) or energy_wms < 0:
        raise ValueError("CustomAllReduce energy must be finite and non-negative")

    raw_samples = raw_result.get("samples_ms")
    if not isinstance(raw_samples, (list, tuple)):
        raise TypeError("CustomAllReduce samples must be a numeric sequence")
    samples_ms = tuple(_positive_finite_number(sample, field="sample") for sample in raw_samples)
    if len(samples_ms) != request.protocol.samples:
        raise ValueError("CustomAllReduce sample count does not match the request protocol")
    if statistics.median(samples_ms) != latency_ms:
        raise ValueError("CustomAllReduce latency must equal the median of its samples")
    if raw_result.get("statistic") != request.protocol.statistic:
        raise ValueError("CustomAllReduce statistic does not match the request protocol")
    if raw_result.get("protocol_digest") != request.protocol.digest:
        raise ValueError("CustomAllReduce protocol identity does not match the request protocol")

    perf_row = raw_result.get("perf_row")
    if not isinstance(perf_row, Mapping):
        raise TypeError("CustomAllReduce result must contain a perf row mapping")
    expected_identity = {
        "allreduce_dtype": expected["dtype"],
        "num_gpus": expected["world_size"],
        "message_size": expected["element_count"],
    }
    if {field: perf_row.get(field) for field in expected_identity} != expected_identity:
        raise ValueError("CustomAllReduce perf row identity does not match the requested world/GPU shape")
    if perf_row.get("framework") != "SGLang" or perf_row.get("version") != request.environment.backend_version:
        raise ValueError("CustomAllReduce perf row framework identity does not match SGLang 0.5.10")
    if perf_row.get("device") is None or _normalized_label(str(perf_row["device"])) != _normalized_label(
        request.environment.gpu_class
    ):
        raise ValueError("CustomAllReduce perf row device does not match the request GPU")
    if perf_row.get("op_name") != "all_reduce":
        raise ValueError("CustomAllReduce perf row operation identity is unsupported")
    if perf_row.get("kernel_source") != "SGLang_CustomAllReduce_graph" or perf_row.get("backend") != "sglang_graph":
        raise ValueError("CustomAllReduce perf row must identify the SGLang CUDA Graph kernel")
    if _positive_finite_number(perf_row.get("latency"), field="perf row latency") != latency_ms:
        raise ValueError("CustomAllReduce perf row latency does not match measured latency")

    provenance = raw_result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("CustomAllReduce provenance must be a mapping")
    if provenance.get("framework") != "SGLang":
        raise ValueError("CustomAllReduce provenance framework must be SGLang")
    if provenance.get("framework_version") != request.environment.backend_version:
        raise ValueError("CustomAllReduce provenance framework version does not match the request runtime")
    measured_device = provenance.get("device")
    if not isinstance(measured_device, str) or _normalized_label(measured_device) != _normalized_label(
        request.environment.gpu_class
    ):
        raise ValueError("CustomAllReduce provenance device does not match the request GPU class")
    if provenance.get("kernel_source") != "SGLang_CustomAllReduce_graph":
        raise ValueError("CustomAllReduce provenance kernel source is unsupported")
    if provenance.get("runtime") != "persistent_sglang_custom_allreduce":
        raise ValueError("CustomAllReduce provenance runtime is unsupported")
    if provenance.get("used_cuda_graph") is not True:
        raise ValueError("CustomAllReduce measurement must use CUDA Graph capture")
    if provenance.get("throttled") is not False:
        raise ValueError("CustomAllReduce measurement must not be thermally throttled")
    if provenance.get("world_size") != 4:
        raise ValueError("CustomAllReduce provenance world size must be four")
    rank_pids = provenance.get("rank_pids")
    if (
        not isinstance(rank_pids, (list, tuple))
        or len(rank_pids) != 4
        or any(isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in rank_pids)
        or len(set(rank_pids)) != 4
    ):
        raise ValueError("CustomAllReduce provenance must contain four unique rank PIDs")
    if provenance.get("model_artifact") != _MODEL_ARTIFACT:
        raise ValueError("CustomAllReduce provenance model artifact does not match the frozen profile")

    return MeasurementRecord.valid(
        key=request.key,
        latency_ms=latency_ms,
        energy_wms=energy_wms,
        samples_ms=samples_ms,
        protocol=request.protocol,
        perf_row=dict(perf_row),
        provenance=dict(provenance),
    )


__all__ = [
    "custom_allreduce_request_to_case",
    "custom_allreduce_resource_for_request",
    "custom_allreduce_result_to_record",
]
