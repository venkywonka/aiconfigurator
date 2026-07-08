# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact DSv4 V1.2 SGLang MoE request and result adapter."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.resolution.types import MeasurementRecord, MeasurementRequest, PerfKey

_NAMESPACE = f"{PerfFile.MOE}/v1"
_QUERY_FIELDS = (
    "num_tokens",
    "hidden_size",
    "inter_size",
    "topk",
    "num_experts",
    "moe_tp_size",
    "moe_ep_size",
    "quant_mode",
    "workload_distribution",
)
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
    "workload_generator": "power_law_v3",
    "seed": 0,
    "rank_simulation": "single-gpu-ep4-rank0",
}


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
        raise ValueError("MoE request is outside the frozen GB200/SGLang 0.5.10 runtime capability envelope")
    if dict(environment.profile_compatibility or {}) != _PROFILE_COMPATIBILITY:
        raise ValueError("MoE request profile compatibility does not match the frozen DSv4 V1.2 deployment")
    if dict(request.semantic_descriptor) != _SEMANTIC_DESCRIPTOR:
        raise ValueError("MoE request semantic descriptor does not match the power_law rank simulation")
    if request.protocol.samples < 3:
        raise ValueError("MoE measurement protocol samples must be at least three")
    if case["hidden_size"] != 4096 or case["inter_size"] != 2048 or case["topk"] != 6 or case["num_experts"] != 256:
        raise ValueError("MoE shape is outside the frozen DSv4 V1.2 capability envelope")
    if case["moe_tp_size"] != 1:
        raise ValueError("MoE frozen capability requires TP1")
    if case["moe_ep_size"] != 4:
        raise ValueError("MoE frozen capability requires EP4")
    if case["quant_mode"] != "fp8_block":
        raise ValueError("MoE frozen capability requires fp8_block")
    if case["workload_distribution"] != "power_law_1.01":
        raise ValueError("MoE frozen capability requires power_law_1.01")


def moe_request_to_case(request: MeasurementRequest) -> dict[str, Any]:
    """Translate one canonical MoE key to one exact rank-local case."""

    if not isinstance(request, MeasurementRequest):
        raise TypeError("request must be a MeasurementRequest")
    if request.key.namespace != _NAMESPACE:
        raise ValueError(f"MoE request namespace must be {_NAMESPACE!r}")
    if set(request.query) != set(_QUERY_FIELDS):
        raise ValueError(f"MoE query fields must be exactly {_QUERY_FIELDS!r}")
    case = {field: request.query[field] for field in _QUERY_FIELDS}
    dimensions = (
        case["num_tokens"],
        case["hidden_size"],
        case["inter_size"],
        case["topk"],
        case["num_experts"],
        case["moe_tp_size"],
        case["moe_ep_size"],
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("MoE dimensions must be positive integers")
    if request.key != PerfKey.build(_NAMESPACE, case, request.environment):
        raise ValueError("MoE request PerfKey does not match its canonical query and environment")
    _validate_capability(request, case)
    return case


def moe_resource_for_request(
    request: MeasurementRequest,
    case: Mapping[str, Any],
) -> ResourceContract:
    """Reserve one exclusive GPU; EP4 is simulated locally without fabric."""

    if moe_request_to_case(request) != dict(case):
        raise ValueError("MoE case does not match the canonical request query")
    return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)


def _positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"MoE {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"MoE {field} must be positive and finite")
    return value


def moe_result_to_record(
    request: MeasurementRequest,
    case: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> MeasurementRecord:
    """Validate one exact rank-local result and preserve the request key."""

    if moe_request_to_case(request) != dict(case):
        raise ValueError("MoE case does not match the canonical request query")
    if not isinstance(raw_result, Mapping):
        raise TypeError("raw MoE result must be a mapping")

    latency_ms = _positive_finite_number(raw_result.get("latency_ms"), field="latency")
    energy_value = raw_result.get("energy_wms", 0.0)
    if isinstance(energy_value, bool) or not isinstance(energy_value, (int, float)):
        raise TypeError("MoE energy must be numeric")
    energy_wms = float(energy_value)
    if not math.isfinite(energy_wms) or energy_wms < 0:
        raise ValueError("MoE energy must be finite and non-negative")

    raw_samples = raw_result.get("samples_ms")
    if not isinstance(raw_samples, (list, tuple)):
        raise TypeError("MoE samples must be a numeric sequence")
    samples_ms = tuple(_positive_finite_number(sample, field="sample") for sample in raw_samples)
    if len(samples_ms) != request.protocol.samples:
        raise ValueError("MoE sample count does not match the request protocol")
    if statistics.median(samples_ms) != latency_ms:
        raise ValueError("MoE latency must equal the median of its samples")
    if raw_result.get("statistic") != request.protocol.statistic:
        raise ValueError("MoE statistic does not match the request protocol")
    if raw_result.get("protocol_digest") != request.protocol.digest:
        raise ValueError("MoE protocol identity does not match the request protocol")

    perf_row = raw_result.get("perf_row")
    if not isinstance(perf_row, Mapping):
        raise TypeError("MoE result must contain a perf row mapping")
    expected_identity = {
        "moe_dtype": case["quant_mode"],
        "num_tokens": case["num_tokens"],
        "hidden_size": case["hidden_size"],
        "inter_size": case["inter_size"],
        "topk": case["topk"],
        "num_experts": case["num_experts"],
        "moe_tp_size": case["moe_tp_size"],
        "moe_ep_size": case["moe_ep_size"],
        "distribution": case["workload_distribution"],
    }
    if {field: perf_row.get(field) for field in expected_identity} != expected_identity:
        raise ValueError("MoE perf row identity does not match the requested TP1/EP4 case")
    if perf_row.get("framework") != "SGLang" or perf_row.get("version") != request.environment.backend_version:
        raise ValueError("MoE perf row framework identity does not match SGLang 0.5.10")
    if perf_row.get("op_name") != "moe" or perf_row.get("kernel_source") != "sglang_fused_moe_triton":
        raise ValueError("MoE perf row kernel identity is unsupported")
    row_latency = _positive_finite_number(perf_row.get("latency"), field="perf row latency")
    if row_latency != latency_ms:
        raise ValueError("MoE perf row latency does not match measured latency")

    provenance = raw_result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("MoE provenance must be a mapping")
    if provenance.get("framework") != "SGLang":
        raise ValueError("MoE provenance framework must be SGLang")
    if provenance.get("framework_version") != request.environment.backend_version:
        raise ValueError("MoE provenance framework version does not match the request runtime")
    measured_device = provenance.get("device")
    if not isinstance(measured_device, str) or _normalized_label(measured_device) != _normalized_label(
        request.environment.gpu_class
    ):
        raise ValueError("MoE provenance device does not match the request GPU class")
    if provenance.get("kernel_source") != "sglang_fused_moe_triton":
        raise ValueError("MoE provenance kernel source is unsupported")
    if provenance.get("used_cuda_graph") is not True:
        raise ValueError("MoE measurement must use CUDA Graph capture")
    if provenance.get("throttled") is not False:
        raise ValueError("MoE measurement must not be thermally throttled")
    if provenance.get("model_artifact") != _MODEL_ARTIFACT:
        raise ValueError("MoE provenance model artifact does not match the frozen profile")
    if (
        provenance.get("workload_generator") != _SEMANTIC_DESCRIPTOR["workload_generator"]
        or provenance.get("seed") != _SEMANTIC_DESCRIPTOR["seed"]
        or provenance.get("rank_simulation") != _SEMANTIC_DESCRIPTOR["rank_simulation"]
    ):
        raise ValueError("MoE provenance rank simulation does not match the semantic descriptor")

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
    "moe_request_to_case",
    "moe_resource_for_request",
    "moe_result_to_record",
]
