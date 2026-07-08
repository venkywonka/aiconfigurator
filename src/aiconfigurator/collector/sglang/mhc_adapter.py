# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen DSv4 V1.2 adapter for exact SGLang mHC measurements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.resolution.types import MeasurementRecord, MeasurementRequest

_NAMESPACE = f"{PerfFile.MHC_MODULE}/v1"
_QUERY_FIELDS = (
    "op",
    "num_tokens",
    "hidden_size",
    "hc_mult",
    "sinkhorn_iters",
    "quant_mode",
)
_SEMANTIC_DESCRIPTOR = {
    "num_sites": 2,
    "tensor_generator": "normal-v1",
    "seed": 0,
}
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
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
_ARCHITECTURE = "DeepseekV4ForCausalLM"


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
        raise ValueError("mHC request is outside the frozen GB200/SGLang 0.5.10 runtime capability envelope")
    if dict(environment.profile_compatibility or {}) != _PROFILE_COMPATIBILITY:
        raise ValueError("mHC request profile compatibility does not match the frozen DSv4 V1.2 deployment")
    if dict(request.semantic_descriptor) != _SEMANTIC_DESCRIPTOR:
        raise ValueError("mHC request semantic descriptor does not describe the exact two-site full module")
    if case["op"] not in {"pre", "post"}:
        raise ValueError("mHC op is outside the frozen pre/post capability envelope")
    if case["hidden_size"] != 4096 or case["hc_mult"] != 4 or case["sinkhorn_iters"] != 20:
        raise ValueError("mHC shape is outside the frozen DSv4 V1.2 capability envelope")
    if case["quant_mode"] != "bfloat16":
        raise ValueError("mHC frozen module capability requires bfloat16")


def mhc_request_to_case(request: MeasurementRequest) -> dict[str, Any]:
    """Translate one canonical physical mHC key to an exact runner case."""

    if not isinstance(request, MeasurementRequest):
        raise TypeError("request must be a MeasurementRequest")
    if request.key.namespace != _NAMESPACE:
        raise ValueError(f"mHC request namespace must be {_NAMESPACE!r}")
    if set(request.query) != set(_QUERY_FIELDS):
        raise ValueError(f"mHC query fields must be exactly {_QUERY_FIELDS!r}")

    op = request.query["op"]
    num_tokens = request.query["num_tokens"]
    hidden_size = request.query["hidden_size"]
    hc_mult = request.query["hc_mult"]
    sinkhorn_iters = request.query["sinkhorn_iters"]
    quant_mode = request.query["quant_mode"]
    dimensions = (num_tokens, hidden_size, hc_mult, sinkhorn_iters)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("mHC num_tokens, hidden_size, hc_mult, and sinkhorn_iters must be positive integers")
    case = {
        "op": op,
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
        "sinkhorn_iters": sinkhorn_iters,
        "quant_mode": quant_mode,
    }
    _validate_capability(request, case)
    return case


def mhc_resource_for_request(
    request: MeasurementRequest,
    case: Mapping[str, Any],
) -> ResourceContract:
    """Reserve one exclusive GPU without a fabric-domain reservation."""

    if mhc_request_to_case(request) != dict(case):
        raise ValueError("mHC case does not match the canonical request query")
    return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)


def _positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"mHC {field} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"mHC {field} must be positive and finite")
    return value


def mhc_result_to_record(
    request: MeasurementRequest,
    case: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> MeasurementRecord:
    """Validate one full-module result and preserve the request's PerfKey."""

    if mhc_request_to_case(request) != dict(case):
        raise ValueError("mHC case does not match the canonical request query")
    if not isinstance(raw_result, Mapping):
        raise TypeError("raw mHC result must be a mapping")

    latency_ms = _positive_finite_number(raw_result.get("latency_ms"), field="latency")
    energy_value = raw_result.get("energy_wms", 0.0)
    if isinstance(energy_value, bool) or not isinstance(energy_value, (int, float)):
        raise TypeError("mHC energy must be numeric")
    energy_wms = float(energy_value)
    if not math.isfinite(energy_wms) or energy_wms < 0:
        raise ValueError("mHC energy must be finite and non-negative")

    raw_samples = raw_result.get("samples_ms")
    if not isinstance(raw_samples, (list, tuple)):
        raise TypeError("mHC samples must be a numeric sequence")
    samples_ms = tuple(_positive_finite_number(sample, field="sample") for sample in raw_samples)
    if len(samples_ms) != request.protocol.samples:
        raise ValueError("mHC sample count does not match the request protocol")
    if statistics.median(samples_ms) != latency_ms:
        raise ValueError("mHC latency must equal the median of its samples")
    if raw_result.get("statistic") != request.protocol.statistic:
        raise ValueError("mHC statistic does not match the request protocol")
    if raw_result.get("protocol_digest") != request.protocol.digest:
        raise ValueError("mHC protocol identity does not match the request protocol")

    perf_row = raw_result.get("perf_row")
    if not isinstance(perf_row, Mapping):
        raise TypeError("mHC result must contain a perf row mapping")
    if perf_row.get("num_sites") != 2:
        raise ValueError("mHC perf row must represent both full-module sites")
    expected_identity = {
        "architecture": _ARCHITECTURE,
        "op_name": case["op"],
        "num_tokens": case["num_tokens"],
        "num_sites": 2,
        "hc_mult": case["hc_mult"],
        "hidden_size": case["hidden_size"],
        "sinkhorn_iters": case["sinkhorn_iters"],
        "quant_mode": case["quant_mode"],
    }
    if {field: perf_row.get(field) for field in expected_identity} != expected_identity:
        raise ValueError("mHC perf row identity does not match the request query")
    row_latency = _positive_finite_number(perf_row.get("latency"), field="perf row latency")
    if row_latency != latency_ms:
        raise ValueError("mHC perf row latency does not match measured latency")

    provenance = raw_result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("mHC provenance must be a mapping")
    if provenance.get("framework") != "SGLang":
        raise ValueError("mHC provenance framework must be SGLang")
    if provenance.get("framework_version") != request.environment.backend_version:
        raise ValueError("mHC provenance framework version does not match the request runtime version")
    measured_device = provenance.get("device")
    if not isinstance(measured_device, str) or _normalized_label(measured_device) != _normalized_label(
        request.environment.gpu_class
    ):
        raise ValueError("mHC provenance device does not match the request GPU class")
    if provenance.get("kernel_source") != "sglang_mhc":
        raise ValueError("mHC provenance kernel source must be sglang_mhc")
    if provenance.get("used_cuda_graph") is not True:
        raise ValueError("mHC measurement must use CUDA Graph")
    if provenance.get("model_artifact") != _PROFILE_COMPATIBILITY["model_artifact"]:
        raise ValueError("mHC provenance model artifact does not match the frozen profile")
    if provenance.get("full_module") is not True or provenance.get("num_sites") != 2:
        raise ValueError("mHC provenance must identify the two-site full-module boundary")
    if provenance.get("tensor_generator") != "normal-v1" or provenance.get("seed") != 0:
        raise ValueError("mHC provenance tensor generator does not match the semantic descriptor")

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
    "mhc_request_to_case",
    "mhc_resource_for_request",
    "mhc_result_to_record",
]
