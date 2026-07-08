# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen DSv4 V1.2 adapter for exact SGLang attention measurements."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.resolution.types import MeasurementRecord, MeasurementRequest, PerfKey

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_ARCHITECTURE = "DeepseekV4ForCausalLM"
_CANONICAL_NUM_HEADS = 16
_PADDED_NUM_HEADS = 64
_TP_SIZE = 4
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
_PROFILE_COMPATIBILITY = {
    "model_artifact": _MODEL_ARTIFACT,
    "serving_mode": "aggregated",
    "tp_size": _TP_SIZE,
    "attention_dp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 4,
    "nextn": 0,
}
_SEMANTIC_DESCRIPTOR = {
    "full_module": True,
    "tensor_generator": "normal-v1",
    "seed": 0,
    "tp_simulation": "single-gpu-tp4",
    "canonical_num_heads": _CANONICAL_NUM_HEADS,
    "padded_num_heads": _PADDED_NUM_HEADS,
}


@dataclass(frozen=True, slots=True)
class _Route:
    mode: str
    attn_kind: str
    compress_ratio: int


_ROUTES = {
    f"{PerfFile.DSV4_CSA_CONTEXT_MODULE}/v1": _Route("context", "csa", 4),
    f"{PerfFile.DSV4_HCA_CONTEXT_MODULE}/v1": _Route("context", "hca", 128),
    f"{PerfFile.DSV4_CSA_GENERATION_MODULE}/v1": _Route("generation", "csa", 4),
    f"{PerfFile.DSV4_HCA_GENERATION_MODULE}/v1": _Route("generation", "hca", 128),
}
_COMMON_QUERY_FIELDS = {
    "tp_size",
    "num_heads",
    "compress_ratio",
    "batch_size",
    "kv_cache_dtype",
    "gemm_type",
}
_CONTEXT_QUERY_FIELDS = _COMMON_QUERY_FIELDS | {"sequence_length", "prefix_length", "mla_dtype"}
_GENERATION_QUERY_FIELDS = _COMMON_QUERY_FIELDS | {"sequence_length"}


def _normalized_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def _route_for_request(request: MeasurementRequest) -> _Route:
    try:
        return _ROUTES[request.key.namespace]
    except KeyError as error:
        raise ValueError("DSv4 attention request namespace is not one of the four frozen module routes") from error


def _positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"DSv4 attention {field} must be a positive integer")
    return value


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"DSv4 attention {field} must be a non-negative integer")
    return value


def _positive_finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"DSv4 attention {field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"DSv4 attention {field} must be positive and finite")
    return number


def _validate_capability(request: MeasurementRequest, route: _Route) -> None:
    environment = request.environment
    if (
        environment.system != "gb200"
        or environment.backend != "sglang"
        or environment.backend_version != "0.5.10"
        or _normalized_label(environment.gpu_class) != "nvidia gb200"
        or any(environment.runtime_versions.get(name) != version for name, version in _RUNTIME_VERSIONS.items())
    ):
        raise ValueError("DSv4 attention request is outside the frozen GB200/SGLang 0.5.10 capability envelope")
    if dict(environment.profile_compatibility or {}) != _PROFILE_COMPATIBILITY:
        raise ValueError("DSv4 attention profile compatibility does not match the frozen V1.2 deployment")
    if dict(request.semantic_descriptor) != _SEMANTIC_DESCRIPTOR:
        raise ValueError("DSv4 attention semantic descriptor does not describe the frozen full module")
    if request.protocol.samples < 3:
        raise ValueError("DSv4 attention protocol samples must be at least three")

    query = request.query
    expected_fields = _CONTEXT_QUERY_FIELDS if route.mode == "context" else _GENERATION_QUERY_FIELDS
    if set(query) != expected_fields:
        raise ValueError(f"DSv4 attention query fields must be exactly {tuple(sorted(expected_fields))!r}")
    if query["tp_size"] != _TP_SIZE or query["num_heads"] != _CANONICAL_NUM_HEADS:
        raise ValueError("DSv4 attention request requires TP4 and canonical rank-local num_heads=16")
    if query["compress_ratio"] != route.compress_ratio:
        raise ValueError("DSv4 attention compression ratio does not match the CSA/HCA namespace")
    _positive_int(query["batch_size"], field="batch_size")
    _positive_int(query["sequence_length"], field="sequence_length")
    if route.mode == "context":
        _non_negative_int(query["prefix_length"], field="prefix_length")
        if query["mla_dtype"] != "bfloat16":
            raise ValueError("DSv4 context attention requires bfloat16 MLA/FMHA inputs")
    if query["kv_cache_dtype"] != "fp8" or query["gemm_type"] != "fp8_block":
        raise ValueError("DSv4 attention requires FP8 KV cache and fp8_block GEMM")
    if route.mode == "generation" and query["sequence_length"] < 2:
        raise ValueError("DSv4 generation attention sequence_length must be at least two")


def dsv4_attn_request_to_case(request: MeasurementRequest) -> dict[str, Any]:
    """Translate one canonical AIC query into an exact padded-head runner case."""

    if not isinstance(request, MeasurementRequest):
        raise TypeError("request must be a MeasurementRequest")
    route = _route_for_request(request)
    _validate_capability(request, route)

    case = dict(request.query)
    case["mode"] = route.mode
    case["attn_kind"] = route.attn_kind
    case["canonical_num_heads"] = case.pop("num_heads")
    case["num_heads"] = _PADDED_NUM_HEADS
    case.setdefault("mla_dtype", "bfloat16")
    if route.mode == "context":
        case["isl"] = case.pop("sequence_length")
        case["prefix"] = case.pop("prefix_length")
    else:
        case["s_total"] = case.pop("sequence_length")
    return case


def dsv4_attn_resource_for_request(
    request: MeasurementRequest,
    case: Mapping[str, Any],
) -> ResourceContract:
    """Reserve one exclusive GPU; TP4 is simulated without a fabric domain."""

    if dsv4_attn_request_to_case(request) != dict(case):
        raise ValueError("DSv4 attention case does not match the canonical request")
    return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)


def _canonical_query_from_row(route: _Route, row: Mapping[str, Any]) -> dict[str, Any]:
    padded_heads = _positive_int(row.get("num_heads"), field="persisted num_heads")
    tp_size = _positive_int(row.get("tp_size"), field="persisted tp_size")
    if padded_heads != _PADDED_NUM_HEADS or tp_size != _TP_SIZE or padded_heads % tp_size:
        raise ValueError("DSv4 attention persisted row does not encode the padded64/TP4 mapping")
    query: dict[str, Any] = {
        "tp_size": tp_size,
        "num_heads": padded_heads // tp_size,
        "compress_ratio": _positive_int(row.get("compress_ratio"), field="persisted compress_ratio"),
        "batch_size": _positive_int(row.get("batch_size"), field="persisted batch_size"),
        "kv_cache_dtype": row.get("kv_cache_dtype"),
        "gemm_type": row.get("gemm_type"),
    }
    if route.mode == "context":
        query.update(
            {
                "sequence_length": _positive_int(row.get("isl"), field="persisted isl"),
                "prefix_length": _non_negative_int(row.get("step"), field="persisted step"),
                "mla_dtype": row.get("mla_dtype"),
            }
        )
    else:
        isl = _positive_int(row.get("isl"), field="persisted isl")
        step = _non_negative_int(row.get("step"), field="persisted step")
        if isl != 1:
            raise ValueError("DSv4 generation attention persisted row must use decode isl=1")
        query["sequence_length"] = isl + step
    return query


def _validate_provenance(
    request: MeasurementRequest,
    route: _Route,
    provenance: Mapping[str, Any],
) -> None:
    measured_device = provenance.get("device")
    if provenance.get("framework") != "SGLang":
        raise ValueError("DSv4 attention provenance framework must be SGLang")
    if provenance.get("framework_version") != request.environment.backend_version:
        raise ValueError("DSv4 attention provenance framework version does not match the request")
    if not isinstance(measured_device, str) or _normalized_label(measured_device) != _normalized_label(
        request.environment.gpu_class
    ):
        raise ValueError("DSv4 attention provenance device does not match the request GPU")
    expected = {
        "kernel_source": "compressed_flashmla",
        "used_cuda_graph": True,
        "throttled": False,
        "model_artifact": _MODEL_ARTIFACT,
        "full_module": True,
        "mode": route.mode,
        "attn_kind": route.attn_kind,
        "tp_simulation": "single-gpu-tp4",
        "canonical_num_heads": _CANONICAL_NUM_HEADS,
        "padded_num_heads": _PADDED_NUM_HEADS,
        "tensor_generator": "normal-v1",
        "seed": 0,
        "model_weight_generator": "proper-normal-v1",
        "model_weight_std": 0.05,
        "model_weight_seed": 1234,
    }
    if any(provenance.get(field) != value for field, value in expected.items()):
        raise ValueError("DSv4 attention provenance does not match the frozen exact-runner contract")


def dsv4_attn_result_to_record(
    request: MeasurementRequest,
    case: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> MeasurementRecord:
    """Validate one padded-head full-module row and preserve canonical identity."""

    expected_case = dsv4_attn_request_to_case(request)
    if expected_case != dict(case):
        raise ValueError("DSv4 attention case does not match the canonical request")
    if not isinstance(raw_result, Mapping):
        raise TypeError("raw DSv4 attention result must be a mapping")

    latency_ms = _positive_finite_number(raw_result.get("latency_ms"), field="latency")
    energy_value = raw_result.get("energy_wms", 0.0)
    if isinstance(energy_value, bool) or not isinstance(energy_value, (int, float)):
        raise TypeError("DSv4 attention energy must be numeric")
    energy_wms = float(energy_value)
    if not math.isfinite(energy_wms) or energy_wms < 0:
        raise ValueError("DSv4 attention energy must be finite and non-negative")

    raw_samples = raw_result.get("samples_ms")
    if not isinstance(raw_samples, (list, tuple)):
        raise TypeError("DSv4 attention samples must be a numeric sequence")
    samples_ms = tuple(_positive_finite_number(sample, field="sample") for sample in raw_samples)
    if len(samples_ms) != request.protocol.samples:
        raise ValueError("DSv4 attention sample count does not match the request protocol")
    if statistics.median(samples_ms) != latency_ms:
        raise ValueError("DSv4 attention latency must equal the median of its samples")
    if raw_result.get("statistic") != request.protocol.statistic:
        raise ValueError("DSv4 attention statistic does not match the request protocol")
    if raw_result.get("protocol_digest") != request.protocol.digest:
        raise ValueError("DSv4 attention protocol identity does not match the request protocol")

    perf_row = raw_result.get("perf_row")
    if not isinstance(perf_row, Mapping):
        raise TypeError("DSv4 attention result must contain a perf row mapping")
    route = _route_for_request(request)
    expected_identity = {
        "model": _MODEL_ARTIFACT,
        "architecture": _ARCHITECTURE,
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
        "num_heads": _PADDED_NUM_HEADS,
        "batch_size": request.query["batch_size"],
        "isl": request.query["sequence_length"] if route.mode == "context" else 1,
        "tp_size": _TP_SIZE,
        "step": request.query["prefix_length"]
        if route.mode == "context"
        else int(request.query["sequence_length"]) - 1,
        "compress_ratio": route.compress_ratio,
    }
    if {field: perf_row.get(field) for field in expected_identity} != expected_identity:
        raise ValueError("DSv4 attention perf row identity does not match the exact runner case")
    if _canonical_query_from_row(route, perf_row) != dict(request.query):
        raise ValueError("DSv4 attention persisted row does not reconstruct the canonical query")
    rebuilt_key = PerfKey.build(request.key.namespace, _canonical_query_from_row(route, perf_row), request.environment)
    if rebuilt_key != request.key:
        raise ValueError("DSv4 attention persisted row does not reconstruct the canonical PerfKey")
    row_latency = _positive_finite_number(perf_row.get("latency"), field="perf row latency")
    if row_latency != latency_ms:
        raise ValueError("DSv4 attention perf row latency does not match measured latency")

    provenance = raw_result.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TypeError("DSv4 attention provenance must be a mapping")
    _validate_provenance(request, route, provenance)

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
    "dsv4_attn_request_to_case",
    "dsv4_attn_resource_for_request",
    "dsv4_attn_result_to_record",
]
