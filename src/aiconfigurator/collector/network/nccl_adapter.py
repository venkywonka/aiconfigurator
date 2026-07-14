# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-free request/result mapping for exact NCCL collectives."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.perf_namespace import perf_namespace
from aiconfigurator.sdk.resolution.types import MeasurementRecord, MeasurementRequest

_NAMESPACE = perf_namespace(str(PerfFile.NCCL))
_OPERATIONS = frozenset({"all_reduce", "all_gather", "reduce_scatter", "alltoall"})
_DTYPES = frozenset({"half", "bfloat16", "int8"})
_QUERY_FIELDS = frozenset({"nccl_dtype", "operation", "num_gpus", "message_size"})


def nccl_request_to_case(request: MeasurementRequest) -> dict[str, Any]:
    """Translate a canonical physical NCCL query without changing its values."""

    if not isinstance(request, MeasurementRequest):
        raise TypeError("request must be a MeasurementRequest")
    if request.key.namespace != _NAMESPACE:
        raise ValueError(f"NCCL request namespace must be {_NAMESPACE!r}")
    if set(request.query) != _QUERY_FIELDS:
        raise ValueError(f"NCCL query fields must be exactly {sorted(_QUERY_FIELDS)!r}")
    dtype = request.query["nccl_dtype"]
    operation = request.query["operation"]
    num_gpus = request.query["num_gpus"]
    element_count = request.query["message_size"]
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported NCCL dtype {dtype!r}")
    if operation not in _OPERATIONS:
        raise ValueError(f"unsupported NCCL operation {operation!r}")
    for name, value in (("num_gpus", num_gpus), ("message_size", element_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"NCCL {name} must be a positive integer")
    if num_gpus < 2:
        raise ValueError("lazy NCCL collection requires at least two GPUs")
    if request.environment.topology_schema is None or request.environment.topology_fingerprint is None:
        raise ValueError("NCCL collection requires an explicit topology identity")
    nccl_version = request.environment.runtime_versions.get("nccl")
    if not isinstance(nccl_version, str) or not nccl_version.strip():
        raise ValueError("NCCL collection requires an explicit NCCL runtime version")
    return {
        "dtype": dtype,
        "nccl_op": operation,
        "element_count": element_count,
        "num_gpus": num_gpus,
    }


def nccl_resource_for_request(
    request: MeasurementRequest,
    case: Mapping[str, Any],
) -> ResourceContract:
    """Reserve an NVLink clique and its stable fabric domain."""

    expected = nccl_request_to_case(request)
    if dict(case) != expected:
        raise ValueError("NCCL case does not match the canonical request query")
    return ResourceContract(
        gpu_count=expected["num_gpus"],
        fabric=FabricRequirement.NVLINK,
        reserve_fabric_domain=True,
    )


def nccl_result_to_record(
    request: MeasurementRequest,
    case: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> MeasurementRecord:
    """Validate a collective result before it enters the parent-owned overlay."""

    expected = nccl_request_to_case(request)
    if dict(case) != expected:
        raise ValueError("NCCL case does not match the canonical request query")
    if not isinstance(raw_result, Mapping):
        raise TypeError("raw NCCL result must be a mapping")
    latency_ms = raw_result.get("latency_ms")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
        raise TypeError("NCCL latency must be numeric")
    latency_ms = float(latency_ms)
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("NCCL latency must be positive and finite")
    energy_wms = raw_result.get("energy_wms", 0.0)
    if isinstance(energy_wms, bool) or not isinstance(energy_wms, (int, float)):
        raise TypeError("NCCL energy must be numeric")
    energy_wms = float(energy_wms)
    if not math.isfinite(energy_wms) or energy_wms < 0:
        raise ValueError("NCCL energy must be finite and non-negative")
    raw_samples = raw_result.get("samples_ms")
    if not isinstance(raw_samples, (list, tuple)):
        raise TypeError("NCCL samples must be a numeric sequence")
    if any(isinstance(sample, bool) or not isinstance(sample, (int, float)) for sample in raw_samples):
        raise TypeError("NCCL samples must be a numeric sequence")
    samples_ms = tuple(float(sample) for sample in raw_samples)
    if len(samples_ms) != request.protocol.samples:
        raise ValueError("NCCL sample count does not match the request protocol")
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples_ms):
        raise ValueError("NCCL samples must be positive and finite")
    if statistics.median(samples_ms) != latency_ms:
        raise ValueError("NCCL latency must equal the median of its samples")
    if raw_result.get("statistic") != request.protocol.statistic:
        raise ValueError("NCCL statistic does not match the request protocol")
    if raw_result.get("protocol_digest") != request.protocol.digest:
        raise ValueError("NCCL protocol identity does not match the request protocol")

    perf_row = raw_result.get("perf_row")
    if not isinstance(perf_row, Mapping):
        raise TypeError("NCCL result must contain a perf row mapping")
    row_identity = {
        "nccl_dtype": perf_row.get("nccl_dtype"),
        "num_gpus": perf_row.get("num_gpus"),
        "message_size": perf_row.get("message_size"),
    }
    expected_identity = {
        "nccl_dtype": expected["dtype"],
        "num_gpus": expected["num_gpus"],
        "message_size": expected["element_count"],
    }
    if row_identity != expected_identity:
        raise ValueError("NCCL perf row identity does not match the request query")
    row_operation = perf_row.get("op_name")
    if row_operation != expected["nccl_op"]:
        raise ValueError("NCCL perf row operation does not match the request query")
    row_latency = perf_row.get("latency")
    if isinstance(row_latency, bool) or not isinstance(row_latency, (int, float)):
        raise TypeError("NCCL perf row latency must be numeric")
    if not math.isfinite(float(row_latency)) or float(row_latency) != latency_ms:
        raise ValueError("NCCL perf row latency does not match measured latency")
    provenance = raw_result.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise TypeError("NCCL provenance must be a mapping")
    if provenance.get("framework") != "NCCL":
        raise ValueError("NCCL provenance framework does not match the request")
    expected_nccl_version = request.environment.runtime_versions["nccl"]
    if provenance.get("framework_version") != expected_nccl_version:
        raise ValueError("NCCL runtime version does not match the request environment")
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
    "nccl_request_to_case",
    "nccl_resource_for_request",
    "nccl_result_to_record",
]
