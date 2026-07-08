# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight request/result mapping for exact TensorRT-LLM GEMM cases."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from typing import Any

from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.perf_namespace import perf_namespace
from aiconfigurator.sdk.resolution.types import MeasurementRecord, MeasurementRequest

_NAMESPACE = perf_namespace(str(PerfFile.GEMM))
_QUERY_FIELDS = ("gemm_type", "m", "n", "k")
_PERF_IDENTITY_FIELDS = ("gemm_dtype", "m", "n", "k")


def gemm_request_to_case(request: MeasurementRequest) -> dict[str, Any]:
    """Translate one canonical BF16 query to the runner's positional case."""

    if not isinstance(request, MeasurementRequest):
        raise TypeError("request must be a MeasurementRequest")
    if request.key.namespace != _NAMESPACE:
        raise ValueError(f"GEMM request namespace must be {_NAMESPACE!r}")
    if set(request.query) != set(_QUERY_FIELDS):
        raise ValueError(f"GEMM query fields must be exactly {_QUERY_FIELDS!r}")
    gemm_type = request.query["gemm_type"]
    if gemm_type != "bfloat16":
        raise ValueError("the generic V1 GEMM pilot supports only bfloat16 (BF16)")
    dimensions = tuple(request.query[field] for field in ("m", "n", "k"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
        raise ValueError("GEMM m, n, and k must be positive integers")
    return {
        "gemm_type": gemm_type,
        "m": dimensions[0],
        "n": dimensions[1],
        "k": dimensions[2],
    }


def gemm_resource_for_request(
    request: MeasurementRequest,
    case: Mapping[str, Any],
) -> ResourceContract:
    """Declare one exclusive GPU and no fabric reservation."""

    if gemm_request_to_case(request) != dict(case):
        raise ValueError("GEMM case does not match the canonical request query")
    return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)


def gemm_result_to_record(
    request: MeasurementRequest,
    case: Mapping[str, Any],
    raw_result: Mapping[str, Any],
) -> MeasurementRecord:
    """Validate raw worker evidence and preserve the request's exact identity."""

    if gemm_request_to_case(request) != dict(case):
        raise ValueError("GEMM case does not match the canonical request query")
    if not isinstance(raw_result, Mapping):
        raise TypeError("raw GEMM result must be a mapping")

    latency_ms = raw_result.get("latency_ms")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)):
        raise TypeError("GEMM latency must be numeric")
    latency_ms = float(latency_ms)
    if not math.isfinite(latency_ms) or latency_ms <= 0:
        raise ValueError("GEMM latency must be positive and finite")

    energy_wms = raw_result.get("energy_wms", 0.0)
    if isinstance(energy_wms, bool) or not isinstance(energy_wms, (int, float)):
        raise TypeError("GEMM energy must be numeric")
    energy_wms = float(energy_wms)
    if not math.isfinite(energy_wms) or energy_wms < 0:
        raise ValueError("GEMM energy must be finite and non-negative")

    raw_samples = raw_result.get("samples_ms")
    if not isinstance(raw_samples, (list, tuple)):
        raise TypeError("GEMM samples must be a numeric sequence")
    if any(isinstance(sample, bool) or not isinstance(sample, (int, float)) for sample in raw_samples):
        raise TypeError("GEMM samples must be a numeric sequence")
    samples_ms = tuple(float(sample) for sample in raw_samples)
    if len(samples_ms) != request.protocol.samples:
        raise ValueError("GEMM sample count does not match the request protocol")
    if any(not math.isfinite(sample) or sample <= 0 for sample in samples_ms):
        raise ValueError("GEMM samples must be positive and finite")
    if statistics.median(samples_ms) != latency_ms:
        raise ValueError("GEMM latency must equal the median of its samples")
    if raw_result.get("statistic") != request.protocol.statistic:
        raise ValueError("GEMM statistic does not match the request protocol")
    if raw_result.get("protocol_digest") != request.protocol.digest:
        raise ValueError("GEMM protocol identity does not match the request protocol")

    perf_row = raw_result.get("perf_row")
    if not isinstance(perf_row, Mapping):
        raise TypeError("GEMM result must contain a perf row mapping")
    expected_identity = {
        "gemm_dtype": case["gemm_type"],
        "m": case["m"],
        "n": case["n"],
        "k": case["k"],
    }
    if {field: perf_row.get(field) for field in _PERF_IDENTITY_FIELDS} != expected_identity:
        raise ValueError("GEMM perf row identity does not match the request query")
    row_latency = perf_row.get("latency")
    if isinstance(row_latency, bool) or not isinstance(row_latency, (int, float)):
        raise TypeError("GEMM perf row latency must be numeric")
    if float(row_latency) != latency_ms:
        raise ValueError("GEMM perf row latency does not match measured latency")

    provenance = raw_result.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise TypeError("GEMM provenance must be a mapping")
    if provenance.get("framework_version") != request.environment.backend_version:
        raise ValueError("GEMM provenance framework version does not match the request runtime")
    measured_device = provenance.get("device")
    if not isinstance(measured_device, str) or (
        " ".join(measured_device.split()).casefold() != " ".join(request.environment.gpu_class.split()).casefold()
    ):
        raise ValueError("GEMM provenance device does not match the request GPU class")
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
    "gemm_request_to_case",
    "gemm_resource_for_request",
    "gemm_result_to_record",
]
