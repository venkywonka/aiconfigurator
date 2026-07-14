# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free exact NCCL case runner."""

from __future__ import annotations

import os
import statistics
from typing import TYPE_CHECKING, Any

from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

if TYPE_CHECKING:
    from aiconfigurator.collector.executor import PersistentNcclRuntime

_SUPPORTED_OPS = {"all_gather", "alltoall", "reduce_scatter", "all_reduce"}
_BYTES_PER_ELEMENT = {"half": 2, "bfloat16": 2, "int8": 1}
_PERSISTENT_RANK_GROUP: Any = None
_PERSISTENT_DEVICE_UUIDS: tuple[str, ...] = ()


def normalize_nccl_runtime_version(raw_version: object) -> str:
    """Normalize every NCCL version representation exposed by PyTorch."""

    if isinstance(raw_version, (tuple, list)) and raw_version:
        if not all(isinstance(part, int) and not isinstance(part, bool) for part in raw_version):
            raise RuntimeError(f"invalid NCCL runtime version tuple {raw_version!r}")
        return ".".join(str(part) for part in raw_version)
    if isinstance(raw_version, int) and not isinstance(raw_version, bool) and raw_version > 0:
        major = raw_version // 10_000
        minor = (raw_version % 10_000) // 100
        patch = raw_version % 100
        return f"{major}.{minor}.{patch}"
    raise RuntimeError(f"invalid NCCL runtime version {raw_version!r}")


def _validate_lazy_protocol(protocol: MeasurementProtocol) -> None:
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "torch-nccl-persistent-v1"
        or protocol.statistic != "median"
    ):
        raise ValueError("NCCL measurement protocol is incompatible with the persistent runner")


def _persistent_rank_group(
    *,
    num_gpus: int,
    protocol: MeasurementProtocol,
) -> Any:
    global _PERSISTENT_DEVICE_UUIDS, _PERSISTENT_RANK_GROUP

    visible = tuple(value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip())
    if len(visible) != num_gpus:
        raise RuntimeError(
            "persistent NCCL worker visibility does not match the requested group size: "
            f"visible={len(visible)}, requested={num_gpus}"
        )
    if _PERSISTENT_RANK_GROUP is None:
        from aiconfigurator.collector.executor import PersistentNcclRankGroup

        _PERSISTENT_RANK_GROUP = PersistentNcclRankGroup(
            device_uuids=visible,
            protocol=protocol,
        )
        _PERSISTENT_DEVICE_UUIDS = visible
    elif visible != _PERSISTENT_DEVICE_UUIDS or _PERSISTENT_RANK_GROUP.protocol_digest != protocol.digest:
        raise RuntimeError("persistent NCCL worker lease identity changed")
    return _PERSISTENT_RANK_GROUP


def close_nccl_worker() -> None:
    """Close the module-local persistent rank group once at worker shutdown."""

    global _PERSISTENT_DEVICE_UUIDS, _PERSISTENT_RANK_GROUP
    group, _PERSISTENT_RANK_GROUP = _PERSISTENT_RANK_GROUP, None
    _PERSISTENT_DEVICE_UUIDS = ()
    if group is not None:
        group.close()


def get_nccl_test_cases(
    dtype: str,
    nccl_op: str,
    test_range: str,
    num_gpus: int,
) -> tuple[tuple[str, str, int, int], ...]:
    """Expand the legacy byte-range sweep into exact element-count cases."""

    if dtype not in _BYTES_PER_ELEMENT:
        raise ValueError(f"unsupported NCCL dtype {dtype!r}")
    if nccl_op not in _SUPPORTED_OPS:
        raise ValueError(f"unsupported NCCL operation {nccl_op!r}")
    minimum, maximum, ratio = (int(value) for value in test_range.split(","))
    if minimum <= 0 or maximum <= minimum or ratio <= 1:
        raise ValueError("NCCL test range must be positive and increasing")
    cases = []
    size_bytes = minimum
    while size_bytes < maximum:
        cases.append((dtype, nccl_op, size_bytes // _BYTES_PER_ELEMENT[dtype], num_gpus))
        size_bytes *= ratio
    return tuple(cases)


def _persistent_provenance(runtime: PersistentNcclRuntime) -> dict[str, Any]:
    try:
        import torch

        framework_version = normalize_nccl_runtime_version(torch.cuda.nccl.version())
        device_name = str(torch.cuda.get_device_name())
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("NCCL worker could not attest its loaded runtime identity") from exc
    return {
        "framework": "NCCL",
        "framework_version": framework_version,
        "device": device_name,
        "kernel_source": "NCCL",
        "runtime": "persistent_torch_distributed",
        "device_uuids": list(getattr(runtime, "device_uuids", _PERSISTENT_DEVICE_UUIDS)),
        "rank_pids": list(getattr(runtime, "rank_pids", ())),
    }


def run_nccl_case(
    dtype: str,
    nccl_op: str,
    element_count: int,
    num_gpus: int,
    *,
    runtime: PersistentNcclRuntime | None = None,
    protocol: MeasurementProtocol | None = None,
) -> RawMeasurement:
    """Measure exactly one NCCL point without writing a perf database file."""

    if dtype not in _BYTES_PER_ELEMENT:
        raise ValueError(f"unsupported NCCL dtype {dtype!r}")
    if nccl_op not in _SUPPORTED_OPS:
        raise ValueError(f"unsupported NCCL operation {nccl_op!r}")
    if element_count <= 0 or num_gpus < 2:
        raise ValueError("NCCL element_count must be positive and num_gpus must be at least two")
    protocol = protocol or MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=5,
        samples=10,
        statistic="median",
        timer="cuda_event",
        tuning_revision="torch-nccl-persistent-v1",
    )
    _validate_lazy_protocol(protocol)
    if runtime is None:
        runtime = _persistent_rank_group(num_gpus=num_gpus, protocol=protocol)
    elif runtime.protocol_digest != protocol.digest:
        raise ValueError("NCCL runtime protocol does not match the requested protocol")
    samples_ms = tuple(runtime.measure(dtype, nccl_op, element_count))
    if len(samples_ms) != protocol.samples:
        raise RuntimeError("persistent NCCL runtime returned an invalid sample count")
    power_stats = None
    provenance = _persistent_provenance(runtime)
    latency_ms = statistics.median(samples_ms)
    row = {
        "nccl_dtype": dtype,
        "op_name": nccl_op,
        "num_gpus": num_gpus,
        "message_size": element_count,
        "latency": latency_ms,
    }
    return RawMeasurement(
        latency_ms=latency_ms,
        energy_wms=float((power_stats or {}).get("power", 0.0)) * latency_ms,
        samples_ms=tuple(samples_ms),
        statistic=protocol.statistic,
        perf_row=row,
        provenance=provenance,
        protocol_digest=runtime.protocol_digest,
        power_stats=power_stats,
    )


run_nccl_case.close_worker = close_nccl_worker


__all__ = [
    "close_nccl_worker",
    "get_nccl_test_cases",
    "normalize_nccl_runtime_version",
    "run_nccl_case",
]
