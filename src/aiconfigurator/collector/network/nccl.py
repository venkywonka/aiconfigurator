# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free exact NCCL case runner."""

from __future__ import annotations

import os
import statistics
import subprocess
from typing import TYPE_CHECKING, Any

from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

if TYPE_CHECKING:
    from aiconfigurator.collector.executor import PersistentNcclRuntime

_OP_BINARY = {
    "all_gather": "all_gather_perf",
    "alltoall": "alltoall_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "all_reduce": "all_reduce_perf",
}
_BYTES_PER_ELEMENT = {"half": 2, "int8": 1}
_PERSISTENT_RANK_GROUP: Any = None
_PERSISTENT_DEVICE_UUIDS: tuple[str, ...] = ()


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
    if nccl_op not in _OP_BINARY:
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


def _parse_nccl_latency_ms(stdout: str) -> float:
    lines = stdout.splitlines()
    for index, line in enumerate(lines):
        if "time" not in line:
            continue
        for candidate in lines[index + 1 :]:
            fields = candidate.split()
            if len(fields) > 5:
                try:
                    return float(fields[5]) * 1e-3
                except ValueError:
                    continue
    raise RuntimeError("nccl-tests output did not contain a latency row")


def _run_nccl_tests_case(
    *,
    dtype: str,
    nccl_op: str,
    message_size_bytes: int,
    num_gpus: int,
    measure_power: bool,
) -> tuple[tuple[float, ...], dict[str, Any] | None, dict[str, Any]]:
    binary = _OP_BINARY[nccl_op]
    inner_loop = 100 if message_size_bytes <= 16_777_216 else 60
    power_monitor = None
    if measure_power:
        from aiconfigurator.collector.benchmark import PowerMonitor

        candidate = PowerMonitor(0)
        if candidate._init_handle():
            power_monitor = candidate
            power_monitor.start_sampling()
    try:
        completed = subprocess.run(
            [
                binary,
                "-b",
                str(message_size_bytes),
                "-e",
                str(message_size_bytes),
                "-t",
                str(num_gpus),
                "-d",
                dtype,
                "-w",
                "40",
                "-a",
                "1",
                "-n",
                str(inner_loop),
                "-c",
                "0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        latency_ms = _parse_nccl_latency_ms(completed.stdout)
    finally:
        power_stats = power_monitor.stop_sampling() if power_monitor is not None else None
    return (latency_ms,), power_stats, {"runtime": "nccl-tests", "binary": binary}


def run_nccl_case(
    dtype: str,
    nccl_op: str,
    element_count: int,
    num_gpus: int,
    *,
    runtime: PersistentNcclRuntime | None = None,
    measure_power: bool = False,
    protocol: MeasurementProtocol | None = None,
) -> RawMeasurement:
    """Measure exactly one NCCL point without writing a perf database file."""

    if dtype not in _BYTES_PER_ELEMENT:
        raise ValueError(f"unsupported NCCL dtype {dtype!r}")
    if nccl_op not in _OP_BINARY:
        raise ValueError(f"unsupported NCCL operation {nccl_op!r}")
    if element_count <= 0 or num_gpus < 2:
        raise ValueError("NCCL element_count must be positive and num_gpus must be at least two")
    if protocol is not None:
        _validate_lazy_protocol(protocol)
        if runtime is None:
            runtime = _persistent_rank_group(num_gpus=num_gpus, protocol=protocol)
        elif runtime.protocol_digest != protocol.digest:
            raise ValueError("NCCL runtime protocol does not match the requested protocol")
    if runtime is None:
        samples_ms, power_stats, provenance = _run_nccl_tests_case(
            dtype=dtype,
            nccl_op=nccl_op,
            message_size_bytes=element_count * _BYTES_PER_ELEMENT[dtype],
            num_gpus=num_gpus,
            measure_power=measure_power,
        )
    else:
        samples_ms = tuple(runtime.measure(dtype, nccl_op, element_count))
        power_stats = None
        provenance = {
            "runtime": "persistent_torch_distributed",
            "device_uuids": list(getattr(runtime, "device_uuids", _PERSISTENT_DEVICE_UUIDS)),
            "rank_pids": list(getattr(runtime, "rank_pids", ())),
        }
        if protocol is not None and len(samples_ms) != protocol.samples:
            raise RuntimeError("persistent NCCL runtime returned an invalid sample count")
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
        statistic=protocol.statistic if protocol is not None else "median",
        perf_row=row,
        provenance=provenance,
        protocol_digest=runtime.protocol_digest if runtime is not None else None,
        power_stats=power_stats,
    )


run_nccl_case.close_worker = close_nccl_worker


__all__ = ["close_nccl_worker", "get_nccl_test_cases", "run_nccl_case"]
