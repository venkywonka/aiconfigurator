# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free exact DSv4 V1.2 SGLang CustomAllReduce runner."""

from __future__ import annotations

import gc
import importlib
import inspect
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aiconfigurator.collector.types import RawMeasurement
from aiconfigurator.sdk.operations.communication import SGLANG_CUSTOM_ALLREDUCE_MAX_BYTES
from aiconfigurator.sdk.resolution.types import MeasurementProtocol

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_SUPPORTED_SGLANG_VERSIONS = frozenset({"0.5.10", "0.5.10rc0"})
_REPEATS_PER_GRAPH = 5
_PHYSICAL_BYTES_PER_ELEMENT = 2
_PERSISTENT_RANK_GROUP: Any = None
_PERSISTENT_DEVICE_UUIDS: tuple[str, ...] = ()


def _physical_indices_for_device_uuids(
    device_uuids: tuple[str, ...],
    *,
    nvml: Any | None = None,
) -> tuple[int, ...]:
    """Resolve assigned GPU UUIDs to exact host indices with identity checks."""

    if not isinstance(device_uuids, tuple) or not device_uuids:
        raise ValueError("rank process visibility requires assigned device UUIDs")
    if any(not isinstance(device_uuid, str) or not device_uuid for device_uuid in device_uuids):
        raise ValueError("rank process device UUIDs must be non-empty strings")
    if len(set(device_uuids)) != len(device_uuids):
        raise ValueError("rank process device UUIDs must be unique")

    nvml_api = nvml if nvml is not None else importlib.import_module("pynvml")
    nvml_api.nvmlInit()
    try:
        physical_indices: list[int] = []
        for device_uuid in device_uuids:
            handle = nvml_api.nvmlDeviceGetHandleByUUID(device_uuid)
            physical_index = nvml_api.nvmlDeviceGetIndex(handle)
            if isinstance(physical_index, bool) or not isinstance(physical_index, int) or physical_index < 0:
                raise ValueError("NVML physical device indices must be nonnegative integers")
            roundtrip_handle = nvml_api.nvmlDeviceGetHandleByIndex(physical_index)
            roundtrip_uuid = nvml_api.nvmlDeviceGetUUID(roundtrip_handle)
            if isinstance(roundtrip_uuid, bytes):
                roundtrip_uuid = roundtrip_uuid.decode("ascii")
            if roundtrip_uuid != device_uuid:
                raise RuntimeError(
                    "NVML UUID round-trip mismatch: "
                    f"expected={device_uuid!r}, observed={roundtrip_uuid!r}, index={physical_index}"
                )
            physical_indices.append(physical_index)
    finally:
        nvml_api.nvmlShutdown()

    if len(set(physical_indices)) != len(physical_indices):
        raise ValueError("assigned device UUIDs must resolve to unique physical device indices")
    return tuple(physical_indices)


def _bind_rank_process_physical_visibility(
    bootstrap: Any,
    *,
    resolve_uuid_indices: Callable[[tuple[str, ...]], tuple[int, ...]] = _physical_indices_for_device_uuids,
) -> None:
    """Bind one spawned rank to the assigned devices using SGLang-safe indices."""

    rank = bootstrap.rank
    world_size = bootstrap.world_size
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
        or not 0 <= rank < world_size
    ):
        raise ValueError("CustomAllReduce rank bootstrap is invalid")

    raw_visibility = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible = tuple(value.strip() for value in raw_visibility.split(",")) if raw_visibility else ()
    if len(visible) != world_size or any(not value for value in visible):
        raise RuntimeError(
            "CustomAllReduce worker visibility must contain exactly four assigned devices: "
            f"visible={len(visible)}, requested={world_size}"
        )
    if len(set(visible)) != len(visible):
        raise ValueError("CustomAllReduce worker visibility must be unique")
    if visible[rank] != bootstrap.device_uuid:
        raise RuntimeError(
            "CustomAllReduce rank UUID does not match its assigned visibility: "
            f"rank={rank}, expected={bootstrap.device_uuid!r}, observed={visible[rank]!r}"
        )

    parsed_indices: list[int | None] = []
    for value in visible:
        try:
            parsed_indices.append(int(value))
        except ValueError:
            parsed_indices.append(None)
    numeric_count = sum(index is not None for index in parsed_indices)
    if numeric_count == world_size:
        physical_indices = tuple(index for index in parsed_indices if index is not None)
    elif numeric_count:
        raise ValueError("CustomAllReduce worker visibility cannot mix UUIDs and physical indices")
    else:
        physical_indices = tuple(resolve_uuid_indices(visible))

    if len(physical_indices) != world_size:
        raise RuntimeError("CustomAllReduce physical visibility does not match its assigned device count")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in physical_indices):
        raise ValueError("CustomAllReduce physical device indices must be nonnegative integers")
    if len(set(physical_indices)) != len(physical_indices):
        raise ValueError("CustomAllReduce worker visibility must resolve to unique physical device indices")
    if numeric_count == world_size:
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in physical_indices)


@dataclass(frozen=True, slots=True)
class PreparedCustomAllReduceCase:
    """Graph plus fixed-address storage retained for every replay."""

    graph: Any
    inputs: tuple[Any, ...]
    outputs: tuple[Any, ...]


def _validate_protocol(protocol: MeasurementProtocol) -> None:
    if (
        protocol.revision != "cuda-event-samples-v1"
        or protocol.timer != "cuda_event"
        or protocol.tuning_revision != "sglang-custom-allreduce-v1"
        or protocol.statistic != "median"
        or protocol.samples < 3
    ):
        raise ValueError("CustomAllReduce measurement protocol is incompatible with the persistent graph runner")


def _validate_case(dtype: str, world_size: int, element_count: int) -> None:
    if dtype != "half":
        raise ValueError("CustomAllReduce dtype must be half")
    if world_size != 4:
        raise ValueError("CustomAllReduce world size must be four (4)")
    if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0:
        raise ValueError("CustomAllReduce element_count must be a positive integer")
    physical_bytes = element_count * _PHYSICAL_BYTES_PER_ELEMENT
    if physical_bytes % 16:
        raise ValueError("CustomAllReduce physical byte size must be a multiple of 16")
    if physical_bytes > SGLANG_CUSTOM_ALLREDUCE_MAX_BYTES:
        raise ValueError("CustomAllReduce element_count exceeds the SGLang 8 MiB maximum")


class _SglangCustomAllReduceRankBackend:
    """Heavy SGLang backend initialized inside one UUID-restricted rank."""

    def __init__(self, bootstrap: Any) -> None:
        self._bootstrap = bootstrap
        self._torch: Any = None
        self._dist: Any = None
        self._ca_comm: Any = None
        self._graph_capture: Any = None
        self._cases: dict[tuple[str, str, int], PreparedCustomAllReduceCase] = {}
        self._destroyed = False

    def initialize(self) -> None:
        _bind_rank_process_physical_visibility(self._bootstrap)
        import torch
        import torch.distributed as dist
        from sglang.srt.distributed.parallel_state import (
            get_tp_group,
            graph_capture,
            init_distributed_environment,
            initialize_model_parallel,
            set_custom_all_reduce,
            set_mscclpp_all_reduce,
            set_torch_symm_mem_all_reduce,
        )
        from sglang.srt.server_args import set_global_server_args_for_scheduler

        class _ServerArgs:
            enable_symm_mem = False

        set_global_server_args_for_scheduler(_ServerArgs())
        rank = self._bootstrap.rank
        torch.cuda.set_device(rank)
        init_method = os.environ.get("AICONFIGURATOR_NCCL_INIT_METHOD")
        if not init_method:
            raise RuntimeError("CustomAllReduce rank rendezvous is not configured")
        init_kwargs: dict[str, Any] = {
            "backend": "nccl",
            "init_method": init_method,
            "rank": rank,
            "world_size": self._bootstrap.world_size,
        }
        if "device_id" in inspect.signature(dist.init_process_group).parameters:
            init_kwargs["device_id"] = torch.device(f"cuda:{rank}")
        dist.init_process_group(**init_kwargs)
        init_distributed_environment(
            world_size=self._bootstrap.world_size,
            rank=rank,
            distributed_init_method=init_method,
            local_rank=rank,
            backend="nccl",
        )
        set_custom_all_reduce(True)
        set_mscclpp_all_reduce(False)
        set_torch_symm_mem_all_reduce(False)
        initialize_model_parallel(
            tensor_model_parallel_size=self._bootstrap.world_size,
            pipeline_model_parallel_size=1,
        )
        tp_group = get_tp_group()
        ca_comm = getattr(tp_group, "ca_comm", None)
        if ca_comm is None or getattr(ca_comm, "disabled", True):
            raise RuntimeError("SGLang CustomAllReduce communicator is unavailable on the assigned topology")
        self._torch = torch
        self._dist = dist
        self._ca_comm = ca_comm
        self._graph_capture = graph_capture

    def _prepare_case(self, dtype: str, operation: str, element_count: int) -> PreparedCustomAllReduceCase:
        if dtype != "half" or operation != "all_reduce":
            raise ValueError("SGLang CustomAllReduce ranks support only half all_reduce")
        torch = self._torch
        device = f"cuda:{self._bootstrap.rank}"
        inputs = tuple(
            torch.ones((element_count,), dtype=torch.bfloat16, device=device) for _ in range(_REPEATS_PER_GRAPH)
        )
        ca_comm = self._ca_comm
        if ca_comm is None or getattr(ca_comm, "disabled", True) or not ca_comm.should_custom_ar(inputs[0]):
            raise RuntimeError("SGLang custom kernel cannot serve this exact shape; refusing NCCL fallback")
        torch.cuda.synchronize(device)
        with self._graph_capture() as graph_capture_context:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                outputs = tuple(ca_comm.custom_all_reduce(value) for value in inputs)
        if any(output is None for output in outputs):
            raise RuntimeError("SGLang custom kernel rejected the shape during CUDA Graph capture")
        return PreparedCustomAllReduceCase(graph=graph, inputs=inputs, outputs=outputs)

    def run(self, dtype: str, operation: str, element_count: int) -> float:
        key = (dtype, operation, element_count)
        case = self._cases.get(key)
        if case is None:
            case = self._prepare_case(*key)
            self._cases[key] = case
        torch = self._torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        case.graph.replay()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / _REPEATS_PER_GRAPH

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        torch = self._torch
        cleanup_errors: list[Exception] = []
        if torch is not None:
            try:
                torch.cuda.synchronize(self._bootstrap.rank)
            except Exception as error:
                cleanup_errors.append(error)

        self._cases.clear()
        self._ca_comm = None
        self._graph_capture = None
        try:
            from sglang.srt.distributed import parallel_state

            for cleanup in (
                parallel_state.destroy_model_parallel,
                parallel_state.destroy_distributed_environment,
            ):
                try:
                    cleanup()
                except Exception as error:
                    cleanup_errors.append(error)
        finally:
            self._dist = None
            self._torch = None
            try:
                gc.collect()
            except Exception as error:
                cleanup_errors.append(error)
            if torch is not None:
                try:
                    torch.cuda.empty_cache()
                except Exception as error:
                    cleanup_errors.append(error)
        if cleanup_errors:
            raise cleanup_errors[0]

    def abort(self) -> None:
        # The parent poisons and terminates the entire four-rank lease.
        return


def custom_allreduce_rank_process_main(bootstrap: Any, command_queue: Any, reply_queue: Any) -> None:
    """Run the proven correlated-rank protocol with the SGLang backend."""

    from aiconfigurator.collector.executor import nccl_rank_process_main

    nccl_rank_process_main(
        bootstrap,
        command_queue,
        reply_queue,
        runtime_factory=_SglangCustomAllReduceRankBackend,
        all_rank_samples=True,
    )


def _visible_device_uuids(world_size: int) -> tuple[str, ...]:
    visible = tuple(value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip())
    if len(visible) != world_size:
        raise RuntimeError(
            "CustomAllReduce worker visibility must contain exactly four assigned devices: "
            f"visible={len(visible)}, requested={world_size}"
        )
    return visible


def _persistent_rank_group(*, world_size: int, protocol: MeasurementProtocol) -> Any:
    global _PERSISTENT_DEVICE_UUIDS, _PERSISTENT_RANK_GROUP

    visible = _visible_device_uuids(world_size)
    if _PERSISTENT_RANK_GROUP is not None and bool(getattr(_PERSISTENT_RANK_GROUP, "poisoned", False)):
        close_custom_allreduce_worker()
    if _PERSISTENT_RANK_GROUP is None:
        from aiconfigurator.collector.executor import PersistentNcclRankGroup

        _PERSISTENT_RANK_GROUP = PersistentNcclRankGroup(
            device_uuids=visible,
            protocol=protocol,
            worker_target=custom_allreduce_rank_process_main,
            collect_all_rank_samples=True,
        )
        _PERSISTENT_DEVICE_UUIDS = visible
    elif visible != _PERSISTENT_DEVICE_UUIDS or _PERSISTENT_RANK_GROUP.protocol_digest != protocol.digest:
        close_custom_allreduce_worker()
        raise RuntimeError("persistent CustomAllReduce worker lease identity changed")
    return _PERSISTENT_RANK_GROUP


def close_custom_allreduce_worker() -> None:
    """Close the module-local four-rank group once at worker shutdown."""

    global _PERSISTENT_DEVICE_UUIDS, _PERSISTENT_RANK_GROUP
    group, _PERSISTENT_RANK_GROUP = _PERSISTENT_RANK_GROUP, None
    _PERSISTENT_DEVICE_UUIDS = ()
    if group is not None:
        group.close()


def _runtime_metadata() -> tuple[str, str]:
    from importlib.metadata import version

    import torch

    return version("sglang"), str(torch.cuda.get_device_name(0))


def get_custom_allreduce_test_cases() -> tuple[()]:
    """Lazy exact collection deliberately exposes no offline grid sweep."""

    return ()


def run_custom_allreduce_case(
    dtype: str,
    world_size: int,
    element_count: int,
    *,
    protocol: MeasurementProtocol,
) -> RawMeasurement:
    """Measure exactly one graph-captured four-rank SGLang collective."""

    _validate_case(dtype, world_size, element_count)
    if not isinstance(protocol, MeasurementProtocol):
        raise TypeError("CustomAllReduce protocol must be a MeasurementProtocol")
    _validate_protocol(protocol)
    _visible_device_uuids(world_size)
    framework_version, device_name = _runtime_metadata()
    if framework_version not in _SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            "CustomAllReduce requires SGLang 0.5.10 or the exact 0.5.10rc0 measurement runtime, "
            f"got {framework_version!r}"
        )
    if " ".join(device_name.split()).casefold() != "nvidia gb200":
        raise RuntimeError(f"CustomAllReduce requires NVIDIA GB200 devices, got {device_name!r}")

    runtime = _persistent_rank_group(world_size=world_size, protocol=protocol)
    samples_ms = tuple(runtime.measure(dtype, "all_reduce", element_count))
    if len(samples_ms) != protocol.samples:
        raise RuntimeError("persistent CustomAllReduce rank group returned an invalid sample count")
    latency_ms = statistics.median(samples_ms)
    rank_pids = list(getattr(runtime, "rank_pids", ()))
    row = {
        "framework": "SGLang",
        "version": framework_version,
        "device": device_name,
        "op_name": "all_reduce",
        "kernel_source": "SGLang_CustomAllReduce_graph",
        "allreduce_dtype": dtype,
        "num_gpus": world_size,
        "message_size": element_count,
        "backend": "sglang_graph",
        "latency": latency_ms,
    }
    return RawMeasurement(
        latency_ms=latency_ms,
        energy_wms=0.0,
        samples_ms=samples_ms,
        statistic=protocol.statistic,
        perf_row=row,
        provenance={
            "framework": "SGLang",
            "framework_version": framework_version,
            "kernel_source": "SGLang_CustomAllReduce_graph",
            "device": device_name,
            "runtime": "persistent_sglang_custom_allreduce",
            "used_cuda_graph": True,
            "throttled": False,
            "world_size": world_size,
            "rank_pids": rank_pids,
            "model_artifact": _MODEL_ARTIFACT,
            "physical_dtype": "bfloat16",
        },
        protocol_digest=protocol.digest,
    )


run_custom_allreduce_case.close_worker = close_custom_allreduce_worker


__all__ = [
    "close_custom_allreduce_worker",
    "custom_allreduce_rank_process_main",
    "get_custom_allreduce_test_cases",
    "run_custom_allreduce_case",
]
