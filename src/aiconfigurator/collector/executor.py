# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent, hardware-aware execution for exact lazy measurements.

The parent process owns request validation, hardware placement, result
conversion, and failure typing.  Workers receive only canonical JSON cases and
return raw result mappings; they never construct or persist performance rows.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import os
import queue
import tempfile
import threading
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any, Protocol

from aiconfigurator.collector.adapters import PreparedMeasurement
from aiconfigurator.collector.scheduler import HardwareAwareScheduler, UnschedulableRequest
from aiconfigurator.collector.types import Assignment, CollectionJob, HardwareInventory
from aiconfigurator.sdk.resolution.session import CancellationToken
from aiconfigurator.sdk.resolution.types import (
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    RecordStatus,
    UnresolvedCode,
    canonical_json,
)

__all__ = [
    "NcclRankBootstrap",
    "NcclRankCommand",
    "NcclRankReply",
    "PersistentMeasurementExecutor",
    "PersistentNcclRankGroup",
    "PersistentNcclRuntime",
    "ProcessWorkerFactory",
    "WorkerBootstrap",
    "WorkerCommand",
    "WorkerReply",
    "bind_and_import_runner",
    "nccl_rank_process_main",
    "worker_process_main",
]


class PersistentNcclRuntime:
    """Persistent collective measurement facade owned by one worker lease.

    The process-group transport is injected so the coordinator and lightweight
    imports remain Torch-free. The concrete multiprocessing rank group is
    installed by the GPU worker factory.
    """

    _OPERATIONS = frozenset({"all_reduce", "all_gather", "reduce_scatter", "alltoall"})
    _DTYPES = frozenset({"half", "int8"})

    def __init__(
        self,
        measure: Callable[[str, str, int], Sequence[float]] | None = None,
        close: Callable[[], None] | None = None,
        protocol: MeasurementProtocol | None = None,
    ) -> None:
        self._measure = measure
        self._close = close
        self._protocol = protocol
        self._closed = False

    @property
    def protocol_digest(self) -> str | None:
        return self._protocol.digest if self._protocol is not None else None

    def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
        if self._closed:
            raise RuntimeError("persistent NCCL runtime is closed")
        if dtype not in self._DTYPES:
            raise ValueError(f"unsupported NCCL dtype {dtype!r}")
        if operation not in self._OPERATIONS:
            raise ValueError(f"unsupported NCCL operation {operation!r}")
        if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0:
            raise ValueError("NCCL element_count must be a positive integer")
        if self._measure is None:
            raise RuntimeError("persistent NCCL rank group has not been initialized")
        try:
            samples = tuple(float(sample) for sample in self._measure(dtype, operation, element_count))
        except BaseException:
            self.close()
            raise
        if not samples:
            raise RuntimeError("persistent NCCL rank group returned no samples")
        return samples

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close is not None:
            self._close()


@dataclass(frozen=True, slots=True)
class NcclRankBootstrap:
    """Immutable identity for one rank in a persistent local NCCL group."""

    rank: int
    world_size: int
    device_uuid: str
    protocol: MeasurementProtocol


@dataclass(frozen=True, slots=True)
class NcclRankCommand:
    """One exact collective case broadcast to every persistent rank."""

    invocation_id: str
    dtype: str
    operation: str
    element_count: int
    protocol: MeasurementProtocol


@dataclass(frozen=True, slots=True)
class NcclRankReply:
    """Per-rank completion acknowledgement; only rank zero carries samples."""

    invocation_id: str
    rank: int
    samples_ms: tuple[float, ...] = ()
    error: str | None = None


def _nccl_alltoall_splits(
    element_count: int,
    world_size: int,
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return exact per-destination sends and per-source receives for one rank."""

    if element_count <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid NCCL alltoall shape or rank")
    base, remainder = divmod(element_count, world_size)
    input_splits = tuple(base + (destination < remainder) for destination in range(world_size))
    output_splits = (input_splits[rank],) * world_size
    return input_splits, output_splits


class _TorchNcclRankBackend:
    """Heavy Torch backend imported only inside a UUID-restricted rank."""

    def __init__(self, bootstrap: NcclRankBootstrap) -> None:
        self._bootstrap = bootstrap
        self._torch: Any = None
        self._dist: Any = None
        self._cases: dict[tuple[str, str, int], tuple[Callable[[], None], Callable[[], None]]] = {}

    def initialize(self) -> None:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(self._bootstrap.rank)
        init_method = os.environ.get("AICONFIGURATOR_NCCL_INIT_METHOD")
        if not init_method:
            raise RuntimeError("NCCL rank rendezvous is not configured")
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=self._bootstrap.rank,
            world_size=self._bootstrap.world_size,
        )
        self._torch = torch
        self._dist = dist

    def _tensor(self, count: int, dtype: Any) -> Any:
        torch = self._torch
        device = f"cuda:{self._bootstrap.rank}"
        generator = torch.Generator(device=device)
        generator.manual_seed(0)
        if dtype is torch.int8:
            return torch.randint(-8, 8, (count,), dtype=dtype, device=device, generator=generator)
        return torch.randn((count,), dtype=dtype, device=device, generator=generator)

    def _prepare_case(
        self,
        dtype_name: str,
        operation: str,
        element_count: int,
    ) -> tuple[Callable[[], None], Callable[[], None]]:
        torch = self._torch
        dist = self._dist
        dtype = {"half": torch.float16, "int8": torch.int8}[dtype_name]
        world_size = self._bootstrap.world_size

        if operation == "all_reduce":
            template = self._tensor(element_count, dtype)
            tensor = template.clone()

            def prepare() -> None:
                tensor.copy_(template)

            def invoke() -> None:
                dist.all_reduce(tensor)

            return prepare, invoke

        if operation == "all_gather":
            tensor = self._tensor(element_count, dtype)
            outputs = [torch.empty_like(tensor) for _ in range(world_size)]
            return (lambda: None), lambda: dist.all_gather(outputs, tensor)

        if operation == "reduce_scatter":
            output = torch.empty((element_count,), dtype=dtype, device=tensor_device(self._bootstrap.rank))
            inputs = [self._tensor(element_count, dtype) for _ in range(world_size)]
            return (lambda: None), lambda: dist.reduce_scatter(output, inputs)

        if operation == "alltoall":
            tensor = self._tensor(element_count, dtype)
            input_splits, output_splits = _nccl_alltoall_splits(
                element_count,
                world_size,
                self._bootstrap.rank,
            )
            output = torch.empty(
                (sum(output_splits),),
                dtype=dtype,
                device=tensor_device(self._bootstrap.rank),
            )
            return (lambda: None), lambda: dist.all_to_all_single(
                output,
                tensor,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
            )
        raise ValueError(f"unsupported NCCL operation {operation!r}")

    def run(self, dtype: str, operation: str, element_count: int) -> float:
        key = (dtype, operation, element_count)
        case = self._cases.get(key)
        if case is None:
            case = self._prepare_case(*key)
            self._cases[key] = case
        prepare, invoke = case
        prepare()
        self._torch.cuda.synchronize()
        start = self._torch.cuda.Event(enable_timing=True)
        end = self._torch.cuda.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    def destroy(self) -> None:
        if self._dist is not None and self._dist.is_initialized():
            self._dist.destroy_process_group()

    def abort(self) -> None:
        # Never enter another process-group operation after one rank fails.
        # The owning parent terminates every rank in the lease.
        return


def tensor_device(rank: int) -> str:
    """Return the local CUDA ordinal after UUID visibility restriction."""

    return f"cuda:{rank}"


def nccl_rank_process_main(
    bootstrap: NcclRankBootstrap,
    command_queue: Any,
    reply_queue: Any,
    *,
    runtime_factory: Callable[[NcclRankBootstrap], Any] = _TorchNcclRankBackend,
) -> None:
    """Initialize one NCCL rank once and serve independent exact cases."""

    backend = runtime_factory(bootstrap)
    command: NcclRankCommand | None = None
    failed = False
    try:
        backend.initialize()
        while True:
            candidate = command_queue.get()
            if candidate is None:
                break
            if not isinstance(candidate, NcclRankCommand):
                raise TypeError("NCCL rank received a malformed command")
            command = candidate
            try:
                if command.protocol != bootstrap.protocol:
                    raise ValueError("NCCL rank command protocol does not match its lease")
                if command.dtype not in PersistentNcclRuntime._DTYPES:
                    raise ValueError(f"unsupported NCCL dtype {command.dtype!r}")
                if command.operation not in PersistentNcclRuntime._OPERATIONS:
                    raise ValueError(f"unsupported NCCL operation {command.operation!r}")
                if (
                    isinstance(command.element_count, bool)
                    or not isinstance(command.element_count, int)
                    or command.element_count <= 0
                ):
                    raise ValueError("NCCL element_count must be a positive integer")
                for _ in range(command.protocol.warmups):
                    backend.run(command.dtype, command.operation, command.element_count)
                samples = tuple(
                    float(backend.run(command.dtype, command.operation, command.element_count))
                    for _ in range(command.protocol.samples)
                )
                reply_queue.put(
                    NcclRankReply(
                        invocation_id=command.invocation_id,
                        rank=bootstrap.rank,
                        samples_ms=samples if bootstrap.rank == 0 else (),
                    )
                )
            except BaseException:
                failed = True
                backend.abort()
                reply_queue.put(
                    NcclRankReply(
                        invocation_id=command.invocation_id,
                        rank=bootstrap.rank,
                        error=traceback.format_exc(),
                    )
                )
                return
    except BaseException:
        failed = True
        backend.abort()
        reply_queue.put(
            NcclRankReply(
                invocation_id=command.invocation_id if command is not None else "",
                rank=bootstrap.rank,
                error=traceback.format_exc(),
            )
        )
    finally:
        if not failed:
            backend.destroy()


class PersistentNcclRankGroup:
    """Spawn and reuse one local rank process per assigned GPU UUID."""

    def __init__(
        self,
        *,
        device_uuids: tuple[str, ...],
        protocol: MeasurementProtocol,
        context_getter: Callable[[str], Any] = multiprocessing.get_context,
        worker_target: Callable[..., None] = nccl_rank_process_main,
        reply_timeout_seconds: float = 120.0,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(device_uuids, tuple) or not device_uuids:
            raise ValueError("persistent NCCL group requires assigned device UUIDs")
        if len(set(device_uuids)) != len(device_uuids):
            raise ValueError("persistent NCCL group device UUIDs must be unique")
        if not isinstance(protocol, MeasurementProtocol):
            raise TypeError("persistent NCCL group protocol must be a MeasurementProtocol")
        if reply_timeout_seconds <= 0 or shutdown_timeout_seconds < 0:
            raise ValueError("persistent NCCL group timeouts must be positive")

        self._device_uuids = device_uuids
        self._protocol = protocol
        self._reply_timeout_seconds = reply_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._context = context_getter("spawn")
        self._reply_queue = self._context.Queue()
        self._command_queues: list[Any] = []
        self._processes: list[Any] = []
        self._closed = False
        self._poisoned = False
        self._rendezvous = tempfile.TemporaryDirectory(prefix="aic-nccl-rendezvous-")
        init_method = f"file://{self._rendezvous.name}/store"
        previous_init_method = os.environ.get("AICONFIGURATOR_NCCL_INIT_METHOD")
        os.environ["AICONFIGURATOR_NCCL_INIT_METHOD"] = init_method
        try:
            for rank, device_uuid in enumerate(device_uuids):
                command_queue = self._context.Queue()
                bootstrap = NcclRankBootstrap(
                    rank=rank,
                    world_size=len(device_uuids),
                    device_uuid=device_uuid,
                    protocol=protocol,
                )
                process = self._context.Process(
                    target=worker_target,
                    args=(bootstrap, command_queue, self._reply_queue),
                    daemon=False,
                )
                process.start()
                self._command_queues.append(command_queue)
                self._processes.append(process)
        except BaseException:
            self._poison()
            raise
        finally:
            if previous_init_method is None:
                os.environ.pop("AICONFIGURATOR_NCCL_INIT_METHOD", None)
            else:
                os.environ["AICONFIGURATOR_NCCL_INIT_METHOD"] = previous_init_method

    @property
    def protocol_digest(self) -> str:
        return self._protocol.digest

    @property
    def device_uuids(self) -> tuple[str, ...]:
        return self._device_uuids

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def rank_pids(self) -> tuple[int | None, ...]:
        return tuple(getattr(process, "pid", None) for process in self._processes)

    def _get_reply(self) -> object:
        try:
            return self._reply_queue.get(timeout=self._reply_timeout_seconds)
        except TypeError:
            return self._reply_queue.get()
        except queue.Empty as error:
            raise TimeoutError("persistent NCCL rank group timed out") from error

    def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
        if self._closed:
            raise RuntimeError("persistent NCCL rank group is closed")
        if self._poisoned:
            raise RuntimeError("persistent NCCL rank group is poisoned")
        if dtype not in PersistentNcclRuntime._DTYPES:
            raise ValueError(f"unsupported NCCL dtype {dtype!r}")
        if operation not in PersistentNcclRuntime._OPERATIONS:
            raise ValueError(f"unsupported NCCL operation {operation!r}")
        if isinstance(element_count, bool) or not isinstance(element_count, int) or element_count <= 0:
            raise ValueError("NCCL element_count must be a positive integer")

        command = NcclRankCommand(
            invocation_id=uuid.uuid4().hex,
            dtype=dtype,
            operation=operation,
            element_count=element_count,
            protocol=self._protocol,
        )
        for command_queue in self._command_queues:
            command_queue.put(command)

        rank_zero_samples: tuple[float, ...] | None = None
        completed_ranks: set[int] = set()
        try:
            for _ in self._processes:
                reply = self._get_reply()
                if not isinstance(reply, NcclRankReply):
                    raise TypeError("persistent NCCL rank returned a malformed reply")
                if reply.invocation_id != command.invocation_id:
                    raise RuntimeError("persistent NCCL rank reply identity mismatch")
                if reply.rank in completed_ranks or not 0 <= reply.rank < len(self._processes):
                    raise RuntimeError("persistent NCCL rank reply has an invalid rank")
                if reply.error is not None:
                    raise RuntimeError(reply.error)
                completed_ranks.add(reply.rank)
                if reply.rank == 0:
                    rank_zero_samples = tuple(reply.samples_ms)
                elif reply.samples_ms:
                    raise RuntimeError("only NCCL rank zero may return timing samples")
            if rank_zero_samples is None or len(rank_zero_samples) != self._protocol.samples:
                raise RuntimeError("NCCL rank zero returned an invalid sample count")
            return rank_zero_samples
        except BaseException:
            self._poison()
            raise

    def _join_or_kill(self, process: Any) -> None:
        process.join(self._shutdown_timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(self._shutdown_timeout_seconds)
        if process.is_alive():
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
                process.join(self._shutdown_timeout_seconds)

    def _cleanup(self, *, forced: bool) -> None:
        for queue_object in (*self._command_queues, self._reply_queue):
            if forced:
                cancel_join_thread = getattr(queue_object, "cancel_join_thread", None)
                if callable(cancel_join_thread):
                    cancel_join_thread()
            close = getattr(queue_object, "close", None)
            if callable(close):
                close()
        self._rendezvous.cleanup()

    def _poison(self) -> None:
        if self._poisoned:
            return
        self._poisoned = True
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(self._shutdown_timeout_seconds)
        survivors = [process for process in self._processes if process.is_alive()]
        for process in survivors:
            kill = getattr(process, "kill", None)
            if callable(kill):
                kill()
        for process in survivors:
            process.join(self._shutdown_timeout_seconds)
        self._cleanup(forced=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._poisoned:
            return
        for command_queue in self._command_queues:
            command_queue.put(None)
        for process in self._processes:
            self._join_or_kill(process)
        self._cleanup(forced=False)


@dataclass(frozen=True, slots=True)
class WorkerBootstrap:
    """Immutable process identity installed before a worker imports GPU code."""

    run_module: str
    run_func: str
    adapter_namespace: str
    protocol_digest: str
    device_uuids: tuple[str, ...]
    topology_fingerprint: str

    @property
    def local_ordinals(self) -> tuple[int, ...]:
        """Ordinals visible inside the UUID-restricted worker process."""

        return tuple(range(len(self.device_uuids)))


@dataclass(frozen=True, slots=True)
class WorkerCommand:
    """One uniquely identified canonical case sent to a worker."""

    invocation_id: str
    request_digest: str
    payload: bytes
    protocol: MeasurementProtocol | None = None


@dataclass(frozen=True, slots=True)
class WorkerReply:
    """Raw worker reply correlated to exactly one invocation and PerfKey."""

    invocation_id: str
    request_digest: str
    raw_result: Any = None
    error: str | None = None


class WorkerChannel(Protocol):
    """Minimal persistent-worker transport used by the coordinator."""

    def send(self, command: WorkerCommand) -> None: ...

    def recv(self) -> object: ...

    def is_alive(self) -> bool: ...

    def close(self) -> None: ...

    def join(self) -> None: ...


def bind_and_import_runner(
    bootstrap: WorkerBootstrap,
    *,
    import_module: Callable[[str], ModuleType | object] = importlib.import_module,
) -> object:
    """Restrict visibility by stable UUID before importing the GPU runner."""

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(bootstrap.device_uuids)
    module = import_module(bootstrap.run_module)
    try:
        return getattr(module, bootstrap.run_func)
    except AttributeError as error:
        raise AttributeError(
            f"runner module {bootstrap.run_module!r} has no attribute {bootstrap.run_func!r}"
        ) from error


def worker_process_main(
    bootstrap: WorkerBootstrap,
    command_queue: Any,
    reply_queue: Any,
    *,
    import_module: Callable[[str], ModuleType | object] = importlib.import_module,
) -> None:
    """Serve exact cases until the parent closes this persistent worker.

    Device UUID visibility is installed before the runner module is imported.
    The wire boundary intentionally contains only immutable command objects and
    plain result dictionaries; row validation and persistence stay parent-owned.
    """

    runner = bind_and_import_runner(bootstrap, import_module=import_module)
    try:
        while True:
            command = command_queue.get()
            if command is None:
                return
            if not isinstance(command, WorkerCommand):
                raise TypeError("worker received a malformed command")

            try:
                if command.protocol is None:
                    raise ValueError("worker command is missing its measurement protocol")
                if command.protocol.digest != bootstrap.protocol_digest:
                    raise ValueError("worker command protocol does not match its persistent lease")
                case = json.loads(command.payload)
                if not isinstance(case, dict):
                    raise TypeError("worker case payload must decode to a JSON object")
                raw_result = runner(**case, protocol=command.protocol)
                if not isinstance(raw_result, Mapping):
                    raise TypeError("exact runner must return a raw result mapping")
                reply = WorkerReply(
                    invocation_id=command.invocation_id,
                    request_digest=command.request_digest,
                    raw_result=dict(raw_result),
                )
            except BaseException:
                reply = WorkerReply(
                    invocation_id=command.invocation_id,
                    request_digest=command.request_digest,
                    error=traceback.format_exc(),
                )
            reply_queue.put(reply)
    finally:
        close_worker = getattr(runner, "close_worker", None)
        if callable(close_worker):
            close_worker()


class _ProcessWorkerChannel:
    """Queue-backed persistent worker channel owned by one hardware lease."""

    def __init__(
        self,
        command_queue: Any,
        reply_queue: Any,
        process: Any,
        *,
        shutdown_timeout_seconds: float,
    ) -> None:
        self._command_queue = command_queue
        self._reply_queue = reply_queue
        self._process = process
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._closed = False
        self._joined = False

    @property
    def ready_handles(self) -> tuple[object, object]:
        return (self._reply_queue._reader, self._process.sentinel)

    def send(self, command: WorkerCommand) -> None:
        if self._closed:
            raise RuntimeError("persistent worker channel is closed")
        self._command_queue.put(command)

    def recv(self) -> object:
        return self._reply_queue.get()

    def is_alive(self) -> bool:
        return bool(self._process.is_alive())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._command_queue.put(None)

    def join(self) -> None:
        if self._joined:
            return
        self._joined = True

        def _bounded_join() -> None:
            try:
                self._process.join(self._shutdown_timeout_seconds)
            except TypeError:
                # Test doubles and a few process-like transports expose join()
                # without multiprocessing's optional timeout.
                self._process.join()

        forced_termination = False
        _bounded_join()
        if self._process.is_alive():
            forced_termination = True
            self._process.terminate()
            _bounded_join()
        if self._process.is_alive():
            kill = getattr(self._process, "kill", None)
            if not callable(kill):
                raise RuntimeError("persistent worker survived termination and cannot be killed")
            kill()
            _bounded_join()
        if self._process.is_alive():
            raise RuntimeError("persistent worker survived forced kill")

        for queue_object in (self._command_queue, self._reply_queue):
            if forced_termination:
                cancel_join_thread = getattr(queue_object, "cancel_join_thread", None)
                if callable(cancel_join_thread):
                    cancel_join_thread()
            close = getattr(queue_object, "close", None)
            if callable(close):
                close()
            if not forced_termination:
                join_thread = getattr(queue_object, "join_thread", None)
                if callable(join_thread):
                    join_thread()


class ProcessWorkerFactory:
    """Create spawn-safe persistent workers and wait on replies or exits."""

    def __init__(
        self,
        *,
        context_getter: Callable[[str], Any] = multiprocessing.get_context,
        worker_target: Callable[..., None] = worker_process_main,
        connection_wait: Callable[[Sequence[object], float], Sequence[object]] | None = None,
        shutdown_timeout_seconds: float = 5.0,
    ) -> None:
        if shutdown_timeout_seconds < 0:
            raise ValueError("worker shutdown timeout must be non-negative")
        if connection_wait is None:
            from multiprocessing.connection import wait as connection_wait

        self._context = context_getter("spawn")
        self._worker_target = worker_target
        self._connection_wait = connection_wait
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

    def __call__(self, bootstrap: WorkerBootstrap) -> WorkerChannel:
        command_queue = self._context.Queue()
        reply_queue = self._context.Queue()
        process = self._context.Process(
            target=self._worker_target,
            args=(bootstrap, command_queue, reply_queue),
            daemon=False,
        )
        process.start()
        return _ProcessWorkerChannel(
            command_queue,
            reply_queue,
            process,
            shutdown_timeout_seconds=self._shutdown_timeout_seconds,
        )

    def wait_ready(
        self,
        channels: Sequence[WorkerChannel],
        timeout_seconds: float,
    ) -> tuple[WorkerChannel, ...]:
        if timeout_seconds < 0:
            raise ValueError("worker wait timeout must be non-negative")
        handles: list[object] = []
        owner_by_handle_id: dict[int, WorkerChannel] = {}
        for channel in channels:
            if not isinstance(channel, _ProcessWorkerChannel):
                raise TypeError("process worker factory can only wait on its own channels")
            for handle in channel.ready_handles:
                handles.append(handle)
                owner_by_handle_id[id(handle)] = channel
        if not handles:
            return ()
        ready_handle_ids = {id(handle) for handle in self._connection_wait(handles, timeout_seconds)}
        return tuple(
            channel for channel in channels if any(id(handle) in ready_handle_ids for handle in channel.ready_handles)
        )


@dataclass(frozen=True, slots=True)
class _PreparedJob:
    index: int
    request: MeasurementRequest
    adapter: Any
    prepared: PreparedMeasurement
    job: CollectionJob


@dataclass(frozen=True, slots=True)
class _ActiveInvocation:
    prepared_job: _PreparedJob
    assignment: Assignment
    channel: WorkerChannel
    command: WorkerCommand


@dataclass(frozen=True, slots=True)
class _LeaseKey:
    adapter_namespace: str
    adapter_module: str
    run_module: str
    run_func: str
    protocol_digest: str
    device_uuids: tuple[str, ...]
    topology_fingerprint: str


class PersistentMeasurementExecutor:
    """Run prepared measurements in hardware-safe waves on persistent workers."""

    def __init__(
        self,
        *,
        inventory: HardwareInventory,
        scheduler: HardwareAwareScheduler,
        resolve_adapter: Callable[[MeasurementRequest], Any],
        worker_factory: Callable[[WorkerBootstrap], WorkerChannel],
        wait_ready: Callable[[Sequence[WorkerChannel], float], Sequence[WorkerChannel]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if scheduler.inventory.topology_fingerprint != inventory.topology_fingerprint:
            raise ValueError("scheduler inventory does not match executor inventory")
        self._inventory = inventory
        self._scheduler = scheduler
        self._resolve_adapter = resolve_adapter
        self._worker_factory = worker_factory
        self._wait_ready = wait_ready
        self._clock = clock
        self._device_by_id = {device.index: device for device in inventory.devices}
        self._leases: dict[_LeaseKey, WorkerChannel] = {}
        self._closed_channel_ids: set[int] = set()
        self._lock = threading.RLock()
        self._closed = False

    def execute(
        self,
        requests: Sequence[MeasurementRequest],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> Sequence[MeasurementRecord]:
        """Prepare all requests, then submit and drain one complete wave at a time."""

        with self._lock:
            if self._closed:
                raise RuntimeError("measurement executor is closed")
            request_tuple = tuple(requests)
            if not request_tuple:
                return ()
            if cancellation.cancelled():
                return tuple(
                    self._failure(request, UnresolvedCode.CANCELLED, "collection cancelled before preparation")
                    for request in request_tuple
                )
            if self._clock() >= deadline_monotonic:
                return tuple(
                    self._failure(request, UnresolvedCode.TIMEOUT, "collection deadline expired before preparation")
                    for request in request_tuple
                )

            records: list[MeasurementRecord | None] = [None] * len(request_tuple)
            prepared_jobs = self._prepare_jobs(request_tuple, records)
            if prepared_jobs:
                waves = self._scheduler.plan(tuple(item.job for item in prepared_jobs))
                by_digest = {item.job.request_digest: item for item in prepared_jobs}
                self._execute_waves(
                    waves,
                    by_digest,
                    records,
                    deadline_monotonic=deadline_monotonic,
                    cancellation=cancellation,
                )

            if any(record is None for record in records):
                raise RuntimeError("executor failed to produce one record per request")
            return tuple(record for record in records if record is not None)

    def close(self) -> None:
        """Close every persistent worker exactly once."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            channels = tuple({id(channel): channel for channel in self._leases.values()}.values())
            self._leases.clear()
            for channel in channels:
                self._close_channel(channel)

    def _prepare_jobs(
        self,
        requests: tuple[MeasurementRequest, ...],
        records: list[MeasurementRecord | None],
    ) -> list[_PreparedJob]:
        digest_counts = Counter(request.key.digest for request in requests)
        prepared_jobs: list[_PreparedJob] = []
        for index, request in enumerate(requests):
            if digest_counts[request.key.digest] > 1:
                records[index] = self._failure(
                    request,
                    UnresolvedCode.INVALID_MEASUREMENT,
                    "duplicate request identity in one executor batch",
                )
                continue
            requested_gpu_class = " ".join(request.environment.gpu_class.split()).casefold()
            if any(
                " ".join(device.name.split()).casefold() != requested_gpu_class for device in self._inventory.devices
            ):
                records[index] = self._failure(
                    request,
                    UnresolvedCode.IDENTITY_MISMATCH,
                    "request GPU class does not match the collector inventory",
                )
                continue
            if request.environment.topology_fingerprint != self._inventory.topology_fingerprint:
                records[index] = self._failure(
                    request,
                    UnresolvedCode.TOPOLOGY_MISMATCH,
                    "request topology fingerprint does not match collector inventory",
                )
                continue

            try:
                adapter = self._resolve_adapter(request)
            except Exception as error:
                records[index] = self._failure(
                    request,
                    UnresolvedCode.MISSING_ADAPTER,
                    f"adapter resolution failed: {error}",
                )
                continue
            try:
                prepared = adapter.prepare(request)
            except ValueError as error:
                records[index] = self._failure(request, UnresolvedCode.UNSUPPORTED_SHAPE, str(error))
                continue
            except Exception as error:
                records[index] = self._failure(
                    request,
                    UnresolvedCode.INVALID_MEASUREMENT,
                    f"adapter preparation failed: {error}",
                )
                continue
            if not isinstance(prepared, PreparedMeasurement):
                records[index] = self._failure(
                    request,
                    UnresolvedCode.INVALID_MEASUREMENT,
                    "adapter preparation did not return PreparedMeasurement",
                )
                continue

            try:
                payload = canonical_json(prepared.case).encode("utf-8")
            except (TypeError, ValueError) as error:
                records[index] = self._failure(
                    request,
                    UnresolvedCode.INVALID_MEASUREMENT,
                    f"adapter case is not canonical JSON: {error}",
                )
                continue
            job = CollectionJob(
                request_digest=request.key.digest,
                adapter_namespace=adapter.lazy.namespace,
                contract=prepared.contract,
                payload=payload,
            )
            try:
                self._scheduler.plan((job,))
            except UnschedulableRequest as error:
                records[index] = self._failure(request, UnresolvedCode.RESOURCE_UNAVAILABLE, str(error))
                continue
            prepared_jobs.append(_PreparedJob(index, request, adapter, prepared, job))
        return prepared_jobs

    def _execute_waves(
        self,
        waves: tuple[tuple[Assignment, ...], ...],
        by_digest: Mapping[str, _PreparedJob],
        records: list[MeasurementRecord | None],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> None:
        for wave_index, wave in enumerate(waves):
            stop_code = self._stop_code(deadline_monotonic, cancellation)
            if stop_code is not None:
                self._fail_remaining_waves(waves[wave_index:], by_digest, records, stop_code)
                return

            active: dict[int, _ActiveInvocation] = {}
            for assignment in wave:
                prepared_job = by_digest[assignment.job.request_digest]
                try:
                    channel = self._acquire_channel(prepared_job, assignment)
                except Exception as error:
                    records[prepared_job.index] = self._failure(
                        prepared_job.request,
                        UnresolvedCode.COLLECTOR_FAILED,
                        f"worker acquisition failed: {error}",
                    )
                    continue

                command = WorkerCommand(
                    invocation_id=uuid.uuid4().hex,
                    request_digest=prepared_job.request.key.digest,
                    payload=prepared_job.job.payload,
                    protocol=prepared_job.request.protocol,
                )
                try:
                    channel.send(command)
                except Exception as error:
                    records[prepared_job.index] = self._failure(
                        prepared_job.request,
                        UnresolvedCode.COLLECTOR_FAILED,
                        f"worker send failed: {error}",
                    )
                    self._evict(channel)
                    continue
                active[id(channel)] = _ActiveInvocation(prepared_job, assignment, channel, command)

            self._drain_wave(
                active,
                records,
                deadline_monotonic=deadline_monotonic,
                cancellation=cancellation,
            )
            stop_code = self._stop_code(deadline_monotonic, cancellation)
            if stop_code is not None and wave_index + 1 < len(waves):
                self._fail_remaining_waves(waves[wave_index + 1 :], by_digest, records, stop_code)
                return

    def _drain_wave(
        self,
        active: dict[int, _ActiveInvocation],
        records: list[MeasurementRecord | None],
        *,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> None:
        while active:
            stop_code = self._stop_code(deadline_monotonic, cancellation)
            if stop_code is not None:
                self._fail_active(active, records, stop_code)
                return

            timeout_seconds = max(0.0, deadline_monotonic - self._clock())
            try:
                ready = tuple(self._wait_ready(tuple(item.channel for item in active.values()), timeout_seconds))
            except Exception as error:
                for invocation in tuple(active.values()):
                    records[invocation.prepared_job.index] = self._failure(
                        invocation.prepared_job.request,
                        UnresolvedCode.COLLECTOR_FAILED,
                        f"worker readiness failed: {error}",
                    )
                    self._evict(invocation.channel)
                active.clear()
                return

            if not ready:
                stop_code = self._stop_code(deadline_monotonic, cancellation) or UnresolvedCode.TIMEOUT
                self._fail_active(active, records, stop_code)
                return

            for channel in ready:
                invocation = active.pop(id(channel), None)
                if invocation is None:
                    continue
                records[invocation.prepared_job.index] = self._receive(invocation)

    def _receive(self, invocation: _ActiveInvocation) -> MeasurementRecord:
        request = invocation.prepared_job.request
        channel = invocation.channel
        try:
            alive = channel.is_alive()
        except Exception as error:
            self._evict(channel)
            return self._failure(
                request,
                UnresolvedCode.COLLECTOR_FAILED,
                f"worker health probe failed: {error}",
            )
        if not alive:
            self._evict(channel)
            return self._failure(request, UnresolvedCode.COLLECTOR_FAILED, "worker exited before replying")
        try:
            reply = channel.recv()
        except Exception as error:
            self._evict(channel)
            return self._failure(request, UnresolvedCode.COLLECTOR_FAILED, f"worker receive failed: {error}")
        if not isinstance(reply, WorkerReply):
            self._evict(channel)
            return self._failure(request, UnresolvedCode.INVALID_MEASUREMENT, "worker returned a malformed reply")
        if (
            reply.invocation_id != invocation.command.invocation_id
            or reply.request_digest != invocation.command.request_digest
        ):
            self._evict(channel)
            return self._failure(
                request,
                UnresolvedCode.IDENTITY_MISMATCH,
                "worker reply identity does not match the active invocation",
            )
        if reply.error is not None:
            self._evict(channel)
            return self._failure(request, UnresolvedCode.COLLECTOR_FAILED, f"worker failed: {reply.error}")
        if isinstance(reply.raw_result, MeasurementRecord) or not isinstance(reply.raw_result, Mapping):
            self._evict(channel)
            return self._failure(
                request,
                UnresolvedCode.INVALID_MEASUREMENT,
                "worker must return a raw result mapping",
            )
        try:
            record = invocation.prepared_job.adapter.record(invocation.prepared_job.prepared, reply.raw_result)
            parent_provenance = {
                "measurement_environment": json.loads(request.environment.canonical),
                "assigned_device_uuids": [self._device_by_id[gpu_id].uuid for gpu_id in invocation.assignment.gpu_ids],
                "assigned_gpu_ids": list(invocation.assignment.gpu_ids),
                "topology_fingerprint": request.environment.topology_fingerprint,
                "invocation_id": invocation.command.invocation_id,
                "request_digest": invocation.command.request_digest,
            }
            return replace(
                record,
                provenance={**dict(record.provenance), **parent_provenance},
            )
        except Exception as error:
            self._evict(channel)
            return self._failure(
                request,
                UnresolvedCode.INVALID_MEASUREMENT,
                f"adapter result conversion failed: {error}",
            )

    def _acquire_channel(self, prepared_job: _PreparedJob, assignment: Assignment) -> WorkerChannel:
        device_uuids = tuple(self._device_by_id[gpu_id].uuid for gpu_id in assignment.gpu_ids)
        lazy = prepared_job.adapter.lazy
        key = _LeaseKey(
            adapter_namespace=lazy.namespace,
            adapter_module=lazy.adapter_module,
            run_module=lazy.run_module,
            run_func=lazy.run_func,
            protocol_digest=prepared_job.request.protocol.digest,
            device_uuids=device_uuids,
            topology_fingerprint=self._inventory.topology_fingerprint,
        )
        channel = self._leases.get(key)
        if channel is not None:
            try:
                if channel.is_alive():
                    return channel
            except Exception as error:
                self._evict(channel)
                raise RuntimeError(f"worker health probe failed: {error}") from error
            self._evict(channel)
        bootstrap = WorkerBootstrap(
            run_module=lazy.run_module,
            run_func=lazy.run_func,
            adapter_namespace=lazy.namespace,
            protocol_digest=prepared_job.request.protocol.digest,
            device_uuids=device_uuids,
            topology_fingerprint=self._inventory.topology_fingerprint,
        )
        channel = self._worker_factory(bootstrap)
        self._leases[key] = channel
        return channel

    def _stop_code(
        self,
        deadline_monotonic: float,
        cancellation: CancellationToken,
    ) -> UnresolvedCode | None:
        if cancellation.cancelled():
            return UnresolvedCode.CANCELLED
        if self._clock() >= deadline_monotonic:
            return UnresolvedCode.TIMEOUT
        return None

    def _fail_active(
        self,
        active: dict[int, _ActiveInvocation],
        records: list[MeasurementRecord | None],
        code: UnresolvedCode,
    ) -> None:
        detail = "collection cancelled" if code is UnresolvedCode.CANCELLED else "collection deadline expired"
        for invocation in tuple(active.values()):
            records[invocation.prepared_job.index] = self._failure(invocation.prepared_job.request, code, detail)
            self._evict(invocation.channel)
        active.clear()

    def _fail_remaining_waves(
        self,
        waves: Sequence[Sequence[Assignment]],
        by_digest: Mapping[str, _PreparedJob],
        records: list[MeasurementRecord | None],
        code: UnresolvedCode,
    ) -> None:
        detail = "collection cancelled before submission" if code is UnresolvedCode.CANCELLED else "deadline expired"
        for wave in waves:
            for assignment in wave:
                prepared_job = by_digest[assignment.job.request_digest]
                records[prepared_job.index] = self._failure(prepared_job.request, code, detail)

    def _evict(self, channel: WorkerChannel) -> None:
        stale_keys = [key for key, candidate in self._leases.items() if candidate is channel]
        for key in stale_keys:
            del self._leases[key]
        self._close_channel(channel)

    def _close_channel(self, channel: WorkerChannel) -> None:
        channel_id = id(channel)
        if channel_id in self._closed_channel_ids:
            return
        self._closed_channel_ids.add(channel_id)
        try:
            channel.close()
        except Exception:
            pass
        try:
            channel.join()
        except Exception:
            pass

    @staticmethod
    def _failure(
        request: MeasurementRequest,
        code: UnresolvedCode,
        detail: str,
    ) -> MeasurementRecord:
        return MeasurementRecord(
            key=request.key,
            status=RecordStatus.FAILED,
            latency_ms=None,
            energy_wms=0.0,
            samples_ms=(),
            protocol=request.protocol,
            perf_row={},
            provenance={},
            failure_code=code,
            failure_reason=detail,
        )
