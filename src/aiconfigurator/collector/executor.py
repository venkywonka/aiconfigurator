# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent, hardware-aware execution for exact lazy measurements.

The parent process owns request validation, hardware placement, result
conversion, and failure typing.  Workers receive only canonical JSON cases and
return raw result mappings; they never construct or persist performance rows.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

from aiconfigurator.collector.adapters import PreparedMeasurement
from aiconfigurator.collector.scheduler import HardwareAwareScheduler, UnschedulableRequest
from aiconfigurator.collector.types import Assignment, CollectionJob, HardwareInventory
from aiconfigurator.sdk.resolution.session import CancellationToken
from aiconfigurator.sdk.resolution.types import (
    MeasurementRecord,
    MeasurementRequest,
    RecordStatus,
    UnresolvedCode,
    canonical_json,
)

__all__ = [
    "PersistentMeasurementExecutor",
    "WorkerBootstrap",
    "WorkerCommand",
    "WorkerReply",
    "bind_and_import_runner",
]


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
            return invocation.prepared_job.adapter.record(invocation.prepared_job.prepared, reply.raw_result)
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
