#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run bounded DSv4 attention cold/warm/reopen GPU lifecycle gates.

This is an experiment gate, not an offline collector. It never enumerates a
grid or mutates curated data. The four exact physical requests flow through the
normal lazy adapter, hardware scheduler, worker command, overlay, and replay
path. The local gate isolates one visible GB200 for kernel execution; the
scheduler gate keeps the full compatible inventory visible and proves disjoint
one-GPU worker leases. A fresh overlay is mandatory so the cold command count
is meaningful.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from unittest import mock

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.executor import (
    PersistentMeasurementExecutor,
    ProcessWorkerFactory,
    WorkerBootstrap,
    WorkerChannel,
)
from aiconfigurator.collector.hardware import discover_hardware
from aiconfigurator.collector.scheduler import HardwareAwareScheduler
from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
from aiconfigurator.collector.types import HardwareInventory
from aiconfigurator.sdk import config
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.operations.dsv4 import (
    ContextDeepSeekV4AttentionModule,
    GenerationDeepSeekV4AttentionModule,
)
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionSession
from aiconfigurator.sdk.resolution.types import MeasurementEnvironment, MeasurementProtocol, RecordStatus

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_ARCHITECTURE = "DeepseekV4ForCausalLM"
_SUPPORTED_SGLANG_VERSIONS = frozenset({"0.5.10", "0.5.10rc0"})
_BASE_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
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


def _json_safe(value: Any) -> Any:
    """Recursively normalize immutable resolution mappings for JSON receipts."""

    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("attention lifecycle receipt mappings require string keys")
            normalized[key] = _json_safe(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AttentionRoute:
    """One exact bounded runtime projection and its physical dataset route."""

    route_id: str
    mode: str
    attn_kind: str
    compress_ratio: int
    namespace: str
    runtime_items: tuple[tuple[str, object], ...]

    @property
    def runtime_kwargs(self) -> dict[str, object]:
        return dict(self.runtime_items)


def _context_runtime_items() -> tuple[tuple[str, object], ...]:
    return (
        ("x", 128),
        ("batch_size", 1),
        ("beam_width", 1),
        ("s", 128),
        ("prefix", 64),
        ("model_name", _MODEL_ARTIFACT),
        ("seq_imbalance_correction_scale", 1.0),
    )


def _generation_runtime_items() -> tuple[tuple[str, object], ...]:
    return (
        ("x", 1),
        ("batch_size", 1),
        ("beam_width", 1),
        ("s", 128),
        ("prefix", 64),
        ("model_name", _MODEL_ARTIFACT),
        ("gen_seq_imbalance_correction_scale", 1.0),
    )


ATTENTION_ROUTE_MATRIX = (
    AttentionRoute(
        "context-csa",
        "context",
        "csa",
        4,
        "dsv4_csa_context_module_perf.txt/v1",
        _context_runtime_items(),
    ),
    AttentionRoute(
        "context-hca",
        "context",
        "hca",
        128,
        "dsv4_hca_context_module_perf.txt/v1",
        _context_runtime_items(),
    ),
    AttentionRoute(
        "generation-csa",
        "generation",
        "csa",
        4,
        "dsv4_csa_generation_module_perf.txt/v1",
        _generation_runtime_items(),
    ),
    AttentionRoute(
        "generation-hca",
        "generation",
        "hca",
        128,
        "dsv4_hca_generation_module_perf.txt/v1",
        _generation_runtime_items(),
    ),
)


@dataclass(frozen=True, slots=True)
class ResultEvidence:
    latency_ms: float
    energy_wms: float
    source: str


@dataclass(frozen=True, slots=True)
class StageEvidence:
    """Per-stage results with both cumulative and newly issued commands."""

    command_count: int
    additional_commands: int
    results: tuple[ResultEvidence, ...]

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(result.source for result in self.results)


@dataclass(frozen=True, slots=True)
class RecordEvidence:
    namespace: str
    key_digest: str
    status: str
    latency_ms: float
    energy_wms: float
    samples_ms: tuple[float, ...]
    protocol_digest: str
    perf_row: Mapping[str, object]
    provenance: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class InventoryEvidence:
    schema_revision: str
    topology_fingerprint: str
    gpu_ids: tuple[int, ...]
    device_uuids: tuple[str, ...]
    gpu_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeaseEvidence:
    route_id: str
    key_digest: str
    contract_gpu_count: int
    assigned_gpu_ids: tuple[int, ...]
    assigned_device_uuids: tuple[str, ...]
    worker_visible_device_uuids: tuple[str, ...]
    worker_cuda_visible_devices: str
    worker_local_ordinals: tuple[int, ...]
    worker_topology_fingerprint: str
    remaining_gpu_ids: tuple[int, ...]
    remaining_device_uuids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WorkerExecutionEvidence:
    bootstrap_count: int
    command_count: int
    commands_per_worker: tuple[int, ...]
    bootstrap_device_uuids: tuple[tuple[str, ...], ...]
    reused_worker_count: int


@dataclass(frozen=True, slots=True)
class AttentionLifecycleReport:
    inventory: InventoryEvidence
    resource_contract_gpu_count: int
    leases: tuple[LeaseEvidence, ...]
    route_ids: tuple[str, ...]
    namespaces: tuple[str, ...]
    key_digests: tuple[str, ...]
    pure: StageEvidence
    cold: StageEvidence
    warm: StageEvidence
    reopened: StageEvidence
    cold_unique_misses: int
    cold_accepted_records: int
    reopened_unique_misses: int
    records: tuple[RecordEvidence, ...]

    def to_dict(self) -> dict[str, object]:
        return _json_safe(asdict(self))


@dataclass(frozen=True, slots=True)
class LocalGpuGateReport:
    runtime: Mapping[str, object]
    lifecycle: AttentionLifecycleReport
    worker_execution: WorkerExecutionEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime": _json_safe(self.runtime),
            "lifecycle": self.lifecycle.to_dict(),
            "worker_execution": _json_safe(asdict(self.worker_execution)),
        }


class _EmptyProfileDatabase:
    """Frozen profile identity plus an explicit no-curated-row control."""

    def __init__(self, environment: MeasurementEnvironment) -> None:
        self.system = environment.system
        self.backend = environment.backend
        self.version = environment.backend_version
        self.measurement_environment = environment
        # Resolution always asks for a SILICON-configured view; mirror the
        # corresponding root-template invariant even though this control owns
        # no curated rows.
        self.enable_shared_layer = True
        self.transfer_policy = None

    @staticmethod
    def query_context_deepseek_v4_attention_module(**query: Any) -> PerformanceResult:
        del query
        return PerformanceResult(1.0, energy=2.0, source="silicon")

    @staticmethod
    def query_generation_deepseek_v4_attention_module(**query: Any) -> PerformanceResult:
        del query
        return PerformanceResult(1.0, energy=2.0, source="silicon")


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
    )


def _environment(
    inventory: HardwareInventory,
    *,
    backend_version: str = "0.5.10",
) -> MeasurementEnvironment:
    if not inventory.devices:
        raise ValueError("DSv4 attention lifecycle requires at least one visible GPU")
    if backend_version not in _SUPPORTED_SGLANG_VERSIONS:
        raise ValueError(f"unsupported DSv4 attention SGLang version {backend_version!r}")
    incompatible = tuple(
        (device.index, device.name)
        for device in inventory.devices
        if " ".join(device.name.split()).casefold() != "nvidia gb200"
    )
    if incompatible:
        raise ValueError(f"DSv4 attention lifecycle requires all inventory GPUs to be NVIDIA GB200: {incompatible!r}")
    device = inventory.devices[0]
    return MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version=backend_version,
        gpu_class=device.name,
        runtime_versions={**_BASE_RUNTIME_VERSIONS, "sglang": backend_version},
        topology_schema=inventory.schema_revision,
        topology_fingerprint=inventory.topology_fingerprint,
        profile_compatibility=_PROFILE_COMPATIBILITY,
    )


def _operations() -> tuple[object, ...]:
    model_config = config.ModelConfig(
        tp_size=4,
        pp_size=1,
        attention_dp_size=1,
        cp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        nextn=0,
        workload_distribution="power_law",
        moe_backend=None,
    )
    model = get_model(_MODEL_ARTIFACT, model_config, backend_name="sglang")
    selected: list[object] = []
    for route in ATTENTION_ROUTE_MATRIX:
        operation_type = (
            ContextDeepSeekV4AttentionModule if route.mode == "context" else GenerationDeepSeekV4AttentionModule
        )
        operations = model.context_ops if route.mode == "context" else model.generation_ops
        selected.append(
            next(
                operation
                for operation in operations
                if isinstance(operation, operation_type) and operation._compress_ratio == route.compress_ratio
            )
        )
    return tuple(selected)


def _resolve_adapter_index() -> Callable[[object], object]:
    adapter_index = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY})

    def resolve_adapter(request):
        routes = adapter_index.routes_for(
            (request.key.namespace, request.environment.backend, request.environment.backend_version)
        )
        if len(routes) != 1:
            raise RuntimeError(f"expected exactly one adapter for {request.key.namespace!r}, found {len(routes)}")
        return routes[0]

    return resolve_adapter


def _stage(
    command_count: Callable[[], int],
    results: Sequence[PerformanceResult],
    *,
    previous_command_count: int,
) -> StageEvidence:
    cumulative_command_count = command_count()
    if cumulative_command_count < previous_command_count:
        raise RuntimeError("lifecycle command counter moved backwards")
    return StageEvidence(
        command_count=cumulative_command_count,
        additional_commands=cumulative_command_count - previous_command_count,
        results=tuple(
            ResultEvidence(
                latency_ms=float(result),
                energy_wms=float(result.energy),
                source=str(result.source),
            )
            for result in results
        ),
    )


def _query_routes(operations, database, *, session: ResolutionSession | None) -> tuple[PerformanceResult, ...]:
    return tuple(
        operation.query_with_resolution(database, session=session, **route.runtime_kwargs)
        for route, operation in zip(ATTENTION_ROUTE_MATRIX, operations, strict=True)
    )


def _executor(
    inventory: HardwareInventory,
    worker_factory: Callable[[WorkerBootstrap], WorkerChannel],
    wait_ready: Callable[[Sequence[WorkerChannel], float], Sequence[WorkerChannel]],
) -> PersistentMeasurementExecutor:
    return PersistentMeasurementExecutor(
        inventory=inventory,
        scheduler=HardwareAwareScheduler(inventory),
        resolve_adapter=_resolve_adapter_index(),
        worker_factory=worker_factory,
        wait_ready=wait_ready,
    )


def _record_evidence(record) -> RecordEvidence:
    if record.status is not RecordStatus.VALID or record.latency_ms is None:
        raise RuntimeError(f"attention lifecycle persisted a non-valid record for {record.key.digest}")
    if not math.isfinite(record.latency_ms) or record.latency_ms <= 0:
        raise RuntimeError(f"attention lifecycle persisted an invalid latency for {record.key.digest}")
    return RecordEvidence(
        namespace=record.key.namespace,
        key_digest=record.key.digest,
        status=record.status.value,
        latency_ms=float(record.latency_ms),
        energy_wms=float(record.energy_wms or 0.0),
        samples_ms=tuple(float(sample) for sample in record.samples_ms),
        protocol_digest=record.protocol.digest,
        perf_row=dict(record.perf_row),
        provenance=dict(record.provenance),
    )


def _inventory_evidence(inventory: HardwareInventory) -> InventoryEvidence:
    return InventoryEvidence(
        schema_revision=inventory.schema_revision,
        topology_fingerprint=inventory.topology_fingerprint,
        gpu_ids=tuple(device.index for device in inventory.devices),
        device_uuids=tuple(device.uuid for device in inventory.devices),
        gpu_names=tuple(device.name for device in inventory.devices),
    )


def _lease_evidence(
    inventory: HardwareInventory,
    records: Sequence[RecordEvidence],
) -> tuple[LeaseEvidence, ...]:
    device_by_id = {device.index: device for device in inventory.devices}
    all_gpu_ids = tuple(device.index for device in inventory.devices)
    leases: list[LeaseEvidence] = []
    for route, record in zip(ATTENTION_ROUTE_MATRIX, records, strict=True):
        assigned_gpu_ids = tuple(int(gpu_id) for gpu_id in record.provenance["assigned_gpu_ids"])
        assigned_device_uuids = tuple(str(uuid) for uuid in record.provenance["assigned_device_uuids"])
        if len(assigned_gpu_ids) != 1 or len(assigned_device_uuids) != 1:
            raise RuntimeError("DSv4 attention resource contract must produce exactly one-GPU leases")
        expected_uuids = tuple(device_by_id[gpu_id].uuid for gpu_id in assigned_gpu_ids)
        if assigned_device_uuids != expected_uuids:
            raise RuntimeError("assigned GPU IDs and UUIDs disagree")
        worker_binding = record.provenance.get("worker_binding")
        if not isinstance(worker_binding, Mapping):
            raise TypeError("attention lifecycle record is missing worker binding evidence")
        worker_visible_device_uuids = tuple(str(uuid) for uuid in worker_binding.get("device_uuids", ()))
        worker_cuda_visible_devices = str(worker_binding.get("cuda_visible_devices", ""))
        worker_local_ordinals = tuple(int(index) for index in worker_binding.get("local_ordinals", ()))
        worker_topology_fingerprint = str(worker_binding.get("topology_fingerprint", ""))
        if worker_visible_device_uuids != assigned_device_uuids:
            raise RuntimeError("worker-visible UUIDs differ from the assigned hardware lease")
        if worker_cuda_visible_devices != ",".join(assigned_device_uuids):
            raise RuntimeError("worker CUDA visibility differs from the assigned hardware lease")
        if worker_local_ordinals != tuple(range(len(assigned_device_uuids))):
            raise RuntimeError("worker local ordinals do not match its restricted visibility")
        if worker_topology_fingerprint != inventory.topology_fingerprint:
            raise RuntimeError("worker topology fingerprint differs from the full inventory")
        remaining_gpu_ids = tuple(gpu_id for gpu_id in all_gpu_ids if gpu_id not in assigned_gpu_ids)
        leases.append(
            LeaseEvidence(
                route_id=route.route_id,
                key_digest=record.key_digest,
                contract_gpu_count=1,
                assigned_gpu_ids=assigned_gpu_ids,
                assigned_device_uuids=assigned_device_uuids,
                worker_visible_device_uuids=worker_visible_device_uuids,
                worker_cuda_visible_devices=worker_cuda_visible_devices,
                worker_local_ordinals=worker_local_ordinals,
                worker_topology_fingerprint=worker_topology_fingerprint,
                remaining_gpu_ids=remaining_gpu_ids,
                remaining_device_uuids=tuple(device_by_id[gpu_id].uuid for gpu_id in remaining_gpu_ids),
            )
        )
    return tuple(leases)


def run_attention_lifecycle(
    *,
    overlay_path: Path,
    inventory: HardwareInventory,
    worker_factory: Callable[[WorkerBootstrap], WorkerChannel],
    wait_ready: Callable[[Sequence[WorkerChannel], float], Sequence[WorkerChannel]],
    command_count: Callable[[], int],
    max_wall_seconds: float,
    backend_version: str = "0.5.10",
) -> AttentionLifecycleReport:
    """Run pure, cold, warm, and close/reopen passes for four exact routes."""

    overlay_path = Path(overlay_path)
    if overlay_path.exists():
        raise FileExistsError(f"lifecycle overlay must start absent: {overlay_path}")
    if not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive and finite")
    if command_count() != 0:
        raise ValueError("lifecycle command counter must start at zero")
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    protocol = _protocol()
    database = _EmptyProfileDatabase(_environment(inventory, backend_version=backend_version))
    operations = _operations()
    requests = tuple(
        operation.measurement_request(database, protocol, **route.runtime_kwargs)
        for route, operation in zip(ATTENTION_ROUTE_MATRIX, operations, strict=True)
    )
    if any(request is None for request in requests):
        raise RuntimeError("one or more frozen attention operations did not produce a measurement request")
    concrete_requests = tuple(request for request in requests if request is not None)
    if tuple(request.key.namespace for request in concrete_requests) != tuple(
        route.namespace for route in ATTENTION_ROUTE_MATRIX
    ):
        raise RuntimeError("attention operation namespaces differ from the frozen route matrix")
    if len({request.key for request in concrete_requests}) != len(ATTENTION_ROUTE_MATRIX):
        raise RuntimeError("attention route matrix did not produce four unique PerfKeys")

    with ExitStack() as stack:
        no_load = classmethod(lambda cls, profile_database: None)
        stack.enter_context(mock.patch.object(ContextDeepSeekV4AttentionModule, "load_data", no_load))
        stack.enter_context(mock.patch.object(GenerationDeepSeekV4AttentionModule, "load_data", no_load))

        pure = _stage(
            command_count,
            _query_routes(operations, database, session=None),
            previous_command_count=0,
        )
        if pure.additional_commands != 0:
            raise RuntimeError("default/pure attention prediction issued a worker command")

        overlay = OverlayStore(overlay_path)
        executor = _executor(inventory, worker_factory, wait_ready)
        session = ResolutionSession(
            overlay,
            executor,
            ResolutionBudget(max_new_keys=4, max_wall_seconds=max_wall_seconds),
            protocol,
        )
        try:
            cold = _stage(
                command_count,
                session.execute_callback(lambda: _query_routes(operations, database, session=session)),
                previous_command_count=pure.command_count,
            )
            records = tuple(overlay.lookup(request.key, protocol) for request in concrete_requests)
            if any(record is None for record in records):
                raise RuntimeError("cold attention lifecycle did not persist all four records")
            record_evidence = tuple(_record_evidence(record) for record in records if record is not None)
            warm = _stage(
                command_count,
                session.execute_callback(lambda: _query_routes(operations, database, session=session)),
                previous_command_count=cold.command_count,
            )
            cold_unique_misses = session.report.unique_misses
            cold_accepted_records = session.report.accepted_records
        finally:
            overlay.close()
            executor.close()

        reopened_overlay = OverlayStore(overlay_path)
        reopened_executor = _executor(inventory, worker_factory, wait_ready)
        reopened_session = ResolutionSession(
            reopened_overlay,
            reopened_executor,
            ResolutionBudget(max_new_keys=4, max_wall_seconds=max_wall_seconds),
            protocol,
        )
        try:
            reopened = _stage(
                command_count,
                reopened_session.execute_callback(
                    lambda: _query_routes(operations, database, session=reopened_session)
                ),
                previous_command_count=warm.command_count,
            )
            reopened_unique_misses = reopened_session.report.unique_misses
        finally:
            reopened_overlay.close()
            reopened_executor.close()

    route_count = len(ATTENTION_ROUTE_MATRIX)
    if cold.additional_commands != route_count:
        raise RuntimeError(
            f"cold attention lifecycle expected {route_count} new commands, got {cold.additional_commands}"
        )
    if warm.additional_commands != 0 or reopened.additional_commands != 0:
        raise RuntimeError("warm or reopened attention lifecycle issued additional worker commands")
    if cold_unique_misses != route_count or cold_accepted_records != route_count:
        raise RuntimeError("cold attention lifecycle did not accept one record per unique route")
    if reopened_unique_misses != 0:
        raise RuntimeError("reopened attention lifecycle observed a persisted-key miss")
    if cold.results != warm.results or cold.results != reopened.results:
        raise RuntimeError("cold, warm, and reopened attention results differ")
    leases = _lease_evidence(inventory, record_evidence)
    if len(inventory.devices) >= route_count:
        assigned_gpu_ids = tuple(lease.assigned_gpu_ids for lease in leases)
        if len(set(assigned_gpu_ids)) != route_count:
            raise RuntimeError("attention lifecycle did not pack routes onto disjoint one-GPU leases")
        if any(len(lease.remaining_gpu_ids) < route_count - 1 for lease in leases):
            raise RuntimeError("attention lifecycle did not preserve three allocatable peer GPUs per lease")

    return AttentionLifecycleReport(
        inventory=_inventory_evidence(inventory),
        resource_contract_gpu_count=1,
        leases=leases,
        route_ids=tuple(route.route_id for route in ATTENTION_ROUTE_MATRIX),
        namespaces=tuple(route.namespace for route in ATTENTION_ROUTE_MATRIX),
        key_digests=tuple(request.key.digest for request in concrete_requests),
        pure=pure,
        cold=cold,
        warm=warm,
        reopened=reopened,
        cold_unique_misses=cold_unique_misses,
        cold_accepted_records=cold_accepted_records,
        reopened_unique_misses=reopened_unique_misses,
        records=record_evidence,
    )


class _CountingChannel:
    def __init__(self, inner: WorkerChannel, on_send: Callable[[], None]) -> None:
        self.inner = inner
        self._on_send = on_send
        self._command_count = 0

    @property
    def command_count(self) -> int:
        return self._command_count

    def send(self, command) -> None:
        self._command_count += 1
        self._on_send()
        self.inner.send(command)

    def recv(self):
        return self.inner.recv()

    def is_alive(self) -> bool:
        return self.inner.is_alive()

    def close(self) -> None:
        self.inner.close()

    def join(self) -> None:
        self.inner.join()


class CountingProcessWorkerRuntime:
    """Count exact worker commands while delegating to spawn workers."""

    def __init__(self) -> None:
        self._factory = ProcessWorkerFactory()
        self._command_count = 0
        self._wrapper_by_inner_id: dict[int, _CountingChannel] = {}
        self._bootstraps: list[WorkerBootstrap] = []
        self._wrappers: list[_CountingChannel] = []

    @property
    def command_count(self) -> int:
        return self._command_count

    @property
    def bootstraps(self) -> tuple[WorkerBootstrap, ...]:
        return tuple(self._bootstraps)

    @property
    def command_counts(self) -> tuple[int, ...]:
        return tuple(wrapper.command_count for wrapper in self._wrappers)

    def _record_send(self) -> None:
        self._command_count += 1

    def __call__(self, bootstrap: WorkerBootstrap) -> _CountingChannel:
        self._bootstraps.append(bootstrap)
        inner = self._factory(bootstrap)
        wrapper = _CountingChannel(inner, self._record_send)
        self._wrapper_by_inner_id[id(inner)] = wrapper
        self._wrappers.append(wrapper)
        return wrapper

    def wait_ready(
        self,
        channels: Sequence[WorkerChannel],
        timeout_seconds: float,
    ) -> tuple[_CountingChannel, ...]:
        wrappers = tuple(channels)
        if any(not isinstance(channel, _CountingChannel) for channel in wrappers):
            raise TypeError("counting runtime received a foreign worker channel")
        ready = self._factory.wait_ready(
            tuple(channel.inner for channel in wrappers if isinstance(channel, _CountingChannel)),
            timeout_seconds,
        )
        return tuple(self._wrapper_by_inner_id[id(channel)] for channel in ready)


def _worker_execution_evidence(worker_runtime: Any) -> WorkerExecutionEvidence:
    bootstraps = tuple(worker_runtime.bootstraps)
    command_counts = tuple(int(count) for count in worker_runtime.command_counts)
    command_count = int(worker_runtime.command_count)
    if len(command_counts) != len(bootstraps):
        raise RuntimeError("worker command evidence does not cover every bootstrap")
    if sum(command_counts) != command_count:
        raise RuntimeError("per-worker command evidence differs from the global command count")
    return WorkerExecutionEvidence(
        bootstrap_count=len(bootstraps),
        command_count=command_count,
        commands_per_worker=command_counts,
        bootstrap_device_uuids=tuple(tuple(bootstrap.device_uuids) for bootstrap in bootstraps),
        reused_worker_count=sum(count > 1 for count in command_counts),
    )


def _single_worker_reuse_evidence(
    worker_runtime: Any,
    *,
    expected_commands: int,
) -> WorkerExecutionEvidence:
    evidence = _worker_execution_evidence(worker_runtime)
    if evidence.bootstrap_count != 1:
        raise RuntimeError(
            f"single-GPU reuse gate expected one persistent worker, got {evidence.bootstrap_count}"
        )
    if evidence.command_count != expected_commands:
        raise RuntimeError(
            f"single-GPU reuse gate expected {expected_commands} commands, got {evidence.command_count}"
        )
    if evidence.commands_per_worker != (expected_commands,):
        raise RuntimeError("single-GPU reuse gate did not route every case through one worker")
    if evidence.reused_worker_count != 1:
        raise RuntimeError("single-GPU reuse gate did not reuse its persistent worker")
    return evidence


def validate_local_gpu_runtime() -> dict[str, object]:
    """Fail closed unless the local process is the frozen one-GB200 runtime."""

    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the offline attention lifecycle gate")
    sglang_version = importlib.metadata.version("sglang")
    if sglang_version not in _SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            f"attention lifecycle requires one of {tuple(sorted(_SUPPORTED_SGLANG_VERSIONS))!r}, got {sglang_version}"
        )
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("attention lifecycle requires exactly one visible CUDA GPU")
    device_name = str(torch.cuda.get_device_name(0))
    if " ".join(device_name.split()).casefold() != "nvidia gb200":
        raise RuntimeError(f"attention lifecycle requires NVIDIA GB200, got {device_name!r}")
    cuda_version = str(torch.version.cuda or "")
    if not cuda_version.startswith("13.0"):
        raise RuntimeError(f"attention lifecycle requires CUDA 13.0, got {cuda_version!r}")
    for module_name in ("flash_mla", "deep_gemm", "sgl_kernel"):
        importlib.import_module(module_name)
    return {
        "machine": platform.machine(),
        "sglang_version": sglang_version,
        "cuda_version": cuda_version,
        "gpu_count": int(torch.cuda.device_count()),
        "gpu_name": device_name,
        "model_artifact": _MODEL_ARTIFACT,
        "architecture": _ARCHITECTURE,
        "offline": True,
    }


def validate_scheduler_gpu_runtime(*, required_gpu_count: int = 4) -> dict[str, object]:
    """Fail closed unless the full visible inventory can host one-GPU leases."""

    if isinstance(required_gpu_count, bool) or not isinstance(required_gpu_count, int):
        raise TypeError("required_gpu_count must be an integer")
    if required_gpu_count < 1:
        raise ValueError("required_gpu_count must be positive")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"{name}=1 is required for the offline attention lifecycle gate")
    sglang_version = importlib.metadata.version("sglang")
    if sglang_version not in _SUPPORTED_SGLANG_VERSIONS:
        raise RuntimeError(
            f"attention lifecycle requires one of {tuple(sorted(_SUPPORTED_SGLANG_VERSIONS))!r}, got {sglang_version}"
        )
    torch = importlib.import_module("torch")
    gpu_count = int(torch.cuda.device_count())
    if not torch.cuda.is_available() or gpu_count < required_gpu_count:
        raise RuntimeError(
            f"scheduler lifecycle requires at least {required_gpu_count} visible CUDA GPUs, got {gpu_count}"
        )
    gpu_names = tuple(str(torch.cuda.get_device_name(index)) for index in range(gpu_count))
    if any(" ".join(name.split()).casefold() != "nvidia gb200" for name in gpu_names):
        raise RuntimeError(f"scheduler lifecycle requires all visible GPUs to be NVIDIA GB200, got {gpu_names!r}")
    cuda_version = str(torch.version.cuda or "")
    if not cuda_version.startswith("13.0"):
        raise RuntimeError(f"attention lifecycle requires CUDA 13.0, got {cuda_version!r}")
    for module_name in ("flash_mla", "deep_gemm", "sgl_kernel"):
        importlib.import_module(module_name)
    return {
        "machine": platform.machine(),
        "sglang_version": sglang_version,
        "cuda_version": cuda_version,
        "gpu_count": gpu_count,
        "required_gpu_count": required_gpu_count,
        "gpu_names": gpu_names,
        "model_artifact": _MODEL_ARTIFACT,
        "architecture": _ARCHITECTURE,
        "offline": True,
        "gate": "full-inventory-one-gpu-leases",
    }


def run_local_gpu_gate(*, overlay_path: Path, max_wall_seconds: float = 1800.0) -> LocalGpuGateReport:
    runtime_evidence = validate_local_gpu_runtime()
    inventory = discover_hardware()
    worker_runtime = CountingProcessWorkerRuntime()
    lifecycle = run_attention_lifecycle(
        overlay_path=overlay_path,
        inventory=inventory,
        worker_factory=worker_runtime,
        wait_ready=worker_runtime.wait_ready,
        command_count=lambda: worker_runtime.command_count,
        max_wall_seconds=max_wall_seconds,
        backend_version=str(runtime_evidence["sglang_version"]),
    )
    worker_execution = _single_worker_reuse_evidence(
        worker_runtime,
        expected_commands=len(ATTENTION_ROUTE_MATRIX),
    )
    return LocalGpuGateReport(
        runtime=runtime_evidence,
        lifecycle=lifecycle,
        worker_execution=worker_execution,
    )


def run_scheduler_gpu_gate(
    *,
    overlay_path: Path,
    max_wall_seconds: float = 1800.0,
    required_gpu_count: int = 4,
) -> LocalGpuGateReport:
    runtime_evidence = validate_scheduler_gpu_runtime(required_gpu_count=required_gpu_count)
    inventory = discover_hardware()
    if len(inventory.devices) != runtime_evidence["gpu_count"]:
        raise RuntimeError("hardware inventory does not cover every CUDA-visible GPU")
    if tuple(device.name for device in inventory.devices) != runtime_evidence["gpu_names"]:
        raise RuntimeError("hardware inventory GPU names differ from the CUDA runtime")
    worker_runtime = CountingProcessWorkerRuntime()
    lifecycle = run_attention_lifecycle(
        overlay_path=overlay_path,
        inventory=inventory,
        worker_factory=worker_runtime,
        wait_ready=worker_runtime.wait_ready,
        command_count=lambda: worker_runtime.command_count,
        max_wall_seconds=max_wall_seconds,
        backend_version=str(runtime_evidence["sglang_version"]),
    )
    expected_bootstraps = {lease.assigned_device_uuids for lease in lifecycle.leases}
    observed_bootstraps = {bootstrap.device_uuids for bootstrap in worker_runtime.bootstraps}
    if observed_bootstraps != expected_bootstraps:
        raise RuntimeError("spawned worker bootstraps differ from the persisted lease evidence")
    worker_execution = _worker_execution_evidence(worker_runtime)
    route_count = len(ATTENTION_ROUTE_MATRIX)
    if worker_execution.bootstrap_count != route_count:
        raise RuntimeError("scheduler gate did not spawn one worker per disjoint attention lease")
    if worker_execution.commands_per_worker != (1,) * route_count:
        raise RuntimeError("scheduler gate did not execute exactly one case per disjoint worker")
    return LocalGpuGateReport(
        runtime=runtime_evidence,
        lifecycle=lifecycle,
        worker_execution=worker_execution,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", type=Path, required=True, help="fresh SQLite overlay path (must not exist)")
    parser.add_argument("--output", type=Path, help="optional JSON evidence path")
    parser.add_argument("--max-wall-seconds", type=float, default=1800.0)
    parser.add_argument("--gate", choices=("local", "scheduler"), default="local")
    parser.add_argument("--required-gpu-count", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.gate == "local":
        report = run_local_gpu_gate(
            overlay_path=args.overlay,
            max_wall_seconds=args.max_wall_seconds,
        )
    else:
        report = run_scheduler_gpu_gate(
            overlay_path=args.overlay,
            max_wall_seconds=args.max_wall_seconds,
            required_gpu_count=args.required_gpu_count,
        )
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
