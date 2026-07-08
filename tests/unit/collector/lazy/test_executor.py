# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the persistent hardware-aware measurement executor."""

from __future__ import annotations

import importlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.collector.scheduler import HardwareAwareScheduler
from aiconfigurator.collector.types import (
    FabricRequirement,
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    LazyOpEntry,
    ResourceContract,
    canonical_topology_fingerprint,
)
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionFailed, ResolutionSession
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    RecordStatus,
    UnresolvedCode,
)

pytestmark = pytest.mark.unit


def _apis():
    executor = importlib.import_module("aiconfigurator.collector.executor")
    adapters = importlib.import_module("aiconfigurator.collector.adapters")
    return adapters, executor


def _protocol(**overrides: Any) -> MeasurementProtocol:
    values = {
        "revision": "cuda-event-v1",
        "warmups": 2,
        "samples": 1,
        "timer": "cuda_event",
        "tuning_revision": "fake-v1",
    }
    values.update(overrides)
    return MeasurementProtocol(**values)


def _environment(inventory_count: int = 1) -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="gb200_nvlink4",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions={"cuda": "13.0"},
        topology_schema="nvidia-smi-v1",
        topology_fingerprint=_inventory(inventory_count).topology_fingerprint,
    )


def _request(
    m: int,
    *,
    op_id: str | None = None,
    protocol: MeasurementProtocol | None = None,
    inventory_count: int = 1,
) -> MeasurementRequest:
    environment = _environment(inventory_count)
    query = {"m": m, "n": 256}
    return MeasurementRequest(
        op_id=op_id or f"gemm-{m}",
        key=PerfKey.build("gemm_perf.txt/v1", query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"consumer": op_id or f"gemm-{m}"},
        protocol=protocol or _protocol(),
    )


_EVIDENCE = HardwareDiscoveryEvidence(
    raw_gpu_query="synthetic query",
    raw_topology="synthetic topology",
    raw_p2p_read="synthetic reads",
    raw_p2p_write="synthetic writes",
)


def _inventory(count: int) -> HardwareInventory:
    devices = tuple(
        GpuDevice(
            index=index,
            uuid=f"GPU-{index}",
            name="NVIDIA GB200",
            pci_bus_id=f"00000000:{index:02X}:00.0",
        )
        for index in range(count)
    )
    pairs = {(left, right): "SYS" for left in range(count) for right in range(count) if left != right}
    capability = dict.fromkeys(pairs, False)
    fingerprint = canonical_topology_fingerprint(
        "executor-test-v1",
        devices,
        pairs,
        capability,
        capability,
    )
    return HardwareInventory(
        schema_revision="executor-test-v1",
        devices=devices,
        links=pairs,
        p2p_read=capability,
        p2p_write=capability,
        fabric_domains={},
        topology_fingerprint=fingerprint,
        evidence=_EVIDENCE,
    )


class _Cancellation:
    def __init__(self) -> None:
        self.value = False

    def cancelled(self) -> bool:
        return self.value


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class _Adapter:
    def __init__(
        self,
        prepared_type: type,
        events: list[str],
        *,
        gpu_count: int = 1,
        adapter_module: str = "aiconfigurator.collector.testing.fake_adapter",
        prepare_error: Exception | None = None,
    ) -> None:
        self.prepared_type = prepared_type
        self.events = events
        self.contract = ResourceContract(gpu_count=gpu_count, fabric=FabricRequirement.NONE)
        self.prepare_error = prepare_error
        self.lazy = LazyOpEntry(
            namespace="gemm_perf.txt/v1",
            run_module="aiconfigurator.collector.testing.fake_runner",
            run_func="run_case",
            adapter_module=adapter_module,
            case_func="request_to_case",
            result_func="result_to_record",
            resource_func="resource_for_request",
            protocol_revision="cuda-event-v1",
            timer="cuda_event",
            tuning_revision="fake-v1",
        )

    def prepare(self, request: MeasurementRequest):
        self.events.append(f"prepare:{request.op_id}")
        if self.prepare_error is not None:
            raise self.prepare_error
        return self.prepared_type(
            request=request,
            case={"m": request.query["m"], "n": request.query["n"]},
            contract=self.contract,
        )

    def record(self, prepared: object, raw_result: Mapping[str, Any]) -> MeasurementRecord:
        request = prepared.request
        self.events.append(f"record:{request.op_id}")
        samples = tuple(raw_result["samples_ms"])
        return MeasurementRecord.valid(
            key=request.key,
            latency_ms=raw_result["latency_ms"],
            energy_wms=0.0,
            samples_ms=samples,
            protocol=request.protocol,
            perf_row=dict(prepared.case),
            provenance={"runner": self.lazy.run_module},
        )


@dataclass
class _Channel:
    bootstrap: object
    events: list[str]
    reply_builder: Callable[[object], object]
    send_error: BaseException | None = None
    pending: list[object] | None = None
    commands: list[object] | None = None
    alive: bool = True
    close_calls: int = 0
    join_calls: int = 0

    def __post_init__(self) -> None:
        self.pending = []
        self.commands = []

    @property
    def label(self) -> str:
        return ",".join(self.bootstrap.device_uuids)

    def send(self, command: object) -> None:
        self.events.append(f"send:{command.request_digest}:{self.label}")
        assert self.commands is not None
        self.commands.append(command)
        if self.send_error is not None:
            error, self.send_error = self.send_error, None
            raise error
        reply = self.reply_builder(command)
        assert self.pending is not None
        if isinstance(reply, list):
            self.pending.extend(reply)
        else:
            self.pending.append(reply)

    def recv(self) -> object:
        self.events.append(f"recv:{self.label}")
        assert self.pending is not None
        reply = self.pending.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def is_alive(self) -> bool:
        return self.alive

    def close(self) -> None:
        self.close_calls += 1
        self.events.append(f"close:{self.label}")

    def join(self) -> None:
        self.join_calls += 1
        self.events.append(f"join:{self.label}")


class _Factory:
    def __init__(
        self,
        executor_api: object,
        events: list[str],
        builders: Sequence[Callable[[object], object]] = (),
        *,
        modes: Sequence[str] = (),
    ) -> None:
        self.api = executor_api
        self.events = events
        self.builders = list(builders)
        self.modes = list(modes)
        self.channels: list[_Channel] = []

    def _valid(self, command: object) -> object:
        return self.api.WorkerReply(
            invocation_id=command.invocation_id,
            request_digest=command.request_digest,
            raw_result={"latency_ms": 1.0, "samples_ms": [1.0]},
        )

    def __call__(self, bootstrap: object) -> _Channel:
        index = len(self.channels)
        self.events.append(f"start:{','.join(bootstrap.device_uuids)}")
        builder = self.builders[index] if index < len(self.builders) else self._valid
        mode = self.modes[index] if index < len(self.modes) else "valid"
        channel = _Channel(bootstrap, self.events, builder)
        if mode == "send":
            channel.send_error = BrokenPipeError("send failed")
        elif mode == "death":
            channel.alive = False
            channel.reply_builder = lambda command: []
        self.channels.append(channel)
        return channel


class _WaitReady:
    def __init__(self, events: list[str], *, reverse: bool = False) -> None:
        self.events = events
        self.reverse = reverse

    def __call__(self, channels: Sequence[_Channel], timeout_seconds: float) -> tuple[_Channel, ...]:
        self.events.append(f"wait:{timeout_seconds:.3f}")
        ready = [channel for channel in channels if channel.pending or not channel.is_alive()]
        if self.reverse:
            ready.reverse()
        return tuple(ready[:1])


def _executor(
    count: int,
    events: list[str],
    *,
    resolver: Callable[[MeasurementRequest], _Adapter],
    factory: _Factory,
    waiter: Callable[[Sequence[_Channel], float], Sequence[_Channel]] | None = None,
    clock: _Clock | None = None,
):
    _, api = _apis()
    return api.PersistentMeasurementExecutor(
        inventory=_inventory(count),
        scheduler=HardwareAwareScheduler(_inventory(count)),
        resolve_adapter=resolver,
        worker_factory=factory,
        wait_ready=waiter or _WaitReady(events),
        clock=clock or _Clock(),
    )


def _assert_failure(record: MeasurementRecord, code: UnresolvedCode) -> None:
    assert record.status is RecordStatus.FAILED
    assert record.failure_code is code


def test_wave_submits_every_job_before_multiplexed_receive_and_returns_request_order() -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    factory = _Factory(api, events)
    executor = _executor(
        2, events, resolver=lambda request: adapter, factory=factory, waiter=_WaitReady(events, reverse=True)
    )
    requests = tuple(_request(m, inventory_count=2) for m in (3, 1, 2))

    records = tuple(executor.execute(requests, deadline_monotonic=20.0, cancellation=_Cancellation()))

    assert [record.key for record in records] == [request.key for request in requests]
    prepare_positions = [index for index, event in enumerate(events) if event.startswith("prepare:")]
    start_positions = [index for index, event in enumerate(events) if event.startswith("start:")]
    send_positions = [index for index, event in enumerate(events) if event.startswith("send:")]
    recv_positions = [index for index, event in enumerate(events) if event.startswith("recv:")]
    assert max(prepare_positions) < min(start_positions)
    assert max(send_positions[:2]) < min(recv_positions[:2])
    assert max(recv_positions[:2]) < send_positions[2]
    assert events[recv_positions[0]] == "recv:GPU-1"
    commands = [command for channel in factory.channels for command in channel.commands or ()]
    assert len({command.invocation_id for command in commands}) == len(commands)
    assert {tuple(sorted(json.loads(command.payload).items())) for command in commands} == {
        (("m", 1), ("n", 256)),
        (("m", 2), ("n", 256)),
        (("m", 3), ("n", 256)),
    }


def test_compatible_leases_reuse_but_protocol_adapter_and_uuid_tuples_isolate() -> None:
    adapters, api = _apis()
    events: list[str] = []
    primary = _Adapter(adapters.PreparedMeasurement, events)
    alternate = _Adapter(
        adapters.PreparedMeasurement,
        events,
        adapter_module="aiconfigurator.collector.testing.other_adapter",
    )
    pair = _Adapter(adapters.PreparedMeasurement, events, gpu_count=2)
    factory = _Factory(api, events)

    def resolve(request: MeasurementRequest) -> _Adapter:
        return {"alternate": alternate, "pair": pair}.get(request.op_id, primary)

    executor = _executor(2, events, resolver=resolve, factory=factory)
    cancellation = _Cancellation()
    executor.execute((_request(1, inventory_count=2),), deadline_monotonic=20.0, cancellation=cancellation)
    executor.execute((_request(2, inventory_count=2),), deadline_monotonic=20.0, cancellation=cancellation)
    executor.execute(
        (_request(3, protocol=_protocol(warmups=7), inventory_count=2),),
        deadline_monotonic=20.0,
        cancellation=cancellation,
    )
    executor.execute(
        (_request(4, op_id="alternate", inventory_count=2),),
        deadline_monotonic=20.0,
        cancellation=cancellation,
    )
    executor.execute(
        (_request(5, op_id="pair", inventory_count=2),),
        deadline_monotonic=20.0,
        cancellation=cancellation,
    )

    assert len(factory.channels) == 4
    assert factory.channels[0].bootstrap.protocol_digest == _protocol().digest
    assert factory.channels[0].bootstrap.device_uuids == ("GPU-0",)
    assert factory.channels[-1].bootstrap.device_uuids == ("GPU-0", "GPU-1")
    assert factory.channels[-1].bootstrap.local_ordinals == (0, 1)
    assert factory.channels[-1].bootstrap.topology_fingerprint == _inventory(2).topology_fingerprint


def test_distinct_lazy_namespaces_use_distinct_leases_and_bootstrap_identities() -> None:
    adapters, api = _apis()
    events: list[str] = []
    primary = _Adapter(adapters.PreparedMeasurement, events)
    alternate = _Adapter(adapters.PreparedMeasurement, events)
    alternate.lazy = replace(alternate.lazy, namespace="other_perf.txt/v1")
    primary_request = _request(1)
    alternate_request = _request(2, op_id="alternate")
    alternate_request = replace(
        alternate_request,
        key=PerfKey.build(alternate.lazy.namespace, alternate_request.query, alternate_request.environment),
    )
    factory = _Factory(api, events)
    executor = _executor(
        1,
        events,
        resolver=lambda request: alternate if request.op_id == "alternate" else primary,
        factory=factory,
    )

    first = tuple(executor.execute((primary_request,), deadline_monotonic=20.0, cancellation=_Cancellation()))
    second = tuple(executor.execute((alternate_request,), deadline_monotonic=20.0, cancellation=_Cancellation()))

    assert first[0].status is second[0].status is RecordStatus.VALID
    assert len(factory.channels) == 2
    assert [channel.bootstrap.adapter_namespace for channel in factory.channels] == [
        "gemm_perf.txt/v1",
        "other_perf.txt/v1",
    ]


def test_duplicate_reply_is_rejected_as_stale_before_the_lease_can_be_reused_again() -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)

    def duplicate(command: object) -> list[object]:
        reply = api.WorkerReply(
            command.invocation_id,
            command.request_digest,
            {"latency_ms": 1.0, "samples_ms": [1.0]},
        )
        return [reply, reply]

    factory = _Factory(api, events, (duplicate,))
    executor = _executor(1, events, resolver=lambda request: adapter, factory=factory)
    first = tuple(executor.execute((_request(1),), deadline_monotonic=20.0, cancellation=_Cancellation()))
    second = tuple(executor.execute((_request(2),), deadline_monotonic=20.0, cancellation=_Cancellation()))
    third = tuple(executor.execute((_request(3),), deadline_monotonic=20.0, cancellation=_Cancellation()))

    assert first[0].status is RecordStatus.VALID
    _assert_failure(second[0], UnresolvedCode.IDENTITY_MISMATCH)
    assert third[0].status is RecordStatus.VALID
    assert factory.channels[0].close_calls == 1
    assert len(factory.channels) == 2


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    [
        ("stale_id", UnresolvedCode.IDENTITY_MISMATCH),
        ("wrong_digest", UnresolvedCode.IDENTITY_MISMATCH),
        ("record_from_worker", UnresolvedCode.INVALID_MEASUREMENT),
        ("malformed", UnresolvedCode.INVALID_MEASUREMENT),
    ],
)
def test_corrupt_replies_fail_closed_evict_and_retry_on_a_fresh_worker(
    fault: str,
    expected_code: UnresolvedCode,
) -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    request = _request(1)

    def corrupt(command: object) -> object:
        if fault == "stale_id":
            return api.WorkerReply("stale", command.request_digest, {"latency_ms": 1.0, "samples_ms": [1.0]})
        if fault == "wrong_digest":
            return api.WorkerReply(command.invocation_id, "wrong", {"latency_ms": 1.0, "samples_ms": [1.0]})
        if fault == "record_from_worker":
            return api.WorkerReply(
                command.invocation_id,
                command.request_digest,
                adapter.record(adapter.prepare(request), {"latency_ms": 1.0, "samples_ms": [1.0]}),
            )
        return object()

    factory = _Factory(api, events, (corrupt,))
    executor = _executor(1, events, resolver=lambda candidate: adapter, factory=factory)
    first = tuple(executor.execute((request,), deadline_monotonic=20.0, cancellation=_Cancellation()))
    second = tuple(executor.execute((_request(2),), deadline_monotonic=20.0, cancellation=_Cancellation()))

    _assert_failure(first[0], expected_code)
    assert second[0].status is RecordStatus.VALID
    assert factory.channels[0].close_calls == 1
    assert len(factory.channels) == 2


@pytest.mark.parametrize("mode", ["send", "eof", "death", "worker_error"])
def test_transport_and_worker_failures_evict_the_lease(mode: str) -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)

    def faulty(command: object) -> object:
        if mode == "eof":
            return EOFError("worker closed")
        if mode == "worker_error":
            return api.WorkerReply(command.invocation_id, command.request_digest, error="kernel failed")
        return api.WorkerReply(command.invocation_id, command.request_digest, {"latency_ms": 1.0, "samples_ms": [1.0]})

    factory = _Factory(api, events, (faulty,), modes=(mode,))
    executor = _executor(1, events, resolver=lambda request: adapter, factory=factory)
    first = tuple(executor.execute((_request(1),), deadline_monotonic=20.0, cancellation=_Cancellation()))
    second = tuple(executor.execute((_request(2),), deadline_monotonic=20.0, cancellation=_Cancellation()))

    _assert_failure(first[0], UnresolvedCode.COLLECTOR_FAILED)
    assert second[0].status is RecordStatus.VALID
    assert factory.channels[0].close_calls == 1
    assert len(factory.channels) == 2


def test_raising_worker_health_probe_fails_typed_and_cleans_up() -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    base_factory = _Factory(api, events)

    def factory(bootstrap: object) -> _Channel:
        channel = base_factory(bootstrap)

        def raise_health_error() -> bool:
            raise RuntimeError("health exploded")

        channel.is_alive = raise_health_error
        return channel

    def ready_without_health_probe(
        channels: Sequence[_Channel],
        timeout_seconds: float,
    ) -> tuple[_Channel, ...]:
        del timeout_seconds
        return tuple(channels)

    executor = _executor(
        1,
        events,
        resolver=lambda request: adapter,
        factory=factory,
        waiter=ready_without_health_probe,
    )

    records = tuple(executor.execute((_request(1),), deadline_monotonic=20.0, cancellation=_Cancellation()))

    _assert_failure(records[0], UnresolvedCode.COLLECTOR_FAILED)
    assert base_factory.channels[0].close_calls == 1
    assert base_factory.channels[0].join_calls == 1


@pytest.mark.parametrize("failing_method", ["close", "join"])
def test_cleanup_failure_preserves_typed_record_and_attempts_close_and_join(failing_method: str) -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    base_factory = _Factory(api, events, builders=(lambda command: object(),))

    def factory(bootstrap: object) -> _Channel:
        channel = base_factory(bootstrap)
        original_cleanup = getattr(channel, failing_method)

        def raise_cleanup_error() -> None:
            original_cleanup()
            raise RuntimeError(f"{failing_method} exploded")

        setattr(channel, failing_method, raise_cleanup_error)
        return channel

    executor = _executor(1, events, resolver=lambda request: adapter, factory=factory)

    records = tuple(executor.execute((_request(1),), deadline_monotonic=20.0, cancellation=_Cancellation()))

    _assert_failure(records[0], UnresolvedCode.INVALID_MEASUREMENT)
    assert base_factory.channels[0].close_calls == 1
    assert base_factory.channels[0].join_calls == 1


def test_ready_sibling_survives_absolute_deadline_timeout() -> None:
    adapters, api = _apis()
    events: list[str] = []
    clock = _Clock()
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    factory = _Factory(api, events)
    calls = 0

    def wait_ready(channels: Sequence[_Channel], timeout_seconds: float) -> tuple[_Channel, ...]:
        nonlocal calls
        calls += 1
        events.append(f"wait:{timeout_seconds:.3f}")
        if calls == 1:
            return (channels[-1],)
        clock.value = 20.0
        return ()

    executor = _executor(2, events, resolver=lambda request: adapter, factory=factory, waiter=wait_ready, clock=clock)
    records = tuple(
        executor.execute(
            (_request(1, inventory_count=2), _request(2, inventory_count=2)),
            deadline_monotonic=15.0,
            cancellation=_Cancellation(),
        )
    )

    assert sorted((record.status for record in records), key=str) == [RecordStatus.FAILED, RecordStatus.VALID]
    failed = next(record for record in records if record.status is RecordStatus.FAILED)
    _assert_failure(failed, UnresolvedCode.TIMEOUT)
    assert sorted(channel.close_calls for channel in factory.channels) == [0, 1]


def test_mid_wave_cancellation_preserves_completed_sibling_and_launches_no_later_wave() -> None:
    adapters, api = _apis()
    events: list[str] = []
    cancellation = _Cancellation()
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    factory = _Factory(api, events)
    calls = 0

    def wait_ready(channels: Sequence[_Channel], timeout_seconds: float) -> tuple[_Channel, ...]:
        nonlocal calls
        del timeout_seconds
        calls += 1
        if calls == 1:
            return (channels[0],)
        cancellation.value = True
        return ()

    executor = _executor(2, events, resolver=lambda request: adapter, factory=factory, waiter=wait_ready)
    records = tuple(
        executor.execute(
            tuple(_request(m, inventory_count=2) for m in (1, 2, 3)),
            deadline_monotonic=20.0,
            cancellation=cancellation,
        )
    )

    assert sum(record.status is RecordStatus.VALID for record in records) == 1
    failures = [record for record in records if record.status is RecordStatus.FAILED]
    assert len(failures) == 2
    assert all(record.failure_code is UnresolvedCode.CANCELLED for record in failures)
    assert sum(event.startswith("send:") for event in events) == 2
    assert sum(channel.close_calls for channel in factory.channels) == 1


def test_unschedulable_and_duplicate_work_fail_without_blocking_or_acquiring() -> None:
    adapters, api = _apis()
    events: list[str] = []
    single = _Adapter(adapters.PreparedMeasurement, events)
    pair = _Adapter(adapters.PreparedMeasurement, events, gpu_count=2)
    factory = _Factory(api, events)
    executor = _executor(
        1, events, resolver=lambda request: pair if request.op_id == "pair" else single, factory=factory
    )

    schedulable = _request(1)
    unschedulable = _request(2, op_id="pair")
    records = tuple(
        executor.execute((unschedulable, schedulable), deadline_monotonic=20.0, cancellation=_Cancellation())
    )
    assert records[1].status is RecordStatus.VALID
    _assert_failure(records[0], UnresolvedCode.RESOURCE_UNAVAILABLE)

    duplicate = _request(3)
    before = len(factory.channels)
    duplicate_records = tuple(
        executor.execute((duplicate, duplicate), deadline_monotonic=20.0, cancellation=_Cancellation())
    )
    assert len(duplicate_records) == 2
    assert all(record.failure_code is UnresolvedCode.INVALID_MEASUREMENT for record in duplicate_records)
    assert len(factory.channels) == before


def test_request_topology_mismatch_fails_typed_before_worker_acquisition() -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    factory = _Factory(api, events)
    executor = _executor(1, events, resolver=lambda request: adapter, factory=factory)

    mismatched = _request(1, inventory_count=2)
    records = tuple(executor.execute((mismatched,), deadline_monotonic=20.0, cancellation=_Cancellation()))

    _assert_failure(records[0], UnresolvedCode.TOPOLOGY_MISMATCH)
    assert factory.channels == []


def test_adapter_prepare_failure_is_typed_and_does_not_block_a_valid_sibling() -> None:
    adapters, api = _apis()
    events: list[str] = []
    valid = _Adapter(adapters.PreparedMeasurement, events)
    invalid = _Adapter(
        adapters.PreparedMeasurement,
        events,
        prepare_error=ValueError("shape is outside the adapter envelope"),
    )
    factory = _Factory(api, events)
    executor = _executor(
        1,
        events,
        resolver=lambda request: invalid if request.op_id == "invalid" else valid,
        factory=factory,
    )

    records = tuple(
        executor.execute(
            (_request(1, op_id="invalid"), _request(2)),
            deadline_monotonic=20.0,
            cancellation=_Cancellation(),
        )
    )

    _assert_failure(records[0], UnresolvedCode.UNSUPPORTED_SHAPE)
    assert records[1].status is RecordStatus.VALID
    assert len(factory.channels) == 1


def test_non_json_adapter_case_is_typed_and_does_not_block_a_valid_sibling() -> None:
    adapters, api = _apis()
    events: list[str] = []
    valid = _Adapter(adapters.PreparedMeasurement, events)
    invalid = _Adapter(adapters.PreparedMeasurement, events)

    def prepare_invalid(request: MeasurementRequest):
        events.append(f"prepare:{request.op_id}")
        return adapters.PreparedMeasurement(
            request=request,
            case={"not_json": object()},
            contract=invalid.contract,
        )

    invalid.prepare = prepare_invalid
    factory = _Factory(api, events)
    executor = _executor(
        1,
        events,
        resolver=lambda request: invalid if request.op_id == "invalid-json" else valid,
        factory=factory,
    )

    records = tuple(
        executor.execute(
            (_request(1, op_id="invalid-json"), _request(2)),
            deadline_monotonic=20.0,
            cancellation=_Cancellation(),
        )
    )

    _assert_failure(records[0], UnresolvedCode.INVALID_MEASUREMENT)
    assert records[1].status is RecordStatus.VALID
    assert len(factory.channels) == 1


def test_expired_deadline_and_pre_cancelled_batch_launch_nothing() -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    factory = _Factory(api, events)
    executor = _executor(1, events, resolver=lambda request: adapter, factory=factory, clock=_Clock(10.0))

    expired = tuple(executor.execute((_request(1),), deadline_monotonic=10.0, cancellation=_Cancellation()))
    cancelled = _Cancellation()
    cancelled.value = True
    pre_cancelled = tuple(executor.execute((_request(2),), deadline_monotonic=20.0, cancellation=cancelled))

    _assert_failure(expired[0], UnresolvedCode.TIMEOUT)
    _assert_failure(pre_cancelled[0], UnresolvedCode.CANCELLED)
    assert factory.channels == []


def test_direct_execution_never_persists_but_session_keeps_valid_partial_result(tmp_path) -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)

    def reply(command: object) -> object:
        if command.request_digest == failed.key.digest:
            return api.WorkerReply(command.invocation_id, command.request_digest, error="injected failure")
        return api.WorkerReply(
            command.invocation_id,
            command.request_digest,
            {"latency_ms": 1.0, "samples_ms": [1.0]},
        )

    successful, failed = _request(1, inventory_count=2), _request(2, inventory_count=2)
    factory = _Factory(api, events, (reply, reply))
    executor = _executor(2, events, resolver=lambda request: adapter, factory=factory)
    overlay = OverlayStore(tmp_path / "overlay.sqlite")

    direct = tuple(executor.execute((successful,), deadline_monotonic=20.0, cancellation=_Cancellation()))
    assert direct[0].status is RecordStatus.VALID
    assert overlay.lookup(successful.key, successful.protocol) is None

    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=2, max_wall_seconds=10.0),
        successful.protocol,
        clock=_Clock(),
    )
    session.record_miss(successful, successful.op_id)
    session.record_miss(failed, failed.op_id)
    with pytest.raises(ResolutionFailed):
        session.resolve_pending()
    assert overlay.lookup(successful.key, successful.protocol) is not None
    overlay.close()


def test_uuid_binding_precedes_runner_import_and_exposes_only_local_ordinals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, api = _apis()
    events: list[str] = []
    sentinel = object()
    module = SimpleNamespace(run_case=sentinel)
    bootstrap = api.WorkerBootstrap(
        run_module="aiconfigurator.collector.testing.fake_runner",
        run_func="run_case",
        adapter_namespace="aiconfigurator.collector.testing.fake_adapter",
        protocol_digest=_protocol().digest,
        device_uuids=("GPU-A", "GPU-B"),
        topology_fingerprint="topology",
    )
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def importer(name: str) -> object:
        events.append(f"import:{name}:{os.environ['CUDA_VISIBLE_DEVICES']}")
        return module

    runner = api.bind_and_import_runner(bootstrap, import_module=importer)

    assert runner is sentinel
    assert events == ["import:aiconfigurator.collector.testing.fake_runner:GPU-A,GPU-B"]
    assert bootstrap.local_ordinals == (0, 1)


def test_concurrent_execute_calls_serialize_and_close_is_idempotent() -> None:
    adapters, api = _apis()
    events: list[str] = []
    adapter = _Adapter(adapters.PreparedMeasurement, events)
    factory = _Factory(api, events)
    release = threading.Event()
    waiting = threading.Event()
    second_attempted = threading.Event()

    def blocking_wait(channels: Sequence[_Channel], timeout_seconds: float) -> tuple[_Channel, ...]:
        del timeout_seconds
        waiting.set()
        assert release.wait(1.0)
        return (channels[0],)

    executor = _executor(1, events, resolver=lambda request: adapter, factory=factory, waiter=blocking_wait)

    def second_call() -> Sequence[MeasurementRecord]:
        second_attempted.set()
        return executor.execute((_request(2),), deadline_monotonic=20.0, cancellation=_Cancellation())

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.execute, (_request(1),), deadline_monotonic=20.0, cancellation=_Cancellation())
        assert waiting.wait(1.0)
        second = pool.submit(second_call)
        assert second_attempted.wait(1.0)
        time.sleep(0.05)
        assert "prepare:gemm-2" not in events
        release.set()
        assert tuple(first.result())[0].status is RecordStatus.VALID
        assert tuple(second.result())[0].status is RecordStatus.VALID

    executor.close()
    executor.close()
    assert len(factory.channels) == 1
    assert factory.channels[0].close_calls == 1
    assert factory.channels[0].join_calls == 1
