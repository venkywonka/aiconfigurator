# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for route-owned protocol identity in one resolution session."""

from __future__ import annotations

from dataclasses import replace
from types import ModuleType

import pytest

from aiconfigurator.collector import adapters as adapter_api
from aiconfigurator.collector.adapters import ResolvedLazyAdapter
from aiconfigurator.collector.executor import PersistentMeasurementExecutor
from aiconfigurator.collector.scheduler import HardwareAwareScheduler
from aiconfigurator.collector.sglang.registry import (
    DSV4_CSA_CONTEXT_LAZY_SPEC,
    MHC_LAZY_SPEC,
    SGLANG_LAZY_REGISTRY,
)
from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC
from aiconfigurator.collector.types import (
    FabricRequirement,
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    LazyOpEntry,
    ResourceContract,
    canonical_topology_fingerprint,
)
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionSession
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
    UnresolvedCode,
)

pytestmark = pytest.mark.unit

_ROUTES = (GEMM_LAZY_SPEC, MHC_LAZY_SPEC, DSV4_CSA_CONTEXT_LAZY_SPEC)


def _template() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="session-template-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="session-template",
        tuning_revision="session-template",
    )


def _route_protocol(template: MeasurementProtocol, lazy: LazyOpEntry) -> MeasurementProtocol:
    """Test oracle: route identity plus session-owned sampling policy."""
    return replace(
        template,
        revision=lazy.protocol_revision,
        timer=lazy.timer,
        tuning_revision=lazy.tuning_revision,
    )


def _environment(*, topology_fingerprint: str = "gb200-nvlink4") -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions={"cuda": "13.0", "sglang": "0.5.10"},
        topology_schema="nvidia-smi-v1",
        topology_fingerprint=topology_fingerprint,
    )


def _request(
    lazy: LazyOpEntry,
    index: int,
    protocol: MeasurementProtocol,
    *,
    environment: MeasurementEnvironment | None = None,
) -> MeasurementRequest:
    selected_environment = environment or _environment()
    query = {"case": index}
    return MeasurementRequest(
        op_id=f"mixed-route-{index}",
        key=PerfKey.build(lazy.namespace, query, selected_environment),
        query=query,
        environment=selected_environment,
        semantic_descriptor={"test_case": index},
        protocol=protocol,
    )


def _record(request: MeasurementRequest, latency_ms: float) -> MeasurementRecord:
    return MeasurementRecord.valid(
        key=request.key,
        latency_ms=latency_ms,
        energy_wms=0.0,
        samples_ms=(latency_ms,) * request.protocol.samples,
        protocol=request.protocol,
        perf_row={"latency": latency_ms},
        provenance={"test": "mixed-route-protocol-binding"},
    )


class _Overlay:
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], MeasurementRecord] = {}
        self.lookup_protocols: list[MeasurementProtocol] = []

    def append(self, record: MeasurementRecord) -> int:
        self.records[(record.key.digest, record.protocol.digest)] = record
        return len(self.records)

    def lookup(self, key: PerfKey, protocol: MeasurementProtocol) -> MeasurementRecord | None:
        self.lookup_protocols.append(protocol)
        return self.records.get((key.digest, protocol.digest))


class _Executor:
    def __init__(self) -> None:
        self.batches: list[tuple[MeasurementRequest, ...]] = []

    def execute(self, requests, *, deadline_monotonic, cancellation):
        del deadline_monotonic, cancellation
        batch = tuple(requests)
        self.batches.append(batch)
        return tuple(_record(request, float(index + 1)) for index, request in enumerate(batch))


class _BindingExecutor(_Executor):
    def __init__(self, lazy: LazyOpEntry) -> None:
        super().__init__()
        self.lazy = lazy
        self.binding_inputs: list[MeasurementRequest] = []

    def bind_request(self, request: MeasurementRequest) -> MeasurementRequest:
        self.binding_inputs.append(request)
        return replace(request, protocol=_route_protocol(request.protocol, self.lazy))


def test_central_binding_combines_sampling_template_with_route_identity() -> None:
    template = _template()
    binder = getattr(adapter_api, "bind_request_protocol", None)

    assert callable(binder), "collector.adapters must expose central bind_request_protocol(request, lazy)"
    for index, lazy in enumerate(_ROUTES):
        request = _request(lazy, index, template)
        bound = binder(request, lazy)

        assert bound.key == request.key
        assert bound.protocol == _route_protocol(template, lazy)
        assert bound.protocol.warmups == template.warmups
        assert bound.protocol.samples == template.samples
        assert bound.protocol.statistic == template.statistic


def test_central_binding_rejects_unsupported_statistic_before_overlay_reuse() -> None:
    template = replace(_template(), statistic="mean")
    request = _request(MHC_LAZY_SPEC, 0, template)

    with pytest.raises(ValueError, match="statistic"):
        adapter_api.bind_request_protocol(request, MHC_LAZY_SPEC)


def test_session_delegates_request_binding_to_executor_before_lookup_or_miss() -> None:
    template = _template()
    request = _request(MHC_LAZY_SPEC, 0, template)
    overlay = _Overlay()
    executor = _BindingExecutor(MHC_LAZY_SPEC)
    session = ResolutionSession(overlay, executor, ResolutionBudget(1, 30.0), template)

    bound = session.bind_request(request)

    assert executor.binding_inputs == [request]
    assert bound.protocol == _route_protocol(template, MHC_LAZY_SPEC)
    assert overlay.lookup_protocols == []


def test_one_callback_batches_mixed_route_protocols_and_preserves_record_identity() -> None:
    template = _template()
    requests = tuple(_request(lazy, index, _route_protocol(template, lazy)) for index, lazy in enumerate(_ROUTES))
    overlay = _Overlay()
    executor = _Executor()
    session = ResolutionSession(overlay, executor, ResolutionBudget(3, 30.0), template)

    def mixed_walk() -> tuple[float | None, ...]:
        results: list[float | None] = []
        for request in requests:
            record = overlay.lookup(request.key, request.protocol)
            if record is None:
                session.record_miss(request, request.op_id)
                results.append(None)
            else:
                results.append(record.latency_ms)
        return tuple(results)

    assert session.execute_callback(mixed_walk) == (1.0, 2.0, 3.0)
    assert executor.batches == [requests]
    assert {record.protocol for record in overlay.records.values()} == {request.protocol for request in requests}


def test_warm_lookup_uses_each_request_protocol_not_the_session_template() -> None:
    template = _template()
    requests = tuple(_request(lazy, index, _route_protocol(template, lazy)) for index, lazy in enumerate(_ROUTES))
    overlay = _Overlay()
    for index, request in enumerate(requests):
        overlay.append(_record(request, float(index + 1)))
    session = ResolutionSession(overlay, _Executor(), ResolutionBudget(3, 30.0), template)

    records = tuple(session.lookup(request.key, request.protocol) for request in requests)

    assert tuple(record.latency_ms for record in records if record is not None) == (1.0, 2.0, 3.0)
    assert overlay.lookup_protocols == [request.protocol for request in requests]


def test_pending_filter_uses_each_bound_protocol_for_existing_overlay_records() -> None:
    template = _template()
    requests = tuple(_request(lazy, index, _route_protocol(template, lazy)) for index, lazy in enumerate(_ROUTES))
    overlay = _Overlay()
    for request in requests:
        overlay.append(_record(request, 1.0))
    executor = _Executor()
    session = ResolutionSession(overlay, executor, ResolutionBudget(3, 30.0), template)

    for request in requests:
        session.record_miss(request, request.op_id)
    session.resolve_pending()

    assert executor.batches == []
    assert overlay.lookup_protocols[-3:] == [request.protocol for request in requests]


def _inventory() -> HardwareInventory:
    devices = (GpuDevice(index=0, uuid="GPU-0", name="NVIDIA GB200", pci_bus_id="00000000:00:00.0"),)
    evidence = HardwareDiscoveryEvidence(
        raw_gpu_query="synthetic query",
        raw_topology="synthetic topology",
        raw_p2p_read="synthetic reads",
        raw_p2p_write="synthetic writes",
    )
    fingerprint = canonical_topology_fingerprint("mixed-protocol-test-v1", devices, {}, {}, {})
    return HardwareInventory(
        schema_revision="mixed-protocol-test-v1",
        devices=devices,
        links={},
        p2p_read={},
        p2p_write={},
        fabric_domains={},
        topology_fingerprint=fingerprint,
        evidence=evidence,
    )


class _Cancellation:
    def cancelled(self) -> bool:
        return False


def test_executor_classifies_protocol_mismatch_as_identity_not_unsupported_shape() -> None:
    inventory = _inventory()
    environment = _environment(topology_fingerprint=inventory.topology_fingerprint)
    request = _request(MHC_LAZY_SPEC, 0, _template(), environment=environment)
    entry = next(entry for entry in SGLANG_LAZY_REGISTRY if entry.lazy is MHC_LAZY_SPEC)
    adapter = ResolvedLazyAdapter(
        entry=entry,
        lazy=MHC_LAZY_SPEC,
        collector_module=MHC_LAZY_SPEC.run_module,
        backend="sglang",
        backend_version="0.5.10",
        adapter_module=ModuleType("aiconfigurator.collector.testing.mixed_protocol_adapter"),
        _case_func=lambda unused: {},
        _resource_func=lambda unused_request, unused_case: ResourceContract(
            gpu_count=1,
            fabric=FabricRequirement.NONE,
        ),
        _result_func=lambda unused_request, unused_case, unused_result: _record(request, 1.0),
    )

    def fail_if_worker_starts(unused_bootstrap):
        raise AssertionError("protocol mismatch must fail before worker acquisition")

    executor = PersistentMeasurementExecutor(
        inventory=inventory,
        scheduler=HardwareAwareScheduler(inventory),
        resolve_adapter=lambda unused_request: adapter,
        worker_factory=fail_if_worker_starts,
        wait_ready=lambda channels, unused_deadline: channels,
        clock=lambda: 0.0,
    )

    bound = executor.bind_request(request)
    assert bound.protocol == _route_protocol(request.protocol, MHC_LAZY_SPEC)
    assert request.protocol == _template()

    records = executor.execute((request,), deadline_monotonic=10.0, cancellation=_Cancellation())

    assert records[0].failure_code is UnresolvedCode.IDENTITY_MISMATCH
    assert "request protocol" in (records[0].failure_reason or "")
