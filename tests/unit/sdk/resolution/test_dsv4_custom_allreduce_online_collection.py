# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 CustomAllReduce normalization and exact-evidence contracts."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from typing import ClassVar

import pytest

from aiconfigurator.collector import (
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    canonical_topology_fingerprint,
)
from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.executor import PersistentMeasurementExecutor, WorkerReply
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.scheduler import HardwareAwareScheduler
from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.communication import CustomAllReduce
from aiconfigurator.sdk.perf_database import LoadedOpData, PerfDataFilename
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionSession
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    PerfKey,
)

pytestmark = pytest.mark.unit

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_NAMESPACE = f"{PerfFile.CUSTOM_ALLREDUCE}/v1"
_EXPECTED_QUERY = {
    "dtype": "half",
    "operation": "all_reduce",
    "world_size": 4,
    "elements": 32768,
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
_SEMANTIC_DESCRIPTOR = {
    "operation": "all_reduce",
    "implementation": "sglang_custom_allreduce",
    "mode": "graph",
}


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="gb200",
        backend="sglang",
        backend_version="0.5.10",
        gpu_class="NVIDIA GB200",
        runtime_versions={
            "cuda": "13.0",
            "model_profile": "dsv4-v1.2",
            "sglang": "0.5.10",
        },
        topology_schema="nvidia-smi-v1",
        topology_fingerprint="gb200-nvlink4",
        profile_compatibility=_PROFILE_COMPATIBILITY,
    )


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-custom-allreduce-v1",
    )


def _four_gpu_nvlink_inventory() -> HardwareInventory:
    devices = tuple(
        GpuDevice(
            index=index,
            uuid=f"GPU-{index}",
            name="NVIDIA GB200",
            pci_bus_id=f"00000000:{index:02X}:00.0",
        )
        for index in range(4)
    )
    links = {(left, right): "NV18" for left in range(4) for right in range(4) if left != right}
    p2p = dict.fromkeys(links, True)
    schema_revision = "nvidia-smi-v1"
    return HardwareInventory(
        schema_revision=schema_revision,
        devices=devices,
        links=links,
        p2p_read=p2p,
        p2p_write=p2p,
        fabric_domains=dict.fromkeys(range(4), "nvlink:0"),
        topology_fingerprint=canonical_topology_fingerprint(
            schema_revision,
            devices,
            links,
            p2p,
            p2p,
        ),
        evidence=HardwareDiscoveryEvidence(
            raw_gpu_query="synthetic GB200 query",
            raw_topology="synthetic fully connected NVLink topology",
            raw_p2p_read="synthetic P2P read matrix",
            raw_p2p_write="synthetic P2P write matrix",
        ),
    )


class _ProfileDatabase:
    system = "gb200"
    backend = "sglang"
    version = "0.5.10"
    enable_shared_layer = True
    system_spec: ClassVar[dict[str, dict[str, int]]] = {
        "gpu": {"sm_version": 100},
        "node": {"num_gpus_per_node": 4},
    }

    def __init__(self) -> None:
        self.measurement_environment = _environment()
        self.queries: list[tuple[common.CommQuantMode, int, int]] = []

    def query_custom_allreduce(
        self,
        quant_mode: common.CommQuantMode,
        tp_size: int,
        size: int,
        database_mode: common.DatabaseMode | None = None,
    ) -> PerformanceResult:
        del database_mode
        self.queries.append((quant_mode, tp_size, size))
        return PerformanceResult(1.25, energy=2.5, source="silicon")


def _exact_data(elements: int = 32768) -> LoadedOpData:
    return LoadedOpData(
        {
            common.CommQuantMode.half: {
                4: {
                    "AUTO": {
                        elements: {
                            "latency": 1.25,
                            "power": 10.0,
                            "energy": 12.5,
                        }
                    }
                }
            }
        },
        PerfDataFilename.custom_allreduce,
        "/frozen/gb200/sglang/0.5.10/custom_allreduce_perf.parquet",
    )


def test_custom_allreduce_runtime_input_feeds_one_physical_query_and_request() -> None:
    database = _ProfileDatabase()
    operation = CustomAllReduce("context_custom_allreduce", 43.0, h=4096, tp_size=4)

    normalized = operation.normalize_perf_query(x=8)
    ordinary = operation.query(database, x=8)
    request = operation.measurement_request(database, _protocol(), x=8)

    assert normalized == _EXPECTED_QUERY
    assert database.queries == [(common.CommQuantMode.half, 4, 32768)]
    assert float(ordinary) == pytest.approx(1.25 * 43)
    assert ordinary.energy == pytest.approx(2.5 * 43)
    assert request is not None
    assert request.query == _EXPECTED_QUERY
    assert request.key == PerfKey.build(_NAMESPACE, _EXPECTED_QUERY, database.measurement_environment)
    assert request.semantic_descriptor == _SEMANTIC_DESCRIPTOR


def test_direct_dispatch_shape_phase_scale_and_consumer_converge_to_one_physical_key() -> None:
    database = _ProfileDatabase()
    context = CustomAllReduce("context_dispatch.custom_allreduce", 43.0, h=4096, tp_size=4)
    generation = CustomAllReduce("generation_dispatch.custom_allreduce", 1.0, h=1, tp_size=4)

    context_request = context.measurement_request(database, _protocol(), x=8)
    generation_request = generation.measurement_request(database, _protocol(), x=32768)

    assert context_request is not None and generation_request is not None
    assert context_request.op_id != generation_request.op_id
    assert context_request.query == generation_request.query == _EXPECTED_QUERY
    assert context_request.key == generation_request.key
    assert context_request.semantic_descriptor == generation_request.semantic_descriptor == _SEMANTIC_DESCRIPTOR
    assert {
        "phase",
        "consumer",
        "op_id",
        "name",
        "scale_factor",
        "gpu_ids",
        "device_uuids",
    }.isdisjoint(context_request.query)


def test_literal_custom_allreduce_row_is_exact_only_and_scaled_for_its_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _ProfileDatabase()
    database._custom_allreduce_data = _exact_data()
    monkeypatch.setattr(CustomAllReduce, "load_data", classmethod(lambda cls, database: None))
    operation = CustomAllReduce("pre_dispatch.custom_allreduce", 2.0, h=1, tp_size=4)

    exact = operation.curated_exact_result(database, x=32768)
    neighbor = operation.curated_exact_result(database, x=32769)

    assert exact is not None
    assert float(exact) == pytest.approx(2.5)
    assert exact.energy == pytest.approx(25.0)
    assert exact.source == "curated_exact"
    assert neighbor is None


@pytest.mark.parametrize("missing_field", ["topology_schema", "topology_fingerprint"])
def test_literal_custom_allreduce_row_rejects_unknown_topology(
    monkeypatch: pytest.MonkeyPatch,
    missing_field: str,
) -> None:
    database = _ProfileDatabase()
    database.measurement_environment = replace(_environment(), **{missing_field: None})
    database._custom_allreduce_data = _exact_data()
    monkeypatch.setattr(CustomAllReduce, "load_data", classmethod(lambda cls, database: None))

    assert (
        CustomAllReduce("custom_allreduce", 1.0, h=1, tp_size=4).curated_exact_result(
            database,
            x=32768,
        )
        is None
    )


def test_custom_allreduce_pure_query_path_remains_legacy_compatible() -> None:
    database = _ProfileDatabase()
    operation = CustomAllReduce("pure_custom_allreduce", 2.0, h=4096, tp_size=4)

    result = operation.query(database, x=8)

    assert database.queries == [(common.CommQuantMode.half, 4, 32768)]
    assert float(result) == pytest.approx(2.5)
    assert result.energy == pytest.approx(5.0)
    assert result.source == "silicon"


def test_custom_allreduce_cold_resolution_measures_once_and_warm_reopen_uses_zero_commands(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiconfigurator.collector import executor as executor_module

    runner = importlib.import_module("aiconfigurator.collector.sglang.custom_allreduce")
    inventory = _four_gpu_nvlink_inventory()
    database = _ProfileDatabase()
    database.measurement_environment = replace(
        _environment(),
        topology_schema=inventory.schema_revision,
        topology_fingerprint=inventory.topology_fingerprint,
    )
    operation = CustomAllReduce("pre_dispatch.custom_allreduce", 2.0, h=4096, tp_size=4)
    protocol = _protocol()
    commands = []
    bootstraps = []
    groups = []

    class _RankGroup:
        def __init__(self, *, device_uuids, protocol, worker_target, collect_all_rank_samples) -> None:
            self.device_uuids = device_uuids
            self.protocol_digest = protocol.digest
            self.worker_target = worker_target
            self.collect_all_rank_samples = collect_all_rank_samples
            self.rank_pids = (101, 102, 103, 104)
            self.measure_calls = []
            groups.append(self)

        def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
            self.measure_calls.append((dtype, operation, element_count))
            return (1.5, 2.0, 2.5)

        def close(self) -> None:
            return None

    class _Channel:
        def __init__(self) -> None:
            self.pending = []
            self.closed = False

        def send(self, command) -> None:
            commands.append(command)
            case = json.loads(command.payload)
            assert command.protocol is not None
            raw_result = runner.run_custom_allreduce_case(**case, protocol=command.protocol)
            self.pending.append(
                WorkerReply(
                    invocation_id=command.invocation_id,
                    request_digest=command.request_digest,
                    raw_result=dict(raw_result),
                )
            )

        def recv(self):
            return self.pending.pop(0)

        def is_alive(self) -> bool:
            return not self.closed

        def close(self) -> None:
            self.closed = True

        def join(self) -> None:
            return None

    runner.close_custom_allreduce_worker()
    monkeypatch.setattr(CustomAllReduce, "load_data", classmethod(lambda cls, database: None))
    monkeypatch.setattr(executor_module, "PersistentNcclRankGroup", _RankGroup)
    monkeypatch.setattr(runner, "_runtime_metadata", lambda: ("0.5.10", "NVIDIA GB200"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(device.uuid for device in inventory.devices))

    adapter_index = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY})

    def _resolve_adapter(candidate):
        routes = adapter_index.routes_for(
            (candidate.key.namespace, candidate.environment.backend, candidate.environment.backend_version)
        )
        assert len(routes) == 1
        return routes[0]

    def _worker_factory(bootstrap):
        bootstraps.append(bootstrap)
        return _Channel()

    def _wait_ready(available, timeout_seconds):
        del timeout_seconds
        return tuple(channel for channel in available if channel.pending)

    executor = PersistentMeasurementExecutor(
        inventory=inventory,
        scheduler=HardwareAwareScheduler(inventory),
        resolve_adapter=_resolve_adapter,
        worker_factory=_worker_factory,
        wait_ready=_wait_ready,
    )
    overlay_path = tmp_path / "custom-allreduce-measurements.sqlite"
    overlay = OverlayStore(overlay_path)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=1, max_wall_seconds=10.0),
        protocol,
    )

    cold = session.execute_callback(lambda: operation.query_with_resolution(database, session=session, x=8))
    warm = session.execute_callback(lambda: operation.query_with_resolution(database, session=session, x=8))

    assert len(bootstraps) == 1
    assert bootstraps[0].device_uuids == tuple(device.uuid for device in inventory.devices)
    assert len(commands) == 1
    assert len(groups) == 1
    assert groups[0].measure_calls == [("half", "all_reduce", 32768)]
    assert session.report.unique_misses == 1
    assert session.report.accepted_records == 1
    assert (float(cold), cold.energy, cold.source) == (4.0, 0.0, "overlay")
    assert (float(warm), warm.energy, warm.source) == (float(cold), cold.energy, cold.source)

    overlay.close()
    executor.close()
    runner.close_custom_allreduce_worker()
    reopened = OverlayStore(overlay_path)

    def _unexpected_worker(bootstrap):
        pytest.fail(f"warm CustomAllReduce overlay unexpectedly launched worker {bootstrap}")

    reopened_executor = PersistentMeasurementExecutor(
        inventory=inventory,
        scheduler=HardwareAwareScheduler(inventory),
        resolve_adapter=_resolve_adapter,
        worker_factory=_unexpected_worker,
        wait_ready=_wait_ready,
    )
    reopened_session = ResolutionSession(
        reopened,
        reopened_executor,
        ResolutionBudget(max_new_keys=1, max_wall_seconds=10.0),
        protocol,
    )

    reopened_warm = reopened_session.execute_callback(
        lambda: operation.query_with_resolution(database, session=reopened_session, x=8)
    )

    assert len(commands) == 1
    assert groups[0].measure_calls == [("half", "all_reduce", 32768)]
    assert reopened_session.report.unique_misses == 0
    assert (float(reopened_warm), reopened_warm.energy, reopened_warm.source) == (
        float(cold),
        cold.energy,
        cold.source,
    )
    reopened.close()
    reopened_executor.close()
