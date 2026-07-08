# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.executor import PersistentMeasurementExecutor, WorkerReply
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.scheduler import HardwareAwareScheduler
from aiconfigurator.collector.trtllm.registry import GEMM_LAZY_SPEC, TRTLLM_LAZY_REGISTRY
from aiconfigurator.collector.types import (
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    canonical_topology_fingerprint,
)
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.gemm import GEMM
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.overlay import OverlayStore
from aiconfigurator.sdk.resolution.session import ResolutionBudget, ResolutionSession
from aiconfigurator.sdk.resolution.types import MeasurementEnvironment, MeasurementProtocol


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=3,
        samples=6,
        statistic="median",
        timer="cuda_event",
        tuning_revision="trtllm-linear-v1",
    )


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="test_system",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="test-gpu",
        runtime_versions={"tensorrt_llm": "1.0"},
        topology_schema="test-topology-v1",
        topology_fingerprint="test-topology",
    )


def _write_database(root: Path) -> PerfDatabase:
    (root / "test_system.yaml").write_text(
        """
data_dir: data/test_system
gpu:
  mem_bw: 1000000000000
  bfloat16_tc_flops: 1000000000000000
  fp8_tc_flops: 2000000000000000
  int8_tc_flops: 2000000000000000
  fp4_tc_flops: 4000000000000000
  sm_version: 100
node:
  num_gpus_per_node: 4
  intra_node_bw: 100000000000
  inter_node_bw: 100000000000
  pcie_bw: 10000000000
misc:
  nccl_version: "2.27.3"
""".lstrip(),
        encoding="utf-8",
    )
    data_dir = root / "data" / "test_system" / "trtllm" / "1.0"
    data_dir.mkdir(parents=True)
    (data_dir / str(PerfFile.GEMM)).write_text(
        "gemm_dtype,m,n,k,latency,power\nbfloat16,8,128,128,1.0,10.0\nbfloat16,16,128,128,2.0,20.0\n",
        encoding="utf-8",
    )
    (data_dir / "computescale_perf.txt").write_text(
        "gemm_dtype,m,k,latency,power\n",
        encoding="utf-8",
    )
    (data_dir / "scale_matrix_perf.txt").write_text(
        "gemm_dtype,m,k,latency,power\n",
        encoding="utf-8",
    )
    database = PerfDatabase("test_system", "trtllm", "1.0", systems_root=str(root))
    database.set_measurement_environment(_environment())
    return database


@pytest.fixture(autouse=True)
def _clear_gemm_cache() -> None:
    GEMM.clear_cache()
    yield
    GEMM.clear_cache()


def test_one_normalization_feeds_query_and_request_with_scale_then_cp_then_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _write_database(tmp_path)
    operation = GEMM(
        "router",
        3.0,
        n=128,
        k=128,
        quant_mode=common.GEMMQuantMode.bfloat16,
        scale_num_tokens=2,
        seq_split=4,
    )

    ordinary_queries = []

    def _query_gemm(m, n, k, quant_mode, database_mode=None):
        ordinary_queries.append((m, n, k, quant_mode, database_mode))
        return PerformanceResult(1.0, energy=2.0)

    monkeypatch.setattr(database, "query_gemm", _query_gemm)
    normalized = operation.normalize_perf_query(x=17, quant_mode=common.GEMMQuantMode.fp8_block)
    ordinary = operation.query(database, x=17, quant_mode=common.GEMMQuantMode.fp8_block)
    request = operation.measurement_request(
        database,
        _protocol(),
        x=17,
        quant_mode=common.GEMMQuantMode.fp8_block,
    )

    assert normalized == {"gemm_type": "fp8_block", "m": 2, "n": 128, "k": 128}
    assert request is not None
    assert request.query == normalized
    assert request.key.namespace == f"{PerfFile.GEMM}/v1"
    assert request.environment is database.measurement_environment
    assert request.semantic_descriptor == {"tensor_generator": "normal-v1", "seed": 0}
    assert ordinary_queries == [(2, 128, 128, common.GEMMQuantMode.fp8_block, None)]
    assert float(ordinary) == pytest.approx(3.0)


def test_literal_curated_row_hits_but_interpolated_and_extrapolated_points_miss(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    operation = GEMM("router", 2.0, 128, 128, common.GEMMQuantMode.bfloat16)

    literal = operation.curated_exact_result(database, x=8)
    interpolated = operation.query(database, x=9)
    interpolation_probe = operation.curated_exact_result(database, x=9)
    extrapolated = operation.query(database, x=32)
    extrapolation_probe = operation.curated_exact_result(database, x=32)

    assert literal is not None
    assert float(literal) == pytest.approx(2.0)
    assert literal.energy == pytest.approx(20.0)
    assert literal.source == "curated_exact"
    assert float(interpolated) > 0
    assert float(extrapolated) > 0
    assert interpolation_probe is None
    assert extrapolation_probe is None


def test_scale_factor_changes_only_conversion_not_physical_key(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    first = GEMM("first", 1.0, 128, 128, common.GEMMQuantMode.bfloat16)
    second = GEMM("second", 5.0, 128, 128, common.GEMMQuantMode.bfloat16)

    first_request = first.measurement_request(database, _protocol(), x=8)
    second_request = second.measurement_request(database, _protocol(), x=8)

    assert first_request is not None and second_request is not None
    assert first_request.key == second_request.key
    assert float(first.curated_exact_result(database, x=8)) == pytest.approx(1.0)
    assert float(second.curated_exact_result(database, x=8)) == pytest.approx(5.0)


def test_fp8_static_composite_has_no_single_measurement_request(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    operation = GEMM("static", 1.0, 128, 128, common.GEMMQuantMode.fp8_static)

    assert operation.measurement_request(database, _protocol(), x=8) is None


def test_absent_quant_mode_remains_an_exact_miss_with_a_simple_request(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    operation = GEMM("missing", 1.0, 128, 128, common.GEMMQuantMode.fp8_block)

    assert operation.curated_exact_result(database, x=8) is None
    request = operation.measurement_request(database, _protocol(), x=8)
    assert request is not None
    assert request.query["gemm_type"] == "fp8_block"


def test_source_registry_attaches_the_identical_packaged_spec() -> None:
    from collector.trtllm.registry import REGISTRY

    gemm_entry = next(entry for entry in REGISTRY if entry.op == "gemm")
    assert gemm_entry.lazy is GEMM_LAZY_SPEC


def test_database_rejects_a_measurement_environment_for_another_route(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    wrong = MeasurementEnvironment(
        system="other-system",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="test-gpu",
        runtime_versions={"tensorrt_llm": "1.0"},
    )

    with pytest.raises(ValueError, match=r"system|route"):
        database.set_measurement_environment(wrong)


def test_bf16_cold_resolution_persists_once_and_warm_reopen_uses_zero_commands(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    device = GpuDevice(
        index=0,
        uuid="GPU-test-0",
        name="test-gpu",
        pci_bus_id="00000000:00:00.0",
    )
    topology_fingerprint = canonical_topology_fingerprint(
        "test-topology-v1",
        (device,),
        {},
        {},
        {},
    )
    inventory = HardwareInventory(
        schema_revision="test-topology-v1",
        devices=(device,),
        links={},
        p2p_read={},
        p2p_write={},
        fabric_domains={},
        topology_fingerprint=topology_fingerprint,
        evidence=HardwareDiscoveryEvidence(
            raw_gpu_query="synthetic query",
            raw_topology="synthetic topology",
            raw_p2p_read="synthetic reads",
            raw_p2p_write="synthetic writes",
        ),
    )
    database.set_measurement_environment(
        MeasurementEnvironment(
            system="test_system",
            backend="trtllm",
            backend_version="1.0",
            gpu_class="test-gpu",
            runtime_versions={"tensorrt_llm": "1.0"},
            topology_schema=inventory.schema_revision,
            topology_fingerprint=inventory.topology_fingerprint,
        )
    )
    operation = GEMM("router", 2.0, 128, 128, common.GEMMQuantMode.bfloat16)
    protocol = _protocol()
    request = operation.measurement_request(database, protocol, x=9)
    assert request is not None
    ordinary = operation.query(database, x=9)
    assert operation.curated_exact_result(database, x=9) is None

    commands = []

    class _Channel:
        def __init__(self) -> None:
            self.pending = []
            self.closed = False

        def send(self, command) -> None:
            commands.append(command)
            case = json.loads(command.payload)
            assert command.protocol is not None
            latency_ms = 4.0
            self.pending.append(
                WorkerReply(
                    invocation_id=command.invocation_id,
                    request_digest=command.request_digest,
                    raw_result={
                        "latency_ms": latency_ms,
                        "energy_wms": 40.0,
                        "samples_ms": (3.0, 4.0, 4.0, 4.0, 5.0, 6.0),
                        "statistic": command.protocol.statistic,
                        "protocol_digest": command.protocol.digest,
                        "perf_row": {
                            "gemm_dtype": case["gemm_type"],
                            "m": case["m"],
                            "n": case["n"],
                            "k": case["k"],
                            "latency": latency_ms,
                        },
                        "provenance": {
                            "worker": "cpu-fake",
                            "device_uuid": "GPU-test-0",
                            "framework_version": "1.0",
                            "device": "test-gpu",
                        },
                    },
                )
            )

        def recv(self):
            return self.pending.pop(0)

        def is_alive(self) -> bool:
            return not self.closed

        def close(self) -> None:
            self.closed = True

        def join(self) -> None:
            pass

    channels = []

    def _worker_factory(bootstrap):
        channel = _Channel()
        channels.append((bootstrap, channel))
        return channel

    def _wait_ready(available, timeout_seconds):
        del timeout_seconds
        return tuple(channel for channel in available if channel.pending)

    adapter_index = LazyAdapterIndex.from_registries({"trtllm": TRTLLM_LAZY_REGISTRY})

    def _resolve_adapter(candidate):
        routes = adapter_index.routes_for(
            (candidate.key.namespace, candidate.environment.backend, candidate.environment.backend_version)
        )
        assert len(routes) == 1
        return routes[0]

    executor = PersistentMeasurementExecutor(
        inventory=inventory,
        scheduler=HardwareAwareScheduler(inventory),
        resolve_adapter=_resolve_adapter,
        worker_factory=_worker_factory,
        wait_ready=_wait_ready,
    )
    overlay_path = tmp_path / "measurements.sqlite"
    overlay = OverlayStore(overlay_path)
    session = ResolutionSession(
        overlay,
        executor,
        ResolutionBudget(max_new_keys=1, max_wall_seconds=10.0),
        protocol,
    )

    cold = session.execute_callback(lambda: operation.query_with_resolution(database, session=session, x=9))
    cold_record = overlay.lookup(request.key, protocol)
    assert cold_record is not None
    assert session.report.unique_misses == 1
    assert session.report.accepted_records == 1
    assert len(commands) == 1
    assert float(ordinary) != float(cold)
    assert float(cold) == pytest.approx(8.0)
    assert cold.source == "overlay"

    warm = session.execute_callback(lambda: operation.query_with_resolution(database, session=session, x=9))
    assert len(commands) == 1
    assert (float(warm), warm.energy, warm.source) == (float(cold), cold.energy, cold.source)

    overlay.close()
    executor.close()
    reopened = OverlayStore(overlay_path)

    def _unexpected_worker(bootstrap):
        pytest.fail(f"warm overlay unexpectedly launched worker {bootstrap}")

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
        lambda: operation.query_with_resolution(database, session=reopened_session, x=9)
    )
    reopened_record = reopened.lookup(request.key, protocol)
    assert reopened_record is not None
    assert len(commands) == 1
    assert reopened_session.report.unique_misses == 0
    assert (float(reopened_warm), reopened_warm.energy, reopened_warm.source) == (
        float(cold),
        cold.energy,
        cold.source,
    )
    assert (
        reopened_record.key,
        reopened_record.latency_ms,
        reopened_record.energy_wms,
        reopened_record.samples_ms,
        reopened_record.perf_row,
        reopened_record.provenance,
    ) == (
        cold_record.key,
        cold_record.latency_ms,
        cold_record.energy_wms,
        cold_record.samples_ms,
        cold_record.perf_row,
        cold_record.provenance,
    )
    reopened.close()
    reopened_executor.close()
