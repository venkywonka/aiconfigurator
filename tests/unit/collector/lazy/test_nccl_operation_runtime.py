# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict-TDD contracts for exact NCCL operation and persistent runner seams."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.executor import PersistentNcclRuntime
from aiconfigurator.collector.network import nccl as nccl_runner
from aiconfigurator.collector.network import nccl_adapter
from aiconfigurator.collector.network.nccl_adapter import nccl_result_to_record
from aiconfigurator.collector.network.registry import NCCL_LAZY_SPEC, NETWORK_LAZY_REGISTRY
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.sdk import common
from aiconfigurator.sdk.operations.communication import NCCL
from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.performance_result import PerformanceResult
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)

pytestmark = pytest.mark.unit

_NAMESPACE = f"{PerfFile.NCCL}/v1"


def _environment() -> MeasurementEnvironment:
    return MeasurementEnvironment(
        system="test_system",
        backend="trtllm",
        backend_version="1.0",
        gpu_class="NVIDIA H100 80GB HBM3",
        runtime_versions={"cuda": "13.0", "nccl": "2.27.3"},
        topology_schema="nvidia-smi-v1",
        topology_fingerprint="nvlink-pair-fingerprint",
    )


def _protocol(*, samples: int = 3, statistic: str = "median") -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=samples,
        statistic=statistic,
        timer="cuda_event",
        tuning_revision="torch-nccl-persistent-v1",
    )


def _write_database(root: Path) -> PerfDatabase:
    (root / "test_system.yaml").write_text(
        """
data_dir: data/test_system
gpu:
  name: NVIDIA H100 80GB HBM3
  mem_bw: 1000000000000
  bfloat16_tc_flops: 1000000000000000
  fp8_tc_flops: 2000000000000000
  int8_tc_flops: 2000000000000000
  fp4_tc_flops: 4000000000000000
  sm_version: 90
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
    data_dir = root / "data" / "test_system" / "nccl" / "2.27.3"
    data_dir.mkdir(parents=True)
    (data_dir / str(PerfFile.NCCL)).write_text(
        "nccl_dtype,num_gpus,message_size,op_name,latency,power\n"
        "half,2,640,all_reduce,1.5,10.0\n"
        "half,2,1280,all_reduce,2.5,20.0\n",
        encoding="utf-8",
    )
    database = PerfDatabase("test_system", "trtllm", "1.0", systems_root=str(root))
    database.set_measurement_environment(_environment())
    return database


def _operation(*, scale_factor: float = 1.0) -> NCCL:
    return NCCL(
        "context_all_reduce",
        scale_factor,
        nccl_op="all_reduce",
        num_elements_per_token=128,
        num_gpus=2,
        comm_quant_mode=common.CommQuantMode.half,
        seq_split=4,
    )


def _manual_request(*, statistic: str = "median") -> MeasurementRequest:
    environment = _environment()
    query = {
        "nccl_dtype": "half",
        "operation": "all_reduce",
        "num_gpus": 2,
        "message_size": 640,
    }
    return MeasurementRequest(
        op_id="context_all_reduce",
        key=PerfKey.build(_NAMESPACE, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"tensor_generator": "normal-v1", "seed": 0},
        protocol=_protocol(statistic=statistic),
    )


@pytest.fixture(autouse=True)
def _clear_nccl_cache() -> None:
    NCCL.clear_cache()
    yield
    NCCL.clear_cache()


def test_nccl_shared_normalization_feeds_query_and_canonical_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = _write_database(tmp_path)
    operation = _operation(scale_factor=3.0)
    ordinary_queries: list[tuple[object, int, str, int]] = []

    def _query_nccl(dtype: object, num_gpus: int, op_name: str, message_size: int) -> PerformanceResult:
        ordinary_queries.append((dtype, num_gpus, op_name, message_size))
        return PerformanceResult(1.0, energy=2.0, source="silicon")

    monkeypatch.setattr(database, "query_nccl", _query_nccl)
    normalized = operation.normalize_perf_query(x=17)
    ordinary = operation.query(database, x=17)
    request = operation.measurement_request(database, _protocol(), x=17)

    assert normalized == {
        "nccl_dtype": "half",
        "operation": "all_reduce",
        "num_gpus": 2,
        "message_size": 640,
    }
    assert request is not None
    assert request.query == normalized
    assert request.key.namespace == _NAMESPACE
    assert request.environment is database.measurement_environment
    assert request.environment.topology_schema == "nvidia-smi-v1"
    assert request.environment.topology_fingerprint == "nvlink-pair-fingerprint"
    assert request.semantic_descriptor["seed"] == 0
    assert ordinary_queries == [(common.CommQuantMode.half, 2, "all_reduce", 640)]
    assert float(ordinary) == pytest.approx(3.0)
    assert ordinary.energy == pytest.approx(6.0)


def test_nccl_literal_curated_hit_does_not_promote_interpolation_to_exact(tmp_path: Path) -> None:
    database = _write_database(tmp_path)
    operation = _operation(scale_factor=2.0)

    literal = operation.curated_exact_result(database, x=17)
    interpolated = operation.query(database, x=21)
    interpolation_probe = operation.curated_exact_result(database, x=21)

    assert literal is not None
    assert float(literal) == pytest.approx(3.0)
    assert literal.energy == pytest.approx(30.0)
    assert literal.source == "curated_exact"
    assert float(interpolated) > 0
    assert interpolation_probe is None


def _raw_result(request: MeasurementRequest) -> dict[str, Any]:
    return {
        "latency_ms": 1.5,
        "energy_wms": 4.0,
        "samples_ms": [1.0, 1.5, 2.0],
        "statistic": "median",
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "nccl_dtype": "half",
            "op_name": "all_reduce",
            "num_gpus": 2,
            "message_size": 640,
            "latency": 1.5,
        },
        "provenance": {"runtime": "persistent_torch_distributed", "worker_pid": 1234},
    }


@pytest.mark.parametrize(
    "invalid_samples",
    (
        [1.0, 2.0],
        [1.0, float("nan"), 2.0],
        [1.0, 0.0, 2.0],
    ),
)
def test_nccl_result_round_trips_key_and_rejects_invalid_samples(
    invalid_samples: list[object],
) -> None:
    request = _manual_request()
    case = {
        "dtype": "half",
        "nccl_op": "all_reduce",
        "element_count": 640,
        "num_gpus": 2,
    }
    valid = _raw_result(request)

    record = nccl_result_to_record(request, case, valid)
    assert record.key == request.key
    assert record.perf_row["op_name"] == request.query["operation"]
    assert record.samples_ms == (1.0, 1.5, 2.0)

    invalid = copy.deepcopy(valid)
    invalid["samples_ms"] = invalid_samples
    with pytest.raises((TypeError, ValueError), match=r"samples|sample count"):
        nccl_result_to_record(request, case, invalid)


@pytest.mark.parametrize("invalid_sample", (True, "1.5"))
def test_nccl_result_rejects_non_numeric_sample_types_before_coercion(invalid_sample: object) -> None:
    request = _manual_request()
    case = {
        "dtype": "half",
        "nccl_op": "all_reduce",
        "element_count": 640,
        "num_gpus": 2,
    }
    raw = _raw_result(request)
    raw["samples_ms"] = (1.0, invalid_sample, 2.0)

    with pytest.raises(TypeError, match="samples"):
        nccl_result_to_record(request, case, raw)


def test_nccl_result_requires_latency_to_equal_sample_median() -> None:
    request = _manual_request()
    case = {
        "dtype": "half",
        "nccl_op": "all_reduce",
        "element_count": 640,
        "num_gpus": 2,
    }
    raw = _raw_result(request)
    raw["latency_ms"] = 1.0
    raw["perf_row"] = dict(raw["perf_row"], latency=1.0)

    with pytest.raises(ValueError, match=r"median|samples"):
        nccl_result_to_record(request, case, raw)


def test_nccl_route_rejects_non_median_statistic_before_resource_construction(monkeypatch) -> None:
    resource_calls: list[object] = []
    original_resource = nccl_adapter.nccl_resource_for_request

    def _tracked_resource(request, case):
        resource_calls.append((request, case))
        return original_resource(request, case)

    monkeypatch.setattr(nccl_adapter, "nccl_resource_for_request", _tracked_resource)
    route = LazyAdapterIndex.from_registries({"trtllm": NETWORK_LAZY_REGISTRY}).routes_for(
        (NCCL_LAZY_SPEC.namespace, "trtllm", "1.0")
    )[0]

    with pytest.raises(ValueError, match=r"median|statistic"):
        route.prepare(_manual_request(statistic="mean"))
    assert resource_calls == []


class _FakeRuntime:
    def __init__(self, protocol_digest: str) -> None:
        self.protocol_digest = protocol_digest
        self.calls: list[tuple[str, str, int]] = []

    def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
        self.calls.append((dtype, operation, element_count))
        return (3.0, 1.0, 2.0)


def test_run_nccl_case_uses_persistent_runtime_without_offline_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(_protocol().digest)

    def _unexpected_offline_run(**kwargs: object) -> None:
        raise AssertionError(f"persistent branch invoked offline runner: {kwargs}")

    monkeypatch.setattr(nccl_runner, "_run_nccl_tests_case", _unexpected_offline_run)
    measured = nccl_runner.run_nccl_case(
        "half",
        "all_reduce",
        640,
        2,
        runtime=runtime,
        measure_power=False,
    )

    assert runtime.calls == [("half", "all_reduce", 640)]
    assert measured.samples_ms == (3.0, 1.0, 2.0)
    assert measured.latency_ms == pytest.approx(2.0)
    assert measured.protocol_digest == _protocol().digest
    assert measured.perf_row == {
        "nccl_dtype": "half",
        "op_name": "all_reduce",
        "num_gpus": 2,
        "message_size": 640,
        "latency": 2.0,
    }
    assert measured.provenance["runtime"] == "persistent_torch_distributed"


def test_lazy_nccl_protocol_creates_reuses_and_closes_one_rank_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiconfigurator.collector import executor as executor_module

    protocol = _protocol()
    groups = []

    class _RankGroup:
        def __init__(self, *, device_uuids, protocol) -> None:
            self.device_uuids = device_uuids
            self.protocol_digest = protocol.digest
            self.rank_pids = (101, 102)
            self.calls = []
            self.close_calls = 0
            groups.append(self)

        def measure(self, dtype, operation, element_count):
            self.calls.append((dtype, operation, element_count))
            return (1.0, 1.5, 2.0)

        def close(self) -> None:
            self.close_calls += 1

    nccl_runner.close_nccl_worker()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    monkeypatch.setattr(executor_module, "PersistentNcclRankGroup", _RankGroup)

    first = nccl_runner.run_nccl_case("half", "all_reduce", 640, 2, protocol=protocol)
    second = nccl_runner.run_nccl_case("half", "all_gather", 1280, 2, protocol=protocol)
    nccl_runner.run_nccl_case.close_worker()
    nccl_runner.run_nccl_case.close_worker()

    assert len(groups) == 1
    assert groups[0].device_uuids == ("GPU-a", "GPU-b")
    assert groups[0].calls == [
        ("half", "all_reduce", 640),
        ("half", "all_gather", 1280),
    ]
    assert groups[0].close_calls == 1
    assert first.protocol_digest == second.protocol_digest == protocol.digest
    assert first.samples_ms == second.samples_ms == (1.0, 1.5, 2.0)
    assert first.provenance["rank_pids"] == [101, 102]


def test_offline_nccl_case_stops_power_monitor_when_subprocess_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from aiconfigurator.collector import benchmark

    events: list[str] = []

    class _PowerMonitor:
        def __init__(self, device_id: int) -> None:
            assert device_id == 0

        @staticmethod
        def _init_handle() -> bool:
            return True

        def start_sampling(self) -> bool:
            events.append("start")
            return True

        def stop_sampling(self) -> None:
            events.append("stop")

    monkeypatch.setattr(benchmark, "PowerMonitor", _PowerMonitor)
    monkeypatch.setattr(
        nccl_runner.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nccl-tests failed")),
    )

    with pytest.raises(RuntimeError, match="nccl-tests failed"):
        nccl_runner._run_nccl_tests_case(
            dtype="half",
            nccl_op="all_reduce",
            message_size_bytes=1280,
            num_gpus=2,
            measure_power=True,
        )

    assert events == ["start", "stop"]


class _FakePersistentRankGroup:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.measure_calls: list[tuple[str, str, int]] = []
        self.close_calls = 0

    def measure(self, dtype: str, operation: str, element_count: int) -> tuple[float, ...]:
        self.measure_calls.append((dtype, operation, element_count))
        if self.fail:
            raise RuntimeError("rank 1 crashed")
        return (1.0, 1.5, 2.0)

    def close(self) -> None:
        self.close_calls += 1


def _persistent_runtime(group: _FakePersistentRankGroup) -> PersistentNcclRuntime:
    return PersistentNcclRuntime(
        measure=group.measure,
        close=group.close,
        protocol=_protocol(),
    )


def test_persistent_nccl_runtime_closes_rank_group_once() -> None:
    group = _FakePersistentRankGroup()
    runtime = _persistent_runtime(group)

    assert runtime.measure("half", "all_reduce", 640) == (1.0, 1.5, 2.0)
    runtime.close()
    runtime.close()

    assert group.measure_calls == [("half", "all_reduce", 640)]
    assert group.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.measure("half", "all_reduce", 640)


def test_persistent_nccl_runtime_retries_a_failed_close_callback() -> None:
    close_calls = 0

    def close_fails_once() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise RuntimeError("persistent rank survived forced shutdown")

    runtime = PersistentNcclRuntime(close=close_fails_once, protocol=_protocol())

    with pytest.raises(RuntimeError, match="survived forced shutdown"):
        runtime.close()
    with pytest.raises(RuntimeError, match="shutdown is incomplete"):
        runtime.measure("half", "all_reduce", 640)

    runtime.close()
    runtime.close()

    assert close_calls == 2
    with pytest.raises(RuntimeError, match="closed"):
        runtime.measure("half", "all_reduce", 640)


def test_persistent_nccl_runtime_fault_closes_and_poison_rank_group() -> None:
    group = _FakePersistentRankGroup(fail=True)
    runtime = _persistent_runtime(group)

    with pytest.raises(RuntimeError, match="rank 1 crashed"):
        runtime.measure("half", "all_reduce", 640)

    assert group.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        runtime.measure("half", "all_reduce", 640)
    assert group.measure_calls == [("half", "all_reduce", 640)]
