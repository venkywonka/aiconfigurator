# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Milestone 2C gates for the generic NCCL runtime and pure prediction path."""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import textwrap

import pytest

from aiconfigurator.collector import (
    FabricRequirement,
    GpuDevice,
    HardwareDiscoveryEvidence,
    HardwareInventory,
    ResourceContract,
    canonical_topology_fingerprint,
)
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.scheduler import CollectionJob, HardwareAwareScheduler
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)

pytestmark = pytest.mark.unit

_NCCL_NAMESPACE = f"{PerfFile.NCCL.value}/v1"


def _nccl_request(*, num_gpus: int = 2) -> MeasurementRequest:
    environment = MeasurementEnvironment(
        system="h100_nvlink4",
        backend="trtllm",
        backend_version="1.2.0",
        gpu_class="NVIDIA H100 80GB HBM3",
        runtime_versions={"cuda": "13.0", "nccl": "2.27.3"},
        topology_schema="scheduler-test-v1",
        topology_fingerprint="two-domain-nvlink-fingerprint",
    )
    query = {
        "nccl_dtype": "half",
        "operation": "all_reduce",
        "num_gpus": num_gpus,
        "message_size": 4096,
    }
    return MeasurementRequest(
        op_id=f"nccl-world-{num_gpus}",
        key=PerfKey.build(_NCCL_NAMESPACE, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"tensor_seed": 0},
        protocol=MeasurementProtocol(
            revision="cuda-event-samples-v1",
            warmups=2,
            samples=3,
            timer="cuda_event",
            tuning_revision="torch-nccl-persistent-v1",
        ),
    )


def _two_domain_inventory() -> HardwareInventory:
    gpu_ids = (0, 1, 2, 3)
    devices = tuple(
        GpuDevice(
            index=gpu_id,
            uuid=f"GPU-{gpu_id}",
            name="NVIDIA H100 80GB HBM3",
            pci_bus_id=f"00000000:{gpu_id:02X}:00.0",
        )
        for gpu_id in gpu_ids
    )
    nvlink_directions = {(0, 1), (1, 0), (2, 3), (3, 2)}
    links = {
        (left, right): "NV4" if (left, right) in nvlink_directions else "SYS"
        for left in gpu_ids
        for right in gpu_ids
        if left != right
    }
    p2p_read = {pair: pair in nvlink_directions for pair in links}
    p2p_write = dict(p2p_read)
    schema_revision = "scheduler-test-v1"
    return HardwareInventory(
        schema_revision=schema_revision,
        devices=devices,
        links=links,
        p2p_read=p2p_read,
        p2p_write=p2p_write,
        fabric_domains={
            0: "nvlink:0",
            1: "nvlink:0",
            2: "nvlink:1",
            3: "nvlink:1",
        },
        topology_fingerprint=canonical_topology_fingerprint(
            schema_revision,
            devices,
            links,
            p2p_read,
            p2p_write,
        ),
        evidence=HardwareDiscoveryEvidence(
            raw_gpu_query="synthetic query",
            raw_topology="synthetic topology",
            raw_p2p_read="synthetic read matrix",
            raw_p2p_write="synthetic write matrix",
        ),
    )


def test_nccl_adapter_registers_exact_case_resource_and_persistent_runner() -> None:
    registry = importlib.import_module("aiconfigurator.collector.network.registry")
    adapter = importlib.import_module("aiconfigurator.collector.network.nccl_adapter")
    runner = importlib.import_module("aiconfigurator.collector.network.nccl")
    executor = importlib.import_module("aiconfigurator.collector.executor")

    lazy = registry.NCCL_LAZY_SPEC
    assert lazy.namespace == _NCCL_NAMESPACE
    assert lazy.run_module == "aiconfigurator.collector.network.nccl"
    assert lazy.adapter_module == "aiconfigurator.collector.network.nccl_adapter"
    assert registry.NETWORK_LAZY_REGISTRY[0].perf_filename == PerfFile.NCCL
    assert registry.NETWORK_LAZY_REGISTRY[0].lazy is lazy

    request = _nccl_request()
    case = adapter.nccl_request_to_case(request)
    assert case == {
        "dtype": "half",
        "nccl_op": "all_reduce",
        "element_count": 4096,
        "num_gpus": 2,
    }
    assert adapter.nccl_resource_for_request(request, case) == ResourceContract(
        gpu_count=2,
        fabric=FabricRequirement.NVLINK,
        reserve_fabric_domain=True,
    )

    signature = inspect.signature(runner.run_nccl_case)
    assert tuple(signature.parameters) == (
        "dtype",
        "nccl_op",
        "element_count",
        "num_gpus",
        "runtime",
        "measure_power",
        "protocol",
    )
    assert signature.parameters["runtime"].default is None
    assert hasattr(executor, "PersistentNcclRuntime")


def test_nccl_resource_saturates_disjoint_gpus_without_entering_its_fabric_domain() -> None:
    adapter = importlib.import_module("aiconfigurator.collector.network.nccl_adapter")
    request = _nccl_request()
    case = adapter.nccl_request_to_case(request)
    collective = CollectionJob(
        request_digest=request.key.digest,
        adapter_namespace=_NCCL_NAMESPACE,
        contract=adapter.nccl_resource_for_request(request, case),
        payload=b"nccl",
    )
    singles = tuple(
        CollectionJob(
            request_digest=f"single-{index}",
            adapter_namespace="fake-single-gpu/v1",
            contract=ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE),
            payload=b"single",
        )
        for index in range(4)
    )

    waves = HardwareAwareScheduler(_two_domain_inventory()).plan((collective, *singles))
    assert len(waves) == 2
    first_collective = next(assignment for assignment in waves[0] if assignment.job is collective)
    first_singles = tuple(assignment for assignment in waves[0] if assignment.job is not collective)
    assert first_collective.gpu_ids == (0, 1)
    assert first_collective.reserved_domains == frozenset({"gpu:0", "gpu:1", "fabric:nvlink:0"})
    assert {assignment.gpu_ids for assignment in first_singles} == {(2,), (3,)}
    assert all("fabric:nvlink:0" not in assignment.reserved_domains for assignment in first_singles)
    assert {assignment.gpu_ids for assignment in waves[1]} == {(0,), (1,)}


def test_nccl_result_requires_operation_and_protocol_to_round_trip() -> None:
    adapter = importlib.import_module("aiconfigurator.collector.network.nccl_adapter")
    request = _nccl_request()
    case = adapter.nccl_request_to_case(request)
    raw = {
        "latency_ms": 1.0,
        "energy_wms": 0.0,
        "samples_ms": (0.9, 1.0, 1.1),
        "statistic": "median",
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "nccl_dtype": "half",
            "op_name": "all_reduce",
            "num_gpus": 2,
            "message_size": 4096,
            "latency": 1.0,
        },
        "provenance": {"runtime": "fake-persistent-group"},
    }

    record = adapter.nccl_result_to_record(request, case, raw)
    assert record.key == request.key
    assert record.perf_row["op_name"] == "all_reduce"

    missing_operation = raw | {"perf_row": dict(raw["perf_row"])}
    del missing_operation["perf_row"]["op_name"]
    with pytest.raises(ValueError, match="operation"):
        adapter.nccl_result_to_record(request, case, missing_operation)

    with pytest.raises((TypeError, ValueError), match="sequence"):
        adapter.nccl_result_to_record(request, case, raw | {"samples_ms": "111"})

    boolean_latency = raw | {"perf_row": dict(raw["perf_row"], latency=True)}
    with pytest.raises((TypeError, ValueError), match="latency"):
        adapter.nccl_result_to_record(request, case, boolean_latency)


def test_public_prediction_api_stays_gpu_and_collector_free() -> None:
    probe = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockGpuAndCollector(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                del path, target
                blocked = (
                    fullname == "torch"
                    or fullname.startswith("torch.")
                    or fullname == "aiconfigurator.collector"
                    or fullname.startswith("aiconfigurator.collector.")
                )
                if blocked:
                    raise AssertionError(f"pure prediction imported {fullname}")
                return None

        sys.meta_path.insert(0, BlockGpuAndCollector())
        from aiconfigurator.sdk.predict import predict_agg_worker

        sentinel = object()

        class PurePredictor:
            def predict_agg_worker(self, **kwargs):
                assert kwargs["ctx_tokens"] == 17
                assert kwargs["marker"] == "pure"
                return sentinel

        result = predict_agg_worker(
            model=object(),
            backend=object(),
            database=object(),
            runtime_config=object(),
            ctx_tokens=17,
            predictor=PurePredictor(),
            marker="pure",
        )
        assert result is sentinel
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
