# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 exact four-GPU SGLang CustomAllReduce contracts."""

from __future__ import annotations

import builtins
import copy
import importlib
import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.types import FabricRequirement, ResourceContract
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)

pytestmark = pytest.mark.unit

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_NAMESPACE = f"{PerfFile.CUSTOM_ALLREDUCE}/v1"
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
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
        runtime_versions=_RUNTIME_VERSIONS,
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


def _query(**overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "operation": "all_reduce",
        "dtype": "half",
        "world_size": 4,
        "elements": 32768,
    }
    query.update(overrides)
    return query


def _request(
    *,
    query: dict[str, object] | None = None,
    environment: MeasurementEnvironment | None = None,
) -> MeasurementRequest:
    query = query or _query()
    environment = environment or _environment()
    return MeasurementRequest(
        op_id="pre_dispatch.custom_allreduce",
        key=PerfKey.build(_NAMESPACE, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor=_SEMANTIC_DESCRIPTOR,
        protocol=_protocol(),
    )


def _custom_allreduce_modules():
    runner = importlib.import_module("aiconfigurator.collector.sglang.custom_allreduce")
    adapter = importlib.import_module("aiconfigurator.collector.sglang.custom_allreduce_adapter")
    return runner, adapter


def _raw_result(request: MeasurementRequest) -> dict[str, object]:
    latency_ms = 1.25
    return {
        "latency_ms": latency_ms,
        "energy_wms": 125.0,
        "samples_ms": (1.2, latency_ms, 1.3),
        "statistic": request.protocol.statistic,
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "framework": "SGLang",
            "version": "0.5.10",
            "device": "NVIDIA GB200",
            "op_name": "all_reduce",
            "kernel_source": "SGLang_CustomAllReduce_graph",
            "allreduce_dtype": "half",
            "num_gpus": 4,
            "message_size": 32768,
            "backend": "sglang_graph",
            "latency": latency_ms,
        },
        "provenance": {
            "framework": "SGLang",
            "framework_version": "0.5.10",
            "kernel_source": "SGLang_CustomAllReduce_graph",
            "device": "NVIDIA GB200",
            "runtime": "persistent_sglang_custom_allreduce",
            "used_cuda_graph": True,
            "throttled": False,
            "world_size": 4,
            "rank_pids": [101, 102, 103, 104],
            "model_artifact": _MODEL_ARTIFACT,
        },
    }


def test_packaged_and_source_registries_share_one_custom_allreduce_route() -> None:
    packaged = importlib.import_module("aiconfigurator.collector.sglang.registry")
    from collector.sglang.registry import REGISTRY as SOURCE_REGISTRY

    lazy = packaged.CUSTOM_ALLREDUCE_LAZY_SPEC
    entry = next(entry for entry in packaged.SGLANG_LAZY_REGISTRY if entry.op == "custom_allreduce")
    source_entry = next(entry for entry in SOURCE_REGISTRY if entry.op == "custom_allreduce")
    routes = LazyAdapterIndex.from_registries({"sglang": packaged.SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )

    assert entry.perf_filename is PerfFile.CUSTOM_ALLREDUCE
    assert entry.lazy is lazy
    assert source_entry.lazy is lazy
    assert len(routes) == 1
    assert routes[0].lazy is lazy
    assert routes[0].collector_module == "aiconfigurator.collector.sglang.custom_allreduce"
    assert lazy.run_func == "run_custom_allreduce_case"
    assert lazy.adapter_module == "aiconfigurator.collector.sglang.custom_allreduce_adapter"
    assert lazy.case_func == "custom_allreduce_request_to_case"
    assert lazy.resource_func == "custom_allreduce_resource_for_request"
    assert lazy.result_func == "custom_allreduce_result_to_record"
    assert lazy.protocol_revision == "cuda-event-samples-v1"
    assert lazy.timer == "cuda_event"
    assert lazy.tuning_revision == "sglang-custom-allreduce-v1"


def test_canonical_case_translation_and_world4_nvlink_resource_contract() -> None:
    _, adapter = _custom_allreduce_modules()
    request = _request()

    case = adapter.custom_allreduce_request_to_case(request)

    assert case == {
        "dtype": "half",
        "world_size": 4,
        "element_count": 32768,
    }
    assert adapter.custom_allreduce_resource_for_request(request, case) == ResourceContract(
        gpu_count=4,
        fabric=FabricRequirement.NVLINK,
        reserve_fabric_domain=True,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dtype": "fp8"}, "half|dtype"),
        ({"world_size": 2}, "world.?size|four|4"),
        ({"elements": 32769}, "multiple|16|element"),
        ({"elements": 8_388_608}, "8 MiB|maximum|element"),
    ],
)
def test_custom_allreduce_domain_validation_precedes_resource_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    message: str,
) -> None:
    packaged = importlib.import_module("aiconfigurator.collector.sglang.registry")
    adapter = importlib.import_module("aiconfigurator.collector.sglang.custom_allreduce_adapter")
    resource_calls: list[object] = []
    original_resource = adapter.custom_allreduce_resource_for_request

    def _tracked_resource(request: MeasurementRequest, case: dict[str, object]) -> ResourceContract:
        resource_calls.append((request, case))
        return original_resource(request, case)

    monkeypatch.setattr(adapter, "custom_allreduce_resource_for_request", _tracked_resource)
    route = LazyAdapterIndex.from_registries({"sglang": packaged.SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )[0]

    with pytest.raises(ValueError, match=message):
        route.prepare(_request(query=_query(**overrides)))
    assert resource_calls == []


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("topology_schema", ""),
        ("topology_schema", "   "),
        ("topology_schema", 7),
        ("topology_fingerprint", ""),
        ("topology_fingerprint", "   "),
        ("topology_fingerprint", 7),
    ],
)
def test_custom_allreduce_rejects_invalid_topology_identity_before_resource_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    packaged = importlib.import_module("aiconfigurator.collector.sglang.registry")
    adapter = importlib.import_module("aiconfigurator.collector.sglang.custom_allreduce_adapter")
    resource_calls: list[object] = []
    original_resource = adapter.custom_allreduce_resource_for_request

    def _tracked_resource(request: MeasurementRequest, case: dict[str, object]) -> ResourceContract:
        resource_calls.append((request, case))
        return original_resource(request, case)

    monkeypatch.setattr(adapter, "custom_allreduce_resource_for_request", _tracked_resource)
    route = LazyAdapterIndex.from_registries({"sglang": packaged.SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )[0]
    environment = replace(_environment(), **{field: value})

    with pytest.raises(ValueError, match="topology identity"):
        route.prepare(_request(environment=environment))
    assert resource_calls == []


def test_result_samples_protocol_provenance_and_key_round_trip() -> None:
    _, adapter = _custom_allreduce_modules()
    request = _request()
    case = adapter.custom_allreduce_request_to_case(request)

    record = adapter.custom_allreduce_result_to_record(request, case, _raw_result(request))
    emitted_query = {
        "operation": record.perf_row["op_name"],
        "dtype": record.perf_row["allreduce_dtype"],
        "world_size": record.perf_row["num_gpus"],
        "elements": record.perf_row["message_size"],
    }

    assert record.key == request.key
    assert record.protocol == request.protocol
    assert record.samples_ms == (1.2, 1.25, 1.3)
    assert emitted_query == request.query
    assert record.provenance["runtime"] == "persistent_sglang_custom_allreduce"
    assert record.provenance["used_cuda_graph"] is True
    assert record.provenance["throttled"] is False
    assert record.provenance["world_size"] == 4
    assert record.provenance["rank_pids"] == (101, 102, 103, 104)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.__setitem__("samples_ms", (1.0, 2.0)), "sample"),
        (lambda raw: raw.__setitem__("protocol_digest", "wrong"), "protocol"),
        (lambda raw: raw["provenance"].__setitem__("used_cuda_graph", False), "CUDA Graph|graph"),
        (lambda raw: raw["provenance"].__setitem__("throttled", True), "throttled"),
        (lambda raw: raw["perf_row"].__setitem__("num_gpus", 2), "identity|world|GPU"),
    ],
)
def test_result_validation_fails_closed(
    mutation,
    message: str,
) -> None:
    _, adapter = _custom_allreduce_modules()
    request = _request()
    case = adapter.custom_allreduce_request_to_case(request)
    raw = copy.deepcopy(_raw_result(request))
    mutation(raw)

    with pytest.raises((TypeError, ValueError), match=message):
        adapter.custom_allreduce_result_to_record(request, case, raw)


def test_exact_runner_is_import_light_and_has_no_offline_output_api(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, _ = _custom_allreduce_modules()

    assert runner.get_custom_allreduce_test_cases() == ()
    assert {
        "output_path",
        "perf_filename",
        "test_range",
        "use_slurm",
        "rank",
    }.isdisjoint(inspect.signature(runner.run_custom_allreduce_case).parameters)
    source = Path(runner.__file__).read_text()
    assert "log_perf" not in source
    assert "collect_all_reduce" not in source
    assert "subprocess" not in source

    module_name = "aiconfigurator.collector.sglang.custom_allreduce"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if (
            name == "torch"
            or name.startswith("torch.")
            or name == "sglang"
            or name.startswith("sglang.")
            or name == "cuda"
            or name.startswith("cuda.")
        ):
            raise AssertionError(f"heavy import at module import time: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    importlib.import_module(module_name)
