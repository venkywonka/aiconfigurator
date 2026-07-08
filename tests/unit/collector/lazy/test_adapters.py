# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for exact lazy-adapter reverse lookup and preparation."""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from types import ModuleType
from typing import Any

import pytest

from aiconfigurator.collector.preflight import (
    CapabilityPreflightError,
    OperationCapability,
    OperationKind,
    preflight_capabilities,
)
from aiconfigurator.collector.registry_types import OpEntry, PerfFile, VersionRoute
from aiconfigurator.collector.types import FabricRequirement, LazyOpEntry, ResourceContract
from aiconfigurator.collector.version_resolver import build_collections
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRecord,
    MeasurementRequest,
    PerfKey,
)

pytestmark = pytest.mark.unit

_NAMESPACE = "gemm_perf.txt/v1"
_ADAPTER_MODULE = "aiconfigurator.collector.testing.fake_adapter"
_RUN_MODULE = "aiconfigurator.collector.testing.forbidden_heavy_runtime"


def _adapter_api():
    return importlib.import_module("aiconfigurator.collector.adapters")


def _lazy_entry(
    *,
    adapter_module: str = _ADAPTER_MODULE,
    run_module: str = _RUN_MODULE,
) -> LazyOpEntry:
    return LazyOpEntry(
        namespace=_NAMESPACE,
        run_module=run_module,
        run_func="run_case",
        adapter_module=adapter_module,
        case_func="request_to_case",
        result_func="result_to_record",
        resource_func="resource_for_request",
        protocol_revision="cuda-event-v1",
        timer="cuda_event",
        tuning_revision="fake-v1",
    )


def _entry(
    *,
    op: str = "gemm",
    lazy: LazyOpEntry | None = None,
    min_version: str = "0.5.10",
    collector_module: str = "collector.sglang.collect_gemm",
) -> OpEntry:
    return OpEntry(
        op=op,
        get_func="get_gemm_test_cases",
        run_func="run_gemm",
        perf_filename=PerfFile.GEMM,
        versions=(VersionRoute(min_version, collector_module),),
        lazy=_lazy_entry() if lazy is None else lazy,
    )


def _environment(**overrides: Any) -> MeasurementEnvironment:
    values = {
        "system": "gb200_nvlink4",
        "backend": "sglang",
        "backend_version": "0.5.10",
        "gpu_class": "NVIDIA GB200",
        "runtime_versions": {"cuda": "13.0", "model_profile": "dsv4-v1.2"},
        "topology_schema": "nvidia-smi-v1",
        "topology_fingerprint": "nvlink4-fingerprint",
    }
    values.update(overrides)
    return MeasurementEnvironment(**values)


def _protocol(**overrides: Any) -> MeasurementProtocol:
    values = {
        "revision": "cuda-event-v1",
        "warmups": 2,
        "samples": 3,
        "timer": "cuda_event",
        "tuning_revision": "fake-v1",
    }
    values.update(overrides)
    return MeasurementProtocol(**values)


def _request(
    *,
    environment: MeasurementEnvironment | None = None,
    protocol: MeasurementProtocol | None = None,
) -> MeasurementRequest:
    environment = environment or _environment()
    query = {"m": 128, "n": 256}
    return MeasurementRequest(
        op_id="gemm-0",
        key=PerfKey.build(_NAMESPACE, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor={"consumer": "shared-gemm"},
        protocol=protocol or _protocol(),
    )


def _install_adapter(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    module = ModuleType(_ADAPTER_MODULE)

    def request_to_case(request: MeasurementRequest) -> dict[str, int]:
        events.append("case")
        environment = request.environment
        expected = (
            environment.system == "gb200_nvlink4"
            and environment.gpu_class == "NVIDIA GB200"
            and environment.topology_schema == "nvidia-smi-v1"
            and environment.topology_fingerprint == "nvlink4-fingerprint"
            and environment.runtime_versions.get("model_profile") == "dsv4-v1.2"
        )
        if not expected:
            raise ValueError("environment is outside the adapter capability envelope")
        return {"rows": request.query["m"], "columns": request.query["n"]}

    def resource_for_request(
        request: MeasurementRequest,
        case: dict[str, int],
    ) -> ResourceContract:
        del request, case
        events.append("resource")
        return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    def result_to_record(
        request: MeasurementRequest,
        case: dict[str, int],
        raw_result: dict[str, Any],
    ) -> MeasurementRecord:
        events.append("record")
        key = request.key
        if raw_result.get("wrong_key"):
            key = PerfKey.build(_NAMESPACE, {"m": 1, "n": 1}, request.environment)
        return MeasurementRecord.valid(
            key=key,
            latency_ms=raw_result["latency_ms"],
            energy_wms=0.0,
            samples_ms=tuple(raw_result["samples_ms"]),
            protocol=request.protocol,
            perf_row={"m": case["rows"], "n": case["columns"]},
            provenance={"runner": "fake"},
        )

    module.request_to_case = request_to_case
    module.resource_for_request = resource_for_request
    module.result_to_record = result_to_record
    monkeypatch.setitem(sys.modules, _ADAPTER_MODULE, module)


def _index(*entries: OpEntry):
    adapters = _adapter_api()
    return adapters.LazyAdapterIndex.from_registries({"sglang": entries})


def test_reverse_index_resolves_exact_route_and_each_wrong_component_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_adapter(monkeypatch, [])
    index = _index(_entry())
    identity = (_NAMESPACE, "sglang", "0.5.10")

    routes = index.routes_for(identity)

    assert len(routes) == 1
    assert routes[0].entry.op == "gemm"
    assert routes[0].lazy.namespace == _NAMESPACE
    assert routes[0].collector_module == "collector.sglang.collect_gemm"
    assert index.routes_for(("other_perf.txt/v1", identity[1], identity[2])) == ()
    assert index.routes_for((identity[0], "trtllm", identity[2])) == ()
    assert index.routes_for((identity[0], identity[1], "0.5.9")) == ()


def test_missing_and_ambiguous_routes_remain_visible_to_capability_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_adapter(monkeypatch, [])
    capability = OperationCapability("gemm", OperationKind.MEASURED, _NAMESPACE)

    for entries, expected_count in (((), 0), ((_entry(), _entry(op="gemm-copy")), 2)):
        index = _index(*entries)
        with pytest.raises(CapabilityPreflightError, match=rf"expected one route.*got {expected_count}"):
            preflight_capabilities(
                (capability,),
                backend="sglang",
                backend_version="0.5.10",
                routes_for=index.routes_for,
            )


def test_version_floor_eligibility_reuses_the_existing_collector_route_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_adapter(monkeypatch, [])
    index = _index(_entry(min_version="0.5.10"))

    assert index.routes_for((_NAMESPACE, "sglang", "0.5.9")) == ()
    assert index.routes_for((_NAMESPACE, "sglang", "0.5.10"))[0].collector_module.endswith("collect_gemm")
    assert index.routes_for((_NAMESPACE, "sglang", "0.5.11"))[0].collector_module.endswith("collect_gemm")


def test_protocol_identity_is_checked_before_case_or_resource_but_sampling_counts_are_flexible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_adapter(monkeypatch, events)
    route = _index(_entry()).routes_for((_NAMESPACE, "sglang", "0.5.10"))[0]

    for field, value in (
        ("revision", "other-protocol"),
        ("timer", "host-clock"),
        ("tuning_revision", "other-tuning"),
    ):
        request = _request(protocol=replace(_protocol(), **{field: value}))
        with pytest.raises(ValueError, match=field):
            route.prepare(request)
        assert events == []

    request = _request(protocol=_protocol(warmups=7, samples=5))
    prepared = route.prepare(request)
    assert prepared.request is request
    assert prepared.case == {"rows": 128, "columns": 256}
    assert prepared.contract == ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)
    assert events == ["case", "resource"]


def test_environment_capability_mismatches_fail_before_resource_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_adapter(monkeypatch, events)
    route = _index(_entry()).routes_for((_NAMESPACE, "sglang", "0.5.10"))[0]
    mismatches = (
        _environment(backend="trtllm"),
        _environment(backend_version="0.5.11"),
        _environment(system="other-system"),
        _environment(gpu_class="NVIDIA H100"),
        _environment(topology_fingerprint="other-fabric"),
        _environment(runtime_versions={"cuda": "13.0", "model_profile": "other-profile"}),
    )

    for environment in mismatches:
        events.clear()
        with pytest.raises(ValueError, match=r"environment|backend"):
            route.prepare(_request(environment=environment))
        assert "resource" not in events


def test_key_case_record_round_trip_preserves_the_identical_perf_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_adapter(monkeypatch, events)
    route = _index(_entry()).routes_for((_NAMESPACE, "sglang", "0.5.10"))[0]
    request = _request()
    prepared = route.prepare(request)
    raw_result = {"latency_ms": 1.2, "samples_ms": (1.1, 1.2, 1.3)}

    record = route.record(prepared, raw_result)

    assert record.key == request.key
    assert record.perf_row == {"m": 128, "n": 256}
    assert events == ["case", "resource", "record"]
    with pytest.raises(ValueError, match=r"PerfKey|key"):
        route.record(prepared, raw_result | {"wrong_key": True})


def test_parent_loads_only_installable_namespaced_lightweight_adapter_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_adapter(monkeypatch, [])
    sys.modules.pop(_RUN_MODULE, None)

    assert _index(_entry()).routes_for((_NAMESPACE, "sglang", "0.5.10"))
    assert _ADAPTER_MODULE in sys.modules
    assert _RUN_MODULE not in sys.modules

    for field, root_relative_module in (
        ("adapter_module", "collector.fake_adapter"),
        ("run_module", "collector.fake_runner"),
    ):
        root_relative = _entry(lazy=_lazy_entry(**{field: root_relative_module}))
        with pytest.raises(ValueError, match=r"namespaced|aiconfigurator"):
            _index(root_relative).routes_for((_NAMESPACE, "sglang", "0.5.10"))
        assert root_relative_module not in sys.modules


def test_entries_without_lazy_metadata_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_adapter(monkeypatch, [])
    offline_only = OpEntry(
        "gemm",
        "get_gemm_test_cases",
        "run_gemm",
        PerfFile.GEMM,
        "collector.sglang.collect_gemm",
    )

    assert _index(offline_only).routes_for((_NAMESPACE, "sglang", "0.5.10")) == ()


def test_legacy_offline_registry_output_and_positional_construction_remain_unchanged() -> None:
    positional = OpEntry(
        "offline-gemm",
        "get_gemm_test_cases",
        "run_gemm",
        PerfFile.GEMM,
        "collector.sglang.collect_gemm",
    )
    lazy = _entry()

    assert positional.lazy is None
    assert build_collections([positional, lazy], "sglang", "0.5.10") == [
        {
            "name": "sglang",
            "type": "offline-gemm",
            "module": "collector.sglang.collect_gemm",
            "get_func": "get_gemm_test_cases",
            "run_func": "run_gemm",
            "perf_filename": PerfFile.GEMM,
        },
        {
            "name": "sglang",
            "type": "gemm",
            "module": "collector.sglang.collect_gemm",
            "get_func": "get_gemm_test_cases",
            "run_func": "run_gemm",
            "perf_filename": PerfFile.GEMM,
        },
    ]
