# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins
import errno
import importlib
import inspect
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.registry_types import PerfFile
from aiconfigurator.collector.sglang import dsv4_attn, dsv4_attn_adapter
from aiconfigurator.collector.sglang.registry import SGLANG_LAZY_REGISTRY
from aiconfigurator.collector.types import FabricRequirement, RawMeasurement, ResourceContract
from aiconfigurator.sdk.resolution.types import (
    MeasurementEnvironment,
    MeasurementProtocol,
    MeasurementRequest,
    PerfKey,
)

pytestmark = pytest.mark.unit

_MODEL_ARTIFACT = "sgl-project/DeepSeek-V4-Flash-FP8"
_ARCHITECTURE = "DeepseekV4ForCausalLM"
_CANONICAL_NUM_HEADS = 16
_PADDED_NUM_HEADS = 64
_TP_SIZE = 4
_MODEL_WEIGHT_GENERATOR = "proper-normal-v1"
_MODEL_WEIGHT_STD = 0.05
_MODEL_WEIGHT_SEED = 1234
_RUNTIME_VERSIONS = {
    "cuda": "13.0",
    "model_profile": "dsv4-v1.2",
    "sglang": "0.5.10",
}
_PROFILE_COMPATIBILITY = {
    "model_artifact": _MODEL_ARTIFACT,
    "serving_mode": "aggregated",
    "tp_size": _TP_SIZE,
    "attention_dp_size": 1,
    "cp_size": 1,
    "pp_size": 1,
    "moe_tp_size": 1,
    "moe_ep_size": 4,
    "nextn": 0,
}
_SEMANTIC_DESCRIPTOR = {
    "full_module": True,
    "tensor_generator": "normal-v1",
    "seed": 0,
    "tp_simulation": "single-gpu-tp4",
    "canonical_num_heads": _CANONICAL_NUM_HEADS,
    "padded_num_heads": _PADDED_NUM_HEADS,
}


@dataclass(frozen=True, slots=True)
class _Route:
    op: str
    namespace: str
    mode: str
    attn_kind: str
    compress_ratio: int


_ROUTES = (
    _Route(
        "dsv4_csa_context_module",
        f"{PerfFile.DSV4_CSA_CONTEXT_MODULE}/v1",
        "context",
        "csa",
        4,
    ),
    _Route(
        "dsv4_hca_context_module",
        f"{PerfFile.DSV4_HCA_CONTEXT_MODULE}/v1",
        "context",
        "hca",
        128,
    ),
    _Route(
        "dsv4_csa_generation_module",
        f"{PerfFile.DSV4_CSA_GENERATION_MODULE}/v1",
        "generation",
        "csa",
        4,
    ),
    _Route(
        "dsv4_hca_generation_module",
        f"{PerfFile.DSV4_HCA_GENERATION_MODULE}/v1",
        "generation",
        "hca",
        128,
    ),
)


def _environment(**overrides: Any) -> MeasurementEnvironment:
    values: dict[str, Any] = {
        "system": "gb200",
        "backend": "sglang",
        "backend_version": "0.5.10",
        "gpu_class": "NVIDIA GB200",
        "runtime_versions": _RUNTIME_VERSIONS,
        "topology_schema": "nvidia-smi-v1",
        "topology_fingerprint": "gb200-nvlink4",
        "profile_compatibility": _PROFILE_COMPATIBILITY,
    }
    values.update(overrides)
    return MeasurementEnvironment(**values)


def _protocol() -> MeasurementProtocol:
    return MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=2,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
    )


def _query(route: _Route, **overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "tp_size": _TP_SIZE,
        "num_heads": _CANONICAL_NUM_HEADS,
        "compress_ratio": route.compress_ratio,
        "batch_size": 2 if route.mode == "context" else 8,
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
    }
    if route.mode == "context":
        query.update(
            {
                "prefix_length": 64,
                "sequence_length": 128,
                "mla_dtype": "bfloat16",
            }
        )
    else:
        query["sequence_length"] = 4096
    query.update(overrides)
    return query


def _request(
    route: _Route,
    *,
    query: dict[str, object] | None = None,
    namespace: str | None = None,
    environment: MeasurementEnvironment | None = None,
    semantic_descriptor: dict[str, object] | None = None,
) -> MeasurementRequest:
    query = query or _query(route)
    environment = environment or _environment()
    return MeasurementRequest(
        op_id=route.op,
        key=PerfKey.build(namespace or route.namespace, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor=semantic_descriptor or _SEMANTIC_DESCRIPTOR,
        protocol=_protocol(),
    )


def _expected_case(route: _Route, query: Mapping[str, object]) -> dict[str, object]:
    case = dict(query)
    case["mode"] = route.mode
    case["attn_kind"] = route.attn_kind
    case["canonical_num_heads"] = case.pop("num_heads")
    case["num_heads"] = _PADDED_NUM_HEADS
    case.setdefault("mla_dtype", "bfloat16")
    if route.mode == "context":
        case["isl"] = case.pop("sequence_length")
        case["prefix"] = case.pop("prefix_length")
    else:
        case["s_total"] = case.pop("sequence_length")
    return case


def _perf_row(route: _Route, query: Mapping[str, object], *, latency_ms: float = 1.25) -> dict[str, object]:
    if route.mode == "context":
        isl = int(query["sequence_length"])
        step = int(query["prefix_length"])
    else:
        isl = 1
        step = int(query["sequence_length"]) - 1
    return {
        "model": _MODEL_ARTIFACT,
        "architecture": _ARCHITECTURE,
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "gemm_type": "fp8_block",
        # The TP-simulated FlashMLA kernel is padded to 64 heads.  The adapter
        # must map this persisted runner identity back to rank-local 16 in the
        # canonical PerfKey instead of silently conflating the two axes.
        "num_heads": _PADDED_NUM_HEADS,
        "batch_size": query["batch_size"],
        "isl": isl,
        "tp_size": _TP_SIZE,
        "step": step,
        "compress_ratio": route.compress_ratio,
        "latency": latency_ms,
    }


def _raw_result(route: _Route, request: MeasurementRequest) -> dict[str, object]:
    latency_ms = 1.25
    return {
        "latency_ms": latency_ms,
        "energy_wms": 125.0,
        "samples_ms": (1.2, latency_ms, 1.3),
        "statistic": request.protocol.statistic,
        "protocol_digest": request.protocol.digest,
        "perf_row": _perf_row(route, request.query, latency_ms=latency_ms),
        "provenance": {
            "framework": "SGLang",
            "framework_version": "0.5.10",
            "kernel_source": "compressed_flashmla",
            "device": "NVIDIA GB200",
            "used_cuda_graph": True,
            "throttled": False,
            "model_artifact": _MODEL_ARTIFACT,
            "full_module": True,
            "mode": route.mode,
            "attn_kind": route.attn_kind,
            "tp_simulation": "single-gpu-tp4",
            "canonical_num_heads": _CANONICAL_NUM_HEADS,
            "padded_num_heads": _PADDED_NUM_HEADS,
            "tensor_generator": "normal-v1",
            "seed": 0,
            "model_weight_generator": _MODEL_WEIGHT_GENERATOR,
            "model_weight_std": _MODEL_WEIGHT_STD,
            "model_weight_seed": _MODEL_WEIGHT_SEED,
        },
    }


def _canonical_query_from_row(route: _Route, row: Mapping[str, object]) -> dict[str, object]:
    query: dict[str, object] = {
        "tp_size": int(row["tp_size"]),
        "num_heads": int(row["num_heads"]) // int(row["tp_size"]),
        "compress_ratio": int(row["compress_ratio"]),
        "batch_size": int(row["batch_size"]),
        "kv_cache_dtype": str(row["kv_cache_dtype"]),
        "gemm_type": str(row["gemm_type"]),
    }
    if route.mode == "context":
        query.update(
            {
                "prefix_length": int(row["step"]),
                "sequence_length": int(row["isl"]),
                "mla_dtype": str(row["mla_dtype"]),
            }
        )
    else:
        query["sequence_length"] = int(row["isl"]) + int(row["step"])
    return query


def test_four_dsv4_attention_namespaces_share_one_packaged_exact_adapter() -> None:
    from collector.sglang.registry import REGISTRY as SOURCE_REGISTRY

    index = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY})
    for route in _ROUTES:
        source_entry = next(entry for entry in SOURCE_REGISTRY if entry.op == route.op)
        resolved = index.routes_for((route.namespace, "sglang", "0.5.10"))

        assert len(resolved) == 1
        assert source_entry.lazy is resolved[0].lazy
        assert resolved[0].collector_module == "aiconfigurator.collector.sglang.dsv4_attn"
        assert resolved[0].lazy.run_func == "run_dsv4_attn_case"
        assert resolved[0].lazy.adapter_module == "aiconfigurator.collector.sglang.dsv4_attn_adapter"
        assert resolved[0].lazy.case_func == "dsv4_attn_request_to_case"
        assert resolved[0].lazy.result_func == "dsv4_attn_result_to_record"
        assert resolved[0].lazy.resource_func == "dsv4_attn_resource_for_request"


@pytest.mark.parametrize("route", _ROUTES, ids=lambda route: f"{route.attn_kind}-{route.mode}")
def test_attention_request_round_trips_through_case_padded_row_and_same_key(route: _Route) -> None:
    request = _request(route)
    expected_query = _query(route)

    assert request.key == PerfKey.build(route.namespace, expected_query, request.environment)
    case = dsv4_attn_adapter.dsv4_attn_request_to_case(request)
    assert case == _expected_case(route, expected_query)
    assert case["canonical_num_heads"] == _CANONICAL_NUM_HEADS
    assert case["num_heads"] == _PADDED_NUM_HEADS
    assert dsv4_attn_adapter.dsv4_attn_resource_for_request(request, case) == ResourceContract(
        gpu_count=1,
        fabric=FabricRequirement.NONE,
    )

    record = dsv4_attn_adapter.dsv4_attn_result_to_record(request, case, _raw_result(route, request))
    assert record.perf_row["num_heads"] == _PADDED_NUM_HEADS
    assert _canonical_query_from_row(route, record.perf_row) == expected_query
    assert (
        record.key
        == request.key
        == PerfKey.build(
            route.namespace,
            _canonical_query_from_row(route, record.perf_row),
            request.environment,
        )
    )


def test_attention_capability_allows_unrelated_runtime_inventory_entries() -> None:
    route = _ROUTES[0]
    environment = replace(
        _environment(),
        runtime_versions={**_RUNTIME_VERSIONS, "tensorrt_llm": "1.3.0rc10"},
    )

    assert dsv4_attn_adapter.dsv4_attn_request_to_case(_request(route, environment=environment)) == _expected_case(
        route,
        _query(route),
    )


@pytest.mark.parametrize(
    ("mismatch", "request_factory"),
    [
        ("namespace", lambda route: _request(route, namespace=f"{PerfFile.MHC_MODULE}/v1")),
        ("system", lambda route: _request(route, environment=replace(_environment(), system="h100_sxm"))),
        ("gpu_class", lambda route: _request(route, environment=replace(_environment(), gpu_class="NVIDIA H100"))),
        (
            "backend_version",
            lambda route: _request(
                route,
                environment=replace(
                    _environment(),
                    backend_version="0.5.11",
                    runtime_versions={**_RUNTIME_VERSIONS, "sglang": "0.5.11"},
                ),
            ),
        ),
        (
            "profile",
            lambda route: _request(
                route,
                environment=replace(
                    _environment(),
                    profile_compatibility={**_PROFILE_COMPATIBILITY, "tp_size": 8},
                ),
            ),
        ),
        (
            "descriptor",
            lambda route: _request(
                route,
                semantic_descriptor={**_SEMANTIC_DESCRIPTOR, "padded_num_heads": _CANONICAL_NUM_HEADS},
            ),
        ),
        ("tp_size", lambda route: _request(route, query=_query(route, tp_size=2))),
        ("num_heads", lambda route: _request(route, query=_query(route, num_heads=_PADDED_NUM_HEADS))),
        ("compress_ratio", lambda route: _request(route, query=_query(route, compress_ratio=7))),
        ("kv_cache_dtype", lambda route: _request(route, query=_query(route, kv_cache_dtype="bfloat16"))),
        ("gemm_type", lambda route: _request(route, query=_query(route, gemm_type="bfloat16"))),
        (
            "mla_dtype",
            lambda route: _request(route, query=_query(route, mla_dtype="fp8"))
            if route.mode == "context"
            else _request(route, query={**_query(route), "mla_dtype": "fp8"}),
        ),
        ("query_fields", lambda route: _request(route, query={**_query(route), "scale_factor": 43})),
    ],
)
def test_attention_capability_mismatch_fails_before_resource_acquisition(
    mismatch: str,
    request_factory,
) -> None:
    route = _ROUTES[0]
    request = request_factory(route)
    resource_calls: list[object] = []

    def _resource(request, case):
        resource_calls.append((request, case))
        return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    with pytest.raises((TypeError, ValueError)):
        case = dsv4_attn_adapter.dsv4_attn_request_to_case(request)
        _resource(request, case)
    assert resource_calls == [], mismatch


@pytest.mark.parametrize(
    ("route", "request_factory", "error_match"),
    [
        pytest.param(
            _ROUTES[0],
            lambda route: replace(
                _request(route),
                protocol=replace(_protocol(), samples=2),
            ),
            "samples",
            id="samples-below-three",
        ),
        pytest.param(
            _ROUTES[2],
            lambda route: _request(route, query=_query(route, sequence_length=1)),
            "sequence_length",
            id="generation-sequence-length-one",
        ),
    ],
)
def test_attention_route_prepare_rejects_before_resource_acquisition(
    route: _Route,
    request_factory,
    error_match: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_calls: list[tuple[MeasurementRequest, Mapping[str, Any]]] = []
    real_resource = dsv4_attn_adapter.dsv4_attn_resource_for_request

    def tracked_resource(request: MeasurementRequest, case: Mapping[str, Any]) -> ResourceContract:
        resource_calls.append((request, case))
        return real_resource(request, case)

    monkeypatch.setattr(dsv4_attn_adapter, "dsv4_attn_resource_for_request", tracked_resource)
    index = LazyAdapterIndex.from_registries({"sglang": SGLANG_LAZY_REGISTRY})
    resolved = index.routes_for((route.namespace, "sglang", "0.5.10"))
    assert len(resolved) == 1

    with pytest.raises(ValueError, match=error_match):
        resolved[0].prepare(request_factory(route))

    assert resource_calls == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw["perf_row"].__setitem__("num_heads", _CANONICAL_NUM_HEADS),
        lambda raw: raw["perf_row"].__setitem__("tp_size", 2),
        lambda raw: raw["perf_row"].__setitem__("compress_ratio", 128),
        lambda raw: raw["perf_row"].__setitem__("gemm_type", "bfloat16"),
        lambda raw: raw["provenance"].__setitem__("padded_num_heads", _CANONICAL_NUM_HEADS),
        lambda raw: raw["provenance"].__setitem__("used_cuda_graph", False),
        lambda raw: raw["provenance"].__setitem__("framework_version", "0.5.11"),
        lambda raw: raw["provenance"].__setitem__("model_artifact", "other/model"),
        lambda raw: raw["provenance"].__setitem__("model_weight_generator", "dummy-load"),
        lambda raw: raw["provenance"].__setitem__("model_weight_std", 0.0),
        lambda raw: raw["provenance"].__setitem__("model_weight_seed", 0),
    ],
)
def test_attention_result_validation_fails_closed(mutation) -> None:
    route = _ROUTES[0]
    request = _request(route)
    case = dsv4_attn_adapter.dsv4_attn_request_to_case(request)
    raw = _raw_result(route, request)
    mutation(raw)

    with pytest.raises((TypeError, ValueError)):
        dsv4_attn_adapter.dsv4_attn_result_to_record(request, case, raw)


@pytest.mark.parametrize("route", _ROUTES, ids=lambda route: f"{route.attn_kind}-{route.mode}")
def test_exact_attention_runner_emits_one_padded_full_module_row_without_writing(
    route: _Route,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    request = _request(route)
    case = _expected_case(route, request.query)
    calls: list[tuple[object, ...]] = []

    def _prepare_dsv4_attn_case(**kwargs):
        calls.append(("prepare", kwargs))
        return dsv4_attn.PreparedDsv4AttentionCase(
            kernel_func=lambda: calls.append(("kernel",)),
            framework_version="0.5.10",
            device_name="NVIDIA GB200",
            device=object(),
            architecture=_ARCHITECTURE,
            model_artifact=_MODEL_ARTIFACT,
            mode=route.mode,
            attn_kind=route.attn_kind,
            compress_ratio=route.compress_ratio,
            tp_size=_TP_SIZE,
            canonical_num_heads=_CANONICAL_NUM_HEADS,
            padded_num_heads=_PADDED_NUM_HEADS,
            mla_dtype="bfloat16",
            kv_cache_dtype="fp8",
            gemm_type="fp8_block",
        )

    @contextmanager
    def _benchmark(**kwargs):
        calls.append(("benchmark", kwargs))
        yield {
            "latency_ms": 1.25,
            "samples_ms": (1.2, 1.25, 1.3),
            "power_stats": {"power": 100.0, "power_limit": 1200.0},
            "throttled": False,
            "used_cuda_graph": True,
            "num_runs_executed": 3,
        }

    monkeypatch.setattr(dsv4_attn, "_prepare_dsv4_attn_case", _prepare_dsv4_attn_case)
    monkeypatch.setattr(dsv4_attn, "benchmark_with_power", _benchmark)
    monkeypatch.chdir(tmp_path)

    result = dsv4_attn.run_dsv4_attn_case(**case, protocol=request.protocol)

    assert isinstance(result, RawMeasurement)
    assert result.latency_ms == 1.25
    assert result.samples_ms == (1.2, 1.25, 1.3)
    assert result.energy_wms == 125.0
    assert result.protocol_digest == request.protocol.digest
    assert result.perf_row == _perf_row(route, request.query)
    assert result.provenance == _raw_result(route, request)["provenance"]
    prepare_kwargs = next(call[1] for call in calls if call[0] == "prepare")
    assert prepare_kwargs["canonical_num_heads"] == _CANONICAL_NUM_HEADS
    assert prepare_kwargs["num_heads"] == _PADDED_NUM_HEADS
    benchmark_kwargs = next(call[1] for call in calls if call[0] == "benchmark")
    assert benchmark_kwargs["num_warmups"] == 2
    assert benchmark_kwargs["num_runs"] == 3
    assert benchmark_kwargs["repeat_n"] == 1
    assert benchmark_kwargs["return_samples"] is True
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert benchmark_kwargs["use_cuda_graph"] is True
    assert not tuple(tmp_path.iterdir())


def test_attention_runner_is_exact_only_import_light_and_has_no_offline_output_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert dsv4_attn.get_dsv4_attn_test_cases() == ()
    assert {
        "output_path",
        "perf_filename",
        "batch_sizes",
        "seq_lens",
        "prefix_lens",
        "num_heads_cases",
    }.isdisjoint(inspect.signature(dsv4_attn.run_dsv4_attn_case).parameters)
    source = Path(dsv4_attn.__file__).read_text()
    assert "log_perf" not in source
    assert "from collector" not in source
    assert "import collector" not in source

    module_name = "aiconfigurator.collector.sglang.dsv4_attn"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name == "sglang" or name.startswith("sglang."):
            raise AssertionError(f"heavy import at module import time: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    importlib.import_module(module_name)


def test_model_runner_construction_retries_with_a_fresh_nccl_port_after_collision() -> None:
    ports = iter((45101, 45102))
    attempted_ports: list[int] = []

    def model_runner_factory(**kwargs):
        attempted_ports.append(kwargs["nccl_port"])
        if len(attempted_ports) == 1:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        return object()

    result = dsv4_attn._construct_model_runner(
        model_runner_factory,
        {"model_config": object()},
        port_factory=lambda: next(ports),
    )

    assert result is not None
    assert attempted_ports == [45101, 45102]


def test_model_runner_construction_does_not_retry_unrelated_failures() -> None:
    port_calls = 0

    def port_factory() -> int:
        nonlocal port_calls
        port_calls += 1
        return 45101

    def model_runner_factory(**kwargs):
        raise RuntimeError("model initialization failed")

    with pytest.raises(RuntimeError, match="model initialization failed"):
        dsv4_attn._construct_model_runner(
            model_runner_factory,
            {"model_config": object()},
            port_factory=port_factory,
        )

    assert port_calls == 1
