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
from types import ModuleType, SimpleNamespace
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
    ProtocolMismatchError,
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


def test_attention_rejects_malformed_sampling_protocol_as_protocol_mismatch() -> None:
    route = _ROUTES[0]
    request = replace(_request(route), protocol=replace(_protocol(), samples=2))

    with pytest.raises(ProtocolMismatchError, match="samples must be at least three"):
        dsv4_attn_adapter.dsv4_attn_request_to_case(request)


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


def _raw_result(
    route: _Route,
    request: MeasurementRequest,
    *,
    framework_version: str = "0.5.10",
) -> dict[str, object]:
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
            "framework_version": framework_version,
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


def test_coherent_release_candidate_round_trips_as_a_distinct_exact_environment() -> None:
    route = _ROUTES[0]
    environment = replace(
        _environment(),
        backend_version="0.5.10rc0",
        runtime_versions={**_RUNTIME_VERSIONS, "sglang": "0.5.10rc0"},
    )
    request = _request(route, environment=environment)
    case = dsv4_attn_adapter.dsv4_attn_request_to_case(request)
    raw_result = _raw_result(route, request, framework_version="0.5.10rc0")

    record = dsv4_attn_adapter.dsv4_attn_result_to_record(request, case, raw_result)

    assert record.key == request.key
    assert request.environment.backend_version == "0.5.10rc0"
    assert request.environment.runtime_versions["sglang"] == "0.5.10rc0"
    assert record.provenance["framework_version"] == "0.5.10rc0"


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
            lambda route: (
                _request(route, query=_query(route, mla_dtype="fp8"))
                if route.mode == "context"
                else _request(route, query={**_query(route), "mla_dtype": "fp8"})
            ),
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
@pytest.mark.parametrize("framework_version", ("0.5.10", "0.5.10rc0"))
def test_exact_attention_runner_emits_one_padded_full_module_row_without_writing(
    route: _Route,
    framework_version: str,
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
            framework_version=framework_version,
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
    assert (
        result.provenance
        == _raw_result(
            route,
            request,
            framework_version=framework_version,
        )["provenance"]
    )
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


def test_attention_runner_accepts_a_broader_offline_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = "deepseek-ai/DeepSeek-V4-Pro"
    protocol = MeasurementProtocol(
        revision="cuda-event-samples-v1",
        warmups=5,
        samples=3,
        statistic="median",
        timer="cuda_event",
        tuning_revision="sglang-dsv4-attn-v1",
    )
    case = {
        "mode": "context",
        "attn_kind": "csa",
        "tp_size": 2,
        "canonical_num_heads": 32,
        "num_heads": 64,
        "compress_ratio": 4,
        "batch_size": 1,
        "mla_dtype": "bfloat16",
        "kv_cache_dtype": "fp8",
        "gemm_type": "bfloat16",
        "isl": 128,
        "prefix": 0,
        "model_path": model_path,
    }

    monkeypatch.setattr(
        dsv4_attn,
        "_prepare_dsv4_attn_case",
        lambda **kwargs: dsv4_attn.PreparedDsv4AttentionCase(
            kernel_func=lambda: None,
            framework_version="0.5.13",
            device_name="NVIDIA B200",
            device=object(),
            architecture=_ARCHITECTURE,
            model_artifact=model_path,
            mode="context",
            attn_kind="csa",
            compress_ratio=4,
            tp_size=2,
            canonical_num_heads=32,
            padded_num_heads=64,
            mla_dtype="bfloat16",
            kv_cache_dtype="fp8",
            gemm_type="bfloat16",
        ),
    )

    @contextmanager
    def _benchmark(**kwargs):
        yield {
            "latency_ms": 2.0,
            "samples_ms": (1.5, 2.0, 2.5),
            "power_stats": None,
            "throttled": False,
            "used_cuda_graph": True,
        }

    monkeypatch.setattr(dsv4_attn, "benchmark_with_power", _benchmark)

    result = dsv4_attn.run_dsv4_attn_case(**case, protocol=protocol)

    assert result.perf_row["model"] == model_path
    assert result.perf_row["tp_size"] == 2
    assert result.perf_row["num_heads"] == 64
    assert result.perf_row["gemm_type"] == "bfloat16"
    assert result.provenance["framework_version"] == "0.5.13"
    assert result.provenance["tp_simulation"] == "single-gpu-tp2"


def test_attention_runner_preserves_canonical_model_identity_for_local_offline_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_model = "/models/DeepSeek-V4-Pro"
    monkeypatch.setattr(
        dsv4_attn,
        "_prepare_dsv4_attn_case",
        lambda **kwargs: dsv4_attn.PreparedDsv4AttentionCase(
            kernel_func=lambda: None,
            framework_version="0.5.13",
            device_name="NVIDIA B200",
            device=object(),
            architecture=_ARCHITECTURE,
            model_artifact=local_model,
            mode="context",
            attn_kind="csa",
            compress_ratio=4,
            tp_size=2,
            canonical_num_heads=32,
            padded_num_heads=64,
            mla_dtype="bfloat16",
            kv_cache_dtype="fp8",
            gemm_type="bfloat16",
        ),
    )

    @contextmanager
    def _benchmark(**kwargs):
        yield {
            "latency_ms": 2.0,
            "samples_ms": (1.5, 2.0, 2.5),
            "power_stats": None,
            "throttled": False,
            "used_cuda_graph": True,
        }

    monkeypatch.setattr(dsv4_attn, "benchmark_with_power", _benchmark)
    result = dsv4_attn.run_dsv4_attn_case(
        mode="context",
        attn_kind="csa",
        tp_size=2,
        canonical_num_heads=32,
        num_heads=64,
        compress_ratio=4,
        batch_size=1,
        mla_dtype="bfloat16",
        kv_cache_dtype="fp8",
        gemm_type="bfloat16",
        isl=128,
        prefix=0,
        protocol=_protocol(),
        model_path=local_model,
    )

    assert result.perf_row["model"] == "deepseek-ai/DeepSeek-V4-Pro"
    assert result.provenance["model_artifact"] == "deepseek-ai/DeepSeek-V4-Pro"


def test_offline_attention_worker_delegates_each_shape_to_the_canonical_runner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from collector.sglang import collect_dsv4_attn

    calls: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []
    runtime = object()
    runtime_events: list[object] = []

    def _run_dsv4_attn_case(**kwargs):
        assert kwargs["runtime"] is runtime
        calls.append(kwargs)
        return RawMeasurement(
            latency_ms=1.0,
            energy_wms=0.0,
            samples_ms=(0.9, 1.0, 1.1),
            statistic="median",
            perf_row={
                "model": kwargs["model_path"],
                "architecture": _ARCHITECTURE,
                "mla_dtype": kwargs["mla_dtype"],
                "kv_cache_dtype": kwargs["kv_cache_dtype"],
                "gemm_type": kwargs["gemm_type"],
                "num_heads": kwargs["num_heads"],
                "batch_size": kwargs["batch_size"],
                "isl": kwargs["isl"],
                "tp_size": kwargs["tp_size"],
                "step": kwargs["prefix"],
                "compress_ratio": kwargs["compress_ratio"],
                "latency": 1.0,
            },
            provenance={
                "framework": "SGLang",
                "framework_version": "0.5.13",
                "kernel_source": "compressed_flashmla",
                "device": "NVIDIA B200",
            },
            protocol_digest=kwargs["protocol"].digest,
        )

    monkeypatch.setattr(collect_dsv4_attn, "_SEQ_LENGTHS", [8, 16])
    monkeypatch.setattr(collect_dsv4_attn, "_PREFIX_LENGTHS", [0, 4])
    monkeypatch.setattr(collect_dsv4_attn, "_filter_pairs", lambda mode, batches, seqs: [(2, sl) for sl in seqs])
    monkeypatch.setattr(collect_dsv4_attn, "_is_valid_shape", lambda mode, bs, sl, prefix: True)
    monkeypatch.setattr(
        collect_dsv4_attn,
        "open_dsv4_attn_runtime",
        lambda **kwargs: runtime_events.append(("open", kwargs)) or runtime,
        raising=False,
    )
    monkeypatch.setattr(
        collect_dsv4_attn,
        "close_dsv4_attn_runtime",
        lambda value: runtime_events.append(("close", value)),
        raising=False,
    )
    monkeypatch.setattr(collect_dsv4_attn, "run_dsv4_attn_case", _run_dsv4_attn_case)
    monkeypatch.setattr(collect_dsv4_attn, "log_perf", lambda **kwargs: logged.append(kwargs))
    perf_filename = tmp_path / "dsv4_csa_context_module_perf.txt"

    collect_dsv4_attn.run_dsv4_attn_worker(
        0,
        2,
        2,
        "fp8",
        "bfloat16",
        "bfloat16",
        "deepseek-ai/DeepSeek-V4-Pro",
        "csa",
        perf_filename=str(perf_filename),
        device="cuda:3",
    )

    assert {(call["isl"], call["prefix"]) for call in calls} == {
        (8, 0),
        (16, 0),
        (8, 4),
        (16, 4),
    }
    assert all(call["tp_size"] == 2 for call in calls)
    assert all(call["canonical_num_heads"] == 32 for call in calls)
    assert all(call["num_heads"] == 64 for call in calls)
    assert all(call["device"] == "cuda:3" for call in calls)
    assert [event[0] for event in runtime_events] == ["open", "close"]
    assert runtime_events[-1] == ("close", runtime)
    assert len(logged) == 4
    assert all(log["perf_filename"] == str(perf_filename) for log in logged)


def test_attention_runner_cleans_prepared_case_when_contract_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _ROUTES[0]
    request = _request(route)
    case = _expected_case(route, request.query)
    case = {
        **case,
        "isl": case.get("isl"),
        "prefix": case.get("prefix"),
        "s_total": case.get("s_total"),
    }
    calls: list[str] = []

    monkeypatch.setattr(
        dsv4_attn,
        "_prepare_dsv4_attn_case",
        lambda **kwargs: dsv4_attn.PreparedDsv4AttentionCase(
            kernel_func=lambda: None,
            framework_version="0.5.11",
            device_name="NVIDIA GB200",
            device=object(),
            architecture=_ARCHITECTURE,
            model_artifact="other/model",
            mode=route.mode,
            attn_kind=route.attn_kind,
            compress_ratio=route.compress_ratio,
            tp_size=_TP_SIZE,
            canonical_num_heads=_CANONICAL_NUM_HEADS,
            padded_num_heads=_PADDED_NUM_HEADS,
            mla_dtype="bfloat16",
            kv_cache_dtype="fp8",
            gemm_type="fp8_block",
            cleanup_func=lambda: calls.append("cleanup"),
        ),
    )
    monkeypatch.setattr(
        dsv4_attn,
        "benchmark_with_power",
        lambda **kwargs: pytest.fail("invalid prepared cases must not benchmark"),
    )

    with pytest.raises(ValueError, match="prepared DSv4 attention case"):
        dsv4_attn.run_dsv4_attn_case(**case, protocol=request.protocol)

    assert calls == ["cleanup"]


def test_attention_runner_preserves_benchmark_failure_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _ROUTES[0]
    request = _request(route)
    case = _expected_case(route, request.query)
    calls: list[str] = []

    def fail_cleanup() -> None:
        calls.append("cleanup")
        raise OSError("cleanup failed")

    monkeypatch.setattr(
        dsv4_attn,
        "_prepare_dsv4_attn_case",
        lambda **kwargs: dsv4_attn.PreparedDsv4AttentionCase(
            kernel_func=lambda: None,
            framework_version="0.5.10rc0",
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
            cleanup_func=fail_cleanup,
        ),
    )

    @contextmanager
    def fail_benchmark(**kwargs):
        raise RuntimeError("benchmark failed")
        yield  # pragma: no cover

    monkeypatch.setattr(dsv4_attn, "benchmark_with_power", fail_benchmark)

    with pytest.raises(RuntimeError, match="benchmark failed"):
        dsv4_attn.run_dsv4_attn_case(**case, protocol=request.protocol)

    assert calls == ["cleanup"]


def test_attention_preparation_cleans_distributed_state_when_model_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _ROUTES[0]
    request = _request(route)
    case = _expected_case(route, request.query)
    case = {
        **case,
        "batch_size": 2,
        "isl": 97,
        "prefix": 0,
        "s_total": case.get("s_total"),
    }
    calls: list[str] = []
    load_kwargs: dict[str, object] = {}

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(empty_cache=lambda: calls.append("empty-cache"))
    torch_module.distributed = SimpleNamespace(is_initialized=lambda: False)
    parallel_state = SimpleNamespace(
        destroy_model_parallel=lambda: calls.append("destroy-model-parallel"),
        destroy_distributed_environment=lambda: calls.append("destroy-distributed-environment"),
    )
    sglang_module = ModuleType("sglang")
    sglang_module.__path__ = []
    srt_module = ModuleType("sglang.srt")
    srt_module.__path__ = []
    distributed_module = ModuleType("sglang.srt.distributed")
    distributed_module.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "sglang", sglang_module)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt_module)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed_module)

    def fail_model_load(*_args, **kwargs):
        load_kwargs.update(kwargs)
        raise RuntimeError("construction failed")

    monkeypatch.setattr(dsv4_attn, "_load_model_runner", fail_model_load)

    with pytest.raises(RuntimeError, match="construction failed"):
        dsv4_attn._prepare_dsv4_attn_case(
            **case,
            device="cuda:0",
            model_path=_MODEL_ARTIFACT,
        )

    assert calls == [
        "destroy-model-parallel",
        "destroy-distributed-environment",
        "empty-cache",
    ]
    assert load_kwargs["required_swa_tokens"] == 512


def test_attention_preparation_preserves_load_error_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _ROUTES[0]
    request = _request(route)
    case = _expected_case(route, request.query)
    case = {
        **case,
        "isl": case.get("isl"),
        "prefix": case.get("prefix"),
        "s_total": case.get("s_total"),
    }
    calls: list[str] = []

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(empty_cache=lambda: calls.append("empty-cache"))
    torch_module.distributed = SimpleNamespace(is_initialized=lambda: False)

    def fail_cleanup() -> None:
        calls.append("destroy-model-parallel")
        raise RuntimeError("cleanup failed")

    distributed_module = ModuleType("sglang.srt.distributed")
    distributed_module.parallel_state = SimpleNamespace(
        destroy_model_parallel=fail_cleanup,
        destroy_distributed_environment=lambda: calls.append("destroy-distributed-environment"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed_module)
    monkeypatch.setattr(
        dsv4_attn,
        "_load_model_runner",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        dsv4_attn._prepare_dsv4_attn_case(
            **case,
            device="cuda:0",
            model_path=_MODEL_ARTIFACT,
        )

    assert calls == ["destroy-model-parallel", "empty-cache"]


def test_attention_preparation_cleans_loaded_runner_when_module_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _ROUTES[0]
    request = _request(route)
    case = _expected_case(route, request.query)
    case = {
        **case,
        "isl": case.get("isl"),
        "prefix": case.get("prefix"),
        "s_total": case.get("s_total"),
    }
    calls: list[str] = []

    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(empty_cache=lambda: calls.append("empty-cache"))
    torch_module.distributed = SimpleNamespace(is_initialized=lambda: False)
    parallel_state = SimpleNamespace(
        destroy_model_parallel=lambda: calls.append("destroy-model-parallel"),
        destroy_distributed_environment=lambda: calls.append("destroy-distributed-environment"),
    )
    distributed_module = ModuleType("sglang.srt.distributed")
    distributed_module.parallel_state = parallel_state
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "sglang", ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", ModuleType("sglang.srt"))
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed_module)
    model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(clear=lambda: calls.append("clear-request-pool")),
        token_to_kv_pool_allocator=SimpleNamespace(clear=lambda: calls.append("clear-kv-pool")),
        model=SimpleNamespace(
            model=SimpleNamespace(
                layers=[
                    SimpleNamespace(
                        self_attn=SimpleNamespace(
                            compress_ratio=route.compress_ratio + 1,
                            n_heads=_PADDED_NUM_HEADS,
                        )
                    )
                ]
            ),
            config=SimpleNamespace(architectures=[_ARCHITECTURE]),
        ),
    )
    monkeypatch.setattr(dsv4_attn, "_load_model_runner", lambda *args, **kwargs: model_runner)

    with pytest.raises(ValueError, match="loaded SGLang attention module"):
        dsv4_attn._prepare_dsv4_attn_case(
            **case,
            device="cuda:0",
            model_path=_MODEL_ARTIFACT,
        )

    assert calls == [
        "clear-request-pool",
        "clear-kv-pool",
        "destroy-model-parallel",
        "destroy-distributed-environment",
        "empty-cache",
    ]


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
    events: list[str] = []

    def model_runner_factory(**kwargs):
        attempted_ports.append(kwargs["nccl_port"])
        events.append(f"attempt:{kwargs['nccl_port']}")
        if len(attempted_ports) == 1:
            raise OSError(errno.EADDRINUSE, "Address already in use")
        return object()

    result = dsv4_attn._construct_model_runner(
        model_runner_factory,
        {"model_config": object()},
        port_factory=lambda: next(ports),
        cleanup_failed_attempt=lambda: events.append("cleanup"),
    )

    assert result is not None
    assert attempted_ports == [45101, 45102]
    assert events == ["attempt:45101", "cleanup", "attempt:45102"]


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
            cleanup_failed_attempt=lambda: None,
            port_factory=port_factory,
        )

    assert port_calls == 1


@pytest.mark.parametrize(
    ("logical_tokens", "expected_full_tokens", "expected_swa_tokens"),
    (
        (257, 5120, 512),
        (641, 7680, 768),
        (642, 7680, 768),
    ),
)
def test_attention_runner_sizes_full_pool_for_exact_swa_request(
    logical_tokens: int,
    expected_full_tokens: int,
    expected_swa_tokens: int,
) -> None:
    full_tokens = dsv4_attn._full_tokens_for_swa_capacity(
        logical_tokens=logical_tokens,
        page_size=256,
        swa_full_tokens_ratio=0.1,
    )

    assert full_tokens == expected_full_tokens
    assert int(full_tokens * 0.1) // 256 * 256 == expected_swa_tokens
    assert expected_swa_tokens >= logical_tokens


@pytest.mark.parametrize(
    ("batch_size", "tokens_per_request", "expected_swa_tokens"),
    (
        (1, 257, 512),
        (2, 97, 512),
        (3, 1, 768),
    ),
)
def test_attention_runner_rounds_swa_capacity_per_request_page(
    batch_size: int,
    tokens_per_request: int,
    expected_swa_tokens: int,
) -> None:
    assert (
        dsv4_attn._page_rounded_swa_capacity_tokens(
            batch_size=batch_size,
            tokens_per_request=tokens_per_request,
            page_size=256,
        )
        == expected_swa_tokens
    )


@pytest.mark.parametrize(
    ("logical_tokens", "page_size", "swa_full_tokens_ratio"),
    ((0, 256, 0.1), (257, 0, 0.1), (257, 256, 0.0), (257, 256, 1.01)),
)
def test_attention_runner_rejects_invalid_swa_capacity_inputs(
    logical_tokens: int,
    page_size: int,
    swa_full_tokens_ratio: float,
) -> None:
    with pytest.raises((TypeError, ValueError), match=r"token|page|ratio"):
        dsv4_attn._full_tokens_for_swa_capacity(
            logical_tokens=logical_tokens,
            page_size=page_size,
            swa_full_tokens_ratio=swa_full_tokens_ratio,
        )


def test_attention_runner_cleanup_releases_model_pools_and_distributed_state() -> None:
    calls: list[str] = []

    class Pool:
        def __init__(self, name: str) -> None:
            self.name = name

        def clear(self) -> None:
            calls.append(self.name)

    class Distributed:
        initialized = True

        @classmethod
        def is_initialized(cls) -> bool:
            return cls.initialized

    class Cuda:
        @staticmethod
        def empty_cache() -> None:
            calls.append("cuda")

    class Torch:
        cuda = Cuda()
        distributed = Distributed()

    model_runner = type(
        "ModelRunner",
        (),
        {
            "req_to_token_pool": Pool("request-pool"),
            "token_to_kv_pool_allocator": Pool("kv-pool"),
        },
    )()

    def cleanup_distributed() -> None:
        calls.append("distributed")
        Distributed.initialized = False

    dsv4_attn._cleanup_model_runner(
        model_runner,
        torch_module=Torch(),
        cleanup_distributed=cleanup_distributed,
        collect_garbage=lambda: calls.append("gc"),
    )

    assert calls == ["request-pool", "kv-pool", "distributed", "cuda", "gc"]
    assert Distributed.is_initialized() is False


def test_attention_runner_cleanup_releases_distributed_state_when_pool_is_partial() -> None:
    calls: list[str] = []
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: calls.append("cuda")),
        distributed=SimpleNamespace(is_initialized=lambda: False),
    )
    partial_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(clear=lambda: calls.append("request-pool")),
    )

    with pytest.raises(RuntimeError, match="model-pool cleanup failed"):
        dsv4_attn._cleanup_model_runner(
            partial_runner,
            torch_module=torch_module,
            cleanup_distributed=lambda: calls.append("distributed"),
            collect_garbage=lambda: calls.append("gc"),
        )

    assert calls == ["request-pool", "distributed", "cuda", "gc"]


def test_attention_runner_cleanup_preserves_pool_failure_when_distributed_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool_failure = RuntimeError("request pool cleanup failed")

    def fail_pool_cleanup() -> None:
        raise pool_failure

    model_runner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(clear=fail_pool_cleanup),
        token_to_kv_pool_allocator=SimpleNamespace(clear=lambda: None),
    )
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: None),
        distributed=SimpleNamespace(is_initialized=lambda: False),
    )
    monkeypatch.setattr(dsv4_attn, "_cleanup_temporary_model_dirs", lambda: None)

    with pytest.raises(RuntimeError, match="model-pool cleanup failed") as failure:
        dsv4_attn._cleanup_model_runner(
            model_runner,
            torch_module=torch_module,
            cleanup_distributed=lambda: (_ for _ in ()).throw(RuntimeError("distributed cleanup failed")),
            collect_garbage=lambda: None,
        )

    assert failure.value.__cause__ is pool_failure


def test_attention_distributed_cleanup_preserves_first_error_when_temp_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: calls.append("cuda")),
        distributed=SimpleNamespace(is_initialized=lambda: False),
    )

    def cleanup_distributed() -> None:
        calls.append("distributed")
        raise RuntimeError("distributed cleanup failed")

    def cleanup_temp_dirs() -> None:
        calls.append("temp-dirs")
        raise OSError("temporary directory cleanup failed")

    monkeypatch.setattr(dsv4_attn, "_cleanup_temporary_model_dirs", cleanup_temp_dirs)

    with pytest.raises(RuntimeError, match="distributed cleanup failed"):
        dsv4_attn._cleanup_distributed_runtime(
            torch_module=torch_module,
            cleanup_distributed=cleanup_distributed,
            collect_garbage=lambda: calls.append("gc"),
        )

    assert calls == ["distributed", "cuda", "gc", "temp-dirs"]


def test_model_runner_port_retry_fails_closed_when_cleanup_fails() -> None:
    attempts = 0

    def model_runner_factory(**kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError(errno.EADDRINUSE, f"Address already in use: {kwargs['nccl_port']}")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        dsv4_attn._construct_model_runner(
            model_runner_factory,
            {"model_config": object()},
            cleanup_failed_attempt=lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
            port_factory=lambda: 45101,
        )

    assert attempts == 1


def test_attention_runtime_cleanup_removes_temporary_model_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dsv4_attn.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        dsv4_attn,
        "_read_model_config",
        lambda model_id: {
            "architectures": [_ARCHITECTURE],
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
        },
    )
    model_dir = Path(dsv4_attn._patched_model_dir(_MODEL_ARTIFACT, "csa", 4))
    assert (model_dir / "config.json").is_file()
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: None),
        distributed=SimpleNamespace(is_initialized=lambda: False),
    )

    dsv4_attn._cleanup_distributed_runtime(
        torch_module=torch_module,
        cleanup_distributed=lambda: None,
        collect_garbage=lambda: None,
    )

    assert not model_dir.exists()


def test_attention_temp_cleanup_attempts_remaining_dirs_after_one_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blocked = tmp_path / "blocked"
    removable = tmp_path / "removable"
    blocked.mkdir()
    removable.mkdir()

    class OrderedDirs(list[Path]):
        def discard(self, item: Path) -> None:
            if item in self:
                self.remove(item)

    tracked = OrderedDirs([blocked, removable])
    calls: list[Path] = []

    def remove_tree(path: Path) -> None:
        calls.append(path)
        if path == blocked:
            raise PermissionError("blocked temp directory")
        path.rmdir()

    monkeypatch.setattr(dsv4_attn, "_TEMPORARY_MODEL_DIRS", tracked)
    monkeypatch.setattr(dsv4_attn.shutil, "rmtree", remove_tree)

    with pytest.raises(PermissionError, match="blocked temp directory"):
        dsv4_attn._cleanup_temporary_model_dirs()

    assert calls == [blocked, removable]
    assert tracked == [blocked]
    assert blocked.is_dir()
    assert not removable.exists()


def test_attention_runner_does_not_patch_sglang_moe_class() -> None:
    source = inspect.getsource(dsv4_attn)
    assert "DeepseekV2MoE =" not in source
    assert "_attention_only_dsv4_moe" not in source
