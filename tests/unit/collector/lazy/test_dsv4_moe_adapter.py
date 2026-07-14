# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DSv4 V1.2 exact one-GPU SGLang MoE adapter and runner contracts."""

from __future__ import annotations

import builtins
import importlib
import inspect
import statistics
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from aiconfigurator.collector.adapters import LazyAdapterIndex
from aiconfigurator.collector.registry_types import PerfFile
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
_NAMESPACE = f"{PerfFile.MOE}/v1"
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
    "workload_generator": "power_law_v3",
    "seed": 0,
    "rank_simulation": "single-gpu-ep4-rank0",
}


def _moe_modules():
    runner = importlib.import_module("aiconfigurator.collector.sglang.moe")
    adapter = importlib.import_module("aiconfigurator.collector.sglang.moe_adapter")
    return runner, adapter


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
        tuning_revision="sglang-moe-v1",
    )


def _query(**overrides: object) -> dict[str, object]:
    query: dict[str, object] = {
        "num_tokens": 19,
        "hidden_size": 4096,
        "inter_size": 2048,
        "topk": 6,
        "num_experts": 256,
        "moe_tp_size": 1,
        "moe_ep_size": 4,
        "quant_mode": "fp8_block",
        "workload_distribution": "power_law_1.01",
    }
    query.update(overrides)
    return query


def _request(
    *,
    query: dict[str, object] | None = None,
    environment: MeasurementEnvironment | None = None,
    semantic_descriptor: dict[str, object] | None = None,
) -> MeasurementRequest:
    query = query or _query()
    environment = environment or _environment()
    return MeasurementRequest(
        op_id="context_moe",
        key=PerfKey.build(_NAMESPACE, query, environment),
        query=query,
        environment=environment,
        semantic_descriptor=semantic_descriptor or _SEMANTIC_DESCRIPTOR,
        protocol=_protocol(),
    )


def _request_with_runtime_mismatch(field: str, value: str) -> MeasurementRequest:
    return _request(
        environment=replace(
            _environment(),
            runtime_versions={**_RUNTIME_VERSIONS, field: value},
        )
    )


def _request_with_profile_mismatch(field: str, value: object) -> MeasurementRequest:
    return _request(
        environment=replace(
            _environment(),
            profile_compatibility={**_PROFILE_COMPATIBILITY, field: value},
        )
    )


def test_moe_rejects_malformed_sampling_protocol_as_protocol_mismatch() -> None:
    _, adapter = _moe_modules()
    request = replace(_request(), protocol=replace(_protocol(), samples=2))

    with pytest.raises(ProtocolMismatchError, match="samples must be at least three"):
        adapter.moe_request_to_case(request)


def _raw_result(request: MeasurementRequest) -> dict[str, object]:
    latency_ms = 1.25
    framework_version = request.environment.backend_version
    return {
        "latency_ms": latency_ms,
        "energy_wms": 125.0,
        "samples_ms": (1.2, latency_ms, 1.3),
        "statistic": request.protocol.statistic,
        "protocol_digest": request.protocol.digest,
        "perf_row": {
            "framework": "SGLang",
            "version": framework_version,
            "device": "NVIDIA GB200",
            "op_name": "moe",
            "kernel_source": "sglang_fused_moe_triton",
            "moe_dtype": "fp8_block",
            "num_tokens": 19,
            "hidden_size": 4096,
            "inter_size": 2048,
            "topk": 6,
            "num_experts": 256,
            "moe_tp_size": 1,
            "moe_ep_size": 4,
            "distribution": "power_law_1.01",
            "latency": latency_ms,
        },
        "provenance": {
            "framework": "SGLang",
            "framework_version": framework_version,
            "kernel_source": "sglang_fused_moe_triton",
            "device": "NVIDIA GB200",
            "used_cuda_graph": True,
            "throttled": False,
            "model_artifact": _MODEL_ARTIFACT,
            "workload_generator": "power_law_v3",
            "seed": 0,
            "rank_simulation": "single-gpu-ep4-rank0",
        },
    }


def test_packaged_and_source_registries_share_one_moe_lazy_route() -> None:
    packaged = importlib.import_module("aiconfigurator.collector.sglang.registry")
    from collector.sglang.registry import REGISTRY as SOURCE_REGISTRY

    source_entry = next(entry for entry in SOURCE_REGISTRY if entry.op == "moe")
    routes = LazyAdapterIndex.from_registries({"sglang": packaged.SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )

    assert source_entry.lazy is packaged.MOE_LAZY_SPEC
    assert len(routes) == 1
    assert routes[0].lazy is packaged.MOE_LAZY_SPEC
    assert routes[0].collector_module == "aiconfigurator.collector.sglang.moe"
    assert packaged.MOE_LAZY_SPEC.tuning_revision == "sglang-moe-v1"


def test_canonical_moe_request_round_trips_through_case_and_emitted_row() -> None:
    _, adapter = _moe_modules()
    request = _request()

    case = adapter.moe_request_to_case(request)
    assert case == _query()
    assert adapter.moe_resource_for_request(request, case) == ResourceContract(
        gpu_count=1,
        fabric=FabricRequirement.NONE,
    )

    record = adapter.moe_result_to_record(request, case, _raw_result(request))
    emitted_query = {
        "num_tokens": record.perf_row["num_tokens"],
        "hidden_size": record.perf_row["hidden_size"],
        "inter_size": record.perf_row["inter_size"],
        "topk": record.perf_row["topk"],
        "num_experts": record.perf_row["num_experts"],
        "moe_tp_size": record.perf_row["moe_tp_size"],
        "moe_ep_size": record.perf_row["moe_ep_size"],
        "quant_mode": record.perf_row["moe_dtype"],
        "workload_distribution": record.perf_row["distribution"],
    }
    assert record.key == request.key == PerfKey.build(_NAMESPACE, emitted_query, request.environment)


def test_unseen_positive_token_count_is_an_in_domain_exact_case() -> None:
    _, adapter = _moe_modules()
    request = _request(query=_query(num_tokens=23))

    assert adapter.moe_request_to_case(request) == _query(num_tokens=23)


@pytest.mark.parametrize("sglang_version", ["0.5.10", "0.5.10rc0"])
def test_moe_preserves_exact_supported_sglang_runtime_identity(sglang_version: str) -> None:
    _, adapter = _moe_modules()
    environment = replace(
        _environment(),
        backend_version=sglang_version,
        runtime_versions={**_RUNTIME_VERSIONS, "sglang": sglang_version},
    )
    request = _request(environment=environment)

    case = adapter.moe_request_to_case(request)
    record = adapter.moe_result_to_record(request, case, _raw_result(request))

    assert record.key == request.key
    assert request.environment.backend_version == sglang_version
    assert record.provenance["framework_version"] == sglang_version


@pytest.mark.parametrize(
    ("mismatch", "request_factory"),
    [
        ("system", lambda: _request(environment=replace(_environment(), system="h100_sxm"))),
        ("gpu-class", lambda: _request(environment=replace(_environment(), gpu_class="NVIDIA H100"))),
        ("backend", lambda: _request(environment=replace(_environment(), backend="vllm"))),
        (
            "backend-version",
            lambda: _request(
                environment=replace(
                    _environment(),
                    backend_version="0.5.10rc1",
                    runtime_versions={**_RUNTIME_VERSIONS, "sglang": "0.5.10rc1"},
                )
            ),
        ),
        ("cuda-runtime", lambda: _request_with_runtime_mismatch("cuda", "12.9")),
        ("model-profile-runtime", lambda: _request_with_runtime_mismatch("model_profile", "other-profile")),
        (
            "missing-sglang-runtime",
            lambda: _request(
                environment=replace(
                    _environment(),
                    runtime_versions={name: version for name, version in _RUNTIME_VERSIONS.items() if name != "sglang"},
                )
            ),
        ),
        ("artifact", lambda: _request_with_profile_mismatch("model_artifact", "other/model")),
        ("serving-mode", lambda: _request_with_profile_mismatch("serving_mode", "disaggregated")),
        ("profile-tp", lambda: _request_with_profile_mismatch("tp_size", 8)),
        ("attention-dp", lambda: _request_with_profile_mismatch("attention_dp_size", 2)),
        ("cp", lambda: _request_with_profile_mismatch("cp_size", 2)),
        ("pp", lambda: _request_with_profile_mismatch("pp_size", 2)),
        ("profile-moe-tp", lambda: _request_with_profile_mismatch("moe_tp_size", 2)),
        ("profile-moe-ep", lambda: _request_with_profile_mismatch("moe_ep_size", 8)),
        ("nextn", lambda: _request_with_profile_mismatch("nextn", 1)),
        ("quantization", lambda: _request(query=_query(quant_mode="bfloat16"))),
        ("distribution", lambda: _request(query=_query(workload_distribution="uniform"))),
        ("moe-tp", lambda: _request(query=_query(moe_tp_size=4))),
        ("moe-ep", lambda: _request(query=_query(moe_ep_size=1))),
        ("shape", lambda: _request(query=_query(hidden_size=7168))),
        ("samples", lambda: replace(_request(), protocol=replace(_protocol(), samples=2))),
        (
            "semantic-descriptor",
            lambda: _request(semantic_descriptor={**_SEMANTIC_DESCRIPTOR, "rank_simulation": "real-ep4"}),
        ),
    ],
)
def test_frozen_moe_capability_mismatch_fails_before_resource_acquisition(
    mismatch: str,
    request_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, adapter = _moe_modules()
    packaged = importlib.import_module("aiconfigurator.collector.sglang.registry")
    request = request_factory()
    resource_calls: list[object] = []

    def _resource(request, case):
        resource_calls.append((request, case))
        return ResourceContract(gpu_count=1, fabric=FabricRequirement.NONE)

    monkeypatch.setattr(adapter, "moe_resource_for_request", _resource)
    route = LazyAdapterIndex.from_registries({"sglang": packaged.SGLANG_LAZY_REGISTRY}).routes_for(
        (_NAMESPACE, "sglang", "0.5.10")
    )[0]

    with pytest.raises(
        ValueError,
        match=r"capability|profile|artifact|descriptor|shape|fp8|TP1|EP4|power_law|route|match|samples",
    ):
        route.prepare(request)
    assert resource_calls == [], mismatch


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["perf_row"].__setitem__("moe_ep_size", 8), "identity|EP4"),
        (lambda raw: raw["provenance"].__setitem__("model_artifact", "other/model"), "artifact"),
        (lambda raw: raw["provenance"].__setitem__("used_cuda_graph", False), "CUDA Graph"),
        (lambda raw: raw["provenance"].__setitem__("throttled", True), "throttled"),
        (lambda raw: raw["provenance"].__setitem__("rank_simulation", "real-ep4"), "rank|simulation"),
    ],
)
def test_moe_result_validation_fails_closed(mutation, message: str) -> None:
    _, adapter = _moe_modules()
    request = _request()
    case = adapter.moe_request_to_case(request)
    raw = _raw_result(request)
    mutation(raw)

    with pytest.raises(ValueError, match=message):
        adapter.moe_result_to_record(request, case, raw)


def test_exact_moe_runner_returns_one_raw_rank_local_row_without_logging(monkeypatch, tmp_path) -> None:
    runner, _ = _moe_modules()
    protocol = _protocol()
    calls: list[tuple[object, ...]] = []

    def _prepare(
        num_tokens,
        hidden_size,
        inter_size,
        topk,
        num_experts,
        moe_tp_size,
        moe_ep_size,
        quant_mode,
        workload_distribution,
        device,
        model_path,
    ):
        calls.append(
            (
                "prepare",
                num_tokens,
                hidden_size,
                inter_size,
                topk,
                num_experts,
                moe_tp_size,
                moe_ep_size,
                quant_mode,
                workload_distribution,
                device,
                model_path,
            )
        )
        return SimpleNamespace(
            kernel_func=lambda: calls.append(("kernel",)),
            framework_version="0.5.10",
            device_name="NVIDIA GB200",
            device=object(),
            model_artifact=_MODEL_ARTIFACT,
            kernel_source="sglang_fused_moe_triton",
            workload_generator="power_law_v3",
            seed=0,
            rank_simulation="single-gpu-ep4-rank0",
            num_tokens=19,
            hidden_size=4096,
            inter_size=2048,
            topk=6,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=4,
            quant_mode="fp8_block",
            workload_distribution="power_law_1.01",
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

    monkeypatch.setattr(runner, "_prepare_moe_case", _prepare)
    monkeypatch.setattr(runner, "benchmark_with_power", _benchmark)
    monkeypatch.chdir(tmp_path)

    result = runner.run_moe_case(**_query(), protocol=protocol)

    assert isinstance(result, RawMeasurement)
    assert result.latency_ms == 1.25
    assert result.samples_ms == (1.2, 1.25, 1.3)
    assert result.energy_wms == 125.0
    assert result.protocol_digest == protocol.digest
    assert result.perf_row == _raw_result(_request())["perf_row"]
    assert result.provenance == _raw_result(_request())["provenance"]
    assert calls[0] == (
        "prepare",
        19,
        4096,
        2048,
        6,
        256,
        1,
        4,
        "fp8_block",
        "power_law_1.01",
        "cuda:0",
        _MODEL_ARTIFACT,
    )
    benchmark_kwargs = next(call[1] for call in calls if call[0] == "benchmark")
    assert benchmark_kwargs["num_warmups"] == 2
    assert benchmark_kwargs["num_runs"] == 3
    assert benchmark_kwargs["return_samples"] is True
    assert benchmark_kwargs["allow_graph_fail"] is False
    assert benchmark_kwargs["use_cuda_graph"] is True
    assert not tuple(tmp_path.iterdir())


def test_power_law_routing_is_seeded_on_the_requested_gpu() -> None:
    runner, _ = _moe_modules()

    class RoutingCreatedError(Exception):
        pass

    class TorchProbe:
        @staticmethod
        def rand(*shape, **kwargs):
            assert shape == (256,)
            assert kwargs == {"device": "cuda:3"}
            raise RoutingCreatedError

    with pytest.raises(RoutingCreatedError):
        runner._power_law_selected_experts(
            19,
            256,
            6,
            4,
            1.01,
            device="cuda:3",
            torch_module=TorchProbe(),
        )


def test_rank0_workload_marks_remote_experts_inactive() -> None:
    runner, _ = _moe_modules()
    source = inspect.getsource(runner._rank0_workloads)

    # SGLang's rank-local fused-MoE contract uses -1 to suppress remote
    # experts; zero would schedule expert 0 even when its routing weight is 0.
    assert "ids[~local_mask] = -1" in source
    assert "weights[~local_mask] = 0.0" in source


def test_fused_moe_call_clamps_inactive_expert_ids_before_weight_indexing() -> None:
    runner, _ = _moe_modules()
    source = inspect.getsource(runner._prepare_moe_case)

    # Rank-local semantics use -1, but SGLang's Triton kernel indexes its
    # weight tensor before applying routing weights and therefore receives 0.
    assert "topk_ids=topk_output.topk_ids.clamp(min=0)" in source


@pytest.mark.parametrize(
    ("parameter_name", "expected"),
    [("swiglu_limit", 10), ("gemm1_clamp_limit", 10)],
)
def test_dsv4_moe_runner_uses_clamp_ten_across_sglang_config_names(parameter_name: str, expected: int) -> None:
    runner, _ = _moe_modules()

    if parameter_name == "swiglu_limit":

        class MoeRunnerConfig:
            def __init__(self, swiglu_limit=None):
                self.value = swiglu_limit

    else:

        class MoeRunnerConfig:
            def __init__(self, gemm1_clamp_limit=None):
                self.value = gemm1_clamp_limit

    config = runner._make_dsv4_moe_runner_config(MoeRunnerConfig)

    assert config.value == expected


def test_moe_runner_recomputes_latency_from_scaled_even_samples(monkeypatch) -> None:
    runner, _ = _moe_modules()
    protocol = replace(_protocol(), samples=4)
    raw_samples = (0.01, 0.1, 0.2, 99.0)

    monkeypatch.setattr(
        runner,
        "_prepare_moe_case",
        lambda *args: SimpleNamespace(
            kernel_func=lambda: None,
            framework_version="0.5.10",
            device_name="NVIDIA GB200",
            device=object(),
            model_artifact=_MODEL_ARTIFACT,
            kernel_source="sglang_fused_moe_triton",
            workload_generator="power_law_v3",
            seed=0,
            rank_simulation="single-gpu-ep4-rank0",
            num_tokens=19,
            hidden_size=4096,
            inter_size=2048,
            topk=6,
            num_experts=256,
            moe_tp_size=1,
            moe_ep_size=4,
            quant_mode="fp8_block",
            workload_distribution="power_law_1.01",
            latency_divisor=5,
        ),
    )

    @contextmanager
    def _benchmark(**kwargs):
        del kwargs
        yield {
            "latency_ms": statistics.median(raw_samples),
            "samples_ms": raw_samples,
            "power_stats": {"power": 100.0},
            "throttled": False,
            "used_cuda_graph": True,
        }

    monkeypatch.setattr(runner, "benchmark_with_power", _benchmark)

    result = runner.run_moe_case(**_query(), protocol=protocol)

    assert result.latency_ms == statistics.median(result.samples_ms)


def test_moe_runner_is_exact_only_import_light_and_has_no_offline_output_api(monkeypatch) -> None:
    runner, _ = _moe_modules()

    assert runner.get_moe_test_cases() == ()
    assert {"output_path", "perf_filename", "num_tokens_cases", "model_cases"}.isdisjoint(
        inspect.signature(runner.run_moe_case).parameters
    )
    source = Path(runner.__file__).read_text()
    assert "log_perf" not in source
    assert "from collector" not in source
    assert "import collector" not in source

    module_name = "aiconfigurator.collector.sglang.moe"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch.") or name == "sglang" or name.startswith("sglang."):
            raise AssertionError(f"heavy import at module import time: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    importlib.import_module(module_name)
